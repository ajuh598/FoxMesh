"""Model versioning: register trained checkpoints, promote one to production, export to the platform.

Layout::

    models/
      index.json            {"production": "v2", "versions": {"v1": {...}, "v2": {...}}}
      v1/model.pt           EMA weights + model config + training and dataset metadata
      v1/metrics.json       evaluation report(s)
      v1/card.md            model card (data, training, metrics, limitations)
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import torch

from . import MODEL_NAME, __version__


class ModelStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.index = (json.loads(self.index_path.read_text()) if self.index_path.exists()
                      else {"production": None, "versions": {}})

    def _save(self) -> None:
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.index, indent=1, ensure_ascii=False))
        tmp.replace(self.index_path)

    def _next_name(self) -> str:
        numbers = [int(m.group(1)) for v in self.index["versions"] if (m := re.fullmatch(r"v(\d+)", v))]
        return f"v{max(numbers, default=0) + 1}"

    # -- registration ------------------------------------------------------------

    def register(self, checkpoint: str | Path, metrics: dict | None = None, notes: str = "",
                 name: str | None = None, parent: str | None = None) -> str:
        """Copy a training checkpoint in as a new version (weights only, no optimiser state)."""
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        version = name or self._next_name()
        if version in self.index["versions"]:
            raise ValueError(f"version {version} already exists")
        vdir = self.root / version
        vdir.mkdir(parents=True)
        params = sum(t.numel() for t in (ckpt.get("ema") or ckpt["model"]).values())
        torch.save({"ema": ckpt.get("ema") or ckpt["model"], "model_config": ckpt["model_config"],
                    "train_config": ckpt.get("train_config"), "dataset": ckpt.get("dataset"),
                    "step": ckpt.get("step"), "epoch": ckpt.get("epoch"), "version": version},
                   vdir / "model.pt")
        summary = {
            "created": time.time(), "parameters": int(params), "parent": parent, "notes": notes,
            "dataset": ckpt.get("dataset"), "steps": ckpt.get("step"),
            "best_val_iou": ckpt.get("best_iou"), "library_version": __version__,
            "metrics": metrics or {},
        }
        (vdir / "metrics.json").write_text(json.dumps(metrics or {}, indent=1, ensure_ascii=False))
        self.index["versions"][version] = summary
        self._save()
        (vdir / "card.md").write_text(self.model_card(version))
        return version

    def set_metrics(self, version: str, metrics: dict) -> None:
        self.index["versions"][version]["metrics"] = metrics
        (self.root / version / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False))
        self._save()
        (self.root / version / "card.md").write_text(self.model_card(version))

    def promote(self, version: str) -> None:
        if version not in self.index["versions"]:
            raise KeyError(version)
        self.index["production"] = version
        self._save()

    def delete(self, version: str) -> None:
        if self.index["production"] == version:
            raise ValueError("cannot delete the production version; promote another first")
        self.index["versions"].pop(version)
        for path in sorted((self.root / version).glob("*")):
            path.unlink()
        (self.root / version).rmdir()
        self._save()

    # -- lookup ------------------------------------------------------------------

    def versions(self) -> list[tuple[str, dict]]:
        return sorted(self.index["versions"].items(), key=lambda kv: kv[1]["created"])

    def resolve(self, version: str = "production") -> str:
        if version == "production":
            if not self.index["production"]:
                raise LookupError("no production version; promote one with `foxmesh-ai models promote`")
            return self.index["production"]
        if version == "latest":
            if not self.index["versions"]:
                raise LookupError("no versions registered")
            return self.versions()[-1][0]
        if version not in self.index["versions"]:
            raise LookupError(f"unknown version {version!r}")
        return version

    def checkpoint(self, version: str = "production") -> Path:
        return self.root / self.resolve(version) / "model.pt"

    # -- documentation and export ------------------------------------------------

    def model_card(self, version: str) -> str:
        info = self.index["versions"][version]
        m = info.get("metrics") or {}
        ds = info.get("dataset") or {}
        lines = [
            f"# {MODEL_NAME} {version}",
            "",
            f"- Created: {time.strftime('%Y-%m-%d %H:%M', time.localtime(info['created']))}",
            f"- Parameters: {info['parameters']:,}",
            f"- Training steps: {info.get('steps')}",
            f"- Dataset: {ds.get('name')} (fingerprint `{ds.get('fingerprint')}`)",
            f"- Parent version: {info.get('parent') or '-'}",
            "",
            "## Model",
            "",
            "Conditional occupancy network. Text (hashed character n-grams + Transformer) and/or an",
            "RGBA image (CNN) produce a condition vector; a FiLM-conditioned MLP predicts inside/outside",
            "and colour for any 3D point. Meshes are extracted with manifold surface nets, so outputs",
            "are closed and edge-manifold.",
            "",
            "## Evaluation",
            "",
        ]
        if m:
            lines += ["| split | modality | items | IoU | Chamfer-L1 | F-score@0.02 | watertight | s/shape |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- |"]
            for key, r in m.items():
                if isinstance(r, dict) and "iou" in r:
                    lines.append(f"| {r['split']} | {r['modality']} | {r['items']} | {r['iou']:.3f} | "
                                 f"{r['chamfer']:.4f} | {r['fscore']:.3f} | {r['watertight_rate']:.2f} | "
                                 f"{r['mean_generation_s']:.2f} |")
        else:
            lines.append("Not evaluated yet (`foxmesh-ai eval`).")
        lines += [
            "",
            "## Intended use and limitations",
            "",
            "- Produces coarse, coloured shapes for prototyping and as a starting point for editing.",
            "- Quality outside the training distribution is much lower than the scores above:",
            "  a model trained only on the synthetic families knows only those object types.",
            "- Colours are per-vertex, not textures. Generation is deterministic for a given input.",
            "",
            f"Notes: {info.get('notes') or '-'}",
        ]
        return "\n".join(lines) + "\n"

    def platform_spec(self, version: str = "production", device: str = "auto", resolution: int = 64) -> dict:
        """A FoxMesh platform model descriptor (``ModelSpec`` JSON) for this version.

        Quality priors come from the evaluation report; the platform replaces them with
        measured values as it runs the model.
        """
        version = self.resolve(version)
        info = self.index["versions"][version]
        metrics = info.get("metrics") or {}
        text = metrics.get("val_text") or next((r for r in metrics.values() if isinstance(r, dict)), {})
        fscore = float(text.get("fscore", 0.3))
        iou = float(text.get("iou", 0.3))
        gpu = device == "cuda" or (device == "auto" and torch.cuda.is_available())
        seconds = float(text.get("mean_generation_s", 5.0)) or 5.0
        return {
            "id": "foxmesh",
            "name": f"{MODEL_NAME} (own model)",
            "version": version,
            "adapter": "foxmesh_ai",
            "description": (f"FoxMesh's own text/image-to-3D model, {version}. Trained on dataset "
                            f"{(info.get('dataset') or {}).get('name')}; quality outside the training "
                            "domain is lower than the evaluation scores."),
            "tasks": ["generate_3d", "image_to_3d"],
            "inputs": ["text", "image"],
            "outputs": ["glb", "obj"],
            "runtime": "local_gpu" if gpu else "cpu",
            "vram_gb": 2 if gpu else 0,
            "est_runtime_s": round(max(seconds * 1.5, 0.5), 2),
            "quality": {"accuracy": round(min(max(fscore, 0.05), 0.95), 3),
                        "mesh": round(min(max(0.5 * iou + 0.5 * float(text.get("watertight_rate", 1.0)), 0.05), 0.95), 3),
                        "texture": 0.3},
            "topology": "dense",
            "watertight": True,
            "produces_textures": False,
            "license": "FoxMesh (own model)",
            "commercial_use": True,
            "enabled": True,
            "config": {"store": str(self.root.resolve()), "version": version, "resolution": resolution},
        }

    def export(self, models_dir: str | Path, version: str = "production", **kwargs) -> Path:
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        path = models_dir / "foxmesh.json"
        path.write_text(json.dumps(self.platform_spec(version, **kwargs), indent=1, ensure_ascii=False))
        return path
