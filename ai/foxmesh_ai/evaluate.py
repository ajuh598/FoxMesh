"""Generation quality evaluation on a dataset split.

Metrics (shapes in the normalised [-1, 1]^3 space):

* IoU of predicted vs true occupancy at the item's uniform sample points
* Chamfer-L1 distance and F-score at 0.02 (1% of the cube side) between the
  generated mesh surface and the ground-truth surface samples
* watertight rate of generated meshes and mean generation time
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from .data.dataset import Dataset
from .data.preprocess import load_sample
from .generate import Generator
from .geometry import chamfer_and_fscore, sample_surface


@dataclass
class EvalReport:
    split: str
    modality: str
    items: int
    iou: float
    chamfer: float
    fscore: float
    watertight_rate: float
    mean_generation_s: float
    per_tag: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _group(item) -> str:
    """Breakdown key: the synthetic family when known, else the first tag."""
    return item.extra.get("family") or (item.tags[0] if item.tags else "untagged")


def evaluate(generator: Generator, dataset: Dataset, split: str = "val", modality: str = "text",
             resolution: int = 48, max_items: int | None = None, surface_points: int = 4096,
             seed: int = 0) -> EvalReport:
    if modality not in ("text", "image"):
        raise ValueError("modality must be 'text' or 'image'")
    items = [i for i in dataset.items(split) if (dataset.root / "cache" / f"{i.id}.npz").exists()]
    items = items[:max_items] if max_items else items
    if not items:
        raise ValueError(f"no preprocessed items in split {split!r}")
    rng = np.random.default_rng(seed)
    rows, failures = [], []
    for item in items:
        sample = load_sample(dataset.root / "cache" / f"{item.id}.npz")
        try:
            started = time.perf_counter()
            if modality == "text":
                cond = generator.condition(prompt=item.captions[0])
            else:
                cond = generator.condition(image=sample["renders"][0])
            pts = torch.from_numpy(sample["uniform_points"].astype(np.float32)).unsqueeze(0).to(generator.device)
            with torch.no_grad():
                logits = generator.model.decode(pts, cond)[0][0].cpu().numpy()
            pred, true = logits > 0, sample["uniform_occ"]
            union = (pred | true).sum()
            iou = 1.0 if union == 0 else float((pred & true).sum() / union)
            # Same post-processing as generation (fragment removal), without smoothing.
            mesh = generator.mesh_from_field(generator.field(cond, resolution), smooth=0)
            elapsed = time.perf_counter() - started
            gen_pts, _, _ = sample_surface(mesh, surface_points, rng)
            gt = sample["surface_points"].astype(np.float32)
            gt = gt[rng.choice(len(gt), min(surface_points, len(gt)), replace=False)]
            chamfer, fscore = chamfer_and_fscore(gen_pts, gt)
            rows.append({"tag": _group(item), "iou": iou, "chamfer": chamfer,
                         "fscore": fscore, "watertight": mesh.watertight, "seconds": elapsed})
        except Exception as exc:
            failures.append(f"{item.id}: {exc}")
            rows.append({"tag": _group(item), "iou": 0.0, "chamfer": 1.0,
                         "fscore": 0.0, "watertight": False, "seconds": 0.0})

    def mean(key, subset=rows):
        return round(float(np.mean([r[key] for r in subset])), 4)

    by_tag = defaultdict(list)
    for r in rows:
        by_tag[r["tag"]].append(r)
    per_tag = {t: {"n": len(rs), "iou": mean("iou", rs), "fscore": mean("fscore", rs)}
               for t, rs in sorted(by_tag.items())}
    return EvalReport(split=split, modality=modality, items=len(rows), iou=mean("iou"),
                      chamfer=mean("chamfer"), fscore=mean("fscore"), watertight_rate=mean("watertight"),
                      mean_generation_s=mean("seconds"), per_tag=per_tag, failures=failures)
