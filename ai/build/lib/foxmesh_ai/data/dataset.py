"""Dataset management: a directory of meshes with captions, images and metadata.

Layout::

    my_dataset/
      dataset.json     name, description, license, created/updated, split ratios
      items.jsonl      one JSON object per item (see ``Item``)
      meshes/          normalised source meshes (GLB, with vertex colours if known)
      images/          optional reference images (PNG)
      cache/           preprocessed training samples (``preprocess.py``), safe to delete

Items are identified by the SHA-256 of their normalised geometry, so importing the
same asset twice is detected. Splits are assigned from the item id, so they are
stable when items are added or removed. ``fingerprint()`` identifies the exact
content of a dataset version and is stored with every trained model.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from ..geometry import MESH_EXTENSIONS, TriMesh, load_mesh, normalize, write_glb

SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


@dataclass
class Item:
    id: str
    mesh: str  # path relative to the dataset root
    captions: list[str]
    images: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    split: str = "train"
    source: str = "unknown"
    license: str = "unknown"
    faces: int = 0
    watertight: bool = False
    extra: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    items: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def mesh_digest(mesh: TriMesh) -> str:
    """Content hash of normalised geometry (rounded, so re-exports of one asset match)."""
    h = hashlib.sha256()
    h.update(np.round(mesh.vertices, 4).astype(np.float32).tobytes())
    h.update(mesh.faces.astype(np.int64).tobytes())
    return h.hexdigest()


def split_for(item_id: str, ratios: tuple[float, float, float], seed: int = 0) -> str:
    bucket = int(hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < ratios[0]:
        return "train"
    if bucket < ratios[0] + ratios[1]:
        return "val"
    return "test"


class Dataset:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "dataset.json").read_text())
        self._items: dict[str, Item] = {}
        manifest = self.root / "items.jsonl"
        if manifest.exists():
            for line in manifest.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = Item(**json.loads(line))
                    self._items[item.id] = item

    # -- creation ------------------------------------------------------------------

    @classmethod
    def create(cls, root: str | Path, name: str, description: str = "", license: str = "mixed",
               split_ratios: tuple[float, float, float] = (0.9, 0.05, 0.05), exist_ok: bool = False
               ) -> "Dataset":
        root = Path(root)
        if (root / "dataset.json").exists():
            if not exist_ok:
                raise FileExistsError(f"dataset already exists at {root}")
            return cls(root)
        for sub in ("meshes", "images", "cache"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        now = time.time()
        meta = {"name": name, "description": description, "license": license,
                "split_ratios": list(split_ratios), "split_seed": 0, "created": now, "updated": now,
                "format": 1}
        (root / "dataset.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
        (root / "items.jsonl").touch()
        return cls(root)

    # -- access --------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Item]:
        return iter(sorted(self._items.values(), key=lambda i: i.id))

    def get(self, item_id: str) -> Item | None:
        return self._items.get(item_id)

    def items(self, split: str | None = None) -> list[Item]:
        return [i for i in self if split is None or i.split == split]

    def path(self, relative: str) -> Path:
        return self.root / relative

    def load(self, item: Item) -> TriMesh:
        return load_mesh(self.path(item.mesh))

    # -- editing -------------------------------------------------------------------

    def add(self, mesh: TriMesh | str | Path, captions: Iterable[str], images: Iterable[str | Path] = (),
            tags: Iterable[str] = (), source: str = "unknown", license: str = "unknown",
            extra: dict | None = None, save: bool = True) -> tuple[Item, bool]:
        """Add one asset. Returns ``(item, added)``; ``added`` is False for duplicates,
        whose captions and tags are merged into the existing item instead."""
        if not isinstance(mesh, TriMesh):
            mesh = load_mesh(mesh)
        mesh, _ = normalize(mesh)
        item_id = mesh_digest(mesh)[:16]
        captions = [c.strip() for c in captions if c and c.strip()]
        if item_id in self._items:
            existing = self._items[item_id]
            existing.captions = list(dict.fromkeys(existing.captions + captions))
            existing.tags = sorted(set(existing.tags) | set(tags))
            if save:
                self.save()
            return existing, False
        mesh_rel = f"meshes/{item_id}.glb"
        write_glb(mesh, self.root / mesh_rel)
        image_rels = []
        for k, image in enumerate(images):
            image = Path(image)
            rel = f"images/{item_id}_{k}{image.suffix.lower()}"
            shutil.copyfile(image, self.root / rel)
            image_rels.append(rel)
        ratios = tuple(self.meta["split_ratios"])
        item = Item(id=item_id, mesh=mesh_rel, captions=captions, images=image_rels,
                    tags=sorted(set(tags)), split=split_for(item_id, ratios, self.meta["split_seed"]),
                    source=source, license=license, faces=int(len(mesh.faces)),
                    watertight=mesh.watertight, extra=extra or {})
        self._items[item_id] = item
        if save:
            self.save()
        return item, True

    def remove(self, item_id: str) -> None:
        item = self._items.pop(item_id, None)
        if item:
            for rel in [item.mesh, *item.images]:
                (self.root / rel).unlink(missing_ok=True)
            (self.root / "cache" / f"{item_id}.npz").unlink(missing_ok=True)
            self.save()

    def import_folder(self, folder: str | Path, source: str = "import", license: str = "unknown",
                      tags: Iterable[str] = ()) -> dict[str, int]:
        """Import every mesh in ``folder`` (recursively).

        Captions come from a sidecar with the same stem: ``chair.txt`` (one caption per
        line) or ``chair.json`` (``{"captions": [...], "tags": [...], "license": ...}``);
        otherwise the file name is used ("red_chair.glb" -> "red chair"). Images named
        ``chair.png`` or ``chair_*.png`` are attached as reference images.
        """
        folder = Path(folder)
        counts = Counter()
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() not in MESH_EXTENSIONS:
                continue
            captions: list[str] = []
            item_tags, item_license = list(tags), license
            if (txt := path.with_suffix(".txt")).exists():
                captions = txt.read_text(encoding="utf-8").splitlines()
            if (js := path.with_suffix(".json")).exists():
                meta = json.loads(js.read_text(encoding="utf-8"))
                captions += meta.get("captions", [])
                item_tags += meta.get("tags", [])
                item_license = meta.get("license", item_license)
            if not captions:
                captions = [path.stem.replace("_", " ").replace("-", " ")]
            images = [p for p in sorted(path.parent.glob(f"{path.stem}*"))
                      if p.suffix.lower() in IMAGE_EXTENSIONS]
            try:
                _, added = self.add(path, captions, images, item_tags, source, item_license, save=False)
                counts["added" if added else "duplicates"] += 1
            except Exception:
                counts["failed"] += 1
        self.save()
        return dict(counts)

    def resplit(self, ratios: tuple[float, float, float], seed: int = 0) -> dict[str, int]:
        if abs(sum(ratios) - 1) > 1e-6:
            raise ValueError("split ratios must sum to 1")
        self.meta["split_ratios"], self.meta["split_seed"] = list(ratios), seed
        for item in self._items.values():
            item.split = split_for(item.id, ratios, seed)
        self.save()
        return self.stats()["splits"]

    def save(self) -> None:
        self.meta["updated"] = time.time()
        (self.root / "dataset.json").write_text(json.dumps(self.meta, indent=1, ensure_ascii=False))
        tmp = self.root / "items.jsonl.tmp"
        with tmp.open("w", encoding="utf-8") as fh:
            for item in self:
                fh.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
        tmp.replace(self.root / "items.jsonl")

    # -- inspection ----------------------------------------------------------------

    def fingerprint(self) -> str:
        """Hash of the dataset's content (items, captions, splits): the dataset version."""
        h = hashlib.sha256()
        for item in self:
            h.update(json.dumps([item.id, item.captions, item.split, item.images],
                                ensure_ascii=False).encode())
        return h.hexdigest()[:16]

    def stats(self) -> dict:
        items = list(self)
        faces = [i.faces for i in items] or [0]
        return {
            "name": self.meta["name"],
            "items": len(items),
            "fingerprint": self.fingerprint(),
            "splits": {s: sum(i.split == s for i in items) for s in SPLITS},
            "sources": dict(Counter(i.source for i in items)),
            "licenses": dict(Counter(i.license for i in items)),
            "tags": dict(Counter(t for i in items for t in i.tags).most_common(30)),
            "captions_per_item": round(sum(len(i.captions) for i in items) / max(len(items), 1), 2),
            "with_images": sum(bool(i.images) for i in items),
            "watertight": sum(i.watertight for i in items),
            "faces": {"min": min(faces), "median": int(np.median(faces)), "max": max(faces)},
            "preprocessed": sum((self.root / "cache" / f"{i.id}.npz").exists() for i in items),
        }

    def validate(self, deep: bool = False) -> ValidationReport:
        """Check files and metadata; ``deep`` also loads every mesh."""
        report = ValidationReport(items=len(self))
        for item in self:
            mesh_path = self.root / item.mesh
            if not mesh_path.exists():
                report.errors.append(f"{item.id}: mesh file missing ({item.mesh})")
                continue
            if not item.captions:
                report.errors.append(f"{item.id}: no captions")
            for image in item.images:
                if not (self.root / image).exists():
                    report.errors.append(f"{item.id}: image missing ({image})")
            if item.split not in SPLITS:
                report.errors.append(f"{item.id}: unknown split {item.split!r}")
            if item.license in ("", "unknown"):
                report.warnings.append(f"{item.id}: license unknown")
            if not item.watertight:
                report.warnings.append(f"{item.id}: mesh is not watertight (occupancy labels may be noisy)")
            if deep:
                try:
                    mesh = load_mesh(mesh_path)
                    if mesh_digest(normalize(mesh)[0])[:16] != item.id:
                        report.warnings.append(f"{item.id}: mesh content changed since import")
                except Exception as exc:
                    report.errors.append(f"{item.id}: cannot load mesh ({exc})")
        if self.items("val") == [] and len(self) >= 20:
            report.warnings.append("validation split is empty")
        return report
