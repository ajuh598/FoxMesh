"""Training and fine-tuning for the FoxMesh model.

Each step samples, per shape, points near the surface and uniformly in space
(occupancy BCE) plus surface points (colour L1). The conditioning modality is
drawn per sample (text only, image only, or both) so one model serves text-to-3D
and image-to-3D. Checkpoints keep EMA weights, which are what ``generate`` uses.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from .data.dataset import Dataset
from .data.preprocess import load_sample
from .model import FoxMeshModel, ModelConfig, build_model
from .text import PAD_ID, tokenize


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 200
    uniform_points: int = 1024
    near_points: int = 1024
    surface_points: int = 512
    color_weight: float = 0.5
    # Probability of each conditioning mode per sample: text only, image only, both.
    p_text: float = 0.5
    p_image: float = 0.25
    p_both: float = 0.25
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    eval_every: int = 1  # epochs
    early_stop_patience: int = 0  # epochs without improvement; 0 disables
    num_workers: int = 0
    seed: int = 0
    device: str = "auto"
    freeze: list[str] = field(default_factory=list)  # "text", "image", "decoder"
    max_steps: int | None = None  # stop early (tests, smoke runs)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def image_tensor(rgba: np.ndarray) -> torch.Tensor:
    """(H, W, 4) uint8 -> (4, H, W) float in [-1, 1], RGB premultiplied by alpha."""
    x = torch.from_numpy(rgba.astype(np.float32) / 255.0)
    rgb, alpha = x[..., :3] * x[..., 3:], x[..., 3:]
    return torch.cat([rgb, alpha], -1).permute(2, 0, 1) * 2 - 1


class ShapeDataset(TorchDataset):
    """Training samples from a preprocessed ``Dataset`` split."""

    def __init__(self, dataset: Dataset, split: str, cfg: TrainConfig, train: bool = True):
        self.items = [i for i in dataset.items(split) if (dataset.root / "cache" / f"{i.id}.npz").exists()]
        if not self.items:
            raise ValueError(f"no preprocessed items in split {split!r}; run preprocessing first")
        self.cache_dir = dataset.root / "cache"
        self.cfg = cfg
        self.train = train
        self._memory: dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _sample(self, item_id: str) -> dict:
        if item_id not in self._memory:
            self._memory[item_id] = load_sample(self.cache_dir / f"{item_id}.npz")
        return self._memory[item_id]

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        s = self._sample(item.id)
        cfg = self.cfg
        rng = np.random.default_rng(None if self.train else index)
        ui = rng.choice(len(s["uniform_points"]), cfg.uniform_points, replace=False)
        ni = rng.choice(len(s["near_points"]), cfg.near_points, replace=False)
        si = rng.choice(len(s["surface_points"]), cfg.surface_points, replace=False)
        points = np.concatenate([s["uniform_points"][ui], s["near_points"][ni]]).astype(np.float32)
        occ = np.concatenate([s["uniform_occ"][ui], s["near_occ"][ni]]).astype(np.float32)

        if self.train:
            r = rng.random()
            use_text = r < cfg.p_text + cfg.p_both
            use_image = r >= cfg.p_text
            caption = item.captions[rng.integers(len(item.captions))]
            view = int(rng.integers(len(s["renders"])))
        else:
            use_text, use_image = True, False
            caption, view = item.captions[0], 0
        image = s["renders"][view]
        if self.train:
            # Small shifts so the encoder does not depend on exact framing.
            dx, dy = rng.integers(-6, 7, 2)
            image = np.roll(image, (int(dy), int(dx)), axis=(0, 1))
        return {
            "points": torch.from_numpy(points),
            "occ": torch.from_numpy(occ),
            "surface": torch.from_numpy(s["surface_points"][si].astype(np.float32)),
            "colors": torch.from_numpy(s["surface_colors"][si].astype(np.float32) / 255.0),
            "has_colors": torch.tensor(bool(s["has_colors"])),
            "tokens": torch.tensor(tokenize(caption) or [PAD_ID]),
            "image": image_tensor(image),
            "text_mask": torch.tensor(float(use_text)),
            "image_mask": torch.tensor(float(use_image)),
        }


def collate(batch: list[dict]) -> dict:
    out = {k: torch.stack([b[k] for b in batch]) for k in batch[0] if k != "tokens"}
    length = max(len(b["tokens"]) for b in batch)
    out["tokens"] = torch.stack([F.pad(b["tokens"], (0, length - len(b["tokens"])), value=PAD_ID)
                                 for b in batch])
    return out


@dataclass
class TrainResult:
    out_dir: str
    best_checkpoint: str
    best_val_iou: float
    epochs: int
    steps: int
    seconds: float
    history: list[dict]


def iou_from_logits(logits: torch.Tensor, occ: torch.Tensor) -> float:
    pred = logits > 0
    target = occ > 0.5
    union = (pred | target).sum().item()
    return 1.0 if union == 0 else (pred & target).sum().item() / union


class Trainer:
    def __init__(self, dataset: Dataset, out_dir: str | Path, cfg: TrainConfig | None = None,
                 model_cfg: ModelConfig | None = None, init_from: str | Path | None = None,
                 resume: bool = False, log=print):
        self.dataset = dataset
        self.out = Path(out_dir)
        self.cfg = cfg or TrainConfig()
        self.log = log
        self.device = pick_device(self.cfg.device)
        torch.manual_seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        self.start_epoch, self.step = 0, 0
        self.history: list[dict] = []
        self.best_iou = -1.0
        ckpt = None
        if resume and (self.out / "checkpoints" / "last.pt").exists():
            ckpt = torch.load(self.out / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
            model_cfg = ModelConfig(**ckpt["model_config"])
        elif init_from:
            base = torch.load(init_from, map_location="cpu", weights_only=False)
            model_cfg = ModelConfig(**base["model_config"])
        self.model_cfg = model_cfg or ModelConfig()
        self.model = build_model(self.model_cfg).to(self.device)
        if init_from and ckpt is None:
            self.model.load_state_dict(base.get("ema") or base["model"])
            self.log(f"initialised from {init_from}")
        self._apply_freeze()
        self.ema = copy.deepcopy(self.model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        if ckpt is not None:
            self.model.load_state_dict(ckpt["model"])
            self.ema.load_state_dict(ckpt["ema"])
            self.opt.load_state_dict(ckpt["optimizer"])
            self.start_epoch, self.step = ckpt["epoch"] + 1, ckpt["step"]
            self.history, self.best_iou = ckpt.get("history", []), ckpt.get("best_iou", -1.0)
            self.log(f"resumed at epoch {self.start_epoch}, step {self.step}")

        self.train_set = ShapeDataset(dataset, "train", self.cfg, train=True)
        val_split = "val" if dataset.items("val") else "train"
        self.val_set = ShapeDataset(dataset, val_split, self.cfg, train=False)
        self.loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True,
                                 num_workers=self.cfg.num_workers, collate_fn=collate,
                                 drop_last=len(self.train_set) > self.cfg.batch_size)
        self.total_steps = self.cfg.max_steps or self.cfg.epochs * max(len(self.loader), 1)

    def _apply_freeze(self) -> None:
        groups = {"text": [self.model.text], "image": [self.model.image],
                  "decoder": [self.model.inp, self.model.blocks, self.model.occ_head, self.model.color_head]}
        for name in self.cfg.freeze:
            for module in groups[name]:
                for p in module.parameters():
                    p.requires_grad_(False)

    def _lr(self) -> float:
        if self.step < self.cfg.warmup_steps:
            return self.cfg.lr * (self.step + 1) / self.cfg.warmup_steps
        progress = min(1.0, (self.step - self.cfg.warmup_steps) / max(1, self.total_steps - self.cfg.warmup_steps))
        return self.cfg.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress)))

    def _loss(self, model: FoxMeshModel, batch: dict) -> tuple[torch.Tensor, dict]:
        b = {k: v.to(self.device) for k, v in batch.items()}
        cond = model.encode(b["tokens"], b["image"], b["text_mask"], b["image_mask"])
        occ_logits, _ = model.decode(b["points"], cond)
        occ_loss = F.binary_cross_entropy_with_logits(occ_logits, b["occ"])
        _, colors = model.decode(b["surface"], cond)
        weight = b["has_colors"].float()[:, None, None]
        color_loss = ((colors - b["colors"]).abs() * weight).sum() / (weight.sum() * colors.shape[1] * 3).clamp(min=1)
        loss = occ_loss + self.cfg.color_weight * color_loss
        return loss, {"loss": loss.item(), "occ_loss": occ_loss.item(), "color_loss": color_loss.item(),
                      "iou": iou_from_logits(occ_logits.detach(), b["occ"])}

    @torch.no_grad()
    def validate(self) -> dict:
        loader = DataLoader(self.val_set, batch_size=self.cfg.batch_size, collate_fn=collate)
        totals, n = {}, 0
        for batch in loader:
            _, metrics = self._loss(self.ema, batch)
            k = len(batch["occ"])
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * k
            n += k
        return {f"val_{k}": v / max(n, 1) for k, v in totals.items()}

    def _save(self, name: str, epoch: int) -> Path:
        path = self.out / "checkpoints" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self.model.state_dict(), "ema": self.ema.state_dict(),
            "optimizer": self.opt.state_dict(), "model_config": self.model_cfg.to_dict(),
            "train_config": asdict(self.cfg), "epoch": epoch, "step": self.step,
            "history": self.history, "best_iou": self.best_iou,
            "dataset": {"name": self.dataset.meta["name"], "fingerprint": self.dataset.fingerprint(),
                        "root": str(self.dataset.root), "items": len(self.dataset),
                        "sources": self.dataset.stats()["sources"]},
        }, path)
        return path

    def train(self) -> TrainResult:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.json").write_text(json.dumps(
            {"train": asdict(self.cfg), "model": self.model_cfg.to_dict(),
             "dataset": self.dataset.stats()}, indent=1, ensure_ascii=False))
        log_file = (self.out / "train_log.jsonl").open("a")
        started = time.time()
        stale = 0
        epoch = self.start_epoch
        self.log(f"training FoxMesh ({self.model.parameter_count():,} params) on {self.device}: "
                 f"{len(self.train_set)} train / {len(self.val_set)} val shapes")
        try:
            for epoch in range(self.start_epoch, self.cfg.epochs):
                self.model.train()
                running: dict[str, float] = {}
                for batch in self.loader:
                    for group in self.opt.param_groups:
                        group["lr"] = self._lr()
                    loss, metrics = self._loss(self.model, batch)
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.opt.step()
                    with torch.no_grad():
                        decay = min(self.cfg.ema_decay, (1 + self.step) / (10 + self.step))
                        for pe, pm in zip(self.ema.parameters(), self.model.parameters()):
                            pe.lerp_(pm, 1 - decay)
                    self.step += 1
                    for key, value in metrics.items():
                        running[key] = running.get(key, 0.0) * 0.9 + value * 0.1 if key in running else value
                    if self.cfg.max_steps and self.step >= self.cfg.max_steps:
                        break
                record = {"epoch": epoch, "step": self.step, "lr": self._lr(),
                          "seconds": round(time.time() - started, 1), **{f"train_{k}": v for k, v in running.items()}}
                if (epoch + 1) % self.cfg.eval_every == 0 or epoch == self.cfg.epochs - 1:
                    record.update(self.validate())
                    if record["val_iou"] > self.best_iou:
                        self.best_iou = record["val_iou"]
                        self._save("best.pt", epoch)
                        stale = 0
                    else:
                        stale += 1
                self.history.append(record)
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                self._save("last.pt", epoch)
                self.log(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()))
                if self.cfg.max_steps and self.step >= self.cfg.max_steps:
                    break
                if self.cfg.early_stop_patience and stale >= self.cfg.early_stop_patience:
                    self.log(f"early stop: no val IoU improvement for {stale} evaluations")
                    break
        finally:
            log_file.close()
        best = self.out / "checkpoints" / "best.pt"
        if not best.exists():
            best = self._save("best.pt", epoch)
        return TrainResult(out_dir=str(self.out), best_checkpoint=str(best), best_val_iou=self.best_iou,
                           epochs=epoch + 1, steps=self.step, seconds=round(time.time() - started, 1),
                           history=self.history)


def finetune_config(base: TrainConfig | None = None, **overrides) -> TrainConfig:
    """Defaults for fine-tuning an existing model: lower LR, short warmup, early stopping."""
    cfg = copy.deepcopy(base) if base else TrainConfig()
    cfg.lr, cfg.warmup_steps, cfg.epochs, cfg.early_stop_patience = 2e-4, 50, 20, 5
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg
