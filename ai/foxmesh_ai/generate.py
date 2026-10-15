"""Inference: prompt and/or image -> coloured, watertight mesh."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .geometry import TriMesh, remove_small_components, surface_nets, taubin_smooth
from .model import build_model
from .text import PAD_ID, tokenize
from .train import image_tensor, pick_device


def prepare_image(image: str | Path | Image.Image | np.ndarray, size: int) -> np.ndarray:
    """Cut the object out and frame it like the training renders (RGBA, object ~90% of the frame).

    Uses the alpha channel when present; otherwise treats the border colour as background.
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    rgba = np.asarray(image.convert("RGBA")).astype(np.float32)
    alpha = rgba[..., 3]
    if (alpha < 250).mean() > 0.01:
        mask = alpha > 127
    else:
        border = np.concatenate([rgba[0, :, :3], rgba[-1, :, :3], rgba[:, 0, :3], rgba[:, -1, :3]])
        dist = np.linalg.norm(rgba[..., :3] - np.median(border, axis=0), axis=-1)
        mask = dist > max(30.0, float(np.percentile(dist, 50)))
        if mask.mean() < 0.01 or mask.mean() > 0.97:
            mask = np.ones_like(mask)
    rows, cols = np.nonzero(mask)
    crop = rgba[rows.min():rows.max() + 1, cols.min():cols.max() + 1].copy()
    crop[..., 3] = mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1] * 255
    h, w = crop.shape[:2]
    side = int(max(h, w) / 0.9)
    canvas = np.zeros((side, side, 4), np.float32)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = crop
    out = Image.fromarray(canvas.astype(np.uint8)).resize((size, size), Image.BILINEAR)
    return np.asarray(out)


class Generator:
    def __init__(self, checkpoint: str | Path, device: str = "auto"):
        self.path = Path(checkpoint)
        ckpt = torch.load(self.path, map_location="cpu", weights_only=False)
        self.device = pick_device(device)
        self.model = build_model(ckpt["model_config"]).to(self.device).eval()
        self.model.load_state_dict(ckpt.get("ema") or ckpt["model"])
        self.info = {k: ckpt.get(k) for k in ("dataset", "step", "epoch", "version")}
        self.image_size = self.model.cfg.image_size

    @torch.no_grad()
    def condition(self, prompt: str | None = None, image=None) -> torch.Tensor:
        tokens = images = None
        if prompt and prompt.strip():
            tokens = torch.tensor([tokenize(prompt) or [PAD_ID]], device=self.device)
        if image is not None:
            images = image_tensor(prepare_image(image, self.image_size)).unsqueeze(0).to(self.device)
        if tokens is None and images is None:
            raise ValueError("give a prompt, an image, or both")
        return self.model.encode(tokens, images)

    @torch.no_grad()
    def field(self, cond: torch.Tensor, resolution: int = 64, chunk: int = 65536) -> np.ndarray:
        """Occupancy logits on a resolution^3 grid over [-1, 1]^3, indexed [x, y, z]."""
        c = (torch.arange(resolution, device=self.device) + 0.5) / resolution * 2 - 1
        grid = torch.stack(torch.meshgrid(c, c, c, indexing="ij"), -1).reshape(1, -1, 3)
        out = [self.model.decode(grid[:, i:i + chunk], cond)[0] for i in range(0, grid.shape[1], chunk)]
        return torch.cat(out, 1).reshape(resolution, resolution, resolution).float().cpu().numpy()

    @torch.no_grad()
    def colors(self, cond: torch.Tensor, points: np.ndarray, chunk: int = 65536) -> np.ndarray:
        pts = torch.from_numpy(points.astype(np.float32)).to(self.device).unsqueeze(0)
        out = [self.model.decode(pts[:, i:i + chunk], cond)[1] for i in range(0, pts.shape[1], chunk)]
        return torch.cat(out, 1)[0].float().cpu().numpy()

    @staticmethod
    def mesh_from_field(logits: np.ndarray, smooth: int = 3, min_component: float = 0.02) -> TriMesh:
        """Closed mesh from occupancy logits; drops floating fragments smaller than
        ``min_component`` of the surface area, then smooths."""
        if (logits > 0).sum() == 0:
            raise RuntimeError("the model predicted an empty shape for this input")
        mesh = surface_nets(logits)
        if min_component:
            mesh = remove_small_components(mesh, min_component)
        if smooth:
            mesh = taubin_smooth(mesh, smooth)
        return mesh

    def generate(self, prompt: str | None = None, image=None, resolution: int = 64, smooth: int = 3,
                 size_m: float | None = 1.0, min_component: float = 0.02) -> TriMesh:
        """Return a closed, coloured mesh. ``size_m`` scales the largest side (None keeps model space)."""
        started = time.perf_counter()
        cond = self.condition(prompt, image)
        mesh = self.mesh_from_field(self.field(cond, resolution), smooth, min_component)
        mesh.colors = self.colors(cond, mesh.vertices)
        if size_m:
            lo, hi = mesh.bounds
            mesh.vertices = ((mesh.vertices - (lo + hi) / 2) * (size_m / float((hi - lo).max()))).astype(np.float32)
        mesh.meta = {"prompt": prompt, "image": image is not None, "resolution": resolution,
                     "seconds": round(time.perf_counter() - started, 3)}
        return mesh
