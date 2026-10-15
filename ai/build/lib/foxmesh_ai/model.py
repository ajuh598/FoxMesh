"""The FoxMesh model: conditional occupancy network with a colour head.

    text  ──► TextEncoder ──┐
                            ├──► condition c ──► FiLM-conditioned MLP f(x, c) ──► occupancy logit, RGB
    image ──► ImageEncoder ─┘                                ▲
                                          Fourier features of the query point x

Either input (or both) can be given. A shape is produced by evaluating the
occupancy on a grid and extracting the zero level set (``generate.py``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .text import MAX_TOKENS, PAD_ID, VOCAB_SIZE


@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    max_tokens: int = MAX_TOKENS
    text_dim: int = 256
    text_layers: int = 2
    text_heads: int = 4
    image_channels: int = 4  # RGBA (alpha = object mask)
    image_size: int = 128
    cond_dim: int = 256
    hidden: int = 256
    blocks: int = 5
    fourier_frequencies: int = 8
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def preset(cls, name: str) -> "ModelConfig":
        """``base`` for GPU training; ``small`` trains in under an hour on a 4-core CPU."""
        presets = {
            "base": {},
            "small": {"hidden": 192, "blocks": 4, "image_size": 64},
        }
        if name not in presets:
            raise ValueError(f"unknown preset {name!r} (choose from {', '.join(presets)})")
        return cls(**presets[name])


class TextEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.text_dim, padding_idx=PAD_ID)
        self.pos = nn.Parameter(torch.zeros(cfg.max_tokens, cfg.text_dim))
        layer = nn.TransformerEncoderLayer(cfg.text_dim, cfg.text_heads, cfg.text_dim * 4,
                                           dropout=cfg.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.text_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.text_dim)
        self.proj = nn.Linear(cfg.text_dim, cfg.cond_dim)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pad = tokens == PAD_ID
        x = self.embed(tokens) + self.pos[: tokens.shape[1]]
        # A row of only padding would make attention produce NaNs; unmask its first slot.
        pad = pad.clone()
        pad[pad.all(dim=1), 0] = False
        x = self.norm(self.encoder(x, src_key_padding_mask=pad))
        keep = (~pad).unsqueeze(-1).float()
        pooled = (x * keep).sum(1) / keep.sum(1).clamp(min=1)
        return self.proj(pooled)


class ImageEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        chans = [cfg.image_channels, 32, 64, 128, 256]
        layers = []
        for cin, cout in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
                       nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU()]
        self.net = nn.Sequential(*layers)
        self.proj = nn.Linear(chans[-1] * 2, cfg.cond_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.net(images)
        pooled = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
        return self.proj(pooled)


class FourierFeatures(nn.Module):
    def __init__(self, frequencies: int):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(frequencies)) * math.pi, persistent=False)

    @property
    def out_dim(self) -> int:
        return 3 + 3 * 2 * len(self.freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = x.unsqueeze(-1) * self.freqs  # (..., 3, F)
        return torch.cat([x, angles.sin().flatten(-2), angles.cos().flatten(-2)], dim=-1)


class FiLMBlock(nn.Module):
    def __init__(self, hidden: int, cond_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.film = nn.Linear(cond_dim, hidden * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(cond).unsqueeze(1).chunk(2, dim=-1)
        x = self.fc1(F.silu(h))
        x = x * (1 + gamma) + beta
        return h + self.fc2(F.silu(x))


class FoxMeshModel(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg
        self.text = TextEncoder(c)
        self.image = ImageEncoder(c)
        self.null_cond = nn.Parameter(torch.zeros(c.cond_dim))
        self.cond_norm = nn.LayerNorm(c.cond_dim)
        self.fourier = FourierFeatures(c.fourier_frequencies)
        self.inp = nn.Linear(self.fourier.out_dim + c.cond_dim, c.hidden)
        self.blocks = nn.ModuleList(FiLMBlock(c.hidden, c.cond_dim) for _ in range(c.blocks))
        self.occ_head = nn.Linear(c.hidden, 1)
        self.color_head = nn.Sequential(nn.Linear(c.hidden + c.cond_dim, c.hidden // 2), nn.SiLU(),
                                        nn.Linear(c.hidden // 2, 3))

    def encode(self, tokens: torch.Tensor | None = None, images: torch.Tensor | None = None,
               text_mask: torch.Tensor | None = None, image_mask: torch.Tensor | None = None
               ) -> torch.Tensor:
        """Condition vector from text and/or image. Masks select per-sample modalities."""
        batch = (tokens if tokens is not None else images).shape[0]
        device = self.null_cond.device
        total = torch.zeros(batch, self.cfg.cond_dim, device=device)
        count = torch.zeros(batch, 1, device=device)
        if tokens is not None:
            tm = (text_mask if text_mask is not None else torch.ones(batch, device=device)).float().unsqueeze(1)
            total = total + self.text(tokens) * tm
            count = count + tm
        if images is not None:
            im = (image_mask if image_mask is not None else torch.ones(batch, device=device)).float().unsqueeze(1)
            total = total + self.image(images) * im
            count = count + im
        cond = torch.where(count > 0, total / count.clamp(min=1), self.null_cond.expand(batch, -1))
        return self.cond_norm(cond)

    def decode(self, points: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """points (B, N, 3) in [-1, 1]; returns occupancy logits (B, N) and colours (B, N, 3)."""
        n = points.shape[1]
        cond_rep = cond.unsqueeze(1).expand(-1, n, -1)
        h = self.inp(torch.cat([self.fourier(points), cond_rep], dim=-1))
        for block in self.blocks:
            h = block(h, cond)
        occ = self.occ_head(F.silu(h)).squeeze(-1)
        color = torch.sigmoid(self.color_head(torch.cat([F.silu(h), cond_rep], dim=-1)))
        return occ, color

    def forward(self, points, tokens=None, images=None, text_mask=None, image_mask=None):
        return self.decode(points, self.encode(tokens, images, text_mask, image_mask))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg: dict | ModelConfig | None = None) -> FoxMeshModel:
    if isinstance(cfg, dict):
        cfg = ModelConfig(**cfg)
    return FoxMeshModel(cfg)
