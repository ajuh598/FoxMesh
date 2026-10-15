"""Automatic result checks. A result is accepted on measured facts, never on a model's self-assessment."""

from __future__ import annotations

import struct
from pathlib import Path

from pydantic import BaseModel, Field

from .blender.worker import PostProcessResult
from .schemas import OutputFormat, Profile, TaskSpec


class Check(BaseModel):
    name: str
    passed: bool
    severity: str = "error"  # "error" fails the result, "warning" only lowers the score
    detail: str = ""


class ValidationResult(BaseModel):
    passed: bool
    score: float = Field(ge=0, le=1)
    checks: list[Check]

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and c.severity == "error"]


def sniff_format(path: Path) -> OutputFormat | None:
    """Identify a 3D file by its content, not its extension."""
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            head = fh.read(512)
    except OSError:
        return None
    if head.startswith(b"glTF"):
        return OutputFormat.GLB
    if head.startswith(b"Kaydara FBX Binary") or head.lstrip().startswith(b"; FBX"):
        return OutputFormat.FBX
    if head.startswith(b"ply"):
        return OutputFormat.PLY
    if head.startswith(b"BLENDER") or head[:4] in (b"\x28\xb5\x2f\xfd", b"\x1f\x8b\x08\x00"):
        return OutputFormat.BLEND
    if len(head) >= 84:
        (count,) = struct.unpack_from("<I", head, 80)
        if size == 84 + 50 * count:
            return OutputFormat.STL
    if head.lstrip().lower().startswith(b"solid"):
        return OutputFormat.STL
    text = head.decode("utf-8", errors="ignore")
    if any(line.startswith(("v ", "o ", "mtllib", "f ")) for line in text.splitlines()):
        return OutputFormat.OBJ
    return None


def validate(task: TaskSpec, post: PostProcessResult) -> ValidationResult:
    checks: list[Check] = []
    stats = post.stats

    checks.append(Check(
        name="postprocess", passed=post.ok,
        detail="; ".join(post.errors) or f"{post.engine} completed",
    ))

    for fmt in task.output_formats:
        path = post.outputs.get(fmt)
        if path is None or not path.exists() or path.stat().st_size == 0:
            checks.append(Check(name=f"output:{fmt.value}", passed=False, detail="file missing or empty"))
            continue
        sniffed = sniff_format(path)
        checks.append(Check(
            name=f"output:{fmt.value}", passed=sniffed is fmt,
            detail=f"{path.stat().st_size} bytes" if sniffed is fmt
            else f"content looks like {sniffed.value if sniffed else 'unknown'}",
        ))

    if stats:
        faces = stats.get("faces", 0)
        checks.append(Check(name="mesh:not_empty", passed=faces > 0, detail=f"{faces} faces"))
        degenerate = stats.get("degenerate_faces", 0)
        if degenerate:
            checks.append(Check(name="mesh:degenerate", passed=False, severity="warning",
                                detail=f"{degenerate} zero-area faces"))
        if task.max_polygons:
            tris = stats.get("triangles", 0)
            checks.append(Check(
                name="mesh:polygon_budget", passed=tris <= task.max_polygons * 1.05,
                detail=f"{tris} triangles (budget {task.max_polygons})",
            ))
        if task.require_watertight:
            nm = stats.get("non_manifold_edges", 0)
            checks.append(Check(
                name="mesh:watertight", passed=bool(stats.get("watertight")),
                detail=f"{nm} non-manifold edges",
            ))
        if task.require_textures:
            checks.append(Check(
                name="material:assigned", passed=stats.get("materials", 0) > 0, severity="warning",
                detail=f"{stats.get('materials', 0)} materials, {stats.get('image_textures', 0)} textures",
            ))
            missing = stats.get("missing_textures", 0)
            if missing:
                checks.append(Check(name="material:missing_textures", passed=False,
                                    detail=f"{missing} texture images missing"))
        if task.profile in (Profile.GAME_ASSET, Profile.HIGH_QUALITY):
            checks.append(Check(name="mesh:uv", passed=bool(stats.get("has_uv")), severity="warning",
                                detail="UV coordinates present" if stats.get("has_uv") else "no UVs"))
        if task.target_size_m and stats.get("dimensions_m"):
            largest = max(stats["dimensions_m"])
            ok = abs(largest - task.target_size_m) <= task.target_size_m * 0.02
            checks.append(Check(name="mesh:size", passed=ok,
                                detail=f"largest dimension {largest:.4g} m (target {task.target_size_m:g} m)"))
    else:
        checks.append(Check(name="mesh:stats", passed=False, severity="warning",
                            detail="no mesh statistics available"))

    if post.thumbnail is not None:
        checks.append(Check(name="thumbnail", passed=post.thumbnail.stat().st_size > 0,
                            severity="warning", detail=post.thumbnail.name))

    weights = {"error": 1.0, "warning": 0.25}
    total = sum(weights[c.severity] for c in checks)
    earned = sum(weights[c.severity] for c in checks if c.passed)
    passed = all(c.passed for c in checks if c.severity == "error")
    return ValidationResult(passed=passed, score=round(earned / total if total else 0.0, 4), checks=checks)
