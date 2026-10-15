"""Run Blender post-processing in a separate process.

Blender runs out of process so a crash, hang or leak in Blender (or in an
imported file) cannot take the API down, and a job can be cancelled by
killing the process.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..schemas import OutputFormat, TaskSpec
from .planner import Step, plan_steps

log = logging.getLogger(__name__)

SCRIPT = Path(__file__).with_name("bpy_pipeline.py")


@dataclass
class PostProcessResult:
    ok: bool
    engine: str  # "blender" or "lite"
    steps: list[str]
    report: dict[str, Any]
    outputs: dict[OutputFormat, Path] = field(default_factory=dict)
    thumbnail: Path | None = None
    duration_s: float = 0.0
    outdir: Path | None = None

    @property
    def stats(self) -> dict[str, Any]:
        return self.report.get("stats") or {}

    @property
    def errors(self) -> list[str]:
        return list(self.report.get("errors", []))

    @property
    def warnings(self) -> list[str]:
        return list(self.report.get("warnings", []))


def output_paths(task: TaskSpec, outdir: Path, stem: str = "model") -> dict[OutputFormat, Path]:
    return {fmt: outdir / f"{stem}.{fmt.value}" for fmt in task.output_formats}


class BlenderWorker:
    def __init__(
        self,
        executable: str,
        mode: str = "blender",
        timeout_s: float = 600.0,
        thumbnails: bool = True,
        thumbnail_size: int = 512,
        thumbnail_samples: int = 16,
    ):
        if mode not in {"blender", "bpy"}:
            raise ValueError("mode must be 'blender' or 'bpy'")
        self.executable = executable
        self.mode = mode
        self.timeout_s = timeout_s
        self.thumbnails = thumbnails
        self.thumbnail_size = thumbnail_size
        self.thumbnail_samples = thumbnail_samples

    def command(self, job_path: Path) -> list[str]:
        if self.mode == "bpy":
            return [self.executable, str(SCRIPT), "--", str(job_path)]
        return [self.executable, "--background", "--factory-startup", "--python-exit-code", "1",
                "--python", str(SCRIPT), "--", str(job_path)]

    def build_job(self, task: TaskSpec, input_path: Path, outdir: Path) -> dict[str, Any]:
        steps = plan_steps(task, thumbnails=self.thumbnails)
        outputs = output_paths(task, outdir)
        return {
            "input_path": str(input_path),
            "steps": [s.value for s in steps],
            "report_path": str(outdir / "blender_report.json"),
            "params": {
                "target_size_m": task.target_size_m,
                "max_polygons": task.max_polygons,
                "exports": [
                    {"format": fmt.value, "path": str(path)}
                    for fmt, path in outputs.items() if fmt is not OutputFormat.BLEND
                ],
                "blend_path": str(outputs[OutputFormat.BLEND]) if OutputFormat.BLEND in outputs else None,
                "thumbnail_path": str(outdir / "thumbnail.png") if Step.THUMBNAIL in steps else None,
                "thumbnail_size": self.thumbnail_size,
                "thumbnail_samples": self.thumbnail_samples,
            },
        }

    async def process(self, task: TaskSpec, input_path: Path, outdir: Path) -> PostProcessResult:
        outdir.mkdir(parents=True, exist_ok=True)
        job = self.build_job(task, input_path, outdir)
        job_path = outdir / "blender_job.json"
        job_path.write_text(json.dumps(job, indent=1))
        report_path = Path(job["report_path"])
        report_path.unlink(missing_ok=True)

        started = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *self.command(job_path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            output, _ = await asyncio.wait_for(proc.communicate(), self.timeout_s)
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            proc.kill()
            await proc.wait()
            if isinstance(exc, asyncio.CancelledError):
                raise
            return PostProcessResult(
                ok=False, engine="blender", steps=job["steps"],
                report={"errors": [f"Blender timed out after {self.timeout_s:g}s"]},
                duration_s=time.monotonic() - started, outdir=outdir,
            )
        duration = time.monotonic() - started
        log_tail = output.decode(errors="replace")[-4000:] if output else ""
        (outdir / "blender.log").write_text(log_tail)

        try:
            report = json.loads(report_path.read_text())
        except (OSError, ValueError):
            report = {"ok": False,
                      "errors": [f"Blender exited with code {proc.returncode} without a report"],
                      "log_tail": log_tail[-1000:]}
        ok = bool(report.get("ok"))
        if ok and proc.returncode != 0:
            # The report is written last, so a complete one means every step finished;
            # a non-zero exit here comes from interpreter teardown.
            report.setdefault("warnings", []).append(f"Blender exited with code {proc.returncode}")
        outputs = {fmt: path for fmt, path in output_paths(task, outdir).items() if path.exists()}
        thumb = outdir / "thumbnail.png"
        return PostProcessResult(
            ok=ok, engine="blender", steps=job["steps"], report=report, outputs=outputs,
            thumbnail=thumb if thumb.exists() else None, duration_s=duration, outdir=outdir,
        )


# ---------------------------------------------------------------------------
# Fallback without Blender
# ---------------------------------------------------------------------------


def obj_stats(path: Path) -> dict[str, Any]:
    """Minimal OBJ statistics, used when Blender is not installed."""
    vertices = faces = triangles = 0
    edges: dict[tuple[int, int], int] = {}
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                vertices += 1
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                idx = [i if i > 0 else vertices + i + 1 for i in idx]
                faces += 1
                triangles += max(len(idx) - 2, 0)
                for a, b in zip(idx, idx[1:] + idx[:1]):
                    key = (a, b) if a < b else (b, a)
                    edges[key] = edges.get(key, 0) + 1
    non_manifold = sum(1 for n in edges.values() if n != 2)
    return {"vertices": vertices, "faces": faces, "triangles": triangles,
            "non_manifold_edges": non_manifold, "watertight": faces > 0 and non_manifold == 0,
            "has_uv": False, "materials": 0, "image_textures": 0, "missing_textures": 0}


class LitePostProcessor:
    """Copies files whose format already matches and reports what it can measure."""

    engine = "lite"

    async def process(self, task: TaskSpec, input_path: Path, outdir: Path) -> PostProcessResult:
        started = time.monotonic()
        outdir.mkdir(parents=True, exist_ok=True)
        source_fmt = input_path.suffix.lstrip(".").lower()
        outputs: dict[OutputFormat, Path] = {}
        errors: list[str] = []
        for fmt, path in output_paths(task, outdir).items():
            if fmt.value == source_fmt:
                shutil.copyfile(input_path, path)
                outputs[fmt] = path
            else:
                errors.append(f"export: cannot produce {fmt.value} from {source_fmt} without Blender")
        stats = obj_stats(input_path) if source_fmt == "obj" else {}
        report = {"ok": not errors, "errors": errors,
                  "warnings": ["Blender not available: post-processing skipped"],
                  "stats": stats}
        return PostProcessResult(
            ok=not errors, engine=self.engine, steps=["copy"], report=report, outputs=outputs,
            duration_s=time.monotonic() - started, outdir=outdir,
        )
