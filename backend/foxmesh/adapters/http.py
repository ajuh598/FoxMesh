"""Adapter for models served by a FoxMesh inference server (see ``foxmesh.inference``).

Contract: ``POST {endpoint}/generate`` with JSON

    {"model": str, "task_type": str, "prompt": str, "image_b64": str | null,
     "output_format": str, "params": {...}}

returning the 3D file as the response body, with the format in the
``X-FoxMesh-Format`` header (falls back to the first declared model output).
"""

from __future__ import annotations

import base64
from pathlib import Path

import httpx

from ..schemas import ModelSpec, OutputFormat, TaskSpec
from .base import GenerationAdapter, GenerationResult, TransientError


class HttpInferenceAdapter(GenerationAdapter):
    key = "http"

    def __init__(self, timeout_s: float = 900.0, transport: httpx.AsyncBaseTransport | None = None):
        self._timeout = timeout_s
        self._transport = transport

    def unavailable_reason(self, model: ModelSpec) -> str | None:
        if not model.config.get("endpoint"):
            return "no inference endpoint configured (config.endpoint)"
        return None

    def _pick_format(self, task: TaskSpec, model: ModelSpec) -> OutputFormat:
        # Ask for a requested format the model emits natively, else its first output.
        for fmt in task.output_formats:
            if fmt in model.outputs:
                return fmt
        return model.outputs[0]

    async def generate(self, task: TaskSpec, model: ModelSpec, workdir: Path) -> GenerationResult:
        endpoint = str(model.config["endpoint"]).rstrip("/")
        fmt = self._pick_format(task, model)
        params = {k: v for k, v in model.config.items() if k not in {"endpoint", "model", "api_key"}}
        if task.max_polygons:
            params.setdefault("max_faces", task.max_polygons)
        payload = {
            "model": model.config.get("model", model.id),
            "task_type": task.task_type.value,
            "prompt": task.prompt,
            "image_b64": (
                base64.b64encode(Path(task.image_path).read_bytes()).decode()
                if task.image_path else None
            ),
            "output_format": fmt.value,
            "params": params,
        }
        headers = {}
        if key := model.config.get("api_key"):
            headers["Authorization"] = f"Bearer {key}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                response = await client.post(f"{endpoint}/generate", json=payload, headers=headers)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise TransientError(f"inference server unreachable: {exc}") from exc
        if response.status_code in (429, 502, 503, 504):
            raise TransientError(f"inference server busy ({response.status_code})")
        if response.status_code >= 400:
            raise RuntimeError(f"inference failed ({response.status_code}): {response.text[:300]}")
        try:
            out_fmt = OutputFormat(response.headers.get("x-foxmesh-format", fmt.value))
        except ValueError:
            out_fmt = fmt
        path = workdir / f"generated.{out_fmt.value}"
        path.write_bytes(response.content)
        return GenerationResult(
            path=path, format=out_fmt,
            metadata={"endpoint": endpoint, "bytes": len(response.content)},
        )
