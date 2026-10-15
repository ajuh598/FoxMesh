"""Run FoxMesh on a FoxMesh inference server (``foxmesh.inference.server``).

    FOXMESH_AI_STORE=/models FOXMESH_AI_VERSION=production \
    FOXMESH_INFERENCE_RUNNERS=foxmesh_ai.serving:FoxMeshRunner \
    uvicorn foxmesh.inference.server:app --port 9000

The platform then reaches it through the ``http`` adapter with ``model: "foxmesh"``.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from foxmesh.inference.server import InferenceRequest, ModelRunner  # type: ignore[import-not-found]


class FoxMeshRunner(ModelRunner):
    name = "foxmesh"
    vram_gb = 2.0
    output_formats = ("glb", "obj")

    def load(self) -> None:
        from .generate import Generator
        from .store import ModelStore

        store = ModelStore(os.environ.get("FOXMESH_AI_STORE", "models"))
        self.generator = Generator(store.checkpoint(os.environ.get("FOXMESH_AI_VERSION", "production")))

    def unload(self) -> None:
        self.generator = None
        super().unload()

    def generate(self, request: InferenceRequest, out_path: Path) -> Path:
        from PIL import Image

        from .geometry import write_mesh

        image = Image.open(io.BytesIO(base64.b64decode(request.image_b64))) if request.image_b64 else None
        mesh = self.generator.generate(request.prompt or None, image,
                                       resolution=int(request.params.get("resolution", 64)))
        fmt = request.output_format if request.output_format in self.output_formats else "glb"
        path = out_path.with_suffix(f".{fmt}")
        write_mesh(mesh, path)
        return path
