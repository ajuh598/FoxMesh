"""Small in-memory mesh with OBJ and GLB writers, shared by the built-in generators."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

Vec = tuple[float, float, float]


@dataclass
class Mesh:
    """Polygon mesh built Z-up (Blender convention); writers convert to Y-up files."""

    vertices: list[Vec] = field(default_factory=list)
    faces: list[tuple[int, ...]] = field(default_factory=list)  # 0-based, CCW from outside
    parts: list[str] = field(default_factory=list)
    # Optional per-vertex texture coordinates, glTF convention: (0, 0) is the image's top-left.
    uvs: list[tuple[float, float]] | None = None
    texture_png: bytes | None = None
    # Write smooth per-vertex normals instead of letting loaders shade flat.
    smooth: bool = False

    def add(self, other: "Mesh", name: str) -> "Mesh":
        offset = len(self.vertices)
        self.vertices.extend(other.vertices)
        self.faces.extend(tuple(i + offset for i in f) for f in other.faces)
        self.parts.append(name)
        return self

    def triangles(self) -> list[int]:
        return [i for face in self.faces for k in range(1, len(face) - 1)
                for i in (face[0], face[k], face[k + 1])]

    def vertex_normals(self, positions: list[Vec]) -> list[Vec]:
        acc = [[0.0, 0.0, 0.0] for _ in positions]
        for face in self.faces:
            for k in range(1, len(face) - 1):
                a, b, c = (positions[i] for i in (face[0], face[k], face[k + 1]))
                u = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
                v = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
                # Area-weighted face normal.
                n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
                for i in (face[0], face[k], face[k + 1]):
                    acc[i][0] += n[0]
                    acc[i][1] += n[1]
                    acc[i][2] += n[2]
        normals = []
        for x, y, z in acc:
            length = math.sqrt(x * x + y * y + z * z) or 1.0
            normals.append((x / length, y / length, z / length))
        return normals

    def write_obj(self, path: Path) -> None:
        # Built Z-up; OBJ convention is Y-up (-Z forward), so rotate -90 degrees about X.
        lines = ["# FoxMesh generator"]
        textured = self.uvs is not None and self.texture_png is not None
        if textured:
            mtl, png = path.with_suffix(".mtl"), path.with_suffix(".png")
            png.write_bytes(self.texture_png)
            mtl.write_text(f"newmtl FoxMesh_Texture\nKd 1 1 1\nmap_Kd {png.name}\n")
            lines += [f"mtllib {mtl.name}", "usemtl FoxMesh_Texture"]
        lines += [f"v {x:.6f} {z:.6f} {-y:.6f}" for x, y, z in self.vertices]
        if textured:
            # OBJ texture space starts at the bottom-left.
            lines += [f"vt {u:.6f} {1 - v:.6f}" for u, v in self.uvs]
            lines += ["f " + " ".join(f"{i + 1}/{i + 1}" for i in f) for f in self.faces]
        else:
            lines += ["f " + " ".join(str(i + 1) for i in f) for f in self.faces]
        path.write_text("\n".join(lines) + "\n")

    def write_glb(self, path: Path) -> None:
        """Minimal binary glTF: indexed triangles, one PBR material, optional texture.

        Vertices stay shared so the mesh keeps its connectivity (watertightness,
        UV islands). Without ``smooth`` normals are omitted and loaders shade flat.
        """
        # glTF is Y-up like OBJ: same -90 degree rotation about X.
        verts = [(x, z, -y) for x, y, z in self.vertices]
        positions = [c for v in verts for c in v]
        indices = self.triangles()
        if self.uvs is not None and len(self.uvs) != len(verts):
            raise ValueError("uvs must have one entry per vertex")

        chunks: list[bytes] = []
        views: list[dict] = []
        accessors: list[dict] = []
        attributes: dict[str, int] = {}

        def add_view(data: bytes, target: int | None = None) -> int:
            offset = sum(len(c) for c in chunks)
            view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
            if target:
                view["target"] = target
            chunks.append(data + b"\0" * (-len(data) % 4))
            views.append(view)
            return len(views) - 1

        def add_accessor(values: list[float], kind: str, ctype: int = 5126, **extra) -> int:
            fmt = "f" if ctype == 5126 else "I"
            target = 34963 if kind == "SCALAR" else 34962
            view = add_view(struct.pack(f"<{len(values)}{fmt}", *values), target)
            width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[kind]
            accessors.append({"bufferView": view, "componentType": ctype,
                              "count": len(values) // width, "type": kind, **extra})
            return len(accessors) - 1

        attributes["POSITION"] = add_accessor(
            positions, "VEC3",
            min=[min(positions[k::3]) for k in range(3)],
            max=[max(positions[k::3]) for k in range(3)],
        )
        if self.smooth:
            attributes["NORMAL"] = add_accessor(
                [c for n in self.vertex_normals(verts) for c in n], "VEC3")
        material: dict = {"name": "FoxMesh_Default", "pbrMetallicRoughness": {
            "baseColorFactor": [0.62, 0.62, 0.65, 1.0], "metallicFactor": 0.0,
            "roughnessFactor": 0.55}}
        extra: dict = {}
        if self.uvs is not None and self.texture_png is not None:
            attributes["TEXCOORD_0"] = add_accessor([c for uv in self.uvs for c in uv], "VEC2")
            image_view = add_view(self.texture_png)
            extra = {
                "images": [{"bufferView": image_view, "mimeType": "image/png"}],
                "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
                "textures": [{"source": 0, "sampler": 0}],
            }
            material = {"name": "FoxMesh_Texture", "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 0.7}}
        index_accessor = add_accessor(indices, "SCALAR", ctype=5125)

        bin_chunk = b"".join(chunks)
        gltf = {
            "asset": {"version": "2.0", "generator": "FoxMesh"},
            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": "model"}],
            "meshes": [{"primitives": [{"attributes": attributes, "indices": index_accessor,
                                        "material": 0, "mode": 4}]}],
            "materials": [material],
            "buffers": [{"byteLength": len(bin_chunk)}],
            "bufferViews": views,
            "accessors": accessors,
            **extra,
        }
        json_chunk = json.dumps(gltf, separators=(",", ":")).encode()
        json_chunk += b" " * (-len(json_chunk) % 4)
        total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
        with path.open("wb") as fh:
            fh.write(struct.pack("<4sII", b"glTF", 2, total))
            fh.write(struct.pack("<I4s", len(json_chunk), b"JSON") + json_chunk)
            fh.write(struct.pack("<I4s", len(bin_chunk), b"BIN\0") + bin_chunk)
