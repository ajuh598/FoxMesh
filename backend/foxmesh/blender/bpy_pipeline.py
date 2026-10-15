"""Blender-side post-processing script. Runs *inside* Blender, never imported by FoxMesh.

    blender --background --factory-startup --python bpy_pipeline.py -- job.json
    python bpy_pipeline.py -- job.json          # with the ``bpy`` module installed

``job.json``::

    {"input_path": str, "steps": [str, ...], "report_path": str,
     "params": {"target_size_m": float | null, "max_polygons": int | null,
                "exports": [{"format": "glb", "path": str}, ...],
                "blend_path": str | null, "thumbnail_path": str | null,
                "thumbnail_size": int, "thumbnail_samples": int}}

Writes a JSON report with per-step results and mesh statistics, and exits
non-zero when a critical step (import or export) fails.
"""

import json
import math
import os
import sys
import time
import traceback

import bpy  # must come first: the pip ``bpy`` module provides bmesh/mathutils
import bmesh  # noqa: E402
from mathutils import Matrix, Vector

CRITICAL_STEPS = {"import", "export", "save_blend"}


class StepWarning(Exception):
    """Non-fatal problem that should be reported as a warning."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mesh_objects():
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def world_bbox(objects):
    lo = Vector((math.inf,) * 3)
    hi = Vector((-math.inf,) * 3)
    for obj in objects:
        # Read vertices rather than ``bound_box``: the latter is stale until the
        # depsgraph re-evaluates, which it does not do between our edits.
        world = obj.matrix_world
        for vert in obj.data.vertices:
            p = world @ vert.co
            lo = Vector(map(min, lo, p))
            hi = Vector(map(max, hi, p))
    if not objects or lo.x == math.inf:
        return Vector((0, 0, 0)), Vector((0, 0, 0))
    return lo, hi


def select_only(objects):
    for obj in bpy.context.scene.objects:
        obj.select_set(False)
    for obj in objects:
        obj.select_set(True)
    if objects:
        bpy.context.view_layer.objects.active = objects[0]


def count_components(bm):
    """Number of connected vertex islands (loose parts)."""
    parent = list(range(len(bm.verts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for edge in bm.edges:
        a, b = find(edge.verts[0].index), find(edge.verts[1].index)
        if a != b:
            parent[a] = b
    return len({find(i) for i in range(len(bm.verts))})


def mesh_stats(objects):
    stats = {
        "objects": len(objects), "vertices": 0, "faces": 0, "triangles": 0,
        "non_manifold_edges": 0, "boundary_edges": 0, "loose_parts": 0,
        "degenerate_faces": 0, "materials": 0, "image_textures": 0,
        "missing_textures": 0, "has_uv": bool(objects), "has_vertex_colors": False,
        "volume_m3": 0.0,
    }
    materials = set()
    for obj in objects:
        mesh = obj.data
        stats["has_uv"] = stats["has_uv"] and len(mesh.uv_layers) > 0
        stats["has_vertex_colors"] |= len(getattr(mesh, "color_attributes", [])) > 0
        for slot in obj.material_slots:
            if slot.material:
                materials.add(slot.material)
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.transform(obj.matrix_world)
        bm.verts.index_update()
        stats["vertices"] += len(bm.verts)
        stats["faces"] += len(bm.faces)
        stats["triangles"] += sum(len(f.verts) - 2 for f in bm.faces)
        stats["boundary_edges"] += sum(1 for e in bm.edges if e.is_boundary)
        stats["non_manifold_edges"] += sum(1 for e in bm.edges if not e.is_manifold)
        stats["degenerate_faces"] += sum(1 for f in bm.faces if f.calc_area() < 1e-12)
        stats["loose_parts"] += count_components(bm) if bm.verts else 0
        try:
            stats["volume_m3"] += bm.calc_volume(signed=True)
        except Exception:
            pass
        bm.free()
    for mat in materials:
        if not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE":
                if node.image is None:
                    stats["missing_textures"] += 1
                    continue
                stats["image_textures"] += 1
                if not node.image.packed_file and node.image.filepath:
                    if not os.path.exists(bpy.path.abspath(node.image.filepath)):
                        stats["missing_textures"] += 1
    stats["materials"] = len(materials)
    stats["watertight"] = bool(objects) and stats["non_manifold_edges"] == 0
    lo, hi = world_bbox(objects)
    stats["dimensions_m"] = [round(v, 6) for v in (hi - lo)]
    stats["bbox_min"] = [round(v, 6) for v in lo]
    stats["volume_m3"] = round(stats["volume_m3"], 9)
    return stats


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def step_import(job, params):
    path = job["input_path"]
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=path)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=path)
    elif ext == ".blend":
        bpy.ops.wm.open_mainfile(filepath=path)
    else:
        raise ValueError(f"unsupported input format {ext}")
    objects = mesh_objects()
    if not objects:
        raise ValueError("file contains no mesh objects")
    return {"format": ext.lstrip("."), "mesh_objects": len(objects)}


def step_inspect(job, params):
    return mesh_stats(mesh_objects())


def _bake_transforms(objects):
    """Apply world transforms into mesh data and detach from parents."""
    for obj in objects:
        if obj.data.users > 1:
            obj.data = obj.data.copy()
        world = obj.matrix_world.copy()
        obj.parent = None
        obj.data.transform(world)
        obj.matrix_world = Matrix.Identity(4)
    # Drop empties left over from importer hierarchies.
    for obj in list(bpy.context.scene.objects):
        if obj.type == "EMPTY" and not obj.children:
            bpy.data.objects.remove(obj, do_unlink=True)


def step_normalize(job, params):
    objects = mesh_objects()
    _bake_transforms(objects)
    lo, hi = world_bbox(objects)
    size = hi - lo
    largest = max(size)
    info = {"original_dimensions_m": [round(v, 6) for v in size]}
    scale = 1.0
    target = params.get("target_size_m")
    if target and largest > 0:
        scale = target / largest
    # Centre on X/Y and stand the object on the ground plane (Z = 0).
    centre = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
    transform = Matrix.Scale(scale, 4) @ Matrix.Translation(-centre)
    for obj in objects:
        obj.data.transform(transform)
        obj.data.update()
    lo, hi = world_bbox(objects)
    info.update({"scale": round(scale, 6), "dimensions_m": [round(v, 6) for v in hi - lo]})
    return info


def step_fix_normals(job, params):
    merge = "check_manifold" in job["steps"]
    merged = removed = 0
    for obj in mesh_objects():
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        before = len(bm.verts)
        if merge:
            # Welding is only safe when shading seams do not matter (printing).
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
            merged += before - len(bm.verts)
        bmesh.ops.dissolve_degenerate(bm, dist=1e-9, edges=bm.edges)
        loose = [v for v in bm.verts if not v.link_edges]
        removed += len(loose)
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update()
    return {"merged_vertices": merged, "removed_loose_vertices": removed}


def step_check_manifold(job, params):
    objects = mesh_objects()
    before = mesh_stats(objects)["non_manifold_edges"]
    filled = 0
    if before:
        for obj in objects:
            bm = bmesh.new()
            bm.from_mesh(obj.data)
            boundary = [e for e in bm.edges if e.is_boundary]
            if boundary:
                result = bmesh.ops.holes_fill(bm, edges=boundary, sides=0)
                filled += len(result.get("faces", []))
                bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(obj.data)
            bm.free()
            obj.data.update()
    after = mesh_stats(objects)
    info = {
        "non_manifold_before": before, "non_manifold_after": after["non_manifold_edges"],
        "holes_filled": filled, "watertight": after["watertight"],
        "loose_parts": after["loose_parts"],
    }
    if not after["watertight"]:
        raise StepWarning(f"mesh is not watertight ({after['non_manifold_edges']} non-manifold edges)")
    return info


def step_uv(job, params):
    unwrapped = []
    for obj in mesh_objects():
        if len(obj.data.uv_layers) > 0:
            continue
        select_only([obj])
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.02)
        bpy.ops.object.mode_set(mode="OBJECT")
        unwrapped.append(obj.name)
    return {"unwrapped": unwrapped}


def step_materials(job, params):
    created = 0
    vertex_color_linked = 0
    default = None
    for obj in mesh_objects():
        has_material = any(slot.material for slot in obj.material_slots)
        if has_material:
            continue
        colors = getattr(obj.data, "color_attributes", None)
        if colors and len(colors):
            # Vertex-coloured output (common for reconstruction models): wire it to base colour.
            mat = bpy.data.materials.new(f"{obj.name}_VertexColor")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            bsdf = nodes.get("Principled BSDF")
            attr = nodes.new("ShaderNodeVertexColor")
            attr.layer_name = colors[0].name
            mat.node_tree.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
            vertex_color_linked += 1
        else:
            if default is None:
                default = bpy.data.materials.new("FoxMesh_Default")
                default.use_nodes = True
                bsdf = default.node_tree.nodes.get("Principled BSDF")
                bsdf.inputs["Base Color"].default_value = (0.62, 0.62, 0.65, 1.0)
                bsdf.inputs["Roughness"].default_value = 0.55
            mat = default
        obj.data.materials.append(mat)
        created += 1
    stats = mesh_stats(mesh_objects())
    info = {
        "assigned": created, "vertex_color_materials": vertex_color_linked,
        "image_textures": stats["image_textures"], "missing_textures": stats["missing_textures"],
    }
    if stats["missing_textures"]:
        raise StepWarning(f"{stats['missing_textures']} texture image(s) are missing")
    return info


def step_decimate(job, params):
    budget = params.get("max_polygons")
    objects = mesh_objects()
    before = mesh_stats(objects)["triangles"]
    if not budget or before <= budget:
        return {"triangles_before": before, "triangles_after": before, "applied": False}
    ratio = max(budget / before * 0.98, 0.001)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in objects:
        mod = obj.modifiers.new("FoxMesh_Decimate", "DECIMATE")
        mod.decimate_type = "COLLAPSE"
        mod.ratio = ratio
        mod.use_collapse_triangulate = True
        depsgraph.update()
        evaluated = obj.evaluated_get(depsgraph)
        new_mesh = bpy.data.meshes.new_from_object(evaluated)
        old = obj.data
        obj.modifiers.remove(mod)
        obj.data = new_mesh
        if old.users == 0:
            bpy.data.meshes.remove(old)
    after = mesh_stats(objects)["triangles"]
    return {"triangles_before": before, "triangles_after": after, "ratio": round(ratio, 4),
            "applied": True}


def step_thumbnail(job, params):
    path = params.get("thumbnail_path")
    if not path:
        return {"skipped": True}
    objects = mesh_objects()
    lo, hi = world_bbox(objects)
    centre = (lo + hi) / 2
    radius = max((hi - lo).length / 2, 1e-3)
    scene = bpy.context.scene

    cam_data = bpy.data.cameras.new("FoxMesh_Camera")
    cam_data.lens = 50
    cam = bpy.data.objects.new("FoxMesh_Camera", cam_data)
    scene.collection.objects.link(cam)
    fov = 2 * math.atan(cam_data.sensor_width / (2 * cam_data.lens))
    direction = Vector((1.0, -1.2, 0.8)).normalized()
    cam.location = centre + direction * (radius / math.sin(fov / 2)) * 1.05
    cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam_data.clip_end = max(1000.0, radius * 20)
    scene.camera = cam

    for name, energy, offset in (("Key", 4.0, (1, -1, 2)), ("Fill", 1.5, (-2, -0.5, 1))):
        light_data = bpy.data.lights.new(f"FoxMesh_{name}", "SUN")
        light_data.energy = energy
        light = bpy.data.objects.new(f"FoxMesh_{name}", light_data)
        scene.collection.objects.link(light)
        light.rotation_euler = (-Vector(offset)).to_track_quat("-Z", "Y").to_euler()

    if scene.world is None:
        scene.world = bpy.data.worlds.new("FoxMesh_World")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    if background:
        background.inputs["Color"].default_value = (0.8, 0.82, 0.85, 1.0)
        background.inputs["Strength"].default_value = 0.6

    size = int(params.get("thumbnail_size", 512))
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = int(params.get("thumbnail_samples", 16))
    scene.cycles.use_denoising = False
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return {"path": path, "size": size}


def step_save_blend(job, params):
    path = params.get("blend_path")
    if not path:
        return {"skipped": True}
    bpy.ops.wm.save_as_mainfile(filepath=path, compress=True, copy=True)
    return {"path": path}


def step_export(job, params):
    objects = mesh_objects()
    select_only(objects)
    written = {}
    for item in params.get("exports", []):
        fmt, path = item["format"], item["path"]
        if fmt == "glb":
            bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True,
                                      export_apply=True)
        elif fmt == "obj":
            bpy.ops.wm.obj_export(filepath=path, export_selected_objects=True,
                                  export_materials=True, path_mode="COPY")
        elif fmt == "fbx":
            bpy.ops.export_scene.fbx(filepath=path, use_selection=True, path_mode="COPY",
                                     embed_textures=True)
        elif fmt == "stl":
            # STL has no units and slicers assume millimetres; Blender works in metres.
            bpy.ops.wm.stl_export(filepath=path, export_selected_objects=True,
                                  global_scale=1000.0, ascii_format=False)
        elif fmt == "ply":
            bpy.ops.wm.ply_export(filepath=path, export_selected_objects=True)
        else:
            raise ValueError(f"unsupported export format {fmt}")
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise RuntimeError(f"export to {fmt} produced no file")
        written[fmt] = path
    return {"written": written}


STEPS = {
    "import": step_import,
    "inspect": step_inspect,
    "normalize": step_normalize,
    "fix_normals": step_fix_normals,
    "check_manifold": step_check_manifold,
    "uv": step_uv,
    "materials": step_materials,
    "decimate": step_decimate,
    "thumbnail": step_thumbnail,
    "save_blend": step_save_blend,
    "export": step_export,
}


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    with open(argv[0], encoding="utf-8") as fh:
        job = json.load(fh)
    params = job.get("params", {})
    report = {"ok": True, "steps": {}, "errors": [], "warnings": [],
              "blender_version": bpy.app.version_string}
    started = time.perf_counter()
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        for name in job["steps"]:
            t0 = time.perf_counter()
            entry = {"ok": True}
            try:
                entry["info"] = STEPS[name](job, params)
            except StepWarning as exc:
                entry.update(ok=True, warning=str(exc))
                report["warnings"].append(f"{name}: {exc}")
            except Exception as exc:
                entry.update(ok=False, error=f"{type(exc).__name__}: {exc}")
                report["errors"].append(f"{name}: {exc}")
                if name in CRITICAL_STEPS:
                    report["ok"] = False
                    entry["traceback"] = traceback.format_exc()
            entry["ms"] = round((time.perf_counter() - t0) * 1000, 1)
            report["steps"][name] = entry
            if not report["ok"]:
                break
        if report["ok"]:
            report["stats"] = mesh_stats(mesh_objects())
    except Exception as exc:
        report["ok"] = False
        report["errors"].append(f"fatal: {exc}")
        report["traceback"] = traceback.format_exc()
    finally:
        report["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
        with open(job["report_path"], "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
    sys.stdout.flush()
    sys.stderr.flush()
    # Skip interpreter teardown: the pip ``bpy`` module can crash while unloading,
    # which would turn a finished job into a failed one.
    os._exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
