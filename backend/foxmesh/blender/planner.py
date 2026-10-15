"""Select only the Blender post-processing steps a task actually needs."""

from __future__ import annotations

from enum import Enum

from ..schemas import OutputFormat, Profile, TaskSpec, TaskType


class Step(str, Enum):
    IMPORT = "import"
    INSPECT = "inspect"
    NORMALIZE = "normalize"  # scale, origin, ground placement
    FIX_NORMALS = "fix_normals"
    CHECK_MANIFOLD = "check_manifold"  # connectivity / closedness for printing
    UV = "uv"  # check UVs, unwrap if missing
    MATERIALS = "materials"  # link textures, ensure a material
    DECIMATE = "decimate"  # polygon budget
    THUMBNAIL = "thumbnail"
    SAVE_BLEND = "save_blend"
    EXPORT = "export"


# Execution order inside the Blender process.
ORDER = list(Step)

PROFILE_STEPS: dict[Profile, set[Step]] = {
    Profile.PROTOTYPE: {Step.NORMALIZE},
    Profile.BALANCED: {Step.NORMALIZE, Step.FIX_NORMALS},
    Profile.HIGH_QUALITY: {Step.NORMALIZE, Step.FIX_NORMALS, Step.UV, Step.MATERIALS},
    Profile.GAME_ASSET: {Step.NORMALIZE, Step.FIX_NORMALS, Step.UV, Step.MATERIALS},
    Profile.PRINT_3D: {Step.NORMALIZE, Step.FIX_NORMALS, Step.CHECK_MANIFOLD},
}

TASK_STEPS: dict[TaskType, set[Step]] = {
    TaskType.PREVIEW: {Step.THUMBNAIL},
    TaskType.CONVERT: set(),
    TaskType.TEXTURE: {Step.UV, Step.MATERIALS},
    TaskType.EDIT_3D: {Step.FIX_NORMALS},
}


def plan_steps(task: TaskSpec, thumbnails: bool = True) -> list[Step]:
    """Return the ordered steps for ``task``; import, inspect and export always run.

    Inspection is cheap and its report drives validation, so it is never skipped.
    """
    steps = {Step.IMPORT, Step.INSPECT}
    if task.task_type in TASK_STEPS:
        steps |= TASK_STEPS[task.task_type]
        if task.task_type is TaskType.EDIT_3D:
            steps |= PROFILE_STEPS[task.profile]
    else:
        steps |= PROFILE_STEPS[task.profile]

    if task.max_polygons:
        steps.add(Step.DECIMATE)
    if task.target_size_m:
        steps.add(Step.NORMALIZE)
    if task.require_watertight:
        steps.add(Step.CHECK_MANIFOLD)
    if OutputFormat.BLEND in task.output_formats:
        steps.add(Step.SAVE_BLEND)
    if any(f is not OutputFormat.BLEND for f in task.output_formats):
        steps.add(Step.EXPORT)
    if thumbnails and task.task_type is not TaskType.CONVERT:
        steps.add(Step.THUMBNAIL)
    return [s for s in ORDER if s in steps]
