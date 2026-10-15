"""Rule-based request analysis.

Turns a free-form request (Korean or English) into a :class:`TaskSpec` using
keyword tables and regular expressions. No language model is involved, so the
step costs microseconds and the result is deterministic and explainable: every
rule that fired is listed in ``TaskSpec.matched_rules``.
"""

from __future__ import annotations

import re
from functools import lru_cache

from .schemas import JobRequest, Modality, OutputFormat, Profile, TaskSpec, TaskType

# Profile keywords, checked in priority order (first match wins).
PROFILE_RULES: list[tuple[Profile, tuple[str, ...]]] = [
    (
        Profile.PRINT_3D,
        ("3d 프린팅", "3d프린팅", "3d 프린트", "3d프린트", "프린팅", "프린터", "출력용",
         "3d print", "3d-print", "3dprint", "printable", "printing", "stl"),
    ),
    (
        Profile.GAME_ASSET,
        ("게임", "로우폴리", "로우 폴리", "game", "unity", "unreal", "godot", "low poly",
         "low-poly", "lowpoly", "real-time", "realtime", "실시간"),
    ),
    (
        Profile.HIGH_QUALITY,
        ("고품질", "고화질", "고해상도", "디테일", "사실적", "정교", "high quality", "high-quality",
         "detailed", "realistic", "photoreal", "hero asset", "cinematic"),
    ),
    (
        Profile.PROTOTYPE,
        ("빠르게", "빠른", "시제품", "프로토타입", "초안", "대충", "간단히", "quick", "fast",
         "prototype", "draft", "rough", "mockup", "placeholder"),
    ),
]

# Latin-only boundaries: Korean particles attach directly ("obj로"), and Python's
# \b treats Hangul as word characters, so \b would miss those.
_L, _R = r"(?<![a-z0-9])", r"(?![a-z0-9])"
FORMAT_RULES: list[tuple[OutputFormat, re.Pattern[str]]] = [
    (OutputFormat.GLB, re.compile(rf"{_L}(glb|gltf){_R}")),
    (OutputFormat.OBJ, re.compile(rf"{_L}obj{_R}")),
    (OutputFormat.FBX, re.compile(rf"{_L}fbx{_R}")),
    (OutputFormat.STL, re.compile(rf"{_L}stl{_R}")),
    (OutputFormat.PLY, re.compile(rf"{_L}ply{_R}")),
    (OutputFormat.BLEND, re.compile(rf"{_L}\.?blend{_R}|블렌드 ?파일|블렌더 ?(파일|프로젝트)")),
]

CONVERT_WORDS = ("변환", "내보내", "convert", "export", "바꿔")
PREVIEW_WORDS = ("렌더", "미리보기", "썸네일", "render", "preview", "thumbnail")
TEXTURE_WORDS = ("텍스처", "텍스쳐", "재질", "머티리얼", "texture", "material", "pbr", "uv")
EDIT_WORDS = ("최적화", "수정", "고쳐", "정리", "줄여", "optimize", "optimise", "repair", "fix",
              "decimate", "reduce", "clean")

LOW_POLY_DEFAULT = 5_000
GAME_POLY_DEFAULT = 20_000

_POLY_RE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(k|천|만)?\s*(?:개\s*)?(?:의\s*)?"
    r"(폴리곤|폴리|삼각형|면|polys?|polygons?|tris|triangles|faces)",
)
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mm|cm|m|밀리미터|밀리|센티미터|센티|미터)(?![a-z])")
_SIZE_UNITS = {
    "mm": 0.001, "밀리미터": 0.001, "밀리": 0.001,
    "cm": 0.01, "센티미터": 0.01, "센티": 0.01,
    "m": 1.0, "미터": 1.0,
}


@lru_cache(maxsize=None)
def _word_pattern(word: str) -> re.Pattern[str]:
    # Latin keywords must match whole words ("fast" must not match "breakfast");
    # Korean keywords match as substrings since particles attach directly to them.
    if word.isascii():
        return re.compile(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])")
    return re.compile(re.escape(word))


def _contains_any(text: str, words: tuple[str, ...]) -> str | None:
    for word in words:
        if _word_pattern(word).search(text):
            return word
    return None


def _parse_polygons(text: str) -> int | None:
    match = _POLY_RE.search(text)
    if not match:
        return None
    value = float(match.group(1).replace(",", ""))
    multiplier = {"k": 1_000, "천": 1_000, "만": 10_000}.get(match.group(2) or "", 1)
    count = int(value * multiplier)
    return count if count > 0 else None


def _parse_size(text: str) -> float | None:
    match = _SIZE_RE.search(text)
    if not match:
        return None
    return float(match.group(1)) * _SIZE_UNITS[match.group(2)]


def _default_formats(profile: Profile) -> list[OutputFormat]:
    if profile is Profile.PRINT_3D:
        # STL for the printer, GLB for the web preview.
        return [OutputFormat.STL, OutputFormat.GLB]
    if profile is Profile.GAME_ASSET:
        return [OutputFormat.GLB, OutputFormat.FBX]
    if profile is Profile.HIGH_QUALITY:
        return [OutputFormat.GLB, OutputFormat.BLEND]
    return [OutputFormat.GLB]


@lru_cache(maxsize=4096)
def _analyze_text(text: str, has_image: bool, has_mesh: bool) -> tuple:
    """Cached keyword analysis of the prompt; returns plain values for cache safety."""
    rules: list[str] = []
    lowered = text.lower()

    # Task type.
    if has_mesh:
        if word := _contains_any(lowered, CONVERT_WORDS):
            task, rule = TaskType.CONVERT, f"task:convert({word})"
        elif word := _contains_any(lowered, PREVIEW_WORDS):
            task, rule = TaskType.PREVIEW, f"task:preview({word})"
        elif word := _contains_any(lowered, TEXTURE_WORDS):
            task, rule = TaskType.TEXTURE, f"task:texture({word})"
        else:
            word = _contains_any(lowered, EDIT_WORDS)
            task, rule = TaskType.EDIT_3D, f"task:edit({word or 'mesh input'})"
    elif has_image:
        task, rule = TaskType.IMAGE_TO_3D, "task:image_to_3d(image input)"
    else:
        task, rule = TaskType.GENERATE_3D, "task:generate_3d(text input)"
    rules.append(rule)

    # Profile.
    profile = Profile.BALANCED
    for candidate, words in PROFILE_RULES:
        if word := _contains_any(lowered, words):
            profile = candidate
            rules.append(f"profile:{candidate.value}({word})")
            break

    # Output formats explicitly mentioned.
    formats = []
    for fmt, pattern in FORMAT_RULES:
        if pattern.search(lowered):
            formats.append(fmt)
            rules.append(f"format:{fmt.value}")

    max_polygons = _parse_polygons(lowered)
    if max_polygons:
        rules.append(f"polygons:{max_polygons}")
    elif _contains_any(lowered, ("로우폴리", "로우 폴리", "low poly", "low-poly", "lowpoly")):
        max_polygons = LOW_POLY_DEFAULT
        rules.append(f"polygons:{LOW_POLY_DEFAULT}(low poly)")

    size = _parse_size(lowered)
    if size:
        rules.append(f"size:{size:g}m")

    wants_textures = _contains_any(lowered, TEXTURE_WORDS) is not None
    return task, profile, tuple(formats), max_polygons, size, wants_textures, tuple(rules)


def analyze(
    request: JobRequest,
    image_path: str | None = None,
    input_mesh_path: str | None = None,
    commercial_use: bool = True,
) -> TaskSpec:
    """Analyse a job request into a structured task specification."""
    has_image = bool(image_path)
    has_mesh = bool(input_mesh_path)
    task, profile, formats, max_polygons, size, wants_textures, rules = _analyze_text(
        request.prompt.strip(), has_image, has_mesh
    )
    rules = list(rules)

    if request.profile is not None:
        profile = request.profile
        rules.append(f"profile:{profile.value}(user)")

    if request.output_formats:
        output_formats = list(request.output_formats)
        rules.append("format:user")
    elif task is TaskType.CONVERT and formats:
        output_formats = list(formats)
    else:
        # Formats named in the prompt are added to the profile defaults, so
        # "3D 프린팅용 ..., 블렌더 파일도" still yields the STL a printer needs.
        output_formats = list(dict.fromkeys([*_default_formats(profile), *formats]))
    # The web viewer needs a GLB unless the task is a pure conversion.
    if task is not TaskType.CONVERT and OutputFormat.GLB not in output_formats:
        output_formats.append(OutputFormat.GLB)

    if request.max_polygons:
        max_polygons = request.max_polygons
    elif max_polygons is None and profile is Profile.GAME_ASSET:
        max_polygons = GAME_POLY_DEFAULT
        rules.append(f"polygons:{GAME_POLY_DEFAULT}(game default)")

    inputs: list[Modality] = []
    if request.prompt.strip():
        inputs.append(Modality.TEXT)
    if has_image:
        inputs.append(Modality.IMAGE)
    if has_mesh:
        inputs.append(Modality.MESH)

    return TaskSpec(
        task_type=task,
        profile=profile,
        prompt=request.prompt.strip(),
        inputs=inputs,
        image_path=image_path,
        input_mesh_path=input_mesh_path,
        output_formats=output_formats,
        max_polygons=max_polygons,
        target_size_m=request.target_size_m or size,
        require_textures=(wants_textures or profile in (Profile.HIGH_QUALITY, Profile.GAME_ASSET))
        and profile is not Profile.PRINT_3D,
        require_watertight=profile is Profile.PRINT_3D,
        deadline_s=request.deadline_s,
        model_id=request.model_id,
        commercial_use=commercial_use,
        matched_rules=rules,
    )
