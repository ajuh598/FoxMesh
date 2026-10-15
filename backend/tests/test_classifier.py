from foxmesh.classifier import analyze
from foxmesh.schemas import JobRequest, Modality, OutputFormat, Profile, TaskType


def spec(prompt: str, **kw):
    return analyze(JobRequest(prompt=prompt), **kw)


def test_doc_example_game_sword():
    s = spec("게임에 사용할 중세풍 검을 만들어 줘")
    assert s.task_type is TaskType.GENERATE_3D
    assert s.profile is Profile.GAME_ASSET
    assert s.output_formats == [OutputFormat.GLB, OutputFormat.FBX]
    assert s.max_polygons == 20_000
    assert s.require_textures and not s.require_watertight
    assert s.inputs == [Modality.TEXT]


def test_print_profile_with_size_and_extra_format():
    s = spec("3D 프린팅용 15cm 컵, 블렌더 파일도 줘")
    assert s.profile is Profile.PRINT_3D
    assert s.target_size_m == 0.15
    # Formats named in the prompt add to, not replace, the profile defaults.
    assert s.output_formats == [OutputFormat.STL, OutputFormat.GLB, OutputFormat.BLEND]
    assert s.require_watertight and not s.require_textures


def test_polygon_budget_parsing():
    assert spec("로우폴리 나무 5천 폴리곤").max_polygons == 5_000
    assert spec("a low poly tree, 10k tris").max_polygons == 10_000
    assert spec("2만개 폴리곤 이하 자동차").max_polygons == 20_000
    assert spec("low poly rock").max_polygons == 5_000


def test_english_prototype_and_word_boundaries():
    assert spec("quick prototype of a chair").profile is Profile.PROTOTYPE
    # "fast" inside "breakfast" must not trigger the prototype profile.
    assert spec("a breakfast table").profile is Profile.BALANCED


def test_high_quality_defaults():
    s = spec("고품질 사실적인 투구")
    assert s.profile is Profile.HIGH_QUALITY
    assert OutputFormat.BLEND in s.output_formats


def test_image_input_is_image_to_3d():
    s = spec("이 사진으로 모델 만들어 줘", image_path="/tmp/x.png")
    assert s.task_type is TaskType.IMAGE_TO_3D
    assert s.inputs == [Modality.TEXT, Modality.IMAGE]


def test_mesh_input_tasks():
    assert spec("OBJ로 변환해 줘", input_mesh_path="/m.glb").task_type is TaskType.CONVERT
    assert spec("OBJ로 변환해 줘", input_mesh_path="/m.glb").output_formats == [OutputFormat.OBJ]
    assert spec("썸네일 렌더", input_mesh_path="/m.glb").task_type is TaskType.PREVIEW
    assert spec("폴리곤 최적화", input_mesh_path="/m.glb").task_type is TaskType.EDIT_3D


def test_user_overrides_win():
    s = analyze(JobRequest(prompt="게임용 검", profile=Profile.PROTOTYPE,
                           output_formats=[OutputFormat.OBJ], max_polygons=800))
    assert s.profile is Profile.PROTOTYPE
    assert s.output_formats == [OutputFormat.OBJ, OutputFormat.GLB]  # GLB kept for the viewer
    assert s.max_polygons == 800
    assert "profile:prototype(user)" in s.matched_rules


def test_fingerprint_ignores_whitespace_and_case():
    assert spec("Game  sword").fingerprint() == spec("game sword").fingerprint()
    assert spec("game sword").fingerprint() != spec("game axe").fingerprint()
