"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { RoutingTable } from "@/components/RoutingTable";
import { ErrorBox, reasonLabel } from "@/components/ui";
import {
  FORMATS, JobRequest, ModelView, OutputFormat, PROFILE_LABELS, Profile, RoutingDecision, TaskSpec, api,
} from "@/lib/api";

type Analysis = { spec: TaskSpec; routing: RoutingDecision; error: string | null };

export default function GeneratePage() {
  const router = useRouter();
  const [prompt, setPrompt] = useState("");
  const [profile, setProfile] = useState<Profile | "">("");
  const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [mesh, setMesh] = useState<File | null>(null);
  const [advanced, setAdvanced] = useState(false);
  const [modelId, setModelId] = useState("");
  const [formats, setFormats] = useState<OutputFormat[]>([]);
  const [maxPolygons, setMaxPolygons] = useState("");
  const [sizeCm, setSizeCm] = useState("");
  const [models, setModels] = useState<ModelView[]>([]);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!image) {
      setImagePreview(null);
      return;
    }
    const url = URL.createObjectURL(image);
    setImagePreview(url);
    return () => URL.revokeObjectURL(url);
  }, [image]);

  useEffect(() => {
    if (advanced && models.length === 0) api.models().then(setModels).catch(() => {});
  }, [advanced, models.length]);

  async function buildRequest(): Promise<JobRequest> {
    const [imageUp, meshUp] = await Promise.all([
      image ? api.upload(image) : null,
      mesh ? api.upload(mesh) : null,
    ]);
    return {
      prompt,
      image_id: imageUp?.id ?? null,
      input_mesh_id: meshUp?.id ?? null,
      profile: profile || null,
      output_formats: advanced && formats.length ? formats : null,
      max_polygons: advanced && maxPolygons ? Number(maxPolygons) : null,
      target_size_m: advanced && sizeCm ? Number(sizeCm) / 100 : null,
      model_id: advanced && modelId ? modelId : null,
    };
  }

  async function run(action: "analyze" | "submit") {
    setBusy(true);
    setError(null);
    try {
      const body = await buildRequest();
      if (action === "analyze") {
        setAnalysis(await api.analyze(body));
      } else {
        const { job } = await api.submit(body);
        router.push(`/jobs/${job.id}`);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const canSubmit = !busy && (prompt.trim() || image || mesh);

  return (
    <div className="stack">
      <h1>새 3D 모델 생성</h1>
      <div className="card stack">
        <label className="field">
          만들고 싶은 3D 모델을 설명하세요
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="예: 게임에 사용할 중세풍 검을 만들어 줘 / 3D 프린팅용 10cm 컵"
          />
        </label>
        <div className="row">
          <label className="field">
            이미지로 3D 생성 (PNG·JPG·WebP)
            <input type="file" accept="image/png,image/jpeg,image/webp" onChange={(e) => setImage(e.target.files?.[0] ?? null)} />
          </label>
          <label className="field">
            기존 3D 파일 (편집·변환·미리보기)
            <input type="file" accept=".glb,.gltf,.obj,.fbx,.stl,.ply,.blend" onChange={(e) => setMesh(e.target.files?.[0] ?? null)} />
          </label>
          <label className="field">
            작업 목적
            <select value={profile} onChange={(e) => setProfile(e.target.value as Profile | "")}>
              <option value="">자동 판단</option>
              {(Object.keys(PROFILE_LABELS) as Profile[]).map((p) => (
                <option key={p} value={p}>{PROFILE_LABELS[p]}</option>
              ))}
            </select>
          </label>
        </div>

        {imagePreview && (
          <div className="row" style={{ alignItems: "flex-start" }}>
            <img src={imagePreview} alt="입력 이미지" className="input-preview" />
            <div className="small muted" style={{ flex: 1, minWidth: 200 }}>
              배경을 자동으로 제거하고 물체의 윤곽으로 입체 모델을 만듭니다. 배경이 단색이거나 투명한 PNG일 때
              가장 잘 됩니다. 설명에 <strong>부조</strong>(높이맵), <strong>로고·키링</strong>(평면 돌출),
              <strong> 인형·통통</strong>(부풀리기, 기본값)을 넣어 형태를 고를 수 있습니다. 이미지만 넣어도 생성됩니다.
            </div>
          </div>
        )}

        <details open={advanced} onToggle={(e) => setAdvanced((e.target as HTMLDetailsElement).open)}>
          <summary>고급 설정 (수동 모델 선택)</summary>
          <div className="row" style={{ marginTop: 12 }}>
            <label className="field">
              AI 모델
              <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
                <option value="">자동 선택</option>
                {models.map((m) => (
                  <option key={m.spec.id} value={m.spec.id} disabled={!m.available}>
                    {m.spec.name}{!m.available ? ` (${reasonLabel(m.unavailable_reason)})` : ""}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              최대 폴리곤
              <input type="number" min={1} value={maxPolygons} onChange={(e) => setMaxPolygons(e.target.value)} />
            </label>
            <label className="field">
              최대 크기 (cm)
              <input type="number" min={0} step="0.1" value={sizeCm} onChange={(e) => setSizeCm(e.target.value)} />
            </label>
          </div>
          <div className="row small" style={{ marginTop: 8 }}>
            출력 형식:
            {FORMATS.map((f) => (
              <label key={f} className="row" style={{ gap: 4 }}>
                <input
                  type="checkbox"
                  checked={formats.includes(f)}
                  onChange={(e) => setFormats((cur) => (e.target.checked ? [...cur, f] : cur.filter((x) => x !== f)))}
                />
                {f.toUpperCase()}
              </label>
            ))}
          </div>
        </details>

        <div className="row">
          <button onClick={() => run("analyze")} disabled={!canSubmit}>선택 기준 미리보기</button>
          <button className="primary" onClick={() => run("submit")} disabled={!canSubmit}>
            {busy ? "처리 중…" : "생성 시작"}
          </button>
        </div>
        <ErrorBox message={error} />
      </div>

      {analysis && (
        <div className="card stack">
          <h2>요청 분석 결과</h2>
          <div className="small">
            작업 <strong>{analysis.spec.task_type}</strong> · 목적 <strong>{PROFILE_LABELS[analysis.spec.profile]}</strong> · 출력{" "}
            {analysis.spec.output_formats.map((f) => f.toUpperCase()).join(", ")}
            {analysis.spec.max_polygons && ` · 폴리곤 ≤ ${analysis.spec.max_polygons.toLocaleString()}`}
            {analysis.spec.target_size_m && ` · 크기 ${analysis.spec.target_size_m * 100}cm`}
          </div>
          <div className="small muted mono">{analysis.spec.matched_rules.join(" · ")}</div>
          <ErrorBox message={analysis.error} />
          <RoutingTable routing={analysis.routing} />
        </div>
      )}
    </div>
  );
}
