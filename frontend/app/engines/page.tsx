"use client";

import { useState } from "react";
import { Bar, ErrorBox, formatSeconds, reasonLabel } from "@/components/ui";
import { ModelView, PROFILE_LABELS, Profile, RoutingPolicy, api } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";

const RUNTIME_LABELS = { local_gpu: "로컬 GPU", remote: "원격 서버", cpu: "CPU" } as const;

function ModelRow({ view, onChange }: { view: ModelView; onChange: () => void }) {
  const { spec, stats } = view;
  const [endpoint, setEndpoint] = useState(String(spec.config.endpoint ?? ""));
  const [error, setError] = useState<string | null>(null);
  const usesEndpoint = spec.adapter === "http";

  async function save(body: Parameters<typeof api.updateModel>[1]) {
    try {
      setError(null);
      await api.updateModel(spec.id, body);
      onChange();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  const failureRate = stats.recent.length ? stats.recent.filter((ok) => !ok).length / stats.recent.length : null;
  return (
    <div className="card stack">
      <div className="row">
        <strong>{spec.name}</strong>
        <span className="small muted mono">{spec.id} v{spec.version}</span>
        {view.available ? <span className="badge ok">실행 가능</span> : <span className="badge muted">{reasonLabel(view.unavailable_reason)}</span>}
        <span className="spacer" />
        <label className="row small">
          <input type="checkbox" checked={spec.enabled} onChange={(e) => save({ enabled: e.target.checked })} />
          활성화
        </label>
      </div>
      {spec.description && <div className="small muted">{spec.description}</div>}
      <div className="grid cols-4 small">
        <div>작업: {spec.tasks.join(", ")}</div>
        <div>입력: {spec.inputs.join(", ")} → {spec.outputs.join(", ").toUpperCase()}</div>
        <div>{RUNTIME_LABELS[spec.runtime]}{spec.vram_gb ? ` · VRAM ${spec.vram_gb}GB` : ""}</div>
        <div>라이선스: {spec.license}{spec.commercial_use ? "" : " (비상업)"}</div>
      </div>
      <div className="grid cols-4 small">
        <div>실행 {stats.runs}회 · 실패 {stats.failures}회</div>
        <div>최근 실패율 {failureRate == null ? "—" : `${(failureRate * 100).toFixed(0)}%`}</div>
        <div>평균 생성 시간 {formatSeconds(stats.mean_runtime_s ?? spec.est_runtime_s)}{stats.mean_runtime_s == null && " (추정)"}</div>
        <div className="row">
          검증 점수 <Bar value={stats.mean_validation_score ?? spec.quality.accuracy} />
          {stats.mean_validation_score == null && "(추정)"}
        </div>
      </div>
      {stats.last_error && <div className="small muted">마지막 오류: {stats.last_error}</div>}
      <div className="row">
        {usesEndpoint && (
          <>
            <input
              placeholder="추론 서버 주소 (http://gpu-host:9000)"
              value={endpoint}
              onChange={(e) => setEndpoint(e.target.value)}
              style={{ minWidth: 280 }}
            />
            <button onClick={() => save({ config: { endpoint: endpoint || null } })}>주소 저장</button>
          </>
        )}
        <button onClick={() => api.resetStats(spec.id).then(onChange)} disabled={stats.runs === 0}>
          통계 초기화
        </button>
      </div>
      <ErrorBox message={error} />
    </div>
  );
}

function PolicyEditor({ profile, policy, onChange }: { profile: Profile; policy: RoutingPolicy; onChange: () => void }) {
  const [draft, setDraft] = useState(policy);
  const keys = ["accuracy", "speed", "quality", "stability", "resource"] as const;
  const labels = { accuracy: "정확도", speed: "속도", quality: "품질", stability: "안정성", resource: "자원 효율" };
  return (
    <tr>
      <td>{PROFILE_LABELS[profile]}</td>
      {keys.map((k) => (
        <td key={k}>
          <input
            type="number" step="0.05" min={0} style={{ width: 72 }} aria-label={labels[k]}
            value={draft.weights[k]}
            onChange={(e) => setDraft({ ...draft, weights: { ...draft.weights, [k]: Number(e.target.value) } })}
          />
        </td>
      ))}
      <td>
        <input type="number" step="0.05" min={0} max={1} style={{ width: 72 }} value={draft.min_quality}
          onChange={(e) => setDraft({ ...draft, min_quality: Number(e.target.value) })} />
      </td>
      <td>
        <input type="number" min={1} style={{ width: 80 }} value={draft.reference_runtime_s}
          onChange={(e) => setDraft({ ...draft, reference_runtime_s: Number(e.target.value) })} />
      </td>
      <td className="row">
        <button onClick={() => api.setPolicy(profile, draft).then(onChange)}>저장</button>
        <button onClick={() => api.resetPolicy(profile).then((p) => { setDraft(p); onChange(); })}>기본값</button>
      </td>
    </tr>
  );
}

export default function EnginesPage() {
  const models = usePolling(api.models, 5000);
  const policies = usePolling(api.policies, 60000);

  return (
    <div className="stack">
      <h1>실행 엔진 관리</h1>
      <ErrorBox message={models.error} />
      {models.data?.map((view) => (
        <ModelRow key={view.spec.id} view={view} onChange={models.refresh} />
      ))}

      <div className="card">
        <h2>자동 선택 정책</h2>
        <p className="small muted">
          작업 목적별 점수 가중치입니다(합계는 자동 정규화). 최소 품질에 못 미치는 모델은 후보에서 빠지며,
          기준 시간에서 속도 점수가 0.5가 됩니다.
        </p>
        {policies.data && (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>목적</th><th>정확도</th><th>속도</th><th>품질</th><th>안정성</th><th>자원</th>
                  <th>최소 품질</th><th>기준 시간(s)</th><th></th>
                </tr>
              </thead>
              <tbody>
                {(Object.keys(policies.data) as Profile[]).map((p) => (
                  <PolicyEditor key={p} profile={p} policy={policies.data![p]} onChange={policies.refresh} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
