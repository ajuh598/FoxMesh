"use client";

import { use } from "react";
import { ModelPreview } from "@/components/ModelPreview";
import { RoutingTable } from "@/components/RoutingTable";
import { ErrorBox, StatusBadge, formatSeconds, formatTime } from "@/components/ui";
import { PROFILE_LABELS, STAGE_LABELS, api } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";

const FINISHED = new Set(["succeeded", "failed", "cancelled"]);

export default function JobPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const { data: job, error, refresh } = usePolling(() => api.job(id), 1000, true);
  const running = job ? !FINISHED.has(job.status) : true;

  if (!job) return <ErrorBox message={error} />;
  const result = job.result;
  const thumb = result?.thumbnail ? api.fileUrl(job.id, result.thumbnail) : null;

  return (
    <div className="stack">
      <div className="row">
        <h1>{job.spec.prompt || job.spec.task_type}</h1>
        <StatusBadge status={job.status} />
        {running && <span className="small muted">{STAGE_LABELS[job.stage] ?? job.stage}…</span>}
        <span className="spacer" />
        {running && (
          <button onClick={() => api.cancel(job.id).then(refresh)}>취소</button>
        )}
      </div>
      <div className="small muted">
        {job.spec.task_type} · {PROFILE_LABELS[job.spec.profile]} · 출력{" "}
        {job.spec.output_formats.map((f) => f.toUpperCase()).join(", ")}
        {job.finished_at && job.started_at && ` · 총 ${formatSeconds(job.finished_at - job.started_at)}`}
      </div>
      <ErrorBox message={job.error} />

      <div className="grid cols-2">
        <div className="card stack">
          <h2>결과 미리보기</h2>
          {result?.files.glb ? (
            <ModelPreview src={api.fileUrl(job.id, result.files.glb)} poster={thumb} />
          ) : thumb ? (
            <img src={thumb} alt="썸네일" className="viewer" style={{ objectFit: "contain" }} />
          ) : (
            <div className="viewer" />
          )}
          {result && (
            <div className="row">
              {Object.entries(result.files).map(([fmt, name]) => (
                <a key={fmt} className="button" href={api.fileUrl(job.id, name!)} download>
                  {fmt.toUpperCase()} 다운로드
                </a>
              ))}
            </div>
          )}
        </div>

        <div className="card stack">
          <h2>진행 상태</h2>
          <ul className="timeline">
            {job.events.map((e, i) => (
              <li key={i}>
                <span className="t">{formatTime(e.ts)}</span>
                <span className="badge muted">{STAGE_LABELS[e.stage] ?? e.stage}</span>
                <span>{e.message}</span>
              </li>
            ))}
          </ul>
          {result && (
            <>
              <h2>검증 ({(result.validation.score * 100).toFixed(0)}점)</h2>
              <table>
                <tbody>
                  {result.validation.checks.map((c) => (
                    <tr key={c.name}>
                      <td>
                        <span className={`badge ${c.passed ? "ok" : c.severity === "error" ? "err" : "warn"}`}>
                          {c.passed ? "통과" : c.severity === "error" ? "실패" : "경고"}
                        </span>
                      </td>
                      <td className="mono">{c.name}</td>
                      <td className="small muted">{c.detail}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="small muted">
                후처리 엔진: {result.engine} · 실행 단계: {result.steps.join(" → ")}
              </div>
            </>
          )}
        </div>
      </div>

      {job.routing && (
        <div className="card">
          <h2>모델 선택 근거</h2>
          <RoutingTable routing={job.routing} />
        </div>
      )}

      {job.attempts.length > 0 && (
        <div className="card">
          <h2>실행 기록</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>#</th><th>모델</th><th className="num">생성</th><th className="num">후처리</th><th>결과</th></tr>
              </thead>
              <tbody>
                {job.attempts.map((a, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td className="mono">{a.model_id ?? "Blender"}</td>
                    <td className="num">{formatSeconds(a.generation_s)}</td>
                    <td className="num">{formatSeconds(a.postprocess_s)}</td>
                    <td className="small">
                      {a.ok ? <span className="badge ok">성공</span> : <span className="badge err">{a.transient ? "일시 오류" : "실패"}</span>}{" "}
                      {a.error}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {result && Object.keys(result.stats).length > 0 && (
        <div className="card">
          <details>
            <summary>메시 통계</summary>
            <pre className="mono" style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(result.stats, null, 2)}</pre>
          </details>
        </div>
      )}
    </div>
  );
}
