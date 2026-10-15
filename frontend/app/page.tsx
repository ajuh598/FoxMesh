"use client";

import Link from "next/link";
import { GpuMeter } from "@/components/GpuMeter";
import { ErrorBox, StatusBadge, formatSeconds, formatTime } from "@/components/ui";
import { PROFILE_LABELS, api } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";

export default function DashboardPage() {
  const { data, error } = usePolling(api.dashboard, 3000);
  const system = usePolling(api.system, 5000);

  return (
    <div className="stack">
      <div className="row">
        <h1>대시보드</h1>
        <span className="spacer" />
        <Link href="/generate" className="button">새 3D 모델 생성</Link>
      </div>
      <ErrorBox message={error && `백엔드에 연결할 수 없습니다 (${error})`} />

      {data && (
        <>
          <div className="grid cols-4">
            {(["running", "queued", "succeeded", "failed"] as const).map((s) => (
              <div className="card stat" key={s}>
                <div className="value">{data.jobs[s]}</div>
                <div className="label"><StatusBadge status={s} /></div>
              </div>
            ))}
          </div>

          <div className="grid cols-2">
            <div className="card">
              <h2>GPU · VRAM</h2>
              <GpuMeter gpu={data.gpu} />
              {system.data && (
                <p className="small muted">
                  Blender {system.data.blender.available ? "연결됨" : "미설치 (기본 처리만 수행)"} · 워커{" "}
                  {system.data.queue.workers}개 · 실행 중 {system.data.queue.running} · 대기 {system.data.queue.queued}
                </p>
              )}
            </div>
            <div className="card">
              <h2>AI 모델 실행 현황</h2>
              <table>
                <thead>
                  <tr><th>모델</th><th className="num">실행</th><th className="num">실패율</th><th className="num">평균 시간</th></tr>
                </thead>
                <tbody>
                  {data.models.map((m) => (
                    <tr key={m.id}>
                      <td>
                        {m.name} {!m.available && <span className="badge muted">사용 불가</span>}
                      </td>
                      <td className="num">{m.runs}</td>
                      <td className="num">{m.failure_rate == null ? "—" : `${(m.failure_rate * 100).toFixed(0)}%`}</td>
                      <td className="num">{formatSeconds(m.mean_runtime_s)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card">
            <h2>최근 프로젝트</h2>
            {data.recent_jobs.length === 0 ? (
              <p className="muted">아직 작업이 없습니다.</p>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr><th></th><th>요청</th><th>유형</th><th>모델</th><th>상태</th><th className="num">소요</th><th className="num">시각</th></tr>
                  </thead>
                  <tbody>
                    {data.recent_jobs.map((j) => (
                      <tr key={j.id}>
                        <td>
                          {j.thumbnail ? <img className="thumb" src={api.fileUrl(j.id, j.thumbnail)} alt="" /> : <div className="thumb" />}
                        </td>
                        <td><Link href={`/jobs/${j.id}`}>{j.prompt || j.task_type}</Link></td>
                        <td className="small">{PROFILE_LABELS[j.profile]}</td>
                        <td className="small mono">{j.model_id ?? "—"}</td>
                        <td><StatusBadge status={j.status} /></td>
                        <td className="num">{formatSeconds(j.duration_s)}</td>
                        <td className="num small">{formatTime(j.created_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {data.recent_errors.length > 0 && (
            <div className="card">
              <h2>오류 기록</h2>
              <ul className="small">
                {data.recent_errors.map((j) => (
                  <li key={j.id}>
                    <Link href={`/jobs/${j.id}`}>{j.prompt || j.id}</Link>: {j.error}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </div>
  );
}
