import { RoutingDecision } from "@/lib/api";
import { Bar, formatSeconds } from "./ui";

const CRITERIA = [
  ["accuracy", "정확도"],
  ["speed", "속도"],
  ["quality", "품질"],
  ["stability", "안정성"],
  ["resource", "자원 효율"],
] as const;

/** Shows why the router picked what it picked: weights, per-model scores and rejections. */
export function RoutingTable({ routing }: { routing: RoutingDecision }) {
  if (!routing.requires_model) {
    return <p className="muted small">AI 모델이 필요 없는 작업입니다. Blender 처리만 실행합니다.</p>;
  }
  return (
    <div className="stack">
      <div className="small muted">
        가중치:{" "}
        {CRITERIA.map(([key, label]) => `${label} ${(routing.weights[key] * 100).toFixed(0)}%`).join(" · ")}
        {" · "}결정 시간 {routing.decided_in_ms.toFixed(1)}ms
      </div>
      {routing.warnings.map((w) => (
        <div key={w} className="badge warn">{w}</div>
      ))}
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>모델</th>
              {CRITERIA.map(([key, label]) => (
                <th key={key}>{label}</th>
              ))}
              <th className="num">총점</th>
              <th className="num">예상 시간</th>
              <th className="num">VRAM</th>
            </tr>
          </thead>
          <tbody>
            {routing.candidates.map((c) => (
              <tr key={c.model_id}>
                <td>
                  {c.model_name}{" "}
                  {c.model_id === routing.selected && <span className="badge ok">선택</span>}
                  {routing.fallbacks.includes(c.model_id) && <span className="badge muted">대체</span>}
                  {c.needs_conversion && <div className="small muted">Blender 변환 필요</div>}
                </td>
                {CRITERIA.map(([key]) => (
                  <td key={key}>
                    <Bar value={c.scores[key]} />
                  </td>
                ))}
                <td className="num"><strong>{c.scores.total.toFixed(3)}</strong></td>
                <td className="num">{formatSeconds(c.est_runtime_s)}</td>
                <td className="num">{c.vram_gb ? `${c.vram_gb}GB` : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {routing.rejected.length > 0 && (
        <details>
          <summary>제외된 모델 {routing.rejected.length}개</summary>
          <ul className="small">
            {routing.rejected.map((r) => (
              <li key={r.model_id}>
                <strong>{r.model_id}</strong>: {r.reasons.join(", ")}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
