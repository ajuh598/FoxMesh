import { JobStatus, STATUS_LABELS } from "@/lib/api";

const STATUS_TONE: Record<JobStatus, string> = {
  queued: "muted",
  running: "info",
  succeeded: "ok",
  failed: "err",
  cancelled: "warn",
};

export function StatusBadge({ status }: { status: JobStatus }) {
  return <span className={`badge ${STATUS_TONE[status]}`}>{STATUS_LABELS[status]}</span>;
}

export function Bar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="bar" title={value.toFixed(2)}>
      <div style={{ width: `${pct}%` }} />
    </div>
  );
}

export function ErrorBox({ message }: { message: string | null | undefined }) {
  if (!message) return null;
  return <div className="error-box">{message}</div>;
}

export function formatTime(ts: number) {
  const d = new Date(ts * 1000);
  return [d.getHours(), d.getMinutes(), d.getSeconds()].map((n) => String(n).padStart(2, "0")).join(":");
}

const REASON_LABELS: Record<string, string> = { disabled: "비활성" };

export function reasonLabel(reason: string | null) {
  return reason ? REASON_LABELS[reason] ?? reason : "";
}

export function formatSeconds(s: number | null | undefined) {
  if (s == null) return "—";
  if (s < 1) return `${Math.round(s * 1000)}ms`;
  if (s < 120) return `${s.toFixed(1)}s`;
  return `${Math.round(s / 60)}분`;
}
