import { GpuSnapshot } from "@/lib/api";

export function GpuMeter({ gpu }: { gpu: GpuSnapshot }) {
  const used = Math.max(gpu.total_vram_gb - gpu.free_vram_gb, 0);
  const pct = gpu.total_vram_gb > 0 ? (used / gpu.total_vram_gb) * 100 : 0;
  return (
    <div className="stack">
      <div className="row">
        <strong>{gpu.devices[0]?.name ?? (gpu.detected ? "GPU" : "GPU 미감지")}</strong>
        <span className="spacer" />
        <span className="small muted">
          {used.toFixed(1)} / {gpu.total_vram_gb.toFixed(1)} GB
        </span>
      </div>
      <div className="meter">
        <div style={{ width: `${pct}%` }} />
      </div>
      <div className="small muted">
        사용 가능 {gpu.free_vram_gb.toFixed(1)} GB · 작업 예약 {gpu.reserved_vram_gb.toFixed(1)} GB
        {!gpu.detected && " · nvidia-smi를 찾지 못해 FOXMESH_VRAM_GB 설정값 사용"}
      </div>
    </div>
  );
}
