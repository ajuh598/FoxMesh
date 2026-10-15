// Types mirror backend/foxmesh/schemas.py and pipeline.py.

export type Profile = "prototype" | "balanced" | "high_quality" | "game_asset" | "print_3d";
export type OutputFormat = "glb" | "obj" | "fbx" | "stl" | "ply" | "blend";
export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export const PROFILE_LABELS: Record<Profile, string> = {
  prototype: "빠른 시제품",
  balanced: "균형",
  high_quality: "고품질",
  game_asset: "게임용 에셋",
  print_3d: "3D 프린팅",
};

export const STATUS_LABELS: Record<JobStatus, string> = {
  queued: "대기",
  running: "실행 중",
  succeeded: "완료",
  failed: "실패",
  cancelled: "취소",
};

export const STAGE_LABELS: Record<string, string> = {
  queued: "대기",
  routing: "모델 선택",
  generating: "AI 생성",
  postprocessing: "Blender 후처리",
  validating: "검증",
  done: "완료",
};

export const FORMATS: OutputFormat[] = ["glb", "obj", "fbx", "stl", "ply", "blend"];

export interface JobRequest {
  prompt: string;
  image_id?: string | null;
  input_mesh_id?: string | null;
  profile?: Profile | null;
  output_formats?: OutputFormat[] | null;
  max_polygons?: number | null;
  target_size_m?: number | null;
  model_id?: string | null;
}

export interface TaskSpec {
  task_type: string;
  profile: Profile;
  prompt: string;
  output_formats: OutputFormat[];
  max_polygons: number | null;
  target_size_m: number | null;
  require_textures: boolean;
  require_watertight: boolean;
  matched_rules: string[];
}

export interface ScoreBreakdown {
  accuracy: number;
  speed: number;
  quality: number;
  stability: number;
  resource: number;
  total: number;
}

export interface Candidate {
  model_id: string;
  model_name: string;
  scores: ScoreBreakdown;
  est_runtime_s: number;
  vram_gb: number;
  needs_conversion: boolean;
}

export interface GpuSnapshot {
  detected: boolean;
  total_vram_gb: number;
  free_vram_gb: number;
  reserved_vram_gb: number;
  devices: { name: string; utilization: number | null }[];
}

export interface RoutingDecision {
  profile: Profile;
  task_type: string;
  selected: string | null;
  fallbacks: string[];
  candidates: Candidate[];
  rejected: { model_id: string; reasons: string[] }[];
  weights: Omit<ScoreBreakdown, "total">;
  warnings: string[];
  gpu: GpuSnapshot;
  decided_in_ms: number;
  requires_model: boolean;
}

export interface Check {
  name: string;
  passed: boolean;
  severity: "error" | "warning";
  detail: string;
}

export interface Validation {
  passed: boolean;
  score: number;
  checks: Check[];
}

export interface Attempt {
  model_id: string | null;
  started_at: number;
  generation_s: number | null;
  postprocess_s: number | null;
  ok: boolean;
  error: string | null;
  transient: boolean;
  validation: Validation | null;
}

export interface Job {
  id: string;
  status: JobStatus;
  stage: string;
  spec: TaskSpec;
  routing: RoutingDecision | null;
  attempts: Attempt[];
  result: {
    model_id: string | null;
    engine: string;
    steps: string[];
    files: Partial<Record<OutputFormat, string>>;
    thumbnail: string | null;
    stats: Record<string, unknown>;
    validation: Validation;
    warnings: string[];
  } | null;
  error: string | null;
  events: { ts: number; stage: string; message: string }[];
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
}

export interface JobSummary {
  id: string;
  status: JobStatus;
  stage: string;
  prompt: string;
  task_type: string;
  profile: Profile;
  model_id: string | null;
  thumbnail: string | null;
  error: string | null;
  created_at: number;
  duration_s: number | null;
}

export interface ModelSpec {
  id: string;
  name: string;
  version: string;
  adapter: string;
  tasks: string[];
  inputs: string[];
  outputs: OutputFormat[];
  runtime: "local_gpu" | "remote" | "cpu";
  vram_gb: number;
  est_runtime_s: number;
  quality: { accuracy: number; mesh: number; texture: number };
  license: string;
  commercial_use: boolean;
  enabled: boolean;
  description: string;
  config: Record<string, unknown>;
}

export interface ModelStats {
  runs: number;
  failures: number;
  mean_runtime_s: number | null;
  mean_validation_score: number | null;
  recent: boolean[];
  last_error: string | null;
}

export interface ModelView {
  spec: ModelSpec;
  stats: ModelStats;
  available: boolean;
  unavailable_reason: string | null;
}

export interface RoutingPolicy {
  weights: Omit<ScoreBreakdown, "total">;
  min_quality: number;
  reference_runtime_s: number;
  tie_margin: number;
}

export interface SystemInfo {
  gpu: GpuSnapshot;
  blender: { available: boolean; executable: string | null; mode: string };
  queue: { queued: number; running: number; workers: number };
  adapters: string[];
}

export interface Dashboard {
  jobs: Record<JobStatus, number>;
  recent_jobs: JobSummary[];
  recent_errors: JobSummary[];
  models: {
    id: string;
    name: string;
    available: boolean;
    runs: number;
    failure_rate: number | null;
    mean_runtime_s: number | null;
  }[];
  gpu: GpuSnapshot;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep statusText */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  system: () => request<SystemInfo>("/api/system"),
  dashboard: () => request<Dashboard>("/api/dashboard"),
  analyze: (body: JobRequest) =>
    request<{ spec: TaskSpec; routing: RoutingDecision; error: string | null }>("/api/analyze", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  submit: (body: JobRequest) =>
    request<{ job: Job; deduplicated: boolean }>("/api/jobs", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  jobs: (limit = 50) => request<JobSummary[]>(`/api/jobs?limit=${limit}`),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  cancel: (id: string) => request<Job>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  models: () => request<ModelView[]>("/api/models"),
  updateModel: (id: string, body: Partial<Pick<ModelSpec, "enabled" | "version" | "config">>) =>
    request<ModelView>(`/api/models/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  resetStats: (id: string) => request<ModelView>(`/api/models/${id}/stats`, { method: "DELETE" }),
  policies: () => request<Record<Profile, RoutingPolicy>>("/api/policies"),
  setPolicy: (profile: Profile, policy: RoutingPolicy) =>
    request<RoutingPolicy>(`/api/policies/${profile}`, {
      method: "PUT",
      body: JSON.stringify(policy),
    }),
  resetPolicy: (profile: Profile) =>
    request<RoutingPolicy>(`/api/policies/${profile}`, { method: "DELETE" }),
  async upload(file: File): Promise<{ id: string; filename: string; size: number }> {
    const res = await fetch(`/api/uploads?filename=${encodeURIComponent(file.name)}`, {
      method: "PUT",
      body: file,
    });
    if (!res.ok) throw new Error(`${res.status}: ${(await res.json()).detail ?? res.statusText}`);
    return res.json();
  },
  fileUrl: (jobId: string, name: string) => `/api/jobs/${jobId}/files/${encodeURIComponent(name)}`,
};
