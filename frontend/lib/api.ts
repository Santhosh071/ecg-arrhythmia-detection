import { clearSession, expireSession, getStoredToken, type AuthSession, type AuthUser } from "@/lib/auth";

export type HealthResponse = {
  status: string;
  models_loaded: boolean;
  agent_ready: boolean;
  db_connected: boolean;
  groq_available: boolean;
};

export type HistorySession = {
  session_id: number;
  timestamp: string;
  risk_level: string;
  anomaly_count: number;
  anomaly_rate: number;
  dominant_class: string;
  recording_sec: number;
  total_beats: number;
};

export type HistoryResponse = {
  patient_id: string;
  sessions: HistorySession[];
  total: number;
};

export type AlertRecord = {
  alert_id: number;
  timestamp: string;
  risk_level: string;
  color: string;
  message: string;
  class_name?: string;
  review_status?: string;
  reviewer_note?: string | null;
  reviewed_by?: string | null;
  reviewed_at?: string | null;
};

export type AlertsResponse = {
  patient_id: string;
  alerts: AlertRecord[];
  total: number;
};

export type ReportRecord = {
  report_id: number;
  session_id?: number | null;
  timestamp: string;
  file_path: string;
  file_name: string;
  report_focus?: string | null;
  status: string;
  summary?: string | null;
};

export type ReportsHistoryResponse = {
  patient_id: string;
  reports: ReportRecord[];
  total: number;
};

export type TrendResponse = {
  patient_id: string;
  trend: string;
  sessions: number;
  avg_anomaly_rate: number;
  latest_rate: number;
  history: HistorySession[];
};

export type PredictRequest = {
  patient_id: string;
  beats: number[][];
  timestamps: number[];
  fs?: number;
  save_session?: boolean;
};

export type PredictBeat = {
  beat_index: number;
  timestamp_sec: number;
  is_anomaly: boolean;
  transformer_anomaly: boolean;
  lstm_anomaly: boolean;
  transformer_score: number;
  lstm_error: number;
  cnn_class_id: number;
  cnn_class_name: string;
  cnn_short_name: string;
  cnn_confidence: number;
  cnn_all_probs: number[];
  risk_level: string;
  alert_color: string;
};

export type PredictResponse = {
  patient_id: string;
  total_beats: number;
  anomaly_count: number;
  anomaly_rate: number;
  class_counts: Record<string, number>;
  dominant_class: string;
  session_risk: string;
  recording_sec: number;
  beats: PredictBeat[];
  session_saved: boolean;
  request_truncated: boolean;
  disclaimer: string;
};

export type AgentQueryRequest = {
  question: string;
  patient_id?: string;
  session_context?: Record<string, unknown>;
};

export type AgentQueryResponse = {
  answer: string;
  patient_id?: string;
  disclaimer: string;
};

export type ReportRequest = {
  patient_id: string;
  session_data: Record<string, unknown>;
};

export type ReportResponse = {
  success: boolean;
  file_path: string;
  patient_id: string;
  message: string;
};

export type AlertReviewRequest = {
  patient_id: string;
  review_status: string;
  reviewer_note: string;
};

export type AlertReviewResponse = {
  patient_id: string;
  alert: AlertRecord;
};

export type LoginRequest = {
  username: string;
  password: string;
};

export type LoginResponse = AuthSession;
export type CurrentUserResponse = AuthUser;

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

function buildHeaders() {
  const token = getStoredToken();
  return {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

function handleUnauthorized(path: string) {
  if (path === "/auth/login") {
    clearSession();
    return;
  }

  expireSession();
  if (typeof window !== "undefined" && !window.location.pathname.startsWith("/login")) {
    window.location.assign("/login?reason=expired");
  }
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: buildHeaders(),
    cache: "no-store",
  });

  if (!response.ok) {
    if (response.status === 401) {
      handleUnauthorized(path);
      throw new Error("Session expired. Sign in again.");
    }
    throw new Error(`Request failed for ${path}: ${response.status}`);
  }

  return (await response.json()) as T;
}

async function postJson<TResponse, TRequest>(path: string, body: TRequest): Promise<TResponse> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: buildHeaders(),
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    let detail = `Request failed for ${path}: ${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) {
        detail = payload.detail;
      }
    } catch {
    }
    if (response.status === 401) {
      handleUnauthorized(path);
      detail = path === "/auth/login" ? detail : "Session expired. Sign in again.";
    }
    throw new Error(detail);
  }

  return (await response.json()) as TResponse;
}

async function patchJson<TResponse, TRequest>(path: string, body: TRequest): Promise<TResponse> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "PATCH",
    headers: buildHeaders(),
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    let detail = `Request failed for ${path}: ${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) {
        detail = payload.detail;
      }
    } catch {
    }
    if (response.status === 401) {
      handleUnauthorized(path);
      detail = "Session expired. Sign in again.";
    }
    throw new Error(detail);
  }

  return (await response.json()) as TResponse;
}

export async function fetchHealth(): Promise<HealthResponse> {
  return getJson<HealthResponse>("/health");
}

export async function login(payload: LoginRequest): Promise<LoginResponse> {
  return postJson<LoginResponse, LoginRequest>("/auth/login", payload);
}

export async function fetchCurrentUser(): Promise<CurrentUserResponse> {
  return getJson<CurrentUserResponse>("/auth/me");
}

export async function fetchHistory(patientId: string): Promise<HistoryResponse> {
  return getJson<HistoryResponse>(`/history/${patientId}`);
}

export async function fetchAlerts(patientId: string): Promise<AlertsResponse> {
  return getJson<AlertsResponse>(`/history/${patientId}/alerts`);
}

export async function fetchReports(patientId: string): Promise<ReportsHistoryResponse> {
  return getJson<ReportsHistoryResponse>(`/history/${patientId}/reports`);
}

export async function reviewAlert(alertId: number, payload: AlertReviewRequest): Promise<AlertReviewResponse> {
  return patchJson<AlertReviewResponse, AlertReviewRequest>(`/history/${payload.patient_id}/alerts/${alertId}`, payload);
}

export async function fetchTrend(patientId: string): Promise<TrendResponse> {
  return getJson<TrendResponse>(`/history/${patientId}/trend`);
}

export async function predictBatch(payload: PredictRequest): Promise<PredictResponse> {
  return postJson<PredictResponse, PredictRequest>("/predict", payload);
}

export async function queryAgent(payload: AgentQueryRequest): Promise<AgentQueryResponse> {
  return postJson<AgentQueryResponse, AgentQueryRequest>("/agent/query", payload);
}

export async function generateReport(payload: ReportRequest): Promise<ReportResponse> {
  return postJson<ReportResponse, ReportRequest>("/report/generate", payload);
}

export async function downloadReport(filePath: string): Promise<void> {
  const token = getStoredToken();
  if (!token) {
    throw new Error("No active session found for opening the report.");
  }

  const reportUrl = `${API_BASE_URL}/report/download?file_path=${encodeURIComponent(filePath)}&auth_token=${encodeURIComponent(token)}`;
  const opened = window.open(reportUrl, "_blank", "noopener,noreferrer");
  if (!opened) {
    throw new Error("Browser blocked the PDF tab. Allow pop-ups and try again.");
  }
}
