import type {
  DashboardResponse,
  FeatureImportance,
  GroupImpact,
  ModelInfo,
  PredictionRow,
  SamplePayload,
} from "./types";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail ?? detail;
    } catch {
      // Keep the HTTP status text when the backend returns non-JSON.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const payload = await fetchJson<{ models: ModelInfo[] }>("/api/models");
  return payload.models;
}

export function fetchDashboard(): Promise<DashboardResponse> {
  return fetchJson<DashboardResponse>("/api/dashboard");
}

export function fetchSamples(datasetId: string, limit = 5): Promise<SamplePayload> {
  return fetchJson<SamplePayload>(`/api/samples/${datasetId}?limit=${limit}`);
}

export async function fetchFeatureImportance(modelKey: string): Promise<FeatureImportance[]> {
  const payload = await fetchJson<{ features: FeatureImportance[] }>(`/api/features/${modelKey}?limit=15`);
  return payload.features;
}

export function fetchGroupImpact(modelKey: string): Promise<GroupImpact> {
  return fetchJson<GroupImpact>(`/api/group-impact/${modelKey}`);
}

export async function runPrediction(options: {
  modelKey: string;
  sampleRow: number;
  editedFeatures?: Record<string, number | null>;
  file?: File | null;
}): Promise<PredictionRow[]> {
  const form = new FormData();
  form.append("model_key", options.modelKey);
  form.append("sample_row", String(options.sampleRow));
  if (options.editedFeatures) {
    form.append("edited_features", JSON.stringify(options.editedFeatures));
  }
  if (options.file) {
    form.append("file", options.file);
  }

  const payload = await fetchJson<{ predictions: PredictionRow[] }>("/api/predict", {
    method: "POST",
    body: form,
  });
  return payload.predictions;
}
