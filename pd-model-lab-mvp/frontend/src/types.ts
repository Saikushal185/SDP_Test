export type ModelInfo = {
  model_key: string;
  dataset_id: string;
  dataset_source: string;
  model_name: string;
  model_family: "classical" | "quantum";
  display_family: string;
  inference_ready: boolean;
  status: "ready" | "repaired" | "unavailable";
  metrics: {
    cv_folds?: number | null;
    mean_accuracy?: number | null;
    mean_recall?: number | null;
    mean_f1?: number | null;
    mean_roc_auc?: number | null;
    fit_strategy?: string;
    fit_sample_count?: number | null;
  };
};

export type DashboardResponse = {
  comparison: Array<{
    dataset_id: string;
    dataset_source: string;
    model_key: string;
    model_name: string;
    model_family: "classical" | "quantum";
    display_family: string;
    inference_ready: boolean;
    status: string;
    mean_accuracy: number | null;
    mean_recall: number | null;
    mean_f1: number | null;
    mean_roc_auc: number | null;
  }>;
  datasets: Array<{
    dataset_id: string;
    dataset_source: string;
    model_count: number;
    ready_count: number;
    feature_count: number;
  }>;
};

export type SamplePayload = {
  dataset_id: string;
  feature_names: string[];
  sample_count: number;
  rows: Array<{
    row_index: number;
    label: number | null;
    features: Record<string, number | null>;
  }>;
};

export type FeatureImportance = {
  feature: string;
  importance: number;
};

export type GroupImpact = {
  model_key: string;
  series: Array<{
    threshold: number;
    group: string;
    positive_rate: number;
    n: number;
  }>;
  summary: Array<{
    group: string;
    positive_rate: number;
    ci_low: number | null;
    ci_high: number | null;
    n: number | null;
  }>;
};

export type PredictionRow = {
  row_index: number;
  source: string;
  model_key: string;
  probability: number;
  confidence: number;
  predicted_label: string;
  explanation?: {
    method: string;
    output_scale?: string;
    base_value?: number | null;
    groups: GroupedExplanation[];
  };
};

export type GroupedExplanation = {
  name: string;
  value: number;
  absValue: number;
  featureCount: number;
};
