"""
Artifact-backed inference pipeline for the Parkinson's detection API.

This module loads persisted models, validates feature alignment, performs
preprocessing, and returns prediction / SHAP JSON payloads for the Flask app.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

TRAINING_DATASET_ID = "pd_speech_features.csv"
LABEL_COLUMN_CANDIDATES = (
    "class",
    "target",
    "status",
    "diagnosis",
    "label",
    "labels",
    "parkinsons",
    "pd",
)
POSITIVE_LABEL_TOKENS = ("parkinson", "pd", "patient", "positive", "disease", "dys")
NEGATIVE_LABEL_TOKENS = ("healthy", "control", "hc", "normal", "negative", "con")

DEFAULT_MODEL_REGISTRY = [
    {
        "key": "xgboost",
        "display_name": "XGBoost Voice Classifier",
        "dataset_id": TRAINING_DATASET_ID,
        "enabled": True,
        "artifact_path": "models/xgboost.pkl",
        "description": "Gradient-boosted tree model trained only on the PD speech feature dataset.",
    },
    {
        "key": "random_forest",
        "display_name": "Random Forest Voice Baseline",
        "dataset_id": TRAINING_DATASET_ID,
        "enabled": True,
        "artifact_path": "models/random_forest.pkl",
        "description": "Tree ensemble baseline trained on the same PD speech feature dataset.",
    },
]


class MLPipeline:
    """Load trained artifacts once at startup and serve inference requests."""

    def __init__(
        self,
        project_root: str,
        model_dir: Optional[str] = None,
        metrics_path: Optional[str] = None,
        registry_path: Optional[str] = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.backend_dir = self.project_root / "backend"
        self.study_dir = self.project_root / "parkinson_feature_study"
        self.model_dir = Path(model_dir) if model_dir else self.backend_dir / "models"
        self.metrics_path = (
            Path(metrics_path)
            if metrics_path
            else self.study_dir / "results" / "metrics" / "deployment_metrics.json"
        )
        self.registry_path = (
            Path(registry_path)
            if registry_path
            else self.backend_dir / "model_registry.json"
        )

        self.random_forest_model = None
        self.xgboost_model = None
        self.models: Dict[str, Any] = {}
        self.model_registry: list[dict[str, Any]] = []
        self.scaler = None
        self.expected_features: list[str] = []
        self.sample_data: Dict[str, float] = {}
        self.metrics: Dict[str, Any] = {}
        self.explainers: Dict[str, shap.TreeExplainer] = {}
        self.dataset_size = 0
        self.is_ready = False
        self.startup_error: Optional[str] = None

    def load_artifacts(self) -> None:
        """Load the persisted models, scaler, feature metadata, and metrics."""
        required_files = {
            "random_forest_model": self.model_dir / "random_forest.pkl",
            "xgboost_model": self.model_dir / "xgboost.pkl",
            "scaler": self.model_dir / "scaler.pkl",
            "feature_columns": self.model_dir / "feature_columns.json",
        }
        missing = [str(path) for path in required_files.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing trained artifacts. Run `python backend/train_and_save.py` first. "
                f"Missing files: {missing}"
            )

        self.random_forest_model = joblib.load(required_files["random_forest_model"])
        self.xgboost_model = joblib.load(required_files["xgboost_model"])
        self.models = {
            "random_forest": self.random_forest_model,
            "xgboost": self.xgboost_model,
        }
        self.scaler = joblib.load(required_files["scaler"])
        self.expected_features = self._load_feature_columns(
            required_files["feature_columns"]
        )
        self.sample_data = self._load_sample_data()
        self.metrics = self._load_metrics()
        self.model_registry = self._load_model_registry()
        self.dataset_size = self._load_dataset_size()
        self.explainers.clear()
        self.is_ready = True
        self.startup_error = None

        logger.info(
            "Loaded trained artifacts from %s with %d expected features",
            self.model_dir,
            len(self.expected_features),
        )

    def _load_feature_columns(self, path: Path) -> list[str]:
        feature_columns = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(feature_columns, list) or not all(
            isinstance(column, str) for column in feature_columns
        ):
            raise ValueError(f"Invalid feature column file: {path}")
        return feature_columns

    def _load_sample_data(self) -> Dict[str, float]:
        sample_path = self.model_dir / "sample_input.json"
        if not sample_path.exists():
            return {}

        raw_sample = json.loads(sample_path.read_text(encoding="utf-8"))
        if not isinstance(raw_sample, dict):
            return {}

        sample_data: Dict[str, float] = {}
        for name, value in raw_sample.items():
            try:
                sample_data[str(name)] = float(value)
            except (TypeError, ValueError):
                continue

        return sample_data

    def _load_metrics(self) -> Dict[str, Any]:
        if self.metrics_path.exists():
            return json.loads(self.metrics_path.read_text(encoding="utf-8"))

        aggregated_csv = self.study_dir / "results" / "metrics" / "aggregated_results.csv"
        if not aggregated_csv.exists():
            raise FileNotFoundError(
                "Missing saved metrics payload. Run `python backend/train_and_save.py` first."
            )

        metrics_df = pd.read_csv(aggregated_csv)
        models: Dict[str, Dict[str, float]] = {}
        for _, row in metrics_df.iterrows():
            model_key = str(row.get("model", "")).strip().lower().replace(" ", "_")
            if model_key not in {"random_forest", "xgboost"}:
                continue
            models[model_key] = {
                "accuracy": float(row.get("accuracy_mean", row.get("accuracy", 0.0))),
                "precision": float(row.get("precision_mean", row.get("precision", 0.0))),
                "recall": float(row.get("recall_mean", row.get("recall", 0.0))),
                "f1": float(row.get("f1_mean", row.get("f1", 0.0))),
                "auc": float(row.get("roc_auc_mean", row.get("auc", 0.0))),
            }

        if not models:
            raise ValueError("No supported model metrics found in aggregated_results.csv")

        best_model = max(models, key=lambda name: models[name]["accuracy"])
        return {"models": models, "best_model": best_model}

    def _load_model_registry(self) -> list[dict[str, Any]]:
        raw_entries: Any = DEFAULT_MODEL_REGISTRY
        if self.registry_path.exists():
            raw_entries = json.loads(self.registry_path.read_text(encoding="utf-8"))

        if not isinstance(raw_entries, list):
            raise ValueError("model_registry.json must contain a list of model entries.")

        default_by_key = {entry["key"]: entry for entry in DEFAULT_MODEL_REGISTRY}
        registry: list[dict[str, Any]] = []
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                continue

            key = self._normalize_model_name(raw_entry.get("key", ""))
            if key not in default_by_key:
                logger.warning("Ignoring unsupported registry model key: %s", raw_entry.get("key"))
                continue

            merged = {**default_by_key[key], **raw_entry, "key": key}
            merged["enabled"] = bool(merged.get("enabled", True))
            merged["dataset_id"] = TRAINING_DATASET_ID
            merged["artifact_path"] = str(
                Path(merged.get("artifact_path", default_by_key[key]["artifact_path"]))
            )
            registry.append(merged)

        if not registry:
            registry = [dict(entry) for entry in DEFAULT_MODEL_REGISTRY]

        return registry

    def _load_dataset_size(self) -> int:
        candidates = [
            self.study_dir / "data" / "raw" / "pd_speech_features_cleaned.csv",
            self.study_dir / "data" / "raw" / "pd_speech_features.csv",
        ]
        for dataset_path in candidates:
            if dataset_path.exists():
                return int(len(pd.read_csv(dataset_path, usecols=[0])))
        return 0

    def get_features_metadata(self) -> Dict[str, Any]:
        return {
            "expected_features": self.expected_features,
            "feature_count": len(self.expected_features),
            "sample_data": self.sample_data,
            "supported_models": self.get_supported_model_keys(),
            "feature_groups": self.get_feature_groups(),
            "training_dataset": TRAINING_DATASET_ID,
        }

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics

    def get_supported_model_keys(self) -> list[str]:
        enabled = [
            entry["key"]
            for entry in self.model_registry
            if entry.get("enabled", True) and entry["key"] in self.models
        ]
        return enabled or ["random_forest", "xgboost"]

    def get_model_registry(self) -> Dict[str, Any]:
        model_metrics = self.metrics.get("models", {})
        entries: list[dict[str, Any]] = []
        for entry in self.model_registry:
            key = entry["key"]
            if key not in self.models:
                continue
            entries.append(
                {
                    "key": key,
                    "display_name": str(entry.get("display_name", key)),
                    "dataset_id": TRAINING_DATASET_ID,
                    "enabled": bool(entry.get("enabled", True)),
                    "artifact_path": str(entry.get("artifact_path", "")),
                    "description": str(entry.get("description", "")),
                    "metrics": model_metrics.get(key, {}),
                    "feature_schema": self.expected_features,
                }
            )

        return {
            "training_dataset": TRAINING_DATASET_ID,
            "models": entries,
            "best_model": self.metrics.get("best_model", "xgboost"),
        }

    def get_model_info(self) -> Dict[str, Any]:
        best_model = self.metrics.get("best_model", "xgboost")
        best_accuracy = float(
            self.metrics.get("models", {}).get(best_model, {}).get("accuracy", 0.0)
        )
        return {
            "dataset_size": self.dataset_size,
            "n_selected_features": len(self.expected_features),
            "models": self.get_supported_model_keys(),
            "model_registry": self.get_model_registry()["models"],
            "best_model": best_model,
            "best_accuracy": best_accuracy,
            "training_dataset": TRAINING_DATASET_ID,
        }

    def get_feature_groups(self) -> list[dict[str, Any]]:
        grouped: dict[str, list[str]] = {}
        for feature in self.expected_features:
            grouped.setdefault(self._friendly_feature_group(feature), []).append(feature)

        return [
            {
                "name": group_name,
                "description": self._feature_group_description(group_name),
                "features": features,
            }
            for group_name, features in grouped.items()
        ]

    def predict(
        self,
        features_dict: Mapping[str, Any],
        model_name: str = "xgboost",
    ) -> Dict[str, Any]:
        scaled_frame = self._prepare_input_frame(features_dict)
        model = self._get_model(model_name)

        probability = float(model.predict_proba(scaled_frame.values)[0, 1])
        prediction = int(probability >= 0.5)
        confidence = float(max(probability, 1.0 - probability))

        return {
            "prediction": prediction,
            "probability": probability,
            "confidence": confidence,
        }

    def batch_evaluate(
        self,
        csv_text: str,
        model_name: str = "xgboost",
        max_predictions: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not isinstance(csv_text, str) or not csv_text.strip():
            raise ValueError("CSV text is required for batch evaluation.")

        dataset = pd.read_csv(io.StringIO(csv_text))
        if dataset.empty:
            raise ValueError("CSV must contain at least one data row.")

        headers = [str(column) for column in dataset.columns]
        missing = [feature for feature in self.expected_features if feature not in headers]
        ignored = [
            column
            for column in headers
            if column not in self.expected_features
            and column.lower() not in LABEL_COLUMN_CANDIDATES
        ]
        label_column = self._find_label_column(headers)

        compatibility = {
            "compatible": len(missing) == 0,
            "required_feature_count": len(self.expected_features),
            "present_feature_count": len(self.expected_features) - len(missing),
            "missing_columns": missing,
            "ignored_columns": ignored,
            "label_column": label_column,
        }

        if missing:
            return {
                **compatibility,
                "model": self._normalize_model_name(model_name),
                "display_name": self._get_model_display_name(model_name),
                "training_dataset": TRAINING_DATASET_ID,
                "row_count": int(len(dataset)),
                "metrics": None,
                "prediction_summary": None,
                "predictions": [],
                "message": "Dataset is not compatible with strict evaluation because required training features are missing.",
            }

        feature_frame = dataset[self.expected_features].apply(pd.to_numeric, errors="coerce")
        invalid_columns = [
            column for column in self.expected_features if feature_frame[column].isna().any()
        ]
        if invalid_columns:
            raise ValueError(
                "Compatible columns must be numeric. Invalid values found in: "
                + ", ".join(invalid_columns[:12])
            )

        model_key = self._normalize_model_name(model_name)
        model = self._get_model(model_key)
        scaled_values = self.scaler.transform(feature_frame.to_numpy())
        probabilities = model.predict_proba(scaled_values)[:, 1].astype(float)
        predictions = (probabilities >= 0.5).astype(int)
        confidences = np.maximum(probabilities, 1.0 - probabilities)

        encoded_labels = None
        metrics = None
        if label_column:
            encoded_labels = self._encode_label_series(dataset[label_column])
            if encoded_labels is not None:
                metrics = self._compute_batch_metrics(encoded_labels, predictions, probabilities)

        prediction_rows = []
        limit = len(dataset) if max_predictions is None else min(len(dataset), int(max_predictions))
        for row_index in range(limit):
            row: dict[str, Any] = {
                "row_index": row_index,
                "prediction": int(predictions[row_index]),
                "probability": float(probabilities[row_index]),
                "confidence": float(confidences[row_index]),
            }
            if encoded_labels is not None:
                row["actual"] = int(encoded_labels[row_index])
                row["correct"] = bool(encoded_labels[row_index] == predictions[row_index])
            prediction_rows.append(row)

        return {
            **compatibility,
            "model": model_key,
            "display_name": self._get_model_display_name(model_key),
            "training_dataset": TRAINING_DATASET_ID,
            "row_count": int(len(dataset)),
            "metrics": metrics,
            "prediction_summary": {
                "positive": int(np.sum(predictions == 1)),
                "negative": int(np.sum(predictions == 0)),
                "mean_probability": float(np.mean(probabilities)),
                "mean_confidence": float(np.mean(confidences)),
            },
            "predictions": prediction_rows,
            "message": "Dataset passed strict feature compatibility and was evaluated successfully.",
        }

    def explain(
        self,
        features_dict: Mapping[str, Any],
        model_name: str = "xgboost",
    ) -> Dict[str, Any]:
        scaled_frame = self._prepare_input_frame(features_dict)
        model_key = self._normalize_model_name(model_name)
        model = self._get_model(model_key)
        explainer = self._get_explainer(model_key, model)

        raw_shap_values = explainer.shap_values(scaled_frame)
        shap_values = self._extract_positive_class_values(raw_shap_values)[0]
        base_value = self._extract_positive_class_base_value(explainer.expected_value)
        probability = float(model.predict_proba(scaled_frame.values)[0, 1])
        prediction = int(probability >= 0.5)

        if len(shap_values) != len(self.expected_features):
            raise ValueError("SHAP output length does not match expected feature count")

        return {
            "shap_values": [float(value) for value in shap_values],
            "feature_names": self.expected_features,
            "base_value": float(base_value),
            "prediction": prediction,
        }

    def _prepare_input_frame(self, features_dict: Mapping[str, Any]) -> pd.DataFrame:
        validated_features = self._validate_and_order_features(features_dict)
        raw_frame = pd.DataFrame([validated_features], columns=self.expected_features)
        scaled_values = self.scaler.transform(raw_frame.to_numpy())
        return pd.DataFrame(scaled_values, columns=self.expected_features)

    def _validate_and_order_features(
        self, features_dict: Mapping[str, Any]
    ) -> Dict[str, float]:
        if not isinstance(features_dict, Mapping):
            raise ValueError("Features payload must be a JSON object.")

        missing = [feature for feature in self.expected_features if feature not in features_dict]
        unexpected = [
            feature for feature in features_dict.keys() if feature not in self.expected_features
        ]

        if missing or unexpected:
            problem_parts = ["Feature mismatch."]
            if missing:
                problem_parts.append(f"Missing: {missing}.")
            if unexpected:
                problem_parts.append(f"Unexpected: {unexpected}.")
            problem_parts.append("The feature columns must exactly match feature_columns.json.")
            raise ValueError(" ".join(problem_parts))

        ordered: Dict[str, float] = {}
        for feature_name in self.expected_features:
            raw_value = features_dict[feature_name]
            try:
                ordered[feature_name] = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Feature `{feature_name}` must be numeric. Received: {raw_value!r}"
                ) from exc

        return ordered

    def _get_model(self, model_name: str):
        model_key = self._normalize_model_name(model_name)
        if model_key in self.models:
            if model_key not in self.get_supported_model_keys():
                raise ValueError(f"Model `{model_key}` is disabled in the registry.")
            return self.models[model_key]
        raise ValueError(
            "Unsupported model. Choose one of: random_forest, xgboost."
        )

    def _normalize_model_name(self, model_name: str) -> str:
        normalized = str(model_name).strip().lower().replace(" ", "_")
        aliases = {
            "randomforest": "random_forest",
            "random-forest": "random_forest",
            "rf": "random_forest",
            "random_forest_voice_baseline": "random_forest",
            "xgb": "xgboost",
            "xg_boost": "xgboost",
            "xgboost_voice_classifier": "xgboost",
        }
        return aliases.get(normalized, normalized)

    def _get_model_display_name(self, model_name: str) -> str:
        model_key = self._normalize_model_name(model_name)
        for entry in self.model_registry:
            if entry["key"] == model_key:
                return str(entry.get("display_name", model_key))
        return model_key

    def _find_label_column(self, headers: Iterable[str]) -> Optional[str]:
        lower_to_original = {header.lower(): header for header in headers}
        for candidate in LABEL_COLUMN_CANDIDATES:
            if candidate in lower_to_original:
                return lower_to_original[candidate]
        return None

    def _encode_label_series(self, series: pd.Series) -> Optional[np.ndarray]:
        cleaned = series.dropna()
        if cleaned.nunique() != 2:
            return None

        if pd.api.types.is_numeric_dtype(series):
            numeric_values = sorted(cleaned.astype(float).unique().tolist())
            mapping = {numeric_values[0]: 0, numeric_values[-1]: 1}
            return series.astype(float).map(mapping).fillna(0).astype(int).to_numpy()

        unique_values = list(cleaned.astype(str).unique())
        negative_value = None
        positive_value = None
        for raw_value in unique_values:
            normalized = self._normalize_label_text(raw_value)
            if any(token in normalized for token in NEGATIVE_LABEL_TOKENS):
                negative_value = raw_value
            if any(token in normalized for token in POSITIVE_LABEL_TOKENS):
                positive_value = raw_value

        if negative_value is None or positive_value is None:
            sorted_values = sorted(unique_values)
            negative_value, positive_value = sorted_values[0], sorted_values[1]

        mapping = {str(negative_value): 0, str(positive_value): 1}
        return series.astype(str).map(mapping).fillna(0).astype(int).to_numpy()

    def _normalize_label_text(self, value: Any) -> str:
        return "".join(character.lower() if character.isalnum() else " " for character in str(value))

    def _compute_batch_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray,
    ) -> dict[str, Optional[float]]:
        metrics: dict[str, Optional[float]] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "auc": None,
        }
        if len(np.unique(y_true)) == 2:
            metrics["auc"] = float(roc_auc_score(y_true, y_score))
        return metrics

    def get_dataset_template(self) -> Dict[str, Any]:
        return {
            "training_dataset": TRAINING_DATASET_ID,
            "required_features": self.expected_features,
            "sample_data": self.sample_data,
            "csv": self._build_template_csv(),
        }

    def _build_template_csv(self) -> str:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=[*self.expected_features, "class"])
        writer.writeheader()
        writer.writerow(
            {
                **{feature: self.sample_data.get(feature, 0.0) for feature in self.expected_features},
                "class": 1,
            }
        )
        return buffer.getvalue()

    def get_cross_dataset_summary(self) -> Dict[str, Any]:
        artifact_roots = [
            self.study_dir / "artifacts",
            self.project_root.parent / "parkinson_feature_study" / "artifacts",
        ]
        artifact_root = next((path for path in artifact_roots if path.exists()), None)
        if artifact_root is None:
            return {
                "training_dataset": TRAINING_DATASET_ID,
                "strict_policy": "External datasets must contain every deployed training feature.",
                "datasets": [],
            }

        comparison_path = artifact_root / "cross_dataset_comparison.csv"
        comparison_df = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
        datasets: list[dict[str, Any]] = []
        dataset_dirs = [path for path in artifact_root.iterdir() if path.is_dir()]
        for dataset_dir in sorted(dataset_dirs, key=lambda path: path.name.lower()):
            schema_path = dataset_dir / "processed" / "feature_schema.csv"
            features: set[str] = set()
            if schema_path.exists():
                schema_df = pd.read_csv(schema_path)
                if "feature_name" in schema_df.columns:
                    features = set(schema_df["feature_name"].astype(str).tolist())

            missing = [feature for feature in self.expected_features if feature not in features]
            overlap = len(self.expected_features) - len(missing)
            dataset_rows = (
                comparison_df[comparison_df["dataset_id"] == dataset_dir.name]
                if not comparison_df.empty and "dataset_id" in comparison_df.columns
                else pd.DataFrame()
            )
            best_row = None
            if not dataset_rows.empty and "mean_f1" in dataset_rows.columns:
                best_row = dataset_rows.sort_values("mean_f1", ascending=False).iloc[0].to_dict()

            datasets.append(
                {
                    "dataset_id": dataset_dir.name,
                    "feature_count": len(features),
                    "required_overlap": overlap,
                    "missing_required_count": len(missing),
                    "strict_compatible": len(missing) == 0,
                    "best_model": best_row.get("model_name") if best_row else None,
                    "best_f1": float(best_row["mean_f1"]) if best_row and pd.notna(best_row.get("mean_f1")) else None,
                    "note": (
                        "Direct strict evaluation is allowed."
                        if len(missing) == 0
                        else "Schema differs from pd_speech_features.csv; upload a matching CSV for direct testing."
                    ),
                }
            )

        return {
            "training_dataset": TRAINING_DATASET_ID,
            "strict_policy": "Models are trained on pd_speech_features.csv and external CSVs must include every required deployed feature.",
            "datasets": datasets,
        }

    def _friendly_feature_group(self, feature_name: str) -> str:
        lowered = feature_name.lower()
        if any(token in lowered for token in ("jitter", "shimmer", "pulse", "period", "ppe", "rpde", "dfa")):
            return "Voice stability"
        if any(token in lowered for token in ("energy", "tkeo", "intensity")):
            return "Energy contour"
        if any(token in lowered for token in ("mfcc", "delta", "log_energy", "coef")):
            return "Cepstral pattern"
        if any(token in lowered for token in ("tqwt", "entropy", "kurtosis", "skew", "value")):
            return "Wavelet texture"
        return "Other acoustic signals"

    def _feature_group_description(self, group_name: str) -> str:
        descriptions = {
            "Voice stability": "Perturbation and cycle-to-cycle variation in the voice signal.",
            "Energy contour": "Changes in speech energy, intensity, and short-time activity.",
            "Cepstral pattern": "Compact frequency-shape descriptors used in speech analysis.",
            "Wavelet texture": "Time-frequency texture captured from wavelet decompositions.",
            "Other acoustic signals": "Additional deployed predictors retained by feature selection.",
        }
        return descriptions.get(group_name, "Grouped deployed acoustic predictors.")

    def _get_explainer(self, model_key: str, model) -> shap.TreeExplainer:
        if model_key not in self.explainers:
            self.explainers[model_key] = shap.TreeExplainer(model)
        return self.explainers[model_key]

    def _extract_positive_class_values(self, shap_values: Any) -> np.ndarray:
        if isinstance(shap_values, list):
            if len(shap_values) < 2:
                raise ValueError("Expected binary classification SHAP values for two classes.")
            return np.asarray(shap_values[1], dtype=float)

        values = np.asarray(shap_values, dtype=float)
        if values.ndim == 1:
            return values.reshape(1, -1)
        if values.ndim == 2:
            return values
        if values.ndim == 3:
            if values.shape[-1] == 2:
                return values[:, :, 1]
            if values.shape[0] == 2:
                return values[1]
        raise ValueError(f"Unsupported SHAP output shape: {values.shape}")

    def _extract_positive_class_base_value(self, expected_value: Any) -> float:
        if isinstance(expected_value, list):
            if len(expected_value) < 2:
                raise ValueError("Expected binary classification base values for two classes.")
            return float(expected_value[1])

        values = np.asarray(expected_value, dtype=float)
        if values.ndim == 0:
            return float(values)
        if values.shape[0] == 2:
            return float(values[1])
        return float(values.reshape(-1)[0])
