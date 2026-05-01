from __future__ import annotations

import io
import json
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
STUDY_ROOT = REPO_ROOT / "parkinson_feature_study"
LOCAL_PACKAGES = STUDY_ROOT / ".python_packages"
ARTIFACT_ROOT = STUDY_ROOT / "artifacts"

for path in (LOCAL_PACKAGES, STUDY_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


class SchemaValidationError(ValueError):
    """Raised when uploaded prediction data does not match a dataset schema."""


@dataclass(frozen=True)
class ModelRecord:
    dataset_id: str
    dataset_source: str
    model_name: str
    model_family: str
    artifact_path: Path
    feature_schema_path: Path
    label_map_path: Path | None
    manifest_inference_ready: bool
    metrics: dict[str, float | str | int | None]

    @property
    def model_key(self) -> str:
        return f"{self.dataset_id}_{self.model_name}"

    @property
    def display_family(self) -> str:
        if self.model_family == "quantum":
            return "Hybrid Quantum"
        return "Classical"

    @property
    def status(self) -> str:
        if self.model_name == "VQC" and self.artifact_path.suffix.lower() == ".json":
            return "repaired"
        if self.manifest_inference_ready:
            return "ready"
        return "unavailable"

    @property
    def inference_ready(self) -> bool:
        return self.manifest_inference_ready or self.status == "repaired"

    def to_api(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "dataset_id": self.dataset_id,
            "dataset_source": self.dataset_source,
            "model_name": self.model_name,
            "model_family": self.model_family,
            "display_family": self.display_family,
            "inference_ready": self.inference_ready,
            "status": self.status,
            "metrics": self.metrics,
        }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _as_float(value: Any) -> float | None:
    try:
        if pd.isna(value) or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_path(value: Any) -> Path | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    return Path(str(value))


def _positive_label(probability: float) -> str:
    return "Parkinson's (PD)" if probability >= 0.5 else "Healthy Control"


class ArtifactRegistry:
    """Scans and serves trained Parkinson model artifacts."""

    def __init__(self, artifact_root: Path = ARTIFACT_ROOT):
        self.artifact_root = Path(artifact_root)
        self.records = self._load_records()

    @classmethod
    def create_default(cls) -> "ArtifactRegistry":
        return cls()

    def _load_records(self) -> dict[str, ModelRecord]:
        records: dict[str, ModelRecord] = {}
        for manifest_path in sorted(self.artifact_root.glob("*/model_manifest.csv")):
            manifest = pd.read_csv(manifest_path)
            for _, row in manifest.iterrows():
                artifact_path = _resolve_path(row.get("artifact_path"))
                schema_path = _resolve_path(row.get("feature_schema_path"))
                if artifact_path is None or schema_path is None:
                    continue

                record = ModelRecord(
                    dataset_id=str(row["dataset_id"]),
                    dataset_source=str(row.get("dataset_source", "")),
                    model_name=str(row["model_name"]),
                    model_family=str(row["model_family"]),
                    artifact_path=artifact_path,
                    feature_schema_path=schema_path,
                    label_map_path=_resolve_path(row.get("label_map_path")),
                    manifest_inference_ready=_as_bool(row.get("inference_ready")),
                    metrics={
                        "cv_folds": _as_float(row.get("cv_folds")),
                        "mean_accuracy": _as_float(row.get("mean_accuracy")),
                        "mean_recall": _as_float(row.get("mean_recall")),
                        "mean_f1": _as_float(row.get("mean_f1")),
                        "mean_roc_auc": _as_float(row.get("mean_roc_auc")),
                        "fit_strategy": str(row.get("fit_strategy", "")),
                        "fit_sample_count": _as_float(row.get("fit_sample_count")),
                    },
                )
                records[record.model_key] = record
        return records

    def list_models(self) -> list[dict[str, Any]]:
        return [record.to_api() for record in sorted(self.records.values(), key=lambda item: item.model_key)]

    def get_model(self, model_key: str) -> ModelRecord:
        try:
            return self.records[model_key]
        except KeyError as exc:
            raise KeyError(f"Unknown model key: {model_key}") from exc

    def dataset_ids(self) -> list[str]:
        return sorted({record.dataset_id for record in self.records.values()})

    def dataset_dir(self, dataset_id: str) -> Path:
        path = self.artifact_root / dataset_id
        if not path.exists():
            raise KeyError(f"Unknown dataset id: {dataset_id}")
        return path

    @lru_cache(maxsize=16)
    def feature_names(self, dataset_id: str) -> tuple[str, ...]:
        schema_path = self.dataset_dir(dataset_id) / "processed" / "feature_schema.csv"
        schema = pd.read_csv(schema_path)
        return tuple(schema["feature_name"].astype(str).tolist())

    def cleaned_features_path(self, dataset_id: str) -> Path:
        return self.dataset_dir(dataset_id) / "processed" / "cleaned_features.csv"

    def load_sample_rows(self, dataset_id: str, limit: int = 5) -> dict[str, Any]:
        feature_names = list(self.feature_names(dataset_id))
        columns = feature_names + (["class"] if dataset_id == "pd_speech_features_local" else [])
        frame = pd.read_csv(self.cleaned_features_path(dataset_id), usecols=lambda name: name in set(columns))
        rows = []
        for index, row in frame.head(limit).iterrows():
            features = {feature: self._clean_number(row[feature]) for feature in feature_names}
            rows.append(
                {
                    "row_index": int(index),
                    "label": int(row["class"]) if "class" in frame.columns and not pd.isna(row["class"]) else None,
                    "features": features,
                }
            )
        return {
            "dataset_id": dataset_id,
            "feature_names": feature_names,
            "sample_count": int(sum(1 for _ in open(self.cleaned_features_path(dataset_id), encoding="utf-8")) - 1),
            "rows": rows,
        }

    def parse_csv(self, contents: bytes) -> pd.DataFrame:
        text = contents.decode("utf-8-sig")
        return pd.read_csv(io.StringIO(text))

    def validate_prediction_frame(self, dataset_id: str, frame: pd.DataFrame) -> pd.DataFrame:
        feature_names = list(self.feature_names(dataset_id))
        expected = set(feature_names)
        actual = set(frame.columns.astype(str))
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing columns: {', '.join(missing[:8])}")
            if extra:
                parts.append(f"unexpected columns: {', '.join(extra[:8])}")
            raise SchemaValidationError("; ".join(parts))

        validated = frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
        return validated

    def predict(self, model_key: str, frame: pd.DataFrame, source: str = "sample") -> list[dict[str, Any]]:
        record = self.get_model(model_key)
        validated = self.validate_prediction_frame(record.dataset_id, frame)
        model = self.load_model(model_key)
        probabilities = self._predict_probabilities(model, validated)
        rows = []
        for index, probability in enumerate(probabilities):
            rows.append(
                {
                    "row_index": int(index),
                    "source": source,
                    "model_key": model_key,
                    "probability": float(probability),
                    "predicted_label": _positive_label(float(probability)),
                }
            )
        return rows

    @lru_cache(maxsize=32)
    def load_model(self, model_key: str) -> Any:
        record = self.get_model(model_key)
        if not record.inference_ready:
            raise RuntimeError(f"Model is not inference-ready: {model_key}")
        if record.model_name == "VQC" and record.artifact_path.suffix.lower() == ".json":
            return self._load_repaired_vqc(record.artifact_path)

        import joblib

        return joblib.load(record.artifact_path)

    def _predict_probabilities(self, model: Any, frame: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(frame), dtype=float)
            if probabilities.ndim == 2:
                return probabilities[:, 1]
            return probabilities
        scores = np.asarray(model.predict(frame), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))

    def _load_repaired_vqc(self, path: Path) -> Any:
        from qiskit_machine_learning.optimizers import OptimizerResult
        from sklearn.decomposition import PCA
        from src.multi_dataset_pipeline import QuantumClassifierBundle

        state = json.loads(path.read_text(encoding="utf-8"))
        bundle = QuantumClassifierBundle(
            state["model_name"],
            random_seed=state["random_seed"],
            variance_threshold=state["variance_threshold"],
            max_qubits=state["max_qubits"],
            vqc_maxiter=state["vqc_maxiter"],
            qsvm_num_steps=state["qsvm_num_steps"],
        )
        bundle.n_components_ = int(state["n_components"])
        preprocessing = state["preprocessing"]

        bundle.imputer.statistics_ = np.asarray(preprocessing["imputer_statistics"], dtype=float)
        bundle.imputer.n_features_in_ = len(bundle.imputer.statistics_)
        bundle.imputer._fit_dtype = np.dtype(float)
        bundle.imputer._fill_dtype = np.dtype(float)
        bundle.imputer.indicator_ = None

        bundle.scaler.mean_ = np.asarray(preprocessing["scaler_mean"], dtype=float)
        bundle.scaler.scale_ = np.asarray(preprocessing["scaler_scale"], dtype=float)
        bundle.scaler.var_ = bundle.scaler.scale_**2
        bundle.scaler.n_features_in_ = len(bundle.scaler.mean_)
        bundle.scaler.n_samples_seen_ = np.ones(len(bundle.scaler.mean_), dtype=int)

        bundle.minmax_scaler.min_ = np.asarray(preprocessing["minmax_min"], dtype=float)
        bundle.minmax_scaler.scale_ = np.asarray(preprocessing["minmax_scale"], dtype=float)
        bundle.minmax_scaler.n_features_in_ = len(bundle.minmax_scaler.min_)
        bundle.minmax_scaler.n_samples_seen_ = 1
        bundle.minmax_scaler.data_min_ = -bundle.minmax_scaler.min_ / bundle.minmax_scaler.scale_
        bundle.minmax_scaler.data_range_ = np.where(
            bundle.minmax_scaler.scale_ != 0,
            np.pi / bundle.minmax_scaler.scale_,
            1.0,
        )
        bundle.minmax_scaler.data_max_ = bundle.minmax_scaler.data_min_ + bundle.minmax_scaler.data_range_

        bundle.pca = PCA(n_components=bundle.n_components_, random_state=bundle.random_seed)
        bundle.pca.components_ = np.asarray(preprocessing["pca_components"], dtype=float)
        bundle.pca.mean_ = np.asarray(preprocessing["pca_mean"], dtype=float)
        bundle.pca.explained_variance_ = np.asarray(preprocessing["pca_explained_variance"], dtype=float)
        bundle.pca.explained_variance_ratio_ = np.asarray(
            preprocessing["pca_explained_variance_ratio"], dtype=float
        )
        bundle.pca.n_components_ = bundle.n_components_
        bundle.pca.n_features_in_ = bundle.pca.components_.shape[1]
        bundle.pca.n_samples_ = 1
        bundle.pca.singular_values_ = np.sqrt(bundle.pca.explained_variance_)

        bundle.model = bundle._build_model()
        fit_result = OptimizerResult()
        fit_result.x = np.asarray(state["model_state"]["weights"], dtype=float)
        bundle.model._fit_result = fit_result
        return bundle

    def feature_importance(self, model_key: str, limit: int = 15) -> list[dict[str, Any]]:
        record = self.get_model(model_key)
        model = self.load_model(model_key)
        feature_names = list(self.feature_names(record.dataset_id))
        estimator = self._final_estimator(model)

        values: np.ndarray | None = None
        if hasattr(estimator, "feature_importances_"):
            values = np.asarray(estimator.feature_importances_, dtype=float)
        elif hasattr(estimator, "coef_"):
            values = np.abs(np.asarray(estimator.coef_, dtype=float)).reshape(-1)

        if values is None or len(values) != len(feature_names):
            values = self._fallback_importance(model_key, feature_names)

        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        total = float(values.sum())
        if total > 0:
            values = values / total
        order = np.argsort(values)[::-1][:limit]
        return [
            {"feature": feature_names[index], "importance": float(values[index])}
            for index in order
        ]

    def _final_estimator(self, model: Any) -> Any:
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            return model.named_steps["model"]
        return model

    def _fallback_importance(self, model_key: str, feature_names: list[str]) -> np.ndarray:
        record = self.get_model(model_key)
        sample = pd.read_csv(self.cleaned_features_path(record.dataset_id), usecols=feature_names).head(12)
        sample = self.validate_prediction_frame(record.dataset_id, sample)
        variance = sample.var(axis=0).fillna(0.0).to_numpy(dtype=float)
        candidate_count = min(12, len(feature_names))
        candidate_indices = np.argsort(variance)[::-1][:candidate_count]
        baseline = self._predict_probabilities(self.load_model(model_key), sample)
        values = np.zeros(len(feature_names), dtype=float)
        rng = np.random.default_rng(42)

        for index in candidate_indices:
            permuted = sample.copy()
            shuffled = permuted.iloc[:, index].to_numpy(copy=True)
            rng.shuffle(shuffled)
            permuted.iloc[:, index] = shuffled
            changed = self._predict_probabilities(self.load_model(model_key), permuted)
            values[index] = float(np.mean(np.abs(changed - baseline)))

        if not np.any(values):
            values = variance
        return values

    def group_impact(self, model_key: str) -> dict[str, Any]:
        record = self.get_model(model_key)
        predictions_path = self.dataset_dir(record.dataset_id) / "results" / "oof_predictions.csv"
        oof = pd.read_csv(predictions_path)
        oof = oof[oof["model_name"] == record.model_name].copy()
        thresholds = np.linspace(0.0, 1.0, 21)
        series = []
        for threshold in thresholds:
            predicted_positive = oof["y_score"] >= threshold
            for truth, group in ((0, "Healthy Control"), (1, "Parkinson's (PD)")):
                group_mask = oof["y_true"] == truth
                n = int(group_mask.sum())
                rate = float(predicted_positive[group_mask].mean()) if n else 0.0
                series.append(
                    {
                        "threshold": round(float(threshold), 2),
                        "group": group,
                        "positive_rate": rate,
                        "n": n,
                    }
                )

        summary = self._impact_summary(oof, threshold=0.5)
        return {"model_key": model_key, "series": series, "summary": summary}

    def _impact_summary(self, oof: pd.DataFrame, threshold: float) -> list[dict[str, Any]]:
        predicted_positive = oof["y_score"] >= threshold
        rows = []
        rates: dict[int, float] = {}
        for truth, group in ((0, "Healthy Control"), (1, "Parkinson's (PD)")):
            mask = oof["y_true"] == truth
            n = int(mask.sum())
            rate = float(predicted_positive[mask].mean()) if n else 0.0
            rates[truth] = rate
            low, high = self._normal_ci(rate, n)
            rows.append(
                {
                    "group": group,
                    "positive_rate": rate,
                    "ci_low": low,
                    "ci_high": high,
                    "n": n,
                }
            )
        rows.append(
            {
                "group": "Delta (PD - HC)",
                "positive_rate": rates.get(1, 0.0) - rates.get(0, 0.0),
                "ci_low": None,
                "ci_high": None,
                "n": None,
            }
        )
        return rows

    def _normal_ci(self, rate: float, n: int) -> tuple[float, float]:
        if n <= 0:
            return 0.0, 0.0
        margin = 1.96 * math.sqrt(max(rate * (1.0 - rate), 0.0) / n)
        return max(0.0, rate - margin), min(1.0, rate + margin)

    def dashboard(self) -> dict[str, Any]:
        comparison_path = self.artifact_root / "cross_dataset_comparison.csv"
        comparison = pd.read_csv(comparison_path)
        rows = []
        for _, row in comparison.iterrows():
            key = f"{row['dataset_id']}_{row['model_name']}"
            record = self.records.get(key)
            rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "dataset_source": row["dataset_source"],
                    "model_key": key,
                    "model_name": row["model_name"],
                    "model_family": row["model_family"],
                    "display_family": "Hybrid Quantum" if row["model_family"] == "quantum" else "Classical",
                    "inference_ready": record.inference_ready if record else _as_bool(row.get("inference_ready")),
                    "status": record.status if record else "unknown",
                    "mean_accuracy": _as_float(row.get("mean_accuracy")),
                    "mean_recall": _as_float(row.get("mean_recall")),
                    "mean_f1": _as_float(row.get("mean_f1")),
                    "mean_roc_auc": _as_float(row.get("mean_roc_auc")),
                }
            )

        datasets = []
        for dataset_id in self.dataset_ids():
            dataset_records = [record for record in self.records.values() if record.dataset_id == dataset_id]
            ready = sum(1 for record in dataset_records if record.inference_ready)
            datasets.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_source": dataset_records[0].dataset_source,
                    "model_count": len(dataset_records),
                    "ready_count": ready,
                    "feature_count": len(self.feature_names(dataset_id)),
                }
            )
        return {"comparison": rows, "datasets": datasets}

    def _clean_number(self, value: Any) -> float | int | None:
        if pd.isna(value):
            return None
        number = float(value)
        if number.is_integer():
            return int(number)
        return number
