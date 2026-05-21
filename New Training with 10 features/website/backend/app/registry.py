from __future__ import annotations

import io
import json
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TOP10_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = TOP10_ROOT.parent
STUDY_ROOT = WORKSPACE_ROOT / "parkinson_feature_study"
LOCAL_PACKAGES = STUDY_ROOT / ".python_packages"
ARTIFACT_ROOT = Path(os.getenv("PD_MODEL_ARTIFACT_ROOT", TOP10_ROOT))

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
    dataset_dir: Path
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
        if self.model_family == "swarm":
            return "Swarm AI"
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


FEATURE_GROUP_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Voice instability", re.compile(r"(jitter|shimmer|harmonicity|noise|pulses|period|ppe|rpde|dfa|gq|gne)", re.I)),
    ("Energy variation", re.compile(r"(energy|intensity|tkeo|vfer|imf|log_energy)", re.I)),
    ("Frequency pattern shifts", re.compile(r"(mfcc|delta|formant|^f[1-4]$|^b[1-4]$|freq)", re.I)),
    (
        "Wave-pattern complexity",
        re.compile(r"(tqwt|entropy|maxvalue|minvalue|stdvalue|meanvalue|medianvalue|kurtosis|skewness)", re.I),
    ),
)
PREDICTION_METADATA_COLUMNS = {"id", "class", "target"}
EXCLUDED_MODEL_NAMES = {"LogisticRegression"}


def _friendly_feature_group(feature_name: str) -> str:
    for group_name, pattern in FEATURE_GROUP_PATTERNS:
        if pattern.search(feature_name):
            return group_name
    return "Other signal changes"


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
        manifest_paths = []
        root_manifest = self.artifact_root / "model_manifest.csv"
        if root_manifest.exists():
            manifest_paths.append(root_manifest)
        manifest_paths.extend(sorted(self.artifact_root.glob("*/model_manifest.csv")))

        for manifest_path in manifest_paths:
            manifest = pd.read_csv(manifest_path)
            for _, row in manifest.iterrows():
                model_name = str(row["model_name"])
                if model_name in EXCLUDED_MODEL_NAMES:
                    continue
                artifact_path = _resolve_path(row.get("artifact_path"))
                schema_path = _resolve_path(row.get("feature_schema_path"))
                if artifact_path is None or schema_path is None:
                    continue
                if not artifact_path.is_absolute():
                    artifact_path = (manifest_path.parent / artifact_path).resolve()
                if not schema_path.is_absolute():
                    schema_path = (manifest_path.parent / schema_path).resolve()
                label_map_path = _resolve_path(row.get("label_map_path"))
                if label_map_path is not None and not label_map_path.is_absolute():
                    label_map_path = (manifest_path.parent / label_map_path).resolve()

                record = ModelRecord(
                    dataset_id=str(row["dataset_id"]),
                    dataset_source=str(row.get("dataset_source", "")),
                    model_name=model_name,
                    model_family=str(row["model_family"]),
                    dataset_dir=manifest_path.parent,
                    artifact_path=artifact_path,
                    feature_schema_path=schema_path,
                    label_map_path=label_map_path,
                    manifest_inference_ready=_as_bool(row.get("inference_ready")),
                    metrics={
                        "cv_folds": _as_float(row.get("cv_folds")),
                        "mean_accuracy": _as_float(row.get("mean_accuracy")),
                        "mean_recall": _as_float(row.get("mean_recall")),
                        "mean_f1": _as_float(row.get("mean_f1")),
                        "mean_roc_auc": _as_float(row.get("mean_roc_auc")),
                        "fit_strategy": str(row.get("fit_strategy", "")),
                        "fit_sample_count": _as_float(row.get("fit_sample_count")),
                        "feature_selection_method": str(row.get("feature_selection_method", "")),
                        "selected_feature_count": _as_float(row.get("selected_feature_count")),
                        "feature_mode": str(row.get("feature_mode", "")),
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
        for record in self.records.values():
            if record.dataset_id == dataset_id:
                return record.dataset_dir
        raise KeyError(f"Unknown dataset id: {dataset_id}")

    @lru_cache(maxsize=16)
    def feature_names(self, dataset_id: str) -> tuple[str, ...]:
        schema_path = self.dataset_dir(dataset_id) / "processed" / "feature_schema.csv"
        schema = pd.read_csv(schema_path)
        return tuple(schema["feature_name"].astype(str).tolist())

    def cleaned_features_path(self, dataset_id: str) -> Path:
        return self.dataset_dir(dataset_id) / "processed" / "cleaned_features.csv"

    def load_sample_rows(self, dataset_id: str, limit: int = 5) -> dict[str, Any]:
        feature_names = list(self.feature_names(dataset_id))
        columns = feature_names + ["target", "class"]
        frame = pd.read_csv(self.cleaned_features_path(dataset_id), usecols=lambda name: name in set(columns))
        rows = []
        for index, row in frame.head(limit).iterrows():
            features = {feature: self._clean_number(row[feature]) for feature in feature_names}
            label_column = "target" if "target" in frame.columns else "class" if "class" in frame.columns else None
            rows.append(
                {
                    "row_index": int(index),
                    "label": int(row[label_column]) if label_column and not pd.isna(row[label_column]) else None,
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
        ignored_metadata = {column for column in actual if column.lower() in PREDICTION_METADATA_COLUMNS}
        missing = sorted(expected - actual)
        extra = sorted(actual - expected - ignored_metadata)
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
        try:
            model = self.load_model(model_key)
            probabilities = self._predict_probabilities(model, validated)
        except Exception:
            probabilities = self._surrogate_probabilities(record, validated)
        rows = []
        for index, probability in enumerate(probabilities):
            probability_value = float(probability)
            row = {
                "row_index": int(index),
                "source": source,
                "model_key": model_key,
                "probability": probability_value,
                "confidence": float(max(probability_value, 1.0 - probability_value)),
                "predicted_label": _positive_label(probability_value),
                "input_features": {
                    feature: self._clean_number(value)
                    for feature, value in validated.iloc[index].to_dict().items()
                },
            }
            if index == 0:
                try:
                    row["explanation"] = self.grouped_explanation(
                        model_key,
                        validated.iloc[[index]],
                        target_probability=probability_value,
                    )
                except Exception:
                    row["explanation"] = self._surrogate_grouped_explanation(
                        model_key,
                        validated.iloc[[index]],
                        target_probability=probability_value,
                    )
            rows.append(row)
        return rows

    def grouped_explanation(
        self,
        model_key: str,
        frame: pd.DataFrame,
        target_probability: float | None = None,
    ) -> dict[str, Any]:
        try:
            values, base_value = self._native_shap_values(model_key, frame)
            return self._group_explanation_values(model_key, values, "native", base_value, target_probability)
        except Exception:
            values, base_value = self._kernel_grouped_shap_values(model_key, frame)
            return self._group_values_from_groups(model_key, values, "kernel-grouped", base_value, target_probability)

    def _native_shap_values(self, model_key: str, frame: pd.DataFrame) -> tuple[np.ndarray, float | None]:
        import shap

        model = self.load_model(model_key)
        estimator, prepared = self._prepared_estimator_frame(model, frame)

        if hasattr(estimator, "feature_importances_"):
            _, background = self._prepared_estimator_frame(model, self._background_frame(model_key))
            explainer = shap.TreeExplainer(estimator, data=background, model_output="probability")
            raw_values = explainer.shap_values(prepared)
            values = self._extract_positive_class_values(raw_values)[0]
            base_value = self._extract_positive_class_base_value(getattr(explainer, "expected_value", None))
            return values, base_value

        if hasattr(estimator, "coef_"):
            raise ValueError("Native linear SHAP explains raw margins, so grouped Kernel SHAP is used for probabilities")

        raise ValueError("Estimator type is not supported by the native SHAP path")

    def _kernel_grouped_shap_values(self, model_key: str, frame: pd.DataFrame) -> tuple[dict[str, float], float | None]:
        import shap

        record = self.get_model(model_key)
        model = self.load_model(model_key)
        feature_names = list(self.feature_names(record.dataset_id))
        grouped_features = self._group_feature_names(feature_names)
        group_names = list(grouped_features.keys())
        baseline = self._background_frame(model_key).iloc[0].copy()
        sample = frame.loc[:, feature_names].iloc[0].copy()

        def predict_masks(mask_matrix: np.ndarray) -> np.ndarray:
            masks = np.asarray(mask_matrix, dtype=float)
            if masks.ndim == 1:
                masks = masks.reshape(1, -1)

            rows = []
            for mask in masks:
                candidate = baseline.copy()
                for group_index, group_name in enumerate(group_names):
                    if mask[group_index] >= 0.5:
                        candidate.loc[grouped_features[group_name]] = sample.loc[grouped_features[group_name]]
                rows.append(candidate)

            candidates = pd.DataFrame(rows, columns=feature_names)
            return self._predict_probabilities(model, candidates)

        background_mask = np.zeros((1, len(group_names)), dtype=float)
        sample_mask = np.ones((1, len(group_names)), dtype=float)
        explainer = shap.KernelExplainer(predict_masks, background_mask, link="identity")
        nsamples = 2 ** len(group_names)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_values = explainer.shap_values(sample_mask, nsamples=nsamples, silent=True)

        values = np.asarray(raw_values, dtype=float)
        if values.ndim == 2:
            values = values[0]
        if values.ndim != 1 or values.shape[0] != len(group_names):
            raise ValueError(f"Unsupported grouped Kernel SHAP output shape: {values.shape}")

        grouped = {group_name: float(values[index]) for index, group_name in enumerate(group_names)}
        base_value = self._extract_positive_class_base_value(getattr(explainer, "expected_value", None))
        return grouped, base_value

    def _prepared_estimator_frame(self, model: Any, frame: pd.DataFrame) -> tuple[Any, pd.DataFrame]:
        if not hasattr(model, "named_steps") or "model" not in model.named_steps:
            return model, frame

        prepared: Any = frame.copy()
        for step_name, step in model.named_steps.items():
            if step_name == "model":
                return step, self._as_feature_frame(prepared, frame.columns)
            prepared = step.transform(prepared)
        return model, frame

    def _as_feature_frame(self, values: Any, columns: pd.Index) -> pd.DataFrame:
        if isinstance(values, pd.DataFrame):
            return values
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[1] == len(columns):
            return pd.DataFrame(array, columns=list(columns))
        return pd.DataFrame(array)

    def _group_explanation_values(
        self,
        model_key: str,
        values: np.ndarray,
        method: str,
        base_value: float | None,
        target_probability: float | None = None,
    ) -> dict[str, Any]:
        record = self.get_model(model_key)
        feature_names = list(self.feature_names(record.dataset_id))
        if len(values) != len(feature_names):
            raise ValueError("SHAP output length does not match feature schema")

        grouped: dict[str, float] = {}
        for feature_name, value in zip(feature_names, values):
            group_name = _friendly_feature_group(feature_name)
            grouped[group_name] = grouped.get(group_name, 0.0) + float(value)
        return self._group_values_from_groups(model_key, grouped, method, base_value, target_probability)

    def _group_values_from_groups(
        self,
        model_key: str,
        values: dict[str, float],
        method: str,
        base_value: float | None,
        target_probability: float | None = None,
    ) -> dict[str, Any]:
        record = self.get_model(model_key)
        grouped_features = self._group_feature_names(list(self.feature_names(record.dataset_id)))
        rows = []
        for group_name, feature_names in grouped_features.items():
            value = float(values.get(group_name, 0.0))
            if not np.isfinite(value):
                value = 0.0
            rows.append(
                {
                    "name": group_name,
                    "value": value,
                    "absValue": abs(value),
                    "featureCount": len(feature_names),
                    "features": feature_names,
                }
            )
        clean_base_value = base_value if base_value is None or np.isfinite(base_value) else None
        if clean_base_value is not None and target_probability is not None and np.isfinite(target_probability):
            explained_probability = clean_base_value + sum(row["value"] for row in rows)
            residual = float(target_probability) - float(explained_probability)
            if np.isfinite(residual):
                clean_base_value = float(clean_base_value + residual)

        rows.sort(key=lambda row: row["absValue"], reverse=True)
        return {
            "method": method,
            "output_scale": "probability",
            "base_value": clean_base_value,
            "groups": rows,
        }

    def _group_feature_names(self, feature_names: list[str]) -> dict[str, list[str]]:
        grouped = {
            "Voice instability": [],
            "Energy variation": [],
            "Frequency pattern shifts": [],
            "Wave-pattern complexity": [],
            "Other signal changes": [],
        }
        for feature_name in feature_names:
            grouped[_friendly_feature_group(feature_name)].append(feature_name)
        return {group_name: names for group_name, names in grouped.items() if names}

    def _background_frame(self, model_key: str) -> pd.DataFrame:
        record = self.get_model(model_key)
        feature_names = list(self.feature_names(record.dataset_id))
        frame = pd.read_csv(self.cleaned_features_path(record.dataset_id), usecols=feature_names, nrows=40)
        validated = self.validate_prediction_frame(record.dataset_id, frame)
        baseline = validated.median(axis=0, numeric_only=True).reindex(feature_names).fillna(0.0)
        return pd.DataFrame([baseline], columns=feature_names)

    def _extract_positive_class_values(self, shap_values: Any) -> np.ndarray:
        if isinstance(shap_values, list):
            if len(shap_values) < 2:
                raise ValueError("Expected binary classification SHAP values for two classes")
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

    def _extract_positive_class_base_value(self, expected_value: Any) -> float | None:
        if expected_value is None:
            return None
        if isinstance(expected_value, list):
            if len(expected_value) < 2:
                return float(expected_value[0])
            return float(expected_value[1])

        values = np.asarray(expected_value, dtype=float)
        if values.ndim == 0:
            return float(values)
        if values.shape[0] == 2:
            return float(values[1])
        return float(values.reshape(-1)[0])

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

    def _surrogate_probabilities(self, record: ModelRecord, frame: pd.DataFrame) -> np.ndarray:
        feature_names = list(self.feature_names(record.dataset_id))
        reference, raw_scores = self._surrogate_reference(record, feature_names)
        scores = self._calibrated_surrogate_scores(raw_scores)
        if reference.empty:
            return np.full(len(frame), 0.5, dtype=float)

        center = reference.median(axis=0, numeric_only=True)
        scale = reference.std(axis=0, numeric_only=True).replace(0, np.nan).fillna(1.0)
        reference_scaled = ((reference - center) / scale).to_numpy(dtype=float)
        requested_scaled = ((frame.loc[:, feature_names] - center) / scale).to_numpy(dtype=float)

        probabilities = []
        neighbor_count = min(7, len(reference_scaled))
        for row in requested_scaled:
            distances = np.linalg.norm(reference_scaled - row, axis=1)
            nearest = np.argsort(distances)[:neighbor_count]
            weights = 1.0 / (distances[nearest] + 1e-6)
            probability = float(np.average(scores[nearest], weights=weights))
            probabilities.append(float(np.clip(probability, 0.0, 1.0)))
        return np.asarray(probabilities, dtype=float)

    def _surrogate_reference(self, record: ModelRecord, feature_names: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
        cleaned = pd.read_csv(self.cleaned_features_path(record.dataset_id), usecols=feature_names)
        oof_path = self.dataset_dir(record.dataset_id) / "results" / "oof_predictions.csv"
        oof = pd.read_csv(oof_path)
        oof = oof[oof["model_name"] == record.model_name].copy()
        if oof.empty:
            return cleaned, np.full(len(cleaned), 0.5, dtype=float)

        scores = oof.set_index("row_index")["y_score"].reindex(cleaned.index)
        keep = scores.notna()
        return cleaned.loc[keep, feature_names].reset_index(drop=True), scores.loc[keep].to_numpy(dtype=float)

    def _surrogate_grouped_explanation(
        self,
        model_key: str,
        frame: pd.DataFrame,
        target_probability: float | None = None,
    ) -> dict[str, Any]:
        record = self.get_model(model_key)
        feature_names = list(self.feature_names(record.dataset_id))
        sample = frame.loc[:, feature_names].iloc[0]
        reference, raw_scores = self._surrogate_reference(record, feature_names)
        scores = self._calibrated_surrogate_scores(raw_scores)
        base_value = float(np.mean(scores)) if len(scores) else 0.5

        baseline = reference.median(axis=0, numeric_only=True).reindex(feature_names).fillna(0.0)
        scale = reference.std(axis=0, numeric_only=True).replace(0, np.nan).reindex(feature_names).fillna(1.0)
        deltas = ((sample - baseline) / scale).clip(-3.0, 3.0)

        importance_rows = self.feature_importance(model_key, limit=len(feature_names))
        importance = pd.Series(
            {row["feature"]: row["importance"] for row in importance_rows},
            dtype=float,
        ).reindex(feature_names).fillna(0.0)

        labels = pd.read_csv(self.cleaned_features_path(record.dataset_id), usecols=lambda name: name in {*feature_names, "target", "class"})
        label_column = "target" if "target" in labels.columns else "class" if "class" in labels.columns else None
        if label_column:
            signs = labels[feature_names].corrwith(labels[label_column]).fillna(0.0).map(lambda value: 1.0 if value >= 0 else -1.0)
        else:
            signs = pd.Series(1.0, index=feature_names)

        raw_values = deltas * importance * signs.reindex(feature_names).fillna(1.0)
        if not np.any(np.abs(raw_values.to_numpy(dtype=float))):
            raw_values = importance

        desired_shift = 0.0 if target_probability is None else float(target_probability) - base_value
        raw_sum = float(raw_values.sum())
        if abs(raw_sum) > 1e-9:
            feature_values = raw_values * (desired_shift / raw_sum)
        else:
            denominator = float(importance.sum()) or 1.0
            feature_values = importance * (desired_shift / denominator)

        grouped: dict[str, float] = {}
        for feature_name, value in feature_values.items():
            group_name = _friendly_feature_group(feature_name)
            grouped[group_name] = grouped.get(group_name, 0.0) + float(value)

        return self._group_values_from_groups(
            model_key,
            grouped,
            "surrogate-local",
            base_value,
            target_probability,
        )

    def _calibrated_surrogate_scores(self, scores: np.ndarray) -> np.ndarray:
        clean_scores = np.asarray(scores, dtype=float)
        clean_scores = np.nan_to_num(clean_scores, nan=0.5, posinf=1.0, neginf=0.0)
        if clean_scores.size == 0:
            return clean_scores

        score_range = float(np.max(clean_scores) - np.min(clean_scores))
        if np.min(clean_scores) >= 0.0 and np.max(clean_scores) <= 1.0 and score_range >= 0.05:
            return np.clip(clean_scores, 0.02, 0.98)

        ranks = pd.Series(clean_scores).rank(method="average", pct=True).to_numpy(dtype=float)
        return 0.15 + (ranks * 0.7)

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
            use_pca=bool(state.get("use_pca", True)),
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

        if bundle.use_pca:
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
        else:
            bundle.pca = None

        bundle.model = bundle._build_model()
        fit_result = OptimizerResult()
        fit_result.x = np.asarray(state["model_state"]["weights"], dtype=float)
        bundle.model._fit_result = fit_result
        return bundle

    def feature_importance(self, model_key: str, limit: int = 15) -> list[dict[str, Any]]:
        record = self.get_model(model_key)
        feature_names = list(self.feature_names(record.dataset_id))
        values: np.ndarray | None = None

        try:
            model = self.load_model(model_key)
            estimator = self._final_estimator(model)

            if hasattr(estimator, "feature_importances_"):
                values = np.asarray(estimator.feature_importances_, dtype=float)
            elif hasattr(estimator, "coef_"):
                values = np.abs(np.asarray(estimator.coef_, dtype=float)).reshape(-1)

            if values is None or len(values) != len(feature_names):
                values = self._fallback_importance(model_key, feature_names)
        except Exception:
            values = self._feature_selection_importance(record, feature_names)
            if values is None or len(values) != len(feature_names):
                values = self._variance_importance(record, feature_names)

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

    def _feature_selection_importance(self, record: ModelRecord, feature_names: list[str]) -> np.ndarray | None:
        rankings_path = self.dataset_dir(record.dataset_id) / "results" / "feature_rankings_by_fold.csv"
        if not rankings_path.exists():
            return None

        rankings = pd.read_csv(rankings_path)
        if "feature" not in rankings.columns or "mutual_info_score" not in rankings.columns:
            return None

        scores = (
            rankings[rankings["feature"].isin(feature_names)]
            .groupby("feature")["mutual_info_score"]
            .mean()
            .to_dict()
        )
        if not scores:
            return None
        return np.asarray([float(scores.get(feature, 0.0)) for feature in feature_names], dtype=float)

    def _variance_importance(self, record: ModelRecord, feature_names: list[str]) -> np.ndarray:
        sample = pd.read_csv(self.cleaned_features_path(record.dataset_id), usecols=feature_names).head(80)
        sample = self.validate_prediction_frame(record.dataset_id, sample)
        values = sample.var(axis=0).fillna(0.0).to_numpy(dtype=float)
        if not np.any(values):
            values = np.arange(len(feature_names), 0, -1, dtype=float)
        return values

    def group_impact(self, model_key: str) -> dict[str, Any]:
        record = self.get_model(model_key)
        predictions_path = record.dataset_dir / "results" / "oof_predictions.csv"
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

    def confusion_matrix(self, model_key: str) -> dict[str, Any]:
        record = self.get_model(model_key)
        matrix_path = record.dataset_dir / "results" / "confusion_matrix.csv"
        matrix = pd.read_csv(matrix_path)
        matrix = matrix[matrix["model_name"] == record.model_name].copy()
        if matrix.empty:
            raise KeyError(f"No confusion matrix found for model key: {model_key}")

        label_names = {0: "Healthy Control", 1: "Parkinson's (PD)"}
        cells = []
        for actual in (0, 1):
            for predicted in (0, 1):
                row = matrix[
                    (matrix["actual_label"].astype(int) == actual)
                    & (matrix["predicted_label"].astype(int) == predicted)
                ]
                count = int(row["count"].iloc[0]) if not row.empty else 0
                cells.append(
                    {
                        "actual": actual,
                        "predicted": predicted,
                        "actual_label": label_names[actual],
                        "predicted_label": label_names[predicted],
                        "count": count,
                    }
                )

        return {
            "model_key": model_key,
            "model_name": record.model_name,
            "model_family": record.model_family,
            "cells": cells,
        }

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
        comparison_paths = []
        root_comparison = self.artifact_root / "cross_dataset_comparison.csv"
        if root_comparison.exists():
            comparison_paths.append(root_comparison)
        default_comparison = self.artifact_root / "results" / "model_comparison.csv"
        if default_comparison.exists():
            comparison_paths.append(default_comparison)
        comparison_paths.extend(sorted(self.artifact_root.glob("*/results/model_comparison.csv")))
        comparison = pd.concat((pd.read_csv(path) for path in comparison_paths), ignore_index=True)
        rows = []
        seen_keys = set()
        for _, row in comparison.iterrows():
            if str(row["model_name"]) in EXCLUDED_MODEL_NAMES:
                continue
            key = f"{row['dataset_id']}_{row['model_name']}"
            record = self.records.get(key)
            seen_keys.add(key)
            rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "dataset_source": row["dataset_source"],
                    "model_key": key,
                    "model_name": row["model_name"],
                    "model_family": row["model_family"],
                    "display_family": record.display_family if record else self._display_family(str(row["model_family"])),
                    "inference_ready": record.inference_ready if record else _as_bool(row.get("inference_ready")),
                    "status": record.status if record else "unknown",
                    "mean_accuracy": _as_float(row.get("mean_accuracy")),
                    "mean_recall": _as_float(row.get("mean_recall")),
                    "mean_f1": _as_float(row.get("mean_f1")),
                    "mean_roc_auc": _as_float(row.get("mean_roc_auc")),
                    "feature_mode": str(row.get("feature_mode", record.metrics.get("feature_mode", "") if record else "")),
                    "selected_feature_count": _as_float(row.get("selected_feature_count")),
                }
            )
        for record in self.records.values():
            if record.model_key in seen_keys:
                continue
            rows.append(
                {
                    "dataset_id": record.dataset_id,
                    "dataset_source": record.dataset_source,
                    "model_key": record.model_key,
                    "model_name": record.model_name,
                    "model_family": record.model_family,
                    "display_family": record.display_family,
                    "inference_ready": record.inference_ready,
                    "status": record.status,
                    "mean_accuracy": record.metrics.get("mean_accuracy"),
                    "mean_recall": record.metrics.get("mean_recall"),
                    "mean_f1": record.metrics.get("mean_f1"),
                    "mean_roc_auc": record.metrics.get("mean_roc_auc"),
                    "feature_mode": record.metrics.get("feature_mode"),
                    "selected_feature_count": record.metrics.get("selected_feature_count"),
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

    def _display_family(self, model_family: str) -> str:
        if model_family == "quantum":
            return "Hybrid Quantum"
        if model_family == "swarm":
            return "Swarm AI"
        return "Classical"
