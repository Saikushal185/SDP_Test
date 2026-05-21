"""
Multi-dataset Parkinson speech training and artifact export pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from .artifacts import (
    build_dataset_artifact_layout,
    write_feature_schema,
    write_label_map,
    write_model_manifest,
)
from .audio_features import extract_acoustic_features
from .dataset_catalog import DatasetSpec, build_default_dataset_specs


logger = logging.getLogger(__name__)

TARGET_COLUMN_CANDIDATES = [
    "label",
    "labels",
    "class",
    "target",
    "status",
    "diagnosis",
    "parkinsons",
    "pd",
]
POSITIVE_LABEL_TOKENS = ("parkinson", "pd", "patient", "positive", "disease", "dys")
NEGATIVE_LABEL_TOKENS = ("healthy", "control", "hc", "normal", "negative", "con")
ID_LIKE_COLUMNS = {"id", "name", "file", "filename", "path", "recording_id"}
XANJEEV_SUBJECT_PATTERN = re.compile(r"([FM]C?\d{2}S\d{2}|[FM]\d{2})", re.IGNORECASE)


@dataclass
class PreparedDataset:
    spec: DatasetSpec
    resolved_source_ref: str
    source_type: str
    features: pd.DataFrame
    labels: np.ndarray
    label_map: Dict[int, str]
    feature_names: List[str]
    cv_folds: int


class QuantumClassifierBundle:
    """Leakage-safe quantum model wrapper with internal preprocessing."""

    def __init__(
        self,
        model_name: str,
        random_seed: int,
        variance_threshold: float = 0.90,
        max_qubits: int = 6,
        vqc_maxiter: int = 25,
        qsvm_num_steps: int = 100,
        use_pca: bool = True,
    ) -> None:
        self.model_name = model_name
        self.random_seed = random_seed
        self.variance_threshold = variance_threshold
        self.max_qubits = max_qubits
        self.vqc_maxiter = vqc_maxiter
        self.qsvm_num_steps = qsvm_num_steps
        self.use_pca = use_pca
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler(feature_range=(0.0, np.pi))
        self.pca: Optional[PCA] = None
        self.model: Any = None
        self.n_components_: Optional[int] = None

    def _lazy_import_qiskit(self) -> Dict[str, Any]:
        from qiskit.primitives import StatevectorSampler
        from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
        from qiskit_machine_learning.algorithms import PegasosQSVC, VQC
        from qiskit_machine_learning.kernels import FidelityStatevectorKernel
        from qiskit_algorithms.optimizers import COBYLA

        return {
            "RealAmplitudes": RealAmplitudes,
            "ZZFeatureMap": ZZFeatureMap,
            "PegasosQSVC": PegasosQSVC,
            "VQC": VQC,
            "FidelityStatevectorKernel": FidelityStatevectorKernel,
            "COBYLA": COBYLA,
            "StatevectorSampler": StatevectorSampler,
        }

    def _transform_for_training(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        X_array = np.asarray(X, dtype=float)
        if fit:
            X_imputed = self.imputer.fit_transform(X_array)
            X_scaled = self.scaler.fit_transform(X_imputed)
            if not self.use_pca:
                if X_scaled.shape[1] > self.max_qubits:
                    raise ValueError(
                        f"Direct quantum feature mode received {X_scaled.shape[1]} features, "
                        f"but max_qubits is {self.max_qubits}."
                    )
                self.pca = None
                self.n_components_ = X_scaled.shape[1]
                return self.minmax_scaler.fit_transform(X_scaled)

            full_pca = PCA(random_state=self.random_seed)
            full_pca.fit(X_scaled)
            cumulative = np.cumsum(full_pca.explained_variance_ratio_)
            n_components = int(np.searchsorted(cumulative, self.variance_threshold) + 1)
            n_components = min(max(n_components, 2), min(self.max_qubits, X_scaled.shape[1]))
            self.n_components_ = n_components
            self.pca = PCA(n_components=n_components, random_state=self.random_seed)
            X_reduced = self.pca.fit_transform(X_scaled)
            return self.minmax_scaler.fit_transform(X_reduced)

        X_imputed = self.imputer.transform(X_array)
        X_scaled = self.scaler.transform(X_imputed)
        if not self.use_pca:
            return self.minmax_scaler.transform(X_scaled)

        X_reduced = self.pca.transform(X_scaled)
        return self.minmax_scaler.transform(X_reduced)

    def _build_model(self) -> Any:
        imports = self._lazy_import_qiskit()
        feature_map = imports["ZZFeatureMap"](feature_dimension=self.n_components_, reps=1)
        if self.model_name == "QSVM":
            kernel = imports["FidelityStatevectorKernel"](feature_map=feature_map)
            return imports["PegasosQSVC"](
                quantum_kernel=kernel,
                num_steps=self.qsvm_num_steps,
                seed=self.random_seed,
            )

        ansatz = imports["RealAmplitudes"](num_qubits=self.n_components_, reps=1)
        return imports["VQC"](
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=imports["COBYLA"](maxiter=self.vqc_maxiter),
            sampler=imports["StatevectorSampler"](),
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "QuantumClassifierBundle":
        X_quantum = self._transform_for_training(X, fit=True)
        self.model = self._build_model()
        self.model.fit(X_quantum, np.asarray(y, dtype=int))
        return self

    def _transform_for_inference(self, X: pd.DataFrame) -> np.ndarray:
        return self._transform_for_training(X, fit=False)

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        X_quantum = self._transform_for_inference(X)
        if hasattr(self.model, "decision_function"):
            return np.asarray(self.model.decision_function(X_quantum), dtype=float)
        return np.asarray(self.model.predict(X_quantum), dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_quantum = self._transform_for_inference(X)
        return np.asarray(self.model.predict(X_quantum), dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_quantum = self._transform_for_inference(X)
        if hasattr(self.model, "predict_proba"):
            probabilities = np.asarray(self.model.predict_proba(X_quantum), dtype=float)
            if probabilities.ndim == 1:
                probabilities = np.column_stack([1.0 - probabilities, probabilities])
            return probabilities

        scores = self.decision_function(X)
        probabilities = 1.0 / (1.0 + np.exp(-np.asarray(scores, dtype=float)))
        return np.column_stack([1.0 - probabilities, probabilities])

    def export_state(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "random_seed": self.random_seed,
            "variance_threshold": self.variance_threshold,
            "max_qubits": self.max_qubits,
            "vqc_maxiter": self.vqc_maxiter,
            "qsvm_num_steps": self.qsvm_num_steps,
            "use_pca": self.use_pca,
            "n_components": self.n_components_,
            "preprocessing": {
                "imputer_statistics": self._serialize_optional_array(getattr(self.imputer, "statistics_", None)),
                "scaler_mean": self._serialize_optional_array(getattr(self.scaler, "mean_", None)),
                "scaler_scale": self._serialize_optional_array(getattr(self.scaler, "scale_", None)),
                "minmax_min": self._serialize_optional_array(getattr(self.minmax_scaler, "min_", None)),
                "minmax_scale": self._serialize_optional_array(getattr(self.minmax_scaler, "scale_", None)),
                "pca_components": self._serialize_optional_array(getattr(self.pca, "components_", None)),
                "pca_mean": self._serialize_optional_array(getattr(self.pca, "mean_", None)),
                "pca_explained_variance": self._serialize_optional_array(
                    getattr(self.pca, "explained_variance_", None)
                ),
                "pca_explained_variance_ratio": self._serialize_optional_array(
                    getattr(self.pca, "explained_variance_ratio_", None)
                ),
            },
            "model_state": self._export_model_state(),
        }

    def _export_model_state(self) -> Dict[str, Any]:
        model_state = {"class_name": type(self.model).__name__ if self.model is not None else None}
        if self.model is None:
            return model_state

        classes = getattr(self.model, "classes_", None)
        if classes is not None:
            model_state["classes"] = self._serialize_optional_array(classes)
        weights = getattr(self.model, "weights", None)
        if weights is not None:
            model_state["weights"] = self._serialize_optional_array(weights)
        if self.model_name == "QSVM":
            model_state["feature_map"] = {
                "name": "ZZFeatureMap",
                "feature_dimension": self.n_components_,
                "reps": 1,
            }
        if self.model_name == "VQC":
            model_state["feature_map"] = {
                "name": "ZZFeatureMap",
                "feature_dimension": self.n_components_,
                "reps": 1,
            }
            model_state["ansatz"] = {
                "name": "RealAmplitudes",
                "num_qubits": self.n_components_,
                "reps": 1,
            }
        return model_state

    def _serialize_optional_array(self, value: Any) -> Any:
        if value is None:
            return None
        return np.asarray(value).tolist()


class MultiDatasetPipeline:
    """Prepare datasets, train models, and export website-ready artifacts."""

    def __init__(self, config: Dict[str, Any], project_root: Optional[Path] = None) -> None:
        self.config = config
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1])
        self.random_seed = config.get("general", {}).get("random_seed", 42)
        self.requested_folds = config.get("cross_validation", {}).get("n_folds", 10)
        self.quantum_config = config.get("quantum", {})
        output_config = config.get("output", {})
        self.artifact_root = self.project_root / output_config.get("artifact_root", "artifacts")
        self.external_data_dir = self.project_root / output_config.get("external_data_dir", "data/external")
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.external_data_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        dataset_specs: Optional[Iterable[DatasetSpec]] = None,
        include_quantum: bool = True,
    ) -> pd.DataFrame:
        specs = list(dataset_specs or build_default_dataset_specs())
        cross_dataset_rows: List[Dict[str, Any]] = []
        markdown_lines = [
            "# Cross-Dataset Parkinson Model Summary",
            "",
            "| Dataset | Best Classical | Best Quantum | Website Candidate |",
            "| --- | --- | --- | --- |",
        ]

        for spec in specs:
            prepared = self.prepare_dataset(spec)
            dataset_comparison = self._train_dataset(prepared, include_quantum=include_quantum)
            dataset_comparison["dataset_id"] = prepared.spec.dataset_id
            dataset_comparison["dataset_source"] = prepared.resolved_source_ref
            cross_dataset_rows.extend(dataset_comparison.to_dict(orient="records"))

            classical_rows = dataset_comparison[dataset_comparison["model_family"] == "classical"]
            quantum_rows = dataset_comparison[dataset_comparison["model_family"] == "quantum"]
            best_classical = classical_rows.sort_values("mean_f1", ascending=False).iloc[0]["model_name"]
            best_quantum = (
                quantum_rows.sort_values("mean_f1", ascending=False).iloc[0]["model_name"]
                if not quantum_rows.empty
                else "N/A"
            )
            website_candidate = dataset_comparison.sort_values(
                ["inference_ready", "mean_f1"], ascending=[False, False]
            ).iloc[0]["model_name"]
            markdown_lines.append(
                f"| {prepared.spec.dataset_id} | {best_classical} | {best_quantum} | {website_candidate} |"
            )

        cross_dataset_df = pd.DataFrame(cross_dataset_rows)
        cross_dataset_path = self.artifact_root / "cross_dataset_comparison.csv"
        cross_dataset_df.to_csv(cross_dataset_path, index=False)

        summary_path = self.artifact_root / "cross_dataset_summary.md"
        summary_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
        logger.info("Saved cross-dataset comparison to %s", cross_dataset_path)
        logger.info("Saved cross-dataset summary to %s", summary_path)

        return cross_dataset_df

    def prepare_dataset(self, spec: DatasetSpec) -> PreparedDataset:
        logger.info("Preparing dataset %s", spec.source_ref)
        if spec.source_type == "local_csv":
            raw_df = pd.read_csv(self.project_root.parent / spec.local_path)
            features, labels, label_map = self._normalize_tabular_dataframe(raw_df, spec.target_column)
            return self._build_prepared_dataset(spec, spec.source_ref, spec.source_type, features, labels, label_map)

        if spec.source_type == "huggingface_tabular":
            tabular = self._load_huggingface_tabular(spec.source_ref)
            resolved_source_ref = spec.source_ref
            if tabular is None and spec.fallback_source_ref:
                logger.warning("Falling back from %s to %s", spec.source_ref, spec.fallback_source_ref)
                try:
                    features, labels, label_map = self._load_huggingface_audio(spec.fallback_source_ref)
                    return self._build_prepared_dataset(
                        spec,
                        spec.fallback_source_ref,
                        "huggingface_audio",
                        features,
                        labels,
                        label_map,
                    )
                except Exception:
                    tabular = self._load_huggingface_tabular(spec.fallback_source_ref)
                resolved_source_ref = spec.fallback_source_ref
            if tabular is None:
                raise ValueError(f"Unable to load a labeled tabular dataset for {spec.source_ref}")
            features, labels, label_map = tabular
            return self._build_prepared_dataset(spec, resolved_source_ref, spec.source_type, features, labels, label_map)

        if spec.source_type == "huggingface_audio":
            features, labels, label_map = self._load_huggingface_audio(spec.source_ref)
            return self._build_prepared_dataset(spec, spec.source_ref, spec.source_type, features, labels, label_map)

        raise ValueError(f"Unsupported dataset source type: {spec.source_type}")

    def _build_prepared_dataset(
        self,
        spec: DatasetSpec,
        resolved_source_ref: str,
        source_type: str,
        features: pd.DataFrame,
        labels: np.ndarray,
        label_map: Dict[int, str],
    ) -> PreparedDataset:
        features = features.reset_index(drop=True)
        labels = np.asarray(labels, dtype=int)
        class_counts = pd.Series(labels).value_counts().sort_index()
        effective_folds = int(min(self.requested_folds, class_counts.min()))
        if effective_folds < 2:
            raise ValueError(f"Dataset {spec.source_ref} does not have enough class members for CV.")

        return PreparedDataset(
            spec=spec,
            resolved_source_ref=resolved_source_ref,
            source_type=source_type,
            features=features,
            labels=labels,
            label_map=label_map,
            feature_names=features.columns.tolist(),
            cv_folds=effective_folds,
        )

    def _load_huggingface_tabular(
        self,
        source_ref: str,
    ) -> Optional[tuple[pd.DataFrame, np.ndarray, Dict[int, str]]]:
        snapshot_dir = self._snapshot_dataset(source_ref)
        tabular_candidates = sorted(
            [
                path
                for path in snapshot_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {".csv", ".tsv", ".xlsx", ".parquet", ".jsonl", ".json"}
            ]
        )

        for candidate in tabular_candidates:
            try:
                raw_df = self._read_tabular_file(candidate)
                return self._normalize_tabular_dataframe(raw_df)
            except Exception:
                continue

        return None

    def _load_huggingface_audio(
        self,
        source_ref: str,
    ) -> tuple[pd.DataFrame, np.ndarray, Dict[int, str]]:
        snapshot_dir = self._snapshot_dataset(source_ref)
        self._extract_archives(snapshot_dir)
        audio_files = sorted(
            [
                path
                for path in snapshot_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".flac"}
            ]
        )
        audio_files = self._filter_audio_files(source_ref, audio_files)
        if not audio_files:
            raise ValueError(f"No audio files found for {source_ref}")

        feature_rows: List[Dict[str, Any]] = []
        labels: List[int] = []
        label_map: Dict[int, str] = {}

        for row_idx, audio_path in enumerate(audio_files):
            label_value, label_name = self._infer_binary_label_from_path(audio_path)
            audio_array, sample_rate = self._decode_audio(audio_path)
            features = extract_acoustic_features(audio_array, sample_rate)
            features["row_index"] = row_idx
            features["path_depth"] = float(len(audio_path.relative_to(snapshot_dir).parts))
            feature_rows.append(features)
            labels.append(label_value)
            label_map[label_value] = label_name

        feature_df = pd.DataFrame(feature_rows)
        feature_df = feature_df.fillna(0.0)
        feature_df = feature_df.sort_index(axis=1)
        return feature_df, np.asarray(labels, dtype=int), label_map

    def _normalize_tabular_dataframe(
        self,
        raw_df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> tuple[pd.DataFrame, np.ndarray, Dict[int, str]]:
        target_column = target_column or self._infer_target_column(raw_df)
        if target_column is None:
            raise ValueError("Unable to infer target column.")

        labels, label_map = self._encode_binary_labels(raw_df[target_column])

        feature_df = raw_df.drop(columns=[target_column]).copy()
        drop_cols = [
            column
            for column in feature_df.columns
            if column.lower() in ID_LIKE_COLUMNS
        ]
        feature_df = feature_df.drop(columns=drop_cols, errors="ignore")
        for column in feature_df.columns:
            feature_df[column] = pd.to_numeric(feature_df[column], errors="coerce")
        feature_df = feature_df.dropna(axis=1, how="all")

        return feature_df, labels, label_map

    def _infer_target_column(self, df: pd.DataFrame) -> Optional[str]:
        lower_to_original = {column.lower(): column for column in df.columns}
        for candidate in TARGET_COLUMN_CANDIDATES:
            if candidate in lower_to_original:
                return lower_to_original[candidate]

        for column in df.columns:
            series = df[column].dropna()
            if series.empty:
                continue
            if series.nunique() == 2:
                return column

        return None

    def _read_tabular_file(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t")
        if suffix == ".xlsx":
            return pd.read_excel(path)
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".jsonl":
            return pd.read_json(path, lines=True)
        if suffix == ".json":
            return pd.read_json(path)
        raise ValueError(f"Unsupported tabular file: {path}")

    def _snapshot_dataset(self, source_ref: str) -> Path:
        from huggingface_hub import snapshot_download

        local_dir = self.external_data_dir / source_ref.replace("/", "__")
        if local_dir.exists() and any(local_dir.rglob("*")):
            return local_dir

        try:
            snapshot_download(
                repo_id=source_ref,
                repo_type="dataset",
                local_dir=local_dir,
            )
            return local_dir
        except Exception:
            cached_snapshot = snapshot_download(
                repo_id=source_ref,
                repo_type="dataset",
                local_files_only=True,
            )
            return Path(cached_snapshot)

    def _extract_archives(self, snapshot_dir: Path) -> None:
        for archive_path in snapshot_dir.rglob("*.zip"):
            extract_dir = archive_path.with_suffix("")
            if extract_dir.exists():
                continue
            try:
                shutil.unpack_archive(str(archive_path), str(extract_dir))
            except Exception as exc:
                logger.warning("Skipping archive extraction for %s: %s", archive_path, exc)

    def _decode_audio(self, audio_path: Path) -> tuple[np.ndarray, int]:
        import soundfile as sf

        y, sr = sf.read(str(audio_path))
        return np.asarray(y, dtype=np.float32), int(sr)

    def _infer_binary_label_from_path(self, path: Path) -> tuple[int, str]:
        normalized = self._normalize_label_name(str(path))
        if any(token in normalized for token in NEGATIVE_LABEL_TOKENS):
            return 0, "control"
        if any(token in normalized for token in POSITIVE_LABEL_TOKENS):
            return 1, "parkinsons"
        raise ValueError(f"Unable to infer binary label from path: {path}")

    def _filter_audio_files(self, source_ref: str, audio_files: List[Path]) -> List[Path]:
        if source_ref == "birgermoell/Italian_Parkinsons_Voice_and_Speech":
            pr1_files = [path for path in audio_files if path.name.upper().startswith("PR1")]
            if pr1_files:
                return pr1_files
        if source_ref == "XANJEEV/Parkinson_Classification_Dataset":
            return self._cap_xanjeev_audio_files(audio_files)
        return audio_files

    def _cap_xanjeev_audio_files(self, audio_files: List[Path], max_files_per_subject: int = 15) -> List[Path]:
        filtered_files: List[Path] = []
        subject_counts: Dict[str, int] = {}

        for audio_path in sorted(audio_files):
            if audio_path.name == ".wav":
                continue

            subject_match = XANJEEV_SUBJECT_PATTERN.search(audio_path.stem)
            subject_key = subject_match.group(1).upper() if subject_match else audio_path.parent.name.upper()
            current_count = subject_counts.get(subject_key, 0)
            if current_count >= max_files_per_subject:
                continue

            filtered_files.append(audio_path)
            subject_counts[subject_key] = current_count + 1

        return filtered_files

    def _select_fit_data(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        model_family: str,
        seed_offset: int = 0,
    ) -> tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
        X_fit = X.reset_index(drop=True)
        y_fit = np.asarray(y, dtype=int)
        if model_family != "quantum":
            return X_fit, y_fit, {"fit_sample_count": len(y_fit), "fit_strategy": "full_dataset"}

        max_train_samples = self.quantum_config.get("max_train_samples")
        if not max_train_samples or len(y_fit) <= int(max_train_samples):
            return X_fit, y_fit, {"fit_sample_count": len(y_fit), "fit_strategy": "full_dataset"}

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=int(max_train_samples),
            random_state=self.random_seed + seed_offset,
        )
        selected_idx, _ = next(splitter.split(np.zeros(len(y_fit)), y_fit))
        selected_idx = np.sort(selected_idx)
        X_subset = X_fit.iloc[selected_idx].reset_index(drop=True)
        y_subset = y_fit[selected_idx]
        return X_subset, y_subset, {"fit_sample_count": len(y_subset), "fit_strategy": "stratified_subsample"}

    def _encode_binary_labels(self, series: pd.Series) -> tuple[np.ndarray, Dict[int, str]]:
        cleaned = series.dropna()
        if cleaned.nunique() != 2:
            raise ValueError("Expected a binary target column.")

        if pd.api.types.is_numeric_dtype(series):
            numeric_values = sorted(series.dropna().astype(float).unique().tolist())
            mapping = {numeric_values[0]: 0, numeric_values[-1]: 1}
            encoded = series.map(mapping).fillna(0).astype(int).to_numpy()
            label_map = {0: str(numeric_values[0]), 1: str(numeric_values[-1])}
            return encoded, label_map

        unique_values = list(cleaned.astype(str).unique())
        normalized_values = [self._normalize_label_name(value) for value in unique_values]
        negative_value = None
        positive_value = None
        for raw_value, normalized in zip(unique_values, normalized_values):
            if any(token in normalized for token in NEGATIVE_LABEL_TOKENS):
                negative_value = raw_value
            if any(token in normalized for token in POSITIVE_LABEL_TOKENS):
                positive_value = raw_value

        if negative_value is None or positive_value is None:
            sorted_values = sorted(unique_values)
            negative_value, positive_value = sorted_values[0], sorted_values[1]

        mapping = {negative_value: 0, positive_value: 1}
        encoded = series.astype(str).map(mapping).astype(int).to_numpy()
        label_map = {0: str(negative_value), 1: str(positive_value)}
        return encoded, label_map

    def _normalize_label_name(self, value: str) -> str:
        return "".join(character.lower() if character.isalnum() else " " for character in str(value))

    def _train_dataset(self, prepared: PreparedDataset, include_quantum: bool) -> pd.DataFrame:
        layout = build_dataset_artifact_layout(self.artifact_root, prepared.spec.dataset_id)

        cleaned_dataset_path = layout.processed_dir / "cleaned_features.csv"
        prepared.features.assign(target=prepared.labels).to_csv(cleaned_dataset_path, index=False)
        feature_schema_path = write_feature_schema(layout, prepared.feature_names)
        label_map_path = write_label_map(layout, prepared.label_map)

        model_factories = self._build_model_factories(include_quantum=include_quantum)
        manifest_rows: List[Dict[str, Any]] = []
        all_fold_metrics: List[pd.DataFrame] = []
        all_oof_records: List[pd.DataFrame] = []
        all_confusion_records: List[pd.DataFrame] = []
        comparison_rows: List[Dict[str, Any]] = []

        for model_name, factory in model_factories.items():
            family = "quantum" if model_name in {"QSVM", "VQC"} else "classical"
            logger.info("Evaluating %s on %s", model_name, prepared.spec.dataset_id)
            evaluation = self._cross_validate_model(
                prepared.features,
                prepared.labels,
                factory,
                model_name=model_name,
                model_family=family,
                cv_folds=prepared.cv_folds,
            )

            all_fold_metrics.append(evaluation["fold_metrics"])
            all_oof_records.append(evaluation["oof_predictions"])
            all_confusion_records.append(evaluation["confusion_matrix"])

            summary_row = evaluation["summary"]
            summary_row["dataset_id"] = prepared.spec.dataset_id
            summary_row["dataset_source"] = prepared.resolved_source_ref
            comparison_rows.append(summary_row)

            fitted_model = factory()
            X_fit, y_fit, fit_metadata = self._select_fit_data(
                prepared.features,
                prepared.labels,
                model_family=family,
                seed_offset=0,
            )
            fitted_model.fit(X_fit, y_fit)
            artifact_base = layout.models_dir / f"{model_name.lower()}.joblib"
            artifact_path, inference_ready = self._save_model_artifact(fitted_model, artifact_base)

            manifest_rows.append(
                {
                    "dataset_id": prepared.spec.dataset_id,
                    "dataset_source": prepared.resolved_source_ref,
                    "model_name": model_name,
                    "model_family": family,
                    "artifact_path": artifact_path,
                    "preprocessor_path": "",
                    "feature_schema_path": feature_schema_path,
                    "label_map_path": label_map_path,
                    "cv_folds": prepared.cv_folds,
                    "mean_accuracy": summary_row["mean_accuracy"],
                    "mean_recall": summary_row["mean_recall"],
                    "mean_f1": summary_row["mean_f1"],
                    "mean_roc_auc": summary_row["mean_roc_auc"],
                    "mean_fit_sample_count": summary_row["mean_fit_sample_count"],
                    "fit_strategy": fit_metadata["fit_strategy"],
                    "fit_sample_count": fit_metadata["fit_sample_count"],
                    "inference_ready": inference_ready,
                }
            )
            summary_row["fit_strategy"] = fit_metadata["fit_strategy"]
            summary_row["fit_sample_count"] = fit_metadata["fit_sample_count"]
            summary_row["inference_ready"] = inference_ready

        pd.concat(all_fold_metrics, ignore_index=True).to_csv(layout.results_dir / "cv_fold_metrics.csv", index=False)
        pd.concat(all_oof_records, ignore_index=True).to_csv(layout.results_dir / "oof_predictions.csv", index=False)
        pd.concat(all_confusion_records, ignore_index=True).to_csv(layout.results_dir / "confusion_matrix.csv", index=False)

        comparison_df = pd.DataFrame(comparison_rows).sort_values(["mean_f1", "mean_accuracy"], ascending=False)
        comparison_df.to_csv(layout.results_dir / "model_comparison.csv", index=False)
        write_model_manifest(layout, manifest_rows)

        self._write_dataset_summary(prepared, comparison_df, layout)
        return comparison_df

    def _build_model_factories(self, include_quantum: bool) -> Dict[str, Callable[[], BaseEstimator]]:
        seed = self.random_seed
        quantum_max_qubits = self.quantum_config.get("max_qubits", 6)
        quantum_variance_threshold = self.quantum_config.get("variance_threshold", 0.90)
        quantum_vqc_maxiter = self.quantum_config.get("vqc_maxiter", 10)
        quantum_qsvm_num_steps = self.quantum_config.get("qsvm_num_steps", 100)
        model_factories: Dict[str, Callable[[], BaseEstimator]] = {
            "LogisticRegression": lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1000, random_state=seed)),
                ]
            ),
            "SVM": lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", SVC(kernel="rbf", probability=True, random_state=seed)),
                ]
            ),
            "RandomForest": lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            random_state=seed,
                            n_jobs=self.config.get("general", {}).get("n_jobs", -1),
                        ),
                    ),
                ]
            ),
            "XGBoost": lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        self._build_xgboost_estimator(),
                    ),
                ]
            ),
            "KNN": lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier(n_neighbors=5)),
                ]
            ),
        }

        if include_quantum:
            model_factories["QSVM"] = lambda: QuantumClassifierBundle(
                "QSVM",
                random_seed=seed,
                variance_threshold=quantum_variance_threshold,
                max_qubits=quantum_max_qubits,
                vqc_maxiter=quantum_vqc_maxiter,
                qsvm_num_steps=quantum_qsvm_num_steps,
            )
            model_factories["VQC"] = lambda: QuantumClassifierBundle(
                "VQC",
                random_seed=seed,
                variance_threshold=quantum_variance_threshold,
                max_qubits=quantum_max_qubits,
                vqc_maxiter=quantum_vqc_maxiter,
                qsvm_num_steps=quantum_qsvm_num_steps,
            )

        return model_factories

    def _build_xgboost_estimator(self) -> Any:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=self.random_seed,
            n_jobs=self.config.get("general", {}).get("n_jobs", -1),
            verbosity=0,
        )

    def _cross_validate_model(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        model_factory: Callable[[], Any],
        model_name: str,
        model_family: str,
        cv_folds: int,
    ) -> Dict[str, Any]:
        splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_seed)
        fold_rows: List[Dict[str, Any]] = []
        oof_rows: List[Dict[str, Any]] = []
        confusion_rows: List[Dict[str, Any]] = []

        oof_pred = np.zeros(len(y), dtype=int)
        oof_score = np.zeros(len(y), dtype=float)

        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
            model = model_factory()
            X_train = X.iloc[train_idx].reset_index(drop=True)
            X_test = X.iloc[test_idx].reset_index(drop=True)
            y_train = y[train_idx]
            y_test = y[test_idx]
            X_fit, y_fit, fit_metadata = self._select_fit_data(
                X_train,
                y_train,
                model_family=model_family,
                seed_offset=fold_idx,
            )

            model.fit(X_fit, y_fit)
            y_pred = np.asarray(model.predict(X_test), dtype=int)
            y_score = self._get_positive_scores(model, X_test)

            oof_pred[test_idx] = y_pred
            oof_score[test_idx] = y_score

            fold_rows.append(
                {
                    "model_name": model_name,
                    "model_family": model_family,
                    "fold": fold_idx,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred, zero_division=0),
                    "f1": f1_score(y_test, y_pred, zero_division=0),
                    "roc_auc": roc_auc_score(y_test, y_score) if len(np.unique(y_test)) == 2 else np.nan,
                    "fit_sample_count": fit_metadata["fit_sample_count"],
                    "fit_strategy": fit_metadata["fit_strategy"],
                }
            )

            for original_idx, truth, pred, score in zip(test_idx, y_test, y_pred, y_score):
                oof_rows.append(
                    {
                        "row_index": int(original_idx),
                        "model_name": model_name,
                        "model_family": model_family,
                        "fold": fold_idx,
                        "y_true": int(truth),
                        "y_pred": int(pred),
                        "y_score": float(score),
                    }
                )

        matrix = confusion_matrix(y, oof_pred, labels=[0, 1])
        for actual_idx, actual_label in enumerate([0, 1]):
            for predicted_idx, predicted_label in enumerate([0, 1]):
                confusion_rows.append(
                    {
                        "model_name": model_name,
                        "model_family": model_family,
                        "actual_label": actual_label,
                        "predicted_label": predicted_label,
                        "count": int(matrix[actual_idx, predicted_idx]),
                    }
                )

        fold_metrics_df = pd.DataFrame(fold_rows)
        return {
            "fold_metrics": fold_metrics_df,
            "oof_predictions": pd.DataFrame(oof_rows),
            "confusion_matrix": pd.DataFrame(confusion_rows),
            "summary": {
                "model_name": model_name,
                "model_family": model_family,
                "mean_accuracy": float(fold_metrics_df["accuracy"].mean()),
                "mean_recall": float(fold_metrics_df["recall"].mean()),
                "mean_f1": float(fold_metrics_df["f1"].mean()),
                "mean_roc_auc": float(fold_metrics_df["roc_auc"].mean()),
                "mean_fit_sample_count": float(fold_metrics_df["fit_sample_count"].mean()),
                "fit_strategy": fold_metrics_df["fit_strategy"].iloc[0],
                "cv_folds": cv_folds,
                "inference_ready": True,
            },
        }

    def _get_positive_scores(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(X), dtype=float)
            if probabilities.ndim == 2:
                return probabilities[:, 1]
            return probabilities

        if hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(X), dtype=float)
            if scores.ndim == 2:
                scores = scores[:, 1]
            return 1.0 / (1.0 + np.exp(-scores))

        return np.asarray(model.predict(X), dtype=float)

    def _save_model_artifact(self, model: Any, artifact_path: Path) -> tuple[Path, bool]:
        try:
            joblib.dump(model, artifact_path)
            return artifact_path, True
        except Exception as exc:
            bundle_path = artifact_path.with_suffix(".json")
            export_state = getattr(model, "export_state", None)
            if callable(export_state):
                try:
                    bundle = dict(export_state())
                    bundle["error"] = str(exc)
                    bundle["serialization"] = "reconstruction_bundle"
                    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
                    return bundle_path, False
                except Exception:
                    pass
            bundle_path.write_text(
                json.dumps(
                    {
                        "model_name": getattr(model, "model_name", artifact_path.stem),
                        "error": str(exc),
                        "serialization": "joblib_failed",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return bundle_path, False

    def _write_dataset_summary(
        self,
        prepared: PreparedDataset,
        comparison_df: pd.DataFrame,
        layout: Any,
    ) -> None:
        classical_df = comparison_df[comparison_df["model_family"] == "classical"]
        quantum_df = comparison_df[comparison_df["model_family"] == "quantum"]

        best_classical = classical_df.sort_values("mean_f1", ascending=False).iloc[0]["model_name"]
        best_quantum = (
            quantum_df.sort_values("mean_f1", ascending=False).iloc[0]["model_name"]
            if not quantum_df.empty
            else "N/A"
        )
        website_candidate = comparison_df.sort_values(
            ["inference_ready", "mean_f1"], ascending=[False, False]
        ).iloc[0]["model_name"]

        summary_df = pd.DataFrame(
            [
                {
                    "dataset_id": prepared.spec.dataset_id,
                    "dataset_source": prepared.resolved_source_ref,
                    "source_type": prepared.source_type,
                    "n_samples": len(prepared.features),
                    "n_features": len(prepared.feature_names),
                    "class_0_count": int(np.sum(prepared.labels == 0)),
                    "class_1_count": int(np.sum(prepared.labels == 1)),
                    "requested_cv_folds": self.requested_folds,
                    "effective_cv_folds": prepared.cv_folds,
                    "best_classical_model": best_classical,
                    "best_quantum_model": best_quantum,
                    "website_candidate_model": website_candidate,
                }
            ]
        )
        summary_df.to_csv(layout.results_dir / "dataset_summary.csv", index=False)


def run_multi_dataset_experiment(
    config: Dict[str, Any],
    include_quantum: bool = True,
) -> pd.DataFrame:
    """Convenience wrapper for scripts."""
    pipeline = MultiDatasetPipeline(config=config)
    return pipeline.run(include_quantum=include_quantum)
