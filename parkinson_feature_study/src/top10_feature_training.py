"""
Reduced-feature retraining experiment for the local PD speech dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif

from .artifacts import DatasetArtifactLayout, write_feature_schema, write_label_map, write_model_manifest
from .multi_dataset_pipeline import MultiDatasetPipeline, QuantumClassifierBundle


REQUESTED_MODEL_NAMES: tuple[str, ...] = (
    "SVM",
    "QSVM",
    "VQC",
    "XGBoost",
    "LogisticRegression",
)
QUANTUM_MODEL_NAMES = {"QSVM", "VQC"}
DATASET_ID = "pd_speech_features_local_top10_mi"
DATASET_SOURCE = "pd_speech_features.csv"


@dataclass(frozen=True)
class ReducedFeatureExperimentResult:
    output_dir: Path
    final_top_features: pd.DataFrame
    model_comparison: pd.DataFrame


def _make_layout(output_dir: Path) -> DatasetArtifactLayout:
    output_dir = Path(output_dir)
    layout = DatasetArtifactLayout(
        dataset_dir=output_dir,
        models_dir=output_dir / "models",
        results_dir=output_dir / "results",
        processed_dir=output_dir / "processed",
    )
    for path in (layout.dataset_dir, layout.models_dir, layout.results_dir, layout.processed_dir):
        path.mkdir(parents=True, exist_ok=True)
    return layout


def _normalize_pd_speech_dataframe(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, dict[int, str]]:
    pipeline = MultiDatasetPipeline(
        config={"general": {"random_seed": 42}, "cross_validation": {"n_folds": 10}},
        project_root=Path(__file__).resolve().parents[1],
    )
    return pipeline._normalize_tabular_dataframe(raw_df, target_column="class")


def rank_features_mutual_info(
    X: pd.DataFrame,
    y: Sequence[int],
    top_k: int | None = 10,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Rank features by mutual information against the target labels."""
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if top_k is not None and top_k > X.shape[1]:
        raise ValueError(f"top_k={top_k} exceeds available feature count {X.shape[1]}.")

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    scores = mutual_info_classif(X_imputed, np.asarray(y, dtype=int), random_state=random_seed)
    ranking = pd.DataFrame(
        {
            "feature": list(X.columns),
            "mutual_info_score": np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0),
        }
    ).sort_values(["mutual_info_score", "feature"], ascending=[False, True], kind="mergesort")
    ranking = ranking.reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking = ranking.loc[:, ["rank", "feature", "mutual_info_score"]]
    if top_k is not None:
        return ranking.head(top_k).reset_index(drop=True)
    return ranking


def select_fold_top_features(
    X: pd.DataFrame,
    y: Sequence[int],
    train_idx: Sequence[int],
    top_k: int = 10,
    random_seed: int = 42,
) -> tuple[list[str], pd.DataFrame]:
    """Select top features for one CV fold using only training rows."""
    train_idx = np.asarray(train_idx, dtype=int)
    ranking = rank_features_mutual_info(
        X.iloc[train_idx].reset_index(drop=True),
        np.asarray(y, dtype=int)[train_idx],
        top_k=None,
        random_seed=random_seed,
    )
    return ranking.head(top_k)["feature"].tolist(), ranking


def _build_xgboost_estimator(random_seed: int, n_jobs: int) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_seed,
        n_jobs=n_jobs,
        verbosity=0,
    )


def build_model_factories(
    random_seed: int = 42,
    quantum_mode: str = "direct_top10",
    n_jobs: int = -1,
) -> dict[str, Callable[[], Any]]:
    use_pca = quantum_mode != "direct_top10"
    max_qubits = 4 if use_pca else 10
    variance_threshold = 0.90 if use_pca else 1.0
    return {
        "SVM": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=True, random_state=random_seed)),
            ]
        ),
        "QSVM": lambda: QuantumClassifierBundle(
            "QSVM",
            random_seed=random_seed,
            variance_threshold=variance_threshold,
            max_qubits=max_qubits,
            vqc_maxiter=3,
            qsvm_num_steps=64,
            use_pca=use_pca,
        ),
        "VQC": lambda: QuantumClassifierBundle(
            "VQC",
            random_seed=random_seed,
            variance_threshold=variance_threshold,
            max_qubits=max_qubits,
            vqc_maxiter=3,
            qsvm_num_steps=64,
            use_pca=use_pca,
        ),
        "XGBoost": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", _build_xgboost_estimator(random_seed=random_seed, n_jobs=n_jobs)),
            ]
        ),
        "LogisticRegression": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=random_seed)),
            ]
        ),
    }


def _get_positive_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X), dtype=float)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return probabilities.reshape(-1)

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))

    return np.asarray(model.predict(X), dtype=float)


def _select_fit_data(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_family: str,
    random_seed: int,
    seed_offset: int,
    max_train_samples: int = 32,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    if model_family != "quantum" or len(y_train) <= max_train_samples:
        return X_train.reset_index(drop=True), y_train, {
            "fit_sample_count": len(y_train),
            "fit_strategy": "full_dataset",
        }

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=max_train_samples,
        random_state=random_seed + seed_offset,
    )
    selected_idx, _ = next(splitter.split(np.zeros(len(y_train)), y_train))
    selected_idx = np.sort(selected_idx)
    return X_train.iloc[selected_idx].reset_index(drop=True), y_train[selected_idx], {
        "fit_sample_count": len(selected_idx),
        "fit_strategy": "stratified_subsample",
    }


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) != 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _cross_validate_one_model(
    model_name: str,
    model_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    selected_by_fold: Mapping[int, list[str]],
    random_seed: int,
    feature_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    model_family = "quantum" if model_name in QUANTUM_MODEL_NAMES else "classical"
    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    oof_pred = np.zeros(len(y), dtype=int)
    oof_score = np.zeros(len(y), dtype=float)

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        features = selected_by_fold[fold]
        X_train = X.iloc[train_idx].loc[:, features].reset_index(drop=True)
        X_test = X.iloc[test_idx].loc[:, features].reset_index(drop=True)
        y_train = y[train_idx]
        y_test = y[test_idx]
        X_fit, y_fit, fit_metadata = _select_fit_data(
            X_train,
            y_train,
            model_family=model_family,
            random_seed=random_seed,
            seed_offset=fold,
        )

        model = model_factory()
        model.fit(X_fit, y_fit)
        y_pred = np.asarray(model.predict(X_test), dtype=int)
        y_score = _get_positive_scores(model, X_test)

        oof_pred[test_idx] = y_pred
        oof_score[test_idx] = y_score
        fold_rows.append(
            {
                "model_name": model_name,
                "model_family": model_family,
                "feature_mode": feature_mode,
                "fold": fold,
                "selected_feature_count": len(features),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "roc_auc": _safe_roc_auc(y_test, y_score),
                **fit_metadata,
            }
        )
        for original_idx, truth, pred, score in zip(test_idx, y_test, y_pred, y_score):
            oof_rows.append(
                {
                    "row_index": int(original_idx),
                    "model_name": model_name,
                    "model_family": model_family,
                    "feature_mode": feature_mode,
                    "fold": fold,
                    "y_true": int(truth),
                    "y_pred": int(pred),
                    "y_score": float(score),
                }
            )

    confusion_rows: list[dict[str, Any]] = []
    matrix = confusion_matrix(y, oof_pred, labels=[0, 1])
    for actual_idx, actual_label in enumerate([0, 1]):
        for predicted_idx, predicted_label in enumerate([0, 1]):
            confusion_rows.append(
                {
                    "model_name": model_name,
                    "model_family": model_family,
                    "feature_mode": feature_mode,
                    "actual_label": actual_label,
                    "predicted_label": predicted_label,
                    "count": int(matrix[actual_idx, predicted_idx]),
                }
            )

    fold_metrics = pd.DataFrame(fold_rows)
    summary = {
        "model_name": model_name,
        "model_family": model_family,
        "feature_mode": feature_mode,
        "mean_accuracy": float(fold_metrics["accuracy"].mean()),
        "mean_recall": float(fold_metrics["recall"].mean()),
        "mean_f1": float(fold_metrics["f1"].mean()),
        "mean_roc_auc": float(fold_metrics["roc_auc"].mean()),
        "mean_fit_sample_count": float(fold_metrics["fit_sample_count"].mean()),
        "fit_strategy": fold_metrics["fit_strategy"].iloc[0],
        "cv_folds": int(fold_metrics["fold"].nunique()),
    }
    return fold_metrics, pd.DataFrame(oof_rows), pd.DataFrame(confusion_rows), summary


def _save_final_model(
    model_name: str,
    model_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    layout: DatasetArtifactLayout,
    random_seed: int,
) -> tuple[Path, bool, dict[str, Any]]:
    model_family = "quantum" if model_name in QUANTUM_MODEL_NAMES else "classical"
    X_selected = X.loc[:, feature_names].reset_index(drop=True)
    X_fit, y_fit, fit_metadata = _select_fit_data(
        X_selected,
        y,
        model_family=model_family,
        random_seed=random_seed,
        seed_offset=0,
    )
    model = model_factory()
    model.fit(X_fit, y_fit)
    artifact_path = layout.models_dir / f"{model_name.lower()}.joblib"
    try:
        joblib.dump(model, artifact_path)
        return artifact_path, True, fit_metadata
    except Exception:
        if artifact_path.exists():
            artifact_path.unlink()
        fallback_path = artifact_path.with_suffix(".json")
        state = model.export_state() if hasattr(model, "export_state") else {"model_name": model_name}
        fallback_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return fallback_path, False, fit_metadata


def run_reduced_feature_experiment(
    source_csv: Path,
    output_dir: Path,
    top_k: int = 10,
    n_splits: int = 10,
    model_factories: Mapping[str, Callable[[], Any]] | None = None,
    random_seed: int = 42,
) -> ReducedFeatureExperimentResult:
    source_csv = Path(source_csv)
    output_dir = Path(output_dir)
    layout = _make_layout(output_dir)

    raw_df = pd.read_csv(source_csv)
    X, y, label_map = _normalize_pd_speech_dataframe(raw_df)
    y = np.asarray(y, dtype=int)
    X.assign(target=y).to_csv(layout.processed_dir / "cleaned_features.csv", index=False)

    final_top_features = rank_features_mutual_info(X, y, top_k=top_k, random_seed=random_seed)
    final_top_features.to_csv(layout.processed_dir / "final_top10_features.csv", index=False)
    selected_final_features = final_top_features["feature"].tolist()
    write_feature_schema(layout, selected_final_features)
    write_label_map(layout, label_map)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    splits = [(train_idx, test_idx) for train_idx, test_idx in splitter.split(X, y)]
    selected_rows: list[pd.DataFrame] = []
    ranking_rows: list[pd.DataFrame] = []
    selected_by_fold: dict[int, list[str]] = {}

    for fold, (train_idx, _) in enumerate(splits, start=1):
        selected, ranking = select_fold_top_features(X, y, train_idx, top_k=top_k, random_seed=random_seed)
        selected_by_fold[fold] = selected
        ranking = ranking.assign(fold=fold).loc[:, ["fold", "rank", "feature", "mutual_info_score"]]
        ranking_rows.append(ranking)
        selected_rows.append(ranking.head(top_k))

    pd.concat(selected_rows, ignore_index=True).to_csv(
        layout.results_dir / "selected_features_by_fold.csv",
        index=False,
    )
    pd.concat(ranking_rows, ignore_index=True).to_csv(
        layout.results_dir / "feature_rankings_by_fold.csv",
        index=False,
    )

    comparison_rows: list[dict[str, Any]] = []
    fold_metric_frames: list[pd.DataFrame] = []
    oof_frames: list[pd.DataFrame] = []
    confusion_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    factory_map = dict(model_factories or build_model_factories(random_seed=random_seed))

    for model_name in REQUESTED_MODEL_NAMES:
        if model_factories is None and model_name in QUANTUM_MODEL_NAMES:
            try:
                fold_metrics, oof, confusion, summary = _cross_validate_one_model(
                    model_name,
                    build_model_factories(random_seed=random_seed, quantum_mode="direct_top10")[model_name],
                    X,
                    y,
                    splits,
                    selected_by_fold,
                    random_seed=random_seed,
                    feature_mode="direct_top10",
                )
                final_factory = build_model_factories(random_seed=random_seed, quantum_mode="direct_top10")[model_name]
            except Exception:
                fold_metrics, oof, confusion, summary = _cross_validate_one_model(
                    model_name,
                    build_model_factories(random_seed=random_seed, quantum_mode="fallback_pca4")[model_name],
                    X,
                    y,
                    splits,
                    selected_by_fold,
                    random_seed=random_seed,
                    feature_mode="fallback_pca4",
                )
                final_factory = build_model_factories(random_seed=random_seed, quantum_mode="fallback_pca4")[model_name]
        else:
            fold_metrics, oof, confusion, summary = _cross_validate_one_model(
                model_name,
                factory_map[model_name],
                X,
                y,
                splits,
                selected_by_fold,
                random_seed=random_seed,
                feature_mode="custom" if model_factories is not None else "direct_top10",
            )
            final_factory = factory_map[model_name]

        artifact_path, inference_ready, fit_metadata = _save_final_model(
            model_name,
            final_factory,
            X,
            y,
            selected_final_features,
            layout,
            random_seed=random_seed,
        )
        summary = {
            **summary,
            "dataset_id": DATASET_ID,
            "dataset_source": DATASET_SOURCE,
            "selected_feature_count": top_k,
            "inference_ready": inference_ready,
        }
        comparison_rows.append(summary)
        fold_metric_frames.append(fold_metrics)
        oof_frames.append(oof)
        confusion_frames.append(confusion)
        manifest_rows.append(
            {
                "dataset_id": DATASET_ID,
                "dataset_source": DATASET_SOURCE,
                "model_name": model_name,
                "model_family": summary["model_family"],
                "artifact_path": artifact_path,
                "preprocessor_path": "",
                "feature_schema_path": layout.processed_dir / "feature_schema.csv",
                "label_map_path": layout.processed_dir / "label_map.csv",
                "cv_folds": n_splits,
                "mean_accuracy": summary["mean_accuracy"],
                "mean_recall": summary["mean_recall"],
                "mean_f1": summary["mean_f1"],
                "mean_roc_auc": summary["mean_roc_auc"],
                "fit_strategy": fit_metadata["fit_strategy"],
                "fit_sample_count": fit_metadata["fit_sample_count"],
                "feature_selection_method": "mutual_information",
                "selected_feature_count": top_k,
                "feature_mode": summary["feature_mode"],
                "inference_ready": inference_ready,
            }
        )

    fold_metrics_df = pd.concat(fold_metric_frames, ignore_index=True)
    model_comparison = pd.DataFrame(comparison_rows).sort_values(
        ["mean_f1", "mean_accuracy"],
        ascending=False,
    )
    fold_metrics_df.to_csv(layout.results_dir / "cv_fold_metrics.csv", index=False)
    model_comparison.to_csv(layout.results_dir / "model_comparison.csv", index=False)
    pd.concat(oof_frames, ignore_index=True).to_csv(layout.results_dir / "oof_predictions.csv", index=False)
    pd.concat(confusion_frames, ignore_index=True).to_csv(layout.results_dir / "confusion_matrix.csv", index=False)
    write_model_manifest(layout, manifest_rows)

    return ReducedFeatureExperimentResult(
        output_dir=output_dir,
        final_top_features=final_top_features,
        model_comparison=model_comparison,
    )
