from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "swarm_digital_twin"
MODEL_DIR = OUT_DIR / "models"
RESULTS_DIR = OUT_DIR / "results"
DATASET_ID = "pd_speech_features_local_top10_mi"
DATASET_SOURCE = "pd_speech_features.csv"
MODEL_NAME = "PSO-SVM"
RANDOM_SEED = 42


def load_top10_data() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    schema = pd.read_csv(ROOT / "processed" / "feature_schema.csv")
    features = schema["feature_name"].astype(str).tolist()
    frame = pd.read_csv(ROOT / "processed" / "cleaned_features.csv")
    return frame.loc[:, features], frame["target"].to_numpy(dtype=int), features


def make_svm(c_value: float, gamma_value: float, probability: bool = False) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    kernel="rbf",
                    C=float(c_value),
                    gamma=float(gamma_value),
                    probability=probability,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def decode_position(position: np.ndarray) -> tuple[float, float]:
    log_c, log_gamma = position
    return 10.0**float(log_c), 10.0**float(log_gamma)


def score_params(position: np.ndarray, X: pd.DataFrame, y: np.ndarray) -> float:
    c_value, gamma_value = decode_position(position)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    for train_idx, test_idx in splitter.split(X, y):
        model = make_svm(c_value, gamma_value)
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict(X.iloc[test_idx])
        scores.append(f1_score(y[test_idx], pred, zero_division=0))
    return float(np.mean(scores))


def run_pso_search(X: pd.DataFrame, y: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    particle_count = 10
    iterations = 8
    lower = np.array([-2.0, -4.0])
    upper = np.array([3.0, 1.0])
    positions = rng.uniform(lower, upper, size=(particle_count, 2))
    velocities = rng.normal(0.0, 0.35, size=(particle_count, 2))

    personal_best = positions.copy()
    personal_scores = np.full(particle_count, -np.inf)
    global_best = positions[0].copy()
    global_score = -np.inf
    history = []

    for iteration in range(1, iterations + 1):
        for index, position in enumerate(positions):
            score = score_params(position, X, y)
            c_value, gamma_value = decode_position(position)
            history.append(
                {
                    "iteration": iteration,
                    "particle": index + 1,
                    "C": c_value,
                    "gamma": gamma_value,
                    "mean_f1": score,
                }
            )
            if score > personal_scores[index]:
                personal_scores[index] = score
                personal_best[index] = position.copy()
            if score > global_score:
                global_score = score
                global_best = position.copy()

        inertia = 0.58
        cognitive = 1.35
        social = 1.35
        r1 = rng.random(size=(particle_count, 2))
        r2 = rng.random(size=(particle_count, 2))
        velocities = (
            inertia * velocities
            + cognitive * r1 * (personal_best - positions)
            + social * r2 * (global_best - positions)
        )
        positions = np.clip(positions + velocities, lower, upper)

    best_c, best_gamma = decode_position(global_best)
    return (
        {
            "C": best_c,
            "gamma": best_gamma,
            "mean_f1": global_score,
            "particles": particle_count,
            "iterations": iterations,
            "cv_folds_for_search": 3,
        },
        pd.DataFrame(history),
    )


def evaluate_model(X: pd.DataFrame, y: np.ndarray, c_value: float, gamma_value: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    fold_rows = []
    oof_rows = []
    oof_pred = np.zeros(len(y), dtype=int)
    oof_score = np.zeros(len(y), dtype=float)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        model = make_svm(c_value, gamma_value)
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict(X.iloc[test_idx]).astype(int)
        decision = model.decision_function(X.iloc[test_idx])
        score = 1.0 / (1.0 + np.exp(-decision))
        oof_pred[test_idx] = pred
        oof_score[test_idx] = score
        fold_rows.append(
            {
                "model_name": MODEL_NAME,
                "model_family": "swarm",
                "feature_mode": "pso_top10",
                "fold": fold,
                "selected_feature_count": X.shape[1],
                "accuracy": float(accuracy_score(y[test_idx], pred)),
                "recall": float(recall_score(y[test_idx], pred, zero_division=0)),
                "f1": float(f1_score(y[test_idx], pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y[test_idx], score)),
                "fit_sample_count": len(train_idx),
                "fit_strategy": "pso_optimized_full_dataset",
            }
        )
        for row_idx, truth, row_pred, row_score in zip(test_idx, y[test_idx], pred, score):
            oof_rows.append(
                {
                    "row_index": int(row_idx),
                    "model_name": MODEL_NAME,
                    "model_family": "swarm",
                    "feature_mode": "pso_top10",
                    "fold": fold,
                    "y_true": int(truth),
                    "y_pred": int(row_pred),
                    "y_score": float(row_score),
                }
            )

    matrix = confusion_matrix(y, oof_pred, labels=[0, 1])
    confusion_rows = []
    for actual_idx, actual_label in enumerate([0, 1]):
        for predicted_idx, predicted_label in enumerate([0, 1]):
            confusion_rows.append(
                {
                    "model_name": MODEL_NAME,
                    "model_family": "swarm",
                    "feature_mode": "pso_top10",
                    "actual_label": actual_label,
                    "predicted_label": predicted_label,
                    "count": int(matrix[actual_idx, predicted_idx]),
                }
            )

    fold_metrics = pd.DataFrame(fold_rows)
    summary = {
        "dataset_id": DATASET_ID,
        "dataset_source": DATASET_SOURCE,
        "model_name": MODEL_NAME,
        "model_family": "swarm",
        "feature_mode": "pso_top10",
        "mean_accuracy": float(fold_metrics["accuracy"].mean()),
        "mean_recall": float(fold_metrics["recall"].mean()),
        "mean_f1": float(fold_metrics["f1"].mean()),
        "mean_roc_auc": float(fold_metrics["roc_auc"].mean()),
        "mean_fit_sample_count": float(fold_metrics["fit_sample_count"].mean()),
        "fit_strategy": "pso_optimized_full_dataset",
        "cv_folds": 10,
        "selected_feature_count": X.shape[1],
        "inference_ready": True,
    }
    return fold_metrics, pd.DataFrame(oof_rows), pd.DataFrame(confusion_rows), summary


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X, y, features = load_top10_data()

    best, history = run_pso_search(X, y)
    history.to_csv(RESULTS_DIR / "pso_search_history.csv", index=False)
    (RESULTS_DIR / "pso_best_params.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    fold_metrics, oof, confusion, summary = evaluate_model(X, y, best["C"], best["gamma"])
    fold_metrics.to_csv(RESULTS_DIR / "cv_fold_metrics.csv", index=False)
    oof.to_csv(RESULTS_DIR / "oof_predictions.csv", index=False)
    confusion.to_csv(RESULTS_DIR / "confusion_matrix.csv", index=False)
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    model = make_svm(best["C"], best["gamma"], probability=True)
    model.fit(X, y)
    artifact_path = MODEL_DIR / "pso_svm.joblib"
    joblib.dump(model, artifact_path)

    manifest = pd.DataFrame(
        [
            {
                "dataset_id": DATASET_ID,
                "dataset_source": DATASET_SOURCE,
                "model_name": MODEL_NAME,
                "model_family": "swarm",
                "artifact_path": str(artifact_path),
                "preprocessor_path": "",
                "feature_schema_path": str(ROOT / "processed" / "feature_schema.csv"),
                "label_map_path": str(ROOT / "processed" / "label_map.csv"),
                "cv_folds": 10,
                "mean_accuracy": summary["mean_accuracy"],
                "mean_recall": summary["mean_recall"],
                "mean_f1": summary["mean_f1"],
                "mean_roc_auc": summary["mean_roc_auc"],
                "fit_strategy": summary["fit_strategy"],
                "fit_sample_count": len(y),
                "feature_selection_method": "mutual_information",
                "selected_feature_count": len(features),
                "feature_mode": "pso_top10",
                "inference_ready": True,
            }
        ]
    )
    manifest.to_csv(OUT_DIR / "model_manifest.csv", index=False)

    readme = f"""# Swarm Digital Twin Extension

This folder contains the PSO-SVM swarm-intelligence model and supporting outputs for the top-10 PD speech feature website.

## PSO-SVM

- Particle Swarm Optimization searched SVM `C` and `gamma` on the same 10 mutual-information-selected features.
- Best `C`: {best["C"]:.6g}
- Best `gamma`: {best["gamma"]:.6g}
- Search mean F1: {best["mean_f1"]:.4f}
- 10-fold mean accuracy: {summary["mean_accuracy"]:.4f}
- 10-fold mean recall: {summary["mean_recall"]:.4f}
- 10-fold mean F1: {summary["mean_f1"]:.4f}
- 10-fold mean ROC-AUC: {summary["mean_roc_auc"]:.4f}

## Digital Twin

The website uses prediction responses to store a local baseline voice profile and compare later predictions against it. No patient data is stored on the backend.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps({"best": best, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
