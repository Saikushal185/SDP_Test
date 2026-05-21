from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.top10_feature_training import (
    REQUESTED_MODEL_NAMES,
    rank_features_mutual_info,
    run_reduced_feature_experiment,
    select_fold_top_features,
)


def _demo_dataset(row_count: int = 40, feature_count: int = 12) -> pd.DataFrame:
    y = np.array([0, 1] * (row_count // 2), dtype=int)
    data = {"id": np.arange(1, row_count + 1), "class": y}
    for index in range(feature_count):
        if index % 3 == 0:
            data[f"f{index}"] = y + (index / 100.0)
        elif index % 3 == 1:
            data[f"f{index}"] = np.arange(row_count) % (index + 2)
        else:
            data[f"f{index}"] = np.linspace(0.0, 1.0, row_count) * (index + 1)
    return pd.DataFrame(data)


def test_rank_features_mutual_info_selects_exactly_requested_valid_features():
    df = _demo_dataset()
    X = df.drop(columns=["id", "class"])
    y = df["class"].to_numpy()

    ranking = rank_features_mutual_info(X, y, top_k=10, random_seed=42)

    assert len(ranking) == 10
    assert ranking["rank"].tolist() == list(range(1, 11))
    assert set(ranking["feature"]).issubset(set(X.columns))
    assert ranking["mutual_info_score"].is_monotonic_decreasing


def test_select_fold_top_features_uses_only_training_rows():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1], dtype=int)
    X = pd.DataFrame(
        {
            "train_signal": [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0],
            "test_only_signal": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            "noise": np.arange(len(y)),
        }
    )
    train_idx = np.arange(8)

    selected, ranking = select_fold_top_features(X, y, train_idx, top_k=1, random_seed=42)

    assert selected == ["train_signal"]
    assert ranking.iloc[0]["feature"] == "train_signal"


def test_run_reduced_feature_experiment_writes_outputs_for_requested_models(tmp_path: Path):
    source_csv = tmp_path / "pd_speech_features.csv"
    _demo_dataset(row_count=40, feature_count=12).to_csv(source_csv, index=False)
    output_dir = tmp_path / "New Training with 10 features"
    model_factories = {
        name: (lambda seed=42: DummyClassifier(strategy="most_frequent"))
        for name in REQUESTED_MODEL_NAMES
    }

    result = run_reduced_feature_experiment(
        source_csv=source_csv,
        output_dir=output_dir,
        top_k=10,
        n_splits=2,
        model_factories=model_factories,
        random_seed=42,
    )

    assert result.output_dir == output_dir
    assert (output_dir / "processed" / "cleaned_features.csv").exists()
    assert (output_dir / "processed" / "final_top10_features.csv").exists()
    assert (output_dir / "results" / "cv_fold_metrics.csv").exists()
    assert (output_dir / "results" / "model_comparison.csv").exists()
    assert (output_dir / "results" / "selected_features_by_fold.csv").exists()
    assert (output_dir / "results" / "feature_rankings_by_fold.csv").exists()
    assert (output_dir / "results" / "oof_predictions.csv").exists()
    assert (output_dir / "results" / "confusion_matrix.csv").exists()
    assert (output_dir / "model_manifest.csv").exists()

    final_features = pd.read_csv(output_dir / "processed" / "final_top10_features.csv")
    comparison = pd.read_csv(output_dir / "results" / "model_comparison.csv")
    fold_metrics = pd.read_csv(output_dir / "results" / "cv_fold_metrics.csv")
    selected_by_fold = pd.read_csv(output_dir / "results" / "selected_features_by_fold.csv")

    assert len(final_features) == 10
    assert set(comparison["model_name"]) == set(REQUESTED_MODEL_NAMES)
    assert set(fold_metrics["model_name"]) == set(REQUESTED_MODEL_NAMES)
    assert (fold_metrics.groupby("model_name")["fold"].nunique() == 2).all()
    assert selected_by_fold.groupby("fold")["feature"].nunique().tolist() == [10, 10]


def test_run_reduced_feature_experiment_writes_json_fallback_when_model_pickle_fails(tmp_path: Path):
    class UnpicklableDummyClassifier(DummyClassifier):
        def __init__(self):
            super().__init__(strategy="most_frequent")
            self.bad_attr = lambda value: value

        def export_state(self):
            return {"model_name": "UnpicklableDummyClassifier"}

    source_csv = tmp_path / "pd_speech_features.csv"
    _demo_dataset(row_count=40, feature_count=12).to_csv(source_csv, index=False)
    output_dir = tmp_path / "New Training with 10 features"
    model_factories = {
        name: (lambda seed=42: UnpicklableDummyClassifier())
        for name in REQUESTED_MODEL_NAMES
    }

    run_reduced_feature_experiment(
        source_csv=source_csv,
        output_dir=output_dir,
        top_k=10,
        n_splits=2,
        model_factories=model_factories,
        random_seed=42,
    )

    assert (output_dir / "models" / "svm.json").exists()
    assert not (output_dir / "models" / "svm.joblib").exists()
