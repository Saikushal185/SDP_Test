from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.artifacts import (
    DatasetArtifactLayout,
    build_dataset_artifact_layout,
    write_feature_schema,
    write_model_manifest,
)


def test_build_dataset_artifact_layout_creates_expected_relative_paths(tmp_path: Path):
    layout = build_dataset_artifact_layout(tmp_path, "pd_speech_features_local")

    assert layout.dataset_dir == tmp_path / "pd_speech_features_local"
    assert layout.models_dir == layout.dataset_dir / "models"
    assert layout.results_dir == layout.dataset_dir / "results"
    assert layout.processed_dir == layout.dataset_dir / "processed"


def test_write_feature_schema_preserves_column_order(tmp_path: Path):
    layout = build_dataset_artifact_layout(tmp_path, "pd_speech_features_local")
    schema_path = write_feature_schema(layout, ["jitter", "shimmer", "mfcc_1"])

    schema = pd.read_csv(schema_path)

    assert schema["feature_name"].tolist() == ["jitter", "shimmer", "mfcc_1"]
    assert schema["position"].tolist() == [0, 1, 2]


def test_write_model_manifest_records_reloadable_artifacts(tmp_path: Path):
    layout = build_dataset_artifact_layout(tmp_path, "pd_speech_features_local")
    model_path = layout.models_dir / "dummy.joblib"
    schema_path = write_feature_schema(layout, ["jitter", "shimmer"])
    label_map_path = layout.processed_dir / "label_map.csv"
    pd.DataFrame(
        [
            {"label_name": "control", "label_value": 0},
            {"label_name": "parkinsons", "label_value": 1},
        ]
    ).to_csv(label_map_path, index=False)

    model = DummyClassifier(strategy="most_frequent")
    model.fit([[0.0, 0.0], [1.0, 1.0]], [0, 1])
    joblib.dump(model, model_path)

    manifest_path = write_model_manifest(
        layout,
        rows=[
            {
                "dataset_id": "pd_speech_features_local",
                "dataset_source": "local",
                "model_name": "Dummy",
                "model_family": "classical",
                "artifact_path": model_path,
                "preprocessor_path": "",
                "feature_schema_path": schema_path,
                "label_map_path": label_map_path,
                "cv_folds": 10,
                "mean_accuracy": 0.9,
                "mean_recall": 0.8,
                "mean_f1": 0.85,
                "mean_roc_auc": 0.88,
                "inference_ready": True,
            }
        ],
    )

    manifest = pd.read_csv(manifest_path)
    reloaded = joblib.load(manifest.loc[0, "artifact_path"])

    assert isinstance(layout, DatasetArtifactLayout)
    assert manifest.loc[0, "feature_schema_path"] == str(schema_path)
    assert reloaded.predict([[0.2, 0.1]]).tolist() == [0]
