import pandas as pd
from pathlib import Path
import numpy as np

from src.multi_dataset_pipeline import MultiDatasetPipeline


def test_normalize_tabular_dataframe_infers_binary_string_labels_and_drops_id_columns(tmp_path):
    pipeline = MultiDatasetPipeline(
        config={"general": {"random_seed": 42}, "cross_validation": {"n_folds": 10}},
        project_root=tmp_path,
    )
    raw_df = pd.DataFrame(
        {
            "id": [101, 102, 103, 104],
            "jitter": ["0.1", "0.2", "0.3", "0.4"],
            "shimmer": [0.01, 0.02, 0.03, 0.04],
            "status": ["healthy", "healthy", "parkinsons", "parkinsons"],
        }
    )

    features, labels, label_map = pipeline._normalize_tabular_dataframe(raw_df)

    assert features.columns.tolist() == ["jitter", "shimmer"]
    assert labels.tolist() == [0, 0, 1, 1]
    assert label_map == {0: "healthy", 1: "parkinsons"}


def test_build_prepared_dataset_caps_cv_folds_at_minority_class_count(tmp_path):
    pipeline = MultiDatasetPipeline(
        config={"general": {"random_seed": 42}, "cross_validation": {"n_folds": 10}},
        project_root=tmp_path,
    )
    prepared = pipeline._build_prepared_dataset(
        spec=type("Spec", (), {"dataset_id": "demo"})(),
        resolved_source_ref="demo",
        source_type="local_csv",
        features=pd.DataFrame({"f1": [0, 1, 2, 3, 4, 5], "f2": [5, 4, 3, 2, 1, 0]}),
        labels=[0, 0, 0, 1, 1, 1],
        label_map={0: "healthy", 1: "parkinsons"},
    )

    assert prepared.cv_folds == 3


def test_filter_audio_files_caps_xanjeev_subject_clips_and_ignores_placeholder_names(tmp_path):
    pipeline = MultiDatasetPipeline(
        config={"general": {"random_seed": 42}, "cross_validation": {"n_folds": 10}},
        project_root=tmp_path,
    )
    audio_files = [
        *[
            Path(f"F_Con/F_Con/wav_arrayMic_FC01S01_{index:04d}.wav")
            for index in range(1, 18)
        ],
        Path("F_Dys/F_Dys/wav_arrayMic_F01_0001.wav"),
        Path("F_Dys/F_Dys/wav_arrayMic_F01_0002.wav"),
        Path("M_Dys/M_Dys/.wav"),
    ]

    filtered = pipeline._filter_audio_files("XANJEEV/Parkinson_Classification_Dataset", audio_files)

    assert filtered == [
        *[
            Path(f"F_Con/F_Con/wav_arrayMic_FC01S01_{index:04d}.wav")
            for index in range(1, 16)
        ],
        Path("F_Dys/F_Dys/wav_arrayMic_F01_0001.wav"),
        Path("F_Dys/F_Dys/wav_arrayMic_F01_0002.wav"),
    ]


def test_select_fit_data_caps_quantum_training_samples_stratified_and_deterministic(tmp_path):
    pipeline = MultiDatasetPipeline(
        config={
            "general": {"random_seed": 42},
            "cross_validation": {"n_folds": 10},
            "quantum": {"max_train_samples": 6},
        },
        project_root=tmp_path,
    )
    X = pd.DataFrame({"f1": np.arange(20), "f2": np.arange(20, 40)})
    y = np.array([0] * 10 + [1] * 10)

    X_fit_a, y_fit_a, metadata_a = pipeline._select_fit_data(X, y, model_family="quantum", seed_offset=1)
    X_fit_b, y_fit_b, metadata_b = pipeline._select_fit_data(X, y, model_family="quantum", seed_offset=1)

    assert len(X_fit_a) == 6
    assert y_fit_a.tolist().count(0) == 3
    assert y_fit_a.tolist().count(1) == 3
    assert X_fit_a.equals(X_fit_b)
    assert y_fit_a.tolist() == y_fit_b.tolist()
    assert metadata_a == metadata_b == {
        "fit_sample_count": 6,
        "fit_strategy": "stratified_subsample",
    }


def test_save_model_artifact_writes_reconstruction_bundle_when_joblib_fails(tmp_path):
    class ExportableUnpicklableModel:
        model_name = "DemoQuantum"

        def __init__(self):
            self.bad_attr = lambda value: value

        def export_state(self):
            return {"model_name": self.model_name, "weights": [0.1, 0.2]}

    pipeline = MultiDatasetPipeline(
        config={"general": {"random_seed": 42}, "cross_validation": {"n_folds": 10}},
        project_root=tmp_path,
    )

    artifact_path, inference_ready = pipeline._save_model_artifact(
        ExportableUnpicklableModel(),
        tmp_path / "demo.joblib",
    )

    payload = __import__("json").loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact_path.suffix == ".json"
    assert inference_ready is False
    assert payload["serialization"] == "reconstruction_bundle"
    assert payload["model_name"] == "DemoQuantum"
    assert payload["weights"] == [0.1, 0.2]
