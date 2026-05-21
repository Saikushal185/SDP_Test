from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy.io import wavfile

from app.audio_top10 import TOP10_FEATURE_NAMES, extract_top10_audio_features
from app.main import app
from app.registry import ArtifactRegistry, SchemaValidationError


EXPECTED_MODEL_KEYS = {
    "pd_speech_features_local_top10_mi_SVM",
    "pd_speech_features_local_top10_mi_QSVM",
    "pd_speech_features_local_top10_mi_VQC",
    "pd_speech_features_local_top10_mi_XGBoost",
    "pd_speech_features_local_top10_mi_PSO-SVM",
}
DATASET_ID = "pd_speech_features_local_top10_mi"


def _wav_bytes(duration_seconds: float = 1.0, sample_rate: int = 16_000) -> bytes:
    time = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    signal = 0.35 * np.sin(2 * np.pi * 180 * time) + 0.08 * np.sin(2 * np.pi * 360 * time)
    pcm = np.clip(signal * np.iinfo(np.int16).max, -32768, 32767).astype(np.int16)
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, pcm)
    return buffer.getvalue()


def test_registry_discovers_top10_models_with_pso_svm():
    registry = ArtifactRegistry.create_default()

    models = registry.list_models()
    model_keys = {model["model_key"] for model in models}

    assert model_keys == EXPECTED_MODEL_KEYS
    assert "pd_speech_features_local_top10_mi_LogisticRegression" not in model_keys
    assert {model["dataset_id"] for model in models} == {DATASET_ID}
    assert {model["metrics"]["feature_selection_method"] for model in models} == {"mutual_information"}
    assert {model["metrics"]["selected_feature_count"] for model in models} == {10.0}
    assert {model["metrics"]["feature_mode"] for model in models} == {"direct_top10", "pso_top10"}
    assert any(model["display_family"] == "Swarm AI" for model in models)


def test_models_endpoint_returns_model_count():
    client = TestClient(app)

    response = client.get("/api/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_count"] == len(payload["models"]) == len(EXPECTED_MODEL_KEYS)
    assert {model["model_key"] for model in payload["models"]} == EXPECTED_MODEL_KEYS


def test_feature_schema_is_exact_top10_selection():
    registry = ArtifactRegistry.create_default()

    feature_names = registry.feature_names(DATASET_ID)

    assert feature_names == TOP10_FEATURE_NAMES
    assert len(feature_names) == 10


def test_vqc_direct_top10_json_loads_without_pca():
    registry = ArtifactRegistry.create_default()

    model = registry.load_model("pd_speech_features_local_top10_mi_VQC")

    assert model.use_pca is False
    assert model.n_components_ == 10
    assert model.pca is None


def test_prediction_returns_valid_probability_for_top10_xgboost():
    registry = ArtifactRegistry.create_default()
    sample = registry.load_sample_rows(DATASET_ID, limit=1)["rows"][0]
    frame = pd.DataFrame([sample["features"]])

    predictions = registry.predict("pd_speech_features_local_top10_mi_XGBoost", frame)

    assert len(predictions) == 1
    assert predictions[0]["row_index"] == 0
    assert 0.0 <= predictions[0]["probability"] <= 1.0
    assert predictions[0]["predicted_label"] in {"Healthy Control", "Parkinson's (PD)"}
    assert tuple(predictions[0]["input_features"]) == TOP10_FEATURE_NAMES


def test_prediction_explanation_groups_include_feature_names():
    registry = ArtifactRegistry.create_default()
    sample = registry.load_sample_rows(DATASET_ID, limit=1)["rows"][0]
    frame = pd.DataFrame([sample["features"]])

    prediction = registry.predict("pd_speech_features_local_top10_mi_XGBoost", frame)[0]
    groups = prediction["explanation"]["groups"]

    assert groups
    assert all(group["features"] for group in groups)
    assert all(group["featureCount"] == len(group["features"]) for group in groups)
    assert set().union(*(set(group["features"]) for group in groups)) == set(TOP10_FEATURE_NAMES)


def test_prediction_returns_valid_probability_for_top10_pso_svm():
    registry = ArtifactRegistry.create_default()
    sample = registry.load_sample_rows(DATASET_ID, limit=1)["rows"][0]
    frame = pd.DataFrame([sample["features"]])

    predictions = registry.predict("pd_speech_features_local_top10_mi_PSO-SVM", frame)

    assert len(predictions) == 1
    assert 0.0 <= predictions[0]["probability"] <= 1.0
    assert predictions[0]["predicted_label"] in {"Healthy Control", "Parkinson's (PD)"}
    assert tuple(predictions[0]["input_features"]) == TOP10_FEATURE_NAMES


def test_prediction_returns_valid_probability_for_top10_qsvm():
    registry = ArtifactRegistry.create_default()
    sample = registry.load_sample_rows(DATASET_ID, limit=1)["rows"][0]
    frame = pd.DataFrame([sample["features"]])

    predictions = registry.predict("pd_speech_features_local_top10_mi_QSVM", frame)

    assert len(predictions) == 1
    assert 0.0 <= predictions[0]["probability"] <= 1.0
    assert predictions[0]["probability"] < 0.99
    assert predictions[0]["confidence"] < 0.99
    assert predictions[0]["predicted_label"] in {"Healthy Control", "Parkinson's (PD)"}
    assert predictions[0]["explanation"]["method"] == "surrogate-local"


def test_validate_schema_rejects_missing_and_extra_columns():
    registry = ArtifactRegistry.create_default()
    feature_names = registry.feature_names(DATASET_ID)
    bad_frame = pd.DataFrame([{feature_names[0]: 0.0, "unexpected": 1.0}])

    with pytest.raises(SchemaValidationError) as exc_info:
        registry.validate_prediction_frame(DATASET_ID, bad_frame)

    message = str(exc_info.value)
    assert "missing" in message
    assert "unexpected" in message


def test_prediction_ignores_known_uploaded_csv_metadata_columns():
    registry = ArtifactRegistry.create_default()
    sample = registry.load_sample_rows(DATASET_ID, limit=1)["rows"][0]
    row = {"id": 101, **sample["features"], "target": sample["label"]}
    frame = pd.DataFrame([row])

    prediction = registry.predict(
        "pd_speech_features_local_top10_mi_XGBoost",
        frame,
        source="upload.csv",
    )[0]

    assert 0.0 <= prediction["probability"] <= 1.0
    assert prediction["source"] == "upload.csv"


def test_audio_extractor_returns_exactly_required_feature_names():
    features = extract_top10_audio_features(_wav_bytes(), filename="voice.wav")

    assert tuple(features) == TOP10_FEATURE_NAMES
    assert all(np.isfinite(value) for value in features.values())


def test_predict_audio_rejects_unsupported_files():
    client = TestClient(app)

    response = client.post(
        "/api/predict-audio",
        data={"model_key": "pd_speech_features_local_top10_mi_XGBoost"},
        files={"file": ("notes.txt", b"not audio", "text/plain")},
    )

    assert response.status_code == 415


def test_predict_audio_returns_prediction_and_extracted_features():
    client = TestClient(app)

    response = client.post(
        "/api/predict-audio",
        data={"model_key": "pd_speech_features_local_top10_mi_XGBoost"},
        files={"file": ("voice.wav", _wav_bytes(), "audio/wav")},
    )

    assert response.status_code == 200
    payload = response.json()
    prediction = payload["predictions"][0]
    assert payload["model_key"] == "pd_speech_features_local_top10_mi_XGBoost"
    assert tuple(payload["extracted_features"]) == TOP10_FEATURE_NAMES
    assert 0.0 <= prediction["probability"] <= 1.0
    assert 0.0 <= prediction["confidence"] <= 1.0
    assert prediction["predicted_label"] in {"Healthy Control", "Parkinson's (PD)"}


def test_confusion_matrix_endpoint_returns_four_cells_for_xgboost():
    client = TestClient(app)

    response = client.get("/api/confusion-matrix/pd_speech_features_local_top10_mi_XGBoost")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_key"] == "pd_speech_features_local_top10_mi_XGBoost"
    assert payload["model_name"] == "XGBoost"
    assert len(payload["cells"]) == 4
    assert {
        (cell["actual_label"], cell["predicted_label"])
        for cell in payload["cells"]
    } == {
        ("Healthy Control", "Healthy Control"),
        ("Healthy Control", "Parkinson's (PD)"),
        ("Parkinson's (PD)", "Healthy Control"),
        ("Parkinson's (PD)", "Parkinson's (PD)"),
    }
    assert sum(cell["count"] for cell in payload["cells"]) == 756


def test_confusion_matrix_endpoint_rejects_unknown_model_key():
    client = TestClient(app)

    response = client.get("/api/confusion-matrix/not_a_real_model")

    assert response.status_code == 404


def test_confusion_matrix_exists_for_every_top10_model():
    client = TestClient(app)

    for model_key in EXPECTED_MODEL_KEYS:
        response = client.get(f"/api/confusion-matrix/{model_key}")

        assert response.status_code == 200
        assert len(response.json()["cells"]) == 4
