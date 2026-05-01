import pandas as pd
import pytest

from app.registry import ArtifactRegistry, SchemaValidationError


def test_registry_discovers_all_dataset_model_pairs():
    registry = ArtifactRegistry.create_default()

    models = registry.list_models()
    model_keys = {model["model_key"] for model in models}

    assert len(models) == 28
    assert "pd_speech_features_local_XGBoost" in model_keys
    assert "pd_speech_features_local_QSVM" in model_keys
    assert "pd_speech_features_local_VQC" in model_keys
    assert {
        model["model_name"]
        for model in models
        if model["dataset_id"] == "pd_speech_features_local"
    } == {
        "LogisticRegression",
        "SVM",
        "RandomForest",
        "XGBoost",
        "KNN",
        "QSVM",
        "VQC",
    }


@pytest.mark.parametrize(
    "model_key",
    [
        "pd_speech_features_local_XGBoost",
        "pd_speech_features_local_QSVM",
        "pd_speech_features_local_VQC",
    ],
)
def test_prediction_returns_valid_probability_for_classical_qsvm_and_repaired_vqc(model_key):
    registry = ArtifactRegistry.create_default()
    sample = registry.load_sample_rows("pd_speech_features_local", limit=1)["rows"][0]
    frame = pd.DataFrame([sample["features"]])

    predictions = registry.predict(model_key, frame)

    assert len(predictions) == 1
    assert predictions[0]["row_index"] == 0
    assert 0.0 <= predictions[0]["probability"] <= 1.0
    assert predictions[0]["predicted_label"] in {"Healthy Control", "Parkinson's (PD)"}


def test_validate_schema_rejects_missing_and_extra_columns():
    registry = ArtifactRegistry.create_default()
    record = registry.get_model("pd_speech_features_local_XGBoost")
    feature_names = registry.feature_names(record.dataset_id)
    bad_frame = pd.DataFrame([{feature_names[0]: 0.0, "unexpected": 1.0}])

    with pytest.raises(SchemaValidationError) as exc_info:
        registry.validate_prediction_frame(record.dataset_id, bad_frame)

    message = str(exc_info.value)
    assert "missing" in message
    assert "unexpected" in message


def test_prediction_ignores_known_uploaded_csv_metadata_columns():
    registry = ArtifactRegistry.create_default()
    sample = registry.load_sample_rows("pd_speech_features_local", limit=1)["rows"][0]
    row = {"id": 101, **sample["features"], "class": sample["label"], "target": 1}
    frame = pd.DataFrame([row])

    prediction = registry.predict("pd_speech_features_local_XGBoost", frame, source="upload.csv")[0]

    assert 0.0 <= prediction["probability"] <= 1.0
    assert prediction["source"] == "upload.csv"


@pytest.mark.parametrize(
    "model_key",
    [
        "pd_speech_features_local_XGBoost",
        "pd_speech_features_local_RandomForest",
        "pd_speech_features_local_LogisticRegression",
        "pd_speech_features_local_VQC",
    ],
)
def test_grouped_explanation_adds_up_to_displayed_probability(model_key):
    registry = ArtifactRegistry.create_default()
    sample = registry.load_sample_rows("pd_speech_features_local", limit=1)["rows"][0]
    frame = pd.DataFrame([sample["features"]])

    prediction = registry.predict(model_key, frame)[0]
    explanation = prediction["explanation"]
    explained_probability = explanation["base_value"] + sum(
        group["value"] for group in explanation["groups"]
    )

    assert explanation["output_scale"] == "probability"
    assert explained_probability == pytest.approx(prediction["probability"], abs=1e-4)


def test_feature_importance_and_group_impact_are_chart_ready():
    registry = ArtifactRegistry.create_default()

    importances = registry.feature_importance("pd_speech_features_local_XGBoost", limit=8)
    impact = registry.group_impact("pd_speech_features_local_XGBoost")

    assert len(importances) == 8
    assert all({"feature", "importance"} <= set(row) for row in importances)
    assert impact["series"]
    assert impact["summary"]
    assert {row["group"] for row in impact["summary"]} >= {
        "Healthy Control",
        "Parkinson's (PD)",
    }


@pytest.mark.parametrize(
    ("model_key", "expected_method"),
    [
        ("pd_speech_features_local_RandomForest", "native"),
        ("pd_speech_features_local_VQC", "kernel-grouped"),
    ],
)
def test_prediction_includes_grouped_explanation_for_native_and_fallback_models(
    model_key,
    expected_method,
):
    registry = ArtifactRegistry.create_default()
    sample = registry.load_sample_rows("pd_speech_features_local", limit=1)["rows"][0]
    frame = pd.DataFrame([sample["features"]])

    prediction = registry.predict(model_key, frame)[0]

    assert 0.0 <= prediction["confidence"] <= 1.0
    assert prediction["explanation"]["method"] == expected_method
    assert prediction["explanation"]["groups"]
    assert all(
        {"name", "value", "absValue", "featureCount"} <= set(group)
        for group in prediction["explanation"]["groups"]
    )
    assert all(
        pd.notna(group["value"]) and pd.notna(group["absValue"])
        for group in prediction["explanation"]["groups"]
    )
