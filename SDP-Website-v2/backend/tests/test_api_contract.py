import importlib
import json
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
LOCAL_DEPS = BACKEND_DIR / ".deps"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if LOCAL_DEPS.exists() and str(LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPS))


class DummyPipeline:
    def __init__(self) -> None:
        self.is_ready = True
        self.startup_error = None

    def predict(self, features, model_name="xgboost"):
        return {
            "prediction": 1,
            "probability": 0.82,
            "confidence": 0.82,
        }

    def explain(self, features, model_name="xgboost"):
        return {
            "shap_values": [0.14, -0.03],
            "feature_names": ["f1", "f2"],
            "base_value": 0.41,
            "prediction": 1,
        }

    def get_metrics(self):
        return {
            "models": {
                "xgboost": {
                    "accuracy": 0.91,
                    "precision": 0.89,
                    "recall": 0.92,
                    "f1": 0.9,
                    "auc": 0.95,
                }
            },
            "best_model": "xgboost",
        }

    def get_features_metadata(self):
        return {
            "expected_features": ["f1", "f2"],
            "feature_count": 2,
            "sample_data": {"f1": 0.1, "f2": 0.2},
            "supported_models": ["random_forest", "xgboost"],
            "feature_groups": [
                {
                    "name": "Voice stability",
                    "description": "Demo group",
                    "features": ["f1", "f2"],
                }
            ],
            "training_dataset": "pd_speech_features.csv",
        }

    def get_model_registry(self):
        return {
            "training_dataset": "pd_speech_features.csv",
            "best_model": "xgboost",
            "models": [
                {
                    "key": "xgboost",
                    "display_name": "Configurable XGBoost",
                    "dataset_id": "pd_speech_features.csv",
                    "enabled": True,
                    "artifact_path": "models/xgboost.pkl",
                    "metrics": self.get_metrics()["models"]["xgboost"],
                    "feature_schema": ["f1", "f2"],
                }
            ],
        }

    def get_model_info(self):
        return {
            "dataset_size": 6,
            "n_selected_features": 2,
            "models": ["xgboost"],
            "model_registry": self.get_model_registry()["models"],
            "best_model": "xgboost",
            "best_accuracy": 0.91,
            "training_dataset": "pd_speech_features.csv",
        }

    def get_dataset_template(self):
        return {
            "training_dataset": "pd_speech_features.csv",
            "required_features": ["f1", "f2"],
            "sample_data": {"f1": 0.1, "f2": 0.2},
            "csv": "f1,f2,class\r\n0.1,0.2,1\r\n",
        }

    def get_cross_dataset_summary(self):
        return {
            "training_dataset": "pd_speech_features.csv",
            "strict_policy": "External datasets must contain every deployed training feature.",
            "datasets": [
                {
                    "dataset_id": "demo_external",
                    "feature_count": 1,
                    "required_overlap": 1,
                    "missing_required_count": 1,
                    "strict_compatible": False,
                    "best_model": "XGBoost",
                    "best_f1": 0.7,
                    "note": "Schema differs.",
                }
            ],
        }

    def batch_evaluate(self, csv_text, model_name="xgboost"):
        if "f2" not in csv_text:
            return {
                "compatible": False,
                "model": model_name,
                "display_name": "Configurable XGBoost",
                "training_dataset": "pd_speech_features.csv",
                "row_count": 1,
                "required_feature_count": 2,
                "present_feature_count": 1,
                "missing_columns": ["f2"],
                "ignored_columns": [],
                "label_column": None,
                "metrics": None,
                "prediction_summary": None,
                "predictions": [],
                "message": "Dataset is not compatible.",
            }
        return {
            "compatible": True,
            "model": model_name,
            "display_name": "Configurable XGBoost",
            "training_dataset": "pd_speech_features.csv",
            "row_count": 2,
            "required_feature_count": 2,
            "present_feature_count": 2,
            "missing_columns": [],
            "ignored_columns": [],
            "label_column": "class",
            "metrics": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "auc": 1.0},
            "prediction_summary": {"positive": 1, "negative": 1, "mean_probability": 0.5},
            "predictions": [{"row_index": 0, "prediction": 1, "probability": 0.82, "confidence": 0.82}],
            "message": "Dataset passed strict feature compatibility.",
        }


def load_app_module():
    with patch.object(threading.Thread, "start", lambda self: None):
        app_module = importlib.import_module("app")
        return importlib.reload(app_module)


class ApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app_module = load_app_module()
        cls.app = cls.app_module.create_app(pipeline_override=DummyPipeline())
        cls.client = cls.app.test_client()

    def test_predict_endpoint_returns_required_contract(self) -> None:
        response = self.client.post(
            "/predict",
            json={"model": "xgboost", "features": {"f1": 0.1, "f2": 0.2}},
        )

        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertEqual(
            set(payload.keys()),
            {"prediction", "probability", "confidence"},
        )
        self.assertIsInstance(payload["prediction"], int)
        self.assertIsInstance(payload["probability"], float)
        self.assertIsInstance(payload["confidence"], float)

    def test_explain_endpoint_returns_local_shap_json(self) -> None:
        response = self.client.post(
            "/explain",
            json={"model": "random_forest", "features": {"f1": 0.1, "f2": 0.2}},
        )

        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertEqual(
            set(payload.keys()),
            {"shap_values", "feature_names", "base_value", "prediction"},
        )
        self.assertEqual(len(payload["shap_values"]), len(payload["feature_names"]))
        self.assertIsInstance(payload["base_value"], float)
        self.assertIsInstance(payload["prediction"], int)

    def test_metrics_endpoint_returns_saved_metrics_json(self) -> None:
        response = self.client.get("/metrics")

        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertIn("models", payload)
        self.assertIn("best_model", payload)
        self.assertIn("xgboost", payload["models"])

    def test_model_registry_exposes_configurable_display_names(self) -> None:
        response = self.client.get("/api/model-registry")

        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertEqual(payload["training_dataset"], "pd_speech_features.csv")
        self.assertEqual(payload["models"][0]["display_name"], "Configurable XGBoost")
        self.assertEqual(payload["models"][0]["dataset_id"], "pd_speech_features.csv")

    def test_dataset_template_can_be_returned_as_csv(self) -> None:
        response = self.client.get("/api/dataset-template?format=csv")

        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertIn("text/csv", response.content_type)
        self.assertIn("f1,f2,class", response.get_data(as_text=True))

    def test_batch_evaluate_reports_strict_incompatible_csv(self) -> None:
        response = self.client.post(
            "/api/batch-evaluate",
            json={"model": "xgboost", "csv_text": "f1,class\n0.1,1\n"},
        )

        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertFalse(payload["compatible"])
        self.assertEqual(payload["missing_columns"], ["f2"])
        self.assertEqual(payload["predictions"], [])

    def test_batch_evaluate_returns_metrics_for_compatible_labeled_csv(self) -> None:
        response = self.client.post(
            "/api/batch-evaluate",
            json={"model": "xgboost", "csv_text": "f1,f2,class\n0.1,0.2,1\n0.2,0.1,0\n"},
        )

        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertTrue(payload["compatible"])
        self.assertEqual(payload["label_column"], "class")
        self.assertIn("accuracy", payload["metrics"])

    def test_cross_dataset_summary_marks_schema_context(self) -> None:
        response = self.client.get("/api/cross-dataset-summary")

        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertEqual(payload["training_dataset"], "pd_speech_features.csv")
        self.assertFalse(payload["datasets"][0]["strict_compatible"])

    def test_predict_returns_validation_error_for_feature_mismatch(self) -> None:
        def raise_validation_error(features, model_name="xgboost"):
            raise ValueError("Feature mismatch. Missing: ['f2']")

        self.app.config["PIPELINE"].predict = raise_validation_error

        response = self.client.post(
            "/predict",
            json={"model": "xgboost", "features": {"f1": 0.1}},
        )

        self.assertEqual(response.status_code, 400, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertIn("Feature mismatch", json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
