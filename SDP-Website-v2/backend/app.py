"""
Flask API for Parkinson's disease prediction, explainability, and metrics.

The app loads persisted training artifacts once on startup and serves JSON
responses that the Next.js frontend can render directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from ml_pipeline import MLPipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _extract_features_payload(payload: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    model_name = str(payload.get("model", "xgboost"))

    if "features" in payload:
        features = payload.get("features")
    else:
        features = {key: value for key, value in payload.items() if key != "model"}

    if not isinstance(features, dict) or not features:
        raise ValueError("No features provided in the request body.")

    return features, model_name


def create_app(pipeline_override: MLPipeline | None = None) -> Flask:
    app = Flask(__name__)
    CORS(app)

    pipeline = pipeline_override or MLPipeline(str(PROJECT_ROOT))
    if pipeline_override is None:
        try:
            pipeline.load_artifacts()
        except Exception as exc:
            pipeline.startup_error = str(exc)
            pipeline.is_ready = False
            logger.error("Failed to load model artifacts: %s", exc, exc_info=True)

    app.config["PIPELINE"] = pipeline

    @app.before_request
    def ensure_pipeline_ready():
        if request.path in {"/health"}:
            return None

        active_pipeline: MLPipeline = app.config["PIPELINE"]
        if active_pipeline.is_ready:
            return None

        return (
            jsonify(
                {
                    "error": active_pipeline.startup_error
                    or "Model artifacts are not loaded yet."
                }
            ),
            503,
        )

    @app.get("/health")
    def health():
        active_pipeline: MLPipeline = app.config["PIPELINE"]
        return jsonify(
            {
                "status": "ok",
                "models_loaded": active_pipeline.is_ready,
                "error": active_pipeline.startup_error,
            }
        )

    @app.get("/features")
    @app.get("/api/features")
    def get_features():
        active_pipeline: MLPipeline = app.config["PIPELINE"]
        return jsonify(active_pipeline.get_features_metadata())

    @app.get("/model-registry")
    @app.get("/api/model-registry")
    def model_registry():
        active_pipeline: MLPipeline = app.config["PIPELINE"]
        return jsonify(active_pipeline.get_model_registry())

    @app.post("/predict")
    @app.post("/api/predict")
    def predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be valid JSON."}), 400

        try:
            features, model_name = _extract_features_payload(payload)
            active_pipeline: MLPipeline = app.config["PIPELINE"]
            return jsonify(active_pipeline.predict(features, model_name))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            logger.error("Prediction failed: %s", exc, exc_info=True)
            return jsonify({"error": "Prediction failed."}), 500

    @app.post("/explain")
    @app.post("/api/explain")
    def explain():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be valid JSON."}), 400

        try:
            features, model_name = _extract_features_payload(payload)
            active_pipeline: MLPipeline = app.config["PIPELINE"]
            return jsonify(active_pipeline.explain(features, model_name))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            logger.error("Explainability failed: %s", exc, exc_info=True)
            return jsonify({"error": "Explainability failed."}), 500

    @app.get("/metrics")
    @app.get("/api/metrics")
    @app.get("/api/performance")
    def metrics():
        active_pipeline: MLPipeline = app.config["PIPELINE"]
        return jsonify(active_pipeline.get_metrics())

    @app.get("/model-info")
    @app.get("/api/model-info")
    def model_info():
        active_pipeline: MLPipeline = app.config["PIPELINE"]
        return jsonify(active_pipeline.get_model_info())

    @app.get("/dataset-template")
    @app.get("/api/dataset-template")
    def dataset_template():
        active_pipeline: MLPipeline = app.config["PIPELINE"]
        payload = active_pipeline.get_dataset_template()
        if request.args.get("format") == "csv":
            return Response(
                payload["csv"],
                mimetype="text/csv",
                headers={
                    "Content-Disposition": "attachment; filename=pd_speech_features_template.csv"
                },
            )
        return jsonify(payload)

    @app.get("/cross-dataset-summary")
    @app.get("/api/cross-dataset-summary")
    def cross_dataset_summary():
        active_pipeline: MLPipeline = app.config["PIPELINE"]
        return jsonify(active_pipeline.get_cross_dataset_summary())

    @app.post("/batch-evaluate")
    @app.post("/api/batch-evaluate")
    def batch_evaluate():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be valid JSON."}), 400

        csv_text = payload.get("csv_text") or payload.get("csv")
        model_name = str(payload.get("model", "xgboost"))
        try:
            active_pipeline: MLPipeline = app.config["PIPELINE"]
            return jsonify(active_pipeline.batch_evaluate(str(csv_text or ""), model_name))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            logger.error("Batch evaluation failed: %s", exc, exc_info=True)
            return jsonify({"error": "Batch evaluation failed."}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True, port=5000)
