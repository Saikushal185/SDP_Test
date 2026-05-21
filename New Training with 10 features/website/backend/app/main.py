from __future__ import annotations

import json
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .audio_top10 import AudioFeatureExtractionError, extract_top10_audio_features
from .registry import ArtifactRegistry, SchemaValidationError


class ModelsResponse(BaseModel):
    model_count: int = Field(examples=[5])
    models: list[dict[str, Any]] = Field(
        examples=[
            [
                {
                    "model_key": "pd_speech_features_local_top10_mi_XGBoost",
                    "model_name": "XGBoost",
                    "status": "ready",
                    "inference_ready": True,
                }
            ]
        ]
    )


app = FastAPI(title="PD Model Lab API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = ArtifactRegistry.create_default()


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_count": len(registry.records)}


@app.get("/api/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    model_records = registry.list_models()
    return ModelsResponse(model_count=len(model_records), models=model_records)


@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    return registry.dashboard()


@app.get("/api/samples/{dataset_id}")
def samples(dataset_id: str, limit: int = 5) -> dict[str, Any]:
    try:
        return registry.load_sample_rows(dataset_id, limit=limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/features/{model_key}")
def features(model_key: str, limit: int = 15) -> dict[str, Any]:
    try:
        return {"model_key": model_key, "features": registry.feature_importance(model_key, limit=limit)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/group-impact/{model_key}")
def group_impact(model_key: str) -> dict[str, Any]:
    try:
        return registry.group_impact(model_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/confusion-matrix/{model_key}")
def confusion_matrix(model_key: str) -> dict[str, Any]:
    try:
        return registry.confusion_matrix(model_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/predict")
async def predict(
    model_key: str = Form(...),
    sample_row: int = Form(0),
    edited_features: str | None = Form(None),
    file: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    try:
        record = registry.get_model(model_key)
        if file is not None:
            frame = registry.parse_csv(await file.read())
            source = file.filename or "uploaded.csv"
        else:
            samples_payload = registry.load_sample_rows(record.dataset_id, limit=max(sample_row + 1, 1))
            sample = samples_payload["rows"][sample_row]["features"]
            if edited_features:
                sample.update(json.loads(edited_features))
            frame = pd.DataFrame([sample])
            source = "sample-edited" if edited_features else "sample"

        predictions = registry.predict(model_key, frame, source=source)
        return {"model_key": model_key, "predictions": predictions}
    except SchemaValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/predict-audio")
async def predict_audio(
    model_key: str = Form(...),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    try:
        features = extract_top10_audio_features(await file.read(), filename=file.filename or "audio")
        frame = pd.DataFrame([features])
        predictions = registry.predict(model_key, frame, source=file.filename or "audio")
        return {
            "model_key": model_key,
            "extracted_features": features,
            "predictions": predictions,
        }
    except AudioFeatureExtractionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except SchemaValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
