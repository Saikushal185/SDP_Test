"""
Train production models once and persist the artifacts used by the Flask API.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
STUDY_DIR = PROJECT_ROOT / "parkinson_feature_study"
MODEL_DIR = BACKEND_DIR / "models"
RESULTS_METRICS_DIR = STUDY_DIR / "results" / "metrics"


def load_config() -> dict[str, Any]:
    config_path = STUDY_DIR / "config.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_dataset(config: dict[str, Any]) -> tuple[pd.DataFrame, list[str], str]:
    data_path = STUDY_DIR / config["data"]["raw_data_path"]
    dataset = pd.read_csv(data_path)

    target_column = config["data"]["target_column"]
    excluded = set(config["data"].get("exclude_columns", []))
    excluded.add(target_column)

    feature_columns = [column for column in dataset.columns if column not in excluded]
    logger.info(
        "Loaded dataset with %d rows and %d usable features",
        len(dataset),
        len(feature_columns),
    )
    return dataset, feature_columns, target_column


def select_top_features(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    config: dict[str, Any],
) -> list[str]:
    rf_config = config["feature_selection"]["classical"]
    random_seed = config["general"]["random_seed"]
    n_features = config["feature_selection"]["n_features_to_select"]

    X = dataset[feature_columns].astype(float).to_numpy()
    y = dataset[target_column].astype(int).to_numpy()

    feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(X)

    selector = RandomForestClassifier(
        n_estimators=rf_config.get("n_estimators", 200),
        max_depth=rf_config.get("max_depth", 10),
        min_samples_split=rf_config.get("min_samples_split", 5),
        min_samples_leaf=rf_config.get("min_samples_leaf", 2),
        random_state=random_seed,
        n_jobs=-1,
    )
    selector.fit(X_scaled, y)

    ranked_indices = np.argsort(selector.feature_importances_)[::-1][:n_features]
    selected_columns = [feature_columns[index] for index in ranked_indices]

    logger.info("Selected %d production features", len(selected_columns))
    return selected_columns


def build_models(config: dict[str, Any]) -> dict[str, Any]:
    random_seed = config["general"]["random_seed"]
    rf_config = config["feature_selection"]["classical"]
    xgb_config = config["models"]["xgboost"]

    return {
        "random_forest": RandomForestClassifier(
            n_estimators=rf_config.get("n_estimators", 200),
            max_depth=rf_config.get("max_depth", 10),
            min_samples_split=rf_config.get("min_samples_split", 5),
            min_samples_leaf=rf_config.get("min_samples_leaf", 2),
            random_state=random_seed,
            n_jobs=-1,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=xgb_config.get("n_estimators", 100),
            max_depth=xgb_config.get("max_depth", 6),
            learning_rate=xgb_config.get("learning_rate", 0.1),
            subsample=xgb_config.get("subsample", 0.8),
            colsample_bytree=xgb_config.get("colsample_bytree", 0.8),
            min_child_weight=xgb_config.get("min_child_weight", 1),
            gamma=xgb_config.get("gamma", 0),
            reg_alpha=xgb_config.get("reg_alpha", 0),
            reg_lambda=xgb_config.get("reg_lambda", 1),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_seed,
            n_jobs=1,
        ),
    }


def compute_metrics(
    dataset: pd.DataFrame,
    selected_columns: list[str],
    target_column: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    X = dataset[selected_columns].astype(float).to_numpy()
    y = dataset[target_column].astype(int).to_numpy()
    skf = StratifiedKFold(
        n_splits=config["cross_validation"]["n_folds"],
        shuffle=config["cross_validation"].get("shuffle", True),
        random_state=config["general"]["random_seed"],
    )

    metric_names = ("accuracy", "precision", "recall", "f1", "auc")
    metric_store: dict[str, dict[str, list[float]]] = {}
    roc_store: dict[str, dict[str, list[float]]] = {}

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        fold_scaler = StandardScaler()
        X_train_scaled = fold_scaler.fit_transform(X_train)
        X_test_scaled = fold_scaler.transform(X_test)

        for model_name, model in build_models(config).items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]

            model_metrics = metric_store.setdefault(
                model_name, {metric_name: [] for metric_name in metric_names}
            )
            model_metrics["accuracy"].append(float(accuracy_score(y_test, y_pred)))
            model_metrics["precision"].append(
                float(precision_score(y_test, y_pred, zero_division=0))
            )
            model_metrics["recall"].append(
                float(recall_score(y_test, y_pred, zero_division=0))
            )
            model_metrics["f1"].append(float(f1_score(y_test, y_pred, zero_division=0)))
            model_metrics["auc"].append(float(roc_auc_score(y_test, y_proba)))

            roc_values = roc_store.setdefault(model_name, {"y_true": [], "y_proba": []})
            roc_values["y_true"].extend(y_test.tolist())
            roc_values["y_proba"].extend(y_proba.tolist())

    models_payload: dict[str, dict[str, float]] = {}
    for model_name, values in metric_store.items():
        models_payload[model_name] = {
            metric_name: float(np.mean(metric_values))
            for metric_name, metric_values in values.items()
        }

    roc_payload: dict[str, dict[str, list[float]]] = {}
    for model_name, values in roc_store.items():
        fpr, tpr, _ = roc_curve(values["y_true"], values["y_proba"])
        if len(fpr) > 100:
            sample_indices = np.linspace(0, len(fpr) - 1, 100, dtype=int)
            fpr = fpr[sample_indices]
            tpr = tpr[sample_indices]
        roc_payload[model_name] = {
            "fpr": [float(point) for point in fpr],
            "tpr": [float(point) for point in tpr],
        }

    best_model = max(models_payload, key=lambda name: models_payload[name]["accuracy"])
    return {
        "models": models_payload,
        "best_model": best_model,
        "roc_data": roc_payload,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def save_artifacts() -> None:
    config = load_config()
    dataset, feature_columns, target_column = load_dataset(config)
    selected_columns = select_top_features(dataset, feature_columns, target_column, config)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    X_selected = dataset[selected_columns].astype(float).to_numpy()
    y = dataset[target_column].astype(int).to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    models = build_models(config)
    for model in models.values():
        model.fit(X_scaled, y)

    joblib.dump(models["random_forest"], MODEL_DIR / "random_forest.pkl")
    joblib.dump(models["xgboost"], MODEL_DIR / "xgboost.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    (MODEL_DIR / "feature_columns.json").write_text(
        json.dumps(selected_columns, indent=2),
        encoding="utf-8",
    )
    (MODEL_DIR / "sample_input.json").write_text(
        json.dumps(dataset[selected_columns].iloc[0].astype(float).to_dict(), indent=2),
        encoding="utf-8",
    )

    metrics_payload = compute_metrics(dataset, selected_columns, target_column, config)
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )
    (RESULTS_METRICS_DIR / "deployment_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )

    feature_importance = [
        {
            "name": selected_columns[position],
            "importance": float(models["random_forest"].feature_importances_[position]),
        }
        for position in range(len(selected_columns))
    ]
    (MODEL_DIR / "feature_importance.json").write_text(
        json.dumps(
            sorted(feature_importance, key=lambda item: item["importance"], reverse=True),
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Saved model artifacts to %s", MODEL_DIR)
    logger.info("Saved deployment metrics to %s", RESULTS_METRICS_DIR)


if __name__ == "__main__":
    save_artifacts()
