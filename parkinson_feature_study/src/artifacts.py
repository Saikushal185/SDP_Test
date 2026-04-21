"""
Artifact layout and manifest helpers for the multi-dataset Parkinson study.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class DatasetArtifactLayout:
    """Directory layout for one dataset's exported artifacts."""

    dataset_dir: Path
    models_dir: Path
    results_dir: Path
    processed_dir: Path


def build_dataset_artifact_layout(root_dir: Path, dataset_id: str) -> DatasetArtifactLayout:
    """Create and return the standard directory layout for a dataset."""
    dataset_dir = Path(root_dir) / dataset_id
    models_dir = dataset_dir / "models"
    results_dir = dataset_dir / "results"
    processed_dir = dataset_dir / "processed"

    for path in (dataset_dir, models_dir, results_dir, processed_dir):
        path.mkdir(parents=True, exist_ok=True)

    return DatasetArtifactLayout(
        dataset_dir=dataset_dir,
        models_dir=models_dir,
        results_dir=results_dir,
        processed_dir=processed_dir,
    )


def _normalize_row_values(row: Mapping[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, Path):
            normalized[key] = str(value)
        else:
            normalized[key] = value
    return normalized


def write_feature_schema(layout: DatasetArtifactLayout, feature_names: Sequence[str]) -> Path:
    """Write ordered feature metadata for inference-time form generation."""
    schema = pd.DataFrame(
        {
            "position": list(range(len(feature_names))),
            "feature_name": list(feature_names),
        }
    )
    schema_path = layout.processed_dir / "feature_schema.csv"
    schema.to_csv(schema_path, index=False)
    return schema_path


def write_model_manifest(
    layout: DatasetArtifactLayout,
    rows: Iterable[Mapping[str, object]],
    filename: str = "model_manifest.csv",
) -> Path:
    """Write a per-dataset model manifest for website integration."""
    manifest = pd.DataFrame([_normalize_row_values(row) for row in rows])
    manifest_path = layout.dataset_dir / filename
    manifest.to_csv(manifest_path, index=False)
    return manifest_path


def write_label_map(layout: DatasetArtifactLayout, label_map: Mapping[int, str]) -> Path:
    """Write the dataset label encoding used during training."""
    label_map_df = pd.DataFrame(
        [
            {"label_value": int(label_value), "label_name": label_name}
            for label_value, label_name in sorted(label_map.items())
        ]
    )
    label_map_path = layout.processed_dir / "label_map.csv"
    label_map_df.to_csv(label_map_path, index=False)
    return label_map_path
