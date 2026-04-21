"""
Dataset catalog for the multi-dataset Parkinson study.

Defines the default local and Hugging Face sources used by the training
pipeline, along with stable dataset identifiers for artifact export.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Optional


def sanitize_dataset_id(raw_value: str) -> str:
    """Convert a dataset reference into a filesystem-safe identifier."""
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", raw_value).strip("_")
    return sanitized or "dataset"


@dataclass(frozen=True)
class DatasetSpec:
    """Configuration for a dataset source."""

    dataset_id: str
    display_name: str
    source_type: str
    source_ref: str
    target_column: Optional[str] = None
    fallback_source_ref: Optional[str] = None
    local_path: Optional[Path] = None


def build_default_dataset_specs() -> List[DatasetSpec]:
    """Return the initial dataset set for the multi-dataset study."""
    return [
        DatasetSpec(
            dataset_id="pd_speech_features_local",
            display_name="Local PD Speech Features",
            source_type="local_csv",
            source_ref="pd_speech_features.csv",
            target_column="class",
            local_path=Path("pd_speech_features.csv"),
        ),
        DatasetSpec(
            dataset_id=sanitize_dataset_id("kongkon123890/uci_parkinsons_voice"),
            display_name="UCI Parkinsons Voice",
            source_type="huggingface_tabular",
            source_ref="kongkon123890/uci_parkinsons_voice",
            target_column=None,
            fallback_source_ref="XANJEEV/Parkinson_Classification_Dataset",
        ),
        DatasetSpec(
            dataset_id=sanitize_dataset_id("birgermoell/Italian_Parkinsons_Voice_and_Speech"),
            display_name="Italian Parkinsons Voice and Speech",
            source_type="huggingface_audio",
            source_ref="birgermoell/Italian_Parkinsons_Voice_and_Speech",
            target_column=None,
        ),
        DatasetSpec(
            dataset_id=sanitize_dataset_id("Hahad14/Parkinsons_Disease_Speech"),
            display_name="Parkinsons Disease Speech",
            source_type="huggingface_audio",
            source_ref="Hahad14/Parkinsons_Disease_Speech",
            target_column=None,
        ),
    ]
