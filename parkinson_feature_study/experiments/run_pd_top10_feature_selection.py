#!/usr/bin/env python
"""Run reduced top-10 feature retraining for the local PD speech dataset."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.top10_feature_training import run_reduced_feature_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PD speech models with top-10 MI features.")
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=WORKSPACE_ROOT / "pd_speech_features.csv",
        help="Path to pd_speech_features.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT / "New Training with 10 features",
        help="Directory where experiment outputs will be written.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of ranked features to select.")
    parser.add_argument("--folds", type=int, default=10, help="Number of stratified CV folds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = run_reduced_feature_experiment(
        source_csv=args.source_csv,
        output_dir=args.output_dir,
        top_k=args.top_k,
        n_splits=args.folds,
    )
    print(f"Outputs saved to: {result.output_dir}")
    print(result.model_comparison.to_string(index=False))


if __name__ == "__main__":
    main()
