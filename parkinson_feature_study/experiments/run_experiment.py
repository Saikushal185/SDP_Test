#!/usr/bin/env python
"""Main experiment runner for the multi-dataset Parkinson study."""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.multi_dataset_pipeline import run_multi_dataset_experiment
from src.preprocessing import load_config, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the multi-dataset Parkinson training pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--classical-only",
        action="store_true",
        help="Skip QSVM and VQC training",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Artifact directory override",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.output_dir:
        config.setdefault("output", {})["artifact_root"] = args.output_dir

    log_level = config.get("general", {}).get("log_level", "INFO")
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("PARKINSON'S DISEASE MULTI-DATASET TRAINING STUDY")
    logger.info("=" * 70)
    logger.info("Start time: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Configuration: %s", args.config)
    logger.info("Quantum models enabled: %s", not args.classical_only)

    try:
        comparison_df = run_multi_dataset_experiment(
            config=config,
            include_quantum=not args.classical_only,
        )
        print(comparison_df.to_string(index=False))

        logger.info("=" * 70)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 70)
        logger.info("End time: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info(
            "Artifacts saved under: %s",
            config.get("output", {}).get("artifact_root", "artifacts"),
        )
    except Exception as exc:
        logger.error("Experiment failed: %s", exc)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
