#!/usr/bin/env python
"""Classical-only wrapper for the multi-dataset Parkinson study."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.multi_dataset_pipeline import run_multi_dataset_experiment
from src.preprocessing import load_config, setup_logging


def run_classical_experiment() -> None:
    config = load_config("config.yaml")
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    logger.info("Running classical-only multi-dataset experiment")
    comparison_df = run_multi_dataset_experiment(config=config, include_quantum=False)
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    run_classical_experiment()
