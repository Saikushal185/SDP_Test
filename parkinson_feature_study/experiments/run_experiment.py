#!/usr/bin/env python
"""
Main Experiment Runner

Orchestrates the complete Parkinson's disease feature-centric comparative study.
Executes the 2x2 cross-paradigm testing framework and generates all outputs.

Usage:
    python run_experiment.py                    # Full experiment
    python run_experiment.py --dry-run          # Quick test run
    python run_experiment.py --config custom.yaml  # Custom config

Author: Research Team
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import load_config, setup_logging
from src.training import TrainingPipeline
from src.evaluation import evaluate_results
from src.interpretability import RiskScoreGenerator


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Parkinson's Disease Feature-Centric Comparative Study"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dataset (overrides config)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with reduced parameters for testing"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the experiment."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    log_level = config.get("general", {}).get("log_level", "INFO")
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("PARKINSON'S DISEASE FEATURE-CENTRIC COMPARATIVE STUDY")
    logger.info("=" * 70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Dry run: {args.dry_run}")
    
    # Set output directory
    output_dir = args.output_dir or config["output"]["results_dir"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metrics_dir = Path(config["output"]["metrics_dir"])
    figures_dir = Path(config["output"]["figures_dir"])
    features_dir = Path(config["output"]["features_dir"])
    
    for dir_path in [metrics_dir, figures_dir, features_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Determine data path
    data_path = args.data
    if data_path is None:
        # Try relative to project root
        project_root = Path(__file__).parent.parent
        data_path = project_root / config["data"]["raw_data_path"]
        
        if not data_path.exists():
            # Try one level up (in case running from experiments/)
            data_path = project_root.parent / "pd_speech_features.csv"
    
    logger.info(f"Data path: {data_path}")
    
    try:
        # Run training pipeline
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 1: TRAINING PIPELINE")
        logger.info("=" * 70)
        
        pipeline = TrainingPipeline(config)
        results = pipeline.run(data_path=str(data_path), dry_run=args.dry_run)
        
        # Save feature selections
        pipeline.save_feature_selections(str(features_dir))
        
        # Evaluate results
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 2: EVALUATION")
        logger.info("=" * 70)
        
        metrics_df, aggregated_df, stats_df = evaluate_results(
            results, config, str(metrics_dir)
        )
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 70)
        
        print("\nAggregated Results (Mean ± Std):")
        print("-" * 70)
        for _, row in aggregated_df.iterrows():
            print(f"\n{row['feature_method']} features + {row['model']}:")
            print(f"  Accuracy:  {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}")
            print(f"  Precision: {row['precision_mean']:.4f} ± {row['precision_std']:.4f}")
            print(f"  Recall:    {row['recall_mean']:.4f} ± {row['recall_std']:.4f}")
            print(f"  F1:        {row['f1_mean']:.4f} ± {row['f1_std']:.4f}")
            print(f"  ROC-AUC:   {row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f}")
        
        # Show significant statistical tests
        significant = stats_df[stats_df["significant"] == True]
        if len(significant) > 0:
            print("\n\nStatistically Significant Differences (p < 0.05):")
            print("-" * 70)
            for _, row in significant.iterrows():
                print(f"\n{row['comparison']}:")
                print(f"  Metric: {row['metric']}")
                print(f"  Difference: {row['difference']:.4f}")
                print(f"  p-value: {row['p_value']:.4f}")
        
        # Generate sample risk assessments
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 3: INTERPRETABILITY")
        logger.info("=" * 70)
        
        # Use best model's predictions for risk assessment demo
        if results.fold_results:
            sample_result = results.fold_results[0]
            risk_generator = RiskScoreGenerator(config)
            assessments = risk_generator.generate_risk_scores(sample_result.y_proba)
            
            report = risk_generator.generate_report(
                assessments,
                model_info={
                    "feature_method": sample_result.feature_method,
                    "model_name": sample_result.model_name,
                    "n_features": str(len(sample_result.selected_features))
                }
            )
            
            # Save report
            report_path = output_path / "sample_risk_report.txt"
            with open(report_path, "w") as f:
                f.write(report)
            logger.info(f"Saved sample risk report to {report_path}")
        
        # Final summary
        logger.info("\n" + "=" * 70)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 70)
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"  - Metrics: {metrics_dir}")
        logger.info(f"  - Figures: {figures_dir}")
        logger.info(f"  - Features: {features_dir}")
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        logger.error("Please ensure pd_speech_features.csv is in data/raw/ directory")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
