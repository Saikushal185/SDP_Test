"""
Evaluation Module

Implements metrics computation, statistical testing, and visualization
for the Parkinson's feature-centric comparative study.

Author: Research Team
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# Configure logging
logger = logging.getLogger(__name__)

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("Matplotlib/Seaborn not available. Plotting disabled.")


@dataclass
class MetricsSummary:
    """Summary statistics for a metric across folds."""
    mean: float
    std: float
    values: List[float]
    ci_lower: float = 0.0
    ci_upper: float = 0.0


class Evaluator:
    """
    Evaluator for computing metrics and statistical tests.
    
    Attributes:
        config: Configuration dictionary
        results: ExperimentResults from training
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.metrics_df: Optional[pd.DataFrame] = None
        self.aggregated_df: Optional[pd.DataFrame] = None
        
        logger.info("Evaluator initialized")
    
    def compute_fold_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute classification metrics for a single fold.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities for positive class
            
        Returns:
            Dictionary of metric names to values
        """
        metrics = {}
        
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = 0.5  # Default for single-class case
        
        return metrics
    
    def compute_all_metrics(self, results: Any) -> pd.DataFrame:
        """
        Compute metrics for all fold results.
        
        Args:
            results: ExperimentResults from training
            
        Returns:
            DataFrame with metrics for each experiment
        """
        records = []
        
        for fold_result in results.fold_results:
            metrics = self.compute_fold_metrics(
                fold_result.y_true,
                fold_result.y_pred,
                fold_result.y_proba
            )
            
            record = {
                "fold": fold_result.fold_idx,
                "feature_method": fold_result.feature_method,
                "model": fold_result.model_name,
                "n_features": len(fold_result.selected_features),
                "training_time": fold_result.training_time,
                **metrics
            }
            records.append(record)
        
        self.metrics_df = pd.DataFrame(records)
        logger.info(f"Computed metrics for {len(records)} experiments")
        
        return self.metrics_df
    
    def aggregate_metrics(self) -> pd.DataFrame:
        """
        Aggregate metrics across folds (mean ± std).
        
        Returns:
            DataFrame with aggregated metrics
        """
        if self.metrics_df is None:
            raise ValueError("No metrics computed. Call compute_all_metrics first.")
        
        # Group by feature method and model
        grouped = self.metrics_df.groupby(["feature_method", "model"])
        
        # Compute mean and std for each metric
        metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        
        agg_records = []
        for (fs_method, model), group in grouped:
            record = {
                "feature_method": fs_method,
                "model": model,
                "n_folds": len(group)
            }
            
            for metric in metric_cols:
                values = group[metric].values
                record[f"{metric}_mean"] = np.mean(values)
                record[f"{metric}_std"] = np.std(values)
                
                # 95% confidence interval
                if len(values) > 1:
                    ci = stats.t.interval(
                        0.95, len(values)-1,
                        loc=np.mean(values),
                        scale=stats.sem(values)
                    )
                    record[f"{metric}_ci_lower"] = ci[0]
                    record[f"{metric}_ci_upper"] = ci[1]
            
            record["training_time_mean"] = group["training_time"].mean()
            agg_records.append(record)
        
        self.aggregated_df = pd.DataFrame(agg_records)
        logger.info("Aggregated metrics across folds")
        
        return self.aggregated_df
    
    def statistical_tests(self) -> pd.DataFrame:
        """
        Perform paired t-tests comparing methods.
        
        Returns:
            DataFrame with statistical test results
        """
        if self.metrics_df is None:
            raise ValueError("No metrics computed. Call compute_all_metrics first.")
        
        alpha = self.config["evaluation"]["statistical_tests"].get("alpha", 0.05)
        test_results = []
        
        metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        
        # Compare classical vs quantum-inspired features (same model)
        models = self.metrics_df["model"].unique()
        
        for model in models:
            model_data = self.metrics_df[self.metrics_df["model"] == model]
            
            classical = model_data[model_data["feature_method"] == "classical"]
            quantum = model_data[model_data["feature_method"] == "quantum_inspired"]
            
            if len(classical) != len(quantum):
                continue
            
            for metric in metric_cols:
                classical_vals = classical.sort_values("fold")[metric].values
                quantum_vals = quantum.sort_values("fold")[metric].values
                
                try:
                    t_stat, p_value = stats.ttest_rel(classical_vals, quantum_vals)
                except Exception:
                    t_stat, p_value = np.nan, np.nan
                
                test_results.append({
                    "comparison": f"Features: Classical vs Quantum ({model})",
                    "metric": metric,
                    "classical_mean": np.mean(classical_vals),
                    "quantum_mean": np.mean(quantum_vals),
                    "difference": np.mean(quantum_vals) - np.mean(classical_vals),
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < alpha if not np.isnan(p_value) else False
                })
        
        # Compare classical vs quantum models (same feature method)
        for fs_method in ["classical", "quantum_inspired"]:
            fs_data = self.metrics_df[self.metrics_df["feature_method"] == fs_method]
            
            # Compare XGBoost vs QNN
            xgb_data = fs_data[fs_data["model"] == "XGBoost"]
            qnn_data = fs_data[fs_data["model"] == "QNN"]
            
            if len(xgb_data) != len(qnn_data):
                continue
            
            for metric in metric_cols:
                xgb_vals = xgb_data.sort_values("fold")[metric].values
                qnn_vals = qnn_data.sort_values("fold")[metric].values
                
                try:
                    t_stat, p_value = stats.ttest_rel(xgb_vals, qnn_vals)
                except Exception:
                    t_stat, p_value = np.nan, np.nan
                
                test_results.append({
                    "comparison": f"Models: XGBoost vs QNN ({fs_method} features)",
                    "metric": metric,
                    "classical_mean": np.mean(xgb_vals),
                    "quantum_mean": np.mean(qnn_vals),
                    "difference": np.mean(qnn_vals) - np.mean(xgb_vals),
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < alpha if not np.isnan(p_value) else False
                })
        
        results_df = pd.DataFrame(test_results)
        logger.info(f"Performed {len(test_results)} statistical tests")
        
        return results_df
    
    def feature_stability_analysis(
        self,
        feature_selections: Dict[str, Dict[int, List[int]]]
    ) -> pd.DataFrame:
        """
        Analyze stability of selected features across folds.
        
        Args:
            feature_selections: Dict mapping method -> fold -> selected indices
            
        Returns:
            DataFrame with stability metrics
        """
        stability_results = []
        
        for method_name, fold_selections in feature_selections.items():
            folds = sorted(fold_selections.keys())
            
            if len(folds) < 2:
                continue
            
            # Compute pairwise Jaccard similarity
            jaccard_scores = []
            for i in range(len(folds)):
                for j in range(i + 1, len(folds)):
                    set_i = set(fold_selections[folds[i]])
                    set_j = set(fold_selections[folds[j]])
                    
                    intersection = len(set_i & set_j)
                    union = len(set_i | set_j)
                    
                    jaccard = intersection / union if union > 0 else 0
                    jaccard_scores.append(jaccard)
            
            # Count feature occurrences
            all_features = []
            for fold_features in fold_selections.values():
                all_features.extend(fold_features)
            
            unique_features, counts = np.unique(all_features, return_counts=True)
            n_folds = len(folds)
            
            stability_results.append({
                "method": method_name,
                "n_folds": n_folds,
                "jaccard_mean": np.mean(jaccard_scores),
                "jaccard_std": np.std(jaccard_scores),
                "unique_features_total": len(unique_features),
                "features_in_all_folds": np.sum(counts == n_folds),
                "features_in_majority": np.sum(counts >= n_folds / 2)
            })
        
        results_df = pd.DataFrame(stability_results)
        logger.info("Computed feature stability analysis")
        
        return results_df
    
    def plot_roc_curves(
        self,
        results: Any,
        output_path: str
    ) -> None:
        """
        Plot ROC curves for all model/feature combinations.
        
        Args:
            results: ExperimentResults from training
            output_path: Path to save the figure
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        colors = {
            "XGBoost": "blue",
            "MLP": "green",
            "QNN": "red"
        }
        linestyles = {
            "classical": "-",
            "quantum_inspired": "--"
        }
        
        for ax_idx, fs_method in enumerate(["classical", "quantum_inspired"]):
            ax = axes[ax_idx]
            
            for fold_result in results.fold_results:
                if fold_result.feature_method != fs_method:
                    continue
                
                fpr, tpr, _ = roc_curve(fold_result.y_true, fold_result.y_proba)
                auc = roc_auc_score(fold_result.y_true, fold_result.y_proba)
                
                ax.plot(
                    fpr, tpr,
                    color=colors.get(fold_result.model_name, "gray"),
                    linestyle=linestyles.get(fs_method, "-"),
                    alpha=0.3
                )
            
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curves ({fs_method.replace('_', ' ').title()} Features)")
            ax.legend(
                [plt.Line2D([0], [0], color=c, label=n) 
                 for n, c in colors.items()],
                colors.keys(),
                loc="lower right"
            )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved ROC curves to {output_path}")
    
    def plot_performance_comparison(
        self,
        output_path: str
    ) -> None:
        """
        Plot bar chart comparing model performance.
        
        Args:
            output_path: Path to save the figure
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available")
            return
        
        if self.aggregated_df is None:
            raise ValueError("No aggregated metrics. Call aggregate_metrics first.")
        
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        
        for ax, metric in zip(axes, metrics):
            data = self.aggregated_df.copy()
            
            # Create grouped bar plot
            x = np.arange(len(data["model"].unique()))
            width = 0.35
            
            classical_data = data[data["feature_method"] == "classical"]
            quantum_data = data[data["feature_method"] == "quantum_inspired"]
            
            bars1 = ax.bar(
                x - width/2,
                classical_data[f"{metric}_mean"],
                width,
                yerr=classical_data[f"{metric}_std"],
                label="Classical Features",
                capsize=3
            )
            bars2 = ax.bar(
                x + width/2,
                quantum_data[f"{metric}_mean"],
                width,
                yerr=quantum_data[f"{metric}_std"],
                label="Quantum Features",
                capsize=3
            )
            
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_xticks(x)
            ax.set_xticklabels(data["model"].unique())
            ax.legend()
            ax.set_ylim(0, 1.1)
        
        plt.suptitle("Model Performance Comparison", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved performance comparison to {output_path}")
    
    def save_results(self, output_dir: str) -> None:
        """
        Save all results to CSV files.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.metrics_df is not None:
            self.metrics_df.to_csv(
                output_path / "cross_validation_results.csv",
                index=False
            )
            logger.info(f"Saved CV results to {output_path}")
        
        if self.aggregated_df is not None:
            self.aggregated_df.to_csv(
                output_path / "aggregated_results.csv",
                index=False
            )
            logger.info(f"Saved aggregated results to {output_path}")


def evaluate_results(
    results: Any,
    config: Dict[str, Any],
    output_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to evaluate experiment results.
    
    Args:
        results: ExperimentResults from training
        config: Configuration dictionary
        output_dir: Directory to save results
        
    Returns:
        Tuple of (metrics_df, aggregated_df, statistical_tests_df)
    """
    evaluator = Evaluator(config)
    
    # Compute metrics
    metrics_df = evaluator.compute_all_metrics(results)
    aggregated_df = evaluator.aggregate_metrics()
    stats_df = evaluator.statistical_tests()
    stability_df = evaluator.feature_stability_analysis(results.feature_selections)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metrics_df.to_csv(output_path / "cross_validation_results.csv", index=False)
    aggregated_df.to_csv(output_path / "aggregated_results.csv", index=False)
    stats_df.to_csv(output_path / "statistical_tests.csv", index=False)
    stability_df.to_csv(output_path / "feature_stability.csv", index=False)
    
    # Generate plots
    figures_dir = Path(config["output"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator.plot_roc_curves(results, str(figures_dir / "roc_curves.png"))
    evaluator.plot_performance_comparison(str(figures_dir / "performance_comparison.png"))
    
    return metrics_df, aggregated_df, stats_df
