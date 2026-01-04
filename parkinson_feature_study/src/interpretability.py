"""
Interpretability Module

Converts model predictions into interpretable risk scores and generates
human-readable output reports.

IMPORTANT: These are NOT medical diagnoses. Outputs represent 
"Parkinson's likelihood" or "speech motor impairment risk" based on
acoustic features.

Author: Research Team
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RiskAssessment:
    """Individual risk assessment result."""
    sample_id: int
    probability: float
    risk_category: str
    confidence: str


class RiskScoreGenerator:
    """
    Converts prediction probabilities into interpretable risk categories.
    
    Risk Categories:
    - Low Risk: p < 0.33
    - Medium Risk: 0.33 ≤ p < 0.67
    - High Risk: p ≥ 0.67
    
    Attributes:
        config: Configuration dictionary
        thresholds: Risk category thresholds
        labels: Risk category labels
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the risk score generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        interp_config = config.get("interpretability", {})
        threshold_config = interp_config.get("risk_thresholds", {})
        
        self.thresholds = {
            "low": threshold_config.get("low", 0.33),
            "medium": threshold_config.get("medium", 0.67)
        }
        
        label_config = interp_config.get("output_labels", {})
        self.labels = {
            "low": label_config.get("low", "Low Risk"),
            "medium": label_config.get("medium", "Medium Risk"),
            "high": label_config.get("high", "High Risk")
        }
        
        self.disclaimer = interp_config.get(
            "disclaimer",
            "These scores represent Parkinson's disease likelihood based on speech "
            "features and should NOT be interpreted as medical diagnosis."
        )
        
        logger.info("RiskScoreGenerator initialized")
    
    def probability_to_category(self, probability: float) -> str:
        """
        Convert a probability to a risk category.
        
        Args:
            probability: Predicted probability (0-1)
            
        Returns:
            Risk category label
        """
        if probability < self.thresholds["low"]:
            return self.labels["low"]
        elif probability < self.thresholds["medium"]:
            return self.labels["medium"]
        else:
            return self.labels["high"]
    
    def get_confidence_level(self, probability: float) -> str:
        """
        Get confidence level based on probability distance from decision boundary.
        
        Args:
            probability: Predicted probability
            
        Returns:
            Confidence level description
        """
        # Distance from 0.5 boundary
        distance = abs(probability - 0.5)
        
        if distance < 0.1:
            return "Low confidence (close to decision boundary)"
        elif distance < 0.25:
            return "Moderate confidence"
        else:
            return "High confidence"
    
    def generate_risk_scores(
        self,
        probabilities: np.ndarray,
        sample_ids: Optional[List[int]] = None
    ) -> List[RiskAssessment]:
        """
        Generate risk assessments for a batch of predictions.
        
        Args:
            probabilities: Array of predicted probabilities
            sample_ids: Optional list of sample identifiers
            
        Returns:
            List of RiskAssessment objects
        """
        if sample_ids is None:
            sample_ids = list(range(len(probabilities)))
        
        assessments = []
        for idx, prob in zip(sample_ids, probabilities):
            assessment = RiskAssessment(
                sample_id=idx,
                probability=float(prob),
                risk_category=self.probability_to_category(prob),
                confidence=self.get_confidence_level(prob)
            )
            assessments.append(assessment)
        
        return assessments
    
    def summarize_risk_distribution(
        self,
        assessments: List[RiskAssessment]
    ) -> Dict[str, Any]:
        """
        Summarize the distribution of risk categories.
        
        Args:
            assessments: List of RiskAssessment objects
            
        Returns:
            Summary statistics dictionary
        """
        categories = [a.risk_category for a in assessments]
        probabilities = [a.probability for a in assessments]
        
        summary = {
            "total_samples": len(assessments),
            "mean_probability": np.mean(probabilities),
            "std_probability": np.std(probabilities),
            "category_counts": {},
            "category_percentages": {}
        }
        
        for label in self.labels.values():
            count = categories.count(label)
            summary["category_counts"][label] = count
            summary["category_percentages"][label] = (
                count / len(assessments) * 100 if assessments else 0
            )
        
        return summary
    
    def generate_report(
        self,
        assessments: List[RiskAssessment],
        model_info: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate a human-readable risk assessment report.
        
        Args:
            assessments: List of RiskAssessment objects
            model_info: Optional model/experiment information
            
        Returns:
            Formatted report string
        """
        summary = self.summarize_risk_distribution(assessments)
        
        report_lines = [
            "=" * 70,
            "PARKINSON'S DISEASE SPEECH-BASED RISK ASSESSMENT REPORT",
            "=" * 70,
            "",
            "⚠️  DISCLAIMER:",
            self.disclaimer,
            "",
            "-" * 70,
        ]
        
        if model_info:
            report_lines.extend([
                "Model Information:",
                f"  Feature Selection Method: {model_info.get('feature_method', 'N/A')}",
                f"  Classification Model: {model_info.get('model_name', 'N/A')}",
                f"  Number of Features Used: {model_info.get('n_features', 'N/A')}",
                "",
                "-" * 70,
            ])
        
        report_lines.extend([
            "SUMMARY STATISTICS",
            f"  Total Samples Assessed: {summary['total_samples']}",
            f"  Mean Parkinson's Likelihood: {summary['mean_probability']:.2%}",
            f"  Standard Deviation: {summary['std_probability']:.2%}",
            "",
            "RISK CATEGORY DISTRIBUTION:",
        ])
        
        for category, count in summary["category_counts"].items():
            pct = summary["category_percentages"][category]
            report_lines.append(f"  {category}: {count} samples ({pct:.1f}%)")
        
        report_lines.extend([
            "",
            "-" * 70,
            "INDIVIDUAL ASSESSMENTS (Sample)",
            ""
        ])
        
        # Show first 10 samples
        for assessment in assessments[:10]:
            report_lines.append(
                f"  Sample {assessment.sample_id}: "
                f"Probability = {assessment.probability:.2%}, "
                f"Category = {assessment.risk_category}, "
                f"{assessment.confidence}"
            )
        
        if len(assessments) > 10:
            report_lines.append(f"  ... and {len(assessments) - 10} more samples")
        
        report_lines.extend([
            "",
            "=" * 70,
            "END OF REPORT",
            "=" * 70
        ])
        
        return "\n".join(report_lines)
    
    def to_dataframe(
        self,
        assessments: List[RiskAssessment]
    ) -> pd.DataFrame:
        """
        Convert assessments to a DataFrame.
        
        Args:
            assessments: List of RiskAssessment objects
            
        Returns:
            DataFrame with assessment results
        """
        records = [
            {
                "sample_id": a.sample_id,
                "probability": a.probability,
                "risk_category": a.risk_category,
                "confidence": a.confidence
            }
            for a in assessments
        ]
        return pd.DataFrame(records)
    
    def save_assessments(
        self,
        assessments: List[RiskAssessment],
        output_path: str,
        model_info: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Save assessments to files (CSV and text report).
        
        Args:
            assessments: List of RiskAssessment objects
            output_path: Directory to save files
            model_info: Optional model information for report
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        df = self.to_dataframe(assessments)
        csv_path = output_dir / "risk_assessments.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved assessments to {csv_path}")
        
        # Save text report
        report = self.generate_report(assessments, model_info)
        report_path = output_dir / "risk_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        logger.info(f"Saved report to {report_path}")


def generate_risk_scores(
    probabilities: np.ndarray,
    config: Dict[str, Any],
    output_dir: Optional[str] = None,
    model_info: Optional[Dict[str, str]] = None
) -> Tuple[List[RiskAssessment], str]:
    """
    Convenience function to generate risk scores and report.
    
    Args:
        probabilities: Predicted probabilities
        config: Configuration dictionary
        output_dir: Optional directory to save results
        model_info: Optional model information
        
    Returns:
        Tuple of (assessments list, report string)
    """
    generator = RiskScoreGenerator(config)
    assessments = generator.generate_risk_scores(probabilities)
    report = generator.generate_report(assessments, model_info)
    
    if output_dir:
        generator.save_assessments(assessments, output_dir, model_info)
    
    return assessments, report
