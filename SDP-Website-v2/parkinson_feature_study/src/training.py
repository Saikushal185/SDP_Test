"""
Training Pipeline Module

Implements the cross-validation training loop for the Parkinson's study.
Handles the 2x2 cross-paradigm testing framework:
- Classical features → Classical models
- Classical features → Quantum-inspired model
- Quantum-inspired features → Classical models
- Quantum-inspired features → Quantum-inspired model

CRITICAL: Feature selection occurs INSIDE each fold to prevent data leakage.

Author: Research Team
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .preprocessing import DataPreprocessor
from .feature_selection import ClassicalFeatureSelector, QIGAFeatureSelector
from .models.classical import XGBoostModel, MLPModel
from .models.quantum_inspired import QuantumNeuralNetwork

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Results from a single cross-validation fold."""
    fold_idx: int
    feature_method: str  # "classical" or "quantum_inspired"
    model_name: str
    train_indices: np.ndarray
    test_indices: np.ndarray
    selected_features: List[int]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    training_time: float = 0.0


@dataclass
class ExperimentResults:
    """Complete results from the experiment."""
    fold_results: List[FoldResult] = field(default_factory=list)
    feature_selections: Dict[str, Dict[int, List[int]]] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)


class TrainingPipeline:
    """
    Training pipeline implementing 2x2 cross-paradigm testing.
    
    This pipeline:
    1. Performs stratified k-fold cross-validation
    2. For each fold, applies BOTH feature selection methods
    3. Trains ALL models with EACH feature set
    4. Records all metrics for comparison
    
    Attributes:
        config: Configuration dictionary
        preprocessor: DataPreprocessor instance
        results: ExperimentResults instance
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the training pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.results = ExperimentResults(config=config)
        
        logger.info("TrainingPipeline initialized")
    
    def _create_feature_selectors(self) -> Dict[str, Any]:
        """Create feature selector instances."""
        return {
            "classical": ClassicalFeatureSelector(self.config),
            "quantum_inspired": QIGAFeatureSelector(self.config)
        }
    
    def _create_models(self) -> Dict[str, Any]:
        """Create model instances."""
        return {
            "XGBoost": XGBoostModel(self.config),
            "MLP": MLPModel(self.config),
            "QNN": QuantumNeuralNetwork(self.config)
        }
    
    def run(
        self,
        data_path: Optional[str] = None,
        dry_run: bool = False
    ) -> ExperimentResults:
        """
        Run the complete training pipeline.
        
        Args:
            data_path: Optional path to dataset. Uses config if not provided.
            dry_run: If True, use reduced parameters for testing
            
        Returns:
            ExperimentResults containing all fold results
        """
        import time
        
        logger.info("Starting training pipeline")
        
        # Load and prepare data
        self.preprocessor.load_data(data_path)
        X, y, feature_names = self.preprocessor.prepare_features()
        
        logger.info(f"Data shape: {X.shape}, Classes: {np.unique(y)}")
        
        # Create CV splits
        cv_splits = self.preprocessor.create_cv_splits(X, y)
        n_folds = len(cv_splits)
        
        # Adjust for dry run
        if dry_run:
            logger.warning("DRY RUN mode: Using reduced parameters")
            cv_splits = cv_splits[:2]  # Only 2 folds
            self.config["feature_selection"]["n_features_to_select"] = 10
            self.config["feature_selection"]["quantum_inspired"]["n_generations"] = 5
            self.config["models"]["qnn"]["n_epochs"] = 10
        
        # Initialize storage for feature selections
        self.results.feature_selections = {
            "classical": {},
            "quantum_inspired": {}
        }
        
        # Process each fold
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing Fold {fold_idx + 1}/{len(cv_splits)}")
            logger.info(f"{'='*60}")
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale features (fit only on training data!)
            X_train_scaled, X_test_scaled = self.preprocessor.scale_features(
                X_train, X_test, fit=True
            )
            
            # Process each feature selection method
            for fs_name, fs_selector in self._create_feature_selectors().items():
                logger.info(f"\n--- Feature Selection: {fs_name} ---")
                
                # Fit feature selector (on training data only!)
                start_time = time.time()
                fs_selector.fit(X_train_scaled, y_train, feature_names)
                fs_time = time.time() - start_time
                
                # Get selected features
                selected_indices = fs_selector.get_selected_indices()
                self.results.feature_selections[fs_name][fold_idx] = selected_indices.tolist()
                
                logger.info(f"Selected {len(selected_indices)} features in {fs_time:.2f}s")
                
                # Transform data with selected features
                X_train_selected = fs_selector.transform(X_train_scaled)
                X_test_selected = fs_selector.transform(X_test_scaled)
                
                # Train each model with these features
                for model_name, model in self._create_models().items():
                    logger.info(f"\n  Training {model_name} with {fs_name} features...")
                    
                    start_time = time.time()
                    model.fit(X_train_selected, y_train)
                    train_time = time.time() - start_time
                    
                    # Predict
                    y_pred = model.predict(X_test_selected)
                    y_proba = model.predict_proba(X_test_selected)[:, 1]
                    
                    # Store results
                    fold_result = FoldResult(
                        fold_idx=fold_idx,
                        feature_method=fs_name,
                        model_name=model_name,
                        train_indices=train_idx,
                        test_indices=test_idx,
                        selected_features=selected_indices.tolist(),
                        y_true=y_test,
                        y_pred=y_pred,
                        y_proba=y_proba,
                        training_time=train_time
                    )
                    self.results.fold_results.append(fold_result)
                    
                    # Quick accuracy check
                    accuracy = np.mean(y_pred == y_test)
                    logger.info(f"  {model_name}: Accuracy = {accuracy:.4f}, Time = {train_time:.2f}s")
        
        logger.info("\n" + "="*60)
        logger.info("Training pipeline complete!")
        logger.info(f"Total experiments: {len(self.results.fold_results)}")
        
        return self.results
    
    def save_feature_selections(self, output_dir: str) -> None:
        """
        Save selected features to CSV files.
        
        Args:
            output_dir: Directory to save feature files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for method_name, fold_selections in self.results.feature_selections.items():
            for fold_idx, selected_features in fold_selections.items():
                filename = f"{method_name}_features_fold_{fold_idx}.csv"
                filepath = output_path / filename
                
                # Get feature names
                feature_names = self.preprocessor.get_feature_names()
                selected_names = [feature_names[i] for i in selected_features]
                
                df = pd.DataFrame({
                    "feature_index": selected_features,
                    "feature_name": selected_names
                })
                df.to_csv(filepath, index=False)
                logger.info(f"Saved {filepath}")


def run_training(
    config: Dict[str, Any],
    data_path: Optional[str] = None,
    dry_run: bool = False
) -> ExperimentResults:
    """
    Convenience function to run the training pipeline.
    
    Args:
        config: Configuration dictionary
        data_path: Optional path to dataset
        dry_run: If True, use reduced parameters
        
    Returns:
        ExperimentResults
    """
    pipeline = TrainingPipeline(config)
    return pipeline.run(data_path, dry_run)
