#!/usr/bin/env python
"""
Classical-only experiment runner for Parkinson's disease study
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import load_config, setup_logging, DataPreprocessor
from src.feature_selection.classical import ClassicalFeatureSelector
from src.models.classical import XGBoostModel, MLPModel

def run_classical_experiment():
    """Run experiment with classical models only"""
    
    # Load configuration
    config = load_config("config.yaml")
    
    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("PARKINSON'S DISEASE CLASSICAL MODELS STUDY")
    logger.info("=" * 70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load and preprocess data
    preprocessor = DataPreprocessor(config)
    data_path = "data/raw/pd_speech_features_cleaned.csv"
    preprocessor.load_data(data_path)
    X, y, feature_names = preprocessor.prepare_features()
    
    logger.info(f"Data shape: {X.shape}, Classes: {np.unique(y)}")
    
    # Initialize models and feature selector
    feature_selector = ClassicalFeatureSelector(config)
    xgb_model = XGBoostModel(config)
    mlp_model = MLPModel(config)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        logger.info(f"\n--- Fold {fold}/5 ---")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Feature selection
        feature_selector.fit(X_train, y_train)
        selected_features = feature_selector.get_selected_feature_names(feature_names)
        X_train_selected = feature_selector.transform(X_train)
        X_test_selected = feature_selector.transform(X_test)
        
        logger.info(f"Selected {len(selected_features)} features")
        
        # Train and evaluate XGBoost
        xgb_model.fit(X_train_selected, y_train)
        xgb_pred = xgb_model.predict(X_test_selected)
        xgb_proba = xgb_model.predict_proba(X_test_selected)[:, 1]
        
        xgb_metrics = {
            'model': 'XGBoost',
            'feature_method': 'classical',
            'fold': fold,
            'accuracy': accuracy_score(y_test, xgb_pred),
            'precision': precision_score(y_test, xgb_pred),
            'recall': recall_score(y_test, xgb_pred),
            'f1': f1_score(y_test, xgb_pred),
            'roc_auc': roc_auc_score(y_test, xgb_proba)
        }
        results.append(xgb_metrics)
        
        # Train and evaluate MLP
        mlp_model.fit(X_train_selected, y_train)
        mlp_pred = mlp_model.predict(X_test_selected)
        mlp_proba = mlp_model.predict_proba(X_test_selected)[:, 1]
        
        mlp_metrics = {
            'model': 'MLP',
            'feature_method': 'classical',
            'fold': fold,
            'accuracy': accuracy_score(y_test, mlp_pred),
            'precision': precision_score(y_test, mlp_pred),
            'recall': recall_score(y_test, mlp_pred),
            'f1': f1_score(y_test, mlp_pred),
            'roc_auc': roc_auc_score(y_test, mlp_proba)
        }
        results.append(mlp_metrics)
        
        logger.info(f"XGBoost - Accuracy: {xgb_metrics['accuracy']:.4f}, F1: {xgb_metrics['f1']:.4f}")
        logger.info(f"MLP - Accuracy: {mlp_metrics['accuracy']:.4f}, F1: {mlp_metrics['f1']:.4f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate aggregated results
    aggregated = results_df.groupby(['model', 'feature_method']).agg({
        'accuracy': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'roc_auc': ['mean', 'std']
    }).round(4)
    
    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    
    print("\nDetailed Results:")
    print(results_df.to_string(index=False))
    
    print("\n\nAggregated Results (Mean ± Std):")
    print("-" * 70)
    for model in ['XGBoost', 'MLP']:
        model_results = results_df[results_df['model'] == model]
        print(f"\n{model}:")
        print(f"  Accuracy:  {model_results['accuracy'].mean():.4f} ± {model_results['accuracy'].std():.4f}")
        print(f"  Precision: {model_results['precision'].mean():.4f} ± {model_results['precision'].std():.4f}")
        print(f"  Recall:    {model_results['recall'].mean():.4f} ± {model_results['recall'].std():.4f}")
        print(f"  F1:        {model_results['f1'].mean():.4f} ± {model_results['f1'].std():.4f}")
        print(f"  ROC-AUC:   {model_results['roc_auc'].mean():.4f} ± {model_results['roc_auc'].std():.4f}")
    
    # Save results
    results_dir = Path("results/metrics")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(results_dir / "classical_results.csv", index=False)
    aggregated.to_csv(results_dir / "classical_aggregated.csv")
    
    logger.info(f"\nResults saved to {results_dir}")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    run_classical_experiment()