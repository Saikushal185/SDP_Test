"""
Classical Models Module

Implements classical machine learning models for the Parkinson's study:
- XGBoost Classifier
- Multi-Layer Perceptron (MLP)

Author: Research Team
"""

import logging
from typing import Dict, Any, Optional

import numpy as np
from sklearn.neural_network import MLPClassifier as SklearnMLP
import xgboost as xgb

# Configure logging
logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    XGBoost classifier wrapper with sklearn-compatible API.
    
    Attributes:
        config: Configuration dictionary
        model: XGBoost classifier instance
        is_fitted: Whether the model has been trained
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the XGBoost model.
        
        Args:
            config: Configuration dictionary with model hyperparameters
        """
        self.config = config
        self.model: Optional[xgb.XGBClassifier] = None
        self.is_fitted: bool = False
        
        # Get model parameters from config
        xgb_config = config["models"]["xgboost"]
        self.params = {
            "n_estimators": xgb_config.get("n_estimators", 100),
            "max_depth": xgb_config.get("max_depth", 6),
            "learning_rate": xgb_config.get("learning_rate", 0.1),
            "subsample": xgb_config.get("subsample", 0.8),
            "colsample_bytree": xgb_config.get("colsample_bytree", 0.8),
            "min_child_weight": xgb_config.get("min_child_weight", 1),
            "gamma": xgb_config.get("gamma", 0),
            "reg_alpha": xgb_config.get("reg_alpha", 0),
            "reg_lambda": xgb_config.get("reg_lambda", 1),
            "objective": xgb_config.get("objective", "binary:logistic"),
            "eval_metric": xgb_config.get("eval_metric", "logloss"),
            "use_label_encoder": xgb_config.get("use_label_encoder", False),
            "random_state": config["general"]["random_seed"],
            "n_jobs": config["general"].get("n_jobs", -1),
            "verbosity": 0
        }
        
        logger.info("XGBoostModel initialized")
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "XGBoostModel":
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features for early stopping
            y_val: Optional validation targets
            
        Returns:
            self
        """
        logger.info(f"Training XGBoost on {X_train.shape[0]} samples, {X_train.shape[1]} features")
        
        self.model = xgb.XGBClassifier(**self.params)
        
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        self.is_fitted = True
        logger.info("XGBoost training complete")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(X)
    
    def get_name(self) -> str:
        """Get model name for reporting."""
        return "XGBoost"


class MLPModel:
    """
    Multi-Layer Perceptron wrapper with sklearn-compatible API.
    
    Attributes:
        config: Configuration dictionary
        model: MLP classifier instance
        is_fitted: Whether the model has been trained
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the MLP model.
        
        Args:
            config: Configuration dictionary with model hyperparameters
        """
        self.config = config
        self.model: Optional[SklearnMLP] = None
        self.is_fitted: bool = False
        
        # Get model parameters from config
        mlp_config = config["models"]["mlp"]
        hidden_sizes = mlp_config.get("hidden_layer_sizes", [64, 32])
        
        self.params = {
            "hidden_layer_sizes": tuple(hidden_sizes),
            "activation": mlp_config.get("activation", "relu"),
            "solver": mlp_config.get("solver", "adam"),
            "alpha": mlp_config.get("alpha", 0.0001),
            "batch_size": mlp_config.get("batch_size", 32),
            "learning_rate": mlp_config.get("learning_rate", "adaptive"),
            "learning_rate_init": mlp_config.get("learning_rate_init", 0.001),
            "max_iter": mlp_config.get("max_iter", 500),
            "early_stopping": mlp_config.get("early_stopping", True),
            "validation_fraction": mlp_config.get("validation_fraction", 0.1),
            "n_iter_no_change": mlp_config.get("n_iter_no_change", 20),
            "random_state": config["general"]["random_seed"],
            "verbose": False
        }
        
        logger.info(f"MLPModel initialized with layers {self.params['hidden_layer_sizes']}")
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "MLPModel":
        """
        Train the MLP model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Unused (MLP uses internal validation)
            y_val: Unused
            
        Returns:
            self
        """
        logger.info(f"Training MLP on {X_train.shape[0]} samples, {X_train.shape[1]} features")
        
        self.model = SklearnMLP(**self.params)
        self.model.fit(X_train, y_train)
        
        self.is_fitted = True
        logger.info(f"MLP training complete (iterations: {self.model.n_iter_})")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(X)
    
    def get_name(self) -> str:
        """Get model name for reporting."""
        return "MLP"


def get_classical_models(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Factory function to create all classical models.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping model names to model instances
    """
    return {
        "XGBoost": XGBoostModel(config),
        "MLP": MLPModel(config)
    }
