"""
Classical Feature Selection Module

Implements tree-based feature importance using Random Forest
for classical feature selection in the Parkinson's study.

Author: Research Team
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Configure logging
logger = logging.getLogger(__name__)


class ClassicalFeatureSelector:
    """
    Classical feature selector using Random Forest feature importance.
    
    This selector uses the mean decrease in impurity (MDI) from a
    Random Forest classifier to rank and select the most important features.
    
    Attributes:
        config: Configuration dictionary
        n_features: Number of top features to select
        rf_model: Trained Random Forest model
        feature_importances: Array of feature importance scores
        selected_indices: Indices of selected features
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the classical feature selector.
        
        Args:
            config: Configuration dictionary with feature selection settings
        """
        self.config = config
        self.n_features = config["feature_selection"]["n_features_to_select"]
        
        # Get Random Forest parameters
        rf_config = config["feature_selection"]["classical"]
        self.rf_params = {
            "n_estimators": rf_config.get("n_estimators", 100),
            "max_depth": rf_config.get("max_depth", 10),
            "min_samples_split": rf_config.get("min_samples_split", 5),
            "min_samples_leaf": rf_config.get("min_samples_leaf", 2),
            "random_state": config["general"]["random_seed"],
            "n_jobs": config["general"].get("n_jobs", -1)
        }
        
        self.rf_model: Optional[RandomForestClassifier] = None
        self.feature_importances: Optional[np.ndarray] = None
        self.selected_indices: Optional[np.ndarray] = None
        
        logger.info(f"ClassicalFeatureSelector initialized (selecting top {self.n_features} features)")
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> "ClassicalFeatureSelector":
        """
        Fit the feature selector on training data.
        
        IMPORTANT: This should only be called on training data within a CV fold
        to prevent data leakage.
        
        Args:
            X_train: Training feature matrix (n_samples, n_features)
            y_train: Training target array (n_samples,)
            feature_names: Optional list of feature names
            
        Returns:
            self
        """
        logger.info(f"Fitting Random Forest on {X_train.shape[0]} samples, {X_train.shape[1]} features")
        
        # Train Random Forest
        self.rf_model = RandomForestClassifier(**self.rf_params)
        self.rf_model.fit(X_train, y_train)
        
        # Get feature importances
        self.feature_importances = self.rf_model.feature_importances_
        
        # Select top-k features
        n_select = min(self.n_features, X_train.shape[1])
        self.selected_indices = np.argsort(self.feature_importances)[::-1][:n_select]
        
        if feature_names is not None:
            selected_names = [feature_names[i] for i in self.selected_indices]
            logger.info(f"Top 5 features: {selected_names[:5]}")
        
        logger.info(f"Selected {len(self.selected_indices)} features")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data by selecting only the chosen features.
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            Transformed matrix with only selected features
        """
        if self.selected_indices is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        
        return X[:, self.selected_indices]
    
    def fit_transform(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Fit the selector and transform the training data.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target array
            feature_names: Optional list of feature names
            
        Returns:
            Transformed training matrix
        """
        self.fit(X_train, y_train, feature_names)
        return self.transform(X_train)
    
    def get_selected_indices(self) -> np.ndarray:
        """
        Get the indices of selected features.
        
        Returns:
            Array of selected feature indices
        """
        if self.selected_indices is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        return self.selected_indices.copy()
    
    def get_feature_importances(self) -> np.ndarray:
        """
        Get the importance scores for all features.
        
        Returns:
            Array of feature importance scores
        """
        if self.feature_importances is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        return self.feature_importances.copy()
    
    def get_selected_feature_names(
        self,
        feature_names: List[str]
    ) -> List[str]:
        """
        Get the names of selected features.
        
        Args:
            feature_names: List of all feature names
            
        Returns:
            List of selected feature names
        """
        if self.selected_indices is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        return [feature_names[i] for i in self.selected_indices]
    
    def get_feature_ranking(
        self,
        feature_names: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Get ranked list of features with their importance scores.
        
        Args:
            feature_names: List of all feature names
            
        Returns:
            List of (feature_name, importance) tuples, sorted by importance
        """
        if self.feature_importances is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        
        ranking = list(zip(feature_names, self.feature_importances))
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking
