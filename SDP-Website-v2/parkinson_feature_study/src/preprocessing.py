"""
Preprocessing Module for Parkinson's Disease Feature Study

This module handles:
- Data loading and validation
- Missing value handling
- Feature standardization
- Stratified K-Fold cross-validation setup

Author: Research Team
"""

import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Configure logging
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles all data preprocessing tasks for the Parkinson's study.
    
    Attributes:
        config: Configuration dictionary with preprocessing settings
        scaler: StandardScaler instance for feature normalization
        feature_names: List of feature column names
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config: Configuration dictionary containing data settings
        """
        self.config = config
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self._data: Optional[pd.DataFrame] = None
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        
        logger.info("DataPreprocessor initialized")
    
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load the dataset from CSV file.
        
        Args:
            data_path: Optional path to data file. Uses config if not provided.
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If required columns are missing
        """
        if data_path is None:
            data_path = self.config["data"]["raw_data_path"]
        
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        logger.info(f"Loading data from {path}")
        self._data = pd.read_csv(path)
        
        # Validate required columns
        target_col = self.config["data"]["target_column"]
        if target_col not in self._data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        logger.info(f"Loaded {len(self._data)} samples with {len(self._data.columns)} columns")
        return self._data
    
    def prepare_features(
        self,
        data: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and target arrays from the dataset.
        
        Args:
            data: Optional DataFrame. Uses loaded data if not provided.
            
        Returns:
            Tuple of (X features, y target, feature_names)
        """
        if data is None:
            if self._data is None:
                raise ValueError("No data loaded. Call load_data() first.")
            data = self._data
        
        # Get columns to exclude
        exclude_cols = set(self.config["data"].get("exclude_columns", []))
        target_col = self.config["data"]["target_column"]
        exclude_cols.add(target_col)
        
        # Identify feature columns
        self.feature_names = [
            col for col in data.columns 
            if col not in exclude_cols
        ]
        
        logger.info(f"Using {len(self.feature_names)} features")
        
        # Extract features and target
        self._X = data[self.feature_names].values.astype(np.float64)
        self._y = data[target_col].values.astype(np.int32)
        
        # Handle missing values
        self._X = self._handle_missing_values(self._X)
        
        return self._X, self._y, self.feature_names
    
    def _handle_missing_values(self, X: np.ndarray) -> np.ndarray:
        """
        Handle missing values in the feature matrix.
        
        Args:
            X: Feature matrix
            
        Returns:
            Feature matrix with missing values handled
        """
        n_missing = np.isnan(X).sum()
        if n_missing > 0:
            logger.warning(f"Found {n_missing} missing values. Imputing with column means.")
            col_means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_means, inds[1])
        
        return X
    
    def create_cv_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create stratified k-fold cross-validation splits.
        
        Args:
            X: Feature matrix (uses stored if not provided)
            y: Target array (uses stored if not provided)
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        if X is None:
            X = self._X
        if y is None:
            y = self._y
            
        if X is None or y is None:
            raise ValueError("No data available. Call prepare_features() first.")
        
        cv_config = self.config["cross_validation"]
        n_folds = cv_config["n_folds"]
        shuffle = cv_config.get("shuffle", True)
        random_seed = self.config["general"]["random_seed"]
        
        skf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=shuffle,
            random_state=random_seed
        )
        
        splits = list(skf.split(X, y))
        logger.info(f"Created {len(splits)} stratified CV splits")
        
        return splits
    
    def scale_features(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        fit: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler.
        
        IMPORTANT: Scaler is fit only on training data to prevent data leakage.
        
        Args:
            X_train: Training features
            X_test: Test features
            fit: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            Tuple of (scaled_X_train, scaled_X_test)
        """
        if fit:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            X_train_scaled = self.scaler.transform(X_train)
        
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature names.
        
        Returns:
            List of feature column names
        """
        return self.feature_names.copy()
    
    def get_class_distribution(
        self,
        y: Optional[np.ndarray] = None
    ) -> Dict[int, int]:
        """
        Get the distribution of classes in the target.
        
        Args:
            y: Target array (uses stored if not provided)
            
        Returns:
            Dictionary mapping class labels to counts
        """
        if y is None:
            y = self._y
        
        if y is None:
            raise ValueError("No target data available.")
        
        unique, counts = np.unique(y, return_counts=True)
        distribution = dict(zip(unique.tolist(), counts.tolist()))
        
        logger.info(f"Class distribution: {distribution}")
        return distribution


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    import yaml
    
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {path}")
    return config


def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
