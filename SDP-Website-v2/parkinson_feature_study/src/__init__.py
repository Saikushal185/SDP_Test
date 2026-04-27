# Package initialization for src module
"""
Parkinson's Disease Feature-Centric Comparative Framework

This package contains modules for:
- preprocessing: Data loading, scaling, and cross-validation setup
- training: Cross-validation training loop with feature selection
- evaluation: Metrics computation and statistical testing
- interpretability: Risk score generation
"""

from . import preprocessing
from . import training
from . import evaluation
from . import interpretability

__version__ = "1.0.0"
__author__ = "Research Team"
