# Feature selection module initialization
"""
Feature Selection Module

Contains:
- ClassicalFeatureSelector: Random Forest-based feature importance
- QIGAFeatureSelector: Quantum-Inspired Genetic Algorithm
"""

from .classical import ClassicalFeatureSelector
from .quantum_inspired import QIGAFeatureSelector

__all__ = ["ClassicalFeatureSelector", "QIGAFeatureSelector"]
