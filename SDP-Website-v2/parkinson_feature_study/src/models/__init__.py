# Models module initialization
"""
Models Module

Contains:
- XGBoostModel: XGBoost classifier wrapper
- MLPModel: Multi-Layer Perceptron wrapper
- QuantumNeuralNetwork: PennyLane-based QNN simulator
"""

from .classical import XGBoostModel, MLPModel
from .quantum_inspired import QuantumNeuralNetwork

__all__ = ["XGBoostModel", "MLPModel", "QuantumNeuralNetwork"]
