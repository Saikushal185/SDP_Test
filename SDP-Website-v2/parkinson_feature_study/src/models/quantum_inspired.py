"""
Quantum-Inspired Neural Network Module

Implements a Variational Quantum Circuit (VQC) classifier using PennyLane.
This is a simulated quantum neural network suitable for tabular data classification.

Note: This runs on a classical simulator, not actual quantum hardware.
The quantum-inspired approach may offer advantages for certain feature spaces.

Author: Research Team
"""

import logging
from typing import Dict, Any, Optional, Tuple

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Try to import PennyLane, fall back to classical if unavailable
try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    logger.warning("PennyLane not available. QNN will use classical fallback.")


class QuantumNeuralNetwork:
    """
    Quantum Neural Network classifier using PennyLane's variational circuits.
    
    Uses angle encoding for input features and a trainable variational circuit.
    Falls back to a simple neural network if PennyLane is unavailable.
    
    Attributes:
        config: Configuration dictionary
        n_qubits: Number of qubits in the circuit
        n_layers: Number of variational layers
        weights: Trained circuit weights
        is_fitted: Whether the model has been trained
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the Quantum Neural Network.
        
        Args:
            config: Configuration dictionary with QNN hyperparameters
        """
        self.config = config
        self.is_fitted: bool = False
        
        # QNN parameters
        qnn_config = config["models"]["qnn"]
        self.n_qubits = qnn_config.get("n_qubits", 4)
        self.n_layers = qnn_config.get("n_layers", 3)
        self.learning_rate = qnn_config.get("learning_rate", 0.01)
        self.n_epochs = qnn_config.get("n_epochs", 100)
        self.batch_size = qnn_config.get("batch_size", 16)
        self.random_seed = config["general"]["random_seed"]
        
        np.random.seed(self.random_seed)
        
        self.weights: Optional[np.ndarray] = None
        self.dev = None
        self.circuit = None
        self._n_features: int = 0
        
        logger.info(f"QuantumNeuralNetwork initialized (qubits={self.n_qubits}, layers={self.n_layers})")
    
    def _create_circuit(self, n_features: int) -> None:
        """
        Create the variational quantum circuit.
        
        Args:
            n_features: Number of input features
        """
        self._n_features = n_features
        
        # Determine number of qubits (at least as many as features, up to limit)
        self.n_qubits = min(max(4, n_features), 8)  # Limit for simulation speed
        
        if not PENNYLANE_AVAILABLE:
            logger.warning("Using classical fallback for QNN")
            return
        
        # Create quantum device (simulator)
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        @qml.qnode(self.dev, interface="autograd")
        def quantum_circuit(inputs: np.ndarray, weights: np.ndarray) -> float:
            """
            Variational quantum circuit for classification.
            
            Args:
                inputs: Input features (angle encoded)
                weights: Trainable circuit weights
                
            Returns:
                Expectation value of PauliZ on first qubit
            """
            # Angle encoding of input features
            for i in range(self.n_qubits):
                feature_idx = i % len(inputs)
                qml.RX(inputs[feature_idx], wires=i)
            
            # Variational layers
            for layer in range(self.n_layers):
                # Rotation gates with trainable parameters
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                
                # Entangling gates (CNOT ladder)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                
                # Ring entanglement
                if self.n_qubits > 2:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
            
            # Measurement
            return qml.expval(qml.PauliZ(0))
        
        self.circuit = quantum_circuit
    
    def _classical_forward(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Classical fallback forward pass when PennyLane unavailable.
        
        Implements a simple neural network that mimics quantum behavior.
        
        Args:
            X: Input features
            weights: Network weights (reshaped for classical use)
            
        Returns:
            Output probabilities
        """
        # Simple 2-layer network as fallback
        hidden_size = self.n_qubits * 2
        
        # Reshape weights for classical network
        total_weights = weights.flatten()
        n_w1 = X.shape[1] * hidden_size
        n_w2 = hidden_size
        
        # Pad if necessary
        if len(total_weights) < n_w1 + n_w2:
            total_weights = np.tile(total_weights, (n_w1 + n_w2) // len(total_weights) + 1)
        
        W1 = total_weights[:n_w1].reshape(X.shape[1], hidden_size)
        W2 = total_weights[n_w1:n_w1 + n_w2].reshape(hidden_size, 1)
        
        # Forward pass
        hidden = np.tanh(X @ W1)
        output = hidden @ W2
        
        # Convert to probabilities
        probs = 1 / (1 + np.exp(-output))
        return probs.flatten()
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "QuantumNeuralNetwork":
        """
        Train the Quantum Neural Network.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
            
        Returns:
            self
        """
        n_samples, n_features = X_train.shape
        logger.info(f"Training QNN on {n_samples} samples, {n_features} features")
        
        # Create circuit
        self._create_circuit(n_features)
        
        # Initialize weights
        weight_shape = (self.n_layers, self.n_qubits, 2)
        self.weights = np.random.uniform(-np.pi, np.pi, weight_shape)
        
        if not PENNYLANE_AVAILABLE:
            # Use classical fallback
            self._fit_classical(X_train, y_train)
        else:
            # Use quantum circuit
            self._fit_quantum(X_train, y_train)
        
        self.is_fitted = True
        logger.info("QNN training complete")
        return self
    
    def _fit_classical(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit using classical fallback.
        """
        # Simple gradient descent
        for epoch in range(self.n_epochs):
            # Forward pass
            probs = self._classical_forward(X, self.weights)
            
            # Binary cross-entropy loss
            eps = 1e-7
            loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
            
            # Simple gradient estimation (finite difference)
            grad = np.zeros_like(self.weights)
            delta = 0.01
            
            for i in range(min(10, self.weights.size)):  # Limit for speed
                idx = np.unravel_index(i, self.weights.shape)
                self.weights[idx] += delta
                loss_plus = -np.mean(y * np.log(self._classical_forward(X, self.weights) + eps) + 
                                    (1 - y) * np.log(1 - self._classical_forward(X, self.weights) + eps))
                self.weights[idx] -= 2 * delta
                loss_minus = -np.mean(y * np.log(self._classical_forward(X, self.weights) + eps) + 
                                     (1 - y) * np.log(1 - self._classical_forward(X, self.weights) + eps))
                self.weights[idx] += delta
                grad[idx] = (loss_plus - loss_minus) / (2 * delta)
            
            # Update weights
            self.weights -= self.learning_rate * grad
            
            if epoch % 20 == 0:
                logger.debug(f"Epoch {epoch}: loss = {loss:.4f}")
    
    def _fit_quantum(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit using PennyLane quantum circuit with parameter-shift gradient.
        """
        from autograd import numpy as anp
        
        # Use PennyLane's numpy for autodiff
        weights = pnp.array(self.weights, requires_grad=True)
        opt = qml.GradientDescentOptimizer(stepsize=self.learning_rate)
        
        # Training loop with batching
        n_samples = len(X)
        
        for epoch in range(self.n_epochs):
            # Shuffle data
            perm = np.random.permutation(n_samples)
            
            for start in range(0, n_samples, self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                
                # Create batch cost function using autograd numpy
                def batch_cost(w):
                    preds = []
                    for i in range(len(X_batch)):
                        inputs = anp.tanh(X_batch[i]) * anp.pi
                        preds.append(self.circuit(inputs, w))
                    preds = anp.array(preds)
                    probs = (preds + 1) / 2
                    eps = 1e-7
                    # Use autograd numpy for log to support autodiff
                    return -anp.mean(y_batch * anp.log(probs + eps) + 
                                   (1 - y_batch) * anp.log(1 - probs + eps))
                
                weights = opt.step(batch_cost, weights)
            
            if epoch % 20 == 0:
                # Calculate loss for logging
                preds = []
                for i in range(len(X)):
                    inputs = np.tanh(X[i]) * np.pi
                    preds.append(float(self.circuit(inputs, weights)))
                preds = np.array(preds)
                probs = (preds + 1) / 2
                eps = 1e-7
                loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
                logger.debug(f"Epoch {epoch}: loss = {loss:.4f}")
        
        self.weights = np.array(weights)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted class labels
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(np.int32)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            Class probabilities (n_samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if not PENNYLANE_AVAILABLE:
            # Classical fallback
            probs_1 = self._classical_forward(X, self.weights)
        else:
            # Quantum circuit
            probs_1 = []
            for i in range(len(X)):
                inputs = np.tanh(X[i]) * np.pi
                pred = self.circuit(inputs, self.weights)
                prob = (pred + 1) / 2  # Map from [-1, 1] to [0, 1]
                probs_1.append(prob)
            probs_1 = np.array(probs_1)
        
        # Ensure valid probabilities
        probs_1 = np.clip(probs_1, 0, 1)
        probs_0 = 1 - probs_1
        
        return np.column_stack([probs_0, probs_1])
    
    def get_name(self) -> str:
        """Get model name for reporting."""
        return "QNN"


def get_quantum_model(config: Dict[str, Any]) -> QuantumNeuralNetwork:
    """
    Factory function to create quantum-inspired model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        QuantumNeuralNetwork instance
    """
    return QuantumNeuralNetwork(config)
