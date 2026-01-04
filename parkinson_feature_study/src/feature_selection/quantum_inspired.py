"""
Quantum-Inspired Feature Selection Module

Implements a Quantum-Inspired Genetic Algorithm (QIGA) for feature selection.
This algorithm uses quantum-inspired probability amplitude representation
and rotation gate updates for optimization.

Note: This is a classical simulation inspired by quantum computing concepts,
NOT actual quantum computing.

References:
- Han & Kim (2002): Genetic Quantum Algorithm and its Application 
  to Combinatorial Optimization Problem

Author: Research Team
"""

import logging
from typing import Dict, Any, List, Tuple, Optional, Callable

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Configure logging
logger = logging.getLogger(__name__)


class QIGAFeatureSelector:
    """
    Quantum-Inspired Genetic Algorithm (QIGA) for feature selection.
    
    Uses quantum-inspired probability amplitudes to represent the population
    and rotation gates for updating based on fitness. Each individual in the
    population represents a binary mask for feature selection.
    
    Attributes:
        config: Configuration dictionary
        n_features_to_select: Target number of features
        population_size: Number of individuals in population
        n_generations: Number of generations to evolve
        best_solution: Best feature mask found
        best_fitness: Fitness of best solution
        selected_indices: Indices of selected features
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the QIGA feature selector.
        
        Args:
            config: Configuration dictionary with QIGA settings
        """
        self.config = config
        self.n_features_to_select = config["feature_selection"]["n_features_to_select"]
        
        # QIGA parameters
        qiga_config = config["feature_selection"]["quantum_inspired"]
        self.population_size = qiga_config.get("population_size", 20)
        self.n_generations = qiga_config.get("n_generations", 50)
        self.mutation_rate = qiga_config.get("mutation_rate", 0.1)
        self.crossover_rate = qiga_config.get("crossover_rate", 0.8)
        self.theta_min = qiga_config.get("theta_min", 0.01)
        self.theta_max = qiga_config.get("theta_max", 0.1)
        self.elitism_count = qiga_config.get("elitism_count", 2)
        self.random_seed = config["general"]["random_seed"]
        
        # State variables
        self.best_solution: Optional[np.ndarray] = None
        self.best_fitness: float = -np.inf
        self.selected_indices: Optional[np.ndarray] = None
        self.fitness_history: List[float] = []
        
        # Random state
        self._rng = np.random.RandomState(self.random_seed)
        
        logger.info(f"QIGAFeatureSelector initialized (population={self.population_size}, "
                   f"generations={self.n_generations})")
    
    def _initialize_quantum_population(self, n_features: int) -> np.ndarray:
        """
        Initialize quantum population with probability amplitudes.
        
        Each individual is represented by probability amplitudes (alpha, beta)
        where |alpha|^2 + |beta|^2 = 1. We store only the angle theta such that
        alpha = cos(theta), beta = sin(theta).
        
        Args:
            n_features: Number of features in the dataset
            
        Returns:
            Quantum population (population_size, n_features) of angles
        """
        # Initialize with uniform superposition (theta = pi/4)
        initial_angle = np.pi / 4
        quantum_pop = np.full((self.population_size, n_features), initial_angle)
        
        # Add small random perturbation
        noise = self._rng.uniform(-0.1, 0.1, quantum_pop.shape)
        quantum_pop += noise
        
        return quantum_pop
    
    def _observe_population(self, quantum_pop: np.ndarray) -> np.ndarray:
        """
        Observe (collapse) quantum population to binary solutions.
        
        Args:
            quantum_pop: Quantum population of angles
            
        Returns:
            Binary population (population_size, n_features)
        """
        # Probability of selecting feature = sin^2(theta)
        probs = np.sin(quantum_pop) ** 2
        random_vals = self._rng.random(quantum_pop.shape)
        binary_pop = (random_vals < probs).astype(np.int32)
        
        return binary_pop
    
    def _evaluate_fitness(
        self,
        binary_pop: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate fitness of each individual using cross-validation accuracy.
        
        Args:
            binary_pop: Binary population (feature masks)
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Fitness scores for each individual
        """
        fitness_scores = np.zeros(self.population_size)
        
        for i in range(self.population_size):
            mask = binary_pop[i]
            n_selected = np.sum(mask)
            
            if n_selected == 0:
                # Penalty for selecting no features
                fitness_scores[i] = 0.0
                continue
            
            # Select features
            X_selected = X_train[:, mask == 1]
            
            # Use a simple classifier for fitness evaluation
            clf = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=self.random_seed,
                n_jobs=-1
            )
            
            # 3-fold CV for faster evaluation during optimization
            try:
                scores = cross_val_score(clf, X_selected, y_train, cv=3, scoring='accuracy')
                accuracy = scores.mean()
            except Exception:
                accuracy = 0.0
            
            # Penalize solutions that deviate significantly from target feature count
            n_target = self.n_features_to_select
            feature_penalty = 0.01 * abs(n_selected - n_target) / n_target
            
            fitness_scores[i] = accuracy - feature_penalty
        
        return fitness_scores
    
    def _update_quantum_population(
        self,
        quantum_pop: np.ndarray,
        binary_pop: np.ndarray,
        fitness_scores: np.ndarray,
        best_solution: np.ndarray
    ) -> np.ndarray:
        """
        Update quantum population using rotation gates.
        
        Uses the difference between current solution and best solution
        to determine rotation direction and magnitude.
        
        Args:
            quantum_pop: Current quantum population
            binary_pop: Current binary solutions
            fitness_scores: Fitness of current solutions
            best_solution: Best solution found so far
            
        Returns:
            Updated quantum population
        """
        n_features = quantum_pop.shape[1]
        
        for i in range(self.population_size):
            for j in range(n_features):
                # Determine rotation angle based on comparison with best
                xi = binary_pop[i, j]
                bi = best_solution[j]
                theta = quantum_pop[i, j]
                
                # Rotation lookup table (simplified)
                if xi == 0 and bi == 1:
                    # Should move towards selecting this feature
                    delta_theta = self._rng.uniform(self.theta_min, self.theta_max)
                elif xi == 1 and bi == 0:
                    # Should move towards not selecting this feature
                    delta_theta = -self._rng.uniform(self.theta_min, self.theta_max)
                else:
                    # Keep exploring
                    delta_theta = self._rng.uniform(-self.theta_min, self.theta_min)
                
                # Apply rotation
                quantum_pop[i, j] = np.clip(
                    theta + delta_theta,
                    0.01,  # Avoid exactly 0 or pi/2
                    np.pi/2 - 0.01
                )
        
        return quantum_pop
    
    def _apply_mutation(self, quantum_pop: np.ndarray) -> np.ndarray:
        """
        Apply quantum mutation (random rotation).
        
        Args:
            quantum_pop: Quantum population
            
        Returns:
            Mutated quantum population
        """
        mutation_mask = self._rng.random(quantum_pop.shape) < self.mutation_rate
        random_angles = self._rng.uniform(0.01, np.pi/2 - 0.01, quantum_pop.shape)
        
        quantum_pop = np.where(mutation_mask, random_angles, quantum_pop)
        return quantum_pop
    
    def _enforce_feature_count(
        self,
        binary_solution: np.ndarray,
        n_target: int
    ) -> np.ndarray:
        """
        Adjust solution to have approximately target number of features.
        
        Args:
            binary_solution: Binary feature mask
            n_target: Target number of features
            
        Returns:
            Adjusted binary solution
        """
        n_selected = np.sum(binary_solution)
        solution = binary_solution.copy()
        
        if n_selected > n_target:
            # Remove excess features randomly
            selected_idx = np.where(solution == 1)[0]
            n_remove = n_selected - n_target
            remove_idx = self._rng.choice(selected_idx, size=int(n_remove), replace=False)
            solution[remove_idx] = 0
        elif n_selected < n_target:
            # Add features randomly
            unselected_idx = np.where(solution == 0)[0]
            n_add = min(n_target - n_selected, len(unselected_idx))
            add_idx = self._rng.choice(unselected_idx, size=int(n_add), replace=False)
            solution[add_idx] = 1
        
        return solution
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> "QIGAFeatureSelector":
        """
        Fit the QIGA feature selector on training data.
        
        IMPORTANT: This should only be called on training data within a CV fold
        to prevent data leakage.
        
        Args:
            X_train: Training feature matrix (n_samples, n_features)
            y_train: Training target array (n_samples,)
            feature_names: Optional list of feature names
            
        Returns:
            self
        """
        n_features = X_train.shape[1]
        logger.info(f"Running QIGA on {X_train.shape[0]} samples, {n_features} features")
        
        # Initialize quantum population
        quantum_pop = self._initialize_quantum_population(n_features)
        
        # Initialize best solution with random selection
        self.best_solution = np.zeros(n_features, dtype=np.int32)
        random_idx = self._rng.choice(n_features, size=self.n_features_to_select, replace=False)
        self.best_solution[random_idx] = 1
        self.best_fitness = -np.inf
        self.fitness_history = []
        
        # Evolution loop
        for gen in range(self.n_generations):
            # Observe binary solutions from quantum population
            binary_pop = self._observe_population(quantum_pop)
            
            # Enforce approximate feature count
            for i in range(self.population_size):
                binary_pop[i] = self._enforce_feature_count(
                    binary_pop[i], self.n_features_to_select
                )
            
            # Evaluate fitness
            fitness_scores = self._evaluate_fitness(binary_pop, X_train, y_train)
            
            # Update best solution
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_solution = binary_pop[best_idx].copy()
            
            self.fitness_history.append(self.best_fitness)
            
            if gen % 10 == 0:
                logger.info(f"Generation {gen}: Best fitness = {self.best_fitness:.4f}")
            
            # Update quantum population
            quantum_pop = self._update_quantum_population(
                quantum_pop, binary_pop, fitness_scores, self.best_solution
            )
            
            # Apply mutation
            quantum_pop = self._apply_mutation(quantum_pop)
        
        # Extract selected feature indices
        self.selected_indices = np.where(self.best_solution == 1)[0]
        
        if feature_names is not None:
            selected_names = [feature_names[i] for i in self.selected_indices[:5]]
            logger.info(f"Top 5 selected features: {selected_names}")
        
        logger.info(f"QIGA complete: Selected {len(self.selected_indices)} features, "
                   f"best fitness = {self.best_fitness:.4f}")
        
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
    
    def get_fitness_history(self) -> List[float]:
        """
        Get the history of best fitness values across generations.
        
        Returns:
            List of best fitness values per generation
        """
        return self.fitness_history.copy()
