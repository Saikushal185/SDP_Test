"""
SHAP Explainability Module
===========================
Generate model explanations using SHAP (SHapley Additive exPlanations).

Provides:
- Feature importance calculation
- Individual prediction explanations
- SHAP values for multiple models
- Aggregated model contributions
"""

import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class SHAPExplainer:
    """SHAP-based explainability for Parkinson's disease models."""
    
    def __init__(self, models: Dict[str, Any], X_background: pd.DataFrame):
        """
        Initialize SHAP explainers for all models.
        
        Args:
            models: Dictionary of trained models {model_name: model}
            X_background: Background dataset for SHAP (training data)
        """
        self.models = models
        self.X_background = X_background
        self.feature_names = X_background.columns.tolist()
        self.explainers = {}
        
        # Initialize SHAP explainers for each model
        self._initialize_explainers()
    
    def _initialize_explainers(self):
        """Create SHAP explainers for each model type."""
        for model_name, model in self.models.items():
            try:
                if 'XGBoost' in model_name or 'xgb' in model_name.lower():
                    # Tree explainer for XGBoost
                    self.explainers[model_name] = shap.TreeExplainer(model)
                else:
                    # Kernel explainer for other models (SVM, MLP)
                    # Use a small background sample for efficiency
                    background_sample = shap.sample(self.X_background, min(100, len(self.X_background)))
                    
                    # Extract the actual model from pipeline if needed
                    if hasattr(model, 'named_steps'):
                        actual_model = model.named_steps['model']
                        # Need to transform background data through the scaler
                        if 'scaler' in model.named_steps:
                            background_transformed = model.named_steps['scaler'].transform(background_sample)
                            background_sample = pd.DataFrame(
                                background_transformed, 
                                columns=self.feature_names
                            )
                        self.explainers[model_name] = shap.KernelExplainer(
                            actual_model.predict_proba, 
                            background_sample
                        )
                    else:
                        self.explainers[model_name] = shap.KernelExplainer(
                            model.predict_proba, 
                            background_sample
                        )
            except Exception as e:
                print(f"Warning: Could not create SHAP explainer for {model_name}: {e}")
    
    def explain_prediction(self, X_patient: pd.DataFrame, 
                          model_name: str = 'XGBoost') -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single patient prediction.
        
        Args:
            X_patient: Single patient feature vector (1 row DataFrame)
            model_name: Name of the model to explain
        
        Returns:
            Dictionary with SHAP values and feature contributions
        """
        if model_name not in self.explainers:
            return self._fallback_explanation(X_patient, model_name)
        
        try:
            explainer = self.explainers[model_name]
            model = self.models[model_name]
            
            # Get prediction
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(X_patient)[0, 1]
            else:
                prediction_proba = model.predict(X_patient)[0]
            
            # Calculate SHAP values
            if 'XGBoost' in model_name or 'xgb' in model_name.lower():
                shap_values = explainer.shap_values(X_patient)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Class 1 (Parkinson's)
            else:
                # For kernel explainer, transform data if using pipeline
                if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
                    X_transformed = model.named_steps['scaler'].transform(X_patient)
                    X_patient_transformed = pd.DataFrame(X_transformed, columns=self.feature_names)
                    shap_values = explainer.shap_values(X_patient_transformed)
                else:
                    shap_values = explainer.shap_values(X_patient)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Class 1
            
            # Get feature contributions
            shap_values_flat = shap_values.flatten() if len(shap_values.shape) > 1 else shap_values
            
            # Create feature contribution dictionary
            contributions = {}
            for idx, feature in enumerate(self.feature_names):
                contributions[feature] = {
                    'value': float(X_patient.iloc[0, idx]),
                    'shap_value': float(shap_values_flat[idx]),
                    'contribution': abs(float(shap_values_flat[idx]))
                }
            
            # Sort by contribution magnitude
            sorted_features = sorted(
                contributions.items(), 
                key=lambda x: x[1]['contribution'], 
                reverse=True
            )
            
            return {
                'model': model_name,
                'prediction_probability': float(prediction_proba),
                'shap_values': shap_values_flat.tolist(),
                'feature_contributions': contributions,
                'top_features': dict(sorted_features[:10]),
                'base_value': float(explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) 
                                   else explainer.expected_value)
            }
        
        except Exception as e:
            print(f"Warning: SHAP explanation failed for {model_name}: {e}")
            return self._fallback_explanation(X_patient, model_name)
    
    def _fallback_explanation(self, X_patient: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """
        Fallback explanation using feature importance from model.
        
        Args:
            X_patient: Patient features
            model_name: Model name
        
        Returns:
            Simple feature importance explanation
        """
        model = self.models[model_name]
        
        # Get prediction
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(X_patient)[0, 1]
        else:
            prediction_proba = model.predict(X_patient)[0]
        
        # Use feature importances if available (for XGBoost)
        contributions = {}
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for idx, feature in enumerate(self.feature_names):
                contributions[feature] = {
                    'value': float(X_patient.iloc[0, idx]),
                    'shap_value': float(importances[idx] * X_patient.iloc[0, idx]),
                    'contribution': float(importances[idx])
                }
        else:
            # Generic contribution based on feature variance
            for idx, feature in enumerate(self.feature_names):
                contributions[feature] = {
                    'value': float(X_patient.iloc[0, idx]),
                    'shap_value': 0.0,
                    'contribution': abs(float(X_patient.iloc[0, idx]))
                }
        
        sorted_features = sorted(
            contributions.items(), 
            key=lambda x: x[1]['contribution'], 
            reverse=True
        )
        
        return {
            'model': model_name,
            'prediction_probability': float(prediction_proba),
            'shap_values': [0.0] * len(self.feature_names),
            'feature_contributions': contributions,
            'top_features': dict(sorted_features[:10]),
            'base_value': 0.5
        }
    
    def get_global_feature_importance(self, model_name: str = 'XGBoost', 
                                     X_sample: pd.DataFrame = None, 
                                     n_samples: int = 100) -> Dict[str, float]:
        """
        Get global feature importance across multiple samples.
        
        Args:
            model_name: Name of the model
            X_sample: Sample of data to analyze (optional)
            n_samples: Number of samples to use
        
        Returns:
            Dictionary of feature importances
        """
        if X_sample is None:
            X_sample = self.X_background.sample(min(n_samples, len(self.X_background)))
        
        try:
            explainer = self.explainers[model_name]
            
            if 'XGBoost' in model_name or 'xgb' in model_name.lower():
                shap_values = explainer.shap_values(X_sample)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:
                model = self.models[model_name]
                if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
                    X_transformed = model.named_steps['scaler'].transform(X_sample)
                    X_sample = pd.DataFrame(X_transformed, columns=self.feature_names)
                
                shap_values = explainer.shap_values(X_sample)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            
            # Calculate mean absolute SHAP values
            mean_shap = np.abs(shap_values).mean(axis=0)
            
            importance_dict = {
                feature: float(importance) 
                for feature, importance in zip(self.feature_names, mean_shap)
            }
            
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        except Exception as e:
            print(f"Warning: Global importance calculation failed: {e}")
            # Fallback to model feature importances
            model = self.models[model_name]
            if hasattr(model, 'feature_importances_'):
                return dict(zip(self.feature_names, model.feature_importances_))
            else:
                return {feature: 0.0 for feature in self.feature_names}
    
    def get_model_contributions(self, X_patient: pd.DataFrame) -> Dict[str, float]:
        """
        Get contribution of each model to the final prediction.
        
        Args:
            X_patient: Patient features
        
        Returns:
            Dictionary of model contributions
        """
        contributions = {}
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_patient)[0, 1]
                else:
                    proba = model.predict(X_patient)[0]
                
                contributions[model_name] = float(proba)
            except Exception as e:
                print(f"Warning: Could not get contribution for {model_name}: {e}")
                contributions[model_name] = 0.5
        
        return contributions
    
    def format_explanation_for_dashboard(self, explanation: Dict[str, Any], 
                                        top_n: int = 5) -> Dict[str, Any]:
        """
        Format explanation for dashboard display.
        
        Args:
            explanation: SHAP explanation dictionary
            top_n: Number of top features to include
        
        Returns:
            Formatted explanation for frontend
        """
        top_features = list(explanation['top_features'].items())[:top_n]
        
        formatted = {
            'prediction': {
                'probability': explanation['prediction_probability'],
                'class': 'Parkinson\'s Disease' if explanation['prediction_probability'] > 0.5 else 'Healthy',
                'confidence': abs(explanation['prediction_probability'] - 0.5) * 2
            },
            'key_features': [
                {
                    'name': self._format_feature_name(feature_name),
                    'value': details['value'],
                    'impact': 'Positive' if details['shap_value'] > 0 else 'Negative',
                    'importance': details['contribution']
                }
                for feature_name, details in top_features
            ],
            'decision_basis': self._generate_decision_basis(top_features)
        }
        
        return formatted
    
    def _format_feature_name(self, feature_name: str) -> str:
        """Convert technical feature name to readable format."""
        # Simple formatting - can be enhanced
        return feature_name.replace('_', ' ').title()
    
    def _generate_decision_basis(self, top_features: List[Tuple[str, Dict]]) -> List[str]:
        """
        Generate human-readable decision basis from top features.
        
        Args:
            top_features: List of (feature_name, details) tuples
        
        Returns:
            List of decision basis statements
        """
        basis = []
        
        for feature_name, details in top_features[:3]:
            readable_name = self._format_feature_name(feature_name)
            if 'jitter' in feature_name.lower():
                basis.append('Consistent motor speech degradation')
            elif 'shimmer' in feature_name.lower():
                basis.append('High-frequency instability in phonation')
            elif 'pitch' in feature_name.lower() or 'freq' in feature_name.lower():
                basis.append('Reduced pitch stability')
        
        if not basis:
            basis = ['Speech feature analysis', 'Motor pattern recognition', 'Acoustic abnormalities']
        
        return basis[:3]
