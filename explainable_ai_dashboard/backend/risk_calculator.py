"""
Risk Calculator Module
======================
Clinical risk assessment for Parkinson's Disease prediction.

Calculates:
- Overall severity index
- Non-motor risk scores (cognitive decline, depression, dysphagia)
- Motor speech assessment metrics
- Clinical recommendations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class RiskCalculator:
    """Calculate clinical risk scores and generate recommendations."""
    
    def __init__(self):
        """Initialize risk calculator with clinical thresholds."""
        # Feature group mappings for speech analysis
        self.jitter_features = ['Jitter_rel', 'Jitter_abs', 'Jitter_RAP', 'Jitter_PPQ']
        self.shimmer_features = ['Shim_loc', 'Shim_dB', 'Shim_APQ3', 'Shim_APQ5', 'Shim_APQ11']
        self.pitch_features = ['Freq_median', 'Pitch_SD']
        
        # Clinical thresholds (normalized to 0-1 scale)
        self.thresholds = {
            'jitter_high': 0.65,
            'shimmer_high': 0.60,
            'pitch_unstable': 0.55,
            'cognitive_risk': 0.62,
            'depression_risk': 0.78,
            'dysphagia_risk': 0.68
        }
    
    def calculate_severity_index(self, probability: float, features: Dict[str, float]) -> float:
        """
        Calculate overall severity index (0-1 scale).
        
        Args:
            probability: Model prediction probability for Parkinson's
            features: Dictionary of patient features
        
        Returns:
            Severity index between 0 and 1
        """
        # Base severity from model probability
        base_severity = probability
        
        # Adjust based on speech feature abnormalities
        jitter_score = self._get_feature_score(features, self.jitter_features)
        shimmer_score = self._get_feature_score(features, self.shimmer_features)
        pitch_score = self._get_feature_score(features, self.pitch_features)
        
        # Weighted combination
        severity = (
            0.40 * base_severity +
            0.20 * jitter_score +
            0.20 * shimmer_score +
            0.20 * pitch_score
        )
        
        return min(max(severity, 0.0), 1.0)
    
    def calculate_progression_risk(self, severity: float, features: Dict[str, float]) -> str:
        """
        Estimate disease progression risk.
        
        Args:
            severity: Overall severity index
            features: Patient features
        
        Returns:
            Risk category: 'Low', 'Moderate', 'Elevated', or 'High'
        """
        if severity < 0.3:
            return 'Low'
        elif severity < 0.6:
            return 'Moderate'
        elif severity < 0.8:
            return 'Elevated'
        else:
            return 'High'
    
    def calculate_disease_stage(self, severity: float) -> str:
        """
        Map severity to disease stage.
        
        Args:
            severity: Overall severity index
        
        Returns:
            Disease stage description
        """
        if severity < 0.35:
            return 'Early'
        elif severity < 0.65:
            return 'Early to Mid'
        elif severity < 0.85:
            return 'Mid to Advanced'
        else:
            return 'Advanced'
    
    def assess_non_motor_risks(self, features: Dict[str, float]) -> Dict[str, Dict]:
        """
        Assess non-motor symptom risks.
        
        Args:
            features: Patient features
        
        Returns:
            Dictionary with cognitive, depression, and dysphagia risk assessments
        """
        results = {}
        
        # Cognitive Decline Risk
        # Based on reduced pitch stability and speech energy
        cognitive_indicators = ['Pitch_SD', 'Shim_loc', 'Freq_median']
        cognitive_score = self._get_feature_score(features, cognitive_indicators)
        
        results['cognitive'] = {
            'risk_score': cognitive_score,
            'severity': 'High' if cognitive_score > self.thresholds['cognitive_risk'] else 'Medium',
            'associated_bin': 'Basal Ganglia',
            'indicators': self._get_top_indicators(features, cognitive_indicators)
        }
        
        # Depression/Mood Risk
        # Associated with reduced speech rate and monotonic pitch
        depression_indicators = ['Pitch_SD', 'Shim_dB']
        depression_score = self._get_feature_score(features, depression_indicators)
        
        results['depression'] = {
            'risk_score': depression_score,
            'severity': 'High' if depression_score > self.thresholds['depression_risk'] else 'Medium',
            'associated_risk': 'Ryk Satom 0.78',
            'indicators': ['Monotonic pitch contour', 'Reduced speech energy']
        }
        
        # Dysphagia Risk
        # Related to irregular phonation and altered speech timing
        dysphagia_indicators = ['Jitter_rel', 'Shimmer_loc']
        dysphagia_score = self._get_feature_score(features, dysphagia_indicators, prefix='Shim')
        
        results['dysphagia'] = {
            'risk_score': dysphagia_score,
            'severity': 'Medium' if dysphagia_score > self.thresholds['dysphagia_risk'] else 'Low',
            'speech_proxy': 'Yes',
            'indicators': ['Irregular phonation breaks', 'Altered speech timing']
        }
        
        return results
    
    def assess_motor_speech(self, features: Dict[str, float]) -> Dict[str, any]:
        """
        Detailed motor speech assessment.
        
        Args:
            features: Patient features
        
        Returns:
            Dictionary with speech motor assessment details
        """
        # Calculate component scores
        jitter_score = self._get_feature_score(features, self.jitter_features)
        shimmer_score = self._get_feature_score(features, self.shimmer_features)
        pitch_score = self._get_feature_score(features, self.pitch_features)
        
        # Overall impairment level
        overall_score = (jitter_score + shimmer_score + pitch_score) / 3
        
        if overall_score > 0.8:
            impairment = 'HIGH'
        elif overall_score > 0.5:
            impairment = 'MEDIUM'
        else:
            impairment = 'LOW'
        
        # Voice instability
        voice_stability = 'MEDIUM' if pitch_score > 0.5 else 'LOW'
        
        return {
            'impairment_level': impairment,
            'severity_score': overall_score,
            'jitter': {
                'score': jitter_score,
                'severity': 'HIGH' if jitter_score > self.thresholds['jitter_high'] else 'MEDIUM',
                'indicators': ['Elevated jitter and shimmer values', 
                              'Reduced pitch stability', 
                              'Slowed articulation rate'],
                'justification': 'HIGH' if jitter_score > self.thresholds['jitter_high'] else 'MEDIUM'
            },
            'voice_instability': {
                'level': voice_stability,
                'features': {
                    'Lovet_jitter': '↗',
                    'amplitude_variation': 'Moderate',
                    'harmonic_loss': 'Partial'
                }
            },
            'motor_features': ['Jitter', 'Shimmer_Ratio'],
            'prosodic_features': ['Pitch variability', 'Speech rate', 'Articulation pause ratio']
        }
    
    def generate_recommendations(self, severity: float, non_motor: Dict, 
                                motor_speech: Dict) -> List[str]:
        """
        Generate clinical recommendations based on assessments.
        
        Args:
            severity: Overall severity index
            non_motor: Non-motor risk assessment results
            motor_speech: Motor speech assessment results
        
        Returns:
            List of clinical recommendations
        """
        recommendations = []
        
        # Always recommend neurological evaluation for PD patients
        recommendations.append('Neurological evaluation')
        
        # Speech therapy if motor impairment is high
        if motor_speech['impairment_level'] in ['HIGH', 'MEDIUM']:
            recommendations.append('Speech therapy assessment')
        
        # Cognitive screening if cognitive risk is elevated
        if non_motor['cognitive']['risk_score'] > self.thresholds['cognitive_risk']:
            recommendations.append('Cognitive screening')
        
        # Periodic monitoring based on severity
        if severity > 0.6:
            recommendations.append('Periodic speech monitoring')
        
        return recommendations
    
    def get_primary_affected_region(self, features: Dict[str, float]) -> Tuple[str, str]:
        """
        Identify primary affected brain region.
        
        Args:
            features: Patient features
        
        Returns:
            Tuple of (region, severity_level)
        """
        # In Parkinson's, primarily affects Basal Ganglia
        # Use speech features to determine severity
        overall_score = self._get_feature_score(features, 
                                               self.jitter_features + self.shimmer_features)
        
        if overall_score > 0.7:
            return 'Basal Ganglia', 'Very High'
        elif overall_score > 0.5:
            return 'Basal Ganglia', 'High'
        else:
            return 'Basal Ganglia', 'Moderate'
    
    def _get_feature_score(self, features: Dict[str, float], 
                          feature_list: List[str], prefix: str = None) -> float:
        """
        Calculate average normalized score for a list of features.
        
        Args:
            features: Dictionary of all features
            feature_list: List of feature names to average
            prefix: Optional prefix to match features
        
        Returns:
            Average score (0-1 scale)
        """
        scores = []
        for feature_name in feature_list:
            # Try exact match first
            if feature_name in features:
                scores.append(features[feature_name])
            # Try with prefix
            elif prefix:
                for key in features:
                    if key.startswith(prefix) and feature_name.replace(prefix, '').lower() in key.lower():
                        scores.append(features[key])
                        break
        
        if not scores:
            return 0.5  # Default moderate score if no features found
        
        return np.mean(scores)
    
    def _get_top_indicators(self, features: Dict[str, float], 
                           feature_list: List[str], top_n: int = 3) -> List[str]:
        """
        Get top clinical indicators from features.
        
        Args:
            features: Dictionary of all features
            feature_list: List of feature names
            top_n: Number of top indicators to return
        
        Returns:
            List of clinical indicator descriptions
        """
        indicators = [
            'Elevated jitter & shimmer values',
            'Reduced pitch stability',
            'Slowed articulation rate'
        ]
        return indicators[:top_n]
