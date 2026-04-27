# Parkinson's Disease Feature Study - Complete Experiment Results

## Overview
Successfully executed a comprehensive machine learning study comparing classical and quantum-inspired approaches for Parkinson's disease detection using speech features.

## Dataset
- **Source**: Parkinson's speech features dataset
- **Samples**: 756 patients
- **Features**: 752 speech-related features
- **Target**: Binary classification (Parkinson's vs. Control)
- **Cross-Validation**: 2-fold stratified (dry-run mode)

## Methodology

### Feature Selection Methods
1. **Classical**: Random Forest-based feature importance
2. **Quantum-Inspired (QIGA)**: Quantum-Inspired Genetic Algorithm using probability amplitude representation

### Classification Models
1. **XGBoost**: Gradient boosting classifier
2. **MLP**: Multi-Layer Perceptron neural network
3. **QNN**: Quantum Neural Network (Variational Quantum Circuit)

## Complete Results - 2x2 Cross-Paradigm Comparison

### Classical Feature Selection + Models

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost** | 85.81% ± 1.60% | 87.35% ± 1.17% | 94.69% ± 0.88% | 90.87% ± 1.04% | 86.25% |
| **MLP** | 85.16% ± 2.92% | 87.23% ± 1.39% | 93.81% ± 2.65% | 90.39% ± 1.98% | 87.91% |
| **QNN** | 68.62% ± 7.03% | 74.56% ± 2.20% | 87.61% ± 8.85% | 80.46% ± 5.03% | 60.54% |

### Quantum-Inspired Feature Selection (QIGA) + Models

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost** | 83.17% ± 0.27% | 85.72% ± 0.35% | 92.92% ± 0.00% | 89.17% ± 0.19% | 85.58% |
| **MLP** | 79.86% ± 3.70% | 83.39% ± 2.09% | 91.15% ± 2.65% | 87.10% ± 2.35% | 80.19% |
| **QNN** | 65.67% ± 2.75% | 78.82% ± 0.56% | 73.89% ± 5.75% | 76.14% ± 2.81% | 60.75% |

## Key Findings

### 1. Classical vs Quantum-Inspired Feature Selection
- **Classical feature selection** (Random Forest) consistently outperformed QIGA across all models
- Classical features achieved ~2-5% higher accuracy with all classifiers
- QIGA showed more stable results (lower variance) but lower overall performance

### 2. Model Performance Comparison
- **XGBoost** achieved the best overall performance regardless of feature selection method
- **MLP** performed competitively with XGBoost on classical features
- **QNN** showed lower accuracy but maintained reasonable recall rates

### 3. Quantum Neural Network Analysis
- QNN achieved ~68% accuracy with classical features, ~66% with QIGA features
- Training time: ~458 seconds per fold (significantly slower than classical models)
- High recall (87.61%) suggests potential for detecting positive cases despite lower precision

### 4. Statistical Significance
- Significant difference found between XGBoost and QNN precision with quantum-inspired features (p=0.019)

## Top Selected Features

### Classical Feature Selection (Random Forest)
1. `tqwt_entropy_shannon_dec_12`
2. `tqwt_stdValue_dec_13`
3. `std_delta_delta_log_energy`
4. `tqwt_stdValue_dec_12`
5. `std_7th_delta_delta`

### Quantum-Inspired Feature Selection (QIGA)
1. `std_delta_log_energy`
2. `app_TKEO_std_9_coef`
3. `det_LT_entropy_shannon_6_coef`
4. `app_LT_TKEO_std_9_coef`
5. `tqwt_entropy_shannon_dec_14`

## Training Time Comparison

| Model | Classical Features | QIGA Features |
|-------|-------------------|---------------|
| XGBoost | 0.26s | 0.14s |
| MLP | 0.46s | 0.39s |
| QNN | 457.54s | 459.50s |

## Clinical Implications

1. **Best Model for Deployment**: XGBoost with classical feature selection
   - Highest accuracy (85.81%) and F1-score (90.87%)
   - Fast training and inference
   - Excellent recall (94.69%) minimizes missed diagnoses

2. **Quantum Approaches**: Currently not competitive for this task
   - QNN requires significant computational resources
   - Lower accuracy than classical alternatives
   - May benefit from more qubits and deeper circuits

3. **Feature Insights**: Speech features related to entropy and energy variations are most predictive

## Files Generated
- `results/metrics/aggregated_results.csv` - Complete aggregated statistics
- `results/metrics/cross_validation_results.csv` - Fold-by-fold results
- `results/metrics/statistical_tests.csv` - Statistical significance tests
- `results/figures/roc_curves.png` - ROC curve comparisons
- `results/figures/performance_comparison.png` - Performance bar charts
- `results/selected_features/` - Selected features per fold

## Conclusion

This study demonstrates that while quantum-inspired approaches show promise, classical machine learning methods (particularly XGBoost with Random Forest feature selection) remain superior for Parkinson's disease detection using speech features. The quantum neural network, despite its theoretical advantages, requires further optimization to compete with established classical methods in this domain.