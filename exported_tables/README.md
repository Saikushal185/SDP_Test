# Exported Tables from cleaned_notebook.py

This folder contains all the result tables from the Parkinson's Disease prediction analysis.

## 📊 Exported Files

### 1. `1_cross_validation_results.csv`
**Cross-Validation Performance Metrics (10-fold CV)**
- Contains accuracy, recall, F1-score, and AUC-ROC for each classical model
- Models: Logistic Regression, SVM, XGBoost
- Used for assessing model stability and generalization

### 2. `2_test_set_results.csv`
**Classical Models Test Set Performance**
- Final performance metrics on the held-out test set
- Models: Logistic Regression, SVM, XGBoost
- Metrics: Accuracy, Recall, F1-Score, AUC-ROC

### 3. `3_quantum_model_results.csv`
**Quantum Models Performance**
- Results from quantum-inspired models (from original analysis)
- Models: QSVM, VQC/QNN
- Metrics: Accuracy, Recall, F1-Score

### 4. `4_all_models_comparison.csv`
**Complete Model Comparison (Ranked by F1-Score)**
- Combines classical and quantum model results
- Sorted by F1-Score for easy comparison
- Includes model type classification (Classical/Quantum)

### 5. `5_feature_importance_all.csv`
**Complete Feature Importance Scores**
- All 754 features with their importance scores from XGBoost
- Sorted by importance (descending)
- Useful for comprehensive feature analysis

### 6. `6_feature_importance_top20.csv`
**Top 20 Most Important Features**
- Curated list of the 20 most influential features
- Includes feature names and importance scores
- Key features for Parkinson's disease prediction

### 7. `7_class_distribution.csv`
**Dataset Class Balance**
- Shows the distribution of Healthy vs Parkinson's samples
- Includes counts and percentages

## 📈 Key Findings

Based on the exported data:

1. **Best Performing Model**: XGBoost
   - Highest Accuracy, F1-Score, and AUC-ROC among all models
   - Shows excellent balance between precision and recall

2. **High Recall Models**: Quantum models (QSVM, VQC/QNN)
   - Achieve very high recall (~97.3%) for detecting Parkinson's
   - Good for minimizing false negatives in medical diagnosis

3. **Top Features**: 
   - Speech feature engineering features dominate (tqwt_entropy, TKEO)
   - Jitter-based features also show high importance

## 🔍 Usage

These CSV files can be:
- Imported into Excel, Google Sheets, or other spreadsheet tools
- Loaded into Python/R for further analysis
- Used for creating custom visualizations
- Referenced in research papers or reports

## 📝 Notes

- All classical model results are from actual test set evaluation
- Quantum model results are from the original notebook execution
- Random seed: 42 for reproducibility
- Train-test split: 80-20 with stratification
