"""
Export Tables from cleaned_notebook.py
========================================
This script runs the analysis and exports all result tables to CSV files
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Create output directory
output_dir = "exported_tables"
os.makedirs(output_dir, exist_ok=True)
print(f"📁 Created output directory: {output_dir}")

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

print("\n📊 Loading and preprocessing data...")
df = pd.read_csv("pd_speech_features.csv")
print(f"Dataset shape: {df.shape}")

# Handle missing values
imputer = SimpleImputer(strategy='median')
df_numeric = df.apply(pd.to_numeric, errors='coerce')
df_imputed = pd.DataFrame(
    imputer.fit_transform(df_numeric),
    columns=df_numeric.columns
)

# Prepare features and target
X = df_imputed.drop(columns=['class'])
y = df_imputed['class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# ============================================================================
# MODEL SETUP
# ============================================================================

print("\n🔧 Setting up models...")

# Cross-validation setup
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = ['accuracy', 'recall', 'f1', 'roc_auc']

# Classical Models
log_reg = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])

svm = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(kernel='rbf', probability=True, random_state=42))
])

xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def cv_results(model, X, y):
    """Perform cross-validation and return mean scores."""
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    return {
        metric: scores[f'test_{metric}'].mean() 
        for metric in scoring
    }

def test_results(model, X_train, y_train, X_test, y_test):
    """Train model and evaluate on test set."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob)
    }

# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================

print("\n🚀 Training and evaluating models...")

# Cross-validation results
print("\n📊 Performing cross-validation...")
cv_log = cv_results(log_reg, X_train, y_train)
cv_svm = cv_results(svm, X_train, y_train)
cv_xgb = cv_results(xgb, X_train, y_train)

cv_df = pd.DataFrame({
    'Logistic Regression': cv_log,
    'SVM': cv_svm,
    'XGBoost': cv_xgb
}).T

# Export CV results
cv_df.to_csv(os.path.join(output_dir, "1_cross_validation_results.csv"))
print("\n✅ Table 1: Cross-Validation Results")
print(cv_df.round(4))

# Test set results
print("\n📊 Evaluating on test set...")
test_log = test_results(log_reg, X_train, y_train, X_test, y_test)
test_svm = test_results(svm, X_train, y_train, X_test, y_test)
test_xgb = test_results(xgb, X_train, y_train, X_test, y_test)

results_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'SVM', 'XGBoost'],
    'Accuracy': [test_log['Accuracy'], test_svm['Accuracy'], test_xgb['Accuracy']],
    'Recall': [test_log['Recall'], test_svm['Recall'], test_xgb['Recall']],
    'F1-Score': [test_log['F1'], test_svm['F1'], test_xgb['F1']],
    'AUC-ROC': [test_log['AUC'], test_svm['AUC'], test_xgb['AUC']]
})

# Export test results
results_df.to_csv(os.path.join(output_dir, "2_test_set_results.csv"), index=False)
print("\n✅ Table 2: Test Set Results")
print(results_df.to_string(index=False))

# Quantum results (from original analysis)
quantum_df = pd.DataFrame([
    ['QSVM', 0.756579, 0.973451, 0.856031, 'N/A'],
    ['VQC/QNN', 0.743421, 0.973451, 0.849421, 'N/A']
], columns=['Model', 'Accuracy', 'Recall', 'F1-Score', 'AUC-ROC'])

# Export quantum results
quantum_df.to_csv(os.path.join(output_dir, "3_quantum_model_results.csv"), index=False)
print("\n✅ Table 3: Quantum Model Results")
print(quantum_df.to_string(index=False))

# Combined results
all_models_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'SVM', 'XGBoost', 'QSVM', 'VQC/QNN'],
    'Type': ['Classical', 'Classical', 'Classical', 'Quantum', 'Quantum'],
    'Accuracy': [test_log['Accuracy'], test_svm['Accuracy'], test_xgb['Accuracy'], 0.756579, 0.743421],
    'Recall': [test_log['Recall'], test_svm['Recall'], test_xgb['Recall'], 0.973451, 0.973451],
    'F1-Score': [test_log['F1'], test_svm['F1'], test_xgb['F1'], 0.856031, 0.849421],
    'AUC': [test_log['AUC'], test_svm['AUC'], test_xgb['AUC'], 'N/A', 'N/A']
}).sort_values('F1-Score', ascending=False)

# Export combined results
all_models_df.to_csv(os.path.join(output_dir, "4_all_models_comparison.csv"), index=False)
print("\n✅ Table 4: All Models Comparison (Ranked by F1-Score)")
print(all_models_df.to_string(index=False))

# Feature importance
print("\n📊 Calculating feature importance...")
xgb_fitted = XGBClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42
)
xgb_fitted.fit(X_train, y_train)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_fitted.feature_importances_
}).sort_values('Importance', ascending=False)

# Export feature importance
feature_importance.to_csv(os.path.join(output_dir, "5_feature_importance_all.csv"), index=False)
print("\n✅ Table 5: Feature Importance (All Features)")
print(f"Total features: {len(feature_importance)}")

# Export top 20 features
top_20_features = feature_importance.head(20)
top_20_features.to_csv(os.path.join(output_dir, "6_feature_importance_top20.csv"), index=False)
print("\n✅ Table 6: Top 20 Most Important Features")
print(top_20_features.to_string(index=False))

# Class distribution
class_dist_df = pd.DataFrame({
    'Class': ['Healthy (0)', "Parkinson's (1)"],
    'Count': [int((y == 0).sum()), int((y == 1).sum())],
    'Percentage': [f"{(y == 0).mean()*100:.1f}%", f"{(y == 1).mean()*100:.1f}%"]
})

class_dist_df.to_csv(os.path.join(output_dir, "7_class_distribution.csv"), index=False)
print("\n✅ Table 7: Class Distribution")
print(class_dist_df.to_string(index=False))

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("📋 EXPORT COMPLETE!")
print("=" * 60)

print(f"\n📁 All tables exported to: {output_dir}/")
print("\nExported files:")
print("  1. 1_cross_validation_results.csv - CV performance metrics")
print("  2. 2_test_set_results.csv - Classical models test performance")
print("  3. 3_quantum_model_results.csv - Quantum models performance")
print("  4. 4_all_models_comparison.csv - Complete comparison ranked by F1")
print("  5. 5_feature_importance_all.csv - All features with importance scores")
print("  6. 6_feature_importance_top20.csv - Top 20 features")
print("  7. 7_class_distribution.csv - Dataset class balance")

print("\n✨ All tables successfully exported!")
print("=" * 60)
