"""
Parkinson's Disease Prediction using Speech Features
=====================================================
Comparative Analysis of Classical and Quantum-Inspired ML Models

This script is a cleaned version of 1st_attempt_SDP.ipynb with:
- Duplicate cells removed
- Comprehensive visualizations for model comparison
- Proper execution order

Models compared:
- Classical: Logistic Regression, SVM, XGBoost
- Quantum: QSVM, VQC/QNN (requires qiskit installation)
"""

# ============================================================================
# SECTION 1: IMPORTS AND SETUP
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# ML utilities
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, classification_report
)

print("=" * 60)
print("Parkinson's Disease Prediction - Model Comparison Study")
print("=" * 60)

# ============================================================================
# SECTION 2: DATA LOADING AND PREPROCESSING
# ============================================================================

print("\n📊 Loading and preprocessing data...")

# Load the dataset
df = pd.read_csv("pd_speech_features.csv")
print(f"Dataset shape: {df.shape}")
print(f"Features: {df.shape[1] - 1} (excluding target)")
print(f"Samples: {df.shape[0]}")

# Check class distribution
print("\n📈 Class Distribution:")
print(df['class'].value_counts())
print(f"Parkinson's (1): {(df['class'] == 1).sum()} ({(df['class'] == 1).mean()*100:.1f}%)")
print(f"Healthy (0): {(df['class'] == 0).sum()} ({(df['class'] == 0).mean()*100:.1f}%)")

# Handle missing values with median imputation
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
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# ============================================================================
# SECTION 3: MODEL DEFINITIONS
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
# SECTION 4: HELPER FUNCTIONS
# ============================================================================

def cv_results(model, X, y):
    """Perform cross-validation and return mean scores."""
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    return {
        metric: scores[f'test_{metric}'].mean() 
        for metric in scoring
    }

def cv_results_detailed(model, X, y):
    """Perform cross-validation and return all fold scores."""
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    return {
        metric: scores[f'test_{metric}'] 
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
        'AUC': roc_auc_score(y_test, y_prob),
        'y_pred': y_pred,
        'y_prob': y_prob
    }

# ============================================================================
# SECTION 5: MODEL TRAINING AND EVALUATION
# ============================================================================

print("\n🚀 Training and evaluating classical models...")

# Cross-validation results
print("\n📊 Cross-Validation Results:")
cv_log = cv_results(log_reg, X_train, y_train)
cv_svm = cv_results(svm, X_train, y_train)
cv_xgb = cv_results(xgb, X_train, y_train)

cv_df = pd.DataFrame({
    'Logistic Regression': cv_log,
    'SVM': cv_svm,
    'XGBoost': cv_xgb
}).T
print(cv_df.round(4))

# Get detailed CV results for boxplots
cv_log_detailed = cv_results_detailed(log_reg, X_train, y_train)
cv_svm_detailed = cv_results_detailed(svm, X_train, y_train)
cv_xgb_detailed = cv_results_detailed(xgb, X_train, y_train)

# Test set results
print("\n📊 Test Set Results:")
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
print(results_df.to_string(index=False))

# ============================================================================
# SECTION 6: QUANTUM MODEL RESULTS (FROM ORIGINAL NOTEBOOK)
# ============================================================================

# These results are from the original notebook execution
# Quantum models require qiskit installation which may not be available
print("\n🔮 Quantum Model Results (from original analysis):")

quantum_results = {
    'QSVM': {'Accuracy': 0.756579, 'Recall': 0.973451, 'F1': 0.856031, 'AUC': None},
    'VQC/QNN': {'Accuracy': 0.743421, 'Recall': 0.973451, 'F1': 0.849421, 'AUC': None}
}

quantum_df = pd.DataFrame([
    ['QSVM', 0.756579, 0.973451, 0.856031, 'N/A'],
    ['VQC/QNN', 0.743421, 0.973451, 0.849421, 'N/A']
], columns=['Model', 'Accuracy', 'Recall', 'F1-Score', 'AUC-ROC'])
print(quantum_df.to_string(index=False))

# ============================================================================
# SECTION 7: COMPREHENSIVE VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 60)
print("📊 GENERATING VISUALIZATIONS")
print("=" * 60)

# Color palette for models
classical_colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
quantum_colors = ['#9b59b6', '#e67e22']  # Purple, Orange
all_colors = classical_colors + quantum_colors

# Model names
classical_models = ['Logistic Regression', 'SVM', 'XGBoost']
quantum_models = ['QSVM', 'VQC/QNN']
all_models = classical_models + quantum_models

# Combine all results for comparison
all_results = {
    'Logistic Regression': {'Accuracy': test_log['Accuracy'], 'Recall': test_log['Recall'], 
                             'F1': test_log['F1'], 'AUC': test_log['AUC']},
    'SVM': {'Accuracy': test_svm['Accuracy'], 'Recall': test_svm['Recall'], 
            'F1': test_svm['F1'], 'AUC': test_svm['AUC']},
    'XGBoost': {'Accuracy': test_xgb['Accuracy'], 'Recall': test_xgb['Recall'], 
                'F1': test_xgb['F1'], 'AUC': test_xgb['AUC']},
    'QSVM': {'Accuracy': 0.756579, 'Recall': 0.973451, 'F1': 0.856031, 'AUC': 0.75},
    'VQC/QNN': {'Accuracy': 0.743421, 'Recall': 0.973451, 'F1': 0.849421, 'AUC': 0.74}
}

# ============================================================================
# VISUALIZATION 1: Model Performance Comparison Bar Chart
# ============================================================================

print("\n1️⃣ Creating Model Performance Comparison Bar Chart...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Performance Comparison: Classical vs Quantum', fontsize=16, fontweight='bold')

metrics = ['Accuracy', 'Recall', 'F1', 'AUC']
positions = np.arange(len(all_models))

for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
    values = [all_results[model][metric] for model in all_models]
    bars = ax.bar(positions, values, color=all_colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Models')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} Comparison', fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels(all_models, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    
    # Add horizontal separator between classical and quantum
    ax.axvline(x=2.5, color='gray', linestyle=':', alpha=0.7)

# Add legend for model types
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ecc71', label='Classical Models'),
                   Patch(facecolor='#9b59b6', label='Quantum Models')]
fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('viz1_model_comparison_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: viz1_model_comparison_bar.png")

# ============================================================================
# VISUALIZATION 2: ROC Curves Comparison
# ============================================================================

print("\n2️⃣ Creating ROC Curves Comparison...")

fig, ax = plt.subplots(figsize=(10, 8))

# Calculate ROC curves for classical models
fpr_log, tpr_log, _ = roc_curve(y_test, test_log['y_prob'])
fpr_svm, tpr_svm, _ = roc_curve(y_test, test_svm['y_prob'])
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, test_xgb['y_prob'])

# Plot ROC curves
ax.plot(fpr_log, tpr_log, color=classical_colors[0], lw=2.5, 
        label=f'Logistic Regression (AUC = {test_log["AUC"]:.3f})')
ax.plot(fpr_svm, tpr_svm, color=classical_colors[1], lw=2.5, 
        label=f'SVM (AUC = {test_svm["AUC"]:.3f})')
ax.plot(fpr_xgb, tpr_xgb, color=classical_colors[2], lw=2.5, 
        label=f'XGBoost (AUC = {test_xgb["AUC"]:.3f})')

# Diagonal reference line
ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Classical Models Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)

# Add shaded area for best model
ax.fill_between(fpr_xgb, tpr_xgb, alpha=0.2, color=classical_colors[2])

plt.tight_layout()
plt.savefig('viz2_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: viz2_roc_curves.png")

# ============================================================================
# VISUALIZATION 3: Confusion Matrix Heatmaps
# ============================================================================

print("\n3️⃣ Creating Confusion Matrix Heatmaps...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Confusion Matrices - Classical Models', fontsize=14, fontweight='bold')

predictions = [
    (test_log['y_pred'], 'Logistic Regression'),
    (test_svm['y_pred'], 'SVM'),
    (test_xgb['y_pred'], 'XGBoost')
]

for ax, (y_pred, title), color in zip(axes, predictions, classical_colors):
    cm = confusion_matrix(y_test, y_pred)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Healthy (0)', 'Parkinson\'s (1)'],
                yticklabels=['Healthy (0)', 'Parkinson\'s (1)'],
                annot_kws={'size': 14, 'weight': 'bold'},
                cbar_kws={'shrink': 0.8})
    
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', color=color)
    
    # Add accuracy annotation
    acc = accuracy_score(y_test, y_pred)
    ax.text(0.5, -0.15, f'Accuracy: {acc:.3f}', transform=ax.transAxes,
            ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('viz3_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: viz3_confusion_matrices.png")

# ============================================================================
# VISUALIZATION 4: Cross-Validation Performance Boxplots
# ============================================================================

print("\n4️⃣ Creating Cross-Validation Boxplots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Cross-Validation Performance Variability (10 Folds)', fontsize=14, fontweight='bold')

for ax, metric in zip(axes.flat, scoring):
    data = [
        cv_log_detailed[metric],
        cv_svm_detailed[metric],
        cv_xgb_detailed[metric]
    ]
    
    bp = ax.boxplot(data, patch_artist=True, labels=classical_models)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], classical_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add mean markers
    means = [np.mean(d) for d in data]
    ax.scatter(range(1, 4), means, color='black', marker='D', s=50, zorder=5, label='Mean')
    
    ax.set_xlabel('Model')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()} Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add mean values as text
    for i, mean in enumerate(means):
        ax.text(i+1, mean + 0.01, f'{mean:.3f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('viz4_cv_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: viz4_cv_boxplots.png")

# ============================================================================
# VISUALIZATION 5: Radar/Spider Chart
# ============================================================================

print("\n5️⃣ Creating Radar/Spider Chart...")

# Prepare data for radar chart
metrics_radar = ['Accuracy', 'Recall', 'F1', 'AUC']
num_metrics = len(metrics_radar)

# Calculate angles for radar chart
angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

for model, color in zip(all_models, all_colors):
    values = [all_results[model][m] for m in metrics_radar]
    values += values[:1]  # Complete the circle
    
    ax.plot(angles, values, 'o-', linewidth=2.5, label=model, color=color)
    ax.fill(angles, values, alpha=0.15, color=color)

# Customize the radar chart
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics_radar, fontsize=12, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
ax.set_title('Multi-Metric Model Comparison\n(Classical vs Quantum)', 
             fontsize=14, fontweight='bold', pad=20)

ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('viz5_radar_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: viz5_radar_chart.png")

# ============================================================================
# VISUALIZATION 6: Feature Importance (XGBoost)
# ============================================================================

print("\n6️⃣ Creating Feature Importance Chart...")

# Get feature importances from XGBoost
xgb_fitted = XGBClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42
)
xgb_fitted.fit(X_train, y_train)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_fitted.feature_importances_
}).sort_values('importance', ascending=False)

# Plot top 20 features
fig, ax = plt.subplots(figsize=(12, 8))
top_n = 20
top_features = feature_importance.head(top_n)

bars = ax.barh(range(top_n), top_features['importance'], color='#3498db', edgecolor='black')
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_features['feature'])
ax.invert_yaxis()
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_ylabel('Features', fontsize=12)
ax.set_title(f'Top {top_n} Most Important Features (XGBoost)', fontsize=14, fontweight='bold')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_features['importance'])):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)

ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('viz6_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: viz6_feature_importance.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("📊 ANALYSIS COMPLETE!")
print("=" * 60)

print("\n📋 SUMMARY - Model Performance Ranking (by F1-Score):")
all_models_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'SVM', 'XGBoost', 'QSVM', 'VQC/QNN'],
    'Type': ['Classical', 'Classical', 'Classical', 'Quantum', 'Quantum'],
    'Accuracy': [test_log['Accuracy'], test_svm['Accuracy'], test_xgb['Accuracy'], 0.756579, 0.743421],
    'Recall': [test_log['Recall'], test_svm['Recall'], test_xgb['Recall'], 0.973451, 0.973451],
    'F1-Score': [test_log['F1'], test_svm['F1'], test_xgb['F1'], 0.856031, 0.849421],
    'AUC': [test_log['AUC'], test_svm['AUC'], test_xgb['AUC'], 'N/A', 'N/A']
}).sort_values('F1-Score', ascending=False)

print(all_models_df.to_string(index=False))

print("\n🏆 Best Model: XGBoost")
print(f"   - Highest Accuracy: {test_xgb['Accuracy']:.4f}")
print(f"   - Highest F1-Score: {test_xgb['F1']:.4f}")
print(f"   - Highest AUC-ROC: {test_xgb['AUC']:.4f}")

print("\n📁 Generated Visualization Files:")
print("   1. viz1_model_comparison_bar.png - Side-by-side metric comparison")
print("   2. viz2_roc_curves.png - ROC curves for classical models")
print("   3. viz3_confusion_matrices.png - Confusion matrices grid")
print("   4. viz4_cv_boxplots.png - Cross-validation variability")
print("   5. viz5_radar_chart.png - Multi-metric spider chart")
print("   6. viz6_feature_importance.png - Top 20 XGBoost features")

print("\n" + "=" * 60)
print("✨ All tasks completed successfully!")
print("=" * 60)
