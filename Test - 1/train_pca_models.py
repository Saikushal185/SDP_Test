"""
Train Classical ML Models on 4 PCA Features
=============================================
Uses the SAME 4 PCA components as the Quantum model (QSVM/VQC)
for a fair apples-to-apples comparison.

Pipeline (identical to run_quantum_fast.py):
  Raw features → StandardScaler → PCA(n_components=4) → Train ML models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ML utilities
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

print("=" * 60)
print("Classical Models on PCA Features (Quantum-Matched)")
print("=" * 60)

# ============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# ============================================================================

print("\n📊 Loading and preprocessing data...")

df = pd.read_csv("pd_speech_features.csv")
print(f"   Dataset shape: {df.shape}")
print(f"   Features: {df.shape[1] - 1} (excluding target)")

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

# Train-test split (SAME random_state as all other scripts)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"   Training set: {len(X_train)} samples")
print(f"   Test set:     {len(X_test)} samples")

# ============================================================================
# SECTION 2: PCA TRANSFORMATION (IDENTICAL TO QUANTUM PIPELINE)
# ============================================================================

print("\n🔧 Applying PCA (same as quantum model pipeline)...")

n_components = 4  # Same as run_quantum_fast.py

# Step 1: StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2: PCA with 4 components
pca = PCA(n_components=n_components, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"   Reduced from {X_train.shape[1]} features → {n_components} PCA components")
print(f"   Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"   Total explained variance: {pca.explained_variance_ratio_.sum():.4f} "
      f"({pca.explained_variance_ratio_.sum()*100:.1f}%)")

# Create DataFrames with PCA feature names for clarity
pca_feature_names = [f'PC{i+1}' for i in range(n_components)]
X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca_feature_names)
X_test_pca_df = pd.DataFrame(X_test_pca, columns=pca_feature_names)

# ============================================================================
# SECTION 3: MODEL DEFINITIONS
# ============================================================================

print("\n🔧 Setting up models (same hyperparameters as original)...")

# Cross-validation setup
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = ['accuracy', 'recall', 'f1', 'roc_auc']

# Models (same hyperparameters as cleaned_notebook.py, but NO scaler in pipeline
# since data is already scaled via StandardScaler → PCA)
log_reg = LogisticRegression(max_iter=1000, random_state=42)

svm = SVC(kernel='rbf', probability=True, random_state=42)

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
# SECTION 5: TRAINING AND EVALUATION ON PCA FEATURES
# ============================================================================

print("\n🚀 Training classical models on 4 PCA features...")

# --- Cross-Validation ---
print("\n📊 Cross-Validation Results (10-fold):")
cv_log = cv_results(log_reg, X_train_pca, y_train)
cv_svm = cv_results(svm, X_train_pca, y_train)
cv_xgb = cv_results(xgb, X_train_pca, y_train)

cv_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'SVM', 'XGBoost'],
    'Accuracy': [cv_log['accuracy'], cv_svm['accuracy'], cv_xgb['accuracy']],
    'Recall': [cv_log['recall'], cv_svm['recall'], cv_xgb['recall']],
    'F1-Score': [cv_log['f1'], cv_svm['f1'], cv_xgb['f1']],
    'AUC-ROC': [cv_log['roc_auc'], cv_svm['roc_auc'], cv_xgb['roc_auc']],
    'Features': ['4 PCA'] * 3
})
print(cv_df.to_string(index=False))

# --- Test Set Evaluation ---
print("\n📊 Test Set Results:")
test_log = test_results(log_reg, X_train_pca, y_train, X_test_pca, y_test)
test_svm = test_results(svm, X_train_pca, y_train, X_test_pca, y_test)
test_xgb = test_results(xgb, X_train_pca, y_train, X_test_pca, y_test)

test_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'SVM', 'XGBoost'],
    'Accuracy': [test_log['Accuracy'], test_svm['Accuracy'], test_xgb['Accuracy']],
    'Recall': [test_log['Recall'], test_svm['Recall'], test_xgb['Recall']],
    'F1-Score': [test_log['F1'], test_svm['F1'], test_xgb['F1']],
    'AUC-ROC': [test_log['AUC'], test_svm['AUC'], test_xgb['AUC']],
    'Features': ['4 PCA'] * 3
})
print(test_df.to_string(index=False))

# ============================================================================
# SECTION 6: COMPARISON — PCA vs ALL FEATURES vs QUANTUM
# ============================================================================

print("\n" + "=" * 60)
print("📊 COMPREHENSIVE COMPARISON")
print("=" * 60)

# Original all-features results (from cleaned_notebook.py / export_tables.py)
# We re-train them here to get exact values
print("\n🔄 Re-training models on ALL features for comparison...")

log_reg_all = LogisticRegression(max_iter=1000, random_state=42)
svm_all = SVC(kernel='rbf', probability=True, random_state=42)
xgb_all = XGBClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42
)

# Scale all features for Logistic Regression and SVM
from sklearn.pipeline import Pipeline

log_reg_pipe = Pipeline([('scaler', StandardScaler()), ('model', log_reg_all)])
svm_pipe = Pipeline([('scaler', StandardScaler()), ('model', svm_all)])

test_log_all = test_results(log_reg_pipe, X_train, y_train, X_test, y_test)
test_svm_all = test_results(svm_pipe, X_train, y_train, X_test, y_test)
test_xgb_all = test_results(xgb_all, X_train, y_train, X_test, y_test)

# Quantum results (from run_quantum_fast.py results)
quantum_file = "exported_tables/quantum_models_actual_results.csv"
if os.path.exists(quantum_file):
    quantum_actual = pd.read_csv(quantum_file)
    qsvm_acc = quantum_actual['Accuracy'].values[0]
    qsvm_recall = quantum_actual['Recall'].values[0]
    qsvm_f1 = quantum_actual['F1-Score'].values[0]
    qsvm_auc = quantum_actual['AUC-ROC'].values[0]
    print(f"   ✅ Loaded actual QSVM results from {quantum_file}")
else:
    # Fallback to hardcoded values from original notebook
    qsvm_acc = 0.756579
    qsvm_recall = 0.973451
    qsvm_f1 = 0.856031
    qsvm_auc = 0.75
    print("   ⚠️ Using hardcoded QSVM results (no actual results file found)")

# Build comprehensive comparison table
comparison_data = []

# Classical on ALL features
for name, res in [('Logistic Regression', test_log_all),
                   ('SVM', test_svm_all),
                   ('XGBoost', test_xgb_all)]:
    comparison_data.append({
        'Model': name,
        'Feature_Set': f'All Features ({X_train.shape[1]})',
        'Accuracy': round(res['Accuracy'], 6),
        'Recall': round(res['Recall'], 6),
        'F1-Score': round(res['F1'], 6),
        'AUC-ROC': round(res['AUC'], 6)
    })

# Classical on PCA features
for name, res in [('Logistic Regression', test_log),
                   ('SVM', test_svm),
                   ('XGBoost', test_xgb)]:
    comparison_data.append({
        'Model': name,
        'Feature_Set': '4 PCA Components',
        'Accuracy': round(res['Accuracy'], 6),
        'Recall': round(res['Recall'], 6),
        'F1-Score': round(res['F1'], 6),
        'AUC-ROC': round(res['AUC'], 6)
    })

# Quantum on PCA features
comparison_data.append({
    'Model': 'QSVM (Quantum)',
    'Feature_Set': '4 PCA Components',
    'Accuracy': round(qsvm_acc, 6),
    'Recall': round(qsvm_recall, 6),
    'F1-Score': round(qsvm_f1, 6),
    'AUC-ROC': round(qsvm_auc, 6) if qsvm_auc is not None else 'N/A'
})

comparison_df = pd.DataFrame(comparison_data)
print("\n📋 Full Comparison Table:")
print(comparison_df.to_string(index=False))

# ============================================================================
# SECTION 7: FAIR COMPARISON — SAME FEATURES (PCA only)
# ============================================================================

print("\n" + "=" * 60)
print("🎯 FAIR COMPARISON (All models on 4 PCA features)")
print("=" * 60)

fair_data = []
for name, res in [('Logistic Regression', test_log),
                   ('SVM', test_svm),
                   ('XGBoost', test_xgb)]:
    fair_data.append({
        'Model': name,
        'Type': 'Classical',
        'Accuracy': round(res['Accuracy'], 6),
        'Recall': round(res['Recall'], 6),
        'F1-Score': round(res['F1'], 6),
        'AUC-ROC': round(res['AUC'], 6)
    })

fair_data.append({
    'Model': 'QSVM',
    'Type': 'Quantum',
    'Accuracy': round(qsvm_acc, 6),
    'Recall': round(qsvm_recall, 6),
    'F1-Score': round(qsvm_f1, 6),
    'AUC-ROC': round(qsvm_auc, 6) if qsvm_auc is not None else 'N/A'
})

fair_df = pd.DataFrame(fair_data).sort_values('F1-Score', ascending=False)
print(fair_df.to_string(index=False))

# ============================================================================
# SECTION 8: EXPORT RESULTS
# ============================================================================

print("\n💾 Exporting results...")

output_dir = "exported_tables"
os.makedirs(output_dir, exist_ok=True)

# 1. CV results on PCA features
cv_df.to_csv(os.path.join(output_dir, "pca_models_cv_results.csv"), index=False)
print(f"   ✅ {output_dir}/pca_models_cv_results.csv")

# 2. Test results on PCA features
test_df.to_csv(os.path.join(output_dir, "pca_models_test_results.csv"), index=False)
print(f"   ✅ {output_dir}/pca_models_test_results.csv")

# 3. Full comparison (PCA vs All Features vs Quantum)
comparison_df.to_csv(os.path.join(output_dir, "pca_vs_all_features_comparison.csv"), index=False)
print(f"   ✅ {output_dir}/pca_vs_all_features_comparison.csv")

# 4. Fair comparison (all on PCA)
fair_df.to_csv(os.path.join(output_dir, "fair_comparison_pca_only.csv"), index=False)
print(f"   ✅ {output_dir}/fair_comparison_pca_only.csv")

# ============================================================================
# SECTION 9: VISUALIZATION
# ============================================================================

print("\n📊 Generating comparison visualization...")

plt.style.use('seaborn-v0_8-whitegrid')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Comparison: Classical (PCA) vs Classical (All) vs Quantum (PCA)',
             fontsize=15, fontweight='bold', y=0.98)

metrics_list = ['Accuracy', 'Recall', 'F1-Score', 'AUC-ROC']

# Colors: All features = solid, PCA = hatched, Quantum = different color
colors_all = ['#2ecc71', '#3498db', '#e74c3c']       # LR, SVM, XGB (all features)
colors_pca = ['#27ae60', '#2980b9', '#c0392b']       # LR, SVM, XGB (PCA features)
color_quantum = '#9b59b6'                              # QSVM

model_names = ['LR\n(All)', 'SVM\n(All)', 'XGB\n(All)',
               'LR\n(PCA)', 'SVM\n(PCA)', 'XGB\n(PCA)',
               'QSVM\n(PCA)']

for ax, metric in zip(axes.flat, metrics_list):
    # Gather values
    values_all = [test_log_all[metric.replace('-', '_').replace('AUC_ROC', 'AUC').replace('F1_Score', 'F1').replace('AUC-ROC', 'AUC').replace('F1-Score', 'F1')],
                  test_svm_all[metric.replace('-', '_').replace('AUC_ROC', 'AUC').replace('F1_Score', 'F1').replace('AUC-ROC', 'AUC').replace('F1-Score', 'F1')],
                  test_xgb_all[metric.replace('-', '_').replace('AUC_ROC', 'AUC').replace('F1_Score', 'F1').replace('AUC-ROC', 'AUC').replace('F1-Score', 'F1')]]

    metric_key = metric.replace('AUC-ROC', 'AUC').replace('F1-Score', 'F1')
    values_pca = [test_log[metric_key], test_svm[metric_key], test_xgb[metric_key]]

    qsvm_val = {'Accuracy': qsvm_acc, 'Recall': qsvm_recall,
                'F1': qsvm_f1, 'AUC': qsvm_auc}
    values_quantum = [qsvm_val[metric_key]]

    all_values = values_all + values_pca + values_quantum
    all_colors = colors_all + colors_pca + [color_quantum]

    x_pos = np.arange(len(all_values))
    bars = ax.bar(x_pos, all_values, color=all_colors, edgecolor='black', linewidth=0.8)

    # Hatch pattern for PCA-trained classical models
    for i in range(3, 6):
        bars[i].set_hatch('//')

    # Different hatch for quantum
    bars[6].set_hatch('xx')

    # Value labels
    for bar, val in zip(bars, all_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(metric, fontweight='bold')
    ax.set_ylim(0, 1.15)

    # Separator lines
    ax.axvline(x=2.5, color='gray', linestyle=':', alpha=0.7)
    ax.axvline(x=5.5, color='gray', linestyle=':', alpha=0.7)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', edgecolor='black', label='Classical (All Features)'),
    Patch(facecolor='#2980b9', edgecolor='black', hatch='//', label='Classical (4 PCA)'),
    Patch(facecolor='#9b59b6', edgecolor='black', hatch='xx', label='Quantum (4 PCA)')
]
fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95), fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('viz_pca_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: viz_pca_model_comparison.png")

# ============================================================================
# SECTION 10: PCA COMPONENT ANALYSIS
# ============================================================================

print("\n" + "=" * 60)
print("🔍 PCA COMPONENT DETAILS")
print("=" * 60)

print(f"\n   Components: {n_components}")
print(f"   Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
print(f"\n   Individual component variance:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"     PC{i+1}: {var:.4f} ({var*100:.1f}%)")

# Top contributing features per PCA component
print(f"\n   Top 5 contributing original features per PCA component:")
feature_names = X.columns.tolist()
for i in range(n_components):
    component = pca.components_[i]
    top_indices = np.argsort(np.abs(component))[::-1][:5]
    print(f"\n     PC{i+1}:")
    for idx in top_indices:
        print(f"       {feature_names[idx]}: {component[idx]:.4f}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("📋 FINAL SUMMARY")
print("=" * 60)

print("\n🎯 Fair Comparison (All models using 4 PCA features):")
print(fair_df.to_string(index=False))

# Find best classical on PCA
best_classical_pca = max(
    [('Logistic Regression', test_log), ('SVM', test_svm), ('XGBoost', test_xgb)],
    key=lambda x: x[1]['F1']
)
print(f"\n🏆 Best Classical Model on PCA: {best_classical_pca[0]}")
print(f"   F1-Score: {best_classical_pca[1]['F1']:.4f}")
print(f"   vs QSVM F1-Score: {qsvm_f1:.4f}")

diff = best_classical_pca[1]['F1'] - qsvm_f1
if diff > 0:
    print(f"   → Classical outperforms Quantum by {diff:.4f} ({diff*100:.1f}%)")
elif diff < 0:
    print(f"   → Quantum outperforms Classical by {abs(diff):.4f} ({abs(diff)*100:.1f}%)")
else:
    print(f"   → Both models perform equally!")

print("\n📁 Exported Files:")
print(f"   1. {output_dir}/pca_models_cv_results.csv")
print(f"   2. {output_dir}/pca_models_test_results.csv")
print(f"   3. {output_dir}/pca_vs_all_features_comparison.csv")
print(f"   4. {output_dir}/fair_comparison_pca_only.csv")
print(f"   5. viz_pca_model_comparison.png")

print("\n" + "=" * 60)
print("✨ Complete!")
print("=" * 60)
