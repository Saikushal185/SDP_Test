"""
Run Quantum Models - Fast Version
==================================
Optimized for quick execution using minimal features and samples
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Quantum Models - Fast Execution Version")
print("=" * 60)

# Qiskit imports
print("\n📦 Importing Qiskit libraries...")
try:
    from qiskit_machine_learning.algorithms import QSVC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit.circuit.library import ZZFeatureMap
    print("✅ Qiskit libraries loaded!")
except ImportError as e:
    print(f"❌ Error: {e}")
    exit(1)

# Load data
print("\n📊 Loading data...")
df = pd.read_csv("pd_speech_features.csv")

# Preprocessing
imputer = SimpleImputer(strategy='median')
df_numeric = df.apply(pd.to_numeric, errors='coerce')
df_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)

X = df_imputed.drop(columns=['class'])
y = df_imputed['class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Original: {len(X_train)} train, {len(X_test)} test")

# Aggressive dimensionality reduction for speed
print("\n🔧 Reducing dimensionality...")
n_components = 4 # Use only 4 features for speed
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=n_components, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Reduced to {n_components} components")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

# Normalize
min_max_scaler = MinMaxScaler()
X_train_quantum = min_max_scaler.fit_transform(X_train_pca)
X_test_quantum = min_max_scaler.transform(X_test_pca)

# Use small subset for training
train_size = 50  # Very small for fast execution
X_train_small = X_train_quantum[:train_size]
y_train_small = y_train.iloc[:train_size].reset_index(drop=True)

print(f"Using only {train_size} training samples for speed")

# ============================================================================
# QSVM
# ============================================================================

print("\n" + "=" * 60)
print("🔮 Training QSVM (Fast Mode)")
print("=" * 60)

try:
    feature_map = ZZFeatureMap(feature_dimension=n_components, reps=1)  # reps=1 for speed
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    
    print("🚀 Training QSVM...")
    qsvm = QSVC(quantum_kernel=quantum_kernel)
    qsvm.fit(X_train_small, y_train_small)
    
    print("📊 Evaluating...")
    y_pred = qsvm.predict(X_test_quantum)
    y_score = qsvm.decision_function(X_test_quantum)
    
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)
    
    print(f"\n✅ QSVM Results:")
    print(f"   Accuracy: {accuracy:.6f}")
    print(f"   Recall:   {recall:.6f}")
    print(f"   F1-Score: {f1:.6f}")
    print(f"   AUC-ROC:  {auc:.6f}")
    
    success = True
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    success = False
    accuracy = recall = f1 = auc = None

# Export results
if success:
    import os
    os.makedirs("exported_tables", exist_ok=True)
    
    result_df = pd.DataFrame([{
        'Model': 'QSVM',
        'Accuracy': accuracy,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc,
        'Note': f'{train_size} training samples, {n_components} PCA components'
    }])
    
    output_file = "exported_tables/quantum_models_actual_results.csv"
    result_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Exported to: {output_file}")
    print("\n📊 Summary:")
    print(result_df.to_string(index=False))
else:
    print("\n❌ Training failed")

print("\n" + "=" * 60)
print("✨ Complete!")
print("=" * 60)
print("\n📝 Note: Used minimal configuration for fast execution")
print(f"   - Features: {n_components} (via PCA)")
print(f"   - Training samples: {train_size}")
print("   - For full accuracy, increase these parameters")
