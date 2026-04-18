"""
Run Quantum Models for Parkinson's Disease Prediction
======================================================
This script trains and evaluates QSVM and VQC quantum models
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
print("Quantum Models for Parkinson's Disease Prediction")
print("=" * 60)

# Qiskit imports
print("\n📦 Importing Qiskit libraries...")
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_machine_learning.algorithms import QSVC, VQC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    print("✅ Qiskit libraries loaded successfully!")
except ImportError as e:
    print(f"❌ Error importing Qiskit: {e}")
    print("Please install: pip install qiskit qiskit-machine-learning qiskit-aer qiskit-algorithms")
    exit(1)

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
# DIMENSIONALITY REDUCTION FOR QUANTUM MODELS
# ============================================================================

print("\n🔧 Performing dimensionality reduction...")
print("   (Quantum models work best with reduced feature space)")

# Use PCA to reduce to manageable number of features for quantum
n_components = 10  # Quantum models are expensive, use fewer features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=n_components, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"   Reduced to {n_components} principal components")
print(f"   Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

# Normalize to [0, 1] for quantum encoding
min_max_scaler = MinMaxScaler()
X_train_quantum = min_max_scaler.fit_transform(X_train_pca)
X_test_quantum = min_max_scaler.transform(X_test_pca)

# ============================================================================
# QUANTUM SVM (QSVM)
# ============================================================================

print("\n" + "=" * 60)
print("🔮 Training QSVM (Quantum Support Vector Machine)")
print("=" * 60)

try:
    # Create quantum feature map
    feature_map = ZZFeatureMap(feature_dimension=n_components, reps=2)
    
    # Create quantum kernel
    print("\n⚙️ Setting up quantum kernel...")
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    
    # Create QSVC
    print("🚀 Training QSVM...")
    qsvm = QSVC(quantum_kernel=quantum_kernel)
    
    # Train on subset for faster execution (quantum is slow)
    # Use first 200 samples for training
    train_subset_size = min(200, len(X_train_quantum))
    X_train_subset = X_train_quantum[:train_subset_size]
    y_train_subset = y_train.iloc[:train_subset_size]
    
    print(f"   Using {train_subset_size} training samples (for speed)")
    qsvm.fit(X_train_subset, y_train_subset)
    
    # Evaluate
    print("\n📊 Evaluating QSVM on test set...")
    y_pred_qsvm = qsvm.predict(X_test_quantum)
    
    # Get decision function for AUC calculation
    y_score_qsvm = qsvm.decision_function(X_test_quantum)
    
    # Calculate metrics
    qsvm_accuracy = accuracy_score(y_test, y_pred_qsvm)
    qsvm_recall = recall_score(y_test, y_pred_qsvm)
    qsvm_f1 = f1_score(y_test, y_pred_qsvm)
    qsvm_auc = roc_auc_score(y_test, y_score_qsvm)
    
    print(f"\n✅ QSVM Results:")
    print(f"   Accuracy: {qsvm_accuracy:.6f}")
    print(f"   Recall:   {qsvm_recall:.6f}")
    print(f"   F1-Score: {qsvm_f1:.6f}")
    print(f"   AUC-ROC:  {qsvm_auc:.6f}")
    
    qsvm_success = True
    
except Exception as e:
    print(f"\n❌ Error training QSVM: {e}")
    qsvm_success = False
    qsvm_accuracy = qsvm_recall = qsvm_f1 = qsvm_auc = None

# ============================================================================
# VARIATIONAL QUANTUM CLASSIFIER (VQC)
# ============================================================================

print("\n" + "=" * 60)
print("🔮 Training VQC (Variational Quantum Classifier)")
print("=" * 60)

try:
    # Create feature map and ansatz
    feature_map_vqc = ZZFeatureMap(feature_dimension=n_components, reps=2)
    ansatz = RealAmplitudes(num_qubits=n_components, reps=3)
    
    # Create VQC without sampler (use default settings)
    print("\n⚙️ Setting up VQC...")
    vqc = VQC(
        feature_map=feature_map_vqc,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=50)  # Reduced iterations for speed
    )
    
    # Train on subset
    print("🚀 Training VQC...")
    print(f"   Using {train_subset_size} training samples (for speed)")
    vqc.fit(X_train_subset, y_train_subset)
    
    # Evaluate
    print("\n📊 Evaluating VQC on test set...")
    y_pred_vqc = vqc.predict(X_test_quantum)
    
    # Calculate metrics
    vqc_accuracy = accuracy_score(y_test, y_pred_vqc)
    vqc_recall = recall_score(y_test, y_pred_vqc)
    vqc_f1 = f1_score(y_test, y_pred_vqc)
    
    # Try to get score for AUC
    try:
        y_score_vqc = vqc.score(X_test_quantum, y_test)
        # Use predictions as proxy for AUC if we can't get probabilities
        vqc_auc = roc_auc_score(y_test, y_pred_vqc)
    except:
        vqc_auc = roc_auc_score(y_test, y_pred_vqc)
    
    print(f"\n✅ VQC Results:")
    print(f"   Accuracy: {vqc_accuracy:.6f}")
    print(f"   Recall:   {vqc_recall:.6f}")
    print(f"   F1-Score: {vqc_f1:.6f}")
    print(f"   AUC-ROC:  {vqc_auc:.6f}")
    
    vqc_success = True
    
except Exception as e:
    print(f"\n❌ Error training VQC: {e}")
    print(f"   Details: {str(e)}")
    vqc_success = False
    vqc_accuracy = vqc_recall = vqc_f1 = vqc_auc = None

# ============================================================================
# EXPORT RESULTS
# ============================================================================

print("\n" + "=" * 60)
print("💾 Exporting Results")
print("=" * 60)

# Create results dataframe
quantum_results = []

if qsvm_success:
    quantum_results.append({
        'Model': 'QSVM',
        'Accuracy': qsvm_accuracy,
        'Recall': qsvm_recall,
        'F1-Score': qsvm_f1,
        'AUC-ROC': qsvm_auc
    })

if vqc_success:
    quantum_results.append({
        'Model': 'VQC',
        'Accuracy': vqc_accuracy,
        'Recall': vqc_recall,
        'F1-Score': vqc_f1,
        'AUC-ROC': vqc_auc
    })

if quantum_results:
    quantum_df = pd.DataFrame(quantum_results)
    
    # Export to CSV
    import os
    output_dir = "exported_tables"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "quantum_models_actual_results.csv")
    quantum_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Results exported to: {output_file}")
    print("\n📊 Final Results:")
    print(quantum_df.to_string(index=False))
else:
    print("\n❌ No quantum models were successfully trained")

print("\n" + "=" * 60)
print("✨ Quantum Model Training Complete!")
print("=" * 60)

# Print comparison note
print("\n📝 Note:")
print("   - Quantum models were trained on reduced feature space (PCA)")
print(f"   - Training subset: {train_subset_size} samples (for computational efficiency)")
print("   - Full quantum computation can take hours; this is optimized for speed")
