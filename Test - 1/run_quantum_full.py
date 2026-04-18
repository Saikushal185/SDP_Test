"""
Run Full Quantum Models - Complete Training
============================================
Uses all training samples for maximum accuracy
This will take significant time (30+ minutes expected)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
import time
import os
warnings.filterwarnings('ignore')

print("=" * 70)
print("FULL QUANTUM MODEL TRAINING")
print("=" * 70)
print("\n⚠️  This will use ALL training samples and may take 30+ minutes")
print("    Please be patient while the quantum circuits are simulated...")
print("\n" + "=" * 70)

# Qiskit imports
print("\n📦 Importing Qiskit libraries...")
try:
    from qiskit_machine_learning.algorithms import QSVC, VQC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    print("✅ Qiskit libraries loaded successfully!")
except ImportError as e:
    print(f"❌ Error: {e}")
    exit(1)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

print("\n📊 Loading and preprocessing data...")
df = pd.read_csv("pd_speech_features.csv")
print(f"   Dataset shape: {df.shape}")

# Handle missing values
imputer = SimpleImputer(strategy='median')
df_numeric = df.apply(pd.to_numeric, errors='coerce')
df_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)

X = df_imputed.drop(columns=['class'])
y = df_imputed['class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")
print(f"   Class distribution train: {y_train.value_counts().to_dict()}")

# ============================================================================
# DIMENSIONALITY REDUCTION
# ============================================================================

print("\n🔧 Performing dimensionality reduction...")
# Use 6 components - balance between accuracy and speed
n_components = 6
print(f"   Target: {n_components} principal components")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=n_components, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"   ✅ Reduced to {n_components} components")
print(f"   Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

# Normalize to [0, 1] for quantum encoding
min_max_scaler = MinMaxScaler()
X_train_quantum = min_max_scaler.fit_transform(X_train_pca)
X_test_quantum = min_max_scaler.transform(X_test_pca)

print(f"\n   Final quantum feature space: {X_train_quantum.shape}")
print(f"   Using ALL {len(X_train_quantum)} training samples")

# Create output directory
output_dir = "exported_tables"
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# QSVM - FULL TRAINING
# ============================================================================

print("\n" + "=" * 70)
print("🔮 QSVM (Quantum Support Vector Machine) - FULL TRAINING")
print("=" * 70)

qsvm_start_time = time.time()

try:
    # Create quantum feature map
    print("\n⚙️  Setting up quantum feature map...")
    feature_map = ZZFeatureMap(feature_dimension=n_components, reps=2)
    print(f"   Feature map: ZZFeatureMap with {n_components} qubits, 2 reps")
    
    # Create quantum kernel
    print("⚙️  Creating quantum kernel...")
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    
    # Create QSVC
    print("⚙️  Initializing QSVC...")
    qsvm = QSVC(quantum_kernel=quantum_kernel)
    
    # Train
    print(f"\n🚀 Training QSVM on {len(X_train_quantum)} samples...")
    print("   ⏰ This may take 15-30 minutes depending on your hardware...")
    print("   ⏰ Started at:", time.strftime("%H:%M:%S"))
    
    qsvm.fit(X_train_quantum, y_train)
    
    training_time = time.time() - qsvm_start_time
    print(f"\n   ✅ Training completed in {training_time/60:.2f} minutes")
    
    # Evaluate
    print("\n📊 Evaluating QSVM on test set...")
    eval_start = time.time()
    
    y_pred_qsvm = qsvm.predict(X_test_quantum)
    y_score_qsvm = qsvm.decision_function(X_test_quantum)
    
    eval_time = time.time() - eval_start
    print(f"   Evaluation time: {eval_time:.2f} seconds")
    
    # Calculate metrics
    qsvm_accuracy = accuracy_score(y_test, y_pred_qsvm)
    qsvm_recall = recall_score(y_test, y_pred_qsvm)
    qsvm_f1 = f1_score(y_test, y_pred_qsvm)
    qsvm_auc = roc_auc_score(y_test, y_score_qsvm)
    
    # Confusion matrix
    cm_qsvm = confusion_matrix(y_test, y_pred_qsvm)
    
    print(f"\n{'='*70}")
    print("✅ QSVM RESULTS (Full Training)")
    print(f"{'='*70}")
    print(f"   Accuracy:  {qsvm_accuracy:.6f} ({qsvm_accuracy*100:.2f}%)")
    print(f"   Recall:    {qsvm_recall:.6f} ({qsvm_recall*100:.2f}%)")
    print(f"   F1-Score:  {qsvm_f1:.6f} ({qsvm_f1*100:.2f}%)")
    print(f"   AUC-ROC:   {qsvm_auc:.6f} ({qsvm_auc*100:.2f}%)")
    print(f"\n   Confusion Matrix:")
    print(f"   {cm_qsvm}")
    print(f"   Training time: {training_time/60:.2f} minutes")
    print(f"{'='*70}")
    
    qsvm_success = True
    
    # Save intermediate results
    qsvm_results = pd.DataFrame([{
        'Model': 'QSVM',
        'Accuracy': qsvm_accuracy,
        'Recall': qsvm_recall,
        'F1-Score': qsvm_f1,
        'AUC-ROC': qsvm_auc,
        'Training_Samples': len(X_train_quantum),
        'PCA_Components': n_components,
        'Training_Time_Minutes': round(training_time/60, 2)
    }])
    qsvm_results.to_csv(os.path.join(output_dir, "qsvm_full_results.csv"), index=False)
    print(f"\n💾 QSVM results saved to: {output_dir}/qsvm_full_results.csv")
    
except Exception as e:
    print(f"\n❌ Error training QSVM: {e}")
    import traceback
    traceback.print_exc()
    qsvm_success = False
    qsvm_accuracy = qsvm_recall = qsvm_f1 = qsvm_auc = None

# ============================================================================
# VQC - FULL TRAINING
# ============================================================================

print("\n" + "=" * 70)
print("🔮 VQC (Variational Quantum Classifier) - FULL TRAINING")
print("=" * 70)

vqc_start_time = time.time()

try:
    # Create feature map and ansatz
    print("\n⚙️  Setting up VQC architecture...")
    feature_map_vqc = ZZFeatureMap(feature_dimension=n_components, reps=2)
    ansatz = RealAmplitudes(num_qubits=n_components, reps=3)
    print(f"   Feature map: ZZFeatureMap with {n_components} qubits, 2 reps")
    print(f"   Ansatz: RealAmplitudes with {n_components} qubits, 3 reps")
    
    # Create VQC with COBYLA optimizer
    print("⚙️  Initializing VQC with COBYLA optimizer...")
    vqc = VQC(
        feature_map=feature_map_vqc,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=200)  # More iterations for better convergence
    )
    
    # Train
    print(f"\n🚀 Training VQC on {len(X_train_quantum)} samples...")
    print("   ⏰ This may take 20-40 minutes depending on your hardware...")
    print("   ⏰ Started at:", time.strftime("%H:%M:%S"))
    
    vqc.fit(X_train_quantum, y_train)
    
    training_time = time.time() - vqc_start_time
    print(f"\n   ✅ Training completed in {training_time/60:.2f} minutes")
    
    # Evaluate
    print("\n📊 Evaluating VQC on test set...")
    eval_start = time.time()
    
    y_pred_vqc = vqc.predict(X_test_quantum)
    
    eval_time = time.time() - eval_start
    print(f"   Evaluation time: {eval_time:.2f} seconds")
    
    # Calculate metrics
    vqc_accuracy = accuracy_score(y_test, y_pred_vqc)
    vqc_recall = recall_score(y_test, y_pred_vqc)
    vqc_f1 = f1_score(y_test, y_pred_vqc)
    
    # Try to get AUC score
    try:
        # VQC may not have decision_function, use predictions
        vqc_auc = roc_auc_score(y_test, y_pred_vqc)
    except:
        vqc_auc = None
        print("   ⚠️  Could not calculate AUC (probability scores not available)")
    
    # Confusion matrix
    cm_vqc = confusion_matrix(y_test, y_pred_vqc)
    
    print(f"\n{'='*70}")
    print("✅ VQC RESULTS (Full Training)")
    print(f"{'='*70}")
    print(f"   Accuracy:  {vqc_accuracy:.6f} ({vqc_accuracy*100:.2f}%)")
    print(f"   Recall:    {vqc_recall:.6f} ({vqc_recall*100:.2f}%)")
    print(f"   F1-Score:  {vqc_f1:.6f} ({vqc_f1*100:.2f}%)")
    if vqc_auc is not None:
        print(f"   AUC-ROC:   {vqc_auc:.6f} ({vqc_auc*100:.2f}%)")
    else:
        print(f"   AUC-ROC:   N/A (using binary predictions only)")
    print(f"\n   Confusion Matrix:")
    print(f"   {cm_vqc}")
    print(f"   Training time: {training_time/60:.2f} minutes")
    print(f"{'='*70}")
    
    vqc_success = True
    
    # Save intermediate results
    vqc_results = pd.DataFrame([{
        'Model': 'VQC',
        'Accuracy': vqc_accuracy,
        'Recall': vqc_recall,
        'F1-Score': vqc_f1,
        'AUC-ROC': vqc_auc if vqc_auc is not None else 'N/A',
        'Training_Samples': len(X_train_quantum),
        'PCA_Components': n_components,
        'Training_Time_Minutes': round(training_time/60, 2)
    }])
    vqc_results.to_csv(os.path.join(output_dir, "vqc_full_results.csv"), index=False)
    print(f"\n💾 VQC results saved to: {output_dir}/vqc_full_results.csv")
    
except Exception as e:
    print(f"\n❌ Error training VQC: {e}")
    import traceback
    traceback.print_exc()
    vqc_success = False
    vqc_accuracy = vqc_recall = vqc_f1 = vqc_auc = None

# ============================================================================
# FINAL SUMMARY AND EXPORT
# ============================================================================

print("\n" + "=" * 70)
print("💾 EXPORTING COMBINED RESULTS")
print("=" * 70)

quantum_results = []

if qsvm_success:
    quantum_results.append({
        'Model': 'QSVM',
        'Accuracy': qsvm_accuracy,
        'Recall': qsvm_recall,
        'F1-Score': qsvm_f1,
        'AUC-ROC': qsvm_auc,
        'Training_Samples': len(X_train_quantum),
        'PCA_Components': n_components
    })

if vqc_success:
    quantum_results.append({
        'Model': 'VQC',
        'Accuracy': vqc_accuracy,
        'Recall': vqc_recall,
        'F1-Score': vqc_f1,
        'AUC-ROC': vqc_auc if vqc_auc is not None else 'N/A',
        'Training_Samples': len(X_train_quantum),
        'PCA_Components': n_components
    })

if quantum_results:
    quantum_df = pd.DataFrame(quantum_results)
    
    # Save to CSV
    output_file = os.path.join(output_dir, "quantum_models_full_results.csv")
    quantum_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Combined results exported to: {output_file}")
    print("\n📊 FINAL QUANTUM MODELS COMPARISON:")
    print("=" * 70)
    print(quantum_df.to_string(index=False))
    print("=" * 70)
else:
    print("\n❌ No quantum models were successfully trained")

# Total time
total_time = time.time() - qsvm_start_time
print(f"\n⏱️  Total execution time: {total_time/60:.2f} minutes")
print(f"   Completed at: {time.strftime('%H:%M:%S')}")

print("\n" + "=" * 70)
print("✨ QUANTUM MODEL TRAINING COMPLETE!")
print("=" * 70)

print("\n📋 Summary:")
print(f"   - Training samples used: {len(X_train_quantum)}")
print(f"   - Test samples: {len(X_test_quantum)}")
print(f"   - Features (PCA): {n_components} components")
print(f"   - Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
print(f"   - Models trained: {len(quantum_results)}")

# Comparison with classical models (approximate)
print("\n📊 Quick Comparison with Classical Models:")
print("   Classical XGBoost: ~87.5% accuracy, ~91.9% F1-score")
if qsvm_success:
    print(f"   QSVM (Quantum):    {qsvm_accuracy*100:.1f}% accuracy, {qsvm_f1*100:.1f}% F1-score")
if vqc_success:
    print(f"   VQC (Quantum):     {vqc_accuracy*100:.1f}% accuracy, {vqc_f1*100:.1f}% F1-score")

print("\n" + "=" * 70)
