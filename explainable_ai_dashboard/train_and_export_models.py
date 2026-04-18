"""
Train and Export Models
=======================
Train the best models and export them to the models folder.
Also copies the dataset to the data folder.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import shutil

# Setup paths
project_root = Path(__file__).parent
models_dir = project_root / 'models'
data_dir = project_root / 'data'

# Create directories
models_dir.mkdir(exist_ok=True)
data_dir.mkdir(exist_ok=True)

print("=" * 60)
print("Training and Exporting Models")
print("=" * 60)

# 1. Load and prepare data
print("\n1. Loading dataset...")
data_source = project_root.parent / 'pd_speech_features.csv'

if not data_source.exists():
    print(f"Error: Dataset not found at {data_source}")
    print("Please ensure pd_speech_features.csv is in the parent directory")
    exit(1)

# Copy dataset to data folder
data_destination = data_dir / 'pd_speech_features.csv'
print(f"   Copying dataset to {data_destination}")
shutil.copy(data_source, data_destination)
print(f"   ✓ Dataset copied successfully!")

# Load data
df = pd.read_csv(data_destination)
print(f"   ✓ Loaded {len(df)} patient records with {df.shape[1]-1} features")

# Convert all columns to numeric, forcing errors to NaN
print("   Cleaning data...")
for col in df.columns:
    if col != 'class':
        df[col] = pd.to_numeric(df[col], errors='coerce')
print(f"   ✓ Data cleaned")

# 2. Preprocess data
print("\n2. Preprocessing data...")
X = df.drop(columns=['class'])
y = df['class']

# Handle missing values
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)

# Save imputer
imputer_path = models_dir / 'imputer.pkl'
with open(imputer_path, 'wb') as f:
    pickle.dump(imputer, f)
print(f"   ✓ Saved imputer to {imputer_path}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, stratify=y, random_state=42
)
print(f"   ✓ Train set: {len(X_train)} samples")
print(f"   ✓ Test set: {len(X_test)} samples")

# 3. Train models
print("\n3. Training models...")

models = {}

# XGBoost (Best performing model)
print("   Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42,
    verbosity=0
)
xgb.fit(X_train, y_train)
models['XGBoost'] = xgb

# Evaluate
train_score = xgb.score(X_train, y_train)
test_score = xgb.score(X_test, y_test)
cv_scores = cross_val_score(xgb, X_imputed, y, cv=5, scoring='accuracy')
print(f"      Train accuracy: {train_score:.4f}")
print(f"      Test accuracy: {test_score:.4f}")
print(f"      CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# SVM
print("   Training SVM...")
svm = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(kernel='rbf', probability=True, random_state=42))
])
svm.fit(X_train, y_train)
models['SVM'] = svm

train_score = svm.score(X_train, y_train)
test_score = svm.score(X_test, y_test)
print(f"      Train accuracy: {train_score:.4f}")
print(f"      Test accuracy: {test_score:.4f}")

# MLP
print("   Training MLP...")
mlp = Pipeline([
    ('scaler', StandardScaler()),
    ('model', MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=42,
        verbose=False
    ))
])
mlp.fit(X_train, y_train)
models['MLP'] = mlp

train_score = mlp.score(X_train, y_train)
test_score = mlp.score(X_test, y_test)
print(f"      Train accuracy: {train_score:.4f}")
print(f"      Test accuracy: {test_score:.4f}")

# 4. Save models
print("\n4. Saving models...")
for model_name, model in models.items():
    model_path = models_dir / f'{model_name.lower()}_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"   ✓ Saved {model_name} to {model_path}")

# 5. Save feature names
feature_names_path = models_dir / 'feature_names.pkl'
with open(feature_names_path, 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
print(f"   ✓ Saved feature names to {feature_names_path}")

# 6. Save background data for SHAP (sample)
background_sample = X_imputed.sample(min(100, len(X_imputed)), random_state=42)
background_path = data_dir / 'background_data.pkl'
with open(background_path, 'wb') as f:
    pickle.dump(background_sample, f)
print(f"   ✓ Saved background data to {background_path}")

# 7. Create model metadata
metadata = {
    'models': list(models.keys()),
    'n_features': len(X.columns),
    'feature_names': X.columns.tolist(),
    'n_train_samples': len(X_train),
    'n_test_samples': len(X_test),
    'best_model': 'XGBoost',
    'xgboost_test_accuracy': xgb.score(X_test, y_test),
    'svm_test_accuracy': svm.score(X_test, y_test),
    'mlp_test_accuracy': mlp.score(X_test, y_test)
}

metadata_path = models_dir / 'metadata.pkl'
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)
print(f"   ✓ Saved metadata to {metadata_path}")

print("\n" + "=" * 60)
print("✓ Model training and export completed successfully!")
print("=" * 60)
print(f"\nModels saved in: {models_dir}")
print(f"Data saved in: {data_dir}")
print(f"\nBest model: XGBoost (Test Accuracy: {metadata['xgboost_test_accuracy']:.4f})")
print("\nYou can now run the API with these pre-trained models!")
print("=" * 60)
