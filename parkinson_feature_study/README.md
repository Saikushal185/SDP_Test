# Parkinson's Disease Feature-Centric Comparative Framework

A research-grade Python framework for studying how feature selection strategies affect model performance across classical and quantum-inspired learning paradigms for Parkinson's disease prediction from speech features.

## 🎯 Research Objective

This framework implements a **2×2 cross-paradigm testing design** to compare:
- **Feature Selection Methods**: Classical (Random Forest) vs Quantum-Inspired (QIGA)
- **Learning Models**: Classical (XGBoost, MLP) vs Quantum-Inspired (QNN)

## 📁 Project Structure

```
parkinson_feature_study/
├── config.yaml                 # All hyperparameters
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/
│   ├── raw/                    # Original dataset (place pd_speech_features.csv here)
│   └── processed/              # Preprocessed data
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Data loading and preprocessing
│   ├── training.py             # Cross-validation training loop
│   ├── evaluation.py           # Metrics and statistical tests
│   ├── interpretability.py     # Risk score generation
│   ├── feature_selection/
│   │   ├── __init__.py
│   │   ├── classical.py        # Random Forest feature selection
│   │   └── quantum_inspired.py # QIGA feature selection
│   └── models/
│       ├── __init__.py
│       ├── classical.py        # XGBoost and MLP
│       └── quantum_inspired.py # QNN simulator
├── experiments/
│   └── run_experiment.py       # Main experiment runner
├── results/
│   ├── metrics/                # CSV files with results
│   ├── figures/                # Plots and visualizations
│   └── selected_features/      # Feature lists from each method
└── notebooks/
    └── analysis.ipynb          # Results exploration
```

## 🚀 Installation

```bash
# Clone or navigate to the project directory
cd parkinson_feature_study

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset

Place the `pd_speech_features.csv` file in the `data/raw/` directory.

**Dataset characteristics:**
- 757 samples (healthy controls and Parkinson's patients)
- 754 acoustic/speech features
- Binary classification target (0=Healthy, 1=Parkinson's)

## ⚙️ Configuration

All hyperparameters are centralized in `config.yaml`:
- Cross-validation settings (k-folds, random seed)
- Feature selection parameters
- Model hyperparameters
- Evaluation metrics

## 🔬 Running Experiments

```bash
# Run full experiment
python experiments/run_experiment.py

# Run with custom config
python experiments/run_experiment.py --config path/to/custom_config.yaml

# Dry run (reduced parameters for testing)
python experiments/run_experiment.py --dry-run
```

## 📈 Expected Outputs

### Metrics (`results/metrics/`)
- `cross_validation_results.csv` - Per-fold metrics for all method combinations
- `aggregated_results.csv` - Mean ± std across folds
- `statistical_tests.csv` - Paired t-test results

### Figures (`results/figures/`)
- `roc_curves.png` - ROC curves for all models
- `performance_comparison.png` - Bar plots comparing methods
- `feature_overlap.png` - Venn diagram of selected features

### Selected Features (`results/selected_features/`)
- `classical_features_fold_*.csv` - Features selected by Random Forest
- `quantum_features_fold_*.csv` - Features selected by QIGA
- `feature_stability.csv` - Jaccard similarity analysis

## ⚠️ Important Notes

1. **No Data Leakage**: Feature selection occurs inside each CV fold
2. **Quantum-Inspired ≠ Quantum Computing**: QIGA and QNN are classical simulations
3. **Not Medical Diagnosis**: Outputs are "Parkinson's likelihood" scores, not clinical diagnoses

## 📚 References

- Quantum-Inspired Genetic Algorithm (QIGA): Han & Kim (2002)
- Variational Quantum Circuits: Cerezo et al. (2021)
- PennyLane: Bergholm et al. (2018)

## 📄 License

This project is for research purposes only.
