# Explainable AI Dashboard for Parkinson's Disease

A comprehensive web-based dashboard for Parkinson's disease prediction with explainable AI features, SHAP interpretability, and clinical risk assessments.

![Dashboard Preview](../uploaded_media_1770113149485.png)

## Features

- **Patient Diagnosis Interface**: Primary diagnosis with probability scores and decision basis
- **Non-Motor Risk Assessment**: Cognitive decline, depression, and dysphagia risk estimation
- **Motor Speech Analysis**: Detailed speech feature analysis with jitter, shimmer, and pitch metrics
- **Model Explanations**: Classical and quantum model contributions with SHAP values
- **Overall Risk Assessment**: Severity gauge with disease stage and progression risk
- **Clinical Recommendations**: Evidence-based recommendations for patient care

## Architecture

### Backend (Python/Flask)
- `backend/explainable_ai_api.py` - Flask REST API
- `backend/shap_explainer.py` - SHAP explainability module
- `backend/risk_calculator.py` - Clinical risk assessment

### Frontend (HTML/CSS/JavaScript)
- `frontend/explainable_ai_dashboard.html` - Main dashboard interface
- `frontend/dashboard.css` - Premium medical-grade styling
- `frontend/dashboard.js` - Interactive data visualization

## Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Ensure you have the dataset:**
   - The API expects `pd_speech_features.csv` in the parent directory
   - Or it will create dummy data for testing

## Usage

### 1. Start the Backend API

```bash
cd backend
python explainable_ai_api.py
```

The API will start on `http://localhost:5000`

### 2. Access the Dashboard

**Option A: Through Flask (Recommended)**
- Navigate to: `http://localhost:5000/dashboard`

**Option B: Open HTML directly**
- Open `frontend/explainable_ai_dashboard.html` in your browser
- Note: API must still be running for data fetching

### 3. Interact with the Dashboard

- Select different patients from the dropdown menu
- View comprehensive diagnosis and risk assessments
- Explore model explanations and feature importance
- Review clinical recommendations

## API Endpoints

- `GET /api/patient/<id>` - Get patient diagnosis with full explanations
- `GET /api/models/performance` - Get model performance metrics
- `POST /api/predict` - Make prediction for new patient data
- `GET /api/patients/list` - List available patients
- `GET /dashboard` - Serve dashboard HTML

## Technical Details

### Models Used
- **XGBoost** - Primary classical model
- **SVM** - Support Vector Machine with RBF kernel
- **MLP** - Multi-layer Perceptron neural network
- **QSVM** - Quantum SVM (simulated results)
- **VQC** - Variational Quantum Classifier (simulated results)

### Explainability Methods
- **SHAP (SHapley Additive exPlanations)** - Feature importance and contribution analysis
- **Risk Scoring** - Clinical severity assessment
- **Feature Analysis** - Speech pattern interpretation

### Design
- Modern medical-grade UI with glassmorphism effects
- Responsive layout for different screen sizes
- Smooth animations and transitions
- Interactive severity gauge visualization

## Customization

### Backend
- Modify risk thresholds in `risk_calculator.py`
- Add new models in `explainable_ai_api.py`
- Customize SHAP explainers in `shap_explainer.py`

### Frontend
- Update color schemes in `dashboard.css` (:root variables)
- Modify layout in `dashboard.html`
- Enhance visualizations in `dashboard.js`

## Troubleshooting

**API Connection Error:**
- Ensure Flask server is running on port 5000
- Check for firewall blocking localhost connections
- Verify CORS is enabled in `explainable_ai_api.py`

**SHAP Calculation Slow:**
- SHAP explanations can take time for complex models
- Reduce background sample size in `shap_explainer.py`
- Use pre-computed explanations for frequent queries

**Model Loading Issues:**
- Ensure dataset path is correct
- Check scikit-learn and xgboost versions
- Verify all dependencies are installed

## Future Enhancements

- [ ] Real-time patient data integration
- [ ] Historical trend analysis
- [ ] Export reports to PDF
- [ ] Multi-language support
- [ ] Mobile application
- [ ] Cloud deployment with authentication

## License

This project is for educational and research purposes.

## Citation

If you use this dashboard in your research, please cite:
```
Explainable AI Dashboard for Parkinson's Disease Prediction
XAI-based clinical decision support system
```

## Contact

For questions or support, please open an issue in the repository.
