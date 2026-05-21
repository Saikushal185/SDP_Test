# Top-10 Feature PD Speech Model Lab

Copied standalone website for the reduced-feature experiment in
`C:\Users\saiku\OneDrive\Desktop\Projects\SDP_Test\New Training with 10 features`.

## What It Includes

- React + Vite frontend in `frontend/`
- FastAPI backend in `backend/`
- Only the five top-10 models: SVM, QSVM, VQC, XGBoost, and LogisticRegression
- Manual single-case prediction with the 10 selected numeric speech features
- Sample-row helper values for testing and demonstration
- Plain-language feature explanations beside each technical column name

## Run Locally

From `C:\Users\saiku\OneDrive\Desktop\Projects\SDP_Test\New Training with 10 features\website`:

```powershell
$env:PYTHONPATH="C:\Users\saiku\OneDrive\Desktop\Projects\SDP_Test\parkinson_feature_study\.python_packages;C:\Users\saiku\OneDrive\Desktop\Projects\SDP_Test\parkinson_feature_study;C:\Users\saiku\OneDrive\Desktop\Projects\SDP_Test\New Training with 10 features\website\backend"
python -m uvicorn app.main:app --app-dir .\backend --host 127.0.0.1 --port 8010
```

In a second terminal:

```powershell
cd "C:\Users\saiku\OneDrive\Desktop\Projects\SDP_Test\New Training with 10 features\website\frontend"
$env:VITE_API_URL="http://127.0.0.1:8010"
npm install
npm run dev -- --host 127.0.0.1 --port 5174
```

Open `http://127.0.0.1:5174`.

## Verification

```powershell
python -m pytest .\backend\tests -q

cd .\frontend
npm run build
```
