# PD Model Lab MVP

New standalone MVP website for the saved Parkinson speech-model artifacts in
`../parkinson_feature_study`.

## What It Includes

- React + Vite frontend in `frontend/`
- FastAPI backend in `backend/`
- Dataset/model selector formatted as `dataset-name_model-name`
- CSV upload plus sample-row edit prediction workflow
- Dashboard metrics, feature importance, group/class impact chart, and recent predictions
- VQC repair loader for JSON reconstruction bundles whose `.joblib` files are not usable

## Run Locally

From `C:\Users\saiku\OneDrive\Desktop\Projects\SDP_Test`:

```powershell
$env:PYTHONPATH="C:\Users\saiku\OneDrive\Desktop\Projects\SDP_Test\parkinson_feature_study\.python_packages;C:\Users\saiku\OneDrive\Desktop\Projects\SDP_Test\parkinson_feature_study;C:\Users\saiku\OneDrive\Desktop\Projects\SDP_Test\pd-model-lab-mvp\backend"
.\parkinson_feature_study\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir .\pd-model-lab-mvp\backend --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd .\pd-model-lab-mvp\frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

## Verification

```powershell
$env:PYTHONPATH="C:\Users\saiku\OneDrive\Desktop\Projects\SDP_Test\parkinson_feature_study\.python_packages;C:\Users\saiku\OneDrive\Desktop\Projects\SDP_Test\parkinson_feature_study;C:\Users\saiku\OneDrive\Desktop\Projects\SDP_Test\pd-model-lab-mvp\backend"
.\parkinson_feature_study\.venv\Scripts\python.exe -m pytest .\pd-model-lab-mvp\backend\tests -q

cd .\pd-model-lab-mvp\frontend
npm run build
```
