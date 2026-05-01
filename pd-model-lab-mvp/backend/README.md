# PD Model Lab Backend

FastAPI backend for serving the saved Parkinson speech-model artifacts in
`../parkinson_feature_study`.

Run from repository root:

```powershell
$env:PYTHONPATH="C:\Users\saiku\OneDrive\Desktop\Projects\SDP_Test\parkinson_feature_study\.python_packages;C:\Users\saiku\OneDrive\Desktop\Projects\SDP_Test\parkinson_feature_study;C:\Users\saiku\OneDrive\Desktop\Projects\SDP_Test\pd-model-lab-mvp\backend"
python -m uvicorn app.main:app --app-dir .\pd-model-lab-mvp\backend --reload --port 8000
```
