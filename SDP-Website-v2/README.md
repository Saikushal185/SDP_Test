# SDP Parkinson Website V2

V2 is a separate rebuild of the Parkinson speech-feature website. The original
`SDP-Website` folder is left untouched.

## What Changed

- Models are presented through `backend/model_registry.json`, so display names
  can be changed without renaming trained artifact files.
- Active inference models are constrained to `pd_speech_features.csv`.
- `/api/batch-evaluate` performs strict external CSV testing and blocks datasets
  that do not contain every required deployed feature.
- `/api/dataset-template` returns a compatible CSV header/sample row.
- `/api/cross-dataset-summary` shows existing artifact context while marking
  incompatible schemas as blocked for direct testing.
- The frontend has a new clinical voice-lab workflow: Overview, Analysis,
  Dataset Test, Prediction, Explainability, and Performance.

## Run Locally

Backend:

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`. The frontend expects the Flask API at
`http://localhost:5000` unless `NEXT_PUBLIC_API_URL` is set.

## Rename Model Display Labels

Edit `backend/model_registry.json`:

```json
{
  "key": "xgboost",
  "display_name": "My XGBoost Label",
  "enabled": true
}
```

Keep `key` stable. It maps to the saved model artifact used by the backend.

## Verify

```bash
python -m unittest discover -s backend/tests -p "test_*.py"
cd frontend
npm run lint
npm run build
```
