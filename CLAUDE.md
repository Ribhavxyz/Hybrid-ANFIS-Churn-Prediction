# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Full-stack telecom customer churn prediction app using a **Hybrid ANFIS (Adaptive Neuro-Fuzzy Inference System)** model. The model achieves ROC-AUC ≈ 0.93 on the IBM Telco Customer Churn dataset.

## Commands

### Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
```
- API docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/

### Frontend (React)
```bash
cd frontend
npm install
npm start        # dev server at http://localhost:3000
npm test         # run tests
npm run build    # production build
```

## Architecture

### ML Model (`backend/anfis_model.joblib`)
The model is pre-trained and loaded at startup. It stores:
- `centers` — Gaussian membership function centers (one per fuzzy rule)
- `sigma` — shared width for all Gaussian functions
- `consequents` — linear consequent parameters per rule
- `scaler` — StandardScaler fitted on training data
- `encoded_features` — ordered list of one-hot encoded column names from training

### ANFIS Inference (`backend/main.py:42`)
Inference is implemented directly in NumPy (no external ANFIS library):
1. Compute firing strength for each rule: `w_i = exp(-0.5 * sum(((x - c_i) / σ)^2))`
2. Normalize firing strengths
3. Output = weighted sum of consequents (`W @ consequents`)

### Prediction Pipeline (`backend/main.py:54`)
Input JSON `{ "data": { ...customer fields... } }` → DataFrame → `pd.get_dummies()` one-hot encode → `reindex` to align with `encoded_features` (fills missing columns with 0) → StandardScaler → ANFIS → returns `churn_probability` (float) and `churn_prediction` ("Yes"/"No").

### Frontend (`frontend/src/App.js`)
Single-page React app with a controlled form. On submit, POSTs to `http://127.0.0.1:8000/predict`. The backend URL is hardcoded — update it if the backend runs on a different host/port.

**Input features sent to the API:**
- Numeric: Age, Number of Dependents, Tenure in Months, Monthly Charge, Total Charges
- Categorical (one-hot encoded by backend): Senior Citizen, Married, Dependents, Internet Service, Phone Service, Multiple Lines, Unlimited Data, Satisfaction Score

## Known Issues (Historical)

- **Internet Service encoding mismatch:** The original model treated Internet Service as binary (`Internet Service_Yes` only), while the frontend dropdown sends three values (DSL / Fiber Optic / No). This caused all three options to silently produce identical predictions. Fixed by `backend/train.py`, which retrains the model with proper multi-category one-hot encoding.

## Key Constraints

- The model file `anfis_model.joblib` must be present in `backend/` at startup; the app will crash without it.
- Adding or renaming input features in the frontend requires retraining the model and regenerating `anfis_model.joblib` so `encoded_features` stays consistent.
- CORS is open (`allow_origins=["*"]`) — intended for local development only.
