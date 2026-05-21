import os
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.environ.get("ANFIS_MODEL_PATH", "anfis_model.joblib"))

try:
    _model = joblib.load(MODEL_PATH)
except FileNotFoundError as exc:
    raise RuntimeError(
        f"{MODEL_PATH} missing. Run `python train.py --csv telco.csv` first."
    ) from exc

centers = _model["centers"]
sigma = _model["sigma"]
consequents = _model["consequents"]
scaler = _model["scaler"]
encoded_features = _model["encoded_features"]


app = FastAPI(title="Hybrid ANFIS Churn Prediction API")

_cors_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


YesNo = Literal["Yes", "No"]


class CustomerFeatures(BaseModel):
    Age: int = Field(..., ge=0, le=120)
    Senior_Citizen: YesNo = Field(..., alias="Senior Citizen")
    Married: YesNo
    Dependents: YesNo
    Number_of_Dependents: int = Field(..., ge=0, le=20, alias="Number of Dependents")
    Tenure_in_Months: int = Field(..., ge=0, le=120, alias="Tenure in Months")
    Internet_Service: Literal["Fiber Optic", "DSL", "No"] = Field(..., alias="Internet Service")
    Phone_Service: YesNo = Field(..., alias="Phone Service")
    Multiple_Lines: YesNo = Field(..., alias="Multiple Lines")
    Unlimited_Data: YesNo = Field(..., alias="Unlimited Data")
    Monthly_Charge: float = Field(..., ge=0, le=1000, alias="Monthly Charge")
    Total_Charges: float = Field(..., ge=0, le=100_000, alias="Total Charges")
    Satisfaction_Score: int = Field(..., ge=1, le=5, alias="Satisfaction Score")

    model_config = {"populate_by_name": True}


class PredictRequest(BaseModel):
    data: CustomerFeatures


class PredictResponse(BaseModel):
    churn_probability: float
    churn_prediction: YesNo


@app.get("/")
def home():
    return {"status": "ANFIS API running"}


def anfis_predict(X: np.ndarray) -> np.ndarray:
    W = np.empty((X.shape[0], len(centers)))
    for r, c in enumerate(centers):
        W[:, r] = np.exp(-0.5 * np.sum(((X - c) / sigma) ** 2, axis=1))
    W = W / (np.sum(W, axis=1, keepdims=True) + 1e-8)
    return np.clip(W @ consequents, 0.0, 1.0)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    payload = request.data.model_dump(by_alias=True)
    df = pd.DataFrame([payload])
    df = pd.get_dummies(df)
    df = df.reindex(columns=encoded_features, fill_value=0)
    X_scaled = scaler.transform(df.values)
    prob = float(anfis_predict(X_scaled)[0])
    return PredictResponse(
        churn_probability=round(prob, 4),
        churn_prediction="Yes" if prob >= 0.5 else "No",
    )
