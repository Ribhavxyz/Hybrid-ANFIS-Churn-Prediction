from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import joblib

# -----------------------------
# Load model parameters
# -----------------------------
data = joblib.load("anfis_model.joblib")

centers = data["centers"]
sigma = data["sigma"]
consequents = data["consequents"]
scaler = data["scaler"]
encoded_features = data["encoded_features"]

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Hybrid ANFIS Churn Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CustomerData(BaseModel):
    data: dict

@app.get("/")
def home():
    return {"status": "ANFIS API running"}

# -----------------------------
# ANFIS inference
# -----------------------------
def anfis_predict(X):
    W = []
    for c in centers:
        w = np.exp(-0.5 * np.sum(((X - c) / sigma) ** 2, axis=1))
        W.append(w)
    W = np.array(W).T
    W = W / (np.sum(W, axis=1, keepdims=True) + 1e-8)
    return W @ consequents

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(customer: CustomerData):

    # Convert input dict → DataFrame
    df = pd.DataFrame([customer.data])

    # One-hot encode input
    df = pd.get_dummies(df)

    # Align with training columns
    df = df.reindex(columns=encoded_features, fill_value=0)

    # Scale
    X_scaled = scaler.transform(df.values)

    # Predict
    prob = float(anfis_predict(X_scaled)[0])
    label = "Yes" if prob >= 0.5 else "No"

    return {
        "churn_probability": round(prob, 4),
        "churn_prediction": label
    }
