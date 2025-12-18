# Hybrid Neuro-Fuzzy Customer Churn Prediction System

## Overview
A full-stack machine learning application that predicts telecom customer churn
using a **Hybrid ANFIS (Adaptive Neuro-Fuzzy Inference System)** model.
The system provides real-time churn probability with interpretable fuzzy inference.

## Tech Stack
- **Machine Learning:** ANFIS (Neuro-Fuzzy), NumPy, Scikit-learn
- **Backend:** FastAPI
- **Frontend:** React (Dark Theme UI)
- **Dataset:** IBM Telco Customer Churn

## Key Features
- Hybrid Neuro-Fuzzy model with Gaussian membership functions
- Strong performance (ROC-AUC ≈ 0.93)
- FastAPI backend with safe model parameter loading
- React frontend with click-based inputs
- End-to-end real-time predictions

## How to Run

### Backend
```bash
cd backend
python -m uvicorn main:app --reload
```
### Frontend
```bash
cd frontend
npm start
```
### Open:
Backend Docs: http://127.0.0.1:8000/docs
Frontend UI: http://localhost:3000

### Output:
Churn Probability (0–1)
Churn Prediction (Yes / No)

### Author: Ribhav Yadav