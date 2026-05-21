"""
Train the Hybrid ANFIS Churn Prediction model.

Usage:
    python train.py --csv path/to/telco_churn.csv

Outputs:
    anfis_model.joblib     -- model artifact loaded by main.py
    figures/confusion_matrix.png
    figures/roc_curve.png
    figures/feature_importance.png
    metrics.json           -- accuracy, roc_auc, precision, recall, f1

This script targets the IBM Telco Customer Churn dataset (the IBM Cognos
version with a Satisfaction Score column).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Feature configuration
# -----------------------------
NUMERIC_FEATURES = [
    "Age",
    "Number of Dependents",
    "Tenure in Months",
    "Monthly Charge",
    "Total Charges",
    "Satisfaction Score",
]

BINARY_FEATURES = [
    "Senior Citizen",
    "Married",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Unlimited Data",
]

MULTI_CATEGORY_FEATURES = [
    "Internet Service",  # values: "DSL" | "Fiber Optic" | "No"
]

TARGET = "Churn Label"


# -----------------------------
# ANFIS (simplified Takagi-Sugeno with constant consequents)
# -----------------------------
class ANFIS:
    """Simplified Takagi-Sugeno ANFIS with Gaussian membership functions.

    Architecture:
      - n_rules Gaussian RBF centers in feature space
      - Shared sigma vector across rules
      - Each rule emits a constant consequent
      - Output = sum_r (w_r * c_r), where w is normalized rule activation
    """

    def __init__(self, n_rules: int = 20, random_state: int = 42):
        self.n_rules = n_rules
        self.random_state = random_state
        self.centers_ = None
        self.sigma_ = None
        self.consequents_ = None

    def _activations(self, X: np.ndarray) -> np.ndarray:
        # W has shape (n_samples, n_rules)
        W = np.empty((X.shape[0], self.n_rules))
        for r, c in enumerate(self.centers_):
            W[:, r] = np.exp(-0.5 * np.sum(((X - c) / self.sigma_) ** 2, axis=1))
        # Normalize per-sample
        W = W / (np.sum(W, axis=1, keepdims=True) + 1e-8)
        return W

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ANFIS":
        # 1. Initialize centers via k-means
        km = KMeans(
            n_clusters=self.n_rules,
            n_init=10,
            random_state=self.random_state,
        ).fit(X)
        self.centers_ = km.cluster_centers_

        # 2. Sigma: per-feature std plus a small constant for stability
        self.sigma_ = X.std(axis=0) + 1e-2

        # 3. Solve for consequents via regularized least squares
        W = self._activations(X)
        # Ridge term keeps consequents bounded
        lam = 1e-2
        A = W.T @ W + lam * np.eye(self.n_rules)
        b = W.T @ y
        self.consequents_ = np.linalg.solve(A, b)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        W = self._activations(X)
        return np.clip(W @ self.consequents_, 0.0, 1.0)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


# -----------------------------
# Data prep
# -----------------------------
def load_and_prepare(csv_path: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)

    # Normalize a few common column-name variants from different Telco CSVs
    rename_map = {
        "tenure": "Tenure in Months",
        "MonthlyCharges": "Monthly Charge",
        "TotalCharges": "Total Charges",
        "SeniorCitizen": "Senior Citizen",
        "Partner": "Married",
        "InternetService": "Internet Service",
        "PhoneService": "Phone Service",
        "MultipleLines": "Multiple Lines",
        "UnlimitedData": "Unlimited Data",
        "SatisfactionScore": "Satisfaction Score",
        "NumberOfDependents": "Number of Dependents",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Derive 3-category Internet Service from Internet Service (Yes/No) + Internet Type
    # The frontend dropdown sends "Fiber Optic" | "DSL" | "No", so we need matching categories.
    # Cable is mapped to "DSL" since the UI doesn't expose it.
    if "Internet Type" in df.columns and set(df["Internet Service"].dropna().unique()).issubset({"Yes", "No"}):
        type_map = {"Fiber Optic": "Fiber Optic", "DSL": "DSL", "Cable": "DSL", "None": "No"}
        df["Internet Service"] = df.apply(
            lambda r: type_map.get(r["Internet Type"], "No") if r["Internet Service"] == "Yes" else "No",
            axis=1,
        )

    # Total Charges is sometimes a string with blanks
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
        df["Total Charges"] = df["Total Charges"].fillna(df["Total Charges"].median())

    # Senior Citizen sometimes 0/1 instead of No/Yes
    if pd.api.types.is_numeric_dtype(df["Senior Citizen"]):
        df["Senior Citizen"] = df["Senior Citizen"].map({0: "No", 1: "Yes"})

    # Standardize "No internet service" / "No phone service" to "No"
    for col in ["Multiple Lines", "Unlimited Data"]:
        if col in df.columns:
            df[col] = df[col].replace(
                {"No internet service": "No", "No phone service": "No"}
            )

    # Target
    y = df[TARGET].map({"Yes": 1, "No": 0}).astype(int).values

    # Numeric block
    num = df[NUMERIC_FEATURES].astype(float)

    # Binary block -> _Yes columns (drop_first=True so each becomes 0/1)
    bin_df = pd.get_dummies(df[BINARY_FEATURES], drop_first=True).astype(int)

    # Multi-category block -> keep ALL dummies (this is the fix for Bug #1)
    multi_df = pd.get_dummies(df[MULTI_CATEGORY_FEATURES], drop_first=False).astype(int)

    X = pd.concat([num, bin_df, multi_df], axis=1)
    return X, y, list(X.columns)


# -----------------------------
# Plots
# -----------------------------
def save_plots(y_true, y_prob, feature_names, model, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=120)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ANFIS (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=120)
    plt.close()

    # Feature "importance": magnitude of center-weighted consequents per feature
    # Heuristic: how much each feature's variation across centers correlates
    # with consequent magnitude.
    importance = np.abs(model.centers_ - model.centers_.mean(axis=0)).mean(axis=0)
    importance = importance * np.abs(model.consequents_).mean()
    order = np.argsort(importance)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(
        [feature_names[i] for i in order][::-1],
        importance[order][::-1],
    )
    ax.set_xlabel("Importance (heuristic)")
    ax.set_title("Feature Importance (ANFIS center-variation)")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=120)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to IBM Telco CSV")
    parser.add_argument("--out", default="anfis_model.joblib")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--n-rules", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[1/5] Loading data from {args.csv}")
    X_df, y, feature_names = load_and_prepare(args.csv)
    print(f"      rows={len(X_df)}, features={X_df.shape[1]}")
    print(f"      features: {feature_names}")
    print(f"      churn rate: {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_df.values, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    print("[2/5] Scaling")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"[3/5] Training ANFIS (n_rules={args.n_rules})")
    model = ANFIS(n_rules=args.n_rules, random_state=args.seed).fit(X_train_s, y_train)

    print("[4/5] Evaluating")
    y_prob = model.predict_proba(X_test_s)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "n_features": X_df.shape[1],
        "n_rules": args.n_rules,
        "feature_names": feature_names,
    }
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"      {k}: {v:.4f}")

    print(f"[5/5] Saving artifacts to {args.out}")
    joblib.dump(
        {
            "centers": model.centers_,
            "sigma": model.sigma_,
            "consequents": model.consequents_,
            "scaler": scaler,
            "encoded_features": feature_names,
        },
        args.out,
    )

    save_plots(y_test, y_prob, feature_names, model, Path(args.figures))
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("      done.")


if __name__ == "__main__":
    main()
