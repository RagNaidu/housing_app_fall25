from pathlib import Path
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Loan Approval Prediction API")

# -----------------------------
# Model path
# -----------------------------
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/models/best_loan_model.pkl"))

model = None  # loaded on startup


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        model = None
        print(f"❌ Model file not found at: {MODEL_PATH}")
        return

    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from: {MODEL_PATH}")


# -----------------------------
# Input schema (must match X columns)
# -----------------------------
class LoanInput(BaseModel):
    age: int
    annual_income: float
    credit_score: int
    experience: int
    loan_amount: float
    loan_duration: int
    number_of_dependents: int
    monthly_debt_payments: float
    credit_card_utilization_rate: float
    number_of_open_credit_lines: int
    number_of_credit_inquiries: int
    debt_to_income_ratio: float
    bankruptcy_history: int
    previous_loan_defaults: int
    payment_history: int
    length_of_credit_history: int
    savings_account_balance: float
    checking_account_balance: float
    total_assets: float
    total_liabilities: float
    monthly_income: float
    utility_bills_payment_history: float
    job_tenure: int
    net_worth: float
    base_interest_rate: float
    interest_rate: float
    monthly_loan_payment: float
    total_debt_to_income_ratio: float
    risk_score: float

    # categorical
    employment_status: str
    education_level: str
    marital_status: str
    home_ownership_status: str
    loan_purpose: str


# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(payload: LoanInput):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model not loaded. Expected model at {MODEL_PATH}.",
        )

    # Convert input to DataFrame
    X = pd.DataFrame([payload.model_dump()])

    try:
        pred = int(model.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = float(model.predict_proba(X)[0][1])
        except Exception:
            proba = None

    return {
        "loan_approved": pred,
        "approval_probability": proba,
    }
