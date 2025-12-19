import os
import requests
import streamlit as st

st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

# -------------------------------------------------
# API URL
# For docker-compose (local): http://api:8000
# For Render: set API_URL env var to FastAPI URL
# -------------------------------------------------
API_BASE_URL = os.getenv("API_URL", "http://api:8000").rstrip("/")
PREDICT_URL = f"{API_BASE_URL}/predict"

st.title("🏦 Loan Approval Prediction")
st.write("Predict whether a loan application will be approved.")

# -------------------------------------------------
# Input Form
# -------------------------------------------------
with st.form("input_form"):
    st.subheader("Applicant Information")

    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    experience = st.number_input("Experience (years)", min_value=0, max_value=50, value=5)
    annual_income = st.number_input("Annual Income", min_value=0.0, value=60000.0)
    monthly_income = st.number_input("Monthly Income", min_value=0.0, value=5000.0)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)

    loan_amount = st.number_input("Loan Amount", min_value=0.0, value=200000.0)
    loan_duration = st.number_input("Loan Duration (months)", min_value=1, value=240)
    monthly_loan_payment = st.number_input("Monthly Loan Payment", min_value=0.0, value=1500.0)

    number_of_dependents = st.number_input("Number of Dependents", min_value=0, value=1)
    job_tenure = st.number_input("Job Tenure (years)", min_value=0, value=3)

    monthly_debt_payments = st.number_input("Monthly Debt Payments", min_value=0.0, value=500.0)
    debt_to_income_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, value=0.25)
    total_debt_to_income_ratio = st.number_input("Total Debt-to-Income Ratio", min_value=0.0, value=0.30)

    credit_card_utilization_rate = st.number_input("Credit Card Utilization Rate", min_value=0.0, max_value=1.0, value=0.4)
    number_of_open_credit_lines = st.number_input("Open Credit Lines", min_value=0, value=5)
    number_of_credit_inquiries = st.number_input("Credit Inquiries", min_value=0, value=1)

    length_of_credit_history = st.number_input("Credit History Length (years)", min_value=0, value=10)
    payment_history = st.number_input("Payment History Score", min_value=0, max_value=100, value=85)

    bankruptcy_history = st.selectbox("Bankruptcy History", [0, 1])
    previous_loan_defaults = st.selectbox("Previous Loan Defaults", [0, 1])

    savings_account_balance = st.number_input("Savings Account Balance", value=15000.0)
    checking_account_balance = st.number_input("Checking Account Balance", value=5000.0)
    total_assets = st.number_input("Total Assets", value=300000.0)
    total_liabilities = st.number_input("Total Liabilities", value=100000.0)
    net_worth = st.number_input("Net Worth", value=200000.0)

    base_interest_rate = st.number_input("Base Interest Rate", value=5.0)
    interest_rate = st.number_input("Interest Rate", value=7.5)
    risk_score = st.number_input("Risk Score", value=0.35)

    utility_bills_payment_history = st.number_input(
        "Utility Bills Payment History", min_value=0.0, max_value=1.0, value=0.95
    )

    employment_status = st.selectbox(
        "Employment Status",
        ["employed", "self-employed", "unemployed", "student", "retired"]
    )

    education_level = st.selectbox(
        "Education Level",
        ["high_school", "bachelors", "masters", "phd", "other"]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["single", "married", "divorced", "widowed"]
    )

    home_ownership_status = st.selectbox(
        "Home Ownership Status",
        ["rent", "own", "mortgage", "other"]
    )

    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["home", "education", "personal", "business", "auto", "other"]
    )

    submitted = st.form_submit_button("Predict")

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if submitted:
    payload = {
        "age": int(age),
        "annual_income": float(annual_income),
        "credit_score": int(credit_score),
        "experience": int(experience),
        "loan_amount": float(loan_amount),
        "loan_duration": int(loan_duration),
        "number_of_dependents": int(number_of_dependents),
        "monthly_debt_payments": float(monthly_debt_payments),
        "credit_card_utilization_rate": float(credit_card_utilization_rate),
        "number_of_open_credit_lines": int(number_of_open_credit_lines),
        "number_of_credit_inquiries": int(number_of_credit_inquiries),
        "debt_to_income_ratio": float(debt_to_income_ratio),
        "bankruptcy_history": int(bankruptcy_history),
        "previous_loan_defaults": int(previous_loan_defaults),
        "payment_history": int(payment_history),
        "length_of_credit_history": int(length_of_credit_history),
        "savings_account_balance": float(savings_account_balance),
        "checking_account_balance": float(checking_account_balance),
        "total_assets": float(total_assets),
        "total_liabilities": float(total_liabilities),
        "monthly_income": float(monthly_income),
        "utility_bills_payment_history": float(utility_bills_payment_history),
        "job_tenure": int(job_tenure),
        "net_worth": float(net_worth),
        "base_interest_rate": float(base_interest_rate),
        "interest_rate": float(interest_rate),
        "monthly_loan_payment": float(monthly_loan_payment),
        "total_debt_to_income_ratio": float(total_debt_to_income_ratio),
        "risk_score": float(risk_score),
        "employment_status": employment_status,
        "education_level": education_level,
        "marital_status": marital_status,
        "home_ownership_status": home_ownership_status,
        "loan_purpose": loan_purpose,
    }

    try:
        resp = requests.post(PREDICT_URL, json=payload, timeout=15)
        resp.raise_for_status()
        out = resp.json()

        pred = out["loan_approved"]
        prob = out.get("approval_probability", None)

        st.write(f"Using API: `{API_BASE_URL}`")

        if pred == 1:
            st.success("✅ Loan Approved")
        else:
            st.warning("❌ Loan Not Approved")

        if prob is not None:
            st.info(f"Approval Probability: {prob:.3f}")

    except Exception as e:
        st.error(f"API call failed: {e}")
        st.error(f"Tried URL: {PREDICT_URL}")
