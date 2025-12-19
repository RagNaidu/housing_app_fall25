from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd

# -----------------------------
# Default paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "data" / "db" / "loan_applications.sqlite3"

# -----------------------------
# SQL JOIN Query
# -----------------------------
JOIN_QUERY = """
SELECT
    f.loan_id,

    -- numeric
    f.age,
    f.annual_income,
    f.credit_score,
    f.experience,
    f.loan_amount,
    f.loan_duration,
    f.number_of_dependents,
    f.monthly_debt_payments,
    f.credit_card_utilization_rate,
    f.number_of_open_credit_lines,
    f.number_of_credit_inquiries,
    f.debt_to_income_ratio,
    f.bankruptcy_history,
    f.previous_loan_defaults,
    f.payment_history,
    f.length_of_credit_history,
    f.savings_account_balance,
    f.checking_account_balance,
    f.total_assets,
    f.total_liabilities,
    f.monthly_income,
    f.utility_bills_payment_history,
    f.job_tenure,
    f.net_worth,
    f.base_interest_rate,
    f.interest_rate,
    f.monthly_loan_payment,
    f.total_debt_to_income_ratio,
    f.risk_score,

    -- categorical (decoded from dimensions)
    j.employment_status AS employment_status,
    e.education_level AS education_level,
    m.marital_status AS marital_status,
    h.home_ownership_status AS home_ownership_status,
    p.loan_purpose AS loan_purpose,

    -- target
    f.loan_approved
FROM fact_loan_application f
LEFT JOIN dim_employment_status j ON f.employment_status_id = j.employment_status_id
LEFT JOIN dim_education_level e ON f.education_level_id = e.education_level_id
LEFT JOIN dim_marital_status m ON f.marital_status_id = m.marital_status_id
LEFT JOIN dim_home_ownership h ON f.home_ownership_id = h.home_ownership_id
LEFT JOIN dim_loan_purpose p ON f.loan_purpose_id = p.loan_purpose_id;
"""

# -----------------------------
# Load dataframe
# -----------------------------
def load_dataframe(db_path: str | Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Load full joined loan dataframe from SQLite."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(JOIN_QUERY, conn)
    return df

# -----------------------------
# Split features and target
# -----------------------------
def split_X_y(df: pd.DataFrame):
    """Split dataframe into X (features) and y (target)."""
    y = df["loan_approved"].astype(int)
    X = df.drop(columns=["loan_id", "loan_approved"])
    return X, y

# -----------------------------
# Test run
# -----------------------------
if __name__ == "__main__":
    df = load_dataframe()
    print("✅ Loaded joined loan dataframe")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head(3))
