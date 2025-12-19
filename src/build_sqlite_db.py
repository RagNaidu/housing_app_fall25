import sqlite3
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# PATHS
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "Loan.csv"
DB_PATH = ROOT / "data" / "db" / "loan_applications.sqlite3"
SCHEMA_PATH = ROOT / "src" / "db_schema.sql"

# --------------------------------------------------
# DIMENSION MAPPING
# --------------------------------------------------
DIMENSIONS = {
    "EmploymentStatus": ("dim_employment_status", "employment_status"),
    "EducationLevel": ("dim_education_level", "education_level"),
    "MaritalStatus": ("dim_marital_status", "marital_status"),
    "HomeOwnershipStatus": ("dim_home_ownership", "home_ownership_status"),
    "LoanPurpose": ("dim_loan_purpose", "loan_purpose")
}

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def insert_unique(conn, table, col, values):
    """Insert unique values into dimension tables"""
    unique_vals = sorted(pd.Series(values).fillna("Unknown").astype(str).unique())
    conn.executemany(
        f"INSERT OR IGNORE INTO {table} ({col}) VALUES (?)",
        [(v,) for v in unique_vals],
    )

def fetch_mapping(conn, table, col):
    """Return mapping dict: value -> dim_id (for FK)"""
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    id_col = df.columns[0]
    return dict(zip(df[col], df[id_col]))

# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------
def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load CSV
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    # Connect to SQLite
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

        # ----------------------------
        # Insert dimension tables
        # ----------------------------
        for col, (table, dim_col) in DIMENSIONS.items():
            insert_unique(conn, table, dim_col, df[col])

        # Build mappings
        maps = {col: fetch_mapping(conn, table, dim_col)
                for col, (table, dim_col) in DIMENSIONS.items()}

        # ----------------------------
        # Build fact table
        # ----------------------------
        fact = pd.DataFrame({
            "application_date": df["ApplicationDate"],
            "age": df["Age"],
            "annual_income": df["AnnualIncome"],
            "credit_score": df["CreditScore"],
            "employment_status_id": df["EmploymentStatus"].map(maps["EmploymentStatus"]),
            "education_level_id": df["EducationLevel"].map(maps["EducationLevel"]),
            "experience": df["Experience"],
            "loan_amount": df["LoanAmount"],
            "loan_duration": df["LoanDuration"],
            "marital_status_id": df["MaritalStatus"].map(maps["MaritalStatus"]),
            "number_of_dependents": df["NumberOfDependents"],
            "home_ownership_id": df["HomeOwnershipStatus"].map(maps["HomeOwnershipStatus"]),
            "monthly_debt_payments": df["MonthlyDebtPayments"],
            "credit_card_utilization_rate": df["CreditCardUtilizationRate"],
            "number_of_open_credit_lines": df["NumberOfOpenCreditLines"],
            "number_of_credit_inquiries": df["NumberOfCreditInquiries"],
            "debt_to_income_ratio": df["DebtToIncomeRatio"],
            "bankruptcy_history": df["BankruptcyHistory"],
            "loan_purpose_id": df["LoanPurpose"].map(maps["LoanPurpose"]),
            "previous_loan_defaults": df["PreviousLoanDefaults"],
            "payment_history": df["PaymentHistory"],
            "length_of_credit_history": df["LengthOfCreditHistory"],
            "savings_account_balance": df["SavingsAccountBalance"],
            "checking_account_balance": df["CheckingAccountBalance"],
            "total_assets": df["TotalAssets"],
            "total_liabilities": df["TotalLiabilities"],
            "monthly_income": df["MonthlyIncome"],
            "utility_bills_payment_history": df["UtilityBillsPaymentHistory"],
            "job_tenure": df["JobTenure"],
            "net_worth": df["NetWorth"],
            "base_interest_rate": df["BaseInterestRate"],
            "interest_rate": df["InterestRate"],
            "monthly_loan_payment": df["MonthlyLoanPayment"],
            "total_debt_to_income_ratio": df["TotalDebtToIncomeRatio"],
            "loan_approved": df["LoanApproved"],
            "risk_score": df["RiskScore"]
        })

        fact.to_sql("fact_loan_application", conn, if_exists="append", index=False)

        # Quick counts
        n_fact = conn.execute("SELECT COUNT(*) FROM fact_loan_application").fetchone()[0]
        print("✅ Loan Applications DB created successfully!")
        print(f"📦 Database: {DB_PATH}")
        print(f"📊 Rows in fact_loan_application: {n_fact}")

# --------------------------------------------------
# RUN
# --------------------------------------------------
if __name__ == "__main__":
    main()
