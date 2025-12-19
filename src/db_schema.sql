-- ----------------------------
-- Dimension Tables
-- ----------------------------

CREATE TABLE IF NOT EXISTS dim_employment_status (
    employment_status_id INTEGER PRIMARY KEY AUTOINCREMENT,
    employment_status TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS dim_education_level (
    education_level_id INTEGER PRIMARY KEY AUTOINCREMENT,
    education_level TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS dim_marital_status (
    marital_status_id INTEGER PRIMARY KEY AUTOINCREMENT,
    marital_status TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS dim_home_ownership (
    home_ownership_id INTEGER PRIMARY KEY AUTOINCREMENT,
    home_ownership_status TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS dim_loan_purpose (
    loan_purpose_id INTEGER PRIMARY KEY AUTOINCREMENT,
    loan_purpose TEXT UNIQUE
);

-- ----------------------------
-- Fact Table
-- ----------------------------

CREATE TABLE IF NOT EXISTS fact_loan_application (
    loan_id INTEGER PRIMARY KEY AUTOINCREMENT,
    application_date TEXT,
    age INTEGER,
    annual_income REAL,
    credit_score INTEGER,
    employment_status_id INTEGER,
    education_level_id INTEGER,
    experience INTEGER,
    loan_amount REAL,
    loan_duration INTEGER,
    marital_status_id INTEGER,
    number_of_dependents INTEGER,
    home_ownership_id INTEGER,
    monthly_debt_payments REAL,
    credit_card_utilization_rate REAL,
    number_of_open_credit_lines INTEGER,
    number_of_credit_inquiries INTEGER,
    debt_to_income_ratio REAL,
    bankruptcy_history INTEGER,
    loan_purpose_id INTEGER,
    previous_loan_defaults INTEGER,
    payment_history INTEGER,
    length_of_credit_history REAL,
    savings_account_balance REAL,
    checking_account_balance REAL,
    total_assets REAL,
    total_liabilities REAL,
    monthly_income REAL,
    utility_bills_payment_history REAL,
    job_tenure REAL,
    net_worth REAL,
    base_interest_rate REAL,
    interest_rate REAL,
    monthly_loan_payment REAL,
    total_debt_to_income_ratio REAL,
    loan_approved INTEGER,
    risk_score REAL,
    FOREIGN KEY (employment_status_id) REFERENCES dim_employment_status(employment_status_id),
    FOREIGN KEY (education_level_id) REFERENCES dim_education_level(education_level_id),
    FOREIGN KEY (marital_status_id) REFERENCES dim_marital_status(marital_status_id),
    FOREIGN KEY (home_ownership_id) REFERENCES dim_home_ownership(home_ownership_id),
    FOREIGN KEY (loan_purpose_id) REFERENCES dim_loan_purpose(loan_purpose_id)
);
