from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA


@dataclass
class FeatureSpec:
    numeric_cols: List[str]
    categorical_cols: List[str]


DEFAULT_FEATURE_SPEC = FeatureSpec(
    numeric_cols=[
        "age", "annual_income", "credit_score", "experience", "loan_amount",
        "loan_duration", "number_of_dependents", "monthly_debt_payments",
        "credit_card_utilization_rate", "number_of_open_credit_lines",
        "number_of_credit_inquiries", "debt_to_income_ratio",
        "length_of_credit_history", "savings_account_balance",
        "checking_account_balance", "total_assets", "total_liabilities",
        "monthly_income", "utility_bills_payment_history", "job_tenure",
        "net_worth", "base_interest_rate", "interest_rate",
        "monthly_loan_payment", "total_debt_to_income_ratio", "risk_score",
    ],
    categorical_cols=[
        "employment_status", "education_level", "marital_status",
        "home_ownership_status", "loan_purpose",
    ],
)


def build_preprocessor(
    spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
) -> ColumnTransformer:
    """
    ColumnTransformer that outputs a numeric matrix
    """
    num = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    cat = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num, spec.numeric_cols),
            ("cat", cat, spec.categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_feature_pipeline(
    use_pca: bool,
    pca_n_components: Optional[int] = None,
) -> Pipeline:
    """
    Preprocessing + optional PCA
    """
    steps = [("preprocess", build_preprocessor())]

    if use_pca:
        if pca_n_components is None:
            steps.append(("pca", PCA(n_components=0.95, random_state=42)))
        else:
            steps.append(("pca", PCA(n_components=pca_n_components, random_state=42)))

    return Pipeline(steps=steps)
