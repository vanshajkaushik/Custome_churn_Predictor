import pytest
import numpy as np
import pandas as pd
from src.data.cleaning import DataCleaner, clean_contract_type, clean_gender

def test_clean_contract_type():
    assert clean_contract_type("month_to_month") == "Month-to-month"
    assert clean_contract_type(" M2M ") == "Month-to-month"
    assert clean_contract_type("1-Yr") == "One year"
    assert clean_contract_type("2_year") == "Two year"

def test_clean_gender():
    assert clean_gender("  male  ") == "Male"
    assert clean_gender("FEMALE") == "Female"
    assert clean_gender("non-binary") == "Non-Binary"

def test_cleaner_imputation_and_deduplication():
    raw_records = pd.DataFrame({
        "customer_id": ["C-1", "C-2", "C-2"],  # Contains duplicate
        "tenure_months": [-5, np.nan, 12],      # Negative & NaN
        "monthly_charges": [999.0, 50.0, 75.0], # Outlier 999.0
        "total_charges": [np.nan, 200.0, 900.0],# NaN total charges
        "satisfaction_score": [9, np.nan, 4],   # Invalid score 9 & NaN
        "contract_type": ["m2m", "1-Year", "Two year"],
        "gender": ["male", "female", "male"],
        "payment_method": [np.nan, "Credit card", "Electronic check"]
    })

    cleaner = DataCleaner()
    cleaned = cleaner.fit_transform(raw_records, is_training=True)

    # 1. Deduplication
    assert len(cleaned) == 2
    assert cleaned["customer_id"].tolist() == ["C-1", "C-2"]

    # 2. Imputation and bounds
    assert cleaned["tenure_months"].min() >= 0
    assert not cleaned["tenure_months"].isna().any()
    assert not cleaned["monthly_charges"].isna().any()
    assert not cleaned["total_charges"].isna().any()
    assert cleaned["satisfaction_score"].isin([1, 2, 3, 4, 5]).all()
    assert not cleaned["payment_method"].isna().any()
