import pytest
import pandas as pd
from src.data.generate_data import generate_customer_churn_data
from src.data.loading import validate_schema, REQUIRED_COLUMNS

def test_data_generation_shape():
    df = generate_customer_churn_data(n_samples=500, random_state=42, inject_defects=False)
    assert len(df) == 500
    assert "churn" in df.columns
    assert df["churn"].isin([0, 1]).all()

def test_schema_validation_success():
    df = generate_customer_churn_data(n_samples=100, random_state=42, inject_defects=False)
    is_valid, missing = validate_schema(df, is_training=True)
    assert is_valid is True
    assert len(missing) == 0

def test_schema_validation_missing_columns():
    df = pd.DataFrame({"customer_id": ["CUST-001"], "tenure_months": [12]})
    is_valid, missing = validate_schema(df, is_training=False)
    assert is_valid is False
    assert len(missing) > 0
