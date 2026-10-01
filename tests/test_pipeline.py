import os
import pytest
import pandas as pd
import numpy as np
from src.models.predict import ChurnPredictor

@pytest.fixture
def predictor():
    return ChurnPredictor()

@pytest.fixture
def sample_customer():
    return {
        "customer_id": "TEST-001",
        "age": 50,
        "gender": "Female",
        "location": "Suburban",
        "marital_status": "Married",
        "dependents": 2,
        "occupation": "Employed",
        "tenure_months": 3,
        "contract_type": "Month-to-month",
        "signup_channel": "Online",
        "customer_segment": "Standard",
        "monthly_charges": 85.0,
        "total_charges": 255.0,
        "average_monthly_spend": 85.0,
        "payment_method": "Electronic check",
        "service_type": "Fiber Optic",
        "number_of_services": 3,
        "add_on_count": 1,
        "login_frequency": 6,
        "support_tickets": 4,
        "complaints": 1,
        "last_activity_days": 28,
        "usage_frequency": 75,
        "satisfaction_score": 2,
        "service_issues": 3,
        "response_time": 14.5
    }

def test_single_prediction_pipeline(predictor, sample_customer):
    res = predictor.predict_single(sample_customer, threshold=0.50)
    assert "churn_probability" in res
    assert 0.0 <= res["churn_probability"] <= 1.0
    assert res["predicted_churn"] in [0, 1]
    assert res["risk_category"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert "recommended_action" in res
    assert len(res["risk_factors"]) > 0

def test_batch_prediction_pipeline(predictor, sample_customer, tmp_path):
    # Construct small batch dataframe
    df = pd.DataFrame([sample_customer, sample_customer.copy()])
    df.loc[1, "customer_id"] = "TEST-002"
    df.loc[1, "tenure_months"] = 48
    df.loc[1, "contract_type"] = "Two year"
    df.loc[1, "satisfaction_score"] = 5

    out_csv = tmp_path / "batch_out.csv"
    batch_res = predictor.predict_batch(df, threshold=0.50, output_csv_path=str(out_csv))

    assert len(batch_res) == 2
    assert "churn_probability" in batch_res.columns
    assert "risk_category" in batch_res.columns
    assert "retention_priority" in batch_res.columns
    assert "recommended_action" in batch_res.columns
    assert os.path.exists(out_csv)

    # Customer 1 (Month-to-month, low CSAT) should have higher probability than Customer 2
    prob1 = batch_res.loc[batch_res["customer_id"] == "TEST-001", "churn_probability"].values[0]
    prob2 = batch_res.loc[batch_res["customer_id"] == "TEST-002", "churn_probability"].values[0]
    assert prob1 > prob2
