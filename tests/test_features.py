import pytest
import pandas as pd
import numpy as np
from src.features.engineering import FeatureEngineer

def test_feature_engineering_transforms():
    df = pd.DataFrame({
        "tenure_months": [2, 10, 18, 30, 60],
        "monthly_charges": [30.0, 60.0, 95.0, 110.0, 40.0],
        "number_of_services": [1, 2, 4, 6, 3],
        "support_tickets": [0, 2, 6, 1, 0],
        "complaints": [0, 0, 2, 1, 0],
        "login_frequency": [5, 12, 4, 25, 18],
        "usage_frequency": [50, 100, 70, 180, 140],
        "last_activity_days": [40, 12, 35, 3, 8],
        "payment_method": ["Electronic check", "Credit card", "Electronic check", "Bank transfer", "Mailed check"]
    })

    fe = FeatureEngineer()
    fe.fit(df)
    res = fe.transform(df)

    # Check newly derived columns exist
    assert "tenure_group" in res.columns
    assert "monthly_charge_band" in res.columns
    assert "service_adoption_rate" in res.columns
    assert "support_intensity" in res.columns
    assert "complaint_rate" in res.columns
    assert "engagement_score" in res.columns
    assert "payment_risk_flag" in res.columns
    assert "recent_inactivity_flag" in res.columns
    assert "customer_value" in res.columns
    assert "high_value_customer" in res.columns

    # Verify boundary checks
    assert res["service_adoption_rate"].between(0.0, 1.0).all()
    assert res["engagement_score"].between(0.0, 100.0).all()
    assert set(res["payment_risk_flag"].unique()).issubset({0, 1})
    assert set(res["recent_inactivity_flag"].unique()).issubset({0, 1})
    assert res["tenure_group"].iloc[0] == "0-6 months"
    assert res["tenure_group"].iloc[-1] == "49+ months"

def test_no_target_leakage():
    df = pd.DataFrame({
        "tenure_months": [12],
        "monthly_charges": [70.0],
        "number_of_services": [3],
        "support_tickets": [1],
        "complaints": [0],
        "login_frequency": [10],
        "usage_frequency": [100],
        "last_activity_days": [15],
        "payment_method": ["Credit card"]
    })
    fe = FeatureEngineer()
    res = fe.fit_transform(df)
    assert "churn" not in res.columns
    assert "cancellation_date" not in res.columns
