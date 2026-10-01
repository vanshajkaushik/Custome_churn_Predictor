"""
Feature engineering module for Customer Churn Prediction System.
Provides scikit-learn compatible transformers for domain-specific feature derivation.
Guarantees zero data leakage by computing quantile thresholds exclusively on training folds.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, List

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transforms cleaned customer data into high-signal predictive features:
    - tenure_group: Binned lifecycle stages (0-6m, 7-12m, 13-24m, 25-48m, 49+m)
    - monthly_charge_band: Low (<$40), Medium ($40-$80), High (>$80)
    - engagement_score: Normalized composite engagement index (0 - 100)
    - service_adoption_rate: Multi-product depth ratio
    - support_intensity: Tickets per month of tenure
    - complaint_rate: Complaints per month of tenure
    - payment_risk_flag: High-friction payment methods
    - recent_inactivity_flag: Inactive > 30 days
    - customer_value: Estimated Annual Contract Value (ACV)
    - high_value_customer: Flag for top value quantile
    """
    def __init__(self, high_value_percentile: float = 80.0):
        self.high_value_percentile = high_value_percentile
        self.value_threshold_: float = 1200.0

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        """Calculates threshold percentiles from training fold."""
        # Estimate annual value = monthly_charges * 12
        if "monthly_charges" in X.columns:
            annual_val = X["monthly_charges"] * 12.0
            self.value_threshold_ = float(np.percentile(annual_val.dropna(), self.high_value_percentile))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Derives features without modifying the original input DataFrame."""
        df = X.copy()

        # 1. Tenure Group
        if "tenure_months" in df.columns:
            bins = [-1, 6, 12, 24, 48, 120]
            labels = ["0-6 months", "7-12 months", "13-24 months", "25-48 months", "49+ months"]
            df["tenure_group"] = pd.cut(df["tenure_months"], bins=bins, labels=labels).astype(str)

        # 2. Monthly Charge Band
        if "monthly_charges" in df.columns:
            mc_bins = [0, 45.0, 85.0, 500.0]
            mc_labels = ["Low", "Medium", "High"]
            df["monthly_charge_band"] = pd.cut(df["monthly_charges"], bins=mc_bins, labels=mc_labels).astype(str)

        # 3. Service Adoption Rate
        if "number_of_services" in df.columns:
            total_possible_services = 6.0
            df["service_adoption_rate"] = np.clip(df["number_of_services"] / total_possible_services, 0.0, 1.0)

        # 4. Support Intensity (tickets per tenure month)
        if "support_tickets" in df.columns and "tenure_months" in df.columns:
            df["support_intensity"] = df["support_tickets"] / (df["tenure_months"] + 1.0)

        # 5. Complaint Rate (complaints per tenure month)
        if "complaints" in df.columns and "tenure_months" in df.columns:
            df["complaint_rate"] = df["complaints"] / (df["tenure_months"] + 1.0)

        # 6. Engagement Score (0 - 100 composite index)
        # Higher login frequency & usage frequency + lower last_activity_days => higher score
        if "login_frequency" in df.columns and "usage_frequency" in df.columns and "last_activity_days" in df.columns:
            norm_login = np.clip(df["login_frequency"] / 30.0, 0.0, 1.0)
            norm_usage = np.clip(df["usage_frequency"] / 200.0, 0.0, 1.0)
            inact_penalty = np.clip(df["last_activity_days"] / 60.0, 0.0, 1.0)
            df["engagement_score"] = np.round((norm_login * 0.40 + norm_usage * 0.40 + (1.0 - inact_penalty) * 0.20) * 100.0, 1)

        # 7. Payment Risk Flag (Electronic check has proven higher friction/churn in telecom)
        if "payment_method" in df.columns:
            df["payment_risk_flag"] = (df["payment_method"] == "Electronic check").astype(int)

        # 8. Recent Inactivity Flag
        if "last_activity_days" in df.columns:
            df["recent_inactivity_flag"] = (df["last_activity_days"] > 25).astype(int)

        # 9. Customer Value & High Value Customer Flag
        if "monthly_charges" in df.columns:
            df["customer_value"] = np.round(df["monthly_charges"] * 12.0, 2)
            df["high_value_customer"] = (df["customer_value"] >= self.value_threshold_).astype(int)

        return df

    def get_feature_names_out(self, input_features=None):
        return None
