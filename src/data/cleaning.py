"""
Data cleaning and preprocessing module for Customer Churn Prediction System.
Handles deduplication, category normalization, invalid numeric bounds, and domain-aware imputation.
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

def clean_contract_type(val: Any) -> str:
    """Standardizes dirty contract type strings."""
    if pd.isna(val):
        return "Month-to-month"
    s = str(val).strip().lower()
    s = re.sub(r"[-_\s]+", " ", s)
    if "m2m" in s or "month" in s:
        return "Month-to-month"
    elif "1" in s or "one" in s:
        return "One year"
    elif "2" in s or "two" in s:
        return "Two year"
    return "Month-to-month"

def clean_gender(val: Any) -> str:
    """Standardizes dirty gender categories."""
    if pd.isna(val):
        return "Unknown"
    s = str(val).strip().title()
    if s in ["Male", "Female", "Non-Binary"]:
        return s
    return "Non-Binary" if "non" in s.lower() else "Male"

class DataCleaner:
    """
    Stateful or stateless data cleaning pipeline.
    Computes reference statistics on training data and applies them cleanly.
    """
    def __init__(self):
        self.stats: Dict[str, Any] = {}

    def fit(self, df: pd.DataFrame) -> "DataCleaner":
        """Learns imputation statistics from training dataframe."""
        df_temp = df.copy()
        # Filter obvious invalid outliers before computing stats
        valid_tenure = df_temp["tenure_months"].dropna()
        valid_tenure = valid_tenure[(valid_tenure >= 0) & (valid_tenure <= 100)]
        self.stats["tenure_median"] = float(valid_tenure.median()) if len(valid_tenure) > 0 else 24.0

        valid_mc = df_temp["monthly_charges"].dropna()
        valid_mc = valid_mc[(valid_mc >= 15.0) & (valid_mc <= 250.0)]
        self.stats["monthly_charges_median"] = float(valid_mc.median()) if len(valid_mc) > 0 else 65.0

        valid_sat = df_temp["satisfaction_score"].dropna()
        valid_sat = valid_sat[(valid_sat >= 1) & (valid_sat <= 5)]
        self.stats["satisfaction_median"] = int(valid_sat.median()) if len(valid_sat) > 0 else 3

        pm_series = df_temp["payment_method"].dropna()
        self.stats["payment_method_mode"] = str(pm_series.mode()[0]) if len(pm_series) > 0 else "Electronic check"
        return self

    def transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """Applies data cleaning and domain-aware imputation."""
        cleaned = df.copy()

        # 1. Deduplication
        if "customer_id" in cleaned.columns:
            cleaned = cleaned.drop_duplicates(subset=["customer_id"], keep="first").reset_index(drop=True)
        else:
            cleaned = cleaned.drop_duplicates().reset_index(drop=True)

        # 2. Categorical normalization
        if "contract_type" in cleaned.columns:
            cleaned["contract_type"] = cleaned["contract_type"].apply(clean_contract_type)

        if "gender" in cleaned.columns:
            cleaned["gender"] = cleaned["gender"].apply(clean_gender)

        # Strip whitespace for other object columns
        obj_cols = cleaned.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            if col != "customer_id":
                cleaned[col] = cleaned[col].astype(str).str.strip()
                # Replace string 'nan' with proper NaN
                cleaned.loc[cleaned[col].str.lower() == "nan", col] = np.nan

        # 3. Numeric range corrections
        if "tenure_months" in cleaned.columns:
            cleaned.loc[cleaned["tenure_months"] < 0, "tenure_months"] = np.nan
            cleaned.loc[cleaned["tenure_months"] > 100, "tenure_months"] = np.nan
            tenure_fill = self.stats.get("tenure_median", 24.0)
            cleaned["tenure_months"] = cleaned["tenure_months"].fillna(tenure_fill).astype(int)

        if "monthly_charges" in cleaned.columns:
            cleaned.loc[cleaned["monthly_charges"] < 15.0, "monthly_charges"] = np.nan
            cleaned.loc[cleaned["monthly_charges"] > 250.0, "monthly_charges"] = np.nan
            mc_fill = self.stats.get("monthly_charges_median", 65.0)
            cleaned["monthly_charges"] = cleaned["monthly_charges"].fillna(mc_fill)

        if "total_charges" in cleaned.columns:
            # Reconstruct missing or invalid total charges using tenure * monthly_charges
            invalid_tc = (cleaned["total_charges"].isna()) | (cleaned["total_charges"] <= 0)
            cleaned.loc[invalid_tc, "total_charges"] = np.maximum(
                cleaned.loc[invalid_tc, "monthly_charges"],
                cleaned.loc[invalid_tc, "monthly_charges"] * np.maximum(1, cleaned.loc[invalid_tc, "tenure_months"])
            )

        if "average_monthly_spend" in cleaned.columns:
            # Ensure average monthly spend matches total_charges / tenure
            cleaned["average_monthly_spend"] = np.round(
                cleaned["total_charges"] / np.maximum(1, cleaned["tenure_months"]), 2
            )

        if "satisfaction_score" in cleaned.columns:
            cleaned.loc[~cleaned["satisfaction_score"].isin([1, 2, 3, 4, 5]), "satisfaction_score"] = np.nan
            sat_fill = self.stats.get("satisfaction_median", 3)
            cleaned["satisfaction_score"] = cleaned["satisfaction_score"].fillna(sat_fill).astype(int)

        if "payment_method" in cleaned.columns:
            pm_fill = self.stats.get("payment_method_mode", "Electronic check")
            cleaned["payment_method"] = cleaned["payment_method"].fillna(pm_fill)

        return cleaned

    def fit_transform(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Fits stats on dataset and transforms."""
        return self.fit(df).transform(df, is_training=is_training)

def clean_and_process_dataset(raw_path: Optional[str] = None, save: bool = True) -> pd.DataFrame:
    """Full end-to-end data cleaning pipeline from raw CSV to data/processed."""
    from src.data.loading import load_raw_data, get_project_root
    raw_df = load_raw_data(raw_path)
    cleaner = DataCleaner()
    cleaned_df = cleaner.fit_transform(raw_df, is_training=True)

    if save:
        out_dir = os.path.join(get_project_root(), "data", "processed")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "customer_churn_cleaned.csv")
        cleaned_df.to_csv(out_path, index=False)
        print(f"Cleaned dataset saved: {len(cleaned_df)} rows at {out_path}")

    return cleaned_df

if __name__ == "__main__":
    df_clean = clean_and_process_dataset()
    from src.data.loading import split_and_save_data
    train_df, test_df = split_and_save_data(df_clean, test_size=0.20, random_state=42)
    print(f"Train set: {len(train_df)} rows, Test set: {len(test_df)} rows")
