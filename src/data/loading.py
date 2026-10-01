"""
Data loading and validation module for Customer Churn Prediction System.
"""

import os
from typing import Tuple, List, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS: List[str] = [
    "customer_id", "age", "gender", "location", "marital_status",
    "dependents", "occupation", "tenure_months", "contract_type",
    "signup_channel", "customer_segment", "monthly_charges",
    "total_charges", "average_monthly_spend", "payment_method",
    "service_type", "number_of_services", "add_on_count",
    "login_frequency", "support_tickets", "complaints",
    "last_activity_days", "usage_frequency", "satisfaction_score",
    "service_issues", "response_time"
]

TARGET_COLUMN: str = "churn"

def get_project_root() -> str:
    """Returns absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Loads raw dataset from data/raw or custom path and verifies structure."""
    if filepath is None:
        filepath = os.path.join(get_project_root(), "data", "raw", "customer_churn_raw.csv")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    return df

def validate_schema(df: pd.DataFrame, is_training: bool = True) -> Tuple[bool, List[str]]:
    """
    Validates that required features are present.
    Returns (is_valid, missing_columns).
    """
    expected = REQUIRED_COLUMNS.copy()
    if is_training:
        expected.append(TARGET_COLUMN)
        
    missing = [col for col in expected if col not in df.columns]
    return (len(missing) == 0, missing)

def load_processed_data(processed_path: Optional[str] = None) -> pd.DataFrame:
    """Loads cleaned processed dataset."""
    if processed_path is None:
        processed_path = os.path.join(get_project_root(), "data", "processed", "customer_churn_cleaned.csv")
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed dataset not found at: {processed_path}")
    return pd.read_csv(processed_path)

def split_and_save_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    save_splits: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs stratified train/test split on churn target and saves to data/processed.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COLUMN]
    )
    
    if save_splits:
        out_dir = os.path.join(get_project_root(), "data", "processed")
        os.makedirs(out_dir, exist_ok=True)
        train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)
        
    return train_df, test_df
