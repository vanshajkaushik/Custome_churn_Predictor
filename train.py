"""
Top-level entrypoint for training the customer churn models.
Runs end-to-end:
1. Verifies raw data or generates if missing.
2. Cleans and splits data.
3. Trains baseline, Decision Tree, and Random Forest pipelines.
4. Generates performance figures and exports evaluation metrics.
"""

import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.generate_data import generate_customer_churn_data
from src.data.cleaning import clean_and_process_dataset
from src.data.loading import split_and_save_data
from src.reporting.generate_eda_charts import run_eda
from src.models.train import train_and_evaluate_all

def main():
    print("==================================================================")
    print("  CUSTOMER CHURN PREDICTION & RETENTION INTELLIGENCE SYSTEM")
    print("==================================================================")
    
    # 1. Check data
    raw_path = os.path.join("data", "raw", "customer_churn_raw.csv")
    if not os.path.exists(raw_path):
        print(f"\n[Step 1/5] Raw data not found. Generating 12,500 customer records...")
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        df_raw = generate_customer_churn_data(12500, random_state=42, inject_defects=True)
        df_raw.to_csv(raw_path, index=False)
        print(f"-> Raw dataset generated at {raw_path}")
    else:
        print(f"\n[Step 1/5] Using existing raw dataset at {raw_path}")

    # 2. Clean & Process
    print("\n[Step 2/5] Cleaning and preprocessing dataset (leakage-free)...")
    df_clean = clean_and_process_dataset(raw_path=raw_path, save=True)
    train_df, test_df = split_and_save_data(df_clean, test_size=0.20, random_state=42, save_splits=True)
    print(f"-> Cleaned records: {len(df_clean)} (Train: {len(train_df)}, Test: {len(test_df)})")

    # 3. EDA Figures
    print("\n[Step 3/5] Generating EDA charts...")
    run_eda()
    print("-> EDA figures saved to reports/figures/")

    # 4. Model Training & Evaluation
    print("\n[Step 4/5] Training Baseline, Decision Tree, and Random Forest models...")
    train_and_evaluate_all()
    print("-> Models trained, evaluated, and saved to models/")

    print("\n[Step 5/5] Pipeline execution successfully finished.")
    print("==================================================================")

if __name__ == "__main__":
    main()
