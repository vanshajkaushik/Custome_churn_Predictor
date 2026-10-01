"""
Dataset generator for Customer Churn Prediction & Retention Intelligence System.
Generates 12,500+ realistic customer records with subtle non-linear churn dynamics
and authentic data-quality defects (missing values, inconsistent categories, duplicates, outliers).
"""

import os
import numpy as np
import pandas as pd

def generate_customer_churn_data(
    n_samples: int = 12500,
    random_state: int = 42,
    inject_defects: bool = True
) -> pd.DataFrame:
    """
    Generates a realistic customer churn dataset.
    """
    rng = np.random.RandomState(random_state)

    # 1. Demographics
    customer_ids = [f"CUST-{i:05d}" for i in range(1, n_samples + 1)]
    ages = rng.randint(18, 81, size=n_samples)
    genders = rng.choice(["Male", "Female", "Non-Binary"], size=n_samples, p=[0.49, 0.49, 0.02])
    locations = rng.choice(["Urban", "Suburban", "Rural"], size=n_samples, p=[0.55, 0.30, 0.15])
    marital_statuses = rng.choice(["Single", "Married", "Divorced"], size=n_samples, p=[0.42, 0.45, 0.13])
    dependents = rng.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.52, 0.22, 0.16, 0.07, 0.03])
    occupations = rng.choice(
        ["Employed", "Self-Employed", "Student", "Retired", "Unemployed"],
        size=n_samples,
        p=[0.58, 0.18, 0.09, 0.10, 0.05]
    )

    # 2. Relationship
    raw_tenure = rng.beta(1.1, 1.3, size=n_samples) * 72
    tenure_months = np.clip(np.round(raw_tenure), 0, 72).astype(int)

    contract_types = []
    for t in tenure_months:
        if t < 12:
            p = [0.70, 0.20, 0.10]
        elif t < 36:
            p = [0.45, 0.35, 0.20]
        else:
            p = [0.25, 0.35, 0.40]
        contract_types.append(rng.choice(["Month-to-month", "One year", "Two year"], p=p))
    contract_types = np.array(contract_types)

    signup_channels = rng.choice(
        ["Online", "Retail Store", "Referral", "Telemarketing"],
        size=n_samples,
        p=[0.48, 0.28, 0.16, 0.08]
    )
    customer_segments = rng.choice(
        ["Standard", "Premium", "Enterprise"],
        size=n_samples,
        p=[0.65, 0.25, 0.10]
    )

    # 3. Product & Services
    service_types = rng.choice(["Fiber Optic", "DSL", "Cable"], size=n_samples, p=[0.45, 0.35, 0.20])
    number_of_services = rng.randint(1, 7, size=n_samples)
    add_on_counts = rng.binomial(n=5, p=0.45, size=n_samples)

    # 4. Financials
    base_price = {"Fiber Optic": 65.0, "DSL": 40.0, "Cable": 45.0}
    monthly_charges = np.array([
        base_price[st] + (num_s * 7.5) + (add_ons * 5.0) + rng.normal(0, 3.5)
        for st, num_s, add_ons in zip(service_types, number_of_services, add_on_counts)
    ])
    monthly_charges = np.clip(np.round(monthly_charges, 2), 19.99, 139.99)

    total_charges = np.maximum(
        monthly_charges,
        monthly_charges * np.maximum(1, tenure_months) + rng.normal(0, 25.0, size=n_samples)
    )
    total_charges = np.round(total_charges, 2)
    average_monthly_spend = np.round(total_charges / np.maximum(1, tenure_months), 2)

    payment_methods = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n_samples,
        p=[0.34, 0.18, 0.24, 0.24]
    )

    # 5. Engagement & Customer Experience
    login_frequency = rng.poisson(lam=12, size=n_samples)
    login_frequency = np.clip(login_frequency, 0, 35)

    last_activity_days = rng.geometric(p=0.08, size=n_samples)
    last_activity_days = np.clip(last_activity_days, 1, 65)

    usage_frequency = np.clip(
        np.round(rng.normal(120, 35, size=n_samples) - (last_activity_days * 0.8)),
        10, 260
    ).astype(int)

    service_issues = rng.poisson(lam=1.2, size=n_samples)
    service_issues = np.clip(service_issues, 0, 8)

    support_tickets = np.clip(
        service_issues + rng.poisson(lam=1.0, size=n_samples) + (ages > 60).astype(int),
        0, 14
    )

    complaints = np.clip(
        (support_tickets > 4).astype(int) * rng.poisson(lam=0.8, size=n_samples) +
        (service_issues > 3).astype(int),
        0, 5
    )

    response_time = np.clip(
        np.round(rng.exponential(scale=6.5, size=n_samples) + (support_tickets * 1.2), 1),
        0.5, 48.0
    )

    sat_raw = 4.2 - (complaints * 0.7) - (service_issues * 0.3) - (response_time * 0.03) + rng.normal(0, 0.5, size=n_samples)
    satisfaction_score = np.clip(np.round(sat_raw), 1, 5).astype(int)

    # 6. Target: Realistic Churn Probability (Logistic Interaction)
    z = (
        -1.35
        + 1.70 * (contract_types == "Month-to-month").astype(float)
        - 0.80 * (contract_types == "One year").astype(float)
        - 1.60 * (contract_types == "Two year").astype(float)
        - 1.45 * (tenure_months / 72.0)
        + 0.95 * ((monthly_charges - 70.0) / 30.0)
        + 1.30 * (satisfaction_score <= 2).astype(float)
        - 0.75 * (satisfaction_score >= 4).astype(float)
        + 0.85 * (support_tickets >= 4).astype(float)
        + 0.90 * (complaints >= 1).astype(float)
        + 0.55 * (last_activity_days > 25).astype(float)
        - 0.60 * (login_frequency > 15).astype(float)
        + 0.45 * (payment_methods == "Electronic check").astype(float)
        + 0.35 * (service_types == "Fiber Optic").astype(float)
        + rng.normal(0, 0.45, size=n_samples)
    )

    churn_prob = 1.0 / (1.0 + np.exp(-z))
    churn = (rng.uniform(0, 1, size=n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": ages,
        "gender": genders,
        "location": locations,
        "marital_status": marital_statuses,
        "dependents": dependents,
        "occupation": occupations,
        "tenure_months": tenure_months,
        "contract_type": contract_types,
        "signup_channel": signup_channels,
        "customer_segment": customer_segments,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "average_monthly_spend": average_monthly_spend,
        "payment_method": payment_methods,
        "service_type": service_types,
        "number_of_services": number_of_services,
        "add_on_count": add_on_counts,
        "login_frequency": login_frequency,
        "support_tickets": support_tickets,
        "complaints": complaints,
        "last_activity_days": last_activity_days,
        "usage_frequency": usage_frequency,
        "satisfaction_score": satisfaction_score,
        "service_issues": service_issues,
        "response_time": response_time,
        "churn": churn
    })

    if not inject_defects:
        return df

    # Inject realistic real-world data flaws:
    df_raw = df.copy()

    # 1. Missing values
    missing_tenure_idx = rng.choice(n_samples, size=int(n_samples * 0.025), replace=False)
    df_raw.loc[missing_tenure_idx, "tenure_months"] = np.nan

    missing_mc_idx = rng.choice(n_samples, size=int(n_samples * 0.020), replace=False)
    df_raw.loc[missing_mc_idx, "monthly_charges"] = np.nan

    missing_tc_idx = rng.choice(n_samples, size=int(n_samples * 0.035), replace=False)
    df_raw.loc[missing_tc_idx, "total_charges"] = np.nan

    missing_sat_idx = rng.choice(n_samples, size=int(n_samples * 0.040), replace=False)
    df_raw.loc[missing_sat_idx, "satisfaction_score"] = np.nan

    missing_pm_idx = rng.choice(n_samples, size=int(n_samples * 0.015), replace=False)
    df_raw.loc[missing_pm_idx, "payment_method"] = np.nan

    # 2. Inconsistent categories and dirty strings
    contract_variants = {
        "Month-to-month": ["Month-to-month", "month_to_month", " M2M ", "month to month"],
        "One year": ["One year", "1 Year", "one-year", "1_yr"],
        "Two year": ["Two year", "2_year", "two-year", "2 Year"]
    }
    for orig, variants in contract_variants.items():
        mask = df_raw["contract_type"] == orig
        indices = df_raw[mask].index
        chosen_var = rng.choice(variants, size=len(indices), p=[0.75, 0.10, 0.10, 0.05])
        df_raw.loc[indices, "contract_type"] = chosen_var

    # Whitespace and case variations in gender and service_type
    dirty_gender_idx = rng.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    df_raw.loc[dirty_gender_idx, "gender"] = df_raw.loc[dirty_gender_idx, "gender"].apply(
        lambda x: f"  {x.lower()}  " if isinstance(x, str) else x
    )

    # 3. Duplicate rows (~180 duplicates)
    dup_idx = rng.choice(n_samples, size=180, replace=False)
    duplicates = df_raw.iloc[dup_idx].copy()
    df_raw = pd.concat([df_raw, duplicates], ignore_index=True)

    # 4. Outliers / invalid numeric values
    outlier_idx = rng.choice(len(df_raw), size=15, replace=False)
    df_raw.loc[outlier_idx[0:5], "tenure_months"] = -1
    df_raw.loc[outlier_idx[5:10], "monthly_charges"] = 999.0
    df_raw.loc[outlier_idx[10:15], "total_charges"] = 0.0

    return df_raw

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "customer_churn_raw.csv")
    df = generate_customer_churn_data(n_samples=12500, random_state=42, inject_defects=True)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} records (including duplicates/defects) at {out_path}")
    print(f"Churn rate in raw base: {df['churn'].mean():.4f}")
