# Model Card: Customer Churn Prediction Pipeline

## 1. Model Details
- **Model Name**: Customer Churn Risk Classifier
- **Model Version**: 1.0.0
- **Model Type**: Supervised Binary Classification (`Pipeline` encapsulating `FeatureEngineer`, `ColumnTransformer`, and `RandomForestClassifier`)
- **Framework**: Python 3.13, Scikit-learn 1.9.1, Joblib 1.6.0
- **Release Date**: September 2026
- **License**: MIT
- **Developers**: ML Engineering & Analytics Team

---

## 2. Intended Use
- **Primary Intended Use**: Proactive identification of enterprise and residential telecom/SaaS customers with elevated defection hazard over a 30–60 day forward horizon.
- **Primary Users**: Customer Success Managers, Retention Specialists, Revenue Operations Leads.
- **Out-of-Scope / Inappropriate Uses**:
  - Automatically terminating accounts or revoking customer services.
  - Making discriminatory pricing decisions based on demographic attributes (e.g. age, gender, marital status).
  - Enforcing debt recovery or credit denial.

---

## 3. Training Data
- **Dataset Source**: Standardized multi-domain customer behavior dataset (12,500 total records).
- **Split Configuration**: Stratified 80% train split (10,000 records) and 20% test split (2,500 records).
- **Target Distribution**:
  - Class 0 (Retained): 66.11%
  - Class 1 (Churned): 33.89%
- **Preprocessing Applied**:
  - Deduplication on `customer_id`.
  - Canonical mapping of contract categories and gender labels.
  - Domain-aware median imputation for missing numeric variables; mode imputation for categoricals.
  - One-hot encoding of categorical variables (`handle_unknown="ignore"`).
  - Robust z-score standard scaling for continuous features.

---

## 4. Feature Taxonomy
- **Demographics** (Analyzed for systemic aggregate patterns, not individual profiling): `age`, `gender`, `location`, `marital_status`, `dependents`, `occupation`.
- **Relationship Lifecycle**: `tenure_months`, `contract_type`, `signup_channel`, `customer_segment`.
- **Financial Dynamics**: `monthly_charges`, `total_charges`, `average_monthly_spend`, `payment_method`.
- **Product Depth**: `service_type`, `number_of_services`, `add_on_count`.
- **Engagement & Experience**: `login_frequency`, `support_tickets`, `complaints`, `last_activity_days`, `usage_frequency`, `satisfaction_score`, `service_issues`, `response_time`.
- **Engineered Features**: `tenure_group`, `monthly_charge_band`, `engagement_score`, `service_adoption_rate`, `support_intensity`, `complaint_rate`, `payment_risk_flag`, `recent_inactivity_flag`, `customer_value`, `high_value_customer`.

---

## 5. Performance Metrics (Held-out Test Set)

| Metric | Random Forest (Tuned) | Decision Tree (Tuned) | Baseline (Majority Class) |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | **0.8591** | 0.8425 | 0.5000 |
| **PR-AUC** | **0.7575** | 0.7191 | 0.3388 |
| **5-Fold CV ROC-AUC** | **0.8625 ± 0.0070** | 0.8441 ± 0.0070 | N/A |
| **Accuracy (th=0.50)** | **78.68%** | 77.08% | 66.12% |
| **Precision (th=0.50)** | **65.76%** | 62.83% | 0.00% |
| **Recall (th=0.50)** | **77.33%** | 79.22% | 0.00% |
| **F1 Score (th=0.50)** | **0.7108** | 0.7008 | 0.0000 |
| **Recall at th=0.35** | **86.30%** | N/A | N/A |

---

## 6. Decision Threshold Optimization
- **Default Probability Threshold**: 0.50 (balanced precision/recall, F1 = 0.711).
- **Operational Risk Threshold**: 0.35 is recommended for retention campaigns with low intervention costs (email, SMS, customer check-in), boosting churn recall to **86.30%**.
- **High-Cost Intervention Threshold**: 0.60 is recommended for expensive incentives (free hardware, deep discounts), achieving **71.64% precision**.

---

## 7. Ethical Considerations & Demographic Fairness
- **Demographic Parity**: Demographic attributes (`age`, `gender`, `marital_status`) contribute less than 4% of cumulative model feature importance. Churn propensity is fundamentally driven by behavioral friction, contract commitment, and service satisfaction.
- **Explainability Requirement**: Every automated risk alert is accompanied by human-interpretable risk drivers and protective factors to ensure retention specialists never act on black-box predictions.

---

## 8. Model Monitoring & Maintenance
- **Concept Drift**: Retrain model every quarter or when customer base shifts by >15%.
- **Data Drift Indicators**:
  - Shift in mean `tenure_months` or contract distribution.
  - Significant divergence in predicted churn probability distribution (monitored via Population Stability Index, PSI).
- **Feedback Loop**: Record retention outreach outcomes (Saved vs. Churned) to track real-world intervention efficacy and update scenario models.
