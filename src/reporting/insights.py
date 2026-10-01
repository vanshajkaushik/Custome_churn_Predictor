"""
Business insights, customer segmentation, retention strategy, and scenario modeling module.
Computes Retention Priority Scores (Risk x Value) and models scenario-based retention ROI.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd

def categorize_risk(churn_prob: float) -> str:
    """
    Categorizes customer churn risk tier based on model probability:
    - CRITICAL: >= 0.70
    - HIGH: 0.50 - 0.69
    - MEDIUM: 0.30 - 0.49
    - LOW: < 0.30
    """
    if churn_prob >= 0.70:
        return "CRITICAL"
    elif churn_prob >= 0.50:
        return "HIGH"
    elif churn_prob >= 0.30:
        return "MEDIUM"
    return "LOW"

def compute_retention_priority(churn_prob: float, customer_value: float, mean_value: float = 850.0) -> float:
    """
    Calculates Retention Priority Score (0 - 100):
    Retention Priority Score = Churn Probability * (Customer Value / Benchmark Value) * 50
    Balances urgency (probability of defection) with financial exposure (annual customer value).
    """
    normalized_value = np.clip(customer_value / mean_value, 0.2, 3.0)
    score = churn_prob * normalized_value * 50.0
    return float(np.clip(np.round(score, 1), 0.0, 100.0))

def recommend_retention_action(row: Dict[str, Any]) -> Dict[str, str]:
    """
    Prescribes targeted business retention interventions based on multi-dimensional customer profile.
    """
    prob = float(row.get("churn_probability", 0.3))
    tenure = float(row.get("tenure_months", 12))
    mc = float(row.get("monthly_charges", 70))
    sat = int(row.get("satisfaction_score", 3))
    tickets = int(row.get("support_tickets", 1))
    complaints = int(row.get("complaints", 0))
    contract = str(row.get("contract_type", "")).lower()
    val = float(row.get("customer_value", mc * 12.0))

    if prob < 0.30:
        return {
            "tier": "Low Risk / Growth",
            "action": "Loyalty & Upsell Program",
            "rationale": "High relationship health. Present multi-service add-ons (streaming, cloud storage) or 2-year renewal reward."
        }

    # High Value + High Risk
    if val >= 1200.0 and prob >= 0.50:
        return {
            "tier": "Critical Financial Exposure",
            "action": "Dedicated Account Manager & VIP Retention Concierge",
            "rationale": "High-value enterprise/premium account with critical churn risk. Proactive outreach within 24 hours."
        }

    # Dissatisfaction / Support issues
    if sat <= 2 or complaints >= 1 or tickets >= 4:
        return {
            "tier": "Service Recovery",
            "action": "Executive Escalation & SLA Resolution Credit",
            "rationale": "Customer experiencing technical/operational friction. Issue $25 billing credit and assign tier-3 specialist."
        }

    # High Price + Month-to-Month
    if "month" in contract and mc > 80.0:
        return {
            "tier": "Pricing Vulnerability",
            "action": "Annual Contract Incentive Offer (15% Savings)",
            "rationale": "High monthly spend on flexible contract. Offer discounted 12-month lock-in to prevent competitor poaching."
        }

    # New Customer Attrition
    if tenure <= 6:
        return {
            "tier": "Onboarding Friction",
            "action": "Early Lifecycle Onboarding Check-in",
            "rationale": "Customer in the vulnerable first 6 months. Deliver tailored product walkthrough and setup assistance."
        }

    return {
        "tier": "Standard Proactive Retention",
        "action": "Targeted Retention Email & Value Summary",
        "rationale": "Moderate risk. Deliver personalized usage report highlighting feature value and seasonal loyalty credit."
    }

def calculate_scenario_retention_impact(
    total_customers: int,
    churn_rate: float,
    targeted_high_risk_share: float = 0.30,
    intervention_success_rate: float = 0.12,
    avg_annual_revenue: float = 840.0
) -> Dict[str, Any]:
    """
    Transparent scenario-based impact calculator.
    Projects customer retention uplift and revenue preservation under selectable intervention success rates.
    Directly models the resume claim: 'projected to improve customer retention by 12%'.
    """
    total_expected_churners = int(total_customers * churn_rate)
    targeted_churners = int(total_expected_churners * targeted_high_risk_share)
    retained_customers = int(targeted_churners * intervention_success_rate)
    
    post_intervention_churners = total_expected_churners - retained_customers
    retention_improvement_pct = (retained_customers / total_expected_churners) * 100.0 if total_expected_churners > 0 else 0.0
    revenue_preserved = retained_customers * avg_annual_revenue

    return {
        "total_customers": total_customers,
        "base_churn_rate_pct": np.round(churn_rate * 100.0, 2),
        "total_expected_churners": total_expected_churners,
        "targeted_high_risk_churners": targeted_churners,
        "assumed_intervention_success_rate_pct": np.round(intervention_success_rate * 100.0, 1),
        "retained_customers_projected": retained_customers,
        "retention_improvement_pct": np.round(retention_improvement_pct, 2),
        "post_intervention_churn_rate_pct": np.round((post_intervention_churners / total_customers) * 100.0, 2),
        "revenue_preserved_projected": np.round(revenue_preserved, 2)
    }
