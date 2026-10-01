"""
Inference and prediction module for Customer Churn Prediction System.
Handles single-instance scoring with explainability and batch scoring with CSV export.
"""

import os
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib

from src.data.cleaning import DataCleaner
from src.explainability.explainer import ChurnExplainer
from src.reporting.insights import categorize_risk, compute_retention_priority, recommend_retention_action

class ChurnPredictor:
    """
    Production inference engine. Encapsulates preprocessing, prediction,
    risk categorization, prioritization, and business action prescription.
    """
    def __init__(self, pipeline_path: Optional[str] = None):
        if pipeline_path is None:
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            pipeline_path = os.path.join(root, "models", "churn_pipeline.joblib")
        
        self.pipeline = joblib.load(pipeline_path)
        self.cleaner = DataCleaner()
        self.explainer = ChurnExplainer(pipeline_path)

    def predict_single(self, customer_data: Dict[str, Any], threshold: float = 0.50) -> Dict[str, Any]:
        """
        Scores a single customer record and returns probability, risk category,
        priority score, explainability breakdown, and recommended action.
        """
        df_raw = pd.DataFrame([customer_data])
        df_clean = self.cleaner.transform(df_raw, is_training=False)

        # Drop ID if present
        df_input = df_clean.drop(columns=["customer_id"], errors="ignore")
        if "churn" in df_input.columns:
            df_input = df_input.drop(columns=["churn"])

        proba = float(self.pipeline.predict_proba(df_input)[0, 1])
        prediction = int(proba >= threshold)
        risk_category = categorize_risk(proba)

        mc = float(customer_data.get("monthly_charges", 70.0))
        cust_value = float(mc * 12.0)
        retention_priority = compute_retention_priority(proba, cust_value)

        enriched_dict = {**customer_data, "churn_probability": proba, "customer_value": cust_value}
        action_plan = recommend_retention_action(enriched_dict)
        explanation = self.explainer.explain_instance(customer_data, baseline_prob=0.339)

        return {
            "churn_probability": np.round(proba, 4),
            "predicted_churn": prediction,
            "risk_category": risk_category,
            "retention_priority_score": retention_priority,
            "recommended_action": action_plan["action"],
            "action_tier": action_plan["tier"],
            "action_rationale": action_plan["rationale"],
            "risk_factors": explanation["risk_factors"],
            "protective_factors": explanation["protective_factors"]
        }

    def predict_batch(
        self,
        df: pd.DataFrame,
        threshold: float = 0.50,
        output_csv_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Processes batch customer DataFrame, generates predictions, priorities, and action recommendations.
        """
        df_clean = self.cleaner.transform(df, is_training=False)
        cust_ids = df_clean["customer_id"].copy() if "customer_id" in df_clean.columns else pd.Series([f"CUST-{i:05d}" for i in range(len(df_clean))])

        df_input = df_clean.drop(columns=["customer_id", "churn"], errors="ignore")
        probas = self.pipeline.predict_proba(df_input)[:, 1]

        results = df_clean.copy()
        results["customer_id"] = cust_ids
        results["churn_probability"] = np.round(probas, 4)
        results["predicted_churn"] = (probas >= threshold).astype(int)
        results["risk_category"] = [categorize_risk(p) for p in probas]

        # Customer Value & Retention Priority
        mc_series = results["monthly_charges"] if "monthly_charges" in results.columns else pd.Series(70.0, index=results.index)
        results["customer_value"] = np.round(mc_series * 12.0, 2)
        results["retention_priority"] = [
            compute_retention_priority(p, v) for p, v in zip(results["churn_probability"], results["customer_value"])
        ]

        # Prescribed actions
        actions = []
        for _, row in results.iterrows():
            rec = recommend_retention_action(row.to_dict())
            actions.append(rec["action"])
        results["recommended_action"] = actions

        # Order by retention priority descending
        results = results.sort_values("retention_priority", ascending=False).reset_index(drop=True)

        if output_csv_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
            results.to_csv(output_csv_path, index=False)
            print(f"Batch predictions exported to: {output_csv_path}")

        return results
