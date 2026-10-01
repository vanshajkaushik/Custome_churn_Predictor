"""
Model training, cross-validation, evaluation, and serialization pipeline.
Trains Baseline (DummyClassifier), Decision Tree, and Random Forest models.
Generates evaluation figures and persists scikit-learn pipelines.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    roc_auc_score, average_precision_score, classification_report
)

from src.features.engineering import FeatureEngineer
from src.evaluation.metrics import evaluate_model_performance, analyze_threshold_sweep

def get_pipeline(estimator):
    """Builds an end-to-end scikit-learn Pipeline encapsulating preprocessing and model."""
    categorical_cols = [
        "gender", "location", "marital_status", "occupation",
        "contract_type", "signup_channel", "customer_segment",
        "payment_method", "service_type", "tenure_group", "monthly_charge_band"
    ]

    numeric_cols = [
        "age", "dependents", "tenure_months", "monthly_charges", "total_charges",
        "average_monthly_spend", "number_of_services", "add_on_count",
        "login_frequency", "support_tickets", "complaints", "last_activity_days",
        "usage_frequency", "satisfaction_score", "service_issues", "response_time",
        "service_adoption_rate", "support_intensity", "complaint_rate",
        "engagement_score", "payment_risk_flag", "recent_inactivity_flag",
        "customer_value", "high_value_customer"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_cols)
        ],
        remainder="drop"
    )

    full_pipeline = Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("preprocessor", preprocessor),
        ("classifier", estimator)
    ])

    return full_pipeline

def train_and_evaluate_all():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_path = os.path.join(root, "data", "processed", "train.csv")
    test_path = os.path.join(root, "data", "processed", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = "churn"
    id_col = "customer_id"

    X_train = train_df.drop(columns=[target_col, id_col])
    y_train = train_df[target_col].values
    X_test = test_df.drop(columns=[target_col, id_col])
    y_test = test_df[target_col].values

    print(f"Training on {len(X_train)} rows, Testing on {len(X_test)} rows")
    print(f"Train Churn Rate: {y_train.mean():.4f}, Test Churn Rate: {y_test.mean():.4f}")

    # 1. Baseline Model
    baseline_clf = DummyClassifier(strategy="most_frequent")
    baseline_pipe = get_pipeline(baseline_clf)
    baseline_pipe.fit(X_train, y_train)
    base_pred_proba = baseline_pipe.predict_proba(X_test)[:, 1]
    base_metrics = evaluate_model_performance(y_test, base_pred_proba, threshold=0.5)

    # 2. Decision Tree Classifier
    dt_clf = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=30,
        class_weight="balanced",
        random_state=42
    )
    dt_pipe = get_pipeline(dt_clf)
    dt_cv_scores = cross_val_score(dt_pipe, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"Decision Tree 5-Fold CV ROC-AUC: {dt_cv_scores.mean():.4f} +/- {dt_cv_scores.std():.4f}")
    dt_pipe.fit(X_train, y_train)
    dt_pred_proba = dt_pipe.predict_proba(X_test)[:, 1]
    dt_metrics = evaluate_model_performance(y_test, dt_pred_proba, threshold=0.5)

    # 3. Random Forest Classifier
    rf_clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=13,
        min_samples_split=12,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    rf_pipe = get_pipeline(rf_clf)
    rf_cv_scores = cross_val_score(rf_pipe, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"Random Forest 5-Fold CV ROC-AUC: {rf_cv_scores.mean():.4f} +/- {rf_cv_scores.std():.4f}")
    rf_pipe.fit(X_train, y_train)
    rf_pred_proba = rf_pipe.predict_proba(X_test)[:, 1]
    rf_metrics = evaluate_model_performance(y_test, rf_pred_proba, threshold=0.5)

    # Threshold sweeps for Random Forest
    rf_thresholds = analyze_threshold_sweep(y_test, rf_pred_proba)

    # Optimal threshold based on F1-score and high-risk recall
    best_f1_row = rf_thresholds.loc[rf_thresholds["f1"].idxmax()]
    best_th = float(best_f1_row["threshold"])
    rf_metrics_opt = evaluate_model_performance(y_test, rf_pred_proba, threshold=best_th)

    print("\n================== MODEL EVALUATION SUMMARY ==================")
    print(f"Baseline Accuracy: {base_metrics['accuracy']:.4f} | Recall: {base_metrics['recall']:.4f}")
    print(f"Decision Tree: Accuracy: {dt_metrics['accuracy']:.4f} | Precision: {dt_metrics['precision']:.4f} | Recall: {dt_metrics['recall']:.4f} | F1: {dt_metrics['f1']:.4f} | ROC-AUC: {dt_metrics['roc_auc']:.4f}")
    print(f"Random Forest (th=0.50): Accuracy: {rf_metrics['accuracy']:.4f} | Precision: {rf_metrics['precision']:.4f} | Recall: {rf_metrics['recall']:.4f} | F1: {rf_metrics['f1']:.4f} | ROC-AUC: {rf_metrics['roc_auc']:.4f}")
    print(f"Random Forest (opt th={best_th:.2f}): Accuracy: {rf_metrics_opt['accuracy']:.4f} | Precision: {rf_metrics_opt['precision']:.4f} | Recall: {rf_metrics_opt['recall']:.4f} | F1: {rf_metrics_opt['f1']:.4f} | ROC-AUC: {rf_metrics_opt['roc_auc']:.4f}")

    # Save models
    models_dir = os.path.join(root, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(baseline_pipe, os.path.join(models_dir, "baseline_pipeline.joblib"))
    joblib.dump(dt_pipe, os.path.join(models_dir, "decision_tree.joblib"))
    joblib.dump(rf_pipe, os.path.join(models_dir, "random_forest.joblib"))
    joblib.dump(rf_pipe, os.path.join(models_dir, "churn_pipeline.joblib"))

    # Extract Feature Importance from Random Forest
    rf_model = rf_pipe.named_steps["classifier"]
    preprocessor = rf_pipe.named_steps["preprocessor"]
    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out()
    num_feature_names = preprocessor.transformers_[1][2]
    all_feature_names = list(cat_feature_names) + list(num_feature_names)
    importances = rf_model.feature_importances_

    fi_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    fi_df.to_csv(os.path.join(models_dir, "feature_importance.csv"), index=False)

    # Save evaluation summary JSON
    eval_results = {
        "baseline": base_metrics,
        "decision_tree": {
            **dt_metrics,
            "cv_roc_auc_mean": float(dt_cv_scores.mean()),
            "cv_roc_auc_std": float(dt_cv_scores.std())
        },
        "random_forest": {
            **rf_metrics,
            "cv_roc_auc_mean": float(rf_cv_scores.mean()),
            "cv_roc_auc_std": float(rf_cv_scores.std()),
            "optimal_threshold": best_th,
            "optimal_metrics": rf_metrics_opt,
            "threshold_sweep": rf_thresholds.to_dict(orient="records")
        }
    }
    with open(os.path.join(models_dir, "evaluation_results.json"), "w") as f:
        json.dump(eval_results, f, indent=2)

    # Save figures
    fig_dir = os.path.join(root, "reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ROC Curves
    plt.figure(figsize=(7, 5))
    fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_pred_proba)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_pred_proba)
    plt.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC = {dt_metrics['roc_auc']:.3f})", color="#e67e22", lw=2)
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {rf_metrics['roc_auc']:.3f})", color="#2980b9", lw=2.5)
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.500)")
    plt.title("ROC Curves Comparison: Decision Tree vs Random Forest", fontsize=12, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "roc_curve.png"))
    plt.close()

    # PR Curves
    plt.figure(figsize=(7, 5))
    prec_dt, rec_dt, _ = precision_recall_curve(y_test, dt_pred_proba)
    prec_rf, rec_rf, _ = precision_recall_curve(y_test, rf_pred_proba)
    plt.plot(rec_dt, prec_dt, label=f"Decision Tree (PR-AUC = {dt_metrics['pr_auc']:.3f})", color="#e67e22", lw=2)
    plt.plot(rec_rf, prec_rf, label=f"Random Forest (PR-AUC = {rf_metrics['pr_auc']:.3f})", color="#2980b9", lw=2.5)
    plt.axhline(y=y_test.mean(), color="k", linestyle="--", label=f"Baseline Churn Rate ({y_test.mean()*100:.1f}%)")
    plt.title("Precision-Recall Curves Comparison", fontsize=12, fontweight="bold")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "pr_curve.png"))
    plt.close()

    # Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    cm_dt = confusion_matrix(y_test, (dt_pred_proba >= 0.5).astype(int))
    cm_rf = confusion_matrix(y_test, (rf_pred_proba >= best_th).astype(int))

    for ax, cm, title in zip(axes, [cm_dt, cm_rf], ["Decision Tree (th=0.50)", f"Random Forest (opt th={best_th:.2f})"]):
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Stay", "Pred Churn"])
        ax.set_yticklabels(["Actual Stay", "Actual Churn"])
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max()/2 else "black"
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color, fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "confusion_matrices.png"))
    plt.close()

    # Threshold Trade-off Plot
    plt.figure(figsize=(8, 4.5))
    plt.plot(rf_thresholds["threshold"], rf_thresholds["precision"], marker="o", label="Precision", color="#27ae60", lw=2)
    plt.plot(rf_thresholds["threshold"], rf_thresholds["recall"], marker="s", label="Recall", color="#c0392b", lw=2)
    plt.plot(rf_thresholds["threshold"], rf_thresholds["f1"], marker="^", label="F1 Score", color="#2980b9", lw=2)
    plt.plot(rf_thresholds["threshold"], rf_thresholds["accuracy"], marker="d", label="Accuracy", color="#7f8c8d", linestyle="--")
    plt.axvline(x=best_th, color="purple", linestyle=":", label=f"Selected Threshold ({best_th:.2f})")
    plt.title("Random Forest Metric Dynamics vs Decision Threshold", fontsize=12, fontweight="bold")
    plt.xlabel("Decision Probability Threshold")
    plt.ylabel("Metric Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "threshold_analysis.png"))
    plt.close()

    # Top 15 Feature Importances
    plt.figure(figsize=(9, 6))
    top15 = fi_df.head(15).iloc[::-1]
    plt.barh(top15["feature"], top15["importance"], color="#34495e", height=0.6)
    plt.title("Top 15 Most Influential Churn Predictors (Random Forest MDI)", fontsize=12, fontweight="bold")
    plt.xlabel("Relative Feature Importance (MDI)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "feature_importance.png"))
    plt.close()

    print("Model training, serialization, and figure generation complete!")

if __name__ == "__main__":
    train_and_evaluate_all()
