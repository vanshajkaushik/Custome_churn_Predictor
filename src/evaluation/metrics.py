"""
Model evaluation metrics and threshold optimization utilities.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

def evaluate_model_performance(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Computes comprehensive performance metrics at a specific probability threshold.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_pred_proba))
    pr_auc = float(average_precision_score(y_true, y_pred_proba))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "specificity": specificity,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }

def analyze_threshold_sweep(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Evaluates metric trade-offs across multiple probability thresholds.
    """
    if thresholds is None:
        thresholds = [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]

    records = []
    for th in thresholds:
        metrics = evaluate_model_performance(y_true, y_pred_proba, threshold=th)
        cm = metrics["confusion_matrix"]
        records.append({
            "threshold": th,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "tp": cm["tp"],
            "fp": cm["fp"],
            "fn": cm["fn"],
            "tn": cm["tn"]
        })
    return pd.DataFrame(records)
