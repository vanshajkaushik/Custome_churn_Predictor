import os
import pytest
import numpy as np
import joblib
from src.evaluation.metrics import evaluate_model_performance, analyze_threshold_sweep
from src.reporting.insights import categorize_risk, compute_retention_priority

def test_evaluate_metrics():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.3, 0.7, 0.4])
    metrics = evaluate_model_performance(y_true, y_prob, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0

def test_threshold_sweep():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.7, 0.4, 0.3, 0.1])
    sweep = analyze_threshold_sweep(y_true, y_prob, thresholds=[0.35, 0.50])
    assert len(sweep) == 2
    assert "precision" in sweep.columns
    assert "recall" in sweep.columns

def test_risk_categorization():
    assert categorize_risk(0.85) == "CRITICAL"
    assert categorize_risk(0.60) == "HIGH"
    assert categorize_risk(0.40) == "MEDIUM"
    assert categorize_risk(0.15) == "LOW"

def test_retention_priority_score():
    high_score = compute_retention_priority(churn_prob=0.8, customer_value=1500.0)
    low_score = compute_retention_priority(churn_prob=0.2, customer_value=300.0)
    assert 0.0 <= high_score <= 100.0
    assert 0.0 <= low_score <= 100.0
    assert high_score > low_score
