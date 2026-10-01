"""
EDA and figure generation script for Customer Churn Prediction System.
Produces high-quality figures answering core business questions and saves to reports/figures/.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Styling setup
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 150

def get_figures_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fig_dir = os.path.join(root, "reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir

def run_eda():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_path = os.path.join(root, "data", "processed", "customer_churn_cleaned.csv")
    df = pd.read_csv(processed_path)
    fig_dir = get_figures_dir()

    print(f"Loaded {len(df)} cleaned records for EDA.")
    print(f"Overall Churn Rate: {df['churn'].mean() * 100:.2f}%")

    # 1. Churn Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["churn"].value_counts()
    ax.bar(["Retained (0)", "Churned (1)"], counts, color=["#2b5c8f", "#d9534f"], width=0.5)
    ax.set_title("Customer Churn Distribution", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Customers")
    for i, v in enumerate(counts):
        ax.text(i, v + 100, f"{v:,} ({v/len(df)*100:.1f}%)", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "churn_distribution.png"))
    plt.close()

    # 2. Churn by Contract Type
    fig, ax = plt.subplots(figsize=(7, 4.5))
    contract_rates = df.groupby("contract_type")["churn"].agg(["mean", "count"])
    bars = ax.bar(contract_rates.index, contract_rates["mean"] * 100, color=["#d9534f", "#f0ad4e", "#5cb85c"], width=0.45)
    ax.set_title("Churn Rate by Contract Type (Key Driver #1)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, 75)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "churn_by_contract.png"))
    plt.close()

    # 3. Churn vs Tenure
    fig, ax = plt.subplots(figsize=(8, 4.5))
    df["tenure_binned"] = pd.cut(df["tenure_months"], bins=[0, 6, 12, 24, 36, 48, 72], labels=["0-6m", "7-12m", "13-24m", "25-36m", "37-48m", "49-72m"])
    tenure_rates = df.groupby("tenure_binned", observed=False)["churn"].mean() * 100
    ax.plot(tenure_rates.index, tenure_rates.values, marker="o", color="#d9534f", linewidth=2.5, markersize=8)
    ax.fill_between(range(len(tenure_rates)), tenure_rates.values, color="#d9534f", alpha=0.15)
    ax.set_title("Churn Rate Decay Across Tenure Lifecycle (Key Driver #2)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Tenure Cohort")
    for i, v in enumerate(tenure_rates.values):
        ax.text(i, v + 1.8, f"{v:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "churn_vs_tenure.png"))
    plt.close()

    # 4. Churn vs Monthly Charges
    fig, ax = plt.subplots(figsize=(7, 4.5))
    df["charge_band"] = pd.cut(df["monthly_charges"], bins=[0, 45, 75, 105, 150], labels=["Low (<$45)", "Mid ($45-$75)", "High ($75-$105)", "Ultra ($105+)"])
    charge_rates = df.groupby("charge_band", observed=False)["churn"].mean() * 100
    bars = ax.bar(charge_rates.index, charge_rates.values, color="#337ab7", width=0.5)
    ax.set_title("Churn Rate by Monthly Spending Band", fontsize=12, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"{h:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "churn_vs_monthly_charges.png"))
    plt.close()

    # 5. Churn vs Customer Satisfaction
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sat_rates = df.groupby("satisfaction_score")["churn"].mean() * 100
    colors = ["#d9534f", "#d9534f", "#f0ad4e", "#5cb85c", "#5cb85c"]
    bars = ax.bar(sat_rates.index.astype(str), sat_rates.values, color=colors, width=0.5)
    ax.set_title("Churn Rate vs Customer CSAT Score (1=Very Dissatisfied, 5=Delighted)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Satisfaction Score")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"{h:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "churn_vs_satisfaction.png"))
    plt.close()

    # 6. Churn vs Support Tickets
    fig, ax = plt.subplots(figsize=(8, 4.5))
    df["ticket_group"] = pd.cut(df["support_tickets"], bins=[-1, 0, 2, 4, 7, 15], labels=["0 tickets", "1-2 tickets", "3-4 tickets", "5-7 tickets", "8+ tickets"])
    ticket_rates = df.groupby("ticket_group", observed=False)["churn"].mean() * 100
    bars = ax.bar(ticket_rates.index, ticket_rates.values, color="#e74c3c", width=0.5)
    ax.set_title("Churn Rate Escalation by Support Ticket Volume", fontsize=12, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"{h:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "churn_vs_support_tickets.png"))
    plt.close()

    # 7. Churn by Customer Segment
    fig, ax = plt.subplots(figsize=(7, 4.5))
    seg_rates = df.groupby("customer_segment")["churn"].mean() * 100
    bars = ax.bar(seg_rates.index, seg_rates.values, color="#8e44ad", width=0.45)
    ax.set_title("Churn Rate by Customer Segment", fontsize=12, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"{h:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "churn_by_customer_segment.png"))
    plt.close()

    # 8. Correlation Matrix of Numerical Features
    fig, ax = plt.subplots(figsize=(10, 8))
    num_cols = [
        "tenure_months", "monthly_charges", "total_charges", "login_frequency",
        "support_tickets", "complaints", "last_activity_days", "usage_frequency",
        "satisfaction_score", "service_issues", "response_time", "churn"
    ]
    corr = df[num_cols].corr()
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(num_cols, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation Matrix of Key Numerical Variables with Churn", fontsize=12, fontweight="bold")
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            val = corr.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if abs(val) > 0.4 else "black", fontsize=7.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "correlation_matrix.png"))
    plt.close()

    print(f"Successfully generated all 8 EDA figures in {fig_dir}")

if __name__ == "__main__":
    run_eda()
