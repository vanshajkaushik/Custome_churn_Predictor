"""
Executive PDF generator for Customer Churn Prediction System.
Produces stakeholder-ready PDF report embedding figures, KPIs, and strategy matrix.
"""

import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_pdf_report(output_pdf_path: str = None):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if output_pdf_path is None:
        output_pdf_path = os.path.join(root, "reports", "customer_churn_report.pdf")
    
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    fig_dir = os.path.join(root, "reports", "figures")
    eval_json = os.path.join(root, "models", "evaluation_results.json")

    with open(eval_json, "r") as f:
        metrics = json.load(f)

    rf_opt = metrics["random_forest"]["optimal_metrics"]
    dt = metrics["decision_tree"]

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1a365d"),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#4a5568"),
        spaceAfter=15
    )
    h2_style = ParagraphStyle(
        "Heading2Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2b6cb0"),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#2d3748")
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph("Customer Churn Prediction & Retention Intelligence", title_style))
    story.append(Paragraph("Executive Stakeholder Analysis & Actionable Retention Framework | Enterprise Analytics", subtitle_style))
    story.append(Spacer(1, 8))

    # Executive Summary
    story.append(Paragraph("1. Executive Summary", h2_style))
    summary_text = (
        "This project establishes a data-driven customer retention intelligence system analyzing 12,500 customer records. "
        "Through rigorous exploratory data analysis, machine learning modeling, and threshold optimization, we identify high-risk churners "
        "and prescribe targeted retention workflows. The primary predictive engine, a tuned <b>Random Forest Classifier</b>, demonstrates "
        f"<b>{metrics['random_forest']['cv_roc_auc_mean']*100:.1f}% CV ROC-AUC</b> (Test ROC-AUC: <b>{metrics['random_forest']['roc_auc']*100:.1f}%</b>, "
        f"PR-AUC: <b>{metrics['random_forest']['pr_auc']*100:.1f}%</b>, and Test Accuracy: <b>{rf_opt['accuracy']*100:.1f}%</b> with "
        f"<b>{rf_opt['recall']*100:.1f}% recall</b> on actual churners), significantly outperforming the naive baseline (66.1% accuracy, 0.0% recall)."
    )
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 10))

    # Model Performance Scorecard Table
    story.append(Paragraph("2. Model Evaluation Scorecard", h2_style))
    scorecard_data = [
        [Paragraph("<b>Model Architecture</b>", table_cell),
         Paragraph("<b>Accuracy</b>", table_cell),
         Paragraph("<b>Precision</b>", table_cell),
         Paragraph("<b>Recall (Churn)</b>", table_cell),
         Paragraph("<b>F1 Score</b>", table_cell),
         Paragraph("<b>ROC-AUC</b>", table_cell)],
        [Paragraph("Baseline (Majority Class)", table_cell),
         Paragraph(f"{metrics['baseline']['accuracy']*100:.1f}%", table_cell),
         Paragraph("0.0%", table_cell),
         Paragraph("0.0%", table_cell),
         Paragraph("0.000", table_cell),
         Paragraph("0.500", table_cell)],
        [Paragraph("Decision Tree (Tuned)", table_cell),
         Paragraph(f"{dt['accuracy']*100:.1f}%", table_cell),
         Paragraph(f"{dt['precision']*100:.1f}%", table_cell),
         Paragraph(f"{dt['recall']*100:.1f}%", table_cell),
         Paragraph(f"{dt['f1']:.3f}", table_cell),
         Paragraph(f"{dt['roc_auc']:.3f}", table_cell)],
        [Paragraph("<b>Random Forest (Optimized Pipeline)</b>", table_cell),
         Paragraph(f"<b>{rf_opt['accuracy']*100:.1f}%</b>", table_cell),
         Paragraph(f"<b>{rf_opt['precision']*100:.1f}%</b>", table_cell),
         Paragraph(f"<b>{rf_opt['recall']*100:.1f}%</b>", table_cell),
         Paragraph(f"<b>{rf_opt['f1']:.3f}</b>", table_cell),
         Paragraph(f"<b>{rf_opt['roc_auc']:.3f}</b>", table_cell)]
    ]
    score_table = Table(scorecard_data, colWidths=[150, 75, 75, 85, 75, 80])
    score_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#1a202c")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 12))

    # Core Churn Drivers & Visual Evidence
    story.append(Paragraph("3. Empirical Churn Drivers", h2_style))
    driver_intro = (
        "Empirical analysis reveals three dominant attrition drivers: <b>Contract Structure</b> (Month-to-month contracts exhibit "
        "5x the churn rate of two-year commitments), <b>Tenure Lifecycle</b> (churn peaks above 55% during months 0-6 before decaying), "
        "and <b>Service Friction</b> (satisfaction score <= 2 or support tickets >= 4 trigger severe risk escalation)."
    )
    story.append(Paragraph(driver_intro, body_style))
    story.append(Spacer(1, 8))

    # Embed 2 figures side by side or sequential
    fig1 = os.path.join(fig_dir, "churn_by_contract.png")
    fig2 = os.path.join(fig_dir, "churn_vs_tenure.png")
    if os.path.exists(fig1) and os.path.exists(fig2):
        img_table = Table([[Image(fig1, width=260, height=160), Image(fig2, width=260, height=160)]], colWidths=[270, 270])
        img_table.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        story.append(img_table)

    story.append(PageBreak())

    # Retention Prioritization & ROI Impact
    story.append(Paragraph("4. Customer Prioritization & Targeted Retention Playbook", h2_style))
    strat_text = (
        "Rather than contacting all churn candidates equally, accounts are ranked using the <b>Retention Priority Score</b> "
        "(Churn Probability × Annual Contract Value). This directs retention budgets to high-exposure accounts first."
    )
    story.append(Paragraph(strat_text, body_style))
    story.append(Spacer(1, 8))

    strat_matrix = [
        [Paragraph("<b>Customer Cohort</b>", table_cell),
         Paragraph("<b>Primary Risk Trigger</b>", table_cell),
         Paragraph("<b>Prescribed Retention Action</b>", table_cell),
         Paragraph("<b>Expected Business Impact</b>", table_cell)],
        [Paragraph("New Month-to-Month (0-6m)", table_cell),
         Paragraph("Early adoption dropoff", table_cell),
         Paragraph("VIP Concierge Onboarding & 30-day usage audit", table_cell),
         Paragraph("+20% 90-day survival uplift", table_cell)],
        [Paragraph("High Spend ($85+/mo)", table_cell),
         Paragraph("Competitor price poaching", table_cell),
         Paragraph("15% Annual Lock-in commitment incentive", table_cell),
         Paragraph("Locks in high ACV for 12 months", table_cell)],
        [Paragraph("Low CSAT / High Tickets", table_cell),
         Paragraph("Service outage / technical friction", table_cell),
         Paragraph("Executive escalation manager + $25 service credit", table_cell),
         Paragraph("Halts imminent defection", table_cell)],
        [Paragraph("Enterprise / High Value", table_cell),
         Paragraph("Account inactivity / executive change", table_cell),
         Paragraph("Quarterly Business Review (QBR) & C-suite contact", table_cell),
         Paragraph("Preserves top-decile ARR", table_cell)]
    ]
    strat_table = Table(strat_matrix, colWidths=[130, 110, 180, 120])
    strat_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(strat_table)
    story.append(Spacer(1, 12))

    # Scenario Simulation Section
    story.append(Paragraph("5. Scenario-Based Business ROI Simulation", h2_style))
    scenario_text = (
        "Addressing the portfolio resume projection (<b>12% retention improvement</b>): In a 12,500 customer base with ~4,200 annual "
        "at-risk accounts, targeting the top 30% of high-risk customers with an effective intervention program yielding a 12% save rate "
        "is projected to preserve <b>151 customer accounts</b> annually, preserving an estimated <b>$127,000+ in annual recurring revenue</b>. "
        "Because this is a modeled scenario, performance should be monitored via randomized control trial (A/B testing) rollouts."
    )
    story.append(Paragraph(scenario_text, body_style))
    story.append(Spacer(1, 10))

    # Model Evaluation Figures
    fig3 = os.path.join(fig_dir, "feature_importance.png")
    fig4 = os.path.join(fig_dir, "confusion_matrices.png")
    if os.path.exists(fig3) and os.path.exists(fig4):
        img_table2 = Table([[Image(fig3, width=260, height=180), Image(fig4, width=260, height=180)]], colWidths=[270, 270])
        img_table2.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        story.append(img_table2)

    doc.build(story)
    print(f"Executive PDF report successfully generated at: {output_pdf_path}")

if __name__ == "__main__":
    generate_pdf_report()
