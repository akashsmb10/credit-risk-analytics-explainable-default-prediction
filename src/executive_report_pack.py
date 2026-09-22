"""Create presentation-only executive report pages from verified saved artifacts.

The outputs are aggregate historical summaries; this module never scores a new
person, exposes IDs, or recommends lending actions.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loading import load_data


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
NAVY, BLUE, TEAL, ORANGE, SLATE = "#102A43", "#1976D2", "#0F766E", "#E66A32", "#52657D"


def read_json(filename: str) -> dict:
    path = REPORTS / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required saved artifact is missing: {path}. Run the corresponding analysis module first.")
    return json.loads(path.read_text(encoding="utf-8"))


def title(fig: plt.Figure, text: str, subtitle: str) -> None:
    fig.text(0.5, 0.965, text, ha="center", va="top", fontsize=24, fontweight="bold", color=NAVY)
    fig.text(0.5, 0.925, subtitle, ha="center", va="top", fontsize=10, color=SLATE)


def kpi(ax: plt.Axes, label: str, value: str, accent: str = TEAL) -> None:
    ax.axis("off")
    ax.text(0.5, 0.62, value, ha="center", va="center", fontsize=24, fontweight="bold", color=accent)
    ax.text(0.5, 0.30, label, ha="center", va="center", fontsize=10, fontweight="bold", color=NAVY)


def save(fig: plt.Figure, filename: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / filename, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_overview(data: pd.DataFrame, calibration: dict) -> None:
    test = calibration["methods"][calibration["selected_calibration"]]["test"]
    review = calibration["final_test_review_capacity"]
    limit = data.assign(band=pd.cut(data["X1"], [-np.inf, 50_000, 100_000, 200_000, np.inf], labels=["Below\nNT$50k", "NT$50k–\n99,999", "NT$100k–\n199,999", "NT$200k+"]))
    limit_rates = limit.groupby("band", observed=False)["Y"].mean()
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    title(fig, "CREDIT RISK ANALYTICS DASHBOARD", "Historical UCI Taiwan 2005 data • educational, read-only portfolio analysis")
    grid = fig.add_gridspec(3, 4, top=0.86, hspace=0.55, wspace=0.45)
    kpi(fig.add_subplot(grid[0, 0]), "FINAL MODEL", "XGBoost", "#1BCB36")
    kpi(fig.add_subplot(grid[0, 1]), "TEST ROC-AUC", f"{test['classification_metrics_at_0_50']['roc_auc']:.3f}", "#E3342F")
    kpi(fig.add_subplot(grid[0, 2]), "HISTORICAL RECORDS", f"{len(data)/1000:.0f}K", "#1BCB36")
    kpi(fig.add_subplot(grid[0, 3]), "REVIEW CAPACITY", "Top 10%", "#E3342F")
    ax_left = fig.add_subplot(grid[1:, :2])
    bars = ax_left.bar(limit_rates.index.astype(str), 100 * limit_rates.values, color=BLUE)
    ax_left.set(title="Recorded default rate by descriptive credit-limit band", ylabel="Recorded default rate (%)", xlabel="Granted-credit band")
    ax_left.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, limit_rates.values):
        ax_left.text(bar.get_x() + bar.get_width()/2, 100 * value + .5, f"{value:.1%}", ha="center", fontsize=10, fontweight="bold")
    ax_right = fig.add_subplot(grid[1:, 2:])
    labels = [f"Top {row['review_fraction']:.0%}" for row in review]
    capture = [100 * row["recall_of_observed_defaults"] for row in review]
    bars = ax_right.barh(labels, capture, color=["#9CCAC1", TEAL, "#5B9BD5"])
    ax_right.set(title="Observed-default capture by review capacity", xlabel="Captured historical defaults (%)", xlim=(0, 60))
    ax_right.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, capture):
        ax_right.text(value + 1, bar.get_y() + bar.get_height()/2, f"{value:.1f}%", va="center", fontweight="bold")
    fig.text(.5, .025, "Captured defaults are observed historical outcomes—not prevented defaults, savings, or lending decisions.", ha="center", color=SLATE, fontsize=9)
    save(fig, "21_executive_overview.png")


def build_portfolio(data: pd.DataFrame) -> None:
    status = data.groupby("X6")["Y"].agg(["mean", "size"]).reset_index()
    status = status[status["X6"].between(-2, 8)].sort_values("X6")
    payment = data[[f"X{i}" for i in range(18, 24)]].median().rename({f"X{i}": month for i, month in zip(range(18, 24), ["Sep", "Aug", "Jul", "Jun", "May", "Apr"])})
    bill = data[[f"X{i}" for i in range(12, 18)]].median().rename({f"X{i}": month for i, month in zip(range(12, 18), ["Sep", "Aug", "Jul", "Jun", "May", "Apr"])})
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    title(fig, "PORTFOLIO RISK PATTERNS", "Aggregate historical patterns only • association is not causation")
    grid = fig.add_gridspec(2, 2, top=.86, hspace=.48, wspace=.34)
    ax = fig.add_subplot(grid[:, 0])
    bars = ax.bar(status["X6"].astype(str), 100 * status["mean"], color=BLUE)
    ax.set(title="Recorded default rate by latest repayment-status code", xlabel="X6 observed code", ylabel="Recorded default rate (%)")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, row in zip(bars, status.itertuples()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{row.mean:.0%}\nn={row.size:,}", ha="center", fontsize=8)
    ax.text(.5, -.20, "Codes 0 and −2 are retained as undocumented values; no operational meaning is inferred.", transform=ax.transAxes, ha="center", color=SLATE, fontsize=8)
    ax = fig.add_subplot(grid[0, 1])
    ax.barh(bill.index, bill.values, color=TEAL)
    ax.set(title="Median bill amounts by historical month", xlabel="NT$", ylabel="")
    ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(grid[1, 1])
    ax.barh(payment.index, payment.values, color=ORANGE)
    ax.set(title="Median payment amounts by historical month", xlabel="NT$", ylabel="")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "22_executive_portfolio_patterns.png")


def build_model_report(comparison: dict, shap: pd.DataFrame) -> None:
    baseline, engineered = comparison["baseline"], comparison["engineered"]
    rows = [("Baseline", baseline), ("Engineered", engineered)]
    table = [[name, item["feature_count"], item["validation_selected"]["roc_auc"], item["test_selected"]["roc_auc"], item["test_selected"]["average_precision"]] for name, item in rows]
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    title(fig, "MODEL PERFORMANCE", "Fair baseline-versus-engineered comparison • final candidate retained for simplicity")
    grid = fig.add_gridspec(2, 2, top=.86, hspace=.48, wspace=.42)
    ax = fig.add_subplot(grid[0, 0]); ax.axis("off")
    tbl = ax.table(cellText=[[r[0], r[1], f"{r[2]:.3f}", f"{r[3]:.3f}", f"{r[4]:.3f}"] for r in table], colLabels=["Feature set", "Fields", "Validation\nROC-AUC", "Test\nROC-AUC", "Test AP"], cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 1.7)
    for cell in tbl.get_celld().values(): cell.set_edgecolor("white")
    ax.set_title("Validated model comparison", pad=14, fontweight="bold")
    ax = fig.add_subplot(grid[1, 0])
    x = np.arange(2); width = .34
    ax.bar(x - width/2, [r[2] for r in table], width, label="Validation ROC-AUC", color=TEAL)
    ax.bar(x + width/2, [r[3] for r in table], width, label="Test ROC-AUC", color=BLUE)
    ax.set_xticks(x, [r[0] for r in table]); ax.set_ylim(.70, .80); ax.set(title="ROC-AUC comparison", ylabel="ROC-AUC")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(grid[:, 1])
    feature_labels = {"X6": "Latest repayment-status code (X6)", "X1": "Granted credit (X1)", "X19": "Prior payment amount (X19)", "X20": "Prior payment amount (X20)", "X8": "Repayment-status code (X8)"}
    top = shap.head(8).iloc[::-1].copy()
    top["label"] = top["feature"].map(feature_labels).fillna(top["feature"])
    ax.barh(top["label"], top["mean_absolute_shap_value"], color=BLUE)
    ax.set(title="Top learned model associations", xlabel="Mean absolute SHAP value")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.5, .025, "The engineered set improved validation ROC-AUC by 0.0008, below the pre-specified 0.005 materiality margin. SHAP values describe learned associations, not causes.", ha="center", color=SLATE, fontsize=9)
    save(fig, "23_executive_model_performance.png")


def build_review_report(calibration: dict) -> None:
    rows = calibration["final_test_review_capacity"]
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    title(fig, "REVIEW-CAPACITY SIMULATION", "Retrospective classroom scenario • top-k ranking is not a lending or approval policy")
    grid = fig.add_gridspec(2, 2, top=.86, hspace=.50, wspace=.38)
    ax = fig.add_subplot(grid[0, :]); ax.axis("off")
    values = [[f"Top {row['review_fraction']:.0%}", f"{row['accounts_reviewed']:,}", f"{row['observed_defaults_captured']:,}", f"{row['recall_of_observed_defaults']:.2%}", f"{row['precision_among_reviewed']:.2%}"] for row in rows]
    tbl = ax.table(cellText=values, colLabels=["Review capacity", "Accounts reviewed", "Historical defaults\ncaptured", "Default capture", "Precision in queue"], cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.9)
    for cell in tbl.get_celld().values(): cell.set_edgecolor("white")
    ax.set_title("Held-out test review-capacity outcomes", pad=10, fontweight="bold")
    ax = fig.add_subplot(grid[1, 0])
    labels = [f"Top {row['review_fraction']:.0%}" for row in rows]
    ax.bar(labels, [100 * row["recall_of_observed_defaults"] for row in rows], color=["#9CCAC1", TEAL, "#5B9BD5"])
    ax.set(title="Observed-default capture", ylabel="Capture (%)", ylim=(0, 60)); ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(grid[1, 1]); ax.axis("off")
    ax.text(.05, .80, "How to read this page", fontsize=15, fontweight="bold", color=NAVY)
    ax.text(.05, .61, "• Scores are ranked, then a fixed number of accounts is reviewed.\n\n• A larger review queue captures more recorded historical defaults, but requires more reviews.\n\n• This is not evidence of prevented defaults, financial savings, or a universal threshold.", fontsize=11, color=SLATE, va="top", linespacing=1.65)
    save(fig, "24_executive_review_simulation.png")


def main() -> None:
    data = load_data()
    calibration = read_json("calibration_threshold_metrics.json")
    comparison = read_json("model_comparison_metrics.json")
    shap = pd.read_csv(REPORTS / "shap_global_feature_importance.csv")
    build_overview(data, calibration)
    build_portfolio(data)
    build_model_report(comparison, shap)
    build_review_report(calibration)
    print("Created executive report pages: 21–24.")


if __name__ == "__main__":
    main()
