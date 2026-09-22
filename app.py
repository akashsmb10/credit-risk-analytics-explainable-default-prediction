"""Read-only educational dashboard built solely from saved project artifacts."""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard_artifacts import load_dashboard_artifacts


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Credit Risk Analytics", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .stApp { background: #f8fafc; }
    [data-testid="stMetric"] { background: #ffffff; border-left: 4px solid #0f766e; padding: 12px; border-radius: 8px; }
    h1, h2, h3 { color: #0f2742; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def artifacts(root_string):
    return load_dashboard_artifacts(Path(root_string))


ARTIFACTS = artifacts(str(ROOT))
JSON = ARTIFACTS["json"]
MARKDOWN = ARTIFACTS["markdown"]
FIGURES = ARTIFACTS["figures_dir"]


def require_json(name):
    value = JSON.get(name)
    if value is None:
        st.warning(f"Saved artifact `{name}` is unavailable. Run its analysis module to restore this view.")
    return value


def show_figure(filename, caption):
    path = FIGURES / filename
    if path.is_file():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Figure unavailable: `{filename}`. Generate the corresponding report artifact to restore it.")


def review_row(rows, fraction=0.10):
    return next((row for row in rows or [] if abs(row.get("review_fraction", -1) - fraction) < 1e-9), None)


with st.sidebar:
    st.header("Educational use only")
    st.warning("Historical UCI Taiwan credit-card data from 2005. This dashboard is retrospective and read-only.")
    st.markdown(
        "**It does not:**\n\n"
        "- accept customer data\n"
        "- predict or rank new people\n"
        "- approve, reject, price, or collect\n"
        "- show IDs, demographics, or individual records"
    )
    st.caption("Captured defaults are historical observed outcomes—not prevented defaults or financial savings.")

st.title("Credit Risk Analytics & Explainable Default Prediction")
st.caption("Read-only educational portfolio dashboard • historical data • no automated decisioning")

tabs = st.tabs([
    "Project Overview", "EDA & Portfolio Patterns", "Model Performance & Calibration",
    "Review-Capacity Simulation", "Explainability", "Fairness & Responsible Use", "Documentation",
])

with tabs[0]:
    st.header("What this project demonstrates")
    st.write("The project estimates and ranks the historical likelihood of a recorded next-month default-payment outcome so a limited manual-review queue can be simulated. It is not a current bank model or a lending policy.")
    st.info("Dataset: 30,000 historical Taiwan credit-card records; amounts are NT$; repayment, bill, and payment history cover April–September 2005. `Y=1` means recorded default payment next month.")
    st.subheader("Project workflow")
    st.markdown("**Data → EDA → model comparison → calibration → review-capacity simulation → SHAP explanations → held-out fairness audit**")
    st.caption("Each stage uses saved artifacts. This dashboard does not recalculate, retrain, or score any records.")
    st.subheader("Final educational setup")
    st.markdown("Frozen baseline **XGBoost** using `X1` and `X6`–`X23` → **isotonic calibration** → retrospective **top-10% ranked review queue**.")
    calibration = require_json("calibration_threshold_metrics.json")
    if calibration:
        method = calibration["methods"][calibration["selected_calibration"]]["test"]
        top10 = review_row(calibration.get("final_test_review_capacity"))
        cols = st.columns(5)
        cols[0].metric("Test ROC-AUC", f"{method['classification_metrics_at_0_50']['roc_auc']:.4f}")
        cols[1].metric("Test average precision", f"{method['classification_metrics_at_0_50']['average_precision']:.4f}")
        cols[2].metric("Test Brier score", f"{method['calibration']['brier_score']:.4f}")
        if top10:
            cols[3].metric("Top-10% capture", f"{top10['recall_of_observed_defaults']:.2%}")
            cols[4].metric("Accounts reviewed", f"{top10['accounts_reviewed']:,}")
            st.warning(f"In this held-out historical simulation, {top10['observed_defaults_captured']:,} observed defaults appeared in the {top10['accounts_reviewed']:,}-account review queue. This does not mean defaults were prevented or savings were achieved.")

with tabs[1]:
    st.header("EDA & portfolio patterns")
    st.caption("These charts describe historical associations in the supplied records. They do not demonstrate causation.")
    left, right = st.columns(2)
    with left:
        show_figure("01_target_class_balance.png", "Recorded next-month default-payment outcome balance")
        show_figure("03_default_rate_credit_limit_band.png", "Historical default rate by descriptive credit-limit band")
    with right:
        show_figure("02_credit_limit_distribution.png", "Granted-credit distributions")
        show_figure("04_default_rate_by_repayment_status.png", "Historical default rate by recorded repayment-status code")
    show_figure("09_default_group_comparison.png", "Selected group comparisons in the historical data")
    st.subheader("Verified findings")
    st.markdown("""
    - 6,636 of 30,000 records (22.12%) have a recorded next-month default-payment outcome.
    - Granted credit is right-skewed: median NT$140,000 and mean NT$167,484.
    - Recorded default rates decline across the descriptive credit-limit bands, but this is association—not evidence that changing a limit changes risk.
    - Latest documented positive repayment-delay codes are strongly associated with higher recorded default rates.
    - Bill and payment amounts are highly skewed, so medians and distributions matter alongside means.
    - Monthly history fields are related time-series blocks; adding many similar variables does not automatically add model value.
    """)
    st.error("Cautions: correlation is not causation. UCI documentation does not define repayment-status codes `0` and `-2`, so this project does not label them as on-time or otherwise infer their meaning. Same-month bills and payments are not treated as a repayment ratio or current balance.")

with tabs[2]:
    st.header("Model performance & calibration")
    comparison = require_json("model_comparison_metrics.json")
    if comparison:
        baseline = comparison["baseline"]
        engineered = comparison["engineered"]
        table = pd.DataFrame([
            {"Feature set": "Baseline", "Features": baseline["feature_count"], "Selected model": baseline["selected_model"], "Validation ROC-AUC": baseline["validation_selected"]["roc_auc"], "Test ROC-AUC": baseline["test_selected"]["roc_auc"], "Test average precision": baseline["test_selected"]["average_precision"], "Test Brier": baseline["test_selected"]["brier_score"]},
            {"Feature set": "Engineered", "Features": engineered["feature_count"], "Selected model": engineered["selected_model"], "Validation ROC-AUC": engineered["validation_selected"]["roc_auc"], "Test ROC-AUC": engineered["test_selected"]["roc_auc"], "Test average precision": engineered["test_selected"]["average_precision"], "Test Brier": engineered["test_selected"]["brier_score"]},
        ])
        st.dataframe(table.style.format({column: "{:.4f}" for column in table.columns[3:]}), use_container_width=True, hide_index=True)
        st.info("The engineered feature set improved validation ROC-AUC by only 0.0008, below the pre-specified 0.005 materiality margin. The simpler baseline therefore remained final.")
        show_figure("10_baseline_vs_engineered_comparison.png", "Fair baseline-versus-engineered comparison")
    calibration = require_json("calibration_threshold_metrics.json")
    if calibration:
        selected = calibration["selected_calibration"]
        st.subheader("Probability calibration")
        st.write(f"Selected method: **{selected.title()}**. It was selected from validation Brier score, with ECE as a tie-breaker; test results did not choose it.")
        show_figure("11_calibration_reliability_curves.png", "Validation and held-out test reliability curves")
    c1, c2 = st.columns(2)
    with c1:
        with st.expander("Ranking quality: ROC-AUC and average precision"):
            st.write("ROC-AUC measures how often observed defaults receive higher scores than non-defaults across possible cutoffs. Average precision focuses more on performance for the less-common default outcome. Neither proves probabilities are numerically accurate.")
    with c2:
        with st.expander("Probability calibration: Brier score and reliability"):
            st.write("Brier score is average squared probability error; lower is better. A reliability curve compares average predicted probability with the observed rate in score bins. Calibration concerns probability levels, while ROC-AUC concerns ranking.")

with tabs[3]:
    st.header("Review-capacity simulation")
    st.caption("A top-k policy is shown because a fixed review team capacity is easier to interpret than treating 0.50 as a universal business threshold.")
    calibration = require_json("calibration_threshold_metrics.json")
    if calibration:
        rows = calibration.get("final_test_review_capacity", [])
        review = pd.DataFrame(rows)
        if not review.empty:
            display = review[["review_fraction", "accounts_reviewed", "observed_defaults_captured", "recall_of_observed_defaults", "precision_among_reviewed"]].copy()
            display["review_fraction"] = display["review_fraction"].map(lambda value: f"Top {value:.0%}")
            st.dataframe(display.style.format({"recall_of_observed_defaults": "{:.2%}", "precision_among_reviewed": "{:.2%}"}), hide_index=True, use_container_width=True)
        top10 = review_row(rows)
        if top10:
            st.success(f"Demonstration policy: top 10% ranked accounts — {top10['accounts_reviewed']:,} reviewed; {top10['observed_defaults_captured']:,} historical defaults captured; {top10['recall_of_observed_defaults']:.2%} capture.")
        show_figure("13_review_capacity_capture.png", "Observed-default capture by capacity")
        st.subheader("Illustrative classroom cost sensitivity")
        st.write("These validation-only examples use transparent units: false positive × 1 plus false negative × 5 or × 10. They are not real banking costs, savings, or a rule for selecting a threshold.")
        thresholds = pd.DataFrame(calibration.get("validation_thresholds", []))
        if not thresholds.empty:
            st.dataframe(thresholds[["threshold", "accounts_flagged", "precision", "recall", "false_positives", "false_negatives", "illustrative_cost_missed_5x_review", "illustrative_cost_missed_10x_review"]].style.format({"threshold": "{:.2f}", "precision": "{:.2%}", "recall": "{:.2%}"}), hide_index=True, use_container_width=True)
        show_figure("12_validation_threshold_tradeoff.png", "Validation threshold trade-offs")
        show_figure("14_validation_cost_sensitivity.png", "Illustrative validation cost sensitivity")

with tabs[4]:
    st.header("Explainability")
    st.caption("SHAP explains learned model associations in the frozen XGBoost model. It is not causality, certainty, fairness proof, or a decision instruction.")
    left, right = st.columns(2)
    with left: show_figure("15_shap_global_beeswarm.png", "Global SHAP beeswarm: direction and size of sampled raw-model contributions")
    with right: show_figure("16_shap_global_mean_absolute_bar.png", "Global mean absolute SHAP values")
    with st.expander("Global versus local SHAP"):
        st.write("Global SHAP averages how much each feature changed the model output across a sample. Local SHAP shows the inputs that pushed one saved historical example’s raw model score up or down. Isotonic calibration changes displayed probability, not the underlying XGBoost feature contributions.")
    metadata = require_json("shap_explainability_metrics.json")
    if metadata:
        cases = metadata.get("local_cases", [])
        case_titles = {
            "high_risk_recorded_default": "High-risk historical default",
            "high_risk_no_recorded_default": "High-risk historical non-default",
            "lower_risk_recorded_default": "Lower-risk missed historical default",
        }
        for case in cases:
            name = case["case"]
            st.subheader(case_titles.get(name, name.replace("_", " ").title()))
            cols = st.columns(3)
            cols[0].metric("Calibrated historical risk", f"{case['calibrated_risk_probability']:.2%}")
            cols[1].metric("Review status", "Top-10% queue" if "top 10%" in case["review_queue_status"] else "Outside queue")
            cols[2].metric("Recorded outcome", "Default" if "recorded default" in case["actual_historical_outcome"] else "No recorded default")
            show_figure(f"17_shap_local_{name}.png", "De-identified local SHAP waterfall: raw-model contributions only")
    st.warning("No person can be identified here. These three fixed examples are teaching cases, not profiles to copy into a real decision process.")

with tabs[5]:
    st.header("Fairness & responsible use")
    st.write("Recorded sex, education, marital-status, and age were excluded from training, calibration, SHAP, thresholds, and ranking. They were used only as held-out audit labels.")
    left, right = st.columns(2)
    with left: show_figure("18_fairness_subgroup_sizes_default_rates.png", "Held-out subgroup sizes and historical default rates")
    with right: show_figure("19_fairness_performance_calibration_comparison.png", "Held-out subgroup performance and calibration diagnostics")
    show_figure("20_fairness_top10_review_capture.png", "Capture within the fixed global top-10% queue")
    st.warning("Key finding to investigate: among sufficiently sized documented education groups, ROC-AUC ranged from 0.750 to 0.798 and Brier score from 0.116 to 0.159. This is not a fairness verdict.")
    st.error("Excluding a demographic field does not automatically prove fairness: account variables can act as proxies, and historical outcomes can reflect prior policies and broader social or economic conditions.")
    st.subheader("Responsible-use checklist")
    st.markdown("- Human review before any action\n- Monitoring for performance and calibration drift\n- Escalation and appeal routes\n- Governance, privacy, and legal review\n- No automated approval, rejection, pricing, collections, or lending decision")

with tabs[6]:
    st.header("Documentation")
    st.caption("These saved artifacts document the analysis; no live data or prediction service is connected.")
    docs = {
        "Model card": "model_card.md", "Data dictionary": "data_dictionary.md",
        "Feature-engineering notes": "feature_engineering_notes.md", "Calibration report": "calibration_threshold_report.md",
        "SHAP report": "shap_explainability_report.md", "Fairness audit": "fairness_audit_report.md",
    }
    for title, filename in docs.items():
        with st.expander(title):
            text = MARKDOWN.get(filename)
            if text is None:
                st.info(f"`reports/{filename}` is unavailable.")
            else:
                st.markdown(text)
    st.subheader("How to run locally")
    st.code("py -3.10 -m pip install -r requirements.txt\nstreamlit run app.py", language="powershell")
    st.caption("The dashboard intentionally contains no input form, prediction endpoint, or customer-level view.")
