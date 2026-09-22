"""Read-only educational dashboard built solely from saved project artifacts."""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard_artifacts import load_dashboard_artifacts


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Credit Risk Analytics", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .stApp { background: #f5f8fc; color: #172b4d; }
    .block-container { max-width: 1440px; padding-top: 2.3rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #102a43 0%, #163e5a 100%); }
    [data-testid="stSidebar"] * { color: #eef6fa; }
    [data-testid="stSidebar"] [data-testid="stAlert"] * { color: #172b4d; }
    [data-testid="stSidebar"] [role="radiogroup"] label { background: rgba(255,255,255,.06); border-radius: 8px; margin: .22rem 0; padding: .35rem .45rem; }
    [data-testid="stMetric"] { background: #ffffff; border: 1px solid #e1e8f0; border-top: 4px solid #0f766e; padding: 1rem; border-radius: 12px; box-shadow: 0 2px 8px rgba(15, 39, 66, .05); }
    [data-testid="stMetricLabel"] { color: #52657d; font-size: .86rem; }
    [data-testid="stMetricValue"] { color: #102a43; }
    h1, h2, h3 { color: #102a43; letter-spacing: -.02em; }
    .portfolio-kicker { color: #0f766e; font-size: .9rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
    .portfolio-hero { background: linear-gradient(115deg, #102a43, #174e63); border-radius: 16px; color: #ffffff; padding: 1.7rem 2rem; margin: .35rem 0 1.6rem; }
    .portfolio-hero h1 { color: #ffffff; margin: 0; }
    .portfolio-hero p { color: #d9edf5; margin: .55rem 0 0; font-size: 1.05rem; }
    .section-note { color: #52657d; font-size: .95rem; }
    .summary-card { background: #ffffff; border: 1px solid #e1e8f0; border-radius: 13px; padding: 1rem 1.1rem; min-height: 108px; box-shadow: 0 2px 8px rgba(15, 39, 66, .04); }
    .summary-card .eyebrow { color: #0f766e; font-size: .72rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
    .summary-card .summary-title { color: #102a43; font-size: 1.1rem; font-weight: 750; margin-top: .35rem; }
    .summary-card .summary-detail { color: #52657d; font-size: .9rem; margin-top: .3rem; }
    .kpi-card { background: #ffffff; border: 1px solid #e1e8f0; border-radius: 13px; border-top: 4px solid #0f766e; padding: 1rem; min-height: 116px; box-shadow: 0 2px 8px rgba(15, 39, 66, .05); }
    .kpi-card .label { color: #52657d; font-size: .8rem; font-weight: 700; text-transform: uppercase; letter-spacing: .03em; }
    .kpi-card .value { color: #102a43; font-size: 2rem; line-height: 1.2; font-weight: 750; margin-top: .35rem; }
    .kpi-card .detail { color: #64748b; font-size: .78rem; margin-top: .35rem; }
    .page-banner { background: #e7f4f1; border-left: 5px solid #0f766e; color: #173f4f; border-radius: 8px; padding: .8rem 1rem; margin: 0 0 1.25rem; }
    .journey-step { border-left: 2px solid #68b6a7; padding: .1rem 0 .7rem 1rem; margin-left: .5rem; color: #52657d; }
    .journey-step strong { color: #102a43; }
    [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid #dbe5ed; }
    [data-baseweb="tab"] { background: #edf3f7; border-radius: 8px 8px 0 0; height: 42px; padding: 0 14px; font-weight: 600; color: #334e68; }
    [aria-selected="true"][data-baseweb="tab"] { background: #dff4ee; color: #0f766e; }
    [data-testid="stDataFrame"] { border: 1px solid #e1e8f0; border-radius: 10px; overflow: hidden; }
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


def summary_card(eyebrow, title, detail):
    """Render a presentation-only card from already saved, non-sensitive artifacts."""
    st.markdown(
        f'<div class="summary-card"><div class="eyebrow">{eyebrow}</div>'
        f'<div class="summary-title">{title}</div><div class="summary-detail">{detail}</div></div>',
        unsafe_allow_html=True,
    )


def kpi_card(label, value, detail):
    """Render a display card; this dashboard never calculates a new risk score."""
    st.markdown(
        f'<div class="kpi-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="detail">{detail}</div></div>',
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown("## ◈ RiskLens")
    st.caption("Credit-risk analytics portfolio")
    st.divider()
    page = st.radio(
        "Explore the project",
        [
            "⌂  Overview", "◫  EDA & patterns", "⌁  Model & calibration",
            "◎  Review simulation", "✦  Explainability", "◌  Fairness & use", "≡  Documentation",
        ],
        label_visibility="visible",
    )
    st.divider()
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

st.markdown("""
<div class="portfolio-kicker">Educational portfolio dashboard</div>
<div class="portfolio-hero">
  <h1>Credit Risk Analytics &amp; Explainable Default Prediction</h1>
  <p>Historical-data analysis • calibrated risk ranking • human-review simulation only</p>
</div>
""", unsafe_allow_html=True)

if page == "⌂  Overview":
    st.header("What this project demonstrates")
    st.markdown('<div class="page-banner"><strong>Portfolio lens:</strong> a reproducible, historical analysis of how a fixed review team could prioritize a limited queue—not a tool for automated lending decisions.</div>', unsafe_allow_html=True)
    st.write("The project estimates and ranks the historical likelihood of a recorded next-month default-payment outcome so a limited manual-review queue can be simulated. It is not a current bank model or a lending policy.")
    st.info("Dataset: 30,000 historical Taiwan credit-card records; amounts are NT$; repayment, bill, and payment history cover April–September 2005. `Y=1` means recorded default payment next month.")
    st.subheader("Project workflow")
    journey_left, journey_right = st.columns(2)
    with journey_left:
        st.markdown('<div class="journey-step"><strong>01 · Understand the portfolio</strong><br>EDA and documented data-quality checks.</div>', unsafe_allow_html=True)
        st.markdown('<div class="journey-step"><strong>02 · Compare feature sets</strong><br>Same split, same candidates, validation-led selection.</div>', unsafe_allow_html=True)
        st.markdown('<div class="journey-step"><strong>03 · Calibrate probabilities</strong><br>Training-only calibration; validation chose the method.</div>', unsafe_allow_html=True)
    with journey_right:
        st.markdown('<div class="journey-step"><strong>04 · Simulate limited review</strong><br>Ranked top-k capacity policy, not a universal cutoff.</div>', unsafe_allow_html=True)
        st.markdown('<div class="journey-step"><strong>05 · Explain associations</strong><br>Global and de-identified local Kernel SHAP.</div>', unsafe_allow_html=True)
        st.markdown('<div class="journey-step"><strong>06 · Audit responsible use</strong><br>Held-out subgroup diagnostics and documented limitations.</div>', unsafe_allow_html=True)
    st.caption("Each stage uses saved artifacts. This dashboard does not recalculate, retrain, or score any records.")
    st.subheader("Final educational setup")
    st.markdown("Frozen baseline **XGBoost** using `X1` and `X6`–`X23` → **isotonic calibration** → retrospective **top-10% ranked review queue**.")
    st.caption("A compact end-to-end workflow designed for clear portfolio communication—not operational lending.")
    overview_cards = st.columns(3)
    with overview_cards[0]:
        summary_card("01 · Model", "Frozen baseline XGBoost", "19 raw historical account fields; no demographics in modelling.")
    with overview_cards[1]:
        summary_card("02 · Probability", "Isotonic calibration", "Selected on validation data before one held-out test check.")
    with overview_cards[2]:
        summary_card("03 · Review simulation", "Top 10% ranked queue", "A fixed capacity demonstration, not a business decision rule.")
    calibration = require_json("calibration_threshold_metrics.json")
    if calibration:
        method = calibration["methods"][calibration["selected_calibration"]]["test"]
        top10 = review_row(calibration.get("final_test_review_capacity"))
        cols = st.columns(5)
        with cols[0]: kpi_card("Test ROC-AUC", f"{method['classification_metrics_at_0_50']['roc_auc']:.4f}", "Ranking quality")
        with cols[1]: kpi_card("Average precision", f"{method['classification_metrics_at_0_50']['average_precision']:.4f}", "Default-focused ranking")
        with cols[2]: kpi_card("Brier score", f"{method['calibration']['brier_score']:.4f}", "Lower is better")
        if top10:
            with cols[3]: kpi_card("Top-10% capture", f"{top10['recall_of_observed_defaults']:.2%}", "Recorded historical defaults")
            with cols[4]: kpi_card("Accounts reviewed", f"{top10['accounts_reviewed']:,}", "Of 6,000 held-out accounts")
            st.warning(f"In this held-out historical simulation, {top10['observed_defaults_captured']:,} observed defaults appeared in the {top10['accounts_reviewed']:,}-account review queue. This does not mean defaults were prevented or savings were achieved.")

if page == "◫  EDA & patterns":
    st.header("EDA & portfolio patterns")
    st.markdown('<div class="page-banner"><strong>Question:</strong> What did the historical portfolio look like before any model was trained?</div>', unsafe_allow_html=True)
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

if page == "⌁  Model & calibration":
    st.header("Model performance & calibration")
    st.markdown('<div class="page-banner"><strong>Question:</strong> Does the model rank recorded defaults well, and are its probability estimates sensible?</div>', unsafe_allow_html=True)
    comparison = require_json("model_comparison_metrics.json")
    if comparison:
        baseline = comparison["baseline"]
        engineered = comparison["engineered"]
        table = pd.DataFrame([
            {"Feature set": "Baseline", "Feature count": baseline["feature_count"], "Selected model": baseline["selected_model"].upper(), "Validation ROC-AUC": baseline["validation_selected"]["roc_auc"], "Test ROC-AUC": baseline["test_selected"]["roc_auc"], "Test average precision": baseline["test_selected"]["average_precision"], "Test Brier score": baseline["test_selected"]["brier_score"]},
            {"Feature set": "Engineered", "Feature count": engineered["feature_count"], "Selected model": engineered["selected_model"].upper(), "Validation ROC-AUC": engineered["validation_selected"]["roc_auc"], "Test ROC-AUC": engineered["test_selected"]["roc_auc"], "Test average precision": engineered["test_selected"]["average_precision"], "Test Brier score": engineered["test_selected"]["brier_score"]},
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

if page == "◎  Review simulation":
    st.header("Review-capacity simulation")
    st.markdown('<div class="page-banner"><strong>Question:</strong> With a limited review team, how many recorded historical defaults appear in the highest-ranked accounts?</div>', unsafe_allow_html=True)
    st.caption("A top-k policy is shown because a fixed review team capacity is easier to interpret than treating 0.50 as a universal business threshold.")
    calibration = require_json("calibration_threshold_metrics.json")
    if calibration:
        rows = calibration.get("final_test_review_capacity", [])
        review = pd.DataFrame(rows)
        if not review.empty:
            display = review[["review_fraction", "accounts_reviewed", "observed_defaults_captured", "recall_of_observed_defaults", "precision_among_reviewed"]].copy()
            display.columns = ["Review capacity", "Accounts reviewed", "Historical defaults captured", "Default capture", "Precision in queue"]
            display["Review capacity"] = display["Review capacity"].map(lambda value: f"Top {value:.0%}")
            st.dataframe(display.style.format({"Default capture": "{:.2%}", "Precision in queue": "{:.2%}"}), hide_index=True, use_container_width=True)
        top10 = review_row(rows)
        if top10:
            st.success(f"Demonstration policy: top 10% ranked accounts — {top10['accounts_reviewed']:,} reviewed; {top10['observed_defaults_captured']:,} historical defaults captured; {top10['recall_of_observed_defaults']:.2%} capture.")
        show_figure("13_review_capacity_capture.png", "Observed-default capture by capacity")
        st.subheader("Illustrative classroom cost sensitivity")
        st.write("These validation-only examples use transparent units: false positive × 1 plus false negative × 5 or × 10. They are not real banking costs, savings, or a rule for selecting a threshold.")
        thresholds = pd.DataFrame(calibration.get("validation_thresholds", []))
        if not thresholds.empty:
            threshold_display = thresholds[["threshold", "accounts_flagged", "precision", "recall", "false_positives", "false_negatives", "illustrative_cost_missed_5x_review", "illustrative_cost_missed_10x_review"]].copy()
            threshold_display.columns = ["Threshold", "Accounts flagged", "Precision", "Recall / capture", "False positives", "Missed defaults", "Illustrative cost (5×)", "Illustrative cost (10×)"]
            with st.expander("View the validation threshold sensitivity table"):
                st.dataframe(threshold_display.style.format({"Threshold": "{:.2f}", "Precision": "{:.2%}", "Recall / capture": "{:.2%}"}), hide_index=True, use_container_width=True)
        show_figure("12_validation_threshold_tradeoff.png", "Validation threshold trade-offs")
        show_figure("14_validation_cost_sensitivity.png", "Illustrative validation cost sensitivity")

if page == "✦  Explainability":
    st.header("Explainability")
    st.markdown('<div class="page-banner"><strong>Question:</strong> Which historical account fields most influenced the frozen model's learned associations?</div>', unsafe_allow_html=True)
    st.caption("SHAP explains learned model associations in the frozen XGBoost model. It is not causality, certainty, fairness proof, or a decision instruction.")
    st.info("Feature guide: `X6` is the latest recorded repayment-status code, `X1` is granted credit, `X12`–`X17` are bill amounts, and `X18`–`X23` are payment amounts.")
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
            cols[1].metric("Review status", "Top-10% queue" if case["review_queue_status"] == "top 10% review queue" else "Outside top-10% queue")
            cols[2].metric("Recorded outcome", "Default" if case["actual_historical_outcome"] == "recorded default next month" else "No recorded default")
            show_figure(f"17_shap_local_{name}.png", "De-identified local SHAP waterfall: raw-model contributions only")
    st.warning("No person can be identified here. These three fixed examples are teaching cases, not profiles to copy into a real decision process.")

if page == "◌  Fairness & use":
    st.header("Fairness & responsible use")
    st.markdown('<div class="page-banner"><strong>Question:</strong> Do held-out performance and review-capture patterns differ across recorded subgroups?</div>', unsafe_allow_html=True)
    st.write("Recorded sex, education, marital-status, and age were excluded from training, calibration, SHAP, thresholds, and ranking. They were used only as held-out audit labels.")
    left, right = st.columns(2)
    with left: show_figure("18_fairness_subgroup_sizes_default_rates.png", "Held-out subgroup sizes and historical default rates")
    with right: show_figure("19_fairness_performance_calibration_comparison.png", "Held-out subgroup performance and calibration diagnostics")
    show_figure("20_fairness_top10_review_capture.png", "Capture within the fixed global top-10% queue")
    st.warning("Key finding to investigate: among sufficiently sized documented education groups, ROC-AUC ranged from 0.750 to 0.798 and Brier score from 0.116 to 0.159. This is not a fairness verdict.")
    st.error("Excluding a demographic field does not automatically prove fairness: account variables can act as proxies, and historical outcomes can reflect prior policies and broader social or economic conditions.")
    st.subheader("Responsible-use checklist")
    st.markdown("- Human review before any action\n- Monitoring for performance and calibration drift\n- Escalation and appeal routes\n- Governance, privacy, and legal review\n- No automated approval, rejection, pricing, collections, or lending decision")

if page == "≡  Documentation":
    st.header("Documentation")
    st.markdown('<div class="page-banner"><strong>Audit trail:</strong> each report below records the project choices, evidence, and limitations.</div>', unsafe_allow_html=True)
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
