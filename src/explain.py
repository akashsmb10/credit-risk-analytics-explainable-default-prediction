"""Approximate model-agnostic SHAP explanations for the frozen baseline model."""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from calibration_threshold_analysis import DEMONSTRATION_REVIEW_FRACTION, fit_calibrated_models
from compare_feature_sets import make_split_indices
from data_loading import ROOT, load_data
from features import FORBIDDEN_MODEL_COLUMNS, MODEL_FEATURES, select_model_data

RANDOM_STATE = 42
BACKGROUND_SIZE = 40
GLOBAL_SAMPLE_SIZE = 160
KERNEL_NSAMPLES = 300


def stratified_sample(X, y, size, random_state=RANDOM_STATE):
    """Return a reproducible, target-stratified sample without including Y in X."""
    parts = []
    for value in sorted(y.unique()):
        count = min(int(round(size * (y == value).mean())), int((y == value).sum()))
        parts.append(X.loc[y[y == value].sample(n=count, random_state=random_state).index])
    return pd.concat(parts).sample(frac=1, random_state=random_state)


def validate_explanation_inputs(X):
    """Fail unless inputs exactly match frozen training order and permitted fields."""
    if list(X.columns) != MODEL_FEATURES:
        raise ValueError("Explanation inputs must use the exact frozen baseline feature order.")
    if FORBIDDEN_MODEL_COLUMNS.intersection(X.columns):
        raise ValueError("Identifier, demographic, or target column entered explanations.")


def validate_shap_output(values, X):
    values = np.asarray(values)
    if values.shape != X.shape:
        raise ValueError(f"Unexpected SHAP shape {values.shape}; expected {X.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("SHAP values must be finite.")
    return values


def choose_local_cases(y_test, calibrated_probability):
    """Reproducibly choose required queue/outcome cases; no identifiers are returned."""
    ranks = pd.DataFrame({"outcome": y_test, "calibrated_probability": calibrated_probability}, index=y_test.index)
    ranks = ranks.sort_values("calibrated_probability", ascending=False, kind="mergesort")
    ranks["rank"] = np.arange(1, len(ranks) + 1)
    ranks["in_queue"] = ranks["rank"] <= int(np.ceil(len(ranks) * DEMONSTRATION_REVIEW_FRACTION))
    cases = {
        "high_risk_recorded_default": ranks[(ranks.in_queue) & (ranks.outcome == 1)].iloc[0],
        "high_risk_no_recorded_default": ranks[(ranks.in_queue) & (ranks.outcome == 0)].iloc[0],
        "lower_risk_recorded_default": ranks[(~ranks.in_queue) & (ranks.outcome == 1)].sort_values("calibrated_probability", kind="mergesort").iloc[0],
    }
    return {name: (row.name, row) for name, row in cases.items()}


def contribution_table(row, values):
    table = pd.DataFrame({"feature": MODEL_FEATURES, "feature_value": row.to_numpy(), "shap_value_raw_probability": values})
    table["direction"] = np.select(
        [table.shap_value_raw_probability > 0, table.shap_value_raw_probability < 0],
        ["pushes raw model risk up", "pushes raw model risk down"],
        default="no material contribution in this approximation",
    )
    return table.sort_values("shap_value_raw_probability", key=np.abs, ascending=False)


def make_figures(global_X, global_values, expected_value, local_X, local_values, local_cases):
    figures = ROOT / "reports" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(global_values, global_X, show=False, max_display=19)
    plt.title("Global SHAP beeswarm — frozen XGBoost raw probability")
    plt.tight_layout()
    beeswarm = figures / "15_shap_global_beeswarm.png"
    plt.savefig(beeswarm, dpi=200, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(9, 7))
    shap.summary_plot(global_values, global_X, plot_type="bar", show=False, max_display=19)
    plt.title("Global mean absolute SHAP — learned associations, not causes")
    plt.tight_layout()
    bar = figures / "16_shap_global_mean_absolute_bar.png"
    plt.savefig(bar, dpi=200, bbox_inches="tight")
    plt.close()
    waterfalls = {}
    for name, (index, _) in local_cases.items():
        explanation = shap.Explanation(values=local_values.loc[index].to_numpy(), base_values=expected_value, data=local_X.loc[index].to_numpy(), feature_names=MODEL_FEATURES)
        plt.figure(figsize=(10, 7))
        shap.plots.waterfall(explanation, max_display=10, show=False)
        plt.title(name.replace("_", " "))
        plt.tight_layout()
        path = figures / f"17_shap_local_{name}.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        waterfalls[name] = path
    return beeswarm, bar, waterfalls


def report_text(summary, metadata):
    text = [
        "# SHAP explainability report", "", "## Compatibility solution", "",
        "SHAP 0.49.1 TreeExplainer fails with saved XGBoost 3.2.0 because XGBoost serializes `base_score` as a vector string while the SHAP tree loader expects one float. To avoid changing or retraining the frozen model, this project uses SHAP KernelExplainer: a slower, approximate model-agnostic method that repeatedly calls the unchanged model. The background is 40 stratified held-out test records and the global sample is 160 stratified held-out test records.", "",
        "## SHAP from zero", "",
        "The base value is the average raw XGBoost probability over the SHAP background. For one record, positive SHAP values push the raw model probability upward and negative values push it downward. Global SHAP averages contribution sizes over records; local SHAP explains one record. These are learned associations, not causes of default.", "",
        "## Top global associations", "", "| Feature | Mean absolute SHAP | Interpretation |", "| --- | ---: | --- |",
    ]
    meanings = {"X1": "Granted credit amount; association only, not an effect of changing the limit.", "X6": "September repayment-status code; 0 and -2 remain undocumented.", "X7": "August repayment-status code; historical association only.", "X8": "July repayment-status code; historical association only.", "X9": "June repayment-status code; historical association only.", "X19": "August recorded payment amount; association only, not proof of repayment quality.", "X20": "July recorded payment amount; association only, not proof of repayment quality.", "X18": "September recorded payment amount; association only, not proof of repayment quality.", "X12": "September bill-statement amount; not a current-balance measure."}
    for row in summary.head(10).itertuples():
        text.append(f"| {row.feature} | {row.mean_absolute_shap_value:.6f} | {meanings.get(row.feature, 'Historical account amount or status; association only, not causation.')} |")
    text += ["", "## Local cases", "", "The following reproducibly selected held-out cases never display IDs or demographic fields. Their tables show calibrated probability, queue status, historical outcome, and raw-model SHAP directions. A waterfall is not a decision or causal explanation."]
    for row in metadata.itertuples():
        text += ["", f"### {row.case.replace('_', ' ')}", "", f"Calibrated probability: **{row.calibrated_risk_probability:.4f}**; queue status: **{row.review_queue_status}**; historical outcome: **{row.actual_historical_outcome}**.", "The shown inputs pushed this fixed model's raw probability up or down. They do not prove why the outcome occurred and should support human review, not determine an outcome."]
    text += ["", "## Calibration and responsible use", "", "Isotonic calibration maps raw model scores to final displayed probabilities, but does not change the frozen XGBoost trees or their SHAP contributions. SHAP is not causality, fairness proof, regulatory compliance, or evidence of suitability for lending. This is historical Taiwan credit-card data from 2005, not current Indian banking data or a live bank portfolio. Explanations can drift and should help a reviewer ask questions, never approve, deny, price, or collect from a person."]
    return "\n".join(text)


def main():
    np.random.seed(RANDOM_STATE)
    data = load_data()
    X, y = select_model_data(data)
    validate_explanation_inputs(X)
    train_idx, _, test_idx = make_split_indices(y)
    X_train, y_train, X_test, y_test = X.loc[train_idx], y.loc[train_idx], X.loc[test_idx], y.loc[test_idx]
    baseline = joblib.load(ROOT / "models" / "selected_model.joblib")
    # This deterministic, training-only calibration recreation is used only to display the already-selected score scale; no artifact is overwritten.
    isotonic = fit_calibrated_models(X_train, y_train, baseline)["isotonic"]
    calibrated_probability = isotonic.predict_proba(X_test)[:, 1]
    raw_probability = baseline.predict_proba(X_test)[:, 1]
    background = stratified_sample(X_test, y_test, BACKGROUND_SIZE)
    global_X = stratified_sample(X_test, y_test, GLOBAL_SAMPLE_SIZE)
    explainer = shap.KernelExplainer(lambda values: baseline.predict_proba(pd.DataFrame(values, columns=MODEL_FEATURES))[:, 1], background)
    global_values = validate_shap_output(explainer.shap_values(global_X, nsamples=KERNEL_NSAMPLES, silent=True), global_X)
    cases = choose_local_cases(y_test, calibrated_probability)
    local_X = X_test.loc[[index for index, _ in cases.values()]]
    local_values = pd.DataFrame(validate_shap_output(explainer.shap_values(local_X, nsamples=KERNEL_NSAMPLES, silent=True), local_X), index=local_X.index, columns=MODEL_FEATURES)
    summary = pd.DataFrame({"feature": MODEL_FEATURES, "mean_absolute_shap_value": np.abs(global_values).mean(axis=0)}).sort_values("mean_absolute_shap_value", ascending=False)
    reports = ROOT / "reports"
    summary.to_csv(reports / "shap_global_feature_importance.csv", index=False)
    records = []
    for name, (index, rank) in cases.items():
        contribution_table(local_X.loc[index], local_values.loc[index]).to_csv(reports / f"shap_local_{name}_contributions.csv", index=False)
        position = X_test.index.get_loc(index)
        records.append({"case": name, "calibrated_risk_probability": float(calibrated_probability[position]), "raw_xgboost_probability": float(raw_probability[position]), "review_queue_status": "top 10% review queue" if bool(rank.in_queue) else "outside top 10% review queue", "actual_historical_outcome": "recorded default next month" if int(rank.outcome) else "no recorded default next month", "rank_within_held_out_test": int(rank['rank'])})
    metadata = pd.DataFrame(records)
    metadata.to_csv(reports / "shap_local_case_metadata.csv", index=False)
    beeswarm, bar, waterfalls = make_figures(global_X, global_values, explainer.expected_value, local_X, local_values, cases)
    (reports / "shap_explainability_report.md").write_text(report_text(summary, metadata), encoding="utf-8")
    (reports / "shap_explainability_metrics.json").write_text(json.dumps({"compatibility_solution": "KernelExplainer fallback: SHAP 0.49.1 TreeExplainer incompatible with XGBoost 3.2.0 vector base_score", "feature_order": MODEL_FEATURES, "background_size": len(background), "global_sample_size": len(global_X), "kernel_nsamples": KERNEL_NSAMPLES, "base_value_raw_probability": float(explainer.expected_value), "top_10_global_features": summary.head(10).to_dict(orient="records"), "local_cases": records, "figures": [str(path.relative_to(ROOT)) for path in [beeswarm, bar, *waterfalls.values()]]}, indent=2), encoding="utf-8")
    print(summary.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
