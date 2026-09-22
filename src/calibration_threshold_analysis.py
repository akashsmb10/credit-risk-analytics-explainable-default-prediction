"""Training-only probability calibration and validation-selected review policy."""

import json
from copy import deepcopy

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix

from compare_feature_sets import RANDOM_STATE, make_split_indices
from data_loading import ROOT, load_data
from evaluate import evaluate_probabilities, review_capacity
from features import select_model_data


BIN_EDGES = np.linspace(0.0, 1.0, 11)
THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)
REVIEW_FRACTIONS = (0.05, 0.10, 0.20)
DEMONSTRATION_REVIEW_FRACTION = 0.10


def probability_bin_table(y_true, probabilities, edges=BIN_EDGES):
    """Return reliability-bin count, mean prediction, and observed outcome rate."""
    frame = pd.DataFrame({"outcome": np.asarray(y_true), "probability": np.asarray(probabilities)})
    frame["bin"] = pd.cut(frame["probability"], bins=edges, include_lowest=True)
    table = frame.groupby("bin", observed=False).agg(
        count=("outcome", "size"),
        mean_predicted_probability=("probability", "mean"),
        observed_default_rate=("outcome", "mean"),
    ).reset_index()
    table["bin"] = table["bin"].astype(str)
    return table


def expected_calibration_error(y_true, probabilities):
    """Weighted absolute prediction-versus-outcome gap across ten fixed bins."""
    table = probability_bin_table(y_true, probabilities)
    nonempty = table[table["count"] > 0]
    if nonempty.empty:
        return 0.0
    return float((nonempty["count"] / nonempty["count"].sum() * (
        nonempty["mean_predicted_probability"] - nonempty["observed_default_rate"]
    ).abs()).sum())


def calibration_metrics(y_true, probabilities):
    metrics = evaluate_probabilities(y_true, probabilities)
    return {
        "brier_score": metrics["brier_score"],
        "expected_calibration_error": expected_calibration_error(y_true, probabilities),
        "probability_bin_table": probability_bin_table(y_true, probabilities).to_dict(orient="records"),
    }


def fit_calibrated_models(X_train, y_train, base_estimator):
    """Fit calibration wrappers using training data only and internal 5-fold CV."""
    return {
        "sigmoid": CalibratedClassifierCV(estimator=clone(base_estimator), method="sigmoid", cv=5, n_jobs=1).fit(X_train, y_train),
        "isotonic": CalibratedClassifierCV(estimator=clone(base_estimator), method="isotonic", cv=5, n_jobs=1).fit(X_train, y_train),
    }


def threshold_table(y_true, probabilities):
    """Calculate validation-only threshold trade-offs with transparent unit costs."""
    y = np.asarray(y_true)
    rows = []
    for threshold in THRESHOLDS:
        labels = (np.asarray(probabilities) >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, labels, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "threshold": float(threshold), "accounts_flagged": int(labels.sum()),
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "false_positives": int(fp), "false_negatives": int(fn),
            "default_capture_rate": float(recall),
            "illustrative_cost_missed_5x_review": int(5 * fn + fp),
            "illustrative_cost_missed_10x_review": int(10 * fn + fp),
        })
    return rows


def review_policy_metrics(y_true, probabilities, fractions=REVIEW_FRACTIONS):
    """Top-k review simulation with the same transparent illustrative cost scenarios."""
    rows = review_capacity(y_true, probabilities, fractions=fractions)
    for row in rows:
        false_negatives = int(np.asarray(y_true).sum() - row["observed_defaults_captured"])
        false_positives = int(row["accounts_reviewed"] - row["observed_defaults_captured"])
        row["false_negatives"] = false_negatives
        row["false_positives"] = false_positives
        row["illustrative_cost_missed_5x_review"] = int(5 * false_negatives + false_positives)
        row["illustrative_cost_missed_10x_review"] = int(10 * false_negatives + false_positives)
    return rows


def select_calibration(validation_results):
    """Choose lowest validation Brier; use ECE only as an explicit tie-breaker."""
    return min(
        validation_results,
        key=lambda name: (
            validation_results[name]["validation"]["calibration"]["brier_score"],
            validation_results[name]["validation"]["calibration"]["expected_calibration_error"],
        ),
    )


def plot_reliability(results, chosen_name):
    figures = ROOT / "reports" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    colors = {"uncalibrated": "#4C78A8", "sigmoid": "#F58518", "isotonic": "#54A24B"}
    for axis, split, title in zip(axes, ["validation", "test"], ["Validation reliability", "Held-out test reliability"]):
        axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        for name, result in results.items():
            table = pd.DataFrame(result[split]["calibration"]["probability_bin_table"])
            table = table[table["count"] > 0]
            label = f"{name}{' (selected)' if name == chosen_name else ''}"
            axis.plot(table["mean_predicted_probability"], table["observed_default_rate"], marker="o", label=label, color=colors[name])
        axis.set(title=title, xlabel="Mean predicted probability", ylabel="Observed default rate", xlim=(0, 1), ylim=(0, 1))
        axis.legend(fontsize=8)
    fig.suptitle("Reliability curves — curves assess probability accuracy, not causal effects")
    plt.tight_layout()
    path = figures / "11_calibration_reliability_curves.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_policy_tables(thresholds, validation_review, test_review):
    figures = ROOT / "reports" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(thresholds)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(table["threshold"], table["precision"], marker="o", label="Precision", color="#4C78A8")
    axes[0].plot(table["threshold"], table["recall"], marker="o", label="Recall / capture", color="#E45756")
    axes[0].plot(table["threshold"], table["f1"], marker="o", label="F1", color="#54A24B")
    axes[0].set(title="Validation threshold trade-off", xlabel="Probability threshold", ylabel="Metric", ylim=(0, 1))
    axes[0].legend()
    axes[1].plot(table["threshold"], table["accounts_flagged"], marker="o", color="#6F4E7C")
    axes[1].set(title="Validation workload by threshold", xlabel="Probability threshold", ylabel="Accounts flagged")
    plt.tight_layout()
    threshold_path = figures / "12_validation_threshold_tradeoff.png"
    plt.savefig(threshold_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    review = pd.DataFrame(validation_review)
    review_test = pd.DataFrame(test_review)
    labels = [f"Top {value:.0%}" for value in review["review_fraction"]]
    positions = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    validation_bars = ax.bar(positions - width / 2, review["recall_of_observed_defaults"], width, label="Validation", color="#4C78A8")
    test_bars = ax.bar(positions + width / 2, review_test["recall_of_observed_defaults"], width, label="Held-out test", color="#E45756")
    for bars in (validation_bars, test_bars):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.1%}", ha="center", va="bottom")
    ax.set(title="Observed-default capture by ranked review capacity", xlabel="Review policy", ylabel="Recall of observed defaults", ylim=(0, 0.65))
    ax.set_xticks(positions, labels)
    ax.legend()
    plt.tight_layout()
    review_path = figures / "13_review_capacity_capture.png"
    plt.savefig(review_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(table["threshold"], table["illustrative_cost_missed_5x_review"], marker="o", label="Missed default = 5× review", color="#F58518")
    ax.plot(table["threshold"], table["illustrative_cost_missed_10x_review"], marker="o", label="Missed default = 10× review", color="#E45756")
    ax.set(title="Illustrative validation cost sensitivity", xlabel="Probability threshold", ylabel="Illustrative cost units")
    ax.legend()
    plt.tight_layout()
    cost_path = figures / "14_validation_cost_sensitivity.png"
    plt.savefig(cost_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return threshold_path, review_path, cost_path


def render_report(results, selected, threshold_rows, validation_review, test_review):
    selected_result = results[selected]
    chosen_review_validation = next(row for row in validation_review if row["review_fraction"] == DEMONSTRATION_REVIEW_FRACTION)
    chosen_review_test = next(row for row in test_review if row["review_fraction"] == DEMONSTRATION_REVIEW_FRACTION)
    lines = [
        "# Calibration and review-policy analysis",
        "",
        "## Calibration, in simple terms",
        "",
        "A calibrated probability is a probability whose predicted level approximately matches the observed event frequency. For example, among accounts assigned roughly 20% risk, we would hope roughly 20% historically defaulted. ROC-AUC only tests whether defaults tend to rank above non-defaults; it can be strong even when probability levels are too high or too low.",
        "",
        "Brier score is the average squared gap between each predicted probability and outcome (lower is better). Expected calibration error (ECE) is the weighted average gap between mean prediction and observed rate in fixed probability bins (lower is better). Both are historical diagnostic measures, not guarantees for future customers.",
        "",
        "## Calibration selection on validation data",
        "",
        "The saved baseline XGBoost supplied uncalibrated predictions. Sigmoid (Platt) and isotonic calibration wrappers were fitted using only the 18,000-row training partition with internal 5-fold cross-validation. Selection uses lowest validation Brier score; validation ECE is the stated tie-breaker. Test results did not select the method.",
        "",
        "| Method | Validation Brier | Validation ECE | Test Brier | Test ECE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, value in results.items():
        lines.append(f"| {name} | {value['validation']['calibration']['brier_score']:.4f} | {value['validation']['calibration']['expected_calibration_error']:.4f} | {value['test']['calibration']['brier_score']:.4f} | {value['test']['calibration']['expected_calibration_error']:.4f} |")
    lines += [
        "",
        f"Selected calibration: **{selected}**. Reliability curves and full bin tables are in `calibration_threshold_metrics.json`; test values are reported only after this validation choice.",
        "",
        "## Thresholds and capacity",
        "",
        "A 0.50 threshold is not automatically correct: it means flag when predicted risk is at least 50%, but it ignores staffing capacity and the relative consequences of false positives (unnecessary reviews) and false negatives (missed observed defaults). The threshold table evaluates 0.10–0.90 on validation only. Its illustrative cost is `false positives × 1 + false negatives × 5` or `× 10`; these are classroom sensitivity assumptions, not bank costs or savings.",
        "",
        "A top-k policy instead reviews a fixed number of the highest-ranked accounts. It is often easier to operate when capacity is fixed because exactly the chosen share is sent to review even if the probability scale changes. This project’s pre-specified demonstration policy is **top 10% ranked accounts**, consistent with the earlier capacity simulation; it was not selected after looking at test outcomes.",
        "",
        "| Split | Policy | Accounts reviewed | Defaults captured | Capture rate | Precision among reviewed | False positives | Missed defaults |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Validation | Top 10% | {chosen_review_validation['accounts_reviewed']} | {chosen_review_validation['observed_defaults_captured']} | {chosen_review_validation['recall_of_observed_defaults']:.2%} | {chosen_review_validation['precision_among_reviewed']:.2%} | {chosen_review_validation['false_positives']} | {chosen_review_validation['false_negatives']} |",
        f"| Held-out test | Top 10% | {chosen_review_test['accounts_reviewed']} | {chosen_review_test['observed_defaults_captured']} | {chosen_review_test['recall_of_observed_defaults']:.2%} | {chosen_review_test['precision_among_reviewed']:.2%} | {chosen_review_test['false_positives']} | {chosen_review_test['false_negatives']} |",
        "",
        "The 0.50 classification metrics in the JSON remain descriptive only. No business classification threshold has been declared as universally correct.",
    ]
    return "\n".join(lines)


def main():
    data = load_data()
    X, y = select_model_data(data)
    train_idx, validation_idx, test_idx = make_split_indices(y)
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_validation, y_validation = X.loc[validation_idx], y.loc[validation_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

    baseline = joblib.load(ROOT / "models" / "selected_model.joblib")
    calibrated = fit_calibrated_models(X_train, y_train, baseline)
    models = {"uncalibrated": baseline, **calibrated}
    results = {}
    for name, model in models.items():
        validation_probability = model.predict_proba(X_validation)[:, 1]
        test_probability = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "validation": {"calibration": calibration_metrics(y_validation, validation_probability), "classification_metrics_at_0_50": evaluate_probabilities(y_validation, validation_probability)},
            "test": {"calibration": calibration_metrics(y_test, test_probability), "classification_metrics_at_0_50": evaluate_probabilities(y_test, test_probability)},
            "_validation_probability": validation_probability, "_test_probability": test_probability,
        }
    selected = select_calibration(results)
    selected_validation_probability = results[selected]["_validation_probability"]
    selected_test_probability = results[selected]["_test_probability"]
    validation_thresholds = threshold_table(y_validation, selected_validation_probability)
    validation_review = review_policy_metrics(y_validation, selected_validation_probability)
    test_review = review_policy_metrics(y_test, selected_test_probability)

    reliability_chart = plot_reliability(results, selected)
    threshold_chart, review_chart, cost_chart = plot_policy_tables(validation_thresholds, validation_review, test_review)
    serializable = {
        name: {key: value for key, value in result.items() if not key.startswith("_")}
        for name, result in results.items()
    }
    report = {
        "experiment": {
            "baseline_model_loaded_from": "models/selected_model.joblib", "baseline_model_overwritten": False,
            "split": "existing stratified 60/20/20", "random_state": RANDOM_STATE,
            "calibration_fit_data": "training partition only; CalibratedClassifierCV internal 5-fold CV",
            "calibration_selection": "lowest validation Brier score; ECE tie-breaker",
            "demonstration_policy": "top 10% ranked accounts", "policy_selected_on": "validation capacity assumption",
            "test_data_used_for_fitting_or_selection": False,
        },
        "split_sizes": {"train": len(train_idx), "validation": len(validation_idx), "test": len(test_idx)},
        "methods": serializable, "selected_calibration": selected,
        "validation_thresholds": validation_thresholds,
        "validation_review_capacity": validation_review,
        "final_test_review_capacity": test_review,
        "charts": [str(path.relative_to(ROOT)) for path in [reliability_chart, threshold_chart, review_chart, cost_chart]],
    }
    reports = ROOT / "reports"
    (reports / "calibration_threshold_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (reports / "calibration_threshold_report.md").write_text(render_report(results, selected, validation_thresholds, validation_review, test_review), encoding="utf-8")
    print(json.dumps({"selected_calibration": selected, "validation_brier": results[selected]["validation"]["calibration"]["brier_score"], "test_brier": results[selected]["test"]["calibration"]["brier_score"], "test_top_10_percent": next(row for row in test_review if row["review_fraction"] == 0.10)}, indent=2))


if __name__ == "__main__":
    main()
