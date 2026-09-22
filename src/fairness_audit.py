"""Held-out subgroup audit; demographic columns are audit labels, never model inputs."""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score

from calibration_threshold_analysis import DEMONSTRATION_REVIEW_FRACTION, fit_calibrated_models
from compare_feature_sets import make_split_indices
from data_loading import ROOT, load_data
from features import MODEL_FEATURES, select_model_data

AUDIT_COLUMNS = ["X2", "X3", "X4", "X5"]
MIN_GROUP_SIZE = 100
MIN_CLASS_COUNT = 20


def build_audit_labels(data):
    """Map only documented UCI demographic labels; unknown codes remain explicit."""
    audit = pd.DataFrame(index=data.index)
    audit["Recorded sex category"] = data["X2"].map({1: "Male (documented code 1)", 2: "Female (documented code 2)"}).fillna("Undocumented/other sex code")
    audit["Recorded education category"] = data["X3"].map({
        1: "Graduate school (documented)", 2: "University (documented)",
        3: "High school (documented)", 4: "Other (documented code 4)",
    }).fillna("Undocumented education code(s)")
    audit["Recorded marital-status category"] = data["X4"].map({
        1: "Married (documented)", 2: "Single (documented)", 3: "Other (documented code 3)",
    }).fillna("Undocumented marital-status code")
    audit["Age band"] = pd.cut(data["X5"], bins=[19, 29, 39, 49, np.inf], labels=["20–29", "30–39", "40–49", "50+"]).astype(str)
    return audit


def subgroup_metrics(y, probability, review_mask, group_name, group_value):
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    flags = probability >= 0.50
    tn, fp, fn, tp = confusion_matrix(y, flags, labels=[0, 1]).ravel()
    positives, negatives = int(y.sum()), int((1 - y).sum())
    reviewed = review_mask.astype(bool)
    captured = int(y[reviewed].sum())
    reviewed_count = int(reviewed.sum())
    stable = len(y) >= MIN_GROUP_SIZE and positives >= MIN_CLASS_COUNT and negatives >= MIN_CLASS_COUNT
    return {
        "audit_group": group_name, "subgroup": str(group_value), "accounts": int(len(y)),
        "observed_defaults": positives, "observed_default_rate": float(y.mean()),
        "mean_calibrated_predicted_risk": float(probability.mean()),
        "roc_auc": float(roc_auc_score(y, probability)) if stable else None,
        "average_precision": float(average_precision_score(y, probability)) if stable else None,
        "brier_score": float(brier_score_loss(y, probability)),
        "precision_at_0_50": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall_tpr_at_0_50": float(tp / positives) if positives else None,
        "fpr_at_0_50": float(fp / negatives) if negatives else None,
        "false_negative_rate_at_0_50": float(fn / positives) if positives else None,
        "selection_flag_rate_at_0_50": float(flags.mean()),
        "top_10_percent_reviewed": reviewed_count,
        "top_10_percent_defaults_captured": captured,
        "top_10_percent_capture_rate": float(captured / positives) if positives else None,
        "top_10_percent_precision": float(captured / reviewed_count) if reviewed_count else None,
        "metric_stability_warning": None if stable else f"Do not treat discrimination metrics as stable: n={len(y)}, defaults={positives}, non-defaults={negatives}; threshold is n≥{MIN_GROUP_SIZE} and at least {MIN_CLASS_COUNT} of each outcome.",
    }


def make_figures(metrics):
    figures = ROOT / "reports" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(metrics)
    groups = frame.audit_group.unique()
    fig, axes = plt.subplots(len(groups), 2, figsize=(14, 4 * len(groups)))
    for row, group in enumerate(groups):
        subset = frame[frame.audit_group == group]
        axes[row, 0].bar(subset.subgroup, subset.accounts, color="#4C78A8")
        axes[row, 0].set(title=f"{group}: held-out subgroup sizes", ylabel="Accounts")
        axes[row, 1].bar(subset.subgroup, subset.observed_default_rate, color="#E45756")
        axes[row, 1].set(title=f"{group}: observed historical default rate", ylabel="Observed default rate", ylim=(0, max(0.4, subset.observed_default_rate.max() * 1.2)))
        axes[row, 1].yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
        for axis in axes[row]: axis.tick_params(axis="x", rotation=22)
    plt.tight_layout()
    size_path = figures / "18_fairness_subgroup_sizes_default_rates.png"
    plt.savefig(size_path, dpi=200, bbox_inches="tight"); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for group in groups:
        subset = frame[frame.audit_group == group]
        axes[0].scatter(subset.mean_calibrated_predicted_risk, subset.observed_default_rate, label=group, s=75)
        axes[1].scatter(subset.roc_auc, subset.brier_score, label=group, s=75)
    axes[0].plot([0, 0.5], [0, 0.5], "--", color="gray", label="perfect group mean match")
    axes[0].set(title="Subgroup mean predicted risk versus outcome rate", xlabel="Mean calibrated predicted risk", ylabel="Observed default rate")
    axes[1].set(title="Stable subgroup discrimination/calibration diagnostics", xlabel="ROC-AUC (missing when unstable)", ylabel="Brier score")
    for axis in axes: axis.legend(fontsize=8)
    plt.tight_layout()
    performance_path = figures / "19_fairness_performance_calibration_comparison.png"
    plt.savefig(performance_path, dpi=200, bbox_inches="tight"); plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for ax, group in zip(axes.flat, groups):
        subset = frame[frame.audit_group == group].copy()
        labels = subset.subgroup.str.replace(" (documented code ", "\n(code ", regex=False)
        bars = ax.barh(labels, subset.top_10_percent_capture_rate, color="#0F766E")
        for bar, value in zip(bars, subset.top_10_percent_capture_rate):
            ax.text(value + 0.008, bar.get_y() + bar.get_height()/2, f"{value:.1%}", va="center", fontsize=9)
        ax.set(title=group.replace("Recorded ", ""), xlabel="Capture of observed defaults", xlim=(0, min(0.55, subset.top_10_percent_capture_rate.max() * 1.35)))
        ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.suptitle("Fixed global top-10% review queue: subgroup default capture", fontsize=16, fontweight="bold")
    review_path = figures / "20_fairness_top10_review_capture.png"
    plt.savefig(review_path, dpi=200, bbox_inches="tight"); plt.close()
    return size_path, performance_path, review_path


def render_report(metrics, overall):
    frame = pd.DataFrame(metrics)
    lines = [
        "# Fairness and responsible-use audit", "", "## Scope", "",
        "This is a held-out descriptive audit of a historical Taiwan 2005 dataset. The frozen XGBoost model, training-only isotonic calibration, and fixed global top-10% review policy were not reselected or optimized using these subgroup results. `X2`–`X5` are audit labels only; they never enter model fitting, calibration fitting, SHAP, threshold selection, or ranking.", "",
        "## How to read the audit", "",
        "Differences can arise from three distinct sources: (1) calibration/performance differences, such as predicted-risk versus observed-rate gaps or Brier/ROC-AUC differences; (2) a fixed global top-10% queue, which allocates a limited number of reviews across all accounts rather than 10% inside every subgroup; and (3) different historical outcome rates. None alone proves a model fair or unfair. ROC-AUC and average precision are withheld when a subgroup has fewer than 100 records or fewer than 20 of either historical outcome.", "",
        "Overall held-out reference: ", f"{overall['accounts']:,} accounts; observed default rate {overall['observed_default_rate']:.2%}; mean calibrated risk {overall['mean_calibrated_predicted_risk']:.2%}; Brier {overall['brier_score']:.4f}; top-10% capture {overall['top_10_percent_capture_rate']:.2%}.", "",
    ]
    for group in frame.audit_group.unique():
        subset = frame[frame.audit_group == group]
        lines += [f"## {group}", "", "| Subgroup | n | Observed rate | Mean predicted risk | ROC-AUC | Brier | TPR at 0.50 | FPR at 0.50 | Top-10% capture | Top-10% precision |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in subset.itertuples():
            auc = f"{row.roc_auc:.3f}" if pd.notna(row.roc_auc) else "Not reported: small/imbalanced"
            lines.append(f"| {row.subgroup} | {row.accounts} | {row.observed_default_rate:.2%} | {row.mean_calibrated_predicted_risk:.2%} | {auc} | {row.brier_score:.3f} | {row.recall_tpr_at_0_50:.2%} | {row.fpr_at_0_50:.2%} | {row.top_10_percent_capture_rate:.2%} | {row.top_10_percent_precision:.2%} |")
    lines += ["", "## Responsible use, from zero", "", "Excluding demographics does not automatically make a model fair because other account variables can be proxies: variables associated with a group may carry some of the same historical patterns. Historical defaults can also reflect past lending policies, access, economic conditions, reporting practices, and other social context rather than an inherent individual attribute. A real deployment would need independent validation, governance, monitoring, human review, an appeal/escalation route, and legal/compliance assessment. This educational project does not justify automatic approval, rejection, pricing, collections, or any other action.", "", "## Cautions", "", "This audit is descriptive and based on one historical source, random held-out split, and recorded categories. It is not fairness certification, causal evidence, regulatory compliance, or evidence for current Indian banking. Small groups are explicitly flagged; apparent gaps should trigger investigation, not conclusions."]
    return "\n".join(lines)


def main():
    data = load_data()
    X, y = select_model_data(data)
    train_idx, _, test_idx = make_split_indices(y)
    # Audit labels are constructed only for held-out test rows, separate from X.
    audit_labels = build_audit_labels(data.loc[test_idx, AUDIT_COLUMNS])
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]
    baseline = joblib.load(ROOT / "models" / "selected_model.joblib")
    isotonic = fit_calibrated_models(X_train, y_train, baseline)["isotonic"]
    probability = isotonic.predict_proba(X_test)[:, 1]
    order = np.argsort(-probability, kind="mergesort")
    review_mask = np.zeros(len(X_test), dtype=bool)
    review_mask[order[:int(np.ceil(len(X_test) * DEMONSTRATION_REVIEW_FRACTION))]] = True

    overall = subgroup_metrics(y_test, probability, review_mask, "Overall", "All held-out records")
    metrics = []
    for group in audit_labels.columns:
        for value in sorted(audit_labels[group].unique()):
            mask = (audit_labels[group] == value).to_numpy()
            metrics.append(subgroup_metrics(y_test.loc[mask], probability[mask], review_mask[mask], group, value))
    charts = make_figures(metrics)
    report = {"scope": {"audit_labels_only": AUDIT_COLUMNS, "model_features": MODEL_FEATURES, "test_data_used_for_reselection": False, "policy": "fixed global top 10% calibrated-score review queue"}, "overall": overall, "subgroups": metrics, "figures": [str(path.relative_to(ROOT)) for path in charts]}
    reports = ROOT / "reports"
    (reports / "fairness_audit_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (reports / "fairness_audit_report.md").write_text(render_report(metrics, overall), encoding="utf-8")
    print(pd.DataFrame(metrics)[["audit_group", "subgroup", "accounts", "observed_default_rate", "mean_calibrated_predicted_risk", "top_10_percent_capture_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
