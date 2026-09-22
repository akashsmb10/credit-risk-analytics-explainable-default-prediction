"""Fair baseline-versus-engineered feature experiment; does not alter baseline artifacts."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_loading import ROOT, load_data
from evaluate import evaluate_probabilities, review_capacity
from features import MODEL_FEATURES, select_engineered_model_data, select_model_data


RANDOM_STATE = 42
VALIDATION_ROC_AUC_MARGIN = 0.005


def make_split_indices(y):
    """Reproduce the existing stratified 60/20/20 split exactly with row indices."""
    indices = np.asarray(y.index)
    train_index, other_index = train_test_split(
        indices, test_size=0.40, random_state=RANDOM_STATE, stratify=y
    )
    validation_index, test_index = train_test_split(
        other_index, test_size=0.50, random_state=RANDOM_STATE,
        stratify=y.loc[other_index],
    )
    return train_index, validation_index, test_index


def make_models():
    """Use the same candidate families and settings as the saved baseline run."""
    return {
        "dummy": DummyClassifier(strategy="prior", random_state=RANDOM_STATE),
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "xgboost": XGBClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=1,
        ),
    }


def evaluate_feature_set(name, X, y, split_indices):
    """Select on validation ROC-AUC, then perform one held-out test evaluation."""
    train_index, validation_index, test_index = split_indices
    X_train, y_train = X.loc[train_index], y.loc[train_index]
    X_validation, y_validation = X.loc[validation_index], y.loc[validation_index]
    X_test, y_test = X.loc[test_index], y.loc[test_index]

    validation = {}
    fitted_models = {}
    for model_name, model in make_models().items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_validation)[:, 1]
        validation[model_name] = evaluate_probabilities(y_validation, probabilities)
        fitted_models[model_name] = model

    selected_name = max(validation, key=lambda model_name: validation[model_name]["roc_auc"])
    selected_model = fitted_models[selected_name]
    validation_probabilities = selected_model.predict_proba(X_validation)[:, 1]
    test_probabilities = selected_model.predict_proba(X_test)[:, 1]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_validate(
        make_models()[selected_name], X_train, y_train, cv=cv,
        scoring={"roc_auc": "roc_auc", "average_precision": "average_precision"},
        n_jobs=1,
    )
    return {
        "feature_set": name,
        "feature_count": int(X.shape[1]),
        "feature_list": list(X.columns),
        "selected_model": selected_name,
        "validation_candidates": validation,
        "validation_selected": evaluate_probabilities(y_validation, validation_probabilities),
        "validation_review_capacity": review_capacity(y_validation, validation_probabilities),
        "test_selected": evaluate_probabilities(y_test, test_probabilities),
        "test_review_capacity": review_capacity(y_test, test_probabilities),
        "training_cross_validation": {
            "folds": 5,
            "roc_auc_mean": float(cv_scores["test_roc_auc"].mean()),
            "roc_auc_std": float(cv_scores["test_roc_auc"].std(ddof=1)),
            "average_precision_mean": float(cv_scores["test_average_precision"].mean()),
            "average_precision_std": float(cv_scores["test_average_precision"].std(ddof=1)),
        },
        "_selected_model": selected_model,
    }


def safe_result(result):
    """Remove non-serializable model object before creating experiment report."""
    return {key: value for key, value in result.items() if key != "_selected_model"}


def make_chart(baseline, engineered, final_name):
    figures = ROOT / "reports" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame({
        "Feature set": ["Baseline", "Engineered"],
        "Validation ROC-AUC": [
            baseline["validation_selected"]["roc_auc"],
            engineered["validation_selected"]["roc_auc"],
        ],
        "Test ROC-AUC": [
            baseline["test_selected"]["roc_auc"],
            engineered["test_selected"]["roc_auc"],
        ],
        "Validation AP": [
            baseline["validation_selected"]["average_precision"],
            engineered["validation_selected"]["average_precision"],
        ],
        "Test AP": [
            baseline["test_selected"]["average_precision"],
            engineered["test_selected"]["average_precision"],
        ],
    })
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = ["#4C78A8", "#E45756"]
    for axis, columns, title in [
        (axes[0], ["Validation ROC-AUC", "Test ROC-AUC"], "ROC-AUC comparison"),
        (axes[1], ["Validation AP", "Test AP"], "Average-precision comparison"),
    ]:
        positions = np.arange(2)
        width = 0.35
        for offset, column in zip([-width / 2, width / 2], columns):
            bars = axis.bar(positions + offset, summary[column], width, label=column.replace(" ", "\n"), color=colors[0 if "Validation" in column else 1])
            for bar in bars:
                axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
        axis.set_xticks(positions, summary["Feature set"])
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.legend(fontsize=8)
    fig.suptitle(f"Fair feature-set experiment — final candidate: {final_name}", fontsize=14)
    plt.tight_layout()
    path = figures / "10_baseline_vs_engineered_comparison.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def markdown_report(baseline, engineered, final_name, meaningful, margin):
    final = baseline if final_name == "baseline" else engineered
    delta = engineered["validation_selected"]["roc_auc"] - baseline["validation_selected"]["roc_auc"]
    lines = [
        "# Baseline versus engineered feature comparison",
        "",
        "## What makes this fair",
        "",
        "Both feature sets use exactly the same stratified 60/20/20 row indices (`random_state=42`), the same target, and the same dummy, logistic-regression, and XGBoost settings. The baseline list is frozen as `X1` and `X6`–`X23`; the engineered set adds only documented row-level summaries. `ID`, `X2`–`X5`, `Y`, payment-to-bill ratios, and reconstructed balances are excluded. Logistic-regression scaling occurs inside its sklearn pipeline and is fit only to training folds. No model or metric from the existing baseline artifact was overwritten.",
        "",
        "## Validation and test roles",
        "",
        "Training data fits candidate models. Validation data chooses the winner by ROC-AUC and calculates the review-capacity simulation. The held-out test set is evaluated once after that selection; it did not choose models, features, or settings. The 5-fold cross-validation scores are calculated only on the training partition for each selected candidate.",
        "",
        "## Results",
        "",
        "| Feature set | Features | Selected model | Validation ROC-AUC | Test ROC-AUC | Validation AP | Test AP | Test Brier | 5-fold train ROC-AUC | 5-fold train AP |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for label, result in [("Baseline", baseline), ("Engineered", engineered)]:
        cv = result["training_cross_validation"]
        lines.append(
            f"| {label} | {result['feature_count']} | {result['selected_model']} | "
            f"{result['validation_selected']['roc_auc']:.4f} | {result['test_selected']['roc_auc']:.4f} | "
            f"{result['validation_selected']['average_precision']:.4f} | {result['test_selected']['average_precision']:.4f} | "
            f"{result['test_selected']['brier_score']:.4f} | {cv['roc_auc_mean']:.4f} ± {cv['roc_auc_std']:.4f} | "
            f"{cv['average_precision_mean']:.4f} ± {cv['average_precision_std']:.4f} |"
        )
    lines += [
        "",
        "### Ranking, calibration, classification, and review capture",
        "",
        "The following test metrics use the inherited 0.50 reporting cutoff only; it is not a chosen business threshold.",
        "",
        "| Feature set | Accuracy | Precision | Recall | F1 | Brier score | Test top-10% defaults captured |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, result in [("Baseline", baseline), ("Engineered", engineered)]:
        metric = result["test_selected"]
        top_ten = next(item for item in result["test_review_capacity"] if item["review_fraction"] == 0.10)
        lines.append(
            f"| {label} | {metric['accuracy']:.4f} | {metric['precision']:.4f} | {metric['recall']:.4f} | "
            f"{metric['f1']:.4f} | {metric['brier_score']:.4f} | {top_ten['observed_defaults_captured']} / "
            f"{top_ten['accounts_reviewed']} ({top_ten['recall_of_observed_defaults']:.2%}) |"
        )
    lines += [
        "",
        "### Interpretation",
        "",
        f"Engineered minus baseline validation ROC-AUC is **{delta:+.4f}**. The pre-specified meaningful-improvement margin is **{margin:.4f}**.",
        "",
        ("The engineered feature set clears the pre-specified validation margin, so it is the final candidate for the next project stage. This remains an historical comparison, not production evidence."
         if meaningful else
         "The engineered feature set does not clear the pre-specified validation margin. Any difference is treated as small/noisy for this project stage, so the simpler baseline remains the final candidate."),
        "",
        f"Final candidate: **{final_name}** using **{final['selected_model']}**. This selection uses validation ROC-AUC only. Calibration and a business threshold have not been selected.",
        "",
        "## Capacity results",
        "",
        "Full validation and test top-5%, top-10%, and top-20% review results, including observed-default capture and random-selection reference, are in `model_comparison_metrics.json`. They are retrospective simulations using known historical outcomes, not prevented-default or savings estimates.",
    ]
    return "\n".join(lines)


def main():
    data = load_data()
    X_baseline, y = select_model_data(data)
    X_engineered, engineered_y = select_engineered_model_data(data)
    if not y.equals(engineered_y):
        raise ValueError("Feature sets do not have the same target values.")
    split_indices = make_split_indices(y)
    baseline = evaluate_feature_set("baseline", X_baseline, y, split_indices)
    engineered = evaluate_feature_set("engineered", X_engineered, y, split_indices)

    delta = engineered["validation_selected"]["roc_auc"] - baseline["validation_selected"]["roc_auc"]
    engineered_wins_meaningfully = delta >= VALIDATION_ROC_AUC_MARGIN
    final_name = "engineered" if engineered_wins_meaningfully else "baseline"

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    chart = make_chart(baseline, engineered, final_name)
    report = {
        "experiment": {
            "split": "stratified random 60/20/20", "random_state": RANDOM_STATE,
            "selection_metric": "validation ROC-AUC", "meaningful_validation_roc_auc_margin": VALIDATION_ROC_AUC_MARGIN,
            "baseline_artifacts_preserved": True, "test_set_used_for_selection": False,
        },
        "split_sizes": {"train": len(split_indices[0]), "validation": len(split_indices[1]), "test": len(split_indices[2])},
        "baseline": safe_result(baseline), "engineered": safe_result(engineered),
        "validation_roc_auc_difference_engineered_minus_baseline": float(delta),
        "engineered_wins_meaningfully": engineered_wins_meaningfully,
        "final_candidate": final_name,
        "comparison_chart": str(chart.relative_to(ROOT)),
    }
    (reports / "model_comparison_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (reports / "model_comparison.md").write_text(
        markdown_report(baseline, engineered, final_name, engineered_wins_meaningfully, VALIDATION_ROC_AUC_MARGIN), encoding="utf-8"
    )

    if engineered_wins_meaningfully:
        joblib.dump(engineered["_selected_model"], ROOT / "models" / "engineered_feature_winner.joblib")

    print(json.dumps({
        "baseline_validation_roc_auc": baseline["validation_selected"]["roc_auc"],
        "engineered_validation_roc_auc": engineered["validation_selected"]["roc_auc"],
        "difference": delta, "final_candidate": final_name,
    }, indent=2))


if __name__ == "__main__":
    main()
