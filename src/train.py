"""Reproducible model training for the educational credit-risk prototype."""
import json
from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_loading import ROOT, load_data
from evaluate import evaluate_probabilities, review_capacity
from features import MODEL_FEATURES, select_model_data

RANDOM_STATE = 42


def split_data(X, y):
    """Make 60/20/20 stratified random splits; no real event dates support a time split."""
    X_train, X_other, y_train, y_other = train_test_split(
        X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y
    )
    return (*train_test_split(
        X_other, y_other, test_size=0.50, random_state=RANDOM_STATE, stratify=y_other
    ), X_train, y_train)


def make_models():
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


def main():
    X, y = select_model_data(load_data())
    X_validation, X_test, y_validation, y_test, X_train, y_train = split_data(X, y)
    models = make_models()
    results = {"split_sizes": {"train": len(y_train), "validation": len(y_validation), "test": len(y_test)},
               "feature_list": MODEL_FEATURES, "validation": {}}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_validation)[:, 1]
        results["validation"][name] = evaluate_probabilities(y_validation, probabilities)
        fitted[name] = model

    # Model choice is based on validation ROC-AUC, before reading test results.
    chosen_name = max(results["validation"], key=lambda n: results["validation"][n]["roc_auc"])
    chosen = fitted[chosen_name]
    validation_probabilities = chosen.predict_proba(X_validation)[:, 1]
    results["chosen_model"] = chosen_name
    results["review_capacity_validation"] = review_capacity(y_validation, validation_probabilities)
    results["test"] = evaluate_probabilities(y_test, chosen.predict_proba(X_test)[:, 1])

    models_dir = ROOT / "models"
    reports_dir = ROOT / "reports"
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    joblib.dump(chosen, models_dir / "selected_model.joblib")
    (reports_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
