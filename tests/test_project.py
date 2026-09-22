import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from data_loading import EXPECTED_COLUMNS, RAW_PATH, load_data
from features import (
    ENGINEERED_FEATURES,
    ENGINEERED_MODEL_FEATURES,
    FORBIDDEN_MODEL_COLUMNS,
    MODEL_FEATURES,
    build_engineered_features,
)
from calibration_threshold_analysis import (
    expected_calibration_error,
    fit_calibrated_models,
    probability_bin_table,
    threshold_table,
)
from explain import validate_explanation_inputs, validate_shap_output
from fairness_audit import AUDIT_COLUMNS, build_audit_labels
from dashboard_artifacts import load_dashboard_artifacts


def test_raw_file_and_columns():
    assert RAW_PATH.is_file()
    assert load_data().columns.tolist() == EXPECTED_COLUMNS


def test_initial_feature_list_excludes_id_and_demographics():
    from features import MODEL_FEATURES

    assert "ID" not in MODEL_FEATURES
    assert not set(["X2", "X3", "X4", "X5"]).intersection(MODEL_FEATURES)


def test_engineered_features_have_expected_names_and_row_count():
    data = load_data()
    engineered = build_engineered_features(data)

    assert engineered.index.equals(data.index)
    assert len(engineered) == len(data)
    assert set(ENGINEERED_FEATURES).issubset(engineered.columns)
    assert list(engineered.columns) == MODEL_FEATURES + ENGINEERED_FEATURES
    assert "repayment_status_latest" not in ENGINEERED_MODEL_FEATURES


def test_engineered_features_exclude_forbidden_columns_and_are_finite():
    import numpy as np

    engineered = build_engineered_features(load_data())
    assert not FORBIDDEN_MODEL_COLUMNS.intersection(engineered.columns)
    assert engineered.notna().all().all()
    assert np.isfinite(engineered.to_numpy(dtype=float)).all()


def test_probability_diagnostics_are_bounded_and_complete():
    import numpy as np

    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.40, 0.60, 0.95])
    table = probability_bin_table(y, probabilities)
    assert table["count"].sum() == len(y)
    assert 0.0 <= expected_calibration_error(y, probabilities) <= 1.0
    rows = threshold_table(y, probabilities)
    assert all(0.0 <= row["precision"] <= 1.0 and 0.0 <= row["recall"] <= 1.0 for row in rows)
    assert all(row["false_positives"] >= 0 and row["false_negatives"] >= 0 for row in rows)


def test_calibration_fit_api_accepts_training_data_only():
    import inspect

    assert list(inspect.signature(fit_calibrated_models).parameters) == ["X_train", "y_train", "base_estimator"]


def test_explanations_use_frozen_permitted_feature_order_and_shape():
    import numpy as np

    data = load_data()
    X = data[MODEL_FEATURES]
    validate_explanation_inputs(X)
    values = validate_shap_output(np.zeros(X.head(3).shape), X.head(3))
    assert values.shape == (3, len(MODEL_FEATURES))
    assert not FORBIDDEN_MODEL_COLUMNS.intersection(X.columns)


def test_demographics_are_audit_labels_not_model_or_explanation_features():
    data = load_data().head(10)
    audit = build_audit_labels(data[AUDIT_COLUMNS])
    assert not set(AUDIT_COLUMNS).intersection(MODEL_FEATURES)
    assert list(audit.index) == list(data.index)
    assert audit.notna().all().all()
    assert "X2" not in audit.columns and "X3" not in audit.columns


def test_dashboard_loads_saved_artifacts_without_loading_raw_records():
    root = Path(__file__).resolve().parents[1]
    artifacts = load_dashboard_artifacts(root)
    assert artifacts["json"]["calibration_threshold_metrics.json"] is not None
    assert artifacts["json"]["fairness_audit_metrics.json"] is not None
    assert artifacts["markdown"]["model_card.md"] is not None
    assert "data" not in artifacts and "models" not in artifacts
