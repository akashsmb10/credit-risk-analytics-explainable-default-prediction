"""Leakage-aware, pre-outcome feature construction for this credit-risk prototype."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler


# Baseline features are frozen for a fair later comparison. Do not change this list.
MODEL_FEATURES = ["X1"] + [f"X{i}" for i in range(6, 24)]
TARGET = "Y"
REPAYMENT_STATUS_COLUMNS = [f"X{i}" for i in range(6, 12)]
BILL_AMOUNT_COLUMNS = [f"X{i}" for i in range(12, 18)]
PAYMENT_AMOUNT_COLUMNS = [f"X{i}" for i in range(18, 24)]
FORBIDDEN_MODEL_COLUMNS = {"ID", "X2", "X3", "X4", "X5", TARGET}

ENGINEERED_FEATURES = [
    "repayment_status_max", "repayment_status_mean", "repayment_positive_delay_count",
    "repayment_status_latest", "repayment_latest_minus_earlier_mean",
    "bill_amount_mean", "bill_amount_median", "bill_amount_max", "bill_amount_min",
    "bill_amount_range", "bill_amount_trend_slope", "payment_amount_mean",
    "payment_amount_median", "payment_amount_max", "payment_amount_min",
    "payment_amount_range", "payment_amount_trend_slope",
]
# X6 is already a baseline feature, so its semantic alias is not added twice.
ENGINEERED_ADDITIONAL_MODEL_FEATURES = [
    feature for feature in ENGINEERED_FEATURES if feature != "repayment_status_latest"
]
ENGINEERED_MODEL_FEATURES = MODEL_FEATURES + ENGINEERED_ADDITIONAL_MODEL_FEATURES


def _six_month_slope(data, columns):
    """Return per-record slope from April (oldest) to September (latest)."""
    oldest_to_latest = data[columns[::-1]].to_numpy(dtype=float)
    months = np.arange(oldest_to_latest.shape[1], dtype=float)
    centered_months = months - months.mean()
    return oldest_to_latest @ centered_months / np.square(centered_months).sum()


def build_engineered_features(data):
    """Return raw eligible inputs plus documented, row-level pre-outcome summaries.

    This function never reads Y and never fits a statistic across records. Every
    summary uses only one record's April-to-September historical fields.
    """
    required = REPAYMENT_STATUS_COLUMNS + BILL_AMOUNT_COLUMNS + PAYMENT_AMOUNT_COLUMNS + ["X1"]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")

    features = data[MODEL_FEATURES].copy()
    repayment = data[REPAYMENT_STATUS_COLUMNS]
    # Codes -2 and 0 remain numeric observed codes; no business meaning is assigned.
    features["repayment_status_max"] = repayment.max(axis=1)
    features["repayment_status_mean"] = repayment.mean(axis=1)
    features["repayment_positive_delay_count"] = (repayment > 0).sum(axis=1)
    features["repayment_status_latest"] = data["X6"]
    features["repayment_latest_minus_earlier_mean"] = data["X6"] - data[REPAYMENT_STATUS_COLUMNS[1:]].mean(axis=1)

    bill = data[BILL_AMOUNT_COLUMNS]
    features["bill_amount_mean"] = bill.mean(axis=1)
    features["bill_amount_median"] = bill.median(axis=1)
    features["bill_amount_max"] = bill.max(axis=1)
    features["bill_amount_min"] = bill.min(axis=1)
    features["bill_amount_range"] = features["bill_amount_max"] - features["bill_amount_min"]
    features["bill_amount_trend_slope"] = _six_month_slope(data, BILL_AMOUNT_COLUMNS)

    payment = data[PAYMENT_AMOUNT_COLUMNS]
    features["payment_amount_mean"] = payment.mean(axis=1)
    features["payment_amount_median"] = payment.median(axis=1)
    features["payment_amount_max"] = payment.max(axis=1)
    features["payment_amount_min"] = payment.min(axis=1)
    features["payment_amount_range"] = features["payment_amount_max"] - features["payment_amount_min"]
    features["payment_amount_trend_slope"] = _six_month_slope(data, PAYMENT_AMOUNT_COLUMNS)
    return features


def select_model_data(data):
    """Return frozen baseline inputs and the documented next-month outcome."""
    return data[MODEL_FEATURES].copy(), data[TARGET].copy()


def select_engineered_model_data(data):
    """Return fair-comparison candidate inputs and the unchanged target."""
    features = build_engineered_features(data)
    return features[ENGINEERED_MODEL_FEATURES].copy(), data[TARGET].copy()


def make_credit_limit_preprocessor():
    """Create, but do not fit, a train-only robust log transform for X1."""
    return ColumnTransformer(
        transformers=[
            ("credit_limit_raw", "passthrough", ["X1"]),
            ("credit_limit_log_robust", Pipeline([
                ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("robust_scale", RobustScaler()),
            ]), ["X1"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
