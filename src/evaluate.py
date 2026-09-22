import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)


def evaluate_probabilities(y_true, probabilities, threshold=0.5):
    """Calculate classification and probability metrics without changing inputs."""
    labels = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, labels)),
        "precision": float(precision_score(y_true, labels, zero_division=0)),
        "recall": float(recall_score(y_true, labels, zero_division=0)),
        "f1": float(f1_score(y_true, labels, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, labels).tolist(),
    }


def review_capacity(y_true, probabilities, fractions=(0.05, 0.10, 0.20)):
    """Evaluate a simulated top-k manual-review policy on known outcomes."""
    order = np.argsort(-np.asarray(probabilities))
    y = np.asarray(y_true)
    total_defaults = y.sum()
    results = []
    for fraction in fractions:
        count = int(np.ceil(len(y) * fraction))
        reviewed = y[order[:count]]
        captured = int(reviewed.sum())
        results.append({
            "review_fraction": fraction,
            "accounts_reviewed": count,
            "observed_defaults_captured": captured,
            "precision_among_reviewed": float(captured / count),
            "recall_of_observed_defaults": float(captured / total_defaults),
            "random_expected_defaults": float(count * total_defaults / len(y)),
        })
    return results
