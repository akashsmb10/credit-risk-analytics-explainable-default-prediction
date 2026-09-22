"""Read-only loaders for saved dashboard artifacts; never load raw customer records."""
import json
from pathlib import Path


REPORT_JSON = [
    "metrics.json", "model_comparison_metrics.json", "calibration_threshold_metrics.json",
    "shap_explainability_metrics.json", "fairness_audit_metrics.json",
]
REPORT_MARKDOWN = [
    "model_card.md", "data_dictionary.md", "feature_engineering_notes.md",
    "calibration_threshold_report.md", "shap_explainability_report.md", "fairness_audit_report.md",
]


def read_json_if_present(path):
    """Return parsed JSON or None for a missing/unreadable optional artifact."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def read_text_if_present(path):
    """Return report text or None for a missing/unreadable optional artifact."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None


def load_dashboard_artifacts(root):
    """Load reports and figures only; intentionally excludes data/, models/, and IDs."""
    root = Path(root)
    reports = root / "reports"
    return {
        "json": {name: read_json_if_present(reports / name) for name in REPORT_JSON},
        "markdown": {name: read_text_if_present(reports / name) for name in REPORT_MARKDOWN},
        "figures_dir": reports / "figures",
    }
