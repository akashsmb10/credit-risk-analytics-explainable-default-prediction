# Mistakes and corrections

No calculation or interpretation mistakes recorded yet.

## Environment issue

- Error: `ModuleNotFoundError: No module named 'pandas'` when using the default Python installation.
- Cause: pandas was available in Python 3.10 but not in the default Python installation.
- Resolution: use Python 3.10 as the notebook kernel.
- Takeaway: different Python installations can have different packages available.

## Explainability compatibility issue

- SHAP 0.49.1 failed with XGBoost 3.2.0 because SHAP could not parse the model's array-valued base score.
- Resolution: use SHAP `KernelExplainer` as a reproducible, model-agnostic fallback. It calls the unchanged saved model and produces approximate SHAP values.
- Limitation: Kernel SHAP is slower and approximate; its global and local values describe learned associations, not causes or decision instructions.
