# Progress

## Current work

- Canonical notebook: `notebooks/01_explore_data.ipynb`.
- Dataset (local, gitignored): `data/raw/default_credit_card_clients.csv`.
- Project state: EDA, baseline comparison, calibration, review-capacity simulation, SHAP explainability, fairness audit, and a read-only dashboard are complete.

## Verified

- CSV contains 30,000 records and 25 columns.
- Notebook EDA and feature-preview sections use the reusable project functions.
- The preview displays records with IDs 1 through 5.
- `df.shape` returned `(30000, 25)`.
- `df.columns.tolist()` returned `ID`, `X1` through `X23`, and `Y`.
- Verified UCI documentation maps the selected columns to `LIMIT_BAL`, `PAY_0`, `BILL_AMT1`, `PAY_AMT1`, and the next-month default-payment outcome.
- The first five selected records were displayed without changing the data.
- `df.info()` reported 25 integer columns and 30,000 non-null entries per column.
- Missing-value counts and duplicate-ID count were zero.
- Target counts were 23,364 for `Y=0` and 6,636 for `Y=1` (77.88% and 22.12%).
- Frozen baseline XGBoost was selected by validation ROC-AUC; isotonic calibration was selected on validation calibration diagnostics.
- The educational demonstration policy is a retrospective top-10% calibrated-score review queue.

## Next step

- Explain why an always-no-default model can appear accurate before proceeding to exploratory analysis.

## Implementation status

- Reusable data loading, feature selection, training, evaluation, SQL analysis, dashboard, tests, model card, data dictionary, and revision sheet have been added.
- The training run produced metrics in `reports/metrics.json` and a selected model in `models/selected_model.joblib`.
- Tests passed: 9 passed at the dashboard packaging stage.

## Reference material

- Dataset source: UCI Default of Credit Card Clients, dataset 350: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- Key reports: `reports/model_card.md`, `reports/calibration_threshold_report.md`, `reports/shap_explainability_report.md`, and `reports/fairness_audit_report.md`.
