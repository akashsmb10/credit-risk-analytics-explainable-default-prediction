# Final project checklist

Packaging review completed for the educational Credit Risk Analytics & Explainable Default Prediction project.

| Check | Status | Evidence / action |
| --- | --- | --- |
| One canonical notebook | PASS | Kept `notebooks/01_explore_data.ipynb`; removed the root duplicate. |
| No stale native-importance artifact | PASS | Removed the obsolete `reports/global_feature_importance.csv`; current SHAP artifacts are clearly labelled Kernel SHAP approximations. |
| Progress and decisions reconciled | PASS | Updated `notes/progress.md`, `notes/decisions.md`, and `notes/mistakes.md` for the completed model, calibration, SHAP, fairness, and dashboard work. |
| Raw data and model artifacts ignored | PASS | `.gitignore` excludes `data/`, `models/`, processed data, credentials, local environments, and caches. |
| Small explanatory reports and figures available | PASS | Saved reports and figures remain available for a GitHub reader; raw CSV and trained model are excluded. |
| Pinned Python dependencies | PASS | `requirements.txt` pins verified package versions. |
| Reproducible commands documented | PASS | README documents environment setup, tests, analysis-report regeneration, and dashboard launch. |
| Read-only dashboard boundary | PASS | Dashboard loads saved reports/figures only; it has no customer input, inference, ID, demographic, or individual-record view. |
| Verified model claims | PASS | README uses saved held-out calibration/policy results and states historical capture is not prevention or savings. |
| Responsible-use limits | PASS | README, model card, calibration, SHAP, and fairness reports state non-production, non-causal, non-fairness-certification limits. |
| Automated tests | PASS | Final packaging run: 9 passed. |
| Streamlit startup | PASS | Verified with `py -3.10 -m streamlit run app.py --server.headless true`. |
| Git remote / publication | NOT CONFIGURED | No repository or remote was created or pushed during this pass; use the upload commands in the README handoff. |

## Final manual reminders before public upload

- Confirm the UCI attribution and dataset license remain appropriate for the intended repository visibility.
- Review every staged file with `git status` and `git diff --cached` before committing.
- Do not add `data/`, `models/`, `.env`, credentials, cache folders, or any external customer data.
