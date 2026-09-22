# Project decisions

- Use a Jupyter notebook to keep explanations, code, and outputs together.
- Use the official UCI Default of Credit Card Clients dataset. Retain the original New Taiwan dollar (NT$) amounts.
- Read the existing local CSV instead of downloading it on each run.
- Use Python 3.10, which has the required pandas and notebook support installed.
- Distinguish recorded data, calculated quantities, model predictions, and simulation assumptions.
- Keep downloaded records, credentials, and local environment files out of version control.
- Treat the project as an educational prototype, not a validated lending system.
- Initial model features are X1 and X6-X23. ID is excluded as an identifier; X2-X5 are excluded from the first model because they are customer attributes.
- Use a stratified random 60/20/20 split with random_state=42 because no suitable chronological event date is available.
- Select the model by validation ROC-AUC before test evaluation. The measured validation winner was XGBoost.
- SHAP 0.49.1 TreeExplainer cannot parse the saved XGBoost 3.2.0 model format. Use reproducible model-agnostic KernelExplainer for approximate SHAP values; label it as approximate and non-causal.
- Keep `notebooks/01_explore_data.ipynb` as the only canonical exploration notebook; the root duplicate was removed during packaging.
