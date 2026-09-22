# SHAP explainability report

## Compatibility solution

SHAP 0.49.1 TreeExplainer fails with saved XGBoost 3.2.0 because XGBoost serializes `base_score` as a vector string while the SHAP tree loader expects one float. To avoid changing or retraining the frozen model, this project uses SHAP KernelExplainer: a slower, approximate model-agnostic method that repeatedly calls the unchanged model. The background is 40 stratified held-out test records and the global sample is 160 stratified held-out test records.

## SHAP from zero

The base value is the average raw XGBoost probability over the SHAP background. For one record, positive SHAP values push the raw model probability upward and negative values push it downward. Global SHAP averages contribution sizes over records; local SHAP explains one record. These are learned associations, not causes of default.

## Top global associations

| Feature | Mean absolute SHAP | Interpretation |
| --- | ---: | --- |
| X6 | 0.063408 | September repayment-status code; 0 and -2 remain undocumented. |
| X1 | 0.019695 | Granted credit amount; association only, not an effect of changing the limit. |
| X19 | 0.014968 | August recorded payment amount; association only, not proof of repayment quality. |
| X20 | 0.014033 | July recorded payment amount; association only, not proof of repayment quality. |
| X8 | 0.012302 | July repayment-status code; historical association only. |
| X12 | 0.012039 | September bill-statement amount; not a current-balance measure. |
| X18 | 0.011986 | September recorded payment amount; association only, not proof of repayment quality. |
| X7 | 0.007883 | August repayment-status code; historical association only. |
| X9 | 0.006944 | June repayment-status code; historical association only. |
| X21 | 0.006054 | Historical account amount or status; association only, not causation. |

## Local cases

The following reproducibly selected held-out cases never display IDs or demographic fields. Their tables show calibrated probability, queue status, historical outcome, and raw-model SHAP directions. A waterfall is not a decision or causal explanation.

### high risk recorded default

Calibrated probability: **0.9563**; queue status: **top 10% review queue**; historical outcome: **recorded default next month**.
The shown inputs pushed this fixed model's raw probability up or down. They do not prove why the outcome occurred and should support human review, not determine an outcome.

### high risk no recorded default

Calibrated probability: **0.8003**; queue status: **top 10% review queue**; historical outcome: **no recorded default next month**.
The shown inputs pushed this fixed model's raw probability up or down. They do not prove why the outcome occurred and should support human review, not determine an outcome.

### lower risk recorded default

Calibrated probability: **0.0170**; queue status: **outside top 10% review queue**; historical outcome: **recorded default next month**.
The shown inputs pushed this fixed model's raw probability up or down. They do not prove why the outcome occurred and should support human review, not determine an outcome.

## Calibration and responsible use

Isotonic calibration maps raw model scores to final displayed probabilities, but does not change the frozen XGBoost trees or their SHAP contributions. SHAP is not causality, fairness proof, regulatory compliance, or evidence of suitability for lending. This is historical Taiwan credit-card data from 2005, not current Indian banking data or a live bank portfolio. Explanations can drift and should help a reviewer ask questions, never approve, deny, price, or collect from a person.