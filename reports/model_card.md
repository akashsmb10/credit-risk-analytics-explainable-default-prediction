# Model card

## Objective
Estimate the probability of a recorded default payment in the following month for existing historical credit-card records, then rank accounts for a simulated limited-capacity review.

## Dataset and target
UCI Default of Credit Card Clients: 30,000 historical Taiwan records, with monetary values in NT$. `Y=1` is default payment next month and `Y=0` is no default payment next month. The history covers April–September 2005; “October 2005” is a calendar inference, not a live forecast.

## Intended and non-intended use
Educational decision-support prototype and portfolio project only. It is not an approval, denial, pricing, collection, or production lending system. It does not represent India, current customers, or any bank's portfolio.

## Feature choices
Initial models use `X1` and `X6`–`X23`. `ID` is excluded because it is an identifier. `X2`–`X5` are excluded initially because they include customer attributes and are not necessary for the first risk model. All selected fields precede the recorded next-month outcome, but historical documentation does not give exact transaction timing; feature limitations remain.

## Validation design and frozen choices
Stratified random 60/20/20 train/validation/test split, `random_state=42`. A chronological split is not available because records do not provide an event date suitable for ordering. Dummy, logistic-regression, and restrained XGBoost models were compared on validation ROC-AUC. XGBoost was selected before the test set was evaluated. The review policy ranks validation accounts by selected-model probability and reviews the highest 5%, 10%, or 20%.

## Measured results
See `reports/metrics.json` for reproducible validation and one-time test metrics. These results are historical measurements, not expected business savings.

## Calibration and demonstration review policy
The frozen baseline XGBoost model was evaluated without retraining. Sigmoid and isotonic calibration wrappers were fitted on the 18,000-row training partition only, each using internal 5-fold cross-validation. Isotonic was selected by the lowest validation Brier score (0.1373; validation ECE 0.0138), compared with uncalibrated XGBoost (0.1380; ECE 0.0175) and sigmoid (0.1390; ECE 0.0316). The held-out test Brier score for selected isotonic calibration was 0.1330 and test ECE was 0.0090.

The educational demonstration policy reviews the top 10% of accounts ranked by selected calibrated probability. This capacity was specified from validation-side project context, not optimized on test data; it is not a universal classification threshold or lending rule. In the one-time historical test check, it reviewed 600 accounts and included 420 of 1,327 observed defaults (31.65% capture; 70.00% observed-default precision). See `reports/calibration_threshold_report.md` and `reports/calibration_threshold_metrics.json` for full bin, threshold, and capacity results.

## Explainability
SHAP 0.49.1 TreeExplainer is incompatible with the saved XGBoost 3.2.0 model format, so this project uses a reproducible, model-agnostic KernelExplainer fallback on a stratified held-out test background and sample. It explains raw XGBoost probability contributions; isotonic calibration changes the displayed risk probability but not the frozen trees or their SHAP contributions. The strongest sampled global associations were `X6` (September repayment status), `X1` (granted credit), `X19` (August payment amount), `X20` (July payment amount), and `X8` (July repayment status). These are model associations, not causes. See `reports/shap_explainability_report.md`.

## Limitations and ethics
Calibration is measured only on historical random splits and can drift in another population or time period. The review-capacity and cost examples are retrospective teaching simulations, not savings estimates.

Kernel SHAP is an approximate fallback whose sampled values can differ from exact TreeSHAP. SHAP does not prove fairness, causality, regulatory compliance, or suitability for automated action.

## Held-out fairness and responsible-use audit
`X2`–`X5` were used only as held-out audit labels and never entered training, isotonic calibration fitting, SHAP, threshold selection, or ranking. The audit found subgroup differences that warrant investigation rather than a verdict: among sufficiently sized documented education groups, ROC-AUC ranged from 0.750 (high school) to 0.798 (graduate school), while Brier score ranged from 0.116 to 0.159. Small/imbalanced categories do not receive discrimination metrics. These descriptive gaps do not establish fairness or unfairness, causality, legal compliance, or suitability for a real lending system. See `reports/fairness_audit_report.md`.

Historical, single-source data can be stale and unrepresentative. Associations do not establish causes. The source’s default definition lacks an operational days-past-due threshold. Model outputs require monitoring, governance, fairness review, and human judgement before any real-world use.
