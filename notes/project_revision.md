# Project revision sheet

## Core problem
Use historical information available before the documented next-month outcome to estimate and rank default-payment risk for a limited manual-review queue.

## Key terms
- Target: `Y`, the documented default-payment-next-month outcome.
- Feature: an input available before the target; `Y` cannot be a feature.
- Leakage: using future or target information when training a model.
- Stratification: preserve the target balance in each random split.
- ROC-AUC: ranking quality across thresholds; 0.5 is random ranking.
- Precision: observed defaults among accounts flagged by a policy.
- Recall: share of observed defaults captured by a policy.
- Calibration: agreement between predicted probabilities and observed frequencies; Brier score is one probability-error measure.

## Interview questions
1. Why exclude ID? It identifies records but does not describe credit behaviour.
2. Why exclude X2-X5 initially? They include customer attributes; the first model prioritises repayment and account data while fairness implications are assessed.
3. Why stratify? Default outcomes are imbalanced, so each split needs a similar outcome mix.
4. Why not accuracy alone? A no-default baseline is 77.88% accurate and captures no defaults.
5. What is the simulator? A retrospective ranking exercise using known validation outcomes, not evidence of prevented defaults.
6. What is the key limitation? Historical 2005 Taiwan data cannot validate a current production lending policy.

## Resume bullets
- Built a reproducible credit-risk prototype on 30,000 UCI Taiwan credit-card records, with documented data-quality checks and leakage-aware feature exclusions.
- Compared dummy, logistic-regression, and XGBoost models using a stratified 60/20/20 split; selected XGBoost on validation ROC-AUC (0.7745).
- Simulated a 10% manual-review capacity that captured 401 of 1,327 observed validation defaults (30.22%), labelled as a retrospective analysis.
