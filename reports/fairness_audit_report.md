# Fairness and responsible-use audit

## Scope

This is a held-out descriptive audit of a historical Taiwan 2005 dataset. The frozen XGBoost model, training-only isotonic calibration, and fixed global top-10% review policy were not reselected or optimized using these subgroup results. `X2`–`X5` are audit labels only; they never enter model fitting, calibration fitting, SHAP, threshold selection, or ranking.

## How to read the audit

Differences can arise from three distinct sources: (1) calibration/performance differences, such as predicted-risk versus observed-rate gaps or Brier/ROC-AUC differences; (2) a fixed global top-10% queue, which allocates a limited number of reviews across all accounts rather than 10% inside every subgroup; and (3) different historical outcome rates. None alone proves a model fair or unfair. ROC-AUC and average precision are withheld when a subgroup has fewer than 100 records or fewer than 20 of either historical outcome.

Overall held-out reference: 
6,000 accounts; observed default rate 22.12%; mean calibrated risk 21.98%; Brier 0.1330; top-10% capture 31.65%.

## Recorded sex category

| Subgroup | n | Observed rate | Mean predicted risk | ROC-AUC | Brier | TPR at 0.50 | FPR at 0.50 | Top-10% capture | Top-10% precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Female (documented code 2) | 3628 | 20.81% | 21.55% | 0.789 | 0.125 | 38.81% | 4.91% | 33.25% | 70.11% |
| Male (documented code 1) | 2372 | 24.11% | 22.65% | 0.777 | 0.146 | 34.79% | 5.28% | 29.55% | 69.83% |
## Recorded education category

| Subgroup | n | Observed rate | Mean predicted risk | ROC-AUC | Brier | TPR at 0.50 | FPR at 0.50 | Top-10% capture | Top-10% precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Graduate school (documented) | 2141 | 19.24% | 18.99% | 0.798 | 0.116 | 31.07% | 2.66% | 26.21% | 78.26% |
| High school (documented) | 989 | 26.49% | 25.58% | 0.750 | 0.159 | 40.08% | 8.12% | 35.50% | 63.70% |
| Other (documented code 4) | 28 | 3.57% | 15.68% | Not reported: small/imbalanced | 0.067 | 0.00% | 3.70% | 0.00% | 0.00% |
| Undocumented education code(s) | 67 | 8.96% | 19.13% | Not reported: small/imbalanced | 0.116 | 16.67% | 11.48% | 16.67% | 12.50% |
| University (documented) | 2775 | 23.28% | 23.14% | 0.786 | 0.137 | 39.94% | 5.78% | 33.75% | 71.01% |
## Recorded marital-status category

| Subgroup | n | Observed rate | Mean predicted risk | ROC-AUC | Brier | TPR at 0.50 | FPR at 0.50 | Top-10% capture | Top-10% precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Married (documented) | 2731 | 23.87% | 22.21% | 0.780 | 0.141 | 36.96% | 5.15% | 32.36% | 71.53% |
| Other (documented code 3) | 65 | 21.54% | 22.98% | Not reported: small/imbalanced | 0.140 | 28.57% | 3.92% | 28.57% | 66.67% |
| Single (documented) | 3196 | 20.68% | 21.75% | 0.791 | 0.126 | 37.37% | 4.93% | 31.01% | 69.02% |
| Undocumented marital-status code | 8 | 0.00% | 28.73% | Not reported: small/imbalanced | 0.123 | nan% | 25.00% | nan% | 0.00% |
## Age band

| Subgroup | n | Observed rate | Mean predicted risk | ROC-AUC | Brier | TPR at 0.50 | FPR at 0.50 | Top-10% capture | Top-10% precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20–29 | 1926 | 21.86% | 23.75% | 0.789 | 0.131 | 42.04% | 6.51% | 35.15% | 68.52% |
| 30–39 | 2206 | 19.90% | 19.69% | 0.787 | 0.124 | 32.57% | 3.34% | 27.33% | 71.43% |
| 40–49 | 1296 | 24.54% | 22.11% | 0.788 | 0.141 | 36.16% | 4.70% | 31.45% | 74.07% |
| 50+ | 572 | 26.05% | 24.56% | 0.753 | 0.156 | 38.26% | 7.80% | 34.90% | 64.20% |

## Responsible use, from zero

Excluding demographics does not automatically make a model fair because other account variables can be proxies: variables associated with a group may carry some of the same historical patterns. Historical defaults can also reflect past lending policies, access, economic conditions, reporting practices, and other social context rather than an inherent individual attribute. A real deployment would need independent validation, governance, monitoring, human review, an appeal/escalation route, and legal/compliance assessment. This educational project does not justify automatic approval, rejection, pricing, collections, or any other action.

## Cautions

This audit is descriptive and based on one historical source, random held-out split, and recorded categories. It is not fairness certification, causal evidence, regulatory compliance, or evidence for current Indian banking. Small groups are explicitly flagged; apparent gaps should trigger investigation, not conclusions.