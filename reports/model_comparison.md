# Baseline versus engineered feature comparison

## What makes this fair

Both feature sets use exactly the same stratified 60/20/20 row indices (`random_state=42`), the same target, and the same dummy, logistic-regression, and XGBoost settings. The baseline list is frozen as `X1` and `X6`–`X23`; the engineered set adds only documented row-level summaries. `ID`, `X2`–`X5`, `Y`, payment-to-bill ratios, and reconstructed balances are excluded. Logistic-regression scaling occurs inside its sklearn pipeline and is fit only to training folds. No model or metric from the existing baseline artifact was overwritten.

## Validation and test roles

Training data fits candidate models. Validation data chooses the winner by ROC-AUC and calculates the review-capacity simulation. The held-out test set is evaluated once after that selection; it did not choose models, features, or settings. The 5-fold cross-validation scores are calculated only on the training partition for each selected candidate.

## Results

| Feature set | Features | Selected model | Validation ROC-AUC | Test ROC-AUC | Validation AP | Test AP | Test Brier | 5-fold train ROC-AUC | 5-fold train AP |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Baseline | 19 | xgboost | 0.7745 | 0.7840 | 0.5347 | 0.5681 | 0.1333 | 0.7795 ± 0.0060 | 0.5534 ± 0.0120 |
| Engineered | 35 | xgboost | 0.7753 | 0.7851 | 0.5400 | 0.5702 | 0.1324 | 0.7836 ± 0.0070 | 0.5566 ± 0.0134 |

### Ranking, calibration, classification, and review capture

The following test metrics use the inherited 0.50 reporting cutoff only; it is not a chosen business threshold.

| Feature set | Accuracy | Precision | Recall | F1 | Brier score | Test top-10% defaults captured |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.8212 | 0.6774 | 0.3655 | 0.4748 | 0.1333 | 418 / 600 (31.50%) |
| Engineered | 0.8225 | 0.6866 | 0.3632 | 0.4751 | 0.1324 | 424 / 600 (31.95%) |

### Interpretation

Engineered minus baseline validation ROC-AUC is **+0.0008**. The pre-specified meaningful-improvement margin is **0.0050**.

The engineered feature set does not clear the pre-specified validation margin. Any difference is treated as small/noisy for this project stage, so the simpler baseline remains the final candidate.

Final candidate: **baseline** using **xgboost**. This selection uses validation ROC-AUC only. Calibration and a business threshold have not been selected.

## Capacity results

Full validation and test top-5%, top-10%, and top-20% review results, including observed-default capture and random-selection reference, are in `model_comparison_metrics.json`. They are retrospective simulations using known historical outcomes, not prevented-default or savings estimates.