# Calibration and review-policy analysis

## Calibration, in simple terms

A calibrated probability is a probability whose predicted level approximately matches the observed event frequency. For example, among accounts assigned roughly 20% risk, we would hope roughly 20% historically defaulted. ROC-AUC only tests whether defaults tend to rank above non-defaults; it can be strong even when probability levels are too high or too low.

Brier score is the average squared gap between each predicted probability and outcome (lower is better). Expected calibration error (ECE) is the weighted average gap between mean prediction and observed rate in fixed probability bins (lower is better). Both are historical diagnostic measures, not guarantees for future customers.

## Calibration selection on validation data

The saved baseline XGBoost supplied uncalibrated predictions. Sigmoid (Platt) and isotonic calibration wrappers were fitted using only the 18,000-row training partition with internal 5-fold cross-validation. Selection uses lowest validation Brier score; validation ECE is the stated tie-breaker. Test results did not select the method.

| Method | Validation Brier | Validation ECE | Test Brier | Test ECE |
| --- | ---: | ---: | ---: | ---: |
| uncalibrated | 0.1380 | 0.0175 | 0.1333 | 0.0134 |
| sigmoid | 0.1390 | 0.0316 | 0.1342 | 0.0235 |
| isotonic | 0.1373 | 0.0138 | 0.1330 | 0.0090 |

Selected calibration: **isotonic**. Reliability curves and full bin tables are in `calibration_threshold_metrics.json`; test values are reported only after this validation choice.

## Thresholds and capacity

A 0.50 threshold is not automatically correct: it means flag when predicted risk is at least 50%, but it ignores staffing capacity and the relative consequences of false positives (unnecessary reviews) and false negatives (missed observed defaults). The threshold table evaluates 0.10–0.90 on validation only. Its illustrative cost is `false positives × 1 + false negatives × 5` or `× 10`; these are classroom sensitivity assumptions, not bank costs or savings.

A top-k policy instead reviews a fixed number of the highest-ranked accounts. It is often easier to operate when capacity is fixed because exactly the chosen share is sent to review even if the probability scale changes. This project’s pre-specified demonstration policy is **top 10% ranked accounts**, consistent with the earlier capacity simulation; it was not selected after looking at test outcomes.

| Split | Policy | Accounts reviewed | Defaults captured | Capture rate | Precision among reviewed | False positives | Missed defaults |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | Top 10% | 600 | 402 | 30.29% | 67.00% | 198 | 925 |
| Held-out test | Top 10% | 600 | 420 | 31.65% | 70.00% | 180 | 907 |

The 0.50 classification metrics in the JSON remain descriptive only. No business classification threshold has been declared as universally correct.