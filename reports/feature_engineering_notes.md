# Feature-engineering notes

## Purpose and leakage boundary

These features summarise the historical April–September 2005 fields that precede the documented `Y` outcome. `build_engineered_features()` never reads `Y`, `ID`, or `X2`–`X5`, and it does not calculate statistics across customers. It therefore cannot learn from validation or test rows while constructing row-level summaries. Any later learned transformation still must be fitted on training data only.

The frozen baseline is `X1` and `X6`–`X23`. `ENGINEERED_MODEL_FEATURES` is that baseline plus the safe summaries. `repayment_status_latest` remains available but is not added twice because it exactly duplicates baseline `X6`.

## Repayment history: X6–X11

| Feature | What it calculates / why it may help | Formula, limitation, and leakage check |
| --- | --- | --- |
| `repayment_status_max` | Largest observed status code; may capture a higher documented positive delay. | Largest of six codes. It does not define delinquency fully; `0` and `-2` remain undocumented. Uses only pre-outcome fields. |
| `repayment_status_mean` | Average numerical level over six observed codes. | Sum of six codes divided by six. It is not “average months late”; undocumented codes are preserved. No target used. |
| `repayment_positive_delay_count` | How often a positive code occurs; may summarise documented positive delays. | Count of codes greater than zero. It does not call `0`/`-2` on-time. No future data used. |
| `repayment_status_latest` | Most recent status, raw `X6`; may be useful because it is latest. | Equals `X6`; it is not duplicated in the model list. `0`/`-2` keep unknown meaning. Pre-outcome only. |
| `repayment_latest_minus_earlier_mean` | Numerical difference between latest code and earlier history. | `X6 − mean(X7…X11)`. Do not call it improvement/deterioration. No target or future fields used. |

## Bill history: X12–X17

| Feature | What it calculates / why it may help | Formula, limitation, and leakage check |
| --- | --- | --- |
| `bill_amount_mean`, `bill_amount_median` | Typical six-month recorded bill level. | Mean/median of six bills. A bill is not current debt or utilisation. Within-row pre-outcome summary only. |
| `bill_amount_max`, `bill_amount_min`, `bill_amount_range` | Highest, lowest, and numerical spread in recorded bills. | Maximum, minimum, maximum minus minimum. Retains negative/zero values without a story. No target used. |
| `bill_amount_trend_slope` | Linear April-to-September numerical bill trend. | Slope in NT$ per month across six points. Not proof of rising debt or causation. Pre-outcome fields only. |

## Payment history: X18–X23

| Feature | What it calculates / why it may help | Formula, limitation, and leakage check |
| --- | --- | --- |
| `payment_amount_mean`, `payment_amount_median` | Typical six-month recorded payment scale. | Mean/median of six payments. Payments are not confirmed minimum/full repayments. No target used. |
| `payment_amount_max`, `payment_amount_min`, `payment_amount_range` | Highest, lowest, and variation in payment amounts. | Maximum, minimum, maximum minus minimum. No bill/payment matching or future field used. |
| `payment_amount_trend_slope` | Linear April-to-September payment trend. | Slope in NT$ per month across six points. Only six observations; does not prove changed ability/willingness. No target used. |

## Credit limit: X1

Raw `X1` remains. Because EDA found it right-skewed, `make_credit_limit_preprocessor()` provides an **unfitted** pipeline factory that retains raw `X1` and adds `log1p(X1)` followed by `RobustScaler`. It is not called here. A future pipeline must fit it only on its training partition. Credit-limit bands are not used as the sole input.

## Explicit exclusions

- Payment-to-bill ratios: no documented within-month bill/payment ordering or pairing.
- “Current balance = bill − payment”: this is not a transaction ledger.
- `ID`, `X2`–`X5`, and `Y`: excluded from inputs.
- Validation/test-fitted transformations: none are fitted in this module.
