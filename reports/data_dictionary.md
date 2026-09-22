# Data dictionary

Source: official [UCI Default of Credit Card Clients metadata](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients). A local provenance copy may exist outside version control, but this public UCI page is the repository reference.

| CSV field | Official meaning | Timing / unit |
| --- | --- | --- |
| ID | Record identifier | No unit or month |
| X1 | LIMIT_BAL, amount of granted credit including supplementary family credit | NT$; UCI does not assign a month |
| X2-X5 | Sex, education, marital status, age | Customer attributes; age is years |
| X6-X11 | Past repayment status | September through April 2005 |
| X12-X17 | Bill statement amounts | September through April 2005, NT$ |
| X18-X23 | Previous payment amounts | September through April 2005, NT$ |
| Y | Default payment next month | Yes = 1, No = 0 |

## Repayment-status documentation

UCI defines `-1` as paid duly; `1` through `8` as one through eight months of payment delay; and `9` as nine months or more. Values such as `0` and `-2` occur in the CSV but are not defined in the supplied UCI documentation, so this project does not assign them a business meaning.

## Interpretation limits

Bill and payment fields have the same listed calendar months but UCI does not specify their within-month timing or pairing. A bill-payment ratio is therefore not presented as a valid repayment measure. No causal claims are made from these data.
