-- Grain: one historical UCI credit-card record per row.
-- Y is a known historical outcome here; it is not available for future prioritisation.

-- Overall portfolio outcome.
SELECT COUNT(*) AS customers, SUM(Y) AS observed_defaults,
       AVG(Y) AS observed_default_rate
FROM credit_cards;

-- Default outcome by granted-credit band; each group includes its sample size.
SELECT CASE
         WHEN X1 < 50000 THEN 'Below NT$50k'
         WHEN X1 < 100000 THEN 'NT$50k–99,999'
         WHEN X1 < 200000 THEN 'NT$100k–199,999'
         ELSE 'NT$200k or more'
       END AS credit_limit_band,
       COUNT(*) AS customers, SUM(Y) AS observed_defaults, AVG(Y) AS observed_default_rate
FROM credit_cards
GROUP BY credit_limit_band;

-- Repayment-status groups. The official documentation defines -1 and 1–9;
-- other stored codes are kept as 'Undocumented code'.
WITH status_groups AS (
  SELECT CASE
           WHEN X6 = -1 THEN 'Paid duly (-1)'
           WHEN X6 BETWEEN 1 AND 8 THEN '1–8 months delay'
           WHEN X6 = 9 THEN '9+ months delay'
           ELSE 'Undocumented code'
         END AS repayment_group,
         Y
  FROM credit_cards
)
SELECT repayment_group, COUNT(*) AS customers, SUM(Y) AS observed_defaults,
       AVG(Y) AS observed_default_rate
FROM status_groups
GROUP BY repayment_group;

-- COUNT(*) counts every row; COUNT(X18) counts non-null payment values.
SELECT COUNT(*) AS all_rows, COUNT(X18) AS non_null_payment_amounts
FROM credit_cards;

-- Rank eligible segments by observed historical default rate.
WITH segments AS (
  SELECT CASE WHEN X1 < 50000 THEN 'Below NT$50k'
              WHEN X1 < 100000 THEN 'NT$50k–99,999'
              WHEN X1 < 200000 THEN 'NT$100k–199,999'
              ELSE 'NT$200k or more' END AS segment,
         COUNT(*) AS customers, AVG(Y) AS observed_default_rate
  FROM credit_cards GROUP BY segment
)
SELECT *, RANK() OVER (ORDER BY observed_default_rate DESC) AS risk_rank
FROM segments WHERE customers >= 100;
