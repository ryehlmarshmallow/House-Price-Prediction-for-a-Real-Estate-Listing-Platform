# Verified results fact sheet

This file is the single reference for numbers quoted in the report deck, slides and
documentation. All values were produced by the committed pipeline and can be re-verified
from the artefacts listed at the end.

## 1. Data

| Item | Value |
|---|---|
| Labelled transactions (Kaggle train) | 1,460 |
| Unlabelled transactions (Kaggle test) | 1,459 |
| Raw attributes | 79 explanatory + identifier + target |
| Neighbourhoods | 25 |
| Sale period | 2006-2010 |
| Median sale price | 163,000 USD |
| Mean sale price | 180,933 USD |
| Price skewness | 1.881 |
| Price skewness after log transform | 0.122 |
| Interquartile outliers in price | 61 (4.2%) |
| Synthetic contextual fields | 21 |
| Missing cells before cleaning (train) | 7,829 |
| Records removed as implausible | 2 |
| Records after cleaning (train) | 1,458 |
| Duplicate records | 0 |

## 2. Exploratory findings

* Price is right-skewed (1.88); the natural log reduces skewness to 0.12.
* The strongest Pearson correlations with price are total floor area (0.83),
  overall quality (0.80), living area (0.74), basement area (0.65) and garage
  capacity (0.64).
* The strongest negative correlations are the crime index (-0.55), age at sale
  (-0.52) and distance to transit (-0.37).
* The amenity score has a Spearman correlation of 0.652 with price; tier-5
  neighbourhoods average an amenity score of 85.5 against 43.5 in tier-1
  neighbourhoods.
* Median price by neighbourhood tier: tier 1 109,000; tier 2 125,500; tier 3
  148,000; tier 4 196,750; tier 5 306,000 USD.
* Median days on market is 31 (mean 38.6, 90th percentile 69); the Spearman
  correlation between days on market and price is -0.435.
* Median price by construction cohort rises from 108,000 for the 1870s cohort to
  224,000 for the 2000s cohort, with the small 2010 cohort at 394,000; the full
  table is in `artifacts/metrics/eda_summary.json`.
* Property type: single-family homes (1,218 listings) have a median of 167,900 USD,
  end-unit townhouses (114) 172,200 USD, interior townhouses (43) 137,500 USD,
  duplexes (52) 135,980 USD and two-family conversions (31) 127,500 USD. Two-storey
  homes (443) have a median of 190,000 USD against 154,750 USD for single-storey
  homes (726). Figure: `artifacts/figures/fig25_price_by_property_type.png`.
* Multicollinearity: total floor area and living area correlate at 0.87; garage
  area and garage capacity at 0.88. VIF pruning retains 47 of 55 numeric features
  with a maximum final VIF of 8.95.

## 3. Model comparison (five-fold out-of-fold)

| Model | RMSE (USD) | MAE (USD) | MAPE (%) | R-squared |
|---|---|---|---|---|
| Elastic net | 20,019 | 13,537 | 7.98 | 0.9365 |
| Lasso | 20,023 | 13,625 | 8.02 | 0.9365 |
| Ridge | 20,172 | 13,753 | 8.13 | 0.9356 |
| Linear regression | 21,135 | 14,500 | 8.56 | 0.9293 |
| XGBoost | 22,471 | 14,419 | 8.25 | 0.9200 |
| LightGBM | 22,518 | 14,665 | 8.49 | 0.9197 |
| Gradient boosting | 22,689 | 14,674 | 8.49 | 0.9185 |
| Random forest | 25,725 | 15,803 | 9.02 | 0.8952 |

Selected model: elastic net on the log price with 102 candidate features.
Conformal quantiles: 80% = 0.1173 log points; 90% = 0.1648 log points.

## 4. Selected model performance

| Metric | Value |
|---|---|
| RMSE | 20,019 USD |
| MAE | 13,537 USD |
| Median absolute error | 9,356 USD |
| MAPE | 7.98% |
| R-squared | 0.9365 |
| Within +/- 15,000 USD | 69.7% |
| Within +/- 8% | 64.9% |
| Over-valuation rate | 48.0% |
| Mean asymmetric cost | 16,708 USD |
| Median asymmetric cost | 11,358 USD |

Temporal holdout (fit 2006-2009, test 2010, n = 175): RMSE 22,269; MAE 14,820;
MAPE 8.71%; R-squared 0.923; within +/- 15,000 USD 63.4%.

## 5. Interval calibration (out-of-fold)

| Nominal | Segment | Empirical coverage | Gap | Mean width (% of price) |
|---|---|---|---|---|
| 80% | All | 0.801 | +0.001 | 23.7 |
| 80% | Entry (<130k) | 0.718 | -0.082 | 24.6 |
| 80% | Mid (130k-200k) | 0.836 | +0.036 | 23.5 |
| 80% | Upper (200k-300k) | 0.831 | +0.031 | 23.3 |
| 80% | Luxury (>=300k) | 0.783 | -0.017 | 22.6 |
| 90% | All | 0.901 | +0.001 | 33.3 |
| 90% | Entry (<130k) | 0.833 | -0.067 | 34.7 |
| 90% | Mid (130k-200k) | 0.941 | +0.041 | 33.1 |
| 90% | Upper (200k-300k) | 0.909 | +0.009 | 32.8 |
| 90% | Luxury (>=300k) | 0.870 | -0.030 | 31.8 |

## 6. Residual diagnostics

| Diagnostic | Value |
|---|---|
| Mean log residual | -0.00055 |
| Standard deviation of log residuals | 0.1112 |
| Skewness of log residuals | -1.141 |
| Jarque-Bera statistic / p-value | 2,861 / < 0.001 |
| Breusch-Pagan LM / p-value | 131.9 / < 0.001 |
| 90th percentile absolute error | 29,122 USD |
| Worst 5% of cases, share of total absolute error | 22.1% |

## 7. Subgroup performance

| Segment | Sales | MAE (USD) | Bias (%) |
|---|---|---|---|
| Value neighbourhoods | 389 | 11,581 | +1.34 |
| Established neighbourhoods | 926 | 12,892 | +0.54 |
| Premium neighbourhoods | 143 | 23,035 | +0.08 |

| Price band | Sales | RMSE (USD) | MAE (USD) | Bias (%) |
|---|---|---|---|---|
| Entry (<130k) | 365 | 14,521 | 10,295 | +4.70 |
| Mid (130k-200k) | 658 | 15,413 | 10,973 | +0.08 |
| Upper (200k-300k) | 320 | 21,864 | 16,147 | -0.91 |
| Luxury (>=300k) | 115 | 41,499 | 31,238 | -3.92 |

## 8. Drivers (mean absolute SHAP share)

| Feature | Share (%) | Family |
|---|---|---|
| tot_rms_abv_grd | 8.73 | Structural |
| total_sf | 6.51 | Structural composite |
| quality_condition_index | 6.31 | Structural composite |
| avg_room_size | 4.51 | Structural |
| age_at_sale | 3.41 | Age and renovation |
| bsmt_fin_type1 | 3.19 | Categorical |
| neigh_lat | 3.05 | Location |
| total_bath | 2.76 | Structural composite |

Family shares: structural 43.35%, categorical 27.82%, structural composites 15.87%,
age and renovation 6.21%, location 3.20%, neighbourhood amenities 2.40%,
macro-financial 1.14%.

## 9. Monitoring replay

Reference model: refitted on 945 sales, January 2006 to December 2008.
Reference out-of-fold RMSE: 20,888 USD.
Stream: 513 sales, January 2009 to July 2010, four equal-volume windows.

| Window | Sales | RMSE | MAE | MAPE | Bias | 90% coverage | RMSE ratio |
|---|---|---|---|---|---|---|---|
| W1 2009-01 to 2009-06 | 129 | 22,306 | 14,813 | 9.41% | -0.16% | 89.1% | 1.068 |
| W2 2009-06 to 2009-09 | 128 | 22,306 | 16,143 | 9.14% | -1.57% | 87.5% | 1.068 |
| W3 2009-09 to 2010-04 | 128 | 23,395 | 15,977 | 9.63% | -2.52% | 83.6% | 1.120 |
| W4 2010-04 to 2010-07 | 128 | 21,839 | 16,048 | 9.34% | -4.99% | 79.7% | 1.046 |

Drift: 5 to 7 features exceed PSI 0.20 in every window, concentrated in the macro
block (mortgage rate, house price index, unemployment, months of supply). Prediction
PSI rises from 0.066 to 0.174. The observed replay raises four R1 warnings, two R4
coverage warnings and the critical rule R7 (sustained coverage erosion); decision:
retrain now.

Stress scenario: mortgage rate +1.5 percentage points, inventory +1 month of supply,
amenity improvement 60%. RMSE ratio peaks at 1.152, coverage falls to 81.2%, bias
reaches -5.23%; additional bias rule R5 fires. Decision: retrain now.

## 10. Deployment status

Locally verified: FastAPI service, Streamlit console, monitoring report. No cloud
credentials were available; no public URL is claimed. Render, Docker and Hugging Face
Spaces configurations are present in `deploy/` and `render.yaml`.

## 11. Artefact map

| Content | Path |
|---|---|
| Model comparison | `artifacts/metrics/model_comparison.csv` |
| Training summary | `artifacts/metrics/training_summary.json` |
| Evaluation diagnostics | `artifacts/metrics/evaluation_diagnostics.json` |
| Cleaning report | `artifacts/metrics/cleaning_report.json` |
| EDA summary | `artifacts/metrics/eda_summary.json` |
| VIF table | `artifacts/metrics/vif.csv` |
| SHAP importance | `artifacts/metrics/shap_importance.csv` |
| Interval coverage | `artifacts/metrics/interval_coverage.csv` |
| Monitoring summary | `artifacts/monitoring/monitoring_summary.json` |
| Monitoring report | `artifacts/monitoring/monitoring_report.html` |
| Figures | `artifacts/figures/fig01...fig24.png` |
| Data dictionary | `docs/data_dictionary.md`, `docs/data_dictionary.xlsx` |
| Application screenshots | `artifacts/reports/ui_home.png`, `artifacts/reports/ui_result.png`, `artifacts/reports/ui_monitoring.png` |
