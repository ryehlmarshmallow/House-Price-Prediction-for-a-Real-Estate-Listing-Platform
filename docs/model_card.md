# Model card

## 1. Model overview

| Item | Value |
|---|---|
| Model | Elastic net regression on the natural-log price |
| Version | 1.0.0 |
| Training date | Recorded in `artifacts/models/model_metadata.json` |
| Training records | 1,458 complete transactions (Ames, Iowa, 2006-2010) |
| Candidate features | 102 inputs: 40 categorical, 29 structural, 12 structural composites, 10 neighbourhood, 5 macro-financial, 2 location, 4 age/renovation |
| Target | `SalePrice` in USD, modelled as `ln(1 + SalePrice)` |
| Uncertainty | Split-conformal intervals from out-of-fold absolute log residuals |

The model is the outcome of a benchmark of eight regression families under identical
five-fold cross-validation. The regularised linear model achieved the lowest
out-of-fold RMSE; the boosted tree ensembles were competitive but not superior on this
small, well-measured tabular sample. The benchmark and the selection rule are reproduced
by `scripts/04_train_models.py`.

## 2. Intended use

* Supporting listing agents with a recommended asking-price range.
* Screening listings that warrant a formal appraisal.
* Portfolio analysis of price drivers at neighbourhood level.
* Educational demonstration of an end-to-end analytics product.

## 3. Out-of-scope use

* Formal appraisals, mortgage underwriting, or legal valuation.
* Properties outside the Ames housing market and the 2006-2011 macro regime.
* New construction with no assessor history or unusual tenure arrangements.
* Automated pricing without human review for low-confidence estimates.
* Use as a fair-lending or credit decision instrument.

## 4. Performance

Out-of-fold metrics on 1,458 transactions:

| Metric | Value |
|---|---|
| RMSE | 20,019 USD |
| MAE | 13,537 USD |
| Median absolute error | 9,356 USD |
| MAPE | 7.98% |
| R-squared | 0.9365 |
| Within +/- 15,000 USD | 69.7% |
| Within +/- 8% | 64.9% |
| Mean asymmetric cost | 16,708 USD |
| 80% interval coverage | 0.801 |
| 90% interval coverage | 0.901 |
| 90% interval mean width | 33.3% of price |

Temporal holdout (fit on 2006-2009, evaluate on 2010, 175 sales): RMSE 22,269 USD,
MAE 14,820 USD, MAPE 8.71%, R-squared 0.923. The degradation relative to cross-validation
is expected, because the holdout is a single year and 2010 sales are concentrated in the
first half of the year.

## 5. Subgroup performance

| Segment | Sales | MAE (USD) | Bias (% of price) |
|---|---|---|---|
| Value neighbourhoods | 389 | 11,581 | +1.34 |
| Established neighbourhoods | 926 | 12,892 | +0.54 |
| Premium neighbourhoods | 143 | 23,035 | +0.08 |

| Price band | Sales | RMSE (USD) | MAE (USD) | Bias (%) |
|---|---|---|---|---|
| Entry (< 130k) | 365 | 14,521 | 10,295 | +4.70 |
| Mid (130k-200k) | 658 | 15,413 | 10,973 | +0.08 |
| Upper (200k-300k) | 320 | 21,864 | 16,147 | -0.91 |
| Luxury (>= 300k) | 115 | 41,499 | 31,238 | -3.92 |

Error is heteroscedastic: absolute error grows with price, and the luxury segment is both
the smallest and the most under-predicted. The 90% interval under-covers the entry band
(83.3%) and the luxury band (86.9%), which is a known limitation of a single global
conformal calibration.

## 6. Drivers

The largest global contributors by mean absolute SHAP value are the number of rooms
above grade (8.7%), total floor area (6.5%), the quality-condition index (6.3%), average
room size (4.5%) and age at sale (3.4%). Aggregated by family, structural attributes
account for 43.4% of the explanation mass, categorical attributes for 27.8%, structural
composites for 15.9%, age and renovation for 6.2%, location for 3.2%, neighbourhood
amenities for 2.4% and macro-financial context for 1.1%.

## 7. Training data and provenance

* Base: Kaggle *House Prices - Advanced Regression Techniques* (Ames, Iowa), 1,460
  labelled transactions with 79 attributes, plus 1,459 unlabelled test transactions.
* Synthetic: 21 contextual fields generated with documented logic and seed 42. The
  generator never uses the sale price, which prevents target leakage.
* Cleaning: structural absence mapped to `None`, neighbourhood-median imputation for lot
  frontage, two implausible area-price records removed (0.14%), no duplicate records.
* The data are historic and observational; no causal interpretation of coefficients is
  warranted.

## 8. Limitations

1. **Small sample.** 1,458 transactions support a parsimonious model; complex
   interactions are estimated imprecisely.
2. **Synthetic context.** Amenity, renovation and macro variables are simulated with
   documented assumptions; their coefficients reflect the simulation, not measured
   real-world effects.
3. **Residual distribution.** Log residuals are left-skewed (skewness -1.14) and
   heteroscedastic (Breusch-Pagan p < 0.001), so Gaussian confidence intervals would be
   misleading; conformal intervals are used instead.
4. **Spatial coverage.** The model does not use precise coordinates for comparable-sales
   adjustment and cannot detect neighbourhood change within the sample period.
5. **No physical inspection.** Condition, quality and renovation are assessor-coded and
   may not reflect the current state of the property.
6. **Single market regime.** The model has not observed a high-interest-rate regime and
   should be recalibrated before use after a structural break.

## 9. Ethical and regulatory considerations

* Property valuation affects wealth and access to housing; the tool is positioned as
  decision support with human oversight, not as an automated decision.
* Geographic and price-band error differences are monitored and reported; systematic
  under-prediction of luxury properties is disclosed.
* No protected attributes are used. Neighbourhood is used as a location proxy and may
  correlate with demographic composition; the tool must not be used to steer or
  discriminate.
* Prediction intervals communicate uncertainty to non-specialist users.

## 10. Monitoring and maintenance

* Input drift: population stability index per feature, threshold 0.20, minimum three
  features.
* Prediction drift: PSI of the score distribution.
* Performance: rolling RMSE, MAE, MAPE and bias per equal-volume window.
* Calibration: empirical coverage of the 80% and 90% intervals.
* Retraining: triggered by sustained coverage erosion, sustained accuracy degradation or
  sustained directional bias across two consecutive windows. The current replay of the
  2009-2010 stream triggers the coverage rule and recommends retraining.
