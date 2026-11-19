# Monitoring methodology and results

## 1. Why a temporal replay

The competition test set carries no sale prices, so performance monitoring cannot use it.
The monitoring study therefore replays a genuine deployment:

* the **reference model** is the selected estimator refitted on the 945 sales completed
  before 1 January 2009, the vintage that would have been in production at that time;
* the **stream** is every sale between January 2009 and July 2010, split into four
  equal-volume chronological windows of 128-129 transactions;
* realised prices are treated as labels that arrive after each window closes, mirroring
  the delay between listing activity and recorded deeds.

This design keeps the monitoring period out of the reference model's training data, so
accuracy and calibration are measured honestly.

## 2. What is measured

| Signal | Statistic | Purpose |
|---|---|---|
| Input drift (numeric) | Population stability index on reference quantile bins; Kolmogorov-Smirnov statistic and p-value | Detect covariate shift |
| Input drift (categorical) | Population stability index on level proportions; total variation distance | Detect mix changes in neighbourhoods and property types |
| Prediction drift | PSI of the score distribution | Detect output shift that users would notice |
| Accuracy | RMSE, MAE, MAPE, R-squared per window | Track degradation against the reference vintage |
| Bias | Mean signed percentage error | Detect systematic over- or under-valuation |
| Calibration | Empirical coverage of the 80% and 90% conformal intervals | Detect uncertainty mis-specification |
| Business KPIs | Share within +/- 15,000 USD, asymmetric cost | Translate errors into operational terms |

PSI conventions: below 0.10 stable, 0.10-0.20 moderate, at or above 0.20 significant.

## 3. Alerting and retraining policy

| Rule | Condition | Severity | Action |
|---|---|---|---|
| R1 input drift | At least 3 features with PSI >= 0.20 in a window | Warning | Review the input pipeline and market context |
| R2 prediction drift | PSI of predictions >= 0.20 | Warning | Confirm the shift is explained by known market moves |
| R3 accuracy breach | Window RMSE > 1.2 x reference RMSE | Critical | Inspect worst errors, prepare a retraining candidate |
| R4 coverage | 90% empirical coverage < 0.85 | Warning | Recalibrate the conformal quantiles |
| R5 bias | Absolute mean signed error > 5% | Warning | Verify the macro feed and sale dates |
| R6 sustained accuracy | R3 in two consecutive windows | Critical | Retrain |
| R7 sustained coverage erosion | R4 in two consecutive windows | Critical | Recalibrate and retrain if erosion persists |
| R8 sustained bias | R5 in two consecutive windows | Critical | Retrain with recent data |
| R9 sustained input drift | R1 in two consecutive windows | Warning | Schedule retraining for the next release |

Any critical alert opens a retraining ticket. Warnings alone trigger an investigation task
and a scheduled retraining. The policy is implemented in
`src/revaluation/monitoring/performance.py` and evaluated on every run.

## 4. Results of the observed replay

Reference out-of-fold RMSE: 20,888 USD (945 sales, 2006-2008).

| Window | Sales | RMSE | MAE | MAPE | Bias | 90% coverage | RMSE ratio |
|---|---|---|---|---|---|---|---|
| W1 2009-01 to 2009-06 | 129 | 22,306 | 14,813 | 9.41% | -0.16% | 89.1% | 1.068 |
| W2 2009-06 to 2009-09 | 128 | 22,306 | 16,143 | 9.14% | -1.57% | 87.5% | 1.068 |
| W3 2009-09 to 2010-04 | 128 | 23,395 | 15,977 | 9.63% | -2.52% | 83.6% | 1.120 |
| W4 2010-04 to 2010-07 | 128 | 21,839 | 16,048 | 9.34% | -4.99% | 79.7% | 1.046 |

Interpretation:

1. **Drift is real but concentrated in the macro block.** Mortgage rates and the house
   price index move outside the reference support: the rate falls from roughly 6.5% to
   5.0% and the index from 104 to 96. Five to seven features breach the PSI threshold in
   every window, all of them macro-financial or amenity variables.
2. **Accuracy degrades only modestly.** The worst window is 1.12 times the reference
   RMSE, below the 1.2 breach threshold, so the accuracy rule does not fire.
3. **Calibration erodes materially.** The nominal 90% interval covers 79.7% of sales in
   the final window, below the 0.85 floor in two consecutive windows. The policy therefore
   raises the critical rule R7 and recommends recalibration and retraining.
4. **Directional bias emerges.** The model under-predicts by 5.0% in the final window, an
   early warning that the reference vintage's price level is stale.

Monitoring decision: **retrain now**, first recalibrating the conformal quantiles and then
refitting on the most recent 24 months of labelled data.

## 5. Adverse stress scenario

The same reference model was scored on a counterfactual stream in which the mortgage rate
rises by 1.5 percentage points, inventory rises by one month of supply, and neighbourhood
amenity distributions improve by 60%, while recorded prices are held at their observed
values. This scenario is not a forecast; it verifies that the alerting machinery reacts to
a sharper context shift. Under stress, 10-12 features drift in the first two windows, the
RMSE ratio rises to 1.152 in W3, coverage falls to 81.2% by W4 and the directional bias
reaches -5.2%, raising the additional bias rule R5.

## 6. Operational runbook

1. Run `uv run python scripts/06_run_monitoring.py` after every model release and at the
   start of each month.
2. Review `artifacts/monitoring/monitoring_report.html` and the Streamlit Monitoring page.
3. If a warning is raised, confirm that the drifting fields originate from the expected
   market move and not from a data-pipeline fault.
4. If a critical rule fires, open a retraining ticket with the window, rule and evidence
   attached.
5. Retraining procedure: refresh the labelled data, rerun the training and evaluation
   scripts, compare the new out-of-fold metrics with the incumbent, and only promote the
   model if RMSE and coverage both improve.
6. Archive the monitoring report with the model version that produced it.

## 7. Reproducibility

The simulation is deterministic: reference and stream are defined by sale dates, the
estimator is cloned from the released artefact, and random draws use seed 42. Re-running
the monitoring script on the same artefacts reproduces every number in this document.

## 8. Relationship to Evidently

The project evaluated the Evidently library, which is the recommended monitoring tool in
the announcement. The released Evidently version depends on internal pandas APIs removed
in pandas 3.1, and would have required pinning the entire project to an older pandas
release. The monitoring component was therefore implemented directly with the same
statistical primitives that Evidently uses for tabular drift (PSI, KS, chi-square) and for
regression performance, and rendered as a static HTML dashboard plus a Streamlit view.
This keeps the dependency set consistent and the thresholds fully auditable.
