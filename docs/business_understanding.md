# Business understanding and KPI framework

## 1. Business problem

A residential listing platform needs a defensible market-value estimate for a property
before a formal appraisal is commissioned. An inflated estimate deters buyers and
lengthens time on market; a depressed estimate transfers value from the seller and
damages the platform's credibility with listing agents. The platform therefore needs a
model that is accurate on average, transparent about its uncertainty, and monitored
after deployment.

The concrete question addressed by this project is:

> Given a property's structural, locational and market-context characteristics at the
> time of listing, what is its estimated market price, and how confident is that
> estimate?

## 2. Stakeholders and decisions

| Stakeholder | Decision supported | Requirement |
|---|---|---|
| Listing agent | Recommended asking price band | Point estimate plus a calibrated interval |
| Seller | Whether to list now or renovate first | Sensitivity to quality, age and renovation |
| Buyer | Whether an asking price is defensible | Interval coverage and bias monitoring |
| Platform operations | Whether to auto-publish the estimate | Confidence thresholds and escalation policy |
| Model risk owner | Whether the model remains fit for purpose | Drift and performance monitoring, retraining triggers |

The operational workflow is:

1. The agent enters the listing profile in the valuation tool.
2. The model returns a point estimate and an 80% and 90% interval.
3. Estimates classified as high confidence may be published automatically.
4. Medium and low confidence estimates are reviewed by a valuer before publication.
5. Monitoring reports run monthly; a sustained breach opens a retraining ticket.

## 3. Key performance indicators

### 3.1 Statistical accuracy

| Indicator | Definition | Target | Achieved (out-of-fold) |
|---|---|---|---|
| RMSE | Root mean squared error in USD | <= 25,000 | 20,019 |
| MAE | Mean absolute error in USD | <= 17,000 | 13,537 |
| MAPE | Mean absolute percentage error | <= 10% | 7.98% |
| R-squared | Share of price variance explained | >= 0.90 | 0.9365 |

### 3.2 Operational KPIs

| Indicator | Definition | Target | Achieved |
|---|---|---|---|
| Tolerance hit rate (absolute) | Share of estimates within +/- 15,000 USD | >= 65% | 69.7% |
| Tolerance hit rate (relative) | Share of estimates within +/- 8% of the sale price | >= 60% | 64.9% |
| Interval coverage (80%) | Realised sales inside the 80% interval | 0.75 - 0.85 | 0.801 |
| Interval coverage (90%) | Realised sales inside the 90% interval | 0.85 - 0.95 | 0.901 |
| Over-valuation rate | Share of estimates above the sale price | 40% - 60% | 48.0% |

The interval-coverage targets are two-sided on purpose: an interval that is too wide is
as operationally useless as one that is too narrow, even though it satisfies coverage.

### 3.3 Economic KPI

Errors are not symmetric. An over-valuation is weighted at 1.5 and an under-valuation
at 1.0, reflecting the assumption that a stale over-priced listing costs the platform
more in lost transactions and agent churn than a conservative estimate costs the seller.
The resulting loss is reported in USD:

```
loss = 1.5 * max(estimate - price, 0) + 1.0 * max(price - estimate, 0)
```

The out-of-fold mean asymmetric loss is 16,708 USD per listing with a median of
11,358 USD. These values are sensitive to the weights, which are documented as a
business assumption rather than estimated from data.

## 4. Error tolerance and escalation

The absolute tolerance of 15,000 USD is approximately the width of a normal
negotiation margin on the median Ames property of 163,000 USD, and is the threshold at
which an estimate is treated as "agent-ready" without manual review. The relative
tolerance of 8% matches the observed median absolute percentage error of the selected
model. Estimates whose 90% interval exceeds 30% of the point estimate are classified as
low confidence and routed to a valuer.

## 5. Data combination and assumptions

The base Kaggle data provide structural attributes and historical sale prices. Five
families of contextual data are simulated because the platform would observe them in
production but the competition data do not contain them:

1. neighbourhood amenity access (school, hospital, transit, park);
2. neighbourhood quality indices (school rating, walkability, crime);
3. renovation history (flag and scope);
4. marketing duration and listing date;
5. macro-financial conditions (mortgage rate, house price index, supply, unemployment).

The generation logic is documented field by field in the data dictionary. The key
modelling assumptions are:

* The neighbourhood tier used by the generator is an expert prior about sub-market
  desirability, not an estimate from the target variable, which avoids target leakage.
* Post-transaction variables such as sale condition, sale type and days on market are
  excluded from the model inputs even though they are available in the base data.
* The synthetic macro series approximates the 2006-2011 U.S. housing cycle and is not a
  live data feed.

## 6. Success criteria for the project

The project is considered successful when:

1. every stage of the pipeline is reproducible from the repository with a single command;
2. the selected model meets the accuracy and calibration targets above;
3. the API and the valuation tool return estimates and intervals that agree with the
   offline evaluation;
4. the monitoring report detects the simulated drift and recommends a retraining action;
5. the report and slides document the decisions, limitations and business implications.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Target leakage from sale-time variables | Explicit exclusion list enforced in code and tested |
| Overfitting on a small sample (1,458 rows) | Five-fold cross-validation, regularisation, model comparison |
| Distribution shift after deployment | PSI and KS monitoring, drift-triggered retraining |
| Geographic or segment bias | Error and coverage reported by neighbourhood segment and price band |
| Synthetic data realism | Documented generation logic, sanity checks against the observed data, sensitivity analysis in the report |
