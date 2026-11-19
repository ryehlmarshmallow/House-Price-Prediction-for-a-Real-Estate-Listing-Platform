# Residential valuation platform

End-to-end analytics product for a real-estate listing platform: synthetic context
generation, data quality control, exploratory analysis, feature engineering, model
development with calibrated uncertainty, a FastAPI service, a Streamlit valuation
console, and post-deployment drift monitoring with retraining triggers.

The project answers a single operational question: given a property's characteristics
and location at listing time, what is its estimated market price, and how confident is
that estimate?

## Team

| Role | Member |
|---|---|
| Data Engineer | `[Member 1 Name]` (`[Student ID 1]`) |
| Data Analyst | `[Member 2 Name]` (`[Student ID 2]`) |
| ML Modeling Engineer | `[Member 3 Name]` (`[Student ID 3]`) |
| ML Evaluation Engineer | `[Member 4 Name]` (`[Student ID 4]`) |
| MLOps Lead | `[Member 5 Name]` (`[Student ID 5]`) |

## Headline results

| Metric | Value |
|---|---|
| Selected model | Elastic net on the log price |
| Out-of-fold RMSE | 20,019 USD |
| Out-of-fold MAE | 13,537 USD |
| Out-of-fold MAPE | 7.98% |
| Out-of-fold R-squared | 0.9365 |
| Within +/- 15,000 USD | 69.7% |
| 80% interval coverage | 0.801 |
| 90% interval coverage | 0.901 |
| Temporal holdout 2010 RMSE | 22,269 USD |
| Monitoring decision (2009-2010 replay) | Retrain now (coverage erosion) |

The benchmark compared linear regression, ridge, lasso, elastic net, random forest,
gradient boosting, XGBoost and LightGBM under identical five-fold cross-validation.
The regularised linear model won on out-of-fold error; the boosted ensembles were
competitive but did not improve on it at this sample size. The full comparison is in
`artifacts/metrics/model_comparison.csv`.

## Repository layout

```
house-prices-advanced-regression-techniques/   Kaggle base data (immutable input)
src/revaluation/
  config.py                 paths, thresholds, business assumptions
  data/load.py              validation of the competition files
  data/synthetic.py         documented generation of contextual fields
  data/clean.py             cleaning rules, outlier and duplicate handling
  data/features.py          engineering, model matrix, VIF
  eda/analysis.py           exploratory figures and statistics
  models/preprocess.py      shared preprocessing pipelines
  models/train.py           cross-validation, tuning, artefact creation
  models/evaluate.py        residual, segment and calibration diagnostics
  models/explain.py         SHAP attribution
  models/metrics.py         regression and business KPIs
  monitoring/drift.py       PSI, KS and total variation
  monitoring/stream.py      temporal replay of the deployment stream
  monitoring/performance.py alerting and retraining policy
  monitoring/dashboard.py   HTML monitoring report
  api/main.py               FastAPI endpoints
  api/service.py            inference contract and conformal intervals
app/streamlit_app.py        valuation console and monitoring views
scripts/                    numbered pipeline steps
docs/                       data dictionary, business case, model card, guides
report/                     LaTeX report and compiled PDF
slides/                     PowerPoint deck and build script
deploy/                     Dockerfiles, Compose, Hugging Face template
tests/                      unit and integration tests
```

## Quick start

Prerequisites: Python managed by `uv` (no global installation is required).

```bash
uv sync
make pipeline          # data, cleaning, features, training, evaluation, monitoring, dictionary
```

The pipeline is deterministic (seed 42) and writes every artefact under `artifacts/`.
Individually:

```bash
uv run python scripts/01_generate_context.py
uv run python scripts/02_clean_data.py
uv run python scripts/03_build_features.py
uv run python scripts/04_train_models.py
uv run python scripts/05_run_evaluation.py
uv run python scripts/06_run_monitoring.py
uv run python scripts/07_export_data_dictionary.py
```

## Applications

API (FastAPI):

```bash
uv run uvicorn revaluation.api.main:app --host 127.0.0.1 --port 8000
# interactive documentation at http://127.0.0.1:8000/docs
```

Console (Streamlit):

```bash
uv run streamlit run app/streamlit_app.py
# http://127.0.0.1:8501
```

The console calls the API at `VALUATION_API_URL` (default `http://127.0.0.1:8000`)
and falls back to an in-process copy of the pipeline when the API is not running.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"neighborhood":"CollgCr","overall_qual":7,"gr_liv_area":1800,"year_built":1995}'
```

The response contains the point estimate, 80% and 90% conformal intervals, a
confidence classification and, when requested with `?explain=true`, the local SHAP
contributions.

## Monitoring

`scripts/06_run_monitoring.py` refits the selected estimator on sales up to December
2008 and replays every 2009-2010 sale through four equal-volume windows. It computes
population stability indices, Kolmogorov-Smirnov statistics, rolling RMSE, MAE, MAPE,
bias and interval coverage, applies nine alerting rules, and writes:

* `artifacts/monitoring/monitoring_report.html` - static dashboard;
* `artifacts/monitoring/monitoring_summary.json` - machine-readable summary;
* `artifacts/monitoring/alerts.json` - triggered alerts and retraining decision;
* `artifacts/monitoring/drift_by_window.csv` and `performance_by_window.csv`.

The observed replay shows macro-financial drift outside the reference support, a
modest accuracy reduction (worst window 1.12 times the reference RMSE) and a material
erosion of interval coverage (79.7% against a nominal 90%), which triggers the
retraining rule. The adverse stress scenario additionally triggers the bias rule.

## Deliverables

| Requirement | Artefact |
|---|---|
| Source code | This repository |
| Data dictionary | `docs/data_dictionary.md`, `docs/data_dictionary.xlsx` |
| Deployed API and valuation tool | `src/revaluation/api`, `app/streamlit_app.py`, deployment configuration in `deploy/` and `render.yaml` |
| Monitoring dashboard or report | `artifacts/monitoring/monitoring_report.html` and the Monitoring page of the console |
| Final report | `report/main.pdf` (LaTeX source in `report/`) |
| Presentation | `slides/residential_valuation_deck.pptx` |

## Deployment status

The complete system runs and is verified locally. The repository contains Dockerfiles,
a Compose stack, a Render blueprint and a Hugging Face Spaces template. No cloud
credentials were available in the development environment, so no public URL has been
provisioned and no live link is claimed. `docs/deployment.md` lists the exact steps and
the variables required; deployment needs only the account owner to authenticate.

## Data provenance

* Kaggle, *House Prices - Advanced Regression Techniques*, Ames, Iowa. The competition
  files are used locally and are not redistributed beyond the course context.
* Synthetic contextual fields are generated by `src/revaluation/data/synthetic.py`
  with seed 42. The generator never uses the target variable.
* Macro anchors approximate the 2006-2011 U.S. housing cycle (mortgage rate, house
  price index, months of supply, unemployment) and are documented in the data dictionary.

## Reproducibility and tests

```bash
uv run pytest -q
uv run ruff check src app scripts tests
```

The LaTeX report is built with:

```bash
cd report && latexmk -pdf -interaction=nonstopmode main.tex
```

## Limitations

The sample is small (1,458 labelled sales), the contextual variables are simulated,
residuals are heteroscedastic, and the model has not observed a high-interest-rate
regime. Estimates are decision support, not appraisals. Details are in
`docs/model_card.md` and in the final report.
