# Architecture

## 1. Runtime view

```
                    +-----------------------------+
                    |  Listing profile (agent)    |
                    +--------------+--------------+
                                   |
                                   v
+---------------------+   HTTP   +---------------------------+
|  Streamlit console  |--------->|  FastAPI valuation service |
|  app/streamlit_app  |          |  /predict, /explain,       |
|                     |<---------|  /model/info, /monitoring  |
+----------+----------+  JSON    +-------------+-------------+
           |                                    |
           | in-process fallback                | loads
           v                                    v
+---------------------+          +-----------------------------+
|  final_model.joblib |<---------|  artifacts/models           |
+---------------------+          +-----------------------------+
```

The console calls the API through `VALUATION_API_URL`. If the API is unreachable it
loads `final_model.joblib` directly and produces an identical response shape, which
keeps a single-service deployment usable.

## 2. Batch pipeline

```
Kaggle files ──> load/validate ──> merge synthetic context ──> clean ──> engineer
                                                                          |
                                                                    train/evaluate
                                                                          |
                                       +----------------------------------+------------------+
                                       v                                                     v
                             artifacts/models/final_model.joblib          artifacts/figures, metrics
                                                                                                |
                                                                                          monitoring replay
                                                                                                |
                                                                                    HTML dashboard + JSON
```

Every stage is a numbered script under `scripts/`. Artefacts are plain files so that
the API, the console, the report and the monitoring dashboard all read the same
objects.

## 3. Data contract

| Layer | Object | Owner |
|---|---|---|
| Raw | competition CSV files | immutable input |
| Synthetic context | `data/synthetic/property_context.parquet` | `data/synthetic.py` |
| Cleaned listings | `data/processed/listings_clean.parquet` | `data/clean.py` |
| Analytical table | `data/processed/modeling_table.parquet` | `data/features.py` |
| Model artefact | `artifacts/models/final_model.joblib` | `models/train.py` |
| Serving defaults | `artifacts/models/serving_defaults.json` | `models/train.py` |
| Monitoring summary | `artifacts/monitoring/monitoring_summary.json` | `monitoring/stream.py` |

## 4. Inference contract

1. The service starts from the training medians and modes in `serving_defaults.json`.
2. Caller-provided fields override those defaults.
3. Amenity and macro fields that are still missing are derived from the documented
   generating distribution for the selected neighbourhood and sale month.
4. `engineer_features` computes the derived variables row-wise, `model_matrix` selects
   the 102 model inputs, and the persisted pipeline returns the log price.
5. Conformal quantiles transform the log prediction into the 80% and 90% intervals.

The same functions are used during training, so the transformation is identical in
batch and in serving.
