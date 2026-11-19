# Deployment guide

## 1. Architecture

The system has three runtime components:

| Component | Entry point | Port | Purpose |
|---|---|---|---|
| Valuation API | `revaluation.api.main:app` (FastAPI) | 8000 | Point estimates, intervals, local explanations |
| Valuation console | `app/streamlit_app.py` (Streamlit) | 8501 | Agent-facing tool, performance and monitoring views |
| Monitoring artefacts | `scripts/06_run_monitoring.py` | - | Drift report, alerts and static HTML dashboard |

The Streamlit application calls the API through `VALUATION_API_URL`. If the API is not
reachable it falls back to an in-process copy of the same pipeline, so a single-service
deployment remains functional. The API is stateless apart from the model artefact.

## 2. Local execution

```bash
uv sync
uv run python scripts/01_generate_context.py
uv run python scripts/02_clean_data.py
uv run python scripts/03_build_features.py
uv run python scripts/04_train_models.py
uv run python scripts/05_run_evaluation.py
uv run python scripts/06_run_monitoring.py
```

Start the services in two terminals:

```bash
uv run uvicorn revaluation.api.main:app --host 127.0.0.1 --port 8000
uv run streamlit run app/streamlit_app.py
```

Verify the API:

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"neighborhood":"CollgCr","overall_qual":7,"gr_liv_area":1800,"year_built":1995}'
```

## 3. Container images

Two Dockerfiles are provided:

* `deploy/Dockerfile.api` - FastAPI service, listens on `$PORT` (default 8000).
* `deploy/Dockerfile.app` - Streamlit console, listens on `$PORT` (default 8501).

Both images copy the trained artefacts and the processed data, so no training step is
required at start-up. Build and run:

```bash
docker build -f deploy/Dockerfile.api -t ames-valuation-api .
docker run --rm -p 8000:8000 ames-valuation-api

docker build -f deploy/Dockerfile.app -t ames-valuation-app .
docker run --rm -p 8501:8501 -e VALUATION_API_URL=http://127.0.0.1:8000 ames-valuation-app
```

Or start both with Compose:

```bash
docker compose -f deploy/docker-compose.yml up --build
```

## 4. Deployment option A: Render (free tier)

The repository contains a Render blueprint (`render.yaml`) with two web services.

Manual procedure:

1. Push the repository to GitHub.
2. In the Render dashboard, choose **New > Blueprint** and connect the repository.
3. Render creates `ames-valuation-api` and `ames-valuation-app` from the blueprint.
4. For `ames-valuation-app`, set `VALUATION_API_URL` to the public URL of the API
   service, for example `https://ames-valuation-api.onrender.com`.
5. Confirm the API health check passes at `/health`.
6. Open the console URL (for example `https://ames-valuation-app.onrender.com`).

Free-tier caveats:

* Services sleep after inactivity; the first request after a pause takes up to 60 seconds.
* The free instance has 512 MB of RAM; the model and dependencies fit comfortably.
* Render assigns the public URL; no domain configuration is required.

## 5. Deployment option B: Hugging Face Spaces (Docker SDK)

1. Create a new Space with the **Docker** SDK and a blank template.
2. Copy `deploy/Dockerfile.app`, `deploy/start_hf.sh` and the repository contents to the
   Space, or connect the Space to the GitHub repository and point `app_port` to 7860 in
   the Space metadata.
3. The Space metadata must contain:

```yaml
---
title: Residential Valuation Console
sdk: docker
app_port: 7860
---
```

4. The console runs with the in-process pipeline; no API secret is required. To use the
   hosted API instead, add `VALUATION_API_URL` as a Space secret.
5. The API can be deployed as a second Space using `deploy/Dockerfile.api` and the same
   metadata with `app_port: 8000`.

## 6. Configuration

| Variable | Default | Purpose |
|---|---|---|
| `VALUATION_API_URL` | `http://127.0.0.1:8000` | API endpoint used by the console |
| `PORT` | 8000 / 8501 | Listening port inside the container |
| `STREAMLIT_SERVER_PORT` | 8501 | Streamlit override when needed |

No credentials are needed to run the system. Cloud deployment requires the account
credentials of the owner of the Render or Hugging Face account; those are intentionally
not stored in the repository.

## 7. Operational checks after deployment

1. `GET /health` returns `{"status": "ok", "model_loaded": true}`.
2. `GET /model/info` returns the artefact metadata and validation metrics.
3. The console returns an estimate and an interval for a known listing.
4. `GET /monitoring/summary` returns the latest monitoring JSON once
   `scripts/06_run_monitoring.py` has been executed.
5. The monitoring HTML report is archived with each release.

## 8. Current status

The complete system has been verified locally: the API, the console and the monitoring
report run end to end against the trained artefacts. No cloud account credentials were
available in the working environment, so no public URL has been provisioned and no live
link is claimed. The configuration above is complete and requires only the account
owner to authenticate and click deploy.
