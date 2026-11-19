---
title: Residential Valuation Console
emoji: false
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Residential valuation console

Streamlit interface for the Ames, Iowa residential valuation model. The container
runs the application with an in-process copy of the trained pipeline; the FastAPI
service is optional and can be configured through the `VALUATION_API_URL` variable.

## Space configuration

1. Create a Docker Space and select this repository as the source.
2. Copy this file to the root of the Space repository as `README.md`.
3. Set the Space variable `PORT` to `7860`.
4. If a hosted API is available, add a secret `VALUATION_API_URL` with its public URL.

The container entry point is the command in `deploy/Dockerfile.app`:

```bash
streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 7860 --server.headless true
```

## Scope

Estimates are decision support for listing agents and buyers. They are not formal
appraisals and must not be used for credit or legal purposes.
