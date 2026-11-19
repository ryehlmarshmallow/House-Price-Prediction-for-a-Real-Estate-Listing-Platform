"""FastAPI application exposing the valuation model (Module 6).

Endpoints
---------
GET  /health              liveness and model availability
GET  /model/info          model card and validation metrics
POST /predict             point estimate plus 80% and 90% intervals
POST /predict/batch       scored list of listing profiles
POST /explain             local SHAP contributions for one profile
GET  /monitoring/summary  latest drift and performance monitoring summary
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from revaluation.api.service import ValuationService
from revaluation.config import MONITORING_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class ListingProfile(BaseModel):
    """A partial listing profile. Missing fields fall back to documented defaults."""

    model_config = ConfigDict(extra="forbid")

    neighborhood: str = Field(..., description="Ames neighbourhood name")
    overall_qual: int = Field(..., ge=1, le=10, description="Overall material and finish quality")
    gr_liv_area: float = Field(..., gt=100, le=12_000, description="Above-grade living area (sqft)")
    year_built: int = Field(..., ge=1870, le=2026, description="Year of construction")
    year_remod_add: int | None = Field(default=None, ge=1870, le=2026)
    overall_cond: int | None = Field(default=None, ge=1, le=10)
    lot_area: float | None = Field(default=None, gt=0, le=500_000)
    lot_frontage: float | None = Field(default=None, ge=0, le=500)
    total_bsmt_sf: float | None = Field(default=None, ge=0, le=10_000)
    first_flr_sf: float | None = Field(default=None, ge=0, le=10_000)
    second_flr_sf: float | None = Field(default=None, ge=0, le=10_000)
    full_bath: int | None = Field(default=None, ge=0, le=6)
    half_bath: int | None = Field(default=None, ge=0, le=4)
    bedroom_abv_gr: int | None = Field(default=None, ge=0, le=12)
    tot_rms_abv_grd: int | None = Field(default=None, ge=1, le=20)
    fireplaces: int | None = Field(default=None, ge=0, le=5)
    garage_cars: int | None = Field(default=None, ge=0, le=5)
    garage_area: float | None = Field(default=None, ge=0, le=3000)
    pool_area: float | None = Field(default=None, ge=0, le=1000)
    wood_deck_sf: float | None = Field(default=None, ge=0, le=2000)
    open_porch_sf: float | None = Field(default=None, ge=0, le=1000)
    ms_zoning: str | None = None
    bldg_type: str | None = None
    house_style: str | None = None
    foundation: str | None = None
    exter_qual: str | None = None
    kitchen_qual: str | None = None
    bsmt_qual: str | None = None
    central_air: str | None = None
    garage_type: str | None = None
    heating_qc: str | None = None
    paved_drive: str | None = None
    sale_year: int | None = Field(default=None, ge=2006, le=2012)
    sale_month: int | None = Field(default=None, ge=1, le=12)
    dist_school_km: float | None = Field(default=None, ge=0, le=20)
    dist_transit_km: float | None = Field(default=None, ge=0, le=30)
    school_rating: int | None = Field(default=None, ge=1, le=10)
    walk_score: int | None = Field(default=None, ge=0, le=100)
    crime_index: int | None = Field(default=None, ge=0, le=100)
    amenity_score: float | None = Field(default=None, ge=0, le=100)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    listings: list[ListingProfile] = Field(..., min_length=1, max_length=200)


class ExplanationRequest(ListingProfile):
    top_k: int = Field(default=5, ge=1, le=10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.service = ValuationService()
        app.state.ready = True
        logger.info("Valuation service ready")
    except Exception as error:  # noqa: BLE001 - report readiness instead of crashing  # pragma: no cover
        app.state.service = None
        app.state.ready = False
        logger.error("Valuation service unavailable: %s", error)
    yield


app = FastAPI(
    title="Residential Valuation API",
    version="1.0.0",
    description=(
        "Point estimates and calibrated prediction intervals for residential "
        "properties in Ames, Iowa. The model is trained on structural, locational "
        "and market-context features observable at listing time."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _service() -> ValuationService:
    service = getattr(app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Model is not available.")
    return service


@app.get("/")
def root() -> dict:
    return {
        "service": "Residential Valuation API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": [
            "/health",
            "/model/info",
            "/predict",
            "/predict/batch",
            "/explain",
            "/monitoring/summary",
        ],
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if getattr(app.state, "ready", False) else "degraded",
        "model_loaded": getattr(app.state, "ready", False),
    }


@app.get("/model/info")
def model_info() -> dict:
    return _service().model_card()


@app.post("/predict")
def predict(
    profile: ListingProfile,
    explain: Annotated[bool, Query(description="Include local feature contributions")] = False,
) -> dict:
    payload = profile.model_dump(exclude_none=True)
    return _service().predict(payload, explain=explain)


@app.post("/predict/batch")
def predict_batch(request: BatchRequest) -> dict:
    payloads = [listing.model_dump(exclude_none=True) for listing in request.listings]
    predictions = _service().predict_batch(payloads)
    return {"count": len(predictions), "predictions": predictions}


@app.post("/explain")
def explain(request: ExplanationRequest) -> dict:
    params = request.model_dump(exclude_none=True)
    top_k = int(params.pop("top_k", 5))
    service = _service()
    drivers = service.explain(params, top_k=top_k)
    return {"drivers": drivers}


@app.get("/monitoring/summary")
def monitoring_summary() -> dict:
    path = MONITORING_DIR / "monitoring_summary.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Monitoring summary not generated yet. Run scripts/06_run_monitoring.py.",
        )
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
