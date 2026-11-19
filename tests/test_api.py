"""Integration tests for the FastAPI service and the inference contract."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from revaluation.api.main import app
from revaluation.api.service import ValuationService


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def profile() -> dict:
    return {
        "neighborhood": "CollgCr",
        "overall_qual": 7,
        "gr_liv_area": 1800.0,
        "year_built": 1995,
        "lot_area": 9000.0,
        "total_bsmt_sf": 900.0,
        "first_flr_sf": 1000.0,
        "second_flr_sf": 800.0,
        "full_bath": 2,
        "half_bath": 1,
        "garage_cars": 2,
        "garage_area": 500.0,
        "fireplaces": 1,
        "sale_year": 2010,
        "sale_month": 6,
    }


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_root_documents_endpoints(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "/predict" in response.json()["endpoints"]


def test_predict_returns_calibrated_interval(client: TestClient, profile: dict) -> None:
    response = client.post("/predict", json=profile)
    assert response.status_code == 200
    body = response.json()
    assert 20_000 < body["estimate"] < 800_000
    lower80, upper80 = body["interval_80"]
    lower90, upper90 = body["interval_90"]
    assert lower90 < lower80 < body["estimate"] < upper80 < upper90
    assert body["confidence"] in {"high", "moderate", "low"}
    assert 0 <= body["confidence_score"] <= 1


def test_predict_rejects_invalid_quality(client: TestClient, profile: dict) -> None:
    response = client.post("/predict", json={**profile, "overall_qual": 15})
    assert response.status_code == 422


def test_predict_rejects_unknown_field(client: TestClient, profile: dict) -> None:
    response = client.post("/predict", json={**profile, "unknown_field": 1})
    assert response.status_code == 422


def test_batch_endpoint(client: TestClient, profile: dict) -> None:
    response = client.post(
        "/predict/batch",
        json={"listings": [profile, {**profile, "overall_qual": 4}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["predictions"][0]["estimate"] > body["predictions"][1]["estimate"]


def test_explanations_are_informative(client: TestClient, profile: dict) -> None:
    response = client.post("/explain", json={**profile, "top_k": 4})
    assert response.status_code == 200
    drivers = response.json()["drivers"]
    assert len(drivers) == 4
    assert any(abs(driver["contribution_log"]) > 1e-6 for driver in drivers)


def test_monitoring_summary_is_available(client: TestClient) -> None:
    response = client.get("/monitoring/summary")
    assert response.status_code == 200
    assert "observed" in response.json()


def test_service_ignores_unknown_payload_fields() -> None:
    service = ValuationService()
    with pytest.raises(ValueError):
        service.assemble({"neighborhood": "NAmes", "not_a_field": 3})


def test_service_minimal_profile_scores() -> None:
    service = ValuationService()
    result = service.predict(
        {"neighborhood": "NoRidge", "overall_qual": 9, "gr_liv_area": 2400, "year_built": 2003}
    )
    assert result["estimate"] > 200_000
    assert result["assumptions"]["sale_year"] == 2010
    assert result["interval_90"][0] < result["estimate"] < result["interval_90"][1]
