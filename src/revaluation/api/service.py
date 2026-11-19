"""Inference service: assembled listing profile to price estimate and interval.

The service loads the persisted scikit-learn pipeline, reconstructs a complete
raw listing record from the caller's partial profile, applies the *same*
cleaning, feature engineering and preprocessing used during training, and
returns a point estimate with split-conformal prediction intervals.

The reconstruction step is deliberate: the model contract is the clean,
engineered variable set, and the service owns the mapping from a user-facing
listing profile to that contract. This prevents train/serve skew and keeps the
API tolerant to partial forms, which is essential for a valuation tool used by
agents who rarely have every field at hand.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from revaluation.config import DEFAULT_SALE_MONTH, DEFAULT_SALE_YEAR, MODELS_DIR
from revaluation.data.features import engineer_features, model_matrix
from revaluation.data.synthetic import expected_context

logger = logging.getLogger(__name__)

CATEGORICAL_INPUTS = (
    "neighborhood",
    "ms_zoning",
    "ms_sub_class",
    "street",
    "alley",
    "lot_shape",
    "land_contour",
    "lot_config",
    "land_slope",
    "utilities",
    "condition1",
    "condition2",
    "bldg_type",
    "house_style",
    "roof_style",
    "roof_matl",
    "exterior1st",
    "exterior2nd",
    "mas_vnr_type",
    "exter_qual",
    "exter_cond",
    "foundation",
    "bsmt_qual",
    "bsmt_cond",
    "bsmt_exposure",
    "bsmt_fin_type1",
    "bsmt_fin_type2",
    "heating",
    "heating_qc",
    "central_air",
    "electrical",
    "kitchen_qual",
    "functional",
    "fireplace_qu",
    "garage_type",
    "garage_finish",
    "garage_qual",
    "garage_cond",
    "paved_drive",
    "pool_qc",
    "fence",
    "misc_feature",
    "renovation_scope",
)

NUMERIC_INPUTS = (
    "lot_frontage",
    "lot_area",
    "overall_qual",
    "overall_cond",
    "year_built",
    "year_remod_add",
    "mas_vnr_area",
    "total_bsmt_sf",
    "first_flr_sf",
    "second_flr_sf",
    "low_qual_fin_sf",
    "gr_liv_area",
    "bsmt_full_bath",
    "bsmt_half_bath",
    "full_bath",
    "half_bath",
    "bedroom_abv_gr",
    "kitchen_abv_gr",
    "tot_rms_abv_grd",
    "fireplaces",
    "garage_cars",
    "garage_area",
    "wood_deck_sf",
    "open_porch_sf",
    "enclosed_porch",
    "three_ssn_porch",
    "screen_porch",
    "pool_area",
    "misc_val",
    "dist_school_km",
    "dist_hospital_km",
    "dist_transit_km",
    "dist_park_km",
    "school_rating",
    "walk_score",
    "crime_index",
    "amenity_density_1_5km",
    "park_area_ha",
    "amenity_score",
    "mortgage_rate_pct",
    "hpi_index",
    "hpi_yoy_pct",
    "months_supply",
    "local_unemployment_pct",
    "neigh_lat",
    "neigh_lon",
)

ALLOWED_INPUTS = set(CATEGORICAL_INPUTS) | set(NUMERIC_INPUTS) | {
    "sale_year",
    "sale_month",
    "yr_sold",
    "mo_sold",
}

CONTEXT_DEFAULTS = (
    "dist_school_km",
    "dist_hospital_km",
    "dist_transit_km",
    "dist_park_km",
    "school_rating",
    "walk_score",
    "crime_index",
    "amenity_density_1_5km",
    "park_area_ha",
    "amenity_score",
    "neigh_lat",
    "neigh_lon",
)


class ValuationService:
    """Encapsulates the trained model and the inference contract."""

    def __init__(
        self,
        model_path: Path | None = None,
        defaults_path: Path | None = None,
        metadata_path: Path | None = None,
    ) -> None:
        model_path = model_path or MODELS_DIR / "final_model.joblib"
        defaults_path = defaults_path or MODELS_DIR / "serving_defaults.json"
        metadata_path = metadata_path or MODELS_DIR / "model_metadata.json"
        if not model_path.exists():
            raise FileNotFoundError(
                "Final model artefact not found. Run `uv run python scripts/04_train_models.py`."
            )
        self.artifact = joblib.load(model_path)
        self.pipeline = self.artifact["pipeline"]
        self.background = self.artifact.get("background")
        self.model_name = self.artifact.get("model_label", self.artifact["model_name"])
        self.conformal: dict[str, float] = self.artifact["conformal"]
        with open(defaults_path, encoding="utf-8") as handle:
            self.defaults: dict = json.load(handle)
        self.metadata: dict = {}
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as handle:
                self.metadata = json.load(handle)
        self.model_version = str(self.artifact.get("trained_at", "unknown"))
        self._explainer = None
        logger.info("Loaded %s from %s", self.model_name, model_path)

    # ------------------------------------------------------------------
    # Input assembly
    # ------------------------------------------------------------------
    def assemble(self, payload: dict) -> tuple[pd.DataFrame, dict]:
        """Build a full raw listing record from a partial profile."""
        unknown = set(payload) - ALLOWED_INPUTS
        if unknown:
            raise ValueError(f"Unsupported fields: {sorted(unknown)}")

        profile = {key: value for key, value in self.defaults.items()}
        profile.update({key: value for key, value in payload.items() if value is not None})

        sale_year = int(
            payload.get("sale_year", payload.get("yr_sold", DEFAULT_SALE_YEAR))
        )
        sale_month = int(
            payload.get("sale_month", payload.get("mo_sold", DEFAULT_SALE_MONTH))
        )
        sale_month = int(np.clip(sale_month, 1, 12))
        profile["yr_sold"] = sale_year
        profile["mo_sold"] = sale_month
        profile["sale_year"] = sale_year
        profile["sale_month"] = sale_month

        context = expected_context(
            str(profile.get("neighborhood", "NAmes")), sale_year, sale_month
        )
        assumed = {}
        for field, value in context.items():
            if payload.get(field) is None:
                profile[field] = value
                if field in CONTEXT_DEFAULTS:
                    assumed[field] = value
        for field, value in payload.items():
            if value is not None:
                profile[field] = value

        if profile.get("id") is None:
            profile["id"] = 0
        frame = pd.DataFrame([profile])
        engineered = engineer_features(frame)
        X, _ = model_matrix(engineered)
        assumptions = {
            "sale_year": sale_year,
            "sale_month": sale_month,
            "neighbourhood": str(profile.get("neighborhood")),
            "derived_context_fields": sorted(assumed),
        }
        return X, assumptions

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, payload: dict, explain: bool = False) -> dict:
        X, assumptions = self.assemble(payload)
        log_prediction = float(self.pipeline.predict(X)[0])
        estimate = float(np.expm1(log_prediction))

        intervals = {}
        for level, quantile in self.conformal.items():
            lower = float(np.expm1(log_prediction - quantile))
            upper = float(np.expm1(log_prediction + quantile))
            intervals[level] = [round(lower, 2), round(upper, 2)]

        q90 = self.conformal.get("0.90", self.conformal.get("0.80", 0.2))
        relative_half_width = float(np.expm1(log_prediction + q90) - estimate) / estimate
        if relative_half_width <= 0.18:
            confidence = "high"
        elif relative_half_width <= 0.30:
            confidence = "moderate"
        else:
            confidence = "low"
        confidence_score = float(np.clip(1.0 - relative_half_width / 0.55, 0.0, 1.0))

        response = {
            "estimate": round(estimate, 2),
            "currency": "USD",
            "interval_80": intervals.get("0.80"),
            "interval_90": intervals.get("0.90"),
            "confidence": confidence,
            "confidence_score": round(confidence_score, 4),
            "relative_interval_width_90": round(
                (np.expm1(log_prediction + q90) - np.expm1(log_prediction - q90)) / estimate, 4
            ),
            "model": self.model_name,
            "model_version": self.model_version,
            "assumptions": assumptions,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        if explain:
            response["drivers"] = self.explain(payload)
        return response

    def predict_batch(self, payloads: list[dict]) -> list[dict]:
        return [self.predict(payload) for payload in payloads]

    def explain(self, payload: dict, top_k: int = 5) -> list[dict]:
        """Local SHAP contributions for the supplied profile."""
        from revaluation.models.explain import local_explanation

        X, _ = self.assemble(payload)
        try:
            return local_explanation(
                self.pipeline, X, top_k=top_k, background=self.background
            )
        except Exception as error:  # noqa: BLE001 - explanation is optional  # pragma: no cover
            logger.warning("Local explanation unavailable: %s", error)
            return []

    def model_card(self) -> dict:
        return {
            "model": self.model_name,
            "version": self.model_version,
            "training_metrics": self.metadata.get("metrics", {}),
            "temporal_holdout_2010": self.metadata.get("temporal_holdout_2010", {}),
            "conformal_quantiles_log": self.conformal,
            "n_features": len(self.artifact.get("feature_columns", [])),
            "target": "SalePrice (USD)",
            "interval_method": "split conformal on out-of-fold log residuals",
        }


_SERVICE: ValuationService | None = None


def get_service() -> ValuationService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = ValuationService()
    return _SERVICE
