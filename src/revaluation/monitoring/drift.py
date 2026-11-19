"""Distribution drift diagnostics (Module 7).

Three complementary statistics are reported for every input feature:

* **Population Stability Index (PSI)** on reference quantile bins. Values below
  0.10 indicate stability, 0.10-0.20 a moderate shift, and above 0.20 a
  material shift that warrants investigation.
* **Kolmogorov-Smirnov** statistic and p-value for numeric features, which
  tests whether the two samples could have been drawn from the same
  distribution.
* **Total variation distance** for categorical features, the natural analogue
  of PSI for unordered levels.

The thresholds are configuration values, not arbitrary constants, so that the
retraining policy can be audited and changed in one place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from revaluation.config import DRIFT_PSI_THRESHOLD

PSI_MODERATE = 0.10
PSI_SIGNIFICANT = DRIFT_PSI_THRESHOLD
MIN_BIN_COUNT = 10


def _safe_psi(reference_pct: np.ndarray, current_pct: np.ndarray) -> float:
    reference_pct = np.clip(reference_pct, 1e-4, None)
    current_pct = np.clip(current_pct, 1e-4, None)
    return float(np.sum((reference_pct - current_pct) * np.log(reference_pct / current_pct)))


def numeric_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI using reference quantile breakpoints."""
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if reference.empty or current.empty:
        return 0.0
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)
    reference_pct = reference_counts / max(reference_counts.sum(), 1)
    current_pct = current_counts / max(current_counts.sum(), 1)
    return _safe_psi(reference_pct, current_pct)


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    reference = reference.astype("object").fillna("__missing__").astype(str)
    current = current.astype("object").fillna("__missing__").astype(str)
    reference_pct = reference.value_counts(normalize=True)
    current_pct = current.value_counts(normalize=True)
    levels = reference_pct.index.union(current_pct.index)
    return _safe_psi(
        reference_pct.reindex(levels, fill_value=0.0).to_numpy(),
        current_pct.reindex(levels, fill_value=0.0).to_numpy(),
    )


def total_variation(reference: pd.Series, current: pd.Series) -> float:
    reference_pct = reference.astype("object").fillna("__missing__").value_counts(
        normalize=True
    )
    current_pct = current.astype("object").fillna("__missing__").value_counts(
        normalize=True
    )
    levels = reference_pct.index.union(current_pct.index)
    return float(
        0.5
        * np.abs(
            reference_pct.reindex(levels, fill_value=0.0).to_numpy()
            - current_pct.reindex(levels, fill_value=0.0).to_numpy()
        ).sum()
    )


def feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Per-feature drift statistics for one monitoring window."""
    records = []
    for feature in features:
        if feature not in reference.columns or feature not in current.columns:
            continue
        reference_values = reference[feature]
        current_values = current[feature]
        is_numeric = pd.api.types.is_numeric_dtype(reference_values) and pd.api.types.is_numeric_dtype(
            current_values
        )
        if is_numeric:
            psi = numeric_psi(reference_values, current_values)
            statistic, pvalue = stats.ks_2samp(
                pd.to_numeric(reference_values, errors="coerce").dropna(),
                pd.to_numeric(current_values, errors="coerce").dropna(),
            )
            tvd = np.nan
            detail = {
                "reference_mean": float(pd.to_numeric(reference_values, errors="coerce").mean()),
                "current_mean": float(pd.to_numeric(current_values, errors="coerce").mean()),
            }
        else:
            psi = categorical_psi(reference_values, current_values)
            breakdown = reference_values.astype("object").fillna("__missing__").value_counts(
                normalize=True
            )
            current_breakdown = current_values.astype("object").fillna(
                "__missing__"
            ).value_counts(normalize=True)
            levels = breakdown.index.union(current_breakdown.index)
            statistic, pvalue = stats.chisquare(
                current_breakdown.reindex(levels, fill_value=0.0).to_numpy()
                * max(len(current_values), 1),
            )
            tvd = total_variation(reference_values, current_values)
            detail = {"reference_mode": str(reference_values.mode().iloc[0]) if not reference_values.mode().empty else None,
                      "current_mode": str(current_values.mode().iloc[0]) if not current_values.mode().empty else None}
        severity = (
            "significant"
            if psi >= PSI_SIGNIFICANT
            else "moderate"
            if psi >= PSI_MODERATE
            else "stable"
        )
        records.append(
            {
                "feature": feature,
                "dtype": "numeric" if is_numeric else "categorical",
                "psi": round(float(psi), 4),
                "ks_statistic": round(float(statistic), 4),
                "ks_pvalue": round(float(pvalue), 6),
                "total_variation": round(float(tvd), 4) if np.isfinite(tvd) else np.nan,
                "severity": severity,
                "reference_mean": detail.get("reference_mean"),
                "current_mean": detail.get("current_mean"),
                "reference_mode": detail.get("reference_mode"),
                "current_mode": detail.get("current_mode"),
            }
        )
    return pd.DataFrame(records).sort_values("psi", ascending=False).reset_index(drop=True)


def population_stability(
    reference: pd.Series, current: pd.Series, bins: int = 10
) -> float:
    """PSI for a score distribution (e.g., model predictions)."""
    return numeric_psi(reference, current, bins=bins)


def drift_summary(table: pd.DataFrame) -> dict:
    """Aggregate a per-feature drift table into a window-level summary."""
    if table.empty:
        return {"n_features": 0, "n_drifted": 0, "share_drifted": 0.0, "max_psi": 0.0}
    drifted = table[table["psi"] >= PSI_SIGNIFICANT]
    return {
        "n_features": len(table),
        "n_drifted": len(drifted),
        "share_drifted": float(len(drifted) / len(table)),
        "max_psi": float(table["psi"].max()),
        "mean_psi": float(table["psi"].mean()),
        "top_drifted": drifted.head(8)["feature"].tolist(),
        "severity_counts": table["severity"].value_counts().to_dict(),
    }
