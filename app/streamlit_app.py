"""Streamlit valuation tool and monitoring console (Module 6).

The application has four pages:

* **Valuation** - agent-facing form that returns a point estimate, calibrated
  prediction intervals and the main local price drivers.
* **Model performance** - out-of-fold diagnostics, model comparison and
  interval calibration.
* **Monitoring** - post-deployment drift and accuracy dashboard with the
  retraining decision.
* **Data and methodology** - data dictionary, cleaning log and pipeline
  description.

The application calls the FastAPI service when it is reachable and falls back
to an in-process copy of the same pipeline otherwise, so the demo remains
functional when only the Streamlit process is running.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from revaluation.config import (
    DOCS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MONITORING_DIR,
    ROOT,
)
from revaluation.data.clean import load_cleaned
from revaluation.data.load import load_train
from revaluation.data.synthetic import NEIGHBOURHOOD_TIERS, synthetic_specs_frame

API_URL = os.getenv("VALUATION_API_URL", "http://127.0.0.1:8000")
CURRENCY = "USD"

st.set_page_config(
    page_title="Residential Valuation Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root { --ink: #1f2d3d; --muted: #5f6b7a; --line: #d7dde6; --accent: #8c5a3c; }
html, body, [class*="css"] { font-family: "DejaVu Serif", "Liberation Serif", Georgia, serif; }
h1, h2, h3 { color: #1f2d3d; letter-spacing: 0.01em; }
h1 { font-size: 1.65rem; }
h2 { font-size: 1.25rem; }
h3 { font-size: 1.05rem; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }
div[data-testid="stMetricValue"] { font-size: 1.6rem; color: #1f2d3d; }
div[data-testid="stMetricLabel"] { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }
.stTabs [data-baseweb="tab"] { font-size: 0.95rem; }
.small-note { color: #5f6b7a; font-size: 0.82rem; }
.hero { border-left: 4px solid #8c5a3c; padding: 0.7rem 1rem; background: #fbf8f4; margin-bottom: 1.2rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def local_service():
    from revaluation.api.service import ValuationService

    return ValuationService()


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    file = Path(path)
    return json.loads(file.read_text()) if file.exists() else {}


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    file = Path(path)
    return pd.read_csv(file) if file.exists() else pd.DataFrame()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def predict(payload: dict) -> tuple[dict, str]:
    """Call the API and fall back to the in-process model."""
    try:
        response = requests.post(
            f"{API_URL}/predict", params={"explain": "true"}, json=payload, timeout=4
        )
        response.raise_for_status()
        return response.json(), "FastAPI service"
    except Exception:  # noqa: BLE001 - fall back to the in-process pipeline
        return local_service().predict(payload, explain=True), "in-process pipeline"


def money(value: float | None) -> str:
    if value is None or not pd.notna(value):
        return "n/a"
    return f"${value:,.0f}"


def sidebar() -> str:
    st.sidebar.title("Residential Valuation")
    st.sidebar.caption("Ames, Iowa. End-to-end analytics platform.")
    page = st.sidebar.radio(
        "View",
        ["Valuation", "Model performance", "Monitoring", "Data and methodology"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.markdown(
        '<p class="small-note">The valuation model estimates market value from '
        "structural, locational and market-context attributes observable at listing "
        "time. Intervals are calibrated with split conformal prediction.</p>",
        unsafe_allow_html=True,
    )
    try:
        info = load_json(str(ROOT / "artifacts" / "models" / "model_metadata.json"))
        if info:
            st.sidebar.caption(
                f"Model: {info.get('selected_model_label', 'n/a')}  |  "
                f"Trained on {info.get('n_train', 'n/a')} sales"
            )
    except Exception as error:  # noqa: BLE001 - metadata is optional in the sidebar
        st.sidebar.caption("Model metadata unavailable.")
        _ = error
    return page


def valuation_page() -> None:
    st.title("Property valuation")
    st.markdown(
        '<div class="hero">Estimate the market value of a residential listing and the '
        "uncertainty around that estimate. Fields marked as optional fall back to the "
        "documented neighbourhood and structural defaults when left blank. The model is "
        "calibrated on the 2006-2010 Ames market, so the transaction date is restricted to "
        "2006-2012.</div>",
        unsafe_allow_html=True,
    )

    neighbourhoods = sorted(NEIGHBOURHOOD_TIERS)
    defaults = {
        "neighborhood": "CollgCr",
        "sale_year": 2010,
        "sale_month": 6,
        "overall_qual": 6,
        "overall_cond": 5,
        "gr_liv_area": 1500.0,
        "lot_area": 9000.0,
        "year_built": 1975,
        "year_remod_add": 2005,
        "total_bsmt_sf": 800.0,
        "first_flr_sf": 900.0,
        "second_flr_sf": 0.0,
        "full_bath": 2,
        "half_bath": 0,
        "bedroom_abv_gr": 3,
        "tot_rms_abv_grd": 7,
        "fireplaces": 0,
        "garage_cars": 2,
        "garage_area": 480.0,
    }

    with st.form("valuation_form"):
        st.subheader("Location and transaction date")
        col1, col2, col3 = st.columns([2, 1, 1])
        neighborhood = col1.selectbox(
            "Neighbourhood", neighbourhoods, index=neighbourhoods.index(defaults["neighborhood"])
        )
        sale_year = col2.number_input(
            "Sale year", min_value=2006, max_value=2012, value=defaults["sale_year"], step=1
        )
        sale_month = col3.selectbox(
            "Sale month", list(range(1, 13)), index=defaults["sale_month"] - 1
        )

        st.subheader("Structure and condition")
        col1, col2, col3, col4 = st.columns(4)
        overall_qual = col1.slider(
            "Overall quality (1-10)", 1, 10, defaults["overall_qual"]
        )
        overall_cond = col2.slider(
            "Overall condition (1-10)", 1, 10, defaults["overall_cond"]
        )
        gr_liv_area = col3.number_input(
            "Above-grade living area (sqft)",
            min_value=300.0,
            max_value=8000.0,
            value=defaults["gr_liv_area"],
            step=50.0,
        )
        lot_area = col4.number_input(
            "Lot area (sqft)",
            min_value=1000.0,
            max_value=100000.0,
            value=defaults["lot_area"],
            step=500.0,
        )

        col1, col2, col3, col4 = st.columns(4)
        year_built = col1.number_input(
            "Year built", min_value=1870, max_value=2025, value=defaults["year_built"], step=1
        )
        year_remod_add = col2.number_input(
            "Year of last remodelling",
            min_value=1870,
            max_value=2025,
            value=defaults["year_remod_add"],
            step=1,
        )
        total_bsmt_sf = col3.number_input(
            "Basement area (sqft)",
            min_value=0.0,
            max_value=5000.0,
            value=defaults["total_bsmt_sf"],
            step=50.0,
        )
        first_flr_sf = col4.number_input(
            "First floor area (sqft)",
            min_value=300.0,
            max_value=5000.0,
            value=defaults["first_flr_sf"],
            step=50.0,
        )

        col1, col2, col3, col4 = st.columns(4)
        full_bath = col1.number_input(
            "Full bathrooms", min_value=0, max_value=6, value=defaults["full_bath"]
        )
        half_bath = col2.number_input(
            "Half bathrooms", min_value=0, max_value=4, value=defaults["half_bath"]
        )
        bedroom_abv_gr = col3.number_input(
            "Bedrooms above grade", min_value=0, max_value=10, value=defaults["bedroom_abv_gr"]
        )
        tot_rms_abv_grd = col4.number_input(
            "Total rooms above grade", min_value=1, max_value=16, value=defaults["tot_rms_abv_grd"]
        )

        col1, col2, col3, col4 = st.columns(4)
        garage_cars = col1.number_input(
            "Garage capacity (cars)", min_value=0, max_value=5, value=defaults["garage_cars"]
        )
        garage_area = col2.number_input(
            "Garage area (sqft)",
            min_value=0.0,
            max_value=2000.0,
            value=defaults["garage_area"],
            step=20.0,
        )
        fireplaces = col3.number_input(
            "Fireplaces", min_value=0, max_value=4, value=defaults["fireplaces"]
        )
        second_flr_sf = col4.number_input(
            "Second floor area (sqft)", min_value=0.0, max_value=4000.0, value=0.0, step=50.0
        )

        with st.expander("Neighbourhood amenities (optional)"):
            st.caption(
                "When omitted, the model uses the median amenity profile of the selected "
                "neighbourhood and the macro-financial conditions of the chosen sale month."
            )
            use_custom = st.checkbox("Provide amenity measurements", value=False)
            col1, col2, col3 = st.columns(3)
            walk_score = col1.number_input("Walk score (0-100)", 0, 100, 45, disabled=not use_custom)
            school_rating = col2.number_input("School rating (1-10)", 1, 10, 6, disabled=not use_custom)
            dist_school = col3.number_input(
                "Distance to school (km)", 0.0, 15.0, 1.5, step=0.1, disabled=not use_custom
            )
            col1, col2 = st.columns(2)
            crime_index = col1.number_input(
                "Crime index (0-100)", 0, 100, 45, disabled=not use_custom
            )
            amenity_score = col2.number_input(
                "Amenity score (0-100)", 0, 100, 55, disabled=not use_custom
            )

        submitted = st.form_submit_button("Estimate market value", type="primary")

    if not submitted:
        st.info("Adjust the listing attributes and submit the form to obtain a valuation.")
        return

    payload = {
        "neighborhood": neighborhood,
        "sale_year": int(sale_year),
        "sale_month": int(sale_month),
        "overall_qual": int(overall_qual),
        "overall_cond": int(overall_cond),
        "gr_liv_area": float(gr_liv_area),
        "lot_area": float(lot_area),
        "year_built": int(year_built),
        "year_remod_add": int(year_remod_add),
        "total_bsmt_sf": float(total_bsmt_sf),
        "first_flr_sf": float(first_flr_sf),
        "second_flr_sf": float(second_flr_sf),
        "full_bath": int(full_bath),
        "half_bath": int(half_bath),
        "bedroom_abv_gr": int(bedroom_abv_gr),
        "tot_rms_abv_grd": int(tot_rms_abv_grd),
        "garage_cars": int(garage_cars),
        "garage_area": float(garage_area),
        "fireplaces": int(fireplaces),
    }
    if use_custom:
        payload.update(
            {
                "walk_score": int(walk_score),
                "school_rating": int(school_rating),
                "dist_school_km": float(dist_school),
                "crime_index": int(crime_index),
                "amenity_score": float(amenity_score),
            }
        )

    result, backend = predict(payload)
    estimate = result["estimate"]
    interval80 = result.get("interval_80") or [None, None]
    interval90 = result.get("interval_90") or [None, None]

    st.divider()
    left, middle, right = st.columns([1.2, 1, 1])
    left.metric("Estimated market value", money(estimate))
    left.caption(f"Point estimate in {CURRENCY}. Served by {backend}.")
    middle.metric("80% interval", f"{money(interval80[0])} - {money(interval80[1])}")
    middle.caption("Four-in-five realised sales are expected to fall inside this range.")
    right.metric("90% interval", f"{money(interval90[0])} - {money(interval90[1])}")
    right.caption(
        f"Confidence assessment: {result.get('confidence', 'n/a')} "
        f"(score {result.get('confidence_score', 0):.2f})."
    )

    if result.get("drivers"):
        st.subheader("Main drivers for this valuation")
        frame = pd.DataFrame(result["drivers"]).copy()
        frame["effect"] = frame["relative_effect_pct"] * frame["contribution_log"].apply(
            lambda value: 1 if value >= 0 else -1
        )
        frame["label"] = frame["feature"].str.replace("_", " ")
        frame = frame.sort_values("effect")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(7.0, 2.9), constrained_layout=True)
        colors = [
            "#3d6b6b" if value >= 0 else "#8c5a3c" for value in frame["effect"]
        ]
        axis.barh(frame["label"], frame["effect"], color=colors, height=0.62)
        axis.axvline(0, color="#1f2d3d", linewidth=0.8)
        axis.set_xlabel("Approximate effect on value (percent)")
        axis.set_title("Local contribution of each attribute", fontsize=9)
        axis.tick_params(axis="y", labelsize=7.5)
        axis.grid(axis="y", visible=False)
        span = max(abs(frame["effect"]).max(), 0.5)
        axis.set_xlim(-span * 1.35, span * 1.35)
        for index, value in enumerate(frame["effect"]):
            axis.annotate(
                f"{value:+.1f}%",
                xy=(value, index),
                xytext=(4 if value >= 0 else -4, 0),
                textcoords="offset points",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=7,
                color="#1f2d3d",
            )
        st.pyplot(figure, use_container_width=False)
        plt.close(figure)
        st.caption(
            "Contribution of each attribute to the log-price estimate, expressed as the "
            "approximate percentage effect on value. Positive values increase the estimate."
        )

    with st.expander("Assumptions and diagnostics"):
        assumptions = result.get("assumptions", {})
        st.write(
            {
                "Neighbourhood": assumptions.get("neighbourhood"),
                "Sale year": assumptions.get("sale_year"),
                "Sale month": assumptions.get("sale_month"),
                "Defaulted context fields": ", ".join(
                    assumptions.get("derived_context_fields", [])
                )
                or "none",
            }
        )
        st.caption(
            "Fields not supplied by the user are replaced by the documented medians of the "
            "training population. The interval width reflects both model error and the "
            "heteroscedasticity of the Ames market."
        )

    eda = load_json(str(METRICS_DIR / "eda_summary.json"))
    neighbourhood_rows = {row["neighborhood"]: row for row in eda.get("neighbourhood", [])}
    reference = neighbourhood_rows.get(neighborhood)
    if reference:
        st.subheader("Neighbourhood reference")
        col1, col2, col3 = st.columns(3)
        col1.metric("Median sale price", money(reference["median_price"]))
        col2.metric("Median price per sqm", money(reference["median_ppsf"]))
        col3.metric("Mean amenity score", f"{reference['amenity']:.1f}")
        st.caption(
            f"Reference statistics from {reference['listings']} training transactions in "
            f"{neighborhood}."
        )


def performance_page() -> None:
    st.title("Model performance")
    summary = load_json(str(METRICS_DIR / "training_summary.json"))
    comparison = load_csv(str(METRICS_DIR / "model_comparison.csv"))
    coverage = load_csv(str(METRICS_DIR / "interval_coverage.csv"))
    diagnostics = load_json(str(METRICS_DIR / "evaluation_diagnostics.json"))

    if not summary:
        st.warning("Training artefacts are not available. Run the training pipeline first.")
        return

    metrics = summary["metrics"]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Selected model", summary["selected_model_label"])
    col2.metric("RMSE", money(metrics["rmse"]))
    col3.metric("MAE", money(metrics["mae"]))
    col4.metric("MAPE", f"{metrics['mape']:.2f}%")
    col5.metric("R-squared", f"{metrics['r2']:.3f}")
    st.caption(
        f"Out-of-fold estimates on {summary['n_train']} transactions with "
        f"{summary['n_features']} candidate features. Intervals calibrated at "
        f"{float(summary['conformal']['0.90']):.3f} log points around the prediction."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Within +/- $15,000", f"{metrics['within_tolerance_usd']:.1f}%")
    col2.metric("Within +/- 8%", f"{metrics['within_tolerance_pct']:.1f}%")
    col3.metric(
        "Mean asymmetric cost", money(metrics["mean_asymmetric_cost"]), help="Over-valuation weighted at 1.5."
    )

    st.subheader("Model comparison")
    display = comparison.copy()
    display = display[
        ["model_label", "rmse", "mae", "mape", "r2", "within_tolerance_pct", "cv_rmse_log"]
    ].rename(
        columns={
            "model_label": "Model",
            "rmse": "RMSE (USD)",
            "mae": "MAE (USD)",
            "mape": "MAPE (%)",
            "r2": "R2",
            "within_tolerance_pct": "Within 8% (%)",
            "cv_rmse_log": "CV RMSE (log)",
        }
    )
    st.dataframe(
        display.style.format(
            {"RMSE (USD)": "{:,.0f}", "MAE (USD)": "{:,.0f}", "MAPE (%)": "{:.2f}", "R2": "{:.4f}", "Within 8% (%)": "{:.1f}", "CV RMSE (log)": "{:.4f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Diagnostics")
    tab1, tab2, tab3 = st.tabs(["Accuracy", "Residuals", "Calibration"])
    with tab1:
        st.image(str(FIGURES_DIR / "fig15_model_comparison.png"), use_container_width=True)
        st.image(str(FIGURES_DIR / "fig16_predicted_vs_actual.png"), width=560)
    with tab2:
        st.image(str(FIGURES_DIR / "fig17_residual_diagnostics.png"), use_container_width=True)
        if diagnostics.get("residual_diagnostics"):
            residual = diagnostics["residual_diagnostics"]
            st.caption(
                f"Jarque-Bera p-value {residual['jarque_bera_pvalue']:.2e}; "
                f"Breusch-Pagan p-value {residual['breusch_pagan_pvalue']:.2e}. "
                "Residuals are heavy-tailed and heteroscedastic, which motivates the log "
                "target and the conformal intervals."
            )
    with tab3:
        st.image(str(FIGURES_DIR / "fig20_interval_coverage.png"), use_container_width=True)
        if not coverage.empty:
            table = coverage.copy()
            table["level"] = table["level"].map(lambda value: f"{value:.0%}")
            st.dataframe(
                table[
                    ["level", "segment", "coverage", "coverage_gap", "mean_width_pct"]
                ].rename(
                    columns={
                        "level": "Nominal coverage",
                        "segment": "Segment",
                        "coverage": "Empirical coverage",
                        "coverage_gap": "Gap",
                        "mean_width_pct": "Mean width (% of price)",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("Global drivers and coefficients")
    col1, col2 = st.columns(2)
    col1.image(str(FIGURES_DIR / "fig18_global_drivers.png"), use_container_width=True)
    col2.image(str(FIGURES_DIR / "fig19_model_coefficients.png"), use_container_width=True)


def monitoring_page() -> None:
    st.title("Post-deployment monitoring")
    summary = load_json(str(MONITORING_DIR / "monitoring_summary.json"))
    if not summary:
        st.warning(
            "No monitoring summary found. Run `uv run python scripts/06_run_monitoring.py`."
        )
        return

    performance = pd.DataFrame(summary["observed"]["performance"])
    drift = pd.DataFrame(summary["observed"]["drift_summary"])
    alerts = summary["observed"]["alerting"]
    reference = summary["reference"]

    st.markdown(
        f'<div class="hero">Reference vintage {reference["period"]} with '
        f'{reference["n"]} sales. Streaming simulation {reference["stream_period"]} '
        f'over {len(performance)} equal-volume windows. Labels are revealed after each '
        "window closes, mirroring the delay in recorded deeds.</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Reference RMSE", money(reference["baseline"]["rmse"]))
    col2.metric("Latest window MAE", money(performance["mae"].iloc[-1]))
    col3.metric("Latest 90% coverage", f"{performance['90_coverage'].iloc[-1] * 100:.1f}%")
    col4.metric(
        "Drifted features (latest)",
        int(drift["n_drifted"].iloc[-1]),
        help="Features whose population stability index exceeds 0.20.",
    )

    decision = alerts["decision"].replace("_", " ").title()
    st.subheader(f"Monitoring decision: {decision}")
    st.write(alerts["action"])
    if alerts["alerts"]:
        st.warning(
            f"{alerts['n_critical']} critical and {alerts['n_warnings']} warning alerts were "
            "raised in the observed replay. The full policy output is listed below."
        )
    else:
        st.success("No alerts were raised in the monitored period.")

    st.subheader("Drift and accuracy")
    tab1, tab2 = st.tabs(["Drift", "Accuracy"])
    with tab1:
        st.image(str(FIGURES_DIR / "fig22_drift_heatmap.png"), use_container_width=True)
        st.image(str(FIGURES_DIR / "fig24_prediction_drift.png"), use_container_width=True)
        st.dataframe(
            drift.rename(
                columns={
                    "window": "Window",
                    "n_drifted": "Drifted features",
                    "max_psi": "Max PSI",
                    "mean_psi": "Mean PSI",
                    "prediction_psi": "Prediction PSI",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    with tab2:
        st.image(str(FIGURES_DIR / "fig23_rolling_performance.png"), use_container_width=True)
        table = performance.copy()
        table["90_coverage"] = table["90_coverage"] * 100
        st.dataframe(
            table[
                [
                    "window",
                    "n",
                    "rmse",
                    "mae",
                    "mape",
                    "bias_pct",
                    "90_coverage",
                    "rmse_ratio_vs_baseline",
                ]
            ].rename(
                columns={
                    "window": "Window",
                    "n": "Sales",
                    "rmse": "RMSE",
                    "mae": "MAE",
                    "mape": "MAPE (%)",
                    "bias_pct": "Bias (%)",
                    "90_coverage": "Coverage 90% (%)",
                    "rmse_ratio_vs_baseline": "RMSE ratio",
                }
            ).style.format(
                {
                    "RMSE": "{:,.0f}",
                    "MAE": "{:,.0f}",
                    "MAPE (%)": "{:.2f}",
                    "Bias (%)": "{:.2f}",
                    "Coverage 90% (%)": "{:.1f}",
                    "RMSE ratio": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Alerts and retraining triggers")
    if alerts["alerts"]:
        st.dataframe(
            pd.DataFrame(alerts["alerts"])[
                ["window", "rule", "severity", "value", "threshold", "action"]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No alerts.")

    report_path = MONITORING_DIR / "monitoring_report.html"
    st.caption(f"Static monitoring report: {report_path.relative_to(ROOT)}")


def data_page() -> None:
    st.title("Data and methodology")
    st.subheader("Data sources")
    train = load_train()
    cleaned = load_cleaned()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Kaggle transactions", f"{len(train):,}")
    col2.metric("Attributes (raw)", train.shape[1])
    col3.metric("Synthetic context fields", 21)
    col4.metric("Cleaned records", f"{len(cleaned):,}")
    st.caption(
        "The base data are the Ames, Iowa transactions of the Kaggle House Prices "
        "competition. Contextual attributes are generated with documented business logic; "
        "the full specification is reproduced below and exported to "
        "docs/data_dictionary.xlsx."
    )

    st.subheader("Synthetic contextual fields")
    specs = synthetic_specs_frame()
    st.dataframe(
        specs[["column", "dtype", "unit", "range"]].rename(
            columns={
                "column": "Column",
                "dtype": "Type",
                "unit": "Unit",
                "range": "Valid range",
            }
        ),
        use_container_width=True,
        hide_index=True,
        height=250,
    )
    with st.expander("Generation logic and business assumptions"):
        for _, row in specs.iterrows():
            st.markdown(
                f"**{row['column']}** ({row['unit']}). {row['logic']} "
                f"*Assumption:* {row['assumption']}"
            )

    st.subheader("Cleaning impact")
    cleaning = load_json(str(METRICS_DIR / "cleaning_report.json"))
    if cleaning:
        report = cleaning["report"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows before cleaning", report["initial_rows"])
        col2.metric("Rows after cleaning", report["final_rows"])
        col3.metric("Implausible records removed", report["outliers_removed"])
        col4.metric("Missing cells after cleaning", sum(report["missing_after"].values()))
        st.caption(
            "Structural absence, neighbourhood-median imputation and the outlier rules are "
            "recorded in artifacts/metrics/cleaning_report.json."
        )

    st.subheader("Pipeline")
    st.markdown(
        "1. Synthetic context generation (seeded, documented).\n"
        "2. Cleaning: structural absence, neighbourhood median imputation, outlier treatment.\n"
        "3. Feature engineering: structural composites, age, renovation, amenity and macro features.\n"
        "4. Model comparison with five-fold cross-validation and randomised tuning.\n"
        "5. Conformal calibration of prediction intervals from out-of-fold residuals.\n"
        "6. Packaging as a FastAPI service and this Streamlit interface.\n"
        "7. Monitoring of input drift, accuracy and interval calibration with retraining triggers."
    )

    st.subheader("Documentation")
    for name in ("data_dictionary.md", "business_understanding.md", "model_card.md", "deployment.md"):
        path = DOCS_DIR / name
        if path.exists():
            with st.expander(name.replace("_", " ").replace(".md", "").title()):
                st.markdown(read_text(path)[:6000])

    st.caption(
        "All code is version-controlled in the repository and executed through uv with a "
        "local virtual environment."
    )


def main() -> None:
    page = sidebar()
    if page == "Valuation":
        valuation_page()
    elif page == "Model performance":
        performance_page()
    elif page == "Monitoring":
        monitoring_page()
    else:
        data_page()
    st.divider()
    st.caption(
        "Estimates are decision support, not appraisals. The model is validated on historic "
        "Ames transactions and does not observe the physical condition of a property."
    )


if __name__ == "__main__":
    main()
