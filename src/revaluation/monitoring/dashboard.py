"""Monitoring dashboard: figures, HTML report and machine-readable summary.

The dashboard is a static, self-contained HTML document so that it can be
archived with each model release and served from the repository or from the
application. It complements the interactive Streamlit view, which reads the
same JSON summary.
"""

from __future__ import annotations

import base64
import html
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

from revaluation.config import FIGURES_DIR, MONITORING_DIR
from revaluation.eda.analysis import PALETTE, _dollar, _save, _style

logger = logging.getLogger(__name__)


def build_figures(
    performance: pd.DataFrame,
    drift_table: pd.DataFrame,
    reference_predictions: np.ndarray,
    latest_predictions: np.ndarray,
    baseline_rmse: float,
) -> dict[str, Path]:
    _style()
    paths: dict[str, Path] = {}

    drift_window = drift_table[drift_table["feature"] != "__prediction__"]
    prediction_rows = drift_table[drift_table["feature"] == "__prediction__"].set_index("window")[
        "psi"
    ]
    pivot = drift_window.pivot_table(index="window", columns="feature", values="psi")
    top_features = (
        drift_window.groupby("feature")["psi"].max().sort_values(ascending=False).head(12).index
    )
    matrix = pivot[top_features].copy()
    matrix["Prediction"] = prediction_rows.reindex(matrix.index)

    order = list(performance["window"])
    matrix = matrix.reindex(order)
    clipped = np.clip(matrix.to_numpy(dtype=float), 0, 1.0)

    figure, axis = plt.subplots(figsize=(7.4, 3.6), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list(
        "drift", ["#eef2f5", PALETTE["sand"], PALETTE["accent"]]
    )
    image = axis.imshow(clipped, aspect="auto", cmap=cmap, vmin=0, vmax=1.0)
    axis.set_xticks(range(matrix.shape[1]))
    axis.set_xticklabels(matrix.columns, rotation=55, ha="right", fontsize=6.3)
    axis.set_yticks(range(matrix.shape[0]))
    axis.set_yticklabels(matrix.index, fontsize=7)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            if not np.isfinite(value):
                continue
            axis.text(
                column,
                row,
                f"{value:.2f}" if value < 10 else f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=5.6,
                color="white" if clipped[row, column] > 0.55 else PALETTE["ink"],
            )
    colorbar = figure.colorbar(image, ax=axis, shrink=0.85)
    colorbar.set_label("PSI (clipped at 1.0 for display)", fontsize=7)
    colorbar.ax.tick_params(labelsize=6.5)
    axis.set_title("Population stability index by monitoring window")
    paths["drift_heatmap"] = FIGURES_DIR / _save(figure, "fig22_drift_heatmap.png")

    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)
    positions = np.arange(len(performance))
    ratios = performance["rmse_ratio_vs_baseline"].to_numpy(dtype=float)
    colors = [PALETTE["accent"] if value > 1.2 else PALETTE["teal"] for value in ratios]
    axes[0].bar(positions, ratios, width=0.6, color=colors)
    axes[0].axhline(1.0, color=PALETTE["muted"], linestyle=":", linewidth=0.9, label="Reference RMSE")
    axes[0].axhline(1.2, color=PALETTE["accent"], linestyle="--", linewidth=0.9, label="Alert threshold (1.2x)")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels([f"W{i + 1}" for i in positions])
    axes[0].set_ylim(0.8, max(1.45, float(ratios.max()) * 1.25))
    axes[0].set_ylabel("RMSE relative to reference")
    axes[0].set_title("Rolling accuracy relative to the reference vintage")
    axes[0].legend(fontsize=6.5, loc="upper left", frameon=False, ncol=2)

    axes[1].plot(
        positions,
        performance["90_coverage"] * 100,
        marker="o",
        markersize=4,
        color=PALETTE["ink"],
        label="Empirical 90% coverage",
    )
    axes[1].axhline(90, color=PALETTE["accent"], linestyle="--", linewidth=0.9, label="Nominal 90%")
    axes[1].axhline(85, color=PALETTE["muted"], linestyle=":", linewidth=0.9, label="Alert threshold")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([f"W{i + 1}" for i in positions])
    axes[1].set_ylim(70, 100)
    axes[1].set_ylabel("Coverage (percent)")
    axes[1].set_title("Calibration of the 90% interval")
    axes[1].legend(fontsize=6.5, loc="lower left")
    paths["performance"] = FIGURES_DIR / _save(figure, "fig23_rolling_performance.png")

    figure, axis = plt.subplots(figsize=(6.4, 3.0), constrained_layout=True)
    axis.hist(
        reference_predictions / 1000.0,
        bins=40,
        density=True,
        alpha=0.55,
        color=PALETTE["muted"],
        label="Reference vintage",
    )
    axis.hist(
        latest_predictions / 1000.0,
        bins=40,
        density=True,
        alpha=0.55,
        color=PALETTE["teal"],
        label="Latest window",
    )
    axis.set_xlabel("Predicted sale price (thousands USD)")
    axis.set_ylabel("Density (per thousand USD)")
    axis.xaxis.set_major_formatter(FuncFormatter(_dollar))
    axis.set_title(
        f"Prediction distribution, latest window (PSI {float(prediction_rows.iloc[-1]):.3f})"
    )
    axis.legend(fontsize=7)
    paths["prediction_distribution"] = FIGURES_DIR / _save(
        figure, "fig24_prediction_drift.png"
    )
    return paths


def _image_tag(path: Path, alt: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(alt)}">'
        f"<figcaption>{html.escape(alt)}</figcaption></figure>"
    )


def _table(frame: pd.DataFrame, columns: list[str], headers: list[str], formats: dict) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            formatter = formats.get(column)
            if callable(formatter):
                text = formatter(value)
            elif isinstance(value, float):
                text = f"{value:,.2f}"
            else:
                text = str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def render_report(
    summary: dict,
    figures: dict[str, Path],
    output: Path | None = None,
) -> Path:
    output = output or MONITORING_DIR / "monitoring_report.html"
    observed = summary["observed"]
    stress = summary["stress"]
    performance = pd.DataFrame(observed["performance"])
    drift_summary = pd.DataFrame(observed["drift_summary"])
    alerts = observed["alerting"]["alerts"]

    money = lambda value: f"{value:,.0f}"
    percent = lambda value: f"{value:.1f}"

    kpis = [
        ("Reference vintage", summary["reference"]["period"], ""),
        ("Windows monitored", str(len(performance)), ""),
        ("Reference OOF RMSE", money(summary["reference"]["baseline"]["rmse"]), "USD"),
        ("Latest window MAE", money(performance["mae"].iloc[-1]), "USD"),
        (
            "Latest 90% coverage",
            f"{performance['90_coverage'].iloc[-1] * 100:.1f}",
            "percent",
        ),
        ("Monitoring decision", summary["observed"]["alerting"]["decision"].replace("_", " "), ""),
    ]
    cards = "".join(
        f'<div class="card"><span class="label">{html.escape(label)}</span>'
        f'<span class="value">{html.escape(value)}</span>'
        f'<span class="unit">{html.escape(unit)}</span></div>'
        for label, value, unit in kpis
    )

    performance_columns = [
        "window",
        "n",
        "rmse",
        "mae",
        "mape",
        "bias_pct",
        "90_coverage",
        "rmse_ratio_vs_baseline",
    ]
    performance_headers = [
        "Window",
        "Sales",
        "RMSE",
        "MAE",
        "MAPE (%)",
        "Bias (%)",
        "Coverage 90%",
        "RMSE ratio",
    ]
    performance_formats = {
        "rmse": money,
        "mae": money,
        "mape": percent,
        "bias_pct": percent,
        "90_coverage": lambda value: f"{value * 100:.1f}",
        "rmse_ratio_vs_baseline": lambda value: f"{value:.3f}",
    }
    performance_table = _table(
        performance, performance_columns, performance_headers, performance_formats
    )

    drift_columns = ["window", "n_drifted", "max_psi", "mean_psi", "prediction_psi"]
    drift_headers = ["Window", "Drifted features", "Maximum PSI", "Mean PSI", "Prediction PSI"]
    drift_formats = {
        "max_psi": lambda value: f"{value:.3f}",
        "mean_psi": lambda value: f"{value:.3f}",
        "prediction_psi": lambda value: f"{value:.3f}",
    }
    drift_summary_table = _table(
        drift_summary, drift_columns, drift_headers, drift_formats
    )

    alert_rows = (
        "".join(
            f"<tr><td>{html.escape(alert['window'])}</td>"
            f"<td>{html.escape(alert['rule'])}</td>"
            f"<td><span class='badge {alert['severity']}'>{alert['severity']}</span></td>"
            f"<td>{html.escape(str(alert['value']))}</td>"
            f"<td>{html.escape(alert['threshold'])}</td>"
            f"<td>{html.escape(alert['action'])}</td></tr>"
            for alert in alerts
        )
        or "<tr><td colspan='6'>No alerts were raised in the monitored period.</td></tr>"
    )

    stress_performance = pd.DataFrame(stress["performance"])
    stress_table = _table(
        stress_performance,
        ["window", "n", "rmse", "mae", "mape", "90_coverage", "rmse_ratio_vs_baseline"],
        ["Window", "Sales", "RMSE", "MAE", "MAPE (%)", "Coverage 90%", "RMSE ratio"],
        performance_formats,
    )

    styles = """
    :root { --ink: #1f2d3d; --muted: #5f6b7a; --line: #d7dde6; --paper: #ffffff; --accent: #8c5a3c; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #f5f6f8; color: var(--ink);
           font-family: "DejaVu Serif", "Liberation Serif", Georgia, serif; line-height: 1.55; }
    main { max-width: 1040px; margin: 0 auto; padding: 42px 32px 64px; background: var(--paper); }
    header { border-bottom: 2px solid var(--ink); padding-bottom: 16px; margin-bottom: 26px; }
    h1 { font-size: 25px; margin: 0 0 6px; font-weight: 600; }
    h2 { font-size: 16px; margin: 34px 0 10px; padding-bottom: 4px; border-bottom: 1px solid var(--line); }
    h3 { font-size: 13px; margin: 20px 0 8px; }
    p, li, td, th { font-size: 11.5px; }
    .subtitle { color: var(--muted); font-size: 11.5px; margin: 0; }
    .kpis { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 22px 0; }
    .card { border: 1px solid var(--line); padding: 12px 14px; background: #fbfbfc; }
    .card .label { display: block; font-size: 9.5px; text-transform: uppercase;
                   letter-spacing: 0.06em; color: var(--muted); }
    .card .value { display: block; font-size: 20px; margin-top: 4px; }
    .card .unit { font-size: 9.5px; color: var(--muted); }
    table { width: 100%; border-collapse: collapse; margin: 10px 0 18px; }
    th, td { border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left; }
    th { background: #eef1f5; font-size: 10px; text-transform: uppercase; letter-spacing: 0.04em; }
    tr:nth-child(even) td { background: #fafbfc; }
    figure { margin: 14px 0 22px; }
    figure img { width: 100%; border: 1px solid var(--line); }
    figcaption { font-size: 9.5px; color: var(--muted); margin-top: 5px; }
    .badge { display: inline-block; padding: 1px 7px; font-size: 9px; letter-spacing: 0.04em;
             text-transform: uppercase; border: 1px solid currentColor; }
    .badge.critical { color: #8c2f2f; }
    .badge.warning { color: #8a5a1f; }
    .decision { border-left: 4px solid var(--accent); padding: 10px 14px; background: #fbf8f4; margin: 12px 0 18px; }
    .decision strong { display: block; font-size: 12px; margin-bottom: 3px; }
    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
    .note { color: var(--muted); font-size: 10px; }
    footer { margin-top: 40px; border-top: 1px solid var(--line); padding-top: 12px;
             color: var(--muted); font-size: 9.5px; }
    @media print { body { background: white; } main { max-width: none; padding: 0; } }
    """

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Post-Deployment Monitoring Report</title>
<style>{styles}</style>
</head>
<body>
<main>
<header>
  <h1>Post-Deployment Monitoring Report</h1>
  <p class="subtitle">Residential valuation model, Ames Iowa. Generated {html.escape(summary['generated_at'])}.
  Reference vintage {html.escape(summary['reference']['period'])}; monitoring period
  {html.escape(performance['window'].iloc[0])} to {html.escape(performance['window'].iloc[-1])}.</p>
</header>

<section>
  <h2>1. Executive summary</h2>
  <p>The deployed model was refitted on sales up to December 2008 and replayed against every
  transaction recorded between January 2009 and July 2010. Input drift is measured with the
  population stability index and the Kolmogorov-Smirnov statistic; accuracy and interval
  calibration are measured on realised sale prices, which arrive with a one-window label lag.
  The monitoring policy returned the decision
  <strong>{html.escape(summary['observed']['alerting']['decision'].replace('_', ' '))}</strong>.</p>
  <div class="kpis">{cards}</div>
  <div class="decision">
    <strong>Recommended action</strong>
    {html.escape(summary['observed']['alerting']['action'])}
  </div>
</section>

<section>
  <h2>2. Input and prediction drift</h2>
  <p>The credit-market variables moved outside the reference support during the monitoring
  period: mortgage rates fell from a reference median of roughly 6.5 percent to roughly 5.0
  percent, and the local price index declined from about 104 to 96. PSI values far above the
  0.20 threshold therefore reflect non-overlapping distributions rather than a modelling error.
  The heat map clips the index at 1.0 for readability; exact values are stored in
  <code>drift_by_window.csv</code>.</p>
  {_image_tag(figures['drift_heatmap'], 'Population stability index by monitoring window')}
  {drift_summary_table}
</section>

<section>
  <h2>3. Accuracy and calibration</h2>
  <p>Accuracy is stable in the first two windows and deteriorates modestly in the last two,
  where the model systematically under-predicts realised prices by up to five percent. The
  consequence is visible in the calibration panel: the nominal 90 percent interval covers only
  80 percent of sales in the final window. Interval recalibration is therefore the first
  corrective action, before a full retraining cycle.</p>
  {_image_tag(figures['performance'], 'Rolling accuracy and interval coverage by window')}
  {performance_table}
  {_image_tag(figures['prediction_distribution'], 'Prediction distribution in the latest window')}
</section>

<section>
  <h2>4. Alerting and retraining policy</h2>
  <p>Nine rules are evaluated on every window. Sustained breaches (two consecutive windows)
  escalate from warning to critical and open a retraining ticket. The table lists every alert
  raised by the observed replay.</p>
  <table>
    <thead><tr><th>Window</th><th>Rule</th><th>Severity</th><th>Value</th><th>Threshold</th><th>Action</th></tr></thead>
    <tbody>{alert_rows}</tbody>
  </table>
</section>

<section>
  <h2>5. Adverse stress scenario</h2>
  <p>To verify that the alerting machinery reacts to a sharper regime change, the same model was
  scored on a counterfactual stream in which the mortgage rate rises by 1.5 percentage points,
  inventory rises by one month of supply, and neighbourhood amenity distributions improve by
  60 percent. Recorded prices are held at their observed values; the scenario therefore measures
  sensitivity to context shift, not a forecast of future accuracy.</p>
  {stress_table}
</section>

<section>
  <h2>6. Methodology and limitations</h2>
  <ul>
    <li>Reference model: the selected estimator refitted on 945 sales completed before 2009;
        out-of-fold RMSE {money(summary['reference']['baseline']['rmse'])} USD.</li>
    <li>Windows: four equal-volume chronological blocks, so that every RMSE and PSI estimate
        rests on roughly the same number of sales.</li>
    <li>Drift: PSI on reference quantile bins (numeric) or level proportions (categorical),
        plus two-sample Kolmogorov-Smirnov for numeric features.</li>
    <li>Uncertainty: split-conformal intervals calibrated on the reference vintage; coverage is
        measured against realised sales, which is the only honest test of calibration.</li>
    <li>The simulation uses recorded Ames sales as delayed labels. It cannot reproduce
        seasonality beyond the sampled period, and the stress scenario holds prices fixed.</li>
  </ul>
</section>

<footer>
  Monitoring dashboard generated by the repository pipeline. The machine-readable summary is
  <code>monitoring_summary.json</code>; the interactive view is available in the Streamlit
  application under the Monitoring page.
</footer>
</main>
</body>
</html>
"""
    output.write_text(document, encoding="utf-8")
    logger.info("Monitoring report written to %s", output)
    return output


def write_summary(
    reference: dict,
    performance: dict,
    drift: dict,
    alerting: dict,
    stress_performance: dict,
    stress_alerting: dict,
    quantiles: dict,
) -> dict:
    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reference": reference,
        "observed": {
            "performance": performance,
            "drift_summary": drift,
            "alerting": alerting,
        },
        "stress": {
            "performance": stress_performance,
            "alerting": stress_alerting,
        },
        "conformal_quantiles": quantiles,
    }
    with open(MONITORING_DIR / "monitoring_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=float)
    with open(MONITORING_DIR / "alerts.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"observed": alerting["alerts"], "stress": stress_alerting["alerts"]},
            handle,
            indent=2,
            default=float,
        )
    return summary
