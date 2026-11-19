"""Model evaluation, residual diagnostics and explainability (Module 5).

This module reads the out-of-fold predictions produced during training so that
every diagnostic is computed on data the selected model has not seen in the
fold that generated the prediction. It produces the figures and tables used in
the report to answer three questions: how accurate is the model, where does it
fail, and which attributes drive the valuation.
"""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan

from revaluation.config import METRICS_DIR, MODELS_DIR, TARGET
from revaluation.data.features import load_modeling_table
from revaluation.eda.analysis import PALETTE, _dollar, _save, _style
from revaluation.models.explain import global_importance, kind_is_linear
from revaluation.models.metrics import business_kpis, regression_metrics

logger = logging.getLogger(__name__)

FRIENDLY = {
    "linear_regression": "Linear regression",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "elastic_net": "Elastic net",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}

GROUP_COLORS = {
    "Structural": "#1f2d3d",
    "Structural composites": "#3d6b6b",
    "Neighbourhood amenities": "#a9673f",
    "Age and renovation": "#7d8a99",
    "Macro-financial": "#c9b79c",
    "Categorical": "#6b7f5e",
    "Location": "#8c5a3c",
}


def run_evaluation() -> dict:
    _style()
    summary = json.loads((METRICS_DIR / "training_summary.json").read_text())
    comparison = pd.read_csv(METRICS_DIR / "model_comparison.csv")
    oof = pd.read_parquet(METRICS_DIR / "oof_predictions.parquet")
    coverage = pd.read_csv(METRICS_DIR / "interval_coverage.csv")
    frame = load_modeling_table()
    training = frame[frame[TARGET].notna()].reset_index(drop=True)
    artifact = joblib.load(MODELS_DIR / "final_model.joblib")
    best = summary["selected_model"]

    diagnostics: dict = {"selected_model": best}
    _figure_comparison(comparison, best)
    _figure_predicted_vs_actual(training, oof, best, diagnostics)
    _figure_residuals(training, oof, best, diagnostics)
    _figure_importance(artifact, training, diagnostics)
    _figure_coefficients(artifact, diagnostics)
    _figure_coverage(coverage)
    _figure_segments(training, oof, best, diagnostics)
    _save_shap_table(artifact, training)
    _business_summary(oof, best, diagnostics)

    with open(METRICS_DIR / "evaluation_diagnostics.json", "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, default=float)
    logger.info("Evaluation diagnostics written")
    return diagnostics


def _figure_comparison(comparison: pd.DataFrame, best: str) -> None:
    table = comparison.sort_values("rmse", ascending=True)
    labels = [FRIENDLY.get(name, name) for name in table["model"]]
    colors = [PALETTE["accent"] if name == best else PALETTE["muted"] for name in table["model"]]
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), constrained_layout=True)
    axes[0].barh(labels[::-1], table["rmse"][::-1], color=colors[::-1], height=0.62)
    axes[0].set_title("Out-of-fold RMSE")
    axes[0].set_xlabel("USD")
    axes[0].xaxis.set_major_formatter(FuncFormatter(_dollar))
    axes[0].tick_params(axis="y", labelsize=7)

    axes[1].barh(labels[::-1], table["mape"][::-1], color=colors[::-1], height=0.62)
    axes[1].set_title("Out-of-fold MAPE")
    axes[1].set_xlabel("Percent")
    axes[1].tick_params(axis="y", labelsize=7)
    for axis in axes:
        axis.grid(axis="y", visible=False)
    _save(figure, "fig15_model_comparison.png")


def _figure_predicted_vs_actual(
    training: pd.DataFrame, oof: pd.DataFrame, best: str, diagnostics: dict
) -> None:
    actual = oof["y_true"].to_numpy()
    predicted = oof[best].to_numpy()
    figure, axis = plt.subplots(figsize=(5.4, 4.2), constrained_layout=True)
    axis.scatter(actual, predicted, s=8, alpha=0.5, color=PALETTE["teal"], linewidths=0)
    limit = [min(actual.min(), predicted.min()) * 0.95, max(actual.max(), predicted.max()) * 1.02]
    axis.plot(limit, limit, color=PALETTE["accent"], linewidth=1, linestyle="--")
    axis.set_xlim(limit)
    axis.set_ylim(limit)
    axis.set_xlabel("Observed sale price")
    axis.set_ylabel("Predicted sale price")
    axis.set_title("Out-of-fold predictions versus observed prices")
    axis.xaxis.set_major_formatter(FuncFormatter(_dollar))
    axis.yaxis.set_major_formatter(FuncFormatter(_dollar))
    _save(figure, "fig16_predicted_vs_actual.png")

    errors = predicted - actual
    diagnostics["prediction_error"] = {
        "mean_error": float(errors.mean()),
        "median_error": float(np.median(errors)),
        "p90_absolute_error": float(np.quantile(np.abs(errors), 0.90)),
        "max_absolute_error": float(np.abs(errors).max()),
        "within_10k": float(np.mean(np.abs(errors) <= 10_000) * 100),
        "within_20k": float(np.mean(np.abs(errors) <= 20_000) * 100),
    }


def _figure_residuals(
    training: pd.DataFrame, oof: pd.DataFrame, best: str, diagnostics: dict
) -> None:
    actual = oof["y_true"].to_numpy()
    predicted = oof[best].to_numpy()
    log_residual = np.log1p(actual) - np.log1p(predicted)
    residuals = actual - predicted

    statistic, pvalue = stats.jarque_bera(log_residual)
    bp = het_breuschpagan(residuals, np.column_stack([np.ones(len(predicted)), predicted]))
    diagnostics["residual_diagnostics"] = {
        "jarque_bera_stat": float(statistic),
        "jarque_bera_pvalue": float(pvalue),
        "breusch_pagan_lm": float(bp[0]),
        "breusch_pagan_pvalue": float(bp[1]),
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std()),
        "log_residual_mean": float(log_residual.mean()),
        "log_residual_std": float(log_residual.std()),
        "skew_log_residual": float(stats.skew(log_residual)),
    }

    figure, axes = plt.subplots(1, 3, figsize=(7.6, 2.7), constrained_layout=True)
    axes[0].scatter(predicted, log_residual, s=6, alpha=0.5, color=PALETTE["teal"], linewidths=0)
    axes[0].axhline(0, color=PALETTE["accent"], linewidth=1, linestyle="--")
    axes[0].set_title("Log residuals versus fitted value")
    axes[0].set_xlabel("Predicted price")
    axes[0].set_ylabel("log(actual) - log(predicted)")
    axes[0].xaxis.set_major_formatter(FuncFormatter(_dollar))

    axes[1].hist(log_residual, bins=40, color=PALETTE["ink"], edgecolor="white", linewidth=0.3)
    axes[1].set_title("Distribution of log residuals")
    axes[1].set_xlabel("Log residual")
    axes[1].set_ylabel("Count")

    stats.probplot(log_residual, dist="norm", plot=axes[2])
    axes[2].set_title("Normal Q-Q plot")
    axes[2].get_lines()[0].set_markersize(2)
    axes[2].get_lines()[0].set_color(PALETTE["muted"])
    axes[2].get_lines()[1].set_color(PALETTE["accent"])
    _save(figure, "fig17_residual_diagnostics.png")

    # Error concentration: the 5% worst predictions carry a large share of the
    # total absolute error, which is the operational case for human review.
    absolute = np.abs(residuals)
    top_share = np.sort(absolute)[-int(0.05 * len(absolute)) :].sum() / absolute.sum()
    diagnostics["prediction_error"]["top5pct_error_share"] = float(top_share)


def _figure_importance(artifact: dict, training: pd.DataFrame, diagnostics: dict) -> None:
    pipeline = artifact["pipeline"]
    X = training.drop(columns=[TARGET])
    importance = global_importance(
        pipeline, X, max_samples=400, background=artifact.get("background")
    )
    importance.to_csv(METRICS_DIR / "shap_importance.csv", index=False)
    diagnostics["top_drivers"] = importance.head(12).to_dict(orient="records")

    top = importance.head(15).iloc[::-1]
    colors = [GROUP_COLORS.get(group, PALETTE["muted"]) for group in top["group"]]
    figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    axis.barh(top["feature"], top["share_pct"], color=colors, height=0.65)
    axis.set_xlabel("Share of total mean absolute SHAP value (percent)")
    axis.set_title("Global drivers of the valuation (SHAP)")
    axis.tick_params(axis="y", labelsize=7)
    handles = [
        Patch(facecolor=color, label=group)
        for group, color in GROUP_COLORS.items()
        if group in set(top["group"])
    ]
    axis.legend(handles=handles, loc="lower right", fontsize=6.5)
    _save(figure, "fig18_global_drivers.png")

    group_share = (
        importance.groupby("group")["mean_abs_shap"].sum().sort_values(ascending=False)
    )
    diagnostics["group_share_pct"] = (
        (100 * group_share / group_share.sum()).round(2).to_dict()
    )


def _figure_coefficients(artifact: dict, diagnostics: dict) -> None:
    pipeline = artifact["pipeline"]
    if not kind_is_linear(pipeline):
        return
    model = pipeline.named_steps["model"]
    preprocess = pipeline.named_steps["preprocess"]
    names = list(preprocess.get_feature_names_out())
    coefficients = pd.Series(model.coef_, index=names)
    table = (
        pd.DataFrame({"encoded_feature": coefficients.index, "coefficient": coefficients.values})
        .assign(abs_coef=lambda d: d["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .head(20)
    )
    table.to_csv(METRICS_DIR / "model_coefficients.csv", index=False)
    diagnostics["top_coefficients"] = table.to_dict(orient="records")

    show = table.head(14).iloc[::-1]
    colors = [
        PALETTE["teal"] if value > 0 else PALETTE["accent"] for value in show["coefficient"]
    ]
    figure, axis = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    axis.barh(show["encoded_feature"], show["coefficient"], color=colors, height=0.62)
    axis.axvline(0, color=PALETTE["ink"], linewidth=0.7)
    axis.set_title("Largest standardised coefficients (log-price units)")
    axis.set_xlabel("Coefficient")
    axis.tick_params(axis="y", labelsize=6.5)
    _save(figure, "fig19_model_coefficients.png")


def _figure_coverage(coverage: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    segments = ["entry (<130k)", "mid (130k-200k)", "upper (200k-300k)", "luxury (>=300k)", "all"]
    positions = np.arange(len(segments))
    for index, level in enumerate(["0.80", "0.90"]):
        subset = coverage[np.isclose(coverage["level"], float(level))].set_index("segment")
        values = [subset.loc[segment, "coverage"] * 100 if segment in subset.index else np.nan for segment in segments]
        axes[0].bar(
            positions + (index - 0.5) * 0.36,
            values,
            width=0.36,
            color=PALETTE["teal"] if level == "0.80" else PALETTE["ink"],
            label=f"Nominal {int(float(level) * 100)}%",
        )
        axes[0].axhline(float(level) * 100, color=PALETTE["accent"], linewidth=0.8, linestyle="--")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(["Entry", "Mid", "Upper", "Luxury", "All"], fontsize=7)
    axes[0].set_ylabel("Empirical coverage (percent)")
    axes[0].set_title("Prediction interval coverage by price segment")
    axes[0].legend(fontsize=6.5, loc="upper center", ncol=2, frameon=False)
    axes[0].set_ylim(50, 118)

    subset = coverage[np.isclose(coverage["level"], 0.90)].set_index("segment")
    widths = [subset.loc[segment, "mean_width_pct"] if segment in subset.index else np.nan for segment in segments]
    axes[1].bar(positions, widths, color=PALETTE["sand"], width=0.62)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(["Entry", "Mid", "Upper", "Luxury", "All"], fontsize=7)
    axes[1].set_ylabel("Mean width (percent of price)")
    axes[1].set_title("90% interval width by price segment")
    _save(figure, "fig20_interval_coverage.png")


def _figure_segments(
    training: pd.DataFrame, oof: pd.DataFrame, best: str, diagnostics: dict
) -> None:
    merged = training.assign(prediction=oof[best].to_numpy())
    merged["absolute_error"] = (merged["prediction"] - merged[TARGET]).abs()
    merged["bias_pct"] = (
        (merged["prediction"] - merged[TARGET]) / merged[TARGET] * 100
    )
    table = (
        merged.groupby("neighbourhood_segment", observed=True)
        .agg(
            n=("id", "size"),
            mae=("absolute_error", "mean"),
            bias=("bias_pct", "mean"),
            median_price=(TARGET, "median"),
        )
        .round(2)
        .reset_index()
    )
    diagnostics["segment_performance"] = table.to_dict(orient="records")

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    axes[0].bar(table["neighbourhood_segment"], table["mae"], color=PALETTE["teal"], width=0.6)
    axes[0].set_title("Mean absolute error by sub-market")
    axes[0].set_ylabel("MAE (USD)")
    axes[0].yaxis.set_major_formatter(FuncFormatter(_dollar))

    axes[1].bar(
        table["neighbourhood_segment"],
        table["bias"],
        color=[PALETTE["accent"] if value > 0 else PALETTE["teal"] for value in table["bias"]],
        width=0.6,
    )
    axes[1].axhline(0, color=PALETTE["ink"], linewidth=0.7)
    axes[1].set_title("Mean signed bias by sub-market")
    axes[1].set_ylabel("Bias (percent of price)")
    _save(figure, "fig21_error_by_segment.png")


def _save_shap_table(artifact: dict, training: pd.DataFrame) -> None:
    pipeline = artifact["pipeline"]
    X = training.drop(columns=[TARGET])
    importance = global_importance(
        pipeline, X, max_samples=250, background=artifact.get("background")
    )
    importance.to_csv(METRICS_DIR / "shap_importance_sample.csv", index=False)


def _business_summary(oof: pd.DataFrame, best: str, diagnostics: dict) -> None:
    actual = oof["y_true"].to_numpy()
    predicted = oof[best].to_numpy()
    metrics = regression_metrics(actual, predicted)
    metrics.update(business_kpis(actual, predicted))
    diagnostics["business_metrics"] = metrics
    diagnostics["error_by_price_band"] = {}
    bands = {
        "entry (<130k)": actual < 130_000,
        "mid (130k-200k)": (actual >= 130_000) & (actual < 200_000),
        "upper (200k-300k)": (actual >= 200_000) & (actual < 300_000),
        "luxury (>=300k)": actual >= 300_000,
    }
    for label, mask in bands.items():
        diagnostics["error_by_price_band"][label] = {
            "n": int(mask.sum()),
            "rmse": float(np.sqrt(np.mean((predicted[mask] - actual[mask]) ** 2))),
            "mae": float(np.mean(np.abs(predicted[mask] - actual[mask]))),
            "bias_pct": float(
                np.mean((predicted[mask] - actual[mask]) / actual[mask]) * 100
            ),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_evaluation()
