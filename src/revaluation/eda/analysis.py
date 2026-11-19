"""Exploratory data analysis and diagnostic visualisation (Module 2).

The analysis answers four questions that structure the rest of the project:

1. What does the price distribution look like, and does it require a
   transformation before linear modelling?
2. Which structural attributes carry the strongest univariate signal?
3. How do location, property age and neighbourhood amenity levels segment the
   market?
4. Which variables are redundant, and what are the implications for
   interpretable linear specifications?

Every figure is written to ``artifacts/figures`` and every headline number to
``artifacts/metrics/eda_summary.json`` so that the report and the application
read the same evidence.
"""

from __future__ import annotations

import json
import logging
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from revaluation.config import (
    FIGURES_DIR,
    METRICS_DIR,
    NEIGHBOURHOOD_TIERS,
    TARGET,
)
from revaluation.data.features import load_modeling_table, vif_table

logger = logging.getLogger(__name__)

PALETTE = {
    "ink": "#1f2d3d",
    "accent": "#8c5a3c",
    "teal": "#3d6b6b",
    "muted": "#7d8a99",
    "light": "#d7dde6",
    "sand": "#c9b79c",
}
SEGMENT_COLORS = {"premium": "#1f2d3d", "established": "#3d6b6b", "value": "#a9673f"}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "axes.edgecolor": PALETTE["ink"],
            "axes.linewidth": 0.7,
            "axes.grid": True,
            "grid.color": PALETTE["light"],
            "grid.linewidth": 0.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    sns.set_palette([PALETTE["ink"], PALETTE["accent"], PALETTE["teal"], PALETTE["muted"]])


def _save(figure: plt.Figure, name: str) -> str:
    path = FIGURES_DIR / name
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved figure %s", name)
    return name


def _dollar(value, _position=None) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.0f}k"
    return f"${value:.0f}"


def run_eda() -> dict:
    _style()
    frame = load_modeling_table()
    train = frame[frame[TARGET].notna()].copy()
    logger.info("EDA on %s labelled listings", len(train))

    stats: dict = {"n_train": len(train), "n_test": int(len(frame) - len(train))}
    _figure_price_distribution(train, stats)
    _figure_price_by_neighbourhood(train, stats)
    _figure_correlations(train, stats)
    _figure_area_price(train)
    _figure_amenity_price(train, stats)
    _figure_age_price(train, stats)
    _figure_missing(train, stats)
    _figure_outliers(train, stats)
    _figure_log_transform(train, stats)
    _figure_collinearity(train, stats)
    _figure_vif(train, stats)
    _figure_macro(stats)
    _figure_days_on_market(train, stats)
    _figure_segments(train, stats)
    _figure_property_type(train, stats)

    with open(METRICS_DIR / "eda_summary.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, default=float)
    logger.info("EDA summary written")
    return stats


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _figure_price_distribution(train: pd.DataFrame, stats: dict) -> None:
    price = train[TARGET].astype(float)
    log_price = np.log(price)
    stats["target"] = {
        "mean": float(price.mean()),
        "median": float(price.median()),
        "std": float(price.std()),
        "min": float(price.min()),
        "max": float(price.max()),
        "skew": float(price.skew()),
        "kurtosis": float(price.kurtosis()),
        "skew_log": float(log_price.skew()),
        "kurtosis_log": float(log_price.kurtosis()),
        "p01": float(price.quantile(0.01)),
        "p99": float(price.quantile(0.99)),
        "iqr_outliers": int(
            ((price < price.quantile(0.25) - 1.5 * (price.quantile(0.75) - price.quantile(0.25)))
             | (price > price.quantile(0.75) + 1.5 * (price.quantile(0.75) - price.quantile(0.25)))).sum()
        ),
    }

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    axes[0].hist(price, bins=45, color=PALETTE["teal"], edgecolor="white", linewidth=0.4)
    axes[0].axvline(
        price.mean(), color=PALETTE["accent"], linestyle="--", linewidth=1, label="Mean"
    )
    axes[0].axvline(
        price.median(), color=PALETTE["ink"], linestyle=":", linewidth=1, label="Median"
    )
    axes[0].legend(loc="upper right")
    axes[0].set_title("Sale price (USD)")
    axes[0].set_xlabel("Sale price")
    axes[0].set_ylabel("Listings")
    axes[0].xaxis.set_major_formatter(FuncFormatter(_dollar))
    axes[1].hist(log_price, bins=45, color=PALETTE["ink"], edgecolor="white", linewidth=0.4)
    axes[1].set_title("Natural logarithm of sale price")
    axes[1].set_xlabel("log(price)")
    axes[1].set_ylabel("Listings")
    axes[1].annotate(
        f"skew {price.skew():.2f} to {log_price.skew():.2f}",
        xy=(0.03, 0.93),
        xycoords="axes fraction",
        fontsize=7,
        color=PALETTE["muted"],
    )
    _save(figure, "fig01_price_distribution.png")


def _figure_price_by_neighbourhood(train: pd.DataFrame, stats: dict) -> None:
    table = (
        train.groupby("neighborhood", observed=True)
        .agg(
            listings=("id", "size"),
            median_price=(TARGET, "median"),
            mean_price=(TARGET, "mean"),
            median_ppsf=("price_per_sqm", "median"),
            amenity=("amenity_score", "mean"),
            tier=("neighborhood", lambda s: NEIGHBOURHOOD_TIERS.get(s.iloc[0], 3)),
        )
        .sort_values("median_price", ascending=False)
    )
    stats["neighbourhood"] = table.reset_index().to_dict(orient="records")

    order = table.index.tolist()
    figure, axis = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    sns.boxplot(
        data=train,
        y="neighborhood",
        x=TARGET,
        order=order,
        color=PALETTE["teal"],
        showfliers=False,
        linewidth=0.7,
        width=0.65,
        ax=axis,
    )
    axis.set_title("Sale price distribution by neighbourhood")
    axis.set_xlabel("Sale price")
    axis.set_ylabel("")
    axis.xaxis.set_major_formatter(FuncFormatter(_dollar))
    for tick, name in zip(axis.get_yticklabels(), order):
        tier = NEIGHBOURHOOD_TIERS.get(name, 3)
        tick.set_color(SEGMENT_COLORS["premium"] if tier >= 5 else PALETTE["ink"])
        tick.set_fontsize(7)
    _save(figure, "fig02_price_by_neighbourhood.png")


def _figure_correlations(train: pd.DataFrame, stats: dict) -> None:
    numeric = train.select_dtypes(include=[np.number])
    derived = [c for c in ("log_price", "price_per_sqm", "id") if c in numeric.columns]
    numeric = numeric.drop(columns=derived, errors="ignore")
    correlations = numeric.corr(numeric_only=True)[TARGET].drop(TARGET)
    top = correlations.abs().sort_values(ascending=False).head(15)
    selected = correlations.loc[top.index].sort_values()
    stats["top_correlations"] = {
        "positive": correlations.sort_values(ascending=False).head(10).round(3).to_dict(),
        "negative": correlations.sort_values().head(10).round(3).to_dict(),
    }

    figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    colors = [PALETTE["teal"] if value >= 0 else PALETTE["accent"] for value in selected]
    axis.barh(selected.index, selected.values, color=colors, height=0.62)
    axis.axvline(0, color=PALETTE["ink"], linewidth=0.7)
    axis.set_title("Pearson correlation with sale price (top 15 absolute values)")
    axis.set_xlabel("Correlation coefficient")
    axis.set_ylabel("")
    axis.tick_params(axis="y", labelsize=7)
    _save(figure, "fig03_price_drivers_correlation.png")


def _figure_area_price(train: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    scatter = axes[0].scatter(
        train["gr_liv_area"],
        train[TARGET],
        c=train["overall_qual"],
        cmap="viridis",
        s=7,
        alpha=0.75,
        linewidths=0,
    )
    axes[0].set_title("Price versus above-grade living area")
    axes[0].set_xlabel("Above-grade living area (sqft)")
    axes[0].set_ylabel("Sale price")
    axes[0].yaxis.set_major_formatter(FuncFormatter(_dollar))
    colorbar = figure.colorbar(scatter, ax=axes[0], shrink=0.85)
    colorbar.set_label("Overall quality", fontsize=7)
    colorbar.ax.tick_params(labelsize=6.5)

    sns.regplot(
        data=train,
        x=np.log(train["gr_liv_area"]),
        y=np.log(train[TARGET]),
        scatter_kws={"s": 6, "alpha": 0.45, "color": PALETTE["teal"], "linewidths": 0},
        line_kws={"color": PALETTE["accent"], "linewidth": 1.1},
        ax=axes[1],
    )
    axes[1].set_title("Log-log relationship with fitted line")
    axes[1].set_xlabel("log(living area)")
    axes[1].set_ylabel("log(price)")
    _save(figure, "fig04_area_price.png")


def _figure_amenity_price(train: pd.DataFrame, stats: dict) -> None:
    corr = float(train["amenity_score"].corr(train[TARGET], method="spearman"))
    stats["amenity_spearman"] = corr
    stats["amenity_by_tier"] = (
        train.assign(tier=train["neighborhood"].map(NEIGHBOURHOOD_TIERS))
        .groupby("tier", observed=True)["amenity_score"]
        .agg(["mean", "std", "count"])
        .round(2)
        .to_dict(orient="index")
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for segment, color in SEGMENT_COLORS.items():
        subset = train[train["neighbourhood_segment"] == segment]
        axes[0].scatter(
            subset["amenity_score"],
            subset[TARGET],
            s=8,
            alpha=0.6,
            color=color,
            label=segment.capitalize(),
            linewidths=0,
        )
    axes[0].set_title("Price versus neighbourhood amenity score")
    axes[0].set_xlabel("Amenity score (0-100)")
    axes[0].set_ylabel("Sale price")
    axes[0].yaxis.set_major_formatter(FuncFormatter(_dollar))
    axes[0].legend(title="Segment", title_fontsize=7)

    sns.boxplot(
        data=train,
        x="neighbourhood_segment",
        y=TARGET,
        order=["premium", "established", "value"],
        color=PALETTE["sand"],
        showfliers=False,
        linewidth=0.7,
        ax=axes[1],
    )
    axes[1].set_title("Price by broad sub-market")
    axes[1].set_ylabel("")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", labelsize=7.5)
    axes[1].yaxis.set_major_formatter(FuncFormatter(_dollar))
    _save(figure, "fig05_amenity_price.png")


def _figure_age_price(train: pd.DataFrame, stats: dict) -> None:
    decades = (train["year_built"] // 10 * 10).astype(int)
    table = (
        pd.DataFrame({"decade": decades, TARGET: train[TARGET], "area": train["gr_liv_area"]})
        .groupby("decade")
        .agg(median_price=(TARGET, "median"), listings=(TARGET, "size"), median_area=("area", "median"))
    )
    stats["price_by_decade"] = table.round(1).reset_index().to_dict(orient="records")

    figure, axis = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    positions = np.arange(len(table))
    bars = axis.bar(
        positions, table["median_price"], color=PALETTE["teal"], width=0.7
    )
    axis.set_xticks(positions)
    axis.set_xticklabels([str(value) for value in table.index])
    axis.set_title("Median sale price by construction decade")
    axis.set_xlabel("Decade built")
    axis.set_ylabel("Median sale price")
    axis.yaxis.set_major_formatter(FuncFormatter(_dollar))
    for bar, count in zip(bars, table["listings"]):
        axis.annotate(
            f"n={int(count)}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            fontsize=6.5,
            color=PALETTE["muted"],
        )
    _save(figure, "fig06_age_price.png")


def _figure_missing(train: pd.DataFrame, stats: dict) -> None:
    missing = train.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(18)
    stats["missing_after_cleaning"] = {str(k): int(v) for k, v in missing.items()}

    report_path = METRICS_DIR / "cleaning_report.json"
    before = {}
    if report_path.exists():
        with open(report_path, encoding="utf-8") as handle:
            before = json.load(handle)["report"]["missing_before"]
    before_series = pd.Series(before).sort_values(ascending=False).head(18)

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    axes[0].barh(before_series.index[::-1], before_series.values[::-1], color=PALETTE["accent"], height=0.6)
    axes[0].set_title("Missing cells before cleaning (top 18)")
    axes[0].set_xlabel("Count")
    axes[0].tick_params(axis="y", labelsize=6.5)
    if missing.empty:
        axes[1].text(0.5, 0.5, "No missing values remain", ha="center", va="center")
        axes[1].axis("off")
    else:
        axes[1].barh(missing.index[::-1], missing.values[::-1], color=PALETTE["teal"], height=0.6)
        axes[1].set_title("Missing cells after cleaning")
        axes[1].set_xlabel("Count")
        axes[1].tick_params(axis="y", labelsize=6.5)
    _save(figure, "fig07_missing_values.png")


def _figure_outliers(train: pd.DataFrame, stats: dict) -> None:
    flags = (train["gr_liv_area"] > 4000) & (train[TARGET] < 300_000)
    stats["outliers_removed_note"] = (
        "Two training records with more than 4,000 sqft sold below USD 300,000 were removed "
        "before modelling; they dominate the least-squares fit and are inconsistent with the "
        "local price-per-square-foot distribution."
    )
    figure, axis = plt.subplots(figsize=(6.6, 3.3), constrained_layout=True)
    axis.scatter(
        train.loc[~flags, "gr_liv_area"],
        train.loc[~flags, TARGET],
        s=7,
        color=PALETTE["muted"],
        alpha=0.55,
        linewidths=0,
        label="Retained",
    )
    axis.scatter(
        train.loc[flags, "gr_liv_area"],
        train.loc[flags, TARGET],
        s=40,
        facecolors="none",
        edgecolors=PALETTE["accent"],
        linewidths=1.1,
        label="Removed as implausible",
    )
    axis.set_title("Implausible area-price combinations")
    axis.set_xlabel("Above-grade living area (sqft)")
    axis.set_ylabel("Sale price")
    axis.yaxis.set_major_formatter(FuncFormatter(_dollar))
    axis.legend(loc="upper left")
    _save(figure, "fig08_outliers.png")


def _figure_log_transform(train: pd.DataFrame, stats: dict) -> None:
    from scipy import stats as scipy_stats

    price = train[TARGET].astype(float)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    scipy_stats.probplot(price, dist="norm", plot=axes[0])
    axes[0].set_title("Q-Q plot: raw price")
    axes[0].get_lines()[0].set_markersize(2)
    axes[0].get_lines()[0].set_color(PALETTE["muted"])
    axes[0].get_lines()[1].set_color(PALETTE["accent"])
    scipy_stats.probplot(np.log(price), dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q plot: log price")
    axes[1].get_lines()[0].set_markersize(2)
    axes[1].get_lines()[0].set_color(PALETTE["muted"])
    axes[1].get_lines()[1].set_color(PALETTE["teal"])
    stats["shapiro_log_p"] = float(scipy_stats.shapiro(np.log(price)).pvalue)
    _save(figure, "fig09_log_transform.png")


def _figure_collinearity(train: pd.DataFrame, stats: dict) -> None:
    candidates = [
        "gr_liv_area",
        "total_sf",
        "first_flr_sf",
        "second_flr_sf",
        "total_bsmt_sf",
        "garage_area",
        "garage_cars",
        "overall_qual",
        "year_built",
        "age_at_sale",
        "amenity_score",
        "dist_school_km",
        "dist_transit_km",
    ]
    matrix = train[candidates].corr().round(3)
    pairs = []
    for i, left in enumerate(candidates):
        for right in candidates[i + 1 :]:
            value = matrix.loc[left, right]
            if abs(value) >= 0.8:
                pairs.append({"left": left, "right": right, "correlation": float(value)})
    stats["high_correlation_pairs"] = pairs

    figure, axis = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    sns.heatmap(
        matrix,
        ax=axis,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 5.5},
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        square=True,
        linewidths=0.4,
        linecolor="white",
    )
    axis.set_title("Correlation structure of the main numeric attributes")
    axis.tick_params(labelsize=6.5)
    _save(figure, "fig10_multicollinearity.png")


def _figure_vif(train: pd.DataFrame, stats: dict) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = vif_table(train)
    table.to_csv(METRICS_DIR / "vif.csv", index=False)
    retained = table[table["retained"]].sort_values("vif_final", ascending=False).head(15)
    stats["vif"] = {
        "n_initial": len(table),
        "n_retained": int(table["retained"].sum()),
        "dropped": table[~table["retained"]][["feature", "vif_initial", "drop_reason"]]
        .head(10)
        .to_dict(orient="records"),
        "max_retained_vif": float(retained["vif_final"].max()) if len(retained) else None,
    }

    figure, axis = plt.subplots(figsize=(6.6, 3.4), constrained_layout=True)
    axis.barh(
        retained["feature"][::-1],
        retained["vif_final"][::-1],
        color=PALETTE["teal"],
        height=0.62,
    )
    axis.axvline(10, color=PALETTE["accent"], linestyle="--", linewidth=1)
    axis.set_xlim(0, max(12, float(retained["vif_final"].max()) * 1.15))
    axis.set_title("Variance inflation factors after iterative pruning")
    axis.set_xlabel("VIF (dashed line = threshold of 10)")
    axis.tick_params(axis="y", labelsize=6.5)
    _save(figure, "fig11_vif.png")


def _figure_macro(stats: dict) -> None:
    from revaluation.data.synthetic import build_macro_series

    macro = build_macro_series()
    macro["date"] = pd.to_datetime(macro["month"] + "-01")
    window = macro[macro["date"] <= "2010-12-01"]
    stats["macro"] = {
        "rate_min": float(window["mortgage_rate_pct"].min()),
        "rate_max": float(window["mortgage_rate_pct"].max()),
        "hpi_start": float(window["hpi_index"].iloc[0]),
        "hpi_end": float(window["hpi_index"].iloc[-1]),
        "hpi_trough": float(window["hpi_index"].min()),
        "supply_max": float(window["months_supply"].max()),
    }
    figure, axes = plt.subplots(3, 1, figsize=(6.6, 4.6), sharex=True, constrained_layout=True)
    series = [
        ("mortgage_rate_pct", "Mortgage rate (%)", PALETTE["accent"]),
        ("hpi_index", "House price index (2006-01 = 100)", PALETTE["ink"]),
        ("months_supply", "Months of supply", PALETTE["teal"]),
    ]
    for axis, (column, label, color) in zip(axes, series):
        axis.plot(window["date"], window[column], color=color, linewidth=1.1)
        axis.set_ylabel(label, fontsize=7.5)
    axes[0].set_title("Simulated local macro-financial conditions, 2006-2010")
    axes[-1].set_xlabel("Month")
    _save(figure, "fig12_macro_trends.png")


def _figure_days_on_market(train: pd.DataFrame, stats: dict) -> None:
    stats["dom"] = {
        "mean": float(train["days_on_market"].mean()),
        "median": float(train["days_on_market"].median()),
        "p90": float(train["days_on_market"].quantile(0.9)),
        "spearman_price": float(train["days_on_market"].corr(train[TARGET], method="spearman")),
    }
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    sns.boxplot(
        data=train,
        x=train["neighborhood"].map(NEIGHBOURHOOD_TIERS).rename("tier"),
        y=train["days_on_market"],
        color=PALETTE["sand"],
        showfliers=False,
        linewidth=0.7,
        ax=axes[0],
    )
    axes[0].set_title("Days on market by neighbourhood tier")
    axes[0].set_xlabel("Neighbourhood tier")
    axes[0].set_ylabel("Days on market")

    monthly = train.groupby("mo_sold")["days_on_market"].median()
    axes[1].plot(monthly.index, monthly.values, marker="o", markersize=3, color=PALETTE["ink"])
    axes[1].set_title("Median days on market by listing month")
    axes[1].set_xlabel("Month of sale")
    axes[1].set_ylabel("Days on market")
    axes[1].set_xticks(range(1, 13))
    _save(figure, "fig13_days_on_market.png")


def _figure_property_type(train: pd.DataFrame, stats: dict) -> None:
    """Price patterns across building types and architectural styles."""
    building = (
        train.groupby("bldg_type", observed=True)
        .agg(listings=("id", "size"), median_price=(TARGET, "median"), median_area=("gr_liv_area", "median"))
        .sort_values("median_price", ascending=False)
    )
    style = (
        train.groupby("house_style", observed=True)
        .agg(listings=("id", "size"), median_price=(TARGET, "median"))
        .sort_values("median_price", ascending=False)
    )
    stats["property_type"] = building.round(1).reset_index().to_dict(orient="records")
    stats["house_style"] = style.round(1).reset_index().to_dict(orient="records")

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    sns.boxplot(
        data=train,
        x="bldg_type",
        y=TARGET,
        order=building.index.tolist(),
        color=PALETTE["teal"],
        showfliers=False,
        linewidth=0.7,
        ax=axes[0],
    )
    axes[0].set_title("Price by building type")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Sale price")
    axes[0].yaxis.set_major_formatter(FuncFormatter(_dollar))
    axes[0].tick_params(axis="x", labelsize=6.5, rotation=20)

    sns.boxplot(
        data=train,
        x="house_style",
        y=TARGET,
        order=style.index.tolist(),
        color=PALETTE["sand"],
        showfliers=False,
        linewidth=0.7,
        ax=axes[1],
    )
    axes[1].set_title("Price by architectural style")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Sale price")
    axes[1].yaxis.set_major_formatter(FuncFormatter(_dollar))
    axes[1].tick_params(axis="x", labelsize=6.5, rotation=20)
    _save(figure, "fig25_price_by_property_type.png")


def _figure_segments(train: pd.DataFrame, stats: dict) -> None:
    train = train.assign(price_band=pd.cut(
        train[TARGET],
        bins=[0, 130_000, 200_000, 300_000, np.inf],
        labels=["Entry", "Mid", "Upper", "Luxury"],
    ))
    table = (
        train.groupby("price_band", observed=True)
        .agg(
            listings=("id", "size"),
            median_price=(TARGET, "median"),
            median_area=("gr_liv_area", "median"),
            median_amenity=("amenity_score", "median"),
        )
        .reset_index()
    )
    stats["price_bands"] = table.to_dict(orient="records")

    figure, axis = plt.subplots(figsize=(6.4, 3.0), constrained_layout=True)
    widths = [0.38, 0.38, 0.38, 0.38]
    positions = np.arange(len(table))
    axis.bar(
        positions - widths[0] / 2,
        table["median_area"],
        width=0.38,
        color=PALETTE["teal"],
        label="Median living area (sqft, left axis)",
    )
    twin = axis.twinx()
    twin.plot(
        positions,
        table["median_amenity"],
        color=PALETTE["accent"],
        marker="o",
        markersize=4,
        label="Median amenity score (right axis)",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(table["price_band"])
    axis.set_ylabel("Living area (sqft)")
    twin.set_ylabel("Amenity score")
    twin.grid(False)
    axis.set_title("Market segments: living area and amenity score")
    handles = axis.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = axis.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    axis.legend(handles, labels, loc="upper left", fontsize=6.8)
    _save(figure, "fig14_segments.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_eda()
