# 1850-2025 reference-style global SST figure from the harmonized
# annual 1991-2020-baselined SST table.
#
# Input:
#   $DATADIR/ManagedData/SeaSurfaceTemperature/processed/
#   merged_global_sst_reconstructions_annual_1850_2025_baseline_1991_2020.csv
#
# Output:
#   $DATADIR/ManagedData/SeaSurfaceTemperature/Figures/
#   global_sea_surface_temperature_1850_2025_reference_style.png
from __future__ import annotations

from itertools import combinations
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_LEVELS = ["CMA-SST", "CMEMS", "DCENT-I", "ERSSTv6", "HadSST4"]
HISTORICAL_DATASET_LEVELS = ["CMA-SST", "DCENT-I", "ERSSTv6", "HadSST4"]
DATASET_COLORS = {
    "CMA-SST": "#2259A6",
    "CMEMS": "#F39C12",
    "DCENT-I": "#6DBB45",
    "ERSSTv6": "#22A7C9",
    "HadSST4": "#009C9C",
}
DISPLAY_NAMES = {
    "CMA-SST": "CMA-SST",
    "CMEMS": "CMEMS",
    "DCENT-I": "DCENT-I",
    "ERSSTv6": "ERSST",
    "HadSST4": "HadSST4",
}
WIDE_DATASET_COLUMNS = {
    "CMA_SST": "CMA-SST",
    "CMEMS_SST": "CMEMS",
    "DCENT_SST_I": "DCENT-I",
    "ERSST_v6": "ERSSTv6",
    "HadSST4": "HadSST4",
}
DATASET_ALIASES = {
    "CMEMS-SST": "CMEMS",
    "DCENT-SST-I": "DCENT-I",
    "ERSST-v6": "ERSSTv6",
    "ERSST v6": "ERSSTv6",
}
OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
STRICT_SUMMARY_CSV = OUTPUT_TABLE_DIR / "sst_summary.csv"
STRICT_OUTPUT_FILES = {
    "CMA-SST": OUTPUT_TABLE_DIR / "sst_CMA_SST.csv",
    "CMEMS": OUTPUT_TABLE_DIR / "sst_CMEMS_SST.csv",
    "DCENT-I": OUTPUT_TABLE_DIR / "sst_DCENT_SST_I.csv",
    "ERSSTv6": OUTPUT_TABLE_DIR / "sst_ERSST_v6.csv",
    "HadSST4": OUTPUT_TABLE_DIR / "sst_HadSST4.csv",
}
DIAGNOSTIC_PERIODS = {
    "1850-2025": (1850, 2025),
    "1900-2025": (1900, 2025),
    "1950-2025": (1950, 2025),
    "1982-2025": (1982, 2025),
    "1991-2025": (1991, 2025),
}
PREINDUSTRIAL_BASELINE = (1850, 1900)
TARGET_BASELINE = (1991, 2020)
MODERN_OVERLAP_START = 1982


def get_sst_managed_dir() -> Path:
    datadir = os.getenv("DATADIR")
    if not datadir:
        raise RuntimeError(
            "DATADIR is not set. Set it before plotting, for example:\n"
            'export DATADIR="$HOME/data/multi-dataset-sst-manager"'
        )
    return Path(datadir) / "ManagedData" / "SeaSurfaceTemperature"


SST_MANAGED_DIR = get_sst_managed_dir()
INPUT_CSV = (
    SST_MANAGED_DIR
    / "processed"
    / "merged_global_sst_reconstructions_annual_1850_2025_baseline_1991_2020.csv"
)
OUTPUT_PNG = (
    SST_MANAGED_DIR
    / "Figures"
    / "global_sea_surface_temperature_1850_2025_reference_style.png"
)
DIAGNOSTIC_FIGURE_DIR = SST_MANAGED_DIR / "Figures" / "annual_diagnostics"
DIAGNOSTIC_TABLE_DIR = SST_MANAGED_DIR / "processed" / "annual_diagnostics"


def normalize_sst_plot_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "sst_anomaly" not in df.columns and "anomaly_sst_C" in df.columns:
        df = df.rename(columns={"anomaly_sst_C": "sst_anomaly"})

    long_required = {"dataset", "year", "sst_anomaly"}
    if long_required.issubset(df.columns):
        normalized = df[["dataset", "year", "sst_anomaly"]].copy()
        normalized["dataset"] = normalized["dataset"].replace(DATASET_ALIASES)
        return normalized

    wide_columns = list(WIDE_DATASET_COLUMNS)
    wide_required = {"year", *wide_columns}
    if wide_required.issubset(df.columns):
        normalized = df.melt(
            id_vars="year",
            value_vars=wide_columns,
            var_name="dataset",
            value_name="sst_anomaly",
        )
        normalized["dataset"] = normalized["dataset"].replace(WIDE_DATASET_COLUMNS)
        return normalized

    missing_long = sorted(long_required.difference(df.columns))
    missing_wide = sorted(wide_required.difference(df.columns))
    raise ValueError(
        "Plot input must contain either long columns "
        "`dataset`, `year`, `sst_anomaly` or the verified wide merged columns. "
        "Missing long columns: "
        + ", ".join(missing_long)
        + ". Missing wide columns: "
        + ", ".join(missing_wide)
        + "."
    )


def prepare_sst_plot_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing merged SST file: {path}\n"
            "Run the SST merge workflow before plotting."
        )
    df = normalize_sst_plot_frame(pd.read_csv(path))
    df = df.copy()
    df["dataset"] = pd.Categorical(
        df["dataset"],
        categories=DATASET_LEVELS,
        ordered=True,
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["sst_anomaly"] = pd.to_numeric(df["sst_anomaly"], errors="coerce")
    df = (
        df.dropna(subset=["dataset", "year", "sst_anomaly"])
        .assign(year=lambda x: x["year"].astype(int))
        .sort_values(["dataset", "year"])
        .reset_index(drop=True)
    )
    found = list(df["dataset"].dropna().astype(str).unique())
    missing_datasets = [name for name in DATASET_LEVELS if name not in found]
    if missing_datasets:
        raise ValueError(
            "The merged SST table is missing required dataset(s): "
            + ", ".join(missing_datasets)
        )
    return df[["dataset", "year", "sst_anomaly"]]


def parse_badc_csv(path: Path) -> pd.DataFrame:
    """Read a BADC-CSV table body, falling back to ordinary CSV if needed."""
    lines = path.read_text().splitlines()
    data_line = next((index for index, line in enumerate(lines) if line.strip() == "data"), None)
    if data_line is None:
        return pd.read_csv(path)
    return pd.read_csv(path, skiprows=data_line + 1)


def prepare_annual_reference_data() -> pd.DataFrame:
    """Load the strict annual reference table, preferring the six-output summary CSV."""
    if STRICT_SUMMARY_CSV.exists():
        return normalize_strict_summary_frame(parse_badc_csv(STRICT_SUMMARY_CSV))
    return prepare_sst_plot_data(INPUT_CSV)


def normalize_strict_summary_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert the required `sst_summary.csv` data block to long plotting form."""
    required = {"year", *WIDE_DATASET_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "Strict annual summary is missing required column(s): "
            + ", ".join(missing)
        )
    normalized = frame.melt(
        id_vars="year",
        value_vars=list(WIDE_DATASET_COLUMNS),
        var_name="dataset",
        value_name="sst_anomaly",
    )
    normalized["dataset"] = normalized["dataset"].replace(WIDE_DATASET_COLUMNS)
    normalized["dataset"] = pd.Categorical(
        normalized["dataset"],
        categories=DATASET_LEVELS,
        ordered=True,
    )
    normalized["year"] = pd.to_numeric(normalized["year"], errors="coerce").astype("Int64")
    normalized["sst_anomaly"] = pd.to_numeric(normalized["sst_anomaly"], errors="coerce")
    return (
        normalized.dropna(subset=["dataset", "year", "sst_anomaly"])
        .assign(year=lambda x: x["year"].astype(int))
        .sort_values(["dataset", "year"])
        .reset_index(drop=True)[["dataset", "year", "sst_anomaly"]]
    )


def pivot_annual_frame(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index="year",
        columns="dataset",
        values="sst_anomaly",
        aggfunc="first",
        observed=False,
    )
    pivot = pivot.reindex(columns=DATASET_LEVELS)
    return pivot.sort_index()


def period_complete_for_datasets(
    df: pd.DataFrame,
    datasets: list[str],
    start_year: int,
    end_year: int,
) -> bool:
    pivot = pivot_annual_frame(df)
    expected_years = set(range(start_year, end_year + 1))
    if not expected_years.issubset(set(pivot.index.astype(int))):
        return False
    period = pivot.loc[start_year:end_year, datasets]
    return bool(period.notna().all().all())


def rebaseline_annual_frame(
    df: pd.DataFrame,
    baseline_start: int,
    baseline_end: int,
    datasets: list[str] | None = None,
) -> pd.DataFrame:
    """Rebaseline long annual data by subtracting each dataset's period mean."""
    selected_datasets = datasets or DATASET_LEVELS
    subset = df[df["dataset"].astype(str).isin(selected_datasets)].copy()
    baseline = subset[
        subset["year"].between(baseline_start, baseline_end)
    ].groupby("dataset", observed=False)["sst_anomaly"].mean()
    missing = [
        dataset for dataset in selected_datasets
        if dataset not in baseline.index or pd.isna(baseline.loc[dataset])
    ]
    if missing:
        raise ValueError(
            f"Cannot rebaseline to {baseline_start}-{baseline_end}; "
            "missing baseline data for " + ", ".join(missing)
        )
    subset["sst_anomaly"] = subset.apply(
        lambda row: row["sst_anomaly"] - baseline.loc[row["dataset"]],
        axis=1,
    )
    subset["dataset"] = pd.Categorical(
        subset["dataset"],
        categories=selected_datasets,
        ordered=True,
    )
    return subset.sort_values(["dataset", "year"]).reset_index(drop=True)


def calculate_availability_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASET_LEVELS:
        sub = df[df["dataset"].astype(str) == dataset].dropna(subset=["sst_anomaly"])
        years = sub["year"].astype(int)
        if sub.empty:
            rows.append(
                {
                    "dataset": dataset,
                    "first_year": np.nan,
                    "last_year": np.nan,
                    "valid_years": 0,
                    "has_1991_2020_baseline": False,
                    "has_1850_1900_baseline": False,
                    "contributes_to_six_csv_contract": True,
                }
            )
            continue
        year_set = set(years)
        rows.append(
            {
                "dataset": dataset,
                "first_year": int(years.min()),
                "last_year": int(years.max()),
                "valid_years": int(years.nunique()),
                "has_1991_2020_baseline": set(range(1991, 2021)).issubset(year_set),
                "has_1850_1900_baseline": set(range(1850, 1901)).issubset(year_set),
                "contributes_to_six_csv_contract": True,
            }
        )
    return pd.DataFrame(rows)


def calculate_baseline_offsets(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for baseline_name, (start_year, end_year) in {
        "1991-2020": TARGET_BASELINE,
        "1850-1900": PREINDUSTRIAL_BASELINE,
    }.items():
        for dataset in DATASET_LEVELS:
            sub = df[
                (df["dataset"].astype(str) == dataset)
                & df["year"].between(start_year, end_year)
            ].dropna(subset=["sst_anomaly"])
            rows.append(
                {
                    "dataset": dataset,
                    "baseline_period": baseline_name,
                    "baseline_start": start_year,
                    "baseline_end": end_year,
                    "available_years": int(sub["year"].nunique()),
                    "expected_years": end_year - start_year + 1,
                    "baseline_mean_C": float(sub["sst_anomaly"].mean()) if not sub.empty else np.nan,
                    "baseline_complete": int(sub["year"].nunique()) == end_year - start_year + 1,
                }
            )
    return pd.DataFrame(rows)


def calculate_pairwise_metrics(df: pd.DataFrame) -> pd.DataFrame:
    pivot = pivot_annual_frame(df)
    rows = []
    for left, right in combinations(DATASET_LEVELS, 2):
        pair = pivot[[left, right]].dropna()
        if pair.empty:
            rows.append(
                {
                    "dataset_a": left,
                    "dataset_b": right,
                    "year_start": np.nan,
                    "year_end": np.nan,
                    "n_years": 0,
                    "mean_residual_a_minus_b_C": np.nan,
                    "rmse_C": np.nan,
                    "mae_C": np.nan,
                    "correlation": np.nan,
                }
            )
            continue
        residual = pair[left] - pair[right]
        rows.append(
            {
                "dataset_a": left,
                "dataset_b": right,
                "year_start": int(pair.index.min()),
                "year_end": int(pair.index.max()),
                "n_years": int(len(pair)),
                "mean_residual_a_minus_b_C": float(residual.mean()),
                "rmse_C": float(np.sqrt(np.mean(np.square(residual)))),
                "mae_C": float(np.mean(np.abs(residual))),
                "correlation": float(pair[left].corr(pair[right])),
            }
        )
    return pd.DataFrame(rows)


def calculate_pairwise_residuals(df: pd.DataFrame) -> pd.DataFrame:
    pivot = pivot_annual_frame(df)
    rows = []
    for left, right in combinations(DATASET_LEVELS, 2):
        pair = pivot[[left, right]].dropna()
        for year, values in pair.iterrows():
            rows.append(
                {
                    "pair": f"{left} - {right}",
                    "dataset_a": left,
                    "dataset_b": right,
                    "year": int(year),
                    "residual_C": float(values[left] - values[right]),
                }
            )
    return pd.DataFrame(rows)


def calculate_source_spread(df: pd.DataFrame) -> pd.DataFrame:
    pivot = pivot_annual_frame(df)
    spread = pd.DataFrame(index=pivot.index)
    spread["year"] = spread.index.astype(int)
    spread["source_count"] = pivot.notna().sum(axis=1).astype(int)
    spread["source_mean_C"] = pivot.mean(axis=1, skipna=True)
    spread["source_min_C"] = pivot.min(axis=1, skipna=True)
    spread["source_max_C"] = pivot.max(axis=1, skipna=True)
    spread["source_spread_C"] = spread["source_max_C"] - spread["source_min_C"]
    spread["source_std_C"] = pivot.std(axis=1, skipna=True)
    return spread.reset_index(drop=True)


def calculate_rolling_differences(df: pd.DataFrame, windows: tuple[int, ...] = (10, 30)) -> pd.DataFrame:
    pivot = pivot_annual_frame(df)
    annual_mean = pivot.mean(axis=1, skipna=True)
    rows = []
    for window in windows:
        residuals = pivot.subtract(annual_mean, axis=0)
        rolled = residuals.rolling(window=window, min_periods=window).mean()
        for dataset in DATASET_LEVELS:
            sub = rolled[dataset].dropna()
            for year, value in sub.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "window_years": window,
                        "year": int(year),
                        "rolling_difference_from_source_mean_C": float(value),
                    }
                )
    return pd.DataFrame(rows)


def calculate_trend_estimates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period_name, (start_year, end_year) in DIAGNOSTIC_PERIODS.items():
        for dataset in DATASET_LEVELS:
            sub = df[
                (df["dataset"].astype(str) == dataset)
                & df["year"].between(start_year, end_year)
            ].dropna(subset=["sst_anomaly"])
            if len(sub) < 2:
                slope_decade = np.nan
            else:
                slope_year = np.polyfit(
                    sub["year"].to_numpy(dtype=float),
                    sub["sst_anomaly"].to_numpy(dtype=float),
                    deg=1,
                )[0]
                slope_decade = slope_year * 10.0
            rows.append(
                {
                    "dataset": dataset,
                    "period": period_name,
                    "period_start": start_year,
                    "period_end": end_year,
                    "n_years": int(sub["year"].nunique()),
                    "trend_C_per_decade": float(slope_decade) if np.isfinite(slope_decade) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def write_diagnostic_table(frame: pd.DataFrame, filename: str) -> Path:
    DIAGNOSTIC_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = DIAGNOSTIC_TABLE_DIR / filename
    frame.to_csv(path, index=False)
    return path


def save_diagnostic_figure(fig: plt.Figure, filename: str) -> Path:
    DIAGNOSTIC_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = DIAGNOSTIC_FIGURE_DIR / filename
    fig.savefig(path, dpi=160, facecolor="white", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    return path


def dynamic_y_limits(values: pd.Series, pad: float = 0.08) -> tuple[float, float]:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return (-1.0, 1.0)
    lower = float(finite.min()) - pad
    upper = float(finite.max()) + pad
    if lower == upper:
        lower -= 0.1
        upper += 0.1
    return lower, upper


def make_dataset_labels(df: pd.DataFrame) -> dict[str, str]:
    labels: dict[str, str] = {}
    for dataset in DATASET_LEVELS:
        sub = df[df["dataset"].astype(str) == dataset]
        if sub.empty:
            continue
        first_year = int(sub["year"].min())
        last_year = int(sub["year"].max())
        labels[dataset] = f"{DISPLAY_NAMES[dataset]} ({first_year}-{last_year})"
    return labels


def set_reference_style(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.figure.set_facecolor("white")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#666666")
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="x", colors="#666666", labelsize=20, width=0.6)
    ax.tick_params(axis="y", colors="#666666", labelsize=20, length=0)
    ax.xaxis.label.set_color("#666666")
    ax.yaxis.label.set_color("#666666")


def plot_global_sst_reconstructions(
    sst_merged: pd.DataFrame,
    title_text: str = "Global sea-surface temperature 1850-2025",
    subtitle_text: str = "Difference from 1991-2020 average",
    uncertainty_half_width: float = 0.04,
    y_limits: tuple[float, float] = (-1.12, 0.60),
    y_breaks: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5),
) -> plt.Figure:
    df = sst_merged.copy()
    labels = make_dataset_labels(df)
    # Match the supplied reference image geometry: 13.7 x 9.0 inches at 100 dpi.
    fig, ax = plt.subplots(figsize=(13.7, 9.0), dpi=100)
    for dataset in DATASET_LEVELS:
        sub = df[df["dataset"].astype(str) == dataset].sort_values("year")
        if sub.empty:
            continue
        x = sub["year"].to_numpy(dtype=float)
        y = sub["sst_anomaly"].to_numpy(dtype=float)
        color = DATASET_COLORS[dataset]
        # Fixed visual envelope, matching the R reference-style ribbon.
        ax.fill_between(
            x,
            y - uncertainty_half_width,
            y + uncertainty_half_width,
            color=color,
            alpha=0.22,
            linewidth=0,
        )
        ax.plot(
            x,
            y,
            color=color,
            linewidth=1.25,
            alpha=0.98,
            label=labels.get(dataset, dataset),
        )
    ax.set_xlim(1842, 2026)
    ax.set_xticks(range(1860, 2021, 20))
    ax.set_ylim(*y_limits)
    ax.set_yticks(y_breaks)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.set_xlabel("Year", fontsize=20, labelpad=2)
    ax.set_ylabel("°C", fontsize=20, labelpad=18)
    set_reference_style(ax)
    fig.suptitle(
        title_text,
        x=0.087,
        y=0.975,
        ha="left",
        va="top",
        fontsize=38,
        fontweight="normal",
        color="#555555",
    )
    fig.text(
        0.087,
        0.915,
        subtitle_text,
        ha="left",
        va="top",
        fontsize=26,
        color="#666666",
    )
    for index, dataset in enumerate(DATASET_LEVELS):
        ax.text(
            0.047,
            0.985 - index * 0.057,
            labels.get(dataset, dataset),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=18,
            color=DATASET_COLORS[dataset],
        )
    # Plot margin tuned to mimic the uploaded R ggplot output.
    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        top=0.835,
        bottom=0.11,
    )
    return fig


def plot_source_coverage(df: pd.DataFrame) -> plt.Figure:
    spread = calculate_source_spread(df)
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=120)
    ax.step(
        spread["year"],
        spread["source_count"],
        where="mid",
        color="#444444",
        linewidth=2.0,
    )
    ax.set_title("Annual SST source coverage", fontsize=18, loc="left", color="#444444")
    ax.set_ylabel("Available datasets", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylim(0, len(DATASET_LEVELS) + 0.5)
    ax.set_yticks(range(0, len(DATASET_LEVELS) + 1))
    set_reference_style(ax)
    return fig


def plot_baseline_sensitivity(offsets: pd.DataFrame) -> plt.Figure:
    sub = offsets[
        (offsets["baseline_period"] == "1850-1900")
        & offsets["dataset"].isin(HISTORICAL_DATASET_LEVELS)
    ].copy()
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=120)
    colors = [DATASET_COLORS[dataset] for dataset in sub["dataset"]]
    ax.bar(sub["dataset"], sub["baseline_mean_C"], color=colors, alpha=0.85)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Baseline sensitivity: 1850-1900 offset", fontsize=18, loc="left", color="#444444")
    ax.set_ylabel("1850-1900 mean on 1991-2020 baseline (degC)", fontsize=12)
    ax.set_xlabel("")
    set_reference_style(ax)
    return fig


def plot_pairwise_residual_heatmap(residuals: pd.DataFrame) -> plt.Figure:
    pairs = sorted(residuals["pair"].unique())
    years = sorted(residuals["year"].unique())
    matrix = np.full((len(pairs), len(years)), np.nan)
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    year_index = {year: index for index, year in enumerate(years)}
    for row in residuals.itertuples(index=False):
        matrix[pair_index[row.pair], year_index[row.year]] = row.residual_C

    fig, ax = plt.subplots(figsize=(13.5, 7.2), dpi=120)
    max_abs = np.nanmax(np.abs(matrix))
    max_abs = max(float(max_abs), 0.05)
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-max_abs,
        vmax=max_abs,
        extent=[min(years) - 0.5, max(years) + 0.5, len(pairs) - 0.5, -0.5],
    )
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs, fontsize=9)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_title("Pairwise annual SST residuals", fontsize=18, loc="left", color="#444444")
    colorbar = fig.colorbar(image, ax=ax, pad=0.012)
    colorbar.set_label("Dataset A - dataset B (degC)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def plot_period_trend_comparison(trends: pd.DataFrame) -> plt.Figure:
    pivot = trends.pivot(index="period", columns="dataset", values="trend_C_per_decade")
    pivot = pivot.reindex(index=list(DIAGNOSTIC_PERIODS), columns=DATASET_LEVELS)
    fig, ax = plt.subplots(figsize=(12.5, 6.8), dpi=120)
    x = np.arange(len(pivot.index))
    width = 0.15
    offsets = np.linspace(-2, 2, len(DATASET_LEVELS)) * width
    for offset, dataset in zip(offsets, DATASET_LEVELS):
        ax.bar(
            x + offset,
            pivot[dataset].to_numpy(dtype=float),
            width=width,
            label=DISPLAY_NAMES[dataset],
            color=DATASET_COLORS[dataset],
            alpha=0.88,
        )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=0)
    ax.set_ylabel("Trend (degC per decade)", fontsize=12)
    ax.set_title("Annual SST trend comparison by period", fontsize=18, loc="left", color="#444444")
    ax.legend(frameon=False, ncol=3, fontsize=10)
    set_reference_style(ax)
    return fig


def plot_modern_overlap(df: pd.DataFrame) -> plt.Figure:
    sub = df[df["year"] >= MODERN_OVERLAP_START].copy()
    fig = plot_global_sst_reconstructions(
        sub,
        title_text=f"Global sea-surface temperature {MODERN_OVERLAP_START}-2025",
        subtitle_text="Modern overlap period; difference from 1991-2020 average",
        uncertainty_half_width=0.0,
        y_limits=dynamic_y_limits(sub["sst_anomaly"], pad=0.1),
        y_breaks=tuple(np.arange(-0.4, 0.8, 0.2)),
    )
    return fig


def plot_preindustrial_sensitivity(df: pd.DataFrame) -> plt.Figure:
    preindustrial = rebaseline_annual_frame(
        df,
        baseline_start=PREINDUSTRIAL_BASELINE[0],
        baseline_end=PREINDUSTRIAL_BASELINE[1],
        datasets=HISTORICAL_DATASET_LEVELS,
    )
    fig = plot_global_sst_reconstructions(
        preindustrial,
        title_text="Global sea-surface temperature 1850-2025",
        subtitle_text="Difference from 1850-1900 average; CMEMS excluded",
        uncertainty_half_width=0.0,
        y_limits=dynamic_y_limits(preindustrial["sst_anomaly"], pad=0.12),
        y_breaks=tuple(np.arange(-0.4, 1.4, 0.2)),
    )
    return fig


def plot_rolling_differences(rolling: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 8.2), dpi=120, sharex=True)
    for ax, window in zip(axes, (10, 30)):
        sub = rolling[rolling["window_years"] == window]
        for dataset in DATASET_LEVELS:
            dataset_sub = sub[sub["dataset"] == dataset]
            if dataset_sub.empty:
                continue
            ax.plot(
                dataset_sub["year"],
                dataset_sub["rolling_difference_from_source_mean_C"],
                color=DATASET_COLORS[dataset],
                linewidth=1.4,
                label=DISPLAY_NAMES[dataset],
            )
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_ylabel(f"{window}-year diff. (degC)", fontsize=11)
        ax.set_title(
            f"{window}-year rolling difference from available-source mean",
            fontsize=14,
            loc="left",
            color="#444444",
        )
        set_reference_style(ax)
    axes[-1].set_xlabel("Year", fontsize=12)
    axes[0].legend(frameon=False, ncol=3, fontsize=10)
    fig.tight_layout()
    return fig


def plot_source_spread(df: pd.DataFrame) -> plt.Figure:
    spread = calculate_source_spread(df)
    fig, ax = plt.subplots(figsize=(12.6, 6.5), dpi=120)
    ax.fill_between(
        spread["year"].to_numpy(dtype=float),
        spread["source_min_C"].to_numpy(dtype=float),
        spread["source_max_C"].to_numpy(dtype=float),
        color="#A9B7C7",
        alpha=0.45,
        linewidth=0,
        label="Source min-max range",
    )
    ax.plot(
        spread["year"],
        spread["source_mean_C"],
        color="#333333",
        linewidth=1.8,
        label="Available-source mean",
    )
    ax.set_title("Global annual SST source spread", fontsize=18, loc="left", color="#444444")
    ax.set_ylabel("SST anomaly (degC, 1991-2020 baseline)", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)
    ax.legend(frameon=False, loc="upper left")
    set_reference_style(ax)
    return fig


def write_annual_diagnostic_suite(df: pd.DataFrame) -> tuple[list[Path], list[Path]]:
    """Write annual diagnostic tables and figures from the strict annual CSV data."""
    table_paths: list[Path] = []
    figure_paths: list[Path] = []

    availability = calculate_availability_table(df)
    offsets = calculate_baseline_offsets(df)
    pairwise_metrics = calculate_pairwise_metrics(df)
    pairwise_residuals = calculate_pairwise_residuals(df)
    spread = calculate_source_spread(df)
    rolling = calculate_rolling_differences(df)
    trends = calculate_trend_estimates(df)

    table_paths.append(write_diagnostic_table(availability, "sst_annual_dataset_availability.csv"))
    table_paths.append(write_diagnostic_table(offsets, "sst_annual_baseline_offsets.csv"))
    table_paths.append(write_diagnostic_table(pairwise_metrics, "sst_annual_pairwise_metrics.csv"))
    table_paths.append(write_diagnostic_table(pairwise_residuals, "sst_annual_pairwise_residuals.csv"))
    table_paths.append(write_diagnostic_table(spread, "sst_annual_source_spread.csv"))
    table_paths.append(write_diagnostic_table(rolling, "sst_annual_rolling_differences.csv"))
    table_paths.append(write_diagnostic_table(trends, "sst_annual_period_trends.csv"))

    figure_paths.append(save_diagnostic_figure(plot_source_coverage(df), "sst_annual_source_coverage.png"))
    figure_paths.append(save_diagnostic_figure(plot_baseline_sensitivity(offsets), "sst_annual_baseline_sensitivity.png"))
    figure_paths.append(save_diagnostic_figure(plot_pairwise_residual_heatmap(pairwise_residuals), "sst_annual_pairwise_residual_heatmap.png"))
    figure_paths.append(save_diagnostic_figure(plot_period_trend_comparison(trends), "sst_annual_period_trend_comparison.png"))
    figure_paths.append(save_diagnostic_figure(plot_modern_overlap(df), "sst_annual_modern_overlap_1982_present.png"))
    figure_paths.append(save_diagnostic_figure(plot_preindustrial_sensitivity(df), "sst_annual_preindustrial_sensitivity_1850_1900.png"))
    figure_paths.append(save_diagnostic_figure(plot_rolling_differences(rolling), "sst_annual_rolling_differences.png"))
    figure_paths.append(save_diagnostic_figure(plot_source_spread(df), "sst_annual_source_spread.png"))

    return figure_paths, table_paths


def main() -> None:
    sst_merged = prepare_annual_reference_data()
    fig = plot_global_sst_reconstructions(
        sst_merged=sst_merged,
        title_text="Global sea-surface temperature 1850-2025",
        subtitle_text="Difference from 1991-2020 average",
        uncertainty_half_width=0.04,
        y_limits=(-1.12, 0.60),
        y_breaks=(-1.0, -0.5, 0.0, 0.5),
    )
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_PNG,
        dpi=100,
        facecolor="white",
        edgecolor="none",
        bbox_inches=None,
    )
    plt.close(fig)
    print(f"Wrote {OUTPUT_PNG}")
    figure_paths, table_paths = write_annual_diagnostic_suite(sst_merged)
    for path in figure_paths:
        print(f"Wrote {path}")
    for path in table_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
