#  Climate indicator manager - a package for managing and building climate indicator dashboards.
#
#  scripts/sea_surface_temperature/plot_sst_gridded_diagnostics.py
#
#  OPTIONAL: plot spatial SST diagnostics from already prepared annual gridded
#  anomaly NetCDFs. This script is intentionally separate from the strict six
#  annual CSV workflow in scripts/data_management/build_sst_outputs.py.

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

try:  # Cartopy is preferred for map axes, but tests should not require it.
    import cartopy.crs as ccrs
except Exception:  # pragma: no cover - fallback depends on optional runtime.
    ccrs = None


DATASETS = {
    "ERSST-v6-gridded": {
        "pattern": "sst_ERSST-v6-gridded_annual_gridded_*_baseline_1991_2020.nc",
        "label": "ERSST-v6",
        "role": "primary_gridded_sst",
    },
    "HadSST4-gridded": {
        "pattern": "sst_HadSST4-gridded_annual_gridded_*_baseline_1991_2020.nc",
        "label": "HadSST4",
        "role": "primary_gridded_sst",
    },
    "CMA-GMST-ocean-sensitivity": {
        "pattern": "sst_CMA-GMST-ocean-sensitivity_annual_gridded_*_baseline_1991_2020.nc",
        "label": "CMA-GMST ocean sensitivity",
        "role": "sensitivity_gridded_gmst_ocean_only",
    },
}
REQUIRED_DATASETS = {"ERSST-v6-gridded", "HadSST4-gridded"}
PERIOD_MEAN_PERIODS = {
    "1850_1900": (1850, 1900),
    "1901_1950": (1901, 1950),
    "1951_1980": (1951, 1980),
    "1981_2010": (1981, 2010),
    "1991_2020": (1991, 2020),
    "2000_2025": (2000, 2025),
}
TREND_PERIODS = {
    "1850_2025": (1850, 2025),
    "1900_2025": (1900, 2025),
    "1950_2025": (1950, 2025),
    "1982_2025": (1982, 2025),
    "1991_2025": (1991, 2025),
}
DATA_VARIABLE = "sst_anomaly_C"


def managed_sst_root() -> Path:
    datadir = os.environ.get("DATADIR")
    if not datadir:
        raise RuntimeError(
            "DATADIR is not set. Set DATADIR before running gridded diagnostics."
        )
    return Path(datadir) / "ManagedData" / "SeaSurfaceTemperature"


def processed_gridded_dir() -> Path:
    return managed_sst_root() / "processed" / "gridded"


def gridded_figure_dir() -> Path:
    return managed_sst_root() / "Figures" / "gridded_diagnostics"


def gridded_table_dir() -> Path:
    return managed_sst_root() / "processed" / "gridded_diagnostics"


def discover_processed_gridded_files() -> dict[str, Path]:
    """Find prepared annual gridded files for approved primary/sensitivity datasets."""
    directory = processed_gridded_dir()
    discovered: dict[str, Path] = {}
    if not directory.exists():
        return discovered
    for dataset, config in DATASETS.items():
        matches = sorted(directory.glob(config["pattern"]))
        if matches:
            discovered[dataset] = matches[-1]
    return discovered


def read_annual_grid(path: Path) -> xr.DataArray:
    """Open a prepared annual gridded SST anomaly file with CF-aware decoding."""
    dataset = xr.open_dataset(path, decode_times=True)
    if DATA_VARIABLE not in dataset:
        dataset.close()
        raise RuntimeError(f"{path} does not contain {DATA_VARIABLE}.")
    data = dataset[DATA_VARIABLE].load()
    dataset.close()
    expected_dims = {"year", "lat", "lon"}
    if not expected_dims.issubset(set(data.dims)):
        raise RuntimeError(f"{path} must contain year, lat, and lon dimensions.")
    return data.transpose("year", "lat", "lon")


def select_period(data: xr.DataArray, start_year: int, end_year: int) -> xr.DataArray:
    selected = data.sel(year=slice(start_year, end_year))
    if selected.sizes.get("year", 0) == 0:
        raise RuntimeError(
            f"Requested period {start_year}-{end_year} does not overlap {data.name or 'grid'}."
        )
    return selected


def calculate_period_mean(data: xr.DataArray, start_year: int, end_year: int) -> xr.DataArray:
    selected = select_period(data, start_year, end_year)
    counts = selected.count("year")
    mean = selected.mean("year", skipna=True).where(counts > 0)
    mean.name = "period_mean_sst_anomaly_C"
    mean.attrs.update({"units": "degC", "period": f"{start_year}-{end_year}"})
    return mean


def calculate_valid_year_fraction(data: xr.DataArray, start_year: int, end_year: int) -> xr.DataArray:
    selected = select_period(data, start_year, end_year)
    fraction = selected.notnull().sum("year") / selected.sizes["year"]
    fraction.name = "valid_year_fraction"
    fraction.attrs.update({"units": "1", "period": f"{start_year}-{end_year}"})
    return fraction


def calculate_linear_trend(
    data: xr.DataArray,
    start_year: int,
    end_year: int,
    min_years: int = 10,
) -> xr.DataArray:
    """Calculate per-cell linear trend in degC per decade without filling gaps."""
    selected = select_period(data, start_year, end_year)
    years = xr.DataArray(
        selected["year"].values.astype(float),
        dims=("year",),
        coords={"year": selected["year"]},
    )
    valid = selected.notnull()
    count = valid.sum("year")
    years_valid = years.where(valid)
    data_valid = selected.where(valid)
    x_mean = years_valid.sum("year") / count
    y_mean = data_valid.sum("year") / count
    numerator = ((years - x_mean) * (selected - y_mean)).where(valid).sum("year")
    denominator = ((years - x_mean) ** 2).where(valid).sum("year")
    trend = (numerator / denominator) * 10.0
    trend = trend.where((count >= min_years) & np.isfinite(trend))
    trend.name = "linear_trend_sst_anomaly_C_per_decade"
    trend.attrs.update({"units": "degC per decade", "period": f"{start_year}-{end_year}"})
    return trend


def calculate_difference_field(
    reference: xr.DataArray,
    comparison: xr.DataArray,
    start_year: int,
    end_year: int,
) -> xr.DataArray:
    """Return reference minus comparison period means on the reference grid."""
    reference_mean = calculate_period_mean(reference, start_year, end_year)
    comparison_mean = calculate_period_mean(comparison, start_year, end_year)
    comparison_on_reference = comparison_mean
    for coord_name in ("lat", "lon"):
        if comparison_on_reference.sizes.get(coord_name, 0) < 2:
            comparison_on_reference = comparison_on_reference.reindex(
                {coord_name: reference_mean[coord_name]}
            )
        else:
            comparison_on_reference = comparison_on_reference.interp(
                {coord_name: reference_mean[coord_name]}
            )
    difference = reference_mean - comparison_on_reference
    difference.name = "ersst_minus_hadsst4_sst_anomaly_C"
    difference.attrs.update({"units": "degC", "period": f"{start_year}-{end_year}"})
    return difference


def cosine_weighted_spatial_mean(field: xr.DataArray) -> float:
    weights = np.cos(np.deg2rad(field["lat"]))
    weighted = field.weighted(weights)
    value = weighted.mean(("lat", "lon"), skipna=True).item()
    return float(value) if np.isfinite(value) else np.nan


def field_missing_fraction(field: xr.DataArray) -> float:
    return float(field.isnull().mean().item())


def _plot_field(
    field: xr.DataArray,
    title: str,
    path: Path,
    cmap: str,
    label: str,
    symmetric: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = field.to_numpy()
    if vmin is None or vmax is None:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            vmin, vmax = 0.0, 1.0
        elif symmetric:
            max_abs = float(np.nanmax(np.abs(finite)))
            max_abs = max(max_abs, 0.01)
            vmin, vmax = -max_abs, max_abs
        else:
            vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
            if vmin == vmax:
                vmin -= 0.01
                vmax += 0.01

    if ccrs is not None:
        fig, ax = plt.subplots(
            figsize=(11.5, 5.7),
            dpi=140,
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        mesh = ax.pcolormesh(
            field["lon"],
            field["lat"],
            field,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
        ax.coastlines(linewidth=0.4)
        ax.set_global()
    else:  # pragma: no cover - used only when cartopy is unavailable.
        fig, ax = plt.subplots(figsize=(11.5, 5.7), dpi=140)
        mesh = ax.pcolormesh(
            field["lon"],
            field["lat"],
            field,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    ax.set_title(title, loc="left", fontsize=13)
    colorbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.06, fraction=0.06)
    colorbar.set_label(label)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def write_inventory_table(grids: dict[str, xr.DataArray], files: dict[str, Path]) -> Path:
    rows = []
    for dataset, data in grids.items():
        rows.append(
            {
                "dataset": dataset,
                "dataset_role": DATASETS[dataset]["role"],
                "source_file": str(files[dataset]),
                "variable": DATA_VARIABLE,
                "year_start": int(data["year"].values.min()),
                "year_end": int(data["year"].values.max()),
                "n_year": int(data.sizes["year"]),
                "n_lat": int(data.sizes["lat"]),
                "n_lon": int(data.sizes["lon"]),
                "missing_fraction": field_missing_fraction(data),
                "baseline": data.attrs.get("baseline", "1991-2020"),
                "units": data.attrs.get("units", "degC"),
            }
        )
    table_dir = gridded_table_dir()
    table_dir.mkdir(parents=True, exist_ok=True)
    path = table_dir / "sst_gridded_diagnostic_inventory.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_summary_table(rows: list[dict[str, object]]) -> Path:
    table_dir = gridded_table_dir()
    table_dir.mkdir(parents=True, exist_ok=True)
    path = table_dir / "sst_gridded_diagnostic_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def run_diagnostics(strict: bool = False) -> dict[str, object]:
    files = discover_processed_gridded_files()
    missing = sorted(REQUIRED_DATASETS - set(files))
    if missing:
        message = "Missing prepared gridded input(s): " + ", ".join(missing)
        if strict:
            raise RuntimeError(message)
        return {"status": "missing_inputs", "missing": missing, "figures": [], "tables": []}

    grids = {dataset: read_annual_grid(path) for dataset, path in files.items()}
    figure_paths: list[Path] = []
    summary_rows: list[dict[str, object]] = []
    figure_dir = gridded_figure_dir()

    for dataset, data in grids.items():
        label = DATASETS[dataset]["label"]
        for period_name, (start_year, end_year) in PERIOD_MEAN_PERIODS.items():
            mean = calculate_period_mean(data, start_year, end_year)
            valid_fraction = calculate_valid_year_fraction(data, start_year, end_year)
            figure_paths.append(
                _plot_field(
                    mean,
                    f"{label} SST anomaly mean, {start_year}-{end_year}",
                    figure_dir / f"sst_gridded_mean_{dataset}_{period_name}.png",
                    cmap="RdBu_r",
                    label="SST anomaly (degC)",
                )
            )
            figure_paths.append(
                _plot_field(
                    valid_fraction,
                    f"{label} valid-year fraction, {start_year}-{end_year}",
                    figure_dir / f"sst_gridded_valid_fraction_{dataset}_{period_name}.png",
                    cmap="viridis",
                    label="Valid-year fraction",
                    symmetric=False,
                    vmin=0.0,
                    vmax=1.0,
                )
            )
            summary_rows.append(
                {
                    "dataset": dataset,
                    "dataset_role": DATASETS[dataset]["role"],
                    "diagnostic": "period_mean",
                    "period": f"{start_year}-{end_year}",
                    "spatial_mean": cosine_weighted_spatial_mean(mean),
                    "missing_fraction": field_missing_fraction(mean),
                    "output_figure": str(figure_paths[-2]),
                }
            )
            summary_rows.append(
                {
                    "dataset": dataset,
                    "dataset_role": DATASETS[dataset]["role"],
                    "diagnostic": "valid_year_fraction",
                    "period": f"{start_year}-{end_year}",
                    "spatial_mean": cosine_weighted_spatial_mean(valid_fraction),
                    "missing_fraction": field_missing_fraction(valid_fraction),
                    "output_figure": str(figure_paths[-1]),
                }
            )

        for period_name, (start_year, end_year) in TREND_PERIODS.items():
            trend = calculate_linear_trend(data, start_year, end_year)
            figure_paths.append(
                _plot_field(
                    trend,
                    f"{label} SST anomaly trend, {start_year}-{end_year}",
                    figure_dir / f"sst_gridded_trend_{dataset}_{period_name}.png",
                    cmap="RdBu_r",
                    label="Trend (degC per decade)",
                )
            )
            summary_rows.append(
                {
                    "dataset": dataset,
                    "dataset_role": DATASETS[dataset]["role"],
                    "diagnostic": "linear_trend",
                    "period": f"{start_year}-{end_year}",
                    "spatial_mean": cosine_weighted_spatial_mean(trend),
                    "missing_fraction": field_missing_fraction(trend),
                    "output_figure": str(figure_paths[-1]),
                }
            )

    ersst = grids["ERSST-v6-gridded"]
    hadsst = grids["HadSST4-gridded"]
    for period_name, (start_year, end_year) in PERIOD_MEAN_PERIODS.items():
        difference = calculate_difference_field(ersst, hadsst, start_year, end_year)
        figure_paths.append(
            _plot_field(
                difference,
                f"ERSST-v6 minus HadSST4 SST anomaly, {start_year}-{end_year}",
                figure_dir / f"sst_gridded_difference_ERSST-v6_minus_HadSST4_{period_name}.png",
                cmap="RdBu_r",
                label="ERSST-v6 - HadSST4 (degC)",
            )
        )
        summary_rows.append(
            {
                "dataset": "ERSST-v6-gridded_minus_HadSST4-gridded",
                "dataset_role": "primary_gridded_sst_difference",
                "diagnostic": "period_mean_difference",
                "period": f"{start_year}-{end_year}",
                "spatial_mean": cosine_weighted_spatial_mean(difference),
                "missing_fraction": field_missing_fraction(difference),
                "output_figure": str(figure_paths[-1]),
            }
        )

    table_paths = [write_inventory_table(grids, files), write_summary_table(summary_rows)]
    return {
        "status": "ok",
        "missing": [],
        "figures": [str(path) for path in figure_paths],
        "tables": [str(path) for path in table_paths],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if either prepared ERSST-v6 or HadSST4 annual gridded input is absent.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_diagnostics(strict=args.strict)
    if result["status"] != "ok":
        print(result["status"] + ": " + ", ".join(result.get("missing", [])))
        return 0
    for path in result["figures"]:
        print(f"Wrote {path}")
    for path in result["tables"]:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
