from pathlib import Path

from climind.fetchers.fetcher_sst_gridded_url import fetch_with_retries


def fetch(url: str, outdir: Path, filename: str) -> None:
    """Fetch the Met Office HadSST4 gridded median anomaly NetCDF file."""
    fetch_with_retries(url, outdir, filename)
