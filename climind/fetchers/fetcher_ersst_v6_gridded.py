from pathlib import Path

from climind.fetchers.fetcher_sst_gridded_url import fetch_with_retries


def fetch(url: str, outdir: Path, filename: str) -> None:
    """Fetch the NOAA PSL ERSSTv6 gridded monthly NetCDF file."""
    fetch_with_retries(url, outdir, filename)
