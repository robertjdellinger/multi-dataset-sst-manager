from __future__ import annotations

import time
from pathlib import Path

import requests


CHUNK_SIZE = 1024 * 1024
RANGE_CHUNK_SIZE = 8 * 1024 * 1024
SMALL_FILE_THRESHOLD = 2 * 1024 * 1024
USER_AGENT = "Mozilla/5.0"


def _headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {"User-agent": USER_AGENT, "Accept-Encoding": "identity"}
    if extra:
        headers.update(extra)
    return headers


def _total_size_from_content_range(value: str | None) -> int | None:
    if not value or "/" not in value:
        return None
    total = value.rsplit("/", 1)[-1]
    return int(total) if total.isdigit() else None


def _remote_metadata(url: str) -> tuple[int | None, bool]:
    try:
        range_response = requests.get(
            url,
            headers=_headers({"Range": "bytes=0-0"}),
            stream=True,
            timeout=(30, 180),
        )
        if range_response.status_code == 206:
            content_length = _total_size_from_content_range(range_response.headers.get("content-range"))
            range_response.close()
            return content_length, bool(content_length)
        length = range_response.headers.get("content-length")
        content_length = int(length) if length and length.isdigit() else None
        range_response.close()
        if range_response.status_code == 200:
            return content_length, False
    except requests.RequestException:
        pass

    try:
        response = requests.head(
            url,
            allow_redirects=True,
            headers=_headers(),
            timeout=(15, 60),
        )
    except requests.RequestException:
        return None, False
    if response.status_code >= 400:
        return None, False
    length = response.headers.get("content-length")
    content_length = int(length) if length and length.isdigit() else None
    accepts_ranges = response.headers.get("accept-ranges", "").lower() == "bytes"
    return content_length, accepts_ranges


def _download_range_chunk(url: str, handle, start: int, end: int, attempts: int) -> None:
    headers = _headers({"Range": f"bytes={start}-{end}"})
    expected = end - start + 1
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with requests.get(url, stream=True, headers=headers, timeout=(30, 180)) as response:
                if response.status_code != 206:
                    raise RuntimeError(f"range request returned HTTP {response.status_code}")
                written = 0
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        handle.write(chunk)
                        written += len(chunk)
                if written != expected:
                    raise RuntimeError(f"range {start}-{end} wrote {written} bytes, expected {expected}")
                return
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt < attempts:
                handle.seek(start)
                handle.truncate(start)
                time.sleep(float(attempt))
    raise RuntimeError(f"could not download range {start}-{end}: {last_error}")


def _download_by_ranges(url: str, tmp_path: Path, total_size: int, attempts: int) -> None:
    with tmp_path.open("wb") as handle:
        for start in range(0, total_size, RANGE_CHUNK_SIZE):
            end = min(start + RANGE_CHUNK_SIZE - 1, total_size - 1)
            _download_range_chunk(url, handle, start, end, attempts)


def _download_small_file(url: str, tmp_path: Path, expected_size: int | None, attempts: int) -> None:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, headers=_headers(), timeout=(30, 240))
            response.raise_for_status()
            content = response.content
            if expected_size and len(content) != expected_size:
                raise RuntimeError(
                    f"incomplete download for {url}: got {len(content)} bytes, expected {expected_size}"
                )
            tmp_path.write_bytes(content)
            return
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(float(attempt))
    raise RuntimeError(f"could not download small file: {last_error}")


def fetch_with_retries(url: str, outdir: Path, filename: str, attempts: int = 3) -> None:
    """Download a gridded SST file atomically and reject incomplete transfers."""
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / Path(filename).name
    tmp_path = out_path.with_name(f"{out_path.name}.part")
    expected_size, accepts_ranges = _remote_metadata(url)

    if expected_size and out_path.exists() and out_path.stat().st_size == expected_size:
        return

    if expected_size and expected_size <= SMALL_FILE_THRESHOLD:
        if tmp_path.exists():
            tmp_path.unlink()
        try:
            _download_small_file(url, tmp_path, expected_size, attempts)
            tmp_path.replace(out_path)
            return
        except RuntimeError as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"Could not download complete file from {url}: {exc}") from exc

    if accepts_ranges and expected_size:
        if tmp_path.exists():
            tmp_path.unlink()
        try:
            _download_by_ranges(url, tmp_path, expected_size, attempts)
            if tmp_path.stat().st_size != expected_size:
                raise RuntimeError(
                    f"incomplete download for {url}: got {tmp_path.stat().st_size} bytes, expected {expected_size}"
                )
            tmp_path.replace(out_path)
            return
        except RuntimeError as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"Could not download complete gridded SST file from {url}: {exc}") from exc

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        if tmp_path.exists():
            tmp_path.unlink()
        try:
            with requests.get(
                url,
                stream=True,
                headers=_headers(),
                timeout=(30, 300),
            ) as response:
                response.raise_for_status()
                response_size = response.headers.get("content-length")
                expected = int(response_size) if response_size and response_size.isdigit() else expected_size
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            handle.write(chunk)

            if expected and tmp_path.stat().st_size != expected:
                raise RuntimeError(
                    f"incomplete download for {url}: got {tmp_path.stat().st_size} bytes, expected {expected}"
                )
            tmp_path.replace(out_path)
            return
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(float(attempt))

    if tmp_path.exists():
        tmp_path.unlink()
    raise RuntimeError(f"Could not download complete gridded SST file from {url}: {last_error}")
