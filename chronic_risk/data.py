"""Download pinned public datasets into the git-ignored data folder."""

from __future__ import annotations

import hashlib
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from chronic_risk.schema import DataSource

TIMEOUT_SECONDS = 120
USER_AGENT = "chronic-disease-risk-prediction/1.0"


class DataDownloadError(RuntimeError):
    """A dataset could not be fetched or did not match its pinned checksum."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_source(source: DataSource, directory: Path) -> Path:
    """Return a verified local copy of ``source``, downloading it only when needed."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / source.filename
    if target.exists() and sha256_file(target) == source.sha256:
        return target

    if not source.url.startswith("https://"):
        raise DataDownloadError(f"Refusing non-HTTPS dataset URL: {source.url}")
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(source.url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    try:
        with (
            urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response,  # noqa: S310
            partial.open("wb") as out,
        ):
            shutil.copyfileobj(response, out)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise DataDownloadError(
            f"Could not download {source.url}: {exc}. Check your connection and try again."
        ) from exc

    actual = sha256_file(partial)
    if actual != source.sha256:
        partial.unlink(missing_ok=True)
        raise DataDownloadError(
            f"Checksum mismatch for {source.filename}: expected {source.sha256}, got {actual}. "
            "The published file may have changed; review it before updating the pinned checksum."
        )
    partial.replace(target)
    return target
