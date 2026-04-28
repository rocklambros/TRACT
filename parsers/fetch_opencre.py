"""Fetch all CRE data from the OpenCRE REST API.

Downloads every page of CREs from ``/rest/v1/all_cres``, saves each page
individually to ``data/raw/opencre/pages/`` for resumability, then merges
into ``data/raw/opencre/opencre_all_cres.json``.

API response envelope (per page)::

    {
        "data": [...],       # list of CRE objects
        "page": N,           # current page (1-indexed)
        "total_pages": N     # total number of pages
    }

Pages beyond ``total_pages`` return HTTP 404 (no content).

Usage::

    python parsers/fetch_opencre.py
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import requests

from tract.config import (
    OPENCRE_API_BASE_URL,
    OPENCRE_PER_PAGE,
    OPENCRE_REQUEST_DELAY_S,
    OPENCRE_REQUEST_TIMEOUT_S,
    OPENCRE_RETRY_BACKOFF_FACTOR,
    OPENCRE_RETRY_INITIAL_DELAY_S,
    OPENCRE_RETRY_MAX_ATTEMPTS,
    OPENCRE_RETRY_MAX_DELAY_S,
    RAW_OPENCRE_DIR,
)
from tract.io import atomic_write_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

USER_AGENT: str = "TRACT/0.1.0 (security-framework-crosswalk research)"


class PageEnvelope(TypedDict):
    """Structure returned by the OpenCRE paginated API."""

    data: list[Any]
    page: int
    total_pages: int


def fetch_page(page: int, session: requests.Session) -> PageEnvelope:
    """Fetch a single page of CREs from the OpenCRE API with exponential backoff.

    Args:
        page: 1-indexed page number to retrieve.
        session: Active ``requests.Session`` to reuse connections.

    Returns:
        A ``PageEnvelope`` dict with ``data``, ``page``, and ``total_pages``.

    Raises:
        ValueError: If the API returns a body that does not match the expected
            envelope structure.
        requests.RequestException: If all retry attempts are exhausted.
    """
    url = OPENCRE_API_BASE_URL
    params: dict[str, int] = {"per_page": OPENCRE_PER_PAGE, "page": page}

    delay: float = OPENCRE_RETRY_INITIAL_DELAY_S
    last_exc: Exception | None = None

    for attempt in range(1, OPENCRE_RETRY_MAX_ATTEMPTS + 1):
        try:
            logger.debug("GET %s params=%s attempt=%d", url, params, attempt)
            response = session.get(url, params=params, timeout=OPENCRE_REQUEST_TIMEOUT_S)
            response.raise_for_status()
            body: Any = response.json()
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == OPENCRE_RETRY_MAX_ATTEMPTS:
                break
            logger.warning(
                "Page %d attempt %d/%d failed: %s — retrying in %.1fs",
                page,
                attempt,
                OPENCRE_RETRY_MAX_ATTEMPTS,
                exc,
                delay,
            )
            time.sleep(delay)
            delay = min(delay * OPENCRE_RETRY_BACKOFF_FACTOR, OPENCRE_RETRY_MAX_DELAY_S)
            continue

        # Schema validation — not a transient error, raise immediately.
        if not isinstance(body, dict):
            raise ValueError(
                f"Expected JSON object (dict) from page {page}, got {type(body).__name__}"
            )
        if "data" not in body or "total_pages" not in body:
            raise ValueError(
                f"Page {page} response missing 'data' or 'total_pages' keys: {list(body.keys())}"
            )
        if not isinstance(body["data"], list):
            raise ValueError(
                f"Page {page} 'data' field is not a list: {type(body['data']).__name__}"
            )
        return PageEnvelope(
            data=body["data"],
            page=body.get("page", page),
            total_pages=int(body["total_pages"]),
        )

    raise requests.RequestException(
        f"All {OPENCRE_RETRY_MAX_ATTEMPTS} attempts failed for page {page}"
    ) from last_exc


def _page_cache_path(pages_dir: Path, page: int) -> Path:
    """Return the cache file path for *page* inside *pages_dir*."""
    return pages_dir / f"page_{page:03d}.json"


def fetch_all_cres() -> None:
    """Fetch all CRE pages and merge into a single output file.

    Creates ``data/raw/opencre/pages/`` for per-page cache files and writes the
    merged result to ``data/raw/opencre/opencre_all_cres.json``.

    Resumable: if a page cache file already exists it is loaded from disk and
    the API is not called for that page.

    Raises:
        requests.RequestException: If a page fetch fails after all retries.
        ValueError: If a page response or cached file has an unexpected structure.
    """
    pages_dir: Path = RAW_OPENCRE_DIR / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    all_cres: list[Any] = []
    total_pages_known: int | None = None
    page: int = 1

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    try:
        while True:
            # Stop if we have fetched all known pages.
            if total_pages_known is not None and page > total_pages_known:
                logger.info("Reached last page (%d) — stopping.", total_pages_known)
                break

            cache_path = _page_cache_path(pages_dir, page)

            if cache_path.exists():
                logger.info("Page %d: loading from cache %s", page, cache_path)
                with open(cache_path, encoding="utf-8") as fh:
                    cached: Any = json.load(fh)
                if not isinstance(cached, dict) or "data" not in cached:
                    raise ValueError(
                        f"Cached page file has unexpected structure: {cache_path}"
                    )
                envelope = PageEnvelope(
                    data=cached["data"],
                    page=cached.get("page", page),
                    total_pages=int(cached["total_pages"]),
                )
            else:
                logger.info("Page %d: fetching from API", page)
                envelope = fetch_page(page, session)
                atomic_write_json(
                    {
                        "data": envelope["data"],
                        "page": envelope["page"],
                        "total_pages": envelope["total_pages"],
                    },
                    cache_path,
                )
                time.sleep(OPENCRE_REQUEST_DELAY_S)

            # Update total_pages from the first response we get.
            if total_pages_known is None:
                total_pages_known = envelope["total_pages"]
                logger.info("API reports %d total pages.", total_pages_known)

            cres_on_page = envelope["data"]
            all_cres.extend(cres_on_page)
            logger.info(
                "Page %d: %d CREs (running total: %d)", page, len(cres_on_page), len(all_cres)
            )
            page += 1
    finally:
        session.close()

    output: dict[str, Any] = {
        "cres": all_cres,
        "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_cres": len(all_cres),
        "total_pages": total_pages_known if total_pages_known is not None else page - 1,
    }

    output_path: Path = RAW_OPENCRE_DIR / "opencre_all_cres.json"
    atomic_write_json(output, output_path)
    logger.info(
        "Wrote %d CREs from %d pages to %s",
        len(all_cres),
        output["total_pages"],
        output_path,
    )


if __name__ == "__main__":
    fetch_all_cres()
