"""Pre-export CRE ID staleness check (spec §6).

Fetches current CRE IDs from upstream OpenCRE, diffs against TRACT's
hub snapshot. Warns if TRACT references hub IDs that no longer exist
upstream (may have been removed/merged).
"""
from __future__ import annotations

import logging
from pathlib import Path

import requests

from tract.config import (
    PHASE5_OPENCRE_STALENESS_TIMEOUT_S,
    PHASE5_OPENCRE_STALENESS_URL,
)
from tract.crosswalk.schema import get_connection

logger = logging.getLogger(__name__)


def _get_tract_hub_ids(db_path: Path) -> set[str]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT id FROM hubs").fetchall()
        return {row["id"] for row in rows}
    finally:
        conn.close()


def _fetch_upstream_cre_ids() -> set[str]:
    resp = requests.get(
        PHASE5_OPENCRE_STALENESS_URL,
        timeout=PHASE5_OPENCRE_STALENESS_TIMEOUT_S,
    )
    resp.raise_for_status()
    data = resp.json()
    ids: set[str] = set()
    _collect_ids(data, ids)
    return ids


def _collect_ids(node: dict | list, ids: set[str]) -> None:
    if isinstance(node, list):
        for item in node:
            _collect_ids(item, ids)
    elif isinstance(node, dict):
        if "id" in node:
            ids.add(node["id"])
        for key in ("links", "children", "contains"):
            if key in node and isinstance(node[key], list):
                for child in node[key]:
                    if isinstance(child, dict) and "document" in child:
                        _collect_ids(child["document"], ids)
                    elif isinstance(child, dict):
                        _collect_ids(child, ids)


def check_staleness(db_path: Path) -> dict:
    tract_ids = _get_tract_hub_ids(db_path)

    try:
        upstream_ids = _fetch_upstream_cre_ids()
    except Exception as e:
        logger.error("Staleness check failed: %s", e)
        return {
            "status": "error",
            "stale_ids": [],
            "upstream_only": [],
            "upstream_hub_count": 0,
            "message": str(e),
        }

    stale = sorted(tract_ids - upstream_ids)
    upstream_only = sorted(upstream_ids - tract_ids)

    if stale:
        logger.warning(
            "Staleness check: %d TRACT hub IDs not found upstream: %s",
            len(stale), stale,
        )
        status = "warn"
        message = f"{len(stale)} TRACT hub IDs not found in upstream OpenCRE"
    else:
        status = "pass"
        message = "All TRACT hub IDs found in upstream OpenCRE"

    if upstream_only:
        logger.info(
            "Staleness check: %d upstream IDs not in TRACT (new hubs, not our concern)",
            len(upstream_only),
        )

    return {
        "status": status,
        "stale_ids": stale,
        "upstream_only": upstream_only,
        "upstream_hub_count": len(upstream_ids),
        "message": message,
    }
