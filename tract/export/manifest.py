"""Export manifest generation (spec §7).

Every export produces export_manifest.json alongside the CSVs,
capturing provenance, filter settings, and per-framework counts.
"""
from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "unknown"


def build_manifest(
    per_framework_stats: dict[str, dict[str, int]],
    confidence_floor: float,
    confidence_overrides: dict[str, float],
    staleness_result: dict,
    model_adapter_hash: str,
) -> dict:
    total_exported = sum(
        s.get("exported", 0) for s in per_framework_stats.values()
    )
    total_excluded = sum(
        s.get("excluded_confidence", 0) + s.get("excluded_ood", 0) +
        s.get("excluded_ground_truth", 0) + s.get("excluded_null_confidence", 0) +
        s.get("excluded_not_accepted", 0)
        for s in per_framework_stats.values()
    )

    return {
        "tract_version": "0.1.0",
        "model_adapter_hash": model_adapter_hash,
        "confidence_floor": confidence_floor,
        "confidence_overrides": confidence_overrides,
        "export_date": datetime.now(timezone.utc).isoformat(),
        "tract_git_sha": _get_git_sha(),
        "staleness_check": {
            "status": staleness_result.get("status", "unknown"),
            "upstream_hub_count": staleness_result.get("upstream_hub_count", 0),
        },
        "per_framework": per_framework_stats,
        "total_exported": total_exported,
        "total_excluded": total_excluded,
    }
