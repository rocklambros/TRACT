#!/usr/bin/env python3
"""Verify OpenCRE import by comparing fork API against export manifest (spec §10.1).

Usage:
    python scripts/verify_opencre_import.py --manifest opencre_export/export_manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import quote

import requests

FORK_BASE_URL = "http://localhost:5001/rest/v1"


def verify_framework(
    opencre_name: str,
    expected_count: int,
    base_url: str = FORK_BASE_URL,
) -> dict:
    """Query the fork API for a standard and count its CRE links."""
    url = f"{base_url}/standard/{quote(opencre_name)}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 404:
            return {"name": opencre_name, "status": "missing", "expected": expected_count, "actual": 0}
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return {"name": opencre_name, "status": "error", "message": str(e),
                "expected": expected_count, "actual": 0}

    linked_cre_count = 0
    if isinstance(data, dict):
        standards = [data]
    elif isinstance(data, list):
        standards = data
    else:
        standards = []

    for std in standards:
        for link in std.get("links", []):
            doc = link.get("document", {})
            if doc.get("doctype") == "CRE":
                linked_cre_count += 1

    status = "match" if linked_cre_count >= expected_count else "mismatch"
    return {
        "name": opencre_name,
        "status": status,
        "expected": expected_count,
        "actual": linked_cre_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify OpenCRE import against manifest")
    parser.add_argument("--manifest", required=True, help="Path to export_manifest.json")
    parser.add_argument("--base-url", default=FORK_BASE_URL, help="Fork API base URL")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    from tract.export.opencre_names import TRACT_TO_OPENCRE_NAME

    print(f"Verifying import against manifest ({manifest.get('export_date', 'unknown')})")
    print(f"Fork API: {args.base_url}")
    print("=" * 60)

    all_ok = True
    for fw_id, fw_stats in sorted(manifest.get("per_framework", {}).items()):
        expected = fw_stats.get("exported", 0)
        if expected == 0:
            continue

        opencre_name = TRACT_TO_OPENCRE_NAME.get(fw_id, fw_id)
        result = verify_framework(opencre_name, expected, args.base_url)

        icon = "✓" if result["status"] == "match" else "✗"
        print(f"  {icon} {opencre_name}: expected={expected}, actual={result['actual']} ({result['status']})")

        if result["status"] not in ("match",):
            all_ok = False

    print()
    if all_ok:
        print("All frameworks verified successfully.")
    else:
        print("Some frameworks have mismatches. Review above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
