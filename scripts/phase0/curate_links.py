"""Apply expert audit corrections to AI training links.

Reads the audit CSV (data/training/ai_link_audit.csv), extracts corrections
from the notes column, and produces curated versions of the training links.

Output:
  - data/training/hub_links_curated.jsonl  (all 4,406 links with AI corrections applied)
  - data/training/hub_links_by_framework_curated.json  (grouped by framework)
  - data/training/audit_corrections_log.json  (detailed log of all changes)

Usage:
    python -m scripts.phase0.curate_links
    python -m scripts.phase0.curate_links --dry-run  # preview corrections only
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent
AUDIT_CSV: Final[Path] = PROJECT_ROOT / "data" / "training" / "ai_link_audit.csv"
ORIGINAL_LINKS: Final[Path] = PROJECT_ROOT / "data" / "training" / "hub_links.jsonl"
ORIGINAL_BY_FW: Final[Path] = PROJECT_ROOT / "data" / "training" / "hub_links_by_framework.json"
CURATED_LINKS: Final[Path] = PROJECT_ROOT / "data" / "training" / "hub_links_curated.jsonl"
CURATED_BY_FW: Final[Path] = PROJECT_ROOT / "data" / "training" / "hub_links_by_framework_curated.json"
CORRECTIONS_LOG: Final[Path] = PROJECT_ROOT / "data" / "training" / "audit_corrections_log.json"
HIERARCHY_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "cre_hierarchy.json"

FW_LABEL_TO_ID: Final[dict[str, str]] = {
    "OWASP-AIX": "owasp_ai_exchange",
    "ATLAS": "mitre_atlas",
    "NIST-AI": "nist_ai_100_2",
    "LLM-T10": "owasp_llm_top10",
    "ML-T10": "owasp_ml_top10",
}

CRE_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:Better(?:\s+[\w-]+)*\s*:\s*)"
    r"(\d{3}-\d{3})\s+"
    r"([A-Z][^.;,\d]*?)(?:\.|;|,|\Z)"
)

CRE_ID_BROAD: Final[re.Pattern[str]] = re.compile(
    r"(\d{3}-\d{3})\s+([A-Z][A-Za-z /&()\-]+)"
)

NO_HUB_MARKERS: Final[list[str]] = [
    "no suitable cre hub",
    "no single cre hub",
    "no ai bom specific cre hub",
    "no single more precise cre hub",
]


def _load_hub_names() -> dict[str, str]:
    """Load CRE hub ID -> name mapping from hierarchy."""
    with open(HIERARCHY_PATH, encoding="utf-8") as f:
        hierarchy = json.load(f)
    hub_names: dict[str, str] = {}
    for hub_id, hub_data in hierarchy["hubs"].items():
        if isinstance(hub_data, dict):
            hub_names[hub_id] = hub_data.get("name", hub_id)
        elif isinstance(hub_data, str):
            hub_names[hub_id] = hub_data
    return hub_names


def _parse_correction(notes: str, old_cre_id: str) -> tuple[str, str] | None:
    """Extract corrected (cre_id, cre_name) from audit notes. Returns None if no correction found."""
    matches = CRE_ID_PATTERN.findall(notes)
    if not matches:
        matches = CRE_ID_BROAD.findall(notes)

    for cre_id, cre_name in matches:
        cre_id = cre_id.strip()
        cre_name = cre_name.strip()
        if cre_id != old_cre_id:
            return cre_id, cre_name

    return None


def _is_no_hub(notes: str) -> bool:
    """Check if the notes indicate no suitable CRE hub exists."""
    notes_lower = notes.lower()
    return any(marker in notes_lower for marker in NO_HUB_MARKERS)


def parse_audit() -> tuple[list[dict], list[dict], list[dict]]:
    """Parse audit CSV into corrections, exclusions, and kept-as-is lists."""
    with open(AUDIT_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    corrections: list[dict] = []
    exclusions: list[dict] = []
    kept_weak: list[dict] = []

    for r in rows:
        verdict = r["verdict"].strip().lower()
        notes = r.get("notes", "").strip()
        fw_id = FW_LABEL_TO_ID[r["framework"]]

        if verdict == "correct":
            continue

        if verdict in ("wrong", "weak"):
            correction = _parse_correction(notes, r["cre_id"])
            if correction:
                new_cre_id, new_cre_name = correction
                corrections.append({
                    "framework_id": fw_id,
                    "framework_label": r["framework"],
                    "section_name": r["section_name"],
                    "old_cre_id": r["cre_id"],
                    "old_hub_name": r["hub_name"],
                    "new_cre_id": new_cre_id,
                    "new_hub_name": new_cre_name,
                    "verdict": verdict,
                    "notes": notes,
                })
            elif _is_no_hub(notes):
                if verdict == "wrong":
                    exclusions.append({
                        "framework_id": fw_id,
                        "framework_label": r["framework"],
                        "section_name": r["section_name"],
                        "cre_id": r["cre_id"],
                        "hub_name": r["hub_name"],
                        "verdict": verdict,
                        "notes": notes,
                    })
                else:
                    kept_weak.append({
                        "framework_id": fw_id,
                        "framework_label": r["framework"],
                        "section_name": r["section_name"],
                        "cre_id": r["cre_id"],
                        "hub_name": r["hub_name"],
                        "notes": notes,
                    })
            else:
                kept_weak.append({
                    "framework_id": fw_id,
                    "framework_label": r["framework"],
                    "section_name": r["section_name"],
                    "cre_id": r["cre_id"],
                    "hub_name": r["hub_name"],
                    "notes": notes,
                })

    return corrections, exclusions, kept_weak


def apply_corrections(
    corrections: list[dict],
    exclusions: list[dict],
    hub_names: dict[str, str],
) -> tuple[list[dict], dict[str, list[dict]]]:
    """Apply corrections to training links. Returns (curated_links, curated_by_fw)."""
    with open(ORIGINAL_LINKS, encoding="utf-8") as f:
        all_links = [json.loads(line) for line in f if line.strip()]

    correction_map: dict[tuple[str, str, str], tuple[str, str]] = {}
    for c in corrections:
        key = (c["framework_id"], c["section_name"], c["old_cre_id"])
        new_name = hub_names.get(c["new_cre_id"], c["new_hub_name"])
        correction_map[key] = (c["new_cre_id"], new_name)

    exclusion_set: set[tuple[str, str, str]] = set()
    for e in exclusions:
        exclusion_set.add((e["framework_id"], e["section_name"], e["cre_id"]))

    fw_id_lookup: dict[str, str] = {
        "MITRE ATLAS": "mitre_atlas",
        "OWASP AI Exchange": "owasp_ai_exchange",
        "NIST AI 100-2": "nist_ai_100_2",
        "OWASP Top10 for LLM": "owasp_llm_top10",
        "OWASP Top10 for ML": "owasp_ml_top10",
    }

    curated: list[dict] = []
    stats = Counter({"total": 0, "corrected": 0, "excluded": 0, "unchanged": 0})

    for link in all_links:
        stats["total"] += 1
        std_name = link.get("standard_name", "")
        fw_id = link.get("framework_id", fw_id_lookup.get(std_name, ""))
        section = link.get("section_name", "")
        cre_id = link.get("cre_id", "")

        excl_key = (fw_id, section, cre_id)
        if excl_key in exclusion_set:
            stats["excluded"] += 1
            continue

        corr_key = (fw_id, section, cre_id)
        if corr_key in correction_map:
            new_id, new_name = correction_map[corr_key]
            corrected_link = dict(link)
            corrected_link["cre_id"] = new_id
            corrected_link["cre_name"] = new_name
            corrected_link["_curated"] = True
            corrected_link["_original_cre_id"] = cre_id
            corrected_link["_original_cre_name"] = link.get("cre_name", "")
            curated.append(corrected_link)
            stats["corrected"] += 1
        else:
            curated.append(link)
            stats["unchanged"] += 1

    curated_by_fw: dict[str, list[dict]] = {}
    for link in curated:
        fw = link.get("framework_id", "")
        if not fw:
            std = link.get("standard_name", "")
            fw = fw_id_lookup.get(std, std.lower().replace(" ", "_"))
        curated_by_fw.setdefault(fw, []).append(link)

    logger.info(
        "Curation: %d total, %d corrected, %d excluded, %d unchanged",
        stats["total"], stats["corrected"], stats["excluded"], stats["unchanged"],
    )
    return curated, curated_by_fw


def _data_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, data: object) -> None:
    content = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".tmp", delete=False, encoding="utf-8",
    )
    try:
        tmp.write(content)
        tmp.write("\n")
        tmp.flush()
        Path(tmp.name).replace(path)
    except BaseException:
        Path(tmp.name).unlink(missing_ok=True)
        raise
    finally:
        tmp.close()


def _atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".tmp", delete=False, encoding="utf-8",
    )
    try:
        for rec in records:
            tmp.write(json.dumps(rec, sort_keys=True, ensure_ascii=False))
            tmp.write("\n")
        tmp.flush()
        Path(tmp.name).replace(path)
    except BaseException:
        Path(tmp.name).unlink(missing_ok=True)
        raise
    finally:
        tmp.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply audit corrections to training links")
    parser.add_argument("--dry-run", action="store_true", help="Preview corrections only")
    args = parser.parse_args()

    hub_names = _load_hub_names()
    logger.info("Loaded %d hub names from hierarchy", len(hub_names))

    corrections, exclusions, kept_weak = parse_audit()
    logger.info(
        "Parsed audit: %d corrections, %d exclusions, %d kept-as-is weak",
        len(corrections), len(exclusions), len(kept_weak),
    )

    for c in corrections:
        if c["new_cre_id"] not in hub_names:
            logger.warning(
                "Corrected hub %s (%s) not in hierarchy — check audit notes for: %s",
                c["new_cre_id"], c["new_hub_name"], c["section_name"],
            )

    if args.dry_run:
        print("\n=== CORRECTIONS ===")
        for c in corrections:
            print(f"  [{c['framework_label']}] {c['section_name']}")
            print(f"    {c['old_cre_id']} {c['old_hub_name']} -> {c['new_cre_id']} {c['new_hub_name']}")
        print(f"\n=== EXCLUSIONS ({len(exclusions)}) ===")
        for e in exclusions:
            print(f"  [{e['framework_label']}] {e['section_name']} ({e['verdict']})")
        print(f"\n=== KEPT WEAK ({len(kept_weak)}) ===")
        for w in kept_weak:
            print(f"  [{w['framework_label']}] {w['section_name']}")
        return

    curated, curated_by_fw = apply_corrections(corrections, exclusions, hub_names)

    _atomic_write_jsonl(CURATED_LINKS, curated)
    logger.info("Wrote %d curated links to %s", len(curated), CURATED_LINKS)

    _atomic_write_json(CURATED_BY_FW, curated_by_fw)
    logger.info("Wrote curated links by framework to %s", CURATED_BY_FW)

    ai_fws = {"owasp_ai_exchange", "mitre_atlas", "nist_ai_100_2", "owasp_llm_top10", "owasp_ml_top10"}
    ai_curated = sum(len(v) for k, v in curated_by_fw.items() if k in ai_fws)

    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "original_total": sum(1 for _ in open(ORIGINAL_LINKS)),
        "curated_total": len(curated),
        "ai_links_original": 198,
        "ai_links_curated": ai_curated,
        "corrections_applied": len(corrections),
        "links_excluded": len(exclusions),
        "weak_kept_as_is": len(kept_weak),
        "data_hash": _data_hash(
            "\n".join(json.dumps(r, sort_keys=True) for r in curated)
        ),
        "corrections": corrections,
        "exclusions": exclusions,
        "kept_weak": kept_weak,
    }
    _atomic_write_json(CORRECTIONS_LOG, log)
    logger.info("Wrote corrections log to %s", CORRECTIONS_LOG)

    print(f"\n{'='*60}")
    print(f"CURATION COMPLETE")
    print(f"  Original:   {log['original_total']} total links ({log['ai_links_original']} AI)")
    print(f"  Curated:    {log['curated_total']} total links ({ai_curated} AI)")
    print(f"  Corrected:  {len(corrections)} links remapped to better hubs")
    print(f"  Excluded:   {len(exclusions)} links with no suitable hub")
    print(f"  Kept weak:  {len(kept_weak)} links kept as-is (ambiguous)")
    print(f"  Data hash:  {log['data_hash'][:16]}...")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
