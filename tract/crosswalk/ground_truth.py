"""Ground truth import — multi-strategy section ID resolver and assignment creation."""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from tract.config import (
    PHASE3_GT_PROVENANCE,
    PHASE3_MODEL_PROVENANCE,
    PHASE3_TEXT_QUALITY_LOW_THRESHOLD,
    PHASE3_UNCOVERED_FRAMEWORK_IDS,
)
from tract.crosswalk.schema import get_connection

logger = logging.getLogger(__name__)


@dataclass
class ResolverResult:
    resolved: dict[str, str]           # gt_key → control_id
    unresolved: list[dict]             # list of unresolvable GT links
    strategy_counts: dict[str, int]    # strategy_name → count


def build_control_lookups(
    conn: sqlite3.Connection,
    framework_id: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Build three lookup dictionaries for a framework's controls.

    Returns (section_id_map, title_map, normalized_map) where each maps
    the lookup key → control_id.
    """
    rows = conn.execute(
        "SELECT id, section_id, title FROM controls WHERE framework_id = ?",
        (framework_id,),
    ).fetchall()

    section_id_map: dict[str, str] = {}
    title_map: dict[str, str] = {}
    normalized_map: dict[str, str] = {}

    for row in rows:
        ctrl_id = row["id"]
        sid = row["section_id"]
        title = row["title"] or ""

        section_id_map[sid] = ctrl_id
        if title:
            title_map[title] = ctrl_id
            title_map[title.lower()] = ctrl_id
        normalized_key = re.sub(r"[^a-z0-9]", "", sid.lower())
        if normalized_key:
            normalized_map[normalized_key] = ctrl_id

    return section_id_map, title_map, normalized_map


def resolve_section_id(
    gt_section_id: str,
    framework_id: str,
    section_id_map: dict[str, str],
    title_map: dict[str, str],
    normalized_map: dict[str, str],
) -> tuple[str | None, str]:
    """Try 5 strategies to resolve a GT section_id to a control_id.

    Returns (control_id, strategy_name) or (None, "unresolved").
    """
    # Strategy 1: Direct match
    if gt_section_id in section_id_map:
        return section_id_map[gt_section_id], "direct"

    # Strategy 2: Prefixed match
    prefixed = f"{framework_id}:{gt_section_id}"
    if prefixed in section_id_map:
        return section_id_map[prefixed], "prefixed"

    # Strategy 3: Title exact match
    if gt_section_id in title_map:
        return title_map[gt_section_id], "title_exact"

    # Strategy 4: Title case-insensitive
    if gt_section_id.lower() in title_map:
        return title_map[gt_section_id.lower()], "title_case_insensitive"

    # Strategy 5: Normalized
    gt_normalized = re.sub(r"[^a-z0-9]", "", gt_section_id.lower())
    if gt_normalized and gt_normalized in normalized_map:
        return normalized_map[gt_normalized], "normalized"

    return None, "unresolved"


def resolve_framework_links(
    conn: sqlite3.Connection,
    framework_id: str,
    links: list[dict],
) -> ResolverResult:
    """Resolve all GT links for a single framework.

    Returns ResolverResult with resolved mappings, unresolved links, and strategy counts.
    """
    section_id_map, title_map, normalized_map = build_control_lookups(conn, framework_id)

    resolved: dict[str, str] = {}
    unresolved: list[dict] = []
    strategy_counts: dict[str, int] = {}

    for link in links:
        gt_sid = link["section_id"]
        gt_key = f"{framework_id}:{gt_sid}"

        control_id, strategy = resolve_section_id(
            gt_sid, framework_id, section_id_map, title_map, normalized_map,
        )

        if control_id is not None:
            resolved[gt_key] = control_id
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        else:
            unresolved.append(link)
            logger.warning(
                "Unresolvable GT link: framework=%s section_id=%s",
                framework_id, gt_sid,
            )

    return ResolverResult(
        resolved=resolved,
        unresolved=unresolved,
        strategy_counts=strategy_counts,
    )


def _backup_database(db_path: Path) -> Path:
    """Copy database to timestamped backup using sqlite3 backup API (WAL-safe)."""
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = db_path.with_name(f"{db_path.name}.backup.{ts}")
    src = sqlite3.connect(str(db_path))
    dst = sqlite3.connect(str(backup))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    logger.info("Database backed up to %s", backup)
    return backup


def import_ground_truth(
    db_path: Path,
    hub_links_path: Path,
    *,
    dry_run: bool = False,
) -> dict[str, object]:
    """Import OpenCRE ground truth links into crosswalk.db.

    Returns summary dict with counts: imported, skipped_duplicate, unresolved,
    per_framework breakdown, and strategy_counts.
    """
    hub_links_data = json.loads(hub_links_path.read_text(encoding="utf-8"))

    if not dry_run:
        _backup_database(db_path)

    conn = get_connection(db_path)
    imported = 0
    skipped_duplicate = 0
    total_unresolved = 0
    per_framework: dict[str, dict[str, int]] = {}
    all_strategy_counts: dict[str, int] = {}

    try:
        for framework_id, links in sorted(hub_links_data.items()):
            result = resolve_framework_links(conn, framework_id, links)

            fw_imported = 0
            fw_skipped = 0

            for link in links:
                gt_key = f"{framework_id}:{link['section_id']}"
                control_id = result.resolved.get(gt_key)
                if control_id is None:
                    continue

                hub_id = link["cre_id"]

                existing = conn.execute(
                    "SELECT 1 FROM assignments WHERE control_id = ? AND hub_id = ?",
                    (control_id, hub_id),
                ).fetchone()

                if existing is not None:
                    fw_skipped += 1
                    skipped_duplicate += 1
                    continue

                conn.execute(
                    "INSERT INTO assignments "
                    "(control_id, hub_id, confidence, provenance, "
                    "source_link_id, review_status) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        control_id,
                        hub_id,
                        1.0,
                        PHASE3_GT_PROVENANCE,
                        link["link_type"],
                        "ground_truth",
                    ),
                )
                fw_imported += 1
                imported += 1

            total_unresolved += len(result.unresolved)

            per_framework[framework_id] = {
                "imported": fw_imported,
                "skipped_duplicate": fw_skipped,
                "unresolved": len(result.unresolved),
                "total_links": len(links),
            }

            for strategy, count in result.strategy_counts.items():
                all_strategy_counts[strategy] = (
                    all_strategy_counts.get(strategy, 0) + count
                )

            logger.info(
                "Framework %s: imported=%d, skipped=%d, unresolved=%d",
                framework_id, fw_imported, fw_skipped, len(result.unresolved),
            )

        if dry_run:
            conn.rollback()
            logger.info("Dry run — rolled back %d inserts", imported)
        else:
            conn.commit()
            logger.info("Committed %d ground truth assignments", imported)

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return {
        "imported": imported,
        "skipped_duplicate": skipped_duplicate,
        "unresolved": total_unresolved,
        "per_framework": per_framework,
        "strategy_counts": all_strategy_counts,
        "dry_run": dry_run,
    }


def run_uncovered_inference(
    db_path: Path,
    model_dir: Path,
    *,
    dry_run: bool = False,
) -> dict[str, object]:
    """Run inference on uncovered framework controls and insert as model_prediction.

    Returns summary dict with per-framework counts and text quality warnings.
    """
    from tract.inference import TRACTPredictor

    if not dry_run:
        _backup_database(db_path)

    predictor = TRACTPredictor(model_dir)
    model_version = predictor._artifacts.model_adapter_hash[:12]

    conn = get_connection(db_path)
    total_inserted = 0
    per_framework: dict[str, dict[str, int]] = {}
    text_quality_warnings: list[dict[str, str]] = []

    try:
        for framework_id in sorted(PHASE3_UNCOVERED_FRAMEWORK_IDS):
            rows = conn.execute(
                "SELECT id, title, description, full_text "
                "FROM controls WHERE framework_id = ?",
                (framework_id,),
            ).fetchall()

            if not rows:
                logger.info("No controls found for framework %s", framework_id)
                per_framework[framework_id] = {"controls": 0, "inserted": 0}
                continue

            control_ids: list[str] = []
            texts: list[str] = []
            for row in rows:
                control_ids.append(row["id"])
                text = " ".join(filter(None, [row["title"], row["description"], row["full_text"]]))
                texts.append(text)

                if len(text) < PHASE3_TEXT_QUALITY_LOW_THRESHOLD:
                    text_quality_warnings.append({
                        "control_id": row["id"],
                        "framework_id": framework_id,
                        "text_length": str(len(text)),
                    })
                    logger.warning(
                        "Short control text (%d chars): %s",
                        len(text), row["id"],
                    )

            batch_results = predictor.predict_batch(texts, top_k=1)

            fw_inserted = 0
            for i, predictions in enumerate(batch_results):
                prediction = predictions[0]
                conn.execute(
                    "INSERT INTO assignments "
                    "(control_id, hub_id, confidence, in_conformal_set, "
                    "is_ood, provenance, model_version, review_status) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        control_ids[i],
                        prediction.hub_id,
                        prediction.calibrated_confidence,
                        1 if prediction.in_conformal_set else 0,
                        1 if prediction.is_ood else 0,
                        PHASE3_MODEL_PROVENANCE,
                        model_version,
                        "pending",
                    ),
                )
                fw_inserted += 1

            total_inserted += fw_inserted
            per_framework[framework_id] = {
                "controls": len(rows),
                "inserted": fw_inserted,
            }
            logger.info(
                "Framework %s: %d controls, %d predictions inserted",
                framework_id, len(rows), fw_inserted,
            )

        if dry_run:
            conn.rollback()
            logger.info("Dry run — rolled back %d inference inserts", total_inserted)
        else:
            conn.commit()
            logger.info("Committed %d inference predictions", total_inserted)

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return {
        "total_inserted": total_inserted,
        "per_framework": per_framework,
        "text_quality_warnings": text_quality_warnings,
        "model_version": model_version,
        "dry_run": dry_run,
    }
