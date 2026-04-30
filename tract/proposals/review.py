"""Hub proposal writing and interactive review.

write_proposal_round: writes proposals to hub_proposals/round_N/ as versioned JSON.
run_review_session: interactive CLI loop for accept/reject/edit/skip per proposal.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from tract.config import HUB_PROPOSALS_DIR
from tract.io import atomic_write_json
from tract.proposals.guardrails import GuardrailResult

logger = logging.getLogger(__name__)


def write_proposal_round(
    results: list[GuardrailResult],
    names: dict[int, str] | None,
    output_dir: Path,
    round_num: int,
) -> Path:
    """Write proposals to hub_proposals/round_N/ as versioned JSON. Atomic write."""
    round_dir = output_dir / f"round_{round_num}"
    round_dir.mkdir(parents=True, exist_ok=True)

    passing = [r for r in results if r.passed]
    proposals = []
    for result in passing:
        name = (names or {}).get(result.cluster.cluster_id, f"Cluster {result.cluster.cluster_id}")
        proposals.append({
            "cluster_id": result.cluster.cluster_id,
            "proposed_name": name,
            "control_ids": result.cluster.control_ids,
            "member_frameworks": sorted(result.cluster.member_frameworks),
            "parent_hub_id": result.parent_hub_id,
            "parent_similarity": result.parent_similarity,
            "nearest_hub_id": result.cluster.nearest_hub_id,
            "nearest_hub_similarity": result.cluster.nearest_hub_similarity,
            "uncertain_placement": result.uncertain_placement,
            "evidence_score": result.evidence_score,
            "rejection_reasons": result.rejection_reasons,
        })

    output_data = {
        "round": round_num,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_clusters_evaluated": len(results),
        "total_passing": len(passing),
        "proposals": proposals,
    }

    atomic_write_json(output_data, round_dir / "proposals.json")
    logger.info("Wrote %d proposals to %s", len(proposals), round_dir)
    return round_dir


def run_review_session(
    round_dir: Path,
    hierarchy: object,
    db_path: Path,
    dry_run: bool = False,
) -> dict:
    """Interactive CLI review loop.

    Per proposal: [a]ccept, [r]eject, [e]dit name, [s]kip
    Returns summary dict of actions taken.
    """
    proposals_path = round_dir / "proposals.json"
    if not proposals_path.exists():
        raise FileNotFoundError(f"No proposals.json in {round_dir}")

    from tract.io import load_json
    data = load_json(proposals_path)
    proposals = data["proposals"]

    if not proposals:
        print("No proposals to review.")
        return {"accepted": 0, "rejected": 0, "skipped": 0}

    if not dry_run:
        pre_review_dir = round_dir / "pre_review"
        pre_review_dir.mkdir(exist_ok=True)
        if db_path.exists():
            shutil.copy2(db_path, pre_review_dir / "crosswalk.db.backup")

    accepted = 0
    rejected = 0
    skipped = 0

    for i, proposal in enumerate(proposals):
        print(f"\n{'='*60}")
        print(f"Proposal {i+1}/{len(proposals)}: {proposal['proposed_name']}")
        print(f"  Cluster ID: {proposal['cluster_id']}")
        print(f"  Controls: {len(proposal['control_ids'])} from {proposal['member_frameworks']}")
        print(f"  Parent hub: {proposal['parent_hub_id']} (similarity: {proposal['parent_similarity']:.3f})")
        if proposal.get("uncertain_placement"):
            print("  WARNING: Uncertain placement (low parent similarity)")
        print(f"  Evidence score: {proposal['evidence_score']:.1f}")
        print(f"  Control IDs: {', '.join(proposal['control_ids'][:5])}")
        if len(proposal['control_ids']) > 5:
            print(f"    ... and {len(proposal['control_ids']) - 5} more")

        if dry_run:
            print("  [DRY RUN] Would prompt for review action")
            skipped += 1
            continue

        while True:
            action = input("\n  [a]ccept / [r]eject / [e]dit name / [s]kip: ").strip().lower()
            if action in ("a", "r", "e", "s"):
                break
            print("  Invalid choice. Use a/r/e/s.")

        if action == "a":
            proposal["review_status"] = "accepted"
            accepted += 1
            logger.info("Accepted proposal %d: %s", proposal["cluster_id"], proposal["proposed_name"])
        elif action == "r":
            proposal["review_status"] = "rejected"
            rejected += 1
        elif action == "e":
            new_name = input("  New name: ").strip()
            if new_name:
                proposal["proposed_name"] = new_name
            proposal["review_status"] = "accepted"
            accepted += 1
        elif action == "s":
            proposal["review_status"] = "skipped"
            skipped += 1

    atomic_write_json(data, proposals_path)
    summary = {"accepted": accepted, "rejected": rejected, "skipped": skipped}
    logger.info("Review complete: %s", summary)
    return summary
