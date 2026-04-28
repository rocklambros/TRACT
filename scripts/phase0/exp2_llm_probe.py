"""Experiment 2: LLM Probe — Opus feasibility ceiling for CRE hub assignment.

Two-stage approach per held-out control:
  Stage 1: Branch shortlisting — 5 API calls (one per CRE root branch),
           each asks Opus to select up to 20 candidate hubs.
  Stage 2: Final ranking — 1 API call with all shortlisted candidates,
           asks Opus to rank top 10.

Uses LOFO cross-validation with hub firewall on 5 AI frameworks.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import time
from typing import Final

import numpy as np

from scripts.phase0.common import (
    AI_FRAMEWORK_NAMES,
    CREHierarchy,
    EvalItem,
    LOFOFold,
    aggregate_lofo_metrics,
    build_evaluation_corpus,
    build_hierarchy,
    build_lofo_folds,
    extract_hub_standard_links,
    load_opencre_cres,
    load_parsed_controls,
    save_results,
    score_predictions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL: Final[str] = "claude-opus-4-20250514"
MAX_CONCURRENT_REQUESTS: Final[int] = 5
SHORTLIST_PER_BRANCH: Final[int] = 20
FINAL_TOP_K: Final[int] = 10


def get_api_key() -> str:
    """Retrieve Anthropic API key from environment or pass manager."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    result = subprocess.run(
        ["pass", "anthropic/api_key"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def build_branch_prompt(
    control_text: str,
    branch_root_name: str,
    branch_hubs: list[dict[str, str]],
) -> str:
    """Build Stage 1 prompt for shortlisting within one branch."""
    hub_lines = "\n".join(
        f"- [{h['id']}] {h['path']} | Linked: {h['linked']}"
        for h in branch_hubs
    )
    return (
        "You are classifying a security control into the Common Requirements "
        "Enumeration (CRE) taxonomy.\n\n"
        f"CONTROL TEXT:\n{control_text}\n\n"
        f"BRANCH: {branch_root_name}\n"
        "The following CRE hubs belong to this branch. Each hub has an ID, "
        "hierarchy path, and linked standard names.\n\n"
        f"{hub_lines}\n\n"
        f"Which of these hubs are relevant to this control? Return up to "
        f"{SHORTLIST_PER_BRANCH} candidates ranked by relevance.\n\n"
        "Respond with ONLY a JSON array of hub IDs, most relevant first. "
        'Example:\n["123-456", "789-012", "345-678"]'
    )


def build_final_prompt(
    control_text: str,
    candidates: list[dict[str, str]],
) -> str:
    """Build Stage 2 prompt for final ranking across all branches."""
    candidate_lines = "\n".join(
        f"- [{c['id']}] {c['path']} | Linked: {c['linked']}"
        for c in candidates
    )
    return (
        "You are classifying a security control into the Common Requirements "
        "Enumeration (CRE) taxonomy.\n\n"
        f"CONTROL TEXT:\n{control_text}\n\n"
        "CANDIDATE HUBS (shortlisted from all branches):\n"
        f"{candidate_lines}\n\n"
        f"Rank the top {FINAL_TOP_K} most relevant hubs for this control.\n\n"
        "Respond with ONLY a JSON array of hub IDs, most relevant first. "
        'Example:\n["123-456", "789-012", "345-678"]'
    )


def parse_hub_ids_from_response(text: str) -> list[str]:
    """Extract hub ID list from LLM response text.

    Tries JSON array extraction first, falls back to regex for ID patterns.
    """
    json_match = re.search(r"\[.*?\]", text, re.DOTALL)
    if json_match:
        try:
            ids = json.loads(json_match.group())
            return [str(i) for i in ids if isinstance(i, (str, int))]
        except json.JSONDecodeError:
            pass
    return re.findall(r"\d+-\d+", text)


def _extract_linked_text(hub_text: str) -> str:
    """Extract linked standard names from a hub text string."""
    if ":" in hub_text:
        linked_part = hub_text.split(":", 1)[1].strip()
        if linked_part:
            return linked_part
    return "(none)"


async def call_opus(
    client: object,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """Make one Opus API call with rate limiting via semaphore."""
    async with semaphore:
        response = await client.messages.create(  # type: ignore[union-attr]
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text  # type: ignore[union-attr]


async def predict_single_control(
    client: object,
    control_text: str,
    tree: CREHierarchy,
    fold: LOFOFold,
    semaphore: asyncio.Semaphore,
) -> list[str]:
    """Run two-stage prediction for a single control.

    Stage 1: one call per root branch to shortlist up to 20 candidates each.
    Stage 2: one call with all shortlisted candidates to rank top 10.
    """
    all_candidates: list[dict[str, str]] = []

    for root_id in tree.roots:
        branch_hub_ids = tree.branch_hub_ids(root_id)
        branch_hubs: list[dict[str, str]] = []
        for hid in branch_hub_ids:
            hub_text = fold.hub_texts.get(hid, tree.hubs.get(hid, ""))
            branch_hubs.append({
                "id": hid,
                "path": tree.hierarchy_path(hid),
                "linked": _extract_linked_text(hub_text),
            })

        prompt = build_branch_prompt(control_text, tree.hubs[root_id], branch_hubs)
        response_text = await call_opus(client, prompt, semaphore)
        shortlisted_ids = parse_hub_ids_from_response(response_text)

        valid_branch_ids = {h["id"] for h in branch_hubs}
        for hid in shortlisted_ids[:SHORTLIST_PER_BRANCH]:
            if hid in valid_branch_ids:
                hub_text = fold.hub_texts.get(hid, tree.hubs.get(hid, ""))
                all_candidates.append({
                    "id": hid,
                    "path": tree.hierarchy_path(hid),
                    "linked": _extract_linked_text(hub_text),
                })

    if not all_candidates:
        return []

    final_prompt = build_final_prompt(control_text, all_candidates)
    final_response = await call_opus(client, final_prompt, semaphore)
    final_ranked = parse_hub_ids_from_response(final_response)

    valid_candidate_ids = {c["id"] for c in all_candidates}
    return [hid for hid in final_ranked if hid in valid_candidate_ids][:FINAL_TOP_K]


async def run_fold_async(
    client: object,
    fold: LOFOFold,
    tree: CREHierarchy,
    max_concurrent: int,
) -> dict[int, list[str]]:
    """Run all controls in one LOFO fold."""
    semaphore = asyncio.Semaphore(max_concurrent)
    predictions: dict[int, list[str]] = {}

    for i, item in enumerate(fold.eval_items):
        ranked = await predict_single_control(
            client, item.control_text, tree, fold, semaphore,
        )
        predictions[i] = ranked
        logger.info(
            "  Control %d/%d: predicted %d hubs (truth: %s)",
            i + 1, len(fold.eval_items), len(ranked), item.ground_truth_hub_id,
        )

    return predictions


async def run_experiment(max_concurrent: int) -> dict:
    """Run the full LLM probe experiment across all LOFO folds."""
    import anthropic

    api_key = get_api_key()
    client = anthropic.AsyncAnthropic(api_key=api_key)

    logger.info("Loading data...")
    cres = load_opencre_cres()
    tree = build_hierarchy(cres)
    links = extract_hub_standard_links(cres)
    parsed_controls = load_parsed_controls()
    corpus = build_evaluation_corpus(links, AI_FRAMEWORK_NAMES, parsed_controls)

    logger.info("Building LOFO folds...")
    folds = build_lofo_folds(tree, links, corpus, AI_FRAMEWORK_NAMES, template="default")

    fold_results: list[dict[int, list[str]]] = []
    raw_predictions: list[dict] = []
    start_time = time.time()

    for fold in folds:
        logger.info("=" * 60)
        logger.info(
            "Fold: held out %s (%d items)",
            fold.held_out_framework, len(fold.eval_items),
        )
        fold_start = time.time()

        preds = await run_fold_async(client, fold, tree, max_concurrent)
        fold_results.append(preds)

        for i, item in enumerate(fold.eval_items):
            pred_list = preds.get(i, [])
            raw_predictions.append({
                "framework": fold.held_out_framework,
                "section_id": item.section_id,
                "ground_truth": item.ground_truth_hub_id,
                "predicted": pred_list,
                "hit_at_1": bool(pred_list and pred_list[0] == item.ground_truth_hub_id),
            })

        fold_elapsed = time.time() - fold_start
        logger.info("Fold completed in %.1f seconds", fold_elapsed)

    total_elapsed = time.time() - start_time
    logger.info("Total elapsed: %.1f seconds", total_elapsed)

    metrics_all198 = aggregate_lofo_metrics(fold_results, folds, track_filter=None)
    metrics_fulltext = aggregate_lofo_metrics(
        fold_results, folds, track_filter="full-text",
    )

    per_fold: list[dict] = []
    for fold, preds in zip(folds, fold_results):
        fold_preds_list = [preds[i] for i in range(len(fold.eval_items))]
        fold_gt = [item.ground_truth_hub_id for item in fold.eval_items]
        fold_metrics = score_predictions(fold_preds_list, fold_gt)
        per_fold.append({
            "framework": fold.held_out_framework,
            "n_items": len(fold.eval_items),
            "metrics": fold_metrics,
        })

    return {
        "experiment": "exp2_llm_probe",
        "model": MODEL,
        "elapsed_seconds": round(total_elapsed, 1),
        "all_198": metrics_all198,
        "full_text": metrics_fulltext,
        "per_fold": per_fold,
        "raw_predictions": raw_predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 2: Opus LLM Probe",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=MAX_CONCURRENT_REQUESTS,
        help="Max concurrent API requests",
    )
    args = parser.parse_args()

    results = asyncio.run(run_experiment(args.max_concurrent))
    save_results(results, "exp2_llm_probe.json")

    logger.info(
        "Opus all-198: hit@1=%.3f [%.3f, %.3f], hit@5=%.3f [%.3f, %.3f]",
        results["all_198"]["hit_at_1"]["mean"],
        results["all_198"]["hit_at_1"]["ci_low"],
        results["all_198"]["hit_at_1"]["ci_high"],
        results["all_198"]["hit_at_5"]["mean"],
        results["all_198"]["hit_at_5"]["ci_low"],
        results["all_198"]["hit_at_5"]["ci_high"],
    )
    logger.info("Experiment 2 complete.")


if __name__ == "__main__":
    main()
