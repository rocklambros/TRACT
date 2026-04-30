"""Experiment 6: Few-shot Sonnet baseline with hub descriptions.

Prompts Claude Sonnet with the full CRE hub catalog (400 hubs with
hierarchy paths, names, and expert descriptions) and 3-shot examples.
Tests whether hub descriptions + few-shot can beat embedding baselines.

Two variants:
  - sonnet-desc: catalog includes expert-reviewed descriptions
  - sonnet-nodesc: catalog uses only path + name + linked standards

Usage:
    python -m scripts.phase0.exp6_fewshot_sonnet
    python -m scripts.phase0.exp6_fewshot_sonnet --variant desc
    python -m scripts.phase0.exp6_fewshot_sonnet --original
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import anthropic

from scripts.phase0.common import (
    AI_FRAMEWORK_NAMES,
    CREHierarchy,
    HubStandardLink,
    LOFOFold,
    PROJECT_ROOT,
    aggregate_lofo_metrics,
    build_evaluation_corpus,
    build_hierarchy,
    build_lofo_folds,
    extract_hub_standard_links,
    finish_wandb,
    get_api_key,
    init_wandb,
    load_curated_links,
    load_opencre_cres,
    load_parsed_controls,
    log_aggregate_metrics,
    log_fold_metrics,
    save_results,
    score_predictions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL: Final[str] = "claude-sonnet-4-20250514"
MAX_CONCURRENT: Final[int] = 10
MAX_TOKENS: Final[int] = 512
API_TIMEOUT_S: Final[int] = 60
DESCRIPTION_TRUNCATE: Final[int] = 150
DESCRIPTIONS_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "hub_descriptions_reviewed.json"

SYSTEM_PROMPT: Final[str] = (
    "You are a security taxonomy classifier. Given a security control "
    "description, assign it to the most relevant CRE (Common Requirements "
    "Enumeration) hub from the provided catalog."
)


# ── Data Loading ──────────────────────────────────────────────────────────


def load_descriptions() -> dict[str, dict]:
    """Load expert-reviewed hub descriptions."""
    with open(DESCRIPTIONS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["descriptions"]


# ── Prompt Building ───────────────────────────────────────────────────────


def format_catalog_with_desc(
    tree: CREHierarchy,
    descriptions: dict[str, dict],
    hub_texts: dict[str, str],
) -> str:
    """Format hub catalog with descriptions (one line per hub)."""
    lines: list[str] = []
    for hub_id in sorted(hub_texts.keys()):
        path = tree.hierarchy_path(hub_id)
        name = tree.hubs[hub_id]
        desc_entry = descriptions.get(hub_id, {})
        desc = desc_entry.get("reviewed_description") or desc_entry.get("description", "")
        if desc and len(desc) > DESCRIPTION_TRUNCATE:
            desc = desc[:DESCRIPTION_TRUNCATE] + "..."
        if desc:
            lines.append(f"[{hub_id}] {path} | {name}: {desc}")
        else:
            lines.append(f"[{hub_id}] {path} | {name}")
    return "\n".join(lines)


def format_catalog_no_desc(
    tree: CREHierarchy,
    hub_texts: dict[str, str],
) -> str:
    """Format hub catalog without descriptions (path + name + linked)."""
    lines: list[str] = []
    for hub_id in sorted(hub_texts.keys()):
        text = hub_texts.get(hub_id, tree.hubs.get(hub_id, ""))
        path = tree.hierarchy_path(hub_id)
        lines.append(f"[{hub_id}] {path} | {text}")
    return "\n".join(lines)


def select_few_shot_examples(
    links: list[HubStandardLink],
    held_out_framework: str,
    tree: CREHierarchy,
    n: int = 3,
) -> list[dict[str, str]]:
    """Select n deterministic few-shot examples from non-held-out training links."""
    candidates = [
        lk for lk in links
        if lk.standard_name != held_out_framework
        and lk.standard_name in AI_FRAMEWORK_NAMES
        and lk.section_name
    ]
    candidates.sort(key=lambda lk: lk.cre_id)

    if len(candidates) < n:
        return []

    indices = [i * len(candidates) // n for i in range(n)]
    examples: list[dict[str, str]] = []
    for idx in indices:
        lk = candidates[idx]
        examples.append({
            "text": lk.section_name,
            "hub_id": lk.cre_id,
            "hub_name": tree.hubs.get(lk.cre_id, lk.cre_name),
        })
    return examples


def build_user_prompt(
    control_text: str,
    catalog: str,
    examples: list[dict[str, str]],
) -> str:
    """Build the user message for hub assignment."""
    example_lines = "\n".join(
        f'Control: "{ex["text"]}" → Hub: {ex["hub_id"]} ({ex["hub_name"]})'
        for ex in examples
    )

    return (
        f"CATALOG:\n{catalog}\n\n"
        f"EXAMPLES:\n{example_lines}\n\n"
        f"CONTROL:\n{control_text}\n\n"
        "Assign this control to the most relevant CRE hub. Return a JSON "
        "array of up to 10 hub IDs ranked by relevance, most relevant first.\n"
        'Response format: ["hub-id-1", "hub-id-2", ...]'
    )


# ── Response Parsing ──────────────────────────────────────────────────────

MAX_RESPONSE_LENGTH: Final[int] = 10_000


def parse_hub_ids(text: str) -> list[str]:
    """Extract hub ID list from LLM response."""
    text = text[:MAX_RESPONSE_LENGTH]
    stripped = text.strip()
    if stripped.startswith("["):
        try:
            ids = json.loads(stripped)
            return [str(i) for i in ids if isinstance(i, (str, int))]
        except json.JSONDecodeError:
            pass
    json_match = re.search(r"\[.*?\]", text, re.DOTALL)
    if json_match:
        try:
            ids = json.loads(json_match.group())
            return [str(i) for i in ids if isinstance(i, (str, int))]
        except json.JSONDecodeError:
            pass
    return re.findall(r"\d+-\d+", text)


# ── Async API Calls ───────────────────────────────────────────────────────


async def call_sonnet(
    client: anthropic.AsyncAnthropic,
    user_prompt: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, int, int]:
    """Make one Sonnet API call. Returns (text, input_tokens, output_tokens)."""
    async with semaphore:
        response = await asyncio.wait_for(
            client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                timeout=API_TIMEOUT_S,
            ),
            timeout=API_TIMEOUT_S + 10,
        )
        text = response.content[0].text
        return text, response.usage.input_tokens, response.usage.output_tokens


async def run_variant(
    client: anthropic.AsyncAnthropic,
    variant: str,
    tree: CREHierarchy,
    links: list[HubStandardLink],
    folds: list[LOFOFold],
    descriptions: dict[str, dict],
    max_concurrent: int,
) -> dict:
    """Run one variant (desc or nodesc) across all LOFO folds."""
    semaphore = asyncio.Semaphore(max_concurrent)
    fold_results: list[dict[int, list[str]]] = []
    total_input_tokens = 0
    total_output_tokens = 0

    for fold in folds:
        logger.info(
            "[%s] Fold: held out %s (%d items)",
            variant, fold.held_out_framework, len(fold.eval_items),
        )

        if variant == "sonnet-desc":
            catalog = format_catalog_with_desc(tree, descriptions, fold.hub_texts)
        else:
            catalog = format_catalog_no_desc(tree, fold.hub_texts)

        examples = select_few_shot_examples(links, fold.held_out_framework, tree)

        predictions: dict[int, list[str]] = {}
        for i, item in enumerate(fold.eval_items):
            prompt = build_user_prompt(item.control_text, catalog, examples)
            text, in_tok, out_tok = await call_sonnet(client, prompt, semaphore)
            total_input_tokens += in_tok
            total_output_tokens += out_tok

            ranked = parse_hub_ids(text)
            valid_hub_ids = set(fold.hub_ids)
            predictions[i] = [h for h in ranked if h in valid_hub_ids][:10]

            if (i + 1) % 10 == 0:
                logger.info("  [%s] Progress: %d/%d", variant, i + 1, len(fold.eval_items))

        fold_results.append(predictions)

    metrics_all = aggregate_lofo_metrics(fold_results, folds, track_filter=None)
    metrics_ft = aggregate_lofo_metrics(fold_results, folds, track_filter="full-text")

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

    cost_usd = (total_input_tokens * 3.0 / 1_000_000) + (total_output_tokens * 15.0 / 1_000_000)

    return {
        "all": metrics_all,
        "full_text": metrics_ft,
        "per_fold": per_fold,
        "tokens": {"input": total_input_tokens, "output": total_output_tokens},
        "cost_usd": round(cost_usd, 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────


async def run_experiment(
    variants: list[str],
    max_concurrent: int,
    curated: bool,
) -> dict:
    """Run the full few-shot Sonnet experiment."""
    import anthropic

    api_key = get_api_key()
    client = anthropic.AsyncAnthropic(
        api_key=api_key, max_retries=3, timeout=API_TIMEOUT_S,
    )

    try:
        logger.info("Loading data (curated=%s)...", curated)
        cres = load_opencre_cres()
        tree = build_hierarchy(cres)

        if curated:
            links = load_curated_links()
        else:
            links = extract_hub_standard_links(cres)

        parsed_controls = load_parsed_controls()
        corpus = build_evaluation_corpus(links, AI_FRAMEWORK_NAMES, parsed_controls)
        descriptions = load_descriptions()
        logger.info("Loaded %d hub descriptions", len(descriptions))

        folds_default = build_lofo_folds(tree, links, corpus, AI_FRAMEWORK_NAMES, template="default")

        results: dict = {
            "experiment": "exp6_fewshot_sonnet",
            "model": MODEL,
            "curated": curated,
            "variants": {},
        }

        start_time = time.time()

        for variant in variants:
            logger.info("=" * 60)
            logger.info("Running variant: %s", variant)
            variant_start = time.time()

            wandb_run = init_wandb(
                f"exp6_{variant}",
                config={"model": MODEL, "variant": variant, "max_concurrent": max_concurrent},
                curated=curated,
                tags=["fewshot-baseline"],
            )

            variant_result = await run_variant(
                client, variant, tree, links, folds_default, descriptions, max_concurrent,
            )

            variant_elapsed = time.time() - variant_start
            variant_result["elapsed_seconds"] = round(variant_elapsed, 1)

            for fold_data in variant_result["per_fold"]:
                log_fold_metrics(wandb_run, variant, fold_data["framework"], fold_data["metrics"])
            log_aggregate_metrics(wandb_run, variant, variant_result["all"], prefix="all")
            log_aggregate_metrics(wandb_run, variant, variant_result["full_text"], prefix="full_text")

            finish_wandb(wandb_run)

            results["variants"][variant] = variant_result

            logger.info(
                "%s all: hit@1=%.3f [%.3f, %.3f], hit@5=%.3f [%.3f, %.3f] | $%.4f",
                variant,
                variant_result["all"]["hit_at_1"]["mean"],
                variant_result["all"]["hit_at_1"]["ci_low"],
                variant_result["all"]["hit_at_1"]["ci_high"],
                variant_result["all"]["hit_at_5"]["mean"],
                variant_result["all"]["hit_at_5"]["ci_low"],
                variant_result["all"]["hit_at_5"]["ci_high"],
                variant_result["cost_usd"],
            )

        results["total_elapsed_seconds"] = round(time.time() - start_time, 1)
    finally:
        await client.close()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 6: Few-shot Sonnet baseline")
    parser.add_argument(
        "--variant", choices=["all", "desc", "nodesc"], default="all",
    )
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--original", action="store_true", help="Use original uncurated links")
    parser.add_argument("--curated", action="store_true", help="Use curated links (default)")
    args = parser.parse_args()

    if args.variant == "all":
        variants = ["sonnet-desc", "sonnet-nodesc"]
    elif args.variant == "desc":
        variants = ["sonnet-desc"]
    else:
        variants = ["sonnet-nodesc"]

    results = asyncio.run(run_experiment(variants, args.max_concurrent, not args.original))
    save_results(results, "exp6_fewshot_sonnet.json")
    logger.info("Experiment 6 complete.")


if __name__ == "__main__":
    main()
