"""Aggregate Phase 0 experiment results and evaluate gate criteria.

Reads results from all four experiments and prints the summary table.
Evaluates:
  (a) Opus hit@5 > 0.50 on all-198 → task is feasible
  (b) Best embedding hit@1 at least 0.10 below Opus hit@1 → room for trained model
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

from scripts.phase0.common import RESULTS_DIR, save_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GATE_A_THRESHOLD: Final[float] = 0.50
GATE_B_THRESHOLD: Final[float] = 0.10

METRIC_NAMES: Final[list[str]] = ["hit_at_1", "hit_at_5", "mrr", "ndcg_at_10"]
METRIC_HEADERS: Final[list[str]] = ["hit@1", "hit@5", "MRR", "NDCG@10"]


def load_result(filename: str) -> dict | None:
    """Load a result JSON file, returning None if missing."""
    path = RESULTS_DIR / filename
    if not path.exists():
        logger.warning("Result file not found: %s", path)
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt_metric(m: dict[str, float]) -> str:
    """Format metric dict as 'mean [ci_low, ci_high]'."""
    return f"{m['mean']:.3f} [{m['ci_low']:.3f}, {m['ci_high']:.3f}]"


def main() -> None:
    exp1 = load_result("exp1_embedding_baseline.json")
    exp2 = load_result("exp2_llm_probe.json")
    exp3 = load_result("exp3_hierarchy_paths.json")
    exp4 = load_result("exp4_hub_descriptions.json")

    rows: list[tuple[str, dict | None]] = []

    if exp1:
        for model_name, model_data in exp1["models"].items():
            rows.append((f"{model_name} (baseline)", model_data.get("all_198")))

    if exp3:
        for model_name, model_data in exp3["models"].items():
            rows.append((
                f"{model_name} + paths",
                model_data.get("path_enriched", {}).get("all_198"),
            ))

    if exp4:
        for model_name, model_data in exp4["models"].items():
            rows.append((
                f"{model_name} + descriptions",
                model_data.get("description_enriched_subset"),
            ))

    if exp2:
        rows.append(("Opus LLM probe", exp2.get("all_198")))

    col_width = 22
    name_width = 30
    header_line = "=" * (name_width + col_width * len(METRIC_HEADERS))
    sep_line = "-" * (name_width + col_width * len(METRIC_HEADERS))

    logger.info("\n%s", header_line)
    logger.info("PHASE 0 SUMMARY — All-198 Track")
    logger.info(header_line)
    logger.info(
        "%s%s",
        f"{'Method':<{name_width}}",
        "".join(f"{h:<{col_width}}" for h in METRIC_HEADERS),
    )
    logger.info(sep_line)

    for name, metrics in rows:
        if metrics is None:
            cells = "(missing)" + " " * (col_width - len("(missing)"))
            logger.info("%s%s", f"{name:<{name_width}}", cells * len(METRIC_NAMES))
            continue

        formatted: list[str] = []
        for m_name in METRIC_NAMES:
            m = metrics.get(m_name)
            formatted.append(fmt_metric(m) if m else "(n/a)")
        logger.info(
            "%s%s",
            f"{name:<{name_width}}",
            "".join(f"{cell:<{col_width}}" for cell in formatted),
        )

    logger.info(sep_line)

    opus_hit5: float | None = None
    opus_hit1: float | None = None
    if exp2 and "all_198" in exp2:
        opus_hit5 = exp2["all_198"]["hit_at_5"]["mean"]
        opus_hit1 = exp2["all_198"]["hit_at_1"]["mean"]

    best_emb_hit1: float | None = None
    best_emb_name: str | None = None
    if exp1:
        for model_name, model_data in exp1["models"].items():
            h1 = model_data["all_198"]["hit_at_1"]["mean"]
            if best_emb_hit1 is None or h1 > best_emb_hit1:
                best_emb_hit1 = h1
                best_emb_name = model_name

    print("\nGATE CRITERIA (all-198 track):")

    if opus_hit5 is not None:
        gate_a_pass = opus_hit5 > GATE_A_THRESHOLD
        print(
            f"  (a) Opus hit@5 = {opus_hit5:.3f} > "
            f"{GATE_A_THRESHOLD:.2f}? {'PASS' if gate_a_pass else 'FAIL'}"
        )
    else:
        print("  (a) Opus hit@5: MISSING (experiment 2 not run)")
        gate_a_pass = False

    if opus_hit1 is not None and best_emb_hit1 is not None:
        gap = opus_hit1 - best_emb_hit1
        gate_b_pass = gap > GATE_B_THRESHOLD
        print(
            f"  (b) Opus hit@1 ({opus_hit1:.3f}) - best embedding hit@1 "
            f"({best_emb_hit1:.3f}, {best_emb_name}) = "
            f"{gap:.3f} > {GATE_B_THRESHOLD:.2f}? "
            f"{'PASS' if gate_b_pass else 'FAIL'}"
        )
    else:
        print("  (b) hit@1 gap: MISSING (need both experiments 1 and 2)")
        gate_b_pass = False

    print()
    if gate_a_pass and gate_b_pass:
        print(">>> BOTH GATES PASS — proceed to Phase 1.")
    elif not gate_a_pass and not gate_b_pass:
        print(">>> BOTH GATES FAIL — reassess architecture.")
    else:
        print(">>> PARTIAL PASS — review results before proceeding.")

    summary: dict = {
        "gate_a": {
            "opus_hit5": opus_hit5,
            "threshold": GATE_A_THRESHOLD,
            "pass": gate_a_pass,
        },
        "gate_b": {
            "opus_hit1": opus_hit1,
            "best_embedding_hit1": best_emb_hit1,
            "best_embedding_model": best_emb_name,
            "gap": (opus_hit1 - best_emb_hit1) if opus_hit1 is not None and best_emb_hit1 is not None else None,
            "threshold": GATE_B_THRESHOLD,
            "pass": gate_b_pass,
        },
        "proceed_to_phase1": gate_a_pass and gate_b_pass,
    }
    save_results(summary, "summary.json")
    print("\nSaved summary to results/phase0/summary.json")


if __name__ == "__main__":
    main()
