"""Stratify the Campaign 2 test-round delta by whether TRACT's own link audit
touched the item's ground truth.

WHY THIS EXISTS. `data/training/audit_corrections_log.json` records 56 gold-label
corrections that TRACT applied to the AI framework links, against 197 AI links
total. All 56 fall inside the four frameworks that make up the Campaign 2 test
split. The audit is disclosed in no markdown file in the repository and has no
test; it was found by an adversarial premortem, not by the campaign.

That matters because the corrections are not neutral with respect to the metric.
49 of 56 move the gold label from a less-linked hub to a more-linked one (median
link degree 3.0 -> 7.5), collapsing 56 links onto 26 distinct hubs. Fine-tuning
learns high-degree hubs best -- they carry more positives and appear in more
batches -- while a zero-shot encoder has no reason to prefer them. So relabelling
toward high-degree hubs mechanically widens a *paired* trained-minus-zero-shot
delta without meaningfully raising absolute accuracy.

Both halves of that prediction hold, and the second is the one worth stating.
Against pristine pre-audit gold the trained model scores 0.5850 rather than
0.5918: the audit is worth well under a point in ABSOLUTE terms. But on the 37
touched items the paired zero-shot scores 0.2162 against 0.5364 on the untouched
ones. The audit barely moves the numerator and guts the baseline, which is
exactly how a paired-improvement gate gets inflated by a labelling change.

This script does not argue that the audit was wrong. Re-reading a link and
finding a better hub is legitimate work, and 24 of the 56 were verdict "wrong"
rather than merely "weak". It argues that an audit applied to 25% of the test
split and to none of the 4,208 general links is a documented asymmetry, and that
the number compared against a 0.10 gate must be reported on both strata.

Read-only. Loads no model, provisions nothing, writes only the JSON report it is
asked for.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Final, TypedDict

import numpy as np

from scripts.phase0.common import (
    AI_FRAMEWORK_ID_MAP,
    AI_FRAMEWORK_NAMES,
    build_evaluation_corpus,
)
from tract.config import PHASE1B_RESULTS_DIR, TRAINING_DIR
from tract.training.evaluate import paired_bootstrap_delta
from tract.training.orchestrate import load_curated_links

logger = logging.getLogger(__name__)

AUDIT_LOG_PATH: Final[Path] = TRAINING_DIR / "audit_corrections_log.json"

# The Campaign 2 held-out test round. Fold directory names are the framework
# names with spaces replaced by underscores; they are listed explicitly rather
# than globbed so that a partially-collected run raises instead of silently
# reporting a delta over whichever folds happen to be on disk.
DEFAULT_RUN: Final[str] = "c2r_TEST_A3_prose_sw_qwen06b"
FOLD_DIRS: Final[dict[str, str]] = {
    "MITRE ATLAS": "fold_MITRE_ATLAS",
    "NIST AI 100-2": "fold_NIST_AI_100-2",
    "OWASP AI Exchange": "fold_OWASP_AI_Exchange",
    "OWASP Top10 for LLM": "fold_OWASP_Top10_for_LLM",
    "OWASP Top10 for ML": "fold_OWASP_Top10_for_ML",
}

# The gate the campaign was pre-registered against (PRD 6.4).
GATE_THRESHOLD: Final[float] = 0.10
N_RESAMPLES: Final[int] = 10_000
BOOTSTRAP_SEED: Final[int] = 42


class ItemRow(TypedDict):
    """One evaluation item with its paired outcome and audit status."""

    framework: str
    fold_dir: str
    section: str
    trained_hit1: int
    zero_shot_hit1: int
    audit_touched: bool


class StratumResult(TypedDict):
    """Paired delta for one stratum of the test corpus."""

    n: int
    n_folds: int
    trained_hit_at_1: float
    zero_shot_hit_at_1: float
    delta_mean: float
    ci_low: float
    ci_high: float
    p_value: float
    p_delta_le_gate: float


def load_audit_touched_keys() -> set[tuple[str, str]]:
    """Return (framework_id, section_name) for every applied gold correction.

    Keyed on section name rather than link identity because the evaluation
    corpus deduplicates by (framework, control_text): several corrected links
    can collapse into one eval item, and an item is "touched" if any of its
    gold labels was rewritten.
    """
    if not AUDIT_LOG_PATH.is_file():
        raise FileNotFoundError(
            f"{AUDIT_LOG_PATH} is missing. This script exists to report the "
            "audit's effect on the gate; without the log there is nothing to "
            "stratify and a delta reported from here would be the undisclosed "
            "pooled number the audit was found in."
        )
    log = json.loads(AUDIT_LOG_PATH.read_text(encoding="utf-8"))
    corrections = log["corrections"]
    applied = log["corrections_applied"]
    if len(corrections) != applied:
        raise ValueError(
            f"audit log says corrections_applied={applied} but carries "
            f"{len(corrections)} correction records. Refusing to stratify "
            "against a log that disagrees with itself."
        )
    return {(c["framework_id"], c["section_name"]) for c in corrections}


def build_rows(run_dir: Path, touched: set[tuple[str, str]]) -> list[ItemRow]:
    """Join per-item paired outcomes to audit status.

    The corpus is rebuilt exactly as `run_experiment` builds it -- from titles,
    before prose is swapped in -- because that is what fixes item identity and
    ordering. `hit1_indicators` is positional, so any divergence in corpus
    construction would silently misalign the join rather than fail. The length
    assertion below is the only thing standing between that and a plausible
    wrong answer, so it raises rather than warns.
    """
    corpus = build_evaluation_corpus(load_curated_links(), AI_FRAMEWORK_NAMES, {})
    by_framework: dict[str, list[Any]] = defaultdict(list)
    for item in corpus:
        by_framework[item.framework_name].append(item)

    rows: list[ItemRow] = []
    for framework, fold_dir in FOLD_DIRS.items():
        result_path = run_dir / fold_dir / "fold_result.json"
        if not result_path.is_file():
            raise FileNotFoundError(
                f"{result_path} is missing. A delta over four of five folds is "
                "not the campaign's number; collect the run or point --run "
                "somewhere complete."
            )
        fold = json.loads(result_path.read_text(encoding="utf-8"))
        trained = fold["hit1_indicators"]
        zero_shot = fold["zero_shot"]["hit1_indicators"]
        items = by_framework[framework]
        if not len(trained) == len(zero_shot) == len(items):
            raise ValueError(
                f"{framework}: fold_result has {len(trained)} trained and "
                f"{len(zero_shot)} zero-shot indicators but the rebuilt corpus "
                f"has {len(items)} items. The positional join is invalid -- the "
                "corpus on disk is not the corpus this run scored."
            )
        framework_id = AI_FRAMEWORK_ID_MAP[framework]
        for item, t, z in zip(items, trained, zero_shot, strict=True):
            rows.append(ItemRow(
                framework=framework,
                fold_dir=fold_dir,
                section=item.control_text,
                trained_hit1=int(t),
                zero_shot_hit1=int(z),
                audit_touched=(framework_id, item.control_text) in touched,
            ))
    return rows


def score_stratum(rows: list[ItemRow], label: str) -> StratumResult:
    """Fold-stratified paired bootstrap over one stratum.

    Uses the campaign's own `paired_bootstrap_delta` so the reported interval is
    produced by the same code that produced the headline, then re-derives the
    resample distribution once more to report P(delta <= gate) -- which the
    campaign's function does not return, and which is the quantity the gate
    decision actually turns on.
    """
    by_fold: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for row in rows:
        zero_shot, trained = by_fold[row["fold_dir"]]
        zero_shot.append(float(row["zero_shot_hit1"]))
        trained.append(float(row["trained_hit1"]))

    fold_a = [np.array(pair[0]) for pair in by_fold.values() if pair[0]]
    fold_b = [np.array(pair[1]) for pair in by_fold.values() if pair[1]]
    stats = paired_bootstrap_delta(
        fold_a, fold_b, n_resamples=N_RESAMPLES, seed=BOOTSTRAP_SEED,
    )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    per_fold_deltas = [b - a for a, b in zip(fold_a, fold_b, strict=True)]
    resampled = np.empty(N_RESAMPLES, dtype=float)
    for i in range(N_RESAMPLES):
        drawn = [f[rng.integers(0, len(f), len(f))] for f in per_fold_deltas]
        resampled[i] = np.concatenate(drawn).mean()

    n = sum(len(f) for f in fold_a)
    result = StratumResult(
        n=n,
        n_folds=len(fold_a),
        trained_hit_at_1=sum(r["trained_hit1"] for r in rows) / n,
        zero_shot_hit_at_1=sum(r["zero_shot_hit1"] for r in rows) / n,
        delta_mean=stats["delta_mean"],
        ci_low=stats["ci_low"],
        ci_high=stats["ci_high"],
        p_value=stats["p_value"],
        p_delta_le_gate=float((resampled <= GATE_THRESHOLD).mean()),
    )
    logger.info(
        "%-10s n=%3d  trained=%.4f  zero-shot=%.4f  delta=%+.4f "
        "[%+.4f, %+.4f]  P(delta<=%.2f)=%.3f",
        label, result["n"], result["trained_hit_at_1"],
        result["zero_shot_hit_at_1"], result["delta_mean"],
        result["ci_low"], result["ci_high"], GATE_THRESHOLD,
        result["p_delta_le_gate"],
    )
    return result


def describe_corrections() -> dict[str, Any]:
    """Quantify the direction of the audit's relabelling.

    Link degree is counted over the curated file the corrections were applied
    to, so `new_cre_id` degree already includes the correction itself. That
    inflates the new-hub degree by one per correction and is left uncorrected
    deliberately: it is the degree the trainer actually saw.
    """
    log = json.loads(AUDIT_LOG_PATH.read_text(encoding="utf-8"))
    corrections = log["corrections"]
    degree = Counter(link.cre_id for link in load_curated_links())
    old = [degree[c["old_cre_id"]] for c in corrections]
    new = [degree[c["new_cre_id"]] for c in corrections]
    return {
        "n_corrections": len(corrections),
        "by_framework": dict(Counter(c["framework_id"] for c in corrections)),
        "by_verdict": dict(Counter(c["verdict"] for c in corrections)),
        "moved_to_higher_degree_hub": sum(1 for o, n in zip(old, new, strict=True) if n > o),
        "moved_to_lower_degree_hub": sum(1 for o, n in zip(old, new, strict=True) if n < o),
        "median_degree_before": float(np.median(old)),
        "median_degree_after": float(np.median(new)),
        "mean_degree_before": float(np.mean(old)),
        "mean_degree_after": float(np.mean(new)),
        "distinct_destination_hubs": len({c["new_cre_id"] for c in corrections}),
        "ai_links_total": log["ai_links_curated"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN,
                        help="Result directory under results/phase1b/.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Write the full report as JSON to this path.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    run_dir = PHASE1B_RESULTS_DIR / args.run
    touched = load_audit_touched_keys()
    rows = build_rows(run_dir, touched)

    n_touched = sum(r["audit_touched"] for r in rows)
    logger.info("Test corpus: %d items, %d audit-touched (%.1f%%)",
                len(rows), n_touched, 100 * n_touched / len(rows))

    report = {
        "run": args.run,
        "gate_threshold": GATE_THRESHOLD,
        "n_items": len(rows),
        "n_audit_touched": n_touched,
        "corrections": describe_corrections(),
        "strata": {
            "pooled": score_stratum(rows, "POOLED"),
            "audit_touched": score_stratum(
                [r for r in rows if r["audit_touched"]], "TOUCHED"),
            "audit_untouched": score_stratum(
                [r for r in rows if not r["audit_touched"]], "UNTOUCHED"),
        },
    }

    if args.out:
        args.out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
