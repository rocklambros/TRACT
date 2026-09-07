"""Test the *mechanism* the audit stratification was explained by, rather than
re-reporting the stratification itself.

WHY THIS EXISTS. `scripts/analysis/audit_stratified_delta.py` established that
TRACT's own link audit touched 25% of the Campaign 2/3 test split, and that the
paired delta on those items is far larger than on the rest. Three artifacts then
went further and explained *why*, in identical terms:

    docs/campaign2-results.md §13, docs/campaign3-rebaseline.md §4, and the
    docstring of audit_stratified_delta.py itself:

        "49 of 56 move the gold label from a less-linked hub to a more-linked
        one (median link degree 3.0 -> 7.5) [...] Fine-tuning learns high-degree
        hubs best -- they carry more positives and appear in more batches --
        while a zero-shot encoder has no reason to prefer them. So relabelling
        toward high-degree hubs mechanically widens a *paired* delta."

That explanation is load-bearing for a spending decision. It is the reason the
inflation was ruled "arithmetic, not provenance" -- widening the delta "whoever
picked the destination" -- which in turn is the reason paid human curation was
treated as safe for the gate. If the mechanism is degree, curation is safe,
because annotators picking better hubs would inflate a metric we already
understand. If it is *discretionary reassignment*, curation reproduces the
inflation and the gate measures the annotators.

The explanation was never tested. It was inferred from two facts inside the
touched stratum (degree went up; zero-shot is low there), which is consistent
with the degree story but equally consistent with several others. This script
tests the three predictions the degree story makes that nothing else does.

    H1. Degree predicts the paired delta GENERALLY. If fine-tuning learns
        high-degree hubs and zero-shot does not prefer them, that holds for
        every item, not only relabelled ones. So among AUDIT-UNTOUCHED items,
        high-degree gold should show a larger delta and a depressed zero-shot.

    H2. The effect is DOSE-DEPENDENT. A correction that raises degree by 10
        should widen the delta more than one that raises it by 1.

    H3. The effect is INDIFFERENT TO VERDICT. Degree is arithmetic, so it
        should not care whether the auditor called the old link "wrong"
        (a genuine error) or "weak" (defensible, but a worse fit than another).

H1 is the decisive one: it is measured entirely on items the audit never
touched, so it cannot be confounded by the audit.

This script also reports the fold composition of the published primary, because
the untouched stratum is not a random sample of the corpus -- one framework
contributes 0 touched items and over half the primary's denominator.

Read-only. Loads no model, provisions nothing, writes only the JSON report it is
asked for. Deterministic: fixed seed, sorted output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Final, TypedDict

import numpy as np

from scripts.phase0.common import (
    AI_FRAMEWORK_ID_MAP,
    AI_FRAMEWORK_NAMES,
    EvalItem,
    build_evaluation_corpus,
    load_curated_links,
)
from tract.config import PHASE1B_RESULTS_DIR, TRAINING_DIR

logger = logging.getLogger(__name__)

AUDIT_LOG_PATH: Final[Path] = TRAINING_DIR / "audit_corrections_log.json"

DEFAULT_RUN: Final[str] = "c3_TEST_A3_prose_sw_qwen06b_seq1024"

# Same explicit fold map as audit_stratified_delta.py, and for the same reason:
# a globbed partial run must raise rather than silently report a delta over
# whichever folds happen to be on disk.
FOLD_DIRS: Final[dict[str, str]] = {
    "MITRE ATLAS": "fold_MITRE_ATLAS",
    "NIST AI 100-2": "fold_NIST_AI_100-2",
    "OWASP AI Exchange": "fold_OWASP_AI_Exchange",
    "OWASP Top10 for LLM": "fold_OWASP_Top10_for_LLM",
    "OWASP Top10 for ML": "fold_OWASP_Top10_for_ML",
}

# 100,000, not 10,000. At 10k the sibling module reproduced its reference
# contrast on 11 of 12 seeds; at 100k, on 12 of 12. This module publishes
# intervals into a results document, so it takes the count shown to be stable.
N_RESAMPLES: Final[int] = 100_000
BOOTSTRAP_SEED: Final[int] = 42
CI_LOW_PCT: Final[float] = 2.5
CI_HIGH_PCT: Final[float] = 97.5

# A separation is called "established" only if the resampled difference excludes
# zero at this level. Everything above it is reported as suggestive. The strata
# here are small (n=17 and n=20 for the verdict split), so this guard is what
# keeps a wide interval from being read as a finding.
ALPHA: Final[float] = 0.05


class ProbeRow(TypedDict):
    """One evaluation item joined to its audit status and gold-hub degree."""

    framework: str
    fold_dir: str
    section: str
    trained_hit1: int
    zero_shot_hit1: int
    audit_touched: bool
    # Degree of the most-linked valid gold hub. hit@1 counts a hit against ANY
    # member of valid_hub_ids, so the max is the learnability of the easiest
    # valid target. deg_primary is carried alongside as a robustness check;
    # the two differ on 5 of 147 items and the tests agree under both.
    gold_degree_max: int
    gold_degree_primary: int
    n_valid_hubs: int
    # None for untouched items.
    verdict: str | None
    degree_change: float | None


# The two gold-degree definitions H1 is measured under. They differ on 5 of 147
# items and the test agrees under both, which is the point of carrying both.
DEGREE_ACCESSORS: Final[tuple[tuple[str, Callable[["ProbeRow"], int]], ...]] = (
    ("gold_degree_max", lambda row: row["gold_degree_max"]),
    ("gold_degree_primary", lambda row: row["gold_degree_primary"]),
)


class DeltaStat(TypedDict):
    """Observed fold-stratified paired delta with a bootstrap interval."""

    n: int
    zero_shot_hit_at_1: float
    trained_hit_at_1: float
    delta_mean: float
    ci_low: float
    ci_high: float


class AuditEntry(TypedDict):
    """Every correction the audit applied to one deduplicated eval item."""

    verdicts: list[str]
    degree_changes: list[float]


class ContrastStat(TypedDict):
    """Difference between two strata's deltas, with the interval on the diff."""

    label_a: str
    label_b: str
    difference: float
    ci_low: float
    ci_high: float
    p_difference_le_zero: float
    established_at_alpha: bool


def _atomic_write_json(path: Path, payload: object) -> None:
    """Write JSON via temp-file-then-rename so a crash cannot truncate a report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_audit_index() -> dict[tuple[str, str], AuditEntry]:
    """Index corrections by (framework_id, section_name).

    Keyed on section name for the same reason audit_stratified_delta.py is: the
    evaluation corpus deduplicates by (framework, control_text), so several
    corrections can collapse onto one eval item. Such an item carries every
    verdict and every degree change that landed on it.
    """
    if not AUDIT_LOG_PATH.is_file():
        raise FileNotFoundError(
            f"{AUDIT_LOG_PATH} is missing. This script tests the mechanism the "
            "audit effect was attributed to; without the log there is nothing "
            "to test and any conclusion drawn here would be unfounded."
        )
    log = json.loads(AUDIT_LOG_PATH.read_text(encoding="utf-8"))
    corrections = log["corrections"]

    # The audit made 65 decisions across three lists, and this index is built
    # from one of them. Reconciling all three is what stops the next audit
    # changing the corpus in a way the stratification cannot see: an EXCLUSION
    # deletes an eval item outright (the one on record removed an OWASP AI
    # Exchange link) and a KEPT-WEAK link was inspected and affirmed, yet both
    # look identical to "untouched" downstream.
    #
    # Only `corrections` feeds the touched set -- exclusions have no item left
    # to mark and kept-weak links were not relabelled -- but a log whose counts
    # disagree with its own lists is not one to stratify against.
    checks = (
        ("corrections", len(corrections), log["corrections_applied"]),
        ("exclusions", len(log["exclusions"]), log["links_excluded"]),
        ("kept_weak", len(log["kept_weak"]), log["weak_kept_as_is"]),
    )
    mismatched = [
        f"{name}: {actual} records against {declared} declared"
        for name, actual, declared in checks if actual != declared
    ]
    if mismatched:
        raise ValueError(
            "audit log does not reconcile -- " + "; ".join(mismatched)
            + ". Refusing to probe a log that disagrees with itself."
        )
    degree = Counter(link.cre_id for link in load_curated_links())
    index: dict[tuple[str, str], AuditEntry] = defaultdict(
        lambda: AuditEntry(verdicts=[], degree_changes=[]),
    )
    for correction in corrections:
        key = (correction["framework_id"], correction["section_name"])
        index[key]["verdicts"].append(str(correction["verdict"]))
        index[key]["degree_changes"].append(
            float(degree[correction["new_cre_id"]] - degree[correction["old_cre_id"]]),
        )
    return dict(index)


def build_rows(run_dir: Path) -> list[ProbeRow]:
    """Join per-item paired outcomes to audit status and gold-hub degree.

    The corpus is rebuilt exactly as `run_experiment` builds it, because
    `hit1_indicators` is positional and any divergence would misalign the join
    silently. The length assertion raises rather than warns for that reason.
    """
    links = load_curated_links()
    degree = Counter(link.cre_id for link in links)
    corpus = build_evaluation_corpus(links, AI_FRAMEWORK_NAMES, {})
    audit = load_audit_index()

    by_framework: dict[str, list[EvalItem]] = defaultdict(list)
    for item in corpus:
        by_framework[item.framework_name].append(item)

    rows: list[ProbeRow] = []
    for framework, fold_dir in FOLD_DIRS.items():
        result_path = run_dir / fold_dir / "fold_result.json"
        if not result_path.is_file():
            raise FileNotFoundError(
                f"{result_path} is missing. A mechanism probe over a partial "
                "run is not a probe of this campaign; collect the run or point "
                "--run somewhere complete."
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
        for item, hit_t, hit_z in zip(items, trained, zero_shot, strict=True):
            key = (framework_id, item.control_text)
            entry = audit.get(key)
            verdicts = entry["verdicts"] if entry else []
            changes = entry["degree_changes"] if entry else []
            degrees = [degree[hub] for hub in item.valid_hub_ids]
            rows.append(ProbeRow(
                framework=framework,
                fold_dir=fold_dir,
                section=item.control_text,
                trained_hit1=int(hit_t),
                zero_shot_hit1=int(hit_z),
                audit_touched=entry is not None,
                gold_degree_max=max(degrees),
                gold_degree_primary=degree[item.ground_truth_hub_id],
                n_valid_hubs=len(item.valid_hub_ids),
                # An item is "wrong" if ANY correction on it was a genuine
                # error; otherwise every correction was discretionary.
                verdict=("wrong" if "wrong" in verdicts else "weak") if entry else None,
                degree_change=float(np.mean(changes)) if changes else None,
            ))
    return rows


def _stratum_rng(
    rows: list[ProbeRow], parent: np.random.Generator, tag: int = 0,
) -> np.random.Generator:
    """A generator determined by the stratum's contents, not by call order.

    Ported from `gate_rule_candidates`, where the defect it fixes was measured:
    one generator threaded through several strata makes each stratum's draws
    depend on how many draws every prior stratum consumed, and the reported
    interval then moves with argument order and with any unrelated upstream
    draw. There that printed +0.4595 where the 500k reference is +0.4324.

    This module had the same shape and publishes into
    docs/campaign3-audit-mechanism.md, so it takes the same fix.
    """
    payload = "|".join(
        f"{r['fold_dir']}\x1f{r['section']}\x1f"
        f"{r['trained_hit1']}\x1f{r['zero_shot_hit1']}"
        for r in rows
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()[:16]
    # getattr rather than attribute access: numpy types seed_seq as the
    # ISeedSequence protocol, which does not declare `entropy` even though
    # SeedSequence carries it. The isinstance guard is what makes the dynamic
    # read safe, and it also covers a caller who built the generator from a
    # bare BitGenerator.
    entropy = getattr(parent.bit_generator.seed_seq, "entropy", None)
    base = entropy if isinstance(entropy, int) else BOOTSTRAP_SEED
    return np.random.default_rng([base, int.from_bytes(digest, "big"), tag])


def delta_distribution(
    rows: list[ProbeRow], rng: np.random.Generator,
) -> tuple[float, np.ndarray[tuple[int, ...], np.dtype[np.float64]]]:
    """Fold-stratified paired delta and its bootstrap resample distribution.

    Resamples within fold and concatenates, matching the campaign's own
    `paired_bootstrap_delta`, so intervals here are comparable to published
    ones rather than being a second, differently-shaped estimator.

    `rng` is a PARENT: the stream actually drawn from is derived from the
    stratum's own contents via `_stratum_rng`, so this stratum's interval is
    the same whether it is computed first, last, or after an unrelated draw.
    """
    if not rows:
        raise ValueError("Refusing to bootstrap an empty stratum.")
    by_fold: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_fold[row["fold_dir"]].append(
            float(row["trained_hit1"] - row["zero_shot_hit1"]),
        )
    folds = [np.array(v, dtype=float) for v in by_fold.values()]
    observed = float(np.concatenate(folds).mean())
    stream = _stratum_rng(rows, rng)
    # Vectorised per fold and accumulated as sums, rather than concatenating
    # inside a Python loop. Summing then dividing by the total item count is
    # exactly the mean of the concatenation, and at N_RESAMPLES=100,000 the
    # loop form is too slow to run.
    total = sum(len(f) for f in folds)
    acc = np.zeros(N_RESAMPLES, dtype=float)
    for fold in folds:
        idx = stream.integers(0, len(fold), (N_RESAMPLES, len(fold)))
        acc += fold[idx].sum(axis=1)
    return observed, acc / total


def score(rows: list[ProbeRow], rng: np.random.Generator) -> DeltaStat:
    """Summarise one stratum."""
    observed, resampled = delta_distribution(rows, rng)
    n = len(rows)
    return DeltaStat(
        n=n,
        zero_shot_hit_at_1=sum(r["zero_shot_hit1"] for r in rows) / n,
        trained_hit_at_1=sum(r["trained_hit1"] for r in rows) / n,
        delta_mean=observed,
        ci_low=float(np.percentile(resampled, CI_LOW_PCT)),
        ci_high=float(np.percentile(resampled, CI_HIGH_PCT)),
    )


def contrast(
    a: list[ProbeRow], b: list[ProbeRow],
    label_a: str, label_b: str, rng: np.random.Generator,
) -> ContrastStat:
    """Bootstrap the DIFFERENCE between two strata's deltas.

    Reporting two overlapping intervals and calling the gap real is the error
    this function exists to prevent: the interval that matters is the one on
    the difference.
    """
    obs_a, res_a = delta_distribution(a, rng)
    obs_b, res_b = delta_distribution(b, rng)
    diff = res_a - res_b
    p_le_zero = float((diff <= 0.0).mean())
    return ContrastStat(
        label_a=label_a,
        label_b=label_b,
        difference=obs_a - obs_b,
        ci_low=float(np.percentile(diff, CI_LOW_PCT)),
        ci_high=float(np.percentile(diff, CI_HIGH_PCT)),
        p_difference_le_zero=p_le_zero,
        established_at_alpha=p_le_zero < ALPHA,
    )


def _log_stat(label: str, stat: DeltaStat) -> None:
    logger.info(
        "  %-42s n=%3d  zs=%.4f  tr=%.4f  delta=%+.4f [%+.4f, %+.4f]",
        label, stat["n"], stat["zero_shot_hit_at_1"], stat["trained_hit_at_1"],
        stat["delta_mean"], stat["ci_low"], stat["ci_high"],
    )


def _log_contrast(stat: ContrastStat) -> None:
    logger.info(
        "  %s minus %s = %+.4f [%+.4f, %+.4f]  P(<=0)=%.3f  -> %s",
        stat["label_a"], stat["label_b"], stat["difference"],
        stat["ci_low"], stat["ci_high"], stat["p_difference_le_zero"],
        "ESTABLISHED" if stat["established_at_alpha"] else "SUGGESTIVE ONLY",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN,
                        help="Result directory under results/phase1b/.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Write the full report as JSON to this path.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = build_rows(PHASE1B_RESULTS_DIR / args.run)
    touched = [r for r in rows if r["audit_touched"]]
    untouched = [r for r in rows if not r["audit_touched"]]
    logger.info("Corpus: %d items, %d touched (%.1f%%)",
                len(rows), len(touched), 100 * len(touched) / len(rows))

    report: dict[str, object] = {"run": args.run, "n_items": len(rows)}

    # --- H1: does degree predict the delta on items the audit never touched? --
    logger.info("")
    logger.info("H1  degree predicts delta generally (measured on UNTOUCHED only)")
    h1: dict[str, object] = {}
    # Accessor functions rather than a variable TypedDict key. Indexing a
    # TypedDict with a loop variable needs a literal-required ignore, and
    # whether that ignore is *itself* an error depends on the mypy version --
    # 1.11 requires it, 2.2 rejects it as unused. A lambda is version-agnostic
    # and drops the suppression entirely.
    for key, degree_of in DEGREE_ACCESSORS:
        median = float(np.median([degree_of(r) for r in untouched]))
        high = [r for r in untouched if degree_of(r) > median]
        low = [r for r in untouched if degree_of(r) <= median]
        # Scored ONCE and reused. Calling score() twice on the same stratum
        # would advance the shared generator and log an interval that differs
        # from the one written to the report.
        stat_high, stat_low = score(high, rng), score(low, rng)
        _log_stat(f"{key} > {median:.1f} (high)", stat_high)
        _log_stat(f"{key} <= {median:.1f} (low)", stat_low)
        c = contrast(high, low, f"{key}-high", f"{key}-low", rng)
        _log_contrast(c)
        h1[key] = {"median": median, "high": stat_high,
                   "low": stat_low, "contrast": c}
    report["h1_degree_predicts_delta_on_untouched"] = h1

    # --- H2: is the effect dose-dependent in the size of the degree change? ---
    logger.info("")
    logger.info("H2  effect scales with the SIZE of the degree increase (TOUCHED only)")
    changes = np.array([r["degree_change"] for r in touched], dtype=float)
    deltas = np.array(
        [r["trained_hit1"] - r["zero_shot_hit1"] for r in touched], dtype=float,
    )
    pearson = float(np.corrcoef(changes, deltas)[0, 1])
    median_change = float(np.median(changes))
    big = [r for r in touched if float(r["degree_change"] or 0.0) > median_change]
    small = [r for r in touched if float(r["degree_change"] or 0.0) <= median_change]
    stat_big, stat_small = score(big, rng), score(small, rng)
    _log_stat(f"degree change > {median_change:+.1f} (big)", stat_big)
    _log_stat(f"degree change <= {median_change:+.1f} (small)", stat_small)
    c2 = contrast(big, small, "big-degree-change", "small-degree-change", rng)
    _log_contrast(c2)
    logger.info("  Pearson r(degree change, per-item delta) = %+.4f (n=%d)",
                pearson, len(touched))
    report["h2_dose_response"] = {
        "pearson_r": pearson, "median_degree_change": median_change,
        "big": stat_big, "small": stat_small, "contrast": c2,
    }

    # --- H3: is the effect indifferent to the auditor's verdict? --------------
    logger.info("")
    logger.info("H3  effect is indifferent to verdict (arithmetic, not judgement)")
    wrong = [r for r in touched if r["verdict"] == "wrong"]
    weak = [r for r in touched if r["verdict"] == "weak"]
    stat_wrong, stat_weak = score(wrong, rng), score(weak, rng)
    _log_stat("verdict=wrong (genuine error)", stat_wrong)
    _log_stat("verdict=weak (discretionary reassignment)", stat_weak)
    c3 = contrast(weak, wrong, "verdict-weak", "verdict-wrong", rng)
    _log_contrast(c3)
    report["h3_verdict_split"] = {
        "wrong": stat_wrong, "weak": stat_weak, "contrast": c3,
        "median_degree_change_wrong": float(
            np.median([r["degree_change"] or 0.0 for r in wrong])),
        "median_degree_change_weak": float(
            np.median([r["degree_change"] or 0.0 for r in weak])),
    }

    # --- composition: what is the published primary actually made of? ---------
    logger.info("")
    logger.info("Fold composition of the published primary (the untouched stratum)")
    composition = Counter(r["fold_dir"] for r in untouched)
    for fold, n in composition.most_common():
        logger.info("  %-26s %3d  (%.1f%% of the primary)",
                    fold, n, 100 * n / len(untouched))
    folds_with_both = sorted(
        {r["fold_dir"] for r in touched} & {r["fold_dir"] for r in untouched},
    )
    matched = [r for r in untouched if r["fold_dir"] in folds_with_both]
    stat_all, stat_matched = score(untouched, rng), score(matched, rng)
    _log_stat("untouched, as published (all folds)", stat_all)
    _log_stat("untouched, fold-matched to touched", stat_matched)
    c4 = contrast(touched, matched, "touched", "untouched-fold-matched", rng)
    _log_contrast(c4)
    report["composition"] = {
        "primary_by_fold": dict(composition),
        "folds_with_both_strata": folds_with_both,
        "folds_with_no_touched_items": sorted(
            {r["fold_dir"] for r in untouched} - {r["fold_dir"] for r in touched},
        ),
        "untouched_all_folds": stat_all,
        "untouched_fold_matched": stat_matched,
        "contrast_fold_matched": c4,
    }

    if args.out:
        _atomic_write_json(args.out, report)
        logger.info("")
        logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
