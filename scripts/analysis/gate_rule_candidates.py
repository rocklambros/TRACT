"""Evaluate candidate replacements for the CAMPAIGN3 §1.2 pooling rule.

WHY THIS EXISTS. §1.2 pre-registers when a Tier-2 (curated) stratum may share a
denominator with Tier-1:

    "The gate is decided on the pooled estimate only if the two strata are
    consistent -- their 95% intervals must overlap."

`docs/campaign3-audit-mechanism.md` §6b shows that rule admits the one
relabelling we can actually observe: Tier-1 +0.1000 [0.0000, 0.2000] and the
audit-relabelled stratum +0.2703 [+0.1081, +0.4324] overlap, so the rule pools
them and reports +0.1429 against a Tier-1 truth of +0.1000.

A replacement has to be chosen before annotation begins, and choosing it on
"sounds stricter" is how the current rule got written. So every candidate here
is scored on BOTH failure directions, because a rule that never pools is safe
and worthless -- it discards exactly the extra n that curation is being funded
to buy:

    SENSITIVITY  Does it REFUSE the audit-touched contrast? That contrast is a
                 real relabelling with a known 2:1 baseline asymmetry, and is
                 the best available stand-in for a curated stratum.

    SPECIFICITY  Does it still PERMIT pooling two strata drawn from the SAME
                 population? Simulated by splitting the audit-untouched items
                 into a pseudo-Tier-2 of n=37 (matching the real small stratum)
                 and a pseudo-Tier-1 of the remaining 73, repeatedly. A rule
                 that refuses here is refusing free power.

A rule must do both. Reporting only sensitivity is how you end up with "never
pool", which is R3 below and is included precisely so its cost is visible.

Read-only. Loads no model, provisions nothing. Deterministic: fixed seed.
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable, Final, TypedDict

import numpy as np

from scripts.analysis.audit_mechanism_probe import (
    ProbeRow,
    _atomic_write_json,
    build_rows,
)
from tract.config import PHASE1B_RESULTS_DIR

logger = logging.getLogger(__name__)

DEFAULT_RUN: Final[str] = "c3_TEST_A3_prose_sw_qwen06b_seq1024"

N_RESAMPLES: Final[int] = 10_000
# The null simulation runs a full bootstrap per replicate, so it uses a smaller
# resample count. 2,000 is enough to place a permit/refuse boundary that is
# never near the decision threshold for any candidate here.
N_RESAMPLES_NULL: Final[int] = 2_000
N_NULL_REPLICATES: Final[int] = 200
SEED: Final[int] = 42

CI_LOW_PCT: Final[float] = 2.5
CI_HIGH_PCT: Final[float] = 97.5

# R2 (equivalence): the two strata are declared consistent only if the whole
# interval on their difference sits inside +/- this margin. 0.05 is half the
# 0.10 gate: a between-stratum difference at half the effect the gate is trying
# to detect is not a rounding error, it is a competing explanation.
EQUIVALENCE_MARGIN: Final[float] = 0.05

# R5 (baseline symmetry): refuse when the two strata's ZERO-SHOT rates differ by
# more than this. Aimed at the measured mechanism rather than at the delta --
# relabelling cost the zero-shot arm -0.3381 and the trained arm -0.1678, and it
# is that asymmetry, not the delta gap, that makes the pooled figure wrong.
BASELINE_MARGIN: Final[float] = 0.10

# R6: significance level for "these two strata have the same zero-shot
# baseline". Sets R6's null permit rate at ~1 - alpha by construction, which is
# why R6 needs no margin calibrated against this particular dataset.
BASELINE_ALPHA: Final[float] = 0.05

# Margins swept for the R4 operating-point table.
BASELINE_MARGIN_SWEEP: Final[tuple[float, ...]] = (
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
)


class Contrast(TypedDict):
    """Everything a pooling rule is allowed to look at."""

    delta_a: float
    delta_b: float
    ci_a: tuple[float, float]
    ci_b: tuple[float, float]
    diff: float
    diff_ci: tuple[float, float]
    zero_shot_a: float
    zero_shot_b: float
    baseline_diff: float
    baseline_diff_ci: tuple[float, float]
    # Two-sided bootstrap p-value for "the two strata have the same zero-shot
    # baseline". Small = the baselines really do differ.
    baseline_p_value: float


# A rule returns True to PERMIT pooling, False to REFUSE.
Rule = Callable[[Contrast], bool]


def bootstrap_deltas(
    rows: list[ProbeRow], n_resamples: int, rng: np.random.Generator,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Vectorised fold-stratified bootstrap of the paired delta.

    Equivalent to resampling within each fold and taking the mean over the
    concatenation: the mean of the pooled draw is the summed per-fold draws
    divided by the total item count. Vectorised because the null simulation
    needs a few hundred of these.
    """
    by_fold: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_fold[row["fold_dir"]].append(
            float(row["trained_hit1"] - row["zero_shot_hit1"]),
        )
    folds = [np.array(v, dtype=float) for v in by_fold.values()]
    total = sum(len(f) for f in folds)
    acc = np.zeros(n_resamples, dtype=float)
    for fold in folds:
        idx = rng.integers(0, len(fold), (n_resamples, len(fold)))
        acc += fold[idx].sum(axis=1)
    return acc / total


def bootstrap_baselines(
    rows: list[ProbeRow], n_resamples: int, rng: np.random.Generator,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Fold-stratified bootstrap of the ZERO-SHOT rate.

    Same resampling scheme as the delta, so the two are directly comparable.
    This exists because the relabelling mechanism shows up roughly twice as
    strongly in the baseline (-0.3381) as in the delta (+0.1703) -- testing the
    baseline buys about double the effect size against the same noise.
    """
    by_fold: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_fold[row["fold_dir"]].append(float(row["zero_shot_hit1"]))
    folds = [np.array(v, dtype=float) for v in by_fold.values()]
    total = sum(len(f) for f in folds)
    acc = np.zeros(n_resamples, dtype=float)
    for fold in folds:
        idx = rng.integers(0, len(fold), (n_resamples, len(fold)))
        acc += fold[idx].sum(axis=1)
    return acc / total


def observed_delta(rows: list[ProbeRow]) -> float:
    return float(np.mean([r["trained_hit1"] - r["zero_shot_hit1"] for r in rows]))


def zero_shot_rate(rows: list[ProbeRow]) -> float:
    return float(np.mean([r["zero_shot_hit1"] for r in rows]))


def make_contrast(
    a: list[ProbeRow], b: list[ProbeRow], n_resamples: int,
    rng: np.random.Generator,
) -> Contrast:
    """Build the contrast a pooling rule sees. `a` is Tier-1, `b` is Tier-2."""
    res_a = bootstrap_deltas(a, n_resamples, rng)
    res_b = bootstrap_deltas(b, n_resamples, rng)
    diff = res_b - res_a
    base_a = bootstrap_baselines(a, n_resamples, rng)
    base_b = bootstrap_baselines(b, n_resamples, rng)
    base_diff = base_b - base_a
    # Two-sided: how much of the resampled baseline difference sits on the far
    # side of zero from the observed one.
    p_base = float(2.0 * min((base_diff <= 0.0).mean(), (base_diff >= 0.0).mean()))
    return Contrast(
        delta_a=observed_delta(a),
        delta_b=observed_delta(b),
        ci_a=(float(np.percentile(res_a, CI_LOW_PCT)),
              float(np.percentile(res_a, CI_HIGH_PCT))),
        ci_b=(float(np.percentile(res_b, CI_LOW_PCT)),
              float(np.percentile(res_b, CI_HIGH_PCT))),
        diff=observed_delta(b) - observed_delta(a),
        diff_ci=(float(np.percentile(diff, CI_LOW_PCT)),
                 float(np.percentile(diff, CI_HIGH_PCT))),
        zero_shot_a=zero_shot_rate(a),
        zero_shot_b=zero_shot_rate(b),
        baseline_diff=zero_shot_rate(b) - zero_shot_rate(a),
        baseline_diff_ci=(float(np.percentile(base_diff, CI_LOW_PCT)),
                          float(np.percentile(base_diff, CI_HIGH_PCT))),
        baseline_p_value=p_base,
    )


# --- the candidates ---------------------------------------------------------

def r0_status_quo(c: Contrast) -> bool:
    """CAMPAIGN3 §1.2 as written: pool iff the two 95% intervals overlap."""
    return max(c["ci_a"][0], c["ci_b"][0]) <= min(c["ci_a"][1], c["ci_b"][1])


def r1_difference_includes_zero(c: Contrast) -> bool:
    """Pool iff the interval on the DIFFERENCE covers zero.

    The obvious repair, and included to show it is not one: failing to prove a
    difference is not evidence of consistency, and with n=37 the difference
    interval is wide enough to cover zero across a large range of real gaps.
    """
    return c["diff_ci"][0] <= 0.0 <= c["diff_ci"][1]


def r2_equivalence(c: Contrast) -> bool:
    """Pool iff the whole difference interval sits within +/- EQUIVALENCE_MARGIN.

    The burden of proof is inverted relative to R0 and R1: consistency must be
    demonstrated, not merely un-refuted. This is the standard equivalence-test
    framing (TOST), and it is the right one whenever the risky action -- pooling
    -- is the one that happens by default when the test is uninformative.
    """
    return (c["diff_ci"][0] >= -EQUIVALENCE_MARGIN
            and c["diff_ci"][1] <= EQUIVALENCE_MARGIN)


def r3_never_pool(c: Contrast) -> bool:
    """Never pool; gate on Tier-1 alone. Included to price the safe option."""
    return False


def r4_baseline_symmetry(c: Contrast) -> bool:
    """Pool iff the two strata's ZERO-SHOT rates agree within BASELINE_MARGIN.

    Targets the measured mechanism rather than its symptom. A relabelling that
    inflates a paired delta does so by moving the baseline; if both strata's
    baselines agree, the paired comparison rests on comparable ground.
    """
    return abs(c["zero_shot_a"] - c["zero_shot_b"]) <= BASELINE_MARGIN


def r5_equivalence_and_baseline(c: Contrast) -> bool:
    """R2 AND R4: consistent deltas AND comparable baselines."""
    return r2_equivalence(c) and r4_baseline_symmetry(c)


def r6_baseline_not_significantly_different(c: Contrast) -> bool:
    """Pool unless the two strata's ZERO-SHOT baselines differ significantly.

    Same weak burden of proof as R1 -- pool unless a difference is demonstrated
    -- but applied to the baseline instead of the delta, which is what makes it
    work. The relabelling moves the baseline about twice as far as it moves the
    delta, so the same test that is underpowered on the delta (z ~ 1.7) is
    decisive on the baseline (z ~ 3.3).

    Its specificity is set by construction: under a true null it permits about
    1 - BASELINE_ALPHA of the time, with no margin to tune against the data.
    """
    return c["baseline_p_value"] >= BASELINE_ALPHA


CANDIDATES: Final[dict[str, tuple[str, Rule]]] = {
    "R0": ("status quo: 95% intervals overlap", r0_status_quo),
    "R1": ("difference interval covers zero", r1_difference_includes_zero),
    "R2": (f"equivalence: difference within +/-{EQUIVALENCE_MARGIN}", r2_equivalence),
    "R3": ("never pool (Tier-1 only)", r3_never_pool),
    "R4": (f"baseline symmetry within {BASELINE_MARGIN}", r4_baseline_symmetry),
    "R5": ("R2 and R4 together", r5_equivalence_and_baseline),
    "R6": (f"baseline difference n.s. at p>={BASELINE_ALPHA}",
           r6_baseline_not_significantly_different),
}


def split_null(
    rows: list[ProbeRow], n_small: int, rng: np.random.Generator,
) -> tuple[list[ProbeRow], list[ProbeRow]]:
    """Split one population into pseudo-Tier-1 and pseudo-Tier-2.

    Both halves come from the same items, so a correct rule PERMITS pooling
    here. n_small matches the real Tier-2 stratum so the null faces the same
    small-sample noise the real decision does.
    """
    idx = rng.permutation(len(rows))
    small = [rows[i] for i in idx[:n_small]]
    large = [rows[i] for i in idx[n_small:]]
    return large, small


def inject_baseline_shift(
    rows: list[ProbeRow], shift: float, rng: np.random.Generator,
) -> list[ProbeRow]:
    """Depress a stratum's zero-shot baseline by ~`shift`, leaving trained alone.

    This is what a relabelling does, in the direction measured on the real
    audit: the new gold is a target the zero-shot encoder finds much harder
    while the trained model finds it comparatively easy. Flipping zero-shot
    hits to misses reproduces that asymmetry without touching the trained arm.

    Used to trace a rule's detection curve -- the smallest inflation it still
    refuses -- so its blind spot is stated rather than assumed away.
    """
    hits = [i for i, r in enumerate(rows) if r["zero_shot_hit1"] == 1]
    n_flip = min(len(hits), int(round(shift * len(rows))))
    if n_flip == 0:
        return list(rows)
    chosen = set(rng.choice(np.array(hits), size=n_flip, replace=False).tolist())
    out: list[ProbeRow] = []
    for i, row in enumerate(rows):
        if i in chosen:
            flipped = dict(row)
            flipped["zero_shot_hit1"] = 0
            out.append(ProbeRow(**flipped))  # type: ignore[typeddict-item]
        else:
            out.append(row)
    return out


def per_framework_deltas(
    rows: list[ProbeRow], n_resamples: int, rng: np.random.Generator,
) -> dict[str, tuple[int, float, float, tuple[float, float]]]:
    """Per-fold n, zero-shot, delta and interval, on one stratum.

    Exists to price the composition exposure: the four curation targets
    (csa_aicm, cosai, aiuc_1, nist_ai_rmf) share NO framework with the five
    test folds, so a curated Tier-2 stratum is 100% composition-shifted from
    Tier-1 by construction. If per-framework deltas differ by more than the
    gate threshold, composition alone can decide the gate.
    """
    by_fold: dict[str, list[ProbeRow]] = defaultdict(list)
    for row in rows:
        by_fold[row["fold_dir"]].append(row)
    out: dict[str, tuple[int, float, float, tuple[float, float]]] = {}
    for fold, items in sorted(by_fold.items()):
        draws = bootstrap_deltas(items, n_resamples, rng)
        out[fold] = (
            len(items), zero_shot_rate(items), observed_delta(items),
            (float(np.percentile(draws, CI_LOW_PCT)),
             float(np.percentile(draws, CI_HIGH_PCT))),
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rows = build_rows(PHASE1B_RESULTS_DIR / args.run)
    tier1 = [r for r in rows if not r["audit_touched"]]
    relabelled = [r for r in rows if r["audit_touched"]]

    rng = np.random.default_rng(SEED)
    real = make_contrast(tier1, relabelled, N_RESAMPLES, rng)
    logger.info("THE WORKED EXAMPLE -- Tier-1 vs the audit-relabelled stratum")
    logger.info("  Tier-1      n=%3d  delta=%+.4f [%+.4f, %+.4f]  zero-shot=%.4f",
                len(tier1), real["delta_a"], *real["ci_a"], real["zero_shot_a"])
    logger.info("  relabelled  n=%3d  delta=%+.4f [%+.4f, %+.4f]  zero-shot=%.4f",
                len(relabelled), real["delta_b"], *real["ci_b"], real["zero_shot_b"])
    logger.info("  difference        %+.4f [%+.4f, %+.4f]   baseline gap=%.4f",
                real["diff"], *real["diff_ci"],
                abs(real["zero_shot_a"] - real["zero_shot_b"]))
    logger.info("")

    logger.info("NULL SIMULATION -- %d random %d/%d splits of the SAME population",
                N_NULL_REPLICATES, len(tier1) - len(relabelled), len(relabelled))
    null_rng = np.random.default_rng(SEED)
    null_contrasts: list[Contrast] = []
    for _ in range(N_NULL_REPLICATES):
        large, small = split_null(tier1, len(relabelled), null_rng)
        null_contrasts.append(
            make_contrast(large, small, N_RESAMPLES_NULL, null_rng),
        )
    logger.info("")

    logger.info("%-4s %-42s %-10s %-12s %s", "id", "rule", "worked ex.",
                "null permit", "verdict")
    results: dict[str, object] = {}
    for key, (label, rule) in CANDIDATES.items():
        caught = not rule(real)
        permit_rate = float(np.mean([rule(c) for c in null_contrasts]))
        if caught and permit_rate >= 0.80:
            verdict = "USABLE"
        elif not caught:
            verdict = "FAILS (admits the inflation)"
        else:
            verdict = "TOO STRICT (discards real power)"
        logger.info("%-4s %-42s %-10s %-12s %s", key, label,
                    "REFUSED" if caught else "pooled",
                    f"{permit_rate:.0%}", verdict)
        results[key] = {
            "label": label, "refuses_worked_example": caught,
            "null_permit_rate": permit_rate, "verdict": verdict,
        }

    logger.info("")
    logger.info("R4 OPERATING POINTS -- how wide must the baseline margin be?")
    logger.info("  (real baseline gap = %.4f; a margin at or above it stops "
                "catching the inflation)", abs(real["baseline_diff"]))
    logger.info("  %-8s %-10s %-12s", "margin", "worked ex.", "null permit")
    sweep: dict[str, object] = {}
    for margin in BASELINE_MARGIN_SWEEP:
        caught = abs(real["baseline_diff"]) > margin
        rate = float(np.mean(
            [abs(c["baseline_diff"]) <= margin for c in null_contrasts],
        ))
        logger.info("  %-8.2f %-10s %-12s", margin,
                    "REFUSED" if caught else "pooled", f"{rate:.0%}")
        sweep[f"{margin:.2f}"] = {
            "refuses_worked_example": caught, "null_permit_rate": rate,
        }

    logger.info("")
    logger.info("DETECTION CURVE -- how big an inflation must be before R6 "
                "refuses it")
    logger.info("  (a curated stratum whose baseline drops LESS than this "
                "slips through)")
    logger.info("  %-14s %-14s %s", "baseline drop", "R6 refuses", "R0 refuses")
    curve: dict[str, object] = {}
    curve_rng = np.random.default_rng(SEED)
    for shift in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
        r6_hits, r0_hits = 0, 0
        for _ in range(N_NULL_REPLICATES // 2):
            large, small = split_null(tier1, len(relabelled), curve_rng)
            small = inject_baseline_shift(small, shift, curve_rng)
            c = make_contrast(large, small, N_RESAMPLES_NULL, curve_rng)
            r6_hits += not r6_baseline_not_significantly_different(c)
            r0_hits += not r0_status_quo(c)
        n = N_NULL_REPLICATES // 2
        logger.info("  %-14.2f %-14s %s", shift,
                    f"{r6_hits / n:.0%}", f"{r0_hits / n:.0%}")
        curve[f"{shift:.2f}"] = {
            "r6_refusal_rate": r6_hits / n, "r0_refusal_rate": r0_hits / n,
        }

    # --- scenario C: composition shift with NO relabelling at all ------------
    logger.info("")
    logger.info("PER-FRAMEWORK DELTAS on the audit-untouched stratum")
    logger.info("  (curation targets share NO framework with these folds, so a "
                "curated")
    logger.info("   Tier-2 is 100%% composition-shifted from Tier-1 by "
                "construction)")
    comp_rng = np.random.default_rng(SEED)
    per_fw = per_framework_deltas(tier1, N_RESAMPLES, comp_rng)
    for fold, (n, zs, delta, ci) in per_fw.items():
        logger.info("  %-26s n=%3d  zs=%.4f  delta=%+.4f [%+.4f, %+.4f]",
                    fold, n, zs, delta, *ci)
    usable = {f: v for f, v in per_fw.items() if v[0] >= 10}
    if usable:
        lo = min(v[2] for v in usable.values())
        hi = max(v[2] for v in usable.values())
        logger.info("  swing across folds with n>=10: %+.4f to %+.4f = %.4f "
                    "(%.1fx the %.2f gate)",
                    lo, hi, hi - lo, (hi - lo) / 0.10, 0.10)

    logger.info("")
    logger.info("SCENARIO C -- composition shift, NO relabelling")
    logger.info("  pseudo-Tier-1 = OWASP AI Exchange (untouched); "
                "pseudo-Tier-2 = other folds (untouched)")
    logger.info("  Both strata carry ORIGINAL gold. A rule should still REFUSE: "
                "pooling here")
    logger.info("  lets framework mix drive the number, which is §1.2's own "
                "stated concern.")
    aix = [r for r in tier1 if r["fold_dir"] == "fold_OWASP_AI_Exchange"]
    rest = [r for r in tier1 if r["fold_dir"] != "fold_OWASP_AI_Exchange"]
    comp = make_contrast(aix, rest, N_RESAMPLES, comp_rng)
    logger.info("  Tier-1 (AIX)   n=%3d  delta=%+.4f  zero-shot=%.4f",
                len(aix), comp["delta_a"], comp["zero_shot_a"])
    logger.info("  Tier-2 (other) n=%3d  delta=%+.4f  zero-shot=%.4f",
                len(rest), comp["delta_b"], comp["zero_shot_b"])
    logger.info("  difference %+.4f [%+.4f, %+.4f]   baseline gap %.4f",
                comp["diff"], *comp["diff_ci"],
                abs(comp["baseline_diff"]))
    logger.info("")
    logger.info("  %-4s %-42s %s", "id", "rule", "composition shift")
    comp_results: dict[str, bool] = {}
    for key, (label, rule) in CANDIDATES.items():
        refused = not rule(comp)
        comp_results[key] = refused
        logger.info("  %-4s %-42s %s", key, label,
                    "REFUSED (correct)" if refused else "pooled (MISSED)")

    report = {
        "run": args.run,
        "baseline_alpha": BASELINE_ALPHA,
        "r4_margin_sweep": sweep,
        "detection_curve": curve,
        "per_framework_untouched": {
            k: {"n": v[0], "zero_shot": v[1], "delta": v[2], "ci": list(v[3])}
            for k, v in per_fw.items()
        },
        "composition_scenario": {
            "n_tier1_aix": len(aix), "n_tier2_other": len(rest),
            **comp, "rule_refusals": comp_results,
        },
        "equivalence_margin": EQUIVALENCE_MARGIN,
        "baseline_margin": BASELINE_MARGIN,
        "n_null_replicates": N_NULL_REPLICATES,
        "worked_example": {
            "n_tier1": len(tier1), "n_relabelled": len(relabelled), **real,
        },
        "candidates": results,
    }
    if args.out:
        _atomic_write_json(args.out, report)
        logger.info("")
        logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
