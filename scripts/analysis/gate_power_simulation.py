"""Monte-Carlo power for the Campaign 3 gate under a framework-random-effects
estimator.

WHY A SIMULATION. The gate is not a t-test. It is a decision rule --
`P(delta <= 0.10) < 0.05` -- evaluated on a *cluster bootstrap* of a *paired
binary* outcome over a small number of frameworks. No closed-form power exists
for that, so this simulates the whole pipeline: draw frameworks, draw items,
run the bootstrap, apply the rule, count passes.

WHY IT IS NEEDED. `results/phase1b/CAMPAIGN3.md` Amendment 1 puts real power at
43-54%. That figure counts ITEMS and treats the five folds as fixed. The gate's
claim is generalisation to UNSEEN frameworks, which makes framework a random
effect, and `docs/campaign3-audit-mechanism.md` §6d shows the primary's interval
widens 2.1x and crosses zero under that framing.

WHAT THE ROUND-1 PREMORTEM FOUND, AND WHAT CHANGED HERE.

  * The simulation used an identical item count for every fold, which makes
    `mean over folds` an UNWEIGHTED (macro) average. The gate reports the
    item-weighted (micro) average. On the real primary the two differ by
    0.1701 -- micro +0.1000, macro +0.2701, against a 0.10 threshold. Fold
    sizes are now explicit and the estimator is `pooled_delta`, which is micro.

  * `TAU_GRID` stopped at 0.20 while the corpus's own point estimate is 0.3702,
    so the branch that says "replace the instrument rather than re-power it"
    was never simulated. The grid now covers it.

  * The clamp that keeps a framework's delta inside the discordant rate
    silently shrank both mu and tau at the high end, unreported. Every cell now
    logs what it actually delivered via `realised_parameters`.

  * Deleting framework resampling outright left the test suite green. The
    resampling is now reachable through `_pass_probability`, and a test asserts
    that removing it changes the answer.

WHAT IT STILL CANNOT DO. The between-framework SD (tau) is the dominant
parameter and CANNOT be estimated from this data:

    all 5 folds     -> tau = 0.3702
    folds n >= 10   -> tau = 0.0782   (k = 3)
    drop the n=2 fold, keep the other four -> tau = 0.0000

Leave-one-fold-out swings it from 0.0000 to 0.4446 depending on which fold
goes. With five frameworks, two of them n <= 4, tau is not identified. So this
script does NOT report a power number. It reports a power SURFACE over tau, and
the reader is expected to notice that the surface spans "comfortable" to
"hopeless".

Read-only, no model, no pods. Deterministic: fixed seed.
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Final, TypedDict

import numpy as np

from scripts.analysis.audit_mechanism_probe import _atomic_write_json, build_rows
from tract.config import PHASE1B_RESULTS_DIR

logger = logging.getLogger(__name__)

NDArrayF = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

DEFAULT_RUN: Final[str] = "c3_TEST_A3_prose_sw_qwen06b_seq1024"

GATE_THRESHOLD: Final[float] = 0.10
# The gate fires when P(delta <= threshold) falls below this.
GATE_ALPHA: Final[float] = 0.05
TARGET_POWER: Final[float] = 0.80

N_STUDIES: Final[int] = 400
N_BOOTSTRAP: Final[int] = 800
SEED: Final[int] = 42

# tau is swept rather than fixed because it is not identified. The top of the
# grid reaches the corpus's own all-folds estimate (0.3702), so the branch where
# no feasible design works is simulated rather than assumed away.
TAU_GRID: Final[tuple[float, ...]] = (
    0.00, 0.05, 0.08, 0.12, 0.16, 0.20, 0.28, 0.37,
)
MU_GRID: Final[tuple[float, ...]] = (0.10, 0.125, 0.15, 0.175, 0.20, 0.25)

# Probability that a given item is a "discordant pair" -- one arm hits and the
# other misses. Measured on the audit-untouched stratum: 22 trained-wins and 11
# trained-losses in 110 items. Ties carry no information about the delta.
#
# Caveat, recorded because the round-1 premortem raised it: the per-fold rate
# ranges 0.2381 (OWASP AI Exchange) to 1.0000 (OWASP Top10 for LLM, n=2), so
# fixing it at the pooled value understates within-fold variance on four of five
# folds. That biases power UP, in the opposite direction from the clamp.
DISCORDANT_RATE: Final[float] = 0.30

# The observed design, from the 110-item audit-untouched primary. Used instead
# of a balanced approximation because the imbalance is severe -- OWASP AI
# Exchange alone is 57.3% of the denominator -- and a balanced simulation
# reports a different estimand entirely.
OBSERVED_FOLD_SIZES: Final[tuple[int, ...]] = (63, 30, 11, 4, 2)

# The same five plus ENISA (33), BIML (17) and ETSI (24) at their dedup eval
# item counts. Kept for comparison only; the roster rotation is NOT part of the
# Phase 2C design (see docs/campaign3-premortem-round1.md A6).
ROSTER_FOLD_SIZES: Final[tuple[int, ...]] = (63, 30, 11, 4, 2, 33, 17, 24)


class PowerCell(TypedDict):
    mu: float
    tau: float
    k: int
    n_items: int
    power: float
    realised_mu: float
    realised_tau: float
    clamped_fraction: float


class RealisedParameters(TypedDict):
    """What a simulated study actually delivered, after clamping."""

    mu: float
    tau: float
    clamped_fraction: float


def pooled_delta(folds: list[NDArrayF]) -> float:
    """Item-weighted (micro) mean, matching the gate's primary.

    The gate reports `np.concatenate(folds).mean()`. A mean over per-fold means
    is a different statistic: on the real primary micro is +0.1000 and macro is
    +0.2701, either side of a 0.10 threshold.
    """
    return float(np.concatenate(folds).mean())


def realised_parameters(folds: list[NDArrayF]) -> RealisedParameters:
    """Report what the draw delivered, not what it was asked for.

    `simulate_study` clamps a framework whose delta exceeds the discordant rate,
    which shrinks both mu and tau. At mu=0.25, tau=0.20 the clamp binds on about
    40% of framework draws. Unreported, that silently mislabels the axes of the
    surface at exactly the cells the funding decision reads.
    """
    means = np.array([f.mean() for f in folds], dtype=float)
    return RealisedParameters(
        mu=float(means.mean()),
        tau=float(means.std(ddof=1)) if len(means) > 1 else 0.0,
        clamped_fraction=float((np.abs(means) > DISCORDANT_RATE).mean()),
    )


def simulate_study(
    fold_sizes: Sequence[int], mu: float, tau: float, discordant: float,
    rng: np.random.Generator,
) -> list[NDArrayF]:
    """Draw one study: one framework per entry in `fold_sizes`.

    Per-framework true deltas are Normal(mu, tau^2). Within a framework each
    item is a McNemar cell: a trained-win (+1), a trained-loss (-1) or a tie
    (0). The discordant rate is held at the observed value and split so that
    p_win - p_loss equals that framework's delta, which is the parameterisation
    the paired statistic actually depends on.
    """
    deltas = rng.normal(mu, tau, size=len(fold_sizes))
    folds: list[NDArrayF] = []
    for delta, n_items in zip(deltas, fold_sizes, strict=True):
        # p_win + p_loss = discordant, p_win - p_loss = delta.
        p_win = (discordant + delta) / 2.0
        p_loss = (discordant - delta) / 2.0
        # A framework whose delta exceeds the discordant rate is infeasible;
        # clamp to the boundary rather than emitting negative probabilities.
        # realised_parameters() reports how often this binds.
        p_win = float(np.clip(p_win, 0.0, 1.0))
        p_loss = float(np.clip(p_loss, 0.0, 1.0 - p_win))
        draw = rng.random(n_items)
        item = np.zeros(n_items, dtype=float)
        item[draw < p_win] = 1.0
        item[(draw >= p_win) & (draw < p_win + p_loss)] = -1.0
        folds.append(item)
    return folds


def _pass_probability(
    folds: list[NDArrayF], n_bootstrap: int, rng: np.random.Generator,
    *, resample_frameworks: bool = True,
) -> float:
    """P(delta <= threshold) under the bootstrap the gate would run.

    `resample_frameworks=False` degrades this to an item-only bootstrap. It
    exists so a test can assert that the cluster resampling changes the answer:
    before this switch existed, deleting the resampling entirely left the suite
    green, which meant nothing pinned the estimator's defining property.

    Fold sizes may differ, so the denominator is accumulated per draw rather
    than assumed constant -- that is what keeps this a micro average.
    """
    k = len(folds)
    sums = np.zeros(n_bootstrap, dtype=float)
    counts = np.zeros(n_bootstrap, dtype=float)
    for slot in range(k):
        picks = (rng.integers(0, k, n_bootstrap) if resample_frameworks
                 else np.full(n_bootstrap, slot, dtype=int))
        for fold_index in range(k):
            mask = picks == fold_index
            n_drawn = int(mask.sum())
            if n_drawn == 0:
                continue
            fold = folds[fold_index]
            idx = rng.integers(0, len(fold), (n_drawn, len(fold)))
            sums[mask] += fold[idx].sum(axis=1)
            counts[mask] += len(fold)
    return float((sums / counts <= GATE_THRESHOLD).mean())


def cluster_bootstrap_pass(
    folds: list[NDArrayF], n_bootstrap: int, rng: np.random.Generator,
) -> bool:
    """Run the gate: resample FRAMEWORKS and items, then apply the rule."""
    return _pass_probability(folds, n_bootstrap, rng) < GATE_ALPHA


def power_at(
    fold_sizes: Sequence[int], mu: float, tau: float, n_studies: int,
    n_bootstrap: int, rng: np.random.Generator,
) -> tuple[float, RealisedParameters]:
    """Power, and the parameters the draws actually delivered."""
    passes = 0
    mus: list[float] = []
    taus: list[float] = []
    clamped: list[float] = []
    for _ in range(n_studies):
        folds = simulate_study(fold_sizes, mu, tau, DISCORDANT_RATE, rng)
        passes += cluster_bootstrap_pass(folds, n_bootstrap, rng)
        realised = realised_parameters(folds)
        mus.append(realised["mu"])
        taus.append(realised["tau"])
        clamped.append(realised["clamped_fraction"])
    return passes / n_studies, RealisedParameters(
        mu=float(np.mean(mus)),
        tau=float(np.mean(taus)),
        clamped_fraction=float(np.mean(clamped)),
    )


def observed_tau_range(run: str) -> dict[str, float]:
    """Report the tau estimates the data supports, and how unstable they are."""
    rows = build_rows(PHASE1B_RESULTS_DIR / run)
    by_fold: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if not row["audit_touched"]:
            by_fold[row["fold_dir"]].append(
                float(row["trained_hit1"] - row["zero_shot_hit1"]),
            )
    folds = {k: np.array(v, dtype=float) for k, v in by_fold.items()}

    def moments(sel: dict[str, NDArrayF]) -> float:
        deltas = np.array([v.mean() for v in sel.values()])
        within = np.array([v.var(ddof=1) / len(v) for v in sel.values()])
        return float(np.sqrt(max(0.0, deltas.var(ddof=1) - within.mean())))

    big = {k: v for k, v in folds.items() if len(v) >= 10}
    swing = [moments({k: v for k, v in folds.items() if k != drop})
             for drop in folds]
    return {
        "tau_all_folds": moments(folds),
        "tau_folds_n_ge_10": moments(big),
        "tau_loo_min": float(min(swing)),
        "tau_loo_max": float(max(swing)),
        "k_all": float(len(folds)),
        "k_big": float(len(big)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    taus = observed_tau_range(args.run)
    logger.info("tau is NOT identified by this data:")
    logger.info("  all %d folds       -> tau = %.4f",
                taus["k_all"], taus["tau_all_folds"])
    logger.info("  folds n>=10 (k=%d)  -> tau = %.4f",
                taus["k_big"], taus["tau_folds_n_ge_10"])
    logger.info("  leave-one-fold-out -> tau spans %.4f to %.4f",
                taus["tau_loo_min"], taus["tau_loo_max"])
    logger.info("  Power is therefore a surface, not a number.")
    logger.info("")

    rng = np.random.default_rng(SEED)
    cells: list[PowerCell] = []

    # A 2x2 separating the two things a bigger design could buy. Comparing only
    # "now" against "everything" would confound more frameworks with more items.
    scenarios: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("now        (k=5, 110 items)", OBSERVED_FOLD_SIZES),
        ("+items     (k=5, 340 items)", (195, 93, 34, 12, 6)),
        ("+frameworks(k=8, 184 items)", ROSTER_FOLD_SIZES),
        ("+both      (k=8, 552 items)", tuple(n * 3 for n in ROSTER_FOLD_SIZES)),
    )

    for mu in MU_GRID:
        logger.info("mu = %+.3f  (true mean delta; gate threshold %.2f)",
                    mu, GATE_THRESHOLD)
        header = "  {:<28}".format("scenario") + "".join(
            f"  tau={t:.2f}" for t in TAU_GRID)
        logger.info("%s", header)
        for label, sizes in scenarios:
            row: list[float] = []
            for tau in TAU_GRID:
                power, realised = power_at(
                    sizes, mu, tau, N_STUDIES, N_BOOTSTRAP, rng)
                cells.append(PowerCell(
                    mu=mu, tau=tau, k=len(sizes), n_items=sum(sizes),
                    power=power, realised_mu=realised["mu"],
                    realised_tau=realised["tau"],
                    clamped_fraction=realised["clamped_fraction"],
                ))
                row.append(power)
            line = "  {:<28}".format(label) + "".join(
                f"    {p:5.0%}" for p in row)
            logger.info("%s", line)
        logger.info("")

    # The axes are only honest if the draws delivered what the labels claim.
    logger.info("REALISED vs REQUESTED (the clamp shrinks both at high tau)")
    logger.info("  %-10s %-10s %-12s %-12s %s",
                "req mu", "req tau", "got mu", "got tau", "clamped")
    for cell in cells:
        if cell["clamped_fraction"] > 0.05 and cell["k"] == 5:
            logger.info("  %-10.3f %-10.2f %-12.4f %-12.4f %.0f%%",
                        cell["mu"], cell["tau"], cell["realised_mu"],
                        cell["realised_tau"], 100 * cell["clamped_fraction"])

    if args.out:
        _atomic_write_json(args.out, {
            "tau_estimates": taus, "gate_threshold": GATE_THRESHOLD,
            "gate_alpha": GATE_ALPHA, "n_studies": N_STUDIES,
            "n_bootstrap": N_BOOTSTRAP, "discordant_rate": DISCORDANT_RATE,
            "cells": cells,
        })
        logger.info("")
        logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
