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
    logs what it actually delivered via `DrawDiagnostics`, measured on
    the drawn deltas rather than on realised fold means.

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
    drawn_mu: float
    drawn_tau: float
    clamped_fraction: float
    fold_mean_spread: float


class DrawDiagnostics(TypedDict):
    """What the DATA-GENERATING PROCESS delivered, measured where it happens.

    The previous version measured realised fold MEANS, which is a different
    thing and wrong in both directions. `clamped_fraction` counted fold means
    above the discordant rate, but the clamp binds on the drawn
    Normal(mu, tau) delta before any items exist -- at fold sizes including
    n=2 and n=4 a mean clears 0.30 by sampling noise, so it reported 0.148 at
    mu=0.10, tau=0.00 where the true probability is exactly 0. And it called
    the SD of fold means "tau", which is sqrt(tau^2 + within-fold noise) and
    read ~0.20 when the true tau was 0.

    `fold_mean_spread` is that observed SD, kept because it is genuinely
    informative -- it is why tau cannot be estimated from five folds -- but
    never again labelled tau.
    """

    drawn_mu: float
    drawn_tau: float
    clamped_fraction: float
    fold_mean_spread: float


def pooled_delta(folds: list[NDArrayF]) -> float:
    """Item-weighted (micro) mean, matching the gate's primary.

    The gate reports `np.concatenate(folds).mean()`. A mean over per-fold means
    is a different statistic: on the real primary micro is +0.1000 and macro is
    +0.2701, either side of a 0.10 threshold.

    This is the REFERENCE definition. `_pass_probability` carries a vectorised
    sums/counts form of the same statistic, because materialising the
    concatenation once per bootstrap replicate is ~800x slower. The two are
    related only by that claim, so
    `TestTheLiveEstimatorIsTheDocumentedOne` pins them together -- mutating the
    fast path to a macro average previously left the entire suite green while
    moving power by 20.5pp at mu=0.25.
    """
    return float(np.concatenate(folds).mean())


def simulate_study(
    fold_sizes: Sequence[int], mu: float, tau: float, discordant: float,
    rng: np.random.Generator,
) -> tuple[list[NDArrayF], DrawDiagnostics]:
    """Draw one study: one framework per entry in `fold_sizes`.

    Per-framework true deltas are Normal(mu, tau^2). Within a framework each
    item is a McNemar cell: a trained-win (+1), a trained-loss (-1) or a tie
    (0). The discordant rate is held at the observed value and split so that
    p_win - p_loss equals that framework's delta, which is the parameterisation
    the paired statistic actually depends on.
    """
    deltas = rng.normal(mu, tau, size=len(fold_sizes))
    # Counted here, on the drawn delta, because here is where the clamp binds.
    n_clamped = int((np.abs(deltas) > discordant).sum())
    folds: list[NDArrayF] = []
    for delta, n_items in zip(deltas, fold_sizes, strict=True):
        # p_win + p_loss = discordant, p_win - p_loss = delta.
        p_win = (discordant + delta) / 2.0
        p_loss = (discordant - delta) / 2.0
        # A framework whose delta exceeds the discordant rate is infeasible;
        # clamp to the boundary rather than emitting negative probabilities.
        # DrawDiagnostics.clamped_fraction reports how often this binds,
        # counted on the drawn delta above rather than on the fold mean.
        p_win = float(np.clip(p_win, 0.0, 1.0))
        p_loss = float(np.clip(p_loss, 0.0, 1.0 - p_win))
        draw = rng.random(n_items)
        item = np.zeros(n_items, dtype=float)
        item[draw < p_win] = 1.0
        item[(draw >= p_win) & (draw < p_win + p_loss)] = -1.0
        folds.append(item)
    means = np.array([f.mean() for f in folds], dtype=float)
    return folds, DrawDiagnostics(
        drawn_mu=float(deltas.mean()),
        drawn_tau=float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
        clamped_fraction=n_clamped / len(deltas),
        fold_mean_spread=float(means.std(ddof=1)) if len(means) > 1 else 0.0,
    )


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
    # Refuse degenerate input rather than dividing by zero. Without this,
    # sums/counts is nan, `nan <= threshold` is False, P(delta <= 0.10) reads
    # 0.0, and `0.0 < GATE_ALPHA` returns a PASS -- the gate passing on no data.
    # In a project whose record is three withdrawn passes, a silent PASS is the
    # worst direction available.
    if not folds:
        raise ValueError(
            "Refusing to evaluate the gate with no folds. A pass computed here "
            "would be a division by zero read as P(delta <= threshold) = 0."
        )
    empty = [i for i, f in enumerate(folds) if len(f) == 0]
    if empty:
        raise ValueError(
            f"Folds {empty} are empty. A fold with no items contributes no "
            "denominator, so any draw selecting only such folds is a nan."
        )

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
) -> tuple[float, DrawDiagnostics]:
    """Power, and what the draws actually delivered."""
    passes = 0
    mus: list[float] = []
    taus: list[float] = []
    clamped: list[float] = []
    spreads: list[float] = []
    for _ in range(n_studies):
        folds, diag = simulate_study(fold_sizes, mu, tau, DISCORDANT_RATE, rng)
        passes += cluster_bootstrap_pass(folds, n_bootstrap, rng)
        mus.append(diag["drawn_mu"])
        taus.append(diag["drawn_tau"])
        clamped.append(diag["clamped_fraction"])
        spreads.append(diag["fold_mean_spread"])
    return passes / n_studies, DrawDiagnostics(
        drawn_mu=float(np.mean(mus)),
        drawn_tau=float(np.mean(taus)),
        clamped_fraction=float(np.mean(clamped)),
        fold_mean_spread=float(np.mean(spreads)),
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
                power, diag = power_at(
                    sizes, mu, tau, N_STUDIES, N_BOOTSTRAP, rng)
                cells.append(PowerCell(
                    mu=mu, tau=tau, k=len(sizes), n_items=sum(sizes),
                    power=power, drawn_mu=diag["drawn_mu"],
                    drawn_tau=diag["drawn_tau"],
                    clamped_fraction=diag["clamped_fraction"],
                    fold_mean_spread=diag["fold_mean_spread"],
                ))
                row.append(power)
            line = "  {:<28}".format(label) + "".join(
                f"    {p:5.0%}" for p in row)
            logger.info("%s", line)
        logger.info("")

    # The axes are only honest if the draws delivered what the labels claim.
    logger.info("DRAWN vs REQUESTED, and the observed fold-mean spread")
    logger.info("  fold_mean_spread is NOT tau: it is sqrt(tau^2 + within-fold")
    logger.info("  noise), and at n=2/n=4 the noise term dominates. That is why")
    logger.info("  tau cannot be estimated from five folds.")
    logger.info("  %-9s %-9s %-11s %-11s %-9s %s",
                "req mu", "req tau", "drawn mu", "drawn tau", "clamped",
                "fold spread")
    for cell in cells:
        if cell["k"] == 5 and cell["tau"] in (0.00, 0.20, 0.37):
            logger.info("  %-9.3f %-9.2f %-11.4f %-11.4f %-9.0f%% %.4f",
                        cell["mu"], cell["tau"], cell["drawn_mu"],
                        cell["drawn_tau"], 100 * cell["clamped_fraction"],
                        cell["fold_mean_spread"])

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
