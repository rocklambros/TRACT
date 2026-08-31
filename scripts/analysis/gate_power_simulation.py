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
widens 2.1x and crosses zero under that framing. Power computed on the fixed-fold
variance is therefore an overestimate of power for the claim actually being made.

WHAT IT CANNOT DO. The between-framework SD (tau) is the dominant parameter and
CANNOT be estimated from this data:

    all 5 folds     -> tau = 0.3702
    folds n >= 10   -> tau = 0.0782   (k = 3)

a 4.7x range, driven by a degenerate n=2 fold whose two items both flipped
(delta exactly +1.0000, sample within-variance exactly 0, so method-of-moments
reads it as a precisely-measured framework at +1.0). With five frameworks, two
of them n <= 4, tau is not identified.

So this script does NOT report a power number. It reports a power SURFACE over
tau, and marks which region the data is consistent with. Quoting a single power
figure from this design would repeat the mistake Amendment 1 corrected.

Read-only, no model, no pods. Deterministic: fixed seed.
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from typing import Final, TypedDict

import numpy as np

from scripts.analysis.audit_mechanism_probe import _atomic_write_json, build_rows
from tract.config import PHASE1B_RESULTS_DIR

logger = logging.getLogger(__name__)

DEFAULT_RUN: Final[str] = "c3_TEST_A3_prose_sw_qwen06b_seq1024"

GATE_THRESHOLD: Final[float] = 0.10
# The gate fires when P(delta <= threshold) falls below this.
GATE_ALPHA: Final[float] = 0.05
TARGET_POWER: Final[float] = 0.80

N_STUDIES: Final[int] = 400
N_BOOTSTRAP: Final[int] = 800
SEED: Final[int] = 42

# Sweeps. tau is swept rather than fixed because it is not identified (see
# module docstring); mu is swept because the planning effect must not be taken
# from the pilot point estimate.
TAU_GRID: Final[tuple[float, ...]] = (0.00, 0.05, 0.08, 0.12, 0.16, 0.20)
MU_GRID: Final[tuple[float, ...]] = (0.10, 0.125, 0.15, 0.175, 0.20, 0.25)

# Probability that a given item is a "discordant pair" -- one arm hits and the
# other misses. Measured on the audit-untouched stratum: 22 trained-wins and 11
# trained-losses in 110 items. Ties carry no information about the delta, so
# this is the fraction of items that can move the estimate at all.
DISCORDANT_RATE: Final[float] = 0.30


class PowerCell(TypedDict):
    mu: float
    tau: float
    k: int
    n_per_framework: int
    power: float


def simulate_study(
    k: int, n_per: int, mu: float, tau: float, discordant: float,
    rng: np.random.Generator,
) -> list[np.ndarray[tuple[int, ...], np.dtype[np.float64]]]:
    """Draw one study: k frameworks of n_per paired items each.

    Per-framework true deltas are Normal(mu, tau^2). Within a framework, each
    item is a McNemar cell: a trained-win (+1), a trained-loss (-1), or a tie
    (0). The discordant rate is held at the observed value and split so that
    p_win - p_loss equals that framework's delta, which is the parameterisation
    the paired statistic actually depends on.
    """
    deltas = rng.normal(mu, tau, size=k)
    folds: list[np.ndarray[tuple[int, ...], np.dtype[np.float64]]] = []
    for delta in deltas:
        # p_win + p_loss = discordant, p_win - p_loss = delta.
        p_win = (discordant + delta) / 2.0
        p_loss = (discordant - delta) / 2.0
        # A framework whose delta exceeds the discordant rate is infeasible;
        # clamp to the boundary rather than emitting negative probabilities.
        p_win = float(np.clip(p_win, 0.0, 1.0))
        p_loss = float(np.clip(p_loss, 0.0, 1.0 - p_win))
        draw = rng.random(n_per)
        item = np.zeros(n_per, dtype=float)
        item[draw < p_win] = 1.0
        item[(draw >= p_win) & (draw < p_win + p_loss)] = -1.0
        folds.append(item)
    return folds


def cluster_bootstrap_pass(
    folds: list[np.ndarray[tuple[int, ...], np.dtype[np.float64]]],
    n_bootstrap: int, rng: np.random.Generator,
) -> bool:
    """Run the gate: resample FRAMEWORKS and items, then apply the rule.

    Resampling frameworks is the whole point -- it propagates between-framework
    variance into the interval, which is what makes the estimate speak to
    unseen frameworks rather than to these ones.
    """
    k = len(folds)
    n_per = len(folds[0])
    matrix = np.stack(folds)                      # (k, n_per)
    picks = rng.integers(0, k, (n_bootstrap, k))  # which frameworks
    items = rng.integers(0, n_per, (n_bootstrap, k, n_per))
    drawn = matrix[picks[:, :, None], items]      # (n_bootstrap, k, n_per)
    stats = drawn.mean(axis=(1, 2))
    return bool((stats <= GATE_THRESHOLD).mean() < GATE_ALPHA)


def power_at(
    k: int, n_per: int, mu: float, tau: float, n_studies: int,
    n_bootstrap: int, rng: np.random.Generator,
) -> float:
    passes = 0
    for _ in range(n_studies):
        folds = simulate_study(k, n_per, mu, tau, DISCORDANT_RATE, rng)
        passes += cluster_bootstrap_pass(folds, n_bootstrap, rng)
    return passes / n_studies


def observed_tau_range(run: str) -> dict[str, float]:
    """Report the two tau estimates the data supports, and why they differ."""
    rows = build_rows(PHASE1B_RESULTS_DIR / run)
    by_fold: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if not row["audit_touched"]:
            by_fold[row["fold_dir"]].append(
                float(row["trained_hit1"] - row["zero_shot_hit1"]),
            )
    folds = {k: np.array(v) for k, v in by_fold.items()}

    def moments(
        sel: dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float64]]],
    ) -> float:
        deltas = np.array([v.mean() for v in sel.values()])
        within = np.array([v.var(ddof=1) / len(v) for v in sel.values()])
        return float(np.sqrt(max(0.0, deltas.var(ddof=1) - within.mean())))

    big = {k: v for k, v in folds.items() if len(v) >= 10}
    return {
        "tau_all_folds": moments(folds),
        "tau_folds_n_ge_10": moments(big),
        "k_all": float(len(folds)),
        "k_big": float(len(big)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    taus = observed_tau_range(args.run)
    logger.info("tau is NOT identified by this data:")
    logger.info("  all %d folds      -> tau = %.4f", taus["k_all"],
                taus["tau_all_folds"])
    logger.info("  folds n>=10 (k=%d) -> tau = %.4f", taus["k_big"],
                taus["tau_folds_n_ge_10"])
    logger.info("  ratio %.1fx. Power is therefore reported as a surface.",
                taus["tau_all_folds"] / max(taus["tau_folds_n_ge_10"], 1e-9))
    logger.info("")

    rng = np.random.default_rng(SEED)
    cells: list[PowerCell] = []

    # A 2x2 that separates the two things curation buys. Comparing only
    # "now" against "curated" would confound more frameworks with more items,
    # and the whole claim under test is which of the two is binding.
    #
    #   k=5, 22/fw : today (110 untouched items)
    #   k=5, 68/fw : 3.1x MORE ITEMS, same five frameworks (340)
    #   k=9, 22/fw : 1.8x MORE FRAMEWORKS, same items per framework (198)
    #   k=9, 68/fw : both -- the realistic post-curation design (612)
    scenarios = (
        ("now        (k=5, 22/fw)", 5, 22),
        ("+items     (k=5, 68/fw)", 5, 68),
        ("+frameworks(k=9, 22/fw)", 9, 22),
        ("curated    (k=9, 68/fw)", 9, 68),
    )

    for mu in MU_GRID:
        logger.info("mu = %+.3f  (true mean delta; gate threshold %.2f)",
                    mu, GATE_THRESHOLD)
        # Built as one string and logged with a single %s: the formatted
        # percentages contain literal '%', which logging's own %-formatting
        # would otherwise try to interpret.
        header = "  {:<22}".format("scenario") + "".join(
            f"  tau={t:.2f}" for t in TAU_GRID)
        logger.info("%s", header)
        for label, k, n_per in scenarios:
            row = []
            for tau in TAU_GRID:
                p = power_at(k, n_per, mu, tau, N_STUDIES, N_BOOTSTRAP, rng)
                cells.append(PowerCell(mu=mu, tau=tau, k=k,
                                       n_per_framework=n_per, power=p))
                row.append(p)
            line = "  {:<22}".format(label) + "".join(
                f"    {p:5.0%}" for p in row)
            logger.info("%s", line)
        logger.info("")

    if args.out:
        _atomic_write_json(
            __import__("pathlib").Path(args.out),
            {"tau_estimates": taus, "gate_threshold": GATE_THRESHOLD,
             "gate_alpha": GATE_ALPHA, "n_studies": N_STUDIES,
             "n_bootstrap": N_BOOTSTRAP, "discordant_rate": DISCORDANT_RATE,
             "cells": cells},
        )
        logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
