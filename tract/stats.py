"""Binomial confidence interval utilities.

Small deliberately: one function, no scipy dependency. The ceiling study
(design doc Part 0.1) is a CPU-only, no-GPU deliverable, and pulling in scipy
for a single closed-form interval would widen the dependency surface of a
script that exists specifically because everything else in the pipeline
needs a GPU.
"""
from __future__ import annotations

import math
from typing import Final, NamedTuple

# Two-sided 95% z-value, standard normal inverse CDF at 0.975. Exact to more
# digits than the interval width will ever resolve. Hardcoded rather than
# computed because there is no dependency in this module that provides
# norm.ppf, and this constant does not change.
Z_95: Final[float] = 1.9599639845400545


class WilsonInterval(NamedTuple):
    """A Wilson score interval for a binomial proportion.

    point: the raw sample proportion (successes / n), not the Wilson center.
    lower, upper: the 95% Wilson bounds.
    half_width: (upper - lower) / 2, the number every ceiling-study gate
        report has to quote to say whether it can decide anything.
    """

    point: float
    lower: float
    upper: float
    half_width: float


def wilson_interval(successes: int, n: int, z: float = Z_95) -> WilsonInterval:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal (Wald) approximation here because n per stratum
    and per framework ranges from 6 (owasp_llm_top10) to 125 (a full
    stratum). The Wald interval undercovers badly at small n and can produce
    bounds outside [0, 1], which the Wilson form cannot.

    Args:
        successes: Number of items scored as a hit. Must satisfy
            0 <= successes <= n.
        n: Number of items scored. Must be > 0.
        z: Two-sided z critical value. Defaults to the 95% value.

    Returns:
        WilsonInterval with point estimate and 95% bounds, both clamped to
        [0, 1].

    Raises:
        ValueError: If n <= 0 or successes is out of [0, n].
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if successes < 0 or successes > n:
        raise ValueError(f"successes={successes} out of range [0, {n}]")

    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))) / denom

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    half_width = (upper - lower) / 2.0

    return WilsonInterval(point=p_hat, lower=lower, upper=upper, half_width=half_width)
