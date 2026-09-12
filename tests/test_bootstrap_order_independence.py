"""The bootstrap must not depend on the order the folds were listed in.

`_build_fold_index_matrix` drew every fold's indices sequentially from ONE
generator, so fold 1's draw depended on how many items fold 0 had consumed. Two
runs over the same data with the folds listed in a different order produced
different resamples and therefore different interval endpoints -- measured at
0.0162 in p, which was noted at premortem checkpoint 2 (item C1) and left open
as something that "moves every published interval".

It was tolerable while the gate rule was `P(delta <= 0.10) < 0.05` evaluated at
0.535, nowhere near its threshold. Gate 2's rule is `ci_low > 0`: a BOUNDARY
test, on a small eval, where 0.0162 is the difference between PASS and FAIL.

The fix is one independent stream per fold, keyed to a hash of THAT FOLD'S
CONTENTS -- the construction `scripts/analysis/gate_rule_candidates._stratum_rng`
already uses in this repository. Keying on position instead would not do: a fold
at index 0 and the same fold at index 1 would still draw differently, which is
the property these tests are about. The estimand does not change and neither
does determinism for a given seed; what changes is that a fold's resamples no
longer depend on its neighbours.
"""

from __future__ import annotations

import numpy as np
import pytest

from tract.training.evaluate import (
    _build_fold_index_matrix,
    fold_stratified_bootstrap_ci,
    paired_bootstrap_delta,
)


def _folds(rng: np.random.Generator, sizes: list[int]) -> list[np.ndarray]:
    return [rng.integers(0, 2, size=n).astype(float) for n in sizes]


class TestTheIndexMatrixItself:
    def test_a_folds_draw_does_not_depend_on_its_neighbours(self) -> None:
        """The defect, stated directly.

        Fold B is the same size in both calls. Its resampled indices -- offset
        removed -- must be identical whether or not a differently-sized fold
        preceded it.
        """
        rng = np.random.default_rng(7)
        big, small = rng.integers(0, 2, 33).astype(float), rng.integers(
            0, 2, 17
        ).astype(float)
        _, alone = _build_fold_index_matrix([small], n_resamples=64, seed=42)
        _, preceded = _build_fold_index_matrix(
            [big, small], n_resamples=64, seed=42
        )
        assert np.array_equal(alone[0], preceded[1] - 33), (
            "the 17-item fold drew different indices because a 33-item fold "
            "consumed the generator first"
        )

    def test_still_deterministic_for_a_seed(self) -> None:
        folds = _folds(np.random.default_rng(7), [33, 17])
        a, _ = _build_fold_index_matrix(folds, n_resamples=64, seed=42)
        b, _ = _build_fold_index_matrix(folds, n_resamples=64, seed=42)
        assert np.array_equal(a, b)

    def test_different_seeds_give_different_draws(self) -> None:
        folds = _folds(np.random.default_rng(7), [33, 17])
        a, _ = _build_fold_index_matrix(folds, n_resamples=64, seed=42)
        b, _ = _build_fold_index_matrix(folds, n_resamples=64, seed=43)
        assert not np.array_equal(a, b)

    def test_indices_stay_inside_their_own_fold(self) -> None:
        """Offsets must still address the concatenated array correctly."""
        _, per_fold = _build_fold_index_matrix(
            _folds(np.random.default_rng(7), [33, 17, 24]),
            n_resamples=32, seed=42,
        )
        assert per_fold[0].min() >= 0 and per_fold[0].max() < 33
        assert per_fold[1].min() >= 33 and per_fold[1].max() < 50
        assert per_fold[2].min() >= 50 and per_fold[2].max() < 74


class TestTheGateStatisticIsOrderIndependent:
    """What actually matters: permuting the fold list must not move ci_low."""

    def test_paired_delta_is_identical_under_fold_permutation(self) -> None:
        rng = np.random.default_rng(0)
        a = _folds(rng, [33, 17, 24])
        b = [x.copy() for x in a]
        b[0][:8] = 1.0  # a real effect in one fold

        forward = paired_bootstrap_delta(a, b, n_resamples=2000, seed=42)
        reverse = paired_bootstrap_delta(
            a[::-1], b[::-1], n_resamples=2000, seed=42
        )
        assert forward["delta_mean"] == pytest.approx(reverse["delta_mean"])
        assert forward["ci_low"] == pytest.approx(reverse["ci_low"]), (
            "ci_low moved when the folds were listed in a different order. "
            "Gate 2's rule is ci_low > 0, so this is the difference between "
            "PASS and FAIL on identical data."
        )
        assert forward["ci_high"] == pytest.approx(reverse["ci_high"])

    def test_stratified_ci_is_identical_under_fold_permutation(self) -> None:
        rng = np.random.default_rng(1)
        folds = _folds(rng, [33, 17, 24])
        forward = fold_stratified_bootstrap_ci(folds, n_resamples=2000, seed=42)
        reverse = fold_stratified_bootstrap_ci(
            folds[::-1], n_resamples=2000, seed=42
        )
        assert forward["ci_low"] == pytest.approx(reverse["ci_low"])
        assert forward["ci_high"] == pytest.approx(reverse["ci_high"])

    def test_the_mean_was_never_the_problem(self) -> None:
        """Sanity: the point estimate was always order-independent.

        Only the interval moved, which is why this went unnoticed -- every
        headline number this project reports is a point estimate plus an
        interval, and the point estimate agreed.
        """
        rng = np.random.default_rng(2)
        folds = _folds(rng, [33, 17])
        assert fold_stratified_bootstrap_ci(folds, n_resamples=500, seed=42)[
            "mean"
        ] == pytest.approx(
            fold_stratified_bootstrap_ci(
                folds[::-1], n_resamples=500, seed=42
            )["mean"]
        )
