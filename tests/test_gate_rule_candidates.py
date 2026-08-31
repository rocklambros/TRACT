"""Tests for the candidate §1.2 pooling rules.

Each rule is a pure predicate over a Contrast, so these run with no corpus and
no result directory. The point of the suite is that every rule's behaviour on
the *worked example* is pinned: if someone later widens a margin, the test that
says "this rule must refuse the audit-touched contrast" is what fails.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.analysis.gate_rule_candidates import (
    BASELINE_ALPHA,
    BASELINE_MARGIN,
    EQUIVALENCE_MARGIN,
    Contrast,
    bootstrap_baselines,
    bootstrap_deltas,
    r0_status_quo,
    r1_difference_includes_zero,
    r2_equivalence,
    r3_never_pool,
    r4_baseline_symmetry,
    r5_equivalence_and_baseline,
    r6_baseline_not_significantly_different,
)
from tests.test_audit_mechanism_probe import _row


def _contrast(
    *, delta_a: float = 0.10, delta_b: float = 0.10,
    ci_a: tuple[float, float] = (0.0, 0.2),
    ci_b: tuple[float, float] = (0.0, 0.2),
    diff: float = 0.0, diff_ci: tuple[float, float] = (-0.02, 0.02),
    zs_a: float = 0.5, zs_b: float = 0.5,
    baseline_diff: float = 0.0,
    baseline_diff_ci: tuple[float, float] = (-0.05, 0.05),
    baseline_p: float = 0.9,
) -> Contrast:
    return Contrast(
        delta_a=delta_a, delta_b=delta_b, ci_a=ci_a, ci_b=ci_b,
        diff=diff, diff_ci=diff_ci, zero_shot_a=zs_a, zero_shot_b=zs_b,
        baseline_diff=baseline_diff, baseline_diff_ci=baseline_diff_ci,
        baseline_p_value=baseline_p,
    )


# The real contrast from docs/campaign3-audit-mechanism.md §6b, to two more
# places than the prose quotes. Any rule adopted for CAMPAIGN3 §1.2 must refuse
# this one -- that is the whole selection criterion.
WORKED_EXAMPLE: Contrast = _contrast(
    delta_a=0.1000, delta_b=0.2703,
    ci_a=(0.0000, 0.2000), ci_b=(0.1081, 0.4595),
    diff=0.1703, diff_ci=(-0.0283, 0.3693),
    zs_a=0.5273, zs_b=0.1892,
    baseline_diff=-0.3381, baseline_diff_ci=(-0.4900, -0.1800),
    baseline_p=0.0,
)


class TestTheWorkedExampleIsTheSelectionCriterion:
    """Pin every rule's verdict on the contrast that motivated the change."""

    def test_status_quo_admits_the_inflation(self) -> None:
        # This is the defect. If this ever starts passing, the intervals in
        # WORKED_EXAMPLE were edited, not the rule.
        assert r0_status_quo(WORKED_EXAMPLE) is True

    def test_naive_difference_test_also_admits_it(self) -> None:
        assert r1_difference_includes_zero(WORKED_EXAMPLE) is True

    @pytest.mark.parametrize("rule", [
        r2_equivalence, r3_never_pool, r4_baseline_symmetry,
        r5_equivalence_and_baseline, r6_baseline_not_significantly_different,
    ])
    def test_every_candidate_replacement_refuses_it(self, rule: object) -> None:
        assert rule(WORKED_EXAMPLE) is False  # type: ignore[operator]


class TestR0StatusQuo:
    def test_permits_when_intervals_overlap(self) -> None:
        assert r0_status_quo(_contrast(ci_a=(0.0, 0.2), ci_b=(0.1, 0.4))) is True

    def test_refuses_when_intervals_are_disjoint(self) -> None:
        assert r0_status_quo(_contrast(ci_a=(0.0, 0.1), ci_b=(0.2, 0.4))) is False

    def test_touching_endpoints_count_as_overlap(self) -> None:
        assert r0_status_quo(_contrast(ci_a=(0.0, 0.2), ci_b=(0.2, 0.4))) is True


class TestR2Equivalence:
    def test_permits_a_difference_interval_inside_the_margin(self) -> None:
        m = EQUIVALENCE_MARGIN
        assert r2_equivalence(_contrast(diff_ci=(-m / 2, m / 2))) is True

    def test_refuses_when_the_interval_spills_past_the_margin(self) -> None:
        m = EQUIVALENCE_MARGIN
        assert r2_equivalence(_contrast(diff_ci=(-m / 2, m * 2))) is False

    def test_a_wide_interval_centred_on_zero_is_refused(self) -> None:
        # The key difference from R1: covering zero is not enough. An
        # uninformative test must refuse, not permit.
        assert r1_difference_includes_zero(_contrast(diff_ci=(-0.5, 0.5))) is True
        assert r2_equivalence(_contrast(diff_ci=(-0.5, 0.5))) is False


class TestR4BaselineSymmetry:
    def test_permits_comparable_baselines(self) -> None:
        assert r4_baseline_symmetry(_contrast(zs_a=0.50, zs_b=0.55)) is True

    def test_refuses_baselines_further_apart_than_the_margin(self) -> None:
        assert r4_baseline_symmetry(
            _contrast(zs_a=0.50, zs_b=0.50 + BASELINE_MARGIN * 2)) is False

    def test_is_symmetric_in_the_two_strata(self) -> None:
        a = _contrast(zs_a=0.20, zs_b=0.60)
        b = _contrast(zs_a=0.60, zs_b=0.20)
        assert r4_baseline_symmetry(a) == r4_baseline_symmetry(b)


class TestR6BaselineSignificance:
    def test_permits_when_baselines_are_not_significantly_different(self) -> None:
        assert r6_baseline_not_significantly_different(
            _contrast(baseline_p=0.40)) is True

    def test_refuses_when_they_are(self) -> None:
        assert r6_baseline_not_significantly_different(
            _contrast(baseline_p=0.001)) is False

    def test_boundary_is_inclusive_at_alpha(self) -> None:
        assert r6_baseline_not_significantly_different(
            _contrast(baseline_p=BASELINE_ALPHA)) is True

    def test_does_not_look_at_the_delta_at_all(self) -> None:
        # R6's whole premise is that the baseline carries the signal. A wildly
        # different delta with matched baselines must still permit.
        c = _contrast(delta_a=0.0, delta_b=0.9, diff=0.9,
                      diff_ci=(0.8, 1.0), baseline_p=0.9)
        assert r6_baseline_not_significantly_different(c) is True


class TestR3AndR5:
    def test_never_pool_refuses_everything(self) -> None:
        assert r3_never_pool(_contrast()) is False
        assert r3_never_pool(WORKED_EXAMPLE) is False

    def test_conjunction_requires_both_halves(self) -> None:
        m = EQUIVALENCE_MARGIN
        good_delta_bad_baseline = _contrast(
            diff_ci=(-m / 2, m / 2), zs_a=0.2, zs_b=0.8)
        assert r2_equivalence(good_delta_bad_baseline) is True
        assert r5_equivalence_and_baseline(good_delta_bad_baseline) is False


class TestBootstrapHelpers:
    def test_baseline_bootstrap_recovers_the_zero_shot_rate(self) -> None:
        rows = [_row("a", 1, 1)] * 3 + [_row("a", 0, 0)] * 1
        draws = bootstrap_baselines(rows, 500, np.random.default_rng(0))
        assert draws.mean() == pytest.approx(0.75, abs=0.05)

    def test_delta_bootstrap_matches_the_observed_mean(self) -> None:
        rows = [_row("a", 1, 0)] * 3 + [_row("a", 0, 0)]
        draws = bootstrap_deltas(rows, 500, np.random.default_rng(0))
        assert draws.mean() == pytest.approx(0.75, abs=0.05)

    def test_both_helpers_are_deterministic_for_a_seed(self) -> None:
        rows = [_row("a", 1, 0), _row("b", 0, 1), _row("b", 1, 1)]
        first = bootstrap_deltas(rows, 200, np.random.default_rng(7))
        second = bootstrap_deltas(rows, 200, np.random.default_rng(7))
        np.testing.assert_array_equal(first, second)
