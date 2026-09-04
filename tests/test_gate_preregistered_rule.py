"""The gate quantity CAMPAIGN3 §3 binds is not the one the code computes.

§3 reads, verbatim:

    **PASS iff `P(true delta <= 0.10) < 0.05`.**

`gate_decision` reported neither side of that. It reported `ci_low_pass`
(`ci_low > threshold` on a two-sided 95% interval, which is the SAME rule at
half the alpha) and `p_value` (which `paired_bootstrap_delta` computes as
`mean(boot <= 0)` -- P(delta <= 0), a different hypothesis under a name a
reader takes for the gate statistic).

So the binding quantity existed in three campaign write-ups and in no line of
code. Every figure quoting it came from an ad-hoc script. These tests pin it
into the library and pin the arithmetic relationship between the two rules, so
that "which alpha is this" stops being answerable only by reading a percentile
call.
"""

from __future__ import annotations

import numpy as np
import pytest

from tract.config import PHASE1B_GATE_HIT1_DELTA, PREREGISTERED_GATE_ALPHA
from tract.training.evaluate import paired_bootstrap_delta


def _gate_decision():  # type: ignore[no-untyped-def]
    """Import gate_decision lazily; tract.training.orchestrate pulls in torch.

    Module-level it would raise at COLLECTION on a runner without torch, and
    pytest aborts the entire run on a collection error -- which is how this
    repository once executed 15 tests while reading as an ordinary red. The
    alpha and the bootstrap both live in torch-free modules, so most of this
    file runs everywhere; only the tests that call gate_decision skip.
    """
    pytest.importorskip("torch", reason="tract.training.orchestrate needs it")
    pytest.importorskip("datasets", reason="tract.training.orchestrate needs it")
    from tract.training.orchestrate import gate_decision

    return gate_decision


def _folds(
    baseline_rate: float, trained_rate: float, n: int, seed: int = 0
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Two paired indicator folds at the requested marginal hit rates.

    Deterministic and unpaired-in-content on purpose: the delta is what is
    under test, not the pairing structure.
    """
    rng = np.random.default_rng(seed)
    base = (rng.random(n) < baseline_rate).astype(float)
    trained = (rng.random(n) < trained_rate).astype(float)
    return [base], [trained]


class TestBootstrapReportsTheGateQuantity:
    """`paired_bootstrap_delta` must be able to answer P(delta <= t)."""

    def test_p_delta_le_threshold_is_returned(self) -> None:
        a, b = _folds(0.40, 0.60, 200)
        out = paired_bootstrap_delta(a, b, threshold=0.10)
        assert "p_delta_le_threshold" in out

    def test_at_threshold_zero_it_reproduces_the_legacy_p_value(self) -> None:
        """The old `p_value` is this quantity at t=0, and must stay equal to it.

        If these diverge, one of the two is not measuring what its name says.
        """
        a, b = _folds(0.40, 0.60, 200)
        out = paired_bootstrap_delta(a, b, threshold=0.0)
        assert out["p_delta_le_threshold"] == pytest.approx(out["p_value"])

    def test_the_two_quantities_differ_at_the_gate_threshold(self) -> None:
        """The whole point: P(d<=0) and P(d<=0.10) are different numbers.

        Campaign 2 reported P(delta <= 0.10) = 0.203 while its bootstrap
        p_value (P(delta <= 0)) was far smaller. A reader taking `p_value` for
        the gate statistic reads a pass where the gate says fail.
        """
        a, b = _folds(0.40, 0.52, 300)
        out = paired_bootstrap_delta(a, b, threshold=PHASE1B_GATE_HIT1_DELTA)
        assert out["p_delta_le_threshold"] > out["p_value"]

    def test_threshold_defaults_to_zero_so_existing_callers_are_unchanged(
        self,
    ) -> None:
        a, b = _folds(0.40, 0.60, 200)
        assert paired_bootstrap_delta(a, b)["p_delta_le_threshold"] == pytest.approx(
            paired_bootstrap_delta(a, b)["p_value"]
        )


class TestGateDecisionImplementsSectionThree:
    """`gate_decision` must publish the pre-registered verdict, by that name."""

    @staticmethod
    def _records(
        baseline_rate: float, trained_rate: float, n: int, seed: int = 0
    ) -> list[dict[str, object]]:
        a, b = _folds(baseline_rate, trained_rate, n, seed)
        return [
            {
                "held_out_framework": "F",
                "hit1_indicators": b[0].tolist(),
                "zero_shot": {"hit1_indicators": a[0].tolist()},
            }
        ]

    def test_reports_the_bound_probability(self) -> None:
        d = _gate_decision()(self._records(0.40, 0.60, 200))
        assert "p_delta_le_threshold" in d

    def test_reports_the_preregistered_verdict(self) -> None:
        d = _gate_decision()(self._records(0.40, 0.60, 200))
        assert "preregistered_pass" in d

    def test_the_verdict_applies_alpha_005_not_0025(self) -> None:
        """Pin the alpha itself, so a mutation to 0.025 goes red.

        The previously-published verdict, `ci_low_pass`, is this rule at
        alpha=0.025 -- strictly harder than the document binds.
        """
        assert PREREGISTERED_GATE_ALPHA == 0.05

    def test_verdict_agrees_with_the_probability_it_is_derived_from(self) -> None:
        d = _gate_decision()(self._records(0.40, 0.60, 200))
        assert d["preregistered_pass"] == (
            d["p_delta_le_threshold"] < PREREGISTERED_GATE_ALPHA
        )

    def test_p_value_is_labelled_as_the_zero_hypothesis(self) -> None:
        """`p_value` stays for compatibility but must not be the only name.

        An explicit `p_delta_le_zero` key makes the two hypotheses
        distinguishable without reading the bootstrap source.
        """
        d = _gate_decision()(self._records(0.40, 0.60, 200))
        assert d["p_delta_le_zero"] == pytest.approx(d["p_value"])


class TestTheTwoRulesDisagreeInTheAlphaBand:
    """The strictness gap is real, not theoretical -- construct a case in it.

    A study whose P(delta <= 0.10) lands strictly between 0.025 and 0.05 PASSES
    the pre-registered rule and FAILS the rule the code published. That band is
    where the substitution changes a verdict, so it needs a test that would
    have caught the substitution.
    """

    def test_a_study_in_the_band_passes_section_three_and_fails_ci_low(
        self,
    ) -> None:
        found = None
        for seed in range(400):
            a, b = _folds(0.40, 0.58, 400, seed=seed)
            out = paired_bootstrap_delta(a, b, threshold=PHASE1B_GATE_HIT1_DELTA)
            if 0.025 < out["p_delta_le_threshold"] < 0.05:
                found = (seed, out)
                break
        assert found is not None, (
            "No study landed in the (0.025, 0.05) band across 400 seeds; "
            "the fixture no longer probes the region where the two rules "
            "disagree, so this test has stopped testing anything."
        )
        seed, out = found
        records = TestGateDecisionImplementsSectionThree._records(
            0.40, 0.58, 400, seed=seed
        )
        d = _gate_decision()(records)
        assert d["preregistered_pass"] is True
        assert d["ci_low_pass"] is False
