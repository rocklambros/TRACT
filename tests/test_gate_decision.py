"""Tests for the Gate 1 decision.

The gate used to be four lines in scripts/phase1b/train.py comparing the
aggregate hit@1 against a hardcoded zero-shot constant (0.399) and passing on
the point estimate. paired_bootstrap_delta existed with no non-test callers.

A point estimate is not a decision when the interval around it contains the
threshold, so both verdicts are computed and reported.
"""
from __future__ import annotations

import pytest

# See tests/test_fold_aggregation.py: orchestrate pulls in torch and datasets,
# and CI runs pytest with -x, so an unguarded import aborts the entire job.
pytest.importorskip("numpy")
pytest.importorskip("torch", reason="needs the phase0 extra")
pytest.importorskip("datasets", reason="needs the phase0 extra")

from tract.training.orchestrate import gate_decision


def _fold(name: str, trained: list[int], zero_shot: list[int]) -> dict:
    return {
        "held_out_framework": name,
        "hit1_indicators": trained,
        "n_eval_items": len(trained),
        "metrics": {"hit_at_1": sum(trained) / len(trained)},
        "zero_shot": {"hit1_indicators": zero_shot},
    }


class TestGateDecision:

    def test_delta_is_trained_minus_baseline(self) -> None:
        """Sign convention: a better trained model gives a positive delta."""
        folds = [_fold("A", [1] * 8 + [0] * 2, [1] * 3 + [0] * 7)]  # 0.8 vs 0.3
        decision = gate_decision(folds)
        assert decision["micro_delta"] == pytest.approx(0.5, abs=1e-9)
        assert decision["per_fold"]["A"]["trained_hit1"] == pytest.approx(0.8)
        assert decision["per_fold"]["A"]["zero_shot_hit1"] == pytest.approx(0.3)
        assert decision["paired"] is True

    def test_large_effect_passes_both_verdicts(self) -> None:
        folds = [
            _fold("A", [1] * 45 + [0] * 5, [1] * 10 + [0] * 40),
            _fold("B", [1] * 40 + [0] * 10, [1] * 12 + [0] * 38),
        ]
        decision = gate_decision(folds)
        assert decision["point_estimate_pass"] is True
        assert decision["ci_low_pass"] is True
        assert decision["verdicts_agree"] is True

    def test_marginal_effect_splits_the_verdicts(self) -> None:
        """The case the old gate could not express.

        A delta above 0.10 on a small eval set, with an interval that still
        contains 0.10. The pre-registered criterion passes; the lower bound does
        not. Both are reported and the disagreement is explicit.
        """
        # 20 items: 13 hits trained, 10 baseline, 3 of them discordant upward.
        trained = [1] * 13 + [0] * 7
        baseline = [1] * 10 + [0] * 10
        decision = gate_decision([_fold("Small", trained, baseline)])

        assert decision["micro_delta"] == pytest.approx(0.15, abs=1e-9)
        assert decision["point_estimate_pass"] is True
        assert decision["ci_low"] < 0.10
        assert decision["ci_low_pass"] is False
        assert decision["verdicts_agree"] is False

    def test_ci_low_is_never_the_easier_test(self) -> None:
        """ci_low <= delta_mean always, so ci_low_pass implies point pass."""
        folds = [
            _fold("A", [1] * 30 + [0] * 20, [1] * 15 + [0] * 35),
            _fold("B", [1] * 4 + [0] * 2, [1] * 2 + [0] * 4),
        ]
        decision = gate_decision(folds)
        assert decision["ci_low"] <= decision["micro_delta"]
        if decision["ci_low_pass"]:
            assert decision["point_estimate_pass"]

    def test_refuses_folds_without_a_paired_baseline(self) -> None:
        """No substituting a baseline measured in some other run."""
        folds = [_fold("A", [1, 0], [1, 1])]
        del folds[0]["zero_shot"]
        with pytest.raises(ValueError, match="no paired zero-shot indicators"):
            gate_decision(folds)

    def test_flags_a_fold_that_regressed(self) -> None:
        """PRD 6.4 requires any fold below its own zero-shot to be flagged."""
        folds = [
            _fold("Good", [1] * 40 + [0] * 10, [1] * 10 + [0] * 40),
            _fold("Regressed", [1] * 2 + [0] * 8, [1] * 5 + [0] * 5),
        ]
        decision = gate_decision(folds)
        assert decision["negative_folds"] == ["Regressed"]
        assert decision["worst_fold"] == "Regressed"
        assert decision["worst_fold_delta"] < 0

    def test_reports_macro_beside_micro(self) -> None:
        """Both diagnostics PRD 6.4 asks for, over deliberately uneven folds."""
        folds = [
            _fold("Tiny", [1] * 6, [0] * 6),                       # n=6,  delta +1.0
            _fold("Big", [1] * 20 + [0] * 40, [1] * 18 + [0] * 42),  # n=60, delta +0.0333
        ]
        decision = gate_decision(folds)
        assert decision["macro_delta"] == pytest.approx((1.0 + 2 / 60) / 2, abs=1e-9)
        assert decision["micro_delta"] == pytest.approx((6 + 2) / 66, abs=1e-9)
        # Macro says +0.52, micro says +0.12. Fold size is the entire difference.
        assert decision["macro_delta"] > decision["micro_delta"] + 0.35
        assert decision["n_total"] == 66

    def test_threshold_is_configurable_but_defaults_to_the_registered_value(
        self,
    ) -> None:
        folds = [_fold("A", [1] * 30 + [0] * 20, [1] * 20 + [0] * 30)]
        assert gate_decision(folds)["threshold"] == pytest.approx(0.10)
        assert gate_decision(folds, threshold=0.05)["threshold"] == pytest.approx(0.05)
