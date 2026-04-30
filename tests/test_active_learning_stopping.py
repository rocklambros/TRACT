"""Tests for active learning stopping criteria."""
from __future__ import annotations

import pytest


class TestEvaluateStopping:
    def test_all_met_returns_stop(self) -> None:
        from tract.active_learning.stopping import evaluate_stopping_criteria

        result = evaluate_stopping_criteria(
            acceptance_rate=0.85,
            canary_accuracy=0.90,
            unique_hubs_accepted=60,
        )
        assert result["should_stop"] is True
        assert all(result["criteria_met"].values())

    def test_low_acceptance_returns_continue(self) -> None:
        from tract.active_learning.stopping import evaluate_stopping_criteria

        result = evaluate_stopping_criteria(
            acceptance_rate=0.70,
            canary_accuracy=0.90,
            unique_hubs_accepted=60,
        )
        assert result["should_stop"] is False
        assert result["criteria_met"]["acceptance_rate"] is False

    def test_low_canary_returns_continue(self) -> None:
        from tract.active_learning.stopping import evaluate_stopping_criteria

        result = evaluate_stopping_criteria(
            acceptance_rate=0.85,
            canary_accuracy=0.40,
            unique_hubs_accepted=60,
        )
        assert result["should_stop"] is False

    def test_low_diversity_returns_continue(self) -> None:
        from tract.active_learning.stopping import evaluate_stopping_criteria

        result = evaluate_stopping_criteria(
            acceptance_rate=0.85,
            canary_accuracy=0.90,
            unique_hubs_accepted=30,
        )
        assert result["should_stop"] is False
        assert result["criteria_met"]["hub_diversity"] is False
