"""Tests for canary item management."""
from __future__ import annotations

import pytest


class TestSelectAICanaries:
    def test_selects_correct_count(self) -> None:
        from tract.active_learning.canary import select_ai_canaries

        controls = [{"control_id": f"c{i}", "framework": "CSA AICM", "control_text": f"text {i}"} for i in range(100)]
        canaries = select_ai_canaries(controls, n=20, seed=42)
        assert len(canaries) == 20

    def test_deterministic(self) -> None:
        from tract.active_learning.canary import select_ai_canaries

        controls = [{"control_id": f"c{i}", "framework": "CSA AICM", "control_text": f"text {i}"} for i in range(100)]
        c1 = select_ai_canaries(controls, seed=42)
        c2 = select_ai_canaries(controls, seed=42)
        assert [c["control_id"] for c in c1] == [c["control_id"] for c in c2]


class TestEvaluateCanaries:
    def test_perfect_accuracy(self) -> None:
        from tract.active_learning.canary import evaluate_canary_accuracy

        canary_labels = {"c1": frozenset({"h1"}), "c2": frozenset({"h2"})}
        review_items = [
            {"control_id": "c1", "review": {"status": "accepted"}, "predictions": [{"hub_id": "h1"}]},
            {"control_id": "c2", "review": {"status": "accepted"}, "predictions": [{"hub_id": "h2"}]},
        ]

        accuracy = evaluate_canary_accuracy(canary_labels, review_items)
        assert accuracy == 1.0

    def test_partial_accuracy(self) -> None:
        from tract.active_learning.canary import evaluate_canary_accuracy

        canary_labels = {"c1": frozenset({"h1"}), "c2": frozenset({"h2"})}
        review_items = [
            {"control_id": "c1", "review": {"status": "accepted"}, "predictions": [{"hub_id": "h1"}]},
            {"control_id": "c2", "review": {"status": "accepted"}, "predictions": [{"hub_id": "h_wrong"}]},
        ]

        accuracy = evaluate_canary_accuracy(canary_labels, review_items)
        assert accuracy == 0.5
