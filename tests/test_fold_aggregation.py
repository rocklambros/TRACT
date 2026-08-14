"""Tests for per-fold result persistence and micro-aggregation.

The RunPod path wrote a per-fold summary that kept only the metric dict and
dropped the per-item hit@1 indicators, and nothing aggregated the five summaries
at all. Averaging them by hand is a MACRO average: it weights a 6-item fold the
same as a 60-item one. TRACT reports a MICRO average, which needs the raw
indicators from every fold.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from tract.training.orchestrate import (
    FOLD_RESULT_FILENAME,
    aggregate_fold_results,
    load_fold_results,
)


def _write_fold(root: Path, framework: str, indicators: list[int]) -> None:
    fold_dir = root / f"fold_{framework.replace(' ', '_')}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "held_out_framework": framework,
        "metrics": {"hit_at_1": sum(indicators) / len(indicators)},
        "hit1_indicators": indicators,
        "n_eval_items": len(indicators),
        "n_training_pairs": 100,
        "elapsed_s": 1.0,
    }
    (fold_dir / FOLD_RESULT_FILENAME).write_text(json.dumps(record), encoding="utf-8")


class TestMicroVersusMacro:

    def test_micro_average_weights_items_not_folds(self) -> None:
        """The whole point of the fix, stated as an assertion.

        One small fold scoring perfectly and one large fold scoring poorly. The
        macro average is the mean of the two rates; the micro average is the
        mean over items. They must not be the same number.
        """
        small = [1] * 6                 # n=6,  hit@1 = 1.0
        large = [1] * 20 + [0] * 40     # n=60, hit@1 = 0.3333

        result = aggregate_fold_results([
            {"held_out_framework": "Tiny", "hit1_indicators": small, "n_eval_items": 6},
            {"held_out_framework": "Big", "hit1_indicators": large, "n_eval_items": 60},
        ])

        expected_micro = (6 + 20) / 66
        expected_macro = (1.0 + 20 / 60) / 2

        assert result["mean"] == pytest.approx(expected_micro, abs=1e-9)
        assert result["macro_mean"] == pytest.approx(expected_macro, abs=1e-9)
        assert result["n_total"] == 66
        # 0.394 vs 0.667 -- averaging fold summaries would have overstated the
        # headline by 27 points here.
        assert abs(result["mean"] - result["macro_mean"]) > 0.25

    def test_fold_sizes_are_recorded(self) -> None:
        result = aggregate_fold_results([
            {"held_out_framework": "A", "hit1_indicators": [1, 0], "n_eval_items": 2},
            {"held_out_framework": "B", "hit1_indicators": [1, 1, 1], "n_eval_items": 3},
        ])
        assert result["fold_sizes"] == {"A": 2, "B": 3}
        assert result["n_folds"] == 2

    def test_confidence_interval_brackets_the_estimate(self) -> None:
        result = aggregate_fold_results([
            {"held_out_framework": "A", "hit1_indicators": [1, 0] * 25, "n_eval_items": 50},
            {"held_out_framework": "B", "hit1_indicators": [1, 1, 0] * 20, "n_eval_items": 60},
        ])
        assert result["ci_low"] < result["mean"] < result["ci_high"]


class TestLoadFoldResults:

    def test_loads_and_sorts_by_framework(self, tmp_path: Path) -> None:
        _write_fold(tmp_path, "OWASP Top10 for LLM", [1, 0, 1])
        _write_fold(tmp_path, "MITRE ATLAS", [1, 1])

        records = load_fold_results(tmp_path)
        assert [r["held_out_framework"] for r in records] == [
            "MITRE ATLAS", "OWASP Top10 for LLM",
        ]

    def test_rejects_a_record_missing_indicators(self, tmp_path: Path) -> None:
        """This is the shape the old RunPod summary wrote."""
        fold_dir = tmp_path / "fold_MITRE_ATLAS"
        fold_dir.mkdir(parents=True)
        (fold_dir / FOLD_RESULT_FILENAME).write_text(
            json.dumps({
                "held_out_framework": "MITRE ATLAS",
                "metrics": {"hit_at_1": 0.5},
                "n_eval_items": 10,
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="hit1_indicators"):
            load_fold_results(tmp_path)

    def test_rejects_inconsistent_indicator_count(self, tmp_path: Path) -> None:
        fold_dir = tmp_path / "fold_MITRE_ATLAS"
        fold_dir.mkdir(parents=True)
        (fold_dir / FOLD_RESULT_FILENAME).write_text(
            json.dumps({
                "held_out_framework": "MITRE ATLAS",
                "metrics": {},
                "hit1_indicators": [1, 0],
                "n_eval_items": 10,
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="internally inconsistent"):
            load_fold_results(tmp_path)

    def test_refuses_to_aggregate_nothing(self, tmp_path: Path) -> None:
        """An empty results directory must not quietly produce a number."""
        with pytest.raises(ValueError, match="Nothing to aggregate"):
            load_fold_results(tmp_path)
