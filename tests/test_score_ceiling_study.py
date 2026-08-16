"""Tests for the ceiling study scorer's scoring and validation logic."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.score_ceiling_study import (
    AnswerRow,
    _load_answers,
    score_items,
)

METADATA: dict[int, tuple[str, str]] = {
    1: ("capec", "validation"),
    2: ("cwe", "validation"),
    3: ("mitre_atlas", "test"),
}

KEY: dict[int, list[str]] = {
    1: ["hub-a"],
    2: ["hub-b", "hub-c"],  # multi-hub anchor
    3: ["hub-d"],
}


def _row(item_index: int, primary: str, acceptable: list[str] | None = None) -> AnswerRow:
    return {
        "item_index": item_index,
        "primary_hub_id": primary,
        "acceptable_hub_ids": acceptable or [],
        "confidence": "high" if primary else "",
        "notes": "",
    }


class TestScoreItems:
    def test_skips_rows_with_empty_primary(self) -> None:
        rows = [_row(1, ""), _row(2, "hub-b")]
        scored = score_items(rows, KEY, METADATA)
        assert [s["item_index"] for s in scored] == [2]

    def test_alpha1_hit_when_primary_matches_gold(self) -> None:
        rows = [_row(1, "hub-a")]
        scored = score_items(rows, KEY, METADATA)
        assert scored[0]["alpha1_hit"] is True
        assert scored[0]["alpha5_hit"] is True

    def test_alpha1_miss_when_primary_wrong(self) -> None:
        rows = [_row(1, "hub-wrong")]
        scored = score_items(rows, KEY, METADATA)
        assert scored[0]["alpha1_hit"] is False

    def test_alpha1_hit_against_either_member_of_a_multihub_anchor(self) -> None:
        rows = [_row(2, "hub-c")]  # not the "first" gold hub, still valid
        scored = score_items(rows, KEY, METADATA)
        assert scored[0]["alpha1_hit"] is True

    def test_alpha5_hit_when_gold_in_acceptable_set_but_not_primary(self) -> None:
        rows = [_row(3, "hub-wrong", acceptable=["hub-d", "hub-other"])]
        scored = score_items(rows, KEY, METADATA)
        assert scored[0]["alpha1_hit"] is False
        assert scored[0]["alpha5_hit"] is True

    def test_alpha5_miss_when_gold_in_neither_primary_nor_acceptable(self) -> None:
        rows = [_row(3, "hub-wrong", acceptable=["hub-other-1", "hub-other-2"])]
        scored = score_items(rows, KEY, METADATA)
        assert scored[0]["alpha5_hit"] is False

    def test_unknown_item_index_raises(self) -> None:
        rows = [_row(999, "hub-a")]
        with pytest.raises(ValueError, match="no entry in the answer key"):
            score_items(rows, KEY, METADATA)

    def test_carries_framework_and_stratum_from_metadata(self) -> None:
        rows = [_row(3, "hub-d")]
        scored = score_items(rows, KEY, METADATA)
        assert scored[0]["framework_id"] == "mitre_atlas"
        assert scored[0]["stratum"] == "test"


class TestLoadAnswersValidation:
    def _write(self, tmp_path: Path, items: list[dict[str, object]]) -> Path:
        path = tmp_path / "answers.json"
        path.write_text(json.dumps({"items": items}), encoding="utf-8")
        return path

    def test_accepts_blank_template_rows(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, [
            {"item_index": 1, "primary_hub_id": "", "acceptable_hub_ids": [],
             "confidence": "", "notes": ""},
        ])
        rows = _load_answers(path)
        assert rows[0]["primary_hub_id"] == ""

    def test_rejects_invalid_confidence(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, [
            {"item_index": 1, "primary_hub_id": "hub-a", "acceptable_hub_ids": [],
             "confidence": "extremely-sure", "notes": ""},
        ])
        with pytest.raises(ValueError, match="confidence"):
            _load_answers(path)

    def test_rejects_more_than_five_acceptable_hubs_on_a_completed_row(
        self, tmp_path: Path,
    ) -> None:
        path = self._write(tmp_path, [
            {"item_index": 1, "primary_hub_id": "hub-a",
             "acceptable_hub_ids": ["a", "b", "c", "d", "e", "f"],
             "confidence": "high", "notes": ""},
        ])
        with pytest.raises(ValueError, match="more than the 5 allowed"):
            _load_answers(path)

    def test_tolerates_too_many_acceptable_hubs_on_an_unfilled_row(
        self, tmp_path: Path,
    ) -> None:
        """An empty primary_hub_id means the row is not yet reviewed. A stray
        list in acceptable_hub_ids on an unfilled row should not block loading
        the rest of the (possibly still in-progress) worksheet."""
        path = self._write(tmp_path, [
            {"item_index": 1, "primary_hub_id": "",
             "acceptable_hub_ids": ["a", "b", "c", "d", "e", "f"],
             "confidence": "", "notes": ""},
        ])
        rows = _load_answers(path)
        assert rows[0]["primary_hub_id"] == ""
