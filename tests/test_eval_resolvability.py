"""Tests for the eval-anchor resolvability guard in tract/text_selection.py.

Preserving item identity across arms is necessary but not sufficient. Several
NIST AI 100-2 sections expand to the same paragraph, so after prose
substitution ten eval items formed groups sharing an anchor while pointing at
different ground-truth hubs. No model can score better than one item per
group, which caps the prose arms in a way the title arm is not capped -- the
arm comparison would then be measuring a data collision, not the anchor.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from tract.text_selection import _keep_items_resolvable


@dataclass
class _Item:
    framework_name: str
    control_text: str
    ground_truth_hub_id: str
    section_id: str = ""
    track: str = "title"


def _pair(titles: list[str], prose: list[str], hubs: list[str]) -> tuple[Any, Any]:
    before = [
        _Item("NIST AI 100-2", t, h) for t, h in zip(titles, hubs)
    ]
    after = [
        replace(i, control_text=p, track="full-text") for i, p in zip(before, prose)
    ]
    return before, after


class TestConflictingCollisions:

    def test_a_collision_with_different_answers_is_reverted(self) -> None:
        before, after = _pair(
            titles=["Prompt injection", "Indirect prompt injection"],
            prose=["Various mitigations exist.", "Various mitigations exist."],
            hubs=["106-447", "205-522"],
        )
        out = _keep_items_resolvable(before, after)

        assert [i.control_text for i in out] == ["Prompt injection",
                                                 "Indirect prompt injection"]
        assert all(i.track == "title" for i in out)

    def test_a_three_way_collision_reverts_all_three(self) -> None:
        before, after = _pair(
            titles=["A", "B", "C"],
            prose=["shared", "shared", "shared"],
            hubs=["1-1", "2-2", "3-3"],
        )
        out = _keep_items_resolvable(before, after)
        assert [i.control_text for i in out] == ["A", "B", "C"]

    def test_only_the_colliding_items_revert(self) -> None:
        """A clean substitution elsewhere must survive the revert."""
        before, after = _pair(
            titles=["A", "B", "C"],
            prose=["shared", "shared", "distinct prose"],
            hubs=["1-1", "2-2", "3-3"],
        )
        out = _keep_items_resolvable(before, after)

        assert [i.control_text for i in out] == ["A", "B", "distinct prose"]
        assert out[2].track == "full-text"


class TestNonConflictingCollisions:

    def test_a_collision_with_the_same_answer_is_kept(self) -> None:
        """Genuinely the same question twice; the grader cannot tell them
        apart either way, so there is nothing to protect against."""
        before, after = _pair(
            titles=["A", "B"],
            prose=["shared prose", "shared prose"],
            hubs=["1-1", "1-1"],
        )
        out = _keep_items_resolvable(before, after)

        assert [i.control_text for i in out] == ["shared prose", "shared prose"]
        assert all(i.track == "full-text" for i in out)

    def test_distinct_prose_is_untouched(self) -> None:
        before, after = _pair(
            titles=["A", "B"],
            prose=["alpha", "beta"],
            hubs=["1-1", "2-2"],
        )
        out = _keep_items_resolvable(before, after)
        assert [i.control_text for i in out] == ["alpha", "beta"]


class TestInvariants:

    def test_item_count_and_order_never_change(self) -> None:
        before, after = _pair(
            titles=["A", "B", "C", "D"],
            prose=["x", "x", "y", "z"],
            hubs=["1-1", "2-2", "3-3", "4-4"],
        )
        out = _keep_items_resolvable(before, after)

        assert len(out) == 4
        assert [i.ground_truth_hub_id for i in out] == ["1-1", "2-2", "3-3", "4-4"]

    def test_ground_truth_is_never_rewritten(self) -> None:
        before, after = _pair(
            titles=["A", "B"], prose=["same", "same"], hubs=["1-1", "2-2"],
        )
        out = _keep_items_resolvable(before, after)
        assert [i.ground_truth_hub_id for i in out] == ["1-1", "2-2"]
