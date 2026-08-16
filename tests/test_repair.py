"""Tests for tract.parsers.repair — named, counted text repairs.

Every repair returns how many times it fired so a parser can declare a
ceiling. A count is not a diff, so the ISO parser test also asserts exact
output on hand-checked rows.
"""

from __future__ import annotations

from tract.parsers.repair import (
    RepairResult,
    fix_hyphen_breaks,
    strip_page_furniture,
)


class TestHyphenBreaks:
    def test_rejoins_a_spaced_hyphen_break(self) -> None:
        result = fix_hyphen_breaks("Policies for information secu - rity")
        assert result.text == "Policies for information security"
        assert result.applied == 1

    def test_rejoins_multiple_breaks_and_counts_each(self) -> None:
        result = fix_hyphen_breaks(
            "shall be de - fined and seg - regated by the owner"
        )
        assert result.text == "shall be defined and segregated by the owner"
        assert result.applied == 2

    def test_preserves_a_real_compound_hyphen(self) -> None:
        # "topic-specific" is correct English and must survive. Only the
        # spaced form " - " is PDF damage.
        result = fix_hyphen_breaks("top - ic-specific policies")
        assert result.text == "topic-specific policies"
        assert result.applied == 1

    def test_leaves_an_em_dash_style_aside_alone(self) -> None:
        # A hyphen flanked by spaces AND followed by a capital or a
        # non-letter is punctuation, not a broken word.
        result = fix_hyphen_breaks("the organization - The owner shall act")
        assert result.applied == 0


class TestPageFurniture:
    def test_drops_matching_lines_and_counts_them(self) -> None:
        lines = [
            "| 5.1 | Policies | Control ... |",
            "## ISO/IEC 27001:2022(E)",
            "| 5.2 | Roles | Control ... |",
            "Table A.1 (continued)",
        ]
        kept, dropped = strip_page_furniture(
            lines, (r"^##\s*ISO/IEC", r"^Table A\.1 \(continued\)"),
        )
        assert dropped == 2
        assert kept == [
            "| 5.1 | Policies | Control ... |",
            "| 5.2 | Roles | Control ... |",
        ]
