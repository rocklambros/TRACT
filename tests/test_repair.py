"""Tests for tract.parsers.repair — named, counted text repairs.

Every repair returns how many times it fired so a parser can declare a
ceiling. A count is not a diff, so the ISO parser test also asserts exact
output on hand-checked rows.
"""

from __future__ import annotations

from tract.parsers.repair import (
    RepairResult,
    build_vocabulary,
    fix_hyphen_breaks,
    split_run_together,
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


class TestRunTogether:
    VOCAB = build_vocabulary([
        "Rules for the acceptable use and procedures for handling information "
        "and other associated assets shall be identified documented and "
        "implemented by the organization",
    ])

    def test_splits_a_known_run_together_token(self) -> None:
        result = split_run_together(
            "Control Rulesfortheacceptableuse of assets", self.VOCAB,
            min_token_length=12,
        )
        assert result.text == "Control Rules for the acceptable use of assets"
        assert result.applied == 1

    def test_leaves_the_token_alone_when_it_cannot_be_fully_segmented(self) -> None:
        # "zzzqqq" is not in the vocabulary, so no complete segmentation
        # exists and the repair must fail closed rather than guess.
        result = split_run_together(
            "Control Rulesforzzzqqqtheacceptableuse here", self.VOCAB,
            min_token_length=12,
        )
        assert "Rulesforzzzqqqtheacceptableuse" in result.text
        assert result.applied == 0

    def test_ignores_ordinary_long_words(self) -> None:
        # "responsibilities" is 16 characters and a real word. The threshold
        # exists so the splitter never looks at it.
        result = split_run_together(
            "roles and responsibilities shall be defined", self.VOCAB,
            min_token_length=20,
        )
        assert result.applied == 0

    def test_vocabulary_is_lowercased_and_length_filtered(self) -> None:
        # min_length is inclusive. The default of 3 must keep "the", "for"
        # and "use", because the run-together tokens this splitter exists for
        # are built out of exactly those words.
        vocab = build_vocabulary(["The Owner shall act"], min_length=3)
        assert "owner" in vocab
        assert "the" in vocab
        # "act" is 3 and kept; a 4-char floor would drop it.
        assert "act" in build_vocabulary(["The Owner shall act"], min_length=3)
        assert "act" not in build_vocabulary(["The Owner shall act"], min_length=4)


class TestCellBleed:
    def test_moves_a_spilled_fragment_back_to_its_own_row(self) -> None:
        from tract.parsers.repair import repair_cell_bleed

        rows = [
            ("5.6", "Contact with special interest groups",
             "Control The organization shall maintain contact with special "
             "interest groups or other specialist security forums and professional"),
            ("5.7", "Threat intelligence",
             "associations. Control Information relating to threats shall be "
             "collected and analysed."),
        ]
        repaired, applied = repair_cell_bleed(rows)

        assert applied == 1
        assert repaired[0][2].endswith("and professional associations.")
        assert repaired[1][2] == (
            "Control Information relating to threats shall be collected "
            "and analysed."
        )

    def test_leaves_well_formed_rows_untouched(self) -> None:
        from tract.parsers.repair import repair_cell_bleed

        rows = [
            ("5.1", "Policies", "Control Policies shall be defined."),
            ("5.2", "Roles", "Control Roles shall be allocated."),
        ]
        repaired, applied = repair_cell_bleed(rows)
        assert applied == 0
        assert repaired == rows

    def test_a_leading_fragment_with_no_previous_row_is_left_alone(self) -> None:
        from tract.parsers.repair import repair_cell_bleed

        rows = [("5.1", "Policies", "orphan fragment. Control Policies apply.")]
        repaired, applied = repair_cell_bleed(rows)
        assert applied == 0
        assert repaired == rows
