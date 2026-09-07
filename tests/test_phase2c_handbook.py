"""The handbook must describe the packet that exists, and it is tracked so it can.

The previous annotator document, `claudedocs/curation-package.md`, is gitignored
-- which is why a contamination-path sweep fixed thirteen tracked files and left
the one an annotator actually reads still pointing at a dead path, and why three
fabricated claims survived in it. This one is tracked, in CI, and its
load-bearing claims are checked against the tooling rather than against a copy
of the tooling's documentation.

It also describes a DIFFERENT round from the curation handbook: traditional
controls onto 78 AI hubs, not AI controls across 522. Sending the wrong one with
the right packet produces a sheet that cannot import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from tract.config import PROJECT_ROOT

HANDBOOK: Final[Path] = PROJECT_ROOT / "docs" / "phase2c-annotator-handbook.md"


@pytest.fixture(scope="module")
def raw() -> str:
    """The file as written, for structural checks like section slicing."""
    return HANDBOOK.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def text(raw: str) -> str:
    """Whitespace-collapsed, so a phrase assertion is wrap-insensitive.

    Markdown is hard-wrapped at ~78 columns, so a multi-word phrase spans a
    newline and a naive substring check fails on the formatting rather than on
    the content. That is a brittle test, not a broken document.
    """
    return " ".join(raw.split())


class TestItIsTracked:
    def test_the_handbook_is_not_gitignored(self) -> None:
        """The whole reason it lives in docs/ rather than claudedocs/."""
        import subprocess

        result = subprocess.run(
            ["git", "check-ignore", str(HANDBOOK)],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        )
        assert result.returncode != 0, (
            "The handbook is gitignored. That is how the previous one escaped "
            "a thirteen-file correction sweep and kept three fabricated claims."
        )


class TestItDescribesTheRealPacket:
    """Checked against the tooling, not against prose about the tooling."""

    def test_it_names_every_file_the_packet_emits(self, text: str) -> None:
        from scripts.build_bridge_packet import (
            ANNOTATION_SHEET_NAME,
            CONTROL_SHEET_NAME,
            HUB_SHEET_NAME,
        )

        for name in (HUB_SHEET_NAME, CONTROL_SHEET_NAME, ANNOTATION_SHEET_NAME):
            assert name in text, f"the handbook never mentions {name}"

    def test_it_names_the_answer_columns_the_importer_requires(
        self, text: str
    ) -> None:
        from scripts.build_bridge_packet import ANSWER_FIELDS

        for field in ANSWER_FIELDS:
            assert f"`{field}`" in text, (
                f"the annotator is never told to fill {field}"
            )

    def test_it_states_the_confidence_scale_the_importer_enforces(
        self, text: str
    ) -> None:
        from scripts.import_bridge_links import CONFIDENCE_MAX, CONFIDENCE_MIN

        assert (CONFIDENCE_MIN, CONFIDENCE_MAX) == (1, 3)
        assert "`1`, `2` or `3`" in text

    def test_it_teaches_the_none_sentinel_the_importer_accepts(
        self, text: str
    ) -> None:
        from scripts.import_bridge_links import NO_HUB_SENTINEL

        assert NO_HUB_SENTINEL in text
        assert "real, correct and expected answer" in text

    def test_it_uses_the_right_generator(self, text: str) -> None:
        """build_curation_packet is the OTHER round, over 522 hubs."""
        assert "scripts.build_bridge_packet" in text
        assert "scripts.build_curation_packet" not in text

    def test_it_names_the_gate_tool_and_not_the_raw_arithmetic(
        self, text: str
    ) -> None:
        assert "gate1_report" in text
        assert "orphan_rate --bridge" in text, (
            "the coordinator must be warned off the raw tool explicitly; it "
            "passes a sheet that violates three quality conditions"
        )

    def test_it_states_the_hub_count_the_packet_ships(self, text: str) -> None:
        from scripts.build_bridge_packet import ai_hub_ids

        assert f"{len(ai_hub_ids())} AI hubs" in text or f"{len(ai_hub_ids())}" in text


class TestTheVolunteerTermsAreStated:
    """V3/V4: the previous handbook asked a volunteer to invoice and told them
    nothing about what happens to their work."""

    def test_it_is_a_volunteer_round_with_no_pay_language(self, text: str) -> None:
        assert "volunteer round" in text
        for term in ("invoice", "/hr", "per hour", "$2,500"):
            assert term not in text, f"pay language remains: {term!r}"

    def test_it_states_what_happens_to_their_work(self, text: str) -> None:
        for promise in ("credit", "withdraw", "OpenCRE"):
            assert promise.lower() in text.lower()

    def test_it_records_that_their_identifier_is_attached(self, text: str) -> None:
        assert "annotator_id" in text


class TestTheBlindingInstructionIsPresent:
    """B1: the previous handbook forbade opencre.org and never named this repo,
    which is public and tracks the answer key."""

    def test_it_names_the_repository(self, text: str) -> None:
        assert "github.com/rocklambros/TRACT" in text

    def test_it_still_names_opencre(self, text: str) -> None:
        assert "opencre.org" in text

    def test_it_warns_against_the_llm_written_hub_reference(self, text: str) -> None:
        assert "results/ceiling_study/hub_reference.md" in text
        assert (PROJECT_ROOT / "results" / "ceiling_study" / "hub_reference.md").is_file()


class TestItCarriesNoUnmeasuredTargets:
    """V1/§1.6: the previous handbook managed people against numbers nobody
    measured -- an agreement band, a rank correlation, a throughput rate."""

    def test_no_throughput_target(self, text: str) -> None:
        assert "minutes per control" not in text
        assert "keyword matching" not in text

    def test_it_says_plainly_that_no_rate_was_measured(self, text: str) -> None:
        assert "No per-item rate has ever been measured" in text

    def test_no_agreement_expectation(self, text: str) -> None:
        for figure in ("0.71", "0.73", "0.35–0.65", "0.35-0.65", "κ"):
            assert figure not in text, f"unmeasured agreement figure: {figure!r}"

    def test_it_does_not_pre_answer_a_stratum(self, text: str) -> None:
        """V5: the previous handbook named bias/fairness as always NONE, with
        the confidence and rationale pre-written. That is the experimenter
        supplying the label."""
        lowered = text.lower()
        assert "bias and fairness" not in lowered
        assert "fairness testing" not in lowered


class TestItDoesNotDiscloseTheTargets:
    """Pre-registration §5: a population generating data is not told the quota."""

    @pytest.mark.parametrize("target", ["55/78", "≥ 40 distinct", "≤ 6 AI hubs", "15%"])
    def test_no_numeric_gate_target_reaches_part_two(
        self, raw: str, target: str
    ) -> None:
        part_two = " ".join(raw[raw.index("# Part 2"):].split())
        assert target not in part_two, (
            f"Part 2 is what the annotator receives and it discloses {target!r}."
        )
