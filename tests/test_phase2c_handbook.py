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

import re
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
    which is public and tracks the answer key.

    These match the INSTRUCTION, not the bare domain. Two reasons. A domain
    mentioned anywhere in the file would satisfy `"opencre.org" in text` --
    including in a sentence recommending it -- so the substring check was the
    weaker assertion. And CodeQL flags a bare domain substring test as
    py/incomplete-url-substring-sanitization, which is a false positive here
    (nothing is sanitising a URL) but is worth not writing in the first place.
    """

    # The instruction and the source must appear together, in that order.
    _FORBIDS = "do not read.{0,600}?%s"

    @pytest.mark.parametrize(
        "source",
        [r"github\.com/rocklambros/TRACT", r"opencre\.org"],
        ids=["repository", "opencre"],
    )
    def test_the_do_not_read_instruction_names_the_source(
        self, text: str, source: str
    ) -> None:
        pattern = re.compile(self._FORBIDS % source, re.IGNORECASE | re.DOTALL)
        assert pattern.search(text), (
            f"the handbook does not tell the annotator not to read {source}. "
            "Naming a source somewhere in the file is not an instruction."
        )

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


class TestTheStatedCountsMatchTheRealPacket:
    """The handbook now quotes concrete numbers, so they must be derived.

    A figure typed into a document drifts the moment the corpus changes, and
    this project has a long record of exactly that -- a test count that was
    wrong three times, a campaign verdict that read "in progress" for five days
    after it failed. These tests build the real packet and compare.
    """

    @pytest.fixture(scope="class")
    def packet(self, tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
        from scripts.build_bridge_packet import build_bridge_packet

        out = tmp_path_factory.mktemp("handbook_packet")
        build_bridge_packet(out, framework_id="nist_800_53")
        return out

    def test_the_hub_count_is_the_packet_s(self, text: str, packet) -> None:  # type: ignore[no-untyped-def]
        import csv

        with (packet / "ai_hubs.csv").open(encoding="utf-8") as handle:
            hubs = list(csv.DictReader(handle))
        assert f"**{len(hubs)}** AI hubs" in text or f"All {len(hubs)} hubs" in text

    def test_the_control_count_is_the_packet_s(self, text: str, packet) -> None:  # type: ignore[no-untyped-def]
        import csv

        with (packet / "annotate.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert f"**{len(rows)}** controls" in text
        assert f"{len(rows)} rows" in text

    def test_the_branch_table_matches(self, text: str, packet) -> None:  # type: ignore[no-untyped-def]
        """The four branch counts are quoted in Part 1 and Part 2."""
        import csv
        from collections import Counter

        with (packet / "ai_hubs.csv").open(encoding="utf-8") as handle:
            counts = Counter(row["branch"] for row in csv.DictReader(handle))
        for branch, n in counts.items():
            assert f"| {n} | {branch} |" in text, (
                f"the handbook's branch table omits or misstates "
                f"{branch!r} ({n} hubs)"
            )

    def test_it_claims_no_truncation_and_that_is_true(
        self, text: str, packet
    ) -> None:  # type: ignore[no-untyped-def]
        """The handbook tells the coordinator to report truncated controls."""
        import csv

        assert "none is truncated" in text.lower()
        with (packet / "annotate.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert not [r for r in rows if len(r["control_text"]) == 2000]

    def test_the_worked_example_control_exists(self, text: str, packet) -> None:  # type: ignore[no-untyped-def]
        """AC-3 is quoted verbatim; it must still be in the packet."""
        import csv

        assert "AC-3 Access Enforcement" in text
        with (packet / "annotate.csv").open(encoding="utf-8") as handle:
            titles = {r["control_id"]: r["control_title"] for r in csv.DictReader(handle)}
        assert titles.get("AC-3") == "AC-3 Access Enforcement"
