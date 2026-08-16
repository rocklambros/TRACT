"""Tests for the OWASP LLM Top 10 2026 Appendix A mapping extractor.

Appendix A is not control text. It is an expert crosswalk from the ten 2026
risks to nine external frameworks, at element level, with primary and
supporting weights and a written rationale per row. It is extracted to its own
artifact so it can never be mistaken for prose the model trains or evaluates
on.

The fixture is the same synthetic beacon-relay document the parser tests use,
whose appendix maps to two invented taxonomies. Every test builds an extractor
and calls extract() or run().
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import ClassVar

import pytest

from parsers.extract_owasp_llm_top10_2026_mappings import (
    MAPPINGS_FILENAME,
    Owasp2026AppendixExtractor,
    TargetFramework,
)
from parsers.parse_owasp_llm_top10_2026 import SOURCE_FILE

FIXTURE = Path(__file__).parent / "fixtures" / "owasp_llm_top10_2026_sample.md"
FIXTURE_SHA256 = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()

SAMPLE_TARGETS: tuple[TargetFramework, ...] = (
    TargetFramework(
        key="ref",
        heading="Synthetic Reference Taxonomy (REF) -v9.9",
        label="Synthetic Reference Taxonomy (REF)",
        version="v9.9",
        framework_id=None,
        element_pattern=re.compile(r"^(?P<id>REF-\d+)\s+(?P<name>.+)$"),
        element_level="invented",
        matrix_column="REF",
    ),
    TargetFramework(
        key="xyz",
        heading="Invented Control Set (XYZ) -v0.1",
        label="Invented Control Set (XYZ)",
        version="v0.1",
        framework_id=None,
        element_pattern=None,
        element_level="invented",
        matrix_column="XYZ",
    ),
)


class SampleExtractor(Owasp2026AppendixExtractor):
    """The extractor pointed at the fixture's appendix and its own counts.

    Mapping counts are exact and two-sided, so a synthetic appendix declares
    what it contains rather than the real gate being widened to cover both.
    """

    targets: ClassVar[tuple[TargetFramework, ...]] = SAMPLE_TARGETS
    expected_mapping_counts: ClassVar[dict[str, int]] = {"ref": 11, "xyz": 1}
    source_sha256: ClassVar[str] = FIXTURE_SHA256


def _extractor_for(
    tmp_path: Path, text: str | None = None,
) -> SampleExtractor:
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    payload = FIXTURE.read_bytes() if text is None else text.encode("utf-8")
    (raw / SOURCE_FILE).write_bytes(payload)
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()

    class Pinned(SampleExtractor):
        source_sha256: ClassVar[str] = digest

    return Pinned(raw_dir=raw, output_dir=out)


@pytest.fixture
def extractor(tmp_path: Path) -> SampleExtractor:
    return _extractor_for(tmp_path)


class TestRowsBecomeMappings:
    def test_every_row_of_every_declared_section_is_extracted(
        self, extractor: SampleExtractor,
    ) -> None:
        mappings = extractor.extract()["mappings"]

        assert len(mappings) == 12

    def test_a_mapping_carries_risk_target_element_weight_and_rationale(
        self, extractor: SampleExtractor,
    ) -> None:
        first = extractor.extract()["mappings"][0]

        assert first["source_control_id"] == "LLM01:2026"
        assert first["target_framework"] == "ref"
        assert first["target_element_id"] == "REF-11"
        assert first["target_element_name"] == "Unauthenticated Input"
        assert first["weight"] == "primary"
        assert first["rationale"].startswith("The relay accepts a bearing")

    def test_the_supporting_marker_is_distinguished_from_the_primary_one(
        self, extractor: SampleExtractor,
    ) -> None:
        weights = {
            (m["source_control_id"], m["target_element_id"]): m["weight"]
            for m in extractor.extract()["mappings"]
        }

        assert weights[("LLM01:2026", "REF-11")] == "primary"
        assert weights[("LLM01:2026", "REF-22")] == "supporting"

    def test_a_continuation_row_carries_the_risk_of_the_row_above(
        self, extractor: SampleExtractor,
    ) -> None:
        """An empty Risk cell means "same risk", not "no risk"."""
        mappings = extractor.extract()["mappings"]
        by_risk = [m["source_control_id"] for m in mappings
                   if m["target_framework"] == "ref"]

        assert by_risk[1] == "LLM01:2026"

    def test_a_rationale_split_across_a_page_break_is_rejoined(
        self, extractor: SampleExtractor,
    ) -> None:
        """The tables break across pages and the text resumes in a "|||" row.

        Dropped, the mapping ships a rationale that stops mid-sentence.
        """
        mappings = {
            m["target_element_id"]: m for m in extractor.extract()["mappings"]
            if m["source_control_id"] == "LLM04:2026"
        }

        assert mappings["REF-55"]["rationale"].endswith(
            "reaches the whole fleet."
        )

    def test_an_element_with_no_id_convention_keeps_its_name(
        self, extractor: SampleExtractor,
    ) -> None:
        """NIST AI 600-1 names its risk categories and numbers none of them."""
        mappings = [m for m in extractor.extract()["mappings"]
                    if m["target_framework"] == "xyz"]

        assert len(mappings) == 1
        assert mappings[0]["target_element_id"] is None
        assert mappings[0]["target_element_name"] == "XYZ Transmitter Assurance"

    def test_all_ten_risks_are_covered(
        self, extractor: SampleExtractor,
    ) -> None:
        risks = {m["source_control_id"] for m in extractor.extract()["mappings"]}

        assert len(risks) == 10


class TestTheCountsAreGated:
    """A row silently dropped by a layout change is the failure mode here.

    The counts are exact and two-sided for the same reason the ISO repair
    counts are: a one-sided ceiling cannot see an extractor that stops
    reaching rows and ships a shorter crosswalk with every gate green.
    """

    def test_fewer_rows_than_measured_is_a_failure(
        self, tmp_path: Path,
    ) -> None:
        class Expecting(SampleExtractor):
            expected_mapping_counts: ClassVar[dict[str, int]] = {
                "ref": 12, "xyz": 1,
            }

        with pytest.raises(ValueError, match="ref"):
            self._extract_with(Expecting, tmp_path)

    def test_more_rows_than_measured_is_a_failure(
        self, tmp_path: Path,
    ) -> None:
        class Expecting(SampleExtractor):
            expected_mapping_counts: ClassVar[dict[str, int]] = {
                "ref": 10, "xyz": 1,
            }

        with pytest.raises(ValueError, match="ref"):
            self._extract_with(Expecting, tmp_path)

    def test_a_target_with_no_measured_count_is_a_failure(
        self, tmp_path: Path,
    ) -> None:
        class Expecting(SampleExtractor):
            expected_mapping_counts: ClassVar[dict[str, int]] = {"ref": 11}

        with pytest.raises(ValueError, match="xyz"):
            self._extract_with(Expecting, tmp_path)

    @staticmethod
    def _extract_with(
        extractor_class: type[Owasp2026AppendixExtractor], tmp_path: Path,
    ) -> dict[str, object]:
        raw = tmp_path / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        (raw / SOURCE_FILE).write_bytes(FIXTURE.read_bytes())
        out = tmp_path / "out"
        out.mkdir(parents=True, exist_ok=True)
        return extractor_class(raw_dir=raw, output_dir=out).extract()


class TestTheCoverageMatrixCrossCheck:
    """The appendix states its own answer twice, so the two must agree.

    The coverage matrix marks each risk-by-framework cell primary, supporting,
    or absent. The per-framework tables carry the same claim row by row. A
    dropped row or a missed continuation shows up as a disagreement, which is
    the only cheap detector of a partial extraction there is.
    """

    def test_the_matrix_agrees_with_the_detail_tables(
        self, extractor: SampleExtractor,
    ) -> None:
        result = extractor.extract()

        assert result["coverage_matrix_cells_checked"] == 20

    def test_a_matrix_cell_the_tables_contradict_is_a_failure(
        self, tmp_path: Path,
    ) -> None:
        text = FIXTURE.read_text(encoding="utf-8").replace(
            "| LLM03 Excessive Steering Authority | ○ | - |",
            "| LLM03 Excessive Steering Authority | ● | - |",
        )

        with pytest.raises(ValueError, match="LLM03"):
            _extractor_for(tmp_path, text).extract()

    def test_a_dropped_detail_row_is_caught_by_the_matrix(
        self, tmp_path: Path,
    ) -> None:
        """With the count gate re-measured to accept the loss, as it would be.

        The count gate catches a dropped row first, and would here too. This
        is the second line: it fires when the shorter count has already been
        accepted as correct, which is exactly how a real drop gets normalised.
        """
        text = FIXTURE.read_text(encoding="utf-8").replace(
            "| LLM06 Unbounded Relay Consumption | ● REF-66 Unbounded Work | "
            "One query causes work with no declared ceiling. |\n",
            "",
        )
        raw = tmp_path / "raw"
        raw.mkdir(parents=True)
        (raw / SOURCE_FILE).write_bytes(text.encode("utf-8"))
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

        class Accepting(SampleExtractor):
            expected_mapping_counts: ClassVar[dict[str, int]] = {
                "ref": 10, "xyz": 1,
            }
            source_sha256: ClassVar[str] = digest

        with pytest.raises(ValueError, match="LLM06"):
            Accepting(raw_dir=raw, output_dir=tmp_path / "out").extract()


class TestTheArtifact:
    def test_run_writes_the_mapping_artifact_next_to_the_corpus(
        self, extractor: SampleExtractor, tmp_path: Path,
    ) -> None:
        path = extractor.run()

        assert path == tmp_path / "out" / MAPPINGS_FILENAME
        assert path.is_file()

    def test_the_artifact_records_the_bytes_it_was_built_from(
        self, extractor: SampleExtractor, tmp_path: Path,
    ) -> None:
        extractor.run()
        written = json.loads(
            (tmp_path / "out" / MAPPINGS_FILENAME).read_text(encoding="utf-8")
        )

        assert written["source_file"] == SOURCE_FILE
        assert written["source_sha256"] == FIXTURE_SHA256

    def test_the_artifact_names_which_targets_exist_in_our_corpus(
        self, extractor: SampleExtractor, tmp_path: Path,
    ) -> None:
        extractor.run()
        written = json.loads(
            (tmp_path / "out" / MAPPINGS_FILENAME).read_text(encoding="utf-8")
        )
        targets = {t["key"]: t for t in written["target_frameworks"]}

        assert targets["ref"]["framework_id"] is None
        assert targets["ref"]["mapping_count"] == 11
        assert targets["ref"]["distinct_elements"] == 9

    def test_an_appendix_with_no_cwe_section_records_no_chain(
        self, extractor: SampleExtractor,
    ) -> None:
        """The chain is measured, not assumed, and says so when it cannot be."""
        assert extractor.extract()["cwe_chain"] is None

    def test_the_artifact_is_never_written_into_the_framework_corpus_dir(
        self,
    ) -> None:
        """A file under data/processed/frameworks/ is merged as control text.

        These rows are a crosswalk, not controls. Merged, they would enter the
        corpus as ten more anchors carrying another framework's element names.
        """
        assert "frameworks" not in MAPPINGS_FILENAME
        assert MAPPINGS_FILENAME == "owasp_llm_top10_2026_mappings.json"

    def test_re_running_over_the_same_bytes_writes_the_same_bytes(
        self, tmp_path: Path,
    ) -> None:
        first = _extractor_for(tmp_path / "a").run().read_bytes()
        second = _extractor_for(tmp_path / "b").run().read_bytes()

        assert first == second


class TestTheDigestPin:
    def test_a_source_that_does_not_match_the_pin_is_refused(
        self, tmp_path: Path,
    ) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / SOURCE_FILE).write_bytes(FIXTURE.read_bytes() + b"\n# drift\n")
        out = tmp_path / "out"
        out.mkdir()

        with pytest.raises(ValueError, match="sha256"):
            SampleExtractor(raw_dir=raw, output_dir=out).extract()


class TestTheRealRegistry:
    """The nine target frameworks the real appendix carries."""

    def test_nine_targets_are_declared_and_each_has_a_measured_count(
        self,
    ) -> None:
        targets = Owasp2026AppendixExtractor.targets
        counts = Owasp2026AppendixExtractor.expected_mapping_counts

        assert len(targets) == 9
        assert {t.key for t in targets} == set(counts)

    def test_the_targets_we_hold_a_corpus_for_name_their_framework_id(
        self,
    ) -> None:
        by_key = {t.key: t for t in Owasp2026AppendixExtractor.targets}

        assert by_key["cwe"].framework_id == "cwe"
        assert by_key["csa_aicm"].framework_id == "csa_aicm"
        # No corpus for either: neither is an OpenCRE-linked framework here.
        assert by_key["mitre_attack"].framework_id is None
        assert by_key["owasp_aivss"].framework_id is None

    def test_the_cwe_section_declares_the_measured_row_count(self) -> None:
        assert Owasp2026AppendixExtractor.expected_mapping_counts["cwe"] == 48
