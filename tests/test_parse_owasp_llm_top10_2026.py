"""Tests for the OWASP Top 10 for LLM Applications 2026 parser.

The fixture is a synthetic ten-entry document for a fictional beacon relay,
not the real OWASP source. It reproduces the heading skeleton the parser keys
on: `## LLM0N:2026 <Title>` entries, the four standard subsections, the extra
subsections some entries carry, and an Appendix A that must terminate the last
entry. The real document is CC BY-SA 4.0 and this repository is CC0, so no
verbatim run of it is tracked here.

Every test instantiates the parser and calls parse() or run(). A test that
asserted a property of the fixture rather than of the parser would pass with
the parser deleted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import ClassVar

import pytest

from parsers.parse_owasp_llm_top10_2026 import (
    ENTRY_IDS,
    SOURCE_FILE,
    OwaspLlmTop102026Parser,
)

FIXTURE = Path(__file__).parent / "fixtures" / "owasp_llm_top10_2026_sample.md"
FIXTURE_SHA256 = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()


class SampleParser(OwaspLlmTop102026Parser):
    """The parser pinned to the fixture's digest rather than the source's.

    The real parser pins `version` to the source sha256 because the document's
    revision history still reads "[2026 release date]" and a date string would
    assert a release that has not happened. A synthetic fixture is different
    bytes, so it declares its own pin instead of the real gate being widened
    to accept two.
    """

    source_sha256: ClassVar[str] = FIXTURE_SHA256
    version: ClassVar[str] = f"sha256:{FIXTURE_SHA256}"


def _stage(tmp_path: Path, text: str | None = None) -> Path:
    """Write the fixture (or a mutated copy) into a raw dir and return it."""
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    payload = FIXTURE.read_bytes() if text is None else text.encode("utf-8")
    (raw / SOURCE_FILE).write_bytes(payload)
    return raw


def _parser_for(tmp_path: Path, text: str | None = None) -> SampleParser:
    """A parser over staged bytes, with its digest pin matching them."""
    raw = _stage(tmp_path, text)
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256((raw / SOURCE_FILE).read_bytes()).hexdigest()

    class Pinned(SampleParser):
        source_sha256: ClassVar[str] = digest
        version: ClassVar[str] = f"sha256:{digest}"

    return Pinned(raw_dir=raw, output_dir=out, audit_dir=tmp_path / "audit")


@pytest.fixture
def parser(tmp_path: Path) -> SampleParser:
    return _parser_for(tmp_path)


class TestTheTenEntries:
    def test_parses_exactly_ten_entries_in_source_order(
        self, parser: SampleParser,
    ) -> None:
        controls = parser.parse()

        assert [c.control_id for c in controls] == list(ENTRY_IDS)

    def test_the_ids_carry_the_2026_edition_tag(
        self, parser: SampleParser,
    ) -> None:
        """The 2025 ids carry all 13 OpenCRE links and must never be reused.

        A 2026 control emitted as "LLM01:2025" would collide with the 2025
        edition's anchor in every downstream join.
        """
        controls = parser.parse()

        assert all(c.control_id.endswith(":2026") for c in controls)
        assert not any(":2025" in c.control_id for c in controls)

    def test_titles_come_from_the_entry_heading(
        self, parser: SampleParser,
    ) -> None:
        controls = {c.control_id: c for c in parser.parse()}

        assert controls["LLM01:2026"].title == "Beacon Spoofing"
        assert controls["LLM09:2026"].title == "Cache Geometry Weaknesses"

    def test_a_missing_entry_is_a_failure_not_a_short_list(
        self, tmp_path: Path,
    ) -> None:
        text = FIXTURE.read_text(encoding="utf-8").replace(
            "## LLM07:2026 Bearing Misinformation",
            "## Bearing Misinformation",
        )

        with pytest.raises(ValueError, match="LLM07:2026"):
            _parser_for(tmp_path, text).parse()


class TestTheAppendixBoundary:
    """Without it LLM10 runs to EOF and swallows the whole back matter.

    Measured on the real source: 937 lines and 132 KB of appendix, references,
    and acknowledgements land inside the last entry, because the appendix sits
    below it rather than above.
    """

    def test_the_last_entry_stops_at_appendix_a(
        self, parser: SampleParser,
    ) -> None:
        last = {c.control_id: c for c in parser.parse()}["LLM10:2026"]
        assert last.full_text is not None

        assert "Appendix A" not in last.full_text
        assert "Acknowledgements" not in last.full_text
        assert "Framework Sources" not in last.full_text

    def test_a_source_without_the_boundary_is_refused(
        self, tmp_path: Path,
    ) -> None:
        """Fail closed. Emitting the back matter as control text is worse."""
        text = FIXTURE.read_text(encoding="utf-8").replace(
            "## Appendix A: Related Framework Mappings",
            "## Related Framework Mappings",
        )

        with pytest.raises(ValueError, match="Appendix A"):
            _parser_for(tmp_path, text).parse()

    def test_a_reference_heading_below_the_appendix_is_not_an_entry(
        self, parser: SampleParser,
    ) -> None:
        """The reference list repeats "LLM01: ..." without the edition tag."""
        controls = parser.parse()

        assert len(controls) == 10
        assert all("Invented reference list" not in (c.full_text or "")
                   for c in controls)


class TestDescriptionIsDefinitionalNotRemediation:
    """description stops at the first remediation heading, full_text does not.

    The cut uses tract.config.REMEDIATION_HEADINGS, the same list
    tract.text_selection.strip_remediation applies downstream, so the parser
    and the anchor selector agree on where remediation starts instead of each
    guessing separately.
    """

    def test_description_carries_the_definitional_prose(
        self, parser: SampleParser,
    ) -> None:
        controls = {c.control_id: c for c in parser.parse()}

        assert "beacon spoofing weakness occurs" in \
            controls["LLM01:2026"].description

    def test_description_carries_the_common_examples_of_risk(
        self, parser: SampleParser,
    ) -> None:
        """They say what the risk is, so they sit on the definitional side."""
        controls = {c.control_id: c for c in parser.parse()}

        assert "Common Examples of Risk" in controls["LLM03:2026"].description
        assert "Excessive function" in controls["LLM03:2026"].description

    def test_description_carries_extra_subsections_before_the_cut(
        self, parser: SampleParser,
    ) -> None:
        """LLM01 carries "Types of ..." between Description and the examples."""
        controls = {c.control_id: c for c in parser.parse()}

        assert "Relayed Spoofing" in controls["LLM01:2026"].description

    def test_description_excludes_prevention_and_scenarios(
        self, parser: SampleParser,
    ) -> None:
        for control in parser.parse():
            assert "Prevention and Mitigation Strategies" \
                not in control.description
            assert "Example Attack Scenarios" not in control.description

    def test_full_text_keeps_the_prevention_and_the_scenarios(
        self, parser: SampleParser,
    ) -> None:
        """Nothing is lost. The cut decides the anchor, not what is kept."""
        controls = {c.control_id: c for c in parser.parse()}
        full = controls["LLM01:2026"].full_text
        assert full is not None

        assert "Prevention and Mitigation Strategies" in full
        assert "Scenario #2: Almanac Poisoning" in full

    def test_an_entry_with_no_remediation_heading_is_refused(
        self, tmp_path: Path,
    ) -> None:
        """Silently, that entry's description would become the whole entry."""
        text = FIXTURE.read_text(encoding="utf-8").replace(
            "## Prevention and Mitigation Strategies\n\nValidate every entry "
            "at ingest, and record the provenance of each one.\n\n"
            "## Example Attack Scenarios\n\n## Scenario #1\n",
            "## Handling Guidance\n\nValidate every entry at ingest.\n\n"
            "## Illustrations\n\n## Case One\n",
        )

        with pytest.raises(ValueError, match="LLM05:2026"):
            _parser_for(tmp_path, text).parse()


class TestDescriptionBudget:
    """A description over the 2000-char cap would evict the parser's full_text.

    BaseParser._sanitize_control replaces a parser-supplied full_text with the
    overflow of an over-long description, so an uncut description would leave
    full_text holding the definitional block instead of the whole entry.
    """

    def test_a_long_definitional_block_is_cut_and_the_entry_is_kept(
        self, tmp_path: Path,
    ) -> None:
        from tract.config import DESCRIPTION_MAX_LENGTH

        filler = ("The relay records a bearing observation and the almanac "
                  "entry that supports it. ") * 40
        text = FIXTURE.read_text(encoding="utf-8").replace(
            "A beacon spoofing weakness occurs",
            filler + "A beacon spoofing weakness occurs",
        )
        controls = {c.control_id: c for c in _parser_for(tmp_path, text).run().controls}
        first = controls["LLM01:2026"]

        assert len(first.description) <= DESCRIPTION_MAX_LENGTH
        assert first.full_text is not None
        # The entry, not the overflow of the description.
        assert "Scenario #2: Almanac Poisoning" in first.full_text

    def test_a_cut_description_does_not_end_mid_word(
        self, tmp_path: Path,
    ) -> None:
        filler = ("The relay records a bearing observation and the almanac "
                  "entry that supports it. ") * 40
        text = FIXTURE.read_text(encoding="utf-8").replace(
            "A beacon spoofing weakness occurs",
            filler + "A beacon spoofing weakness occurs",
        )
        controls = {c.control_id: c for c in _parser_for(tmp_path, text).parse()}

        assert not controls["LLM01:2026"].description.endswith("-")
        assert controls["LLM01:2026"].description.split()[-1] in filler


class TestTheDigestPin:
    """version pins to the source sha256 because the document is pre-release.

    Its revision history still reads "[2026 release date]", so a date string
    in `version` would assert a release that has not happened.
    """

    def test_the_version_field_carries_the_digest(
        self, parser: SampleParser,
    ) -> None:
        output = parser.run()

        assert output.version == f"sha256:{FIXTURE_SHA256}"
        assert output.version.startswith("sha256:")

    def test_a_source_that_does_not_match_the_pin_is_refused(
        self, tmp_path: Path,
    ) -> None:
        raw = _stage(
            tmp_path,
            FIXTURE.read_text(encoding="utf-8") + "\n## Drifted heading\n",
        )
        out = tmp_path / "out"
        out.mkdir()

        with pytest.raises(ValueError, match="sha256"):
            SampleParser(
                raw_dir=raw, output_dir=out, audit_dir=tmp_path / "audit",
            ).parse()

    @staticmethod
    def test_the_pinned_digest_is_the_one_recorded_in_the_ledger() -> None:
        """The module constant must match the owner-supplied staging hash."""
        from parsers.parse_owasp_llm_top10_2026 import SOURCE_SHA256

        assert SOURCE_SHA256 == (
            "3d3c9f21809c5f882a668b87424ac6b2e2a270caab4b29aa5265df3475433a96"
        )


class TestRunWritesAValidArtifact:
    def test_run_writes_the_2026_file_and_never_the_2025_one(
        self, parser: SampleParser, tmp_path: Path,
    ) -> None:
        parser.run()

        assert (tmp_path / "out" / "owasp_llm_top10_2026.json").is_file()
        assert not (tmp_path / "out" / "owasp_llm_top10.json").exists()

    def test_the_artifact_records_the_source_it_read(
        self, parser: SampleParser, tmp_path: Path,
    ) -> None:
        parser.run()
        written = json.loads(
            (tmp_path / "out" / "owasp_llm_top10_2026.json").read_text(
                encoding="utf-8")
        )

        assert [f["path"] for f in written["source_files"]] == [SOURCE_FILE]
        assert written["source_files"][0]["sha256"] == FIXTURE_SHA256

    def test_every_entry_clears_the_declared_prose_floor(
        self, parser: SampleParser,
    ) -> None:
        from tract.parsers.base import BaseParser

        controls = parser.run().controls

        assert BaseParser.honest_prose_fraction(controls) == 1.0
        assert OwaspLlmTop102026Parser.min_prose_fraction == 1.0

    def test_re_running_over_the_same_bytes_writes_the_same_bytes(
        self, tmp_path: Path,
    ) -> None:
        """No clock is read, so a re-parse diff shows real changes only."""
        first = _parser_for(tmp_path / "a").run()
        second = _parser_for(tmp_path / "b").run()

        assert first.model_dump(mode="json") == second.model_dump(mode="json")
