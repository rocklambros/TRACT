"""SP 800-63B numbers its headings, and the section number is the join key.

Three things this file is built to catch, in the order they would hurt.

The revision. OpenCRE's 79 curated links carry revision 3B section numbers.
Revision 4B renumbered the document and matches 0 of the 25 distinct ids, so a
fetch of the wrong revision produces a corpus whose join is zero while every
count still looks healthy. TestStructureGate exercises the refusal, and
TestRequiredSectionIds derives the gate's own id set from the tracked link file
so a transcription error in the parser cannot make the gate vacuous.

The anchor. BaseParser._sanitize_control rewrites full_text behind a parser's
back whenever description exceeds DESCRIPTION_MAX_LENGTH. Twenty-two of the
118 section bodies are over that limit. TestDescriptionCap asserts on the
shipped artifact, not on the cap function, because the base class is what
would undo the cap.

The fixture. Every HTML string here is invented. Nothing in this file is
transcribed from the source document, so a test cannot pass by accident of
quoting the same words the parser reads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pytest

from parsers.parse_nist_800_63 import (
    REQUIRED_SECTION_IDS,
    UNANSWERABLE_SECTION_ID,
    Nist80063Parser,
)
from tract.config import DESCRIPTION_MAX_LENGTH, TRAINING_DIR

CURATED_LINKS_PATH: Final[Path] = TRAINING_DIR / "hub_links_curated.jsonl"

# Invented, not transcribed. The shape is the source's: an unnumbered front
# heading, a numbered chapter with no body of its own, numbered subsections
# with bodies, a lettered appendix section, and an unnumbered appendix heading.
HTML: Final[str] = """<html>
<head><title>NIST Special Publication 800-63B</title></head>
<body>
<h2 id="front">Table of Contents</h2>
<p>Front matter that belongs to no numbered section.</p>
<h2 id="sec5">5 Chapter About Credential Handling</h2>
<p>An opening paragraph that belongs to the chapter itself.</p>
<h3 id="sub51">5.1 Subsection With No Body Of Its Own</h3>
<h4 id="sub511">5.1.1 Subsection Carrying A Statement</h4>
<p>The first invented paragraph, long enough to count as a statement rather
than a restatement of the heading above it.</p>
<h5 id="sub5112">5.1.1.2 Deeper Subsection Carrying Another Statement</h5>
<p>A second invented paragraph, deliberately distinct from the first so a body
that ran past its heading would be visible in an assertion.</p>
<h2 id="appa">Appendix A Unnumbered Appendix Heading</h2>
<p>Appendix front matter under a heading with no dotted number.</p>
<h3 id="appa3">A.3 Lettered Appendix Section</h3>
<p>An invented appendix paragraph, present so the lettered id form is covered
by the same assertions as the dotted one.</p>
</body></html>
"""

# Under the parser's real floor of 100, so a fixture-backed structural test has
# to declare its own rather than widening the one that ships.
FIXTURE_SECTIONS: Final[int] = 5


def _curated_section_ids() -> set[str]:
    """Every distinct section_id OpenCRE's curated links carry for 800-63."""
    ids: set[str] = set()
    with CURATED_LINKS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("framework_id") == "nist_800_63":
                ids.add(str(row["section_id"]))
    return ids


def _document(sections: list[tuple[str, str, str]]) -> str:
    """An HTML document with one numbered h3 per (id, title, body) triple."""
    parts = ["<html><head><title>Invented</title></head><body>"]
    for number, title, body in sections:
        parts.append(f"<h3>{number} {title}</h3>")
        parts.append(f"<p>{body}</p>")
    parts.append("</body></html>")
    return "\n".join(parts)


def _full_document(long_body_chars: int = 0) -> str:
    """A 118-section document holding every required id, for end-to-end runs.

    Deterministic and invented. This exists so parse(), both structural gates,
    the description cap, the prose floor, the count check and the write all run
    in CI, rather than only on a laptop that happens to hold data/raw.
    """
    sections: list[tuple[str, str, str]] = []
    for position, number in enumerate(sorted(REQUIRED_SECTION_IDS)):
        body = (
            f"Invented statement number {position} for a required section, "
            f"written long enough to clear the honest-prose floor without "
            f"restating its own heading."
        )
        if long_body_chars and position == 0:
            body = body + " " + "word " * (long_body_chars // 5)
        sections.append((number, f"Required Section {position}", body))
    for filler in range(118 - len(REQUIRED_SECTION_IDS)):
        sections.append((
            f"20.{filler}",
            f"Filler Section {filler}",
            f"Invented filler statement number {filler}, written long enough "
            f"to clear the honest-prose floor on its own.",
        ))
    return _document(sections)


class TestSectionsFromHtml:
    def test_the_section_number_is_the_control_id(self) -> None:
        controls = Nist80063Parser.sections_from_html(HTML)
        assert [c.control_id for c in controls] == [
            "5", "5.1", "5.1.1", "5.1.1.2", "A.3",
        ]

    def test_title_is_the_heading_text_without_the_number(self) -> None:
        controls = Nist80063Parser.sections_from_html(HTML)
        assert controls[3].title == "Deeper Subsection Carrying Another Statement"

    def test_body_stops_at_the_next_heading(self) -> None:
        controls = Nist80063Parser.sections_from_html(HTML)
        assert controls[2].description.startswith("The first invented paragraph")
        assert "second invented paragraph" not in controls[2].description

    def test_an_unnumbered_appendix_heading_is_not_a_section(self) -> None:
        titles = {c.title for c in Nist80063Parser.sections_from_html(HTML)}
        assert "Unnumbered Appendix Heading" not in titles
        assert "Table of Contents" not in titles

    def test_a_section_with_no_body_restates_its_own_title(self) -> None:
        """ProseIndex excludes these, which is the intended outcome.

        Storing the title is not the same as answering a link with it: a
        description equal to the title fails the prose rule, so the control is
        counted under dropped_by_prose_rule and never becomes an anchor.
        """
        empty = next(
            c for c in Nist80063Parser.sections_from_html(HTML)
            if c.control_id == "5.1"
        )
        assert empty.description == empty.title
        assert empty.full_text is None

    def test_parent_id_is_set_only_when_the_parent_is_a_numbered_section(
        self,
    ) -> None:
        by_id = {
            c.control_id: c for c in Nist80063Parser.sections_from_html(HTML)
        }
        assert by_id["5.1.1"].parent_id == "5.1"
        assert by_id["5"].parent_id is None
        # "A" is the appendix's unnumbered heading, so it is no control's id.
        assert by_id["A.3"].parent_id is None

    def test_two_headings_claiming_one_number_raise(self) -> None:
        duplicated = _document([
            ("9.1", "First Claim", "An invented statement under the first."),
            ("9.1", "Second Claim", "An invented statement under the second."),
        ])
        with pytest.raises(ValueError, match=r"'9\.1' appears on more than one"):
            Nist80063Parser.sections_from_html(duplicated)

    def test_a_document_with_no_numbering_yields_nothing(self) -> None:
        """The shape revision 4B presents: headings without dotted numbers."""
        slugged = (
            "<html><body><h2>Authenticator Requirements</h2>"
            "<p>An invented paragraph under a heading with no number.</p>"
            "</body></html>"
        )
        assert Nist80063Parser.sections_from_html(slugged) == []


class TestRequiredSectionIds:
    """The gate's id set is derived from the link file, not trusted."""

    def test_the_required_set_is_the_curated_set_minus_the_one_gap(self) -> None:
        curated = _curated_section_ids()
        assert set(REQUIRED_SECTION_IDS) | {UNANSWERABLE_SECTION_ID} == curated

    def test_the_declared_gap_is_a_link_the_file_actually_carries(self) -> None:
        """Fails if the fragment is repaired upstream and this entry goes stale."""
        assert UNANSWERABLE_SECTION_ID in _curated_section_ids()

    def test_the_required_set_holds_no_duplicates(self) -> None:
        assert len(set(REQUIRED_SECTION_IDS)) == len(REQUIRED_SECTION_IDS)

    def test_the_gap_is_not_also_required(self) -> None:
        """Requiring it would make the structural gate unpassable.

        The set-equality test above cannot see this on its own: the union it
        compares is identical whether or not the fragment is also in the
        required tuple, so without this line the failure only shows up on a
        checkout that holds data/raw.
        """
        assert UNANSWERABLE_SECTION_ID not in REQUIRED_SECTION_IDS


class TestStructureGate:
    def test_a_document_missing_a_required_section_is_refused(
        self, tmp_path: Path,
    ) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "sp800_63b.html").write_text(HTML, encoding="utf-8")
        parser = Nist80063Parser(
            raw_dir=raw, output_dir=tmp_path / "out", audit_dir=tmp_path / "aud",
        )
        parser.min_sections = FIXTURE_SECTIONS
        parser.required_section_ids = ("5.1.1.2", "9.9.9")
        with pytest.raises(ValueError, match=r"9\.9\.9"):
            parser.parse()

    def test_a_thin_document_is_refused(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw2"
        raw.mkdir()
        (raw / "sp800_63b.html").write_text(HTML, encoding="utf-8")
        parser = Nist80063Parser(
            raw_dir=raw, output_dir=tmp_path / "out2", audit_dir=tmp_path / "aud2",
        )
        parser.required_section_ids = ()
        with pytest.raises(ValueError, match="numbered section"):
            parser.parse()

    def test_a_document_meeting_both_halves_is_accepted(
        self, tmp_path: Path,
    ) -> None:
        """The gate has to be able to pass, or its failures prove nothing."""
        raw = tmp_path / "raw3"
        raw.mkdir()
        (raw / "sp800_63b.html").write_text(HTML, encoding="utf-8")
        parser = Nist80063Parser(
            raw_dir=raw, output_dir=tmp_path / "out3", audit_dir=tmp_path / "aud3",
        )
        parser.min_sections = FIXTURE_SECTIONS
        parser.required_section_ids = ("5.1.1.2", "A.3")
        assert len(parser.parse()) == FIXTURE_SECTIONS

    def test_the_shipped_floor_is_the_one_a_default_parser_uses(self) -> None:
        """A default instance must not inherit a test's widened gate."""
        parser = Nist80063Parser()
        assert parser.min_sections == 100
        assert parser.required_section_ids == REQUIRED_SECTION_IDS


class TestDescriptionCap:
    """Ruling R14. The parser owns its anchor rather than a margin."""

    def test_a_statement_inside_the_budget_is_not_cut(self) -> None:
        text = "A short invented statement."
        assert Nist80063Parser._cap_description(text) == (text, None)

    def test_an_oversized_statement_is_cut_under_the_limit(self) -> None:
        text = "word " * (DESCRIPTION_MAX_LENGTH // 2)
        capped, full = Nist80063Parser._cap_description(text)
        assert full == text
        assert len(capped) < DESCRIPTION_MAX_LENGTH
        assert not capped.endswith(" ")

    def test_a_statement_with_no_space_still_lands_under_the_limit(self) -> None:
        """The word-boundary search finds nothing, so the hard cut must hold."""
        text = "x" * (DESCRIPTION_MAX_LENGTH * 2)
        capped, full = Nist80063Parser._cap_description(text)
        assert full == text
        assert len(capped) < DESCRIPTION_MAX_LENGTH

    def test_the_base_class_never_rewrites_a_shipped_full_text(
        self, tmp_path: Path,
    ) -> None:
        """Asserts on the artifact run() wrote, which is what the cap protects."""
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "sp800_63b.html").write_text(
            _full_document(long_body_chars=4000), encoding="utf-8",
        )
        parser = Nist80063Parser(
            raw_dir=raw, output_dir=tmp_path / "out", audit_dir=tmp_path / "aud",
        )
        output = parser.run()
        capped = [c for c in output.controls if c.full_text is not None]
        assert len(capped) == 1, "the fixture carries exactly one oversized body"
        assert all(
            len(c.description) < DESCRIPTION_MAX_LENGTH for c in output.controls
        )
        # The base class would have replaced full_text with the sanitised
        # description. It did not, so the parser's own value survived.
        assert capped[0].full_text is not None
        assert len(capped[0].full_text) > len(capped[0].description)

    def test_the_repair_audit_carries_the_text_on_both_sides(
        self, tmp_path: Path,
    ) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "sp800_63b.html").write_text(
            _full_document(long_body_chars=4000), encoding="utf-8",
        )
        audit = tmp_path / "aud"
        parser = Nist80063Parser(
            raw_dir=raw, output_dir=tmp_path / "out", audit_dir=audit,
        )
        parser.run()
        records = [
            json.loads(line)
            for line in (audit / "nist_800_63.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(records) == 1
        record = records[0]
        assert record["repair"] == "description_capped_to_protect_full_text"
        assert isinstance(record["before"], str)
        assert isinstance(record["after"], str)
        assert record["before"].startswith(record["after"][:200])
        assert len(record["before"]) > len(record["after"])

    def test_the_audit_is_written_even_when_no_cap_fires(
        self, tmp_path: Path,
    ) -> None:
        """A missing file must mean the parser never ran, not that nothing fired."""
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "sp800_63b.html").write_text(_full_document(), encoding="utf-8")
        audit = tmp_path / "aud"
        parser = Nist80063Parser(
            raw_dir=raw, output_dir=tmp_path / "out", audit_dir=audit,
        )
        parser.run()
        assert (audit / "nist_800_63.jsonl").read_text(encoding="utf-8") == ""


class TestRunOnASyntheticDocument:
    """The whole pipeline, in CI, without data/raw."""

    def test_run_writes_a_complete_artifact(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "sp800_63b.html").write_text(_full_document(), encoding="utf-8")
        parser = Nist80063Parser(
            raw_dir=raw, output_dir=tmp_path / "out", audit_dir=tmp_path / "aud",
        )
        output = parser.run()
        assert len(output.controls) == 118
        assert [s.path for s in output.source_files] == ["sp800_63b.html"]
        assert output.framework_id == "nist_800_63"
        assert set(REQUIRED_SECTION_IDS) <= {c.control_id for c in output.controls}
        assert (tmp_path / "out" / "nist_800_63.json").is_file()

    def test_a_short_document_fails_the_count_check(self, tmp_path: Path) -> None:
        """expected_count is exact, so a document that clears the structural
        floor at 100 can still be short enough to be wrong."""
        raw = tmp_path / "raw"
        raw.mkdir()
        sections = [
            (number, f"Required Section {position}",
             f"Invented statement {position}, long enough to clear the "
             f"honest-prose floor without restating its heading.")
            for position, number in enumerate(sorted(REQUIRED_SECTION_IDS))
        ]
        sections += [
            (f"20.{n}", f"Filler {n}",
             f"Invented filler statement {n}, long enough to clear the "
             f"honest-prose floor on its own.")
            for n in range(100 - len(REQUIRED_SECTION_IDS))
        ]
        (raw / "sp800_63b.html").write_text(_document(sections), encoding="utf-8")
        parser = Nist80063Parser(
            raw_dir=raw, output_dir=tmp_path / "out", audit_dir=tmp_path / "aud",
        )
        with pytest.raises(ValueError, match="expected 118"):
            parser.run()

    def test_a_prose_free_document_fails_the_prose_floor(
        self, tmp_path: Path,
    ) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        sections = [
            (number, f"Required Section {position}", "")
            for position, number in enumerate(sorted(REQUIRED_SECTION_IDS))
        ]
        sections += [
            (f"20.{n}", f"Filler {n}", "")
            for n in range(118 - len(REQUIRED_SECTION_IDS))
        ]
        raw_html = _document(sections).replace("<p></p>\n", "")
        (raw / "sp800_63b.html").write_text(raw_html, encoding="utf-8")
        parser = Nist80063Parser(
            raw_dir=raw, output_dir=tmp_path / "out", audit_dir=tmp_path / "aud",
        )
        with pytest.raises(ValueError, match="honest prose fraction"):
            parser.run()


class TestRunOnTheRealDocument:
    def test_run_writes_from_the_real_document(self, tmp_path: Path) -> None:
        parser = Nist80063Parser(
            output_dir=tmp_path, audit_dir=tmp_path / "aud",
        )
        try:
            output = parser.run()
        except FileNotFoundError:
            pytest.skip("data/raw is gitignored and absent in this checkout")
        assert len(output.controls) == 118
        assert [s.path for s in output.source_files] == ["sp800_63b.html"]
        # R14, on the real bodies. 22 of the 118 are over the limit.
        assert all(
            len(c.description) < DESCRIPTION_MAX_LENGTH for c in output.controls
        )
        assert sum(1 for c in output.controls if c.full_text is not None) == 22

    def test_every_curated_id_the_document_can_answer_is_a_control(
        self, tmp_path: Path,
    ) -> None:
        parser = Nist80063Parser(
            output_dir=tmp_path, audit_dir=tmp_path / "aud",
        )
        try:
            output = parser.run()
        except FileNotFoundError:
            pytest.skip("data/raw is gitignored and absent in this checkout")
        found = {c.control_id for c in output.controls}
        answerable = _curated_section_ids() - {UNANSWERABLE_SECTION_ID}
        assert answerable <= found
        assert UNANSWERABLE_SECTION_ID not in found
