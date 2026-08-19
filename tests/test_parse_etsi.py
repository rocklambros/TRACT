"""ETSI is restricted: every fixture here is synthetic, none of it is source.

ETSI's copyright notification reserves reproduction, so nothing in this file
quotes the deliverable. The document identifier in the fixture is an invented
deliverable number in the real one's shape, the clause headings are invented,
and every statement is generated from a filler vocabulary that exists nowhere
outside this module. Every assertion about the real PDF is a count, a length, a
digest or a negative, so the suite can gate on the source without carrying it.

The fixture mirrors the source's SHAPE rather than its words, because three of
the four defects this parser fixes are shapes: a running header whose page
number falls in the clause range, running furniture inside a clause body, and a
change-history table past the last numbered clause. A fixture without them
leaves the regression tests passing against the defect, which is decoration.
The released reader gave clauses 5, 6 and 7 the document identifier as their
heading and 22,639 characters of front matter as one statement, while every
gate stayed green.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Final

import pytest

from parsers.parse_etsi import (
    ANNEX_HEADING,
    CLAUSE,
    DESCRIPTION_BUDGET,
    DOCUMENT_IDENTIFIER,
    EXPECTED_CLAUSES,
    NAME_SECTION_IDS,
    RUNNING_FOOTER,
    RUNNING_HEADER,
    SOURCE_FILE,
    SOURCE_SHA256,
    Clause,
    EtsiParser,
)
from tests.synthetic_pdf import build_pdf
from tract.config import (
    DESCRIPTION_MAX_LENGTH,
    HONEST_PROSE_MIN_CHARS,
    MAX_ANCHOR_CHARS,
)
from tract.corpus_report import SYNTHETIC_TEXT_ORIGIN, TEXT_ORIGIN_METADATA_KEY
from tract.text_selection import prepare_anchor

# An invented deliverable number and version in the real one's shape. ETSI
# publishes no SAI 999 and no V9.9.9, so this string is the shape without the
# document.
DOCUMENT_ID: Final[str] = "ETSI GR SAI 999 V9.9.9 (2099-01)"

# Nonsense tokens. A filler vocabulary of real security words could collide
# with a 12-word window of the source, which is what the tree-wide fingerprint
# gate scans for.
_FILLER: Final[tuple[str, ...]] = (
    "quorix", "vandel", "tessary", "morvale", "brantic", "solimer",
    "delvane", "pratule", "cindrel", "harveto", "nusquam", "veldric",
)

# The source lays 25 numbered clauses over three top-level ones, seven of which
# are headings whose text lives entirely in their subclauses. [measured] The
# fixture carries the same tree so the roll-up, the count gate and the
# truncation behaviour all run in CI.
#
# The two 2,400-character bodies sit on the first child of a rolled-up parent,
# which is where the source puts its own: MAX_ANCHOR_CHARS cuts at 2,150 and a
# roll-up opens with its first child, so parent and child present one anchor.
_TREE: Final[tuple[tuple[str, str, int], ...]] = (
    ("5", "Alpha area", 0),
    ("5.1", "Opening topic", 220),
    ("5.2", "Grouped topic", 0),
    ("5.2.1", "Overview", 220),
    ("5.2.2", "First member", 240),
    ("5.2.3", "Second member", 200),
    ("5.3", "Second grouped topic", 0),
    ("5.3.1", "Overview", 210),
    ("5.3.2", "Third member", 230),
    ("5.3.3", "Fourth member", 200),
    ("6", "Beta area", 0),
    ("6.1", "Second opening topic", 2400),
    ("6.2", "Third grouped topic", 0),
    ("6.2.1", "Overview", 2400),
    ("6.2.2", "Fifth member", 250),
    ("6.2.3", "Sixth member", 210),
    ("6.3", "Fourth grouped topic", 0),
    ("6.3.1", "Overview", 200),
    ("6.3.2", "Seventh member", 240),
    ("6.3.3", "Eighth member", 220),
    ("6.4", "Fifth grouped topic", 0),
    ("6.4.1", "Overview", 200),
    ("6.4.2", "Ninth member", 230),
    ("6.4.3", "Tenth member", 210),
    ("7", "Closing area", 250),
)

# Eight content lines to a page, so the running header and footer land inside
# clause bodies as often as they do in the source.
_LINES_PER_PAGE: Final[int] = 8
_LINE_CHARS: Final[int] = 84
# Seven pages of preamble, so the pages numbered 5, 6 and 7 are the ones whose
# header can be read as a top-level clause. That is the whole defect. The
# preamble carries a contents page, because every clause appears there too and
# the contents page comes first in page order, so a fixture without one leaves
# CLAUSE's heading bound untested.
_FRONT_MATTER_PAGES: Final[int] = 7
# Each contents row runs past the heading bound, as the source's do at 137 to
# 170 characters against a bound of 81.
_CONTENTS_ROW_CHARS: Final[int] = 100

# The tail past the last numbered clause. The date row has the shape CLAUSE
# matches, which is the fourth false heading in the source.
_TAIL: Final[tuple[str, ...]] = (
    "Annex A:",
    "Change History",
    "Date Version Information about changes",
    "6 June 2099 0.0.7 An invented summary of an invented editorial change.",
    "History",
    "Document history",
    "V9.9.9 January 2099 Publication",
)


# A clause list where string order and clause order disagree, and where one
# number is a bare prefix of another without being its parent. The source has
# neither shape, so both of the rules that handle them are untestable against
# it, and a rule nothing exercises is a rule that quietly stops holding.
_TWO_DIGIT_TEXT: Final[str] = (
    "6.1 Grouped topic\n"
    "6.1.1 Only child\n"
    "The only child carries a statement long enough to clear the prose floor.\n"
    "6.2 Second sibling\n"
    "The second sibling carries a statement of its own, also long enough.\n"
    "6.10 Tenth sibling\n"
    "The tenth sibling carries a different statement, also long enough.\n"
    "Annex A:\n"
)
# As strings these sort 6.1, 6.1.1, 6.10, 6.2. As clauses they do not.
_TWO_DIGIT_ORDER: Final[tuple[str, ...]] = ("6.1", "6.1.1", "6.2", "6.10")


def _statement(tag: str, chars: int) -> str:
    """Invented prose of at least `chars` characters, unique to `tag`.

    Carries no digit anywhere. A wrapped line opening with a bare 5, 6 or 7
    would match CLAUSE and turn the fixture's own body text into a second
    heading for a number that already has one, which the parser refuses. The
    clause number is spelled as a filler slug for that reason.
    """
    slug = "".join(_FILLER[int(part) % len(_FILLER)][:3] for part in tag.split("."))
    parts = [f"Clause {slug} sets an invented requirement for this fixture."]
    index = 0
    while len(" ".join(parts)) < chars:
        marker = (
            _FILLER[index % len(_FILLER)]
            + _FILLER[(index // len(_FILLER)) % len(_FILLER)][:2]
        )
        picks = " ".join(
            _FILLER[(index * 5 + offset) % len(_FILLER)] for offset in range(6)
        )
        parts.append(f"Point {slug} {marker} records {picks}.")
        index += 1
    return " ".join(parts)


def _wrap(text: str) -> list[str]:
    """Break a statement into lines the fixture page can hold."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > _LINE_CHARS and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _contents_rows(tree: tuple[tuple[str, str, int], ...]) -> list[str]:
    """One dot-leader row per clause, in the shape the source's contents use."""
    rows = ["Contents"]
    for number, title, _ in tree:
        stem = f"{number} {title} "
        rows.append(stem + "." * (_CONTENTS_ROW_CHARS - len(stem)))
    return rows


def _content_lines(tree: tuple[tuple[str, str, int], ...]) -> list[str]:
    """Preamble, contents, one heading plus body per clause, then the tail.

    The front matter is padded to exactly seven pages whatever the tree holds,
    so the pages numbered 5, 6 and 7 always carry preamble and their headers
    always precede the real top-level headings.
    """
    contents = _contents_rows(tree)
    lines = [
        f"Preamble line {n} of the invented front matter for this fixture."
        for n in range(_FRONT_MATTER_PAGES * _LINES_PER_PAGE - len(contents))
    ]
    lines.extend(contents)
    for number, title, chars in tree:
        lines.append(f"{number} {title}")
        if chars:
            lines.extend(_wrap(_statement(number, chars)))
    lines.extend(_TAIL)
    return lines


def _pdf(tree: tuple[tuple[str, str, int], ...] = _TREE) -> bytes:
    """A paginated PDF carrying a running header and footer on every page."""
    lines = _content_lines(tree)
    pages = [
        lines[start:start + _LINES_PER_PAGE]
        for start in range(0, len(lines), _LINES_PER_PAGE)
    ]
    runs: list[list[tuple[float, float, str]]] = []
    for number, page in enumerate(pages, start=1):
        page_runs: list[tuple[float, float, str]] = [
            (72.0, 40.0, f"{number} {DOCUMENT_ID}"),
        ]
        page_runs += [
            (72.0, 70.0 + offset * 14.0, line)
            for offset, line in enumerate(page)
        ]
        page_runs.append((72.0, 750.0, "ETSI"))
        runs.append(page_runs)
    return build_pdf(runs)


def _text(tree: tuple[tuple[str, str, int], ...] = _TREE) -> str:
    """The same content as plain text, for the pure-parsing tests."""
    return "\n".join(_content_lines(tree))


def _parser(tmp_path: Path, payload: bytes) -> EtsiParser:
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    (raw / SOURCE_FILE).write_bytes(payload)
    (tmp_path / "out").mkdir(exist_ok=True)
    instance = EtsiParser(
        raw_dir=raw, output_dir=tmp_path / "out", audit_dir=tmp_path / "aud",
    )
    instance.expected_sha256 = hashlib.sha256(payload).hexdigest()
    return instance


def _curated_rows() -> list[dict[str, str]]:
    """Every curated ETSI link, from the tracked link file."""
    path = Path(__file__).resolve().parent.parent / (
        "data/training/hub_links_curated.jsonl"
    )
    with path.open(encoding="utf-8") as handle:
        return [
            row for row in (json.loads(line) for line in handle if line.strip())
            if row.get("framework_id") == "etsi"
        ]


def _real(tmp_path: Path) -> EtsiParser:
    """A parser pointed at the real source, skipping where it is absent.

    resolve_raw_dir raises rather than returning None when the tree is absent,
    which on a fresh clone would surface as an error rather than a skip.
    """
    try:
        present = (EtsiParser.resolve_raw_dir() / SOURCE_FILE).is_file()
    except FileNotFoundError:
        present = False
    if not present:
        pytest.skip("data/raw is gitignored and absent in this checkout")
    return EtsiParser(output_dir=tmp_path / "out", audit_dir=tmp_path / "aud")


class TestPatterns:
    """The patterns, at the smallest scope that shows what each one is for."""

    def test_the_clause_pattern_reads_a_running_header_as_a_clause(self) -> None:
        """The defect itself. CLAUSE alone cannot tell the two apart."""
        match = CLAUSE.match(f"5 {DOCUMENT_ID}")
        assert match is not None
        assert match.group(1) == "5"
        assert match.group(2) == DOCUMENT_ID

    def test_the_identifier_guard_separates_them(self) -> None:
        assert DOCUMENT_IDENTIFIER.match(DOCUMENT_ID) is not None
        assert DOCUMENT_IDENTIFIER.match("Alpha area") is None

    def test_the_header_pattern_needs_the_page_number(self) -> None:
        assert RUNNING_HEADER.match(f"12 {DOCUMENT_ID}") is not None
        assert RUNNING_HEADER.match(DOCUMENT_ID) is None

    def test_the_footer_pattern_is_the_whole_line(self) -> None:
        assert RUNNING_FOOTER.match("ETSI") is not None
        assert RUNNING_FOOTER.match("ETSI GR SAI 999") is None

    def test_the_annex_heading_is_not_a_clause(self) -> None:
        assert ANNEX_HEADING.match("Annex A:") is not None
        assert ANNEX_HEADING.match("Annex A: Change History and more") is None

    def test_pages_five_six_and_seven_are_the_only_ones_at_risk(self) -> None:
        """Names why the defect hits three clauses rather than thirty."""
        at_risk = [n for n in range(1, 40) if CLAUSE.match(f"{n} {DOCUMENT_ID}")]
        assert at_risk == [5, 6, 7]


class TestClausesFromText:
    def test_every_clause_in_the_tree_is_found(self) -> None:
        clauses = EtsiParser.clauses_from_text(_text())
        assert sorted(clauses) == sorted(number for number, _, _ in _TREE)

    def test_the_preamble_is_not_a_mapping_unit(self) -> None:
        clauses = EtsiParser.clauses_from_text(_text())
        assert not [n for n in clauses if n.startswith("4")]

    def test_a_running_header_is_not_a_clause_heading(self) -> None:
        """The defect, at the smallest scope that shows it.

        The header comes first in page order, so without the guard it takes
        the number and the real heading is the one that loses the slot.
        """
        text = f"5 {DOCUMENT_ID}\nPreamble.\n\n5 Alpha area\n{_statement('5', 200)}\nAnnex A:\n"
        clauses = EtsiParser.clauses_from_text(text)
        assert clauses["5"].heading == "Alpha area"

    def test_a_repeated_clause_number_is_refused(self) -> None:
        """Silence here is how the headers won the slot for three clauses."""
        text = (
            f"5 Alpha area\n{_statement('5', 200)}\n"
            f"5 Second alpha area\n{_statement('5.1', 200)}\nAnnex A:\n"
        )
        with pytest.raises(ValueError, match="matches on two different lines"):
            EtsiParser.clauses_from_text(text)

    def test_a_document_with_no_annex_is_refused(self) -> None:
        """Without the boundary, clause 7 swallows the change-history table."""
        with pytest.raises(ValueError, match="no annex heading"):
            EtsiParser.clauses_from_text(f"5 Alpha area\n{_statement('5', 200)}\n")

    def test_the_change_history_table_is_outside_the_clause_range(self) -> None:
        """Its date row has CLAUSE's shape and would claim clause 6."""
        clauses = EtsiParser.clauses_from_text(_text())
        assert clauses["6"].heading == "Beta area"
        assert "0.0.7" not in clauses["7"].body
        assert "Document history" not in clauses["7"].body

    def test_running_furniture_is_removed_from_a_body(self) -> None:
        clauses = EtsiParser.clauses_from_text(_text())
        bodies = [clause.body for clause in clauses.values()]
        assert bodies, "the fixture produced no bodies to check"
        for body in bodies:
            assert DOCUMENT_ID not in body
            assert "ETSI" not in body

    def test_the_fixture_actually_puts_furniture_inside_a_body(self) -> None:
        """Otherwise the check above passes against a fixture with no defect."""
        raw = _text().split("\n")
        start = raw.index("6.1 Second opening topic")
        end = next(i for i in range(start + 1, len(raw)) if CLAUSE.match(raw[i]))
        assert sum(1 for line in raw[start:end] if line == "ETSI") == 0
        # The plain-text form carries no furniture, so the shape only exists
        # once the fixture goes through a page layout. That is what the PDF
        # tests below cover, and this asserts the two forms differ on purpose.
        assert f"1 {DOCUMENT_ID}" not in _text()

    def test_a_parent_with_no_text_takes_its_descendants(self) -> None:
        clauses = EtsiParser.clauses_from_text(_text())
        parent = clauses["5.2"]
        assert parent.own_body == ""
        assert parent.assembled
        for child in ("5.2.1", "5.2.2", "5.2.3"):
            assert clauses[child].body in parent.body

    def test_a_roll_up_counts_each_descendant_once(self) -> None:
        """An empty parent contributes an empty string, so nothing doubles."""
        clauses = EtsiParser.clauses_from_text(_text())
        grandchild = clauses["5.2.1"].body
        assert clauses["5"].body.count(grandchild) == 1

    def test_a_clause_with_its_own_text_does_not_roll_up(self) -> None:
        clauses = EtsiParser.clauses_from_text(_text())
        assert not clauses["5.1"].assembled
        assert clauses["5.2.1"].body not in clauses["5.1"].body

    def test_exactly_the_seven_heading_only_clauses_are_assembled(self) -> None:
        clauses = EtsiParser.clauses_from_text(_text())
        assembled = sorted(n for n, c in clauses.items() if c.assembled)
        assert assembled == ["5", "5.2", "5.3", "6", "6.2", "6.3", "6.4"]

    def test_a_leaf_clause_never_borrows_a_sibling(self) -> None:
        clauses = EtsiParser.clauses_from_text(_text())
        assert clauses["5.2.2"].body not in clauses["5.2.3"].body

    def test_a_two_digit_sibling_is_not_a_child(self) -> None:
        """6.10 follows 6.1 in string order and is its sibling, not its child."""
        clauses = EtsiParser.clauses_from_text(_TWO_DIGIT_TEXT)
        assert sorted(clauses) == sorted(_TWO_DIGIT_ORDER)
        assert clauses["6.1"].assembled
        assert clauses["6.1.1"].body in clauses["6.1"].body
        assert clauses["6.10"].body not in clauses["6.1"].body

    def test_a_parent_with_a_stub_lead_keeps_it_and_still_rolls_up(self) -> None:
        """The roll-up threshold is the prose floor, not emptiness.

        A lead too short to stand as a statement still belongs to the clause,
        so it leads the roll-up rather than being replaced by it.
        """
        stub = "A lead line."
        text = (
            f"5 Alpha area\n{stub}\n5.1 Opening topic\n"
            f"{_statement('5.1', 200)}\nAnnex A:\n"
        )
        clauses = EtsiParser.clauses_from_text(text)
        assert len(stub) < HONEST_PROSE_MIN_CHARS
        assert clauses["5"].own_body == stub
        assert clauses["5"].assembled
        assert clauses["5"].body.startswith(stub)
        assert clauses["5.1"].body in clauses["5"].body

    def test_a_contents_row_is_not_a_clause_heading(self) -> None:
        """Every clause appears on the contents page, and it comes first.

        Without CLAUSE's heading bound each of those rows would claim its
        clause number ahead of the real heading, which the duplicate check
        then refuses.
        """
        clauses = EtsiParser.clauses_from_text(_text())
        assert clauses["5"].heading == "Alpha area"
        for clause in clauses.values():
            assert "...." not in clause.heading

    def test_the_fixture_carries_a_contents_row_for_every_clause(self) -> None:
        """Otherwise the check above passes against a fixture with no page."""
        rows = _contents_rows(_TREE)
        assert len(rows) == len(_TREE) + 1
        matched = [row for row in rows if CLAUSE.match(row)]
        assert matched == [], "a contents row must not fit the heading bound"
        assert all(len(row) == _CONTENTS_ROW_CHARS for row in rows[1:])


class TestClause:
    def test_assembled_is_carried_not_inferred(self) -> None:
        assert Clause("H", "body", "body").assembled is False
        assert Clause("H", "", "child text").assembled is True


class TestBuildControls:
    def test_a_clause_with_no_text_at_all_is_refused(self) -> None:
        """A heading with no body and no subclause leaves the prose index."""
        with pytest.raises(ValueError, match="no text of its own"):
            EtsiParser.build_controls({"5": Clause("Alpha area", "", "")}, {})

    def test_controls_are_ordered_numerically_not_as_strings(self) -> None:
        """5.10 sorts before 5.2 as a string and after it as a clause."""
        controls = EtsiParser.build_controls(
            EtsiParser.clauses_from_text(_TWO_DIGIT_TEXT), {},
        )
        assert [c.control_id for c in controls] == list(_TWO_DIGIT_ORDER)
        assert sorted(_TWO_DIGIT_ORDER) != list(_TWO_DIGIT_ORDER), (
            "the fixture must be a case where the two orders disagree"
        )

    def test_a_roll_up_is_recorded_as_a_repair_with_both_sides(self) -> None:
        repairs: list[dict[str, object]] = []
        EtsiParser.build_controls(
            EtsiParser.clauses_from_text(_TWO_DIGIT_TEXT), {}, repairs,
        )
        assembled = [
            r for r in repairs
            if r["repair"] == "statement_assembled_from_subclauses"
        ]
        assert [r["control_id"] for r in assembled] == ["6.1"]
        assert assembled[0]["before"] == ""
        assert isinstance(assembled[0]["after"], str)


class TestNameShapedIds:
    def test_a_declared_name_becomes_an_alternate_title_on_its_clause(
        self,
    ) -> None:
        controls = EtsiParser.build_controls(
            EtsiParser.clauses_from_text(_text()), {"Some technique": "5.1"},
        )
        first = next(c for c in controls if c.control_id == "5.1")
        assert first.metadata is not None
        assert first.metadata["alt_titles"] == ["Some technique"]

    def test_no_other_clause_gains_an_alternate(self) -> None:
        controls = EtsiParser.build_controls(
            EtsiParser.clauses_from_text(_text()), {"Some technique": "5.1"},
        )
        holders = [
            c.control_id for c in controls
            if (c.metadata or {}).get("alt_titles")
        ]
        assert holders == ["5.1"]

    def test_two_names_on_one_clause_are_both_kept_in_order(self) -> None:
        controls = EtsiParser.build_controls(
            EtsiParser.clauses_from_text(_text()),
            {"Second name": "5.1", "First name": "5.1"},
        )
        first = next(c for c in controls if c.control_id == "5.1")
        assert (first.metadata or {})["alt_titles"] == ["First name", "Second name"]

    def test_a_declared_clause_that_is_absent_is_refused(self) -> None:
        with pytest.raises(ValueError, match="NAME_SECTION_IDS"):
            EtsiParser.build_controls(
                EtsiParser.clauses_from_text(_text()),
                {"Nowhere technique": "9.9"},
            )

    def test_the_declared_map_is_the_two_rows_the_link_file_needs(self) -> None:
        """Derived from the link file, so the map cannot drift from it.

        A section_id that is not a clause number reaches its clause only
        through this map. One that is a clause number needs no entry, and an
        entry for it would put a name in front of the id channel.
        """
        name_shaped = {
            str(row["section_id"]) for row in _curated_rows()
            if not str(row["section_id"])[0].isdigit()
        }
        assert sorted(NAME_SECTION_IDS) == sorted(name_shaped)

    def test_every_declared_target_is_a_clause_the_source_defines(self) -> None:
        for clause_id in NAME_SECTION_IDS.values():
            assert CLAUSE.match(f"{clause_id} Heading") is not None

    def test_each_declared_name_shares_a_hub_with_the_clause_it_names(
        self,
    ) -> None:
        """Which clause a name belongs to, decided from the link file alone.

        A name-shaped row and a numbered row that target the same CRE are two
        curator statements about one place in the document. That agreement is
        the evidence for the mapping, it lives in a tracked file, and it names
        no ETSI text. Pointing an entry at the neighbouring clause breaks it.
        """
        rows = _curated_rows()
        for name, clause_id in NAME_SECTION_IDS.items():
            named = {r["cre_id"] for r in rows if r["section_id"] == name}
            numbered = {r["cre_id"] for r in rows if r["section_id"] == clause_id}
            assert named, f"{name} is not a section_id in the link file"
            assert named <= numbered, (
                f"{name} targets {sorted(named - numbered)}, which no link on "
                f"clause {clause_id} targets"
            )


class TestSyntheticPdf:
    """parse() and run() through pdfplumber, in CI, without data/raw."""

    @pytest.fixture()
    def parser(self, tmp_path: Path) -> EtsiParser:
        return _parser(tmp_path, _pdf())

    def test_the_fixture_carries_the_headers_that_cause_the_defect(
        self, parser: EtsiParser,
    ) -> None:
        """A fixture without them leaves every test below decorative."""
        import pdfplumber

        with pdfplumber.open(parser.raw_dir / SOURCE_FILE) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        headers = [
            line for page in pages for line in page.split("\n")
            if CLAUSE.match(line.strip())
            and DOCUMENT_IDENTIFIER.match(CLAUSE.match(line.strip()).group(2))  # type: ignore[union-attr]
        ]
        assert len(headers) == 3, "pages 5, 6 and 7 carry the at-risk headers"
        assert sum(
            1 for page in pages for line in page.split("\n")
            if RUNNING_FOOTER.match(line.strip())
        ) == len(pages)

    def test_the_three_top_level_clauses_keep_their_own_headings(
        self, parser: EtsiParser,
    ) -> None:
        controls = {c.control_id: c for c in parser.parse()}
        assert controls["5"].title == "Alpha area"
        assert controls["6"].title == "Beta area"
        assert controls["7"].title == "Closing area"

    def test_no_clause_heading_is_the_document_identifier(
        self, parser: EtsiParser,
    ) -> None:
        for control in parser.parse():
            assert DOCUMENT_IDENTIFIER.match(control.title) is None

    def test_no_shipped_text_carries_the_running_furniture(
        self, parser: EtsiParser,
    ) -> None:
        """A framework-identifying token in an anchor is a learnable shortcut."""
        for control in parser.parse():
            for field in (control.description, control.full_text or ""):
                assert DOCUMENT_ID not in field
                assert not re.search(r"\bETSI\b", field)

    def test_a_rolled_up_clause_is_marked_synthetic(
        self, parser: EtsiParser,
    ) -> None:
        controls = {c.control_id: c for c in parser.parse()}
        marked = sorted(
            number for number, control in controls.items()
            if (control.metadata or {}).get(TEXT_ORIGIN_METADATA_KEY)
            == SYNTHETIC_TEXT_ORIGIN
        )
        assert marked == ["5", "5.2", "5.3", "6", "6.2", "6.3", "6.4"]

    def test_hierarchy_is_recorded(self, parser: EtsiParser) -> None:
        controls = {c.control_id: c for c in parser.parse()}
        assert controls["5"].parent_id is None
        assert controls["5.2.1"].parent_id == "5.2"
        assert {c.hierarchy_level for c in controls.values()} == {"clause"}

    def test_a_short_clause_list_is_refused(self, tmp_path: Path) -> None:
        """The count band would accept 24 of 25. The clause gate does not."""
        parser = _parser(tmp_path, _pdf(_TREE[:-1]))
        with pytest.raises(ValueError, match="numbered clause"):
            parser.parse()

    def test_a_short_clause_list_clears_the_count_band(self) -> None:
        """Names why the clause gate exists, in both directions."""
        from tract.config import COUNT_TOLERANCE

        assert abs(24 - EXPECTED_CLAUSES) / EXPECTED_CLAUSES <= COUNT_TOLERANCE

    def test_a_pdf_that_is_not_the_pinned_one_is_refused(
        self, tmp_path: Path,
    ) -> None:
        parser = _parser(tmp_path, _pdf())
        parser.expected_sha256 = "0" * 64
        with pytest.raises(ValueError, match="not the pinned"):
            parser.parse()

    def test_run_writes_a_complete_artifact(self, parser: EtsiParser) -> None:
        output = parser.run()
        assert len(output.controls) == EXPECTED_CLAUSES
        assert [s.path for s in output.source_files] == [SOURCE_FILE]
        assert output.framework_id == "etsi"
        assert output.mapping_unit_level == "clause"
        assert (parser.output_dir / "etsi.json").is_file()


class TestDescriptionCap:
    """Ruling R14. The parser owns its anchor rather than a margin."""

    def test_a_statement_inside_the_budget_is_not_cut(self) -> None:
        text = "An invented statement well under the budget."
        assert EtsiParser._cap_description(text) == (text, None)

    def test_an_oversized_statement_is_cut_under_the_limit(self) -> None:
        text = "word " * DESCRIPTION_MAX_LENGTH
        capped, full = EtsiParser._cap_description(text)
        assert full == text
        assert len(capped) < DESCRIPTION_MAX_LENGTH
        assert not capped.endswith(" ")

    def test_a_statement_with_no_space_still_lands_under_the_limit(self) -> None:
        text = "x" * (DESCRIPTION_MAX_LENGTH * 2)
        capped, full = EtsiParser._cap_description(text)
        assert full == text
        assert len(capped) < DESCRIPTION_MAX_LENGTH

    def test_the_budget_is_strictly_under_the_limit(self) -> None:
        assert DESCRIPTION_BUDGET < DESCRIPTION_MAX_LENGTH

    def test_the_base_class_never_rewrites_a_shipped_full_text(
        self, tmp_path: Path,
    ) -> None:
        """Asserts on the artifact run() wrote, which is what the cap protects."""
        output = _parser(tmp_path, _pdf()).run()
        capped = [c for c in output.controls if c.full_text is not None]
        assert len(capped) == 4, "the fixture carries four oversized statements"
        assert all(
            len(c.description) < DESCRIPTION_MAX_LENGTH for c in output.controls
        )
        for control in capped:
            assert control.full_text is not None
            assert len(control.full_text) > len(control.description)
            assert control.full_text.startswith(control.description)


class TestRepairAudit:
    """A count says a repair fired. Only the pair says what moved."""

    def _records(self, tmp_path: Path) -> list[dict[str, object]]:
        parser = _parser(tmp_path, _pdf())
        parser.run()
        path = parser.audit_dir / "etsi.jsonl"
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]

    def test_both_repair_kinds_are_recorded(self, tmp_path: Path) -> None:
        records = self._records(tmp_path)
        kinds = {str(r["repair"]) for r in records}
        assert kinds == {
            "statement_assembled_from_subclauses",
            "description_capped_to_protect_full_text",
        }

    def test_every_roll_up_carries_its_text_on_both_sides(
        self, tmp_path: Path,
    ) -> None:
        rolled = [
            r for r in self._records(tmp_path)
            if r["repair"] == "statement_assembled_from_subclauses"
        ]
        assert len(rolled) == 7
        for record in rolled:
            assert record["before"] == ""
            assert isinstance(record["after"], str)
            assert len(str(record["after"])) > 0

    def test_every_cap_carries_its_text_on_both_sides(
        self, tmp_path: Path,
    ) -> None:
        capped = [
            r for r in self._records(tmp_path)
            if r["repair"] == "description_capped_to_protect_full_text"
        ]
        assert len(capped) == 4
        for record in capped:
            before, after = str(record["before"]), str(record["after"])
            assert before.startswith(after)
            assert len(before) > len(after)

    def test_the_audit_is_written_even_with_nothing_to_report(
        self, tmp_path: Path,
    ) -> None:
        """A missing file must mean the parser never ran."""
        flat = tuple(
            (number, title, 200 if chars == 0 else chars)
            for number, title, chars in _TREE
        )
        parser = _parser(tmp_path, _pdf(flat))
        parser.run()
        assert (parser.audit_dir / "etsi.jsonl").is_file()
        records = (parser.audit_dir / "etsi.jsonl").read_text(encoding="utf-8")
        assert "statement_assembled_from_subclauses" not in records


class TestAnchorCollapse:
    """A rolled-up parent opens with its first child, and the budget cuts both.

    Reported rather than fixed. Rolling up is what makes the four links on
    5.2, 5.3 and 6.3 resolve at all, and the alternative is 4 unresolved links
    against 2 anchors an encoder cannot tell apart. distinct_anchors cannot see
    this, because no curated link targets either colliding parent.
    """

    def test_two_parent_child_pairs_present_one_anchor(
        self, tmp_path: Path,
    ) -> None:
        output = _parser(tmp_path, _pdf()).run()
        anchors: dict[str, list[str]] = {}
        for control in output.controls:
            anchor, _ = prepare_anchor(control.full_text or control.description)
            anchors.setdefault(anchor, []).append(control.control_id)
        collapsed = sorted(
            sorted(ids) for ids in anchors.values() if len(ids) > 1
        )
        assert collapsed == [["6", "6.1"], ["6.2", "6.2.1"]]
        assert len(anchors) == EXPECTED_CLAUSES - 2

    def test_the_stored_statements_are_all_distinct(
        self, tmp_path: Path,
    ) -> None:
        """The collapse is the budget's doing, not the parser's."""
        output = _parser(tmp_path, _pdf()).run()
        stored = [c.full_text or c.description for c in output.controls]
        assert len(set(stored)) == EXPECTED_CLAUSES
        assert MAX_ANCHOR_CHARS < max(len(s) for s in stored)


class TestRun:
    """Real-source assertions, all negative or numeric, so nothing is quoted."""

    def test_run_writes_from_the_real_pdf(self, tmp_path: Path) -> None:
        output = _real(tmp_path).run()
        assert len(output.controls) == EXPECTED_CLAUSES
        assert [s.path for s in output.source_files] == [SOURCE_FILE]
        assert output.source_files[0].sha256 == SOURCE_SHA256

    def test_no_real_clause_takes_the_document_identifier_as_a_heading(
        self, tmp_path: Path,
    ) -> None:
        """The regression test for the released defect, stated as a negative.

        Asserting the correct headings would put ETSI's text into a tracked
        file. Asserting that no heading is the identifier catches the same
        defect and quotes nothing.
        """
        output = _real(tmp_path).run()
        by_id = {c.control_id: c for c in output.controls}
        for control in output.controls:
            assert DOCUMENT_IDENTIFIER.match(control.title) is None
        # All three shared one heading under the defect, because all three took
        # the same running header. Three distinct headings is the positive half
        # of the same check and it names no ETSI text.
        assert len({by_id[n].title for n in ("5", "6", "7")}) == 3

    def test_no_real_statement_carries_the_running_furniture(
        self, tmp_path: Path,
    ) -> None:
        """656 characters of it, across 14 of the 25 clauses, before this.

        The probe is the running header's shape rather than the bare
        identifier. The document cites other ETSI deliverables in its own
        prose, and 4 of the 25 statements legitimately carry the publisher's
        name. What no statement may carry is a page number immediately in
        front of a deliverable identifier, which is what a header collapsed
        into a body looks like once the lines are joined.
        """
        for control in _real(tmp_path).run().controls:
            for field in (control.description, control.full_text or ""):
                assert not re.search(
                    r"\d\s+ETSI\s+(?:GR|GS|TS|TR|EG|EN)\s", field
                )

    def test_the_last_clause_is_not_the_change_history_table(
        self, tmp_path: Path,
    ) -> None:
        """Clause 7 shipped 2,776 characters, 1,889 of them the annex.

        Under the released reader it shipped 22,639. The statement itself is
        licensed, so the check is on its length and on the absence of the
        version strings the table is built from.
        """
        by_id = {c.control_id: c for c in _real(tmp_path).run().controls}
        last = by_id["7"]
        statement = last.full_text or last.description
        assert len(statement) < 1_000
        assert "0.0.1" not in statement
        assert "Document history" not in statement

    def test_exactly_two_alternates_are_registered(self, tmp_path: Path) -> None:
        """A name registered on one clause answers links naming the other.

        Two of the 24 technique names name two clauses each, so registering
        all of them would put a wrong anchor in front of the encoder while the
        resolution rate still read 1.0000.
        """
        output = _real(tmp_path).run()
        named = sorted(
            name
            for control in output.controls
            for name in (control.metadata or {}).get("alt_titles", [])
        )
        assert named == sorted(NAME_SECTION_IDS)

    def test_each_declared_name_occurs_in_the_clause_it_is_declared_on(
        self, tmp_path: Path,
    ) -> None:
        """Corroborates the link-file evidence against the document itself.

        The names come from NAME_SECTION_IDS rather than from a literal here,
        and they are OpenCRE section ids already carried by a tracked file, so
        this quotes nothing the repository does not already hold.
        """
        by_id = {c.control_id: c for c in _real(tmp_path).run().controls}
        for name, clause_id in NAME_SECTION_IDS.items():
            statement = by_id[clause_id].full_text or by_id[clause_id].description
            assert name.lower() in statement.lower()

    def test_the_measured_shape_of_the_real_parse(self, tmp_path: Path) -> None:
        """Every number here fails in both directions if the parse moves."""
        output = _real(tmp_path).run()
        assembled = [
            c.control_id for c in output.controls
            if (c.metadata or {}).get(TEXT_ORIGIN_METADATA_KEY)
            == SYNTHETIC_TEXT_ORIGIN
        ]
        assert assembled == ["5", "5.2", "5.3", "6", "6.2", "6.3", "6.4"]
        assert sum(1 for c in output.controls if c.full_text is not None) == 17
        assert max(len(c.description) for c in output.controls) < (
            DESCRIPTION_MAX_LENGTH
        )
        statements = [c.full_text or c.description for c in output.controls]
        assert min(len(s) for s in statements) == 452
        assert max(len(s) for s in statements) == 38_005
        assert len(set(statements)) == EXPECTED_CLAUSES

    def test_the_real_anchors_share_no_leading_boilerplate(
        self, tmp_path: Path,
    ) -> None:
        """Ruling R13, reported even though it is zero."""
        statements = [
            c.full_text or c.description for c in _real(tmp_path).run().controls
        ]
        shortest, longest = min(statements), max(statements)
        shared = 0
        while (
            shared < len(shortest)
            and shared < len(longest)
            and shortest[shared] == longest[shared]
        ):
            shared += 1
        assert shared == 0
