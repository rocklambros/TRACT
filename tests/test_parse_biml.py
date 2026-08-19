"""BIML's two documents reuse one id space, and OpenCRE leaves 8 ids unprefixed.

Measured: 'Data Confidentiality' names two different risks across the two PDFs
and 'Hosting' names three link rows. With a bare label as the title, ProseIndex
-- which resolves title before id -- gives all of them one anchor. A label is
not unique inside one document either, so the title carries the tag as well.

The other half of this file is about where a risk starts and stops. Both PDFs
wrap their columns, so a mid-sentence cross-reference lands at a line start
nine times, and reading one as a definition gave ara's data:2, which carries a
curated link, three unrelated summary paragraphs. Bodies also need a terminator
other than the next definition: without one, ara's system:10 absorbed the whole
reference list.

Every string here is written for this file. None is copied from either PDF.
TestSyntheticPdf drives parse() through pdfplumber against two PDFs this file
builds, so the multi-document read, the digest branch, the census gate and the
audit write all run in CI, where data/raw is absent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from parsers.parse_biml import (
    ARA,
    LLM24,
    MAX_BODY_CHARS,
    SOURCE_FILES,
    UNPREFIXED_IDS,
    BimlParser,
)
from tract.config import DESCRIPTION_MAX_LENGTH
from tests.synthetic_pdf import build_pdf

# A block-style document: the tag sits alone on its line and the statement
# follows. Line 4 is the shape that broke the naive rule -- the sentence above
# it ends with "see" and the wrapped continuation opens with a tag.
ARA_TEXT = """[raw:3:storage]
A stored pool of training records can be reached by anyone the storage grants,
and the grant is rarely reviewed after the pool is built. See
[system:8:insider], which names the same hazard from the other side.

[output:1:direct]
Output handed straight to a caller can be altered in transit, and the caller
has no way to tell an altered answer from an intended one. See
[data:4:storage] below.

[inference:4:hosting]
Where a model runs decides who can reach it, so a hosted model inherits every
trust boundary of whatever hosts it.
"""

LLM24_TEXT = """[raw:3:data feudalism]
A small number of parties control the records everyone else trains on, and the
terms of access are theirs to set.

[inference:9:hosting]
A model served by a third party puts the prompt, the completion and the system
message inside somebody else's trust boundary.
"""

# Every declared target, so require_targets has something to check rather than
# something to trip over. UNPREFIXED_IDS names five ara tags and two BIML-LLM24
# tags, and NAME_CONFLICTS names ara's output:1.
#
# 'storage' appears twice in the ara set and 'data confidentiality' twice in the
# BIML-LLM24 set, because the real documents repeat a label inside one document
# 34 times. 'hosting' appears in both sets, because the real documents repeat a
# label across the two documents as well.
ARA_TAGS: tuple[tuple[str, str], ...] = (
    ("raw:3", "storage"),
    ("data:4", "storage"),
    ("model:2", "trojan"),
    ("input:2", "controlled input stream"),
    ("inference:4", "hosting"),
    ("alg:11", "parameters"),
    ("output:1", "direct"),
    ("output:2", "provenance"),
)
LLM24_TAGS: tuple[tuple[str, str], ...] = (
    ("inference:9", "hosting"),
    ("output:4", "data confidentiality"),
    ("raw:5", "data confidentiality"),
    ("raw:3", "data feudalism"),
)

# 10pt text on 14pt leading, and a page tall enough for the longest block below.
_LEADING: float = 14.0
_TOP: float = 60.0
_LEFT: float = 72.0
_LINES_PER_PAGE: int = 44


def _pdf(lines: list[str]) -> bytes:
    """One PDF whose text lines come back in the order given."""
    pages: list[list[tuple[float, float, str]]] = []
    for start in range(0, len(lines), _LINES_PER_PAGE):
        chunk = lines[start:start + _LINES_PER_PAGE]
        pages.append([
            (_LEFT, _TOP + n * _LEADING, text)
            for n, text in enumerate(chunk)
            if text
        ])
    return build_pdf(pages)


def _document(tags: tuple[tuple[str, str], ...]) -> bytes:
    """A document in the inline style, with the furniture the real ones carry."""
    lines: list[str] = []
    for tag, label in tags:
        lines.append(
            f"[{tag}:{label}] A statement for {label} long enough to stand on"
        )
        lines.append(
            "its own as a risk, and it names [system:8:insider] mid-sentence. See"
        )
        # The wrapped cross-reference: a tag at a line start whose remainder
        # continues the sentence above it with a lowercase word.
        lines.append("[system:8:insider] below.")
        lines.append("BIML 7")
    return _pdf(lines)


class TestRisksFromText:
    def test_a_block_style_tag_defines_a_risk(self) -> None:
        risks = BimlParser.risks_from_text(ARA_TEXT, ARA)
        assert [tag for tag, _, _ in risks] == [
            "raw:3", "output:1", "inference:4",
        ]

    def test_a_wrapped_cross_reference_does_not_define_a_risk(self) -> None:
        """The line opens with [system:8:insider] and continues the sentence
        above it, so it is a reference and not a definition."""
        risks = BimlParser.risks_from_text(ARA_TEXT, ARA)
        assert "system:8" not in {tag for tag, _, _ in risks}

    def test_a_cross_reference_followed_by_a_lowercase_word_is_not_a_definition(
        self,
    ) -> None:
        """ara ends a summary line with "See" and opens the next with
        "[alg:2:reproducibility] below.", and does the same with "and" twice
        more. Reading the remainder as a body gave alg:2 three unrelated
        paragraphs."""
        risks = BimlParser.risks_from_text(ARA_TEXT, ARA)
        assert "data:4" not in {tag for tag, _, _ in risks}

    def test_a_wrapped_cross_reference_stays_in_the_body_above_it(self) -> None:
        risks = {tag: body for tag, _, body in BimlParser.risks_from_text(
            ARA_TEXT, ARA,
        )}
        assert "[system:8:insider]" in risks["raw:3"]

    def test_a_body_runs_to_the_next_definition(self) -> None:
        risks = {tag: body for tag, _, body in BimlParser.risks_from_text(
            ARA_TEXT, ARA,
        )}
        assert "rarely reviewed" in risks["raw:3"]
        assert "Output handed straight" not in risks["raw:3"]

    def test_a_body_keeps_its_line_breaks(self) -> None:
        """sanitize_text rejoins a word the PDF wrapped across a hyphen, and it
        can only see the break while the newline is still there."""
        risks = {tag: body for tag, _, body in BimlParser.risks_from_text(
            ARA_TEXT, ARA,
        )}
        assert "\n" in risks["raw:3"]

    def test_the_controls_heading_ends_a_body(self) -> None:
        text = (
            "[raw:3:storage] A statement about where records sit.\n"
            "Associated controls. Note that the labels refer to the risks:\n"
            "Boilerplate that belongs to no risk at all.\n"
        )
        risks = dict(
            (tag, body) for tag, _, body in BimlParser.risks_from_text(text, ARA)
        )
        assert "Boilerplate" not in risks["raw:3"]
        assert "where records sit" in risks["raw:3"]

    def test_a_page_break_at_a_sentence_boundary_ends_a_body(self) -> None:
        text = (
            "[raw:3:storage] A statement that finishes on this line.\n"
            "BIML 7\n"
            "A heading for the next part of the document\n"
        )
        risks = dict(
            (tag, body) for tag, _, body in BimlParser.risks_from_text(text, ARA)
        )
        assert "next part" not in risks["raw:3"]

    def test_a_page_break_inside_a_sentence_does_not_end_a_body(self) -> None:
        """Five running heads fall mid-sentence in the real documents, where the
        paragraph genuinely continues on the next page."""
        text = (
            "[raw:3:storage] A statement that runs past the foot of the page and\n"
            "26 Berryville Institute of Machine Learning\n"
            "carries on here, which is where it finishes.\n"
        )
        risks = dict(
            (tag, body) for tag, _, body in BimlParser.risks_from_text(text, ARA)
        )
        assert "carries on here" in risks["raw:3"]

    def test_the_first_definition_wins_over_a_reused_tag(self) -> None:
        """A later block reuses a tag to name the CONTROL for that risk,
        sometimes under a different descriptor."""
        text = (
            "[data:4:storage] The risk statement.\n"
            "Associated controls. Note that the labels refer to the risks:\n"
            "[data:4:disimilarity] The control statement.\n"
        )
        risks = BimlParser.risks_from_text(text, ARA)
        assert [(tag, label) for tag, label, _ in risks] == [
            ("data:4", "storage"),
        ]

    def test_text_with_no_definitional_tag_raises(self) -> None:
        with pytest.raises(ValueError, match="no definitional tag"):
            BimlParser.risks_from_text("Prose with no tag anywhere.\n", ARA)


class TestScopedIdentity:
    def _controls(self) -> list[object]:
        return list(BimlParser.build_controls(
            {ARA: ARA_TEXT, LLM24: LLM24_TEXT},
        )[0])

    def test_control_id_carries_the_document(self) -> None:
        controls = BimlParser.build_controls(
            {ARA: ARA_TEXT, LLM24: LLM24_TEXT},
        )[0]
        assert f"{ARA}: raw:3" in {c.control_id for c in controls}
        assert f"{LLM24}: raw:3" in {c.control_id for c in controls}

    def test_titles_cannot_collide_across_documents(self) -> None:
        controls = BimlParser.build_controls(
            {ARA: ARA_TEXT, LLM24: LLM24_TEXT},
        )[0]
        titles = [c.title for c in controls]
        assert len(titles) == len(set(titles))
        assert f"Hosting ({ARA}: inference:4)" in titles
        assert f"Hosting ({LLM24}: inference:9)" in titles

    def test_titles_cannot_collide_inside_one_document(self) -> None:
        """ara names two risks 'storage' and BIML-LLM24 names three 'data
        confidentiality', two of which carry curated links. The document alone
        does not separate those, so the tag is in the title too."""
        text = (
            "[raw:3:storage] The first statement about storage.\n"
            "[data:4:storage] The second statement about storage.\n"
        )
        controls = BimlParser.build_controls({ARA: text})[0]
        titles = [c.title for c in controls]
        assert len(titles) == len(set(titles)) == 2

    def test_no_title_is_the_bare_label(self) -> None:
        """A link's section_name is a bare label, so a bare label as the title
        would put every collided row through the title channel."""
        controls = BimlParser.build_controls(
            {ARA: ARA_TEXT, LLM24: LLM24_TEXT},
        )[0]
        assert "Hosting" not in {c.title for c in controls}
        assert "Storage" not in {c.title for c in controls}

    def test_an_unprefixed_id_is_an_alternate_on_exactly_one_document(
        self,
    ) -> None:
        controls = BimlParser.build_controls(
            {ARA: ARA_TEXT, LLM24: LLM24_TEXT},
        )[0]
        holders = [
            c.control_id for c in controls
            if c.metadata and "raw:3" in (c.metadata.get("alt_ids") or [])
        ]
        assert holders == [f"{ARA}: raw:3"]

    def test_the_named_conflict_resolves_by_name(self) -> None:
        controls = BimlParser.build_controls(
            {ARA: ARA_TEXT, LLM24: LLM24_TEXT},
        )[0]
        target = next(c for c in controls if c.control_id == f"{ARA}: output:1")
        assert target.metadata is not None
        assert "Direct Output" in target.metadata["alt_titles"]
        assert not any(
            "output:2" in (c.metadata or {}).get("alt_ids", [])
            for c in controls
        )

    def test_a_label_with_an_acronym_keeps_its_capitals(self) -> None:
        """str.title() would spell 'API encoding' as 'Api Encoding'."""
        controls = BimlParser.build_controls(
            {ARA: "[assembly:4:API encoding] A statement about encoding.\n"},
        )[0]
        assert controls[0].title.startswith("API Encoding")


class TestBodyCap:
    """Ruling R14: the parser owns its anchor, not a margin."""

    def _long(self, chars: int) -> str:
        line = "A sentence about one risk that fills the measure of a column."
        count = chars // (len(line) + 1) + 1
        return "[raw:3:storage] " + "\n".join([line] * count) + "\n"

    def test_a_body_under_the_limit_is_untouched(self) -> None:
        controls, audit = BimlParser.build_controls({ARA: ARA_TEXT})
        assert [r for r in audit if r["repair"] == "body_capped"] == []
        assert all(len(c.description) < MAX_BODY_CHARS for c in controls)

    def test_a_long_body_is_cut_below_the_description_limit(self) -> None:
        controls, _ = BimlParser.build_controls({ARA: self._long(4000)})
        assert len(controls[0].description) < DESCRIPTION_MAX_LENGTH

    def test_the_cut_is_audited_with_both_texts(self) -> None:
        """A count says a repair fired. It does not say what moved."""
        _, audit = BimlParser.build_controls({ARA: self._long(4000)})
        record = next(r for r in audit if r["repair"] == "body_capped")
        assert record["control_id"] == f"{ARA}: raw:3"
        before = record["before"]
        after = record["after"]
        assert isinstance(before, str) and isinstance(after, str)
        assert before.startswith(after)
        assert len(before) > len(after)

    def test_the_cut_lands_on_a_line_boundary(self) -> None:
        _, audit = BimlParser.build_controls({ARA: self._long(4000)})
        record = next(r for r in audit if r["repair"] == "body_capped")
        after = record["after"]
        assert isinstance(after, str)
        assert after.endswith("column.")

    def test_a_body_with_no_short_enough_line_raises(self) -> None:
        text = "[raw:3:storage] X" + "x" * (MAX_BODY_CHARS + 10) + "\n"
        with pytest.raises(ValueError, match="no line short enough"):
            BimlParser.build_controls({ARA: text})

    def test_the_base_class_never_rewrites_a_shipped_description(
        self, tmp_path: Path,
    ) -> None:
        """BaseParser._sanitize_control moves anything over the limit into
        full_text, which would hand ProseIndex an anchor the parser never
        chose."""
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / SOURCE_FILES[ARA]).write_bytes(_document(ARA_TAGS))
        (raw / SOURCE_FILES[LLM24]).write_bytes(_document(LLM24_TAGS))
        parser = BimlParser(
            raw_dir=raw, output_dir=tmp_path / "out",
            audit_dir=tmp_path / "audit",
        )
        parser.expected_sha256 = None  # type: ignore[misc]
        parser.expected_count = len(ARA_TAGS) + len(LLM24_TAGS)  # type: ignore[misc]
        parser.expected_tags = {  # type: ignore[misc]
            ARA: len(ARA_TAGS), LLM24: len(LLM24_TAGS),
        }
        (tmp_path / "out").mkdir()
        output = parser.run()
        assert all(c.full_text is None for c in output.controls)
        assert all(
            len(c.description) < DESCRIPTION_MAX_LENGTH
            for c in output.controls
        )


class TestAudit:
    def test_the_conflict_is_recorded(self, tmp_path: Path) -> None:
        parser = BimlParser(output_dir=tmp_path, audit_dir=tmp_path / "audit")
        _, audit = BimlParser.build_controls({ARA: ARA_TEXT, LLM24: LLM24_TEXT})
        parser.write_repair_audit(audit)
        records = [
            json.loads(line)
            for line in (tmp_path / "audit" / "biml.jsonl").read_text(
                encoding="utf-8",
            ).splitlines()
        ]
        conflict = next(
            r for r in records if r["repair"] == "name_conflict"
        )
        assert conflict["opencre_section_id"] == "output:2"
        assert conflict["resolved_to"] == f"{ARA}: output:1"
        assert conflict["resolved_by"] == "section_name"


class TestDeclarations:
    def test_an_alternate_whose_target_is_absent_is_refused(self) -> None:
        with pytest.raises(ValueError, match="alt_ids"):
            BimlParser.build_controls({ARA: ARA_TEXT}, require_targets=True)

    def test_a_name_conflict_whose_target_is_absent_is_refused(self) -> None:
        """Every alt_ids target present, the alt_titles target missing."""
        text = "\n".join(
            f"[{tag}:{label}] A statement about {label}."
            for tag, label in ARA_TAGS if tag != "output:1"
        ) + "\n"
        llm = "\n".join(
            f"[{tag}:{label}] A statement about {label}."
            for tag, label in LLM24_TAGS
        ) + "\n"
        with pytest.raises(ValueError, match="NAME_CONFLICTS"):
            BimlParser.build_controls(
                {ARA: text, LLM24: llm}, require_targets=True,
            )

    def test_declarations_are_not_checked_unless_asked(self) -> None:
        controls, _ = BimlParser.build_controls({ARA: ARA_TEXT})
        assert controls


class TestSyntheticPdf:
    """parse() through pdfplumber, across both documents, with no data/raw."""

    @pytest.fixture()
    def parser(self, tmp_path: Path) -> BimlParser:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / SOURCE_FILES[ARA]).write_bytes(_document(ARA_TAGS))
        (raw / SOURCE_FILES[LLM24]).write_bytes(_document(LLM24_TAGS))
        instance = BimlParser(
            raw_dir=raw,
            output_dir=tmp_path / "out",
            audit_dir=tmp_path / "audit",
        )
        instance.expected_sha256 = None  # type: ignore[misc]
        instance.expected_count = len(ARA_TAGS) + len(LLM24_TAGS)  # type: ignore[misc]
        instance.expected_tags = {  # type: ignore[misc]
            ARA: len(ARA_TAGS), LLM24: len(LLM24_TAGS),
        }
        return instance

    def test_both_documents_are_read(self, parser: BimlParser) -> None:
        controls = parser.parse()
        documents = [
            c.metadata["document"] for c in controls if c.metadata is not None
        ]
        assert documents.count(ARA) == len(ARA_TAGS)
        assert documents.count(LLM24) == len(LLM24_TAGS)

    def test_the_two_hosting_risks_keep_separate_titles(
        self, parser: BimlParser,
    ) -> None:
        titles = {c.title for c in parser.parse()}
        assert f"Hosting ({ARA}: inference:4)" in titles
        assert f"Hosting ({LLM24}: inference:9)" in titles

    def test_every_declared_unprefixed_id_lands(
        self, parser: BimlParser,
    ) -> None:
        holders: dict[str, str] = {}
        for control in parser.parse():
            for alternate in (control.metadata or {}).get("alt_ids", []) or []:
                holders[alternate] = control.control_id
        assert holders == {
            unprefixed: f"{document}: {tag}"
            for unprefixed, (document, tag) in UNPREFIXED_IDS.items()
        }

    def test_every_description_opens_a_sentence(
        self, parser: BimlParser,
    ) -> None:
        """The same invariant TestRun asserts against the real PDFs, held here
        so CI carries it. A wrapped cross-reference read as a definition hands
        its risk the remainder of somebody else's sentence, which opens with
        punctuation or a lowercase word."""
        wrong = {
            c.control_id: c.description[0] for c in parser.parse()
            if not (c.description[0].isupper() or c.description[0] in ("“", '"'))
        }
        assert wrong == {}

    def test_a_document_short_of_its_census_is_refused(
        self, parser: BimlParser,
    ) -> None:
        """expected_count covers the SUM, so 78 + 68 also passes as 80 + 66.
        The per-document census is what beats that."""
        parser.expected_tags = {  # type: ignore[misc]
            ARA: len(ARA_TAGS) + 1, LLM24: len(LLM24_TAGS),
        }
        with pytest.raises(ValueError, match=re.escape(ARA)):
            parser.parse()

    def test_a_document_over_its_census_is_refused(
        self, parser: BimlParser,
    ) -> None:
        """The census is exact, not a floor: the bytes are pinned by sha256, so
        a surplus is the parser changing rather than the publisher."""
        parser.expected_tags = {  # type: ignore[misc]
            ARA: len(ARA_TAGS), LLM24: len(LLM24_TAGS) - 1,
        }
        with pytest.raises(ValueError, match=re.escape(LLM24)):
            parser.parse()

    def test_a_digest_that_does_not_match_is_refused(
        self, parser: BimlParser,
    ) -> None:
        parser.expected_sha256 = {  # type: ignore[misc]
            ARA: "0" * 64, LLM24: "0" * 64,
        }
        with pytest.raises(ValueError, match="not the pinned"):
            parser.parse()

    def test_a_hyphen_wrapped_word_is_rejoined(self, tmp_path: Path) -> None:
        """121 words in BIML-LLM24 and 7 in ara arrive split across a hyphen."""
        raw = tmp_path / "raw"
        raw.mkdir()
        lines: list[str] = []
        for tag, label in ARA_TAGS:
            lines.append(f"[{tag}:{label}] A statement about an unreviewed configu-")
            lines.append(f"ration of whatever holds {label}.")
        (raw / SOURCE_FILES[ARA]).write_bytes(_pdf(lines))
        (raw / SOURCE_FILES[LLM24]).write_bytes(_document(LLM24_TAGS))
        parser = BimlParser(
            raw_dir=raw, output_dir=tmp_path / "out",
            audit_dir=tmp_path / "audit",
        )
        parser.expected_sha256 = None  # type: ignore[misc]
        parser.expected_count = len(ARA_TAGS) + len(LLM24_TAGS)  # type: ignore[misc]
        parser.expected_tags = {  # type: ignore[misc]
            ARA: len(ARA_TAGS), LLM24: len(LLM24_TAGS),
        }
        (tmp_path / "out").mkdir()
        target = next(
            c for c in parser.run().controls if c.control_id == f"{ARA}: raw:3"
        )
        assert "configuration of whatever" in target.description

    def test_run_writes_and_writes_the_audit(
        self, parser: BimlParser, tmp_path: Path,
    ) -> None:
        (tmp_path / "out").mkdir()
        output = parser.run()
        assert len(output.controls) == len(ARA_TAGS) + len(LLM24_TAGS)
        assert sorted(s.path for s in output.source_files) == sorted(
            SOURCE_FILES.values()
        )
        lines = (tmp_path / "audit" / "biml.jsonl").read_text(
            encoding="utf-8",
        ).splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["repair"] == "name_conflict"


class TestPins:
    def test_the_class_ships_a_real_digest_pin(self) -> None:
        """Without this the class could ship expected_sha256 = None, which is
        what the synthetic fixture sets on an instance."""
        assert BimlParser.expected_sha256 is not None
        assert set(BimlParser.expected_sha256) == set(SOURCE_FILES)
        assert all(
            len(digest) == 64 for digest in BimlParser.expected_sha256.values()
        )

    def test_the_census_covers_both_documents(self) -> None:
        assert set(BimlParser.expected_tags) == set(SOURCE_FILES)
        assert sum(BimlParser.expected_tags.values()) == BimlParser.expected_count


class TestRun:
    """The real PDFs. data/raw is gitignored, so this skips in CI."""

    @pytest.fixture(scope="class")
    def controls(self) -> list[object]:
        parser = BimlParser()
        try:
            return list(parser.parse())
        except FileNotFoundError:
            pytest.skip("data/raw is gitignored and absent in this checkout")

    def test_the_census_holds(self, controls: list[object]) -> None:
        documents = [
            c.metadata["document"]  # type: ignore[attr-defined]
            for c in controls
        ]
        assert len(controls) == 146
        assert documents.count(ARA) == 78
        assert documents.count(LLM24) == 68

    def test_every_description_opens_a_sentence(
        self, controls: list[object],
    ) -> None:
        """A wrapped cross-reference read as a definition gives its risk the
        remainder of somebody else's sentence, which starts with punctuation or
        a lowercase word. ara's data:2 carries a curated link and did exactly
        that."""
        openers = {
            c.control_id: c.description[0]  # type: ignore[attr-defined]
            for c in controls
        }
        wrong = {
            k: v for k, v in openers.items()
            if not (v.isupper() or v in ("“", '"'))
        }
        assert wrong == {}

    def test_no_description_carries_the_controls_heading(
        self, controls: list[object],
    ) -> None:
        """Without the heading terminator, 11 bodies swallow it and the generic
        control text behind it."""
        carriers = [
            c.control_id for c in controls  # type: ignore[attr-defined]
            if "Associated controls" in c.description  # type: ignore[attr-defined]
        ]
        assert carriers == []

    def test_no_description_reaches_the_description_limit(
        self, controls: list[object],
    ) -> None:
        """Ruling R14. Without the terminators the longest body is 39,093
        characters and seven of them cross the limit."""
        longest = max(
            len(c.description) for c in controls  # type: ignore[attr-defined]
        )
        assert longest < DESCRIPTION_MAX_LENGTH

    def test_every_declared_target_exists(self, controls: list[object]) -> None:
        ids = {c.control_id for c in controls}  # type: ignore[attr-defined]
        declared = {
            f"{document}: {tag}" for document, tag in UNPREFIXED_IDS.values()
        }
        assert declared <= ids
        assert f"{ARA}: output:1" in ids
