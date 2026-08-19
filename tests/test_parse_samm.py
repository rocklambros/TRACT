"""SAMM joins at the stream, and the statement must not overflow the budget.

Measured on the pinned archive: the stream description plus its three
activities' `longDescription` runs 2,548 to 6,678 characters, so all 30
descriptions exceed DESCRIPTION_MAX_LENGTH, `_sanitize_control` moves the
overflow into `full_text`, and ProseIndex prefers `full_text` over
`description` unconditionally. The anchor is then a text `prepare_anchor` cuts
at 2,150 characters, on a framework that has no truncation today.
`shortDescription` keeps every statement inside both budgets: 347 to 986.

TestTheTrapIsReal is the positive control for that whole paragraph. Without it
`test_full_text_is_left_unset` asserts a property no plausible wrong
implementation could violate, because nothing else here proves that a long
description reaches `full_text` at all.

The order assertions carry two deliberate fixture choices. The zip members are
written in an order that is neither alphabetical nor level-ascending, and each
activity's `level` GUID sorts in the reverse of its maturity level. A fixture
without both cannot tell a parser that reads the filename level from one that
trusts `namelist()` order or sorts on the GUID field.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from parsers.parse_samm import (
    ARCHIVE_NAME,
    EXPECTED_LEVELS,
    OPENCRE_TITLE_VARIANTS,
    SammParser,
)
from tract.config import (
    DESCRIPTION_MAX_LENGTH,
    HONEST_PROSE_MIN_CHARS,
    MAX_ANCHOR_CHARS,
    PROCESSED_FRAMEWORKS_DIR,
)
from tract.corpus_report import (
    CURATED_LINKS_PATH,
    JOIN_FLOORS,
    SYNTHETIC_TEXT_ORIGIN,
    TEXT_ORIGIN_METADATA_KEY,
    build_corpus_report,
    check_join_floors,
)
from tract.parsers.base import BaseParser

# The practice GUID both fixture streams point at. Opaque upstream, carried
# through as parent_id because it is the source's own foreign key.
PRACTICE = "4753e55e943c4d418303bf90d599c6b1"

STREAMS: dict[str, dict[str, Any]] = {
    "D-SA-A": {
        "practice": PRACTICE,
        "id": "253b012094cf4e0988e08fd22609227d",
        "name": "Architecture Design",
        "letter": "A",
        "description": "The design of a software architecture significantly "
                       "affects the security posture of the software.",
        "order": 1,
        "type": "Stream",
    },
    "D-SA-B": {
        "practice": PRACTICE,
        "id": "b6b1ba4f8e1c4b7e9e3f0d2a5c7e9011",
        "name": "Technology Management",
        "letter": "B",
        "description": "The technologies a team standardises on decide which "
                       "classes of defect are possible at all.",
        "order": 2,
        "type": "Stream",
    },
}

# Descending, so that sorting activities on the `level` FIELD reverses the
# maturity order the filename states. The source-structures document calls
# `level` an integer. It is a GUID. [measured]
LEVEL_GUIDS: dict[int, str] = {
    1: "cccccccc7dec4cfdad983cf6d1d17b61",
    2: "bbbbbbbb7dec4cfdad983cf6d1d17b61",
    3: "aaaaaaaa7dec4cfdad983cf6d1d17b61",
}


def _activity(stem: str, level: int) -> dict[str, Any]:
    """One activity, with a long description a correct parser never reads."""
    return {
        "stream": STREAMS[stem]["id"],
        "level": LEVEL_GUIDS[level],
        "title": f"{stem} level {level}",
        "shortDescription": f"Teams apply {stem} level {level} design "
                            f"practices during architecture review sessions.",
        "longDescription": "x" * 3000,
        "type": "Activity",
    }


def _statement(stem: str, levels: tuple[int, ...] = EXPECTED_LEVELS) -> str:
    """The statement a correct parser builds for one fixture stream."""
    parts = [str(STREAMS[stem]["description"])]
    parts += [str(_activity(stem, level)["shortDescription"]) for level in levels]
    return "\n\n".join(parts)


def _archive(members: list[tuple[str, Any]]) -> bytes:
    """A zip carrying the given members, in the order given.

    `sort_keys=False` is load-bearing for the same reason it is in the DSOMM
    fixture: safe_dump alphabetises by default, and a fixture that re-sorts on
    the way in cannot see a parser that sorts on the way out.
    """
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as handle:
        for name, body in members:
            handle.writestr(name, yaml.safe_dump(body, sort_keys=False))
    return payload.getvalue()


# Neither alphabetical nor level-ascending, and the two streams interleave. A
# parser that trusts namelist() insertion order builds D-SA-A's statement in
# the order 3, 2, 1 and trips the maturity-level check.
SCRAMBLED: list[tuple[str, Any]] = [
    ("core-abc/model/activities/D-SA-3-A.yml", _activity("D-SA-A", 3)),
    ("core-abc/model/streams/D-SA-B.yml", STREAMS["D-SA-B"]),
    ("core-abc/model/activities/D-SA-1-B.yml", _activity("D-SA-B", 1)),
    ("core-abc/model/activities/D-SA-2-A.yml", _activity("D-SA-A", 2)),
    ("core-abc/model/streams/D-SA-A.yml", STREAMS["D-SA-A"]),
    ("core-abc/model/activities/D-SA-3-B.yml", _activity("D-SA-B", 3)),
    ("core-abc/model/activities/D-SA-1-A.yml", _activity("D-SA-A", 1)),
    ("core-abc/model/activities/D-SA-2-B.yml", _activity("D-SA-B", 2)),
]


def _parser_for(
    tmp_path: Path,
    members: list[tuple[str, Any]],
    expected_count: int = 2,
    name: str = "raw",
) -> SammParser:
    raw = tmp_path / name
    raw.mkdir(exist_ok=True)
    (raw / ARCHIVE_NAME).write_bytes(_archive(members))

    instance = SammParser(raw_dir=raw, output_dir=tmp_path / "out")
    instance.expected_count = expected_count  # type: ignore[misc]
    # A fixture archive is not the pinned one, so the real digest is stood
    # down here rather than the gate being widened to accept two archives.
    instance.expected_sha256 = None  # type: ignore[misc]
    # The real variant table names streams no fixture carries. Stood down for
    # the same reason, and TestTitleVariants puts it back one entry at a time.
    instance.title_variants = {}  # type: ignore[misc]
    return instance


@pytest.fixture()
def parser(tmp_path: Path) -> SammParser:
    return _parser_for(tmp_path, SCRAMBLED)


class TestParse:
    def test_control_id_is_the_stream_filename_stem(
        self, parser: SammParser,
    ) -> None:
        """The stem is what every curated section_id spells. [measured 30/30]"""
        assert [c.control_id for c in parser.parse()] == ["D-SA-A", "D-SA-B"]

    def test_title_is_the_streams_own_name(self, parser: SammParser) -> None:
        assert [c.title for c in parser.parse()] == [
            "Architecture Design", "Technology Management",
        ]

    def test_statement_carries_all_three_maturity_levels(
        self, parser: SammParser,
    ) -> None:
        text = parser.parse()[0].description
        for level in EXPECTED_LEVELS:
            assert f"D-SA-A level {level} design practices" in text

    def test_statement_is_the_description_then_the_levels_in_order(
        self, parser: SammParser,
    ) -> None:
        """Containment alone passes on any permutation, including reversed.

        The expected order is spelled out in _statement rather than read back
        from the parser, so a swapped-order implementation has nothing to
        agree with.
        """
        assert parser.parse()[0].description == _statement("D-SA-A")

    def test_a_streams_statement_holds_no_other_streams_text(
        self, parser: SammParser,
    ) -> None:
        """D-SA-A and D-SA-B share a practice code and differ only by letter.

        Keying activities on the practice code alone merges the two streams'
        six activities into one bucket.
        """
        first, second = parser.parse()
        assert "D-SA-B" not in first.description
        assert "D-SA-A" not in second.description
        assert second.description == _statement("D-SA-B")

    def test_statement_uses_short_not_long_descriptions(
        self, parser: SammParser,
    ) -> None:
        for control in parser.parse():
            assert "x" * 100 not in control.description
            assert len(control.description) <= DESCRIPTION_MAX_LENGTH
            assert len(control.description) <= MAX_ANCHOR_CHARS
            assert len(control.description) >= HONEST_PROSE_MIN_CHARS

    def test_full_text_is_left_unset_so_the_anchor_is_the_statement(
        self, parser: SammParser,
    ) -> None:
        assert [c.full_text for c in parser.parse()] == [None, None]

    def test_the_practice_and_letter_are_recorded(
        self, parser: SammParser,
    ) -> None:
        first = parser.parse()[0]
        assert first.parent_id == PRACTICE
        assert first.hierarchy_level == "stream"
        assert first.metadata is not None
        assert first.metadata["stream_letter"] == "A"

    def test_build_controls_orders_activities_it_is_handed_unsorted(
        self,
    ) -> None:
        """The declared interface, called directly and handed a wrong order.

        Nothing routed through parse() can see this: _read_members walks
        sorted(namelist()), so an activity list arrives level-ascending
        whatever order the zip was written in, and dropping the sort here
        survives every test built on the fixture archive. build_controls is a
        public classmethod, so its contract is what a caller may hand it.
        """
        streams = {"D-SA-A": STREAMS["D-SA-A"]}
        activities = {
            "D-SA-A": [
                (level, _activity("D-SA-A", level)) for level in (3, 1, 2)
            ],
        }
        control = SammParser.build_controls(streams, activities, {})[0]
        assert control.description == _statement("D-SA-A")

    def test_the_assembled_statement_is_marked_synthetic(
        self, parser: SammParser,
    ) -> None:
        """Four source records, joined in an order this parser chose.

        The corpus report separates parser-written anchors from
        publisher-written ones on this key alone, so an unmarked statement is
        counted as text a publisher wrote as one paragraph.
        """
        for control in parser.parse():
            assert control.metadata is not None
            assert control.metadata[TEXT_ORIGIN_METADATA_KEY] == (
                SYNTHETIC_TEXT_ORIGIN
            )


class TestTheTrapIsReal:
    """The positive control for the whole composition decision.

    Every other assertion about `full_text` is worth nothing unless a long
    description really does reach it behind the parser's back.
    """

    def test_a_long_description_is_moved_into_full_text_by_run(
        self, parser: SammParser, tmp_path: Path,
    ) -> None:
        long_statement = "The architecture is reviewed. " * 100
        assert len(long_statement) > DESCRIPTION_MAX_LENGTH
        control = parser.parse()[0].model_copy(
            update={"description": long_statement},
        )
        sanitized = parser._sanitize_control(control)
        assert sanitized.full_text is not None
        assert len(sanitized.description) <= DESCRIPTION_MAX_LENGTH

    def test_the_shipped_statement_stays_under_both_budgets(
        self, parser: SammParser, tmp_path: Path,
    ) -> None:
        (tmp_path / "out").mkdir()
        for control in parser.run().controls:
            assert control.full_text is None
            assert len(control.description) <= DESCRIPTION_MAX_LENGTH
            assert len(control.description) <= MAX_ANCHOR_CHARS


class TestRefusals:
    """Every one of these is a silent short statement if it does not raise."""

    def test_a_stream_with_no_activities_is_refused(
        self, tmp_path: Path,
    ) -> None:
        instance = _parser_for(
            tmp_path,
            [("core-abc/model/streams/D-SA-A.yml", STREAMS["D-SA-A"])],
            expected_count=1,
        )
        with pytest.raises(ValueError, match="no activities"):
            instance.parse()

    def test_a_stream_missing_a_maturity_level_is_refused(
        self, tmp_path: Path,
    ) -> None:
        instance = _parser_for(
            tmp_path,
            [
                ("core-abc/model/streams/D-SA-A.yml", STREAMS["D-SA-A"]),
                ("core-abc/model/activities/D-SA-1-A.yml",
                 _activity("D-SA-A", 1)),
                ("core-abc/model/activities/D-SA-3-A.yml",
                 _activity("D-SA-A", 3)),
            ],
            expected_count=1,
        )
        with pytest.raises(ValueError, match="maturity levels"):
            instance.parse()

    def test_an_archive_with_no_stream_members_is_refused(
        self, tmp_path: Path,
    ) -> None:
        """Activity stems match no section_id, so there is no join without them."""
        instance = _parser_for(
            tmp_path,
            [("core-abc/model/activities/D-SA-1-A.yml",
              _activity("D-SA-A", 1))],
            expected_count=1,
        )
        with pytest.raises(ValueError, match="no model/streams"):
            instance.parse()

    def test_a_stream_without_a_name_is_refused(self, tmp_path: Path) -> None:
        """The name is the title channel's key for 27 of the 30 links."""
        nameless = {**STREAMS["D-SA-A"], "name": "  "}
        members = [("core-abc/model/streams/D-SA-A.yml", nameless)]
        members += [
            (f"core-abc/model/activities/D-SA-{level}-A.yml",
             _activity("D-SA-A", level))
            for level in EXPECTED_LEVELS
        ]
        instance = _parser_for(tmp_path, members, expected_count=1)
        with pytest.raises(ValueError, match="has no name"):
            instance.parse()

    def test_a_stream_without_a_description_is_refused(
        self, tmp_path: Path,
    ) -> None:
        """All 30 carry one, 110 characters at the shortest. [measured]"""
        bare = {**STREAMS["D-SA-A"], "description": ""}
        members = [("core-abc/model/streams/D-SA-A.yml", bare)]
        members += [
            (f"core-abc/model/activities/D-SA-{level}-A.yml",
             _activity("D-SA-A", level))
            for level in EXPECTED_LEVELS
        ]
        instance = _parser_for(tmp_path, members, expected_count=1)
        with pytest.raises(ValueError, match="has no description"):
            instance.parse()

    def test_an_activity_without_a_short_description_is_refused(
        self, tmp_path: Path,
    ) -> None:
        """All 90 carry one, 30 characters at the shortest. [measured]

        Skipping it silently would leave the statement short by a third with
        nothing in any column to say so.
        """
        members = [("core-abc/model/streams/D-SA-A.yml", STREAMS["D-SA-A"])]
        for level in EXPECTED_LEVELS:
            body = _activity("D-SA-A", level)
            if level == 2:
                body["shortDescription"] = "   "
            members.append(
                (f"core-abc/model/activities/D-SA-{level}-A.yml", body)
            )
        instance = _parser_for(tmp_path, members, expected_count=1)
        with pytest.raises(ValueError, match="has no shortDescription"):
            instance.parse()

    def test_a_member_that_is_not_a_mapping_is_refused(
        self, tmp_path: Path,
    ) -> None:
        members = [
            ("core-abc/model/streams/D-SA-A.yml", ["not", "a", "mapping"]),
        ]
        instance = _parser_for(tmp_path, members, expected_count=1)
        with pytest.raises(ValueError, match="where a mapping was expected"):
            instance.parse()


class TestTitleVariants:
    """OpenCRE spells three of the 30 stream names differently. [measured]

    V-AA-A "Achitecture validation", V-AA-B "Achitecture mitigation" and
    G-PC-A "Policy & Standards". Two are misspellings and one is an ampersand.
    Declared as alternates so all 30 links resolve through the title channel
    the curator wrote, rather than 27 by title and 3 by id with the id-side
    wrong-anchor detector flagging the name disagreement it cannot resolve.
    """

    def test_a_declared_variant_reaches_the_controls_alt_titles(
        self, tmp_path: Path,
    ) -> None:
        instance = _parser_for(tmp_path, SCRAMBLED)
        instance.title_variants = {  # type: ignore[misc]
            "D-SA-A": ("Achitecture design",),
        }
        first, second = instance.parse()
        assert first.metadata is not None
        assert first.metadata["alt_titles"] == ["Achitecture design"]
        assert second.metadata is not None
        assert "alt_titles" not in second.metadata

    def test_a_variant_naming_an_absent_stream_is_refused(
        self, tmp_path: Path,
    ) -> None:
        """A stale entry does nothing at all, so it has to be loud."""
        instance = _parser_for(tmp_path, SCRAMBLED)
        instance.title_variants = {"Z-ZZ-A": ("Gone upstream",)}  # type: ignore[misc]
        with pytest.raises(ValueError, match="names no stream"):
            instance.parse()

    def test_a_variant_equal_to_the_real_name_is_refused(
        self, tmp_path: Path,
    ) -> None:
        """ProseIndex never lets an alternate displace a real title.

        A variant that restates the name is therefore dead on arrival, and a
        dead entry reads as a live one in this table.
        """
        instance = _parser_for(tmp_path, SCRAMBLED)
        instance.title_variants = {  # type: ignore[misc]
            "D-SA-A": ("architecture design",),
        }
        with pytest.raises(ValueError, match="already the stream's own name"):
            instance.parse()

    def test_an_empty_variant_is_refused(self, tmp_path: Path) -> None:
        instance = _parser_for(tmp_path, SCRAMBLED)
        instance.title_variants = {"D-SA-A": ("  ",)}  # type: ignore[misc]
        with pytest.raises(ValueError, match="declares an empty"):
            instance.parse()

    def test_the_declared_table_is_exactly_what_the_link_file_misspells(
        self,
    ) -> None:
        """Fails in both directions, which is the point of deriving it.

        A link name that stops matching its stream's title and is not declared
        fails here, and a declared variant no link spells fails here too. Both
        sides read tracked files, so this runs in a checkout with no data/raw.
        """
        titles = {
            control["control_id"]: control["title"]
            for control in json.loads(
                (PROCESSED_FRAMEWORKS_DIR / "samm.json").read_text(
                    encoding="utf-8",
                )
            )["controls"]
        }
        assert len(titles) == 30, titles

        observed: dict[str, list[str]] = {}
        with CURATED_LINKS_PATH.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("framework_id") != "samm":
                    continue
                section_id = str(row.get("section_id") or "")
                name = str(row.get("section_name") or "").strip()
                title = titles[section_id]
                if name.lower() != title.lower():
                    observed.setdefault(section_id, []).append(name)

        assert observed == {
            stem: list(names) for stem, names in OPENCRE_TITLE_VARIANTS.items()
        }


class TestRun:
    def test_run_writes_and_clears_the_prose_floor(
        self, parser: SammParser, tmp_path: Path,
    ) -> None:
        (tmp_path / "out").mkdir()
        output = parser.run()
        assert len(output.controls) == 2
        assert (tmp_path / "out" / "samm.json").exists()
        assert BaseParser.honest_prose_fraction(output.controls) == 1.0
        assert [s.path for s in output.source_files] == [ARCHIVE_NAME]

    def test_a_title_length_statement_trips_the_prose_floor(
        self, tmp_path: Path,
    ) -> None:
        """Nothing else here proves the floor is set to a non-zero value."""
        stub = {**STREAMS["D-SA-A"], "description": "Design."}
        members: list[tuple[str, Any]] = [
            ("core-abc/model/streams/D-SA-A.yml", stub),
        ]
        for level in EXPECTED_LEVELS:
            body = _activity("D-SA-A", level)
            body["shortDescription"] = f"Do {level}."
            members.append(
                (f"core-abc/model/activities/D-SA-{level}-A.yml", body)
            )
        instance = _parser_for(tmp_path, members, expected_count=1)
        (tmp_path / "out").mkdir()
        with pytest.raises(ValueError, match="below the declared floor"):
            instance.run()

    def test_reparse_is_byte_identical(
        self, parser: SammParser, tmp_path: Path,
    ) -> None:
        (tmp_path / "out").mkdir()
        parser.run()
        first = (tmp_path / "out" / "samm.json").read_bytes()
        parser.run()
        assert (tmp_path / "out" / "samm.json").read_bytes() == first


class TestDigestGate:
    def test_a_different_archive_is_refused(self, parser: SammParser) -> None:
        parser.expected_sha256 = "0" * 64  # type: ignore[misc]
        with pytest.raises(ValueError, match="not the pinned"):
            parser.parse()

    def test_the_shipped_pin_is_the_one_the_fetcher_downloads(self) -> None:
        """Without this the class could ship `expected_sha256 = None`.

        Every other test here stands the gate down on purpose, so none of them
        can see a parser with no pin at all.
        """
        from scripts.fetch_frameworks import SOURCES

        pins = {
            source.expected_sha256
            for source in SOURCES
            if source.framework_id == "samm"
        }
        assert pins == {SammParser.expected_sha256}


class TestRealCorpus:
    """The artifact this parser ships, and the join it buys.

    Reads only tracked files, so it holds in a checkout with no data/raw and
    no licensed overlay.
    """

    def test_the_shipped_artifact_is_thirty_streams_of_prose(self) -> None:
        data = json.loads(
            (PROCESSED_FRAMEWORKS_DIR / "samm.json").read_text(encoding="utf-8")
        )
        controls = data["controls"]
        assert data["framework_id"] == "samm"
        assert data["mapping_unit_level"] == "stream"
        assert len(controls) == 30
        assert [c["control_id"] for c in controls] == sorted(
            c["control_id"] for c in controls
        )
        lengths = [len(c["description"]) for c in controls]
        # 347..986 on the pinned archive. The bounds asserted here are the
        # budgets the composition was chosen to clear, not the run's own
        # numbers, so re-measuring does not require editing this test.
        assert min(lengths) >= HONEST_PROSE_MIN_CHARS, min(lengths)
        assert max(lengths) <= MAX_ANCHOR_CHARS, max(lengths)
        assert max(lengths) <= DESCRIPTION_MAX_LENGTH, max(lengths)
        for control in controls:
            assert not control.get("full_text")
            assert control["description"] != control["title"]
            assert control["metadata"][TEXT_ORIGIN_METADATA_KEY] == (
                SYNTHETIC_TEXT_ORIGIN
            )

    def test_every_curated_link_resolves_to_its_own_stream(self) -> None:
        row = build_corpus_report().by_id("samm")
        assert row.links == 30
        assert row.unresolved == 0
        assert row.resolution_rate == pytest.approx(1.0)
        assert row.by_title == 30, "the three misspelled names need alternates"
        assert row.by_id == 0
        assert row.distinct_anchors == 30, "30 links must not collapse"
        assert row.distinct_anchors_pre_truncation == 30
        assert row.truncated == 0
        assert row.nested_anchors == 0
        assert row.dropped_by_prose_rule == 0
        assert row.wrong_anchor_risk == 0
        assert row.anchor_source_synthetic == 30
        assert check_join_floors(
            build_corpus_report(), {"samm": JOIN_FLOORS["samm"]},
        ) == []
