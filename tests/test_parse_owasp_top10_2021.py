"""Ten categories, and neither A00 nor A11 is one of them.

The fixture carries all ten because parse() refuses a short list, which is the
whole reason the completeness check exists. TestGuards removes one to prove the
refusal fires.

Two fixture members exist only to catch a member filter that reads the
filename instead of the path. The pinned archive holds twelve files named
`A01_2021-Broken_Access_Control.md`, one per translation, and a filter that
matched on the stem alone would ingest Arabic. [measured: 12 of them]

TestTheClobberIsReal is the positive control for the docstring paragraph about
`_sanitize_control`. Without it, `test_full_text_carries_the_whole_entry`
asserts a property that holds on the fixture and fails on two of the ten real
categories, which is exactly the gap the brief for this task shipped.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from parsers.parse_owasp_top10_2021 import (
    ARCHIVE_NAME,
    CATEGORY_IDS,
    OPENCRE_TITLE_VARIANTS,
    OwaspTop102021Parser,
)
from tract.config import (
    DESCRIPTION_MAX_LENGTH,
    MAX_ANCHOR_CHARS,
    PROCESSED_FRAMEWORKS_DIR,
)
from tract.corpus_report import CURATED_LINKS_PATH, build_corpus_report
from tract.parsers.base import BaseParser

TITLES: dict[str, str] = {
    "A01": "Broken Access Control",
    "A02": "Cryptographic Failures",
    "A03": "Injection",
    "A04": "Insecure Design",
    "A05": "Security Misconfiguration",
    "A06": "Vulnerable and Outdated Components",
    "A07": "Identification and Authentication Failures",
    "A08": "Software and Data Integrity Failures",
    "A09": "Security Logging and Monitoring Failures",
    "A10": "Server-Side Request Forgery (SSRF)",
}

# Section order matches the source: Factors, Overview, Description, then the
# two remediation headings. A parser that takes the first section, the last
# section, or everything after the Overview reads differently here.
BODY = """
## Factors

| CWEs Mapped | Max Incidence Rate |
|---|---|
| 34 | 55.97% |

## Overview

Moving up from the fifth position, 94% of applications were tested for some
form of this weakness.

## Description

{title} covers the case where a system does not enforce the policy it claims
to enforce, and failures typically lead to unauthorized information
disclosure, modification, or destruction of data.

## How to Prevent

Enforcement is only effective in trusted server-side code.

## References

- OWASP Proactive Controls
"""

META = """# How to start an AppSec Program with the OWASP Top 10

Previously, the OWASP Top 10 was never designed to be the basis of anything.

## Description

A program is not a category.
"""

NEXT_STEPS = """# A11:2021 – Next Steps

The Top 10 is not the end of the journey.

## Description

There is more to application security than ten risks.
"""

# French keeps the English filename upstream, so this member differs from its
# English sibling only in path. Its title is the string that must never appear.
FRENCH_TITLE = "Controle d Acces Defaillant"


def _category(code: str, title: str) -> str:
    icon = '![icon](assets/i.png){: style="height:80px" align="right"}'
    return f"# {code}:2021 – {title}    {icon}\n" + BODY.format(title=title)


def _archive(codes: tuple[str, ...]) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        for code in codes:
            archive.writestr(
                f"Top10-abc/2021/docs/en/{code}_2021-Entry.md",
                _category(code, TITLES[code]),
            )
        archive.writestr("Top10-abc/2021/docs/en/A00_2021-How_to_start.md", META)
        archive.writestr("Top10-abc/2021/docs/en/A11_2021-Next_Steps.md", NEXT_STEPS)
        # Same stem as an English member, different language directory.
        archive.writestr(
            "Top10-abc/2021/docs/fr/A01_2021-Entry.md",
            _category("A01", FRENCH_TITLE),
        )
        # Same tree shape, earlier edition.
        archive.writestr(
            "Top10-abc/2017/docs/en/A01_2017-Injection.md",
            _category("A01", "Injection 2017"),
        )
    return payload.getvalue()


def _parser_for(tmp_path: Path, codes: tuple[str, ...], name: str) -> (
    OwaspTop102021Parser
):
    raw = tmp_path / name
    raw.mkdir(exist_ok=True)
    (raw / ARCHIVE_NAME).write_bytes(_archive(codes))
    instance = OwaspTop102021Parser(raw_dir=raw, output_dir=tmp_path / "out")
    # A fixture archive is not the pinned one, so the real digest is stood
    # down here rather than the gate being widened to accept two archives.
    instance.expected_sha256 = None  # type: ignore[misc]
    return instance


@pytest.fixture()
def parser(tmp_path: Path) -> OwaspTop102021Parser:
    return _parser_for(tmp_path, tuple(TITLES), "raw")


class TestParse:
    def test_only_the_ten_english_2021_categories_are_read(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        assert [c.control_id for c in parser.parse()] == list(CATEGORY_IDS)

    def test_a_translation_sharing_the_filename_is_not_read(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        """A stem-only member filter ingests twelve translations. [measured]"""
        titles = [c.title for c in parser.parse()]
        assert FRENCH_TITLE not in titles
        assert titles.count(TITLES["A01"]) == 1

    def test_an_earlier_edition_is_not_read(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        assert "Injection 2017" not in [c.title for c in parser.parse()]

    def test_title_drops_the_code_the_en_dash_and_the_icon(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        assert parser.parse()[0].title == "Broken Access Control"

    def test_description_is_the_description_section(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        text = parser.parse()[0].description
        assert text.startswith("Broken Access Control covers the case")
        assert "Moving up from the fifth position" not in text
        assert "trusted server-side code" not in text
        assert "CWEs Mapped" not in text

    def test_full_text_carries_the_whole_entry_below_the_heading(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        control = parser.parse()[0]
        assert control.full_text is not None
        assert control.full_text.startswith("## Factors")
        assert "Moving up from the fifth position" in control.full_text
        assert "trusted server-side code" in control.full_text
        # The heading line is consumed, not carried.
        assert "A01:2021" not in control.full_text

    def test_only_the_declared_categories_carry_alt_titles(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        declared = {
            c.control_id: (c.metadata or {}).get("alt_titles")
            for c in parser.parse()
        }
        assert {k: v for k, v in declared.items() if v} == {
            key: list(names) for key, names in OPENCRE_TITLE_VARIANTS.items()
        }


class TestGuards:
    def test_a_short_list_is_refused(self, tmp_path: Path) -> None:
        """The band would accept 9 of 10. The exact tuple does not.

        COUNT_TOLERANCE is 0.10 and abs(9 - 10) / 10 is 0.1, so
        _check_expected_count would pass a parser that lost a category.
        """
        codes = tuple(c for c in TITLES if c != "A07")
        short = _parser_for(tmp_path, codes, "short")
        with pytest.raises(ValueError, match="expected categories"):
            short.parse()

    def test_a_missing_description_section_is_refused(
        self, tmp_path: Path,
    ) -> None:
        raw = tmp_path / "broken"
        raw.mkdir()
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr(
                "Top10-abc/2021/docs/en/A01_2021-X.md",
                "# A01:2021 – X\n\n## Overview\n\nNo body.\n",
            )
        (raw / ARCHIVE_NAME).write_bytes(payload.getvalue())
        broken = OwaspTop102021Parser(raw_dir=raw, output_dir=tmp_path / "out")
        broken.expected_sha256 = None  # type: ignore[misc]
        with pytest.raises(ValueError, match="no '## Description' section"):
            broken.parse()

    def test_markdown_with_no_category_heading_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no 'A0N:2021"):
            OwaspTop102021Parser.control_from_markdown("# Next Steps\n\nText.\n")

    def test_a_different_archive_is_refused(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        parser.expected_sha256 = "0" * 64  # type: ignore[misc]
        with pytest.raises(ValueError, match="not the pinned"):
            parser.parse()

    def test_the_shipped_pin_is_the_one_the_fetcher_downloads(self) -> None:
        """Every other test stands the gate down, so none can see no pin."""
        from scripts.fetch_frameworks import SOURCES

        pins = {
            source.expected_sha256
            for source in SOURCES
            if source.framework_id == "owasp_top10_2021"
        }
        assert pins == {OwaspTop102021Parser.expected_sha256}


class TestTitleVariants:
    """OpenCRE spells three of the ten category names differently. [measured]

    A01 "Broken Access Controls" is plural, A09 "Logging and Monitoring
    Failures" drops the Security prefix, and A10 "Server Side Request Forgery
    (SSRF)" drops the hyphen. Declared as alternates so all 17 links resolve
    through the title channel the curator wrote.
    """

    def test_a_variant_naming_an_absent_category_is_refused(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        """A stale entry does nothing at all, so it has to be loud."""
        parser.title_variants = {"A99": ("Gone upstream",)}  # type: ignore[misc]
        with pytest.raises(ValueError, match="names no category"):
            parser.parse()

    def test_a_variant_equal_to_the_real_title_is_refused(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        """ProseIndex never lets an alternate displace a real title."""
        parser.title_variants = {  # type: ignore[misc]
            "A03": ("injection",),
        }
        with pytest.raises(ValueError, match="already the category's own title"):
            parser.parse()

    def test_an_empty_variant_is_refused(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        parser.title_variants = {"A03": ("  ",)}  # type: ignore[misc]
        with pytest.raises(ValueError, match="declares an empty"):
            parser.parse()

    def test_the_declared_table_is_exactly_what_the_link_file_respells(
        self,
    ) -> None:
        """Fails in both directions, which is the point of deriving it.

        A link name that stops matching its category title and is not declared
        fails here, and a declared variant no link spells fails here too. Both
        sides read tracked files, so this runs with no data/raw present.
        """
        titles = {
            control["control_id"]: control["title"]
            for control in json.loads(
                (PROCESSED_FRAMEWORKS_DIR / "owasp_top10_2021.json").read_text(
                    encoding="utf-8",
                )
            )["controls"]
        }
        assert len(titles) == 10, titles

        observed: dict[str, list[str]] = {}
        with CURATED_LINKS_PATH.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("framework_id") != "owasp_top10_2021":
                    continue
                section_id = str(row.get("section_id") or "")
                name = str(row.get("section_name") or "").strip()
                if name.lower() != titles[section_id].lower():
                    names = observed.setdefault(section_id, [])
                    if name not in names:
                        names.append(name)

        assert observed == {
            key: list(names) for key, names in OPENCRE_TITLE_VARIANTS.items()
        }


class TestRun:
    def test_run_writes_and_clears_the_prose_floor(
        self, parser: OwaspTop102021Parser, tmp_path: Path,
    ) -> None:
        (tmp_path / "out").mkdir()
        output = parser.run()
        assert len(output.controls) == 10
        assert (tmp_path / "out" / "owasp_top10_2021.json").exists()
        assert BaseParser.honest_prose_fraction(output.controls) == 1.0
        assert [s.path for s in output.source_files] == [ARCHIVE_NAME]

    def test_a_title_length_statement_trips_the_prose_floor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Nothing else here proves the floor is set to a non-zero value."""
        raw = tmp_path / "thin"
        raw.mkdir()
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            for code, title in TITLES.items():
                archive.writestr(
                    f"Top10-abc/2021/docs/en/{code}_2021-Entry.md",
                    f"# {code}:2021 – {title}\n\n## Description\n\nShort.\n",
                )
        (raw / ARCHIVE_NAME).write_bytes(payload.getvalue())
        thin = OwaspTop102021Parser(raw_dir=raw, output_dir=tmp_path / "out")
        thin.expected_sha256 = None  # type: ignore[misc]
        (tmp_path / "out").mkdir()
        with pytest.raises(ValueError, match="below the declared floor"):
            thin.run()

    def test_the_anchor_survives_sanitisation(
        self, parser: OwaspTop102021Parser, tmp_path: Path,
    ) -> None:
        """ProseIndex prefers full_text, so full_text is what the model reads."""
        (tmp_path / "out").mkdir()
        for control in parser.run().controls:
            assert control.full_text is not None
            assert "trusted server-side code" in control.full_text
            assert len(control.full_text) > len(control.description)


class TestTheClobberIsReal:
    """_sanitize_control replaces full_text when description overflows.

    Two of the ten real categories hit this: A02's Description sanitises to
    2,377 characters and A04's to 2,944, both over DESCRIPTION_MAX_LENGTH.
    For those two the whole-entry full_text this parser writes is discarded
    and the anchor becomes the Description alone. [measured on the pinned
    archive]

    The fixture above cannot show that, because its descriptions are short.
    Without this class the docstring's claim would be untested and the two
    real categories would diverge from it in silence.
    """

    def test_an_overflowing_description_displaces_the_whole_entry(
        self, tmp_path: Path,
    ) -> None:
        raw = tmp_path / "long"
        raw.mkdir()
        long_body = ("Access control enforces policy such that users. " * 60)
        assert len(long_body) > DESCRIPTION_MAX_LENGTH
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            for code, title in TITLES.items():
                body = long_body if code == "A01" else "A shorter statement. " * 8
                archive.writestr(
                    f"Top10-abc/2021/docs/en/{code}_2021-Entry.md",
                    f"# {code}:2021 – {title}\n\n## Description\n\n{body}\n\n"
                    f"## How to Prevent\n\nUse trusted server-side code.\n",
                )
        (raw / ARCHIVE_NAME).write_bytes(payload.getvalue())
        instance = OwaspTop102021Parser(raw_dir=raw, output_dir=tmp_path / "out")
        instance.expected_sha256 = None  # type: ignore[misc]
        (tmp_path / "out").mkdir()
        controls = {c.control_id: c for c in instance.run().controls}

        overflowed = controls["A01"]
        assert overflowed.full_text is not None
        # The remediation heading is in the entry and not in the Description,
        # so its absence is the proof that the entry was displaced.
        assert "trusted server-side code" not in overflowed.full_text
        assert len(overflowed.description) <= DESCRIPTION_MAX_LENGTH

        kept = controls["A02"]
        assert kept.full_text is not None
        assert "trusted server-side code" in kept.full_text


class TestRealCorpus:
    """The artifact this parser ships and the join it buys.

    Reads only tracked files, so it holds in a checkout with no data/raw.
    """

    def test_the_shipped_artifact_is_ten_categories_of_prose(self) -> None:
        data = json.loads(
            (PROCESSED_FRAMEWORKS_DIR / "owasp_top10_2021.json").read_text(
                encoding="utf-8",
            )
        )
        controls = data["controls"]
        assert data["framework_id"] == "owasp_top10_2021"
        assert data["mapping_unit_level"] == "category"
        assert [c["control_id"] for c in controls] == list(CATEGORY_IDS)
        assert [c["title"] for c in controls] == list(TITLES.values())
        assert all(c["full_text"] for c in controls)

    def test_two_categories_ship_a_displaced_anchor(self) -> None:
        """The measured consequence of TestTheClobberIsReal on real text.

        A02 and A04 are the two whose Description overflows. Their full_text
        is the Description, so it holds no remediation text; the other eight
        carry the whole entry and do. A parser change that moved the boundary
        either way fails here.
        """
        controls = {
            c["control_id"]: c
            for c in json.loads(
                (PROCESSED_FRAMEWORKS_DIR / "owasp_top10_2021.json").read_text(
                    encoding="utf-8",
                )
            )["controls"]
        }
        displaced = {
            key for key, control in controls.items()
            if "## How to Prevent" not in (control["full_text"] or "")
        }
        assert displaced == {"A02", "A04"}

    def test_the_join_is_seventeen_of_seventeen_through_the_title_channel(
        self,
    ) -> None:
        """Every value here is measured and every one can move.

        by_title is 17 only because OPENCRE_TITLE_VARIANTS declares the three
        respellings. Drop the table and it is 11, with A10 flagged by the
        id-side wrong-anchor detector, which has no budget entry.
        """
        report = build_corpus_report()
        row = next(
            r for r in report.per_framework
            if r.framework_id == "owasp_top10_2021"
        )
        assert row.links == 17
        assert row.by_title == 17
        assert row.by_id == 0
        assert row.unresolved == 0
        assert row.resolution_rate == 1.0
        assert row.wrong_anchor_risk == 0
        assert row.distinct_anchors == 10
        assert row.distinct_anchors_pre_truncation == 10
        assert row.distinct_hubs == 16
        assert row.anchor_source_full_text == 17
        assert row.anchor_source_description == 0
        assert row.anchor_source_synthetic == 0
        assert row.dropped_by_prose_rule == 0
        # Every entry exceeds MAX_ANCHOR_CHARS, so every link is cut.
        assert row.truncated == 17
        assert MAX_ANCHOR_CHARS == 2150
