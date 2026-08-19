"""WSTG joins on the ID table, and the archive disagrees with itself in four ways.

Measured on the pinned archive: 130 test files under the testing tree, 116 of
them carrying an ID table, 115 distinct ids, one id owning two files, eight
members that are withdrawal notices rather than tests, and four section_ids in
the curated links that appear in none of the 199 markdown members.

The real-source class reads only tracked files, so the suite holds in a
checkout with no data/raw.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import pytest

from parsers.parse_wstg import (
    DESCRIPTION_BUDGET,
    MEMBER,
    SOURCE_SHA256,
    SourceMember,
    WstgParser,
)
from tract.config import (
    CONTROL_DAMAGED_METADATA_KEY,
    DESCRIPTION_MAX_LENGTH,
    MAX_ANCHOR_CHARS,
    OVERLAY_FRAMEWORK_IDS,
    PROCESSED_FRAMEWORKS_DIR,
)
from tract.corpus_report import (
    CURATED_LINKS_PATH,
    SYNTHETIC_TEXT_ORIGIN,
    TEXT_ORIGIN_METADATA_KEY,
)
from tract.text_selection import ProseIndex, prepare_anchor

FRAMEWORK_ID = "wstg"
BASE = "document/4-Web_Application_Security_Testing"
ROOT = "wstg-abc"

# The four section_ids OpenCRE links that the archive never spells, and the
# number of links each carries. Asserted against the tracked link file rather
# than hardcoded into the parser, so an upstream repair shows up as a failure.
BOGUS_SECTION_IDS: dict[str, int] = {
    "WSTG-APPE-D": 2,
    "WSTG-BUSL-$$": 3,
    "WSTG-INFO-##": 1,
    "WSTG-INPV-00": 3,
}
LINK_COUNT = 118


def _test_file(test_id: str, title: str, summary: str) -> str:
    return (
        f"# {title}\n\n"
        f"|ID          |\n|------------|\n|{test_id}|\n\n"
        f"## Summary\n\n{summary}\n\n"
        f"## How to Test\n\nRun the tool and read the output carefully.\n\n"
        f"## References\n\n- Somewhere\n"
    )


def _withdrawn(test_id: str, title: str, notice: str, successor: str | None) -> str:
    trailer = f"\n\n[merged]: # ({successor})" if successor else ""
    return (
        f"# {title}\n\n"
        f"|ID          |\n|------------|\n|{test_id}|\n\n"
        f"{notice}{trailer}\n"
    )


def _archive(members: dict[str, str]) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        for name, text in members.items():
            archive.writestr(f"{ROOT}/{name}", text)
    return payload.getvalue()


def _parser(
    tmp_path: Path, members: dict[str, str], expected: int, tag: str = "a",
) -> WstgParser:
    raw = tmp_path / f"raw_{tag}"
    raw.mkdir()
    (raw / "wstg.zip").write_bytes(_archive(members))
    instance = WstgParser(
        raw_dir=raw,
        output_dir=tmp_path / f"out_{tag}",
        audit_dir=tmp_path / f"audit_{tag}",
    )
    (tmp_path / f"out_{tag}").mkdir()
    # The fixtures are hand-built, so the pinned digest cannot apply. Every
    # test that needs the pin exercises it explicitly.
    instance.expected_count = expected  # type: ignore[misc]
    instance.expected_sha256 = None  # type: ignore[misc]
    return instance


LONG_SUMMARY = (
    "Search engines crawl and index billions of pages, and the index can "
    "retain content the owner has since removed from the live site."
)

LIVE_MEMBERS: dict[str, str] = {
    f"{BASE}/01-Information_Gathering/01-Conduct_Search.md": _test_file(
        "WSTG-INFO-01", "Conduct Search Engine Discovery", LONG_SUMMARY,
    ),
    f"{BASE}/07-Input_Validation_Testing/13-Testing_for_Buffer_Overflow.md":
        _test_file(
            "WSTG-INPV-13", "Testing for Buffer Overflow",
            "A buffer overflow overwrites adjacent memory and can hand an "
            "attacker control of execution flow.",
        ),
    f"{BASE}/07-Input_Validation_Testing/13-Testing_for_Format_String_Injection.md":
        _test_file(
            "WSTG-INPV-13", "Testing for Format String Injection",
            "A format string bug lets an attacker read and write process "
            "memory through the conversion specifiers.",
        ),
    f"{BASE}/07-Input_Validation_Testing/05.1-Testing_for_Oracle.md":
        "# Testing for Oracle\n\nA sub-test with no ID table of its own.\n",
    f"{BASE}/01-Information_Gathering/README.md":
        "# Information Gathering\n\nCategory intro.\n",
}


@pytest.fixture()
def parser(tmp_path: Path) -> WstgParser:
    return _parser(tmp_path, LIVE_MEMBERS, expected=2)


def _by_id(controls: list[Any]) -> dict[str, Any]:
    return {c.control_id: c for c in controls}


class TestMemberFilter:
    def test_only_files_with_an_id_table_become_controls(
        self, parser: WstgParser,
    ) -> None:
        assert sorted(c.control_id for c in parser.parse()) == [
            "WSTG-INFO-01", "WSTG-INPV-13",
        ]

    def test_the_template_tree_is_not_a_member(self, tmp_path: Path) -> None:
        """`template/999-Foo_Testing/` carries the same ID table shape.

        Three files there spell WSTG-FOO-001 through WSTG-FOO-003 in a
        two-row table. [measured] An id-shaped filter reads them. The path
        filter is what keeps them out.
        """
        members = dict(LIVE_MEMBERS)
        members["template/999-Foo_Testing/1-Testing_for_a_Cat_in_a_Box.md"] = (
            _test_file("WSTG-FOO-001", "Testing for a Cat in a Box",
                       "A box may or may not contain a cat, and the owner "
                       "cannot tell without opening it.")
        )
        found = {c.control_id for c in _parser(tmp_path, members, 2).parse()}
        assert "WSTG-FOO-001" not in found
        assert found == {"WSTG-INFO-01", "WSTG-INPV-13"}

    def test_a_second_tree_reusing_the_path_tail_is_not_a_member(
        self, tmp_path: Path,
    ) -> None:
        """MEMBER anchors the tree on a segment boundary, not a substring."""
        members = dict(LIVE_MEMBERS)
        members[f"vendor/mirror/{BASE}/01-Information_Gathering/01-Copy.md"] = (
            _test_file("WSTG-INFO-99", "A vendored copy", LONG_SUMMARY)
        )
        found = {c.control_id for c in _parser(tmp_path, members, 2).parse()}
        assert "WSTG-INFO-99" not in found

    def test_a_deeper_file_under_a_category_is_not_a_member(
        self, tmp_path: Path,
    ) -> None:
        """`[^/]*` stops the stem spanning another separator."""
        members = dict(LIVE_MEMBERS)
        members[f"{BASE}/01-Information_Gathering/nested/01-Deep.md"] = (
            _test_file("WSTG-INFO-98", "Too deep", LONG_SUMMARY)
        )
        found = {c.control_id for c in _parser(tmp_path, members, 2).parse()}
        assert "WSTG-INFO-98" not in found

    def test_a_tree_readme_is_not_a_member(self, tmp_path: Path) -> None:
        members = dict(LIVE_MEMBERS)
        members[f"{BASE}/README.md"] = _test_file(
            "WSTG-INFO-97", "Tree readme", LONG_SUMMARY,
        )
        found = {c.control_id for c in _parser(tmp_path, members, 2).parse()}
        assert "WSTG-INFO-97" not in found

    def test_a_category_readme_is_not_a_member(self) -> None:
        """Asserted on the pattern, because no output can show it.

        None of the 13 category READMEs carries an ID table [measured], so
        dropping the lookahead changes nothing this parser emits today and no
        test on the artifact could see it. The guard is still worth keeping:
        a README that grew one would otherwise arrive as a control. Pinning it
        here says so rather than leaving an untested line.
        """
        readme = f"{ROOT}/{BASE}/01-Information_Gathering/README.md"
        test = f"{ROOT}/{BASE}/01-Information_Gathering/01-Conduct_Search.md"
        assert MEMBER.match(readme) is None
        assert MEMBER.match(test) is not None

    def test_an_archive_with_no_id_table_anywhere_is_refused(
        self, tmp_path: Path,
    ) -> None:
        members = {
            f"{BASE}/01-Information_Gathering/01-No_Table.md":
                "# No table here\n\n## Summary\n\nProse without an id.\n",
        }
        with pytest.raises(ValueError, match="only join key"):
            _parser(tmp_path, members, 1).parse()


class TestStatement:
    def test_title_is_the_h1_not_the_id(self, parser: WstgParser) -> None:
        first = _by_id(parser.parse())["WSTG-INFO-01"]
        assert first.title == "Conduct Search Engine Discovery"

    def test_statement_stops_before_how_to_test(
        self, parser: WstgParser,
    ) -> None:
        first = _by_id(parser.parse())["WSTG-INFO-01"]
        assert "Search engines crawl" in first.description
        assert "Run the tool" not in first.description
        assert "Somewhere" not in first.description

    def test_the_cut_is_case_insensitive(self, tmp_path: Path) -> None:
        """`13-Test_for_Path_Confusion.md` spells it `## How To Test`.

        Case-sensitively that statement runs to 2,106 characters of which
        1,545 are the procedure, against 561 characters of statement.
        [measured] One control, and the whole point of the cut rule.
        """
        members = {
            f"{BASE}/02-Configuration_and_Deployment_Management_Testing/13-Path.md":
                f"# Test for Path Confusion\n\n"
                f"|ID          |\n|------------|\n|WSTG-CONF-13|\n\n"
                f"## Summary\n\n{LONG_SUMMARY}\n\n"
                f"## How To Test\n\nA procedure spelled with a capital T.\n",
        }
        control = _parser(tmp_path, members, 1).parse()[0]
        assert "Search engines crawl" in control.description
        assert "capital T" not in control.description

    def test_the_statement_starts_below_the_id_table(
        self, parser: WstgParser,
    ) -> None:
        first = _by_id(parser.parse())["WSTG-INFO-01"]
        assert "WSTG-INFO-01" not in first.description
        assert not first.description.startswith("|")

    def test_the_statement_keeps_its_section_headings(
        self, parser: WstgParser,
    ) -> None:
        """Kept in the artifact, removed on the way to the encoder.

        The stored form stays inspectable, which is what lets a test tell a
        `## How to Test` heading that escaped the cut from the prose mention
        of that section name WSTG-ATHZ-02 carries in a sentence. [measured]
        """
        first = _by_id(parser.parse())["WSTG-INFO-01"]
        assert first.description.startswith("## Summary")
        assert prepare_anchor(first.description)[0].startswith("Summary ")

    def test_a_member_with_no_h1_is_refused(self, tmp_path: Path) -> None:
        members = {
            f"{BASE}/01-Information_Gathering/01-No_H1.md":
                "|ID          |\n|------------|\n|WSTG-INFO-01|\n\n"
                "## Summary\n\nProse with no heading above it.\n",
        }
        with pytest.raises(ValueError, match="no H1"):
            _parser(tmp_path, members, 1).parse()


class TestMerge:
    def test_a_shared_id_merges_both_files(self, parser: WstgParser) -> None:
        shared = _by_id(parser.parse())["WSTG-INPV-13"]
        assert shared.title == (
            "Testing for Buffer Overflow / Testing for Format String Injection"
        )
        assert "adjacent memory" in shared.description
        assert "conversion specifiers" in shared.description

    def test_a_merged_control_is_marked_synthetic(
        self, parser: WstgParser,
    ) -> None:
        controls = _by_id(parser.parse())
        merged = controls["WSTG-INPV-13"]
        assert merged.metadata is not None
        assert merged.metadata[TEXT_ORIGIN_METADATA_KEY] == SYNTHETIC_TEXT_ORIGIN
        solo = controls["WSTG-INFO-01"]
        assert solo.metadata is not None
        assert TEXT_ORIGIN_METADATA_KEY not in solo.metadata

    def test_members_are_merged_in_filename_order(
        self, parser: WstgParser,
    ) -> None:
        shared = _by_id(parser.parse())["WSTG-INPV-13"]
        assert shared.description.index("adjacent memory") < (
            shared.description.index("conversion specifiers")
        )

    def test_a_withdrawn_member_contributes_no_text_to_a_merge(
        self, tmp_path: Path,
    ) -> None:
        """The real WSTG-INPV-13: one file withdrawn, one with prose.

        The notice is not a statement about the surviving test and it would
        sit at the front of the anchor.
        """
        members = {
            f"{BASE}/07-Input_Validation_Testing/13-Testing_for_Buffer_Overflow.md":
                _withdrawn("WSTG-INPV-13", "Testing for Buffer Overflow",
                           "This content has been removed", None),
            f"{BASE}/07-Input_Validation_Testing/13-Testing_for_Format_String_Injection.md":
                _test_file("WSTG-INPV-13", "Testing for Format String Injection",
                           "A format string bug lets an attacker read and "
                           "write process memory through the specifiers."),
        }
        control = _parser(tmp_path, members, 1).parse()[0]
        assert "has been removed" not in control.description
        assert "format string bug" in control.description
        # Both names survive in the title: the id is what OpenCRE curated and
        # both spellings should reach the title channel.
        assert control.title == (
            "Testing for Buffer Overflow / Testing for Format String Injection"
        )

    def test_a_merged_control_is_not_marked_damaged(
        self, tmp_path: Path,
    ) -> None:
        """One live member is enough. Damage is the absence of any statement."""
        members = {
            f"{BASE}/07-Input_Validation_Testing/13-A.md": _withdrawn(
                "WSTG-INPV-13", "A", "This content has been removed", None,
            ),
            f"{BASE}/07-Input_Validation_Testing/13-B.md": _test_file(
                "WSTG-INPV-13", "B", LONG_SUMMARY,
            ),
        }
        control = _parser(tmp_path, members, 1).parse()[0]
        assert control.metadata is not None
        assert CONTROL_DAMAGED_METADATA_KEY not in control.metadata


class TestWithdrawnTests:
    """Eight members are notices rather than tests, five naming a successor."""

    WITHDRAWN = {
        f"{BASE}/04-Authentication_Testing/01-Credentials.md": _withdrawn(
            "WSTG-ATHN-01", "Testing for Credentials Transported over an "
            "Encrypted Channel",
            "This content has been merged into: [Testing for Sensitive "
            "Information Sent via Unencrypted Channels](../09/03.md)",
            "WSTG-CRYP-03",
        ),
        f"{BASE}/09-Testing_for_Weak_Cryptography/03-Sensitive.md": _test_file(
            "WSTG-CRYP-03", "Testing for Sensitive Information Sent via "
            "Unencrypted Channels",
            "Sensitive data must be protected when it is transmitted through "
            "the network, or an attacker on the path can read it.",
        ),
        f"{BASE}/11-Client-side_Testing/08-Flashing.md": _withdrawn(
            "WSTG-CLNT-08", "Testing for Cross Site Flashing",
            "This content has been removed.", None,
        ),
    }

    def _controls(self, tmp_path: Path) -> dict[str, Any]:
        return _by_id(_parser(tmp_path, self.WITHDRAWN, 2).parse())

    def test_a_withdrawn_test_naming_a_successor_emits_no_control(
        self, tmp_path: Path,
    ) -> None:
        assert "WSTG-ATHN-01" not in self._controls(tmp_path)

    def test_its_id_becomes_an_alt_id_on_the_successor(
        self, tmp_path: Path,
    ) -> None:
        target = self._controls(tmp_path)["WSTG-CRYP-03"]
        assert target.metadata is not None
        assert target.metadata["alt_ids"] == ["WSTG-ATHN-01"]

    def test_the_retired_id_reaches_the_successor_prose(
        self, tmp_path: Path,
    ) -> None:
        """End to end through the channel the alias exists to use.

        Without the alias this id resolves to "This content has been merged
        into: ...", which is a redirect notice standing in for a control
        statement on three curated links.
        """
        controls = self._controls(tmp_path)
        index = ProseIndex([{
            "framework_name": WstgParser.framework_name,
            "controls": [c.model_dump(mode="json") for c in controls.values()],
        }])
        # The index keys on the framework NAME, not the id: canonical_framework
        # folds "WSTG" and "OWASP Web Security Testing Guide (WSTG)" onto one
        # another and leaves the lowercase id alone.
        name = WstgParser.framework_name
        retired = index.by_id(name, "WSTG-ATHN-01")
        successor = index.by_id(name, "WSTG-CRYP-03")
        assert retired is not None
        assert successor is not None
        assert retired.text == successor.text
        assert "Sensitive data must be protected" in retired.text
        assert "has been merged into" not in retired.text

    def test_a_withdrawn_test_naming_no_successor_stays_a_damaged_control(
        self, tmp_path: Path,
    ) -> None:
        control = self._controls(tmp_path)["WSTG-CLNT-08"]
        assert control.metadata is not None
        assert control.metadata[CONTROL_DAMAGED_METADATA_KEY] == "true"
        assert "damage_reason" in control.metadata

    def test_a_live_control_is_never_marked_damaged(
        self, tmp_path: Path,
    ) -> None:
        control = self._controls(tmp_path)["WSTG-CRYP-03"]
        assert control.metadata is not None
        assert CONTROL_DAMAGED_METADATA_KEY not in control.metadata

    def test_a_successor_that_does_not_exist_is_refused(
        self, tmp_path: Path,
    ) -> None:
        members = {
            f"{BASE}/04-Authentication_Testing/01-Gone.md": _withdrawn(
                "WSTG-ATHN-01", "Gone", "This content has been merged into: X",
                "WSTG-NOPE-99",
            ),
            f"{BASE}/01-Information_Gathering/01-Live.md": _test_file(
                "WSTG-INFO-01", "Live", LONG_SUMMARY,
            ),
        }
        with pytest.raises(ValueError, match="which no member of the archive"):
            _parser(tmp_path, members, 1).parse()

    def test_a_redirect_chain_is_refused(self, tmp_path: Path) -> None:
        """One hop only. A chain has to be read and declared, not followed."""
        members = {
            f"{BASE}/04-Authentication_Testing/01-A.md": _withdrawn(
                "WSTG-ATHN-01", "A", "Merged.", "WSTG-ATHN-02",
            ),
            f"{BASE}/04-Authentication_Testing/02-B.md": _withdrawn(
                "WSTG-ATHN-02", "B", "Merged.", "WSTG-INFO-01",
            ),
            f"{BASE}/01-Information_Gathering/01-Live.md": _test_file(
                "WSTG-INFO-01", "Live", LONG_SUMMARY,
            ),
        }
        with pytest.raises(ValueError, match="itself withdrawn"):
            _parser(tmp_path, members, 1).parse()


class TestTheDescriptionCapProtectsTheAnchor:
    """Ruling R14: the parser owns its anchor, not whatever margin is left.

    45 of the 115 statements sanitise past DESCRIPTION_MAX_LENGTH. [measured]
    Without the cap `_sanitize_control` replaces this parser's full_text with
    the whole description on every one of them.
    """

    @staticmethod
    def _long_summary(chars: int) -> str:
        # Sentence-terminated so a paragraph pack and a word cut differ, and
        # padded to an odd length so a cut on the budget cannot coincide with
        # a word boundary by luck.
        sentence = "A tester probes the endpoint for an injection flaw. "
        return (sentence * (chars // len(sentence) + 2))[:chars]

    def _control(self, tmp_path: Path, summary: str, tag: str = "cap") -> Any:
        members = {
            f"{BASE}/01-Information_Gathering/01-Long.md": _test_file(
                "WSTG-INFO-01", "Long", summary,
            ),
        }
        return _parser(tmp_path, members, 1, tag).parse()[0]

    def test_a_description_inside_the_budget_is_untouched(
        self, tmp_path: Path,
    ) -> None:
        control = self._control(tmp_path, LONG_SUMMARY)
        assert LONG_SUMMARY in control.description
        assert control.full_text is None

    def test_an_overflowing_description_is_capped_and_full_text_carries_it(
        self, tmp_path: Path,
    ) -> None:
        control = self._control(tmp_path, self._long_summary(4000))
        assert len(control.description) <= DESCRIPTION_BUDGET
        assert control.full_text is not None
        assert len(control.full_text) > DESCRIPTION_BUDGET

    def test_a_description_between_the_budget_and_the_limit_is_still_capped(
        self, tmp_path: Path,
    ) -> None:
        """The window the base class would let through unnoticed.

        A statement of 1,900 characters fits DESCRIPTION_MAX_LENGTH, so
        `_sanitize_control` would not overwrite full_text today. It is capped
        anyway, because the margin is 100 characters of source formatting and
        the parser is not entitled to spend it.
        """
        control = self._control(tmp_path, self._long_summary(1900))
        assert DESCRIPTION_BUDGET < len(self._long_summary(1900)) + 8
        assert len(control.description) <= DESCRIPTION_BUDGET

    def test_the_anchor_survives_sanitisation(self, tmp_path: Path) -> None:
        """The R14 regression itself, measured through run() rather than parse().

        run() is where `_sanitize_control` fires. If the cap ever stops
        clearing DESCRIPTION_MAX_LENGTH, full_text on disk becomes the
        description and this fails.
        """
        members = {
            f"{BASE}/01-Information_Gathering/01-Long.md": _test_file(
                "WSTG-INFO-01", "Long", self._long_summary(6000),
            ),
        }
        parser = _parser(tmp_path, members, 1, "anchor")
        control = parser.run().controls[0]
        assert control.full_text is not None
        assert "Run the tool" not in control.full_text
        assert control.full_text.startswith("## Summary")
        # The cap fired, so the two fields differ and full_text is the longer.
        assert len(control.full_text) > len(control.description)

    def test_the_cap_is_recorded_as_a_before_and_after_pair(
        self, tmp_path: Path,
    ) -> None:
        members = {
            f"{BASE}/01-Information_Gathering/01-Long.md": _test_file(
                "WSTG-INFO-01", "Long", self._long_summary(4000),
            ),
        }
        parser = _parser(tmp_path, members, 1, "rec")
        parser.run()
        records = [
            json.loads(line)
            for line in (tmp_path / "audit_rec" / "wstg.jsonl").read_text(
                encoding="utf-8",
            ).splitlines()
        ]
        capped = [
            r for r in records
            if r["repair"] == "description_capped_to_protect_full_text"
        ]
        assert len(capped) == 1
        # Text on both sides. A pair of integers cannot be checked against
        # anything.
        assert isinstance(capped[0]["before"], str)
        assert isinstance(capped[0]["after"], str)
        assert len(capped[0]["before"]) > len(capped[0]["after"])
        assert capped[0]["before"].startswith(capped[0]["after"][:200])

    def test_a_single_paragraph_over_budget_is_cut_on_a_word_boundary(
        self, tmp_path: Path,
    ) -> None:
        control = self._control(tmp_path, self._long_summary(4000), "word")
        assert len(control.description) <= DESCRIPTION_BUDGET
        assert not control.description.endswith(" ")
        # A word cut, not a hard slice through a token.
        assert control.description[-1].isalnum() or control.description[-1] in ".,"

    def test_the_cap_lands_on_a_paragraph_boundary_when_it_can(
        self, tmp_path: Path,
    ) -> None:
        first = "A" * 1000
        second = "B" * 1000
        members = {
            f"{BASE}/01-Information_Gathering/01-Paras.md": _test_file(
                "WSTG-INFO-01", "Paras", f"{first}\n\n{second}",
            ),
        }
        control = _parser(tmp_path, members, 1, "para").parse()[0]
        assert control.description.endswith(first)
        assert "B" not in control.description

    def test_the_budget_leaves_real_headroom(self) -> None:
        assert DESCRIPTION_BUDGET < DESCRIPTION_MAX_LENGTH
        assert DESCRIPTION_MAX_LENGTH - DESCRIPTION_BUDGET >= 100


class TestAudit:
    def test_the_merge_is_recorded(
        self, parser: WstgParser, tmp_path: Path,
    ) -> None:
        parser.run()
        records = [
            json.loads(line)
            for line in (tmp_path / "audit_a" / "wstg.jsonl").read_text(
                encoding="utf-8",
            ).splitlines()
        ]
        merges = [r for r in records if r["repair"] == "members_merged_under_one_id"]
        assert len(merges) == 1
        assert merges[0]["control_id"] == "WSTG-INPV-13"
        assert len(merges[0]["members"]) == 2
        assert isinstance(merges[0]["after"], str)
        assert "adjacent memory" in merges[0]["after"]
        assert [isinstance(b, str) for b in merges[0]["before"]] == [True, True]

    def test_the_alias_is_recorded_with_text_on_both_sides(
        self, tmp_path: Path,
    ) -> None:
        parser = _parser(tmp_path, TestWithdrawnTests.WITHDRAWN, 2, "alias")
        parser.run()
        records = [
            json.loads(line)
            for line in (tmp_path / "audit_alias" / "wstg.jsonl").read_text(
                encoding="utf-8",
            ).splitlines()
        ]
        aliases = [
            r for r in records
            if r["repair"] == "withdrawn_test_aliased_to_its_successor"
        ]
        assert len(aliases) == 1
        assert aliases[0]["control_id"] == "WSTG-ATHN-01"
        assert aliases[0]["successor_id"] == "WSTG-CRYP-03"
        assert "has been merged into" in aliases[0]["before"]
        assert "Sensitive data must be protected" in aliases[0]["after"]

    def test_the_audit_file_is_written_even_with_no_repairs(
        self, tmp_path: Path,
    ) -> None:
        members = {
            f"{BASE}/01-Information_Gathering/01-One.md": _test_file(
                "WSTG-INFO-01", "One",
                "A statement long enough to clear every prose bar that this "
                "project applies to a description.",
            ),
        }
        parser = _parser(tmp_path, members, 1, "solo")
        parser.run()
        assert (tmp_path / "audit_solo" / "wstg.jsonl").read_text(
            encoding="utf-8",
        ) == ""


class TestRun:
    def test_run_writes(self, parser: WstgParser, tmp_path: Path) -> None:
        output = parser.run()
        assert len(output.controls) == 2
        assert [s.path for s in output.source_files] == ["wstg.zip"]

    def test_output_is_byte_identical_on_a_re_run(
        self, parser: WstgParser, tmp_path: Path,
    ) -> None:
        parser.run()
        path = tmp_path / "out_a" / "wstg.json"
        first = path.read_bytes()
        parser.run()
        assert path.read_bytes() == first

    def test_a_wrong_archive_is_refused(self, tmp_path: Path) -> None:
        parser = _parser(tmp_path, LIVE_MEMBERS, 2, "pin")
        parser.expected_sha256 = "0" * 64  # type: ignore[misc]
        with pytest.raises(ValueError, match="not the pinned"):
            parser.parse()

    def test_a_title_length_statement_trips_the_prose_floor(
        self, tmp_path: Path,
    ) -> None:
        """min_prose_fraction is 1.0, so one lost statement fails the run."""
        members = {
            f"{BASE}/01-Information_Gathering/01-Short.md": _test_file(
                "WSTG-INFO-01", "Short", "Too short.",
            ),
        }
        with pytest.raises(ValueError, match="honest prose fraction"):
            _parser(tmp_path, members, 1, "floor").run()


class TestSourceMember:
    """The classifier the whole withdrawal path rests on."""

    @staticmethod
    def _member(text: str) -> SourceMember:
        return SourceMember("WSTG-INFO-01", "x.md", text)

    def test_a_body_with_a_section_is_live(self) -> None:
        member = self._member(_test_file("WSTG-INFO-01", "T", LONG_SUMMARY))
        assert member.is_withdrawn is False
        assert member.successor is None

    def test_a_body_with_no_section_is_withdrawn(self) -> None:
        member = self._member(
            _withdrawn("WSTG-INFO-01", "T", "This content has been removed.",
                       None)
        )
        assert member.is_withdrawn is True

    def test_the_successor_is_read_from_the_trailer(self) -> None:
        member = self._member(
            _withdrawn("WSTG-INFO-01", "T", "Merged.", "WSTG-INFO-08")
        )
        assert member.successor == "WSTG-INFO-08"

    def test_a_notice_mentioning_an_id_in_prose_is_not_a_trailer(self) -> None:
        """Only the machine-readable trailer counts.

        WSTG-IDNT-05 names its successor in the sentence as well, so a looser
        pattern would read the same id twice and a different sentence could
        supply a wrong one.
        """
        member = self._member(
            _withdrawn("WSTG-INFO-01", "T",
                       "This test has been merged into WSTG-IDNT-04 due to "
                       "overlapping scope.", None)
        )
        assert member.successor is None


class TestOpenCreLinkShape:
    """Derived from the tracked link file, so it fails in both directions."""

    @staticmethod
    def _links() -> list[dict[str, Any]]:
        rows = [
            json.loads(line)
            for line in CURATED_LINKS_PATH.read_text(
                encoding="utf-8",
            ).splitlines()
            if line.strip()
        ]
        return [r for r in rows if r["framework_id"] == FRAMEWORK_ID]

    def test_every_curated_link_names_its_own_section_id(self) -> None:
        """Why no OPENCRE_TITLE_VARIANTS table exists for this framework.

        All 118 links carry section_name == section_id, so the title channel
        cannot answer and by_title is 0 by construction. If OpenCRE ever
        respells one as a test name, this fails and the parser needs the
        variants table Task 4 established.
        """
        rows = self._links()
        assert len(rows) == LINK_COUNT
        divergent = {
            (r["section_id"], r["section_name"])
            for r in rows
            if str(r["section_name"]).strip() != str(r["section_id"]).strip()
        }
        assert divergent == set()

    def test_the_unresolvable_section_ids_are_exactly_the_four_known_ones(
        self,
    ) -> None:
        """The arithmetic ceiling, derived rather than asserted.

        Fails if upstream repairs one of the four, and fails if a fifth
        appears. Either way the declared ceiling needs re-deriving.
        """
        shipped = {c["control_id"] for c in _shipped()}
        alt = {
            alt_id
            for c in _shipped()
            for alt_id in (c.get("metadata") or {}).get("alt_ids", [])
        }
        reachable = shipped | alt
        missing: dict[str, int] = {}
        for row in self._links():
            if row["section_id"] not in reachable:
                missing[row["section_id"]] = missing.get(row["section_id"], 0) + 1
        assert missing == BOGUS_SECTION_IDS
        assert sum(missing.values()) == 9
        assert (LINK_COUNT - 9) / LINK_COUNT == pytest.approx(0.923728813559322)

    def test_every_other_link_reaches_a_shipped_id(self) -> None:
        shipped = {c["control_id"] for c in _shipped()}
        alt = {
            alt_id
            for c in _shipped()
            for alt_id in (c.get("metadata") or {}).get("alt_ids", [])
        }
        unreached = {
            r["section_id"] for r in self._links()
            if r["section_id"] not in shipped | alt
        }
        assert unreached == set(BOGUS_SECTION_IDS)

    def test_the_aliases_carry_curated_links(self) -> None:
        """The reason the alias path exists rather than a comment about it.

        Three of the five retired ids are linked. Without the alias each one
        anchors a curated link on a redirect notice.
        """
        linked = {r["section_id"] for r in self._links()}
        alt = {
            alt_id
            for c in _shipped()
            for alt_id in (c.get("metadata") or {}).get("alt_ids", [])
        }
        assert len(alt & linked) == 3
        assert alt & linked == {
            "WSTG-ATHN-01", "WSTG-ERRH-02", "WSTG-INPV-03",
        }


def _shipped() -> list[dict[str, Any]]:
    data = json.loads(
        (PROCESSED_FRAMEWORKS_DIR / f"{FRAMEWORK_ID}.json").read_text(
            encoding="utf-8",
        )
    )
    controls: list[dict[str, Any]] = data["controls"]
    return controls


def _anchor(control: dict[str, Any]) -> str:
    text: str = control.get("full_text") or control["description"]
    prepared: str = prepare_anchor(text)[0]
    return prepared


class TestRealCorpus:
    """The artifact this parser ships. Reads only tracked files."""

    def test_the_artifact_is_tracked_rather_than_overlay_routed(self) -> None:
        """CC-BY-SA-4.0 is in no restricted tier, so the file ships in git."""
        assert FRAMEWORK_ID not in OVERLAY_FRAMEWORK_IDS
        assert (PROCESSED_FRAMEWORKS_DIR / f"{FRAMEWORK_ID}.json").exists()

    def test_the_shipped_census(self) -> None:
        data = json.loads(
            (PROCESSED_FRAMEWORKS_DIR / f"{FRAMEWORK_ID}.json").read_text(
                encoding="utf-8",
            )
        )
        assert data["framework_id"] == FRAMEWORK_ID
        assert data["mapping_unit_level"] == "test"
        assert len(data["controls"]) == WstgParser.expected_count == 110
        assert [s["sha256"] for s in data["source_files"]] == [SOURCE_SHA256]

    def test_no_description_reaches_the_limit_that_rewrites_full_text(
        self,
    ) -> None:
        """Ruling R14, measured on what shipped rather than on the source."""
        longest = max(len(c["description"]) for c in _shipped())
        assert longest <= DESCRIPTION_BUDGET
        assert longest < DESCRIPTION_MAX_LENGTH

    def test_no_statement_carries_a_cut_heading(self) -> None:
        """The cut rule, checked on the artifact.

        On the heading, not the phrase. WSTG-ATHZ-02 says "covered in the How
        to Test section" in prose [measured], which is the statement talking
        about the procedure rather than containing it, so a substring test on
        the words would fail on correct output. The `##` marker is what
        distinguishes them, which is why the stored statement keeps it.

        Both stored fields, not just the anchor. `full_text` is what the
        encoder reads, but a leak into the capped `description` ships to every
        consumer that reads the artifact directly, and an earlier version of
        this test looked at `full_text or description` and so could not see
        one.
        """
        offenders = sorted(
            c["control_id"] for c in _shipped()
            for field in (c["description"], c.get("full_text") or "")
            for heading in ("## How to Test", "## How To Test", "## Tools",
                            "## References", "## Remediation",
                            "## Related Test Cases")
            if heading in field
        )
        assert offenders == []

    def test_the_statement_reaches_past_the_summary_heading(self) -> None:
        """The cut has to leave something. A statement of one heading is not one.

        `## Summary` is ten characters, so a paragraph pack that kept only the
        heading would land here.
        """
        stubs = [
            c["control_id"] for c in _shipped()
            for field in (c["description"], c.get("full_text") or c["description"])
            if len(field.strip()) < 30
        ]
        assert stubs == []

    def test_the_five_aliases_are_the_withdrawn_tests_naming_a_successor(
        self,
    ) -> None:
        aliases = {
            c["control_id"]: (c.get("metadata") or {})["alt_ids"]
            for c in _shipped()
            if (c.get("metadata") or {}).get("alt_ids")
        }
        assert aliases == {
            "WSTG-CONF-06": ["WSTG-INPV-03"],
            "WSTG-CRYP-03": ["WSTG-ATHN-01"],
            "WSTG-ERRH-01": ["WSTG-ERRH-02"],
            "WSTG-IDNT-04": ["WSTG-IDNT-05"],
            "WSTG-INFO-08": ["WSTG-INFO-09"],
        }

    def test_no_alias_collides_with_a_shipped_id(self) -> None:
        """An alternate that spells a real id would be dropped in silence."""
        shipped = {c["control_id"] for c in _shipped()}
        alt = [
            alt_id
            for c in _shipped()
            for alt_id in (c.get("metadata") or {}).get("alt_ids", [])
        ]
        assert set(alt) & shipped == set()
        assert len(alt) == len(set(alt)) == 5

    def test_only_the_merged_control_is_synthetic(self) -> None:
        synthetic = [
            c["control_id"] for c in _shipped()
            if (c.get("metadata") or {}).get(TEXT_ORIGIN_METADATA_KEY)
            == SYNTHETIC_TEXT_ORIGIN
        ]
        assert synthetic == ["WSTG-INPV-13"]

    def test_only_the_two_unrecoverable_withdrawals_are_damaged(self) -> None:
        damaged = sorted(
            c["control_id"] for c in _shipped()
            if (c.get("metadata") or {}).get(CONTROL_DAMAGED_METADATA_KEY)
        )
        assert damaged == ["WSTG-CLNT-08", "WSTG-CONF-08"]

    def test_no_live_control_ships_a_withdrawal_notice_as_its_anchor(
        self,
    ) -> None:
        """The defect the alias path removes, asserted on the artifact."""
        damaged = {
            c["control_id"] for c in _shipped()
            if (c.get("metadata") or {}).get(CONTROL_DAMAGED_METADATA_KEY)
        }
        offenders = [
            c["control_id"] for c in _shipped()
            if c["control_id"] not in damaged
            and "has been merged into" in _anchor(c)
        ]
        assert offenders == []

    def test_the_shared_anchor_prefix_stays_negligible(self) -> None:
        """Ruling R13, measured rather than assumed, and reported either way.

        Across all 110 prepared anchors the shared prefix is 0 characters,
        because the two withdrawn controls open on their notice. Across the
        108 that open on a section it is 8, "Summary " surviving
        strip_markup's removal of the `##` marker. That is 0.4% of the 2,150
        character budget, against the 364 characters (17%) that made the Top
        10's Factors table worth removing structurally, so nothing is
        stripped. Pinned so a source edition that grows a shared header shows
        up as a failure rather than as a quiet loss of budget.
        """
        damaged = {
            c["control_id"] for c in _shipped()
            if (c.get("metadata") or {}).get(CONTROL_DAMAGED_METADATA_KEY)
        }
        every = [_anchor(c) for c in _shipped()]
        opening = [
            _anchor(c) for c in _shipped() if c["control_id"] not in damaged
        ]
        assert len(every) == 110
        assert len(opening) == 108
        assert len(os.path.commonprefix(every)) == 0
        assert len(os.path.commonprefix(opening)) == 8
        assert os.path.commonprefix(opening) == "Summary "
        # The threshold the ruling turns on, stated so the comparison is not
        # left to the reader.
        assert 8 / MAX_ANCHOR_CHARS < 0.01
