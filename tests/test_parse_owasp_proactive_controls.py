"""Ten controls, C1 through C10, from the current mkdocs tree only.

The pinned archive holds the `c<N>-` filename pattern three times, not twice.
`docs/the-top-10/` is the current edition, `docs/archive/2018/` is the
superseded v3 wording under different stems, and `docs/archive/2024/the-top-10/`
is a near-copy of the current tree under IDENTICAL stems. [measured: 10 members
each] The third one is the dangerous one, because a member filter written on
`the-top-10/c\\d+-` rather than on `docs/the-top-10/` reads twenty files and
emits every control id twice. TestMemberFilter carries all three.

TestTheDescriptionCapProtectsTheAnchor is the positive control for the
docstring paragraph about `_sanitize_control`. Six of the ten real Descriptions
sanitise past DESCRIPTION_MAX_LENGTH, so without the cap six of ten anchors
would silently become a truncated Description instead of the entry. [measured]

TestRealCorpus reads only tracked files, so it holds in a checkout with no
data/raw present.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from parsers.parse_owasp_proactive_controls import (
    ARCHIVE_NAME,
    CONTROL_IDS,
    DESCRIPTION_BUDGET,
    OwaspProactiveControlsParser,
)
from tract.config import (
    DESCRIPTION_MAX_LENGTH,
    OVERLAY_FRAMEWORK_IDS,
    PROCESSED_FRAMEWORKS_DIR,
)
from tract.corpus_report import CURATED_LINKS_PATH
from tract.parsers.base import BaseParser
from tract.text_selection import prepare_anchor

FRAMEWORK_ID = "owasp_proactive_controls"

TITLES: dict[str, str] = {
    "C1": "Implement Access Control",
    "C2": "Use Cryptography to Protect Data",
    "C3": "Validate all Input & Handle Exceptions",
    "C4": "Address Security from the Start",
    "C5": "Secure By Default Configurations",
    "C6": "Keep your Components Secure",
    "C7": "Secure Digital Identities",
    "C8": "Leverage Browser Security Features",
    "C9": "Implement Security Logging and Monitoring",
    "C10": "Stop Server Side Request Forgery",
}

# Stems as they appear in the pinned archive. docs/archive/2024/the-top-10/
# reuses every one of them, which is why the member filter cannot key on them.
STEMS: dict[str, str] = {
    "C1": "c1-accesscontrol",
    "C2": "c2-crypto",
    "C3": "c3-validate-input-and-handle-exceptions",
    "C4": "c4-secure-architecture",
    "C5": "c5-secure-by-default",
    "C6": "c6-use-secure-dependencies",
    "C7": "c7-secure-digital-identities",
    "C8": "c8-leverage-browser-security-features",
    "C9": "c9-security-logging-and-monitoring",
    "C10": "c10-stop-server-side-request-forgery",
}

ROOT = "www-project-proactive-controls-abc"

# Section order matches the source: Description, Threats, Implementation,
# Vulnerabilities Prevented, References, Tools. A parser that takes the first
# section, the last section, or everything after the heading reads differently
# here.
BODY = """
## Description

{title} is the control this entry defines, and the statement runs long enough
to clear the prose floor without any help from the sections below it.

Access decisions, key handling and input trust boundaries all sit inside the
part of the entry that says what the control is.

## Threats

An attacker abuses the missing control.

## Implementation

### 1) Do the thing properly up front

Guidance on how to satisfy the control rather than what it is.

## Vulnerabilities Prevented

- Some weakness class.

## References

- A link.

## Tools

- A scanner.
"""

# The 2018 edition. Same `c<N>-` prefix, different stems, superseded wording.
ARCHIVED_2018 = """# C1: Security Requirements

## Description

The superseded 2018 wording that must never reach the corpus.
"""

# The 2024 archive copy. IDENTICAL stems to the current tree, which is the
# decoy a stem-keyed filter cannot see.
ARCHIVED_2024_MARKER = "the archived 2024 copy that must never reach the corpus"


def _entry(code: str, title: str) -> str:
    return f"# {code}: {title}\n" + BODY.format(title=title)


def _archive(
    codes: tuple[str, ...] = tuple(TITLES),
    *,
    with_decoys: bool = True,
    extra: dict[str, str] | None = None,
) -> bytes:
    """A fixture archive. `extra` replaces a member rather than repeating it.

    Built through a dict on purpose. `ZipFile.writestr` happily writes a second
    member under a name that already exists, and then `namelist()` reports both
    while `read()` returns one, so a test meaning to swap a file would instead
    parse two and assert against whichever the reader picked.
    """
    members: dict[str, str] = {}
    for code in codes:
        members[f"{ROOT}/docs/the-top-10/{STEMS[code]}.md"] = _entry(
            code, TITLES[code],
        )
    if with_decoys:
        members[f"{ROOT}/docs/archive/2018/c1-security-requirements.md"] = (
            ARCHIVED_2018
        )
        for code in codes:
            members[
                f"{ROOT}/docs/archive/2024/the-top-10/{STEMS[code]}.md"
            ] = (
                f"# {code}: {TITLES[code]}\n\n## Description\n\n"
                f"{ARCHIVED_2024_MARKER}\n"
            )
        # Binary exports live beside the docs tree and carry no markdown.
        members[f"{ROOT}/v3/OWASP_Top_10_Proactive_Controls_V3.pdf"] = "x"
    members.update(extra or {})

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        for name in sorted(members):
            archive.writestr(name, members[name])
    return payload.getvalue()


def _instance(tmp_path: Path, raw: Path) -> OwaspProactiveControlsParser:
    """Audit dir is redirected so a test never writes the real one."""
    instance = OwaspProactiveControlsParser(
        raw_dir=raw,
        output_dir=tmp_path / "out",
        audit_dir=tmp_path / "audit",
    )
    # A fixture archive is not the pinned one, so the real digest is stood
    # down here rather than the gate being widened to accept two archives.
    instance.expected_sha256 = None  # type: ignore[misc]
    return instance


def _parser_for(
    tmp_path: Path, name: str, payload: bytes,
) -> OwaspProactiveControlsParser:
    raw = tmp_path / name
    raw.mkdir(exist_ok=True)
    (raw / ARCHIVE_NAME).write_bytes(payload)
    return _instance(tmp_path, raw)


def _shipped() -> dict[str, dict[str, Any]]:
    """The tracked artifact, keyed by control id."""
    return {
        control["control_id"]: control
        for control in json.loads(
            (PROCESSED_FRAMEWORKS_DIR / f"{FRAMEWORK_ID}.json").read_text(
                encoding="utf-8",
            )
        )["controls"]
    }


def _curated_links() -> list[dict[str, Any]]:
    rows = []
    with CURATED_LINKS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("framework_id") == FRAMEWORK_ID:
                rows.append(row)
    return rows


def _shared_prefix(texts: list[str]) -> int:
    """Characters every one of `texts` shares from the front."""
    count = 0
    for chars in zip(*texts):
        if len(set(chars)) > 1:
            break
        count += 1
    return count


def _audit_records(tmp_path: Path) -> list[dict[str, object]]:
    path = tmp_path / "audit" / f"{FRAMEWORK_ID}.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture()
def parser(tmp_path: Path) -> OwaspProactiveControlsParser:
    return _parser_for(tmp_path, "raw", _archive())


class TestMemberFilter:
    def test_only_the_current_edition_is_read(
        self, parser: OwaspProactiveControlsParser,
    ) -> None:
        assert [c.control_id for c in parser.parse()] == list(CONTROL_IDS)

    def test_the_2018_archive_is_not_read(
        self, parser: OwaspProactiveControlsParser,
    ) -> None:
        controls = parser.parse()
        assert "Security Requirements" not in [c.title for c in controls]
        assert not any("superseded" in c.description for c in controls)

    def test_the_2024_archive_sharing_every_stem_is_not_read(
        self, parser: OwaspProactiveControlsParser,
    ) -> None:
        """A filter keyed on `the-top-10/c<N>-` reads twenty files. [measured]

        Ten of them repeat ids C1..C10, so the exact-set check cannot catch it
        and the artifact ships each control twice.
        """
        controls = parser.parse()
        assert len(controls) == 10
        assert not any(
            ARCHIVED_2024_MARKER in (c.full_text or "") for c in controls
        )
        assert not any(ARCHIVED_2024_MARKER in c.description for c in controls)

    def test_a_member_outside_the_docs_tree_is_not_read(
        self, tmp_path: Path,
    ) -> None:
        """`docs/the-top-10/` must be a path segment, not a substring."""
        payload = _archive(extra={
            f"{ROOT}/vendor/mirror-docs/the-top-10/c11-extra.md":
                "# C11: Not a control\n\n## Description\n\nA vendor mirror.\n",
        })
        instance = _parser_for(tmp_path, "vendor", payload)
        assert [c.control_id for c in instance.parse()] == list(CONTROL_IDS)


class TestParse:
    def test_title_is_the_heading_after_the_code(
        self, parser: OwaspProactiveControlsParser,
    ) -> None:
        assert [c.title for c in parser.parse()] == list(TITLES.values())

    def test_description_is_the_description_section_only(
        self, parser: OwaspProactiveControlsParser,
    ) -> None:
        text = parser.parse()[0].description
        assert text.startswith("Implement Access Control is the control")
        assert "An attacker abuses" not in text
        assert "Do the thing properly up front" not in text

    def test_full_text_is_the_whole_entry_below_the_heading(
        self, parser: OwaspProactiveControlsParser,
    ) -> None:
        control = parser.parse()[0]
        assert control.full_text is not None
        assert control.full_text.startswith("## Description")
        # The heading line is consumed, not carried.
        assert "# C1:" not in control.full_text
        # The sections below Description survive, which is what makes the
        # entry worth more than the Description on its own.
        assert "An attacker abuses" in control.full_text
        assert "## Tools" in control.full_text

    def test_ids_are_ordered_numerically_rather_than_lexically(
        self, parser: OwaspProactiveControlsParser,
    ) -> None:
        """Sorted member names put c10 between c1 and c2."""
        assert [c.control_id for c in parser.parse()][-1] == "C10"

    def test_no_control_declares_alternate_keys(
        self, parser: OwaspProactiveControlsParser,
    ) -> None:
        """Every section_name is the section_id, so no alternate can help.

        TestOpenCreLinkShape derives that claim from the tracked link file.
        """
        assert all(c.metadata is None for c in parser.parse())

    def test_hierarchy_level_is_control(
        self, parser: OwaspProactiveControlsParser,
    ) -> None:
        assert {c.hierarchy_level for c in parser.parse()} == {"control"}


class TestGuards:
    def test_a_short_catalogue_is_refused(self, tmp_path: Path) -> None:
        """The band would accept 9 of 10. The exact set does not.

        COUNT_TOLERANCE is 0.10 and abs(9 - 10) / 10 is 0.1, so
        _check_expected_count would pass a parser that lost a control. These
        ten carry 7.6 curated links each.
        """
        codes = tuple(c for c in TITLES if c != "C7")
        instance = _parser_for(tmp_path, "short", _archive(codes))
        with pytest.raises(ValueError, match="did not read control id"):
            instance.parse()

    def test_an_id_outside_the_ten_is_refused(self, tmp_path: Path) -> None:
        payload = _archive(extra={
            f"{ROOT}/docs/the-top-10/c11-renumbered.md":
                "# C11: Renumbered\n\n## Description\n\nA new control.\n",
        })
        instance = _parser_for(tmp_path, "extra", payload)
        with pytest.raises(ValueError, match="outside C1..C10"):
            instance.parse()

    def test_a_missing_heading_is_refused(self, tmp_path: Path) -> None:
        payload = _archive(extra={
            f"{ROOT}/docs/the-top-10/c1-accesscontrol.md":
                "## Description\n\nNo H1 at all.\n",
        })
        instance = _parser_for(tmp_path, "noheading", payload)
        with pytest.raises(ValueError, match="no '# Cn: Title' heading"):
            instance.parse()

    def test_a_missing_description_section_is_refused(
        self, tmp_path: Path,
    ) -> None:
        payload = _archive(extra={
            f"{ROOT}/docs/the-top-10/c1-accesscontrol.md":
                "# C1: Implement Access Control\n\n## Threats\n\nNo body.\n",
        })
        instance = _parser_for(tmp_path, "nodesc", payload)
        with pytest.raises(ValueError, match="no '## Description' section"):
            instance.parse()

    def test_a_trailing_description_is_bounded_by_the_next_heading(
        self, tmp_path: Path,
    ) -> None:
        """With `\\Z` alone the pattern would swallow the rest of the entry."""
        payload = _archive(extra={
            f"{ROOT}/docs/the-top-10/c1-accesscontrol.md":
                "# C1: Implement Access Control\n\n## Description\n\n"
                "A statement of the control that is long enough to count as "
                "prose against the sixty character floor.\n\n"
                "## Threats\n\nAn attacker abuses the missing control.\n",
        })
        instance = _parser_for(tmp_path, "bounded", payload)
        control = instance.parse()[0]
        assert "An attacker abuses" not in control.description
        assert "An attacker abuses" in (control.full_text or "")

    def test_a_different_archive_is_refused(
        self, parser: OwaspProactiveControlsParser,
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
            if source.framework_id == FRAMEWORK_ID
        }
        assert pins == {OwaspProactiveControlsParser.expected_sha256}


class TestTheDescriptionCapProtectsTheAnchor:
    """_sanitize_control rewrites full_text when description overflows.

    Six of the ten real Descriptions sanitise past DESCRIPTION_MAX_LENGTH:
    C2 at 2,905, C3 at 2,325, C4 at 2,239, C7 at 5,725, C8 at 2,019 and C9 at
    2,678. [measured on the pinned archive] Each one would hand full_text back
    to the base class, which replaces the entry with the Description. The
    parser caps description on a paragraph boundary, which is the only way a
    parser can stop that rewrite.

    This class builds the overflow rather than asserting on stored lengths.
    A capped artifact reads as comfortably inside the limit whether the cap is
    the parser's or the base class's, so the stored number cannot tell them
    apart.
    """

    SENTENCE = (
        "Access control enforces the policy the application claims to "
        "enforce across every request path it exposes to a caller."
    )

    @classmethod
    def _body(cls, paragraphs: int, sentences: int = 1, pad: int = 0) -> str:
        """`pad` shifts the budget boundary off a space.

        The sentence is 118 characters, so an unpadded run of them puts a
        space at exactly DESCRIPTION_BUDGET - 1 and a word cut lands where a
        hard cut lands. That coincidence let two wrong implementations pass.
        One padding character moves the boundary into the middle of a token.
        """
        one = " ".join([cls.SENTENCE] * sentences)
        body = "\n\n".join([one] * paragraphs)
        return ("x" * pad + " " + body) if pad else body

    @classmethod
    def _overflowing(
        cls, paragraphs: int = 24, sentences: int = 1, pad: int = 0,
        *, over: int = DESCRIPTION_MAX_LENGTH,
    ) -> bytes:
        body = cls._body(paragraphs, sentences, pad)
        assert len(body) > over, len(body)
        entry = (
            "# C1: Implement Access Control\n\n## Description\n\n"
            f"{body}\n\n## Threats\n\nAn attacker abuses the missing "
            "control.\n\n## Tools\n\n- A scanner.\n"
        )
        return _archive(extra={
            f"{ROOT}/docs/the-top-10/c1-accesscontrol.md": entry,
        })

    def test_an_overflowing_description_no_longer_displaces_the_entry(
        self, tmp_path: Path,
    ) -> None:
        instance = _parser_for(tmp_path, "long", self._overflowing())
        (tmp_path / "out").mkdir()
        controls = {c.control_id: c for c in instance.run().controls}

        overflowed = controls["C1"]
        assert overflowed.full_text is not None
        # "## Tools" is in the entry and not in the Description, so its
        # presence is the proof that the entry survived.
        assert "## Tools" in overflowed.full_text
        assert len(overflowed.description) < DESCRIPTION_MAX_LENGTH

    def test_the_cap_lands_on_a_paragraph_boundary(
        self, tmp_path: Path,
    ) -> None:
        instance = _parser_for(tmp_path, "long", self._overflowing())
        control = {c.control_id: c for c in instance.parse()}["C1"]
        assert control.description.endswith("to a caller.")
        assert len(control.description) <= DESCRIPTION_BUDGET
        # Whole paragraphs, so the survivors are separated as they were.
        assert control.description.count("\n\n") >= 1

    def test_a_single_paragraph_over_budget_is_cut_on_a_word_boundary(
        self, tmp_path: Path,
    ) -> None:
        """Whole-paragraph packing alone would yield an empty description."""
        payload = self._overflowing(paragraphs=1, sentences=20, pad=1)
        instance = _parser_for(tmp_path, "onepara", payload)
        control = {c.control_id: c for c in instance.parse()}["C1"]
        body = self._body(1, 20, pad=1)

        assert 0 < len(control.description) <= DESCRIPTION_BUDGET
        assert "\n\n" not in control.description
        # A prefix of the source that stops where a space was, so the last
        # token is whole rather than a fragment.
        assert body.startswith(control.description)
        assert body[len(control.description)] == " "
        # And not the hard cut, which ends "Access contro". Without this the
        # two are the same string whenever the boundary lands on a space.
        assert control.description != body[:DESCRIPTION_BUDGET].strip()

    def test_a_description_between_the_budget_and_the_limit_is_still_capped(
        self, tmp_path: Path,
    ) -> None:
        """The headroom is the ruling, so the headroom needs a test.

        A cap written against DESCRIPTION_MAX_LENGTH rather than
        DESCRIPTION_BUDGET passes every other test here: the overflow fixtures
        are far past both numbers. This one sits between them, which is the
        only place the two differ.
        """
        payload = self._overflowing(paragraphs=16, over=DESCRIPTION_BUDGET)
        instance = _parser_for(tmp_path, "between", payload)
        control = {c.control_id: c for c in instance.parse()}["C1"]

        assert len(self._body(16)) > DESCRIPTION_BUDGET
        assert len(self._body(16)) < DESCRIPTION_MAX_LENGTH
        assert len(control.description) <= DESCRIPTION_BUDGET
        assert [
            r["control_id"] for r in _audit_records(tmp_path)
            if r["repair"] == "description_capped_to_protect_full_text"
        ] == ["C1"]

    def test_the_cap_is_recorded_as_a_before_and_after_pair(
        self, tmp_path: Path,
    ) -> None:
        instance = _parser_for(tmp_path, "long", self._overflowing())
        instance.parse()
        capped = [
            r for r in _audit_records(tmp_path)
            if r["repair"] == "description_capped_to_protect_full_text"
        ]
        assert [r["control_id"] for r in capped] == ["C1"]
        assert len(str(capped[0]["before"])) > len(str(capped[0]["after"]))
        assert str(capped[0]["after"]) in str(capped[0]["before"])
        assert capped[0]["field"] == "description"

    def test_a_description_inside_the_limit_is_not_capped(
        self, parser: OwaspProactiveControlsParser, tmp_path: Path,
    ) -> None:
        parser.parse()
        assert _audit_records(tmp_path) == []

    def test_the_audit_is_written_even_when_nothing_fired(
        self, parser: OwaspProactiveControlsParser, tmp_path: Path,
    ) -> None:
        """A missing file must mean the parser never ran."""
        parser.parse()
        assert (tmp_path / "audit" / f"{FRAMEWORK_ID}.jsonl").exists()


class TestRun:
    def test_run_writes_and_clears_the_prose_floor(
        self, parser: OwaspProactiveControlsParser, tmp_path: Path,
    ) -> None:
        (tmp_path / "out").mkdir()
        output = parser.run()
        assert len(output.controls) == 10
        assert (tmp_path / "out" / f"{FRAMEWORK_ID}.json").exists()
        assert BaseParser.honest_prose_fraction(output.controls) == 1.0
        assert [s.path for s in output.source_files] == [ARCHIVE_NAME]

    def test_a_title_length_statement_trips_the_prose_floor(
        self, tmp_path: Path,
    ) -> None:
        """Nothing else here proves the floor is set to a non-zero value."""
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            for code, title in TITLES.items():
                archive.writestr(
                    f"{ROOT}/docs/the-top-10/{STEMS[code]}.md",
                    f"# {code}: {title}\n\n## Description\n\nShort.\n",
                )
        instance = _parser_for(tmp_path, "thin", payload.getvalue())
        (tmp_path / "out").mkdir()
        with pytest.raises(ValueError, match="below the declared floor"):
            instance.run()

    def test_the_anchor_survives_sanitisation(
        self, parser: OwaspProactiveControlsParser, tmp_path: Path,
    ) -> None:
        """ProseIndex prefers full_text, so full_text is what the model reads."""
        (tmp_path / "out").mkdir()
        for control in parser.run().controls:
            assert control.full_text is not None
            assert "An attacker abuses" in control.full_text
            assert len(control.full_text) > len(control.description)

    def test_output_is_byte_identical_on_a_re_run(
        self, parser: OwaspProactiveControlsParser, tmp_path: Path,
    ) -> None:
        (tmp_path / "out").mkdir()
        path = tmp_path / "out" / f"{FRAMEWORK_ID}.json"
        parser.run()
        first = path.read_bytes()
        parser.run()
        assert path.read_bytes() == first


class TestOpenCreLinkShape:
    """Derived from the tracked link file, so it fails in both directions."""

    def test_every_curated_link_names_its_own_section_id(self) -> None:
        """The reason no alt_titles table exists.

        All 76 links carry section_name == section_id, so the title channel
        cannot answer and no alternate title could make it. If OpenCRE ever
        respells one as a control name, this fails and the parser needs the
        alt_titles table Task 4 established. A stale table that stopped being
        needed fails here too, because the parser declares no metadata and
        TestParse asserts that.
        """
        rows = _curated_links()
        assert len(rows) == 76
        divergent = {
            (row["section_id"], row["section_name"])
            for row in rows
            if str(row["section_name"]).strip() != str(row["section_id"]).strip()
        }
        assert divergent == set()

    def test_the_links_reach_all_ten_controls(self) -> None:
        rows = _curated_links()
        assert {row["section_id"] for row in rows} == set(CONTROL_IDS)

    def test_detector_b_is_inert_because_the_name_is_the_id(self) -> None:
        """`wrong_anchor_risk == 0` has to mean something.

        Detector B only fires where the link carries a name that says
        something the id does not. Here it never does, so this framework has
        no entry in JOIN_WRONG_ANCHOR_BUDGET and reports a zero over a zero
        denominator. Recording why keeps that zero from reading as a pass.
        """
        rows = _curated_links()
        assert all(
            str(row["section_name"]).strip().casefold()
            == str(row["section_id"]).strip().casefold()
            for row in rows
        )


class TestRealCorpus:
    """The artifact this parser ships and the join it buys.

    Reads only tracked files, so it holds in a checkout with no data/raw.
    """

    def test_the_artifact_is_tracked_rather_than_overlay_routed(self) -> None:
        """CC-BY-SA-4.0 is in no restricted tier, so the file ships in git."""
        assert FRAMEWORK_ID not in OVERLAY_FRAMEWORK_IDS
        assert (PROCESSED_FRAMEWORKS_DIR / f"{FRAMEWORK_ID}.json").exists()

    def test_the_shipped_artifact_is_ten_controls_of_prose(self) -> None:
        data = json.loads(
            (PROCESSED_FRAMEWORKS_DIR / f"{FRAMEWORK_ID}.json").read_text(
                encoding="utf-8",
            )
        )
        controls = data["controls"]
        assert data["framework_id"] == FRAMEWORK_ID
        assert data["mapping_unit_level"] == "control"
        assert [c["control_id"] for c in controls] == list(CONTROL_IDS)
        assert [c["title"] for c in controls] == list(TITLES.values())
        assert all(c["full_text"] for c in controls)

    def test_no_control_ships_a_displaced_anchor(self) -> None:
        """Six of ten would fail this without the cap. [measured]

        Every entry carries a Threats heading below its Description, so its
        absence would mean the base class had replaced full_text with the
        Description.
        """
        displaced = {
            key for key, control in _shipped().items()
            if "## Threats" not in (control["full_text"] or "")
        }
        assert displaced == set()

    def test_no_description_reaches_the_limit_that_rewrites_full_text(
        self,
    ) -> None:
        """Ruling R14: the parser owns its anchor, not a two-character margin."""
        lengths = {
            key: len(control["description"])
            for key, control in _shipped().items()
        }
        assert max(lengths.values()) <= DESCRIPTION_BUDGET, lengths
        assert DESCRIPTION_BUDGET < DESCRIPTION_MAX_LENGTH

    def test_every_anchor_opens_on_the_description_section(self) -> None:
        for key, control in _shipped().items():
            assert (control["full_text"] or "").startswith("## Description"), key

    def test_the_shared_anchor_prefix_stays_negligible(self) -> None:
        """Ruling R13, measured rather than assumed.

        The ten prepared anchors share 12 leading characters, "Description "
        surviving strip_markup's removal of the `##` marker. That is 0.6% of
        the 2,150 character budget, against the 364 characters (17%) that made
        the Top 10's Factors table worth removing structurally. Pinned so a
        source edition that grows a shared header shows up as a failure rather
        than as a quiet loss of budget.
        """
        anchors = [
            prepare_anchor(control["full_text"] or "")[0]
            for control in _shipped().values()
        ]
        assert len(anchors) == 10
        assert _shared_prefix(anchors) == 12
