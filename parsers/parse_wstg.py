"""Parser for the OWASP Web Security Testing Guide.

The join key is the value in the two-row ID table under each test's H1, read
from that table rather than derived from the path: the directory prefix is a
zero-padded number (`01-Information_Gathering`) while the id prefix is a
four-letter mnemonic (`WSTG-INFO-01`).

OpenCRE sets section_name equal to section_id for all 118 curated links, so the
link side carries no human title at all. [measured] The parser's title is the
file's H1, which no link name spells, so the whole join runs through the id
channel and the title channel cannot misfire.

Nine of the 118 links can never resolve. Their section_ids -- WSTG-APPE-D,
WSTG-BUSL-$$, WSTG-INFO-## and WSTG-INPV-00 -- appear in none of the 199
markdown members of the pinned archive, carrying 2, 3, 1 and 3 links.
[measured] That is an upstream extraction artifact, not something a parser can
fix, and it sets this framework's ceiling at 109 of 118.

Census of the pinned archive. [measured]

    144  markdown members under document/4-Web_Application_Security_Testing/
    -14  category READMEs and the tree README
    130  test files
    -14  sub-tests that share a parent's id and carry no ID table of their own
    116  members carrying an ID table
     -1  WSTG-INPV-13 is the ID table value of two of them
    115  distinct ids
     -5  withdrawn tests that name a successor, aliased rather than emitted
    110  controls

A decoy tree carries the same ID table shape. `template/999-Foo_Testing/` holds
three files whose tables read WSTG-FOO-001 through WSTG-FOO-003. [measured] An
id-shaped filter alone reads them, so MEMBER anchors on the
`document/4-Web_Application_Security_Testing/` path as a whole and the id
pattern is never asked to carry the exclusion.

Eight members are withdrawn tests rather than tests. Their body holds a
sentence saying the content moved or was removed and nothing else, and the
structural signal is exact: those eight and only those eight carry no `##`
section heading at all. [measured] Five of the eight name their successor in a
machine-readable `[merged]: # (WSTG-XXXX-NN)` trailer.

Those five are the reason this parser emits 110 controls rather than 115. Three
of them are linked -- WSTG-ATHN-01, WSTG-INPV-03 and WSTG-ERRH-02 -- and
emitting them as controls would hand three curated links an anchor reading
"This content has been merged into: Test HTTP Methods". That resolves, counts
toward the 109, and trains on a redirect notice. The id is therefore declared
as an `alt_ids` entry on the successor's control, which is what that channel
was built for: the retired id reaches the successor's real prose, no second
copy of that prose enters the corpus, and `distinct_anchors` sees one anchor
rather than two spellings of one. Dropping them instead would push resolution
to 106 of 118, below the floor, and discard something the source states.

The remaining three withdrawn tests name no successor. WSTG-CONF-08 and
WSTG-CLNT-08 carry no curated link and ship as controls marked damaged, so the
prose floor measures the tests that have text rather than averaging over
withdrawals. The third is a member of WSTG-INPV-13, whose other member has
prose, so it contributes no text to that merge.

WSTG-INPV-13 is the ID table value of both `13-Testing_for_Buffer_Overflow.md`
and `13-Testing_for_Format_String_Injection.md`. They become one control,
because the id is what the link targets and emitting two controls with the same
control_id would let whichever came last silently win. A merge assembles text
across a file boundary, so the control carries `text_origin: synthetic` and the
merge writes an audit record naming both members and quoting both statements.

The statement is the body from the end of the ID table to the first cut
heading. The cut list is REMEDIATION_HEADINGS plus WSTG's own three procedural
sections, and it matches case-insensitively.
The casing matters on real input: `13-Test_for_Path_Confusion.md` spells the
heading `## How To Test`, and a case-sensitive cut leaves that statement at
2,106 characters of which 1,545 are the test procedure, against 561 characters
of statement. [measured] One control, and the whole point of the cut rule.

Ruling R14 is why `description` is capped. `BaseParser._sanitize_control` calls
`sanitize_text(description, return_full=True)` and assigns the result to
`full_text`, keeping the parser's own value only when the description fits
DESCRIPTION_MAX_LENGTH. 45 of the 115 statements sanitise past that limit
[measured], so without the cap 45 anchors would silently become a truncated
description. The parser owns its anchor, so the headroom is declared and
verified here rather than left to whatever margin the source happens to leave.

`full_text` is set only where the cap actually dropped text, matching
`sanitize_text`'s own convention that a full form exists only when the short
one is short. ProseIndex prefers full_text over description unconditionally, so
either way the anchor is the whole statement up to the MAX_ANCHOR_CHARS budget,
and 38 of the statements exceed that budget and are cut there. [measured]
Setting full_text to the whole file body instead would put the test procedure
in the anchor and truncate 104 of 115 rather than 38.

Ruling R13 was checked and nothing is stripped. Across all prepared anchors the
shared leading prefix is 0 characters, because the withdrawn tests open on
their notice. Across the 107 that open on a section it is 8 characters,
"Summary " left by strip_markup removing the `##` marker, which is 0.4% of the
2,150 character budget against the 364 characters (17%) that made the Top 10's
Factors table worth removing structurally. [measured] Both numbers are pinned
by tests so a shared header appearing upstream shows up as a failure.
"""
from __future__ import annotations

import hashlib
import logging
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from typing import ClassVar, Final

from tract.config import (
    CONTROL_DAMAGE_REASON_METADATA_KEY,
    CONTROL_DAMAGED_METADATA_KEY,
    CONTROL_DAMAGED_METADATA_VALUE,
    DESCRIPTION_MAX_LENGTH,
    REMEDIATION_HEADINGS,
)
from tract.corpus_report import SYNTHETIC_TEXT_ORIGIN, TEXT_ORIGIN_METADATA_KEY
from tract.parsers.base import BaseParser
from tract.sanitize import sanitize_text
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARCHIVE_NAME: Final[str] = "wstg.zip"
SOURCE_SHA256: Final[str] = (
    "e093f1648fbf4195f2a8fccac4f80315fb6b6281af85aa557edb34d0f9c58b33"
)

# The testing tree at its one legal depth. `^(?:[^/]+/)?` allows the archive's
# single commit-hash root directory and nothing deeper, so a vendored mirror at
# `vendor/copy/document/4-.../` is not a member: an unanchored `/document/`
# would read it. `[^/]+/` is the category directory and `[^/]*` stops the stem
# spanning a further separator, so a deeper tree cannot arrive through the same
# filter either. The `README.md` lookahead drops the 13 category READMEs. The
# tree's own README sits one level up and fails the category-directory group
# instead.
MEMBER: Final[re.Pattern[str]] = re.compile(
    r"^(?:[^/]+/)?document/4-Web_Application_Security_Testing/"
    r"[^/]+/(?!README\.md$)[^/]*\.md$"
)
ID_TABLE: Final[re.Pattern[str]] = re.compile(
    r"^\|\s*(WSTG-[A-Z]+-\d+)\s*\|\s*$", re.M
)
H1: Final[re.Pattern[str]] = re.compile(r"^#\s+(.+?)\s*$", re.M)

# A withdrawn test is one whose body holds no section at all. Structural rather
# than a match on the withdrawal sentence, whose wording differs across the
# eight ("has been removed.", "has been removed", "has been merged into:",
# "has been merged into WSTG-IDNT-04"). [measured]
SECTION: Final[re.Pattern[str]] = re.compile(r"^##\s+\S", re.M)
# The successor a withdrawn test names, in the trailer upstream writes as a
# markdown comment so it renders as nothing and still parses.
REDIRECT: Final[re.Pattern[str]] = re.compile(
    r"^\[merged\]:\s*#\s*\((WSTG-[A-Z]+-\d+)\)\s*$", re.M
)

# Where the statement ends. REMEDIATION_HEADINGS supplies Remediation,
# References and the rest. These three are WSTG's own procedural sections and
# say how to run the test rather than what the test is for.
_EXTRA_CUTS: Final[tuple[str, ...]] = (
    "How to Test", "Related Test Cases", "Tools",
)
# Case-insensitive because the source is. See the module docstring on
# `## How To Test`.
CUT: Final[re.Pattern[str]] = re.compile(
    r"^##\s+(?:"
    + "|".join(re.escape(h) for h in (*REMEDIATION_HEADINGS, *_EXTRA_CUTS))
    + r")\s*$",
    re.M | re.I,
)
PARAGRAPH: Final[re.Pattern[str]] = re.compile(r"\n\s*\n")

# Headroom below DESCRIPTION_MAX_LENGTH, which is the length at which
# _sanitize_control discards this parser's full_text. 200 characters rather
# than a boundary hug: sanitisation is not guaranteed to shrink, because the
# ligature map expands (a single "ffl" glyph becomes three characters), so a
# cap at the limit itself is safe only by inspection of the current source.
# Matches the value Task 6 established for the same base-class hazard.
DESCRIPTION_BUDGET: Final[int] = DESCRIPTION_MAX_LENGTH - 200


@dataclass(frozen=True)
class SourceMember:
    """One markdown file that carries an ID table.

    Attributes:
        test_id: The WSTG id read from the file's two-row ID table.
        member: The member name inside the archive, kept for audit records.
        text: The file's full markdown.
    """

    test_id: str
    member: str
    text: str

    @property
    def title(self) -> str:
        """The file's H1.

        Raises:
            ValueError: If the file has no H1.
        """
        heading = H1.search(self.text)
        if heading is None:
            raise ValueError(
                f"wstg: {self.member} carries an ID table but no H1. The H1 is "
                f"the control title, and OpenCRE supplies none of its own for "
                f"this framework: every one of its 118 links sets section_name "
                f"equal to section_id."
            )
        return heading.group(1).strip()

    @property
    def body(self) -> str:
        """Everything below the ID table."""
        table = ID_TABLE.search(self.text)
        return self.text[table.end():].strip() if table else self.text.strip()

    @property
    def is_withdrawn(self) -> bool:
        """Whether this file is a withdrawal notice rather than a test."""
        return SECTION.search(self.body) is None

    @property
    def successor(self) -> str | None:
        """The id this withdrawal notice redirects to, or None."""
        marker = REDIRECT.search(self.body)
        return marker.group(1) if marker else None

    @property
    def statement(self) -> str:
        """The body up to the first procedural heading.

        Section headings are kept rather than stripped. `strip_markup` removes
        them on the way to the encoder, so the anchor is the same either way,
        and keeping them leaves the stored statement inspectable: a test can
        assert `## How to Test` never reaches an artifact, which a stripped
        form cannot distinguish from the prose mention of that section name
        that WSTG-ATHZ-02 carries in a sentence. [measured]
        """
        body = self.body
        cut = CUT.search(body)
        return (body[: cut.start()] if cut else body).strip()


class WstgParser(BaseParser):
    framework_id: ClassVar[str] = "wstg"
    # canonical_framework maps OpenCRE's "OWASP Web Security Testing Guide
    # (WSTG)" onto this through the existing FRAMEWORK_NAME_ALIASES entry.
    framework_name: ClassVar[str] = "WSTG"
    version: ClassVar[str] = "4.2"
    source_url: ClassVar[str] = (
        "https://owasp.org/www-project-web-security-testing-guide/"
    )
    mapping_unit_level: ClassVar[str] = "test"
    # 115 distinct ids over 116 members, less the five withdrawn tests that
    # alias onto their successor instead of shipping a redirect notice as an
    # anchor. [measured] See the census in the module docstring.
    expected_count: ClassVar[int] = 110
    fetched_date: ClassVar[str] = "2026-08-15"
    # Exactly 1.0 is attainable and is the honest floor. The two withdrawn
    # tests that ship as controls are marked damaged, which takes them out of
    # both sides of the ratio, so every control this fraction measures is one
    # with a statement. Anything below 1.0 means a test with prose lost it.
    min_prose_fraction: ClassVar[float] = 1.0
    expected_sha256: ClassVar[str | None] = SOURCE_SHA256

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(ARCHIVE_NAME)
        self._check_digest(payload)
        members = self._read_members(payload)
        controls, audit = self.build_controls(members)
        self.write_repair_audit(audit)
        logger.info(
            "%s: %d controls from %d members carrying an ID table",
            self.framework_id, len(controls), len(members),
        )
        return controls

    def _check_digest(self, payload: bytes) -> None:
        """Refuse an archive that is not the pinned one.

        Raises:
            ValueError: If the digest does not match `expected_sha256`.
        """
        if self.expected_sha256 is None:
            return
        actual = hashlib.sha256(payload).hexdigest()
        if actual != self.expected_sha256:
            raise ValueError(
                f"{self.framework_id}: {ARCHIVE_NAME} has sha256 {actual}, "
                f"not the pinned {self.expected_sha256}. Every measured figure "
                f"in this parser is a reading of one archive."
            )

    def _read_members(self, payload: bytes) -> list[SourceMember]:
        """Every archive member that carries an ID table, in member order.

        Raises:
            ValueError: If no member carries an ID table at all.
        """
        members: list[SourceMember] = []
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            for name in sorted(n for n in archive.namelist() if MEMBER.search(n)):
                text = archive.read(name).decode("utf-8")
                table = ID_TABLE.search(text)
                if table is None:
                    logger.debug(
                        "%s: %s has no ID table. It is a sub-test that "
                        "shares its parent's id", self.framework_id, name,
                    )
                    continue
                members.append(SourceMember(table.group(1), name, text))
        if not members:
            raise ValueError(
                f"{self.framework_id}: no member of {ARCHIVE_NAME} carries a "
                f"WSTG-XXXX-NN ID table. The table is the only join key. The "
                f"path prefix does not spell the id."
            )
        return members

    @classmethod
    def build_controls(
        cls, members: list[SourceMember],
    ) -> tuple[list[Control], list[dict[str, object]]]:
        """One Control per live id, plus the audit trail for every repair.

        Three repairs can fire, and each one writes what moved rather than
        that something moved: a withdrawn test aliased onto its successor, two
        files merged under one id, and a description capped to keep the base
        class from overwriting the anchor.

        Raises:
            ValueError: If a withdrawn test names a successor that no live id
                spells, or names one that is itself withdrawn.
        """
        grouped: dict[str, list[SourceMember]] = defaultdict(list)
        for member in members:
            grouped[member.test_id].append(member)

        redirects = cls._redirects(grouped)
        audit: list[dict[str, object]] = []
        controls: list[Control] = []

        for test_id in sorted(grouped):
            if test_id in redirects:
                continue
            controls.append(
                cls._control(test_id, sorted(
                    grouped[test_id], key=lambda m: m.member,
                ), redirects, audit)
            )

        cls._record_redirects(grouped, redirects, controls, audit)
        return controls, audit

    @staticmethod
    def _redirects(
        grouped: dict[str, list[SourceMember]],
    ) -> dict[str, str]:
        """Withdrawn id -> successor id, for ids with no live member left.

        An id keeps its own control when any of its members still has prose,
        so WSTG-INPV-13 is not a redirect even though one of its two files is
        a withdrawal notice.

        Raises:
            ValueError: If a successor is missing or itself withdrawn.
        """
        redirects: dict[str, str] = {}
        for test_id, members in grouped.items():
            if any(not m.is_withdrawn for m in members):
                continue
            successors = [m.successor for m in members if m.successor]
            if not successors:
                continue
            redirects[test_id] = successors[0]

        for test_id, successor in sorted(redirects.items()):
            if successor not in grouped:
                raise ValueError(
                    f"wstg: withdrawn test {test_id} redirects to {successor}, "
                    f"which no member of the archive spells. The redirect "
                    f"would drop {test_id} out of the corpus with nothing to "
                    f"reach in its place."
                )
            if successor in redirects:
                raise ValueError(
                    f"wstg: withdrawn test {test_id} redirects to {successor}, "
                    f"which is itself withdrawn and redirects to "
                    f"{redirects[successor]}. This parser resolves one hop, so "
                    f"a chain has to be read and declared rather than followed."
                )
        return redirects

    @classmethod
    def _control(
        cls,
        test_id: str,
        members: list[SourceMember],
        redirects: dict[str, str],
        audit: list[dict[str, object]],
    ) -> Control:
        """One control from the members that share an id."""
        titles = [m.title for m in members]
        # A withdrawal notice inside a merge contributes nothing. Its sentence
        # is not a statement about the surviving test, and it would sit at the
        # front of the anchor.
        live = [m for m in members if not m.is_withdrawn]
        sourced = live or members
        statement = "\n\n".join(m.statement for m in sourced if m.statement)

        if len(members) > 1:
            audit.append({
                "control_id": test_id,
                "repair": "members_merged_under_one_id",
                "members": [m.member for m in members],
                "titles": titles,
                # The text itself, not its length. write_repair_audit's own
                # docstring says why: a count says a repair fired, not what
                # moved, and this is the file a reviewer reads to check one.
                "before": [m.statement for m in members],
                "after": statement,
                "withdrawn_members": [
                    m.member for m in members if m.is_withdrawn
                ],
                "reason": (
                    "one WSTG id is the ID table value of more than one file. "
                    "Emitting one control per file would make two controls "
                    "share a control_id and let the later one win"
                ),
            })

        description, capped = cls._cap_description(statement)
        if capped:
            audit.append({
                "control_id": test_id,
                "repair": "description_capped_to_protect_full_text",
                "field": "description",
                "before": statement,
                "after": description,
            })

        metadata: dict[str, str | list[str]] = {
            "source_members": [m.member for m in members],
        }
        alt_ids = sorted(k for k, v in redirects.items() if v == test_id)
        if alt_ids:
            # The retired id reaches this control's prose. ProseIndex applies
            # alternates in a second pass and never lets one displace a real
            # id, so this can only add reach.
            metadata["alt_ids"] = alt_ids
        if len(members) > 1:
            # A merged control's text was assembled by this parser, not read
            # from one file. The instrument reads text_origin to populate
            # anchor_source_synthetic. Without the key the column reads zero
            # and the synthetic arm is invisible, which is the defect the
            # column exists to expose.
            metadata[TEXT_ORIGIN_METADATA_KEY] = SYNTHETIC_TEXT_ORIGIN
        if not live:
            metadata[CONTROL_DAMAGED_METADATA_KEY] = CONTROL_DAMAGED_METADATA_VALUE
            metadata[CONTROL_DAMAGE_REASON_METADATA_KEY] = (
                "the source withdrew this test and names no successor, so its "
                "body is the withdrawal notice rather than a statement"
            )

        return Control(
            control_id=test_id,
            title=" / ".join(titles),
            description=description,
            # Only where the cap dropped text, matching sanitize_text's own
            # convention that a full form exists when the short one is short.
            full_text=statement if capped else None,
            hierarchy_level="test",
            parent_id=members[0].member.split("/")[-2],
            metadata=metadata,
        )

    @staticmethod
    def _record_redirects(
        grouped: dict[str, list[SourceMember]],
        redirects: dict[str, str],
        controls: list[Control],
        audit: list[dict[str, object]],
    ) -> None:
        """Write what each alias makes reachable, as text on both sides."""
        by_id = {c.control_id: c for c in controls}
        for test_id in sorted(redirects):
            successor = redirects[test_id]
            target = by_id[successor]
            audit.append({
                "control_id": test_id,
                "repair": "withdrawn_test_aliased_to_its_successor",
                "successor_id": successor,
                "members": [m.member for m in grouped[test_id]],
                "before": "\n\n".join(
                    m.statement for m in grouped[test_id] if m.statement
                ),
                "after": target.full_text or target.description,
                "reason": (
                    "the source withdrew this test and names its successor, so "
                    "the id is declared as an alt_id on that successor rather "
                    "than shipped as a control whose anchor is a redirect "
                    "notice"
                ),
            })

    @classmethod
    def _cap_description(cls, text: str) -> tuple[str, bool]:
        """Keep description clear of the length that discards full_text.

        Whole paragraphs first, because a cut between paragraphs leaves a
        statement that still reads as one. A single paragraph over the budget
        falls back to a word boundary, since dropping it whole would leave an
        empty description and fail the prose floor for a formatting change.

        The result is verified against the function that would do the damage
        rather than against a character count, so the guarantee does not rest
        on sanitisation happening to shrink this corpus.

        Returns the text and whether the cut fired.

        Raises:
            ValueError: If the capped text would still displace full_text.
        """
        capped, was_capped = cls._pack(text)
        # Measurement, not mutation: `capped` is what run() will sanitise, and
        # a non-None second element is exactly the condition under which
        # _sanitize_control overwrites full_text.
        _, overflow = sanitize_text(
            capped, max_length=DESCRIPTION_MAX_LENGTH, return_full=True,
        )
        if overflow is not None:
            raise ValueError(
                f"wstg: a description capped to {len(capped)} characters still "
                f"sanitises to {len(overflow)}, past DESCRIPTION_MAX_LENGTH of "
                f"{DESCRIPTION_MAX_LENGTH}. _sanitize_control would replace "
                f"full_text with it and the statement anchor would be lost. "
                f"Lower DESCRIPTION_BUDGET."
            )
        return capped, was_capped

    @staticmethod
    def _pack(text: str) -> tuple[str, bool]:
        """Whole paragraphs under DESCRIPTION_BUDGET, else a word cut."""
        if len(text) <= DESCRIPTION_BUDGET:
            return text, False

        kept: list[str] = []
        used = 0
        for paragraph in (p.strip() for p in PARAGRAPH.split(text)):
            if not paragraph:
                continue
            # The separator costs two characters in the joined form.
            cost = len(paragraph) + (2 if kept else 0)
            if used + cost > DESCRIPTION_BUDGET:
                break
            kept.append(paragraph)
            used += cost
        packed = "\n\n".join(kept)
        # Mirrors sanitize_text's own guard, and it is not decorative here. A
        # WSTG statement opens with its `## Summary` heading on its own line,
        # so the first paragraph is seven characters. A statement whose one
        # prose paragraph runs past the budget packs to "## Summary" alone,
        # which is not a description and fails the prose floor. Below half the
        # budget the word cut is the better answer.
        if len(packed) >= DESCRIPTION_BUDGET // 2:
            return packed, True

        cut = text[:DESCRIPTION_BUDGET].rsplit(" ", 1)[0]
        # Mirrors sanitize_text's own guard: a body with no space in its first
        # half would otherwise be cut to almost nothing.
        if len(cut) < DESCRIPTION_BUDGET // 2:
            cut = text[:DESCRIPTION_BUDGET]
        return cut.strip(), True


def main() -> None:
    WstgParser().run()


if __name__ == "__main__":
    main()
