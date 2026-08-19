"""Parser for the OWASP Proactive Controls, current mkdocs edition.

Ten controls, C1 through C10, one markdown file each under
`docs/the-top-10/`. Every one of the 76 curated links carries a section_id of
`C1`..`C10` and a section_name that is the same two or three characters, so the
join runs entirely through the id channel and the title channel never fires.
That is also why this parser declares no alt_titles table: an alternate title
can only help where a curator wrote a name, and here the name is the id.
tests/test_parse_owasp_proactive_controls.py derives that claim from the
tracked link file, so a respelling upstream fails rather than passes quietly.

This framework contributed zero training links until the gate moved to the
resolved anchor. It used to be named in a framework deny list, and every one
of its section_names is shorter than the ten-character floor that also applied
to the title. Both retired gates tested a title. All 76 links now train, on
the prose this parser produces, because the gate reads
PHASE1B_MIN_ANCHOR_TEXT_LENGTH against the anchor the encoder is handed.

Three trees in the pinned archive carry the `c<N>-` filename pattern, not two.
[measured]

    docs/the-top-10/              the current edition, 10 members
    docs/archive/2018/            the superseded v3 wording, 10 members under
                                  different stems (c1-security-requirements)
    docs/archive/2024/the-top-10/ a near-copy of the current tree, 10 members
                                  under IDENTICAL stems

The third is the one that bites. A member filter written on the filename, or
on `the-top-10/c\\d+-`, reads twenty files and emits every control id twice,
which the exact-set completeness check below cannot see because the set is
still C1..C10. MEMBER therefore anchors on `docs/the-top-10/` as a whole path
segment. `v2/` and `v3/` hold the binary exports (pptx, pdf, docx) and no
markdown at all, so they were never the hazard.

`description` is the `## Description` section, capped on a paragraph boundary
at DESCRIPTION_BUDGET. The cap is not cosmetic. `BaseParser._sanitize_control`
calls `sanitize_text(description, return_full=True)` and assigns whatever that
returns to `full_text`, keeping the parser's own value only when the
description fits DESCRIPTION_MAX_LENGTH. Six of the ten Descriptions sanitise
past that limit -- C2 at 2,905, C3 at 2,325, C4 at 2,239, C7 at 5,725, C8 at
2,019 and C9 at 2,678 [measured] -- so without the cap six of ten anchors
would silently become a truncated Description instead of the entry. Ruling R14:
the parser owns its anchor, so the headroom is declared and verified here
rather than left to whatever margin the source happens to leave.

`full_text` is the whole entry below the heading line, and `full_text` is
normally the anchor: ProseIndex prefers it over description unconditionally.
Measured on the pinned archive the ten entries run 2,002 to 16,381 characters
against a MAX_ANCHOR_CHARS budget of 2,150, so eight of the ten are cut at the
budget and the 61 links they carry are recorded as truncated. C9 and C10 fit.

Ruling R13 was checked and does not apply. The ten prepared anchors share 12
leading characters, "Description " left by strip_markup removing the `##`
marker, which is 0.6% of the anchor budget against the 364 characters (17%)
that made the Top 10's Factors table worth removing structurally. Nothing is
stripped here, and the 12 is pinned by a test so a shared header appearing
upstream shows up as a failure.
"""
from __future__ import annotations

import hashlib
import logging
import re
import zipfile
from io import BytesIO
from typing import ClassVar, Final

from tract.config import DESCRIPTION_MAX_LENGTH
from tract.parsers.base import BaseParser
from tract.sanitize import sanitize_text
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARCHIVE_NAME: Final[str] = "owasp_proactive_controls.zip"
SOURCE_SHA256: Final[str] = (
    "6db1aafd6ecd758f05cf6b4133ec7085eb95016ec41afc5f462b4683c603b19d"
)

# `docs/the-top-10/` as a whole path segment. `(?:^|/)` keeps a vendored
# mirror at `vendor/mirror-docs/the-top-10/` out, and requiring `the-top-10`
# to follow `docs/` directly keeps `docs/archive/2024/the-top-10/` out.
# `[^/]*` stops the stem spanning a directory separator.
MEMBER: Final[re.Pattern[str]] = re.compile(
    r"(?:^|/)docs/the-top-10/c\d+-[^/]*\.md$"
)
HEADING: Final[re.Pattern[str]] = re.compile(r"^#\s+(C\d+):\s*(.+?)\s*$", re.M)
DESCRIPTION: Final[re.Pattern[str]] = re.compile(
    r"^##\s+Description\s*$(.*?)(?=^##\s|\Z)", re.M | re.S
)
PARAGRAPH: Final[re.Pattern[str]] = re.compile(r"\n\s*\n")

CONTROL_IDS: Final[tuple[str, ...]] = tuple(f"C{n}" for n in range(1, 11))

# Headroom below DESCRIPTION_MAX_LENGTH, which is the length at which
# _sanitize_control discards this parser's full_text. 200 characters rather
# than a boundary hug: sanitisation is not guaranteed to shrink, because the
# ligature map expands (a single "ffl" glyph becomes three characters), so a
# cap at the limit itself is safe only by inspection of the current source.
# The largest packed description on the pinned archive is 1,825 characters, so
# the budget costs nothing on real input. [measured]
DESCRIPTION_BUDGET: Final[int] = DESCRIPTION_MAX_LENGTH - 200


class OwaspProactiveControlsParser(BaseParser):
    framework_id: ClassVar[str] = "owasp_proactive_controls"
    framework_name: ClassVar[str] = "OWASP Proactive Controls"
    version: ClassVar[str] = "2024"
    source_url: ClassVar[str] = "https://top10proactive.owasp.org/"
    mapping_unit_level: ClassVar[str] = "control"
    expected_count: ClassVar[int] = 10
    fetched_date: ClassVar[str] = "2026-08-15"
    # All ten descriptions run 529 to 5,725 characters before the cap and none
    # equals its title, so the attainable value is exactly 1.0 and a floor of
    # 1.0 fires the moment any control loses its Description section.
    # [measured]
    min_prose_fraction: ClassVar[float] = 1.0
    expected_sha256: ClassVar[str | None] = SOURCE_SHA256

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(ARCHIVE_NAME)
        self._check_digest(payload)
        controls: list[Control] = []
        repairs: list[dict[str, object]] = []
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            for name in sorted(n for n in archive.namelist() if MEMBER.search(n)):
                controls.append(
                    self.control_from_markdown(
                        archive.read(name).decode("utf-8"), name, repairs,
                    )
                )

        # Both directions, not just the superset check. `expected_count = 10`
        # against COUNT_TOLERANCE = 0.10 means nine controls gives a deviation
        # of exactly 0.1, and `0.1 <= 0.10` is True, so a short catalogue would
        # pass silently. These ten carry 7.6 curated links each, so one lost
        # control is roughly eight lost links.
        found = {c.control_id for c in controls}
        unknown = found - set(CONTROL_IDS)
        if unknown:
            raise ValueError(
                f"{self.framework_id}: read control id(s) {sorted(unknown)} "
                f"outside C1..C10. Either the edition renumbered or a decoy "
                f"directory reached the member filter."
            )
        missing = set(CONTROL_IDS) - found
        if missing:
            raise ValueError(
                f"{self.framework_id}: did not read control id(s) "
                f"{sorted(missing)}. The band around expected_count would let "
                f"a nine-control catalogue through without a word, so the "
                f"completeness check is the exact set, not the count."
            )

        # Written unconditionally, after the completeness check, so the file on
        # disk always describes a full ten-control run and a missing file means
        # the parser never ran rather than the cap never firing.
        self.write_repair_audit(repairs)
        # Numerically, so C10 sorts after C9 rather than after C1.
        return sorted(controls, key=lambda c: int(c.control_id[1:]))

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
                f"not the pinned {self.expected_sha256}. Every measured "
                f"figure in this parser is a reading of one archive."
            )

    @classmethod
    def control_from_markdown(
        cls,
        text: str,
        member: str,
        repairs: list[dict[str, object]] | None = None,
    ) -> Control:
        """One control from one markdown file.

        `repairs` is an optional sink. The description cap appends a
        before/after pair to it, which parse() hands to write_repair_audit. A
        count would say the cap fired; only the pair says what was dropped.

        Raises:
            ValueError: If the file has no heading or no Description section.
        """
        heading = HEADING.search(text)
        if heading is None:
            raise ValueError(
                f"owasp_proactive_controls: {member} has no '# Cn: Title' "
                f"heading. The code in that heading is the only identifier "
                f"OpenCRE links against."
            )
        control_id = heading.group(1)
        body = DESCRIPTION.search(text)
        if body is None:
            raise ValueError(
                f"owasp_proactive_controls: {control_id} in {member} has no "
                f"'## Description' section, so its statement would be the "
                f"Threats section, which describes the attack rather than "
                f"the control."
            )

        statement = body.group(1).strip()
        capped, was_capped = cls._cap_description(statement)
        if repairs is not None and was_capped:
            repairs.append({
                "control_id": control_id,
                "repair": "description_capped_to_protect_full_text",
                "field": "description",
                "before": statement,
                "after": capped,
            })
        return Control(
            control_id=control_id,
            title=heading.group(2).strip(),
            description=capped,
            full_text=text[heading.end():].strip(),
            hierarchy_level="control",
        )

    @staticmethod
    def _cap_description(text: str) -> tuple[str, bool]:
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
        capped, was_capped = OwaspProactiveControlsParser._pack(text)
        # Measurement, not mutation: `capped` is what run() will sanitise, and
        # a non-None second element is exactly the condition under which
        # _sanitize_control overwrites full_text.
        _, overflow = sanitize_text(
            capped, max_length=DESCRIPTION_MAX_LENGTH, return_full=True,
        )
        if overflow is not None:
            raise ValueError(
                f"owasp_proactive_controls: a description capped to "
                f"{len(capped)} characters still sanitises to "
                f"{len(overflow)}, past DESCRIPTION_MAX_LENGTH of "
                f"{DESCRIPTION_MAX_LENGTH}. _sanitize_control would replace "
                f"full_text with it and the entry anchor would be lost. "
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
        if kept:
            return "\n\n".join(kept), True

        cut = text[:DESCRIPTION_BUDGET].rsplit(" ", 1)[0]
        # Mirrors sanitize_text's own guard: a body with no space in its first
        # half would otherwise be cut to almost nothing.
        if len(cut) < DESCRIPTION_BUDGET // 2:
            cut = text[:DESCRIPTION_BUDGET]
        return cut.strip(), True


def main() -> None:
    OwaspProactiveControlsParser().run()


if __name__ == "__main__":
    main()
