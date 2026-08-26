"""Parser for the OWASP Top 10 2021.

The archive carries every Top 10 edition and every translation: 1,223 members,
710 of them markdown, 196 MB. [measured] Only the twelve
`2021/docs/en/A\\d\\d_2021-*.md` members are read. The path prefix is load
bearing rather than cosmetic, because twelve members share the filename
`A01_2021-Broken_Access_Control.md`, one per language, and a filter written on
the stem alone reads Arabic. [measured]

Twelve files match the A-prefix pattern. `A00` is *How to start an AppSec
Program* and `A11` is *Next Steps*; neither is a category and neither carries a
curated link. `A00` is excluded by its H1, which has no `A0N:2021` code; `A11`
is excluded by CATEGORY_IDS, because its H1 does carry one.

`description` is the `## Description` section, which drops `## Factors` (a
table of incidence and CVE counts), `## Overview` (release commentary about
where the category moved in the rankings) and the two remediation headings in
tract.config.REMEDIATION_HEADINGS that this framework is the original reason
for.

`full_text` is the entry below the heading line with the `## Factors` section
removed, and `full_text` is the anchor: ProseIndex prefers it over description
unconditionally. Measured on the pinned archive the ten stripped entries run
4,546 to 10,662 characters against a MAX_ANCHOR_CHARS budget of 2,150, so
every one of the 17 curated links resolves onto a truncated anchor and the
corpus report records `truncated == 17`.

Why the Factors section is removed. It is a markdown table of CWE counts,
incidence rates and CVE totals, 529 characters in every one of the ten
entries, and the first 364 characters of A01's and A03's anchors were
byte-identical because of it. [measured] That is about a fifth of the anchor
budget spent on text that is the same string in every category, which is
shared non-discriminative signal pulling ten category embeddings toward each
other -- the opposite of what a hub-assignment anchor is for. Removing it
takes the prefix the ten anchors share from 69 characters to 12, and every
anchor now opens on the Overview's prose about the risk. [measured]

The removal is structural, anchored on the literal `## Factors` heading and
bounded by the next `##` heading. A future edition that drops or renames the
section produces no match and this parser RAISES rather than passing the entry
through. Passing through would put a stats table back at the head of some
anchors and not others, which is the exact inconsistency this removal exists
to end, and a parser that already refuses a nine-category list should not
accept a silent change to what every anchor opens with.

This is a removal of publisher boilerplate, not an assembly of new text, so
`text_origin` stays unset. Do not set it to synthetic: every remaining
character is the publisher's, in the publisher's order, and the corpus report
uses that key to separate parser-written statements from publisher-written
ones.

`description` is capped at DESCRIPTION_MAX_LENGTH by this parser, on a word
boundary. That is not cosmetic. `_sanitize_control` calls
`sanitize_text(description, return_full=True)` and assigns whatever that
returns to `full_text`, keeping the parser's own value only when the
description fits. A02's Description sanitises to 2,263 characters and A04's to
2,938, both over the 2,000 limit, so before the cap those two shipped the
Description as their anchor and the entry was discarded. [measured] The cap
keeps every description inside the limit, so the base class cannot rewrite
`full_text` and all ten anchors are entries. Nothing is lost: the complete
Description text is inside `full_text`, which is the field the model reads.
Both transforms write a before/after record through `write_repair_audit`.

The found id tuple must equal CATEGORY_IDS exactly. COUNT_TOLERANCE is 10%, so
_check_expected_count accepts 9 categories of 10, and those ten carry 1.7
curated links each. The tuple check is what beats the band.

Three of the 17 curated links spell their category differently from the source
H1. See OPENCRE_TITLE_VARIANTS.
"""
from __future__ import annotations

import hashlib
import logging
import re
import zipfile
from collections.abc import Mapping
from io import BytesIO
from typing import ClassVar, Final

from tract.config import DESCRIPTION_MAX_LENGTH
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARCHIVE_NAME: Final[str] = "owasp_top10_2021.zip"
SOURCE_SHA256: Final[str] = (
    "7f4747a7d7958d58ae3a4c7f7329740b9363c4788655bc3f28da8fdbedf48b5d"
)

# Anchored on the language directory, not on the filename. `[^/]*` keeps the
# stem from spanning a directory separator.
MEMBER: Final[re.Pattern[str]] = re.compile(
    r"(?:^|/)2021/docs/en/A\d\d_2021-[^/]*\.md$"
)
# The en dash is the source's own separator. A hyphen and an em dash are
# accepted too, so a typographic edit upstream does not silently drop a
# category.
HEADING: Final[re.Pattern[str]] = re.compile(
    r"^#\s+(A\d\d):2021\s*[–—-]\s*(.+?)\s*$", re.M
)
DESCRIPTION: Final[re.Pattern[str]] = re.compile(
    r"^##\s+Description\s*$(.*?)(?=^##\s|\Z)", re.M | re.S
)
# Anchored on the literal heading and bounded by the next one, so a renamed or
# absent section matches nothing and the parser raises instead of cutting a
# span it did not identify. `\Z` is deliberately NOT an alternative here: a
# Factors section with no following heading would swallow the rest of the
# entry, which is the one failure this pattern must not have.
FACTORS: Final[re.Pattern[str]] = re.compile(
    r"^##\s+Factors\s*$.*?(?=^##\s)", re.M | re.S
)

CATEGORY_IDS: Final[tuple[str, ...]] = tuple(f"A{n:02d}" for n in range(1, 11))

# Category id -> the names OpenCRE's curated links spell for it, where those
# differ from the source H1. Declared as alt_titles so the link resolves
# through the title channel the curator wrote.
#
# Seven of the ten names match exactly and carry 11 of the 17 links, so without
# this table the join is 11 by title and 6 by id. [measured] All six still
# resolve, because every section_id is the category's own id. What they cost is
# the channel, and for A10 that costs more than a label: the id-side
# wrong-anchor detector compares the link name to the reached control's title
# by containment, and "Server Side Request Forgery (SSRF)" neither contains nor
# is contained by "Server-Side Request Forgery (SSRF)". A10 is therefore
# flagged, and owasp_top10_2021 has no entry in JOIN_WRONG_ANCHOR_BUDGET, so
# the flag is a build failure over a hyphen. The anchor is provably right: the
# id IS the category's own id.
#
#   A01  "Broken Access Controls"            plural, against the singular H1
#   A09  "Logging and Monitoring Failures"   drops the "Security" prefix
#   A10  "Server Side Request Forgery (SSRF)" drops the hyphen
#
# A01 and A09 are not flagged today, because each name and its title contain
# one another. They are declared anyway: the table is derived from the link
# file by tests/test_parse_owasp_top10_2021.py, which asserts it holds every
# divergence and no more, and a table of "the divergences that happen to trip
# a detector" cannot be derived that way.
OPENCRE_TITLE_VARIANTS: Final[Mapping[str, tuple[str, ...]]] = {
    "A01": ("Broken Access Controls",),
    "A09": ("Logging and Monitoring Failures",),
    "A10": ("Server Side Request Forgery (SSRF)",),
}


class OwaspTop102021Parser(BaseParser):
    framework_id: ClassVar[str] = "owasp_top10_2021"
    framework_name: ClassVar[str] = "OWASP Top 10 2021"
    version: ClassVar[str] = "2021"
    source_url: ClassVar[str] = "https://owasp.org/Top10/"
    mapping_unit_level: ClassVar[str] = "category"
    expected_count: ClassVar[int] = 10
    fetched_date: ClassVar[str] = "2026-08-15"
    # All ten descriptions run 582 to 2,944 characters and none equals its
    # title, so the attainable value is exactly 1.0 and a floor of 1.0 fires
    # the moment any category loses its Description section. [measured]
    min_prose_fraction: ClassVar[float] = 1.0
    expected_sha256: ClassVar[str | None] = SOURCE_SHA256
    title_variants: ClassVar[Mapping[str, tuple[str, ...]]] = (
        OPENCRE_TITLE_VARIANTS
    )

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(ARCHIVE_NAME)
        self._check_digest(payload)
        controls: list[Control] = []
        repairs: list[dict[str, object]] = []
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            for name in sorted(n for n in archive.namelist() if MEMBER.search(n)):
                text = archive.read(name).decode("utf-8")
                heading = HEADING.search(text)
                if heading is None:
                    logger.info(
                        "%s: %s has no A0N:2021 heading, so it is front "
                        "matter rather than a category", self.framework_id,
                        name.rsplit("/", 1)[-1],
                    )
                    continue
                if heading.group(1) not in CATEGORY_IDS:
                    logger.info(
                        "%s: skipping %s, which is numbered outside the ten "
                        "categories", self.framework_id, heading.group(1),
                    )
                    continue
                controls.append(
                    self.control_from_markdown(
                        text, self.title_variants, repairs,
                    )
                )
        found = tuple(c.control_id for c in controls)
        if found != CATEGORY_IDS:
            raise ValueError(
                f"{self.framework_id}: expected categories "
                f"{list(CATEGORY_IDS)}, found {list(found)}. A short list "
                f"would ship a partial Top 10, and at 9 of 10 the count "
                f"deviation is 10.0% against a COUNT_TOLERANCE of 10%, so "
                f"_check_expected_count would accept it."
            )
        self._check_title_variants(controls)
        # Written unconditionally, after the completeness check, so the file on
        # disk always describes a full ten-category run.
        self.write_repair_audit(repairs)
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
                f"not the pinned {self.expected_sha256}. Every measured "
                f"figure in this parser is a reading of one archive."
            )

    def _check_title_variants(self, controls: list[Control]) -> None:
        """Refuse a variant table entry that reaches nothing.

        Raises:
            ValueError: If a declared id is absent, or a declared variant is
                empty or restates the category's own title.
        """
        titles = {c.control_id: c.title for c in controls}
        for key in sorted(self.title_variants):
            if key not in titles:
                raise ValueError(
                    f"{self.framework_id}: the title variant table names no "
                    f"category {key!r}. Categories read: {sorted(titles)}. A "
                    f"variant for a category that does not exist reaches no "
                    f"control and still reads as a live alternate."
                )
            for variant in self.title_variants[key]:
                if not variant.strip():
                    raise ValueError(
                        f"{self.framework_id}: category {key} declares an "
                        f"empty title variant. An empty key can never be "
                        f"looked up."
                    )
                if variant.strip().lower() == titles[key].strip().lower():
                    raise ValueError(
                        f"{self.framework_id}: category {key} declares the "
                        f"title variant {variant!r}, which is already the "
                        f"category's own title. ProseIndex indexes real "
                        f"titles first and never lets an alternate displace "
                        f"one, so this entry is dead."
                    )

    @classmethod
    def control_from_markdown(
        cls,
        text: str,
        variants: Mapping[str, tuple[str, ...]] | None = None,
        repairs: list[dict[str, object]] | None = None,
    ) -> Control:
        """One category from one markdown file.

        `repairs` is an optional sink. Both text-moving transforms append a
        before/after pair to it, which parse() hands to write_repair_audit. A
        count would say a repair fired; only the pair says what moved.

        Raises:
            ValueError: If the file has no heading, no Description section, or
                no removable Factors section.
        """
        heading = HEADING.search(text)
        if heading is None:
            raise ValueError(
                "owasp_top10_2021: markdown with no 'A0N:2021 - Title' "
                "heading reached control_from_markdown."
            )
        control_id = heading.group(1)
        body = DESCRIPTION.search(text)
        if body is None:
            raise ValueError(
                f"owasp_top10_2021: {control_id} has no '## Description' "
                f"section. Unhandled, its statement would fall back to the "
                f"Overview, which is commentary about the survey rather than "
                f"about the risk."
            )
        entry = text[heading.end():].strip()
        stripped, removed = FACTORS.subn("", entry, count=1)
        if removed != 1:
            raise ValueError(
                f"owasp_top10_2021: {control_id} has no '## Factors' section "
                f"followed by another '##' heading, so there is nothing this "
                f"parser can identify as the statistics table. All ten "
                f"entries carried one at 529 characters when this landed. "
                f"Passing the entry through would put a table of CWE counts "
                f"at the head of this anchor and not the others; removing a "
                f"span the pattern did not match would be worse. Re-read the "
                f"edition and decide."
            )
        stripped = stripped.strip()
        statement, capped = cls._cap_description(body.group(1).strip())
        if repairs is not None:
            repairs.append({
                "control_id": control_id,
                "repair": "factors_section_removed",
                "field": "full_text",
                "before": entry,
                "after": stripped,
            })
            if capped:
                repairs.append({
                    "control_id": control_id,
                    "repair": "description_capped_to_protect_full_text",
                    "field": "description",
                    "before": body.group(1).strip(),
                    "after": statement,
                })
        # Every H1 carries a trailing icon image with a style attribute. It is
        # markup rather than name, and it would otherwise be indexed as part
        # of the title key.
        title = heading.group(2).split("![", 1)[0].strip()
        metadata: dict[str, str | list[str]] = {}
        declared = (variants or {}).get(control_id)
        if declared:
            metadata["alt_titles"] = list(declared)
        return Control(
            control_id=control_id,
            title=title,
            description=statement,
            full_text=stripped,
            hierarchy_level="category",
            metadata=metadata or None,
        )

    @staticmethod
    def _cap_description(text: str) -> tuple[str, bool]:
        """Keep description inside DESCRIPTION_MAX_LENGTH, on a word boundary.

        Sanitisation only shrinks this corpus (whitespace collapse over
        markdown indentation), so a raw cut at the limit lands under it once
        run() sanitises. The test suite asserts the shipped result rather than
        trusting that, because a pathological expansion would silently hand
        full_text back to the base class.

        Returns the text and whether the cut fired.
        """
        if len(text) <= DESCRIPTION_MAX_LENGTH:
            return text, False
        cut = text[:DESCRIPTION_MAX_LENGTH].rsplit(" ", 1)[0]
        # Mirrors sanitize_text's own guard: a body with no space in its first
        # half would otherwise be cut to almost nothing.
        if len(cut) < DESCRIPTION_MAX_LENGTH // 2:
            cut = text[:DESCRIPTION_MAX_LENGTH]
        return cut, True


def main() -> None:
    OwaspTop102021Parser().run()


if __name__ == "__main__":
    main()
