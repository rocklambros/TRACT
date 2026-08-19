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

`full_text` is the whole entry below the heading line, and `full_text` is
normally the anchor: ProseIndex prefers it over description unconditionally.
Measured on the pinned archive the ten entries run 4,821 to 9,706 characters
against a MAX_ANCHOR_CHARS budget of 2,150, so every one of the 17 curated
links resolves onto a truncated anchor and the corpus report records
`truncated == 17`. The anchor is the opening 2,150 characters of the category
page, which spends its first ~900 on the Factors table and the Overview before
reaching the Description.

Two categories are an exception, and it is not this parser's choice.
`_sanitize_control` calls `sanitize_text(description, return_full=True)` and
assigns whatever that returns to `full_text`, keeping the parser's own value
only when the description fits DESCRIPTION_MAX_LENGTH. A02's Description
sanitises to 2,377 characters and A04's to 2,944, both over the 2,000
character limit, so for those two the whole entry is discarded and the anchor
is the Description alone. [measured] Both replacements still exceed
MAX_ANCHOR_CHARS, so the join counters do not move and nothing downstream can
see the substitution. tests/test_parse_owasp_top10_2021.py pins which two they
are, in both directions, so a change to either budget shows up as a failure
rather than as a quiet change of anchor.

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
                    self.control_from_markdown(text, self.title_variants)
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
    ) -> Control:
        """One category from one markdown file.

        Raises:
            ValueError: If the file has no heading or no Description section.
        """
        heading = HEADING.search(text)
        if heading is None:
            raise ValueError(
                "owasp_top10_2021: markdown with no 'A0N:2021 - Title' "
                "heading reached control_from_markdown."
            )
        body = DESCRIPTION.search(text)
        if body is None:
            raise ValueError(
                f"owasp_top10_2021: {heading.group(1)} has no '## Description' "
                f"section. Unhandled, its statement would fall back to the "
                f"Overview, which is commentary about the survey rather than "
                f"about the risk."
            )
        # Every H1 carries a trailing icon image with a style attribute. It is
        # markup rather than name, and it would otherwise be indexed as part
        # of the title key.
        title = heading.group(2).split("![", 1)[0].strip()
        metadata: dict[str, str | list[str]] = {}
        declared = (variants or {}).get(heading.group(1))
        if declared:
            metadata["alt_titles"] = list(declared)
        return Control(
            control_id=heading.group(1),
            title=title,
            description=body.group(1).strip(),
            full_text=text[heading.end():].strip(),
            hierarchy_level="category",
            metadata=metadata or None,
        )


def main() -> None:
    OwaspTop102021Parser().run()


if __name__ == "__main__":
    main()
