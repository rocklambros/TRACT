"""Parser for the OWASP Top 10 for LLM Applications 2026.

A separate framework from `owasp_llm_top10`, never a newer version of it. The
2025 edition's LLM0x:2025 ids carry all 13 of OpenCRE's links for this
standard, so writing 2026 entries over that file would break the join for an
AI test fold. This parser writes `owasp_llm_top10_2026.json` and touches
nothing else.

It is also the project's pretraining-contamination control. BAAI/bge-large-
en-v1.5 predates this document, so it is the one corpus that can separate an
encoder mapping meaning from an encoder recalling text it was pretrained on.
That only holds while the framework stays out of training, which is enforced
by `tract.config.HOLDOUT_FRAMEWORK_IDS` and asserted by
`tests/test_holdout_framework.py`.

Three decisions worth stating.

**The parser stops at `## Appendix A`.** The appendix and all the back matter
sit below the last entry rather than above it, so without that boundary LLM10
runs to end of file and takes the 871 lines and 124 KB that sit below the
appendix heading with it, appendix tables, references, and acknowledgements
all shipping as control text. Measured on the pinned source. A source that has lost the
boundary heading is refused rather than parsed.

**`description` is the definitional block, `full_text` is the whole entry.**
Every entry is laid out as Description, sometimes extra subsections, Common
Examples of Risk, Prevention and Mitigation Strategies, Example Attack
Scenarios. The cut falls at the first subheading in
`tract.config.REMEDIATION_HEADINGS`, which in this document is always
Prevention and Mitigation Strategies. Three reasons. The assignment paradigm
maps what a control *is*, and prevention text says how to satisfy it, which is
a different question and pulls the anchor toward controls that share
countermeasures rather than meaning. The same heading list is what
`tract.text_selection.strip_remediation` applies downstream, so cutting here
structurally makes the parser and the anchor selector agree instead of each
guessing separately from a regex over flattened text. And the encoder's
512-token budget is fixed, so including remediation would push definitional
prose out of the window rather than add to it. Nothing is lost either way:
`full_text` carries the entry from its heading to the next one, prevention and
scenarios included.

Common Examples of Risk stays on the definitional side. It enumerates what the
risk looks like, not what to do about it, and it is what tops up the six
entries whose Description section alone is under the 2,000-character
description cap.

**`version` pins to the source sha256, not to a date.** The document's own
revision history still reads "[2026 release date]", so it is pre-release and
any date string here would assert a release that has not happened. The digest
is checked on every parse, so a re-pinned or edited source stops the parser
rather than silently changing what the artifact describes.
"""
from __future__ import annotations

import logging
import re
from typing import ClassVar, Final

from tract.config import DESCRIPTION_MAX_LENGTH, REMEDIATION_HEADINGS
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SOURCE_FILE: Final[str] = "2026_OWASP-GenAI-LLM-Top-10.repaired.md"

# The owner-supplied staging digest, recorded in the run ledger and in
# scripts/fetch_frameworks.py. Also this framework's `version`.
SOURCE_SHA256: Final[str] = (
    "3d3c9f21809c5f882a668b87424ac6b2e2a270caab4b29aa5265df3475433a96"
)

# "## LLM01:2026 Prompt Injection". The edition tag is required: the reference
# list below Appendix A repeats every entry label as "## LLM01: Prompt
# Injection" without it, and those are citations rather than entries.
ENTRY_HEADING: Final[re.Pattern[str]] = re.compile(
    r"^##\s+(LLM(?:0[1-9]|10):2026)\s+(\S.*?)\s*$"
)
# Where the entries stop. Matched on the prefix rather than the full heading
# so a change to the appendix subtitle does not silently disable the boundary.
APPENDIX_HEADING: Final[re.Pattern[str]] = re.compile(r"^##\s+Appendix A\b")
# The document writes every subsection at the same level as its entries.
SUBHEADING: Final[re.Pattern[str]] = re.compile(r"^##\s+(\S.*?)\s*$")

# Exact and ordered. A source that yields nine entries, eleven, or the same ten
# under different ids is a source this parser has not read before.
ENTRY_IDS: Final[tuple[str, ...]] = tuple(
    f"LLM{index:02d}:2026" for index in range(1, 11)
)

# Every entry opens with this subsection. Checked so a restructured source
# stops here rather than emitting a description that starts somewhere else.
DESCRIPTION_HEADING: Final[str] = "Description"

# Where the definitional block ends. Shared with strip_remediation so the two
# cuts cannot disagree.
REMEDIATION_SUBHEADINGS: Final[frozenset[str]] = frozenset(REMEDIATION_HEADINGS)

# Trailing partial word left by a hard character cut.
_TRAILING_PARTIAL_WORD: Final[re.Pattern[str]] = re.compile(r"\s\S*$")


class OwaspLlmTop102026Parser(BaseParser):
    framework_id: ClassVar[str] = "owasp_llm_top10_2026"
    framework_name: ClassVar[str] = "OWASP Top 10 for LLM Applications 2026"
    version: ClassVar[str] = f"sha256:{SOURCE_SHA256}"
    # The publisher's project site. The document itself is pre-release and has
    # no URL of its own yet, which is why its Source entry carries url=None and
    # the file is staged by hand.
    source_url: ClassVar[str] = "https://genai.owasp.org"
    mapping_unit_level: ClassVar[str] = "risk"
    expected_count: ClassVar[int] = 10
    fetched_date: ClassVar[str] = "2026-08-16"
    # 1.0, and measured at 1.0. Unlike ISO 27001, which has three genuinely
    # one-sentence controls, every entry here opens with a multi-paragraph
    # Description section; the shortest definitional block is 2,631 characters
    # against a 60-character prose bar. There is no short-entry exception to
    # carve out, so anything under 1.0 would mean the extraction broke.
    min_prose_fraction: ClassVar[float] = 1.0
    # Class-level so a test over a synthetic fixture declares its own digest
    # rather than the real gate being widened to accept two sources.
    source_sha256: ClassVar[str] = SOURCE_SHA256

    def parse(self) -> list[Control]:
        text = self.read_source(SOURCE_FILE)
        self._check_digest()

        lines = text.splitlines()
        boundary = self._appendix_line(lines)
        entries = self._entry_spans(lines, boundary)

        controls: list[Control] = []
        truncated: list[str] = []
        for control_id, title, body in entries:
            definitional = self._definitional_block(control_id, body)
            description, was_cut = self._cut_to_budget(definitional)
            if was_cut:
                truncated.append(control_id)
            controls.append(Control(
                control_id=control_id,
                title=title,
                description=description,
                full_text="\n".join(body).strip(),
                hierarchy_level="risk",
            ))

        logger.info(
            "%s: %d entries, boundary at source line %d, %d description(s) cut "
            "to the %d-character budget: %s",
            self.framework_id, len(controls), boundary + 1, len(truncated),
            DESCRIPTION_MAX_LENGTH, ", ".join(truncated) or "none",
        )
        return controls

    def _check_digest(self) -> None:
        """Refuse a source that is not the bytes this parser was written for.

        `version` is that digest, so a mismatch means the artifact would
        describe itself with a hash of something else.

        Raises:
            ValueError: If the read file's sha256 is not the declared pin.
        """
        actual = self.recorded_sha256(SOURCE_FILE)
        if actual == self.source_sha256:
            return
        raise ValueError(
            f"{self.framework_id}: {SOURCE_FILE} has sha256 {actual}, not the "
            f"pinned {self.source_sha256}. The staged source changed. This "
            f"document is pre-release, so its version IS the digest: re-measure "
            f"the structure against the new bytes, then move SOURCE_SHA256 and "
            f"the Source entry in scripts/fetch_frameworks.py together."
        )

    def _appendix_line(self, lines: list[str]) -> int:
        """Index of the `## Appendix A` heading that terminates the entries.

        Raises:
            ValueError: If no appendix heading is present.
        """
        for index, line in enumerate(lines):
            if APPENDIX_HEADING.match(line):
                return index
        raise ValueError(
            f"{self.framework_id}: no '## Appendix A' heading in "
            f"{SOURCE_FILE}. That heading is what ends the last entry. Without "
            f"it LLM10 runs to end of file and takes the 871 lines of appendix "
            f"tables, references, and acknowledgements below it, which would "
            f"ship as control text. Fix the source rather than the boundary."
        )

    def _entry_spans(
        self, lines: list[str], boundary: int,
    ) -> list[tuple[str, str, list[str]]]:
        """(control_id, title, body lines) for each of the ten entries.

        Raises:
            ValueError: If the entry ids are not exactly ENTRY_IDS in order.
        """
        heads: list[tuple[int, str, str]] = []
        for index, line in enumerate(lines[:boundary]):
            match = ENTRY_HEADING.match(line)
            if match:
                heads.append((index, match.group(1), match.group(2)))

        found = tuple(control_id for _, control_id, _ in heads)
        if found != ENTRY_IDS:
            raise ValueError(
                f"{self.framework_id}: expected entries {list(ENTRY_IDS)} in "
                f"order, found {list(found)}. Either the source changed or the "
                f"heading pattern stopped matching it. A short list here would "
                f"otherwise ship a partial Top 10."
            )

        spans: list[tuple[str, str, list[str]]] = []
        for position, (start, control_id, title) in enumerate(heads):
            end = heads[position + 1][0] if position + 1 < len(heads) else boundary
            spans.append((control_id, title, lines[start + 1:end]))
        return spans

    def _definitional_block(self, control_id: str, body: list[str]) -> str:
        """The entry text up to its first remediation subheading.

        Heading markers are dropped and the heading text is kept, which is what
        strip_markup does downstream anyway. The names carry meaning ("Common
        Examples of Risk"), the hashes do not.

        Raises:
            ValueError: If the entry does not open with Description, or has no
                remediation subheading to cut at.
        """
        subheadings = [
            (index, match.group(1))
            for index, line in enumerate(body)
            if (match := SUBHEADING.match(line))
        ]
        if not subheadings or subheadings[0][1] != DESCRIPTION_HEADING:
            first = subheadings[0][1] if subheadings else "none"
            raise ValueError(
                f"{self.framework_id}: entry {control_id} opens with "
                f"{first!r}, not {DESCRIPTION_HEADING!r}. The layout changed, "
                f"and the description would start somewhere other than the "
                f"definitional text."
            )

        cut = next(
            (index for index, name in subheadings
             if name in REMEDIATION_SUBHEADINGS),
            None,
        )
        if cut is None:
            raise ValueError(
                f"{self.framework_id}: entry {control_id} has no subheading in "
                f"REMEDIATION_HEADINGS, so there is nowhere to end the "
                f"definitional block. Unhandled, its description would become "
                f"the whole entry including every attack scenario. Add the "
                f"source's new heading to tract.config.REMEDIATION_HEADINGS."
            )

        kept = [
            match.group(1) if (match := SUBHEADING.match(line)) else line
            for line in body[:cut]
        ]
        return "\n".join(kept).strip()

    @staticmethod
    def _cut_to_budget(text: str) -> tuple[str, bool]:
        """Cut a description to DESCRIPTION_MAX_LENGTH. Returns (text, was_cut).

        Not cosmetic. BaseParser._sanitize_control replaces a parser-supplied
        full_text with the overflow of an over-long description, so an uncut
        definitional block would evict the entry text this parser puts there
        and leave full_text holding a longer copy of the description.

        Sanitization only ever shortens (whitespace collapses), so a raw cut at
        the budget guarantees the stored description clears it.
        """
        if len(text) <= DESCRIPTION_MAX_LENGTH:
            return text, False
        head = text[:DESCRIPTION_MAX_LENGTH]
        match = _TRAILING_PARTIAL_WORD.search(head)
        # A whole-word cut that throws away more than half the budget means the
        # text has no whitespace to cut at, so keep the hard cut instead. Same
        # rule as tract.sanitize.sanitize_text.
        if match and match.start() >= DESCRIPTION_MAX_LENGTH // 2:
            head = head[:match.start()]
        return head.strip(), True


def main() -> None:
    OwaspLlmTop102026Parser().run()


if __name__ == "__main__":
    main()
