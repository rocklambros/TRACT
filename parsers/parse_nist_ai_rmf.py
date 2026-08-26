"""Parser for NIST AI RMF 1.0 subcategories.

The source is a PDF converted to markdown. Tables 1 through 4 hold the
subcategories, one per table cell, and the converter rendered each cell as a
run of hard-wrapped lines delimited by a blank line. A subcategory statement
is therefore one block of wrapped lines, and the wrap position carries no
meaning: it records where the PDF's column ended.

The previous parser captured the title as `[^\\n]*`, which stops at the first
wrap. Every one of the 72 subcategories is wrapped, so every control shipped as
two halves of one sentence. [measured 2026-08-19]

    title:       'Legal and regulatory requirements involving AI'
    description: 'are understood, managed, and documented.'

Neither half is a control statement, and the damage went further than the
split. The description ran from the wrap to the next subcategory marker, so it
also swallowed whatever the converter emitted in between: `GOVERN 1.4` carried
`Continued on next page`, a page number, the running header and the repeated
table caption, and `MEASURE 2.13` carried the text of the `MEASURE 3` category
cell. `MEASURE 2.11` lost its closing `**` to the `rstrip("*")` and shipped a
title ending `the **MAP`.

This parser reads the cell instead of the line. It scans the blocks between one
subcategory marker and the next, keeps blocks until one ends a sentence, and
refuses any block that opens a new structural element. Two cells need the
second block: the converter left a stray blank line inside `MEASURE 2.12`, and
it turned the en dash in `MAP 1.4` into a list bullet on its own line.
[measured 2026-08-19] Both are identified by the same signal, a first block
that does not end in a period, and neither needs a rule of its own.

BLOCK_STOP is what keeps that scan honest. Without it an unterminated cell
walks through `Continued on next page`, the page number, the running header and
the caption, none of which ends in a period, and stops inside the next cell
with a plausible-looking sentence assembled from four sources. The parser
raises instead.

The title is the subcategory identifier. The source gives subcategories no
title: Tables 1 through 4 have two columns, and the subcategory column holds a
bare statement under an identifier. Any title here would be a truncation of the
statement, which is the defect this change removes. parsers/parse_nist_ssdf.py
took the same route for the same reason.

The repair moves text across the title and description boundary, so every
rejoin writes an audit record carrying both halves and the result as text.
"""
from __future__ import annotations

import logging
import re
from typing import ClassVar, Final

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SOURCE_NAME: Final[str] = "nist_ai_rmf_1.0.md"

# Anchored at the start of a line, because a subcategory marker is the first
# thing in its table cell. The unanchored form would also read a mid-sentence
# mention of a subcategory, and the narrative between the tables makes several.
SUBCATEGORY: Final[re.Pattern[str]] = re.compile(
    r"^\*\*(?P<func>GOVERN|MAP|MEASURE|MANAGE)\s+"
    r"(?P<cat>\d+)\.(?P<sub>\d+)[:.]?\*\*",
    re.MULTILINE,
)

# The converter separates table cells with a blank line. Nothing else in this
# source marks a cell boundary.
BLOCK_BREAK: Final[re.Pattern[str]] = re.compile(r"\n[ \t]*\n")

# A block that opens a new structural element and can never continue a
# statement. The four page-furniture forms repeat on every table page, and the
# category cell is the left column of the same table.
BLOCK_STOP: Final[re.Pattern[str]] = re.compile(
    r"^(?:"
    r"#{1,6}\s"                                       # a document heading
    r"|\*\*\d+\.\d+\*\*"                              # a numbered section head
    r"|\*\*(?:GOVERN|MAP|MEASURE|MANAGE)\s+\d+[:.]?\*\*"  # a category cell
    r"|\*\*Categories\*\*"                            # the column header row
    r"|Table\s+\d+:"                                  # the repeated caption
    r"|Continued on next page"
    r"|Page\s+\d+"
    r"|NIST AI 100-1 AI RMF"                          # the running header
    r")"
)

# A cell holds one or two sentences and always closes on a period. Used to
# decide whether the cell continued past a blank line the converter left in it,
# never to cut inside a block.
SENTENCE_END: Final[str] = "."

FUNCTION_NAMES: Final[dict[str, str]] = {
    "GOVERN": "Govern",
    "MAP": "Map",
    "MEASURE": "Measure",
    "MANAGE": "Manage",
}


class NistAiRmfParser(BaseParser):
    framework_id: ClassVar[str] = "nist_ai_rmf"
    framework_name: ClassVar[str] = "NIST AI Risk Management Framework"
    version: ClassVar[str] = "1.0"
    source_url: ClassVar[str] = "https://doi.org/10.6028/NIST.AI.100-1"
    mapping_unit_level: ClassVar[str] = "subcategory"
    expected_count: ClassVar[int] = 72
    fetched_date: ClassVar[str] = "2026-04-28"
    # 72 of 72 subcategories clear HONEST_PROSE_MIN_CHARS once the wrapped
    # sentence is rejoined, giving 1.0. [measured 2026-08-19] The floor
    # replaces 0.76, which measured where the converter wrapped its lines
    # rather than what the source states.
    #
    # 1.0 is attainable here because the source gives every subcategory a
    # statement, and it is the honest floor for the same reason: anything below
    # it means a cell lost text. The margin is not uniform. MAP 1.5,
    # "Organizational risk tolerances are determined and documented.", is 61
    # characters against a threshold of 60. [measured 2026-08-19]
    # tests/test_parse_nist_ai_rmf.py pins that length, so a change in
    # sanitisation shows up there and names the control rather than surfacing
    # as an unexplained floor failure here.
    min_prose_fraction: ClassVar[float] = 1.0

    def parse(self) -> list[Control]:
        text = self.read_source(SOURCE_NAME)
        matches = list(SUBCATEGORY.finditer(text))
        if not matches:
            raise ValueError(
                f"{self.framework_id}: no subcategory marker in {SOURCE_NAME}. "
                f"The marker is the only join key this source offers, and the "
                f"tables carry nothing else that names a subcategory."
            )

        controls: list[Control] = []
        audit: list[dict[str, object]] = []

        for i, match in enumerate(matches):
            func = match.group("func")
            control_id = f"{func} {match.group('cat')}.{match.group('sub')}"
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            statement, blocks_used = self._statement(
                control_id, text[match.end():end],
            )
            self._record_rejoin(control_id, statement, blocks_used, audit)

            controls.append(Control(
                control_id=control_id,
                # The source names no title. See the module docstring.
                title=control_id,
                description=self._rejoin(statement),
                hierarchy_level="subcategory",
                parent_id=f"{func} {match.group('cat')}",
                parent_name=FUNCTION_NAMES[func],
                metadata={"function": func},
            ))

        self.write_repair_audit(audit)
        logger.info(
            "%s: %d subcategories, %d rejoined from a hard line wrap",
            self.framework_id, len(controls), len(audit),
        )
        return controls

    @classmethod
    def _statement(cls, control_id: str, segment: str) -> tuple[str, int]:
        """The subcategory cell between one marker and the next.

        Returns the statement with its source line breaks intact and the number
        of blocks it spans.

        Raises:
            ValueError: If the cell does not close on a sentence, which means
                the converter's layout changed and the block scan no longer
                describes it.
        """
        kept: list[str] = []
        for block in (b.strip() for b in BLOCK_BREAK.split(segment)):
            if not block or BLOCK_STOP.match(block):
                break
            kept.append(block)
            if block.endswith(SENTENCE_END):
                break

        statement = "\n".join(kept)
        if not statement.endswith(SENTENCE_END):
            raise ValueError(
                f"nist_ai_rmf: the cell for {control_id} does not close on a "
                f"sentence. Read so far: {statement!r}. Every subcategory in "
                f"this source is one or two complete sentences inside one "
                f"table cell, so either the converter's layout changed or the "
                f"scan stopped on a block it should have kept."
            )
        return statement, len(kept)

    @staticmethod
    def _rejoin(statement: str) -> str:
        """Undo the converter's hard wrap.

        Mirrors tract.sanitize's order, hyphenation before whitespace, so a
        word the converter broke across a line rejoins as one word rather than
        keeping a hyphen and gaining a space. No line of the pinned source ends
        in a hyphen [measured 2026-08-19], so this is a guard against a future
        revision rather than a transform that fires today.
        """
        return " ".join(re.sub(r"(\w)-\n(\w)", r"\1\2", statement).split())

    @classmethod
    def _record_rejoin(
        cls,
        control_id: str,
        statement: str,
        blocks_used: int,
        audit: list[dict[str, object]],
    ) -> None:
        """Write what the rejoin moved, as text on both sides.

        A count would say the repair fired. It would not say which fragment
        landed on which control, and a fragment attributed to the wrong
        subcategory is a wrong compliance assertion carrying a plausible
        provenance record.
        """
        head, _, tail = statement.partition("\n")
        if not tail:
            return
        audit.append({
            "control_id": control_id,
            "repair": "line_wrapped_statement_rejoined",
            # The two halves the converter's wrap created. The previous parser
            # stored the first as the title and the second as the description.
            "before": [head, tail],
            "after": cls._rejoin(statement),
            # Above 1 only where the converter also left a blank line inside
            # the cell, which happens twice in the pinned source.
            "source_blocks": blocks_used,
            "reason": (
                "the source is a PDF converted to markdown and the converter "
                "hard-wrapped each table cell, so the line break inside a "
                "subcategory records the width of a PDF column rather than a "
                "boundary in the text"
            ),
        })


def main() -> None:
    NistAiRmfParser().run()


if __name__ == "__main__":
    main()
