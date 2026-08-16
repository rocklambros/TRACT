"""Parser for ISO/IEC 27001:2022 Annex A.

The source is a markdown conversion of the PDF and it carries three kinds of
damage, all measured across its 93 control rows: 29 rows with a spaced
hyphen break of which 17 corrupt the title, 22 rows with a run-together token
of 20 characters or more, and 4 pairs where a cell bled into the next row.

Unrepaired, this text tokenizes to fragments and would score worse than the
28-character titles it replaces.

The output file is gitignored. This repository is CC0 and committing normative
ISO control statements under it would assert rights the project does not hold.
See tests/test_licensed_text_not_tracked.py.
"""
from __future__ import annotations

import logging
import re
from typing import ClassVar, Final

from tract.parsers.base import BaseParser
from tract.parsers.repair import (
    build_vocabulary,
    fix_hyphen_breaks,
    repair_cell_bleed,
    split_run_together,
    strip_page_furniture,
)
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SOURCE_FILE: Final[str] = "ISO_IEC_27001_2022_en.md"

# Running headers and footers repeat through Annex A and would otherwise be
# extracted as content.
PAGE_FURNITURE: Final[tuple[str, ...]] = (
    r"^##\s*ISO/IEC\s+27001",
    r"^Table A\.1 \(continued\)",
)

# A control row's first cell is "5.1", a section header's is "5".
CONTROL_ID: Final[re.Pattern[str]] = re.compile(r"^\d+\.\d+$")
# Each control statement opens with this keyword in the source table.
STATEMENT_MARKER: Final[str] = "Control"

# A run-together token is treated as one word by build_vocabulary, since it
# is an unbroken run of letters with nothing to mark the joins. Left in the
# vocabulary unfiltered, that whole token becomes a trivially valid
# one-segment "word" and split_run_together's fewest-segments search prefers
# it over the real multi-word decomposition, so the row that most needs the
# repair silently defeats it. Matches split_run_together's own default
# min_token_length so the excluded entries are exactly the ones long enough
# to be a run-together candidate themselves.
RUN_TOGETHER_MIN_LENGTH: Final[int] = 20

# Measured ceilings. run() refuses to write when a repair exceeds its
# declared count, so a transform that starts eating good text is caught here
# rather than by someone reading the corpus later.
MAX_HYPHEN_REPAIRS: Final[int] = 40
MAX_RUN_TOGETHER_REPAIRS: Final[int] = 30
MAX_BLEED_REPAIRS: Final[int] = 6


class Iso27001Parser(BaseParser):
    framework_id: ClassVar[str] = "iso_27001"
    framework_name: ClassVar[str] = "ISO/IEC 27001:2022 Annex A"
    version: ClassVar[str] = "2022"
    source_url: ClassVar[str] = "https://www.iso.org/standard/27001"
    mapping_unit_level: ClassVar[str] = "control"
    expected_count: ClassVar[int] = 93
    fetched_date: ClassVar[str] = "2026-08-15"
    # Measured at 90/93 = 0.9677 on the real source. 3 rows (5.16, 7.8, 7.9)
    # carry a genuine single-sentence statement under the shared 60-character
    # honest-prose threshold; hand-checked against the source and confirmed
    # real, not a truncated fragment. The floor sits just below the measured
    # value, close enough to 1.0 that a regression toward title-only
    # extraction, the failure mode this check exists for, still trips it.
    min_prose_fraction: ClassVar[float] = 0.96

    def parse(self) -> list[Control]:
        text = self.read_source(SOURCE_FILE)
        rows = self._extract_rows(text)
        rows, bleed = repair_cell_bleed(rows, marker=STATEMENT_MARKER)
        self._check_repair("cell bleed", bleed, MAX_BLEED_REPAIRS)

        # Built from the statements themselves. A run-together token's parts
        # are ordinary words that appear elsewhere in the same table. The raw
        # build also picks up each row's own still-joined token as a "word"
        # (see RUN_TOGETHER_MIN_LENGTH); drop anything that long before it is
        # used for segmentation.
        raw_vocabulary = build_vocabulary(
            [t for _, _, t in rows] + [t for _, t, _ in rows]
        )
        vocabulary = frozenset(
            word for word in raw_vocabulary if len(word) < RUN_TOGETHER_MIN_LENGTH
        )

        controls: list[Control] = []
        hyphen_total = 0
        split_total = 0
        for control_id, title, statement in rows:
            fixed_title = fix_hyphen_breaks(title)
            fixed_body = fix_hyphen_breaks(statement)
            hyphen_total += fixed_title.applied + fixed_body.applied

            split = split_run_together(fixed_body.text, vocabulary)
            split_total += split.applied

            body = split.text.strip()
            if body.startswith(STATEMENT_MARKER):
                body = body[len(STATEMENT_MARKER):].strip()

            controls.append(Control(
                control_id=control_id,
                title=fixed_title.text.strip(),
                description=body,
            ))

        self._check_repair("hyphen break", hyphen_total, MAX_HYPHEN_REPAIRS)
        self._check_repair("run-together", split_total, MAX_RUN_TOGETHER_REPAIRS)
        return controls

    def _extract_rows(self, text: str) -> list[tuple[str, str, str]]:
        """Pull (control_id, title, statement) from the Table A.1 rows."""
        lines, dropped = strip_page_furniture(text.splitlines(), PAGE_FURNITURE)
        logger.info("Dropped %d page-furniture lines", dropped)

        rows: list[tuple[str, str, str]] = []
        for line in lines:
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) != 3:
                continue
            if not CONTROL_ID.match(cells[0]):
                continue
            rows.append((cells[0], cells[1], cells[2]))
        if not rows:
            raise ValueError(
                f"{self.framework_id}: no Table A.1 control rows matched. The "
                f"source layout changed; re-check {SOURCE_FILE}."
            )
        return rows

    def _check_repair(self, name: str, applied: int, ceiling: int) -> None:
        if applied > ceiling:
            raise ValueError(
                f"{self.framework_id}: the {name} repair fired {applied} times "
                f"against a declared ceiling of {ceiling}. Either the source "
                f"changed or the repair is eating good text. Inspect before "
                f"raising the ceiling."
            )
        logger.info("%s repair applied %d times (ceiling %d)", name, applied, ceiling)


def main() -> None:
    Iso27001Parser().run()


if __name__ == "__main__":
    main()
