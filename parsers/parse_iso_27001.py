"""Parser for ISO/IEC 27001:2022 Annex A.

The source is a markdown conversion of the PDF and it carries three kinds of
damage, all measured across its 93 control rows: 29 rows with a spaced
hyphen break of which 17 corrupt the title, 22 rows with a run-together token
of 20 characters or more, and 4 pairs where a cell bled into the next row.

Unrepaired, this text tokenizes to fragments and would score worse than the
28-character titles it replaces.

Repair counts are declared exactly and checked in both directions, and the
damage no repair reached is declared alongside them. A ceiling only catches a
transform that runs away. The failure that ships bad text quietly is the
opposite one: a source refresh moves the damage, the repair stops reaching it,
and the output is truncated with every gate green. See EXPECTED_REPAIRS,
EXPECTED_RESIDUAL_DAMAGE, and _find_residual_run_together below.

The output file is gitignored. This repository is CC0 and committing normative
ISO control statements under it would assert rights the project does not hold.
See tests/test_licensed_text_not_tracked.py.
"""
from __future__ import annotations

import logging
import re
from typing import ClassVar, Final

from tract.config import (
    CONTROL_DAMAGE_REASON_METADATA_KEY,
    CONTROL_DAMAGED_METADATA_KEY,
    CONTROL_DAMAGED_METADATA_VALUE,
    CONTROL_ELISION_MARKER,
)
from tract.parsers.base import BaseParser
from tract.parsers.repair import (
    BleedJoin,
    build_vocabulary,
    fix_hyphen_breaks,
    prune_decomposable,
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
    # The source writes this as a markdown heading, "## Table A.1
    # (continued)", so an anchored pattern without the hashes matched none of
    # its seven occurrences.
    r"^#*\s*Table A\.1 \(continued\)",
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

# Measured against the pinned source, exactly, not as ceilings. data/raw is
# immutable and every transform here is deterministic, so these counts do not
# move unless the source or the repair changed, and either deserves a stop.
#
# The previous ceilings were one-sided and one was unreachable: run-together
# sat at 30 against an actual 10, so it could not have fired under any input
# this parser accepts. A one-sided ceiling also cannot see the failure that
# matters most, which is a repair that stops firing after a source refresh and
# ships truncated text with every gate green.
#
# Moving a number here requires a written repair_deviation_reason on the
# parser and a look at the audit file, not an edit to the literal.
EXPECTED_REPAIRS: Final[dict[str, int]] = {
    "cell bleed": 4,
    "hyphen break": 32,
    "run-together": 10,
}

# Rows whose title or description still carries an unsplit run-together token
# after every repair has run. split_run_together only fires when the
# vocabulary supplies a complete word-by-word decomposition and fails closed
# otherwise, so this is real remaining damage rather than an upper bound on
# it. Declared and checked for the same reason as the repair counts: a number
# that drifts silently is not a control.
#
# Of the 14, three are title-only damage the vocabulary cannot segment
# ("Addressinginformationsecurity" at 5.20, "managementplanningandpreparation"
# at 5.24, "Redundancyofinformation" at 8.14).
EXPECTED_RESIDUAL_DAMAGE: Final[int] = 14

# Controls whose source text lost content that no transform can recover, keyed
# to the reason. The cell-bleed repair joins a truncated head to the fragment
# that opens the next row, and the shape gate cannot see that the source also
# dropped the clause between them: head and tail still read as one sentence.
#
# These are hand-verified against the raw markdown. NEVER add an entry by
# writing the missing text from memory. Inventing a normative control
# statement is worse than shipping a disclosed gap, because a reader cannot
# tell an invented requirement from a real one.
KNOWN_DAMAGED_CONTROLS: Final[dict[str, str]] = {
    "7.5": (
        "unrecoverable from this source: the markdown conversion dropped the "
        "clause between 'such as natural' and 'infrastructure shall be "
        "designed and implemented.' Neither 'disasters' nor 'unintentional' "
        "appears anywhere in the raw file, so the words are gone rather than "
        "misplaced. The owner must supply the clause from the licensed PDF "
        "before this control carries a complete requirement."
    ),
}

# A control's description is flagged as still damaged if, after every repair
# has run, it still contains an unbroken letters-only run this long. This is
# the same length as RUN_TOGETHER_MIN_LENGTH on purpose: anything shorter is
# an ordinary long word, and anything at or above it is exactly the shape a
# genuine run-together token has.
RESIDUAL_RUN_TOGETHER_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"[A-Za-z]{{{RUN_TOGETHER_MIN_LENGTH},}}"
)

class Iso27001Parser(BaseParser):
    framework_id: ClassVar[str] = "iso_27001"
    framework_name: ClassVar[str] = "ISO/IEC 27001:2022 Annex A"
    version: ClassVar[str] = "2022"
    source_url: ClassVar[str] = "https://www.iso.org/standard/27001"
    mapping_unit_level: ClassVar[str] = "control"
    expected_count: ClassVar[int] = 93
    fetched_date: ClassVar[str] = "2026-08-15"
    # Below the brief's 1.0 because three Annex A statements are genuinely one
    # short sentence, each hand-verified against the raw source:
    #   5.16  "The full life cycle of identities shall be managed." (51 chars)
    #   7.8   "Equipment shall be sited securely and protected." (48 chars)
    #   7.9   "Off-site assets shall be protected." (35 chars), recovered from
    #         a double cell bleed onto 7.8's raw row. Accurate, just short.
    # Measured 89/92 = 0.9674 with 7.5 excluded as damaged. The floor sits
    # just under that rather than at it, close enough to 1.0 that a regression
    # toward title-only extraction still trips it.
    #
    # This number is the whole audit trail. The rationale used to live in two
    # module constants that nothing read, which is a control in appearance
    # only. run() reads this one.
    min_prose_fraction: ClassVar[float] = 0.96
    # Class-level so a test over a sample of the table declares its own
    # measured counts rather than widening the real gate to cover both.
    expected_repairs: ClassVar[dict[str, int]] = EXPECTED_REPAIRS
    expected_residual_damage: ClassVar[int] = EXPECTED_RESIDUAL_DAMAGE
    # Set to a written reason when a repair count legitimately moves, in
    # the same spirit as BaseParser.count_deviation_reason.
    repair_deviation_reason: ClassVar[str | None] = None

    def parse(self) -> list[Control]:
        text = self.read_source(SOURCE_FILE)
        rows = self._extract_rows(text)
        rows, joins = repair_cell_bleed(rows, marker=STATEMENT_MARKER)
        rows = self._disclose_damaged_joins(rows, joins)
        self.write_repair_audit([self._audit_record(j) for j in joins])
        self._log_joins(joins)
        self._check_repair("cell bleed", sum(1 for j in joins if j.applied))

        # Hyphen repair runs over every row FIRST, before the vocabulary is
        # built. Built the other way round, "secu - rity" contributes "secu"
        # and "rity" as words and the splitter then prefers that pair over the
        # whole word it never saw, so the rows that most needed the repair
        # were the ones that defeated it.
        hyphen_total = 0
        repaired_rows: list[tuple[str, str, str]] = []
        for control_id, title, statement in rows:
            fixed_title = fix_hyphen_breaks(title)
            fixed_body = fix_hyphen_breaks(statement)
            hyphen_total += fixed_title.applied + fixed_body.applied
            repaired_rows.append(
                (control_id, fixed_title.text, fixed_body.text)
            )
        self._check_repair("hyphen break", hyphen_total)

        vocabulary = self._build_vocabulary(repaired_rows)

        controls: list[Control] = []
        split_total = 0
        for control_id, title, statement in repaired_rows:
            # Titles carry the same damage as bodies. Five ship joined
            # ("Addressinginformationsecurity", "Responsibilitiesaftertermination")
            # and the title is what OpenCRE joins a link on, so a joined one
            # cannot match its own link.
            split_title = split_run_together(title, vocabulary)
            split = split_run_together(statement, vocabulary)
            split_total += split_title.applied + split.applied

            body = split.text.strip()
            if body.startswith(STATEMENT_MARKER):
                body = body[len(STATEMENT_MARKER):].strip()

            controls.append(Control(
                control_id=control_id,
                title=split_title.text.strip(),
                description=body,
                metadata=self._damage_metadata(control_id, joins),
            ))

        self._check_repair("run-together", split_total)

        self._check_residual_damage(
            self._find_residual_run_together(controls)
        )
        return controls

    @staticmethod
    def _build_vocabulary(rows: list[tuple[str, str, str]]) -> frozenset[str]:
        """Segmentation vocabulary from already hyphen-repaired rows.

        Two filters, because a run-together token is an ordinary word as far
        as build_vocabulary can tell and either shape defeats the splitter.

        The length cut removes a token long enough to be a run-together
        candidate in its own right. The decomposability cut removes the short
        ones the length cut misses, which is most of them: "andenvironmental"
        is 16 characters and "theorganization" is 15.
        """
        raw = build_vocabulary(
            [body for _, _, body in rows] + [title for _, title, _ in rows]
        )
        by_length = frozenset(
            word for word in raw if len(word) < RUN_TOGETHER_MIN_LENGTH
        )
        return prune_decomposable(by_length)

    @staticmethod
    def _disclose_damaged_joins(
        rows: list[tuple[str, str, str]], joins: list[BleedJoin],
    ) -> list[tuple[str, str, str]]:
        """Rewrite a known-damaged row so its statement shows the gap.

        The fragment does belong to the predecessor, so leaving it on the
        successor would create a second wrong statement. It is moved, and an
        elision marker goes where the source lost text, which is the opposite
        of inventing content: the emitted statement declares its own hole
        rather than reading as though nothing is missing.
        """
        damaged_text = {
            j.predecessor_id: (
                f"{j.predecessor_before} {CONTROL_ELISION_MARKER} {j.fragment}"
            )
            for j in joins
            if j.applied and j.predecessor_id in KNOWN_DAMAGED_CONTROLS
        }
        if not damaged_text:
            return rows
        return [
            (cid, title, damaged_text.get(cid, body)) for cid, title, body in rows
        ]

    @staticmethod
    def _damage_metadata(
        control_id: str, joins: list[BleedJoin],
    ) -> dict[str, str | list[str]] | None:
        """Damage marker for one control, or None when its text is intact.

        Two sources of damage. A control in KNOWN_DAMAGED_CONTROLS lost text
        the source cannot supply. A control whose incoming join was refused
        still carries a fragment belonging to its predecessor, which the
        repair declined to move because the shapes were inconsistent.
        """
        reason = KNOWN_DAMAGED_CONTROLS.get(control_id)
        if reason is None:
            refused = [
                j for j in joins
                if j.successor_id == control_id and not j.applied
            ]
            if not refused:
                return None
            reason = (
                f"leading fragment from {refused[0].predecessor_id} was not "
                f"moved: {refused[0].refusal_reason}"
            )
        return {
            CONTROL_DAMAGED_METADATA_KEY: CONTROL_DAMAGED_METADATA_VALUE,
            CONTROL_DAMAGE_REASON_METADATA_KEY: reason,
        }

    @staticmethod
    def _audit_record(join: BleedJoin) -> dict[str, object]:
        """One inspectable before/after pair for the gitignored audit file."""
        return {
            "predecessor_id": join.predecessor_id,
            "successor_id": join.successor_id,
            "fragment": join.fragment,
            "predecessor_before": join.predecessor_before,
            "predecessor_after": join.predecessor_after,
            "applied": join.applied,
            "refusal_reason": join.refusal_reason,
            "known_damaged": join.predecessor_id in KNOWN_DAMAGED_CONTROLS,
        }

    def _log_joins(self, joins: list[BleedJoin]) -> None:
        """Log every bleed decision at WARNING with both ids and the text.

        This repair reattributes a compliance statement from one control id to
        another. That is not a detail worth an INFO line nobody reads.
        """
        for join in joins:
            if join.applied:
                logger.warning(
                    "%s: moved %r from %s to %s%s",
                    self.framework_id, join.fragment, join.successor_id,
                    join.predecessor_id,
                    " (marked damaged, source lost a clause)"
                    if join.predecessor_id in KNOWN_DAMAGED_CONTROLS else "",
                )
                continue
            logger.warning(
                "%s: refused to move %r from %s to %s: %s. %s keeps text that "
                "is not its own and is marked damaged",
                self.framework_id, join.fragment, join.successor_id,
                join.predecessor_id, join.refusal_reason, join.successor_id,
            )

    @staticmethod
    def _find_residual_run_together(controls: list[Control]) -> list[str]:
        """Control ids whose title or description still carries a joined token.

        split_run_together only fires when the corpus vocabulary supplies a
        complete word-by-word decomposition, and a row it cannot fully segment
        is left untouched rather than partially repaired. The repair counts
        say how many rows it fixed, not how many still need fixing, so this
        scan is the number that says what is left.

        Titles are scanned as well as descriptions. The title is what OpenCRE
        joins a link on, so a joined title is a link that cannot resolve.
        """
        return [
            c.control_id for c in controls
            if RESIDUAL_RUN_TOGETHER_PATTERN.search(c.description)
            or RESIDUAL_RUN_TOGETHER_PATTERN.search(c.title)
        ]

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
                f"source layout changed. Re-check {SOURCE_FILE}."
            )
        return rows

    def _check_repair(self, name: str, applied: int) -> None:
        """Compare one repair's count against its measured value, both ways.

        Raises:
            ValueError: If the repair is undeclared, or its count moved and no
                repair_deviation_reason is set.
        """
        expected = self.expected_repairs.get(name)
        if expected is None:
            raise ValueError(
                f"{self.framework_id}: the {name} repair has no measured "
                f"count in expected_repairs, so it runs ungated. Measure it "
                f"against the pinned source and declare the number."
            )
        if applied == expected:
            logger.info("%s repair applied %d times (measured %d)",
                        name, applied, expected)
            return
        if self.repair_deviation_reason:
            logger.warning(
                "%s: the %s repair fired %d times against a measured %d. "
                "Permitted: %s",
                self.framework_id, name, applied, expected,
                self.repair_deviation_reason,
            )
            return
        raise ValueError(
            f"{self.framework_id}: the {name} repair fired {applied} times "
            f"against a measured {expected}. Firing more means it is reaching "
            f"text it should not. Firing fewer means it stopped reaching "
            f"damage that is still there and the output ships truncated. Read "
            f"the audit file before moving the number, and record why in "
            f"repair_deviation_reason."
        )

    def _check_residual_damage(self, residual: list[str]) -> None:
        """Compare the rows still carrying damage against the measured count.

        Raises:
            ValueError: If the count moved and no repair_deviation_reason is
                set. A fall is as much a surprise as a rise: it means the
                repair reached rows it did not reach when this was measured,
                and that is worth reading before it is accepted.
        """
        if len(residual) == self.expected_residual_damage:
            logger.info(
                "%s: %d control(s) still carry an unsplit run-together token, "
                "as measured: %s",
                self.framework_id, len(residual), ", ".join(residual) or "none",
            )
            return
        if self.repair_deviation_reason:
            logger.warning(
                "%s: %d control(s) carry residual run-together damage against "
                "a measured %d. Permitted: %s",
                self.framework_id, len(residual), self.expected_residual_damage,
                self.repair_deviation_reason,
            )
            return
        raise ValueError(
            f"{self.framework_id}: {len(residual)} control(s) carry residual "
            f"run-together damage against a measured "
            f"{self.expected_residual_damage}: {', '.join(residual) or 'none'}. "
            f"Record why in repair_deviation_reason before moving the number."
        )


def main() -> None:
    Iso27001Parser().run()


if __name__ == "__main__":
    main()
