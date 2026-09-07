"""The OpenCRE export filtered by a blocklist of one.

`tract/export/filters.py` and `tract/export/canonical.py` both selected rows
`WHERE review_status = 'accepted' AND provenance != 'ground_truth_T1-AI'`. That
excludes exactly one value and admits every other by default.

Measured on the committed `results/phase1c/crosswalk.db`: **558 rows with
`provenance='active_learning_round_2'` pass every clause**, against `PRD.md`'s
claim that *"nothing model-derived is downstream in the published model, the
published dataset, the OpenCRE fork import, or the Phase 5B export"*. Premortem
round 1 parked this as a tail risk whose trigger was "any move toward RFC
submission"; Phase 2C's design makes upstream proposal part of the round, so the
trigger fired.

The owner's decision (2026-09-06) was to make the filter an allowlist and
correct the PRD sentence, **not** to drop the rows. So the export still carries
active-learning output; what changed is that it now names what it permits and
refuses a provenance nobody has classified, rather than shipping it by default.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tract.config import (
    PHASE5_EXPORTABLE_PROVENANCES,
    PHASE5_GROUND_TRUTH_PROVENANCE,
    PHASE5_WITHHELD_PROVENANCES,
    PROJECT_ROOT,
)

CROSSWALK_DB = PROJECT_ROOT / "results" / "phase1c" / "crosswalk.db"


class TestTheConstantsAreCoherent:
    def test_the_two_sets_are_disjoint(self) -> None:
        assert not (PHASE5_EXPORTABLE_PROVENANCES & PHASE5_WITHHELD_PROVENANCES)

    def test_ground_truth_is_withheld_not_exported(self) -> None:
        assert PHASE5_GROUND_TRUTH_PROVENANCE in PHASE5_WITHHELD_PROVENANCES
        assert PHASE5_GROUND_TRUTH_PROVENANCE not in PHASE5_EXPORTABLE_PROVENANCES

    def test_the_allowlist_is_not_empty(self) -> None:
        """An empty allowlist exports nothing, which would pass every other test."""
        assert PHASE5_EXPORTABLE_PROVENANCES


class TestEveryProvenanceInTheDatabaseIsClassified:
    """The check that would have caught this: nothing unaccounted for."""

    @pytest.fixture(scope="class")
    def provenances(self) -> set[str]:
        if not CROSSWALK_DB.is_file():
            pytest.skip(f"{CROSSWALK_DB} absent")
        conn = sqlite3.connect(CROSSWALK_DB)
        try:
            rows = conn.execute(
                "SELECT DISTINCT provenance FROM assignments"
            ).fetchall()
        finally:
            conn.close()
        return {r[0] for r in rows if r[0] is not None}

    def test_no_provenance_is_unclassified(self, provenances: set[str]) -> None:
        known = PHASE5_EXPORTABLE_PROVENANCES | PHASE5_WITHHELD_PROVENANCES
        unclassified = provenances - known
        assert not unclassified, (
            f"Provenance values in crosswalk.db that neither set names: "
            f"{sorted(unclassified)}. Under the old blocklist these would have "
            "been exported by default. Classify each one."
        )

    def test_the_database_still_contains_what_this_suite_assumes(
        self, provenances: set[str]
    ) -> None:
        """Guards the guard: an empty table passes the test above."""
        assert provenances, "no provenance values at all; the test above is vacuous"


class TestTheFiltersUseTheAllowlist:
    """Source inspection, because both filters are SQL strings.

    A blocklist and an allowlist are one operator apart, and the difference is
    invisible in a passing export.
    """

    @pytest.mark.parametrize(
        "relative", ["tract/export/filters.py", "tract/export/canonical.py"]
    )
    def test_no_negated_provenance_comparison_remains(self, relative: str) -> None:
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "provenance != ?" not in source, (
            f"{relative} still filters by inequality, which admits every "
            "unlisted provenance by default."
        )

    @pytest.mark.parametrize(
        "relative", ["tract/export/filters.py", "tract/export/canonical.py"]
    )
    def test_it_references_the_allowlist(self, relative: str) -> None:
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "PHASE5_EXPORTABLE_PROVENANCES" in source


class TestThePRDNoLongerClaimsTheExportIsModelFree:
    def test_the_prd_states_what_the_export_actually_carries(self) -> None:
        text = (PROJECT_ROOT / "PRD.md").read_text(encoding="utf-8")
        assert "active_learning_round_2" in text, (
            "PRD.md must name what the export carries. It claimed nothing "
            "model-derived was downstream while 558 model-derived rows passed "
            "every clause of the export filter."
        )
