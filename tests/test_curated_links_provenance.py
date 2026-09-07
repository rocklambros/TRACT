"""`hub_links_curated.jsonl` is gold for 20+ call sites and had no guard of its own.

WHAT IT IS. 4,405 curated (framework, section, hub) links. `load_curated_links`
reads it, and everything downstream treats the result as ground truth: the
evaluation corpus, the training positives, the bridge pipeline, the degree
analysis, the crosswalk export. A Tier-3 triple that reaches this file is
inherited silently by every one of them, and by every gate denominator computed
from them.

WHY THE EXISTING GUARD IS NOT ENOUGH. `tests/test_tier3_quarantine.py` already
checks that no `review_export.json` decision became a curated link. That is a
blocklist against ONE known Tier-3 source. It cannot catch contamination from a
source nobody has thought of yet -- and this repository has at least one other:
`results/bridge/` holds 46 model-proposed, human-ratified hub pairs which are
Tier 3 on their face (`results/bridge/PROVENANCE.md`) and which already reached
`cre_hierarchy.json` as `related_hub_ids`.

SO THIS TESTS PROVENANCE POSITIVELY. Every curated triple must trace to a known
non-model origin. Two exist, and they are exhaustive:

  1. `data/training/hub_links.jsonl` -- the pre-review baseline, committed
     2026-04-28, before both the curated file and any model review.
  2. `data/training/audit_corrections_log.json` -- the AI link audit of
     2026-04-29, whose 56 corrections are human-authored and documented in
     docs/campaign2-results.md Section 13.

Measured at the time of writing: 4,368 distinct triples, 22 of them absent from
the baseline, and all 22 carry a hub id introduced by a documented audit
correction. The unaccounted set is EMPTY. A triple that is neither has no
legitimate origin in this repository, and a positive test says so whatever the
source turns out to be.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pytest

from tract.config import PROJECT_ROOT

BASELINE_PATH: Final[Path] = PROJECT_ROOT / "data" / "training" / "hub_links.jsonl"
AUDIT_PATH: Final[Path] = (
    PROJECT_ROOT / "data" / "training" / "audit_corrections_log.json"
)

Triple = tuple[str, str, str]


def unaccounted_triples(
    curated: set[Triple],
    baseline: set[Triple],
    audit_hub_ids: set[str],
) -> set[Triple]:
    """Curated triples with no legitimate origin.

    Extracted rather than inlined so the guard can be exercised against a
    deliberately contaminated input. A guard that has only ever been run
    against clean data has not been shown to fail, and this project has
    already shipped one tripwire that could not go red.
    """
    return {t for t in curated - baseline if t[2] not in audit_hub_ids}


@pytest.fixture(scope="module")
def curated() -> set[Triple]:
    from scripts.phase0.common import load_curated_links

    return {
        (link.standard_name, link.section_id, link.cre_id)
        for link in load_curated_links()
    }


@pytest.fixture(scope="module")
def baseline() -> set[Triple]:
    if not BASELINE_PATH.is_file():
        pytest.skip(f"{BASELINE_PATH} absent; provenance cannot be established")
    out: set[Triple] = set()
    with BASELINE_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                out.add((row["standard_name"], row["section_id"], row["cre_id"]))
    return out


@pytest.fixture(scope="module")
def audit_hub_ids() -> set[str]:
    if not AUDIT_PATH.is_file():
        pytest.skip(f"{AUDIT_PATH} absent; provenance cannot be established")
    payload = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    return {c["new_cre_id"] for c in payload["corrections"]}


class TestTheGuardCanActuallyFail:
    """Exercise the check against contaminated input, not only clean input."""

    def test_a_triple_with_no_origin_is_reported(self) -> None:
        contaminated = {("MITRE ATLAS", "AML.T0001", "999-999")}
        assert unaccounted_triples(contaminated, set(), set()) == contaminated

    def test_a_baseline_triple_is_accepted(self) -> None:
        t: Triple = ("MITRE ATLAS", "AML.T0001", "111-111")
        assert unaccounted_triples({t}, {t}, set()) == set()

    def test_an_audit_introduced_hub_is_accepted(self) -> None:
        t: Triple = ("MITRE ATLAS", "AML.T0001", "222-222")
        assert unaccounted_triples({t}, set(), {"222-222"}) == set()

    def test_the_audit_allowance_is_scoped_to_its_own_hub_ids(self) -> None:
        """A different hub id must not be waved through by the audit clause."""
        t: Triple = ("MITRE ATLAS", "AML.T0001", "333-333")
        assert unaccounted_triples({t}, set(), {"222-222"}) == {t}


class TestEveryCuratedLinkHasALegitimateOrigin:
    def test_no_curated_triple_is_unaccounted_for(
        self,
        curated: set[Triple],
        baseline: set[Triple],
        audit_hub_ids: set[str],
    ) -> None:
        """The load-bearing assertion.

        Positive, not a blocklist: it does not ask "did this come from
        review_export" but "can this be traced to a human origin at all".
        """
        orphans = unaccounted_triples(curated, baseline, audit_hub_ids)
        assert not orphans, (
            f"{len(orphans)} curated links trace to neither the pre-review "
            f"baseline nor a documented audit correction, so their provenance "
            f"is unknown and every downstream gate inherits them: "
            f"{sorted(orphans)[:20]}"
        )

    def test_the_residual_is_explained_by_the_audit_and_is_not_empty(
        self,
        curated: set[Triple],
        baseline: set[Triple],
        audit_hub_ids: set[str],
    ) -> None:
        """Guards the guard: the two clauses must both still do work.

        If the curated file ever became a subset of the baseline, the audit
        clause would be vacuous and the test above would keep passing while
        having stopped discriminating.
        """
        residual = curated - baseline
        assert residual, (
            "No curated triple is outside the baseline. The audit clause is "
            "now vacuous, so this suite no longer tests what it claims to."
        )
        assert all(t[2] in audit_hub_ids for t in residual)


class TestTheBridgeSetHasNotReachedTheGoldFile:
    """Name the specific Tier-3 path that the positive test covers generically.

    Bridges are hub->hub, so they cannot become a (framework, section, hub)
    triple directly. The reachable mechanism is transitive: a control linked to
    AI hub X acquires a link to the traditional hub X bridges to. This asserts
    no such triple exists, so the mechanism is documented as considered rather
    than merely covered by accident.
    """

    def test_no_curated_link_lands_on_a_bridge_target_without_baseline_support(
        self,
        curated: set[Triple],
        baseline: set[Triple],
        audit_hub_ids: set[str],
    ) -> None:
        report = PROJECT_ROOT / "results" / "bridge" / "bridge_report.json"
        if not report.is_file():
            pytest.skip(f"{report} absent")
        payload = json.loads(report.read_text(encoding="utf-8"))
        targets = {
            c["trad_hub_id"] for c in payload["candidates"]
            if c["status"] == "accepted"
        }
        # Indexed by exact key rather than a chain of .get() fallbacks. A
        # fallback chain that misses turns this guard into a silent skip, which
        # is the shape of defect this repository's premortems keep finding.
        assert targets, (
            "bridge_report.json has no accepted candidates; either the schema "
            "changed or the file is not what this test believes it is."
        )

        leaked = {
            t for t in unaccounted_triples(curated, baseline, audit_hub_ids)
            if t[2] in targets
        }
        assert not leaked, (
            f"Curated links land on Tier-3 bridge targets with no baseline or "
            f"audit origin: {sorted(leaked)[:20]}"
        )
