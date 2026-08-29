"""The AI link audit must stay disclosed, and its effect on the gate measurable.

`data/training/audit_corrections_log.json` records 56 gold-label corrections
TRACT applied to its own AI framework links. All 56 land inside the four
frameworks that make up the Campaign 2 test split, covering 25% of the 147-item
test corpus. Removing those items drops the headline delta from +0.1361 to
exactly the 0.10 gate value.

The audit went undisclosed through an entire campaign: it appears in no markdown
file, had no test, and was found by an adversarial premortem rather than by the
campaign that depended on it. These tests exist so that cannot recur silently.
They assert the audit is documented, that the stratification tooling still runs,
and that the two strata remain distinguishable -- not that any particular delta
holds, since re-running the campaign is allowed to move the numbers.

None of these tests load a model.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Final

import pytest

from tract.config import PROJECT_ROOT, TRAINING_DIR

AUDIT_LOG: Final[Path] = TRAINING_DIR / "audit_corrections_log.json"
AUDIT_CSV: Final[Path] = TRAINING_DIR / "ai_link_audit.csv"

# The frameworks the Campaign 2 test round held out. An audit correction landing
# outside this set would be a different situation -- it would touch training
# supervision rather than test gold -- so the set is named rather than derived.
AI_TEST_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({
    "mitre_atlas",
    "nist_ai_100_2",
    "owasp_ai_exchange",
    "owasp_llm_top10",
    "owasp_ml_top10",
})

# Where a reader looking for the campaign's caveats would actually go.
DISCLOSURE_TARGETS: Final[tuple[Path, ...]] = (
    PROJECT_ROOT / "docs" / "campaign2-results.md",
    PROJECT_ROOT / "results" / "phase1b" / "CAMPAIGN2.md",
)


@pytest.fixture(scope="module")
def audit_log() -> dict:
    if not AUDIT_LOG.is_file():
        pytest.skip(f"{AUDIT_LOG} absent")
    return json.loads(AUDIT_LOG.read_text(encoding="utf-8"))


def test_audit_log_is_internally_consistent(audit_log: dict) -> None:
    """corrections_applied must match the number of correction records.

    A log that disagrees with itself cannot support a stratification, and the
    stratified delta is the number the corrected campaign record reports.
    """
    assert len(audit_log["corrections"]) == audit_log["corrections_applied"]
    assert audit_log["ai_links_curated"] > 0


def test_every_correction_names_a_real_replacement(audit_log: dict) -> None:
    """Each correction must move gold from one CRE id to a different one.

    The corrections were parsed out of a free-text notes column by regex
    (scripts/phase0/curate_links.py). A regex that failed to match would
    plausibly yield an empty or unchanged id rather than an error, so the
    no-op case is asserted against explicitly.
    """
    cre_id = re.compile(r"^\d+-\d+$")
    for correction in audit_log["corrections"]:
        old, new = correction["old_cre_id"], correction["new_cre_id"]
        assert cre_id.match(old), f"malformed old_cre_id {old!r}"
        assert cre_id.match(new), f"malformed new_cre_id {new!r}"
        assert old != new, (
            f"correction on {correction['section_name']!r} is a no-op; a "
            "regex that failed to find a replacement would look exactly "
            "like this"
        )


def test_corrections_are_confined_to_the_test_split(audit_log: dict) -> None:
    """All corrections sit in AI test frameworks -- the asymmetry to disclose.

    This is not a defect to fix; it is the fact the disclosure exists to state.
    The test pins it so that if a future audit touches training frameworks too,
    the disclosure text is forced to be rewritten rather than quietly outgrown.
    """
    touched = {c["framework_id"] for c in audit_log["corrections"]}
    assert touched <= AI_TEST_FRAMEWORK_IDS, (
        f"audit now touches {touched - AI_TEST_FRAMEWORK_IDS} outside the AI "
        "test split; docs/campaign2-audit-disclosure.md describes an audit "
        "confined to the test split and no longer describes this file"
    )


def test_audit_is_disclosed_in_prose(audit_log: dict) -> None:
    """The audit must be named in a document a reader of the results would open.

    Campaign 2 published a gate decision computed on gold that this audit had
    rewritten, while no markdown file in the repository mentioned it. Grepping
    for the artifact names is a blunt check, but it is exactly the check that
    would have failed during Campaign 2 and did not exist.
    """
    names = ("audit_corrections_log", "ai_link_audit")
    found = [
        path.name for path in DISCLOSURE_TARGETS
        if path.is_file()
        and any(n in path.read_text(encoding="utf-8") for n in names)
    ]
    assert found, (
        "the AI link audit rewrote 25% of the Campaign 2 test gold and is "
        f"named in none of {[p.name for p in DISCLOSURE_TARGETS]}. Any result "
        "computed on that gold is being reported without its caveat."
    )


def test_stratification_tooling_runs_and_separates_the_strata() -> None:
    """The stratified report must build and keep the two strata distinguishable.

    Asserts the shape of the finding, not its magnitude: audit-touched items
    carry a larger paired delta than untouched ones, because the relabelling
    moved gold onto high-degree hubs that fine-tuning learns and a zero-shot
    encoder does not privilege. If a re-run ever erased that gap the disclosure
    would need rewriting, and this test is what would force it.
    """
    pytest.importorskip("numpy")
    from tract.config import PHASE1B_RESULTS_DIR

    run_dir = PHASE1B_RESULTS_DIR / "c2r_TEST_A3_prose_sw_qwen06b"
    if not (run_dir / "fold_MITRE_ATLAS" / "fold_result.json").is_file():
        pytest.skip("Campaign 2 test round results not present")

    from scripts.analysis.audit_stratified_delta import (
        build_rows,
        load_audit_touched_keys,
        score_stratum,
    )

    rows = build_rows(run_dir, load_audit_touched_keys())
    assert len(rows) == 147

    touched = [r for r in rows if r["audit_touched"]]
    untouched = [r for r in rows if not r["audit_touched"]]
    assert touched and untouched

    hit = score_stratum(touched, "touched")
    miss = score_stratum(untouched, "untouched")
    assert hit["delta_mean"] > miss["delta_mean"], (
        "audit-touched items no longer carry a larger delta than untouched "
        "ones; the disclosure's stated mechanism no longer holds"
    )
    # The gate decision turns on this quantity and the campaign never reported
    # it. Its presence is the point; its value is allowed to move.
    assert 0.0 <= miss["p_delta_le_gate"] <= 1.0
