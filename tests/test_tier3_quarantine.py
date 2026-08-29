"""`results/review/review_export.json` is Tier 3 and must never become gold.

WHAT IT IS. 898 assignments that a TRACT model proposed and a human then ratified
in the model's presence. 693 (77.2%) were accepted exactly as proposed. It spans
11 frameworks, and 215 of those items -- 211 `mitre_atlas` and 4
`owasp_llm_top10` -- belong to frameworks in the Campaign 2 TEST split.

WHY IT MATTERS. Under the Campaign 3 provenance tiers, a label produced by or
ratified in the presence of a model is Tier 3 and may not sit in a gate
denominator at any ratio. This file is the single largest pool of ready-made
(framework, section, hub) triples in the repository, it covers exactly the
frameworks whose eval set is too small, and turning it into training or eval
labels is a ten-minute job. The pressure to do that will be real, because the
honest alternative is 25 expert-hours of blind curation.

IT IS NOT HYPOTHETICAL. 63 of the 898 items land on a (framework, section) that
OpenCRE independently curated. The pipeline agrees with that gold on 47 of 63 --
74.6%. On 12 of the 16 disagreements the reviewer marked the model's proposal
`accepted`, ratifying a hub that contradicts OpenCRE's own label; on the other 4
they reassigned to a third hub that also disagrees. A 74.6% agreement rate is
the measured cost of admitting these labels, and it is the reason for the rule
rather than a decoration on it.

These tests do not forbid using the file. They forbid it becoming ground truth
silently.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Final

import pytest

from tract.config import PROJECT_ROOT

REVIEW_EXPORT: Final[Path] = PROJECT_ROOT / "results" / "review" / "review_export.json"

# The frameworks Campaign 2 held out and scored. Contamination here is worse
# than elsewhere: it would put model-derived labels in the denominator of the
# number the project reports.
TEST_SPLIT_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({
    "mitre_atlas", "nist_ai_100_2", "owasp_ai_exchange",
    "owasp_llm_top10", "owasp_ml_top10",
})

# framework_id in review_export.json -> standard_name in hub_links_curated.jsonl
FRAMEWORK_ID_TO_STANDARD: Final[dict[str, str]] = {
    "mitre_atlas": "MITRE ATLAS",
    "nist_ai_100_2": "NIST AI 100-2",
    "owasp_ai_exchange": "OWASP AI Exchange",
    "owasp_llm_top10": "OWASP Top10 for LLM",
    "owasp_ml_top10": "OWASP Top10 for ML",
}


@pytest.fixture(scope="module")
def review_items() -> list[dict]:
    if not REVIEW_EXPORT.is_file():
        pytest.skip(f"{REVIEW_EXPORT} absent")
    return json.loads(REVIEW_EXPORT.read_text(encoding="utf-8"))["predictions"]


@pytest.fixture(scope="module")
def curated_gold() -> dict[tuple[str, str], set[str]]:
    """(standard_name, section_id) -> hub ids, from the curated link file."""
    from scripts.phase0.common import load_curated_links

    gold: dict[tuple[str, str], set[str]] = defaultdict(set)
    for link in load_curated_links():
        gold[(link.standard_name, link.section_id)].add(link.cre_id)
    return dict(gold)


def test_the_file_is_still_what_this_quarantine_describes(
    review_items: list[dict],
) -> None:
    """Pin the shape, so the prose above cannot quietly stop being true.

    If the file is regenerated or extended, these numbers move and the
    quarantine's rationale has to be rewritten rather than inherited.
    """
    assert len(review_items) == 898
    test_split = [
        i for i in review_items
        if i["framework_id"] in TEST_SPLIT_FRAMEWORK_IDS
    ]
    assert len(test_split) == 215, (
        f"{len(test_split)} items now belong to test-split frameworks, not 215. "
        "The contamination surface changed; re-derive it before trusting this file."
    )


def test_no_review_decision_has_become_a_curated_link(
    review_items: list[dict], curated_gold: dict[tuple[str, str], set[str]],
) -> None:
    """The load-bearing assertion: model-derived labels are not in the gold file.

    A curated link matching a review decision is not by itself a leak -- on 47
    of the 63 overlapping items the model simply agreed with a hub OpenCRE had
    already assigned. Telling the two apart needs a baseline that predates the
    review, and `data/training/hub_links.jsonl` is one: it was committed
    2026-04-28, before both the curated file and review_export.json.

    There is a SECOND legitimate origin, and missing it produces false alarms.
    The AI link audit ran 2026-04-29, between the baseline and the curated file,
    and introduced 26 hub ids that the baseline does not contain (see
    docs/campaign2-results.md §13). review_export.json was generated 2026-05-03,
    by which time those corrections were already the gold the model was shown --
    so the model agreeing with one is expected, not contamination. Four such
    triples exist and are legitimate.

    So the invariant is: every curated (framework, section, hub) triple that a
    review decision would also produce must trace to the pre-review baseline OR
    to a documented audit correction. A triple with neither origin has only one
    plausible source.
    """
    baseline_path = PROJECT_ROOT / "data" / "training" / "hub_links.jsonl"
    if not baseline_path.is_file():
        pytest.skip(f"{baseline_path} absent; cannot separate leak from agreement")

    baseline: set[tuple[str, str, str]] = set()
    with baseline_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            baseline.add((row["standard_name"], row["section_id"], row["cre_id"]))

    audit_path = PROJECT_ROOT / "data" / "training" / "audit_corrections_log.json"
    audit_hubs: set[str] = set()
    if audit_path.is_file():
        audit_hubs = {
            c["new_cre_id"] for c in
            json.loads(audit_path.read_text(encoding="utf-8"))["corrections"]
        }

    leaked: list[tuple[str, str, str]] = []
    for item in review_items:
        standard = FRAMEWORK_ID_TO_STANDARD.get(item["framework_id"])
        if standard is None:
            continue  # framework has no curated links at all; nothing to leak into
        section = item.get("section_id", "")
        decided = item.get("reviewer_hub_id") or item["assigned_hub_id"]
        triple = (standard, section, decided)
        if decided not in curated_gold.get((standard, section), set()):
            continue  # the decision is not in the gold file at all -- the good case
        if triple in baseline:
            continue  # pre-existing OpenCRE link the model happened to match
        if decided in audit_hubs:
            continue  # audit correction, already gold when the model saw it
        leaked.append(triple)

    assert not leaked, (
        f"{len(leaked)} (framework, section, hub) triples are in the curated "
        f"link file, match a review decision, and trace to neither the "
        f"pre-review baseline nor a documented audit correction: "
        f"e.g. {leaked[:3]}. The ready source is "
        "results/review/review_export.json, which is Tier 3 -- model-proposed "
        "and human-ratified in the model's presence. A gate computed on these "
        "is measuring the model against itself."
    )


def test_ratified_disagreements_did_not_overwrite_opencre_gold(
    review_items: list[dict], curated_gold: dict[tuple[str, str], set[str]],
) -> None:
    """Where the pipeline contradicted OpenCRE, OpenCRE's label must still stand.

    The sharpest version of the risk. On 12 items the reviewer accepted a
    model proposal that disagrees with the independently curated hub. If one of
    those decisions were ever applied to the link file, the test set's gold
    would have been replaced by a model's opinion that a human waved through.
    """
    overwritten: list[tuple[str, str, str]] = []
    for item in review_items:
        standard = FRAMEWORK_ID_TO_STANDARD.get(item["framework_id"])
        if standard is None:
            continue
        key = (standard, item.get("section_id", ""))
        gold = curated_gold.get(key)
        if not gold:
            continue
        decided = item.get("reviewer_hub_id") or item["assigned_hub_id"]
        if decided not in gold and decided in curated_gold.get(key, set()):
            overwritten.append((standard, key[1], decided))

    assert not overwritten, (
        f"a review decision that contradicts OpenCRE's curated hub is now IN "
        f"the curated file: {overwritten[:3]}"
    )


def test_the_quarantine_is_documented_where_someone_would_look(
    review_items: list[dict],
) -> None:
    """A test nobody can find is a test nobody honours.

    The file sits in a results directory beside three other artifacts that ARE
    safe to reuse. Someone reaching for 898 ready-made labels needs to meet the
    reason not to before they meet the file.
    """
    marker = REVIEW_EXPORT.parent / "PROVENANCE.md"
    assert marker.is_file(), (
        f"{marker} is missing. review_export.json is Tier 3 and sits "
        "unlabelled next to reusable artifacts."
    )
    text = marker.read_text(encoding="utf-8")
    for needle in ("Tier 3", "review_export.json"):
        assert needle in text, f"{marker} does not mention {needle!r}"
