"""Tests for the LLM judge panel analysis.

The analysis is the deliverable that has to be correct before the real data
arrives, because once real numbers exist nobody will re-derive them by hand.
So the fixtures here are constructed so the right answer is known in advance:
a panel that agrees perfectly with the human, one that agrees perfectly with
OpenCRE, one split two to one, and one where everybody disagrees with
everybody.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.analyze_panel_agreement import (
    Annotator,
    ItemMeta,
    _agreement,
    _distance_table,
    _majority_annotator,
    _section_capec_contingency,
    build_report,
    majority_vote,
    primary_of,
)
from tract.panel import (
    distance_category,
    extract_json_array,
    model_slug,
    parse_hub_reference,
    parse_judge_response,
)

# ── Fixtures ─────────────────────────────────────────────────────────────
# A miniature taxonomy with a known shape, so every distance category has an
# unambiguous expected answer:
#
#   Alpha                        (100-000)
#     Alpha > Net                (110-000)
#       Alpha > Net > Scan       (111-000)
#       Alpha > Net > Sniff      (112-000)
#     Alpha > Crypto             (120-000)
#   Beta                         (200-000)
#     Beta > Logs                (210-000)

HUB_REFERENCE = """\
# CRE Hub Reference

## Alpha (100-000)

### 100-000 -- Alpha
Path: Alpha
Description: (none on record for this hub)

### 110-000 -- Net
Path: Alpha > Net
Description: Networking.

### 111-000 -- Scan
Path: Alpha > Net > Scan
Description: Scanning.

### 112-000 -- Sniff
Path: Alpha > Net > Sniff
Description: Sniffing.

### 120-000 -- Crypto
Path: Alpha > Crypto
Description: Crypto.

## Beta (200-000)

### 200-000 -- Beta
Path: Beta
Description: (none on record for this hub)

### 210-000 -- Logs
Path: Beta > Logs
Description: Logs.
"""

# Six items: four CAPEC, two from another framework, so per-framework
# slicing and the CAPEC contingency both have something to bite on.
ITEM_META: dict[int, ItemMeta] = {
    1: ItemMeta("capec", "validation"),
    2: ItemMeta("capec", "validation"),
    3: ItemMeta("capec", "validation"),
    4: ItemMeta("capec", "validation"),
    5: ItemMeta("cwe", "validation"),
    6: ItemMeta("mitre_atlas", "test"),
}

# OpenCRE says every item is 210-000 (Beta > Logs). That is deliberately a
# different branch from the "reading" the human and panel share below, which
# is what the real CAPEC disagreement looks like.
VALID_GOLD: dict[int, list[str]] = {i: ["210-000"] for i in ITEM_META}
PRIMARY_GOLD: dict[int, str] = {i: "210-000" for i in ITEM_META}


def _annotator(name: str, hubs: dict[int, str], kind: str = "panel") -> Annotator:
    return Annotator(
        name=name,
        kind=kind,  # type: ignore[arg-type]
        answers={
            index: {
                "item_index": index,
                "primary_hub_id": hub,
                "acceptable_hub_ids": [hub] if hub else [],
                "confidence": "high",
                "notes": "",
            }
            for index, hub in hubs.items()
        },
        provenance={"model_id": name, "cost_usd": 0.0},
    )


HUMAN_HUBS = {1: "111-000", 2: "111-000", 3: "112-000", 4: "120-000", 5: "111-000", 6: "210-000"}
HUMAN = _annotator("human", HUMAN_HUBS, kind="human")

# Agrees with the human on every item.
PANEL_LIKE_HUMAN = [_annotator(f"m{i}", dict(HUMAN_HUBS)) for i in range(3)]

# Agrees with OpenCRE on every item.
PANEL_LIKE_OPENCRE = [
    _annotator(f"m{i}", {index: "210-000" for index in ITEM_META}) for i in range(3)
]

ALL_INDICES = sorted(ITEM_META)
CAPEC_INDICES = [i for i in ALL_INDICES if ITEM_META[i].framework_id == "capec"]


# ── Hub reference and hierarchy distance ─────────────────────────────────

def test_parse_hub_reference_reads_every_hub_and_path() -> None:
    paths = parse_hub_reference(HUB_REFERENCE)
    assert len(paths) == 7
    assert paths["111-000"] == ["Alpha", "Net", "Scan"]
    assert paths["210-000"] == ["Beta", "Logs"]


def test_parse_hub_reference_rejects_heading_without_path() -> None:
    broken = "### 999-999 -- Orphan\nDescription: no path line\n"
    with pytest.raises(ValueError, match="no Path line"):
        parse_hub_reference(broken)


def test_parse_hub_reference_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="no parseable hubs"):
        parse_hub_reference("# nothing here\n")


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("111-000", "111-000", "same"),
        ("111-000", "110-000", "ancestor/descendant"),  # Scan is under Net
        ("110-000", "111-000", "ancestor/descendant"),  # and the reverse
        ("111-000", "112-000", "sibling"),              # Scan and Sniff share Net
        ("111-000", "120-000", "same branch"),          # both under Alpha, not siblings
        ("111-000", "210-000", "different branch"),     # Alpha versus Beta
        ("111-000", "404-404", "unknown"),              # id not in the reference
    ],
)
def test_distance_category(left: str, right: str, expected: str) -> None:
    paths = parse_hub_reference(HUB_REFERENCE)
    assert distance_category(left, right, paths) == expected


def test_distance_table_counts_only_disagreements() -> None:
    paths = parse_hub_reference(HUB_REFERENCE)
    hubs = {i: HUMAN_HUBS[i] for i in ALL_INDICES}
    counts, total = _distance_table(hubs, PRIMARY_GOLD, ALL_INDICES, paths)
    # Item 6 matches gold exactly, so five disagreements remain, all of them
    # Alpha-branch answers against a Beta-branch gold.
    assert total == 5
    assert counts["different branch"] == 5


def test_distance_table_reproduces_published_human_split() -> None:
    """The real human answers must still give 9.3 / 3.7 / 17.8 / 69.2.

    This is the anchor for the whole section. If the categories drift, the
    panel rows stop being comparable to the human's published figures.
    """
    root = Path(__file__).resolve().parents[1]
    study = root / "results" / "ceiling_study"
    paths = parse_hub_reference((study / "hub_reference.md").read_text(encoding="utf-8"))
    key = json.loads((study / "ceiling_answer_key.json").read_text(encoding="utf-8"))
    human = json.loads((study / "answers_human_rock.json").read_text(encoding="utf-8"))

    valid = {int(e["item_index"]): set(e["valid_gold_hub_ids"]) for e in key["answers"]}
    primary = {int(e["item_index"]): str(e["primary_gold_hub_id"]) for e in key["answers"]}
    # Only alpha-1 misses are categorised, matching the human analysis.
    hubs = {
        int(r["item_index"]): str(r["primary_hub_id"])
        for r in human["items"]
        if str(r["primary_hub_id"]) and str(r["primary_hub_id"]) not in valid[int(r["item_index"])]
    }
    counts, total = _distance_table(hubs, primary, sorted(hubs), paths)

    assert total == 107
    assert counts["ancestor/descendant"] == 10
    assert counts["sibling"] == 4
    assert counts["same branch"] == 19
    assert counts["different branch"] == 74
    assert round(100 * counts["different branch"] / total, 1) == 69.2


# ── Majority vote ────────────────────────────────────────────────────────

def test_majority_vote_unanimous() -> None:
    hub, votes = majority_vote(PANEL_LIKE_HUMAN, 1)
    assert (hub, votes) == ("111-000", 3)


def test_majority_vote_two_to_one() -> None:
    split = [
        _annotator("a", {1: "111-000"}),
        _annotator("b", {1: "111-000"}),
        _annotator("c", {1: "210-000"}),
    ]
    assert majority_vote(split, 1) == ("111-000", 2)


def test_majority_vote_three_way_tie_is_reported_as_one_vote() -> None:
    """A 1-1-1 split must not be presented as a majority carrying weight."""
    tied = [
        _annotator("a", {1: "111-000"}),
        _annotator("b", {1: "112-000"}),
        _annotator("c", {1: "210-000"}),
    ]
    hub, votes = majority_vote(tied, 1)
    assert votes == 1
    assert hub == "111-000"  # deterministic: first model in panel order


def test_majority_vote_three_two_split_on_five_member_panel() -> None:
    """The case five members exist to resolve: a 3-2 split has a real winner."""
    five = [
        _annotator("a", {1: "111-000"}),
        _annotator("b", {1: "111-000"}),
        _annotator("c", {1: "111-000"}),
        _annotator("d", {1: "210-000"}),
        _annotator("e", {1: "210-000"}),
    ]
    assert majority_vote(five, 1) == ("111-000", 3)


def test_odd_panel_cannot_produce_a_two_two_tie() -> None:
    """An even panel can deadlock on exactly the contested items that matter.

    With four members split 2-2 the reported majority carries only two votes,
    which is the ambiguity a five-member panel is sized to avoid.
    """
    four = [
        _annotator("a", {1: "111-000"}),
        _annotator("b", {1: "111-000"}),
        _annotator("c", {1: "210-000"}),
        _annotator("d", {1: "210-000"}),
    ]
    _, tied_votes = majority_vote(four, 1)
    assert tied_votes == 2, "a 2-2 split must not be presented as a clear majority"

    five = [*four, _annotator("e", {1: "210-000"})]
    hub, votes = majority_vote(five, 1)
    assert (hub, votes) == ("210-000", 3)


def test_majority_vote_ignores_unanswered_members() -> None:
    partial = [
        _annotator("a", {1: ""}),
        _annotator("b", {1: "112-000"}),
        _annotator("c", {1: "112-000"}),
    ]
    assert majority_vote(partial, 1) == ("112-000", 2)


def test_majority_vote_with_no_answers_at_all() -> None:
    assert majority_vote([_annotator("a", {1: ""})], 1) == ("", 0)


# ── Agreement ────────────────────────────────────────────────────────────

def test_agreement_is_perfect_when_panel_copies_human() -> None:
    interval, hits, n = _agreement(HUMAN, PANEL_LIKE_HUMAN[0], ALL_INDICES)
    assert (hits, n) == (6, 6)
    assert interval is not None and interval.point == 1.0


def test_agreement_is_minimal_when_panel_copies_opencre() -> None:
    interval, hits, n = _agreement(HUMAN, PANEL_LIKE_OPENCRE[0], ALL_INDICES)
    # Only item 6, where the human happened to match OpenCRE.
    assert (hits, n) == (1, 6)
    assert interval is not None and interval.point == pytest.approx(1 / 6)


def test_agreement_skips_items_either_side_left_blank() -> None:
    partial = _annotator("p", {**HUMAN_HUBS, 1: "", 2: ""})
    _, hits, n = _agreement(HUMAN, partial, ALL_INDICES)
    assert n == 4, "unanswered items must leave the denominator, not count as misses"
    assert hits == 4


def test_agreement_with_no_overlap_returns_none() -> None:
    empty = _annotator("p", {i: "" for i in ALL_INDICES})
    interval, hits, n = _agreement(HUMAN, empty, ALL_INDICES)
    assert (interval, hits, n) == (None, 0, 0)


def test_majority_annotator_holds_the_vote() -> None:
    majority = _majority_annotator(PANEL_LIKE_HUMAN, ALL_INDICES)
    assert primary_of(majority, 3) == "112-000"
    assert "3 of 3" in majority.answers[3]["notes"]


# ── The CAPEC contingency, the headline ──────────────────────────────────

def test_contingency_panel_agrees_with_human_against_opencre() -> None:
    """Hypothesis (a): the labels are poor.

    All four CAPEC items have the human and panel converging on a hub that
    OpenCRE did not choose, so all four must land in the headline cell.
    """
    _, headline = _section_capec_contingency(
        PANEL_LIKE_HUMAN, HUMAN, PRIMARY_GOLD, VALID_GOLD, ITEM_META
    )
    assert headline == 4


def test_contingency_panel_siding_with_opencre_yields_zero_headline() -> None:
    """Hypothesis (b): the human is idiosyncratic.

    The panel matches OpenCRE everywhere, so the headline cell must be empty
    and the disagreement must be attributed to the human alone.
    """
    lines, headline = _section_capec_contingency(
        PANEL_LIKE_OPENCRE, HUMAN, PRIMARY_GOLD, VALID_GOLD, ITEM_META
    )
    assert headline == 0
    body = "\n".join(lines)
    assert "| panel matches OpenCRE, human does not | 4 |" in body


def test_contingency_two_to_one_split_follows_the_majority() -> None:
    """A 2-1 panel must be scored on its majority, not on any single member."""
    split = [
        _annotator("a", dict(HUMAN_HUBS)),
        _annotator("b", dict(HUMAN_HUBS)),
        _annotator("c", {index: "210-000" for index in ITEM_META}),
    ]
    _, headline = _section_capec_contingency(
        split, HUMAN, PRIMARY_GOLD, VALID_GOLD, ITEM_META
    )
    assert headline == 4, "the two-member majority reading should carry the vote"


def test_contingency_minority_agreeing_with_human_does_not_carry() -> None:
    """Two members siding with the human against three must not set the headline.

    The headline counts the panel *majority*, so a loud minority that happens
    to match the human cannot inflate it.
    """
    split = [
        _annotator("a", dict(HUMAN_HUBS)),
        _annotator("b", dict(HUMAN_HUBS)),
        _annotator("c", {index: "210-000" for index in ITEM_META}),
        _annotator("d", {index: "210-000" for index in ITEM_META}),
        _annotator("e", {index: "210-000" for index in ITEM_META}),
    ]
    lines, headline = _section_capec_contingency(
        split, HUMAN, PRIMARY_GOLD, VALID_GOLD, ITEM_META
    )
    assert headline == 0
    assert "| panel matches OpenCRE, human does not | 4 |" in "\n".join(lines)


def test_contingency_everyone_disagrees_with_everyone() -> None:
    """No two annotators agree, so nothing may land in the headline cell."""
    scattered = [
        _annotator("a", {1: "111-000", 2: "111-000", 3: "111-000", 4: "111-000"}),
        _annotator("b", {1: "112-000", 2: "112-000", 3: "112-000", 4: "112-000"}),
        _annotator("c", {1: "120-000", 2: "120-000", 3: "120-000", 4: "120-000"}),
    ]
    # Human picks a fourth distinct hub on every CAPEC item.
    human = _annotator(
        "human", {1: "100-000", 2: "100-000", 3: "100-000", 4: "100-000"}, kind="human"
    )
    lines, headline = _section_capec_contingency(
        scattered, human, PRIMARY_GOLD, VALID_GOLD, ITEM_META
    )
    assert headline == 0
    assert "| all three differ | 4 |" in "\n".join(lines)


def test_contingency_ignores_items_the_human_skipped() -> None:
    human = _annotator("human", {**HUMAN_HUBS, 1: "", 2: ""}, kind="human")
    lines, headline = _section_capec_contingency(
        PANEL_LIKE_HUMAN, human, PRIMARY_GOLD, VALID_GOLD, ITEM_META
    )
    assert headline == 2, "only items 3 and 4 remain scorable among the CAPEC four"
    assert "2 of 2 CAPEC items" in "\n".join(lines)


# ── Judge response parsing ───────────────────────────────────────────────

def test_parse_judge_response_happy_path() -> None:
    text = json.dumps([
        {"item_index": 1, "primary_hub_id": "111-000", "acceptable_hub_ids": ["111-000"],
         "confidence": "high", "notes": ""},
        {"item_index": 2, "primary_hub_id": "112-000", "acceptable_hub_ids": [],
         "confidence": "low", "notes": "vague"},
    ])
    rows, counters = parse_judge_response(text, [1, 2], {"111-000", "112-000"}, 5)
    assert [r["primary_hub_id"] for r in rows] == ["111-000", "112-000"]
    assert counters == {"invented_primary": 0, "truncated_acceptable": 0, "missing": 0}


def test_parse_judge_response_tolerates_fences_and_prose() -> None:
    text = 'Here you go:\n```json\n[{"item_index": 1, "primary_hub_id": "111-000"}]\n```\nDone.'
    rows, _ = parse_judge_response(text, [1], {"111-000"}, 5)
    assert rows[0]["primary_hub_id"] == "111-000"


def test_parse_judge_response_keeps_invented_ids_rather_than_blanking() -> None:
    """Blanking an invented id would drop the item and inflate the score."""
    text = json.dumps([{"item_index": 1, "primary_hub_id": "999-999"}])
    rows, counters = parse_judge_response(text, [1], {"111-000"}, 5)
    assert rows[0]["primary_hub_id"] == "999-999"
    assert counters["invented_primary"] == 1


def test_parse_judge_response_fills_missing_items_as_unanswered() -> None:
    text = json.dumps([{"item_index": 1, "primary_hub_id": "111-000"}])
    rows, counters = parse_judge_response(text, [1, 2, 3], {"111-000"}, 5)
    assert [r["item_index"] for r in rows] == [1, 2, 3]
    assert rows[1]["primary_hub_id"] == ""
    assert counters["missing"] == 2


def test_parse_judge_response_preserves_requested_order() -> None:
    text = json.dumps([
        {"item_index": 3, "primary_hub_id": "112-000"},
        {"item_index": 1, "primary_hub_id": "111-000"},
    ])
    rows, _ = parse_judge_response(text, [1, 3], {"111-000", "112-000"}, 5)
    assert [r["item_index"] for r in rows] == [1, 3]
    assert [r["primary_hub_id"] for r in rows] == ["111-000", "112-000"]


def test_parse_judge_response_truncates_overlong_acceptable_lists() -> None:
    text = json.dumps([{
        "item_index": 1, "primary_hub_id": "111-000",
        "acceptable_hub_ids": [f"11{i}-000" for i in range(8)],
    }])
    rows, counters = parse_judge_response(text, [1], {"111-000"}, 5)
    assert len(rows[0]["acceptable_hub_ids"]) == 5
    assert counters["truncated_acceptable"] == 1


def test_parse_judge_response_on_unparseable_output() -> None:
    rows, counters = parse_judge_response("I refuse.", [1, 2], {"111-000"}, 5)
    assert all(r["primary_hub_id"] == "" for r in rows)
    assert counters["missing"] == 2


def test_parse_judge_response_normalises_bad_confidence() -> None:
    text = json.dumps([{"item_index": 1, "primary_hub_id": "111-000", "confidence": "certain"}])
    rows, _ = parse_judge_response(text, [1], {"111-000"}, 5)
    assert rows[0]["confidence"] == "", "an out-of-vocabulary value must not reach the scorer"


def test_extract_json_array_handles_brackets_inside_strings() -> None:
    text = '[{"item_index": 1, "notes": "see hub [Net] and ] this"}]'
    parsed = extract_json_array(text)
    assert isinstance(parsed, list) and len(parsed) == 1


def test_extract_json_array_returns_none_on_garbage() -> None:
    assert extract_json_array("no array here") is None
    assert extract_json_array("[unclosed") is None


def test_model_slug() -> None:
    assert model_slug("zai-org/GLM-5.2") == "zai_org_glm_5_2"
    assert model_slug("moonshotai/Kimi-K3") == "moonshotai_kimi_k3"


# ── Report assembly ──────────────────────────────────────────────────────

def _report(panel: list[Annotator], missing: list[tuple[str, str]]) -> str:
    paths = parse_hub_reference(HUB_REFERENCE)
    return build_report(
        HUMAN, panel, VALID_GOLD, PRIMARY_GOLD, ITEM_META, paths, {}, {}, missing
    )


def test_report_names_models_that_did_not_run() -> None:
    body = _report(PANEL_LIKE_HUMAN[:2], [("deepseek-ai/DeepSeek-V4-Pro", "no key")])
    assert "deepseek-ai/DeepSeek-V4-Pro" in body
    assert "no key" in body
    assert "weaker evidence" in body, "a short panel must be labelled as weaker evidence"


def test_report_states_probe_not_run_when_absent() -> None:
    body = _report(PANEL_LIKE_HUMAN, [])
    assert "**Not run.**" in body
    assert "upper bounds" in body


def test_report_carries_the_headline_number() -> None:
    body = _report(PANEL_LIKE_HUMAN, [])
    assert "**Headline: 4 of 4 CAPEC items" in body


def test_report_records_pins_for_reproducibility() -> None:
    member = Annotator(
        "zai-org/GLM-5.2",
        "panel",
        dict(PANEL_LIKE_HUMAN[0].answers),
        {
            "model_id": "zai-org/GLM-5.2",
            "route": "openrouter",
            "route_model_id": "z-ai/glm-5.2",
            "served_model_snapshot": ["z-ai/glm-5.2"],
            "temperature": 0.0,
            "reasoning_effort": "low",
            "cost_usd": 1.2345,
        },
    )
    body = _report([member], [])
    for pin in ("z-ai/glm-5.2", "openrouter", "low", "$1.2345"):
        assert pin in body, f"{pin} must appear so the judge is reproducible"
