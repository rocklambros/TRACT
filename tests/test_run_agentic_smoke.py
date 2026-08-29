"""Tests for the agentic smoke-test runner.

Everything here stays on the item-assembly and refusal paths, which is the
whole of the runner that can be exercised without allocating a model. Model
loading runs on a pod, so `_score_one_model` is deliberately untested here and
its correctness rests on `evaluate_on_fold`, which has its own tests.
"""

from __future__ import annotations

import json

import pytest

from scripts.phase1b.run_agentic_smoke import (
    AGENTIC_FRAMEWORK_ID,
    AGENTIC_FRAMEWORK_NAME,
    FAIL_AT_OR_BELOW,
    INVESTIGATE_MAX,
    SMOKE_FIXTURE,
    _build_eval_items,
)
from scripts.phase0.common import CURATED_LINKS_PATH
from tract.config import PROCESSED_DIR
from tract.hierarchy import CREHierarchy
from tract.io import load_json


@pytest.fixture(scope="module")
def hierarchy() -> CREHierarchy:
    return CREHierarchy.model_validate(
        load_json(PROCESSED_DIR / "cre_hierarchy.json")
    )


@pytest.fixture(scope="module")
def fixture_data() -> dict:
    return load_json(SMOKE_FIXTURE)


def test_fixture_still_declares_itself_not_a_metric(fixture_data: dict) -> None:
    """The runner's premise. If this flips, the runner must be re-reviewed."""
    assert fixture_data["is_a_metric"] is False


def test_fixture_is_six_items_over_four_hubs(fixture_data: dict) -> None:
    items = fixture_data["items"]
    assert len(items) == 6
    assert len({i["hub_id"] for i in items}) == 4
    # Three of six answering one hub is what makes 0.500 the guess-one-hub
    # baseline, which is why the fixture forbids treating this as a metric.
    counts = fixture_data["hub_distribution"]
    assert max(counts.values()) == 3


def test_agentic_framework_has_no_curated_links() -> None:
    """The held-out property the fixture claims. Verified, not trusted.

    If this framework ever acquires a curated link, these six controls stop
    being held out and the smoke test silently becomes a training-set probe.
    """
    # Read through the constant the training code reads, not a literal path.
    # An earlier version of this test hardcoded data/processed/, where the file
    # does not exist, and a grep against that missing path returned no matches
    # -- which reads exactly like "zero agentic links" and is not evidence of
    # anything. Asserting the corpus is non-empty is what turns the second
    # assertion into a real check.
    n_links = 0
    with CURATED_LINKS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            n_links += 1
            record = json.loads(line)
            assert record.get("framework_id") != AGENTIC_FRAMEWORK_ID, (
                f"{AGENTIC_FRAMEWORK_ID} now carries a curated link. The "
                "agentic smoke test is no longer held out."
            )
    assert n_links > 4000, (
        f"Only {n_links} curated links read from {CURATED_LINKS_PATH}. A short "
        "or empty corpus would make the held-out assertion above vacuous."
    )


def test_build_eval_items_resolves_all_six_to_prose(
    hierarchy: CREHierarchy, fixture_data: dict,
) -> None:
    items = _build_eval_items(
        fixture_data, hierarchy,
        use_prose=True, use_stopwords=True, max_seq_length=512,
    )
    assert len(items) == 6
    assert {i.framework_name for i in items} == {AGENTIC_FRAMEWORK_NAME}
    # A title fallback here would answer a different question than the fixture
    # asks, so assert on the anchor length rather than on "no exception".
    for item in items:
        assert len(item.control_text) > 500, (
            f"{item.section_id} fell back to a short anchor "
            f"({len(item.control_text)} chars); prose was expected."
        )


def test_ground_truth_is_strict_single_hub(
    hierarchy: CREHierarchy, fixture_data: dict,
) -> None:
    """No multi-label credit. Each item has exactly one acceptable answer."""
    items = _build_eval_items(
        fixture_data, hierarchy,
        use_prose=True, use_stopwords=True, max_seq_length=512,
    )
    for item in items:
        assert item.valid_hub_ids == frozenset({item.ground_truth_hub_id})
        assert item.ground_truth_hub_id in hierarchy.hubs


def test_every_answer_hub_has_a_branch_root(
    hierarchy: CREHierarchy, fixture_data: dict,
) -> None:
    """The fail condition compares branch roots, so they must exist."""
    for entry in fixture_data["items"]:
        node = hierarchy.hubs[entry["hub_id"]]
        assert node.branch_root_id
        assert node.branch_root_id in hierarchy.hubs


def test_unknown_hub_in_fixture_is_refused(
    hierarchy: CREHierarchy, fixture_data: dict,
) -> None:
    """A fixture written against a different hub set must not score."""
    broken = json.loads(json.dumps(fixture_data))
    broken["items"][0]["hub_id"] = "999-999"
    with pytest.raises(ValueError, match="not in the hierarchy"):
        _build_eval_items(
            broken, hierarchy,
            use_prose=True, use_stopwords=True, max_seq_length=512,
        )


def test_unknown_control_in_fixture_is_refused(
    hierarchy: CREHierarchy, fixture_data: dict,
) -> None:
    broken = json.loads(json.dumps(fixture_data))
    broken["items"][0]["control_id"] = "ASI99"
    with pytest.raises(ValueError, match="no control in"):
        _build_eval_items(
            broken, hierarchy,
            use_prose=True, use_stopwords=True, max_seq_length=512,
        )


def test_pre_declared_thresholds_match_the_fixture(fixture_data: dict) -> None:
    """The runner restates the fixture's thresholds; they must not drift."""
    condition = fixture_data["pre_declared_pass_condition"]
    assert condition["declared_before_any_campaign_2_arm_ran"] is True
    assert str(FAIL_AT_OR_BELOW) in condition["fail"]
    assert str(INVESTIGATE_MAX) in condition["investigate"]
