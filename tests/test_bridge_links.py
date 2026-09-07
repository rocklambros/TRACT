"""The Tier-2 bridge link corpus, and the tag surviving the merge.

Phase 2C produces human-curated traditional-control -> AI-hub links. They are
Tier 2 (independently human-authored, no model output shown to the annotator)
and they live in their own file, which design decision D3 justifies on the
grounds that the tier boundary is then a file boundary.

That rationale does not survive contact with the training pipeline unless the
tag is carried explicitly, which is premortem finding C2. `assign_quality_tier`
takes a link dict and decides:

    standard_name in AI_FRAMEWORK_NAMES        -> T1-AI
    link_type == "AutomaticallyLinkedTo"       -> T3
    otherwise                                  -> T1

A bridge link is a TRADITIONAL framework pointing at an AI hub. "NIST 800-53 v5"
is not in AI_FRAMEWORK_NAMES and the link carries no link_type, so it falls
through to **T1** -- indistinguishable from an OpenCRE-curated gold link, one
function call after the file boundary that was supposed to keep them apart.

So these tests cover two things: that bridge links never reach the evaluation
corpus, and that they never arrive at the training pipeline wearing T1.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tract.bridge.links import (
    BRIDGE_LINK_TYPE,
    BridgeLink,
    bridge_training_records,
    load_bridge_links,
    merge_for_training,
)
from tract.training.data_quality import QualityTier, assign_quality_tier


def _link(**overrides: object) -> BridgeLink:
    base = {
        "framework_id": "nist_800_53",
        "standard_name": "NIST 800-53 v5",
        "section_id": "AC-3",
        "section_name": "Access Enforcement",
        "cre_id": "342-641",
        "tier": 2,
        "annotator_id": "a1",
        "created_at": "2026-09-01T00:00:00Z",
        "confidence": 3,
        "rationale": "enforcement of approved authorizations",
    }
    base.update(overrides)
    return BridgeLink(**base)  # type: ignore[arg-type]


class TestBridgeLinksNeverReachTheEvaluationCorpus:
    """The property Phase 2C rests on, asserted on the real 147-item corpus."""

    def test_the_corpus_is_unchanged_when_bridges_are_merged(self) -> None:
        import logging

        from scripts.phase0.common import (
            AI_FRAMEWORK_NAMES,
            build_evaluation_corpus,
            load_curated_links,
        )

        logging.disable(logging.INFO)
        try:
            curated = load_curated_links()
            before = build_evaluation_corpus(curated, AI_FRAMEWORK_NAMES, {})
            after = build_evaluation_corpus(
                merge_for_training(curated, [_link()]), AI_FRAMEWORK_NAMES, {}
            )
        finally:
            logging.disable(logging.NOTSET)

        assert len(after) == len(before) == 147
        assert [i.control_text for i in after] == [i.control_text for i in before]
        assert [i.ground_truth_hub_id for i in after] == [
            i.ground_truth_hub_id for i in before
        ]

    def test_the_merge_actually_added_something(self) -> None:
        """Guards the guard: an inert merge would pass the test above."""
        curated = [
            _hub_link("111-111", "OWASP ASVS", "V1.1", "Architecture"),
        ]
        merged = merge_for_training(curated, [_link()])
        assert len(merged) == len(curated) + 1


def _hub_link(cre_id: str, standard: str, section: str, name: str):  # type: ignore[no-untyped-def]
    from scripts.phase0.common import HubStandardLink

    return HubStandardLink(
        cre_id=cre_id,
        cre_name="",
        standard_name=standard,
        section_id=section,
        section_name=name,
    )


class TestTheTierTagSurvivesIntoTraining:
    """Finding C2. Without this the file boundary buys nothing."""

    def test_a_bridge_record_is_tiered_t2_not_t1(self) -> None:
        record = bridge_training_records([_link()])[0]
        tier = assign_quality_tier(record, "Access enforcement for approved users")
        assert tier is QualityTier.T2

    def test_the_same_record_without_the_marker_would_be_t1(self) -> None:
        """Shows the defect is real rather than asserting the fix twice.

        This is the shape the plan's own C1 code produced: a plain link dict
        for a traditional framework, which falls through every branch to T1.
        """
        record = dict(bridge_training_records([_link()])[0])
        record.pop(BRIDGE_LINK_TYPE_FIELD := "link_type")
        assert BRIDGE_LINK_TYPE_FIELD not in record
        tier = assign_quality_tier(record, "Access enforcement for approved users")
        assert tier is QualityTier.T1

    def test_the_marker_is_not_the_automatic_link_marker(self) -> None:
        """T3 means AutomaticallyLinkedTo. A bridge is human-authored."""
        assert BRIDGE_LINK_TYPE != "AutomaticallyLinkedTo"

    def test_a_bridge_record_with_no_anchor_is_still_dropped(self) -> None:
        """The tier tag must not exempt a bridge from the anchor floor."""
        record = bridge_training_records([_link()])[0]
        assert assign_quality_tier(record, None) is QualityTier.DROPPED
        assert assign_quality_tier(record, "tiny") is QualityTier.DROPPED

    def test_an_ai_framework_bridge_is_still_tiered_by_provenance(self) -> None:
        """Bridges are traditional->AI. A mislabelled one must not read T1-AI.

        Provenance beats framework identity: a human-authored bridge is Tier 2
        whatever standard it names, and silently promoting it to T1-AI would
        put it in an AI gate denominator.
        """
        record = bridge_training_records(
            [_link(standard_name="MITRE ATLAS", framework_id="mitre_atlas")]
        )[0]
        assert assign_quality_tier(record, "a sufficiently long anchor") is (
            QualityTier.T2
        )


class TestLoaderValidatesItsInput:
    def test_round_trips_a_well_formed_file(self, tmp_path: Path) -> None:
        path = tmp_path / "bridge.jsonl"
        link = _link()
        path.write_text(
            json.dumps(link.__dict__, sort_keys=True) + "\n", encoding="utf-8"
        )
        assert load_bridge_links(path) == [link]

    def test_rejects_a_link_that_is_not_tier_two(self, tmp_path: Path) -> None:
        path = tmp_path / "bridge.jsonl"
        payload = dict(_link().__dict__, tier=1)
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="tier"):
            load_bridge_links(path)

    def test_rejects_a_missing_field_naming_it(self, tmp_path: Path) -> None:
        path = tmp_path / "bridge.jsonl"
        payload = dict(_link().__dict__)
        del payload["annotator_id"]
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="annotator_id"):
            load_bridge_links(path)

    def test_rejects_an_unknown_field_rather_than_ignoring_it(
        self, tmp_path: Path
    ) -> None:
        """A typo'd field name that is silently dropped loses annotator data."""
        path = tmp_path / "bridge.jsonl"
        payload = dict(_link().__dict__, anotator_id="a1")
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="anotator_id"):
            load_bridge_links(path)

    def test_reports_the_line_number_of_a_bad_record(self, tmp_path: Path) -> None:
        path = tmp_path / "bridge.jsonl"
        good = json.dumps(_link().__dict__)
        bad = json.dumps(dict(_link().__dict__, tier=3))
        path.write_text(f"{good}\n{good}\n{bad}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="line 3"):
            load_bridge_links(path)

    def test_an_absent_file_raises_rather_than_returning_empty(
        self, tmp_path: Path
    ) -> None:
        """Returning [] would silently train with no bridge supervision."""
        with pytest.raises(FileNotFoundError):
            load_bridge_links(tmp_path / "nope.jsonl")
