"""Tests for the blind expert-agreement ceiling study (design doc Part 0.1).

Three properties matter more than the rest, because getting any of them
wrong quietly invalidates the study rather than raising an error:

1. ceiling_items.json must carry zero ground truth, ever.
2. The sample must be exactly reproducible from its recorded seed.
3. Every prefix of the shuffled 250 must span both strata, so a partial
   completion is still an analyzable sample.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tract.ceiling_study import (
    AnchorRecord,
    apportion_with_caps,
    build_answer_key,
    build_answer_template,
    build_ceiling_study,
    sample_ceiling_items,
)


def _make_record(framework_id: str, index: int, hub_id: str) -> AnchorRecord:
    return AnchorRecord(
        framework_id=framework_id,
        framework_name=f"{framework_id}-display",
        control_id=f"{framework_id}-{index}",
        control_title=f"Title {index}",
        control_text=f"Control text body number {index} for {framework_id}.",
        text_source="description",
        anchor_key=f"control text body number {index} for {framework_id}.",
        primary_gold_hub_id=hub_id,
        gold_hub_ids=(hub_id,),
    )



# Matches the real corpus's unique-anchor pool sizes (measured 2026-08-15) so
# the synthetic pool exercises the same cap-binding path as production: the
# test stratum's total capacity (133) exceeds its 125-item quota only because
# owasp_ai_exchange has slack, while mitre_atlas, nist_ai_100_2 and
# owasp_llm_top10 are each capped below their proportional share.
_SYNTHETIC_POOL_SIZES: dict[str, int] = {
    "capec": 339, "cwe": 240, "nist_800_53": 298,
    "mitre_atlas": 43, "owasp_ai_exchange": 62,
    "nist_ai_100_2": 22, "owasp_llm_top10": 6,
}


def _synthetic_pool() -> dict[str, list[AnchorRecord]]:
    """A stand-in pool shaped like the real one's anchor counts, for unit tests."""
    return {
        framework_id: [
            _make_record(framework_id, i, f"hub-{framework_id}-{i}")
            for i in range(size)
        ]
        for framework_id, size in _SYNTHETIC_POOL_SIZES.items()
    }


_SYNTHETIC_WEIGHTS: dict[str, int] = {
    "capec": 1755, "cwe": 596, "nist_800_53": 298,
    "mitre_atlas": 65, "owasp_ai_exchange": 62, "nist_ai_100_2": 45,
    "owasp_llm_top10": 13,
}


class TestApportionWithCaps:
    def test_uncapped_largest_remainder(self) -> None:
        result = apportion_with_caps(
            {"a": 1755, "b": 596, "c": 298}, 125,
            {"a": 339, "b": 240, "c": 298},
        )
        assert result == {"a": 83, "b": 28, "c": 14}
        assert sum(result.values()) == 125

    def test_matches_measured_test_stratum_allocation(self) -> None:
        """Reproduces the real corpus's cap-binding case by hand-checked values."""
        weights = {
            "mitre_atlas": 65, "owasp_ai_exchange": 62,
            "nist_ai_100_2": 45, "owasp_llm_top10": 13,
        }
        caps = {
            "mitre_atlas": 43, "owasp_ai_exchange": 62,
            "nist_ai_100_2": 22, "owasp_llm_top10": 6,
        }
        result = apportion_with_caps(weights, 125, caps)
        assert result == {
            "mitre_atlas": 43, "owasp_ai_exchange": 54,
            "nist_ai_100_2": 22, "owasp_llm_top10": 6,
        }

    def test_never_exceeds_cap(self) -> None:
        result = apportion_with_caps(
            {"a": 100, "b": 1}, 10, {"a": 3, "b": 100},
        )
        assert result["a"] == 3
        assert sum(result.values()) == 10

    def test_sums_to_total_across_many_ratios(self) -> None:
        for total in (0, 1, 7, 50, 125, 250):
            weights = {"a": 11, "b": 7, "c": 3, "d": 1}
            caps = {"a": 200, "b": 200, "c": 200, "d": 200}
            result = apportion_with_caps(weights, total, caps)
            assert sum(result.values()) == total
            for key, cap in caps.items():
                assert 0 <= result[key] <= cap

    def test_infeasible_total_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds combined capacity"):
            apportion_with_caps({"a": 1, "b": 1}, 10, {"a": 2, "b": 2})

    def test_missing_cap_raises(self) -> None:
        with pytest.raises(ValueError, match="no cap"):
            apportion_with_caps({"a": 1, "b": 1}, 1, {"a": 5})

    def test_zero_total_returns_zeros(self) -> None:
        result = apportion_with_caps({"a": 5, "b": 5}, 0, {"a": 10, "b": 10})
        assert result == {"a": 0, "b": 0}


class TestSamplerDeterminism:
    """The pool and weights are pure Python data. Same seed, same output."""

    def test_same_seed_same_items(self) -> None:
        pool = _synthetic_pool()
        items_a, summary_a = sample_ceiling_items(pool, _SYNTHETIC_WEIGHTS, seed=42)
        items_b, summary_b = sample_ceiling_items(pool, _SYNTHETIC_WEIGHTS, seed=42)
        assert items_a == items_b
        assert summary_a == summary_b

    def test_different_seed_different_order(self) -> None:
        pool = _synthetic_pool()
        items_a, _ = sample_ceiling_items(pool, _SYNTHETIC_WEIGHTS, seed=42)
        items_b, _ = sample_ceiling_items(pool, _SYNTHETIC_WEIGHTS, seed=43)
        control_ids_a = [i["control_id"] for i in items_a]
        control_ids_b = [i["control_id"] for i in items_b]
        assert control_ids_a != control_ids_b

    def test_totals_and_no_duplicate_anchors(self) -> None:
        pool = _synthetic_pool()
        items, summary = sample_ceiling_items(pool, _SYNTHETIC_WEIGHTS, seed=42)
        assert len(items) == 250
        assert summary["n_items"] == 250
        assert sum(summary["validation_allocation"].values()) == 125
        assert sum(summary["test_allocation"].values()) == 125
        keys = [(i["framework_id"], i["control_id"]) for i in items]
        assert len(keys) == len(set(keys))

    def test_item_index_is_1_based_contiguous(self) -> None:
        pool = _synthetic_pool()
        items, _ = sample_ceiling_items(pool, _SYNTHETIC_WEIGHTS, seed=42)
        assert [i["item_index"] for i in items] == list(range(1, 251))


class TestPrefixInvariant:
    """Every prefix of length >= 2 in the shuffled order spans both strata.

    This is the property that makes stopping partway through review still
    yield an analyzable sample, which is the entire reason the study is
    shuffled rather than emitted stratum-by-stratum.
    """

    def test_every_prefix_of_length_2_or_more_has_both_strata(self) -> None:
        pool = _synthetic_pool()
        items, _ = sample_ceiling_items(pool, _SYNTHETIC_WEIGHTS, seed=42)
        strata = [i["stratum"] for i in items]
        for k in range(2, len(strata) + 1):
            prefix_strata = set(strata[:k])
            assert prefix_strata == {"validation", "test"}, (
                f"prefix of length {k} has strata {prefix_strata}"
            )

    def test_holds_across_several_seeds(self) -> None:
        pool = _synthetic_pool()
        for seed in (1, 2, 42, 999, 123456):
            items, _ = sample_ceiling_items(pool, _SYNTHETIC_WEIGHTS, seed=seed)
            strata = [i["stratum"] for i in items]
            for k in range(2, len(strata) + 1):
                assert set(strata[:k]) == {"validation", "test"}


class TestAnswerKeyAndTemplate:
    def test_answer_key_matches_items_by_index(self) -> None:
        pool = _synthetic_pool()
        items, _ = sample_ceiling_items(pool, _SYNTHETIC_WEIGHTS, seed=42)
        key = build_answer_key(items, pool)
        assert [e["item_index"] for e in key] == [i["item_index"] for i in items]
        for entry in key:
            assert entry["primary_gold_hub_id"] in entry["valid_gold_hub_ids"]

    def test_template_starts_blank(self) -> None:
        pool = _synthetic_pool()
        items, _ = sample_ceiling_items(pool, _SYNTHETIC_WEIGHTS, seed=42)
        template = build_answer_template(items)
        assert len(template) == 250
        for row in template:
            assert row["primary_hub_id"] == ""
            assert row["acceptable_hub_ids"] == []
            assert row["confidence"] == ""
            assert row["notes"] == ""


class TestNoGroundTruthLeak:
    """The load-bearing safety property: ceiling_items.json is blind."""

    def test_no_gold_hub_id_in_serialized_items(self) -> None:
        pool = _synthetic_pool()
        items, _ = sample_ceiling_items(pool, _SYNTHETIC_WEIGHTS, seed=42)
        key = build_answer_key(items, pool)

        items_blob = json.dumps(items)
        gold_ids: set[str] = set()
        for entry in key:
            gold_ids.add(entry["primary_gold_hub_id"])
            gold_ids.update(entry["valid_gold_hub_ids"])

        leaked = [gold_id for gold_id in gold_ids if gold_id in items_blob]
        assert leaked == [], f"gold hub id(s) leaked into items export: {leaked}"

    def test_items_have_no_hub_shaped_keys(self) -> None:
        pool = _synthetic_pool()
        items, _ = sample_ceiling_items(pool, _SYNTHETIC_WEIGHTS, seed=42)
        forbidden_keys = {
            "gold_hub_id", "gold_hub_ids", "hub_id", "hub_ids",
            "ground_truth", "answer", "predicted_hub_id", "candidates",
        }
        for item in items:
            assert forbidden_keys.isdisjoint(item.keys())


@pytest.mark.skipif(
    not Path("data/training/hub_links_curated.jsonl").exists(),
    reason="requires the committed corpus (data/training, data/processed)",
)
class TestBuildCeilingStudyIntegration:
    """End-to-end against the real corpus. Slow-ish, skipped if data is absent."""

    def test_produces_250_items_matching_measured_allocation(self) -> None:
        items, key, template, hub_reference_md, summary = build_ceiling_study()
        assert len(items) == 250
        assert len(key) == 250
        assert len(template) == 250
        assert summary["validation_allocation"] == {"capec": 83, "cwe": 28, "nist_800_53": 14}
        assert summary["test_allocation"] == {
            "mitre_atlas": 43, "owasp_ai_exchange": 54,
            "nist_ai_100_2": 22, "owasp_llm_top10": 6,
        }
        assert "# CRE Hub Reference" in hub_reference_md
        assert hub_reference_md.count("### ") == 522

    def test_deterministic_end_to_end(self) -> None:
        items_a, key_a, _, _, _ = build_ceiling_study()
        items_b, key_b, _, _, _ = build_ceiling_study()
        assert items_a == items_b
        assert key_a == key_b

    def test_no_leak_against_real_corpus(self) -> None:
        items, key, _, _, _ = build_ceiling_study()
        items_blob = json.dumps(items)
        gold_ids: set[str] = set()
        for entry in key:
            gold_ids.add(entry["primary_gold_hub_id"])
            gold_ids.update(entry["valid_gold_hub_ids"])
        leaked = [gold_id for gold_id in gold_ids if gold_id in items_blob]
        assert leaked == []
