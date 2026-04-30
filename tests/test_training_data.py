"""Tests for training data generation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from datasets import Dataset

from tract.training.data import (
    HubAwareTemperatureSampler,
    TrainingPair,
    build_training_pairs,
    mine_hard_negatives,
    pairs_to_dataset,
)
from tract.training.data_quality import QualityTier, TieredLink

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "phase1a_mini_cres.json"


@pytest.fixture
def hierarchy():
    from tract.hierarchy import CREHierarchy

    with open(FIXTURE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return CREHierarchy.from_opencre(
        cres=data["cres"],
        fetch_timestamp=data["fetch_timestamp"],
        data_hash="abc123",
    )


class TestTrainingPair:

    def test_frozen_dataclass(self) -> None:
        pair = TrainingPair(
            control_text="SQL Injection",
            hub_id="760-764",
            hub_representation="Root > AppSec | Injection protection",
            framework="OWASP Top 10 2021",
            link_type="LinkedTo",
            quality_tier="T1",
        )
        assert pair.control_text == "SQL Injection"
        with pytest.raises(AttributeError):
            pair.control_text = "changed"  # type: ignore[misc]


class TestHardNegativeMining:

    def test_returns_sibling_hub_ids(self, hierarchy) -> None:
        leaf_ids = hierarchy.leaf_hub_ids()
        hub_with_siblings = None
        for lid in leaf_ids:
            if hierarchy.get_siblings(lid):
                hub_with_siblings = lid
                break
        if hub_with_siblings is None:
            pytest.skip("No leaf with siblings in test hierarchy")
        negatives = mine_hard_negatives(hub_with_siblings, hierarchy, n=3)
        assert isinstance(negatives, list)
        assert hub_with_siblings not in negatives
        for neg_id in negatives:
            assert neg_id in hierarchy.hubs

    def test_returns_at_most_n(self, hierarchy) -> None:
        hub_id = list(hierarchy.hubs.keys())[0]
        negatives = mine_hard_negatives(hub_id, hierarchy, n=2)
        assert len(negatives) <= 2

    def test_no_duplicates(self, hierarchy) -> None:
        hub_id = list(hierarchy.hubs.keys())[0]
        negatives = mine_hard_negatives(hub_id, hierarchy, n=5)
        assert len(negatives) == len(set(negatives))

    def test_root_with_no_siblings(self, hierarchy) -> None:
        for rid in hierarchy.roots:
            if not hierarchy.get_siblings(rid):
                negatives = mine_hard_negatives(rid, hierarchy, n=3)
                assert isinstance(negatives, list)
                break


class TestBuildTrainingPairs:

    def test_excludes_framework(self, hierarchy) -> None:
        hub_texts = {hid: f"path | {node.name}" for hid, node in hierarchy.hubs.items()}
        hub_id = list(hierarchy.hubs.keys())[0]
        links = [
            TieredLink(
                link={"cre_id": hub_id, "standard_name": "MITRE ATLAS",
                      "section_name": "Adversarial attack technique", "link_type": "LinkedTo"},
                tier=QualityTier.T1_AI,
            ),
            TieredLink(
                link={"cre_id": hub_id, "standard_name": "CWE",
                      "section_name": "CWE-79 Cross-site Scripting", "link_type": "LinkedTo"},
                tier=QualityTier.T1,
            ),
        ]
        pairs = build_training_pairs(links, hub_texts, excluded_framework="MITRE ATLAS")
        assert len(pairs) == 1
        assert pairs[0].framework == "CWE"

    def test_skips_short_text(self, hierarchy) -> None:
        hub_texts = {hid: f"path | {node.name}" for hid, node in hierarchy.hubs.items()}
        hub_id = list(hierarchy.hubs.keys())[0]
        links = [
            TieredLink(
                link={"cre_id": hub_id, "standard_name": "CWE",
                      "section_name": "ab", "link_type": "LinkedTo"},
                tier=QualityTier.T1,
            ),
        ]
        pairs = build_training_pairs(links, hub_texts)
        assert len(pairs) == 0

    def test_skips_missing_hub(self, hierarchy) -> None:
        hub_texts = {}
        links = [
            TieredLink(
                link={"cre_id": "NONEXISTENT", "standard_name": "CWE",
                      "section_name": "Real text here", "link_type": "LinkedTo"},
                tier=QualityTier.T1,
            ),
        ]
        pairs = build_training_pairs(links, hub_texts)
        assert len(pairs) == 0

    def test_keeps_same_text_multiple_hubs(self, hierarchy) -> None:
        """Multi-hub mappings are valid CRE graph structure, not noise."""
        hub_ids = list(hierarchy.hubs.keys())[:5]
        hub_texts = {hid: f"path | {node.name}" for hid, node in hierarchy.hubs.items()}
        links = [
            TieredLink(
                link={"cre_id": hub_ids[i], "standard_name": "CAPEC",
                      "section_name": "Brute Force", "link_type": "LinkedTo"},
                tier=QualityTier.T1,
            )
            for i in range(5)
        ]
        pairs = build_training_pairs(links, hub_texts)
        assert len(pairs) == 5, "All 5 text→hub pairs should be kept"
        assert len({p.hub_id for p in pairs}) == 5

    def test_dedup_same_text_same_hub_keeps_best_tier(self, hierarchy) -> None:
        hub_ids = list(hierarchy.hubs.keys())[:1]
        hub_texts = {hid: f"path | {node.name}" for hid, node in hierarchy.hubs.items()}
        links = [
            TieredLink(
                link={"cre_id": hub_ids[0], "standard_name": "CWE",
                      "section_name": "SQL Injection", "link_type": "LinkedTo"},
                tier=QualityTier.T3,
            ),
            TieredLink(
                link={"cre_id": hub_ids[0], "standard_name": "CAPEC",
                      "section_name": "SQL Injection", "link_type": "LinkedTo"},
                tier=QualityTier.T1,
            ),
        ]
        pairs = build_training_pairs(links, hub_texts)
        assert len(pairs) == 1, "Same text + same hub = deduplicated"
        assert pairs[0].quality_tier == "T1"

    def test_dedup_case_insensitive_same_hub(self, hierarchy) -> None:
        hub_ids = list(hierarchy.hubs.keys())[:1]
        hub_texts = {hid: f"path | {node.name}" for hid, node in hierarchy.hubs.items()}
        links = [
            TieredLink(
                link={"cre_id": hub_ids[0], "standard_name": "ATLAS",
                      "section_name": "Validate AI Model", "link_type": "LinkedTo"},
                tier=QualityTier.T1_AI,
            ),
            TieredLink(
                link={"cre_id": hub_ids[0], "standard_name": "ATLAS",
                      "section_name": "Validate AI model", "link_type": "LinkedTo"},
                tier=QualityTier.T1_AI,
            ),
        ]
        pairs = build_training_pairs(links, hub_texts)
        assert len(pairs) == 1, "Case-insensitive dedup on same hub"

    def test_keeps_same_text_different_hubs_case_insensitive(self, hierarchy) -> None:
        """Same text, different hubs = different CRE neighborhoods, keep both."""
        hub_ids = list(hierarchy.hubs.keys())[:2]
        hub_texts = {hid: f"path | {node.name}" for hid, node in hierarchy.hubs.items()}
        links = [
            TieredLink(
                link={"cre_id": hub_ids[0], "standard_name": "ATLAS",
                      "section_name": "Validate AI Model", "link_type": "LinkedTo"},
                tier=QualityTier.T1_AI,
            ),
            TieredLink(
                link={"cre_id": hub_ids[1], "standard_name": "ATLAS",
                      "section_name": "Validate AI model", "link_type": "LinkedTo"},
                tier=QualityTier.T1_AI,
            ),
        ]
        pairs = build_training_pairs(links, hub_texts)
        assert len(pairs) == 2, "Same text, different hubs = keep both"


def _make_sampler_dataset(
    n: int,
    hub_ids: list[str] | None = None,
    is_ai: list[bool] | None = None,
) -> Dataset:
    """Helper to build a minimal Dataset for sampler tests."""
    if hub_ids is None:
        hub_ids = [f"h{i % 20}" for i in range(n)]
    if is_ai is None:
        is_ai = [False] * n
    return Dataset.from_dict({
        "anchor": [f"text_{i}" for i in range(n)],
        "positive": [f"hub_{hub_ids[i]}" for i in range(n)],
        "hub_id": hub_ids,
        "is_ai": is_ai,
    })


class TestHubAwareTemperatureSampler:

    def test_no_hub_collisions_in_full_batches(self) -> None:
        hub_ids = [f"h{i % 20}" for i in range(40)]
        ds = _make_sampler_dataset(40, hub_ids=hub_ids)
        sampler = HubAwareTemperatureSampler(dataset=ds, batch_size=8, drop_last=False)
        for batch_indices in sampler:
            if len(batch_indices) == 8:
                batch_hubs = [hub_ids[i] for i in batch_indices]
                assert len(batch_hubs) == len(set(batch_hubs)), \
                    f"Hub collision in full batch: {batch_hubs}"

    def test_all_indices_appear_exactly_once(self) -> None:
        n = 50
        ds = _make_sampler_dataset(n)
        sampler = HubAwareTemperatureSampler(dataset=ds, batch_size=10, drop_last=False)
        all_indices: list[int] = []
        for batch in sampler:
            all_indices.extend(batch)
        assert sorted(all_indices) == list(range(n))

    def test_ai_upweighting_with_temperature(self) -> None:
        n = 200
        hub_ids = [f"h{i}" for i in range(n)]
        is_ai = [i < 10 for i in range(n)]  # 5% AI
        ds = _make_sampler_dataset(n, hub_ids=hub_ids, is_ai=is_ai)
        sampler = HubAwareTemperatureSampler(
            dataset=ds, batch_size=20, drop_last=False,
            temperature=2.0, seed=42,
        )
        batches = list(sampler)
        first_half_indices: list[int] = []
        for b in batches[:len(batches) // 2]:
            first_half_indices.extend(b)
        ai_in_first_half = sum(1 for i in first_half_indices if is_ai[i])
        ai_fraction = ai_in_first_half / len(first_half_indices)
        assert ai_fraction > 0.05, f"AI fraction should be > 5%, got {ai_fraction:.3f}"

    def test_deterministic_with_same_seed(self) -> None:
        hub_ids = ["h1", "h2", "h3", "h4", "h5"] * 3
        ds = _make_sampler_dataset(15, hub_ids=hub_ids)
        batches1 = list(HubAwareTemperatureSampler(dataset=ds, batch_size=4, seed=42))
        batches2 = list(HubAwareTemperatureSampler(dataset=ds, batch_size=4, seed=42))
        assert batches1 == batches2

    def test_len(self) -> None:
        ds = _make_sampler_dataset(5)
        sampler = HubAwareTemperatureSampler(dataset=ds, batch_size=2)
        assert len(sampler) == 3

    def test_drop_last(self) -> None:
        ds = _make_sampler_dataset(5)
        sampler = HubAwareTemperatureSampler(dataset=ds, batch_size=2, drop_last=True)
        batches = list(sampler)
        for batch in batches:
            assert len(batch) == 2

    def test_rejects_missing_hub_id_column(self) -> None:
        ds = Dataset.from_dict({
            "anchor": ["a", "b"],
            "positive": ["p1", "p2"],
        })
        with pytest.raises(ValueError, match="hub_id"):
            HubAwareTemperatureSampler(dataset=ds, batch_size=2)

    def test_epoch_changes_order(self) -> None:
        ds = _make_sampler_dataset(20)
        sampler = HubAwareTemperatureSampler(dataset=ds, batch_size=4, seed=42)
        batches_e0 = list(sampler)
        sampler.set_epoch(1)
        batches_e1 = list(sampler)
        assert batches_e0 != batches_e1

    def test_set_metadata_works_without_hub_id_column(self) -> None:
        """Simulates trainer path: metadata set via class method, dataset has no hub_id."""
        hub_ids = [f"h{i}" for i in range(10)]
        is_ai = [i < 3 for i in range(10)]
        ds_stripped = Dataset.from_dict({
            "anchor": [f"text_{i}" for i in range(10)],
            "positive": [f"pos_{i}" for i in range(10)],
        })
        try:
            HubAwareTemperatureSampler.set_metadata(hub_ids=hub_ids, is_ai=is_ai)
            sampler = HubAwareTemperatureSampler(dataset=ds_stripped, batch_size=4)
            all_indices: list[int] = []
            for batch in sampler:
                all_indices.extend(batch)
            assert sorted(all_indices) == list(range(10))
        finally:
            HubAwareTemperatureSampler.clear_metadata()

    def test_no_anchor_text_collisions_in_full_batches(self) -> None:
        """Same anchor text mapped to different hubs must not share a batch."""
        n = 30
        hub_ids = [f"h{i}" for i in range(n)]
        anchor_keys = [f"text_{i}" for i in range(n)]
        anchor_keys[10] = anchor_keys[0]
        anchor_keys[20] = anchor_keys[0]
        ds = Dataset.from_dict({
            "anchor": [f"anchor_{i}" for i in range(n)],
            "positive": [f"pos_{i}" for i in range(n)],
            "hub_id": hub_ids,
            "anchor_key": anchor_keys,
        })
        sampler = HubAwareTemperatureSampler(dataset=ds, batch_size=8, drop_last=False)
        for batch_indices in sampler:
            if len(batch_indices) == 8:
                batch_keys = [anchor_keys[i] for i in batch_indices]
                assert len(batch_keys) == len(set(batch_keys)), \
                    f"Anchor text collision in batch: {batch_keys}"

    def test_anchor_text_all_indices_still_appear(self) -> None:
        """Text collision avoidance must not lose any examples."""
        n = 20
        hub_ids = [f"h{i}" for i in range(n)]
        anchor_keys = [f"text_{i % 5}" for i in range(n)]
        ds = Dataset.from_dict({
            "anchor": [f"anchor_{i}" for i in range(n)],
            "positive": [f"pos_{i}" for i in range(n)],
            "hub_id": hub_ids,
            "anchor_key": anchor_keys,
        })
        sampler = HubAwareTemperatureSampler(dataset=ds, batch_size=4, drop_last=False)
        all_indices: list[int] = []
        for batch in sampler:
            all_indices.extend(batch)
        assert sorted(all_indices) == list(range(n))


class TestPairsToDataset:

    def test_produces_correct_columns(self, hierarchy) -> None:
        hub_texts = {hid: f"{node.hierarchy_path} | {node.name}"
                     for hid, node in hierarchy.hubs.items()}
        leaf_ids = hierarchy.leaf_hub_ids()
        pairs = [
            TrainingPair(
                control_text=f"Control text {i}",
                hub_id=leaf_ids[i % len(leaf_ids)],
                hub_representation=hub_texts[leaf_ids[i % len(leaf_ids)]],
                framework="CWE",
                link_type="LinkedTo",
                quality_tier="T1",
            )
            for i in range(4)
        ]
        ds = pairs_to_dataset(pairs, hierarchy, hub_texts, n_hard_negatives=3)
        assert "anchor" in ds.column_names
        assert "positive" in ds.column_names
        assert "negative_1" in ds.column_names
        assert "negative_2" in ds.column_names
        assert "negative_3" in ds.column_names
        assert "hub_id" in ds.column_names
        assert "is_ai" in ds.column_names
        assert "anchor_key" in ds.column_names
        assert len(ds) == 4
