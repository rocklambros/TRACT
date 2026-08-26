"""Tests for training data generation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# tract.training.data imports torch at module scope, and torch, datasets and
# sentence-transformers all live in the optional `phase0` extra rather than
# requirements.txt. The default CI test job installs the base set only, so
# skip this module there instead of failing collection. Run it with
# `pip install -e '.[phase0]'`. Every other test module keeps its top-level
# imports free of the ML stack for the same reason.
pytest.importorskip("torch", reason="needs the phase0 extra")
pytest.importorskip("datasets", reason="needs the phase0 extra")

import torch
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


def test_tier_priority_includes_al() -> None:
    from tract.training.data import TIER_PRIORITY
    assert "AL" in TIER_PRIORITY
    assert TIER_PRIORITY["AL"] == 3


class TestSamplerAttributeContract:
    """__iter__ seeds its RNG from self.generator and self.seed.

    The trainer always supplies a generator, so the generator branch is the
    production path and the bare-seed branch is the fallback taken only by
    direct construction. Both attributes are asserted to exist AND to change
    the emitted order, because an attribute that is present but ignored is the
    same silent no-op as one that is missing.
    """

    def test_generator_and_seed_survive_construction(self) -> None:
        generator = torch.Generator().manual_seed(11)
        dataset = _make_sampler_dataset(20)
        sampler = HubAwareTemperatureSampler(
            dataset=dataset, batch_size=4, generator=generator, seed=7,
        )
        assert sampler.generator is generator
        assert sampler.seed == 7
        assert sampler.epoch == 0

    def test_seed_selects_the_order_when_no_generator_is_given(self) -> None:
        dataset = _make_sampler_dataset(60)
        first = list(HubAwareTemperatureSampler(dataset=dataset, batch_size=6, seed=1))
        second = list(HubAwareTemperatureSampler(dataset=dataset, batch_size=6, seed=2))
        assert first != second

    def test_generator_overrides_the_seed_when_given(self) -> None:
        dataset = _make_sampler_dataset(60)
        seeded = list(HubAwareTemperatureSampler(dataset=dataset, batch_size=6, seed=5))
        generated = [
            list(HubAwareTemperatureSampler(
                dataset=dataset, batch_size=6, seed=5,
                generator=torch.Generator().manual_seed(101),
            ))
            for _ in range(2)
        ]
        # Same generator seed reproduces, so the generator branch is
        # deterministic rather than merely different.
        assert generated[0] == generated[1]
        # And it is the generator, not self.seed, that chose the order: both
        # samplers carry seed=5 and disagree.
        assert generated[0] != seeded


def _dispatch_batch_sampler(
    args: object, dataset: Dataset, batch_size: int = 8,
) -> object:
    """Run the library's real get_batch_sampler against a stub trainer.

    get_batch_sampler reads only ``self.args``, so this drives the genuine
    dispatch without constructing a trainer and therefore without loading a
    model. Reimplementing the dispatch here instead would test this file
    rather than sentence-transformers.
    """
    from sentence_transformers import SentenceTransformerTrainer

    class _StubTrainer:
        args: object

    stub = _StubTrainer()
    stub.args = args
    return SentenceTransformerTrainer.get_batch_sampler(
        stub,
        dataset,
        batch_size=batch_size,
        drop_last=False,
        valid_label_columns=None,
        generator=torch.Generator().manual_seed(42),
        seed=42,
    )


class TestTrainerReachesTheCustomSampler:
    """The sampler is wired in by handing the CLASS to the training arguments.

    sentence-transformers instantiates it inside get_batch_sampler
    (``inspect.isclass(...) and issubclass(..., DefaultBatchSampler)``). If
    either end of that contract breaks, training falls back to the library
    default and hub-aware temperature sampling stops running while every run
    record still reports sampling_temperature. Nothing raises. So these tests
    observe HubAwareTemperatureSampler.__iter__ execute rather than checking
    that a keyword argument was passed.
    """

    @pytest.fixture
    def iter_calls(self, monkeypatch: pytest.MonkeyPatch) -> list[object]:
        """Record every entry into the sampler's own __iter__, then delegate."""
        calls: list[object] = []
        original = HubAwareTemperatureSampler.__iter__

        def recording_iter(sampler: HubAwareTemperatureSampler):  # type: ignore[no-untyped-def]
            calls.append(sampler)
            yield from original(sampler)

        monkeypatch.setattr(HubAwareTemperatureSampler, "__iter__", recording_iter)
        return calls

    # 45 is not a multiple of the batch size of 8 on purpose, so the
    # completeness assertion below covers the trailing partial batch. At 40 it
    # did not, and a mutation deleting the final `yield batch` survived it.
    N_EXAMPLES = 45

    @pytest.fixture
    def metadata_hub_ids(self) -> list[str]:
        return [f"CRE-{i % 20}" for i in range(self.N_EXAMPLES)]

    def test_library_dispatch_instantiates_and_iterates_the_class(
        self, tmp_path: Path, iter_calls: list[object], metadata_hub_ids: list[str],
    ) -> None:
        from sentence_transformers import SentenceTransformerTrainingArguments

        dataset = _make_sampler_dataset(
            self.N_EXAMPLES, hub_ids=metadata_hub_ids,
        ).remove_columns(["hub_id", "is_ai"])
        HubAwareTemperatureSampler.set_metadata(
            hub_ids=metadata_hub_ids, is_ai=[False] * self.N_EXAMPLES,
        )
        try:
            args = SentenceTransformerTrainingArguments(
                output_dir=str(tmp_path), report_to="none",
                batch_sampler=HubAwareTemperatureSampler,
            )
            assert args.batch_sampler is HubAwareTemperatureSampler, (
                "__post_init__ coerced the sampler class away, so the trainer "
                "will never see it"
            )

            sampler = _dispatch_batch_sampler(args, dataset)
            assert isinstance(sampler, HubAwareTemperatureSampler), (
                f"get_batch_sampler returned {type(sampler).__name__}, so "
                "hub-aware temperature sampling is not running"
            )

            batches = list(sampler)
            assert iter_calls, (
                "the sampler was constructed but its __iter__ never ran, so "
                "the batches came from somewhere else"
            )
            assert sorted(i for batch in batches for i in batch) == list(
                range(self.N_EXAMPLES)
            )
        finally:
            HubAwareTemperatureSampler.clear_metadata()

    def test_enum_batch_sampler_does_not_reach_the_custom_sampler(
        self, tmp_path: Path, iter_calls: list[object],
    ) -> None:
        """Negative control: the default arm must not run the custom sampler.

        Without this, a test asserting the custom sampler ran would pass even
        if every configuration ran it.
        """
        from sentence_transformers import SentenceTransformerTrainingArguments

        dataset = _make_sampler_dataset(
            self.N_EXAMPLES,
        ).remove_columns(["hub_id", "is_ai"])
        args = SentenceTransformerTrainingArguments(
            output_dir=str(tmp_path), report_to="none",
            # The string form of BatchSamplers.BATCH_SAMPLER. Spelled as a
            # string because the enum's import path moved in 5.7.
            batch_sampler="batch_sampler",
        )
        sampler = _dispatch_batch_sampler(args, dataset)
        assert not isinstance(sampler, HubAwareTemperatureSampler)
        list(sampler)
        assert iter_calls == []

    def test_train_model_wires_the_class_into_the_training_arguments(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        iter_calls: list[object], metadata_hub_ids: list[str],
    ) -> None:
        """Guard the wiring in loop.py, not only the library's dispatch.

        Runs the real train_model with the model, loss and trainer stubbed, so
        the training arguments are built by production code. The stub trainer
        then asks the library which sampler that configuration selects, which
        is what the real trainer does when it builds its dataloader.
        """
        # tract.training.loop imports peft at module scope, which the other
        # tests in this file do not need. Guard it here rather than at module
        # scope so a missing peft costs one skip, not the whole file.
        pytest.importorskip("peft", reason="needs the phase0 extra")
        from tract.training import loop as loop_module
        from tract.training.config import TrainingConfig

        captured: dict[str, object] = {}

        class _FakeTrainer:
            def __init__(
                self, model: object, args: object, train_dataset: Dataset,
                eval_dataset: Dataset | None, loss: object,
            ) -> None:
                captured["args"] = args
                captured["dataset"] = train_dataset

            def train(self) -> None:
                captured["sampler"] = _dispatch_batch_sampler(
                    captured["args"], captured["dataset"],  # type: ignore[arg-type]
                )
                list(captured["sampler"])  # type: ignore[call-overload]

        class _StubModel:
            def named_parameters(self) -> object:
                return iter(())

        monkeypatch.setattr(
            loop_module, "load_model_with_lora", lambda config: _StubModel(),
        )
        monkeypatch.setattr(
            loop_module, "MultipleNegativesRankingLoss", lambda model: object(),
        )
        monkeypatch.setattr(loop_module, "SentenceTransformerTrainer", _FakeTrainer)

        dataset = _make_sampler_dataset(self.N_EXAMPLES, hub_ids=metadata_hub_ids)
        # lora_rank=0 is the full fine-tune arm, which has no adapter for
        # _assert_adapter_learned to inspect on a stub model.
        config = TrainingConfig(name="sampler-wiring-probe", lora_rank=0, max_epochs=1)

        loop_module.train_model(config, dataset, tmp_path)

        assert captured["args"].batch_sampler is HubAwareTemperatureSampler, (  # type: ignore[attr-defined]
            "train_model did not put the sampler class in the training "
            "arguments, so the trainer will use the library default"
        )
        assert isinstance(captured["sampler"], HubAwareTemperatureSampler)
        assert iter_calls, "the wired sampler never iterated"
