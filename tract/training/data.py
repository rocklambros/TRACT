"""Training data generation for contrastive fine-tuning.

Handles:
- TrainingPair construction from filtered hub links
- Hard negative mining from CRE hierarchy (siblings, then cousins)
- HubAwareTemperatureSampler: collision-free batching + AI upweighting
- Conversion to sentence-transformers Dataset format
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar, Iterator

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers.sampler import DefaultBatchSampler

from tract.hierarchy import CREHierarchy
from tract.training.data_quality import QualityTier, TieredLink

logger = logging.getLogger(__name__)

AI_FRAMEWORK_NAMES: frozenset[str] = frozenset({
    "MITRE ATLAS", "NIST AI 100-2", "OWASP AI Exchange",
    "OWASP Top10 for LLM", "OWASP Top10 for ML",
})


@dataclass(frozen=True)
class TrainingPair:
    control_text: str
    hub_id: str
    hub_representation: str
    framework: str
    link_type: str
    quality_tier: str


def mine_hard_negatives(
    hub_id: str,
    hierarchy: CREHierarchy,
    n: int = 3,
) -> list[str]:
    """Return up to n hard negative hub IDs from hierarchy structure.

    Priority: siblings first (same parent), then cousins (parent's siblings' children).
    """
    siblings = [s.hub_id for s in hierarchy.get_siblings(hub_id)]
    if len(siblings) >= n:
        return siblings[:n]

    cousins: list[str] = []
    parent = hierarchy.get_parent(hub_id)
    if parent:
        for uncle in hierarchy.get_siblings(parent.hub_id):
            for child in hierarchy.get_children(uncle.hub_id):
                if child.hub_id != hub_id and child.hub_id not in siblings:
                    cousins.append(child.hub_id)

    all_negatives = siblings + cousins
    seen: set[str] = set()
    deduped: list[str] = []
    for neg_id in all_negatives:
        if neg_id not in seen:
            seen.add(neg_id)
            deduped.append(neg_id)
    return deduped[:n]


TIER_PRIORITY: dict[str, int] = {
    "T1": 0,
    "T1-AI": 1,
    "T3": 2,
}


def build_training_pairs(
    tiered_links: list[TieredLink],
    hub_texts: dict[str, str],
    excluded_framework: str | None = None,
) -> list[TrainingPair]:
    """Build TrainingPair objects from filtered links, deduplicated per text+hub.

    A control text may legitimately map to multiple CRE hubs (the CRE graph
    has multi-hop structure). We keep ALL valid text→hub pairs — MNRL false
    negatives from same-text collisions are prevented by the sampler, not
    by dropping data. Only exact (text, hub) duplicates are collapsed,
    keeping the highest-quality-tier link.

    Args:
        tiered_links: Quality-filtered links with tier metadata.
        hub_texts: Firewalled hub text representations.
        excluded_framework: Framework to exclude (the LOFO held-out framework).
    """
    raw_pairs: list[TrainingPair] = []
    skipped = 0

    for tiered in tiered_links:
        link = tiered.link
        standard_name = link.get("standard_name", "")

        if excluded_framework and standard_name == excluded_framework:
            continue

        control_text = link.get("section_name") or link.get("section_id", "")
        if not control_text or len(control_text) < 3:
            skipped += 1
            continue

        hub_id = link["cre_id"]
        hub_rep = hub_texts.get(hub_id)
        if not hub_rep:
            skipped += 1
            continue

        raw_pairs.append(TrainingPair(
            control_text=control_text,
            hub_id=hub_id,
            hub_representation=hub_rep,
            framework=standard_name,
            link_type=link.get("link_type", ""),
            quality_tier=tiered.tier.value,
        ))

    if skipped:
        logger.info("Skipped %d links (empty text or missing hub)", skipped)

    pair_groups: dict[tuple[str, str], list[TrainingPair]] = defaultdict(list)
    for pair in raw_pairs:
        key = (pair.control_text.lower().strip(), pair.hub_id)
        pair_groups[key].append(pair)

    pairs: list[TrainingPair] = []
    n_deduped = 0
    n_multi_hub_texts = 0

    text_hub_counts: dict[str, set[str]] = defaultdict(set)
    for pair in raw_pairs:
        text_hub_counts[pair.control_text.lower().strip()].add(pair.hub_id)

    for (text_key, hub_id), group in pair_groups.items():
        best = min(group, key=lambda p: TIER_PRIORITY.get(p.quality_tier, 99))
        pairs.append(best)
        n_deduped += len(group) - 1

    for text_key, hubs in text_hub_counts.items():
        if len(hubs) > 1:
            n_multi_hub_texts += 1

    logger.info(
        "Built %d training pairs (excluded=%s): %d raw, %d deduped, "
        "%d texts map to multiple hubs (handled by sampler)",
        len(pairs), excluded_framework, len(raw_pairs), n_deduped,
        n_multi_hub_texts,
    )
    return pairs


class HubAwareTemperatureSampler(DefaultBatchSampler):
    """Batch sampler preventing hub AND anchor-text collisions with AI upsampling.

    Prevents two sources of MNRL false negatives:
    1. Hub collisions — two examples sharing the same target hub in a batch
       means one hub is both positive and in-batch negative.
    2. Anchor-text collisions — the same control text mapping to different
       hubs in a batch means each hub becomes a false negative for the other.

    Also upweights AI-domain examples via temperature-scaled class selection.

    The trainer's data collator tokenizes ALL dataset columns, so hub_id/is_ai/
    anchor_key must be stripped before passing to the trainer. Use set_metadata()
    to inject metadata before trainer construction, then strip those columns.
    """

    _hub_ids_override: ClassVar[list[str] | None] = None
    _is_ai_override: ClassVar[list[bool] | None] = None
    _anchor_keys_override: ClassVar[list[str] | None] = None

    @classmethod
    def set_metadata(
        cls,
        hub_ids: list[str],
        is_ai: list[bool],
        anchor_keys: list[str] | None = None,
    ) -> None:
        cls._hub_ids_override = hub_ids
        cls._is_ai_override = is_ai
        cls._anchor_keys_override = anchor_keys

    @classmethod
    def clear_metadata(cls) -> None:
        cls._hub_ids_override = None
        cls._is_ai_override = None
        cls._anchor_keys_override = None

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 64,
        drop_last: bool = False,
        valid_label_columns: list[str] | None = None,
        generator: torch.Generator | None = None,
        seed: int = 0,
        temperature: float = 2.0,
    ) -> None:
        super().__init__(
            dataset, batch_size=batch_size, drop_last=drop_last,
            valid_label_columns=valid_label_columns,
            generator=generator, seed=seed,
        )
        self.temperature = temperature

        if self._hub_ids_override is not None:
            self.hub_ids = self._hub_ids_override
            self.is_ai = self._is_ai_override or [False] * len(dataset)
            self.anchor_keys = self._anchor_keys_override
        elif "hub_id" in dataset.column_names:
            self.hub_ids = dataset["hub_id"]
            self.is_ai = (
                dataset["is_ai"] if "is_ai" in dataset.column_names
                else [False] * len(dataset)
            )
            self.anchor_keys = (
                dataset["anchor_key"] if "anchor_key" in dataset.column_names
                else None
            )
        else:
            raise ValueError("Dataset must have a 'hub_id' column or use set_metadata()")
        self.n = len(dataset)

    def __iter__(self) -> Iterator[list[int]]:
        if self.generator is not None:
            seed = int(torch.randint(0, 2**31, (1,), generator=self.generator).item())
        else:
            seed = self.seed + self.epoch
        rng = np.random.default_rng(seed)

        ai_indices = [i for i in range(self.n) if self.is_ai[i]]
        trad_indices = [i for i in range(self.n) if not self.is_ai[i]]
        rng.shuffle(ai_indices)
        rng.shuffle(trad_indices)

        n_ai = len(ai_indices)
        n_trad = len(trad_indices)

        if n_ai > 0 and n_trad > 0 and self.temperature > 0:
            w_ai = (n_ai / self.n) ** (1.0 / self.temperature)
            w_trad = (n_trad / self.n) ** (1.0 / self.temperature)
            p_ai = w_ai / (w_ai + w_trad)
        else:
            p_ai = n_ai / self.n if self.n > 0 else 0.0

        ordered: list[int] = []
        ai_ptr, trad_ptr = 0, 0
        for _ in range(self.n):
            if ai_ptr >= n_ai:
                ordered.append(trad_indices[trad_ptr])
                trad_ptr += 1
            elif trad_ptr >= n_trad:
                ordered.append(ai_indices[ai_ptr])
                ai_ptr += 1
            elif rng.random() < p_ai:
                ordered.append(ai_indices[ai_ptr])
                ai_ptr += 1
            else:
                ordered.append(trad_indices[trad_ptr])
                trad_ptr += 1

        batch: list[int] = []
        hubs_in_batch: set[str] = set()
        texts_in_batch: set[str] = set()
        deferred: list[int] = []

        for idx in ordered:
            hub = self.hub_ids[idx]
            text_key = self.anchor_keys[idx] if self.anchor_keys else None
            hub_ok = hub not in hubs_in_batch
            text_ok = text_key is None or text_key not in texts_in_batch
            if hub_ok and text_ok:
                batch.append(idx)
                hubs_in_batch.add(hub)
                if text_key is not None:
                    texts_in_batch.add(text_key)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                    hubs_in_batch = set()
                    texts_in_batch = set()
            else:
                deferred.append(idx)

        remaining = deferred
        while remaining:
            next_remaining: list[int] = []
            for idx in remaining:
                hub = self.hub_ids[idx]
                text_key = self.anchor_keys[idx] if self.anchor_keys else None
                hub_ok = hub not in hubs_in_batch
                text_ok = text_key is None or text_key not in texts_in_batch
                if hub_ok and text_ok:
                    batch.append(idx)
                    hubs_in_batch.add(hub)
                    if text_key is not None:
                        texts_in_batch.add(text_key)
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
                        hubs_in_batch = set()
                        texts_in_batch = set()
                else:
                    next_remaining.append(idx)
            if len(next_remaining) == len(remaining):
                batch.extend(next_remaining)
                break
            remaining = next_remaining

        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        return math.ceil(self.n / self.batch_size)


def pairs_to_dataset(
    pairs: list[TrainingPair],
    hierarchy: CREHierarchy,
    hub_texts: dict[str, str],
    n_hard_negatives: int = 3,
) -> Dataset:
    """Convert TrainingPairs to a sentence-transformers Dataset with hard negatives.

    Output columns: anchor, positive, negative_1..N, hub_id, is_ai, anchor_key
    """
    records: list[dict] = []
    for pair in pairs:
        record: dict = {
            "anchor": pair.control_text,
            "positive": pair.hub_representation,
            "hub_id": pair.hub_id,
            "is_ai": pair.framework in AI_FRAMEWORK_NAMES,
            "anchor_key": pair.control_text.lower().strip(),
        }
        negatives = mine_hard_negatives(pair.hub_id, hierarchy, n=n_hard_negatives)
        for i, neg_id in enumerate(negatives):
            neg_text = hub_texts.get(neg_id, "")
            if neg_text:
                record[f"negative_{i + 1}"] = neg_text

        for i in range(len(negatives), n_hard_negatives):
            record[f"negative_{i + 1}"] = ""

        records.append(record)

    ds = Dataset.from_list(records)
    logger.info(
        "Built dataset: %d examples, %d AI (%.1f%%), columns=%s",
        len(ds),
        sum(1 for r in records if r["is_ai"]),
        100 * sum(1 for r in records if r["is_ai"]) / max(len(records), 1),
        ds.column_names,
    )
    return ds
