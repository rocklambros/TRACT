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
from typing import Iterator

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


def build_training_pairs(
    tiered_links: list[TieredLink],
    hub_texts: dict[str, str],
    excluded_framework: str | None = None,
) -> list[TrainingPair]:
    """Build TrainingPair objects from filtered links.

    Args:
        tiered_links: Quality-filtered links with tier metadata.
        hub_texts: Firewalled hub text representations.
        excluded_framework: Framework to exclude (the LOFO held-out framework).
    """
    pairs: list[TrainingPair] = []
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

        pairs.append(TrainingPair(
            control_text=control_text,
            hub_id=hub_id,
            hub_representation=hub_rep,
            framework=standard_name,
            link_type=link.get("link_type", ""),
            quality_tier=tiered.tier.value,
        ))

    if skipped:
        logger.info("Skipped %d links (empty text or missing hub)", skipped)
    logger.info("Built %d training pairs (excluded=%s)", len(pairs), excluded_framework)
    return pairs


class HubAwareTemperatureSampler(DefaultBatchSampler):
    """Batch sampler preventing hub collisions with temperature-weighted AI upsampling.

    Combines two functions:
    1. No two examples in a batch share the same target hub (eliminates
       false negatives in MNRL).
    2. AI-domain examples are upweighted via temperature-scaled class
       selection (from ~3.2% natural to ~15.5% at T=2).

    Compatible with SentenceTransformerTrainer — accepts the trainer's
    constructor signature and is passed as batch_sampler= in training args.
    """

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

        if "hub_id" not in dataset.column_names:
            raise ValueError("Dataset must have a 'hub_id' column for hub-aware batching")
        self.hub_ids: list[str] = dataset["hub_id"]
        self.is_ai: list[bool] = (
            dataset["is_ai"] if "is_ai" in dataset.column_names
            else [False] * len(dataset)
        )
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
        deferred: list[int] = []

        for idx in ordered:
            hub = self.hub_ids[idx]
            if hub not in hubs_in_batch:
                batch.append(idx)
                hubs_in_batch.add(hub)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                    hubs_in_batch = set()
            else:
                deferred.append(idx)

        remaining = deferred
        while remaining:
            next_remaining: list[int] = []
            for idx in remaining:
                hub = self.hub_ids[idx]
                if hub not in hubs_in_batch:
                    batch.append(idx)
                    hubs_in_batch.add(hub)
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
                        hubs_in_batch = set()
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

    Output columns: anchor, positive, negative_1..N, hub_id, is_ai
    """
    records: list[dict] = []
    for pair in pairs:
        record: dict = {
            "anchor": pair.control_text,
            "positive": pair.hub_representation,
            "hub_id": pair.hub_id,
            "is_ai": pair.framework in AI_FRAMEWORK_NAMES,
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
