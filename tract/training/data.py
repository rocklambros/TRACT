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
from typing import Any, ClassVar, Iterator

import numpy as np
import torch
from datasets import Dataset

from tract.hierarchy import CREHierarchy
from tract.text_selection import ProseIndex, SelectionStats, select_control_text
from tract.training.data_quality import TieredLink
from tract.training.st_compat import resolve_symbol

logger = logging.getLogger(__name__)

# sentence-transformers moved this class from `.sampler` to `.base.sampler` in
# 5.4, and the training pin (5.7.0) and the serving pin (3.2.0) sit on opposite
# sides of that move. Resolving through the shim keeps one import working under
# both. See tract/training/st_compat.py for the verified path matrix.
DefaultBatchSampler = resolve_symbol("DefaultBatchSampler")

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
    "AL": 3,
}


def build_training_pairs(
    tiered_links: list[TieredLink],
    hub_texts: dict[str, str],
    excluded_framework: str | None = None,
    prose_index: ProseIndex | None = None,
    stopwords: frozenset[str] | None = None,
    description_only: bool = False,
    max_chars: int | None = None,
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
        prose_index: When given, each anchor is the control's full text rather
            than its section title. The pipeline previously took section_name
            unconditionally, which trains on three-word titles while production
            is handed paragraphs.
        stopwords: When given, low-information words are filtered from anchors.
            The same set must be applied to hub_texts by the caller.
    """
    raw_pairs: list[TrainingPair] = []
    skipped = 0
    selection_stats = SelectionStats()

    for tiered in tiered_links:
        link = tiered.link
        standard_name = link.get("standard_name", "")

        if excluded_framework and standard_name == excluded_framework:
            continue

        try:
            control_text = select_control_text(
                prose_index,
                standard_name,
                link.get("section_id"),
                link.get("section_name"),
                stats=selection_stats,
                stopwords=stopwords,
                description_only=description_only,
            ).text
        except ValueError:
            skipped += 1
            continue
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
    selection_stats.log_summary("Training anchors")

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


# The base class arrives from st_compat.resolve_symbol rather than a literal
# import, because sentence-transformers 5.4 moved it and the training and
# serving pins straddle that move. mypy therefore sees a module-level variable
# of type Any and raises both `valid-type` and `misc` in every environment,
# whether or not the ML stack is installed. That determinism is the point: the
# previous literal import resolved to a real class on a developer machine and to
# Any in the lint job, so one comment had to carry `unused-ignore` to stay
# correct under both. Narrowed to those two codes on purpose -- a bare
# `type: ignore` would also swallow a genuine signature conflict with the base.
class HubAwareTemperatureSampler(DefaultBatchSampler):  # type: ignore[valid-type, misc]
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
    _strata_override: ClassVar[list[str] | None] = None
    # sentence-transformers takes the sampler CLASS and constructs it itself,
    # so anything from TrainingConfig has to arrive through a class attribute.
    # config.sampling_temperature was recorded in every run record and never
    # reached the sampler, which always used the 2.0 default -- the same
    # dead-field shape as control_text_source.
    _temperature_override: ClassVar[float | None] = None
    _strata_temperature_override: ClassVar[float | None] = None

    @classmethod
    def set_metadata(
        cls,
        hub_ids: list[str],
        is_ai: list[bool],
        anchor_keys: list[str] | None = None,
        strata: list[str] | None = None,
        temperature: float | None = None,
        strata_temperature: float | None = None,
    ) -> None:
        cls._temperature_override = temperature
        cls._strata_temperature_override = strata_temperature
        cls._hub_ids_override = hub_ids
        cls._is_ai_override = is_ai
        cls._anchor_keys_override = anchor_keys
        cls._strata_override = strata

    @classmethod
    def clear_metadata(cls) -> None:
        cls._hub_ids_override = None
        cls._is_ai_override = None
        cls._strata_override = None
        cls._temperature_override = None
        cls._strata_temperature_override = None
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
        strata_temperature: float = 0.0,
    ) -> None:
        super().__init__(
            dataset, batch_size=batch_size, drop_last=drop_last,
            valid_label_columns=valid_label_columns,
            generator=generator, seed=seed,
        )
        # __iter__ seeds its RNG from self.generator and self.seed. The base
        # class assigns both, but only from sentence-transformers 5.3 onward,
        # and mypy resolves whichever version the machine happens to carry --
        # against the 3.2 serving pin those reads have no attribute to bind
        # to. Owning the two that arrive as arguments keeps the contract
        # __iter__ depends on inside this class, so a base-class change cannot
        # drop it silently. self.epoch stays with SetEpochMixin, which owns it:
        # the trainer calls set_epoch() between epochs to mutate it.
        self.generator = generator
        self.seed = seed
        self.temperature = (
            self._temperature_override
            if self._temperature_override is not None else temperature
        )
        # 0 disables stratum balancing and leaves the binary is_ai behaviour
        # untouched, so an existing run reproduces exactly.
        self.strata_temperature = (
            self._strata_temperature_override
            if self._strata_temperature_override is not None
            else strata_temperature
        )

        if self._hub_ids_override is not None:
            self.hub_ids = self._hub_ids_override
            self.is_ai = self._is_ai_override or [False] * len(dataset)
            self.anchor_keys = self._anchor_keys_override
            self.strata = self._strata_override
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
            self.strata = (
                dataset["branch"] if "branch" in dataset.column_names else None
            )
        else:
            raise ValueError("Dataset must have a 'hub_id' column or use set_metadata()")
        self.n = len(dataset)

    def _order_by_ai(self, rng: Any) -> list[int]:
        """The original binary AI-vs-traditional temperature interleave."""
        ai_indices = [i for i in range(self.n) if self.is_ai[i]]
        trad_indices = [i for i in range(self.n) if not self.is_ai[i]]
        rng.shuffle(ai_indices)
        rng.shuffle(trad_indices)
        n_ai, n_trad = len(ai_indices), len(trad_indices)

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
        return ordered

    def _order_by_strata(self, rng: Any) -> list[int]:
        """Temperature-flatten an arbitrary stratum distribution.

        Generalises the binary is_ai interleave to N classes so the CRE
        branch can be balanced. 72.1% of training links point at "Technical
        application security controls" and 3.3% at "Cross-cutting concerns",
        and CAPEC alone is 42.5% of links with none of its 702
        adversary-as-subject anchors pointing at the threat branch. The model
        therefore learns "attack narrative -> the control that stops it",
        which is right for CAPEC and wrong for the MITRE ATLAS techniques
        that want the threat itself -- measured at -29.4 hit@1 points on that
        stratum, enough to make the whole ATLAS fold negative.

        p^(1/T): T=1 leaves the natural distribution, larger T flattens
        toward uniform. Sampling is without replacement -- each example still
        appears exactly once per epoch, only the ORDER changes, so no example
        is duplicated or dropped and epoch size is unchanged.
        """
        assert self.strata is not None
        buckets: dict[str, list[int]] = {}
        for i in range(self.n):
            buckets.setdefault(str(self.strata[i]), []).append(i)

        # Order WITHIN a stratum by the AI interleave, so enabling branch
        # balancing does not silently switch off AI upsampling. Without this
        # the balanced arm differed from the baseline in two ways at once and
        # a null result could not be attributed to branch balance -- while
        # to_dict still recorded sampling_temperature as though it applied.
        for key, idx_list in buckets.items():
            member = set(idx_list)
            ai_order = self._order_by_ai(rng)
            buckets[key] = [i for i in ai_order if i in member]

        keys = sorted(buckets)
        weights = [
            (len(buckets[k]) / self.n) ** (1.0 / self.strata_temperature)
            for k in keys
        ]
        total = sum(weights)
        probs = [w / total for w in weights] if total > 0 else None

        pointers = dict.fromkeys(keys, 0)
        ordered: list[int] = []
        for _ in range(self.n):
            live = [k for k in keys if pointers[k] < len(buckets[k])]
            if not live:
                break
            if probs is None:
                chosen = live[0]
            else:
                live_p = [probs[keys.index(k)] for k in live]
                s = sum(live_p)
                chosen = live[0] if s <= 0 else str(
                    rng.choice(live, p=[p / s for p in live_p])
                )
            ordered.append(buckets[chosen][pointers[chosen]])
            pointers[chosen] += 1
        return ordered

    def __iter__(self) -> Iterator[list[int]]:
        if self.generator is not None:
            seed = int(torch.randint(0, 2**31, (1,), generator=self.generator).item())
        else:
            seed = self.seed + self.epoch
        rng = np.random.default_rng(seed)

        if self.strata is not None and self.strata_temperature > 0:
            ordered = self._order_by_strata(rng)
        else:
            ordered = self._order_by_ai(rng)

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
                # Every survivor collides with the current batch, so no
                # further deferral can help. Emit them in batch_size chunks
                # rather than as one oversized batch: extending here yielded
                # 111 examples where 32 was configured under the AI ordering
                # and 175 under branch stratification, measured on the real
                # MITRE ATLAS fold. Peak memory scales with the largest batch,
                # and config.py already records that a worst-case batch OOMed
                # an 80GB H100 -- at a longer token budget this one is fatal
                # rather than merely wasteful.
                #
                # Chunking accepts hub collisions WITHIN these leftovers, which
                # costs some MNRL false negatives on a small tail. That is the
                # better trade: the alternative silently breaks the batch-size
                # contract the memory guard depends on.
                if next_remaining:
                    logger.debug(
                        "Batch sampler: %d examples could not be placed "
                        "without collision; emitting in chunks of %d.",
                        len(next_remaining), self.batch_size,
                    )
                for start in range(0, len(next_remaining), self.batch_size):
                    chunk = next_remaining[start:start + self.batch_size]
                    room = self.batch_size - len(batch)
                    if room and len(chunk) <= room:
                        batch.extend(chunk)
                    else:
                        if batch:
                            yield batch
                            batch = []
                        batch = list(chunk)
                    if len(batch) >= self.batch_size:
                        yield batch
                        batch = []
                break
            remaining = next_remaining

        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        n_batches: int = math.ceil(self.n / self.batch_size)
        return n_batches


def top_level_branch(hub_id: str, hierarchy: CREHierarchy) -> str:
    """The root of the CRE tree this hub hangs from.

    Coarse on purpose. The failure it addresses is branch-level -- attack
    narrative routed to the control that mitigates it rather than to the
    threat itself -- and finer strata would leave most buckets too small to
    sample from.
    """
    node = hierarchy.hubs.get(hub_id)
    if node is None:
        return "unknown"
    path = node.hierarchy_path or node.name
    return path.split(">")[0].strip() or node.name


def pairs_to_dataset(
    pairs: list[TrainingPair],
    hierarchy: CREHierarchy,
    hub_texts: dict[str, str],
    n_hard_negatives: int = 3,
) -> Dataset:
    """Convert TrainingPairs to a sentence-transformers Dataset with hard negatives.

    Output columns: anchor, positive, negative_1..N, hub_id, is_ai, anchor_key
    """
    records: list[dict[str, Any]] = []
    for pair in pairs:
        record: dict[str, Any] = {
            "anchor": pair.control_text,
            "positive": pair.hub_representation,
            "hub_id": pair.hub_id,
            "is_ai": pair.framework in AI_FRAMEWORK_NAMES,
            "anchor_key": pair.control_text.lower().strip(),
            # Top-level CRE branch of the target, so the sampler can flatten a
            # distribution that is 72.1% "Technical application security
            # controls" and 3.3% "Cross-cutting concerns".
            "branch": top_level_branch(pair.hub_id, hierarchy),
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
