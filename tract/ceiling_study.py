"""Blind expert-agreement ceiling study (design doc Part 0.1).

The only ceiling evidence in this project before this module was the Phase 3
hidden-calibration result: 13 of 20 items, Wilson half-width 0.193, too wide
to gate anything. This module builds the replacement: a 250-item study drawn
from the frameworks whose control text is stable under the pending corpus
rebuild, stratified so the owner can stop partway and still have an
analyzable sample.

Two quantities the exported items are scored against later, by a human, not
by this module:

- alpha-1: the annotator's single best hub matches OpenCRE. Ceiling on hit@1.
- alpha-5: OpenCRE's hub is inside the annotator's set of up to five.
  Ceiling on hit@5.

`ceiling_items.json` carries no ground truth. `build_ceiling_study()` keeps
the item stream and the answer key as two separate return values for exactly
that reason: nothing downstream of this function can accidentally staple the
gold hub back onto an item.

The sampling frame moved when the training gate moved to the resolved anchor.
`_load_eligible_links` mirrors training, and training now admits links whose
section title was too short for the retired floor. `sample_ceiling_items`
draws with `rng.sample` over the resulting pool, so the frame size alone
changes which items come out at the same seed.

MEASURED, against `results/ceiling_study/ceiling_items.json`, the 250 items
the owner and the five LLM panels scored. Two counts, because they answer
different questions and an earlier version of this docstring reported one of
each as though they were a pair:

  owasp_ai_exchange 62 -> 63 anchors, on the anchor gate alone
      207 of 250 positions hold, 43 carry a different control
  capec 339 -> 349 and cwe 240 -> 245, when the contested recovery is on
      82 of 250 positions hold, 168 carry a different control, and 77 of
      the scored controls are absent from the fresh sample altogether

Every answer file keys on `item_index`, so 168 is the number that governs
whether an answer still lines up. 77 is the smaller question: how many of the
annotated controls a fresh draw drops, ignoring where the rest landed.

So `build_ceiling_study()` no longer redraws the scored sample, and reverting
the recovery commit does not restore it either. Those artifacts stand as a
record, and every one of their 250 anchors is still in the pool, which a test
asserts, so alpha-1 and alpha-5 remain measurable against the text that was
scored. What is gone is the ability to regenerate the sample from this code.
Any new draw is not comparable to the scored 250 without saying so.

Ruling R22 turns that paragraph into machinery. `results/ceiling_study/
ceiling_items.json` is the study of record, `load_ceiling_items()` is how a
consumer reaches it, and `require_pinned_study_unmodified()` compares the file
against the digest recorded in `ceiling_study_provenance.json` on every one of
those reads. `require_unmoved_ceiling_study()` refuses to let a fresh draw
overwrite the pinned artifact, and `new_study_path()` is where a fresh draw
goes instead. The divergence is a measurement this module can state, not a
condition it resolves in favour of whichever ran last.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Mapping, TypedDict

from tract.config import (
    CEILING_STUDY_N_ITEMS,
    CEILING_STUDY_NEW_DIR,
    CEILING_STUDY_PINNED_ITEMS,
    CEILING_STUDY_PROVENANCE_PATH,
    CEILING_STUDY_SEED,
    CEILING_STUDY_STRATUM_SIZE,
    CEILING_STUDY_TEST_FRAMEWORKS,
    CEILING_STUDY_VALIDATION_FRAMEWORKS,
    PROCESSED_DIR,
)
from tract.hierarchy import CREHierarchy
from tract.io import load_json
from tract.text_selection import ProseIndex, select_control_text
from tract.training.data_quality import (
    QualityTier,
    assign_quality_tier,
    curated_link_filter_report,
)

logger = logging.getLogger(__name__)

# Same priority training gives quality tiers when a text collapses two links
# onto one anchor (tract/training/data.py:TIER_PRIORITY): prefer human
# LinkedTo over AutomaticallyLinkedTo. Duplicated rather than imported
# because tract.training.data pulls in torch, sentence-transformers and
# datasets, and this module has to run without a GPU or those packages.
_TIER_PRIORITY: Final[dict[str, int]] = {
    QualityTier.T1.value: 0,
    QualityTier.T1_AI.value: 1,
    QualityTier.T3.value: 2,
    QualityTier.AL.value: 3,
}

Stratum = Literal["validation", "test"]


class CeilingItem(TypedDict):
    """One row of ceiling_items.json. No ground truth, ever."""

    item_index: int
    framework_id: str
    framework_name: str
    control_id: str
    control_title: str
    control_text: str
    text_source: str
    stratum: Stratum


class AnswerKeyEntry(TypedDict):
    """One row of ceiling_answer_key.json."""

    item_index: int
    primary_gold_hub_id: str
    valid_gold_hub_ids: list[str]


class TemplateAnswer(TypedDict):
    """One row of ceiling_answers_TEMPLATE.json, the owner's worksheet."""

    item_index: int
    primary_hub_id: str
    acceptable_hub_ids: list[str]
    confidence: str
    notes: str


@dataclass(frozen=True)
class AnchorRecord:
    """One deduplicated control anchor, before sampling.

    gold_hub_ids holds every CRE hub this anchor's resolved text links to in
    the filtered pool (usually one, sometimes several when the anchor is a
    multi-hub AutomaticallyLinkedTo target). primary_gold_hub_id is the
    highest-quality-tier link among them, tie-broken by hub id.
    """

    framework_id: str
    framework_name: str
    control_id: str
    control_title: str
    control_text: str
    text_source: str
    anchor_key: str
    primary_gold_hub_id: str
    gold_hub_ids: tuple[str, ...]


def _load_eligible_links(allowed_framework_ids: frozenset[str]) -> list[dict[str, str]]:
    """Curated links, quality-filtered, restricted to the eligible frameworks.

    Calls tract.training.data_quality.curated_link_filter_report, the function
    training calls, rather than repeating the gate beside it. The previous
    version inlined an assign_quality_tier(record) call under a docstring
    claiming it mirrored training. When that function gained a resolved-anchor
    argument, the copy here would have kept compiling against the old contract
    and quietly stopped mirroring: the study pool would keep the section-title
    gate while training moved to the anchor gate, and nothing would raise.
    """
    report, _ = curated_link_filter_report()
    return [
        tiered.link for tiered in report.kept
        if tiered.link.get("framework_id") in allowed_framework_ids
    ]


def _link_priority(
    link: dict[str, str], prose_index: ProseIndex,
) -> tuple[int, str, str]:
    """Sort key preferring higher-quality tiers, then section id, then name.

    Used both to pick which raw link represents a multi-link anchor's
    control_id/control_title and, via the same ordering, its primary hub.

    Takes the index because assign_quality_tier now needs the anchor. The one
    caller, build_anchor_pool, already holds it.
    """
    selection = prose_index.lookup(
        link.get("standard_name", ""),
        link.get("section_id"),
        link.get("section_name"),
    )
    tier = assign_quality_tier(
        link, selection.text if selection else None,
    ).value
    return (
        _TIER_PRIORITY.get(tier, 99),
        link.get("section_id", ""),
        link.get("section_name", ""),
    )


def build_anchor_pool(
    links: list[dict[str, str]], prose_index: ProseIndex,
) -> dict[str, list[AnchorRecord]]:
    """Resolve links to prose and dedupe to one AnchorRecord per control.

    Grouping is scoped per framework_id. Verified against this pool (no
    cross-framework text collisions among the 7 eligible frameworks as of
    the 2026-08-15 corpus), but scoping per framework is also the correct
    behavior on its own terms: an anchor belongs to the framework whose link
    produced it, and a cross-framework merge would leave the merged anchor
    with no single home framework for stratified sampling.

    Returns:
        framework_id -> list of AnchorRecord, each anchor exactly once,
        ordered by anchor_key for deterministic downstream sampling.
    """
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    resolved: dict[tuple[str, str], tuple[str, str]] = {}  # -> (text, source)

    for link in links:
        standard_name = link["standard_name"]
        selection = select_control_text(
            prose_index,
            standard_name,
            link.get("section_id"),
            link.get("section_name"),
        )
        anchor_key = selection.text.lower().strip()
        group_key = (link["framework_id"], anchor_key)
        groups.setdefault(group_key, []).append(link)
        resolved[group_key] = (selection.text, selection.source)

    pool: dict[str, list[AnchorRecord]] = {}
    for (framework_id, anchor_key), members in groups.items():
        text, source = resolved[(framework_id, anchor_key)]
        representative = min(members, key=lambda m: _link_priority(m, prose_index))
        gold_hub_ids = tuple(sorted({m["cre_id"] for m in members}))

        record = AnchorRecord(
            framework_id=framework_id,
            framework_name=representative["standard_name"],
            control_id=representative.get("section_id", ""),
            control_title=representative.get("section_name", ""),
            control_text=text,
            text_source=source,
            anchor_key=anchor_key,
            primary_gold_hub_id=representative["cre_id"],
            gold_hub_ids=gold_hub_ids,
        )
        pool.setdefault(framework_id, []).append(record)

    for framework_id, records in pool.items():
        records.sort(key=lambda r: r.anchor_key)

    return pool


def apportion_with_caps(
    weights: Mapping[str, int], total: int, caps: Mapping[str, int],
) -> dict[str, int]:
    """Largest-remainder apportionment of `total` across `weights`, capped.

    Standard Hamilton apportionment breaks when a proportional quota exceeds
    a category's actual capacity -- exactly what happens here: mitre_atlas,
    nist_ai_100_2 and owasp_llm_top10 all have far fewer unique anchors than
    their link-count-proportional quota would assign. This fixes any
    over-quota key at its cap, removes it from the pool, and re-derives
    quotas for the rest, repeating until no quota exceeds its cap. What is
    left is split across the still-uncapped keys by largest remainder.

    Args:
        weights: Positive integer weight per key (used for proportionality).
        total: The total to distribute. Must not exceed sum(caps.values()).
        caps: Maximum allocation per key. Must cover every key in weights.

    Returns:
        key -> allocated count. Sums to `total` exactly. Each value is
        between 0 and caps[key] inclusive.

    Raises:
        ValueError: If total is negative, a cap is missing or negative, or
            the caps cannot possibly sum to `total`.
    """
    keys = list(weights)
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    for key in keys:
        if key not in caps:
            raise ValueError(f"no cap given for {key!r}")
        if caps[key] < 0:
            raise ValueError(f"negative cap for {key!r}: {caps[key]}")
        if weights[key] <= 0:
            raise ValueError(f"weight for {key!r} must be positive, got {weights[key]}")
    capacity = sum(caps[k] for k in keys)
    if capacity < total:
        raise ValueError(
            f"total {total} exceeds combined capacity {capacity} across {keys}"
        )
    if not keys:
        if total != 0:
            raise ValueError("no keys to apportion a nonzero total across")
        return {}

    fixed: dict[str, int] = {}
    remaining_keys = set(keys)
    remaining_total = total

    while True:
        remaining_weight = sum(weights[k] for k in remaining_keys)
        if remaining_weight == 0:
            break
        exceeded: str | None = None
        for key in sorted(remaining_keys):
            quota = remaining_total * weights[key] / remaining_weight
            if quota > caps[key] + 1e-9:
                exceeded = key
                break
        if exceeded is None:
            break
        fixed[exceeded] = caps[exceeded]
        remaining_total -= caps[exceeded]
        remaining_keys.discard(exceeded)

    remaining_weight = sum(weights[k] for k in remaining_keys)
    quotas = {
        k: (remaining_total * weights[k] / remaining_weight if remaining_weight else 0.0)
        for k in remaining_keys
    }
    floors = {k: math.floor(q) for k, q in quotas.items()}
    shortfall = remaining_total - sum(floors.values())
    # Largest fractional remainder first. Key name breaks ties so the result
    # does not depend on set iteration order.
    ranked = sorted(remaining_keys, key=lambda k: (-(quotas[k] - floors[k]), k))

    result: dict[str, int] = dict(fixed)
    result.update(floors)
    for key in ranked[:shortfall]:
        result[key] += 1

    for key in keys:
        if not (0 <= result[key] <= caps[key]):
            raise ValueError(
                f"apportionment produced {result[key]} for {key!r}, "
                f"outside [0, {caps[key]}]"
            )
    if sum(result.values()) != total:
        raise ValueError(
            f"apportionment sums to {sum(result.values())}, expected {total}"
        )
    return result


def _balanced_interleave(
    stratum_a: list[CeilingItem], stratum_b: list[CeilingItem], rng: random.Random,
) -> list[CeilingItem]:
    """Merge two equal-length lists so every prefix of length >= 2 has both.

    Pairs items positionally (a[i], b[i]) and randomizes which of the two
    comes first within each pair. Every completed pair contributes exactly
    one item from each stratum, so as soon as one pair is complete -- a
    prefix of length 2 -- both strata are represented, and every longer
    prefix only adds to that, never removes it.

    Requires len(stratum_a) == len(stratum_b): the design's two strata are
    both exactly CEILING_STUDY_STRATUM_SIZE by construction, and an unequal
    pairing cannot make the guarantee above hold near the end of the list.
    """
    if len(stratum_a) != len(stratum_b):
        raise ValueError(
            f"strata must be equal length to interleave with the both-strata "
            f"guarantee, got {len(stratum_a)} and {len(stratum_b)}"
        )
    merged: list[CeilingItem] = []
    for a_item, b_item in zip(stratum_a, stratum_b):
        pair = [a_item, b_item] if rng.random() < 0.5 else [b_item, a_item]
        merged.extend(pair)
    return merged


def _sample_stratum(
    pool: dict[str, list[AnchorRecord]],
    frameworks: tuple[str, ...],
    link_weights: Mapping[str, int],
    stratum_name: Stratum,
    stratum_size: int,
    rng: random.Random,
) -> tuple[list[CeilingItem], dict[str, int]]:
    """Allocate and draw one stratum's worth of anchors, framework by framework.

    Returns (items, allocation) so the caller can report exactly what was
    drawn without recomputing the (deterministic, but still redundant)
    apportionment a second time.
    """
    caps = {fw: len(pool.get(fw, [])) for fw in frameworks}
    weights = {fw: link_weights[fw] for fw in frameworks}
    allocation = apportion_with_caps(weights, stratum_size, caps)

    drawn: list[CeilingItem] = []
    for fw in frameworks:
        n = allocation[fw]
        framework_pool = pool.get(fw, [])
        chosen = rng.sample(framework_pool, k=n)
        for record in chosen:
            drawn.append({
                "item_index": 0,  # filled in after final interleave
                "framework_id": record.framework_id,
                "framework_name": record.framework_name,
                "control_id": record.control_id,
                "control_title": record.control_title,
                "control_text": record.control_text,
                "text_source": record.text_source,
                "stratum": stratum_name,
            })
    rng.shuffle(drawn)
    return drawn, allocation


class SamplingSummary(TypedDict):
    """Per-framework allocation, for the caller to report and log."""

    seed: int
    n_items: int
    validation_allocation: dict[str, int]
    test_allocation: dict[str, int]
    validation_pool_sizes: dict[str, int]
    test_pool_sizes: dict[str, int]


def sample_ceiling_items(
    pool: dict[str, list[AnchorRecord]],
    link_weights: Mapping[str, int],
    seed: int = CEILING_STUDY_SEED,
    stratum_size: int = CEILING_STUDY_STRATUM_SIZE,
) -> tuple[list[CeilingItem], SamplingSummary]:
    """Draw and interleave the 250-item ceiling study from an anchor pool.

    Deterministic in `seed`: same pool, same weights, same seed produces the
    same 250 items in the same order every time. All randomness comes from a
    single random.Random(seed) instance consumed in a fixed order --
    validation allocation draws (framework by framework, in
    CEILING_STUDY_VALIDATION_FRAMEWORKS order), then test allocation draws,
    then each stratum's internal shuffle, then the pairwise interleave -- so
    the sequence of operations is itself part of what the determinism test
    is checking, not just the seed.

    Args:
        pool: framework_id -> deduplicated AnchorRecord list, from
            build_anchor_pool.
        link_weights: framework_id -> count of filtered links (NOT anchors).
            Proportional allocation is computed against link counts, per the
            study's pre-registration, so a framework whose links collapse
            onto relatively few anchors (heavy multi-hub reuse, e.g. capec)
            is not penalized twice for that collapse.
        seed: RNG seed. Recorded in the output for the study to be
            reproducible from the artifact alone.
        stratum_size: Items per stratum. Total items is 2x this.
    """
    rng = random.Random(seed)

    validation_weights = {
        fw: link_weights[fw] for fw in CEILING_STUDY_VALIDATION_FRAMEWORKS
    }
    validation_items, validation_allocation = _sample_stratum(
        pool, CEILING_STUDY_VALIDATION_FRAMEWORKS, validation_weights,
        "validation", stratum_size, rng,
    )

    test_weights = {fw: link_weights[fw] for fw in CEILING_STUDY_TEST_FRAMEWORKS}
    test_items, test_allocation = _sample_stratum(
        pool, CEILING_STUDY_TEST_FRAMEWORKS, test_weights,
        "test", stratum_size, rng,
    )

    merged = _balanced_interleave(validation_items, test_items, rng)
    for i, item in enumerate(merged, start=1):
        item["item_index"] = i

    summary: SamplingSummary = {
        "seed": seed,
        "n_items": len(merged),
        "validation_allocation": validation_allocation,
        "test_allocation": test_allocation,
        "validation_pool_sizes": {
            fw: len(pool.get(fw, [])) for fw in CEILING_STUDY_VALIDATION_FRAMEWORKS
        },
        "test_pool_sizes": {
            fw: len(pool.get(fw, [])) for fw in CEILING_STUDY_TEST_FRAMEWORKS
        },
    }
    return merged, summary


def build_answer_key(items: list[CeilingItem], pool: dict[str, list[AnchorRecord]]) -> list[AnswerKeyEntry]:
    """Build the hidden key from the same AnchorRecords the items came from.

    Looked up by (framework_id, control_id, control_text) rather than by
    object identity, so the key can be rebuilt from a saved items file plus
    a freshly-rebuilt pool without needing the AnchorRecord objects to
    survive a serialization round trip.
    """
    by_lookup: dict[tuple[str, str, str], AnchorRecord] = {}
    for framework_id, records in pool.items():
        for record in records:
            by_lookup[(framework_id, record.control_id, record.control_text)] = record

    key: list[AnswerKeyEntry] = []
    for item in items:
        lookup_key = (item["framework_id"], item["control_id"], item["control_text"])
        matched = by_lookup.get(lookup_key)
        if matched is None:
            raise ValueError(
                f"item_index={item['item_index']} has no matching anchor record "
                f"for {lookup_key[:2]!r}"
            )
        key.append({
            "item_index": item["item_index"],
            "primary_gold_hub_id": matched.primary_gold_hub_id,
            "valid_gold_hub_ids": list(matched.gold_hub_ids),
        })
    return key


def build_answer_template(items: list[CeilingItem]) -> list[TemplateAnswer]:
    """The blank worksheet the owner fills in, one row per item."""
    return [
        {
            "item_index": item["item_index"],
            "primary_hub_id": "",
            "acceptable_hub_ids": [],
            "confidence": "",
            "notes": "",
        }
        for item in items
    ]


def load_prose_index() -> ProseIndex:
    """Thin wrapper so callers do not need to know ProseIndex's default path."""
    return ProseIndex.load()


def load_hierarchy() -> CREHierarchy:
    return CREHierarchy.load(PROCESSED_DIR / "cre_hierarchy.json")


def load_hub_descriptions() -> dict[str, str]:
    """hub_id -> best available expert description text.

    Prefers a reviewer's edited text over the generated one when both exist.
    Hubs with no description at all (every non-leaf hub, plus any leaf never
    run through the description pipeline) are simply absent from the dict.
    """
    data = load_json(PROCESSED_DIR / "hub_descriptions_reviewed.json")
    out: dict[str, str] = {}
    for hub_id, entry in data.get("descriptions", {}).items():
        text = entry.get("reviewed_description") or entry.get("description")
        if text:
            out[hub_id] = text
    return out


def render_hub_reference(
    hierarchy: CREHierarchy, descriptions: dict[str, str],
) -> str:
    """All 522 hubs grouped by their 5 top branches, outline-ordered.

    Within a branch, hubs are ordered by hierarchy_path string. That sort is
    a valid depth-first, alphabetical-at-each-level traversal: a parent's
    path is always a strict prefix of its children's paths, and a string
    sort places a prefix immediately before anything that extends it.
    """
    lines: list[str] = [
        "# CRE Hub Reference",
        "",
        "All 522 hubs in the CRE hierarchy, grouped by top-level branch and",
        "ordered by hierarchy path. Expert descriptions are shown where one",
        "exists (the 400 leaf hubs that have been through description review).",
        "Intermediate and root hubs carry no description of their own.",
        "",
    ]

    for root_id in sorted(hierarchy.roots, key=lambda r: hierarchy.hubs[r].name):
        root = hierarchy.hubs[root_id]
        branch_hub_ids = hierarchy.get_branch_hub_ids(root_id)
        branch_nodes = sorted(
            (hierarchy.hubs[hid] for hid in branch_hub_ids),
            key=lambda n: n.hierarchy_path,
        )
        lines.append(f"## {root.name} ({root_id})")
        lines.append("")
        for node in branch_nodes:
            lines.append(f"### {node.hub_id} -- {node.name}")
            lines.append(f"Path: {node.hierarchy_path}")
            description = descriptions.get(node.hub_id)
            if description:
                lines.append(f"Description: {description}")
            else:
                lines.append("Description: (none on record for this hub)")
            lines.append("")

    return "\n".join(lines) + "\n"


def eligible_framework_ids() -> frozenset[str]:
    return frozenset(CEILING_STUDY_VALIDATION_FRAMEWORKS) | frozenset(
        CEILING_STUDY_TEST_FRAMEWORKS
    )


def count_filtered_links(links: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for link in links:
        fid = link["framework_id"]
        counts[fid] = counts.get(fid, 0) + 1
    return counts


def build_ceiling_study(
    seed: int = CEILING_STUDY_SEED,
) -> tuple[list[CeilingItem], list[AnswerKeyEntry], list[TemplateAnswer], str, SamplingSummary]:
    """End-to-end build: load data, sample, and render every export artifact.

    Returns (items, answer_key, template, hub_reference_markdown, summary).
    Pure with respect to seed and the on-disk corpus -- no writes here, so a
    caller (script or test) decides where and whether to persist the result.
    """
    links = _load_eligible_links(eligible_framework_ids())
    prose_index = load_prose_index()
    pool = build_anchor_pool(links, prose_index)
    link_weights = count_filtered_links(links)

    items, summary = sample_ceiling_items(pool, link_weights, seed=seed)
    if len(items) != CEILING_STUDY_N_ITEMS:
        raise ValueError(f"expected {CEILING_STUDY_N_ITEMS} items, got {len(items)}")

    answer_key = build_answer_key(items, pool)
    template = build_answer_template(items)

    hierarchy = load_hierarchy()
    descriptions = load_hub_descriptions()
    hub_reference_md = render_hub_reference(hierarchy, descriptions)

    return items, answer_key, template, hub_reference_md, summary


# ── Ruling R22: the annotated study is an artifact, not a redraw ──────────

_ITEM_STRING_FIELDS: Final[tuple[str, ...]] = (
    "framework_id", "framework_name", "control_id", "control_title",
    "control_text", "text_source", "stratum",
)

# A study name becomes a directory name, so the allowlist is the traversal
# defence: no dot, no separator, no drive letter can survive it. Kept
# deliberately narrow rather than sanitised after the fact, because a
# sanitiser that silently rewrites a name hands back a path the caller did
# not ask for, which is the accident this whole ruling exists to stop.
_STUDY_NAME_ALPHABET: Final[frozenset[str]] = frozenset(
    "abcdefghijklmnopqrstuvwxyz0123456789_-"
)
_STUDY_NAME_MAX_CHARS: Final[int] = 64

# What `drawn_from.recovery` in the provenance record is allowed to say.
# "reproduced" means the recorded inputs were replayed and returned the pinned
# items. "inferred" means they were read off the commit that pinned the study
# without a replay. "unrecoverable" means they are gone, and the schema check
# below refuses to let that word sit beside a digest, because a provenance
# that is a guess is worse than an absent one.
_RECOVERY_STATES: Final[frozenset[str]] = frozenset(
    {"reproduced", "inferred", "unrecoverable"}
)


@dataclass(frozen=True)
class StudyDivergence:
    """How far a fresh draw has moved from a pinned one.

    Two numbers, because they answer different questions and the run ledger
    conflated them. `positions_replaced` is the one that governs whether an
    answer file still lines up, since every answer keys on `item_index`.
    `pinned_items_absent` is the smaller, kinder number: how many annotated
    controls the fresh sample dropped entirely, ignoring where they landed.
    """

    n_pinned: int
    n_fresh: int
    positions_held: int
    positions_replaced: int
    pinned_items_absent: int

    def describe(self) -> str:
        # A fresh draw LONGER than the pinned one moves no position and drops
        # no anchor, so both counts read zero while the study has still
        # changed. The size clause leads in that case, because a message
        # opening "0 of 4 positions now carry a different control" reads as a
        # match.
        size = (
            ""
            if self.n_fresh == self.n_pinned
            else f"the fresh draw holds {self.n_fresh} items against "
                 f"{self.n_pinned} pinned, and "
        )
        return (
            f"{size}{self.positions_replaced} of {self.n_pinned} item "
            f"positions now carry a different control "
            f"({self.positions_held} hold), and {self.pinned_items_absent} of "
            f"the pinned controls are absent from the fresh sample of "
            f"{self.n_fresh} altogether"
        )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _coerce_items(raw: Any, source: str) -> list[CeilingItem]:
    """Validate a decoded items payload into CeilingItem rows.

    Args:
        raw: The decoded JSON, either the whole document or its "items" list.
        source: What to name in an error message, usually a path.

    Raises:
        ValueError: If the payload is not a list of well-formed item rows, or
            if item_index is not 1..N contiguous. Contiguity is checked
            because every answer file in this directory keys on item_index,
            so a gap or a repeat silently mis-joins a human's answers.
    """
    records = raw.get("items") if isinstance(raw, dict) else raw
    if not isinstance(records, list) or not records:
        raise ValueError(
            f"{source} carries no ceiling-study items: expected a non-empty "
            f"'items' list, got {type(records).__name__}"
        )

    items: list[CeilingItem] = []
    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(
                f"{source} item at position {position} is "
                f"{type(record).__name__}, not an object"
            )
        index = record.get("item_index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError(
                f"{source} item at position {position} has item_index "
                f"{index!r}, which is not an integer"
            )
        for field_name in _ITEM_STRING_FIELDS:
            value = record.get(field_name)
            if not isinstance(value, str):
                raise ValueError(
                    f"{source} item_index={index} has {field_name}={value!r}, "
                    f"which is not a string"
                )
        items.append({
            "item_index": index,
            "framework_id": str(record["framework_id"]),
            "framework_name": str(record["framework_name"]),
            "control_id": str(record["control_id"]),
            "control_title": str(record["control_title"]),
            "control_text": str(record["control_text"]),
            "text_source": str(record["text_source"]),
            "stratum": record["stratum"],
        })

    indices = [item["item_index"] for item in items]
    if indices != list(range(1, len(items) + 1)):
        raise ValueError(
            f"{source} item_index is not 1..{len(items)} contiguous in order, "
            f"so an answer file keyed on it would mis-join. First five: "
            f"{indices[:5]}"
        )
    return items


def _item_identity(item: CeilingItem) -> tuple[str, str, str]:
    """What makes two item rows the same drawn control."""
    return (item["framework_id"], item["control_id"], item["control_text"])


def _anchor_identity(item: CeilingItem) -> tuple[str, str]:
    """The anchor key build_anchor_pool would give this row.

    Deliberately ignores control_id. An anchor is its resolved text, and a
    parser that renames a section id without touching the prose has not
    dropped the control the owner scored.
    """
    return (item["framework_id"], item["control_text"].lower().strip())


def ceiling_study_divergence(
    fresh: list[CeilingItem], pinned: list[CeilingItem],
) -> StudyDivergence:
    """Measure a fresh draw against the pinned one. Never raises on a mismatch.

    Reporting is this function's whole job. The refusal lives in
    require_unmoved_ceiling_study, which calls this and then decides.
    """
    fresh_by_index = {item["item_index"]: _item_identity(item) for item in fresh}
    held = sum(
        1 for item in pinned
        if fresh_by_index.get(item["item_index"]) == _item_identity(item)
    )
    fresh_anchors = {_anchor_identity(item) for item in fresh}
    absent = sum(1 for item in pinned if _anchor_identity(item) not in fresh_anchors)
    return StudyDivergence(
        n_pinned=len(pinned),
        n_fresh=len(fresh),
        positions_held=held,
        positions_replaced=len(pinned) - held,
        pinned_items_absent=absent,
    )


def require_unmoved_ceiling_study(fresh: list[CeilingItem], path: Path) -> None:
    """Refuse to replace a drawn study with a draw that is not the same study.

    Shaped after tract.corpus_report.require_unmoved_corpus, for the same
    reason and with the same both-directions property: a fresh draw that
    REPRODUCES the artifact passes, because byte-identical regeneration is
    the check being protected, and a guard that blocked it would retire that
    check. Only a draw that moved is refused.

    Raises:
        ValueError: If path exists and holds a different sample, or exists and
            cannot be read for the comparison to happen at all.
    """
    if not path.exists():
        return

    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"refusing to overwrite {path}: it exists but is not valid JSON "
            f"({error}), so this run cannot tell whether it holds the study "
            f"someone annotated. An artifact that cannot be read is not an "
            f"artifact this run may replace on its own. Inspect it, restore "
            f"it from git, or draw into a new study with --study-name."
        ) from error

    pinned = _coerce_items(existing, str(path))
    divergence = ceiling_study_divergence(fresh, pinned)
    if divergence.positions_replaced == 0 and divergence.n_fresh == divergence.n_pinned:
        return

    raise ValueError(
        f"refusing to overwrite {path}: a draw at this seed no longer "
        f"reproduces it. {divergence.describe()}. Those items were annotated "
        f"by hand and every answer file beside them keys on item_index, so "
        f"overwriting would leave 250 answers pointing at controls nobody "
        f"scored, with no diff that says so. A fresh draw is a NEW study: "
        f"give it a name with --study-name."
    )


def require_new_study_destination(out_path: Path) -> None:
    """Refuse a destination that would land inside the annotated study.

    The whole of results/ceiling_study/ is off limits, not only
    ceiling_items.json, because the answer key and six answer files sit
    beside it under names a new draw would also want.

    Raises:
        ValueError: If out_path is the pinned artifact or a sibling of it.
    """
    resolved = out_path.resolve()
    pinned = CEILING_STUDY_PINNED_ITEMS.resolve()
    if resolved == pinned:
        raise ValueError(
            f"refusing to write a new study to {out_path}: that is the pinned "
            f"artifact, the 250 items a domain expert annotated by hand. "
            f"A fresh draw goes under {CEILING_STUDY_NEW_DIR} with its own name."
        )
    if resolved.parent == pinned.parent:
        raise ValueError(
            f"refusing to write a new study to {out_path}: {pinned.parent} "
            f"holds the annotated study, its hidden key and its answer files. "
            f"A fresh draw goes under {CEILING_STUDY_NEW_DIR} with its own name."
        )


def new_study_dir(name: str) -> Path:
    """Where a NEW draw goes. Never the pinned directory.

    Raises:
        ValueError: If the name is empty, over length, or carries a character
            outside the allowlist.
    """
    if not name:
        raise ValueError("a new ceiling study needs a name, got an empty string")
    if len(name) > _STUDY_NAME_MAX_CHARS:
        raise ValueError(
            f"study name is {len(name)} characters, over the "
            f"{_STUDY_NAME_MAX_CHARS} allowed: {name!r}"
        )
    offending = sorted({char for char in name if char not in _STUDY_NAME_ALPHABET})
    if offending:
        raise ValueError(
            f"study name {name!r} carries {offending}, outside the allowed "
            f"lowercase letters, digits, underscore and hyphen. The name "
            f"becomes a directory, so anything else is a path, not a name."
        )
    return CEILING_STUDY_NEW_DIR / name


def load_pinned_study_provenance(path: Path | None = None) -> dict[str, Any]:
    """Read what the pinned study was drawn from, and check the record is honest.

    Raises:
        ValueError: If the record is malformed, or claims a recovery state its
            recorded values contradict. "unrecoverable" beside a digest is the
            case that matters: a provenance that reconstructs a plausible
            value is worse than one that admits the value is gone, so the two
            are not allowed to coexist.
    """
    source = path or CEILING_STUDY_PROVENANCE_PATH
    record = load_json(source)
    if not isinstance(record, dict):
        raise ValueError(
            f"{source} is a {type(record).__name__}, not a provenance object"
        )

    artifact = record.get("pinned_artifact")
    if not isinstance(artifact, dict):
        raise ValueError(f"{source} has no 'pinned_artifact' object")
    for field_name in ("path", "sha256"):
        value = artifact.get(field_name)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"{source} pinned_artifact.{field_name} is {value!r}, "
                f"expected a non-empty string"
            )
    if not isinstance(artifact.get("n_items"), int):
        raise ValueError(
            f"{source} pinned_artifact.n_items is "
            f"{artifact.get('n_items')!r}, expected an integer"
        )

    drawn = record.get("drawn_from")
    if not isinstance(drawn, dict):
        raise ValueError(f"{source} has no 'drawn_from' object")
    recovery = drawn.get("recovery")
    if recovery not in _RECOVERY_STATES:
        raise ValueError(
            f"{source} drawn_from.recovery is {recovery!r}, not one of "
            f"{sorted(_RECOVERY_STATES)}"
        )
    if recovery == "unrecoverable":
        reconstructed = sorted(
            key for key in ("seed", "corpus_sha256", "curated_links_sha256")
            if drawn.get(key) is not None
        )
        if reconstructed:
            raise ValueError(
                f"{source} drawn_from.recovery says 'unrecoverable' while "
                f"still carrying {reconstructed}. A recorded provenance that "
                f"is a guess is worse than an absent one, so state one or the "
                f"other."
            )
    return record


def require_pinned_study_unmodified(
    path: Path | None = None, provenance_path: Path | None = None,
) -> None:
    """Refuse to read the annotated study if it no longer hashes to its record.

    The one tripwire on "do not modify ceiling_items.json". It fires on every
    read through load_ceiling_items, so an edit is caught by the next consumer
    rather than by a reviewer noticing a diff.

    Raises:
        ValueError: If the file's digest differs from the recorded one.
    """
    target = path or CEILING_STUDY_PINNED_ITEMS
    record = load_pinned_study_provenance(provenance_path)
    recorded = str(record["pinned_artifact"]["sha256"])
    measured = _sha256_file(target)
    if measured == recorded:
        return
    raise ValueError(
        f"refusing to read {target}: it no longer matches the study of "
        f"record.\n"
        f"  recorded in the provenance  {recorded}\n"
        f"  this file                   {measured}\n"
        f"These 250 items were annotated by hand and the six answer files "
        f"beside them key on item_index, so a changed item file re-points "
        f"every answer at a control nobody scored. Restore it from git. If "
        f"the study is genuinely being replaced, that is a NEW study with a "
        f"new name, and the provenance record moves with it."
    )


def load_ceiling_items(path: Path | None = None) -> list[CeilingItem]:
    """Read a ceiling-study items file. The study of record is verified too.

    This is how a consumer reaches "the ceiling study": from the tracked
    artifact, never from a fresh draw. build_ceiling_study() answers what a
    study drawn TODAY would look like, which stopped being the same question
    the moment the curated-link pool moved.

    The digest check runs only when the path resolves to the pinned artifact.
    Other item files with this schema exist (the contamination control, and
    any new study under results/ceiling_study/studies/), and they are not
    evidence of an expert's afternoon, so they get schema validation only.

    Raises:
        ValueError: If the file is malformed, or is the pinned artifact and
            no longer matches its provenance record.
    """
    target = path or CEILING_STUDY_PINNED_ITEMS
    if target.resolve() == CEILING_STUDY_PINNED_ITEMS.resolve():
        require_pinned_study_unmodified(target)
    return _coerce_items(load_json(target), str(target))
