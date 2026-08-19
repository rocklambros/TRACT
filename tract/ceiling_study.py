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
the owner and the five LLM panels scored:

  owasp_ai_exchange 62 -> 63 anchors, on the anchor gate alone
      207 of 250 items hold their position, 43 are replaced
  capec 339 -> 349 and cwe 240 -> 245, when the contested recovery is on
      82 of 250 items hold their position, 77 are replaced

So `build_ceiling_study()` no longer redraws the scored sample, and reverting
the recovery commit does not restore it either. Those artifacts stand as a
record, and every one of their 250 anchors is still in the pool, which a test
asserts, so alpha-1 and alpha-5 remain measurable against the text that was
scored. What is gone is the ability to regenerate the sample from this code.
Any new draw is not comparable to the scored 250 without saying so.
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Final, Literal, Mapping, TypedDict

from tract.config import (
    CEILING_STUDY_N_ITEMS,
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
