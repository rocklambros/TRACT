"""Measure how much of Campaign 2's delta a domain oracle can account for.

THE QUESTION. Across all 4,405 curated links the framework-hub graph has exactly
two connected components -- 380 hubs supervised only by general-security
frameworks and 78 supervised only by AI-security ones, intersection empty. That
partition is not a labelling choice: enumerate the components with no AI/general
roster supplied and they fall out on their own, and all 147 test golds land in
the 78-hub component.

So a model can score on the AI test split without any semantic mapping, by
learning "this text is from an AI framework, so answer in the AI region" and
collapsing the effective label space from 522 candidates to 78. Campaign 2's
+0.1361 is consistent with that and nothing in the campaign excludes it.

THE MEASUREMENT. Hand the *zero-shot* encoder the domain for free -- restrict its
ranking pool to the 78-hub AI component -- and see how much of the trained
model's advantage evaporates:

  - restricted zero-shot lands near the trained 0.5918
        -> the delta was distractor rejection. The training taught domain
           detection, and curating more AI items measures the artifact more
           precisely. Do not fund curation.
  - restricted zero-shot stays near the full-pool 0.4558
        -> narrowing explains little, the gain is within-domain discrimination,
           and curation is justified.

WHY THIS AND NOT A RETRAIN. The three-arm firewall retrain this replaces could
not have answered it. Rotating ENISA/ETSI/BIML out of training leaves the
intersection at exactly zero with 73 of 78 AI hubs still AI-only supervised --
it deletes 40-49% of in-domain supervision while the cue survives untouched.
Restricting the candidate pool to a "domain-neutral" mix fails for a blunter
reason: the cue is written into the text being ranked. A bare regex \\bAI\\b over
the exact string `build_firewalled_hub_text` produces matches 78/78 AI-component
hubs and 0/380 general ones. You cannot hide a cue that is spelled out in the
candidates.

THE BUILT-IN CONTROL. `--require-baseline` asserts the full-pool zero-shot
reproduces the campaign's paired 0.4558 before the restricted number is
believed. If the reconstruction of the anchors, filters or hub texts is wrong,
the full-pool score moves and the restricted score means nothing. The control is
on by default and the run aborts rather than reporting a number it cannot anchor.

Persists FULL ranked lists for both pools. Campaign 2 stored only
`hit1_indicators` for the zero-shot and truncated the trained model's
predictions to the top 10, which is the only reason this question needed a GPU
at all instead of a query against artifacts already on disk.

Loads a model, so it runs on a pod. Encode-only: no training, no gradients.
"""
from __future__ import annotations

import argparse
import logging
import time
from collections import defaultdict
from typing import Any, Final


from scripts.phase0.common import (
    AI_FRAMEWORK_NAMES,
    build_evaluation_corpus,
    load_curated_links,
)
from tract.config import PHASE1B_RESULTS_DIR, PROCESSED_DIR
from tract.framework_identity import filter_set
from tract.hierarchy import CREHierarchy
from tract.io import atomic_write_json, load_json
from tract.text_selection import ProseIndex, apply_prose_to_corpus
from tract.training.evaluate import evaluate_on_fold
from tract.training.firewall import build_all_hub_texts

logger = logging.getLogger(__name__)

# The encoder and revision Campaign 2's arm A3 trained from. Pinned rather than
# floating: a different revision would answer a different question and the
# baseline control would fail without saying why.
BASE_MODEL: Final[str] = "Qwen/Qwen3-Embedding-0.6B"
BASE_REVISION: Final[str] = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"

# The paired zero-shot micro hit@1 that Campaign 2 reported alongside +0.1361.
# Reproducing it is what licenses believing the restricted figure.
CAMPAIGN2_ZERO_SHOT_HIT1: Final[float] = 0.4558
BASELINE_TOLERANCE: Final[float] = 0.02

OUTPUT_NAME: Final[str] = "domain_shortcut_probe"


def ai_component_hub_ids(links: list[Any]) -> list[str]:
    """Hubs in the AI-supervised connected component of the framework-hub graph.

    Derived structurally rather than from `AI_FRAMEWORK_NAMES`, because the
    roster names only the five frameworks that rotate, while ENISA, ETSI and
    BIML supervise the same region and never rotate out. Taking the component
    means the region is defined by the corpus rather than by a constant that
    happens to describe the eval split.
    """
    adjacency: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for link in links:
        hub, framework = ("H", link.cre_id), ("F", link.standard_name)
        adjacency[hub].add(framework)
        adjacency[framework].add(hub)

    components: list[set[tuple[str, str]]] = []
    seen: set[tuple[str, str]] = set()
    for node in adjacency:
        if node in seen:
            continue
        stack, component = [node], set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.add(current)
            stack.extend(adjacency[current] - seen)
        components.append(component)

    ai_components = [
        c for c in components
        if any(n[0] == "F" and n[1] in AI_FRAMEWORK_NAMES for n in c)
    ]
    if len(ai_components) != 1:
        raise ValueError(
            f"Expected the AI test frameworks to share exactly one component, "
            f"found {len(ai_components)}. The two-component structure this "
            "probe is built on no longer holds and the restricted pool is "
            "undefined."
        )
    return sorted(n[1] for n in ai_components[0] if n[0] == "H")


def build_corpus() -> list[Any]:
    """Rebuild arm A3's evaluation corpus: prose anchors, stop-words filtered.

    Mirrors `run_experiment` exactly, including fixing item identity from titles
    before swapping anchors in. Building the corpus from prose directly would
    let the anchor change the item count, which breaks the pairing.
    """
    corpus = build_evaluation_corpus(load_curated_links(), AI_FRAMEWORK_NAMES, {})
    return apply_prose_to_corpus(
        corpus,
        ProseIndex.load(),
        filter_set(use_stopwords=True, use_framework_identity=False),
        description_only=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-name", default=OUTPUT_NAME)
    parser.add_argument(
        "--require-baseline", action="store_true", default=True,
        help="Abort unless the full-pool zero-shot reproduces Campaign 2's "
             "0.4558. On by default.",
    )
    parser.add_argument(
        "--no-require-baseline", dest="require_baseline", action="store_false",
        help="Report the restricted number even if the control fails. Only for "
             "diagnosing WHY the control failed -- the result is not usable.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    start = time.time()

    hierarchy = CREHierarchy.model_validate(
        load_json(PROCESSED_DIR / "cre_hierarchy.json"),
    )
    all_hub_ids = sorted(hierarchy.hubs.keys())
    links = load_curated_links()
    restricted_hub_ids = ai_component_hub_ids(links)
    logger.info(
        "Ranking pools: full=%d hubs, AI component=%d hubs (chance %.4f -> %.4f)",
        len(all_hub_ids), len(restricted_hub_ids),
        1 / len(all_hub_ids), 1 / len(restricted_hub_ids),
    )

    corpus = build_corpus()
    by_framework: dict[str, list[Any]] = defaultdict(list)
    for item in corpus:
        by_framework[item.framework_name].append(item)

    # Every test gold must be reachable inside the restricted pool, or the
    # restricted score is bounded below the full-pool score for a reason that
    # has nothing to do with domain narrowing.
    restricted = set(restricted_hub_ids)
    unreachable = [
        item.ground_truth_hub_id for items in by_framework.values()
        for item in items if item.ground_truth_hub_id not in restricted
    ]
    if unreachable:
        raise ValueError(
            f"{len(unreachable)} test golds fall outside the AI component "
            f"(e.g. {unreachable[:3]}). A restricted pool that cannot contain "
            "the answer measures pool membership, not domain narrowing."
        )
    logger.info("All %d test golds are reachable in the restricted pool", len(corpus))

    from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    logger.info("Loading %s @ %s", BASE_MODEL, BASE_REVISION)
    model = SentenceTransformer(BASE_MODEL, revision=BASE_REVISION)

    folds: list[dict[str, Any]] = []
    for framework in sorted(AI_FRAMEWORK_NAMES):
        items = by_framework.get(framework, [])
        if not items:
            raise ValueError(f"{framework} contributes no eval items")

        # Firewalled per fold to match the campaign. At hub_rep_format
        # "path+name" the exclusion is a no-op on the text, but it is applied
        # rather than skipped so this stays correct if the format changes.
        hub_texts = build_all_hub_texts(hierarchy, excluded_framework=framework)

        full_metrics, full_ranked, full_hits = evaluate_on_fold(
            model, items, all_hub_ids, hub_texts,
        )
        rest_metrics, rest_ranked, rest_hits = evaluate_on_fold(
            model, items, restricted_hub_ids, hub_texts,
        )
        logger.info(
            "%-22s n=%3d  full=%.4f  restricted=%.4f  (+%.4f)",
            framework, len(items), full_metrics["hit_at_1"],
            rest_metrics["hit_at_1"],
            rest_metrics["hit_at_1"] - full_metrics["hit_at_1"],
        )
        folds.append({
            "framework": framework,
            "n": len(items),
            "full_pool": {
                "metrics": full_metrics,
                "hit1_indicators": [int(h) for h in full_hits],
                "ranked": full_ranked,
            },
            "restricted_pool": {
                "metrics": rest_metrics,
                "hit1_indicators": [int(h) for h in rest_hits],
                "ranked": rest_ranked,
            },
        })

    n_total = sum(f["n"] for f in folds)
    full_micro = sum(sum(f["full_pool"]["hit1_indicators"]) for f in folds) / n_total
    rest_micro = sum(sum(f["restricted_pool"]["hit1_indicators"]) for f in folds) / n_total
    drift = abs(full_micro - CAMPAIGN2_ZERO_SHOT_HIT1)

    logger.info("=" * 66)
    logger.info("full-pool zero-shot micro hit@1       %.4f  (campaign %.4f, drift %.4f)",
                full_micro, CAMPAIGN2_ZERO_SHOT_HIT1, drift)
    logger.info("AI-restricted zero-shot micro hit@1   %.4f", rest_micro)
    logger.info("free-domain-oracle gain               %+.4f", rest_micro - full_micro)
    logger.info("trained model (campaign, full pool)   0.5918")
    logger.info("=" * 66)

    output_dir = PHASE1B_RESULTS_DIR / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json({
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_REVISION,
        "n_items": n_total,
        "n_hubs_full": len(all_hub_ids),
        "n_hubs_restricted": len(restricted_hub_ids),
        "restricted_hub_ids": restricted_hub_ids,
        "full_pool_micro_hit1": full_micro,
        "restricted_pool_micro_hit1": rest_micro,
        "domain_oracle_gain": rest_micro - full_micro,
        "campaign2_zero_shot_micro_hit1": CAMPAIGN2_ZERO_SHOT_HIT1,
        "campaign2_trained_micro_hit1": 0.5918,
        "baseline_drift": drift,
        "baseline_reproduced": bool(drift <= BASELINE_TOLERANCE),
        "elapsed_s": time.time() - start,
        "folds": folds,
    }, output_dir / "probe_result.json")
    logger.info("Wrote %s", output_dir / "probe_result.json")

    if drift > BASELINE_TOLERANCE:
        message = (
            f"Full-pool zero-shot is {full_micro:.4f}, off the campaign's "
            f"{CAMPAIGN2_ZERO_SHOT_HIT1:.4f} by {drift:.4f} (tolerance "
            f"{BASELINE_TOLERANCE}). The reconstruction does not match the run "
            "it is compared against, so the restricted figure is not "
            "interpretable."
        )
        if args.require_baseline:
            raise ValueError(message)
        logger.error("%s Reporting anyway per --no-require-baseline.", message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
