"""Run one LOFO fold. Entrypoint for the RunPod per-pod invocation.

Usage:
    python -m scripts.phase1b.run_fold --framework "MITRE ATLAS"
    python -m scripts.phase1b.run_fold --framework "MITRE ATLAS" --zero-shot

Exists as a module rather than an inline `python -c` string. The orchestrator
used to build roughly twenty-five statements of Python inside an f-string
inside a shell command inside SSH, with four levels of quote escaping. That is
unreviewable, unrunnable locally, and untestable, and it silently dropped the
per-item hit@1 indicators the aggregate confidence interval is computed from.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scripts.phase0.common import (
    AI_FRAMEWORK_NAMES,
    build_evaluation_corpus,
    load_curated_links,
)
from tract.config import (
    LOFO_WANDB_ENTITY,
    LOFO_WANDB_PROJECT,
    PROCESSED_DIR,
)
from tract.hierarchy import CREHierarchy
from tract.io import load_json
from tract.stopwords import load_stopwords
from tract.text_selection import (
    ProseIndex,
    SelectionStats,
    apply_prose_to_corpus,
)
from tract.training.config import TrainingConfig
from tract.training.data_quality import load_and_filter_curated_links
from tract.training.orchestrate import FOLD_RESULT_FILENAME, run_single_fold
from tract.training.tracking import finish_run, init_run, log_fold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _arm_label(config: TrainingConfig) -> str:
    """Short, stable name for the text-selection arm.

    Derived from the flags rather than passed in, so the label cannot drift
    from the configuration it names.
    """
    if not config.use_prose:
        return "title-only"
    parts = ["prose"]
    if config.use_description_only:
        parts.append("desconly")
    if config.use_stopword_filter:
        parts.append("stopwords")
    return "-".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single LOFO fold")
    parser.add_argument("--framework", required=True,
                        help="Framework to hold out for this fold")
    parser.add_argument("--config-name", default="phase1b_primary",
                        help="Experiment name; also the results subdirectory")
    parser.add_argument("--output-dir", default=None,
                        help="Defaults to results/phase1b/<config-name>")
    parser.add_argument("--no-prose", action="store_true",
                        help="Anchor on section titles instead of full control "
                             "text. The pre-2026-08 behaviour; baseline arm.")
    parser.add_argument("--stopwords", action="store_true",
                        help="Filter corpus-derived boilerplate from control and "
                             "hub text. Ablation arm.")
    parser.add_argument("--description-only", action="store_true",
                        help="Cut each control at its first remediation heading. "
                             "The 512-token budget is fixed by the architecture; "
                             "this changes which tokens it spends.")
    parser.add_argument("--zero-shot", action="store_true",
                        help="Also evaluate the untrained base model on this "
                             "fold, paired item-for-item with the trained one")
    parser.add_argument("--wandb", action="store_true",
                        help="Track this fold from inside the run. Off by "
                             "default: pods carry no credentials, and the "
                             "orchestrator logs every collected fold from the "
                             "operator's machine instead (runpod_parallel "
                             "track). Use this only for a local or manual run "
                             "where a key is already present. It fails closed "
                             "-- an unusable key raises rather than silently "
                             "running untracked.")
    parser.add_argument("--wandb-project", default=LOFO_WANDB_PROJECT,
                        help="WandB project for this campaign")
    args = parser.parse_args()

    if args.framework not in AI_FRAMEWORK_NAMES:
        # A typo here would otherwise hold out nothing, train on everything and
        # report an inflated score against an empty eval set.
        raise ValueError(
            f"Unknown framework {args.framework!r}. "
            f"Expected one of: {sorted(AI_FRAMEWORK_NAMES)}"
        )

    config = TrainingConfig(
        name=args.config_name,
        use_prose=not args.no_prose,
        use_stopword_filter=args.stopwords,
        use_description_only=args.description_only,
    )
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else Path("results") / "phase1b" / args.config_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    tiered_links, raw_hash = load_and_filter_curated_links()
    hierarchy = CREHierarchy.model_validate(
        load_json(PROCESSED_DIR / "cre_hierarchy.json")
    )
    hub_ids = sorted(hierarchy.hubs.keys())

    # Supply the prose. build_evaluation_corpus already prefers a control's full
    # text over its section title when given this mapping; every Phase 1B caller
    # passed an empty one, so the evaluation measured three-word titles while
    # `tract assign` is handed paragraphs.
    # Build the corpus from titles first: that fixes the item set, the ground
    # truth and the ordering. Only then swap in prose. Building it per arm would
    # let the anchor decide how many items exist, because the corpus
    # de-duplicates on control text, and arms with different item sets cannot be
    # compared with a paired test.
    links = load_curated_links()
    corpus = build_evaluation_corpus(links, AI_FRAMEWORK_NAMES, {})
    selection_stats = SelectionStats()
    corpus = apply_prose_to_corpus(
        corpus,
        ProseIndex.load() if config.use_prose else None,
        load_stopwords() if config.use_stopword_filter else None,
        stats=selection_stats,
        description_only=config.use_description_only,
    )
    selection_stats.log_summary("Eval items")
    eval_items = [i for i in corpus if i.framework_name == args.framework]
    if not eval_items:
        raise ValueError(
            f"No eval items for {args.framework!r}. The fold would score "
            "against an empty set, which is not a result."
        )

    logger.info("Fold %s: %d eval items, raw_hash=%s",
                args.framework, len(eval_items), raw_hash)

    # The arm is a runtime flag on one commit, so it has to be in the run name
    # and the tags. Two arms landing as indistinguishable runs is the same
    # failure the fold-record arm check exists to prevent, one layer up.
    arm = _arm_label(config)
    run = None
    if args.wandb:
        run = init_run(
            project=args.wandb_project,
            entity=LOFO_WANDB_ENTITY,
            name=f"{arm}/{args.framework}",
            config={
                **config.to_dict(),
                "held_out_framework": args.framework,
                "arm": arm,
                "n_eval_items": len(eval_items),
                "curated_links_hash": raw_hash,
                "eval_prose_fraction": selection_stats.prose_fraction,
            },
            tags=[arm, args.framework, "lofo"],
        )

    exit_code = 0
    try:
        result = run_single_fold(
            config=config,
            held_out_framework=args.framework,
            tiered_links=tiered_links,
            hierarchy=hierarchy,
            eval_items=eval_items,
            hub_ids=hub_ids,
            output_dir=output_dir,
            include_zero_shot=args.zero_shot,
        )
    except BaseException:
        # Mark the run failed rather than leaving it displayed as running.
        # A pod that dies mid-fold is the case the aggregate must not mistake
        # for a fold still in progress.
        exit_code = 1
        finish_run(run, exit_code=1)
        raise

    fold_dir = output_dir / f"fold_{args.framework.replace(' ', '_')}"
    # Log the persisted record, not the in-memory result: the record is what
    # aggregation reads, so tracking and aggregation cannot disagree.
    log_fold(run, load_json(fold_dir / FOLD_RESULT_FILENAME))
    finish_run(run, exit_code=exit_code)

    logger.info("FOLD COMPLETE: %s hit@1=%.4f -> %s",
                args.framework, result["metrics"]["hit_at_1"],
                fold_dir / FOLD_RESULT_FILENAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
