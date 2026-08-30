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
    FOLD_RESULT_FILENAME,
    PHASE1B_BASE_MODEL,
    PHASE1B_MAX_SEQ_LENGTH,
    max_anchor_chars,
    LOFO_WANDB_ENTITY,
    LOFO_WANDB_PROJECT,
    PROCESSED_DIR,
)
from tract.hierarchy import CREHierarchy
from tract.io import load_json
from tract.framework_identity import filter_set
from tract.text_selection import (
    ProseIndex,
    SelectionStats,
    apply_prose_to_corpus,
)
from tract.training.config import TrainingConfig
from tract.training.data_quality import (
    assert_corpus_matches_training_links,
    load_and_filter_curated_links,
)
from tract.training.orchestrate import run_single_fold
from tract.training.tracking import (
    finish_run,
    init_run,
    log_fold,
    stable_run_id,
)

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
    if config.use_framework_identity_filter:
        parts.append("fwid")
    return "-".join(parts)


def validation_frameworks() -> set[str]:
    """Every framework that is NOT part of the pre-registered AI test set.

    Selection has to happen somewhere other than the set that reports the
    number. These 15 frameworks carry 3,932 of the 4,127 curated links and
    yield 1,579 eval items against the test set's 147, which moves the
    minimum detectable effect from 11.4 hit@1 points to 3.5 -- the difference
    between an instrument that can rank the arms and one that cannot.
    """
    links = load_curated_links()
    everything = {
        link.standard_name for link in links if link.standard_name
    }
    return everything - set(AI_FRAMEWORK_NAMES)


def _campaign_label(config: TrainingConfig) -> str:
    """Arm label plus the dimensions that also define a configuration.

    The anchor arm alone stopped being unique once the encoder and the branch
    balance became variables: without these, a rebalanced run and an
    unbalanced one land under one name and aggregate together.
    """
    label = _arm_label(config)
    if config.branch_balance_temperature:
        label += f"-bal{config.branch_balance_temperature:g}"
    if config.base_model != PHASE1B_BASE_MODEL:
        label += "-" + config.base_model.split("/")[-1]
    if config.max_seq_length != PHASE1B_MAX_SEQ_LENGTH:
        label += f"-seq{config.max_seq_length}"
    if config.hub_rep_format != "path+name":
        label += "-" + config.hub_rep_format.replace("path+name+", "")
    return label


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
    parser.add_argument("--framework-identity", action="store_true",
                        help="Strip the acronyms that name a framework "
                             "(OWASP, CWE, CAPEC, CCM) from control and hub "
                             "text. Ablation arm: a bi-encoder that reads the "
                             "publisher can answer from it instead of the "
                             "mapping.")
    parser.add_argument("--description-only", action="store_true",
                        help="Cut each control at its first remediation heading. "
                             "The 512-token budget is fixed by the architecture; "
                             "this changes which tokens it spends.")
    parser.add_argument("--zero-shot", action="store_true",
                        help="Also evaluate the untrained base model on this "
                             "fold, paired item-for-item with the trained one")
    parser.add_argument("--base-model", default=None,
                        help="Encoder to fine-tune. Defaults to the pinned "
                             "BGE-large (512 tokens). A long-context model "
                             "also needs --max-seq-length or the character "
                             "budget derived from 512 binds first.")
    parser.add_argument("--max-seq-length", type=int, default=None,
                        help="Encoder token budget. The anchor character cut "
                             "is derived from it.")
    parser.add_argument("--split", choices=("test", "validation"),
                        default="test",
                        help="test: LOFO over the 5 AI frameworks, the "
                             "pre-registered 147-item set PRD 6.4 reports. "
                             "validation: LOFO over the 15 non-AI frameworks, "
                             "1,579 items. Arm selection belongs on validation "
                             "-- selecting on the test set is what makes a "
                             "16-arm winner selection-optimistic, and at n=147 "
                             "the minimum detectable effect is 11.4 hit@1 "
                             "points against effects of 1-3.")
    parser.add_argument("--hub-rep", default=None,
                        choices=("path+name", "path+name+desc",
                                 "path+name+standards"),
                        help="What the model matches AGAINST. The default is a "
                             "bare label -- for many hubs the name is the last "
                             "path segment repeated -- so there is nothing on "
                             "the target side to comprehend. PRD:372 requires "
                             "path+name+desc be measured; it never has been, "
                             "because this flag did not exist on the RunPod "
                             "path and CLAUDE.md forbids the local one.")
    parser.add_argument("--branch-balance", type=float, default=None,
                        help="Temperature flattening the CRE-branch "
                             "distribution during batch ordering. 0 disables "
                             "it; ~3 pulls the 3.3%% threat branch up toward "
                             "the 72.1%% controls branch.")
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

    # Before the framework name is even validated, because this is the check
    # that decides whether the run means anything. A clone without the
    # gitignored overlay trains on 4,019 of the 4,389 links -- 370 belong to
    # the four overlay frameworks -- and reports the same figures in the same
    # shape. The refusal existed for eight days and nothing called it; it was
    # a checklist row in the Jetson briefing and a set of its own tests.
    #
    # The orchestrator refuses earlier, in provision, which is where the money
    # is saved. This is the half that holds for a fold launched by hand, by a
    # resumed fleet, or by a caller that does not go through runpod_parallel.
    corpus_digest = assert_corpus_matches_training_links()
    logger.info("Corpus digest %s matches the training links.", corpus_digest[:12])

    # The eval population follows the split. A typo would otherwise hold out
    # nothing, train on everything and report an inflated score against an
    # empty eval set, so the name is checked against the split's own roster.
    eval_frameworks = (
        set(AI_FRAMEWORK_NAMES) if args.split == "test"
        else validation_frameworks()
    )
    if args.framework not in eval_frameworks:
        raise ValueError(
            f"Unknown framework {args.framework!r} for split "
            f"{args.split!r}. Expected one of: {sorted(eval_frameworks)}"
        )

    config = TrainingConfig(
        name=args.config_name,
        use_prose=not args.no_prose,
        use_stopword_filter=args.stopwords,
        use_framework_identity_filter=args.framework_identity,
        use_description_only=args.description_only,
        **({"base_model": args.base_model} if args.base_model else {}),
        **({"max_seq_length": args.max_seq_length} if args.max_seq_length else {}),
        **({"branch_balance_temperature": args.branch_balance}
           if args.branch_balance is not None else {}),
        **({"hub_rep_format": args.hub_rep} if args.hub_rep else {}),
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
    corpus = build_evaluation_corpus(links, eval_frameworks, {})
    selection_stats = SelectionStats()
    corpus = apply_prose_to_corpus(
        corpus,
        ProseIndex.load() if config.use_prose else None,
        filter_set(
            use_stopwords=config.use_stopword_filter,
            use_framework_identity=config.use_framework_identity_filter,
        ),
        stats=selection_stats,
        description_only=config.use_description_only,
        max_chars=max_anchor_chars(config.max_seq_length),
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
    arm = _campaign_label(config)
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
            # Same key the orchestrator uses, so a pod-side run and a
            # later `track` of the same fold are one run, not two.
            run_id=stable_run_id(args.config_name, arm, args.framework),
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
            # The stats apply_prose_to_corpus populated above. Without them the
            # fold falls back to a length heuristic that undercounts truncation
            # by 26% -- 39 reported against 55 real across Campaign 2's test
            # round -- because prepare_anchor rstrips after cutting.
            corpus_selection=selection_stats,
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
