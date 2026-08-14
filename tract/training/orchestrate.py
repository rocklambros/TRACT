"""Multi-fold, multi-config experiment runner.

Orchestrates the full Phase 1B pipeline:
1. Load and filter training data
2. For each LOFO fold:
   a. Build firewalled hub texts
   b. Generate training pairs with hard negatives
   c. Train LoRA model
   d. Evaluate on held-out framework
3. Aggregate across folds with fold-stratified bootstrap CIs
"""
from __future__ import annotations

import hashlib
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.phase0.common import (
    AI_FRAMEWORK_NAMES,
    EvalItem,
    build_evaluation_corpus,
    load_curated_links,
)
from tract.config import (
    MAX_ANCHOR_CHARS,
    PHASE1B_GATE_HIT1_DELTA,
    PHASE1B_RESULTS_DIR,
    PROCESSED_DIR,
)
from tract.hierarchy import CREHierarchy
from tract.io import atomic_write_json, load_json
from tract.training.config import TrainingConfig
from tract.training.data import (
    build_training_pairs,
    pairs_to_dataset,
)
from tract.training.data_quality import (
    CURATED_PATH,
    load_and_filter_curated_links,
)
from tract.training.evaluate import (
    evaluate_on_fold,
    fold_stratified_bootstrap_ci,
    paired_bootstrap_delta,
)
from tract.stopwords import STOPWORDS_PATH, load_stopwords
from tract.text_selection import (
    ProseIndex,
    SelectionStats,
    TextSelection,
    apply_prose_to_corpus,
)
from tract.training.firewall import assert_firewall, build_all_hub_texts
from tract.training.loop import save_checkpoint, train_model
from tract.training.data_quality import TieredLink

logger = logging.getLogger(__name__)

# Written per fold, and the only artifact carrying the per-item hit@1
# indicators needed to micro-average across folds.
FOLD_RESULT_FILENAME = "fold_result.json"


def _free_gpu_memory() -> None:
    """Release the baseline model's VRAM before training allocates its own."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _artifact_sha256(path: Path) -> str | None:
    """Hash an input artifact, or None if the arm did not read it."""
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_single_fold(
    config: TrainingConfig,
    held_out_framework: str,
    tiered_links: list[TieredLink],
    hierarchy: CREHierarchy,
    eval_items: list[EvalItem],
    hub_ids: list[str],
    output_dir: Path,
    standard_sections: dict[str, list[str]] | None = None,
    include_zero_shot: bool = False,
) -> dict[str, Any]:
    """Train and evaluate one LOFO fold. Returns fold result dict.

    Args:
        include_zero_shot: Also evaluate the untrained base model on this fold,
            producing a per-item indicator array paired with the trained one.
    """
    logger.info("=== FOLD: %s ===", held_out_framework)
    fold_start = time.time()

    include_desc = config.hub_rep_format == "path+name+desc"
    descriptions = None
    if include_desc:
        desc_data = load_json(PROCESSED_DIR / "hub_descriptions_reviewed.json")
        descriptions = {
            hid: d["description"]
            for hid, d in desc_data.get("descriptions", {}).items()
        }

    include_standards = config.hub_rep_format == "path+name+standards"
    if include_standards and standard_sections is None:
        # Previously both builds below were called with an identical argument
        # list, so hub_texts == base_hub_texts, every appended slice was empty,
        # and the breach check was unreachable. Failing loudly is better than
        # silently producing a format the config asked for and never built.
        raise ValueError(
            "hub_rep_format='path+name+standards' requires standard_sections; "
            "none were supplied, so the standards format cannot be built and "
            "the firewall would have nothing to check"
        )

    # Both sides of the comparison get the same treatment. assert_firewall
    # matches control text against hub text by exact substring, so filtering one
    # and not the other would make a real leak unmatchable and the check would
    # pass on a breach.
    stopwords = load_stopwords() if config.use_stopword_filter else None
    prose_index = ProseIndex.load() if config.use_prose else None

    hub_texts = build_all_hub_texts(
        hierarchy,
        excluded_framework=held_out_framework,
        include_description=include_desc,
        descriptions=descriptions,
        include_standards=include_standards,
        standard_sections=standard_sections,
        stopwords=stopwords,
    )

    # Any augmentation makes the appended slice the leakage surface, so build the
    # plain "path | name" base to subtract from it. Descriptions are CRE-authored
    # and were previously left unfirewalled, but passing base_hub_texts=None for
    # them does not merely skip the check -- assert_firewall then derives hub
    # names from the full text, so the description becomes part of the "hub name"
    # and a control text leaked into it is skipped as a hub-name match. The base
    # must be the base format, not the format minus one augmentation.
    base_hub_texts = None
    if include_standards or include_desc:
        base_hub_texts = build_all_hub_texts(
            hierarchy,
            excluded_framework=held_out_framework,
            include_description=False,
            include_standards=False,
            stopwords=stopwords,
        )
    # Hub names come from the hierarchy, not from parsing the hub text, which
    # may have been transformed. Filtered with the same stop words so the
    # exemption compares like with like.
    hub_names = {node.name for node in hierarchy.hubs.values()}
    if stopwords:
        from tract.stopwords import filter_stopwords

        hub_names = {filter_stopwords(name, stopwords) for name in hub_names}
    assert_firewall(
        hub_texts, eval_items, held_out_framework, base_hub_texts,
        hub_names=hub_names,
    )

    pairs = build_training_pairs(
        tiered_links, hub_texts, excluded_framework=held_out_framework,
        prose_index=prose_index, stopwords=stopwords,
        description_only=config.use_description_only,
    )
    dataset = pairs_to_dataset(pairs, hierarchy, hub_texts, n_hard_negatives=config.hard_negatives)

    fold_output = output_dir / f"fold_{held_out_framework.replace(' ', '_')}"
    fold_output.mkdir(parents=True, exist_ok=True)

    # Recompute what the eval anchors actually resolved to, so the fold record
    # can state it. Cheap: dictionary lookups over at most a few hundred items.
    eval_selection = SelectionStats()
    for _item in eval_items:
        _sel = prose_index.lookup(
            _item.framework_name, _item.section_id, _item.control_text,
        ) if prose_index else None
        eval_selection.record(
            _item.framework_name,
            TextSelection(_item.control_text,
                          _sel.source if _sel else "title",
                          len(_item.control_text) >= MAX_ANCHOR_CHARS),
        )
    eval_selection.log_summary(f"Fold {held_out_framework} eval anchors")
    n_truncated = eval_selection.n_truncated

    zero_shot: dict[str, Any] | None = None
    if include_zero_shot:
        # Measure the untrained base model on the SAME eval items, hub ids and
        # firewalled hub texts, in the same process, before training touches
        # anything. paired_bootstrap_delta requires the two indicator arrays to
        # be aligned item by item; a baseline computed in a separate job cannot
        # guarantee that alignment, and an unpaired delta is what put the
        # pre-registered gate inside its own confidence interval.
        from sentence_transformers import SentenceTransformer

        logger.info("Fold %s: measuring zero-shot baseline", held_out_framework)
        base_model = SentenceTransformer(config.base_model)
        base_model.max_seq_length = config.max_seq_length
        zs_metrics, _zs_predictions, zs_hit1 = evaluate_on_fold(
            base_model, eval_items, hub_ids, hub_texts,
        )
        zero_shot = {
            "metrics": zs_metrics,
            "hit1_indicators": [int(x) for x in zs_hit1],
        }
        logger.info("Fold %s zero-shot: hit@1=%.3f", held_out_framework,
                    zs_metrics["hit_at_1"])
        del base_model
        _free_gpu_memory()

    model = train_model(config, dataset, fold_output)

    metrics, predictions, hit1_indicators = evaluate_on_fold(
        model, eval_items, hub_ids, hub_texts,
    )
    logger.info("Fold %s: hit@1=%.3f, hit@5=%.3f, MRR=%.3f, NDCG@10=%.3f",
                held_out_framework, metrics["hit_at_1"], metrics["hit_at_5"],
                metrics["mrr"], metrics["ndcg_at_10"])

    save_checkpoint(model, config, metrics, fold_output / "model", _get_git_sha())

    pred_data = []
    for item, pred in zip(eval_items, predictions):
        pred_data.append({
            "control_text": item.control_text,
            "ground_truth_hub_id": item.ground_truth_hub_id,
            "predicted_top10": pred[:10],
            "framework": item.framework_name,
        })
    atomic_write_json(pred_data, fold_output / "predictions.json")
    atomic_write_json(metrics, fold_output / "metrics.json")

    elapsed = time.time() - fold_start
    logger.info("Fold %s complete in %.1fs", held_out_framework, elapsed)

    result: dict[str, Any] = {
        "held_out_framework": held_out_framework,
        "metrics": metrics,
        "predictions": predictions,
        "hit1_indicators": hit1_indicators,
        "n_eval_items": len(eval_items),
        "n_training_pairs": len(pairs),
        "elapsed_s": elapsed,
    }
    if zero_shot is not None:
        result["zero_shot"] = zero_shot

    # Persist the per-item indicators, not just the fold's summary metrics.
    # Averaging five fold summaries is a MACRO average: it weights a 6-item fold
    # the same as a 60-item one. The aggregate hit@1 TRACT reports is a MICRO
    # average, which needs the raw indicator array from every fold. Any path that
    # only keeps metrics.json cannot reconstruct it, and the model card now
    # refuses to build without it.
    fold_record = {k: v for k, v in result.items() if k != "predictions"}
    fold_record["hit1_indicators"] = [int(x) for x in hit1_indicators]
    fold_record["git_sha"] = _get_git_sha()
    # Which arm produced this. The record previously carried none, and the arms
    # differ by a runtime flag on one commit, so the git_sha check below cannot
    # tell them apart: four folds from one arm beside one from another passed
    # every guard and aggregated into a number describing no single
    # configuration.
    fold_record["config"] = config.to_dict()
    # And how much prose the arm actually got. A run that quietly fell back to
    # titles is otherwise indistinguishable in the metrics from one that did
    # not, which is exactly how the title-only evaluation went unnoticed.
    fold_record["text_selection"] = {
        "prose_fraction": eval_selection.prose_fraction,
        "by_source": dict(eval_selection.by_source),
        "n_truncated_at_encoder_budget": n_truncated,
    }
    # And which files produced those anchors. TrainingConfig.data_hash exists
    # but nothing ever assigned it, so every record carried "data_hash": "" --
    # a field named for the guarantee CLAUDE.md requires, holding nothing. The
    # git SHA pins the code; these pin the data the code was pointed at, which
    # is the half that changes when a parser is re-run. Hashing the files
    # rather than the parsed objects means anyone with the repo can re-derive
    # them with sha256sum.
    fold_record["inputs"] = {
        "curated_links_sha256": _artifact_sha256(CURATED_PATH),
        "all_controls_sha256": (
            _artifact_sha256(PROCESSED_DIR / "all_controls.json")
            if prose_index is not None else None
        ),
        "stopwords_sha256": (
            _artifact_sha256(STOPWORDS_PATH) if stopwords is not None else None
        ),
    }
    atomic_write_json(fold_record, fold_output / FOLD_RESULT_FILENAME)

    return result


def load_fold_results(
    results_dir: Path,
    expected_frameworks: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Load every persisted per-fold record under results_dir.

    Sorted by held-out framework so aggregation is order-independent and
    therefore reproducible regardless of the order folds finished in.

    Args:
        expected_frameworks: The exact set of folds this run must contain.
            Defaults to AI_FRAMEWORK_NAMES. A missing fold is not a smaller
            result, it is a different experiment: "leave one framework out"
            over four frameworks is not the claim being published.
    """
    expected = set(
        AI_FRAMEWORK_NAMES if expected_frameworks is None else expected_frameworks
    )
    records = []
    for path in sorted(results_dir.glob(f"fold_*/{FOLD_RESULT_FILENAME}")):
        record = load_json(path)
        missing = {"held_out_framework", "hit1_indicators", "n_eval_items"} - set(record)
        if missing:
            raise ValueError(
                f"{path} is missing {sorted(missing)}. It was written by a version "
                "that dropped the per-item indicators, so the aggregate hit@1 "
                "cannot be micro-averaged from it."
            )
        if len(record["hit1_indicators"]) != record["n_eval_items"]:
            raise ValueError(
                f"{path}: {len(record['hit1_indicators'])} indicators for "
                f"{record['n_eval_items']} eval items. The fold record is "
                "internally inconsistent; refusing to aggregate it."
            )
        zero_shot = record.get("zero_shot") or {}
        zs_indicators = zero_shot.get("hit1_indicators")
        if zs_indicators is not None and len(zs_indicators) != record["n_eval_items"]:
            # Equal length is what makes the delta paired. A baseline array of
            # the wrong length cannot be aligned item-for-item with the trained
            # one, and a mis-paired delta gets a paired interval it has not
            # earned.
            raise ValueError(
                f"{path}: {len(zs_indicators)} zero-shot indicators for "
                f"{record['n_eval_items']} eval items. The baseline is not "
                "paired with the trained run; refusing to aggregate it."
            )
        records.append(record)

    if not records:
        raise ValueError(
            f"No {FOLD_RESULT_FILENAME} files under {results_dir}. Nothing to "
            "aggregate: either no fold completed or the results were not collected."
        )

    found = {r["held_out_framework"] for r in records}
    if len(found) != len(records):
        raise ValueError(
            f"Duplicate folds under {results_dir}: {len(records)} records for "
            f"{len(found)} frameworks."
        )
    if found != expected:
        # Averaging whatever came back is how a four-fold number gets published
        # as a five-fold LOFO result, with the pods that held the missing fold
        # already terminated.
        raise ValueError(
            f"Fold set mismatch under {results_dir}. Missing: "
            f"{sorted(expected - found) or 'none'}; unexpected: "
            f"{sorted(found - expected) or 'none'}. Refusing to aggregate a "
            "partial cross-validation into a headline number."
        )

    arms = {
        (
            r.get("config", {}).get("use_prose"),
            r.get("config", {}).get("use_stopword_filter"),
        )
        for r in records
    }
    if len(arms) > 1:
        raise ValueError(
            f"Folds under {results_dir} come from different arms: "
            f"(use_prose, use_stopword_filter) = {sorted(arms)}. The arms "
            "differ by a runtime flag on one commit, so the git_sha check "
            "below cannot separate them. Aggregating them would produce a "
            "number describing no single configuration."
        )

    # Same argument as the arm check, one layer down: the arms differ by a
    # runtime flag and the input data differs by a parser re-run, and neither
    # moves the git SHA. A fold trained before a parser was fixed and one
    # trained after are two experiments wearing the same commit.
    input_sets = {
        tuple(sorted((r.get("inputs") or {}).items())) for r in records
    }
    if len(input_sets) > 1:
        differing = sorted({
            key
            for key in {k for s in input_sets for k, _ in s}
            if len({dict(s).get(key) for s in input_sets}) > 1
        })
        raise ValueError(
            f"Folds under {results_dir} were built from different input data: "
            f"{differing} differ across folds. Re-run every fold against one "
            "snapshot, or aggregate the runs separately."
        )
    if input_sets == {()}:
        logger.warning(
            "No fold records an 'inputs' block; the aggregate cannot be tied "
            "to the data snapshot that produced it."
        )

    shas = {r.get("git_sha", "unknown") for r in records}
    if len(shas) > 1:
        raise ValueError(
            f"Folds under {results_dir} were produced by different code: "
            f"git_sha {sorted(shas)}. This is the signature of a stale "
            f"{FOLD_RESULT_FILENAME} left from an earlier run. Clear the "
            "directory and re-run, or aggregate the runs separately."
        )
    if shas == {"unknown"}:
        logger.warning(
            "Every fold records git_sha='unknown'; the aggregate cannot be tied "
            "to a commit."
        )
    return records


def aggregate_fold_results(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Micro-average hit@1 across folds with a fold-stratified bootstrap CI.

    Pools per-item indicators rather than averaging per-fold rates, so each
    eval item carries equal weight. With TRACT's fold sizes the difference is
    not cosmetic: the smallest fold would otherwise count for 1/5 of the
    headline number instead of its share of the items.
    """
    fold_hit1s = [np.asarray(r["hit1_indicators"], dtype=float) for r in fold_results]
    aggregate: dict[str, Any] = dict(fold_stratified_bootstrap_ci(fold_hit1s))

    # Report the macro figure alongside it. They differ only through fold-size
    # imbalance, so a wide gap is a signal about the folds, not a second result.
    macro = float(np.mean([float(np.mean(f)) for f in fold_hit1s]))
    aggregate["macro_mean"] = macro
    aggregate["n_folds"] = len(fold_hit1s)
    aggregate["fold_sizes"] = {
        r["held_out_framework"]: int(r["n_eval_items"]) for r in fold_results
    }
    logger.info(
        "AGGREGATE hit@1 (micro): %.4f [%.4f, %.4f] over n=%d | macro: %.4f",
        aggregate["mean"], aggregate["ci_low"], aggregate["ci_high"],
        aggregate["n_total"], macro,
    )
    return aggregate


def gate_decision(
    fold_results: list[dict[str, Any]],
    threshold: float = PHASE1B_GATE_HIT1_DELTA,
) -> dict[str, Any]:
    """Evaluate Gate 1 against the paired zero-shot baseline.

    PRD Section 6.4 pre-registers the criterion as "Micro-averaged
    (sample-weighted) hit@1 delta > 0.10 over zero-shot baseline", with
    per-fold delta, macro average and worst-fold delta reported as diagnostics,
    and any fold whose delta is negative flagged for investigation.

    Two verdicts are computed and BOTH are returned. They are never merged:

    - ``point_estimate_pass`` is the pre-registered criterion exactly as
      written: does the micro delta exceed the threshold.
    - ``ci_low_pass`` applies the same threshold to the lower bound of the
      paired bootstrap confidence interval on that delta.

    The second is strictly the harder test. A point estimate above the
    threshold whose interval still contains it means the eval set cannot
    distinguish the model from the gate, which is a fact about the evidence
    rather than a different metric. ``verdicts_agree`` says whether the choice
    between them matters for this run. Substituting one for the other is a
    decision for the owner of the pre-registration, not for this function, so
    both are reported and neither is dropped.

    The baseline must be the paired one recorded per fold by run_single_fold
    with include_zero_shot=True: same items, same hub ids, same firewalled hub
    texts. Pairing cancels item-level difficulty, and an unpaired delta on this
    eval set produces an interval wide enough to swallow the threshold.
    """
    missing = [
        r["held_out_framework"] for r in fold_results
        if not r.get("zero_shot", {}).get("hit1_indicators")
    ]
    if missing:
        raise ValueError(
            f"Folds {sorted(missing)} carry no paired zero-shot indicators, so "
            "the delta cannot be computed against the baseline they were "
            "measured beside. Re-run them with include_zero_shot=True; do not "
            "substitute a baseline figure from another run."
        )

    trained = [np.asarray(r["hit1_indicators"], dtype=float) for r in fold_results]
    baseline = [
        np.asarray(r["zero_shot"]["hit1_indicators"], dtype=float)
        for r in fold_results
    ]
    # paired_bootstrap_delta reports B - A, so the baseline is A.
    paired = paired_bootstrap_delta(baseline, trained)

    per_fold = {}
    for record, tr, bl in zip(fold_results, trained, baseline):
        per_fold[record["held_out_framework"]] = {
            "delta": float(np.mean(tr) - np.mean(bl)),
            "trained_hit1": float(np.mean(tr)),
            "zero_shot_hit1": float(np.mean(bl)),
            "n_eval_items": int(len(tr)),
        }

    deltas = [v["delta"] for v in per_fold.values()]
    worst_framework = min(per_fold, key=lambda k: per_fold[k]["delta"])
    negative_folds = sorted(k for k, v in per_fold.items() if v["delta"] < 0)

    point_estimate_pass = bool(paired["delta_mean"] > threshold)
    ci_low_pass = bool(paired["ci_low"] > threshold)

    decision = {
        "threshold": threshold,
        "micro_delta": paired["delta_mean"],
        "ci_low": paired["ci_low"],
        "ci_high": paired["ci_high"],
        "p_value": paired["p_value"],
        "paired": True,
        "point_estimate_pass": point_estimate_pass,
        "ci_low_pass": ci_low_pass,
        "verdicts_agree": point_estimate_pass == ci_low_pass,
        "macro_delta": float(np.mean(deltas)),
        "worst_fold": worst_framework,
        "worst_fold_delta": per_fold[worst_framework]["delta"],
        "negative_folds": negative_folds,
        "per_fold": per_fold,
        "n_total": int(sum(len(t) for t in trained)),
    }

    logger.info("=" * 62)
    logger.info("GATE 1  (PRD 6.4: micro hit@1 delta > %.2f over zero-shot)", threshold)
    logger.info("  micro delta      : %+.4f  [%+.4f, %+.4f]  (paired, n=%d)",
                paired["delta_mean"], paired["ci_low"], paired["ci_high"],
                decision["n_total"])
    logger.info("  macro delta      : %+.4f", decision["macro_delta"])
    logger.info("  worst fold       : %s %+.4f",
                worst_framework, decision["worst_fold_delta"])
    logger.info("  pre-registered   : %s  (delta %.4f %s %.2f)",
                "PASS" if point_estimate_pass else "FAIL",
                paired["delta_mean"], ">" if point_estimate_pass else "<=", threshold)
    logger.info("  CI lower bound   : %s  (ci_low %.4f %s %.2f)",
                "PASS" if ci_low_pass else "FAIL",
                paired["ci_low"], ">" if ci_low_pass else "<=", threshold)
    if not decision["verdicts_agree"]:
        logger.warning(
            "  VERDICTS DISAGREE: the point estimate clears the gate but its "
            "confidence interval contains the threshold. The eval set cannot "
            "separate this model from the gate. Owner decision."
        )
    if negative_folds:
        logger.warning("  FOLDS BELOW ZERO-SHOT: %s -- flagged per PRD 6.4",
                       negative_folds)
    logger.info("=" * 62)
    return decision


def run_experiment(
    config: TrainingConfig,
    include_zero_shot: bool = True,
) -> dict[str, Any]:
    """Run a full LOFO experiment with the given config.

    Args:
        include_zero_shot: Measure the paired zero-shot baseline on each fold.
            Required for the Gate 1 delta; on by default so a gate decision is
            never assembled from a baseline measured somewhere else.
    """
    logger.info("Starting experiment: %s", config.name)
    exp_start = time.time()

    tiered_links, raw_hash = load_and_filter_curated_links()

    hierarchy = CREHierarchy.model_validate(load_json(PROCESSED_DIR / "cre_hierarchy.json"))
    hub_ids = sorted(hierarchy.hubs.keys())

    links = load_curated_links()
    # Corpus identity is fixed from titles, then anchors are swapped. See
    # apply_prose_to_corpus: building per arm lets the anchor change the item
    # count, which breaks the paired delta.
    corpus = build_evaluation_corpus(links, AI_FRAMEWORK_NAMES, {})
    corpus = apply_prose_to_corpus(
        corpus,
        ProseIndex.load() if config.use_prose else None,
        load_stopwords() if config.use_stopword_filter else None,
        description_only=config.use_description_only,
    )

    eval_by_fw: dict[str, list[EvalItem]] = {}
    for item in corpus:
        eval_by_fw.setdefault(item.framework_name, []).append(item)

    output_dir = PHASE1B_RESULTS_DIR / config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_results: list[dict[str, Any]] = []
    for fw_name in sorted(AI_FRAMEWORK_NAMES):
        fw_items = eval_by_fw.get(fw_name, [])
        if not fw_items:
            logger.warning("No eval items for %s, skipping", fw_name)
            continue

        result = run_single_fold(
            config=config,
            held_out_framework=fw_name,
            tiered_links=tiered_links,
            hierarchy=hierarchy,
            eval_items=fw_items,
            hub_ids=hub_ids,
            output_dir=output_dir,
            include_zero_shot=include_zero_shot,
        )
        fold_results.append(result)

    experiment_result = {
        "config": config.to_dict(),
        "aggregate_hit1": aggregate_fold_results(fold_results),
        "per_fold": {r["held_out_framework"]: r["metrics"] for r in fold_results},
        "raw_hash": raw_hash,
        "git_sha": _get_git_sha(),
        "total_elapsed_s": time.time() - exp_start,
    }
    if include_zero_shot:
        experiment_result["gate"] = gate_decision(fold_results)
    atomic_write_json(experiment_result, output_dir / "aggregate_metrics.json")

    return experiment_result
