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

import logging
import math
import os
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from scripts.phase0.common import (
    AI_FRAMEWORK_NAMES,
    EvalItem,
    build_evaluation_corpus,
    load_curated_links,
)
from tract.config import (
    FOLD_RESULT_FILENAME,
    max_anchor_chars,
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
    fold_input_digests,
    load_and_filter_curated_links,
)
from tract.training.evaluate import (
    evaluate_on_fold,
    fold_stratified_bootstrap_ci,
    paired_bootstrap_delta,
)
from tract.framework_identity import (
    assert_identity_symmetry,
    filter_set,
    load_framework_corpora,
    load_hub_vocabulary,
)
from tract.text_selection import (
    ProseIndex,
    SelectionStats,
    TextSelection,
    apply_prose_to_corpus,
    merged_corpus_path,
)
from tract.training.firewall import assert_firewall, build_all_hub_texts
from tract.training.loop import save_checkpoint, train_model
from tract.training.data_quality import TieredLink

logger = logging.getLogger(__name__)

# FOLD_RESULT_FILENAME moved to tract.config; this module imports it above and
# uses it below. Import it FROM tract.config, not from here -- mypy --strict
# rejects the implicit re-export, and the RunPod orchestrator needs the name
# without paying for this module's torch and datasets imports.


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


# The flags that make one arm different from another. load_fold_results
# refuses to aggregate across them.
ARM_DEFINING_KEYS: tuple[str, ...] = (
    "use_prose",
    "use_stopword_filter",
    # Whether the anchors still say "OWASP". A run that scrubbed framework
    # acronyms and one that did not are answering different questions, and
    # averaging them reports a number describing neither.
    "use_framework_identity_filter",
    "use_description_only",
    # Not anchor arms, but they define a configuration just as completely. A
    # rebalanced run and an unbalanced one, or two different encoders, must
    # never aggregate into one number. This list started with two of the four
    # anchor flags and let prose and prose-desconly merge; the failure mode is
    # silent and the result describes no configuration that was actually run.
    "branch_balance_temperature",
    "base_model",
    "max_seq_length",
    # What the model matches AGAINST. Omitted for three months, so a run with
    # semantic hub descriptions would have averaged with one matching a bare
    # label -- the two most different experiments this project can run.
    "hub_rep_format",
)

# hit@1 is an indicator: for each eval item the top-ranked hub either was a
# valid answer or it was not. evaluate_on_fold writes 1.0 or 0.0 and nothing
# else, so any other value in the array did not come from a measurement.
INDICATOR_DOMAIN: Final[frozenset[float]] = frozenset({0.0, 1.0})

# How many distinct offending values an error quotes before it stops counting.
# Enough to tell one stray entry from a wholly fabricated array, few enough
# that a 1,400-item fold does not print a wall of numbers into a CI log.
MAX_REPORTED_OFFENDING_VALUES: Final[int] = 8

# metrics.hit_at_1 and mean(hit1_indicators) are the same integer count over
# the same denominator, computed in two modules (score_predictions in
# scripts.phase0.common, and here), so on every one of the 32 committed fold
# records they agree exactly. This tolerance is for float summation order and
# nothing else: one flipped indicator in the largest committed fold (CAPEC,
# 349 items) moves the mean by 2.9e-3, six orders of magnitude above it, and
# is meant to be caught.
INDICATOR_METRIC_TOLERANCE: Final[float] = 1e-9

# A hit rate is a fraction of items, and a difference of two hit rates
# measured on the same items cannot leave [-1, 1]. Named because three
# separate layers check them and must not drift apart.
HIT_RATE_BOUNDS: Final[tuple[float, float]] = (0.0, 1.0)
HIT_RATE_DELTA_BOUNDS: Final[tuple[float, float]] = (-1.0, 1.0)


def _get_git_sha() -> str:
    """Short SHA of the code that produced this fold.

    Prefers TRACT_GIT_SHA. The orchestrator rsyncs the working tree to each
    pod with --exclude=.git, so `git rev-parse` there has no repository to
    read and returned "unknown" for every fold of every RunPod run. That made
    the whole fleet agree on "unknown", which load_fold_results treats as a
    warning rather than a mismatch, so the stale-fold check was dead on the
    only path that spends money. The orchestrator knows the SHA of the tree it
    shipped and passes it in.
    """
    injected = os.environ.get("TRACT_GIT_SHA", "").strip()
    if injected:
        return injected
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"






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
    stopwords = filter_set(
        use_stopwords=config.use_stopword_filter,
        use_framework_identity=config.use_framework_identity_filter,
    )
    # Checked against the corpus THIS fold reads, not the one the artifact was
    # built from. A checkout holding the licensed overlay trains on frameworks
    # the committed identity set never saw, and a set that scrubs "OWASP" while
    # leaving "ETSI" is the original defect wearing a different name. Skipped
    # when nothing is filtered, so the default path pays nothing.
    if stopwords is not None:
        assert_identity_symmetry(
            stopwords,
            load_framework_corpora(merged_corpus_path()),
            load_hub_vocabulary(),
        )
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
        max_chars=max_anchor_chars(config.max_seq_length),
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
        base_model = SentenceTransformer(
            config.base_model, revision=config.base_model_revision,
        )
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
    fold_record["inputs"] = fold_input_digests(
        with_prose=prose_index is not None,
        with_stopwords=config.use_stopword_filter,
        with_framework_identity=config.use_framework_identity_filter,
    )
    atomic_write_json(fold_record, fold_output / FOLD_RESULT_FILENAME)

    return result


def lexical_overlap_diagnostic(
    results_dir: Path, hierarchy: CREHierarchy,
) -> dict[str, Any]:
    """Split hit@1 by whether the anchor already contains the answer.

    The task is semantic mapping, not string retrieval, and an eval item whose
    anchor lexically covers its own ground-truth hub name is not testing the
    former. On the AI test set 26 of 147 titles are character-identical to
    their hub name, and the title arm scores 0.923 on exactly those against
    0.654 for prose -- 79% of that arm's apparent lead. Reporting one pooled
    number lets a lexical shortcut masquerade as comprehension, so both are
    reported and the non-echo figure is the one that answers the question the
    project is actually asking.

    Reads predictions.json, which carries the anchor text the model saw. The
    split is therefore computed PER ARM, against that arm's own anchors: it
    answers "how much of this arm's score is lexical" and not "how do two arms
    compare on a fixed subset". The subsets differ (32 of 147 under titles, 38
    under prose, because prose can also contain the hub's words), so a
    cross-arm comparison needs one arm's partition applied to all of them --
    which is a different question and belongs to analysis, not to the
    per-run record.
    """
    function_words = {
        "the", "a", "an", "of", "for", "to", "in", "and", "or", "with", "by",
        "on", "at", "is", "are", "be", "as", "that", "this", "from", "its",
        "it", "their", "which", "when", "where", "all", "any", "not", "no",
        "if", "then", "than", "into", "via", "using", "use", "used",
    }

    def content(text: str) -> set[str]:
        from tract.stopwords import tokenize

        return {t.lower() for t in tokenize(text)} - function_words

    names = {hid: node.name for hid, node in hierarchy.hubs.items()}
    echo_hits = echo_n = clean_hits = clean_n = 0

    for path in sorted(results_dir.glob("fold_*/predictions.json")):
        preds = load_json(path)
        record = load_json(path.parent / FOLD_RESULT_FILENAME)
        indicators = record["hit1_indicators"]
        for item, hit in zip(preds, indicators):
            hub_words = content(names.get(item["ground_truth_hub_id"], ""))
            anchor_words = content(item["control_text"])
            is_echo = bool(hub_words) and hub_words <= anchor_words
            if is_echo:
                echo_hits += hit
                echo_n += 1
            else:
                clean_hits += hit
                clean_n += 1

    total = echo_n + clean_n
    return {
        "n_total": total,
        "n_lexical_echo": echo_n,
        "echo_fraction": (echo_n / total) if total else 0.0,
        "hit_at_1_echo": (echo_hits / echo_n) if echo_n else None,
        # The number that describes semantic mapping rather than string match.
        "hit_at_1_non_echo": (clean_hits / clean_n) if clean_n else None,
        "n_non_echo": clean_n,
    }


def _numeric_value(value: Any) -> float | None:
    """The value of one indicator entry as a double, or None if it has none.

    None here means "this entry is not a number at all" -- a string, a null, a
    nested list, or an integer too large for a double -- and the callers below
    turn that into a raise. It is a classification, not a failure signal.
    """
    if not isinstance(value, (int, float)):
        return None
    try:
        return float(value)
    except OverflowError:
        # A JSON integer wider than a double. Whatever it is, it is not a hit.
        return None


def _out_of_domain_values(indicators: Sequence[Any]) -> tuple[list[str], int]:
    """The distinct entries that are neither 0 nor 1, as text to quote.

    Returns the values to show and how many further distinct ones were left
    out. Numbers sort numerically and everything else follows them by repr, so
    the sample reads in the order a human would have written it rather than in
    whatever order json.load happened to hand the array over.
    """
    ranked: dict[str, tuple[int, float, str]] = {}
    for value in indicators:
        numeric = _numeric_value(value)
        if numeric is not None and numeric in INDICATOR_DOMAIN:
            continue
        shown = repr(value)
        if numeric is not None and not math.isnan(numeric):
            ranked[shown] = (0, numeric, shown)
        else:
            ranked[shown] = (1, 0.0, shown)

    ordered = [shown for shown, _ in sorted(ranked.items(), key=lambda kv: kv[1])]
    return (
        ordered[:MAX_REPORTED_OFFENDING_VALUES],
        max(0, len(ordered) - MAX_REPORTED_OFFENDING_VALUES),
    )


def _assert_indicator_domain(
    indicators: Sequence[Any], path: Path, field: str,
) -> None:
    """Refuse an indicator array that is not made of hits and misses.

    Every other guard in load_fold_results checks SHAPE: that a field is
    present, that two arrays are the same length, that the folds agree on an
    arm, on their inputs and on a commit. A red-team pass wrote five fold
    records whose hit1_indicators were all 7.0 and walked them through the
    real load -> aggregate -> gate path: every guard passed, the aggregate
    logged "hit@1 (micro): 7.0000", and the gate returned
    point_estimate_pass=true, ci_low_pass=true, verdicts_agree=true. Nothing
    between the file and the headline number had ever asked what the values
    meant.
    """
    offending, withheld = _out_of_domain_values(indicators)
    if not offending:
        return
    suffix = f" (and {withheld} more distinct values)" if withheld else ""
    raise ValueError(
        f"{path}: {field} holds values outside the 0/1 indicator domain: "
        f"{', '.join(offending)}{suffix}. hit@1 is a per-item indicator -- the "
        "top-ranked hub was a valid answer or it was not -- so an array "
        "carrying anything else did not come from evaluate_on_fold and must "
        "not be micro-averaged into a hit rate."
    )


def _assert_summary_matches_indicators(
    metrics: Any, indicators: Sequence[Any], path: Path, where: str,
) -> None:
    """Make the record's two statements of its own hit@1 agree.

    A fold record says hit@1 twice: once as metrics.hit_at_1, which is what a
    human reads, and once as the array the micro average, the bootstrap CI and
    the gate are all computed from. Nothing made them agree. The domain check
    above catches an array that was never indicators; this catches the subtler
    producer-side case where a well-formed array no longer describes the
    summary sitting beside it in the same file -- a rescore applied to one and
    not the other, or a record assembled from two runs.

    Called after _assert_indicator_domain, so every entry is already 0 or 1.
    """
    if not indicators:
        raise ValueError(
            f"{path}: {where} covers an empty indicator array. A fold with no "
            "eval items cannot state a hit rate, and n_eval_items=0 satisfies "
            "the length check by arithmetic accident."
        )
    if not isinstance(metrics, dict) or "hit_at_1" not in metrics:
        raise ValueError(
            f"{path}: {where} carries no 'hit_at_1' to check its "
            f"{len(indicators)} indicators against. Every producer of this "
            "file writes one, so a record without it was assembled somewhere "
            "else -- which is the case this cross-check exists for. Skipping "
            "the check when the field is absent is how a guard dies."
        )
    reported = _numeric_value(metrics["hit_at_1"])
    if reported is None:
        raise ValueError(
            f"{path}: {where} reports hit_at_1={metrics['hit_at_1']!r}, which "
            "is not a number, so it states nothing the indicators can be "
            "checked against."
        )
    measured = float(sum(indicators)) / len(indicators)
    if abs(reported - measured) > INDICATOR_METRIC_TOLERANCE:
        raise ValueError(
            f"{path}: {where} reports hit_at_1={reported!r} while the "
            f"{len(indicators)} indicators beside it average {measured!r}. "
            "The record states its own hit@1 twice and the two statements "
            f"disagree by more than {INDICATOR_METRIC_TOLERANCE:g}, so one of "
            "them describes an evaluation the other did not; refusing to "
            "aggregate either."
        )


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
        _assert_indicator_domain(record["hit1_indicators"], path, "hit1_indicators")
        _assert_summary_matches_indicators(
            record.get("metrics"), record["hit1_indicators"], path, "metrics",
        )
        zero_shot = record.get("zero_shot") or {}
        zs_indicators = zero_shot.get("hit1_indicators")
        if zs_indicators is not None:
            if len(zs_indicators) != record["n_eval_items"]:
                # Equal length is what makes the delta paired. A baseline array
                # of the wrong length cannot be aligned item-for-item with the
                # trained one, and a mis-paired delta gets a paired interval it
                # has not earned.
                raise ValueError(
                    f"{path}: {len(zs_indicators)} zero-shot indicators for "
                    f"{record['n_eval_items']} eval items. The baseline is not "
                    "paired with the trained run; refusing to aggregate it."
                )
            # The gate reports trained MINUS baseline, so a fabricated baseline
            # buys the same headline as a fabricated trained run and reads as
            # the more innocent half of the record.
            _assert_indicator_domain(
                zs_indicators, path, "zero_shot.hit1_indicators",
            )
            if "metrics" in zero_shot:
                # Checked when present rather than required: a fold that was
                # not run with include_zero_shot carries no block at all, and
                # the block's own contract is the indicator array -- the
                # summary is a convenience some producers write.
                _assert_summary_matches_indicators(
                    zero_shot["metrics"], zs_indicators, path, "zero_shot.metrics",
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

    # Every flag that defines an arm belongs here. This listed two of them
    # while the campaign runs four arms, so prose and prose-desconly both
    # hashed to (True, False) and would have aggregated into a single number
    # describing neither. Derived from a named tuple of keys so adding a fifth
    # arm flag without extending the guard is a visible omission rather than a
    # silent one.
    arms = {
        tuple(r.get("config", {}).get(key) for key in ARM_DEFINING_KEYS)
        for r in records
    }
    if len(arms) > 1:
        raise ValueError(
            f"Folds under {results_dir} come from different arms: "
            f"{ARM_DEFINING_KEYS} = {sorted(arms)}. The arms "
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

    # load_fold_results now refuses an indicator array that is not 0/1, but it
    # is not on every path into this function: run_experiment hands over the
    # in-memory dicts run_single_fold returned, which never touch a file and
    # never meet that check. The 7.0 replay reached "AGGREGATE hit@1 (micro):
    # 7.0000" precisely because one guard was doing all the work, so this layer
    # states the invariant it can state on its own. A NaN mean fails this too,
    # since every comparison against NaN is false.
    low, high = HIT_RATE_BOUNDS
    mean = float(aggregate["mean"])
    if not low <= mean <= high:
        raise ValueError(
            f"Aggregate micro hit@1 is {mean!r}, outside [{low}, {high}]. It "
            "is a fraction of eval items, so it cannot be. The per-item "
            "indicators these folds were pooled from are not measurements of "
            "hit@1; refusing to report a hit rate from them."
        )

    # Report the macro figure alongside it. They differ only through fold-size
    # imbalance, so a wide gap is a signal about the folds, not a second result.
    macro = float(np.mean([float(np.mean(f)) for f in fold_hit1s]))
    # Checked independently of the micro rather than inferred from it, because
    # one large clean fold hides a poisoned small one in the pool: 100 zeros
    # beside a single 3.0 gives a micro of 0.0297, which no range check on the
    # micro alone would ever flag, and a macro of 1.5.
    if not low <= macro <= high:
        raise ValueError(
            f"Aggregate macro hit@1 is {macro!r}, outside [{low}, {high}], "
            f"while the micro figure ({mean!r}) is inside it. At least one "
            "fold's indicators are not hit@1 measurements and pooling hid it."
        )
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
    n_configurations: int = 1,
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
    # The third statement of the same invariant, at the layer that publishes
    # the verdict. run_experiment calls this with in-memory fold dicts that
    # passed through neither the loader nor the aggregate, and the 7.0 replay
    # ended here: a delta of 7.0 returned point_estimate_pass=true,
    # ci_low_pass=true and verdicts_agree=true. A delta is a difference of two
    # rates measured on the same items and cannot leave [-1, 1]. Checked before
    # the family-wise interval below, so a second 10,000-resample pass is not
    # spent on an array that was never a measurement.
    delta_low, delta_high = HIT_RATE_DELTA_BOUNDS
    micro_delta = float(paired["delta_mean"])
    if not delta_low <= micro_delta <= delta_high:
        raise ValueError(
            f"Micro hit@1 delta is {micro_delta!r}, outside "
            f"[{delta_low}, {delta_high}]. Either the trained indicators or "
            "the paired zero-shot ones are not per-item hit@1 measurements, "
            "so this is not a gate decision; refusing to return one."
        )
    # A second interval at the family-wise level, so a campaign that ran many
    # configurations reports one the selection cannot inflate.
    corrected = (
        paired if n_configurations == 1
        else paired_bootstrap_delta(
            baseline, trained,
            ci_level=1.0 - (1.0 - (1.0 - 0.05) ** (1.0 / n_configurations)),
        )
    )

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

    # Sidak-correct the threshold test when several configurations competed.
    # Simulated on this eval set (mean SE 0.0438, inter-arm item correlation
    # 0.678), 16 configurations under a null where every one has a true delta
    # of 0.08 produce at least one point estimate above 0.10 about 73% of the
    # time. Without n_configurations the gate cannot tell "this arm works"
    # from "this arm won a 16-way raffle".
    if n_configurations < 1:
        raise ValueError("n_configurations must be >= 1")
    alpha_effective = 1.0 - (1.0 - 0.05) ** (1.0 / n_configurations)
    point_estimate_pass = bool(paired["delta_mean"] > threshold)
    ci_low_pass = bool(paired["ci_low"] > threshold)

    decision = {
        "threshold": threshold,
        "micro_delta": paired["delta_mean"],
        "ci_low": paired["ci_low"],
        "ci_high": paired["ci_high"],
        # Selection context. n_configurations=1 leaves these equal to the
        # uncorrected pair; anything higher widens the interval and marks the
        # point estimate as selection-optimistic, which is what it is.
        "n_configurations": n_configurations,
        "alpha_effective": alpha_effective,
        "ci_low_familywise": corrected["ci_low"],
        "ci_high_familywise": corrected["ci_high"],
        "familywise_pass": bool(corrected["ci_low"] > threshold),
        "selection_optimistic": bool(n_configurations > 1),
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
        filter_set(
            use_stopwords=config.use_stopword_filter,
            use_framework_identity=config.use_framework_identity_filter,
        ),
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
