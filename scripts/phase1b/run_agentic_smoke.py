"""Run the agentic smoke test against the winning arm's committed fold models.

This is NOT a metric and may not become one. `data/eval/agentic_smoke_test.json`
records that in the fixture itself, along with the arithmetic: three of its six
items answer hub `220-442`, so always guessing that one hub scores 0.500, and
only 6/6 clears p<0.05 on a one-sided binomial. Six items cannot rank an arm and
no arm is re-selected on whatever this prints.

What it does ask is a real question. `hub_links_curated.jsonl` carries zero
links for `owasp_agentic_top10`, so none of these six controls has ever been a
training anchor, while the four hubs they answer DO appear in training through
23 links from eight other frameworks. So this is a TEXT generalisation test:
does agentic control prose route to hubs the model already knows?

Every fold model is scored, not one. The five test-round checkpoints differ only
in which AI framework each held out, and none of them held out the agentic
framework -- it has no links to hold out. Picking one would be an unregistered
choice made after the results existed; scoring all five makes the spread across
them part of the finding.

`excluded_framework` is passed to build_all_hub_texts for each fold, but at the
campaign's `hub_rep_format="path+name"` it changes NOTHING: hub text is
`{hierarchy_path} | {name}`, entirely CRE-native, and firewall.py only consults
the exclusion when standards sections are appended. Verified rather than
assumed -- building the hub texts under all five fold exclusions and under no
exclusion at all yields byte-identical output for all 522 hubs.

So all five models here are scored against one hub set, and there is no
"unfiltered set" to swap in. An earlier version of this docstring claimed the
per-fold exclusion was doing work; it was not, and a reader would have taken a
firewall as applied that cannot apply in this format. The argument is kept
because it becomes live the moment `--hub-rep path+name+standards` is used, and
dropping it would leave that future run silently unfirewalled.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Final

from scripts.phase0.common import EvalItem
from tract.config import PHASE1B_RESULTS_DIR, PROCESSED_DIR, max_anchor_chars
from tract.framework_identity import filter_set
from tract.hierarchy import CREHierarchy
from tract.io import atomic_write_json, load_json
from tract.text_selection import ProseIndex, SelectionStats, apply_prose_to_corpus
from tract.training.evaluate import evaluate_on_fold
from tract.training.firewall import build_all_hub_texts

logger = logging.getLogger(__name__)

SMOKE_FIXTURE: Final[Path] = Path("data/eval/agentic_smoke_test.json")

# The name this framework carries in the merged corpus, which is the key
# ProseIndex is built on. The fixture uses the short `framework_id` instead, so
# the two are joined here rather than assumed equal.
AGENTIC_FRAMEWORK_NAME: Final[str] = "OWASP Top 10 for Agentic Applications 2026"
AGENTIC_FRAMEWORK_ID: Final[str] = "owasp_agentic_top10"

# Pre-declared in the fixture, restated here so a reader of this file sees the
# thresholds without opening the JSON. Both come from
# `pre_declared_pass_condition` and neither is computed from the results.
FAIL_AT_OR_BELOW: Final[int] = 1
INVESTIGATE_MAX: Final[int] = 3


def _build_eval_items(
    fixture: dict[str, Any],
    hierarchy: CREHierarchy,
    use_prose: bool,
    use_stopwords: bool,
    max_seq_length: int,
) -> list[EvalItem]:
    """Six EvalItems carrying the same text treatment the arm's own items got.

    Built from titles first and then swapped to prose, which is the order
    run_fold.py uses. The order matters for a different reason there (it fixes
    the item set before the anchor can change it); here it matters because
    ProseIndex resolves by title before id, so the title has to be the anchor
    the lookup sees.
    """
    controls_by_id: dict[str, dict[str, Any]] = {}
    corpus_path = ProseIndex.load().source_path
    if corpus_path is None:
        raise ValueError(
            "ProseIndex reported no source path, so the corpus this run read "
            "cannot be recorded. Refusing to produce an unattributable result."
        )
    for record in load_json(corpus_path)["frameworks"]:
        if record.get("framework_id") == AGENTIC_FRAMEWORK_ID:
            controls_by_id = {c["control_id"]: c for c in record["controls"]}
            break
    if not controls_by_id:
        raise ValueError(
            f"{AGENTIC_FRAMEWORK_ID!r} is absent from {corpus_path}. Without the "
            "licensed overlay staged there is no agentic prose to score, and a "
            "title-only run would answer a different question than the fixture "
            "asks. Run `python -m scripts.stage_licensed_overlay --unpack`."
        )

    items: list[EvalItem] = []
    for entry in fixture["items"]:
        control_id = entry["control_id"]
        control = controls_by_id.get(control_id)
        if control is None:
            raise ValueError(
                f"Fixture item {control_id!r} has no control in {corpus_path}. "
                "The fixture and the corpus disagree about what exists; that is "
                "a data defect, not a scoring edge case."
            )
        hub_id = entry["hub_id"]
        if hub_id not in hierarchy.hubs:
            raise ValueError(
                f"Fixture item {control_id!r} answers hub {hub_id!r}, which is "
                "not in the hierarchy. The fixture was written against a "
                "different hub set and its answers cannot be scored."
            )
        items.append(EvalItem(
            control_text=str(control.get("title") or ""),
            ground_truth_hub_id=hub_id,
            # One hub per item, by construction: the fixture collapsed the
            # 39-row source CSV to distinct (control, hub) pairs. Strict
            # scoring, no multi-label credit.
            valid_hub_ids=frozenset({hub_id}),
            ground_truth_hub_name=entry["hub_name"],
            framework_name=AGENTIC_FRAMEWORK_NAME,
            section_id=control_id,
            track="full-text",
        ))

    if not use_prose:
        return items

    stats = SelectionStats()
    items = apply_prose_to_corpus(
        items,
        ProseIndex.load(),
        filter_set(use_stopwords=use_stopwords, use_framework_identity=False),
        stats=stats,
        max_chars=max_anchor_chars(max_seq_length),
    )
    stats.log_summary("Agentic smoke anchors")
    n_title = sum(1 for i in items if len(i.control_text) < 200)
    if n_title:
        logger.warning(
            "%d of %d anchors look title-length. The standing rule is that a "
            "title fallback is a last resort and is counted, not defaulted to.",
            n_title, len(items),
        )
    return items


def _held_out_framework(fold_dir: Path) -> str:
    """The framework this fold held out, read from the record it wrote.

    Reconstructing it from the directory name -- `fold_X` with underscores
    turned back into spaces -- is lossy in both directions. It happens to
    round-trip for the current five AI frameworks, which is exactly why the
    mistake survives: `fold_NIST_AI_100-2` gives back "NIST AI 100-2" today,
    and a framework whose display name contains an underscore, or a fold
    directory written under a stricter slug rule, would give back a name that
    matches nothing and silently disable whatever is keyed on it.
    """
    record = load_json(fold_dir / "fold_result.json")
    name = record.get("held_out_framework")
    if not name:
        raise ValueError(
            f"{fold_dir / 'fold_result.json'} records no held_out_framework. "
            "The directory name is not a substitute: it is a slug, and "
            "reversing a slug guesses. Re-run the fold or fix the record."
        )
    return str(name)


def _score_one_model(
    model_dir: Path,
    held_out_framework: str,
    items: list[EvalItem],
    hierarchy: CREHierarchy,
    hub_ids: list[str],
    use_stopwords: bool,
    max_seq_length: int,
) -> dict[str, Any]:
    """Score one fold checkpoint. Loads a model, so this runs on a pod."""
    from sentence_transformers import SentenceTransformer

    stopwords = filter_set(
        use_stopwords=use_stopwords, use_framework_identity=False,
    )
    hub_texts = build_all_hub_texts(
        hierarchy, excluded_framework=held_out_framework, stopwords=stopwords,
    )
    logger.info("Loading %s", model_dir)
    model = SentenceTransformer(str(model_dir))
    model.max_seq_length = max_seq_length

    metrics, predictions, hit1 = evaluate_on_fold(
        model, items, hub_ids, hub_texts,
    )

    per_item = []
    for item, ranked, hit in zip(items, predictions, hit1):
        top1 = ranked[0]
        truth = item.ground_truth_hub_id
        per_item.append({
            "control_id": item.section_id,
            "ground_truth_hub_id": truth,
            "ground_truth_hub_name": item.ground_truth_hub_name,
            "top1_hub_id": top1,
            "top1_hub_name": hierarchy.hubs[top1].name,
            "correct": bool(hit),
            # The fixture fails the run on a top-1 in the wrong BRANCH even
            # when the count would otherwise be acceptable. A near miss inside
            # the right branch and a miss in a different branch are not the
            # same error and the pass condition does not treat them alike.
            "same_branch": (
                hierarchy.hubs[top1].branch_root_id
                == hierarchy.hubs[truth].branch_root_id
            ),
            "rank_of_truth": (
                ranked.index(truth) + 1 if truth in ranked else None
            ),
        })

    n_correct = sum(1 for p in per_item if p["correct"])
    return {
        "held_out_framework": held_out_framework,
        "model_dir": str(model_dir),
        "n_correct": n_correct,
        "n_items": len(items),
        "n_wrong_branch": sum(1 for p in per_item if not p["same_branch"]),
        "hit_at_1": metrics["hit_at_1"],
        "hit_at_5": metrics["hit_at_5"],
        "mrr": metrics["mrr"],
        "per_item": per_item,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True,
                        help="Results directory under results/phase1b holding "
                             "the fold checkpoints to score.")
    parser.add_argument("--stopwords", action="store_true")
    parser.add_argument("--no-prose", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Assemble items and locate checkpoints, then stop "
                             "without loading a model. Safe off-pod.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    fixture = load_json(SMOKE_FIXTURE)
    if fixture.get("is_a_metric") is not False:
        raise ValueError(
            "The fixture no longer declares is_a_metric=false. This runner "
            "exists on the premise that it is not a metric; if that changed, "
            "the change has to be deliberate and the runner reviewed."
        )
    hierarchy = CREHierarchy.model_validate(
        load_json(PROCESSED_DIR / "cre_hierarchy.json")
    )
    hub_ids = sorted(hierarchy.hubs.keys())

    items = _build_eval_items(
        fixture, hierarchy,
        use_prose=not args.no_prose,
        use_stopwords=args.stopwords,
        max_seq_length=args.max_seq_length,
    )
    logger.info("Assembled %d smoke items over %d hubs", len(items), len(hub_ids))

    results_dir = PHASE1B_RESULTS_DIR / args.config_name
    fold_dirs = sorted(
        d for d in results_dir.glob("fold_*")
        if (d / "model" / "model").is_dir()
    )
    if not fold_dirs:
        raise ValueError(
            f"No fold checkpoints under {results_dir}. Expected "
            "fold_*/model/model directories; a collect that pulled only JSON "
            "leaves nothing to score."
        )
    logger.info("Found %d fold checkpoints", len(fold_dirs))

    if args.dry_run:
        for item in items:
            logger.info("  %s -> %s (%d chars)", item.section_id,
                        item.ground_truth_hub_id, len(item.control_text))
        for d in fold_dirs:
            logger.info("  would score %s", d.name)
        return 0

    per_fold = [
        _score_one_model(
            d / "model" / "model",
            _held_out_framework(d),
            items, hierarchy, hub_ids,
            use_stopwords=args.stopwords,
            max_seq_length=args.max_seq_length,
        )
        for d in fold_dirs
    ]

    counts = [r["n_correct"] for r in per_fold]
    n_items = len(items)
    # Reported as a range because five models were scored. Collapsing them to a
    # mean would invent a single number the fixture never asked for, and the
    # fixture's own pass condition is stated over counts.
    verdict = "pass"
    if min(counts) <= FAIL_AT_OR_BELOW or any(r["n_wrong_branch"] for r in per_fold):
        verdict = "fail"
    elif min(counts) <= INVESTIGATE_MAX:
        verdict = "investigate"

    summary = {
        "config_name": args.config_name,
        "is_a_metric": False,
        "n_items": n_items,
        "n_models_scored": len(per_fold),
        "counts_by_fold": {r["held_out_framework"]: r["n_correct"] for r in per_fold},
        "range": [min(counts), max(counts)],
        "verdict_against_pre_declared_condition": verdict,
        "per_fold": per_fold,
    }

    out = args.output or (results_dir / "agentic_smoke_test.json")
    atomic_write_json(summary, out)
    logger.info("=" * 62)
    for r in per_fold:
        logger.info("  %-22s %d of %d correct, %d wrong-branch",
                    r["held_out_framework"], r["n_correct"], n_items,
                    r["n_wrong_branch"])
    logger.info("  RANGE: %d-%d of %d across %d models -> %s",
                min(counts), max(counts), n_items, len(per_fold), verdict)
    logger.info("  Not a metric. No arm is re-selected on this.")
    logger.info("=" * 62)
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
