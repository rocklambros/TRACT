"""Build the blind expert-agreement ceiling study export (design doc Part 0.1).

    python -m scripts.build_ceiling_study                    regenerate the study of record
    python -m scripts.build_ceiling_study --study-name post_rebuild_2026   a NEW study
    python -m scripts.build_ceiling_study --out-dir /tmp/scratch           a scratch draw

Writes five files to the destination directory:

    ceiling_items.json           the 250 items, shuffled, no ground truth
    ceiling_answer_key.json      hidden: item_index -> OpenCRE gold hub(s)
    hub_reference.md             all 522 hubs, the owner's lookup
    ceiling_answers_TEMPLATE.json  blank worksheet the owner edits in place
    README.md                    instructions

Ruling R22. With no flag this targets results/ceiling_study/, which already
holds the 250 items a domain expert annotated by hand, and
require_unmoved_ceiling_study refuses the write unless this draw reproduces
them exactly. It does not today: the curated-link pool moved after the study
was scored, so the guard fires and names how far. That refusal is the point.
A fresh draw is a NEW study with its own name, so pass --study-name and it
lands under results/ceiling_study/studies/<name>/ where nothing can mistake
it for the annotated one.

results/ is gitignored project-wide, with results/ceiling_study/ negated back
in, so the five files there stage with a plain `git add` after review. That
review is what makes a study pre-registered rather than a note to self.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tract.ceiling_study import (
    build_ceiling_study,
    new_study_dir,
    require_new_study_destination,
    require_unmoved_ceiling_study,
)
from tract.config import (
    CEILING_STUDY_DIR,
    CEILING_STUDY_MAX_ACCEPTABLE_HUBS,
    CEILING_STUDY_N_ITEMS,
    CEILING_STUDY_SEED,
    CEILING_STUDY_TEST_FRAMEWORKS,
    CEILING_STUDY_VALIDATION_FRAMEWORKS,
    EXIT_USER_ERROR,
)
from tract.io import atomic_write_json, atomic_write_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ANSWER_KEY_WARNING = (
    "Opening this file before the ceiling study is finished, or before you "
    "have made a final decision to stop partway, invalidates the study. "
    "This is the hidden key: item_index maps to the OpenCRE gold hub(s) for "
    "that control. Review ceiling_answers_TEMPLATE.json first, in item order, "
    "without looking here."
)


def _build_readme(summary_line: str) -> str:
    return f"""# Ceiling study: blind expert-agreement review

## What this measures

Whether two qualified annotators agree on which CRE hub a control belongs
to. Nobody has measured this before at a size that means anything -- the
only prior evidence is 13 of 20 hidden calibration items from Phase 3, a
point estimate whose Wilson 95% interval is [0.433, 0.819]. This study
replaces that with 250 items, powered to a Wilson half-width of 0.059.

{summary_line}

## What to do

1. Open `ceiling_items.json`. Work through the items **in item_index order**.
   Do not skip around -- the order is shuffled so that stopping anywhere is
   still a valid random sample, and working out of order defeats that.
2. For each item, use `hub_reference.md` to find the CRE hub you believe the
   control belongs to. Search by keyword, or browse by branch -- the file is
   organized as an outline (5 top branches, then depth-first by name).
3. Fill in `ceiling_answers_TEMPLATE.json` in place, one entry per item:
   - `primary_hub_id`: your single best hub id. This measures alpha-1
     (agreement at rank 1, the ceiling on hit@1).
   - `acceptable_hub_ids`: up to {CEILING_STUDY_MAX_ACCEPTABLE_HUBS} hub ids
     you would also accept as correct, including the primary one if it
     belongs in the set. This measures alpha-5 (agreement within a
     shortlist, the ceiling on hit@5).
   - `confidence`: "high", "medium", or "low".
   - `notes`: optional, free text.
4. **Do not open `ceiling_answer_key.json`.** It has a warning at the top of
   the file for the same reason.

## Stopping partway is fine

The 250 items are shuffled so that every prefix -- the first 20, the first
83, whatever you actually get through -- is itself a valid stratified
sample across both strata. `scripts/score_ceiling_study.py` scores whatever
fraction of `primary_hub_id` fields are non-empty and reports how many of
250 were completed. There is no requirement to finish before scoring.

## Time budget

Expect roughly one to three minutes per item. A short CAPEC or CWE entry
reads faster than a longer NIST 800-53 control body. Nothing here is timed,
this is only so you can plan the session.

## Scoring

Once you have done as many items as you intend to, from the repository root:

    python -m scripts.score_ceiling_study

Reads your filled-in `ceiling_answers_TEMPLATE.json` against
`ceiling_answer_key.json`, and reports alpha-1 and alpha-5 with Wilson 95%
intervals, pooled and per stratum and per framework, against the 13/20
Phase 3 datum, and states plainly whether the interval is narrow enough to
decide anything.
"""


def _resolve_destination(args: argparse.Namespace) -> Path:
    """Where this run writes, and whether it is allowed to.

    Three routes, one guard each. The default targets the annotated study and
    is gated on reproducing it. A named study and a scratch --out-dir are
    gated on staying out of the annotated study's directory, which --out-dir
    can otherwise name and --study-name cannot.

    Raises:
        ValueError: If the destination would land on the pinned study.
    """
    if args.study_name is not None:
        destination = new_study_dir(args.study_name)
    elif args.out_dir is not None:
        destination = args.out_dir
    else:
        return CEILING_STUDY_DIR
    require_new_study_destination(destination / "ceiling_items.json")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--study-name", type=str, default=None,
        help=(
            "Draw a NEW study under results/ceiling_study/studies/<name>/. "
            "Lowercase letters, digits, underscore and hyphen only."
        ),
    )
    group.add_argument(
        "--out-dir", type=Path, default=None,
        help="Write a scratch draw to an arbitrary directory that nothing is gated on.",
    )
    args = parser.parse_args()

    logger.info("Building ceiling study (seed=%d, n_items=%d)",
                CEILING_STUDY_SEED, CEILING_STUDY_N_ITEMS)
    try:
        out_dir = _resolve_destination(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USER_ERROR

    items, answer_key, template, hub_reference_md, summary = build_ceiling_study()

    # Only the default route can collide with the annotated study, and it is
    # checked after the draw so the refusal can say how far the draw moved.
    if out_dir == CEILING_STUDY_DIR:
        try:
            require_unmoved_ceiling_study(items, out_dir / "ceiling_items.json")
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return EXIT_USER_ERROR

    items_out = {
        "seed": summary["seed"],
        "n_items": summary["n_items"],
        "sampling": {
            "validation_stratum": {
                "frameworks": list(CEILING_STUDY_VALIDATION_FRAMEWORKS),
                "allocation": summary["validation_allocation"],
                "pool_sizes": summary["validation_pool_sizes"],
            },
            "test_stratum": {
                "frameworks": list(CEILING_STUDY_TEST_FRAMEWORKS),
                "allocation": summary["test_allocation"],
                "pool_sizes": summary["test_pool_sizes"],
            },
        },
        "items": items,
    }
    atomic_write_json(items_out, out_dir / "ceiling_items.json")
    logger.info("Wrote %s", out_dir / "ceiling_items.json")

    key_out = {
        "WARNING": ANSWER_KEY_WARNING,
        "seed": summary["seed"],
        "answers": answer_key,
    }
    atomic_write_json(key_out, out_dir / "ceiling_answer_key.json")
    logger.info("Wrote %s", out_dir / "ceiling_answer_key.json")

    atomic_write_text(hub_reference_md, out_dir / "hub_reference.md")
    logger.info("Wrote %s", out_dir / "hub_reference.md")

    template_out = {"items": template}
    atomic_write_json(template_out, out_dir / "ceiling_answers_TEMPLATE.json")
    logger.info("Wrote %s", out_dir / "ceiling_answers_TEMPLATE.json")

    validation_breakdown = ", ".join(
        f"{fw} {summary['validation_allocation'][fw]}"
        for fw in CEILING_STUDY_VALIDATION_FRAMEWORKS
    )
    test_breakdown = ", ".join(
        f"{fw} {summary['test_allocation'][fw]}"
        for fw in CEILING_STUDY_TEST_FRAMEWORKS
    )
    summary_line = (
        f"Validation stratum (125 items): {validation_breakdown}. "
        f"Test stratum (125 items): {test_breakdown}."
    )
    atomic_write_text(_build_readme(summary_line), out_dir / "README.md")
    logger.info("Wrote %s", out_dir / "README.md")

    print(f"\n{summary_line}\n")
    print(f"5 files written to {out_dir}")
    print(
        "Review them, then stage the study before the owner starts:\n"
        f"  git add {out_dir}/ceiling_items.json "
        f"{out_dir}/ceiling_answer_key.json "
        f"{out_dir}/hub_reference.md "
        f"{out_dir}/ceiling_answers_TEMPLATE.json "
        f"{out_dir}/README.md\n"
        "A new study needs its own provenance record beside it, naming the "
        "corpus digest, the curated-links digest and the seed it was drawn "
        "from. See results/ceiling_study/ceiling_study_provenance.json."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
