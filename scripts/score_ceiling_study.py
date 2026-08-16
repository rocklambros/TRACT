"""Score the blind expert-agreement ceiling study (design doc Part 0.1).

    python -m scripts.score_ceiling_study
    python -m scripts.score_ceiling_study --model-hit-at-5 0.62

Reads the owner's filled-in answers against the hidden key and reports
alpha-1 and alpha-5 with Wilson 95% intervals: pooled, per stratum, and per
framework. Scores whatever fraction of items has a non-empty
`primary_hub_id` -- completion is not required to run this.

alpha-1: primary_hub_id matches any of the item's valid gold hub ids.
alpha-5: {primary_hub_id} | acceptable_hub_ids intersects the valid gold set.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Literal, TypedDict

from tract.config import (
    CEILING_STUDY_DIR,
    CEILING_STUDY_MAX_ACCEPTABLE_HUBS,
    CEILING_STUDY_N_ITEMS,
    CEILING_STUDY_TARGET_HALF_WIDTH,
    EXIT_USER_ERROR,
)
from tract.io import load_json
from tract.stats import WilsonInterval, wilson_interval

# The Phase 3 hidden-calibration datum this study replaces (design doc
# Part 0.1). Hardcoded: it is a fixed historical measurement, not something
# this script recomputes.
PRIOR_CALIBRATION_SUCCESSES = 13
PRIOR_CALIBRATION_N = 20

_VALID_CONFIDENCE = {"", "high", "medium", "low"}


class AnswerRow(TypedDict):
    item_index: int
    primary_hub_id: str
    acceptable_hub_ids: list[str]
    confidence: str
    notes: str


class ScoredItem(TypedDict):
    item_index: int
    framework_id: str
    stratum: str
    alpha1_hit: bool
    alpha5_hit: bool


def _load_items_metadata(path: Path) -> dict[int, tuple[str, str]]:
    """item_index -> (framework_id, stratum), from the (ground-truth-free) items file."""
    data = load_json(path)
    out: dict[int, tuple[str, str]] = {}
    for item in data["items"]:
        out[int(item["item_index"])] = (str(item["framework_id"]), str(item["stratum"]))
    return out


def _load_answer_key(path: Path) -> dict[int, list[str]]:
    """item_index -> valid gold hub ids."""
    data = load_json(path)
    out: dict[int, list[str]] = {}
    for entry in data["answers"]:
        out[int(entry["item_index"])] = [str(h) for h in entry["valid_gold_hub_ids"]]
    return out


def _load_answers(path: Path) -> list[AnswerRow]:
    """Owner's worksheet, validated but tolerant of unfilled rows."""
    data: Any = load_json(path)
    rows: list[AnswerRow] = []
    for raw in data["items"]:
        item_index = int(raw["item_index"])
        primary = str(raw.get("primary_hub_id") or "")
        acceptable = [str(h) for h in (raw.get("acceptable_hub_ids") or [])]
        confidence = str(raw.get("confidence") or "")
        notes = str(raw.get("notes") or "")

        if confidence not in _VALID_CONFIDENCE:
            raise ValueError(
                f"item_index={item_index}: confidence {confidence!r} is not one "
                f"of 'high', 'medium', 'low', or empty"
            )
        if primary and len(acceptable) > CEILING_STUDY_MAX_ACCEPTABLE_HUBS:
            raise ValueError(
                f"item_index={item_index}: {len(acceptable)} acceptable_hub_ids, "
                f"more than the {CEILING_STUDY_MAX_ACCEPTABLE_HUBS} allowed"
            )

        rows.append({
            "item_index": item_index,
            "primary_hub_id": primary,
            "acceptable_hub_ids": acceptable,
            "confidence": confidence,
            "notes": notes,
        })
    return rows


def score_items(
    answers: list[AnswerRow],
    key: dict[int, list[str]],
    metadata: dict[int, tuple[str, str]],
) -> list[ScoredItem]:
    """Score only rows with a non-empty primary_hub_id. Order not guaranteed."""
    scored: list[ScoredItem] = []
    for row in answers:
        if not row["primary_hub_id"]:
            continue
        item_index = row["item_index"]
        if item_index not in key:
            raise ValueError(f"item_index={item_index} has no entry in the answer key")
        if item_index not in metadata:
            raise ValueError(f"item_index={item_index} has no entry in ceiling_items.json")

        gold = set(key[item_index])
        framework_id, stratum = metadata[item_index]

        alpha1_hit = row["primary_hub_id"] in gold
        acceptable_set = set(row["acceptable_hub_ids"]) | {row["primary_hub_id"]}
        alpha5_hit = not acceptable_set.isdisjoint(gold)

        scored.append({
            "item_index": item_index,
            "framework_id": framework_id,
            "stratum": stratum,
            "alpha1_hit": alpha1_hit,
            "alpha5_hit": alpha5_hit,
        })
    return scored


def _wilson_for(
    scored: list[ScoredItem], field: Literal["alpha1_hit", "alpha5_hit"],
) -> WilsonInterval | None:
    if not scored:
        return None
    successes = sum(1 for s in scored if s[field])
    return wilson_interval(successes, len(scored))


def _format_interval(label: str, interval: WilsonInterval | None, n: int) -> str:
    if interval is None:
        return f"  {label}: no completed items (n=0)"
    return (
        f"  {label}: {interval.point:.3f} (n={n}), "
        f"95% CI [{interval.lower:.3f}, {interval.upper:.3f}], "
        f"half-width {interval.half_width:.3f}"
    )


def _report_group(title: str, scored: list[ScoredItem]) -> None:
    print(f"\n{title} (n={len(scored)})")
    print(_format_interval("alpha-1", _wilson_for(scored, "alpha1_hit"), len(scored)))
    print(_format_interval("alpha-5", _wilson_for(scored, "alpha5_hit"), len(scored)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--items", type=Path, default=CEILING_STUDY_DIR / "ceiling_items.json",
        help="Path to ceiling_items.json (ground-truth-free item metadata).",
    )
    parser.add_argument(
        "--key", type=Path, default=CEILING_STUDY_DIR / "ceiling_answer_key.json",
        help="Path to ceiling_answer_key.json (the hidden gold hub ids).",
    )
    parser.add_argument(
        "--answers", type=Path,
        default=CEILING_STUDY_DIR / "ceiling_answers_TEMPLATE.json",
        help="Path to the owner's filled-in answers file.",
    )
    parser.add_argument(
        "--model-hit-at-5", type=float, default=None,
        help=(
            "The best configuration's measured hit@5 on the same roster. "
            "When given, evaluates the S1 stop-gate trigger directly."
        ),
    )
    args = parser.parse_args()

    for path in (args.items, args.key, args.answers):
        if not path.exists():
            print(f"error: file not found: {path}", file=sys.stderr)
            return EXIT_USER_ERROR

    metadata = _load_items_metadata(args.items)
    key = _load_answer_key(args.key)
    answers = _load_answers(args.answers)

    scored = score_items(answers, key, metadata)
    n_completed = len(scored)

    print(f"Completed {n_completed} of {CEILING_STUDY_N_ITEMS} items.")
    if n_completed == 0:
        print("Nothing scored yet -- fill in primary_hub_id for at least one item.")
        return 0

    _report_group("Pooled", scored)
    for stratum in ("validation", "test"):
        subset = [s for s in scored if s["stratum"] == stratum]
        _report_group(f"Stratum: {stratum}", subset)

    frameworks = sorted({s["framework_id"] for s in scored})
    print("\nPer-framework breakdown:")
    for framework_id in frameworks:
        subset = [s for s in scored if s["framework_id"] == framework_id]
        _report_group(f"  Framework: {framework_id}", subset)

    # ── S1 gate power: is the pooled alpha-5 interval narrow enough to be
    # usable as a gate at all, independent of any specific model score.
    pooled_alpha5 = _wilson_for(scored, "alpha5_hit")
    print("\nS1 gate power check (design target half-width "
          f"{CEILING_STUDY_TARGET_HALF_WIDTH:.3f}, n={CEILING_STUDY_N_ITEMS}):")
    if pooled_alpha5 is None:
        print("  cannot decide: no completed items to compute alpha-5 from")
    elif n_completed < CEILING_STUDY_N_ITEMS:
        print(
            f"  study incomplete ({n_completed}/{CEILING_STUDY_N_ITEMS}): "
            f"achieved half-width {pooled_alpha5.half_width:.3f} on the completed "
            "subset. This will narrow as more items are scored. The design "
            f"target ({CEILING_STUDY_TARGET_HALF_WIDTH:.3f}) assumes all "
            f"{CEILING_STUDY_N_ITEMS}."
        )
    elif pooled_alpha5.half_width <= CEILING_STUDY_TARGET_HALF_WIDTH:
        print(
            f"  narrow enough: half-width {pooled_alpha5.half_width:.3f} <= "
            f"target {CEILING_STUDY_TARGET_HALF_WIDTH:.3f}"
        )
    else:
        print(
            f"  NOT narrow enough: half-width {pooled_alpha5.half_width:.3f} > "
            f"target {CEILING_STUDY_TARGET_HALF_WIDTH:.3f}. The alpha-5 point "
            "estimate itself moved further from 0.5 than the design assumed, "
            "which widens the achieved interval past the plan."
        )

    # ── S1 gate trigger: only evaluable against an actual measured hit@5.
    print("\nS1 gate trigger (best configuration's hit@5 vs the alpha-5 CI):")
    if args.model_hit_at_5 is None:
        print("  not evaluated: pass --model-hit-at-5 to check the trigger directly")
    elif pooled_alpha5 is None:
        print("  cannot decide: no completed items to compute alpha-5 from")
    else:
        gap = abs(args.model_hit_at_5 - pooled_alpha5.point)
        if pooled_alpha5.half_width > gap:
            print(
                f"  cannot decide: the alpha-5 CI half-width "
                f"({pooled_alpha5.half_width:.3f}) is wider than the gap between "
                f"hit@5 ({args.model_hit_at_5:.3f}) and the alpha-5 point estimate "
                f"({pooled_alpha5.point:.3f}), which is {gap:.3f}. The measurement "
                "noise is bigger than the effect being tested. More completed "
                "items are needed before this gate can fire either way."
            )
        elif pooled_alpha5.lower <= args.model_hit_at_5 <= pooled_alpha5.upper:
            print(
                f"  S1 TRIGGERED: hit@5 {args.model_hit_at_5:.3f} falls inside "
                f"the alpha-5 95% CI [{pooled_alpha5.lower:.3f}, "
                f"{pooled_alpha5.upper:.3f}]. Stop architecture work. The model "
                "is already at the limit of its labels."
            )
        else:
            print(
                f"  S1 not triggered: hit@5 {args.model_hit_at_5:.3f} falls "
                f"outside the alpha-5 95% CI [{pooled_alpha5.lower:.3f}, "
                f"{pooled_alpha5.upper:.3f}]."
            )

    # ── Comparison against the Phase 3 13/20 datum.
    prior = wilson_interval(PRIOR_CALIBRATION_SUCCESSES, PRIOR_CALIBRATION_N)
    pooled_alpha1 = _wilson_for(scored, "alpha1_hit")
    print(
        f"\nPhase 3 prior (13/20): {prior.point:.3f}, "
        f"95% CI [{prior.lower:.3f}, {prior.upper:.3f}], "
        f"half-width {prior.half_width:.3f}"
    )
    if pooled_alpha1 is None:
        print("  cannot compare: no completed items to compute alpha-1 from")
    else:
        overlap = not (pooled_alpha1.upper < prior.lower or pooled_alpha1.lower > prior.upper)
        print(
            f"  this study's pooled alpha-1: {pooled_alpha1.point:.3f}, "
            f"95% CI [{pooled_alpha1.lower:.3f}, {pooled_alpha1.upper:.3f}]"
        )
        print(f"  intervals overlap: {overlap}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
