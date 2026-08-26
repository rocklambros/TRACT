"""Shared logic for the LLM judge panel over the ceiling study.

Parsing of judge responses, of the hub reference, and the hierarchy-distance
categories the human ceiling analysis used. Kept here rather than in either
script because the runner and the analysis have to agree exactly on what a
hub id is and on what "sibling" means, and two copies would drift.
"""
from __future__ import annotations

import json
import re
from typing import Final, Literal, TypedDict

# CRE hub ids are NNN-NNN throughout the corpus. Used to reject invented ids
# before they reach a set-membership test, where a malformed string would
# silently behave like a wrong-but-real answer.
HUB_ID_RE: Final[re.Pattern[str]] = re.compile(r"\d{3}-\d{3}")

_HUB_HEADING_RE: Final[re.Pattern[str]] = re.compile(r"^### (\S+) -- (.*)$")
_HUB_PATH_RE: Final[re.Pattern[str]] = re.compile(r"^Path: (.*)$")

DistanceCategory = Literal[
    "same", "ancestor/descendant", "sibling", "same branch", "different branch", "unknown"
]

# Ordered nearest-first. The analysis reports the same four disagreement
# categories the human ceiling analysis reported, so the panel's numbers sit
# in the same frame as the human's 9.3 / 3.7 / 17.8 / 69.2.
DISAGREEMENT_CATEGORIES: Final[tuple[DistanceCategory, ...]] = (
    "ancestor/descendant",
    "sibling",
    "same branch",
    "different branch",
)


class PanelAnswer(TypedDict):
    """One judge's answer for one item, in the human worksheet's schema."""

    item_index: int
    primary_hub_id: str
    acceptable_hub_ids: list[str]
    confidence: str
    notes: str


def model_slug(model_id: str) -> str:
    """Filesystem-safe slug for a HuggingFace-style `org/name` model id."""
    return re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")


def parse_hub_reference(text: str) -> dict[str, list[str]]:
    """hub_id -> hierarchy path components, from hub_reference.md.

    The reference is the annotator's only view of the taxonomy, so it is also
    the authority on which ids exist. Reading the hierarchy from anywhere else
    would let the analysis credit a judge for an id the judge was never shown.

    Raises:
        ValueError: If a hub heading carries no Path line, which would mean
            the reference file changed shape and every distance category
            computed from it would be quietly wrong.
    """
    paths: dict[str, list[str]] = {}
    pending: str | None = None
    for line in text.splitlines():
        heading = _HUB_HEADING_RE.match(line)
        if heading:
            if pending is not None:
                raise ValueError(f"hub {pending} has a heading but no Path line")
            pending = heading.group(1)
            continue
        path_match = _HUB_PATH_RE.match(line)
        if path_match and pending is not None:
            paths[pending] = [part.strip() for part in path_match.group(1).split(">")]
            pending = None
    if pending is not None:
        raise ValueError(f"hub {pending} has a heading but no Path line")
    if not paths:
        raise ValueError("hub reference contained no parseable hubs")
    return paths


def distance_category(
    left: str, right: str, paths: dict[str, list[str]]
) -> DistanceCategory:
    """How far apart two hubs are in the CRE hierarchy.

    Matches the categories the human ceiling analysis used, and reproduces its
    published 9.3 / 3.7 / 17.8 / 69.2 split exactly when applied to the human
    answers against the key's `primary_gold_hub_id`.
    """
    if left == right:
        return "same"
    left_path = paths.get(left)
    right_path = paths.get(right)
    if left_path is None or right_path is None:
        return "unknown"
    if left_path[: len(right_path)] == right_path or right_path[: len(left_path)] == left_path:
        return "ancestor/descendant"
    if left_path[:-1] == right_path[:-1]:
        return "sibling"
    if left_path[0] == right_path[0]:
        return "same branch"
    return "different branch"


def extract_json_array(text: str) -> list[object] | None:
    """First balanced JSON array in *text*, tolerating code fences and prose.

    Judges are told to return bare JSON and mostly do, but a reasoning model
    that leaks a sentence before the array should not cost a whole batch.
    Brace counting rather than a regex because hub descriptions quoted back
    into `notes` contain brackets.

    Parsed with strict=False because GLM emits raw control characters inside
    string values, which the strict JSON grammar rejects outright. Losing a
    25-item batch to an unescaped newline in a `notes` field is not a
    trade worth making.
    """
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    start = cleaned.find("[")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for pos in range(start, len(cleaned)):
        char = cleaned[pos]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(cleaned[start : pos + 1], strict=False)
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, list) else None
    return None


def parse_judge_response(
    text: str,
    expected_indices: list[int],
    valid_hub_ids: set[str],
    max_acceptable: int,
) -> tuple[list[PanelAnswer], dict[str, int]]:
    """Judge output -> one answer row per expected item, in the given order.

    An item the judge did not answer comes back with an empty
    `primary_hub_id`, which the scorer counts as incomplete rather than
    wrong. That is the honest treatment: scoring a missing answer as a miss
    would penalise a transport failure as if it were a judgement.

    An invented `primary_hub_id` is kept rather than blanked. Blanking it
    would drop the item from the denominator and inflate the judge's score,
    which is the one failure mode this parser must not have.

    Returns the rows and counters for invented primaries, truncated
    `acceptable_hub_ids` lists, and items with no answer at all.
    """
    counters = {"invented_primary": 0, "truncated_acceptable": 0, "missing": 0}
    block = extract_json_array(text)

    by_index: dict[int, dict[str, object]] = {}
    if block is not None:
        for entry in block:
            if isinstance(entry, dict) and "item_index" in entry:
                try:
                    by_index[int(str(entry["item_index"]))] = entry
                except ValueError:
                    continue

    rows: list[PanelAnswer] = []
    for index in expected_indices:
        entry = by_index.get(index)
        if entry is None:
            counters["missing"] += 1
            rows.append({
                "item_index": index,
                "primary_hub_id": "",
                "acceptable_hub_ids": [],
                "confidence": "",
                "notes": "",
            })
            continue

        primary = str(entry.get("primary_hub_id") or "").strip()
        if primary and primary not in valid_hub_ids:
            counters["invented_primary"] += 1

        raw_acceptable = entry.get("acceptable_hub_ids") or []
        acceptable: list[str] = []
        if isinstance(raw_acceptable, list):
            for hub in raw_acceptable:
                hub_id = str(hub).strip()
                if hub_id and hub_id not in acceptable:
                    acceptable.append(hub_id)
        if len(acceptable) > max_acceptable:
            counters["truncated_acceptable"] += 1
            acceptable = acceptable[:max_acceptable]

        confidence = str(entry.get("confidence") or "").strip().lower()
        if confidence not in {"", "high", "medium", "low"}:
            confidence = ""

        notes = str(entry.get("notes") or "").strip().replace("\x00", "")

        rows.append({
            "item_index": index,
            "primary_hub_id": primary,
            "acceptable_hub_ids": acceptable,
            "confidence": confidence,
            "notes": notes[:500],
        })

    if not rows or all(not row["primary_hub_id"] for row in rows):
        counters["missing"] = len(expected_indices)
    return rows, counters


def parse_hub_names(text: str) -> dict[str, str]:
    """hub_id -> hub name, from hub_reference.md.

    Ground truth for the exposure control, taken from the same file the
    annotator reads so a "correct" name is the one the study itself uses.
    """
    names: dict[str, str] = {}
    for line in text.splitlines():
        heading = _HUB_HEADING_RE.match(line)
        if heading:
            names[heading.group(1)] = heading.group(2).strip()
    if not names:
        raise ValueError("hub reference contained no parseable hub names")
    return names
