"""Three-way agreement structure across the human, the LLM panel, and OpenCRE.

    python -m scripts.analyze_panel_agreement
    python -m scripts.analyze_panel_agreement --out results/ceiling_study/panel_agreement.md

The human ceiling study returned pooled alpha-1 of 0.572, but CAPEC's alpha-1
is 0.181 on 83 items and CAPEC is 42.8% of the training graph. Two
explanations are live:

    (a) OpenCRE's CAPEC links are poor.
    (b) the single human annotator's reading of CAPEC is idiosyncratic.

They make opposite predictions. Under (b), independent judges should land
where OpenCRE landed and the human should be the outlier. Under (a), judges
that never saw the human's answers should land near the human and away from
OpenCRE. The headline number is the count of CAPEC items where the human and
the panel majority agree with each other while both disagree with OpenCRE.

Reuses scripts.score_ceiling_study.score_items for alpha-1 and alpha-5, and
tract.stats.wilson_interval for every interval, so the panel's numbers are
produced by the same code that produced the human's.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple

from scripts.score_ceiling_study import AnswerRow, ScoredItem, score_items
from tract.config import (
    CEILING_STUDY_DIR,
    EXIT_USER_ERROR,
    PANEL_CONTAMINATION_PROBE_FRAMEWORK,
    PANEL_MODELS,
)
from tract.io import atomic_write_text, load_json
from tract.panel import (
    DISAGREEMENT_CATEGORIES,
    DistanceCategory,
    distance_category,
    model_slug,
    parse_hub_reference,
)
from tract.stats import WilsonInterval, wilson_interval

# The framework the whole study turns on, and the one whose 83 items carry
# the headline contingency.
CAPEC: Final[str] = "capec"


class Annotator(NamedTuple):
    """One annotator's answers plus the provenance needed to reproduce them."""

    name: str
    kind: Literal["human", "panel"]
    answers: dict[int, AnswerRow]
    provenance: dict[str, Any]


class ItemMeta(NamedTuple):
    framework_id: str
    stratum: str


def _load_items_metadata(path: Path) -> dict[int, ItemMeta]:
    data = load_json(path)
    return {
        int(item["item_index"]): ItemMeta(str(item["framework_id"]), str(item["stratum"]))
        for item in data["items"]
    }


def _load_key(path: Path) -> tuple[dict[int, list[str]], dict[int, str]]:
    """(item_index -> valid gold hubs, item_index -> the single primary gold).

    Both are needed. alpha-1 and alpha-5 score against the valid set, because
    an item can have several defensible gold hubs. The hierarchy-distance
    breakdown measures against `primary_gold_hub_id`, which is what the human
    ceiling analysis used and what reproduces its published split.
    """
    data = load_json(path)
    valid: dict[int, list[str]] = {}
    primary: dict[int, str] = {}
    for entry in data["answers"]:
        index = int(entry["item_index"])
        valid[index] = [str(hub) for hub in entry["valid_gold_hub_ids"]]
        primary[index] = str(entry["primary_gold_hub_id"])
    return valid, primary


def _load_annotator(path: Path, name: str, kind: Literal["human", "panel"]) -> Annotator:
    data = load_json(path)
    answers: dict[int, AnswerRow] = {}
    for raw in data["items"]:
        index = int(raw["item_index"])
        answers[index] = {
            "item_index": index,
            "primary_hub_id": str(raw.get("primary_hub_id") or ""),
            "acceptable_hub_ids": [str(h) for h in raw.get("acceptable_hub_ids") or []],
            "confidence": str(raw.get("confidence") or ""),
            "notes": str(raw.get("notes") or ""),
        }
    return Annotator(name, kind, answers, dict(data.get("run") or {}))


def _fmt(interval: WilsonInterval | None, n: int) -> str:
    if interval is None or n == 0:
        return "n/a (n=0)"
    return (
        f"{interval.point:.3f} [{interval.lower:.3f}, {interval.upper:.3f}] "
        f"(n={n})"
    )


def _alpha(
    scored: list[ScoredItem], field: Literal["alpha1_hit", "alpha5_hit"]
) -> WilsonInterval | None:
    if not scored:
        return None
    successes = sum(1 for row in scored if row[field])
    return wilson_interval(successes, len(scored))


def primary_of(annotator: Annotator, index: int) -> str:
    """An annotator's primary hub for one item, empty if unanswered.

    A single accessor because `AnswerRow` is a TypedDict and the
    dict.get(index, {}).get(...) idiom does not type-check against one.
    """
    row = annotator.answers.get(index)
    return row["primary_hub_id"] if row is not None else ""


def _agreement(
    left: Annotator, right: Annotator, indices: list[int]
) -> tuple[WilsonInterval | None, int, int]:
    """Rank-1 agreement between two annotators over *indices*.

    Restricted to items both actually answered. An item one of them skipped
    is dropped rather than counted as a disagreement, because a missing
    answer is not a judgement and scoring it as one would understate
    agreement in proportion to transport failures.
    """
    both = [
        index
        for index in indices
        if primary_of(left, index) and primary_of(right, index)
    ]
    if not both:
        return None, 0, 0
    hits = sum(1 for index in both if primary_of(left, index) == primary_of(right, index))
    return wilson_interval(hits, len(both)), hits, len(both)


def majority_vote(panel: list[Annotator], index: int) -> tuple[str, int]:
    """The panel's modal primary hub for one item, and how many backed it.

    Ties are broken by the model order in PANEL_MODELS rather than at random,
    so the report is reproducible. A 1-1-1 split therefore resolves to the
    first-listed model's answer and is reported separately as unanimity 1, so
    a reader can see that the "majority" carried a single vote.
    """
    votes = [
        primary_of(annotator, index)
        for annotator in panel
        if primary_of(annotator, index)
    ]
    if not votes:
        return "", 0
    counts = Counter(votes)
    best = max(counts.values())
    for vote in votes:  # votes are in panel order, so first-listed wins ties
        if counts[vote] == best:
            return vote, best
    return "", 0


def _distance_table(
    annotator_hub: dict[int, str],
    reference_hub: dict[int, str],
    indices: list[int],
    paths: dict[str, list[str]],
) -> tuple[dict[DistanceCategory, int], int]:
    """Disagreement counts by hierarchy distance, over items both answered."""
    counts: Counter[DistanceCategory] = Counter()
    total = 0
    for index in indices:
        left = annotator_hub.get(index, "")
        right = reference_hub.get(index, "")
        if not left or not right or left == right:
            continue
        total += 1
        counts[distance_category(left, right, paths)] += 1
    return dict(counts), total


def _render_distance(counts: dict[DistanceCategory, int], total: int) -> str:
    if total == 0:
        return "no disagreements to categorise"
    parts = []
    for category in DISAGREEMENT_CATEGORIES:
        n = counts.get(category, 0)
        parts.append(f"{category} {n} ({100 * n / total:.1f}%)")
    unknown = counts.get("unknown", 0)
    if unknown:
        parts.append(f"unknown {unknown}")
    return ", ".join(parts)


def _scored_for(
    annotator: Annotator,
    key: dict[int, list[str]],
    metadata: dict[int, ItemMeta],
) -> list[ScoredItem]:
    """alpha-1 / alpha-5 rows, via the human study's own scorer."""
    meta_tuples = {index: (m.framework_id, m.stratum) for index, m in metadata.items()}
    return score_items(list(annotator.answers.values()), key, meta_tuples)


def _section_per_model_alpha(
    panel: list[Annotator],
    human: Annotator,
    key: dict[int, list[str]],
    metadata: dict[int, ItemMeta],
) -> list[str]:
    lines = [
        "## 1. Agreement with OpenCRE (alpha-1 and alpha-5)",
        "",
        "Wilson 95% intervals, from `tract.stats.wilson_interval`. Scored by",
        "`scripts.score_ceiling_study.score_items`, unmodified, so these numbers",
        "are produced by the same code that produced the human's 0.572.",
        "",
    ]
    everyone = [human, *panel]
    strata = sorted({m.stratum for m in metadata.values()})
    frameworks = sorted({m.framework_id for m in metadata.values()})

    lines.append("| annotator | scope | alpha-1 | alpha-5 |")
    lines.append("|---|---|---|---|")
    for annotator in everyone:
        scored = _scored_for(annotator, key, metadata)
        label = annotator.name
        lines.append(
            f"| {label} | pooled | {_fmt(_alpha(scored, 'alpha1_hit'), len(scored))} "
            f"| {_fmt(_alpha(scored, 'alpha5_hit'), len(scored))} |"
        )
        for stratum in strata:
            subset = [row for row in scored if row["stratum"] == stratum]
            lines.append(
                f"| {label} | stratum: {stratum} | "
                f"{_fmt(_alpha(subset, 'alpha1_hit'), len(subset))} | "
                f"{_fmt(_alpha(subset, 'alpha5_hit'), len(subset))} |"
            )
    lines.append("")

    lines.append("### Per framework, alpha-1")
    lines.append("")
    lines.append("| annotator | " + " | ".join(frameworks) + " |")
    lines.append("|---" * (len(frameworks) + 1) + "|")
    for annotator in everyone:
        scored = _scored_for(annotator, key, metadata)
        cells = []
        for framework in frameworks:
            subset = [row for row in scored if row["framework_id"] == framework]
            cells.append(_fmt(_alpha(subset, "alpha1_hit"), len(subset)))
        lines.append(f"| {annotator.name} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("### Per framework, alpha-5")
    lines.append("")
    lines.append("| annotator | " + " | ".join(frameworks) + " |")
    lines.append("|---" * (len(frameworks) + 1) + "|")
    for annotator in everyone:
        scored = _scored_for(annotator, key, metadata)
        cells = []
        for framework in frameworks:
            subset = [row for row in scored if row["framework_id"] == framework]
            cells.append(_fmt(_alpha(subset, "alpha5_hit"), len(subset)))
        lines.append(f"| {annotator.name} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _section_human_vs_panel(
    panel: list[Annotator],
    human: Annotator,
    metadata: dict[int, ItemMeta],
) -> list[str]:
    all_indices = sorted(metadata)
    capec_indices = [i for i in all_indices if metadata[i].framework_id == CAPEC]

    lines = [
        "## 2. Human versus panel",
        "",
        "The least contaminated comparison in this report. Neither side saw the",
        "other's answers, and neither is derived from OpenCRE. Two annotators",
        "converging on a hub that OpenCRE did not choose is evidence about the",
        "label, not about either annotator.",
        "",
        "| panel member | agreement with human, pooled | on CAPEC |",
        "|---|---|---|",
    ]
    for member in panel:
        pooled, _, pooled_n = _agreement(human, member, all_indices)
        capec, _, capec_n = _agreement(human, member, capec_indices)
        lines.append(
            f"| {member.name} | {_fmt(pooled, pooled_n)} | {_fmt(capec, capec_n)} |"
        )

    if len(panel) >= 2:
        majority = _majority_annotator(panel, all_indices)
        pooled, _, pooled_n = _agreement(human, majority, all_indices)
        capec, _, capec_n = _agreement(human, majority, capec_indices)
        lines.append(
            f"| **panel majority** | **{_fmt(pooled, pooled_n)}** | "
            f"**{_fmt(capec, capec_n)}** |"
        )
    lines.append("")
    return lines


def _majority_annotator(panel: list[Annotator], indices: list[int]) -> Annotator:
    """A synthetic annotator holding the panel's per-item majority vote."""
    answers: dict[int, AnswerRow] = {}
    for index in indices:
        hub, votes = majority_vote(panel, index)
        answers[index] = {
            "item_index": index,
            "primary_hub_id": hub,
            "acceptable_hub_ids": [hub] if hub else [],
            "confidence": "",
            "notes": f"{votes} of {len(panel)} panel members",
        }
    return Annotator("panel majority", "panel", answers, {})


def _section_panel_vs_panel(
    panel: list[Annotator], metadata: dict[int, ItemMeta]
) -> list[str]:
    all_indices = sorted(metadata)
    capec_indices = [i for i in all_indices if metadata[i].framework_id == CAPEC]
    lines = [
        "## 3. Panel versus panel",
        "",
        "Whether disagreement is structural or model-specific. Families that",
        "disagree with OpenCRE in the same direction, while agreeing with each",
        "other, are describing a property of the labels. Families that disagree",
        "with OpenCRE in different directions are describing several different",
        "confusions, and the majority vote over them means much less.",
        "",
        "| pair | pooled | on CAPEC |",
        "|---|---|---|",
    ]
    for left, right in itertools.combinations(panel, 2):
        pooled, _, pooled_n = _agreement(left, right, all_indices)
        capec, _, capec_n = _agreement(left, right, capec_indices)
        lines.append(
            f"| {left.name} vs {right.name} | {_fmt(pooled, pooled_n)} | "
            f"{_fmt(capec, capec_n)} |"
        )
    lines.append("")

    unanimous = sum(
        1
        for index in capec_indices
        if all(primary_of(a, index) for a in panel)
        and len({primary_of(a, index) for a in panel}) == 1
    )
    if capec_indices:
        lines.append(
            f"All {len(panel)} panel members chose the identical hub on "
            f"{unanimous} of {len(capec_indices)} CAPEC items "
            f"({100 * unanimous / len(capec_indices):.1f}%)."
        )
        lines.append("")
    return lines


def _section_capec_contingency(
    panel: list[Annotator],
    human: Annotator,
    primary_gold: dict[int, str],
    valid_gold: dict[int, list[str]],
    metadata: dict[int, ItemMeta],
) -> tuple[list[str], int]:
    """The headline: who agreed with whom on each of the 83 CAPEC items."""
    capec_indices = [i for i in sorted(metadata) if metadata[i].framework_id == CAPEC]
    majority = _majority_annotator(panel, capec_indices)

    cells: Counter[str] = Counter()
    both_against_opencre = 0
    scorable = 0
    for index in capec_indices:
        human_hub = primary_of(human, index)
        panel_hub = primary_of(majority, index)
        if not human_hub or not panel_hub:
            continue
        scorable += 1
        gold = set(valid_gold.get(index, []))
        human_ok = human_hub in gold
        panel_ok = panel_hub in gold
        agree = human_hub == panel_hub

        if human_ok and panel_ok:
            cells["both match OpenCRE"] += 1
        elif human_ok and not panel_ok:
            cells["human matches OpenCRE, panel does not"] += 1
        elif panel_ok and not human_ok:
            cells["panel matches OpenCRE, human does not"] += 1
        elif agree:
            cells["human and panel agree, both differ from OpenCRE"] += 1
            both_against_opencre += 1
        else:
            cells["all three differ"] += 1

    lines = [
        "## 4. The CAPEC three-way contingency",
        "",
        f"All {len(capec_indices)} CAPEC items, {scorable} of them answered by both",
        "the human and at least one panel member. Agreement with OpenCRE is",
        "membership in the item's valid gold set; agreement between human and",
        "panel is an identical primary hub.",
        "",
        "| cell | n | share of scorable |",
        "|---|---|---|",
    ]
    order = [
        "both match OpenCRE",
        "human matches OpenCRE, panel does not",
        "panel matches OpenCRE, human does not",
        "human and panel agree, both differ from OpenCRE",
        "all three differ",
    ]
    for label in order:
        n = cells.get(label, 0)
        share = f"{100 * n / scorable:.1f}%" if scorable else "n/a"
        marker = "**" if label.startswith("human and panel agree") else ""
        lines.append(f"| {marker}{label}{marker} | {marker}{n}{marker} | {share} |")
    lines.append("")
    lines.append(
        f"**Headline: {both_against_opencre} of {scorable} CAPEC items where the "
        "human and the panel majority chose the same hub and OpenCRE chose a "
        "different one.**"
    )
    lines.append("")
    lines.append(
        "Read it against the two hypotheses. Under (b), a single human's "
        f"idiosyncratic reading, this cell should be near zero: {len(panel)} "
        "unrelated model families have no reason to reproduce one person's "
        "private confusion. Every item in it is an item where two independent "
        "readings converged and the published label did not."
    )
    lines.append("")
    return lines, both_against_opencre


def _section_distances(
    panel: list[Annotator],
    human: Annotator,
    primary_gold: dict[int, str],
    valid_gold: dict[int, list[str]],
    metadata: dict[int, ItemMeta],
    paths: dict[str, list[str]],
) -> list[str]:
    all_indices = sorted(metadata)
    capec_indices = [i for i in all_indices if metadata[i].framework_id == CAPEC]

    lines = [
        "## 5. Where the disagreements land in the hierarchy",
        "",
        "Same four categories the human ceiling analysis used, measured the same",
        "way. Only alpha-1 misses are categorised, that is items where the",
        "annotator's primary hub is not in the item's valid gold set, and the",
        "comparison is against the key's `primary_gold_hub_id`. Both details",
        "matter: including items that hit a non-primary gold hub, or comparing",
        "against the whole valid set, changes the split. Measured this way the",
        "human's row below is 9.3 / 3.7 / 17.8 / 69.2, which is what the human",
        "ceiling analysis published, which is what makes the panel rows",
        "comparable to it.",
        "",
        "| annotator | scope | disagreements categorised |",
        "|---|---|---|",
    ]
    everyone = [human, *panel]
    if len(panel) >= 2:
        everyone.append(_majority_annotator(panel, all_indices))
    for annotator in everyone:
        hubs = {
            index: row["primary_hub_id"]
            for index, row in annotator.answers.items()
            if row["primary_hub_id"]
            and row["primary_hub_id"] not in set(valid_gold.get(index, []))
        }
        for scope, indices in (("pooled", all_indices), ("CAPEC", capec_indices)):
            counts, total = _distance_table(hubs, primary_gold, indices, paths)
            lines.append(
                f"| {annotator.name} | {scope} | {_render_distance(counts, total)} |"
            )
    lines.append("")
    lines.append(
        "A disagreement in a different branch is not a near miss. It means the "
        "two readings disagree about what kind of thing the control is, not "
        "about how finely to file it."
    )
    lines.append("")
    return lines


def _section_contamination(
    probe_paths: dict[str, Path],
    name_probe_paths: dict[str, Path],
    valid_gold: dict[int, list[str]],
    metadata: dict[int, ItemMeta],
) -> list[str]:
    """Closed-book recall probe: did the judges memorise OpenCRE?"""
    lines = [
        "## 6. Contamination probe",
        "",
        "### What it tests",
        "",
        "OpenCRE's mappings are published on opencre.org and in a public GitHub",
        "repository. If a panel member memorised them, its agreement with",
        "OpenCRE measures recall, not judgement, and every number in section 1",
        "is inflated. The probe asks each model, with the hub taxonomy withheld,",
        "to state the hub id OpenCRE publishes for each control. Chance is 1 in",
        "522 hubs, so a memorising model is easy to see.",
        "",
        f"The negative control is `{PANEL_CONTAMINATION_PROBE_FRAMEWORK}`'s 2026",
        "edition (`data/processed/frameworks/owasp_llm_top10_2026.json`, fetched",
        "2026-08-16, source document dated August 2026). All three panel members",
        "were released between April and June 2026, so that document postdates",
        "every one of their training cutoffs, and OpenCRE has never mapped it.",
        "Any hub id a model produces for it is confabulation by construction,",
        "which calibrates how readily each model emits a plausible-looking id",
        "when it cannot possibly know.",
        "",
    ]
    if not probe_paths:
        lines.extend([
            "### Result",
            "",
            "**Not run.** No `contamination_probe_*.json` files are present, so no",
            "contamination number is reported. Section 1's agreement figures",
            "should be read as upper bounds until this runs.",
            "",
        ])
        return lines

    lines.append("### Result")
    lines.append("")
    lines.append("| model | scope | items | exact-id recall | ids emitted |")
    lines.append("|---|---|---|---|---|")
    for name, path in sorted(probe_paths.items()):
        data = load_json(path)
        rows = data.get("recalls", [])
        for scope, subset in (
            ("contaminable study items", [r for r in rows if int(r["item_index"]) in valid_gold]),
            ("CAPEC only", [
                r for r in rows
                if metadata.get(int(r["item_index"]), ItemMeta("", "")).framework_id == CAPEC
            ]),
        ):
            if not subset:
                continue
            emitted = sum(1 for r in subset if r.get("recalled_hub_id"))
            hits = sum(
                1
                for r in subset
                if r.get("recalled_hub_id")
                and r["recalled_hub_id"] in set(valid_gold.get(int(r["item_index"]), []))
            )
            interval = wilson_interval(hits, len(subset))
            lines.append(
                f"| {name} | {scope} | {len(subset)} | "
                f"{_fmt(interval, len(subset))} | {emitted} |"
            )
    lines.append("")
    lines.append(
        "Chance recall is 1/522 = 0.0019. A rate indistinguishable from chance "
        "means the panel could not recite OpenCRE's mapping when asked directly."
    )
    lines.append("")

    if name_probe_paths:
        lines.append("### Exposure control: can a judge name a hub it is given the id of?")
        lines.append("")
        lines.append(
            "A judge that emits no hub id above has either never memorised "
            "OpenCRE or is simply obeying the instruction not to guess, and the "
            "mapping arm alone cannot tell those apart. This asks for the NAME "
            "of a hub whose id is supplied, which is a far weaker memory than a "
            "mapping and is checkable against `hub_reference.md`. A judge that "
            "can name hubs but cannot recall mappings saw the taxonomy and not "
            "the links. A judge that can do neither never saw OpenCRE in a form "
            "it retained, and contamination is moot for it."
        )
        lines.append("")
        lines.append("| model | hubs asked | names emitted | names correct |")
        lines.append("|---|---|---|---|")
        for name, path in sorted(name_probe_paths.items()):
            rows = load_json(path).get("hub_names", [])
            emitted = sum(1 for r in rows if r.get("recalled_name"))
            exact = sum(1 for r in rows if r.get("exact_match"))
            lines.append(f"| {name} | {len(rows)} | {emitted} | {exact} |")
        lines.append("")

    lines.append("### What this does and does not establish")
    lines.append("")
    lines.extend([
        "It establishes that the models cannot reproduce OpenCRE's mapping on",
        "demand from the control text, and that they cannot name a hub from its",
        "id either. Those two together are hard to reconcile with a memorised",
        "copy of the mapping table, which is the form of contamination that",
        "would directly inflate section 1.",
        "",
        "It does not establish that the models never saw OpenCRE. Verbatim recall",
        "is a much stronger property than having read a document once, and a",
        "model can be influenced by training exposure it cannot recite. Nor does",
        "it rule out contamination that runs through the frameworks themselves:",
        "CAPEC and CWE entries are heavily represented on the public web, and a",
        "model's sense of what a CAPEC entry is 'about' is shaped by that",
        "exposure whether or not OpenCRE was in the corpus. That channel would",
        "push the panel toward the same reading of a CAPEC entry the human",
        "reached, and this probe cannot bound it.",
        "",
        "The probe also cannot distinguish a model that never saw OpenCRE from",
        "one that saw it and has no addressable memory of six-digit ids, which",
        "are exactly the kind of token string models retain badly. The negative",
        "result is therefore weaker than it looks in one specific way: it is",
        "evidence against id-level memorisation, not against topic-level",
        "familiarity with the taxonomy's shape.",
        "",
    ])
    return lines


def _human_spread(
    human: Annotator, valid_gold: dict[int, list[str]], metadata: dict[int, ItemMeta]
) -> str:
    scored = _scored_for(human, valid_gold, metadata)
    a1 = _alpha(scored, "alpha1_hit")
    a5 = _alpha(scored, "alpha5_hit")
    return f"{a5.point - a1.point:+.3f}" if a1 and a5 else "n/a"


def build_report(
    human: Annotator,
    panel: list[Annotator],
    valid_gold: dict[int, list[str]],
    primary_gold: dict[int, str],
    metadata: dict[int, ItemMeta],
    paths: dict[str, list[str]],
    probe_paths: dict[str, Path],
    name_probe_paths: dict[str, Path],
    missing: list[tuple[str, str]],
) -> str:
    """The whole report. Returns markdown."""
    capec_n = sum(1 for m in metadata.values() if m.framework_id == CAPEC)
    lines: list[str] = [
        "# LLM judge panel: three-way agreement over the ceiling study",
        "",
        "Version 1.0. Owner: Rock Lambros.",
        "",
        "## What this is",
        "",
        "The human ceiling study returned pooled alpha-1 of 0.572, and CAPEC's",
        f"alpha-1 of 0.181 on {capec_n} items against a framework that is 42.8% of",
        "the training graph. This panel exists to discriminate two explanations:",
        "",
        "  (a) OpenCRE's CAPEC links are poor.",
        "  (b) the single human annotator's reading of CAPEC is idiosyncratic.",
        "",
        "Each judge answered the runbook prompt verbatim, in `item_index` order,",
        "with no sight of the human's answers, the other judges' answers, or the",
        "key.",
        "",
    ]

    lines.append("## Panel roster and pins")
    lines.append("")
    lines.append(
        "| model | ran | route | served snapshot | temp | reasoning | cost USD |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for member in panel:
        run = member.provenance
        snapshot = ", ".join(run.get("served_model_snapshot") or []) or "not recorded"
        lines.append(
            f"| `{run.get('model_id', member.name)}` | yes | "
            f"{run.get('route', '?')} (`{run.get('route_model_id', '?')}`) | "
            f"`{snapshot}` | {run.get('temperature', '?')} | "
            f"{run.get('reasoning_effort', '?')} | "
            f"${float(run.get('cost_usd') or 0):.4f} |"
        )
    for name, reason in missing:
        lines.append(f"| `{name}` | **no** | {reason} | | | | |")
    lines.append("")

    total_cost = sum(float(m.provenance.get("cost_usd") or 0) for m in panel)
    failed = sum(int(m.provenance.get("n_failed_attempts") or 0) for m in panel)
    lines.append(
        f"Total panel cost ${total_cost:.4f} across {len(panel)} models, "
        f"including {failed} failed or retried requests."
    )
    lines.append("")
    if len(panel) < len(PANEL_MODELS):
        lines.append(
            f"**A {len(panel)}-model panel is weaker evidence than a "
            f"{len(PANEL_MODELS)}-model one.** Convergence across fewer families "
            "is easier to explain by shared pretraining data, and the majority "
            "vote is correspondingly less meaningful."
        )
        lines.append("")

    lines.append("### Judge competence")
    lines.append("")
    lines.append(
        "Not every panel member is an equally credible judge, and a weak one "
        "should not carry a majority vote unexamined. Two signals, both "
        "independent of the answer key:"
    )
    lines.append("")
    lines.append(
        "| model | invented hub ids | unanswered | alpha-5 minus alpha-1 |"
    )
    lines.append("|---|---|---|---|")
    for member in panel:
        counters = member.provenance.get("parse_counters") or {}
        scored = _scored_for(member, valid_gold, metadata)
        a1 = _alpha(scored, "alpha1_hit")
        a5 = _alpha(scored, "alpha5_hit")
        spread = f"{a5.point - a1.point:+.3f}" if a1 and a5 else "n/a"
        lines.append(
            f"| {member.name} | {int(counters.get('invented_primary') or 0)} | "
            f"{int(counters.get('missing') or 0)} | {spread} |"
        )
    lines.append("")
    lines.append(
        "An invented hub id is one the reference does not contain, so the judge "
        "was not reading the taxonomy it was given. These are recorded as the "
        "judge gave them rather than blanked, because blanking one would drop "
        "the item from the denominator and inflate that model's score."
    )
    lines.append("")
    lines.append(
        "A near-zero alpha-5 minus alpha-1 spread means the judge's shortlist "
        "adds nothing to its first choice, which is what a judge that is "
        "guessing looks like. The human's spread is "
        f"{_human_spread(human, valid_gold, metadata)}."
    )
    lines.append("")

    lines.extend(_section_per_model_alpha(panel, human, valid_gold, metadata))
    lines.extend(_section_human_vs_panel(panel, human, metadata))
    if len(panel) >= 2:
        lines.extend(_section_panel_vs_panel(panel, metadata))
    contingency, _ = _section_capec_contingency(
        panel, human, primary_gold, valid_gold, metadata
    )
    lines.extend(contingency)
    lines.extend(
        _section_distances(panel, human, primary_gold, valid_gold, metadata, paths)
    )
    lines.extend(
        _section_contamination(probe_paths, name_probe_paths, valid_gold, metadata)
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, default=CEILING_STUDY_DIR / "ceiling_items.json")
    parser.add_argument("--key", type=Path, default=CEILING_STUDY_DIR / "ceiling_answer_key.json")
    parser.add_argument(
        "--human", type=Path, default=CEILING_STUDY_DIR / "answers_human_rock.json"
    )
    parser.add_argument(
        "--hub-reference", type=Path, default=CEILING_STUDY_DIR / "hub_reference.md"
    )
    parser.add_argument("--out", type=Path, default=CEILING_STUDY_DIR / "panel_agreement.md")
    parser.add_argument(
        "--panel-dir", type=Path, default=CEILING_STUDY_DIR,
        help="Directory holding answers_panel_*.json and contamination_probe_*.json.",
    )
    args = parser.parse_args()

    for path in (args.items, args.key, args.human, args.hub_reference):
        if not path.exists():
            print(f"error: file not found: {path}", file=sys.stderr)
            return EXIT_USER_ERROR

    metadata = _load_items_metadata(args.items)
    valid_gold, primary_gold = _load_key(args.key)
    paths = parse_hub_reference(args.hub_reference.read_text(encoding="utf-8"))
    human = _load_annotator(args.human, "human (Rock)", "human")

    panel: list[Annotator] = []
    missing: list[tuple[str, str]] = []
    probe_paths: dict[str, Path] = {}
    name_probe_paths: dict[str, Path] = {}
    for model in PANEL_MODELS:
        slug = model_slug(model)
        answers_path = args.panel_dir / f"answers_panel_{slug}.json"
        if answers_path.exists():
            panel.append(_load_annotator(answers_path, model, "panel"))
        else:
            missing.append((model, f"no {answers_path.name} on disk, run not completed"))
        probe_path = args.panel_dir / f"contamination_probe_{slug}.json"
        if probe_path.exists():
            probe_paths[model] = probe_path
        name_probe_path = args.panel_dir / f"hub_name_probe_{slug}.json"
        if name_probe_path.exists():
            name_probe_paths[model] = name_probe_path

    if not panel:
        print(
            "error: no panel answer files found in "
            f"{args.panel_dir}. Run scripts.run_panel first.",
            file=sys.stderr,
        )
        return EXIT_USER_ERROR

    report = build_report(
        human,
        panel,
        valid_gold,
        primary_gold,
        metadata,
        paths,
        probe_paths,
        name_probe_paths,
        missing,
    )
    atomic_write_text(report, args.out)
    print(f"wrote {args.out} ({len(panel)} of {len(PANEL_MODELS)} panel models)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
