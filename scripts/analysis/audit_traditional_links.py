"""Audit traditional (non-AI) framework links for data quality issues.

Analyzes hub_links_curated.jsonl to quantify:
- Bare-ID vs descriptive section names
- Per-framework quality breakdown
- Link type distribution (LinkedTo vs AutomaticallyLinkedTo)
- Hub concentration
- Section name length distribution
"""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Final

CURATED_PATH: Final[Path] = Path("/home/rock/github_projects/TRACT/data/training/hub_links_curated.jsonl")

AI_FRAMEWORKS: Final[frozenset[str]] = frozenset({
    "MITRE ATLAS",
    "NIST AI 100-2",
    "OWASP AI Exchange",
    "OWASP Top10 for LLM",
    "OWASP Top10 for ML",
})

# Patterns for bare-ID section names (no semantic content)
BARE_ID_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"^CWE-\d+$"),
    re.compile(r"^CAPEC-\d+$"),
    re.compile(r"^[A-Z]{1,4}-?\d+(\.\d+)*$"),         # e.g. "AC-1", "V1.1.1", "CM-2"
    re.compile(r"^\d+(\.\d+)*$"),                       # e.g. "1.2.3"
    re.compile(r"^[A-Z]+\d+(\.\d+)*-[A-Z]+$"),         # coded patterns
]

# Patterns for section names that include an ID prefix but also have descriptive text
ID_PLUS_DESC_PATTERN = re.compile(
    r"^("
    r"[A-Z]{1,4}-?\d+(\.\d+)*\s+"            # ID prefix like "CM-2 ", "AC-1 "
    r"|V\d+(\.\d+)*\s+"                       # ASVS prefix like "V1.1.1 "
    r"|\d+(\.\d+)*\s+"                        # Numeric prefix like "1.2 "
    r")"
)


def is_bare_id(name: str) -> bool:
    """Check if a section name is just an identifier with no semantic content."""
    name = name.strip()
    if not name:
        return True
    if len(name) <= 3:
        return True
    for pat in BARE_ID_PATTERNS:
        if pat.match(name):
            return True
    return False


def classify_section_name(name: str) -> str:
    """Classify a section name into quality categories."""
    name = name.strip()
    if not name:
        return "empty"
    if is_bare_id(name):
        return "bare_id"
    if len(name) <= 10:
        return "short"
    if ID_PLUS_DESC_PATTERN.match(name):
        return "id_plus_description"
    return "descriptive"


def load_traditional_links() -> list[dict]:
    links = []
    with open(CURATED_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["standard_name"] not in AI_FRAMEWORKS:
                links.append(rec)
    return links


def main() -> None:
    links = load_traditional_links()
    print(f"Total traditional links: {len(links)}")
    print(f"Unique frameworks: {len(set(l['standard_name'] for l in links))}")
    print()

    # ── 1. Per-framework breakdown ──────────────────────────────────
    print("=" * 90)
    print("1. PER-FRAMEWORK BREAKDOWN")
    print("=" * 90)
    print()

    fw_groups: dict[str, list[dict]] = defaultdict(list)
    for l in links:
        fw_groups[l["standard_name"]].append(l)

    # Collect for summary
    fw_stats: list[dict] = []

    for fw in sorted(fw_groups.keys(), key=lambda k: -len(fw_groups[k])):
        fw_links = fw_groups[fw]
        total = len(fw_links)

        linked_to = sum(1 for l in fw_links if l["link_type"] == "LinkedTo")
        auto_linked = sum(1 for l in fw_links if l["link_type"] == "AutomaticallyLinkedTo")
        other_type = total - linked_to - auto_linked

        classifications = Counter(classify_section_name(l["section_name"]) for l in fw_links)
        descriptive = classifications.get("descriptive", 0)
        id_plus_desc = classifications.get("id_plus_description", 0)
        bare_id = classifications.get("bare_id", 0)
        short = classifications.get("short", 0)
        empty = classifications.get("empty", 0)

        useful = descriptive + id_plus_desc
        pct_useful = useful / total * 100 if total > 0 else 0

        fw_stats.append({
            "framework": fw,
            "total": total,
            "linked_to": linked_to,
            "auto_linked": auto_linked,
            "descriptive": descriptive,
            "id_plus_desc": id_plus_desc,
            "bare_id": bare_id,
            "short": short,
            "empty": empty,
            "pct_useful": pct_useful,
        })

        print(f"  {fw} ({total} links)")
        print(f"    Link type: LinkedTo={linked_to}, Auto={auto_linked}" +
              (f", Other={other_type}" if other_type else ""))
        print(f"    Quality:   descriptive={descriptive}, id+desc={id_plus_desc}, "
              f"bare_id={bare_id}, short={short}, empty={empty}")
        print(f"    Useful:    {useful}/{total} ({pct_useful:.1f}%)")
        print()

    # ── 2. Bare-ID analysis ─────────────────────────────────────────
    print("=" * 90)
    print("2. BARE-ID ANALYSIS")
    print("=" * 90)
    print()

    all_classifications = Counter(classify_section_name(l["section_name"]) for l in links)
    total = len(links)
    for cat in ["descriptive", "id_plus_description", "bare_id", "short", "empty"]:
        count = all_classifications.get(cat, 0)
        print(f"  {cat:25s}: {count:5d}  ({count/total*100:5.1f}%)")

    usable = all_classifications.get("descriptive", 0) + all_classifications.get("id_plus_description", 0)
    not_usable = total - usable
    print()
    print(f"  USABLE (descriptive + id+desc):   {usable:5d}  ({usable/total*100:5.1f}%)")
    print(f"  NOT USABLE (bare_id + short + empty): {not_usable:5d}  ({not_usable/total*100:5.1f}%)")
    print()

    # Examples of each category
    for cat in ["descriptive", "id_plus_description", "bare_id", "short", "empty"]:
        examples = [l for l in links if classify_section_name(l["section_name"]) == cat][:3]
        if examples:
            print(f"  Examples of '{cat}':")
            for e in examples:
                print(f"    [{e['standard_name']}] section_name={e['section_name']!r}")
            print()

    # ── 3. Section name length distribution ─────────────────────────
    print("=" * 90)
    print("3. SECTION NAME LENGTH DISTRIBUTION")
    print("=" * 90)
    print()

    lengths = [len(l["section_name"]) for l in links]
    buckets = [0, 5, 10, 20, 30, 50, 100, 200, 500]
    print(f"  Min: {min(lengths)}, Max: {max(lengths)}, "
          f"Mean: {statistics.mean(lengths):.1f}, Median: {statistics.median(lengths):.1f}")
    print()

    print("  Length range    Count    Pct")
    print("  " + "-" * 40)
    for i in range(len(buckets)):
        lo = buckets[i]
        hi = buckets[i + 1] if i + 1 < len(buckets) else float("inf")
        count = sum(1 for x in lengths if lo <= x < hi)
        bar = "#" * max(1, count * 50 // total)
        label = f"{lo}-{int(hi)-1}" if hi != float("inf") else f"{lo}+"
        print(f"  {label:14s}  {count:5d}  ({count/total*100:5.1f}%)  {bar}")
    print()

    # ── 4. Hub concentration ────────────────────────────────────────
    print("=" * 90)
    print("4. HUB CONCENTRATION")
    print("=" * 90)
    print()

    hub_counts = Counter(l["cre_id"] for l in links)
    print(f"  Unique hubs targeted: {len(hub_counts)}")
    print(f"  Links per hub: min={min(hub_counts.values())}, max={max(hub_counts.values())}, "
          f"mean={statistics.mean(hub_counts.values()):.1f}, median={statistics.median(hub_counts.values()):.1f}")
    print()

    print("  Top 15 most-linked hubs:")
    for hub_id, count in hub_counts.most_common(15):
        name = next((l["cre_name"] for l in links if l["cre_id"] == hub_id), "?")
        print(f"    {hub_id} ({name}): {count} links")
    print()

    # Hubs with only bare-ID links
    hub_quality: dict[str, dict[str, int]] = defaultdict(lambda: {"usable": 0, "not_usable": 0})
    for l in links:
        cat = classify_section_name(l["section_name"])
        if cat in ("descriptive", "id_plus_description"):
            hub_quality[l["cre_id"]]["usable"] += 1
        else:
            hub_quality[l["cre_id"]]["not_usable"] += 1

    hubs_all_bad = [h for h, q in hub_quality.items() if q["usable"] == 0]
    hubs_all_good = [h for h, q in hub_quality.items() if q["not_usable"] == 0]
    hubs_mixed = [h for h, q in hub_quality.items() if q["usable"] > 0 and q["not_usable"] > 0]
    print(f"  Hubs with ALL usable links:       {len(hubs_all_good)}")
    print(f"  Hubs with MIXED quality links:     {len(hubs_mixed)}")
    print(f"  Hubs with NO usable links (all bad): {len(hubs_all_bad)}")
    print()

    # ── 5. Samples ──────────────────────────────────────────────────
    print("=" * 90)
    print("5. REPRESENTATIVE SAMPLES")
    print("=" * 90)
    print()

    good_links = [l for l in links if classify_section_name(l["section_name"]) == "descriptive"]
    print("  5 'good' links (fully descriptive):")
    import random
    random.seed(42)
    for l in random.sample(good_links, min(5, len(good_links))):
        print(f"    [{l['standard_name']}] → {l['cre_name']} ({l['cre_id']})")
        print(f"      section: {l['section_name']!r}")
        print(f"      type: {l['link_type']}")
        print()

    bad_links = [l for l in links if classify_section_name(l["section_name"]) in ("bare_id", "short", "empty")]
    print("  5 'bad' links (bare ID / short / empty):")
    for l in random.sample(bad_links, min(5, len(bad_links))):
        print(f"    [{l['standard_name']}] → {l['cre_name']} ({l['cre_id']})")
        print(f"      section: {l['section_name']!r}")
        print(f"      type: {l['link_type']}")
        print()

    # ── 6. Framework quality ranking ────────────────────────────────
    print("=" * 90)
    print("6. FRAMEWORK QUALITY RANKING (by % usable links)")
    print("=" * 90)
    print()

    ranked = sorted(fw_stats, key=lambda s: -s["pct_useful"])
    print(f"  {'Framework':<45s} {'Total':>6s} {'Usable':>7s} {'Pct':>7s} {'LinkTo':>7s} {'Auto':>6s}")
    print("  " + "-" * 80)
    for s in ranked:
        useful = s["descriptive"] + s["id_plus_desc"]
        print(f"  {s['framework']:<45s} {s['total']:>6d} {useful:>7d} {s['pct_useful']:>6.1f}% "
              f"{s['linked_to']:>7d} {s['auto_linked']:>6d}")

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print()
    print(f"  Total traditional links: {len(links)}")
    print(f"  Usable for training: {usable} ({usable/total*100:.1f}%)")
    print(f"  Not usable (bare ID/short/empty): {not_usable} ({not_usable/total*100:.1f}%)")
    print()

    # Break down the "not usable" by framework
    not_usable_by_fw = Counter()
    for l in links:
        cat = classify_section_name(l["section_name"])
        if cat not in ("descriptive", "id_plus_description"):
            not_usable_by_fw[l["standard_name"]] += 1

    print("  Not-usable links by framework:")
    for fw, count in not_usable_by_fw.most_common():
        print(f"    {fw}: {count}")


if __name__ == "__main__":
    main()
