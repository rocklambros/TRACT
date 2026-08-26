"""A parsed framework's prose must actually reach the links that need it.

ISO 27001 shipped a parser producing 93 controls at 0.967 honest prose, and
every one of its 94 curated links resolved to nothing. `ProseIndex` keys the
corpus side on `canonical_framework(framework_name)` and the link side on
`canonical_framework(standard_name)`. OpenCRE says "ISO 27001", the parser says
"ISO/IEC 27001:2022 Annex A", no alias bridged them, and the pipeline fell back
to three-word section titles without a word of complaint.

Prose fraction was measured. Reachability was not. A parser can be correct and
still be wired to nothing, and no existing test could tell the difference. That
is what these two check.

The first test needs no corpus and no data/raw, so it runs everywhere the
repository does. The second measures the real join and is the one that would
catch a break in normalisation rather than in naming.
"""
from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Final

import pytest

from tract.config import (
    OPENCRE_FRAMEWORK_ID_MAP,
    OVERLAY_FRAMEWORK_IDS,
    PARSERS_DIR,
    RESTRICTED_FRAMEWORK_IDS,
    TRAINING_DIR,
)
from tract.text_selection import ProseIndex, canonical_framework, merged_corpus_path

logger = logging.getLogger(__name__)

CURATED_LINKS_PATH: Final[Path] = TRAINING_DIR / "hub_links_curated.jsonl"

# Frameworks that have a parser AND curated links today. Named rather than
# derived so the test cannot go quiet: if the corpus loses one of these, the
# floor assertion below turns red instead of the loop simply having less to do.
# ISO is not here because its prose is restricted and absent from the tracked
# corpus. It is checked wherever the licensed overlay exists.
PARSER_BACKED_WITH_LINKS: Final[frozenset[str]] = frozenset({
    "asvs", "capec", "cwe", "mitre_atlas", "nist_800_53", "nist_ai_100_2",
    "owasp_ai_exchange", "owasp_cheat_sheets", "owasp_llm_top10",
    "owasp_ml_top10",
})

# Measured against the full corpus on 2026-08-16, worst first:
#   iso_27001 92/94 (0.979), nist_800_53 298/300 (0.993), cwe 612/613 (0.998),
#   everything else 1.000.
# The two ISO misses are A.7.8 and A.7.9, whose statements are shorter than
# their own titles plus PROSE_MIN_EXTRA_CHARS, so ProseIndex excludes them on
# purpose. A broken join scores 0.000, so this floor separates the two cases
# with a wide margin and does not need editing every time the corpus is rebuilt.
#
# ISO's 92/94 is no longer only a comment. tests/test_corpus_report.py::
# TestIsoStillResolves asserts it, along with the 91 distinct anchors and the
# 2 controls the prose rule drops, wherever the licensed overlay is present.
MIN_RESOLUTION_RATE: Final[float] = 0.90


def _parser_framework_names() -> dict[str, str]:
    """framework_id -> the framework_name its parser declares.

    Read with ast rather than by importing. Several parsers import defusedxml,
    pdfplumber or datasets, none of which the lint environment installs, and a
    test that cannot run without the ML stack is a test that does not run.
    """
    names: dict[str, str] = {}
    for path in sorted(PARSERS_DIR.glob("parse_*.py")):
        framework_id = path.stem[len("parse_"):]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets: list[ast.Name] = []
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target]
            elif isinstance(node, ast.Assign):
                targets = [t for t in node.targets if isinstance(t, ast.Name)]
            for target in targets:
                if target.id == "framework_name" and isinstance(node.value, ast.Constant):
                    names[framework_id] = str(node.value.value)
    return names


def _link_names_by_framework() -> dict[str, set[str]]:
    """framework_id -> the standard_name spellings its curated links carry."""
    by_framework: dict[str, set[str]] = {}
    with open(CURATED_LINKS_PATH, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            standard_name = str(json.loads(line).get("standard_name", ""))
            framework_id = OPENCRE_FRAMEWORK_ID_MAP.get(standard_name)
            if framework_id is None:
                continue
            by_framework.setdefault(framework_id, set()).add(standard_name)
    return by_framework


def _load_corpus() -> tuple[Path, ProseIndex]:
    """The same corpus the run path loads, resolved the same way.

    merged_corpus_path() rather than a second copy of the rule. A test that
    picks its own corpus can pass while the pipeline reads a different one,
    which is the shape of the defect this file exists to catch.
    """
    path = merged_corpus_path()
    return path, ProseIndex.load(path)


def test_every_parser_backed_link_name_reaches_its_parser_name() -> None:
    """The naming check, offline.

    This is the one that would have caught the ISO defect on a clean checkout
    with no corpus built and no data/raw present. It compares the two strings
    ProseIndex has to make meet.
    """
    parser_names = _parser_framework_names()
    mismatches: list[tuple[str, str, str, str]] = []
    checked = 0
    for framework_id, standard_names in sorted(_link_names_by_framework().items()):
        expected = parser_names.get(framework_id)
        if expected is None:
            continue  # no parser yet; the link legitimately has no prose to reach
        for standard_name in sorted(standard_names):
            checked += 1
            resolved = canonical_framework(standard_name)
            if resolved != expected:
                mismatches.append((framework_id, standard_name, resolved, expected))

    assert checked >= len(PARSER_BACKED_WITH_LINKS), (
        f"only {checked} link-side names were checked against a parser. The "
        f"link file or OPENCRE_FRAMEWORK_ID_MAP changed shape and this test "
        f"stopped covering what it claims to."
    )
    assert not mismatches, (
        f"{len(mismatches)} framework(s) have curated links whose "
        f"standard_name does not canonicalise to their parser's "
        f"framework_name, so ProseIndex can never join them and every link "
        f"falls back to its section title: {mismatches}. Add the missing "
        f"entry to FRAMEWORK_NAME_ALIASES in tract/config.py."
    )


def test_every_parser_backed_framework_resolves_its_links() -> None:
    """The end-to-end check against the real corpus and the real links.

    Naming is one way the join breaks. Section-id normalisation and the prose
    threshold are others, and only a measurement catches those.
    """
    if not CURATED_LINKS_PATH.exists():
        pytest.skip(f"{CURATED_LINKS_PATH} absent")

    corpus_path, index = _load_corpus()
    parser_names = _parser_framework_names()
    frameworks = json.loads(
        corpus_path.read_text(encoding="utf-8")
    ).get("frameworks", [])
    present = {str(framework.get("framework_id")) for framework in frameworks}

    # A framework whose prose this corpus withholds is in the same position as
    # one it omits: ProseIndex indexes a control only when its description
    # exceeds its title, so an overlay member reduced to titles can resolve
    # nothing and 0/214 is the correct answer rather than a broken join. This
    # arrived with rulings R4 and R10, which moved csa_ccm and dsomm into
    # OVERLAY_FRAMEWORK_IDS, so the tracked corpus carries them without prose
    # while the gitignored overlay carries them with it. Derived from the
    # corpus rather than listed, so the same test still demands resolution
    # wherever the overlay is present.
    withheld = {
        str(framework.get("framework_id"))
        for framework in frameworks
        if str(framework.get("framework_id")) in OVERLAY_FRAMEWORK_IDS
        and not any(
            str(control.get("full_text") or "").strip()
            or str(control.get("description") or "").strip()
            != str(control.get("title") or "").strip()
            for control in framework.get("controls") or []
        )
    }
    if withheld:
        logger.info(
            "prose withheld from %s for %s, so their links cannot resolve here",
            corpus_path.name, sorted(withheld),
        )

    totals: dict[str, int] = {}
    resolved: dict[str, int] = {}
    with open(CURATED_LINKS_PATH, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            link = json.loads(line)
            standard_name = str(link.get("standard_name", ""))
            framework_id = OPENCRE_FRAMEWORK_ID_MAP.get(standard_name)
            if framework_id is None or framework_id not in parser_names:
                continue
            if framework_id not in present or framework_id in withheld:
                continue
            totals[framework_id] = totals.get(framework_id, 0) + 1
            if index.lookup(
                standard_name, link.get("section_id"), link.get("section_name"),
            ):
                resolved[framework_id] = resolved.get(framework_id, 0) + 1

    rates = {
        framework_id: resolved.get(framework_id, 0) / total
        for framework_id, total in sorted(totals.items())
    }
    logger.info("prose reachability from %s: %s", corpus_path.name, rates)

    missing = sorted(PARSER_BACKED_WITH_LINKS - set(totals))
    assert not missing, (
        f"{missing} have a parser and curated links but were not measured. "
        f"They are absent from {corpus_path}, which means the corpus lost a "
        f"framework rather than the join being fine."
    )
    starved = {
        framework_id: f"{resolved.get(framework_id, 0)}/{totals[framework_id]}"
        for framework_id, rate in rates.items() if rate < MIN_RESOLUTION_RATE
    }
    assert not starved, (
        f"{len(starved)} parser-backed framework(s) resolve fewer than "
        f"{MIN_RESOLUTION_RATE:.0%} of their curated links through ProseIndex, "
        f"so those links train and evaluate on section titles while the "
        f"parser's prose sits unread: {starved}"
    )


class TestTheRunPathReadsTheCorpusThatHoldsTheProse:
    """A restricted framework's prose only exists in the gitignored overlay.

    merge_all_controls has documented the "prefer the overlay" read order
    since the overlay was introduced, and no caller implemented it. Every
    ProseIndex.load() took the tracked corpus, which excludes every restricted
    framework by construction, so the ISO alias on its own would have moved
    the number in a test and nothing in a run.
    """

    def test_the_overlay_wins_when_it_exists(self, tmp_path: Path) -> None:
        overlay = tmp_path / "licensed" / "all_controls.json"
        overlay.parent.mkdir(parents=True)
        overlay.write_text("{}", encoding="utf-8")
        tracked = tmp_path / "all_controls.json"
        tracked.write_text("{}", encoding="utf-8")

        import tract.text_selection as text_selection

        original_licensed = text_selection.PROCESSED_LICENSED_DIR
        original_processed = text_selection.PROCESSED_DIR
        try:
            text_selection.PROCESSED_LICENSED_DIR = overlay.parent
            text_selection.PROCESSED_DIR = tmp_path
            assert text_selection.merged_corpus_path() == overlay
            overlay.unlink()
            assert text_selection.merged_corpus_path() == tracked
        finally:
            text_selection.PROCESSED_LICENSED_DIR = original_licensed
            text_selection.PROCESSED_DIR = original_processed

    def test_the_default_load_sees_every_restricted_framework_present(self) -> None:
        """On a checkout holding the sources, the run path must see them.

        Skips where the overlay does not exist, because there the restricted
        prose genuinely is not on disk and there is nothing to reach. The
        reachability tests above are the ones that never skip.
        """
        path = merged_corpus_path()
        if path.parent.name != "licensed":
            pytest.skip("no licensed overlay in this checkout")

        present = {
            str(framework.get("framework_id"))
            for framework in json.loads(
                path.read_text(encoding="utf-8")
            ).get("frameworks", [])
        }
        missing = sorted(set(RESTRICTED_FRAMEWORK_IDS) - present)
        assert not missing, (
            f"{missing} are restricted and absent from the overlay the run "
            f"path loads. Re-run parsers/merge_all_controls.py."
        )
