"""The OWASP LLM Top 10 2026 corpus must reach no training or fold roster.

BGE-large-v1.5 was pretrained before this document existed, so it is the one
corpus in the project that can separate an encoder mapping meaning from an
encoder recalling text it saw in pretraining. That property survives exactly
as long as the document stays out of training. One roster entry destroys it,
and the loss is silent: every metric still computes, and the contamination
control quietly stops controlling for anything.

So the guard is not a comment. It is a sweep over the module-level constants
that define rosters, plus the curated link file, plus the merge step that
would otherwise pull the framework into the corpus every trainer reads.

The sweep is checked in both directions. `test_the_sweep_flags_a_planted_id`
plants the id in a fake namespace and asserts the helper reports it, because a
gate nobody has seen fire is a gate nobody knows works.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tract.config import (
    FRAMEWORK_LICENSES,
    HOLDOUT_FRAMEWORK_IDS,
    TRAINING_DIR,
)

HOLDOUT_ID = "owasp_llm_top10_2026"
# The names a link or a roster could carry for this framework. The id is the
# machine spelling; the others are the human ones a roster written by hand
# would plausibly use.
HOLDOUT_NAMES: frozenset[str] = frozenset({
    HOLDOUT_ID,
    "OWASP Top 10 for LLM Applications 2026",
    "OWASP Top10 for LLM 2026",
})

# Constants that must name the holdout, with the reason each one is exempt.
# Anything not listed here is swept.
EXEMPT: dict[str, str] = {
    "HOLDOUT_FRAMEWORK_IDS": "this is the list itself",
    "FRAMEWORK_LICENSES": (
        "every framework with a processed artifact must carry a licence, "
        "holdout included, and tests/test_framework_licenses.py enforces it"
    ),
}


def _strings_in(value: Any) -> list[str]:
    """Every string reachable one level inside a container constant."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [
            s for item in list(value.keys()) + list(value.values())
            for s in _strings_in(item)
        ]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [s for item in value for s in _strings_in(item)]
    return []


def roster_offenders(module: ModuleType | dict[str, Any]) -> list[str]:
    """Module-level constants that name the holdout framework.

    Returns the attribute names, so a failure says which roster to go and fix
    rather than only that one exists.
    """
    namespace = module if isinstance(module, dict) else vars(module)
    offenders = []
    for name, value in namespace.items():
        if name.startswith("_") or name in EXEMPT:
            continue
        if HOLDOUT_NAMES & set(_strings_in(value)):
            offenders.append(name)
    return sorted(offenders)


class TestTheSweepCanFire:
    """Ledger lesson 3: compute the attainable range and assert it fires."""

    def test_the_sweep_flags_a_planted_id(self) -> None:
        planted = {
            "SOME_FOLD_ROSTER": ["mitre_atlas", HOLDOUT_ID],
            "UNRELATED": ["capec"],
        }

        assert roster_offenders(planted) == ["SOME_FOLD_ROSTER"]

    def test_the_sweep_flags_a_planted_display_name(self) -> None:
        planted = {"FOLD_FRAMEWORKS": ["OWASP Top 10 for LLM Applications 2026"]}

        assert roster_offenders(planted) == ["FOLD_FRAMEWORKS"]

    def test_the_sweep_reaches_inside_a_mapping(self) -> None:
        planted = {"ALIASES": {"owasp top10 for llm 2026": HOLDOUT_ID}}

        assert roster_offenders(planted) == ["ALIASES"]

    def test_the_sweep_ignores_the_2025_edition(self) -> None:
        """The two ids share a prefix and are different frameworks."""
        planted = {"FOLD_FRAMEWORKS": ["owasp_llm_top10", "OWASP Top10 for LLM"]}

        assert roster_offenders(planted) == []


class TestNoRosterNamesTheHoldout:
    def test_the_holdout_list_names_the_2026_framework(self) -> None:
        assert HOLDOUT_ID in HOLDOUT_FRAMEWORK_IDS

    def test_it_still_carries_a_recorded_licence(self) -> None:
        """Held out of training is not held out of the licensing record."""
        assert FRAMEWORK_LICENSES[HOLDOUT_ID] == "CC-BY-SA-4.0"

    def test_no_config_constant_names_it(self) -> None:
        import tract.config as config

        assert roster_offenders(config) == []

    def test_no_lofo_fold_roster_names_it(self) -> None:
        from scripts.phase1b import runpod_parallel

        assert roster_offenders(runpod_parallel) == []

    def test_neither_lofo_split_holds_it_out(self) -> None:
        """fold_roster() is what provision and run actually read."""
        from scripts.phase1b.runpod_parallel import fold_roster

        assert not HOLDOUT_NAMES & set(fold_roster("test"))
        assert not HOLDOUT_NAMES & set(fold_roster("validation"))

    def test_the_ceiling_study_does_not_sample_it(self) -> None:
        from tract.ceiling_study import eligible_framework_ids

        assert HOLDOUT_ID not in eligible_framework_ids()

    def test_opencre_extraction_does_not_claim_it(self) -> None:
        """A framework in this set gets sections extracted from OpenCRE."""
        from tract.config import (
            AI_PARSER_FRAMEWORK_IDS,
            OPENCRE_EXTRACT_FRAMEWORK_IDS,
            OPENCRE_FRAMEWORK_ID_MAP,
        )

        assert HOLDOUT_ID not in AI_PARSER_FRAMEWORK_IDS
        assert HOLDOUT_ID not in OPENCRE_EXTRACT_FRAMEWORK_IDS
        assert HOLDOUT_ID not in set(OPENCRE_FRAMEWORK_ID_MAP.values())


class TestNoCuratedTrainingLinkNamesIt:
    """The 2026 edition has no OpenCRE links and must never acquire one.

    All 13 OpenCRE links for the OWASP LLM Top 10 key to the 2025 ids. A 2026
    link would put the holdout's prose on a training anchor.
    """

    @pytest.mark.parametrize("filename", [
        "hub_links_curated.jsonl", "hub_links.jsonl",
    ])
    def test_no_link_file_references_the_holdout(self, filename: str) -> None:
        path = TRAINING_DIR / filename
        if not path.exists():
            pytest.skip(f"{filename} is not present in this checkout")

        offenders = []
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            link = json.loads(line)
            if HOLDOUT_NAMES & {
                str(link.get("framework_id", "")),
                str(link.get("standard_name", "")),
            }:
                offenders.append(number)

        assert not offenders, (
            f"{filename} lines {offenders} link the pretraining-contamination "
            f"holdout. Remove them, or the control stops controlling."
        )

    def test_no_by_framework_index_references_the_holdout(self) -> None:
        path = TRAINING_DIR / "hub_links_by_framework_curated.json"
        if not path.exists():
            pytest.skip("hub_links_by_framework_curated.json is not present")

        index = json.loads(path.read_text(encoding="utf-8"))

        assert not HOLDOUT_NAMES & set(index)


class TestTheMergeExcludesIt:
    """The merged corpus is the other way this prose reaches a trainer.

    Ledger lesson 1: guarding one channel while another stays open. Every
    roster could be clean and the framework would still arrive in
    all_controls.json, which is what the prose index reads.
    """

    @staticmethod
    def _framework(framework_id: str, control_id: str) -> dict[str, object]:
        return {
            "framework_id": framework_id,
            "framework_name": framework_id,
            "version": "1",
            "source_url": "https://example.invalid",
            "fetched_date": "2026-08-16",
            "mapping_unit_level": "control",
            "controls": [{
                "control_id": control_id,
                "title": "Example",
                "description": "A synthetic control statement for this test.",
            }],
        }

    def test_the_holdout_is_absent_from_both_merged_corpora(
        self, tmp_path: Path,
    ) -> None:
        from parsers.merge_all_controls import MERGED_FILENAME, main

        frameworks = tmp_path / "frameworks"
        frameworks.mkdir()
        for framework_id, control_id in (
            ("capec", "CAPEC-1"), (HOLDOUT_ID, "LLM01:2026"),
        ):
            (frameworks / f"{framework_id}.json").write_text(
                json.dumps(self._framework(framework_id, control_id)),
                encoding="utf-8",
            )
        out = tmp_path / "out"
        out.mkdir()
        licensed = tmp_path / "licensed"
        licensed.mkdir()

        main(frameworks_dir=frameworks, output_dir=out, licensed_dir=licensed)
        merged = json.loads(
            (out / MERGED_FILENAME).read_text(encoding="utf-8")
        )

        assert [f["framework_id"] for f in merged["frameworks"]] == ["capec"]
        assert not (licensed / MERGED_FILENAME).exists()

    def test_the_holdout_alone_does_not_produce_an_empty_corpus(
        self, tmp_path: Path,
    ) -> None:
        """Excluding everything is a different failure from excluding one."""
        from parsers.merge_all_controls import main

        frameworks = tmp_path / "frameworks"
        frameworks.mkdir()
        (frameworks / f"{HOLDOUT_ID}.json").write_text(
            json.dumps(self._framework(HOLDOUT_ID, "LLM01:2026")),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="holdout"):
            main(
                frameworks_dir=frameworks,
                output_dir=tmp_path / "out",
                licensed_dir=tmp_path / "licensed",
            )
