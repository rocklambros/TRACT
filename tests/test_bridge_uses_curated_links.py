"""The bridge pipeline read the pre-curation link set.

`tract bridge` passed `data/training/hub_links_by_framework.json` (2026-04-28,
raw) while `hub_links_by_framework_curated.json` (2026-04-29, post-audit) sat
beside it and is referenced thirteen lines further down the same function.

The two differ:

    hub_links_by_framework.json          83 AI hubs, 463 hubs total
    hub_links_by_framework_curated.json  78 AI hubs, 458 hubs total

Everything else in the project -- training, evaluation, the gate denominator,
`load_curated_links()` -- uses the curated set. So the Phase 2B bridge analysis,
the hub classification it recorded, and the counts published on the model card
were all computed against links the audit had already changed.

This is the same defect class as the degree statistic measured on the
post-audit graph: a number taken from whichever file was nearest rather than
from the one the rest of the system uses.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING = PROJECT_ROOT / "data" / "training"
RAW = TRAINING / "hub_links_by_framework.json"
CURATED = TRAINING / "hub_links_by_framework_curated.json"


def _ai_and_total(path: Path) -> tuple[int, int]:
    from tract.config import BRIDGE_AI_FRAMEWORK_IDS
    data = json.loads(path.read_text(encoding="utf-8"))
    ai: set[str] = set()
    every: set[str] = set()
    for framework_id, links in data.items():
        for link in links:
            every.add(link["cre_id"])
            if framework_id in BRIDGE_AI_FRAMEWORK_IDS:
                ai.add(link["cre_id"])
    return len(ai), len(every)


class TestTheCliPassesTheCuratedSet:

    def test_bridge_command_does_not_name_the_raw_file(self) -> None:
        source = (PROJECT_ROOT / "tract" / "cli.py").read_text(encoding="utf-8")
        # The curated name contains the raw name as a prefix, so match the
        # exact quoted literal rather than a substring.
        assert '"hub_links_by_framework.json"' not in source, (
            "tract bridge is reading the pre-curation link set. Everything "
            "else in the project uses the curated one, so the bridge analysis "
            "and the counts it publishes describe a corpus the audit changed."
        )

    def test_bridge_command_names_the_curated_file(self) -> None:
        source = (PROJECT_ROOT / "tract" / "cli.py").read_text(encoding="utf-8")
        assert '"hub_links_by_framework_curated.json"' in source


@pytest.mark.skipif(not (RAW.is_file() and CURATED.is_file()),
                    reason="link files absent")
class TestTheTwoFilesReallyDiffer:
    """If they ever converge this test says so, and the fix becomes moot."""

    def test_the_raw_set_is_larger(self) -> None:
        assert _ai_and_total(RAW) == (83, 463)

    def test_the_curated_set_is_the_post_audit_one(self) -> None:
        assert _ai_and_total(CURATED) == (78, 458)

    def test_the_curated_grouped_file_agrees_with_the_curated_jsonl(
        self,
    ) -> None:
        """The grouped and line-delimited curated files must describe one corpus."""
        from scripts.phase0.common import AI_FRAMEWORK_ID_MAP, load_curated_links
        extra = {"ENISA": "enisa", "ETSI": "etsi", "BIML": "biml"}
        from tract.config import BRIDGE_AI_FRAMEWORK_IDS
        by_hub: dict[str, set[str]] = defaultdict(set)
        for link in load_curated_links():
            fid = AI_FRAMEWORK_ID_MAP.get(link.standard_name) or extra.get(
                link.standard_name)
            by_hub[link.cre_id].add(fid or "")
        jsonl_ai = {h for h, f in by_hub.items()
                    if f & set(BRIDGE_AI_FRAMEWORK_IDS)}
        grouped_ai, _ = _ai_and_total(CURATED)
        assert len(jsonl_ai) == grouped_ai == 78
