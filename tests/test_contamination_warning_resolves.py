"""A guardrail that names a path which does not resolve is not a guardrail.

`CLAUDE.md`, `scripts/build_curation_packet.py` and
`tests/test_curation_packet.py` all warn against sending
`results/review/hub_reference.md` to an annotator, because 400 of its hub
descriptions are LLM-written from the gold links and would make the round
Tier 3 under `results/phase1b/CAMPAIGN3.md` §2.

That path does not exist. `ls results/review/` returns hub_reference.**json**
(clean -- `PROVENANCE.md` records it as derived from cre_hierarchy.json with no
model output). The file the warning is about lives at
`results/ceiling_study/hub_reference.md`, 422 KB, whose own header reads
"Expert descriptions are shown where one exists (the 400 leaf hubs that have
been through description review)".

So an operator who greps the documented path finds nothing and concludes the
hazard was already removed, while the hazard sits one directory over. A warning
spelled against a non-resolving path is indistinguishable from one that is
working -- which is worse than no warning, because it stops the search.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Every tracked file that warns about the hazard, and must name it correctly.
WARNING_SITES = (
    PROJECT_ROOT / "CLAUDE.md",
    PROJECT_ROOT / "scripts" / "build_curation_packet.py",
    PROJECT_ROOT / "tests" / "test_curation_packet.py",
)

HAZARD = PROJECT_ROOT / "results" / "ceiling_study" / "hub_reference.md"
DEAD_PATH = "results/review/hub_reference.md"


class TestTheHazardFileIsNamedCorrectly:

    @pytest.mark.parametrize("site", WARNING_SITES, ids=lambda p: p.name)
    def test_no_site_names_the_non_resolving_path(self, site: Path) -> None:
        if not site.is_file():
            pytest.skip(f"{site.name} absent")
        text = site.read_text(encoding="utf-8")
        assert DEAD_PATH not in text, (
            f"{site.name} warns against {DEAD_PATH}, which does not exist. "
            f"The hazard is at {HAZARD.relative_to(PROJECT_ROOT)}. An operator "
            "who greps the documented path finds nothing and stops looking."
        )

    @pytest.mark.parametrize("site", WARNING_SITES, ids=lambda p: p.name)
    def test_every_hub_reference_path_named_resolves(self, site: Path) -> None:
        if not site.is_file():
            pytest.skip(f"{site.name} absent")
        named = re.findall(r"results/[\w/]*hub_reference\.\w+", 
                           site.read_text(encoding="utf-8"))
        for path in set(named):
            assert (PROJECT_ROOT / path).exists(), (
                f"{site.name} names {path}, which does not resolve"
            )

    def test_the_hazard_file_is_untracked_but_present(self) -> None:
        # If it is ever removed the warnings become archaeology; if it is ever
        # tracked, the licensed-text and Tier-3 gates need to cover it.
        if not HAZARD.is_file():
            pytest.skip("ceiling study not generated on this checkout")
        assert HAZARD.stat().st_size > 100_000
