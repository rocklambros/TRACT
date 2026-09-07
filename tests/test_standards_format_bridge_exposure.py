"""`path+name+standards` is a bridge-reaches-evaluation path, and the firewall
will not catch it. Today it is dead code; these tests make sure it cannot be
revived quietly.

THE MECHANISM. `build_firewalled_hub_text(include_standards=True)` appends the
names of standard sections linked to a hub, dropping only those belonging to
the held-out framework. Phase 2C adds traditional-control -> AI-hub links. After
it lands, an AI hub carries NIST 800-53 (and similar) sections.

Now hold out MITRE ATLAS and evaluate. Those NIST sections are not ATLAS, so:

  - the exclusion filter correctly keeps them, and
  - `assert_firewall` correctly passes, because no ATLAS text leaked.

No breach is raised, and yet bridge-derived text has entered the hub
representations the fold is scored against. The firewall is doing exactly what
it is specified to do. The specification forbids HELD-OUT-FRAMEWORK leakage; it
does not forbid the supervision under test from reaching the evaluation. Those
are different properties and only the first one has an assertion.

WHY THIS IS NOT AN INCIDENT TODAY. Nothing supplies `standard_sections`. The
only non-test reference passes it straight through, so `run_single_fold` raises
on the format it is asked for, and the declared A3 ablation arm in
`scripts/phase1b/ablation.py` has never run. The exposure is entirely
prospective, which is the cheapest moment to fence it.

WHAT THESE TESTS DO. They pin the dead state, they demonstrate that the firewall
returns clean on the bridge shape so nobody assumes otherwise, and they go red
the day someone wires the format up -- pointing at the bridge-exclusion rule
that has to be designed first.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import pytest

from tract.config import PROJECT_ROOT
from tract.training.firewall import assert_firewall, build_all_hub_texts

# The single legitimate reference: orchestrate forwards its own parameter to the
# builder. Anything else is a caller that supplies sections, which is the event
# these tests exist to catch.
ALLOWED_PASSTHROUGH: Final[frozenset[str]] = frozenset(
    {"tract/training/orchestrate.py", "tract/training/firewall.py"}
)

SUPPLIES_SECTIONS: Final[re.Pattern[str]] = re.compile(
    r"standard_sections\s*=\s*(?!None\b)"
)


class TestTheFormatIsStillUnreachable:
    def test_no_production_caller_supplies_standard_sections(self) -> None:
        """Source inspection is the right instrument: the finding IS the absence.

        If this goes red, someone wired the standards format up. Before it can
        ship, the bridge-exclusion rule described in this module's docstring has
        to exist, because the firewall will not raise on the leak it creates.
        """
        offenders: list[str] = []
        for path in sorted(PROJECT_ROOT.rglob("*.py")):
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            if rel.startswith(("tests/", ".venv/", "wandb/", "build/")):
                continue
            if rel in ALLOWED_PASSTHROUGH:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if SUPPLIES_SECTIONS.search(text):
                offenders.append(rel)
        assert not offenders, (
            "These files now supply standard_sections, which makes "
            "hub_rep_format='path+name+standards' reachable. After Phase 2C "
            "that puts bridge-derived traditional sections into AI hub text at "
            "evaluation time, and assert_firewall does NOT raise on it. Design "
            "the bridge-exclusion rule before enabling this: " + repr(offenders)
        )

    def test_the_declared_a3_ablation_arm_names_the_unreachable_format(
        self,
    ) -> None:
        """The arm is configured, so a reader could assume it produced results.

        It did not. It raises. Pinned so the ablation table is never read as
        containing an A3 number.
        """
        ablation = PROJECT_ROOT / "scripts" / "phase1b" / "ablation.py"
        text = ablation.read_text(encoding="utf-8")
        assert 'hub_rep_format="path+name+standards"' in text
        assert "ablation_A3_standards" in text


class TestTheFirewallDoesNotCoverThisShape:
    """Demonstrate the gap rather than describing it."""

    class Item:
        def __init__(self, text: str, framework: str) -> None:
            self.control_text = text
            self.framework = framework

    @pytest.fixture()
    def hierarchy(self):  # type: ignore[no-untyped-def]
        """Same mini fixture the firewall suite uses, for the same reason.

        A fixture rather than the real hierarchy: this asserts a property of
        the builder, and it must not start depending on which hubs happen to be
        in data/processed today.
        """
        import json

        from tract.hierarchy import CREHierarchy

        fixture = Path(__file__).parent / "fixtures" / "phase1a_mini_cres.json"
        data = json.loads(fixture.read_text(encoding="utf-8"))
        return CREHierarchy.from_opencre(
            cres=data["cres"],
            fetch_timestamp=data["fetch_timestamp"],
            data_hash="abc123",
        )

    def test_bridge_derived_traditional_text_passes_the_firewall_clean(
        self, hierarchy
    ) -> None:  # type: ignore[no-untyped-def]
        """Hold out an AI framework; attach a traditional section to a hub.

        This is exactly the post-Phase-2C shape. The firewall returns clean:
        the leaked-in text belongs to NIST 800-53, not to the held-out
        framework, so there is no breach to find. The point of the test is that
        it PASSES -- that is the finding.
        """
        hub_id = next(iter(hierarchy.hubs))
        bridged = "The organization employs boundary protection mechanisms"
        sections = {hub_id: [f"NIST 800-53: {bridged}"]}

        hub_texts = build_all_hub_texts(
            hierarchy,
            excluded_framework="MITRE ATLAS",
            include_standards=True,
            standard_sections=sections,
        )
        base_hub_texts = build_all_hub_texts(
            hierarchy, excluded_framework="MITRE ATLAS", include_standards=False,
        )

        assert hub_texts[hub_id] != base_hub_texts[hub_id], (
            "the bridged section must actually be appended, or this test is "
            "asserting nothing"
        )
        assert bridged in hub_texts[hub_id]

        # An ordinary ATLAS eval item. No breach is raised, and the traditional
        # text is sitting in the hub representation it will be scored against.
        items = [self.Item("Adversarial ML model evasion", "MITRE ATLAS")]
        assert_firewall(
            hub_texts, items, "MITRE ATLAS", base_hub_texts=base_hub_texts,
        )

    def test_the_same_shape_from_the_held_out_framework_does_raise(
        self, hierarchy
    ) -> None:  # type: ignore[no-untyped-def]
        """The contrast that makes the point above meaningful.

        The firewall is not broken. It catches held-out-framework text in the
        same position. It is scoped to a different property than the one Phase
        2C needs.
        """
        hub_id = next(iter(hierarchy.hubs))
        leaked = "Adversarial ML model evasion via crafted inputs"
        sections = {hub_id: [f"ATLAS-mislabelled: {leaked}"]}

        hub_texts = build_all_hub_texts(
            hierarchy,
            excluded_framework="MITRE ATLAS",
            include_standards=True,
            standard_sections=sections,
        )
        base_hub_texts = build_all_hub_texts(
            hierarchy, excluded_framework="MITRE ATLAS", include_standards=False,
        )
        items = [self.Item(leaked, "MITRE ATLAS")]
        with pytest.raises(AssertionError, match="Firewall breach"):
            assert_firewall(
                hub_texts, items, "MITRE ATLAS", base_hub_texts=base_hub_texts,
            )
