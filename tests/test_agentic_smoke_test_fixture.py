"""The agentic smoke test is only worth running while it stays held out.

`data/eval/agentic_smoke_test.json` carries six OWASP Agentic Top 10 controls
hand-mapped to CRE hubs. Its whole value is that no item has ever been a
training anchor, and that property is one import away from being destroyed
silently: adding `owasp_agentic_top10` links to hub_links_curated.jsonl is an
obviously good idea on its own terms, and it would turn this set into a
memorisation check without changing a single line of the fixture.

So the leakage assertion is the point of this file. The rest guards the
fixture's own claims, because a pass condition that can be edited after seeing
a result is not a pre-declared pass condition.

Owner: TRACT
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import pytest

from tract.config import PROCESSED_DIR, PROJECT_ROOT

FIXTURE_PATH: Final[Path] = PROJECT_ROOT / "data" / "eval" / "agentic_smoke_test.json"
CURATED_LINKS: Final[Path] = PROJECT_ROOT / "data" / "training" / "hub_links_curated.jsonl"
FRAMEWORK_ID: Final[str] = "owasp_agentic_top10"

# Six items over four hubs. Recorded here as well as in the fixture so that
# changing one without the other fails rather than passes quietly.
EXPECTED_ITEM_COUNT: Final[int] = 6
EXPECTED_HUB_COUNT: Final[int] = 4


@pytest.fixture(scope="module")
def fixture() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return data


class TestTheSetStaysHeldOut:
    """The one assertion this file exists for."""

    def test_no_item_is_a_training_anchor(self, fixture: dict[str, Any]) -> None:
        """Fails the day someone imports the agentic bridge into training.

        That import is a reasonable thing to want. It is also the thing that
        silently converts this set from a held-out check into a memorisation
        check. When this fails, the fix is to retire the smoke test, not to
        weaken the assertion.
        """
        linked = {
            json.loads(line)["framework_id"]
            for line in CURATED_LINKS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        assert FRAMEWORK_ID not in linked, (
            f"{FRAMEWORK_ID} now has curated training links, so the items in "
            f"{FIXTURE_PATH.name} are no longer held out and the smoke test "
            f"measures memorisation. Retire the smoke test or rebuild it from "
            f"controls that are still unlinked."
        )


class TestTheFixtureResolves:
    """Every item must name a real control and a real hub."""

    def test_shape_matches_its_own_claims(self, fixture: dict[str, Any]) -> None:
        items = fixture["items"]
        assert len(items) == EXPECTED_ITEM_COUNT
        hubs = {i["hub_id"] for i in items}
        assert len(hubs) == EXPECTED_HUB_COUNT
        assert fixture["hub_distribution"] == {
            hub: sum(1 for i in items if i["hub_id"] == hub) for hub in sorted(hubs)
        }

    def test_every_control_exists_and_carries_prose(
        self, fixture: dict[str, Any]
    ) -> None:
        """An item with no prose is an item the model cannot be asked about."""
        source = json.loads(
            (PROCESSED_DIR / "frameworks" / f"{FRAMEWORK_ID}.json").read_text(
                encoding="utf-8"
            )
        )
        controls = {c["control_id"]: c for c in source["controls"]}
        for item in fixture["items"]:
            control = controls.get(item["control_id"])
            assert control is not None, f"{item['control_id']} is not in the corpus"
            text = control.get("full_text") or ""
            assert len(text) > 200, (
                f"{item['control_id']} carries {len(text)} characters of prose. "
                f"The smoke test asks the model to route control text, so an "
                f"item without text cannot be scored."
            )

    def test_every_hub_exists_with_the_recorded_name(
        self, fixture: dict[str, Any]
    ) -> None:
        """Guards against a hub id that drifted out of the hierarchy."""
        hubs = json.loads(
            (PROCESSED_DIR / "cre_hierarchy.json").read_text(encoding="utf-8")
        )["hubs"]
        for item in fixture["items"]:
            hub = hubs.get(item["hub_id"])
            assert hub is not None, f"hub {item['hub_id']} is not in the hierarchy"
            assert hub["name"] == item["hub_name"], (
                f"hub {item['hub_id']} is named {hub['name']!r} in the "
                f"hierarchy and {item['hub_name']!r} in the fixture"
            )


class TestThePassConditionIsPreDeclared:
    """A condition editable after the fact is not a pre-declared condition."""

    def test_the_fixture_declares_it_is_not_a_metric(
        self, fixture: dict[str, Any]
    ) -> None:
        assert fixture["is_a_metric"] is False
        assert fixture["why_not_a_metric"].strip()

    def test_every_outcome_band_is_declared(self, fixture: dict[str, Any]) -> None:
        """All four keys, so no outcome can be adjudicated after the result."""
        condition = fixture["pre_declared_pass_condition"]
        assert condition["declared_before_any_campaign_2_arm_ran"] is True
        for band in ("pass", "investigate", "fail", "on_fail"):
            assert condition.get(band, "").strip(), f"{band} is not declared"

    def test_the_majority_baseline_is_stated(self, fixture: dict[str, Any]) -> None:
        """The reason this cannot gate has to travel with the file.

        Three of six items share one hub, so always guessing it scores 0.500.
        Anyone reading a 4-of-6 result without that number in front of them
        will read it as the model being right most of the time.
        """
        items = fixture["items"]
        majority = max(fixture["hub_distribution"].values())
        assert majority / len(items) == 0.5
        assert "0.500" in fixture["why_not_a_metric"]
