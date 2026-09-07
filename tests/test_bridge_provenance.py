"""The Phase 2B bridge set is Tier 3 and must be recorded as such.

`results/review/PROVENANCE.md` covers `review_export.json`, and
`tests/test_tier3_quarantine.py` enforces that it never reaches a gate
denominator. Nothing covered `results/bridge/`, whose 46 accepted edges are
published three ways: into `cre_hierarchy.json` as `related_hub_ids`, into the
HuggingFace dataset, and onto the model card with a code example teaching
consumers to iterate the field.

They were produced by cosine top-k with LLM-written rationales and ratified by
one reviewer in the model's presence -- `CAMPAIGN3.md` §2's Tier 3 on its face.

These pin the provenance record and the two constraints Phase 2C inherits from
it: nothing here may seed an annotator packet, and `related_hub_ids` must not
reach one.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BRIDGE_DIR = PROJECT_ROOT / "results" / "bridge"
PROVENANCE = BRIDGE_DIR / "PROVENANCE.md"
REPORT = BRIDGE_DIR / "bridge_report.json"
HIERARCHY = PROJECT_ROOT / "data" / "processed" / "cre_hierarchy.json"


class TestProvenanceIsRecorded:

    def test_the_directory_carries_a_provenance_file(self) -> None:
        assert PROVENANCE.is_file(), (
            "results/bridge/ publishes into the hierarchy, the dataset and the "
            "model card. It needs the same provenance record review_export.json "
            "has."
        )

    def test_it_states_the_tier(self) -> None:
        text = PROVENANCE.read_text(encoding="utf-8")
        assert "Tier 3" in text
        assert "gate denominator" in text

    def test_it_forbids_seeding_a_phase2c_packet(self) -> None:
        text = PROVENANCE.read_text(encoding="utf-8")
        assert "may seed a Phase 2C packet" in text or "seed a Phase 2C" in text
        assert "related_hub_ids" in text


@pytest.mark.skipif(not REPORT.is_file(), reason="bridge report absent")
class TestTheRecordedFiguresAreTrue:
    """A provenance file asserting unverified numbers is the defect it exists
    to prevent. Each figure below is recomputed from the committed report."""

    def _candidates(self) -> list[dict]:
        return json.loads(REPORT.read_text(encoding="utf-8"))["candidates"]

    def test_accept_rate(self) -> None:
        c = self._candidates()
        accepted = [x for x in c if x["status"] == "accepted"]
        assert (len(accepted), len(c)) == (46, 63)

    def test_a_single_cosine_threshold_reproduces_59_of_63(self) -> None:
        c = self._candidates()
        best = max(
            sum(1 for x in c
                if (x["cosine_similarity"] >= t) == (x["status"] == "accepted"))
            for t in {x["cosine_similarity"] for x in c}
        )
        assert best == 59, (
            f"{best}/63 decisions reproduced by one threshold, not 59. The "
            "PROVENANCE figure is stale."
        )

    def test_acceptance_falls_monotonically_with_presented_rank(self) -> None:
        # 19/21, 15/21, 12/21 -- the signature of a reviewer following a
        # model-ordered shortlist.
        cands = json.loads(
            (BRIDGE_DIR / "bridge_candidates.json").read_text(encoding="utf-8")
        )["candidates"]
        rank = {(x["ai_hub_id"], x["trad_hub_id"]): x.get("rank_for_ai_hub")
                for x in cands}
        by_rank: dict[int, list[bool]] = {}
        for x in self._candidates():
            k = rank.get((x["ai_hub_id"], x["trad_hub_id"]))
            if k:
                by_rank.setdefault(k, []).append(x["status"] == "accepted")
        rates = [sum(v) for _, v in sorted(by_rank.items())]
        assert rates == [19, 15, 12]


@pytest.mark.skipif(not HIERARCHY.is_file(), reason="hierarchy absent")
class TestRelatedHubIdsIsEntirelyThisSet:

    def test_the_published_field_is_100_percent_model_proposed(self) -> None:
        hubs = json.loads(HIERARCHY.read_text(encoding="utf-8"))["hubs"]
        endpoints = sum(len(v.get("related_hub_ids", [])) for v in hubs.values())
        carrying = sum(1 for v in hubs.values() if v.get("related_hub_ids"))
        # 46 accepted edges, recorded bidirectionally.
        assert (carrying, endpoints) == (51, 92)
