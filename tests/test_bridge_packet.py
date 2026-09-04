"""The annotator packet must be model-free, and provably so by content.

Two corrections to the plan's original tests are baked in here.

C4 -- THE GUARD MUST CHECK VALUES, NOT COLUMN NAMES. The plan asserted that no
CSV header contains "similarity", "rank", "related_hub" and so on. A column
called `see_also` carrying bare CRE ids passes that check completely.
`cre_hierarchy.json` carries `related_hub_ids` on 51 hubs -- Phase 2B's 46
model-proposed edges, Tier 3 per results/bridge/PROVENANCE.md -- and those are
bare ids like "342-641". So the load-bearing test scans every CELL for hub-id
values that the sheet has no business naming.

A1 -- ALL 78 HUBS, UNRANKED. The plan scoped the sheet to the top 20 hubs by
eval weight. That made Gate 1 unpassable (it needs 23 de-orphaned, and a link
carries one cre_id, so a flawless annotator reaches 20), and "eval weight"
counts appearances in the held-out split -- a selection rule derived from the
test set, which is the leakage shape that withdrew two prior campaigns. There is
no top_n_hubs parameter.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

from scripts.build_bridge_packet import build_bridge_packet

def _real_hub_ids() -> frozenset[str]:
    """Every hub id in the hierarchy.

    Matching by SHAPE (`\\b\\d{3}-\\d{3}\\b`) was the first attempt and it is
    wrong here: NIST 800-53 prose cites "SP 800-160-1", "800-189" and "M-19-03"
    constantly, and four controls tripped the guard on document references. The
    question is not "does this look like an id" but "does this name a real
    hub", which is both stricter and free of false positives -- measured, no
    hub id resembles a NIST document number, and none occurs in that prose.
    """
    import json

    from tract.config import PROCESSED_DIR

    hierarchy = json.loads(
        (PROCESSED_DIR / "cre_hierarchy.json").read_text(encoding="utf-8")
    )
    return frozenset(hierarchy["hubs"])

FORBIDDEN_HEADER_TERMS = (
    "similarity", "cosine", "rank", "top_k", "suggested",
    "candidate", "model", "predict", "score", "related_hub",
)


@pytest.fixture(scope="module")
def packet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("packet")
    build_bridge_packet(out, framework_id="nist_800_53")
    return out


def _csvs(packet: Path) -> list[Path]:
    return sorted(packet.glob("*.csv"))


class TestTheHubSheetIsComplete:
    def test_it_carries_all_78_ai_hubs(self, packet: Path) -> None:
        """A1: unranked and complete, so Gate 1 is reachable."""
        rows = list(csv.DictReader((packet / "ai_hubs.csv").open(encoding="utf-8")))
        assert len(rows) == 78

    def test_it_exposes_no_ranking_column(self, packet: Path) -> None:
        """Any order column would reintroduce a test-set-derived selection."""
        header = (packet / "ai_hubs.csv").read_text(encoding="utf-8").splitlines()[0]
        for term in ("weight", "rank", "priority", "order", "freq", "count"):
            assert term not in header.lower(), f"ai_hubs.csv exposes {term!r}"

    def test_it_carries_the_fields_an_annotator_needs(self, packet: Path) -> None:
        reader = csv.DictReader((packet / "ai_hubs.csv").open(encoding="utf-8"))
        assert set(reader.fieldnames or []) == {
            "hub_id", "hub_name", "hierarchy_path", "branch",
        }


class TestNoModelOutputByName:
    def test_no_forbidden_column_in_any_sheet(self, packet: Path) -> None:
        for path in _csvs(packet):
            header = path.read_text(encoding="utf-8").splitlines()[0].lower()
            for term in FORBIDDEN_HEADER_TERMS:
                assert term not in header, f"{path.name} leaks {term}"


class TestNoModelOutputByValue:
    """C4. The test the header check could not be."""

    def test_no_cell_outside_the_hub_id_column_contains_a_hub_id(
        self, packet: Path
    ) -> None:
        """A `see_also` column of bare ids passes every header check.

        The hub sheet's own `hub_id` column is the one legitimate place a hub id
        appears. Anywhere else -- any column, any sheet -- is a model-proposed
        edge that reached the annotator.
        """
        hub_ids = _real_hub_ids()
        offenders: list[str] = []
        for path in _csvs(packet):
            with path.open(encoding="utf-8") as handle:
                for lineno, row in enumerate(csv.DictReader(handle), start=2):
                    for column, value in row.items():
                        if path.name == "ai_hubs.csv" and column == "hub_id":
                            continue
                        if not value:
                            continue
                        named = sorted(
                            h for h in hub_ids if re.search(rf"\b{re.escape(h)}\b", value)
                        )
                        if named:
                            offenders.append(
                                f"{path.name}:{lineno} column {column!r} names {named}"
                            )
        assert not offenders, (
            "Hub ids appear outside the hub_id column. cre_hierarchy.json "
            "carries related_hub_ids on 51 hubs (Phase 2B's model-proposed "
            "edges, Tier 3); a sheet naming them makes the round Tier 3 too: "
            + "; ".join(offenders[:10])
        )

    def test_the_scan_would_catch_a_planted_id(self, packet: Path, tmp_path: Path) -> None:
        """Prove the scanner fires, rather than trusting a clean pass.

        This project has shipped a tripwire that could not go red. The regex is
        the whole guard, so it gets exercised against a value that must trip it.
        """
        hub_ids = _real_hub_ids()
        leaked = sorted(hub_ids)[0]
        planted = tmp_path / "leaky.csv"
        planted.write_text(
            f"hub_id,see_also\n999-999,{leaked}\n", encoding="utf-8"
        )
        found = [
            v
            for row in csv.DictReader(planted.open(encoding="utf-8"))
            for k, v in row.items()
            if k != "hub_id"
            and v
            and any(re.search(rf"\b{re.escape(h)}\b", v) for h in hub_ids)
        ]
        assert found == [leaked]

    def test_nist_document_references_do_not_trip_the_scan(
        self, packet: Path
    ) -> None:
        """The false positive the shape-based regex produced, pinned.

        NIST 800-53 prose cites SP 800-160-1, SP 800-189 and OMB M-19-03. None
        is a hub id, and a guard that flags them would be silenced by whoever
        hit it next.
        """
        hub_ids = _real_hub_ids()
        for citation in ("SP 800-160-1", "SP 800-189", "OMB M-19-03", "ISO 15408-2"):
            assert not any(
                re.search(rf"\b{re.escape(h)}\b", citation) for h in hub_ids
            ), f"{citation!r} reads as a hub id"

    def test_related_hub_ids_are_present_in_the_source(self) -> None:
        """Guards the guard: if the source stopped carrying them, the test above
        would pass for the wrong reason and stop protecting anything."""
        import json

        from tract.config import PROCESSED_DIR

        hierarchy = json.loads(
            (PROCESSED_DIR / "cre_hierarchy.json").read_text(encoding="utf-8")
        )
        with_related = [
            h for h in hierarchy["hubs"].values() if h.get("related_hub_ids")
        ]
        assert len(with_related) == 51, (
            f"{len(with_related)} hubs carry related_hub_ids, not the 51 this "
            "suite was written against. Re-check what the packet excludes."
        )


class TestItRefusesLicensedFrameworks:
    def test_refuses_a_restricted_framework(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="restricted"):
            build_bridge_packet(tmp_path, framework_id="etsi")

    def test_refuses_the_other_restricted_framework(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="restricted"):
            build_bridge_packet(tmp_path, framework_id="iso_27001")

    def test_it_refuses_before_reading_any_prose(self, tmp_path: Path) -> None:
        """The check must precede the read, or the prose is already in memory."""
        with pytest.raises(ValueError, match="restricted"):
            build_bridge_packet(tmp_path, framework_id="etsi")
        assert not list(tmp_path.glob("*.csv")), (
            "A refused framework still wrote sheets; the guard runs too late."
        )

    def test_refuses_an_unknown_framework_by_name(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not_a_framework"):
            build_bridge_packet(tmp_path, framework_id="not_a_framework")


class TestTheControlSheet:
    def test_it_carries_prose_not_just_titles(self, packet: Path) -> None:
        """CLAUDE.md: prefer a control's full text over its title everywhere."""
        rows = list(csv.DictReader((packet / "controls.csv").open(encoding="utf-8")))
        assert rows
        assert set(rows[0]) == {"control_id", "control_title", "control_text"}
        with_text = [r for r in rows if len(r["control_text"].strip()) > 40]
        assert len(with_text) > len(rows) // 2, (
            "Most controls have no usable prose, so annotators would be mapping "
            "titles -- the fallback CLAUDE.md calls a last resort."
        )

    def test_no_ground_truth_column(self, packet: Path) -> None:
        header = (packet / "controls.csv").read_text(encoding="utf-8").splitlines()[0]
        for term in ("cre", "hub", "gold", "answer", "label"):
            assert term not in header.lower()
