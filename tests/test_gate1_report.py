"""Gate 1's four quality conditions must be computed, not read.

Until checkpoint 2 they were prose. `GATE1_CONFIDENCE_FLOOR` had exactly one
reader in the repository -- a test asserting the constant lay inside its own
scale -- and `orphan_rate` added every bridge link with no confidence
predicate. Q1, Q2 and Q4 had no implementation anywhere.

The demonstration that made this Critical: a sheet mapping ONE control (AC-1,
"Policy and Procedures") onto all 78 hub ids, copied from the first column of
the packet the volunteer receives, at confidence 1 with rationale ".", imports
cleanly and takes the orphan rate from 78/78 to 0/78. It violates Q1 (one
control, not 40), Q2 (78 hubs on one control, not 6) and Q3 (every confidence
below the floor) simultaneously, and nothing objected.

That sheet is the primary fixture here, because a gate report that does not
fail it is not a gate report.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from scripts.analysis.gate1_report import gate1_report
from tract.bridge.links import BridgeLink
from tract.config import (
    PHASE2C_Q1_MIN_DISTINCT_CONTROLS,
    PHASE2C_Q2_MAX_HUBS_PER_CONTROL,
    PHASE2C_Q3_CONFIDENCE_FLOOR,
)


def _hubs(n: int) -> list[str]:
    from tract.config import TRAINING_DIR

    payload: dict[str, list[dict[str, str]]] = json.loads(
        (TRAINING_DIR / "hub_links_by_framework_curated.json").read_text(
            encoding="utf-8"
        )
    )
    from tract.config import BRIDGE_AI_FRAMEWORK_IDS

    ai: set[str] = set()
    trad: set[str] = set()
    for fid, links in payload.items():
        target = ai if fid in BRIDGE_AI_FRAMEWORK_IDS else trad
        for link in links:
            target.add(link["cre_id"])
    return sorted(ai - trad)[:n]


def _link(
    control: str, hub: str, *, confidence: int = 3, annotator: str = "a1"
) -> BridgeLink:
    return BridgeLink(
        framework_id="nist_800_53",
        standard_name="NIST 800-53 v5",
        section_id=control,
        section_name=control,
        cre_id=hub,
        tier=2,
        annotator_id=annotator,
        created_at="2026-09-04T12:00:00Z",
        confidence=confidence,
        rationale="r",
    )


def _write(tmp_path: Path, links: list[BridgeLink]) -> Path:
    path = tmp_path / "bridge.jsonl"
    path.write_text(
        "".join(json.dumps(asdict(link), sort_keys=True) + "\n" for link in links),
        encoding="utf-8",
    )
    return path


def _good_round() -> list[BridgeLink]:
    """A sheet that should pass: 45 controls, <=3 hubs each, confidence 3."""
    hubs = _hubs(30)
    links: list[BridgeLink] = []
    for i in range(45):
        for j in range(2):
            links.append(_link(f"AC-{i}", hubs[(i + j) % len(hubs)]))
    # Q4: double-annotate 20% of controls.
    for i in range(9):
        links.append(_link(f"AC-{i}", hubs[i % len(hubs)], annotator="a2"))
    return links


class TestTheLazySheetFails:
    """The 15-minute attack, as the primary fixture."""

    @pytest.fixture()
    def lazy(self, tmp_path: Path) -> Path:
        hubs = _hubs(78)
        return _write(
            tmp_path, [_link("AC-1", h, confidence=1) for h in hubs]
        )

    def test_the_overall_verdict_is_fail(self, lazy: Path) -> None:
        assert gate1_report(lazy)["passed"] is False

    def test_q1_fails_on_one_distinct_control(self, lazy: Path) -> None:
        q = gate1_report(lazy)["conditions"]["Q1_distinct_controls"]
        assert q["submitted"] == 1, "the sheet named one control"
        assert q["value"] == 0, "and no link on it clears the confidence floor"
        assert q["passed"] is False

    def test_q2_fails_on_78_hubs_for_one_control(self, lazy: Path) -> None:
        """Q2 judges what was SUBMITTED, not what survives Q3.

        Measuring only the counting subset would let this sheet pass Q2 with a
        value of 0: every link is filtered out, so no control has any hubs.
        The annotator still mapped one control onto 78 hubs, and dropping their
        low-confidence links does not unask whether that was a judgement about
        the control or about the region.
        """
        q = gate1_report(lazy)["conditions"]["Q2_max_hubs_per_control"]
        assert q["submitted"] == 78
        assert q["value"] == 0
        assert q["passed"] is False

    def test_q3_excludes_every_link_below_the_floor(self, lazy: Path) -> None:
        report = gate1_report(lazy)
        assert report["n_links_total"] == 78
        assert report["n_links_counting"] == 0

    def test_the_orphan_rate_does_not_move_on_excluded_links(
        self, lazy: Path
    ) -> None:
        """The defect in one line: 78 low-confidence links de-orphaned 78 hubs.

        Q3 says they are data, not evidence. So the orphan count must be
        computed on the counting subset, not on every imported row.
        """
        report = gate1_report(lazy)
        assert report["orphans_after"] == report["orphans_before"] == 78


class TestAWellFormedRoundPasses:
    @pytest.fixture()
    def good(self, tmp_path: Path) -> Path:
        return _write(tmp_path, _good_round())

    def test_q1_passes(self, good: Path) -> None:
        q = gate1_report(good)["conditions"]["Q1_distinct_controls"]
        assert q["value"] >= PHASE2C_Q1_MIN_DISTINCT_CONTROLS
        assert q["passed"] is True

    def test_q2_passes(self, good: Path) -> None:
        q = gate1_report(good)["conditions"]["Q2_max_hubs_per_control"]
        assert q["value"] <= PHASE2C_Q2_MAX_HUBS_PER_CONTROL
        assert q["passed"] is True

    def test_q4_reports_a_double_annotation_rate_and_an_agreement(
        self, good: Path
    ) -> None:
        q = gate1_report(good)["conditions"]["Q4_double_annotated"]
        assert q["value"] > 0.15
        assert q["passed"] is True
        assert q["agreement"] is not None, (
            "Q4 requires the agreement number to EXIST and be published. "
            "None means it was not computed."
        )

    def test_every_counted_link_clears_the_confidence_floor(
        self, good: Path
    ) -> None:
        report = gate1_report(good)
        assert report["n_links_counting"] == report["n_links_total"]


class TestQ4CannotBeSatisfiedByOneAnnotator:
    def test_a_single_annotator_reports_no_agreement(self, tmp_path: Path) -> None:
        """One person cannot agree with themselves.

        The pre-registration calls Q4 this project's FIRST human-human
        agreement number. A round with one annotator must report that it does
        not have one, rather than a rate of 1.0.
        """
        links = [_link(f"AC-{i}", _hubs(5)[i % 5]) for i in range(45)]
        report = gate1_report(_write(tmp_path, links))
        q = report["conditions"]["Q4_double_annotated"]
        assert q["value"] == 0.0
        assert q["passed"] is False
        assert q["agreement"] is None


class TestTheReportRefusesToGuess:
    def test_an_empty_corpus_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="no links"):
            gate1_report(path)

    def test_every_condition_carries_a_verdict(self, tmp_path: Path) -> None:
        """A missing condition must not read as a satisfied one."""
        report = gate1_report(_write(tmp_path, _good_round()))
        expected = {
            "Q1_distinct_controls",
            "Q2_max_hubs_per_control",
            "Q3_confidence_floor",
            "Q4_double_annotated",
        }
        assert set(report["conditions"]) == expected
        for name, condition in report["conditions"].items():
            assert isinstance(condition["passed"], bool), name

    def test_the_verdict_is_the_conjunction_of_everything(
        self, tmp_path: Path
    ) -> None:
        """passed must not be the orphan rate alone."""
        report = gate1_report(_write(tmp_path, _good_round()))
        expected = report["orphan_reduction_passed"] and all(
            c["passed"] for c in report["conditions"].values()
        )
        assert report["passed"] == expected


class TestThresholdsMatchThePreRegistration:
    """A gate whose constants drift from the document is not that gate."""

    def test_the_document_states_the_same_numbers(self) -> None:
        from tract.config import (
            PHASE2C_GATE1_MAX_ORPHANS,
            PHASE2C_GATE1_MIN_DEORPHANED,
            PHASE2C_Q4_MIN_DOUBLE_ANNOTATED,
            PROJECT_ROOT,
        )

        text = (PROJECT_ROOT / "docs" / "phase2c-preregistration.md").read_text(
            encoding="utf-8"
        )
        assert f"≤ {PHASE2C_GATE1_MAX_ORPHANS}/78" in text
        assert f"**{PHASE2C_GATE1_MIN_DEORPHANED}**" in text
        assert f"≥ {PHASE2C_Q1_MIN_DISTINCT_CONTROLS} distinct" in text
        assert f"≤ {PHASE2C_Q2_MAX_HUBS_PER_CONTROL} AI hubs" in text
        assert f"confidence ≥ {PHASE2C_Q3_CONFIDENCE_FLOOR}" in text
        assert f"≥ {int(PHASE2C_Q4_MIN_DOUBLE_ANNOTATED * 100)}%" in text
