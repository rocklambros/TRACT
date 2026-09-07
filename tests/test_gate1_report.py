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


class TestThePrescribedWorkflowCanActuallyPass:
    """The importer and the gate contradicted each other.

    `import_bridge_links` refuses to overwrite and tells the operator to write
    one file per annotator. `gate1_report` read ONE path with no merge, and Q4
    counts annotators WITHIN a corpus -- so under the prescribed workflow Q4 was
    structurally 0.0 and a flawless round failed on a condition it could not
    satisfy. The only way to satisfy it was a manual concatenation the tooling
    never mentioned, and that concatenation then broke Q2, which measured the
    union of every annotator's hubs for a control rather than each annotator's.
    """

    @pytest.fixture()
    def per_annotator_dir(self, tmp_path: Path) -> Path:
        hubs = _hubs(30)
        a1 = [
            _link(f"AC-{i}", hubs[i % len(hubs)], annotator="a1") for i in range(45)
        ]
        a2 = [
            _link(f"AC-{i}", hubs[i % len(hubs)], annotator="a2") for i in range(9)
        ]
        d = tmp_path / "corpora"
        d.mkdir()
        _write(d, a1).rename(d / "a1.jsonl")
        _write(d, a2).rename(d / "a2.jsonl")
        return d

    def test_a_directory_of_per_annotator_corpora_is_read(
        self, per_annotator_dir: Path
    ) -> None:
        report = gate1_report(per_annotator_dir)
        assert len(report["sources"]) == 2
        assert report["n_annotators"] == 2

    def test_q4_is_satisfiable_under_the_prescribed_workflow(
        self, per_annotator_dir: Path
    ) -> None:
        """The load-bearing one. This was structurally impossible before."""
        q = gate1_report(per_annotator_dir)["conditions"]["Q4_double_annotated"]
        assert q["value"] == pytest.approx(9 / 45)
        assert q["passed"] is True
        assert q["agreement"] is not None

    def test_q2_measures_one_annotator_not_the_union(self, tmp_path: Path) -> None:
        """Two annotators, four hubs each, same control. Neither exceeds 6.

        Keyed on control alone this measures 8 and fails. Since Q4 mandates the
        overlap, that made the two conditions mutually unsatisfiable.
        """
        hubs = _hubs(8)
        links = [_link("AC-1", h, annotator="a1") for h in hubs[:4]]
        links += [_link("AC-1", h, annotator="a2") for h in hubs[4:8]]
        links += [
            _link(f"AC-{i}", hubs[i % 8], annotator="a1") for i in range(2, 46)
        ]
        q = gate1_report(_write(tmp_path, links))["conditions"][
            "Q2_max_hubs_per_control"
        ]
        assert q["value"] == 4, "each annotator gave four hubs, not eight"
        assert q["passed"] is True

    def test_the_same_annotator_twice_is_refused(self, tmp_path: Path) -> None:
        """Re-importing one sheet under two names would inflate the overlap."""
        d = tmp_path / "corpora"
        d.mkdir()
        links = [_link("AC-1", _hubs(1)[0], annotator="a1")]
        _write(d, links).rename(d / "one.jsonl")
        _write(d, links).rename(d / "two.jsonl")
        with pytest.raises(ValueError, match="already recorded"):
            gate1_report(d)

    def test_an_empty_directory_raises(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(ValueError, match="no .jsonl"):
            gate1_report(d)


class TestItRefusesTheGoldDirectory:
    """`data/training/` holds the gold link files, not bridge corpora.

    Gate 1 reads a directory and globs *.jsonl. The handbook originally told
    the coordinator to point it at `data/training/`, which holds
    hub_links.jsonl, hub_links_curated.jsonl and hub_links_training.jsonl -- so
    it handed the CURATED GOLD LINKS to a bridge loader.

    It failed, because the loader validates its fields, but the operator was
    left reading `missing required field(s) ['annotator_id', 'confidence',
    'created_at', 'rationale', 'tier']` instead of being told they had pointed
    at the wrong directory.
    """

    def test_a_directory_of_gold_links_is_refused_by_name(
        self, tmp_path: Path
    ) -> None:
        for name in ("hub_links.jsonl", "hub_links_curated.jsonl"):
            (tmp_path / name).write_text('{"cre_id": "1"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="not a bridge corpus directory"):
            gate1_report(tmp_path)

    def test_the_error_names_where_to_point_instead(self, tmp_path: Path) -> None:
        (tmp_path / "hub_links_training.jsonl").write_text("{}\n", encoding="utf-8")
        with pytest.raises(ValueError) as excinfo:
            gate1_report(tmp_path)
        assert "bridge" in str(excinfo.value), (
            "the refusal must tell the operator which directory to use"
        )

    def test_a_real_bridge_directory_is_still_accepted(
        self, tmp_path: Path
    ) -> None:
        """Guards the guard: refusing everything would pass the tests above."""
        hubs = _hubs(30)
        d = tmp_path / "bridge"
        d.mkdir()
        _write(d, [_link(f"AC-{i}", hubs[i % 30]) for i in range(45)]).rename(
            d / "vol-01.jsonl"
        )
        assert gate1_report(d)["n_links_total"] == 45

    def test_the_default_is_the_bridge_directory(self) -> None:
        """A bare `gate1_report` must not need the operator to remember a path."""
        import inspect

        from scripts.analysis.gate1_report import main
        from tract.config import BRIDGE_CORPUS_DIR

        assert "BRIDGE_CORPUS_DIR" in inspect.getsource(main)
        assert BRIDGE_CORPUS_DIR.name == "bridge"


class TestQ4ReportsAgreementHonestly:
    """The headline agreement figure was the most flattering denominator.

    Measured on the real round: 23 controls both annotators linked to the SAME
    hub, 3 to different hubs, **28 where one linked and the other said NONE**,
    and 242 where both said NONE.

    Jaccard over "controls both linked" gives 23/26 = 0.8846 and silently drops
    the 28 -- the largest disagreement category. Counting NONE-NONE gives
    265/300 = 0.8833, which is the negative-class inflation the
    pre-registration named in advance: "the negative class dominates and raw
    agreement will read ~95% at chance-level skill". Chance-corrected, the same
    data gives Cohen's kappa = 0.539.

    So Q4 reports the kappa as the headline and every denominator beside it. A
    single agreement number on this task is a choice of denominator, and the
    choice must be visible rather than made silently.
    """

    @pytest.fixture()
    def two_annotators(self, tmp_path: Path) -> Path:
        """Reproduces the real round's shape at small scale."""
        hubs = _hubs(10)
        d = tmp_path / "bridge"
        d.mkdir()
        a: list[BridgeLink] = []
        b: list[BridgeLink] = []
        for i in range(40):          # both link, same hub -> agreement
            a.append(_link(f"AC-{i}", hubs[i % 10], annotator="a1"))
            b.append(_link(f"AC-{i}", hubs[i % 10], annotator="a2"))
        for i in range(40, 45):      # both link, different hub -> disagreement
            a.append(_link(f"AC-{i}", hubs[0], annotator="a1"))
            b.append(_link(f"AC-{i}", hubs[1], annotator="a2"))
        for i in range(45, 55):      # only a1 links -> disagreement, was dropped
            a.append(_link(f"AC-{i}", hubs[2], annotator="a1"))
        _write(d, a).rename(d / "a1.jsonl")
        _write(d, b).rename(d / "a2.jsonl")
        # The sidecars the importer writes. Without them there is no reviewed
        # denominator, and kappa is correctly None -- a corpus holds links only,
        # so the controls judged NONE are invisible to it.
        for who, n_linked in (("a1", 55), ("a2", 45)):
            (d / f"{who}.reviewed.json").write_text(
                json.dumps({
                    "annotator_id": who, "created_at": "2026-09-07T12:00:00Z",
                    "framework_id": "nist_800_53", "n_reviewed": 300,
                    "n_linked": n_linked, "n_no_hub": 300 - n_linked,
                    "no_hub_controls": [],
                }),
                encoding="utf-8",
            )
        return d

    def test_it_reports_a_chance_corrected_figure(
        self, two_annotators: Path
    ) -> None:
        q = gate1_report(two_annotators)["conditions"]["Q4_double_annotated"]
        assert q["cohen_kappa"] is not None, (
            "the pre-registration asks for a chance-corrected figure; raw "
            "agreement on a task dominated by NONE reads high at chance skill"
        )

    def test_it_reports_every_denominator(self, two_annotators: Path) -> None:
        q = gate1_report(two_annotators)["conditions"]["Q4_double_annotated"]
        for key in ("agreement_both_linked", "agreement_either_linked",
                    "n_both_linked", "n_one_linked_only", "n_same_hub"):
            assert key in q, f"Q4 omits {key}"

    def test_the_excluded_disagreements_are_counted_and_visible(
        self, two_annotators: Path
    ) -> None:
        """The 10 controls only one annotator linked must be reported."""
        q = gate1_report(two_annotators)["conditions"]["Q4_double_annotated"]
        assert q["n_one_linked_only"] == 10

    def test_the_two_denominators_actually_differ_here(
        self, two_annotators: Path
    ) -> None:
        """If they were equal the distinction would be decorative."""
        q = gate1_report(two_annotators)["conditions"]["Q4_double_annotated"]
        assert q["agreement_both_linked"] > q["agreement_either_linked"], (
            "the both-linked denominator must be the more flattering one, or "
            "this test is not exercising the case it was written for"
        )

    def test_a_single_annotator_still_reports_nothing(
        self, tmp_path: Path
    ) -> None:
        """One person cannot agree with themselves, in any denominator."""
        links = [_link(f"AC-{i}", _hubs(5)[i % 5]) for i in range(45)]
        q = gate1_report(_write(tmp_path, links))["conditions"]["Q4_double_annotated"]
        assert q["cohen_kappa"] is None
        assert q["agreement_both_linked"] is None
