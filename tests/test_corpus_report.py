"""The corpus report is the only instrument in the parser plan.

Counting links resolved cannot tell 615 links unstacked onto 615 anchors from
615 links collapsed onto 40. Both make the same number rise. The tests below
pin the columns that can tell them apart: distinct anchors against the
fallback anchors the trainer already gets, links per anchor, truncation,
nesting by containment, controls the prose rule excludes from the index, and
the three wrong-anchor detectors.

Every path here is anchored to PROJECT_ROOT. A test that resolves a relative
path passes or fails on the directory pytest started in.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import pytest

from tract.config import (
    PROCESSED_FRAMEWORKS_DIR,
    PROJECT_ROOT,
    PROSE_MIN_EXTRA_CHARS,
)
from tract.corpus_report import (
    COARSE_NAME_RATIO,
    CORPUS_EVIDENCE_DIR,
    CURATED_LINKS_PATH,
    DETECTOR_B_INAPPLICABLE,
    FINE_NAME_RATIO,
    JOIN_CEILINGS,
    JOIN_FLOORS,
    JOIN_WRONG_ANCHOR_BUDGET,
    CorpusReport,
    _load_links,
    build_corpus_report,
    check_join_floors,
    name_level_mismatch_frameworks,
    require_full_corpus,
    require_unmoved_corpus,
    wrong_anchor_applicable,
)
from tract.text_selection import merged_corpus_path

TRACKED_CORPUS = PROJECT_ROOT / "data" / "processed" / "all_controls.json"

LONG = "A control statement long enough to clear every prose bar. " * 4


def _corpus_carries_prose(framework_id: str) -> bool:
    """Whether the corpus THIS checkout reads holds statements for a framework.

    parsers/merge_all_controls.py reduces every OVERLAY_FRAMEWORK_IDS member to
    identifiers and titles in the tracked corpus and keeps its prose only in
    the gitignored overlay. ProseIndex indexes a control when its description
    exceeds its title, so a reduced framework resolves nothing and every
    per-framework assertion about its join measures the licence tier instead of
    the parser. Derived from the corpus rather than from a list of framework
    ids, so the same assertions stay live wherever the overlay is present.
    """
    corpus = json.loads(merged_corpus_path().read_text(encoding="utf-8"))
    for framework in corpus.get("frameworks", []):
        if str(framework.get("framework_id")) != framework_id:
            continue
        return any(
            str(control.get("full_text") or "").strip()
            or str(control.get("description") or "").strip()
            != str(control.get("title") or "").strip()
            for control in framework.get("controls") or []
        )
    return False


def _corpus(
    directory: Path,
    controls: list[dict[str, object]],
    name: str = "corpus",
) -> Path:
    """A corpus in the shape the real files use: a dict, not a list.

    Version 2's parity test assumed a list and silently indexed nothing. The
    fixtures use the real shape so the loader is exercised the way production
    exercises it.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    payload = {
        "framework_count": 1,
        "frameworks": [
            {
                "framework_id": "demo",
                "framework_name": "Demo",
                "controls": controls,
            }
        ],
        "generated_date": "2026-01-01",
        "total_controls": len(controls),
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _corpus_as_list(
    directory: Path, controls: list[dict[str, object]], name: str = "legacy",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(
        json.dumps(
            [{
                "framework_id": "demo",
                "framework_name": "Demo",
                "controls": controls,
            }],
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _links(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    path = tmp_path / "links.jsonl"
    path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
        encoding="utf-8",
    )
    return path


def _row(section_id: str, section_name: str, cre_id: str = "1-1") -> dict[str, str]:
    return {
        "framework_id": "demo",
        "standard_name": "Demo",
        "section_id": section_id,
        "section_name": section_name,
        "cre_id": cre_id,
        "link_type": "LinkedTo",
    }


class TestCorpusShape:
    """The fact that made version 2's parity test assert nothing."""

    def test_the_tracked_corpus_is_a_dict(self) -> None:
        data = json.loads(TRACKED_CORPUS.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert sorted(data) == [
            "framework_count", "frameworks", "generated_date", "total_controls",
        ]

    def test_the_loader_reads_both_shapes(self, tmp_path: Path) -> None:
        controls: list[dict[str, object]] = [
            {"control_id": "C-1", "title": "One", "description": LONG},
        ]
        links = _links(tmp_path, [_row("C-1", "One")])
        as_dict = build_corpus_report(links, _corpus(tmp_path / "d", controls))
        as_list = build_corpus_report(
            links, _corpus_as_list(tmp_path / "l", controls),
        )
        assert as_dict.per_framework[0].by_title == 1
        assert as_list.per_framework[0].by_title == 1
        assert as_dict.corpus_framework_count == 1

    def test_a_corpus_with_no_records_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"total_controls": 0}), encoding="utf-8")
        with pytest.raises(ValueError, match="no list under 'frameworks'"):
            build_corpus_report(_links(tmp_path, [_row("C-1", "One")]), path)

    def test_a_list_under_the_wrong_key_raises_instead_of_being_used(
        self, tmp_path: Path,
    ) -> None:
        """The silent zero the "first list value" fallback used to produce.

        {"controls": [...]} was accepted as a list of framework records, so
        every framework missed, every count read zero, and nothing failed or
        logged. A renamed 'frameworks' key did the same. This is the failure
        the whole module exists to make impossible, so it raises and names the
        keys it found.
        """
        path = tmp_path / "renamed.json"
        path.write_text(json.dumps({
            "controls": [
                {"control_id": "C-1", "title": "One", "description": LONG},
            ],
            "meta": {},
        }, sort_keys=True), encoding="utf-8")
        with pytest.raises(ValueError, match=r"Keys found: \['controls', 'meta'\]"):
            build_corpus_report(_links(tmp_path, [_row("C-1", "One")]), path)

    def test_a_corpus_that_is_neither_shape_names_what_it_parsed_as(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "scalar.json"
        path.write_text(json.dumps("not a corpus"), encoding="utf-8")
        with pytest.raises(ValueError, match="It parsed as str"):
            build_corpus_report(_links(tmp_path, [_row("C-1", "One")]), path)


class TestLinkFileFailures:
    """Every failure names the file, the line and the record.

    A bare JSONDecodeError or KeyError from a 4,405-line file tells a reader
    nothing about which line to open.
    """

    def test_malformed_json_names_the_line(self, tmp_path: Path) -> None:
        path = tmp_path / "links.jsonl"
        path.write_text(
            json.dumps(_row("C-1", "One"), sort_keys=True) + "\n"
            + "{not json\n",
            encoding="utf-8",
        )
        corpus = _corpus(tmp_path, [])
        with pytest.raises(ValueError, match="line 2 is not valid JSON"):
            build_corpus_report(path, corpus)

    def test_a_missing_framework_id_names_the_line_and_the_keys(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "links.jsonl"
        path.write_text(
            json.dumps({"section_id": "C-1", "cre_id": "1-1"}, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        corpus = _corpus(tmp_path, [])
        with pytest.raises(
            ValueError, match=r"line 1 has no 'framework_id'",
        ):
            build_corpus_report(path, corpus)

    def test_a_json_array_line_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "links.jsonl"
        path.write_text(json.dumps([1, 2, 3]) + "\n", encoding="utf-8")
        corpus = _corpus(tmp_path, [])
        with pytest.raises(ValueError, match="line 1 holds a list"):
            build_corpus_report(path, corpus)


class TestAnchorCollapse:
    def test_distinct_anchors_separates_collapse_from_coverage(
        self, tmp_path: Path,
    ) -> None:
        """Two corpora resolve every link. Only one of them is good."""
        rows = [_row(f"C-{n}", f"Control {n}") for n in range(1, 5)]
        spread = _corpus(tmp_path / "a", [
            {"control_id": f"C-{n}", "title": f"Control {n}",
             "description": f"{LONG} Variant {n}."}
            for n in range(1, 5)
        ])
        collapsed = _corpus(tmp_path / "b", [
            {"control_id": f"C-{n}", "title": f"Control {n}",
             "description": LONG}
            for n in range(1, 5)
        ])

        links = _links(tmp_path, rows)
        good = build_corpus_report(links, spread).per_framework[0]
        bad = build_corpus_report(links, collapsed).per_framework[0]

        assert good.links == bad.links == 4
        assert good.unresolved == bad.unresolved == 0
        assert good.distinct_anchors == 4
        assert bad.distinct_anchors == 1
        assert bad.links_per_anchor == pytest.approx(4.0)

    def test_nesting_is_containment_not_a_prefix(self, tmp_path: Path) -> None:
        """ETSI clause 5.2 rolled up over 5.2.2 does not start the parent's text.

        Version 2 counted strict prefixes, so this case read 0. The child here
        sits in the middle of the parent, which is what a clause rollup looks
        like when the parent opens with its own lead paragraph.
        """
        child = LONG + " The child clause statement."
        parent = "Lead-in paragraph for the parent clause. " + child + " Tail."
        corpus = _corpus(tmp_path, [
            {"control_id": "5.2.2", "title": "Child", "description": child},
            {"control_id": "5.2", "title": "Parent", "description": parent},
        ])
        links = _links(tmp_path, [_row("5.2.2", "Child"), _row("5.2", "Parent")])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.distinct_anchors == 2
        assert row.nested_anchors == 1
        assert row.contained_anchors == 0

    def test_contained_keeps_the_strict_prefix_count(self, tmp_path: Path) -> None:
        """A domain aggregate that opens with its own first member."""
        member = LONG + " Member statement."
        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "Member", "description": member},
            {"control_id": "D-1", "title": "Domain",
             "description": member + " And the rest of the domain."},
        ])
        links = _links(tmp_path, [_row("C-1", "Member"), _row("D-1", "Domain")])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.distinct_anchors == 2
        assert row.contained_anchors == 1
        assert row.nested_anchors == 1

    def test_truncation_can_merge_two_anchors_into_one(
        self, tmp_path: Path,
    ) -> None:
        """Two anchors sharing a long prefix collapse after MAX_ANCHOR_CHARS.

        distinct_anchors falls with no other column moving, which is how the
        collapse hid. distinct_anchors_pre_truncation is the witness.
        """
        from tract.config import MAX_ANCHOR_CHARS

        shared = "Shared opening text. " * (MAX_ANCHOR_CHARS // 21 + 2)
        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "One", "description": shared + " Tail A."},
            {"control_id": "C-2", "title": "Two", "description": shared + " Tail B."},
        ])
        links = _links(tmp_path, [_row("C-1", "One"), _row("C-2", "Two")])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.by_title == 2
        assert row.truncated == 2
        assert row.distinct_anchors == 1
        assert row.distinct_anchors_pre_truncation == 2


class TestFallbackAnchors:
    def test_unresolved_links_still_give_the_trainer_an_anchor(
        self, tmp_path: Path,
    ) -> None:
        """The column that turns +452 into +152.

        select_control_text falls back to section_name, so a framework with
        zero resolved links does not train on zero anchors.
        """
        corpus = _corpus(tmp_path, [])
        links = _links(tmp_path, [
            _row("C-1", "Access control policy"),
            _row("C-2", "Access control policy"),
            _row("C-3", "Cryptographic key management"),
        ])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.unresolved == 3
        assert row.distinct_anchors == 0
        assert row.fallback_anchors == 2

    def test_a_link_with_no_name_falls_back_to_its_id(self, tmp_path: Path) -> None:
        corpus = _corpus(tmp_path, [])
        links = _links(tmp_path, [_row("C-1", "")])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.fallback_anchors == 1


class TestProseRuleExclusion:
    def test_control_whose_description_restates_its_title_is_counted(
        self, tmp_path: Path,
    ) -> None:
        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "Access control",
             "description": "Access control."},
        ])
        links = _links(tmp_path, [_row("C-1", "Access control")])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.dropped_by_prose_rule == 1
        assert row.unresolved == 1
        assert row.distinct_anchors == 0
        assert row.fallback_anchors == 1

    def test_the_total_counts_frameworks_with_no_curated_links(
        self, tmp_path: Path,
    ) -> None:
        """522 was the link-bearing subset. The corpus holds 558."""
        path = tmp_path / "corpus.json"
        path.write_text(json.dumps({
            "framework_count": 2,
            "frameworks": [
                {"framework_id": "demo", "framework_name": "Demo",
                 "controls": [{"control_id": "C-1", "title": "A",
                               "description": "A."}]},
                {"framework_id": "silent", "framework_name": "Silent",
                 "controls": [{"control_id": "S-1", "title": "B",
                               "description": "B."}]},
            ],
            "generated_date": "2026-01-01",
            "total_controls": 2,
        }, sort_keys=True), encoding="utf-8")
        report = build_corpus_report(_links(tmp_path, [_row("C-1", "A")]), path)
        assert report.by_id("demo").dropped_by_prose_rule == 1
        assert report.totals.dropped_by_prose_rule == 2


class TestAnchorSource:
    def test_the_four_sources_partition_the_resolved_links(
        self, tmp_path: Path,
    ) -> None:
        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "Full", "description": "short",
             "full_text": LONG + " From full text."},
            {"control_id": "C-2", "title": "Described", "description": LONG},
            {"control_id": "C-3", "title": "Title restated as full text",
             "description": "short",
             "full_text": "Title restated as full text"},
            {"control_id": "C-4", "title": "Built", "description": LONG + " Built.",
             "metadata": {"text_origin": "synthetic"}},
        ])
        links = _links(tmp_path, [
            _row("C-1", "Full"), _row("C-2", "Described"),
            _row("C-3", "Title restated as full text"), _row("C-4", "Built"),
        ])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.anchor_source_full_text == 1
        assert row.anchor_source_description == 1
        assert row.anchor_source_title == 1
        assert row.anchor_source_synthetic == 1
        assert (
            row.anchor_source_full_text
            + row.anchor_source_description
            + row.anchor_source_title
            + row.anchor_source_synthetic
        ) == row.by_title + row.by_id


class TestHubSide:
    def test_hub_concentration_is_reported(self, tmp_path: Path) -> None:
        corpus = _corpus(tmp_path, [
            {"control_id": f"C-{n}", "title": f"T{n}",
             "description": f"{LONG} {n}."} for n in range(1, 5)
        ])
        links = _links(tmp_path, [
            _row("C-1", "T1", "hub-a"), _row("C-2", "T2", "hub-a"),
            _row("C-3", "T3", "hub-a"), _row("C-4", "T4", "hub-b"),
        ])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.distinct_hubs == 2
        assert row.links_per_hub == pytest.approx(2.0)


class TestWrongAnchorRisk:
    """Three detectors. Two of them reach the id branch, which is the fix."""

    def test_detector_a_title_hit_that_disagrees_with_the_id(
        self, tmp_path: Path,
    ) -> None:
        corpus = _corpus(tmp_path, [
            {"control_id": "2.3", "title": "Poisoning attacks",
             "description": LONG + " Predictive."},
            {"control_id": "3.2.2", "title": "Generative poisoning",
             "description": LONG + " Generative.",
             "metadata": {"alt_titles": ["Poisoning attacks"]}},
        ])
        links = _links(tmp_path, [_row("3.2.2", "Poisoning attacks")])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.by_title == 1
        assert row.wrong_anchor_risk == 1

    def test_detector_a_does_not_fire_when_the_channels_agree(
        self, tmp_path: Path,
    ) -> None:
        corpus = _corpus(tmp_path, [
            {"control_id": "2.3", "title": "Poisoning attacks",
             "description": LONG + " Predictive."},
        ])
        links = _links(tmp_path, [_row("2.3", "Poisoning attacks")])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.by_title == 1
        assert row.wrong_anchor_risk == 0

    def test_detector_b_id_hit_whose_control_does_not_carry_the_name(
        self, tmp_path: Path,
    ) -> None:
        """The id branch, where version 2 was blind."""
        corpus = _corpus(tmp_path, [
            {"control_id": "IPY", "title": "Interoperability and portability",
             "description": LONG + " Domain."},
        ])
        links = _links(tmp_path, [_row("IPY", "Data centre power redundancy")])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.by_id == 1
        assert row.wrong_anchor_risk == 1

    def test_detector_b_does_not_fire_on_an_identifier_shaped_name(
        self, tmp_path: Path,
    ) -> None:
        """wstg and owasp_proactive_controls have section_name == section_id.

        Those links make no independent claim, so nothing is checked and
        nothing is flagged. This is why their attainable range is zero, and it
        is asserted rather than assumed.
        """
        corpus = _corpus(tmp_path, [
            {"control_id": "WSTG-INFO-01", "title": "Information gathering",
             "description": LONG + " Gathering."},
        ])
        links = _links(tmp_path, [_row("WSTG-INFO-01", "WSTG-INFO-01")])
        report = build_corpus_report(links, corpus)
        assert report.per_framework[0].by_id == 1
        assert report.per_framework[0].wrong_anchor_risk == 0
        assert all(
            not entry.wrong_anchor_checked for entry in report.resolution_rows
        )

    def test_detector_c_a_parent_id_and_a_child_id_reach_one_paragraph(
        self, tmp_path: Path,
    ) -> None:
        """The NIST AI 100-2 failure that put title first in the lookup order.

        Both controls exist and both carry the same paragraph, which is what a
        parser produces when a subsection's text is copied to each of the three
        mitigations it contains. Section names equal section ids here, so
        detector B cannot apply and the flag is C's alone.
        """
        shared = LONG + " Mitigations subsection."
        corpus = _corpus(tmp_path, [
            {"control_id": "3.3.2", "title": "Mitigations",
             "description": shared},
            {"control_id": "3.3.2.1", "title": "Adversarial training",
             "description": shared},
        ])
        links = _links(tmp_path, [_row("3.3.2", "3.3.2"), _row("3.3.2.1", "3.3.2.1")])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.by_id == 2
        assert row.distinct_anchors == 1
        assert row.wrong_anchor_risk == 2

    def test_the_applicable_denominator_is_reported(self, tmp_path: Path) -> None:
        """A zero over a zero denominator proves nothing, so it is countable."""
        from tract.corpus_report import wrong_anchor_applicable

        corpus = _corpus(tmp_path, [
            {"control_id": "WSTG-INFO-01", "title": "Information gathering",
             "description": LONG + " Gathering."},
        ])
        links = _links(tmp_path, [_row("WSTG-INFO-01", "WSTG-INFO-01")])
        assert wrong_anchor_applicable(build_corpus_report(links, corpus)) == {
            "demo": 0,
        }


class TestDetectorBApplicability:
    """Detector B is off where the link file names a different level from its ids.

    Two directions, and the predicate covers both.

    Names COARSER, ruling R11. DSOMM's link file carries 18 sub-dimension names
    against 183 activity uuids, and `section_name` equals the resolved control's
    title for 0 of 214 links. The 198 detector B reported were a fact about the
    source, not 198 wrong anchors.

    Names FINER, ruling R21. ETSI carries 24 technique names against 16 clause
    ids, so the id reaches a parent while the name describes one of its
    children. B compares a technique against a clause title, which is the same
    mismatch mirrored, and it reported 32 of 36 against a pre-registered budget
    of 1.

    Membership is derived from a measurable property rather than declared and
    trusted, and the first test below is the ratchet that holds the two equal.
    """

    def _framework_links(
        self, tmp_path: Path, rows: list[dict[str, str]], name: str = "links",
    ) -> Path:
        path = tmp_path / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
            encoding="utf-8",
        )
        return path

    def _link(
        self, framework_id: str, section_id: str, section_name: str,
        cre_id: str = "1-1",
    ) -> dict[str, str]:
        return {
            "framework_id": framework_id,
            "standard_name": "Demo",
            "section_id": section_id,
            "section_name": section_name,
            "cre_id": cre_id,
            "link_type": "LinkedTo",
        }

    def _shaped_links(
        self, tmp_path: Path, distinct_ids: int, distinct_names: int, name: str,
    ) -> Path:
        """A link file with exactly the requested id and name cardinalities.

        Built rather than written out because the fine boundary is 17 ids over
        20 names, and twenty literal rows would hide the one property the test
        is about. Every row past the smaller cardinality reuses that side's last
        value, so the ratio is exact rather than approximate.
        """
        rows = max(distinct_ids, distinct_names)
        return self._framework_links(tmp_path, [
            self._link(
                "shaped",
                f"a-{min(n, distinct_ids - 1)}",
                f"Name {min(n, distinct_names - 1)}",
            )
            for n in range(rows)
        ], name=name)

    def test_the_shaped_fixture_has_the_cardinalities_it_claims(
        self, tmp_path: Path,
    ) -> None:
        """The helper every boundary test below rests on, checked once.

        A helper that quietly produced 20 ids over 20 names would make each
        boundary test assert something other than what it reads as.
        """
        grouped = _load_links(self._shaped_links(tmp_path, 17, 20, "shape"))
        links = grouped["shaped"]
        assert len({r["section_id"] for r in links}) == 17
        assert len({r["section_name"] for r in links}) == 20

    def test_the_declared_set_equals_the_derived_set(self) -> None:
        """The ratchet, over the real curated link file.

        Fails in both directions by construction. A framework declared without
        the property is in the declared set and not the derived one. A
        framework that acquires the property and is not declared is in the
        derived set and not the declared one. Neither can land quietly.
        """
        assert DETECTOR_B_INAPPLICABLE == name_level_mismatch_frameworks()

    def _real_ratios(self) -> dict[str, float]:
        grouped = _load_links(CURATED_LINKS_PATH)
        assert len(grouped) == 22
        ratios: dict[str, float] = {}
        for framework_id, links in grouped.items():
            ids = {str(r.get("section_id") or "").strip() for r in links}
            names = {str(r.get("section_name") or "").strip() for r in links}
            ids.discard("")
            names.discard("")
            ratios[framework_id] = len(ids) / len(names)
        return ratios

    def test_the_real_link_file_splits_into_three_groups(self) -> None:
        """The measurement both thresholds rest on, asserted rather than cited.

        22 frameworks carry curated links. One sits above the coarse threshold,
        three sit below the fine one, and the other eighteen cluster around 1:1.
        Membership of each group is named here, so a link-file change that moves
        a framework between groups fails before it reaches the exemption set.
        """
        ratios = self._real_ratios()
        assert {k for k, v in ratios.items() if v >= COARSE_NAME_RATIO} == {
            "dsomm",
        }
        assert {k for k, v in ratios.items() if v <= FINE_NAME_RATIO} == {
            "enisa", "etsi", "nist_ai_100_2",
        }
        assert ratios["dsomm"] == pytest.approx(183 / 18)
        assert ratios["nist_ai_100_2"] == pytest.approx(20 / 28)
        assert ratios["etsi"] == pytest.approx(16 / 24)
        assert ratios["enisa"] == pytest.approx(10 / 33)

    def test_both_thresholds_sit_in_a_gap_the_data_opens(self) -> None:
        """Neither number is fitted to one framework's current shape.

        The nearest measured value on each side of each threshold, and the
        headroom between them. This fails if a threshold moves toward the data
        and it fails if the data moves toward a threshold, which is what makes
        the two numbers a measurement rather than a preference.
        """
        ratios = self._real_ratios()
        middle = sorted(
            v for v in ratios.values() if FINE_NAME_RATIO < v < COARSE_NAME_RATIO
        )
        assert len(middle) == 18

        # Fine side. mitre_atlas at 43/44 is the floor of the 1:1 cluster, which
        # is tighter than the 0.99 ruling R21 quotes, and is the value the
        # headroom above is measured against.
        assert middle[0] == pytest.approx(43 / 44)
        assert FINE_NAME_RATIO - ratios["nist_ai_100_2"] > 0.13
        assert middle[0] - FINE_NAME_RATIO > 0.12

        # Coarse side. biml at 20/17 is the top of the cluster.
        assert middle[-1] == pytest.approx(20 / 17)
        assert COARSE_NAME_RATIO - middle[-1] > 0.8
        assert ratios["dsomm"] - COARSE_NAME_RATIO > 8.0

    def test_the_reciprocal_of_the_coarse_threshold_would_miss_the_defect(
        self,
    ) -> None:
        """0.5 is the tidy fine threshold and it is empirically wrong.

        Stated as a test rather than only as a comment, because 1 / 2.0 is the
        change a later reader is most likely to make on symmetry grounds alone.
        """
        ratios = self._real_ratios()
        reciprocal = {k for k, v in ratios.items() if v <= 1.0 / COARSE_NAME_RATIO}
        assert "etsi" not in reciprocal
        assert "nist_ai_100_2" not in reciprocal

    def test_the_coarse_ratio_decides_membership_at_its_boundary(
        self, tmp_path: Path,
    ) -> None:
        """Exactly at the ratio is a member, just under it is not."""
        assert name_level_mismatch_frameworks(
            self._shaped_links(tmp_path, 4, 2, "coarse_at")
        ) == frozenset({"shaped"})
        assert name_level_mismatch_frameworks(
            self._shaped_links(tmp_path, 3, 2, "coarse_under")
        ) == frozenset()

    def test_the_fine_ratio_decides_membership_at_its_boundary(
        self, tmp_path: Path,
    ) -> None:
        """The mirror. 17/20 is exactly 0.85 and is a member, 18/20 is not."""
        assert name_level_mismatch_frameworks(
            self._shaped_links(tmp_path, 17, 20, "fine_at")
        ) == frozenset({"shaped"})
        assert name_level_mismatch_frameworks(
            self._shaped_links(tmp_path, 18, 20, "fine_over")
        ) == frozenset()

    def test_a_framework_between_the_two_thresholds_is_not_a_member(
        self, tmp_path: Path,
    ) -> None:
        """The eighteen 1:1 frameworks keep detector B.

        A predicate widened until it admits ETSI would take the whole corpus
        with it, and the equality ratchet alone cannot see that: it compares two
        sets built from the same link file. This reads the middle of the range
        directly.
        """
        for ids, names, label in ((1, 1, "one"), (43, 44, "atlas"), (20, 17, "biml")):
            assert name_level_mismatch_frameworks(
                self._shaped_links(tmp_path, ids, names, label)
            ) == frozenset(), label

    def test_a_link_file_with_no_names_does_not_divide_by_zero(
        self, tmp_path: Path,
    ) -> None:
        """Detector B never reads a name it was not given, so nothing to declare."""
        path = self._framework_links(tmp_path, [
            self._link("nameless", "a-1", ""),
            self._link("nameless", "a-2", ""),
        ])
        assert name_level_mismatch_frameworks(path) == frozenset()

    def test_detector_b_is_skipped_for_a_member_and_not_for_anyone_else(
        self, tmp_path: Path,
    ) -> None:
        """Identical link content under two framework_ids, one of them declared.

        The only variable between the two rows is membership, so a pass here
        cannot come from the corpus or the link shape.
        """
        corpus = _corpus(tmp_path, [
            {"control_id": "SD-1",
             "title": "Inventory of production components",
             "description": LONG + " Activity."},
        ])
        links = self._framework_links(tmp_path, [
            self._link("dsomm", "SD-1", "Deployment"),
            self._link("demo", "SD-1", "Deployment"),
        ])
        report = build_corpus_report(links, corpus)
        applicable = wrong_anchor_applicable(report)

        assert report.by_id("demo").by_id == 1
        assert report.by_id("demo").wrong_anchor_risk == 1
        assert applicable["demo"] == 1

        assert report.by_id("dsomm").by_id == 1
        assert report.by_id("dsomm").wrong_anchor_risk == 0
        # The denominator shrinks with the detector. B was the only applicable
        # check on this link, so counting it would report 0 of 1 as a pass.
        assert applicable["dsomm"] == 0

    def test_detector_c_still_fires_for_a_member(self, tmp_path: Path) -> None:
        """Only B is switched off. C is the id-side check B cannot make.

        An implementation that exempts the framework from the whole
        wrong-anchor column instead of from one detector reads 0 here.
        """
        shared = LONG + " Shared subsection."
        corpus = _corpus(tmp_path, [
            {"control_id": "5.2", "title": "Build", "description": shared},
            {"control_id": "5.2.1", "title": "Build pipeline",
             "description": shared},
        ])
        links = self._framework_links(tmp_path, [
            self._link("dsomm", "5.2", "5.2"),
            self._link("dsomm", "5.2.1", "5.2.1"),
        ])
        report = build_corpus_report(links, corpus)
        assert report.by_id("dsomm").distinct_anchors == 1
        assert report.by_id("dsomm").wrong_anchor_risk == 2
        assert wrong_anchor_applicable(report)["dsomm"] == 2

    def test_detector_a_still_fires_for_a_member(self, tmp_path: Path) -> None:
        """The title-channel check, which B's exemption must not reach."""
        corpus = _corpus(tmp_path, [
            {"control_id": "2.3", "title": "Poisoning attacks",
             "description": LONG + " Predictive."},
            {"control_id": "3.2.2", "title": "Generative poisoning",
             "description": LONG + " Generative.",
             "metadata": {"alt_titles": ["Poisoning attacks"]}},
        ])
        links = self._framework_links(tmp_path, [
            self._link("dsomm", "3.2.2", "Poisoning attacks"),
        ])
        report = build_corpus_report(links, corpus)
        assert report.by_id("dsomm").by_title == 1
        assert report.by_id("dsomm").wrong_anchor_risk == 1
        assert wrong_anchor_applicable(report)["dsomm"] == 1

    def test_the_real_dsomm_row_reports_zero_over_a_live_denominator(
        self,
    ) -> None:
        """0 of 3, not 198 of 213, and not 0 of 0.

        The three survivors are the uuid-suffixed WAF ids that detector C
        reaches. A zero over a zero denominator would mean nothing checked
        this framework at all, which is the reading wrong_anchor_applicable
        exists to prevent.
        """
        if not merged_corpus_path().exists():
            pytest.skip("no merged corpus in this checkout")
        if not _corpus_carries_prose("dsomm"):
            # Ruling R10 put DSOMM in OVERLAY_FRAMEWORK_IDS, so the tracked
            # corpus carries its titles and withholds its statements. Nothing
            # resolves, the denominator is 0 by construction, and that is the
            # licence tiering working rather than the detector failing. The
            # live case is asserted wherever the gitignored overlay exists.
            pytest.skip("this checkout's corpus withholds DSOMM's prose")
        report = build_corpus_report()
        assert report.by_id("dsomm").wrong_anchor_risk == 0
        assert wrong_anchor_applicable(report)["dsomm"] == 3

    def test_detector_b_is_skipped_for_a_fine_name_member(
        self, tmp_path: Path,
    ) -> None:
        """ETSI's shape: the id reaches a clause, the name is a technique in it.

        Identical link content under two framework_ids, one of them declared, so
        the only variable between the two rows is membership.
        """
        corpus = _corpus(tmp_path, [
            {"control_id": "6.3", "title": "Model hardening",
             "description": LONG + " Clause."},
        ])
        links = self._framework_links(tmp_path, [
            self._link("etsi", "6.3", "Mitigating model stealing"),
            self._link("demo", "6.3", "Mitigating model stealing"),
        ])
        report = build_corpus_report(links, corpus)
        applicable = wrong_anchor_applicable(report)

        assert report.by_id("demo").by_id == 1
        assert report.by_id("demo").wrong_anchor_risk == 1
        assert applicable["demo"] == 1

        assert report.by_id("etsi").by_id == 1
        assert report.by_id("etsi").wrong_anchor_risk == 0
        # B was the only applicable check, so the denominator goes with it.
        assert applicable["etsi"] == 0

    def test_detector_c_still_fires_for_a_fine_name_member(
        self, tmp_path: Path,
    ) -> None:
        """A rolled-up ETSI clause and its child presenting one anchor.

        An implementation that exempts a fine-name framework from the whole
        wrong-anchor column instead of from detector B reads 0 here.
        """
        shared = LONG + " Rolled-up clause."
        corpus = _corpus(tmp_path, [
            {"control_id": "6.2", "title": "Model stealing", "description": shared},
            {"control_id": "6.2.1", "title": "Query throttling",
             "description": shared},
        ])
        links = self._framework_links(tmp_path, [
            self._link("etsi", "6.2", "6.2"),
            self._link("etsi", "6.2.1", "6.2.1"),
        ])
        report = build_corpus_report(links, corpus)
        assert report.by_id("etsi").distinct_anchors == 1
        assert report.by_id("etsi").wrong_anchor_risk == 2
        assert wrong_anchor_applicable(report)["etsi"] == 2

    def test_detector_a_still_fires_for_a_fine_name_member(
        self, tmp_path: Path,
    ) -> None:
        """The title-channel check, which the fine exemption must not reach."""
        corpus = _corpus(tmp_path, [
            {"control_id": "6.3", "title": "Mitigating model stealing",
             "description": LONG + " Clause 6.3."},
            {"control_id": "6.3.1", "title": "Query throttling",
             "description": LONG + " Clause 6.3.1.",
             "metadata": {"alt_titles": ["Mitigating model stealing"]}},
        ])
        links = self._framework_links(tmp_path, [
            self._link("etsi", "6.3.1", "Mitigating model stealing"),
        ])
        report = build_corpus_report(links, corpus)
        assert report.by_id("etsi").by_title == 1
        assert report.by_id("etsi").wrong_anchor_risk == 1
        assert wrong_anchor_applicable(report)["etsi"] == 1

    def _framework_join(self, tmp_path: Path, framework_id: str) -> CorpusReport:
        """The curated-link join against one framework's own tracked artifact.

        Not against data/processed/all_controls.json. Ruling R15 keeps that
        shared derived file out of a parser task's commit, so it lags the
        parsers that have landed and a test reading it asserts state no commit
        carries. Every column the join reports is computed per framework, so a
        one-framework corpus produces the same row, and the three figures below
        were confirmed identical against the full 31-framework overlay.
        tests/test_parse_enisa.py builds its join the same way.
        """
        record_path = PROCESSED_FRAMEWORKS_DIR / f"{framework_id}.json"
        if not record_path.exists():
            pytest.skip(f"{framework_id} has no processed artifact in this checkout")
        corpus = tmp_path / f"{framework_id}.json"
        corpus.write_text(
            json.dumps(
                {"frameworks": [json.loads(record_path.read_text(encoding="utf-8"))]},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return build_corpus_report(corpus_path=corpus)

    def test_the_real_etsi_row_meets_its_pre_registered_budget(
        self, tmp_path: Path,
    ) -> None:
        """1 of 9, against the 1 JOIN_WRONG_ANCHOR_BUDGET registered.

        Before the predicate became symmetric this read 32 of 36. Detector B
        supplied 31 of those 32, and the survivor is the 6.3.1 row the budget
        entry names, resolved through the title channel by detector A. The
        denominator stays live at 9, five title-channel checks and four
        id-channel ones, so the 1 is not a zero over nothing.

        ETSI is in RESTRICTED_FRAMEWORK_IDS, so its processed artifact is
        gitignored and this skips in a checkout that has no licensed text. The
        skip is stated rather than implied, and the predicate tests above assert
        etsi's membership from the tracked link file, so the exemption itself
        stays gated everywhere.
        """
        report = self._framework_join(tmp_path, "etsi")
        row = report.by_id("etsi")
        assert row.by_title == 5
        assert row.by_id == 31
        assert row.wrong_anchor_risk == 1
        assert wrong_anchor_applicable(report)["etsi"] == 9
        assert row.wrong_anchor_risk == JOIN_WRONG_ANCHOR_BUDGET["etsi"]
        flagged = [
            entry for entry in report.resolution_rows
            if entry.framework_id == "etsi" and entry.wrong_anchor
        ]
        assert [entry.section_id for entry in flagged] == ["6.3.1"]
        assert [entry.channel for entry in flagged] == ["title"]

    def test_the_real_nist_ai_100_2_row_keeps_its_title_channel_findings(
        self, tmp_path: Path,
    ) -> None:
        """8 of 29, down from 20 of 45, and every survivor is detector A.

        The twelve that left were id-channel detector B flags on links whose
        names sit one level finer than their ids. The eight that stay are
        title-channel links pointing at Sec. 2.2.4 whose names resolve to their
        own technique instead, which is a real disagreement between the two
        channels and has to survive an exemption aimed at detector B.
        """
        report = self._framework_join(tmp_path, "nist_ai_100_2")
        assert report.by_id("nist_ai_100_2").wrong_anchor_risk == 8
        assert wrong_anchor_applicable(report)["nist_ai_100_2"] == 29
        flagged = [
            entry for entry in report.resolution_rows
            if entry.framework_id == "nist_ai_100_2" and entry.wrong_anchor
        ]
        assert {entry.channel for entry in flagged} == {"title"}

    def test_the_real_enisa_row_does_not_move(self, tmp_path: Path) -> None:
        """Declared, and unchanged, which is the point of declaring it.

        Every enisa link resolves through the title channel, so detector B never
        ran for it and the exemption costs nothing. Membership follows the link
        file rather than the run, and this is the assertion that says so. An
        implementation that switched off the whole wrong-anchor column for a
        member would take the denominator of 68 with it.
        """
        report = self._framework_join(tmp_path, "enisa")
        assert "enisa" in DETECTOR_B_INAPPLICABLE
        assert report.by_id("enisa").by_id == 0
        assert report.by_id("enisa").wrong_anchor_risk == 0
        assert wrong_anchor_applicable(report)["enisa"] == 68


class TestFloors:
    def test_a_framework_below_its_floor_is_reported(self, tmp_path: Path) -> None:
        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "One", "description": LONG},
        ])
        links = _links(tmp_path, [_row("C-1", "One"), _row("C-2", "Two")])
        report = build_corpus_report(links, corpus)
        assert check_join_floors(report, {"demo": 0.50}) == []
        assert len(check_join_floors(report, {"demo": 0.90})) == 1

    def test_a_floor_for_an_absent_framework_raises(self, tmp_path: Path) -> None:
        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "One", "description": LONG},
        ])
        links = _links(tmp_path, [_row("C-1", "One")])
        report = build_corpus_report(links, corpus)
        with pytest.raises(KeyError, match="no curated links"):
            check_join_floors(report, {"absent": 0.50})

    def test_the_restricted_group_is_dropped_by_name_not_by_deletion(
        self, tmp_path: Path,
    ) -> None:
        """Rule 7. CI has 29 frameworks, the overlay has 31."""
        from tract.corpus_report import floors_for_report

        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "One", "description": LONG},
        ])
        links = _links(tmp_path, [_row("C-1", "One")])
        report = build_corpus_report(links, corpus)
        applicable, skipped = floors_for_report(
            report, {"demo": 0.50, "etsi": 1.00}, frozenset({"etsi"}),
        )
        assert applicable == {"demo": 0.50}
        assert skipped == ["etsi"]


class TestDerivedFloors:
    """The criterion is committed before the run it gates, and it is checkable."""

    def test_every_floor_has_a_ceiling(self) -> None:
        assert sorted(JOIN_FLOORS) == sorted(JOIN_CEILINGS)

    def test_no_floor_exceeds_its_arithmetic_ceiling(self) -> None:
        """Not `floor <= 1.0`, which is true of every literal in the dict.

        The previous plan carried three impossible floors: dsomm 1.00 against a
        maximum of 0.9953, wstg 0.96 against 0.9322, enisa 0.80 against 0.721.
        """
        for framework_id, floor in JOIN_FLOORS.items():
            ceiling = JOIN_CEILINGS[framework_id]
            assert 0.0 < floor <= ceiling, framework_id

    def test_each_floor_is_its_ceiling_rounded_down(self) -> None:
        for framework_id, ceiling in JOIN_CEILINGS.items():
            expected = math.floor(round(ceiling * 100, 6)) / 100
            assert JOIN_FLOORS[framework_id] == pytest.approx(expected), (
                framework_id
            )

    def test_the_floors_cover_the_eleven_pending_frameworks(self) -> None:
        assert {
            "biml", "csa_ccm", "dsomm", "enisa", "etsi", "nist_800_63",
            "nist_ssdf", "owasp_proactive_controls", "owasp_top10_2021",
            "samm", "wstg",
        } <= set(JOIN_FLOORS)


class TestChannelParity:
    def test_report_and_lookup_agree_on_every_curated_link(self) -> None:
        """The report must describe the join the pipeline performs."""
        from tract.corpus_report import (
            CURATED_LINKS_PATH, _load_records, _lookup_with_channel,
        )
        from tract.text_selection import (
            ProseIndex, canonical_framework, merged_corpus_path,
        )

        corpus = merged_corpus_path()
        records = _load_records(corpus)
        # The bug this test exists to prevent: an empty index agrees with
        # itself on everything.
        assert records, f"{corpus} produced no framework records"
        index = ProseIndex(records)
        assert len(index) > 1000, (
            f"prose index holds {len(index)} controls, and a near-empty index "
            f"makes every assertion below vacuous"
        )

        compared = 0
        with CURATED_LINKS_PATH.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                canonical = canonical_framework(row.get("standard_name", ""))
                mine, _ = _lookup_with_channel(
                    index, canonical, row.get("section_id"),
                    row.get("section_name"),
                )
                theirs = index.lookup(
                    row.get("standard_name", ""), row.get("section_id"),
                    row.get("section_name"),
                )
                assert (mine is None) == (theirs is None), row
                if mine is not None and theirs is not None:
                    assert mine.text == theirs.text, row
                compared += 1
        assert compared == 4405, compared

    def test_the_report_resolves_a_useful_number_of_links(self) -> None:
        """A second guard against an index that silently held nothing."""
        from tract.corpus_report import (
            FULL_CORPUS_FRAMEWORK_COUNT, TRACKED_CORPUS_FRAMEWORK_COUNT,
        )
        from tract.config import PROCESSED_LICENSED_DIR

        report = build_corpus_report()
        resolved = report.totals.by_title + report.totals.by_id
        assert report.totals.links == 4405
        assert resolved >= 3600, resolved

        # `>= 29` passed at both censuses, so it could not tell a checkout
        # with the overlay from one without. The census is one of exactly two
        # known values, and which one is decided by whether the overlay file
        # is on disk. A corpus that quietly lost a framework fails here.
        overlay = PROCESSED_LICENSED_DIR / "all_controls.json"
        expected = (
            FULL_CORPUS_FRAMEWORK_COUNT if overlay.exists()
            else TRACKED_CORPUS_FRAMEWORK_COUNT
        )
        assert report.corpus_framework_count == expected, (
            f"corpus holds {report.corpus_framework_count} frameworks against "
            f"{expected} expected with overlay present={overlay.exists()}"
        )


class TestEvidenceGuards:
    """A tagged artifact is committed evidence, so both guards run first."""

    def test_a_partial_corpus_cannot_overwrite_the_baseline(
        self, tmp_path: Path,
    ) -> None:
        """--tag before on a checkout with no overlay is ledger lesson 5.

        The report still builds, because floors_for_report handles a partial
        corpus by name. Only the tagged write is refused.
        """
        from tract.corpus_report import require_full_corpus

        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "One", "description": LONG},
        ])
        report = build_corpus_report(_links(tmp_path, [_row("C-1", "One")]), corpus)
        assert report.corpus_framework_count == 1
        # Match text unique to the census guard. `require_full_corpus` and
        # `require_portable_paths` both open "refusing to write tagged
        # evidence", so matching that prefix alone accepts either one. A
        # mutation that skipped the census guard entirely already passed a test
        # written that way, because portable-paths fired on the tmp_path corpus
        # and the message still matched.
        with pytest.raises(ValueError, match=r"from a corpus of 1 frameworks"):
            require_full_corpus(report)

    def test_the_full_corpus_passes_the_census_guard(self) -> None:
        """The guard must be reachable in the passing direction too."""
        from tract.corpus_report import (
            FULL_CORPUS_FRAMEWORK_COUNT, require_full_corpus,
        )
        from tract.config import PROCESSED_LICENSED_DIR

        if not (PROCESSED_LICENSED_DIR / "all_controls.json").exists():
            pytest.skip("no licensed overlay in this checkout")
        report = build_corpus_report()
        assert report.corpus_framework_count == FULL_CORPUS_FRAMEWORK_COUNT
        require_full_corpus(report)

    def test_an_absolute_path_cannot_reach_a_committed_artifact(
        self, tmp_path: Path,
    ) -> None:
        """tmp_path sits outside the repository, so both paths stay absolute."""
        from tract.corpus_report import require_portable_paths

        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "One", "description": LONG},
        ])
        report = build_corpus_report(_links(tmp_path, [_row("C-1", "One")]), corpus)
        with pytest.raises(ValueError, match="absolute path"):
            require_portable_paths(report)

    def test_the_repo_corpus_records_relative_paths(self) -> None:
        from tract.corpus_report import require_portable_paths

        report = build_corpus_report()
        assert not Path(report.corpus_path).is_absolute(), report.corpus_path
        assert not Path(report.links_path).is_absolute(), report.links_path
        require_portable_paths(report)

    def test_no_committed_evidence_file_carries_an_absolute_path(self) -> None:
        """A CC0 repository must not publish the author's home directory.

        Checks the bytes on disk rather than the object in memory, because the
        artifact is what ships.
        """
        from tract.corpus_report import CORPUS_EVIDENCE_DIR

        written = sorted(CORPUS_EVIDENCE_DIR.glob("*"))
        assert written, f"no evidence under {CORPUS_EVIDENCE_DIR}"
        for path in written:
            body = path.read_text(encoding="utf-8")
            for marker in ("/Users/", "/home/", "/root/", "C:\\\\"):
                assert marker not in body, (
                    f"{path.name} carries {marker!r}, which is an absolute "
                    f"machine path in a committed artifact"
                )


class TestUnmovedCorpusGuard:
    """Ruling R12. A tag may be reproduced, never silently replaced.

    require_full_corpus checks the framework COUNT, and a parser rewrites what
    the frameworks contain without changing how many there are. The DSOMM
    parser moved the corpus sha256 from 2440d7c0 to 5b0a4289 with the count at
    31 throughout, so the documented `--tag before` command could replace the
    plan's reference baseline with a report built from different bytes and no
    existing guard could see it.
    """

    def _report(self, tmp_path: Path) -> CorpusReport:
        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "One", "description": LONG},
        ])
        return build_corpus_report(_links(tmp_path, [_row("C-1", "One")]), corpus)

    def _artifact(self, path: Path, corpus_sha256: object) -> None:
        """An existing tagged artifact recording some corpus digest."""
        payload: dict[str, object] = {"per_framework": [], "totals": {}}
        if corpus_sha256 is not None:
            payload["corpus_sha256"] = corpus_sha256
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def test_a_tag_that_does_not_exist_yet_is_written_without_an_override(
        self, tmp_path: Path,
    ) -> None:
        """First capture of a tag is not a replacement of anything."""
        report = self._report(tmp_path)
        require_unmoved_corpus(report, tmp_path / "absent.json")

    def test_the_same_corpus_reproduces_the_tag_without_an_override(
        self, tmp_path: Path,
    ) -> None:
        """The direction that matters most.

        Regenerating a tag from the corpus that produced it is the
        byte-identical reproduction property Task 1 established and every task
        since has checked. A guard that blocked it would retire that check.
        """
        report = self._report(tmp_path)
        summary = tmp_path / "before.json"
        self._artifact(summary, report.corpus_sha256)
        require_unmoved_corpus(report, summary)

    def test_a_moved_corpus_is_refused(self, tmp_path: Path) -> None:
        report = self._report(tmp_path)
        summary = tmp_path / "before.json"
        self._artifact(summary, "2440d7c062055f66" + "0" * 48)
        with pytest.raises(ValueError, match="built from a different corpus"):
            require_unmoved_corpus(report, summary)

    def test_the_refusal_names_both_digests_and_the_file(
        self, tmp_path: Path,
    ) -> None:
        """A refusal a reader cannot act on is an obstacle, not a guard."""
        report = self._report(tmp_path)
        summary = tmp_path / "before.json"
        recorded = "2440d7c062055f66" + "0" * 48
        self._artifact(summary, recorded)
        with pytest.raises(ValueError) as error:
            require_unmoved_corpus(report, summary)
        message = str(error.value)
        assert recorded in message
        assert report.corpus_sha256 in message
        assert "before.json" in message
        assert "--replace-baseline" in message

    def test_the_refusal_leaves_the_existing_artifact_byte_identical(
        self, tmp_path: Path,
    ) -> None:
        """The guard must run before the write, not alongside it."""
        report = self._report(tmp_path)
        summary = tmp_path / "before.json"
        self._artifact(summary, "2440d7c062055f66" + "0" * 48)
        before = summary.read_bytes()
        with pytest.raises(ValueError):
            require_unmoved_corpus(report, summary)
        assert summary.read_bytes() == before

    def test_an_artifact_with_no_recorded_digest_is_refused(
        self, tmp_path: Path,
    ) -> None:
        """Unreadable provenance is not permission to overwrite."""
        report = self._report(tmp_path)
        summary = tmp_path / "before.json"
        self._artifact(summary, None)
        with pytest.raises(ValueError, match="no 'corpus_sha256'"):
            require_unmoved_corpus(report, summary)

    def test_an_unparseable_artifact_is_refused(self, tmp_path: Path) -> None:
        report = self._report(tmp_path)
        summary = tmp_path / "before.json"
        summary.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="not valid JSON"):
            require_unmoved_corpus(report, summary)

    def test_the_committed_baseline_is_what_the_guard_now_refuses(self) -> None:
        """The live case, asserted rather than described.

        results/corpus/before.json records the pre-DSOMM corpus. Today's corpus
        is a different one, so `--tag before` without an override must refuse.
        Both hold the same framework count, which is why require_full_corpus
        passes on the very run this guard has to stop.
        """
        if not merged_corpus_path().exists():
            pytest.skip("no merged corpus in this checkout")
        summary = CORPUS_EVIDENCE_DIR / "before.json"
        if not summary.exists():
            pytest.skip("no committed baseline in this checkout")
        recorded = json.loads(summary.read_text(encoding="utf-8"))
        if not str(merged_corpus_path()).endswith(str(recorded["corpus_path"])):
            # The baseline was taken from the gitignored overlay and this
            # checkout reads the tracked corpus, which holds two frameworks
            # fewer by licence rather than by drift. The counts differ for a
            # reason the guard is not about, so the premise below is false and
            # asserting it would report a licence tier as a corpus move.
            pytest.skip(
                f"baseline recorded {recorded['corpus_path']} and this checkout "
                f"reads {merged_corpus_path()}"
            )
        report = build_corpus_report()
        if recorded["corpus_sha256"] == report.corpus_sha256:
            pytest.skip("baseline and corpus agree; nothing for the guard to stop")
        assert recorded["corpus_framework_count"] == report.corpus_framework_count
        require_full_corpus(report)          # passes, and cannot see the move
        with pytest.raises(ValueError, match="built from a different corpus"):
            require_unmoved_corpus(report, summary)


class TestReplaceBaselineOverride:
    """The override is a decision. It is not a way around the census guard."""

    def _argv(self, *extra: str) -> list[str]:
        return ["corpus_report.py", *extra]

    def test_the_flag_exists_and_defaults_to_off(self) -> None:
        import scripts.corpus_report as script

        parser = script.build_parser()
        assert parser.parse_args([]).replace_baseline is False
        assert parser.parse_args(["--replace-baseline"]).replace_baseline is True

    def test_the_help_text_says_what_it_destroys(self) -> None:
        """A flag whose help does not name the loss is a trap, not an option."""
        import scripts.corpus_report as script

        help_text = script.build_parser().format_help()
        assert "DESTRUCTIVE" in help_text
        assert "baseline" in help_text
        assert "no copy kept" in help_text

    def test_the_override_reaches_the_write_and_the_guard_does_not_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With the flag, a moved corpus writes. Without it, it raises.

        Both directions over one fixture and one tag, so a pass cannot come
        from the corpus or from the tag name. require_full_corpus is stood
        down here only because this checkout's tracked corpus is short of the
        full set; that guard has its own test below and it is NOT stood down
        there.
        """
        import scripts.corpus_report as script

        evidence = tmp_path / "evidence"
        evidence.mkdir()
        monkeypatch.setattr(script, "CORPUS_EVIDENCE_DIR", evidence)
        monkeypatch.setattr(script, "require_full_corpus", lambda r: None)

        corpus = PROJECT_ROOT / "data" / "processed" / "all_controls.json"
        report = build_corpus_report(CURATED_LINKS_PATH, corpus)

        summary = evidence / "probe.json"
        summary.write_text(
            json.dumps({"corpus_sha256": "0" * 64}), encoding="utf-8",
        )
        stale = summary.read_bytes()

        monkeypatch.setattr(
            "sys.argv", self._argv("--tag", "probe", "--corpus", str(corpus)),
        )
        with pytest.raises(ValueError, match="built from a different corpus"):
            script.main()
        assert summary.read_bytes() == stale

        monkeypatch.setattr(
            "sys.argv",
            self._argv("--tag", "probe", "--corpus", str(corpus),
                       "--replace-baseline"),
        )
        script.main()
        assert summary.read_bytes() != stale
        assert json.loads(summary.read_text(encoding="utf-8"))["corpus_sha256"] == (
            report.corpus_sha256
        )

    def test_the_override_does_not_bypass_the_census_guard(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A recapture from a partial checkout is the same defect with a flag.

        Matched on text unique to require_full_corpus, not on the shared
        "refusing to write tagged evidence" opening. require_portable_paths
        opens with the same words, and a tmp_path corpus trips it too, so the
        looser pattern passed whichever guard fired and could not tell a
        working census guard from a bypassed one. Mutation R9 survived it.
        """
        import scripts.corpus_report as script

        evidence = tmp_path / "evidence"
        evidence.mkdir()
        monkeypatch.setattr(script, "CORPUS_EVIDENCE_DIR", evidence)

        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "One", "description": LONG},
        ])
        links = _links(tmp_path, [_row("C-1", "One")])
        monkeypatch.setattr(
            "sys.argv",
            self._argv("--tag", "probe", "--corpus", str(corpus),
                       "--links", str(links), "--replace-baseline"),
        )
        with pytest.raises(ValueError) as error:
            script.main()
        message = str(error.value)
        assert "from a corpus of 1 frameworks against" in message, message
        assert "absolute path" not in message, (
            "require_portable_paths fired, so this says nothing about whether "
            "the census guard still binds under the override"
        )
        assert not (evidence / "probe.json").exists()

    def test_an_out_write_stays_unguarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--out is scratch. Nothing is gated on it, so nothing guards it.

        Written twice from different corpora over the same path, which is the
        exact move --tag now refuses.
        """
        import scripts.corpus_report as script

        scratch = tmp_path / "scratch.json"
        tracked = PROJECT_ROOT / "data" / "processed" / "all_controls.json"
        small = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "One", "description": LONG},
        ])
        links = _links(tmp_path, [_row("C-1", "One")])

        monkeypatch.setattr(
            "sys.argv",
            self._argv("--out", str(scratch), "--corpus", str(tracked)),
        )
        script.main()
        first = json.loads(scratch.read_text(encoding="utf-8"))["corpus_sha256"]

        monkeypatch.setattr(
            "sys.argv",
            self._argv("--out", str(scratch), "--corpus", str(small),
                       "--links", str(links)),
        )
        script.main()
        second = json.loads(scratch.read_text(encoding="utf-8"))["corpus_sha256"]
        assert first != second


class TestAlternateIdsAgainstTheRealCorpus:
    """The new channel, exercised on the corpus rather than on a fixture.

    cwe resolves 612 of 613 curated links. The miss is section_id "937", an
    obsolete CWE category whose description is shorter than its title plus
    PROSE_MIN_EXTRA_CHARS, so ProseIndex never indexes it. [measured]
    Attaching "937" as an alt_id to any indexed CWE control closes the gap,
    which is a property of the channel rather than of that control.

    cwe is deliberately the subject: it is in the tracked corpus, so these run
    on a checkout with no licensed overlay.
    """

    def _cwe_only(self, tmp_path: Path) -> tuple[Path, Path]:
        from tract.corpus_report import CURATED_LINKS_PATH, _load_records
        from tract.text_selection import merged_corpus_path

        records = [
            record for record in _load_records(merged_corpus_path())
            if record.get("framework_name") == "CWE"
        ]
        assert len(records) == 1, "expected exactly one CWE record"
        corpus = tmp_path / "cwe.json"
        corpus.write_text(
            json.dumps(
                {"framework_count": 1, "frameworks": records,
                 "generated_date": "2026-01-01",
                 "total_controls": len(records[0]["controls"])},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        rows = [
            line for line in
            CURATED_LINKS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line)["framework_id"] == "cwe"
        ]
        links = tmp_path / "cwe.jsonl"
        links.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return corpus, links

    @staticmethod
    def _is_indexed(control: dict[str, object]) -> bool:
        """The prose rule, spelled out, so the donor is one ProseIndex holds."""
        return bool(str(control.get("full_text") or "").strip()) or (
            len(str(control.get("description") or "").strip())
            > len(str(control.get("title") or "").strip())
            + PROSE_MIN_EXTRA_CHARS
        )

    def test_the_unresolved_cwe_link_stays_unresolved_without_an_alt_id(
        self, tmp_path: Path,
    ) -> None:
        corpus, links = self._cwe_only(tmp_path)
        row = build_corpus_report(links, corpus).by_id("cwe")
        assert row.links == 613
        assert row.unresolved == 1
        assert row.by_title + row.by_id == 612
        assert row.by_id == 18

    def test_an_alt_id_closes_it(self, tmp_path: Path) -> None:
        corpus, links = self._cwe_only(tmp_path)
        data = json.loads(corpus.read_text(encoding="utf-8"))
        controls = data["frameworks"][0]["controls"]
        target = next(c for c in controls if self._is_indexed(c))
        metadata = dict(target.get("metadata") or {})
        metadata["alt_ids"] = ["937"]
        target["metadata"] = metadata
        corpus.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")

        row = build_corpus_report(links, corpus).by_id("cwe")
        assert row.unresolved == 0
        # 18 before, so the id channel carried the new one and the title
        # channel did not quietly answer it instead.
        assert row.by_id == 19
        assert row.resolution_rate == pytest.approx(1.0)

    @pytest.mark.parametrize("donor_after_the_real_id", [False, True])
    def test_an_alt_id_cannot_take_a_live_cwe_id(
        self, tmp_path: Path, donor_after_the_real_id: bool,
    ) -> None:
        """The guarantee, on real data: 79 is a real CWE and must not move.

        Run in both corpus orders. The donor before CWE-79 fails an
        implementation that writes alternates in the first pass under
        first-writer-wins. The donor after it fails one that writes them in the
        first pass under the real ids' last-writer-wins rule. One order alone
        leaves half the failure space unreachable.
        """
        corpus, links = self._cwe_only(tmp_path)
        data = json.loads(corpus.read_text(encoding="utf-8"))
        controls = data["frameworks"][0]["controls"]
        position = next(
            i for i, c in enumerate(controls) if str(c["control_id"]) == "79"
        )
        real = controls[position]
        assert self._is_indexed(real), "CWE-79 must be in the index to be taken"
        candidates = (
            controls[position + 1:] if donor_after_the_real_id
            else controls[:position]
        )
        other = next(
            c for c in candidates
            if str(c["control_id"]) != "79" and self._is_indexed(c)
        )
        metadata = dict(other.get("metadata") or {})
        metadata["alt_ids"] = ["79"]
        other["metadata"] = metadata
        corpus.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")

        from tract.corpus_report import _load_records
        from tract.text_selection import ProseIndex

        index = ProseIndex(_load_records(corpus))
        hit = index.by_id("CWE", "79")
        assert hit is not None
        expected = str(real.get("full_text") or "").strip() or str(
            real["description"]
        ).strip()
        assert hit.text == expected
        # The donor's own text is distinct, so the assertion above discriminates
        # rather than passing because both controls read the same.
        donor_text = str(other.get("full_text") or "").strip() or str(
            other["description"]
        ).strip()
        assert donor_text != expected


class TestIsoStillResolves:
    """92 of 94 lived only in a comment. Now it is a gate.

    Skipped as a named group when the licensed overlay is absent, per Rule 7,
    because gating on file existence never skips: the tracked corpus always
    exists and the restricted rows would hard-fail in CI on data that cannot
    legally be there.
    """

    def test_iso_resolves_92_of_94_with_91_distinct_anchors(self) -> None:
        from tract.corpus_report import FULL_CORPUS_FRAMEWORK_COUNT

        report = build_corpus_report()
        if report.corpus_framework_count < FULL_CORPUS_FRAMEWORK_COUNT:
            pytest.skip(
                f"corpus has {report.corpus_framework_count} frameworks "
                f"against {FULL_CORPUS_FRAMEWORK_COUNT} in the full set, so "
                f"the licensed overlay is absent from this checkout and the "
                f"restricted rows cannot be asserted"
            )
        row = report.by_id("iso_27001")
        assert row.links == 94
        assert row.by_title + row.by_id == 92
        assert row.distinct_anchors == 91
        assert row.dropped_by_prose_rule == 2


class TestGuardMessagesAreDistinguishable:
    """Two guards opening with the same words made a passing test meaningless.

    `require_full_corpus` and `require_portable_paths` both begin "refusing to
    write tagged evidence". A mutation that skipped the census guard was caught
    by portable-paths instead, the message still satisfied
    `pytest.raises(match="refusing to write tagged evidence")`, and the test
    went green while asserting nothing about the guard it was named for.

    The messages are not rewritten here, because other tests match on them and
    the wording carries real diagnostic value. What is asserted is that no
    guard's message can be identified by a prefix that another guard shares, so
    a future `match=` written lazily fails loudly rather than passing for the
    wrong reason.
    """

    def _messages(self) -> dict[str, str]:
        """One raised message per guard, each triggered on its own condition."""
        import inspect

        from tract import corpus_report

        sources = {
            name: inspect.getsource(obj)
            for name, obj in vars(corpus_report).items()
            if name.startswith("require_") and inspect.isfunction(obj)
        }
        assert len(sources) >= 3, (
            f"expected at least three require_* guards, found {sorted(sources)}"
        )
        opens: dict[str, str] = {}
        for name, src in sources.items():
            match = re.search(r'f?"(refusing to [^"]+)"', src)
            assert match, f"{name} raises no message beginning 'refusing to'"
            opens[name] = match.group(1)
        return opens

    def test_no_guard_message_is_a_prefix_of_another(self) -> None:
        opens = self._messages()
        for name, text in opens.items():
            for other_name, other in opens.items():
                if name == other_name:
                    continue
                shared = os.path.commonprefix([text, other])
                assert shared != text and shared != other, (
                    f"{name} and {other_name} share the whole of the shorter "
                    f"message, so pytest.raises(match=...) on it cannot tell "
                    f"them apart:\n  {name}: {text!r}\n  {other_name}: {other!r}"
                )

    def test_every_guard_is_identifiable_within_its_first_forty_characters(
        self,
    ) -> None:
        """Forty characters is about what a reader copies into a `match=`."""
        opens = self._messages()
        heads = {name: text[:40] for name, text in opens.items()}
        collisions = [
            (a, b) for a in heads for b in heads if a < b and heads[a] == heads[b]
        ]
        assert not collisions, (
            f"these guards are indistinguishable in their first forty "
            f"characters: {collisions}. Give one of each pair a distinct "
            f"opening so a short match cannot select the wrong guard."
        )
