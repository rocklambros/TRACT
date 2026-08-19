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
from pathlib import Path

import pytest

from tract.config import PROJECT_ROOT, PROSE_MIN_EXTRA_CHARS
from tract.corpus_report import (
    JOIN_CEILINGS,
    JOIN_FLOORS,
    build_corpus_report,
    check_join_floors,
)

TRACKED_CORPUS = PROJECT_ROOT / "data" / "processed" / "all_controls.json"

LONG = "A control statement long enough to clear every prose bar. " * 4


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
        with pytest.raises(ValueError, match="refusing to write tagged evidence"):
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
