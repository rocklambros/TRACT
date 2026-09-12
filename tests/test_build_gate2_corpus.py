"""The Gate 2 training corpus must be a real, deterministic, declared file.

The draft plan named two arms, "round 1 (80 links)" and "round 2 (173 links)",
and neither existed. `load_bridge_links` requires `path.is_file()`; the rounds on
disk are per-annotator pairs. The confidence floor that defines "80" lives only
in `scripts/analysis/gate1_report.py` and never reaches training, so pointing
`--bridge-links` at a naive concatenation trains on 86 rows including six the
pre-registration classifies as "data, not evidence" -- while every artifact and
the write-up say 80.

The merge is therefore not a convenience. It is the step where the number in the
document becomes the number in the weights, and it has to be reproducible by
someone who was not here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_gate2_corpus import (
    CorpusStats,
    build_manifest,
    merge_round,
    write_corpus,
)
from tract.bridge.links import load_bridge_links


def _link(
    section: str,
    cre: str,
    *,
    annotator: str = "vol-01",
    confidence: int = 3,
    framework: str = "nist_800_53",
) -> dict[str, object]:
    return {
        "framework_id": framework,
        "standard_name": "NIST 800-53 v5",
        "section_id": section,
        "section_name": f"name of {section}",
        "cre_id": cre,
        "tier": 2,
        "annotator_id": annotator,
        "created_at": "2026-09-07",
        "confidence": confidence,
        "rationale": "because",
    }


def _round(tmp_path: Path, **by_annotator: list[dict[str, object]]) -> Path:
    d = tmp_path / "round"
    d.mkdir(parents=True, exist_ok=True)
    for annotator, links in by_annotator.items():
        (d / f"{annotator}.jsonl").write_text(
            "".join(json.dumps(x, sort_keys=True) + "\n" for x in links),
            encoding="utf-8",
        )
        (d / f"{annotator}.reviewed.json").write_text(
            json.dumps({
                "annotator_id": annotator,
                "n_reviewed": 300,
                "no_hub_controls": [],
                "source_sha256": "0" * 64,
            }),
            encoding="utf-8",
        )
    return d


class TestTheConfidenceFloorReachesTraining:
    """Q3 is a counting floor in the gate. Here it has to be a training floor."""

    def test_low_confidence_links_are_dropped(self, tmp_path: Path) -> None:
        d = _round(tmp_path, **{"vol-01": [
            _link("AC-1", "010-108", confidence=3),
            _link("AC-2", "010-109", confidence=2),
            _link("AC-3", "010-110", confidence=1),
        ]})
        links, stats = merge_round(d, min_confidence=2)
        assert [b.section_id for b in links] == ["AC-1", "AC-2"]
        assert stats.n_below_floor == 1
        assert stats.n_kept == 2

    def test_the_floor_is_reported_not_silently_applied(
        self, tmp_path: Path
    ) -> None:
        """A corpus that shrank without saying so is how 86 gets called 80."""
        d = _round(tmp_path, **{"vol-01": [
            _link("AC-1", "010-108", confidence=1),
        ]})
        _, stats = merge_round(d, min_confidence=2)
        assert stats.n_raw == 1
        assert stats.n_kept == 0
        assert stats.min_confidence == 2


class TestDeterminism:
    """Two runs, two machines, byte-identical output -- or the digest is noise."""

    def test_output_is_byte_identical_across_runs(self, tmp_path: Path) -> None:
        d = _round(
            tmp_path,
            **{
                "vol-02": [_link("SI-4", "020-200", annotator="vol-02"),
                           _link("AC-1", "010-108", annotator="vol-02")],
                "vol-01": [_link("SR-3", "030-300"), _link("AC-1", "010-108")],
            },
        )
        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        links, _ = merge_round(d, min_confidence=2)
        write_corpus(links, a)
        links2, _ = merge_round(d, min_confidence=2)
        write_corpus(links2, b)
        assert a.read_bytes() == b.read_bytes()

    def test_order_does_not_depend_on_filename_iteration(
        self, tmp_path: Path
    ) -> None:
        """Sorted by content, not by whatever order the glob returned."""
        d = _round(
            tmp_path,
            **{"vol-02": [_link("AC-9", "010-999", annotator="vol-02")],
               "vol-01": [_link("AC-1", "010-108")]},
        )
        links, _ = merge_round(d, min_confidence=2)
        assert [b.section_id for b in links] == ["AC-1", "AC-9"]


class TestTheDedupRuleIsDeclared:
    def test_identical_edges_from_two_annotators_collapse_to_one(
        self, tmp_path: Path
    ) -> None:
        """Downstream `build_training_pairs` collapses (text, hub) anyway.

        Doing it here too means the corpus count and the trained count agree,
        which is what makes `n_kept` quotable.
        """
        d = _round(
            tmp_path,
            **{"vol-01": [_link("AC-1", "010-108", confidence=2)],
               "vol-02": [_link("AC-1", "010-108", annotator="vol-02",
                                confidence=3)]},
        )
        links, stats = merge_round(d, min_confidence=2)
        assert len(links) == 1
        assert stats.n_agreed_edges == 1

    def test_the_surviving_record_is_the_higher_confidence_one(
        self, tmp_path: Path
    ) -> None:
        d = _round(
            tmp_path,
            **{"vol-01": [_link("AC-1", "010-108", confidence=2)],
               "vol-02": [_link("AC-1", "010-108", annotator="vol-02",
                                confidence=3)]},
        )
        links, _ = merge_round(d, min_confidence=2)
        assert links[0].confidence == 3
        assert links[0].annotator_id == "vol-02"

    def test_same_control_different_hubs_are_both_kept_and_counted(
        self, tmp_path: Path
    ) -> None:
        """A control legitimately maps to several hubs -- Q2 permits up to six.

        These are not duplicates and collapsing them would discard an
        annotator's judgement. They ARE a disagreement, so they are counted.
        """
        d = _round(
            tmp_path,
            **{"vol-01": [_link("SI-6", "010-108")],
               "vol-02": [_link("SI-6", "020-222", annotator="vol-02")]},
        )
        links, stats = merge_round(d, min_confidence=2)
        assert len(links) == 2
        assert stats.n_conflicting_controls == 1


class TestItRefusesToEatGold:
    def test_a_directory_holding_gold_links_is_refused(
        self, tmp_path: Path
    ) -> None:
        """gate1_report learned this the hard way; so does the merger.

        `data/training/` holds hub_links_curated.jsonl. Globbing *.jsonl there
        loads 4,405 Tier-1 gold links as though volunteers had written them.
        """
        d = tmp_path / "training"
        d.mkdir()
        (d / "hub_links_curated.jsonl").write_text("{}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="gold"):
            merge_round(d, min_confidence=2)

    def test_an_empty_directory_raises_rather_than_returning_nothing(
        self, tmp_path: Path
    ) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(ValueError, match="no .*jsonl"):
            merge_round(d, min_confidence=2)


class TestTheOutputLoadsBackAsBridgeLinks:
    def test_round_trips_through_load_bridge_links(self, tmp_path: Path) -> None:
        """The merged file is what --bridge-links is pointed at."""
        d = _round(tmp_path, **{"vol-01": [_link("AC-1", "010-108")]})
        out = tmp_path / "merged.jsonl"
        links, _ = merge_round(d, min_confidence=2)
        write_corpus(links, out)
        reloaded = load_bridge_links(out)
        assert len(reloaded) == 1
        assert reloaded[0].cre_id == "010-108"
        assert reloaded[0].tier == 2


class TestTheManifestIsTheDigestsPublicReferent:
    def test_manifest_carries_counts_and_digests(self, tmp_path: Path) -> None:
        d = _round(
            tmp_path,
            **{"vol-01": [_link("AC-1", "010-108"),
                          _link("AC-2", "010-109", confidence=1)]},
        )
        out = tmp_path / "merged.jsonl"
        links, stats = merge_round(d, min_confidence=2)
        write_corpus(links, out)
        manifest = build_manifest(
            round_label="r2", round_dir=d, corpus_path=out, stats=stats
        )
        assert manifest["round_label"] == "r2"
        assert manifest["n_kept"] == 1
        assert manifest["n_below_floor"] == 1
        assert len(manifest["corpus_sha256"]) == 64
        assert manifest["source_files"]["vol-01.jsonl"]
        assert "git_sha" in manifest

    def test_manifest_carries_no_annotator_free_text(
        self, tmp_path: Path
    ) -> None:
        """The manifest is committed; the corpus is not.

        Counts and digests are the scientific record. A volunteer's verbatim
        rationale is published by a deliberate act, not by a manifest.
        """
        d = _round(tmp_path, **{"vol-01": [_link("AC-1", "010-108")]})
        out = tmp_path / "merged.jsonl"
        links, stats = merge_round(d, min_confidence=2)
        write_corpus(links, out)
        blob = json.dumps(
            build_manifest(round_label="r2", round_dir=d,
                           corpus_path=out, stats=stats)
        )
        assert "because" not in blob
        assert "rationale" not in blob


class TestAgainstTheRealCorpus:
    """The numbers Amendment 1 commits to, re-derived from what is on disk."""

    @pytest.mark.parametrize(
        ("label", "directory", "n_kept", "n_raw"),
        [("r1", "data/training/bridge", 57, 86),
         ("r2", "data/training/bridge-r2", 124, 192)],
    )
    def test_real_round_counts(
        self, label: str, directory: str, n_kept: int, n_raw: int
    ) -> None:
        d = Path(directory)
        if not d.is_dir():
            pytest.skip(f"{directory} is gitignored and absent on this machine")
        links, stats = merge_round(d, min_confidence=2)
        assert stats.n_raw == n_raw
        assert len(links) == n_kept, (
            "distinct (section, hub) edges at confidence >= 2. The plan's "
            "'80 links' and '173 links' are ROW counts before dedup."
        )


def test_stats_is_a_dataclass_not_a_dict() -> None:
    """Domain objects are typed here; CLAUDE.md forbids bare dicts for them."""
    assert CorpusStats.__dataclass_fields__  # type: ignore[attr-defined]


class TestTheWithdrawalMechanism:
    """The handbook promised a right to withdraw before publication.

    The annotators are anonymous by their own request, so there is no channel
    to solicit an individual acknowledgement -- asking would require the named
    person anonymity removes. The promise is therefore kept by being
    MECHANICALLY POSSIBLE: one contributor comes out of the derived corpus,
    the source record stays intact, and Gate 1 re-runs against the remainder.
    """

    def test_an_excluded_annotators_links_are_gone(self, tmp_path: Path) -> None:
        d = _round(
            tmp_path,
            **{"vol-01": [_link("AC-1", "010-108")],
               "vol-02": [_link("SI-4", "020-200", annotator="vol-02")]},
        )
        links, stats = merge_round(
            d, min_confidence=2, exclude_annotators=frozenset({"vol-01"})
        )
        assert [b.section_id for b in links] == ["SI-4"]
        assert stats.withdrawn == {"vol-01": 1}
        assert "vol-01" not in stats.per_annotator_raw

    def test_the_source_file_is_not_destroyed(self, tmp_path: Path) -> None:
        """Withdrawal removes a contribution from the corpus, not the record.

        Deleting the file would work and would also erase what was submitted,
        which the Gate 1 report already depends on.
        """
        d = _round(
            tmp_path,
            **{"vol-01": [_link("AC-1", "010-108")],
               "vol-02": [_link("SI-4", "020-200", annotator="vol-02")]},
        )
        merge_round(d, min_confidence=2,
                    exclude_annotators=frozenset({"vol-01"}))
        assert (d / "vol-01.jsonl").is_file()

    def test_excluding_everyone_raises_rather_than_shipping_a_null(
        self, tmp_path: Path
    ) -> None:
        d = _round(tmp_path, **{"vol-01": [_link("AC-1", "010-108")]})
        with pytest.raises(ValueError, match="no links at all"):
            merge_round(d, min_confidence=2,
                        exclude_annotators=frozenset({"vol-01"}))

    def test_an_unknown_pseudonym_raises(self, tmp_path: Path) -> None:
        """A withdrawal that silently excluded nobody is worse than an error."""
        d = _round(
            tmp_path,
            **{"vol-01": [_link("AC-1", "010-108")],
               "vol-02": [_link("SI-4", "020-200", annotator="vol-02")]},
        )
        with pytest.raises(ValueError, match="match no file"):
            merge_round(d, min_confidence=2,
                        exclude_annotators=frozenset({"vol-99"}))

    def test_no_exclusion_records_an_empty_withdrawn_map(
        self, tmp_path: Path
    ) -> None:
        """Not None. `0, never None` is the convention FilterReport.n_bridge set."""
        d = _round(tmp_path, **{"vol-01": [_link("AC-1", "010-108")]})
        _, stats = merge_round(d, min_confidence=2)
        assert stats.withdrawn == {}
