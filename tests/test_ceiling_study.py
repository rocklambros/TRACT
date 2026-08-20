"""Tests for the blind expert-agreement ceiling study (design doc Part 0.1).

Three properties matter more than the rest, because getting any of them
wrong quietly invalidates the study rather than raising an error:

1. ceiling_items.json must carry zero ground truth, ever.
2. The sample must be exactly reproducible from its recorded seed.
3. Every prefix of the shuffled 250 must span both strata, so a partial
   completion is still an analyzable sample.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import pytest

from scripts import analyze_panel_agreement, run_panel, score_ceiling_study
from tract import ceiling_study
from tract.ceiling_study import (
    AnchorRecord,
    CeilingItem,
    _link_priority,
    _load_eligible_links,
    apportion_with_caps,
    build_anchor_pool,
    build_answer_key,
    build_answer_template,
    build_ceiling_study,
    ceiling_study_divergence,
    eligible_framework_ids,
    load_ceiling_items,
    load_pinned_study_provenance,
    new_study_dir,
    require_new_study_destination,
    require_pinned_study_unmodified,
    require_unmoved_ceiling_study,
    sample_ceiling_items,
)
from tract.config import (
    CEILING_STUDY_DIR,
    CEILING_STUDY_N_ITEMS,
    CEILING_STUDY_NEW_DIR,
    CEILING_STUDY_PINNED_ITEMS,
    CEILING_STUDY_SEED,
    CONTESTED_RECOVERY_DEFAULT,
    EXIT_USER_ERROR,
    PANEL_MODELS,
    PROJECT_ROOT,
)
from tract.io import atomic_write_json
from tract.text_selection import ProseIndex
from tract.training.data_quality import curated_link_filter_report, link_key

CEILING_ITEMS_PATH = CEILING_STUDY_DIR / "ceiling_items.json"
HUMAN_ANSWERS_PATH = CEILING_STUDY_DIR / "answers_human_rock.json"


def _make_record(framework_id: str, index: int, hub_id: str) -> AnchorRecord:
    return AnchorRecord(
        framework_id=framework_id,
        framework_name=f"{framework_id}-display",
        control_id=f"{framework_id}-{index}",
        control_title=f"Title {index}",
        control_text=f"Control text body number {index} for {framework_id}.",
        text_source="description",
        anchor_key=f"control text body number {index} for {framework_id}.",
        primary_gold_hub_id=hub_id,
        gold_hub_ids=(hub_id,),
    )



# Matches the real corpus's unique-anchor pool sizes (measured 2026-08-15) so
# the synthetic pool exercises the same cap-binding path as production: the
# test stratum's total capacity (133) exceeds its 125-item quota only because
# owasp_ai_exchange has slack, while mitre_atlas, nist_ai_100_2 and
# owasp_llm_top10 are each capped below their proportional share.
_SYNTHETIC_POOL_SIZES: dict[str, int] = {
    "capec": 339, "cwe": 240, "nist_800_53": 298,
    "mitre_atlas": 43, "owasp_ai_exchange": 62,
    "nist_ai_100_2": 22, "owasp_llm_top10": 6,
}


def _synthetic_pool() -> dict[str, list[AnchorRecord]]:
    """A stand-in pool shaped like the real one's anchor counts, for unit tests."""
    return {
        framework_id: [
            _make_record(framework_id, i, f"hub-{framework_id}-{i}")
            for i in range(size)
        ]
        for framework_id, size in _SYNTHETIC_POOL_SIZES.items()
    }


_SYNTHETIC_WEIGHTS: dict[str, int] = {
    "capec": 1755, "cwe": 596, "nist_800_53": 298,
    "mitre_atlas": 65, "owasp_ai_exchange": 62, "nist_ai_100_2": 45,
    "owasp_llm_top10": 13,
}


class TestApportionWithCaps:
    def test_uncapped_largest_remainder(self) -> None:
        result = apportion_with_caps(
            {"a": 1755, "b": 596, "c": 298}, 125,
            {"a": 339, "b": 240, "c": 298},
        )
        assert result == {"a": 83, "b": 28, "c": 14}
        assert sum(result.values()) == 125

    def test_matches_measured_test_stratum_allocation(self) -> None:
        """Reproduces the real corpus's cap-binding case by hand-checked values."""
        weights = {
            "mitre_atlas": 65, "owasp_ai_exchange": 62,
            "nist_ai_100_2": 45, "owasp_llm_top10": 13,
        }
        caps = {
            "mitre_atlas": 43, "owasp_ai_exchange": 62,
            "nist_ai_100_2": 22, "owasp_llm_top10": 6,
        }
        result = apportion_with_caps(weights, 125, caps)
        assert result == {
            "mitre_atlas": 43, "owasp_ai_exchange": 54,
            "nist_ai_100_2": 22, "owasp_llm_top10": 6,
        }

    def test_never_exceeds_cap(self) -> None:
        result = apportion_with_caps(
            {"a": 100, "b": 1}, 10, {"a": 3, "b": 100},
        )
        assert result["a"] == 3
        assert sum(result.values()) == 10

    def test_sums_to_total_across_many_ratios(self) -> None:
        for total in (0, 1, 7, 50, 125, 250):
            weights = {"a": 11, "b": 7, "c": 3, "d": 1}
            caps = {"a": 200, "b": 200, "c": 200, "d": 200}
            result = apportion_with_caps(weights, total, caps)
            assert sum(result.values()) == total
            for key, cap in caps.items():
                assert 0 <= result[key] <= cap

    def test_infeasible_total_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds combined capacity"):
            apportion_with_caps({"a": 1, "b": 1}, 10, {"a": 2, "b": 2})

    def test_missing_cap_raises(self) -> None:
        with pytest.raises(ValueError, match="no cap"):
            apportion_with_caps({"a": 1, "b": 1}, 1, {"a": 5})

    def test_zero_total_returns_zeros(self) -> None:
        result = apportion_with_caps({"a": 5, "b": 5}, 0, {"a": 10, "b": 10})
        assert result == {"a": 0, "b": 0}


class TestSamplerDeterminism:
    """The pool and weights are pure Python data. Same seed, same output."""

    def test_same_seed_same_items(self) -> None:
        pool = _synthetic_pool()
        items_a, summary_a = sample_ceiling_items(pool, _SYNTHETIC_WEIGHTS, seed=42)
        items_b, summary_b = sample_ceiling_items(pool, _SYNTHETIC_WEIGHTS, seed=42)
        assert items_a == items_b
        assert summary_a == summary_b

    def test_different_seed_different_order(self) -> None:
        pool = _synthetic_pool()
        items_a, _ = sample_ceiling_items(pool, _SYNTHETIC_WEIGHTS, seed=42)
        items_b, _ = sample_ceiling_items(pool, _SYNTHETIC_WEIGHTS, seed=43)
        control_ids_a = [i["control_id"] for i in items_a]
        control_ids_b = [i["control_id"] for i in items_b]
        assert control_ids_a != control_ids_b

    def test_totals_and_no_duplicate_anchors(self) -> None:
        pool = _synthetic_pool()
        items, summary = sample_ceiling_items(pool, _SYNTHETIC_WEIGHTS, seed=42)
        assert len(items) == 250
        assert summary["n_items"] == 250
        assert sum(summary["validation_allocation"].values()) == 125
        assert sum(summary["test_allocation"].values()) == 125
        keys = [(i["framework_id"], i["control_id"]) for i in items]
        assert len(keys) == len(set(keys))

    def test_item_index_is_1_based_contiguous(self) -> None:
        pool = _synthetic_pool()
        items, _ = sample_ceiling_items(pool, _SYNTHETIC_WEIGHTS, seed=42)
        assert [i["item_index"] for i in items] == list(range(1, 251))


class TestPrefixInvariant:
    """Every prefix of length >= 2 in the shuffled order spans both strata.

    This is the property that makes stopping partway through review still
    yield an analyzable sample, which is the entire reason the study is
    shuffled rather than emitted stratum-by-stratum.
    """

    def test_every_prefix_of_length_2_or_more_has_both_strata(self) -> None:
        pool = _synthetic_pool()
        items, _ = sample_ceiling_items(pool, _SYNTHETIC_WEIGHTS, seed=42)
        strata = [i["stratum"] for i in items]
        for k in range(2, len(strata) + 1):
            prefix_strata = set(strata[:k])
            assert prefix_strata == {"validation", "test"}, (
                f"prefix of length {k} has strata {prefix_strata}"
            )

    def test_holds_across_several_seeds(self) -> None:
        pool = _synthetic_pool()
        for seed in (1, 2, 42, 999, 123456):
            items, _ = sample_ceiling_items(pool, _SYNTHETIC_WEIGHTS, seed=seed)
            strata = [i["stratum"] for i in items]
            for k in range(2, len(strata) + 1):
                assert set(strata[:k]) == {"validation", "test"}


class TestAnswerKeyAndTemplate:
    def test_answer_key_matches_items_by_index(self) -> None:
        pool = _synthetic_pool()
        items, _ = sample_ceiling_items(pool, _SYNTHETIC_WEIGHTS, seed=42)
        key = build_answer_key(items, pool)
        assert [e["item_index"] for e in key] == [i["item_index"] for i in items]
        for entry in key:
            assert entry["primary_gold_hub_id"] in entry["valid_gold_hub_ids"]

    def test_template_starts_blank(self) -> None:
        pool = _synthetic_pool()
        items, _ = sample_ceiling_items(pool, _SYNTHETIC_WEIGHTS, seed=42)
        template = build_answer_template(items)
        assert len(template) == 250
        for row in template:
            assert row["primary_hub_id"] == ""
            assert row["acceptable_hub_ids"] == []
            assert row["confidence"] == ""
            assert row["notes"] == ""


class TestNoGroundTruthLeak:
    """The load-bearing safety property: ceiling_items.json is blind."""

    def test_no_gold_hub_id_in_serialized_items(self) -> None:
        pool = _synthetic_pool()
        items, _ = sample_ceiling_items(pool, _SYNTHETIC_WEIGHTS, seed=42)
        key = build_answer_key(items, pool)

        items_blob = json.dumps(items)
        gold_ids: set[str] = set()
        for entry in key:
            gold_ids.add(entry["primary_gold_hub_id"])
            gold_ids.update(entry["valid_gold_hub_ids"])

        leaked = [gold_id for gold_id in gold_ids if gold_id in items_blob]
        assert leaked == [], f"gold hub id(s) leaked into items export: {leaked}"

    def test_items_have_no_hub_shaped_keys(self) -> None:
        pool = _synthetic_pool()
        items, _ = sample_ceiling_items(pool, _SYNTHETIC_WEIGHTS, seed=42)
        forbidden_keys = {
            "gold_hub_id", "gold_hub_ids", "hub_id", "hub_ids",
            "ground_truth", "answer", "predicted_hub_id", "candidates",
        }
        for item in items:
            assert forbidden_keys.isdisjoint(item.keys())


@pytest.mark.skipif(
    not Path("data/training/hub_links_curated.jsonl").exists(),
    reason="requires the committed corpus (data/training, data/processed)",
)
class TestBuildCeilingStudyIntegration:
    """End-to-end against the real corpus. Slow-ish, skipped if data is absent."""

    def test_produces_250_items_matching_measured_allocation(self) -> None:
        items, key, template, hub_reference_md, summary = build_ceiling_study()
        assert len(items) == 250
        assert len(key) == 250
        assert len(template) == 250
        assert summary["validation_allocation"] == {"capec": 83, "cwe": 28, "nist_800_53": 14}
        assert summary["test_allocation"] == {
            "mitre_atlas": 43, "owasp_ai_exchange": 54,
            "nist_ai_100_2": 22, "owasp_llm_top10": 6,
        }
        assert "# CRE Hub Reference" in hub_reference_md
        assert hub_reference_md.count("### ") == 522

    def test_deterministic_end_to_end(self) -> None:
        items_a, key_a, _, _, _ = build_ceiling_study()
        items_b, key_b, _, _, _ = build_ceiling_study()
        assert items_a == items_b
        assert key_a == key_b

    def test_no_leak_against_real_corpus(self) -> None:
        items, key, _, _, _ = build_ceiling_study()
        items_blob = json.dumps(items)
        gold_ids: set[str] = set()
        for entry in key:
            gold_ids.add(entry["primary_gold_hub_id"])
            gold_ids.update(entry["valid_gold_hub_ids"])
        leaked = [gold_id for gold_id in gold_ids if gold_id in items_blob]
        assert leaked == []


@pytest.mark.skipif(
    not Path("data/training/hub_links_curated.jsonl").exists(),
    reason="requires the committed corpus (data/training, data/processed)",
)
class TestTheStudyPoolIsTheTrainingPool:
    """The mirror the docstring promises, asserted rather than described.

    _load_eligible_links used to inline its own assign_quality_tier call under
    a docstring claiming it mirrored training. Nothing checked the claim, so
    the day the two gates differed the study would have sampled from a pool
    training never used and no test would have moved.
    """

    def test_the_two_pools_hold_the_same_links(self) -> None:
        eligible = eligible_framework_ids()
        report, _ = curated_link_filter_report()
        training = {
            link_key(tiered.link) for tiered in report.kept
            if tiered.link.get("framework_id") in eligible
        }
        study = {link_key(link) for link in _load_eligible_links(eligible)}
        assert study == training

    def test_link_priority_cannot_be_called_without_the_index(self) -> None:
        """A defaulted index would let the priority call stop mirroring too.

        _link_priority ranks the link that represents a multi-link anchor, and
        it ranks by quality tier. The tier now depends on the resolved anchor,
        so an index-free call would rank on a stale contract while the pool
        around it moved.
        """
        parameter = inspect.signature(_link_priority).parameters["prose_index"]
        assert parameter.default is inspect.Parameter.empty

    def test_no_anchor_in_the_pool_is_a_section_title(self) -> None:
        """The gate admits only resolved links, so the pool is prose throughout.

        build_anchor_pool calls select_control_text, which falls back to the
        title when the index misses. A link that reached the pool unresolved
        put a bare section title in front of a human reviewer who scored it as
        a control statement.
        """
        index = ProseIndex.load()
        pool = build_anchor_pool(_load_eligible_links(eligible_framework_ids()), index)
        titles = [
            record.anchor_key for records in pool.values() for record in records
            if record.text_source == "title"
        ]
        assert titles == []

    def test_the_sampling_frame_is_the_size_this_task_left_it(self) -> None:
        """The frame the scored 250 were drawn from is not the frame today.

        MEASURED. owasp_ai_exchange goes 62 -> 63 on the anchor gate alone,
        and capec 339 -> 349 with cwe 240 -> 245 only when the contested
        recovery is on, so the two commits are pinned apart here rather than
        averaged. None of the seven eligible frameworks is licensed, so these
        hold with the overlay and without it.

        Pinned because sample_ceiling_items draws with rng.sample over the
        pool, so the frame size alone changes which items are drawn at a fixed
        seed. That is how items were replaced under the scored study without a
        single existing test moving.
        """
        pool = build_anchor_pool(_load_eligible_links(eligible_framework_ids()),
                                 ProseIndex.load())
        assert {fw: len(records) for fw, records in pool.items()} == {
            "capec": 349 if CONTESTED_RECOVERY_DEFAULT else 339,
            "cwe": 245 if CONTESTED_RECOVERY_DEFAULT else 240,
            "nist_800_53": 298,
            "mitre_atlas": 43, "nist_ai_100_2": 22,
            "owasp_ai_exchange": 63, "owasp_llm_top10": 6,
        }

    def test_every_scored_study_item_is_still_an_anchor_in_the_pool(self) -> None:
        """The human's 250 answers stay joinable to the pool they scored.

        The frame moved, so build_ceiling_study() no longer redraws the
        committed sample. That is recorded in the module docstring and in the
        run ledger rather than hidden. What must not also break is the join:
        if a scored anchor stopped existing, alpha-1 and alpha-5 would be
        measured against text no longer in the corpus.
        """
        items = json.loads(
            CEILING_ITEMS_PATH.read_text(encoding="utf-8")
        )
        records = items["items"] if isinstance(items, dict) else items
        pool = build_anchor_pool(_load_eligible_links(eligible_framework_ids()),
                                 ProseIndex.load())
        present = {
            (framework_id, record.anchor_key)
            for framework_id, group in pool.items() for record in group
        }
        missing = [
            (item["framework_id"], item["control_id"]) for item in records
            if (item["framework_id"], item["control_text"].lower().strip())
            not in present
        ]
        assert missing == [], (
            "these scored ceiling-study items no longer exist as anchors, so "
            f"the measured alpha-1 cannot be reproduced against them: {missing}"
        )


# ── Ruling R22: the annotated study is an artifact, not a redraw ──────────


def _item(index: int, framework_id: str = "capec", text: str | None = None,
          control_id: str | None = None) -> CeilingItem:
    """One well-formed item row, for building study files in tmp_path."""
    return {
        "item_index": index,
        "framework_id": framework_id,
        "framework_name": "CAPEC",
        "control_id": control_id if control_id is not None else f"CAPEC-{index}",
        "control_title": f"Title {index}",
        "control_text": text if text is not None else f"Body of control {index}.",
        "text_source": "description",
        "stratum": "validation",
    }


def _write_items(path: Path, items: list[CeilingItem]) -> None:
    atomic_write_json({"seed": 42, "n_items": len(items), "items": items}, path)


def _write_provenance(path: Path, items_path: Path, **overrides: object) -> None:
    """A minimal valid provenance record pointing at items_path."""
    drawn: dict[str, object] = {
        "recovery": "reproduced",
        "seed": 42,
        "corpus_sha256": "a" * 64,
        "curated_links_sha256": "b" * 64,
    }
    drawn.update(overrides)
    atomic_write_json(
        {
            "pinned_artifact": {
                "path": str(items_path),
                "sha256": hashlib.sha256(items_path.read_bytes()).hexdigest(),
                "n_items": 3,
            },
            "drawn_from": drawn,
        },
        path,
    )


class TestCeilingItemsLoader:
    """load_ceiling_items is how a consumer reaches the study of record."""

    def test_loads_the_pinned_items_in_file_order(self) -> None:
        loaded = load_ceiling_items()
        raw = json.loads(CEILING_ITEMS_PATH.read_text(encoding="utf-8"))["items"]
        assert len(loaded) == CEILING_STUDY_N_ITEMS
        assert [item["item_index"] for item in loaded] == list(range(1, 251))
        assert [item["control_id"] for item in loaded] == [
            row["control_id"] for row in raw
        ]

    def test_a_gap_in_item_index_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "items.json"
        _write_items(path, [_item(1), _item(2), _item(4)])
        with pytest.raises(ValueError, match="contiguous"):
            load_ceiling_items(path)

    def test_a_repeated_item_index_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "items.json"
        _write_items(path, [_item(1), _item(2), _item(2)])
        with pytest.raises(ValueError, match="contiguous"):
            load_ceiling_items(path)

    def test_a_reordered_file_raises(self, tmp_path: Path) -> None:
        """Order is the study's design: every prefix must span both strata."""
        path = tmp_path / "items.json"
        _write_items(path, [_item(2), _item(1), _item(3)])
        with pytest.raises(ValueError, match="contiguous"):
            load_ceiling_items(path)

    def test_a_non_string_control_text_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "items.json"
        broken = dict(_item(1))
        broken["control_text"] = 42  # type: ignore[typeddict-item]
        atomic_write_json({"items": [broken]}, path)
        with pytest.raises(ValueError, match="control_text=42"):
            load_ceiling_items(path)

    def test_a_boolean_item_index_raises(self, tmp_path: Path) -> None:
        """bool is an int in Python, so the check has to say so explicitly."""
        path = tmp_path / "items.json"
        broken = dict(_item(1))
        broken["item_index"] = True  # type: ignore[typeddict-item]
        atomic_write_json({"items": [broken]}, path)
        with pytest.raises(ValueError, match="not an integer"):
            load_ceiling_items(path)

    def test_an_empty_items_list_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "items.json"
        atomic_write_json({"items": []}, path)
        with pytest.raises(ValueError, match="carries no ceiling-study items"):
            load_ceiling_items(path)

    def test_a_file_that_is_not_the_pinned_one_skips_the_digest_check(
        self, tmp_path: Path,
    ) -> None:
        """Other files share this schema and are not an expert's afternoon.

        The contamination control and any new study under studies/ must load
        without a provenance record, or the tripwire would block every path
        that is not the annotated study.
        """
        path = tmp_path / "items.json"
        _write_items(path, [_item(1), _item(2), _item(3)])
        assert [item["item_index"] for item in load_ceiling_items(path)] == [1, 2, 3]


class TestPinnedStudyTripwire:
    """require_pinned_study_unmodified is the guard on 'do not edit evidence'."""

    def test_a_matching_digest_passes(self, tmp_path: Path) -> None:
        items_path = tmp_path / "items.json"
        provenance = tmp_path / "provenance.json"
        _write_items(items_path, [_item(1), _item(2), _item(3)])
        _write_provenance(provenance, items_path)
        require_pinned_study_unmodified(items_path, provenance)

    def test_an_edited_items_file_raises_and_names_both_digests(
        self, tmp_path: Path,
    ) -> None:
        items_path = tmp_path / "items.json"
        provenance = tmp_path / "provenance.json"
        _write_items(items_path, [_item(1), _item(2), _item(3)])
        _write_provenance(provenance, items_path)
        recorded = hashlib.sha256(items_path.read_bytes()).hexdigest()

        _write_items(items_path, [_item(1), _item(2), _item(3, text="edited")])
        edited = hashlib.sha256(items_path.read_bytes()).hexdigest()
        assert edited != recorded

        with pytest.raises(ValueError) as caught:
            require_pinned_study_unmodified(items_path, provenance)
        assert recorded in str(caught.value)
        assert edited in str(caught.value)

    def test_the_tracked_study_still_matches_its_recorded_digest(self) -> None:
        """The live tripwire, over the two tracked files it ties together.

        Not a tautology: the digest lives in ceiling_study_provenance.json and
        the bytes live in ceiling_items.json, and nothing keeps them in step
        except this assertion. It fails the day either file is rewritten,
        which is the day 250 hand-made answers stop meaning what they say.
        """
        require_pinned_study_unmodified()

    def test_a_read_of_the_pinned_path_runs_the_tripwire(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """load_ceiling_items must not be a plain json.load in disguise.

        Moves the module's idea of both the pinned path and the provenance
        path onto tmp_path, records a digest that does not match, and asserts
        the read fails. Without the dispatch this file parses cleanly and the
        call returns three items.
        """
        items_path = tmp_path / "ceiling_items.json"
        provenance = tmp_path / "provenance.json"
        _write_items(items_path, [_item(1), _item(2), _item(3)])
        _write_provenance(provenance, items_path)
        _write_items(items_path, [_item(1), _item(2), _item(3, text="edited")])

        monkeypatch.setattr(
            ceiling_study, "CEILING_STUDY_PINNED_ITEMS", items_path,
        )
        monkeypatch.setattr(
            ceiling_study, "CEILING_STUDY_PROVENANCE_PATH", provenance,
        )
        with pytest.raises(ValueError, match="no longer matches the study of record"):
            load_ceiling_items(items_path)

    def test_a_read_of_another_path_does_not_run_the_tripwire(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The other direction, so the dispatch is not 'always check'."""
        items_path = tmp_path / "ceiling_items.json"
        provenance = tmp_path / "provenance.json"
        other = tmp_path / "some_other_study.json"
        _write_items(items_path, [_item(1)])
        _write_provenance(provenance, items_path)
        _write_items(other, [_item(1), _item(2)])

        monkeypatch.setattr(
            ceiling_study, "CEILING_STUDY_PINNED_ITEMS", items_path,
        )
        monkeypatch.setattr(
            ceiling_study, "CEILING_STUDY_PROVENANCE_PATH", provenance,
        )
        assert len(load_ceiling_items(other)) == 2


class TestConsumersReadTheStudyOfRecord:
    """Ruling R22 item 1: a consumer reads the artifact, never a fresh draw."""

    def test_the_scorer_validates_the_items_file_it_reads(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "items.json"
        _write_items(path, [_item(1), _item(3)])
        with pytest.raises(ValueError, match="contiguous"):
            score_ceiling_study._load_items_metadata(path)

    def test_the_panel_analysis_validates_the_items_file_it_reads(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "items.json"
        _write_items(path, [_item(1), _item(3)])
        with pytest.raises(ValueError, match="contiguous"):
            analyze_panel_agreement._load_items_metadata(path)

    def test_the_scorer_reads_the_pinned_study_by_default(self) -> None:
        metadata = score_ceiling_study._load_items_metadata(
            CEILING_STUDY_PINNED_ITEMS
        )
        assert sorted(metadata) == list(range(1, CEILING_STUDY_N_ITEMS + 1))

    def test_the_panel_run_refuses_a_pinned_study_that_moved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The one path that spends money against the study checks it first."""
        items_path = tmp_path / "ceiling_items.json"
        provenance = tmp_path / "provenance.json"
        _write_items(items_path, [_item(1)])
        _write_provenance(provenance, items_path)
        _write_items(items_path, [_item(1, text="edited")])

        monkeypatch.setattr(
            ceiling_study, "CEILING_STUDY_PROVENANCE_PATH", provenance,
        )
        monkeypatch.setattr(run_panel, "CEILING_STUDY_PINNED_ITEMS", items_path)
        monkeypatch.setattr(
            sys, "argv",
            ["run_panel", "--model", next(iter(PANEL_MODELS)),
             "--items", str(items_path)],
        )
        assert run_panel.main() == EXIT_USER_ERROR


class TestPinnedStudyProvenance:
    """The record of what the annotated study was drawn from."""

    def test_the_recorded_digest_and_count_match_the_pinned_file(self) -> None:
        record = load_pinned_study_provenance()
        artifact = record["pinned_artifact"]
        items = json.loads(CEILING_ITEMS_PATH.read_text(encoding="utf-8"))
        assert artifact["n_items"] == len(items["items"])
        assert artifact["sha256"] == hashlib.sha256(
            CEILING_ITEMS_PATH.read_bytes()
        ).hexdigest()

    def test_the_recorded_path_is_the_pinned_artifact(self) -> None:
        record = load_pinned_study_provenance()
        recorded = PROJECT_ROOT / str(record["pinned_artifact"]["path"])
        assert recorded.resolve() == CEILING_STUDY_PINNED_ITEMS.resolve()

    def test_the_recorded_seed_is_the_seed_the_artifact_carries(self) -> None:
        """Two files, two independent statements of the seed. They must agree."""
        record = load_pinned_study_provenance()
        items = json.loads(CEILING_ITEMS_PATH.read_text(encoding="utf-8"))
        assert record["drawn_from"]["seed"] == items["seed"] == CEILING_STUDY_SEED

    def test_the_measured_divergence_is_recorded_with_the_inputs_it_used(
        self,
    ) -> None:
        """A count without its corpus goes stale silently. Both, or neither."""
        divergence = load_pinned_study_provenance()["divergence_when_pinned"]
        assert divergence["positions_replaced"] == 168
        assert divergence["positions_held"] == 82
        assert divergence["pinned_items_absent"] == 77
        assert (
            divergence["positions_held"] + divergence["positions_replaced"]
            == CEILING_STUDY_N_ITEMS
        )
        assert len(str(divergence["corpus_sha256"])) == 64
        assert len(str(divergence["curated_links_sha256"])) == 64

    def test_an_unknown_recovery_state_raises(self, tmp_path: Path) -> None:
        items_path = tmp_path / "items.json"
        provenance = tmp_path / "provenance.json"
        _write_items(items_path, [_item(1)])
        _write_provenance(provenance, items_path, recovery="probably_fine")
        with pytest.raises(ValueError, match="drawn_from.recovery"):
            load_pinned_study_provenance(provenance)

    def test_unrecoverable_may_not_sit_beside_a_reconstructed_value(
        self, tmp_path: Path,
    ) -> None:
        """A provenance that is a guess is worse than an absent one."""
        items_path = tmp_path / "items.json"
        provenance = tmp_path / "provenance.json"
        _write_items(items_path, [_item(1)])
        _write_provenance(provenance, items_path, recovery="unrecoverable")
        with pytest.raises(ValueError, match="worse than an absent one"):
            load_pinned_study_provenance(provenance)

    def test_unrecoverable_on_its_own_is_accepted(self, tmp_path: Path) -> None:
        """The other direction: admitting the inputs are gone is allowed."""
        items_path = tmp_path / "items.json"
        provenance = tmp_path / "provenance.json"
        _write_items(items_path, [_item(1)])
        atomic_write_json(
            {
                "pinned_artifact": {
                    "path": str(items_path),
                    "sha256": hashlib.sha256(items_path.read_bytes()).hexdigest(),
                    "n_items": 1,
                },
                "drawn_from": {"recovery": "unrecoverable"},
            },
            provenance,
        )
        record = load_pinned_study_provenance(provenance)
        assert record["drawn_from"]["recovery"] == "unrecoverable"

    def test_a_missing_sha256_raises(self, tmp_path: Path) -> None:
        provenance = tmp_path / "provenance.json"
        atomic_write_json(
            {"pinned_artifact": {"path": "x", "n_items": 1},
             "drawn_from": {"recovery": "reproduced"}},
            provenance,
        )
        with pytest.raises(ValueError, match="pinned_artifact.sha256"):
            load_pinned_study_provenance(provenance)

    def test_a_non_integer_n_items_raises(self, tmp_path: Path) -> None:
        provenance = tmp_path / "provenance.json"
        atomic_write_json(
            {"pinned_artifact": {"path": "x", "sha256": "y", "n_items": "250"},
             "drawn_from": {"recovery": "reproduced"}},
            provenance,
        )
        with pytest.raises(ValueError, match="n_items"):
            load_pinned_study_provenance(provenance)


class TestStudyDivergence:
    """Two counts, because the run ledger reported one of each as a pair."""

    def test_an_identical_sample_diverges_by_zero(self) -> None:
        items = [_item(i) for i in range(1, 6)]
        divergence = ceiling_study_divergence(items, items)
        assert divergence.positions_replaced == 0
        assert divergence.positions_held == 5
        assert divergence.pinned_items_absent == 0

    def test_a_disjoint_sample_replaces_every_position(self) -> None:
        pinned = [_item(i, text=f"old {i}") for i in range(1, 6)]
        fresh = [_item(i, text=f"new {i}") for i in range(1, 6)]
        divergence = ceiling_study_divergence(fresh, pinned)
        assert divergence.positions_replaced == 5
        assert divergence.positions_held == 0
        assert divergence.pinned_items_absent == 5

    def test_a_permutation_moves_positions_without_dropping_anything(self) -> None:
        """The distinction the two counts exist to make.

        A reshuffled sample still holds every scored control, so
        pinned_items_absent is 0, while every answer keyed on item_index now
        points at a different control, so positions_replaced is not.
        """
        pinned = [_item(i, text=f"body {i}") for i in range(1, 6)]
        texts = [f"body {i}" for i in (5, 4, 3, 2, 1)]
        fresh = [
            _item(i, text=texts[i - 1], control_id=f"CAPEC-{texts[i - 1][-1]}")
            for i in range(1, 6)
        ]
        divergence = ceiling_study_divergence(fresh, pinned)
        assert divergence.pinned_items_absent == 0
        assert divergence.positions_replaced == 4  # index 3 maps to itself
        assert divergence.positions_held == 1

    def test_a_renamed_control_id_moves_a_position_but_drops_no_anchor(
        self,
    ) -> None:
        """An anchor is its text. A parser renaming a section id drops nothing."""
        pinned = [_item(1, text="same body")]
        fresh = [_item(1, text="same body", control_id="CAPEC-renamed")]
        divergence = ceiling_study_divergence(fresh, pinned)
        assert divergence.positions_replaced == 1
        assert divergence.pinned_items_absent == 0

    def test_describe_names_both_counts(self) -> None:
        pinned = [_item(i, text=f"old {i}") for i in range(1, 4)]
        fresh = [_item(i, text=f"new {i}") for i in range(1, 4)]
        text = ceiling_study_divergence(fresh, pinned).describe()
        assert "3 of 3 item positions" in text
        assert "3 of the pinned controls are absent" in text


class TestRequireUnmovedCeilingStudy:
    """Shaped after require_unmoved_corpus, and both directions matter."""

    def test_an_absent_destination_is_allowed(self, tmp_path: Path) -> None:
        require_unmoved_ceiling_study([_item(1)], tmp_path / "absent.json")

    def test_a_draw_that_reproduces_the_artifact_is_allowed(
        self, tmp_path: Path,
    ) -> None:
        """Byte-identical regeneration is the property being protected."""
        path = tmp_path / "items.json"
        items = [_item(i) for i in range(1, 6)]
        _write_items(path, items)
        require_unmoved_ceiling_study(items, path)

    def test_a_draw_that_moved_is_refused_and_names_the_count(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "items.json"
        _write_items(path, [_item(i) for i in range(1, 6)])
        moved = [_item(i) for i in range(1, 5)] + [_item(5, text="different")]
        with pytest.raises(ValueError, match="1 of 5 item positions"):
            require_unmoved_ceiling_study(moved, path)

    def test_a_shorter_draw_is_refused(self, tmp_path: Path) -> None:
        """Same prefix, fewer items. The dropped tail reads as a moved position."""
        path = tmp_path / "items.json"
        _write_items(path, [_item(i) for i in range(1, 6)])
        with pytest.raises(ValueError, match="no longer reproduces"):
            require_unmoved_ceiling_study([_item(i) for i in range(1, 5)], path)

    def test_a_longer_draw_is_refused_and_says_why(self, tmp_path: Path) -> None:
        """The case only the length clause catches.

        Same prefix, one extra item. No position moved and no anchor was
        dropped, so both divergence counts read zero and a guard that looked
        only at positions_replaced would let a 5-item draw replace a 4-item
        study. Mutation M8 found exactly this: the shorter-draw test above
        passes with the length clause deleted, because a dropped tail already
        shows up as a replaced position.
        """
        path = tmp_path / "items.json"
        pinned = [_item(i) for i in range(1, 5)]
        _write_items(path, pinned)
        fresh = [_item(i) for i in range(1, 6)]

        divergence = ceiling_study_divergence(fresh, pinned)
        assert divergence.positions_replaced == 0
        assert divergence.pinned_items_absent == 0

        with pytest.raises(ValueError, match="holds 5 items against 4 pinned"):
            require_unmoved_ceiling_study(fresh, path)

    def test_unreadable_json_is_refused_rather_than_overwritten(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "items.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="not valid JSON"):
            require_unmoved_ceiling_study([_item(1)], path)

    def test_a_destination_holding_a_malformed_study_is_refused(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "items.json"
        atomic_write_json({"items": []}, path)
        with pytest.raises(ValueError, match="carries no ceiling-study items"):
            require_unmoved_ceiling_study([_item(1)], path)


class TestNewStudyDestination:
    """A fresh draw is a NEW study, and it cannot land on the old one."""

    def test_the_pinned_artifact_is_refused(self) -> None:
        with pytest.raises(ValueError, match="that is the pinned artifact"):
            require_new_study_destination(CEILING_STUDY_PINNED_ITEMS)

    def test_a_sibling_of_the_pinned_artifact_is_refused(self) -> None:
        """The key and six answer files live beside it under names a draw wants."""
        with pytest.raises(ValueError, match="holds the annotated study"):
            require_new_study_destination(CEILING_STUDY_DIR / "ceiling_answer_key.json")

    def test_a_destination_under_the_studies_directory_is_allowed(self) -> None:
        require_new_study_destination(
            CEILING_STUDY_NEW_DIR / "post_rebuild" / "ceiling_items.json"
        )

    def test_a_destination_outside_the_repository_is_allowed(
        self, tmp_path: Path,
    ) -> None:
        require_new_study_destination(tmp_path / "ceiling_items.json")

    def test_a_valid_name_lands_under_the_studies_directory(self) -> None:
        assert new_study_dir("post_rebuild_2026") == (
            CEILING_STUDY_NEW_DIR / "post_rebuild_2026"
        )

    def test_an_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="needs a name"):
            new_study_dir("")

    @pytest.mark.parametrize("name", ["../evil", "a/b", "Post_Rebuild", "a.b", "x y"])
    def test_a_name_outside_the_alphabet_raises(self, name: str) -> None:
        with pytest.raises(ValueError, match="outside the allowed"):
            new_study_dir(name)

    def test_an_over_length_name_raises(self) -> None:
        with pytest.raises(ValueError, match="over the 64 allowed"):
            new_study_dir("a" * 65)

    def test_a_name_at_the_length_limit_is_allowed(self) -> None:
        assert new_study_dir("a" * 64).name == "a" * 64


class TestTheScoredAnswersStillJoin:
    """The 250 hand-made answers and the pinned items are one unit."""

    def test_every_answer_keys_onto_a_pinned_item(self) -> None:
        answers = json.loads(HUMAN_ANSWERS_PATH.read_text(encoding="utf-8"))
        answered = [int(row["item_index"]) for row in answers["items"]]
        pinned = [item["item_index"] for item in load_ceiling_items()]
        assert sorted(answered) == pinned

    def test_the_provenance_names_the_answer_files_it_binds(self) -> None:
        record = load_pinned_study_provenance()
        named = [
            PROJECT_ROOT / str(path)
            for path in record["pinned_artifact"]["answers"]
        ]
        assert named, "the provenance record names no answer file"
        for path in named:
            assert path.exists(), f"provenance names a missing answer file: {path}"


@pytest.mark.skipif(
    not Path("data/training/hub_links_curated.jsonl").exists(),
    reason="requires the committed corpus (data/training, data/processed)",
)
class TestTheGuardTracksTheLivePool:
    """The pool keeps moving, so the guard is asserted as a biconditional.

    Pinning today's divergence here would break the day Task 15 lands and
    would say nothing about whether the guard works. What must hold at every
    pool size is that the refusal fires exactly when the draw moved.
    """

    def test_the_guard_fires_exactly_when_a_fresh_draw_moved(self) -> None:
        fresh, _, _, _, _ = build_ceiling_study()
        pinned = load_ceiling_items()
        divergence = ceiling_study_divergence(fresh, pinned)
        assert 0 <= divergence.positions_replaced <= CEILING_STUDY_N_ITEMS
        if divergence.positions_replaced == 0 and divergence.n_fresh == len(pinned):
            require_unmoved_ceiling_study(fresh, CEILING_ITEMS_PATH)
        else:
            with pytest.raises(ValueError, match="no longer reproduces"):
                require_unmoved_ceiling_study(fresh, CEILING_ITEMS_PATH)
