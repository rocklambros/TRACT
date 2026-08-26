"""A corpus rebuild must be reversible and must diff the field the model reads.

Three things the previous version could not do. It hashed `description` while
ProseIndex prefers `full_text` unconditionally, so it could re-point every link
and report 0 changed. It stored one digest per key while nine keys hold 39 extra
records with distinct text. Its only mutation was shutil.copy2 over files that
git cannot restore.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.rebuild_corpus import (
    BASELINE_PATH,
    DECLARED_DROPPED_KEYS,
    DECLARED_MOVED_KEYS,
    EXPECTED_CHANGED_FRAMEWORK_IDS,
    EXPECTED_UNCHANGED_RECORDS,
    RebuildReport,
    _parser_classes,
    assert_expected_frameworks_only,
    build_baseline,
    classify_removed,
    content_digest,
    corpus_from_framework_dir,
    diff_against_baseline,
    irrecoverable_members,
    resolve_required,
    restore_snapshot,
    snapshot_processed,
    sole_parser_class,
)
from tract.config import OVERLAY_FRAMEWORK_IDS, PROCESSED_FRAMEWORKS_DIR
from tract.parsers.base import BaseParser

REPO_ROOT = Path(__file__).resolve().parent.parent


def _baseline(*pairs: tuple[str, dict[str, object]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key, control in pairs:
        out.setdefault(key, []).append(content_digest(control))
    return out


class TestTheDiffSeesEveryAnchorField:
    def test_identical_content_reports_no_change(self) -> None:
        control = {"control_id": "C-1", "description": "statement one"}
        report = diff_against_baseline(
            {"demo": [control]}, _baseline(("demo:C-1", control)),
        )
        assert report.changed == []
        assert report.unchanged == 1

    def test_a_moved_full_text_is_a_change(self) -> None:
        """The defect this replaces. description is equal, full_text is not.

        ProseIndex.__init__ takes full_text when it is non-empty and never
        looks at description, so this control's anchor moved entirely while a
        description-only hash reports nothing.
        """
        old = {"control_id": "C-1", "description": "same", "full_text": "before"}
        new = {"control_id": "C-1", "description": "same", "full_text": "after"}
        report = diff_against_baseline({"demo": [new]}, _baseline(("demo:C-1", old)))
        assert report.changed == ["demo:C-1"]
        assert report.unchanged == 0

    def test_a_moved_title_is_a_change(self) -> None:
        """title is the join channel ProseIndex keys the by-name lookup on."""
        old = {"control_id": "C-1", "description": "same", "title": "before"}
        new = {"control_id": "C-1", "description": "same", "title": "after"}
        report = diff_against_baseline({"demo": [new]}, _baseline(("demo:C-1", old)))
        assert report.changed == ["demo:C-1"]

    def test_a_moved_alt_id_is_a_change(self) -> None:
        """alt_ids decides which control a link resolves to."""
        old = {"control_id": "C-1", "description": "same",
               "metadata": {"alt_ids": ["PO.1.1"]}}
        new = {"control_id": "C-1", "description": "same",
               "metadata": {"alt_ids": ["PO.1.2"]}}
        report = diff_against_baseline({"demo": [new]}, _baseline(("demo:C-1", old)))
        assert report.changed == ["demo:C-1"]

    def test_a_moved_alt_title_is_a_change(self) -> None:
        old = {"control_id": "C-1", "description": "same",
               "metadata": {"alt_titles": ["Poisoning"]}}
        new = {"control_id": "C-1", "description": "same",
               "metadata": {"alt_titles": ["Evasion"]}}
        report = diff_against_baseline({"demo": [new]}, _baseline(("demo:C-1", old)))
        assert report.changed == ["demo:C-1"]

    def test_alt_lists_are_order_insensitive(self) -> None:
        old = {"control_id": "C-1", "description": "same",
               "metadata": {"alt_titles": ["b", "a"]}}
        new = {"control_id": "C-1", "description": "same",
               "metadata": {"alt_titles": ["a", "b"]}}
        report = diff_against_baseline({"demo": [new]}, _baseline(("demo:C-1", old)))
        assert report.unchanged == 1

    def test_a_missing_field_and_an_empty_field_agree(self) -> None:
        """BaseParser.run writes with exclude_none, so absent means empty.

        A digest that distinguished the two would report every control whose
        parser stopped emitting a null full_text as changed.
        """
        absent = {"control_id": "C-1", "description": "same"}
        empty = {"control_id": "C-1", "description": "same", "full_text": None,
                 "metadata": {"alt_ids": None}}
        assert content_digest(absent) == content_digest(empty)


class TestCollidingKeysAreCountedPerRecord:
    """Nine keys hold 39 extra records, all with distinct text. [measured]"""

    def test_the_baseline_counts_records_not_keys(self) -> None:
        """The committed baseline declared 4,222 against 4,261 records.

        Nine keys absorbed 48 records and stored 9 digests, every one the
        FIRST writer's, so 39 records with distinct text were compared against
        nothing. All nine sit in enisa and etsi, the two frameworks this
        rebuild replaces. [measured]
        """
        corpus = {"frameworks": [{"framework_id": "enisa", "controls": [
            {"control_id": "Table 3:", "description": "poisoning"},
            {"control_id": "Table 3:", "description": "data disclosure"},
            {"control_id": "4.1", "description": "something else"},
        ]}]}
        baseline = build_baseline(corpus)
        assert baseline["n_keys"] == 2
        assert baseline["n_records"] == 3
        assert len(baseline["digests"]["enisa:Table 3:"]) == 2

    def test_two_records_under_one_key_are_two_units(self) -> None:
        first = {"control_id": "Table 3:", "description": "poisoning"}
        second = {"control_id": "Table 3:", "description": "data disclosure"}
        report = diff_against_baseline(
            {"enisa": [first, second]},
            _baseline(("enisa:Table 3:", first), ("enisa:Table 3:", second)),
        )
        assert report.unchanged == 2
        assert report.changed == []

    def test_losing_one_of_two_shadowed_records_is_visible(self) -> None:
        first = {"control_id": "Table 3:", "description": "poisoning"}
        second = {"control_id": "Table 3:", "description": "data disclosure"}
        report = diff_against_baseline(
            {"enisa": [first]},
            _baseline(("enisa:Table 3:", first), ("enisa:Table 3:", second)),
        )
        assert report.unchanged == 1
        assert report.removed == ["enisa:Table 3:"]


class TestRenamesAreNotLosses:
    def test_the_same_content_under_a_new_id_is_a_rename(self) -> None:
        old = {"control_id": "c1", "description": "validate every input"}
        new = {"control_id": "C1", "description": "validate every input"}
        report = diff_against_baseline(
            {"owasp_proactive_controls": [new]},
            _baseline(("owasp_proactive_controls:c1", old)),
        )
        assert report.renamed == [
            ("owasp_proactive_controls:c1", "owasp_proactive_controls:C1"),
        ]
        assert report.removed == []
        assert report.added == []

    def test_a_rename_does_not_cross_frameworks(self) -> None:
        old = {"control_id": "c1", "description": "validate every input"}
        new = {"control_id": "C1", "description": "validate every input"}
        report = diff_against_baseline(
            {"wstg": [new]}, _baseline(("owasp_proactive_controls:c1", old)),
        )
        assert report.renamed == []
        assert report.removed == ["owasp_proactive_controls:c1"]
        assert report.added == ["wstg:C1"]


class TestRemovedKeysAreClassifiedByIdLineage:
    """`removed` alone cannot tell a retired prefix from a lost control.

    Every pre-rebuild control_id in the eleven carries a redundant
    `<framework_id>:` prefix that the OpenCRE extraction wrote and no new
    parser reproduces, so all 436 baseline keys are literally removed. A stop
    rule reading that number without the lineage split cannot act on it.
    """

    def test_a_retired_prefix_is_not_a_loss(self) -> None:
        buckets = classify_removed(
            ["nist_ssdf:nist_ssdf:PO.1.1"],
            {"nist_ssdf": [{"control_id": "PO.1.1", "description": "x"}]},
        )
        assert buckets["prefix_only"] == ["nist_ssdf:nist_ssdf:PO.1.1"]
        assert buckets["id_reshaped"] == []
        assert buckets["gone"] == []

    def test_a_case_change_is_a_reshaped_id(self) -> None:
        buckets = classify_removed(
            ["wstg:wstg:wstg-athn-01"],
            {"wstg": [{"control_id": "WSTG-ATHN-01", "description": "x"}]},
        )
        assert buckets["id_reshaped"] == ["wstg:wstg:wstg-athn-01"]
        assert buckets["prefix_only"] == []

    def test_a_separator_change_is_a_reshaped_id(self) -> None:
        buckets = classify_removed(
            ["nist_800_63:nist_800_63:5-1-1-1"],
            {"nist_800_63": [{"control_id": "x",
                              "metadata": {"alt_ids": ["5.1.1.1"]}}]},
        )
        assert buckets["id_reshaped"] == ["nist_800_63:nist_800_63:5-1-1-1"]

    def test_a_control_with_no_successor_id_is_gone(self) -> None:
        buckets = classify_removed(
            ["enisa:enisa:Table 3:"],
            {"enisa": [{"control_id": "apply-a-rbac-model", "description": "x"}]},
        )
        assert buckets["gone"] == ["enisa:enisa:Table 3:"]

    def test_lineage_does_not_cross_frameworks(self) -> None:
        """samm:D-SA-A surviving under wstg would be a loss, not a rename."""
        buckets = classify_removed(
            ["samm:samm:D-SA-A"],
            {"wstg": [{"control_id": "D-SA-A", "description": "x"}]},
        )
        assert buckets["gone"] == ["samm:samm:D-SA-A"]

    def test_every_removed_key_lands_in_exactly_one_bucket(self) -> None:
        removed = ["nist_ssdf:nist_ssdf:PO.1.1", "wstg:wstg:wstg-athn-01",
                   "enisa:enisa:Table 3:"]
        buckets = classify_removed(removed, {
            "nist_ssdf": [{"control_id": "PO.1.1"}],
            "wstg": [{"control_id": "WSTG-ATHN-01"}],
            "enisa": [{"control_id": "apply-a-rbac-model"}],
        })
        flat = buckets["prefix_only"] + buckets["id_reshaped"] + buckets["gone"]
        assert sorted(flat) == sorted(removed)


class TestEveryParserModuleIsReachable:
    """A module the discovery misses reports its framework as unchanged.

    Needs no data/raw: it imports the modules and reads their class attributes.
    """

    def test_every_parse_module_yields_exactly_one_parser(self) -> None:
        from tract.config import PARSERS_DIR

        classes = _parser_classes()
        modules = sorted(PARSERS_DIR.glob("parse_*.py"))
        assert len(classes) == len(modules) == 32
        assert {path.stem[len("parse_"):] for path in modules} == set(classes)

    def test_one_candidate_is_returned(self) -> None:
        class Only(BaseParser):  # type: ignore[misc]
            framework_id = "only"

        assert sole_parser_class("parse_only.py", [Only]) is Only

    def test_two_candidates_raise_rather_than_taking_the_first(self) -> None:
        """Taking found[0] loses a framework and calls its controls unchanged."""
        class First(BaseParser):  # type: ignore[misc]
            framework_id = "first"

        class Second(BaseParser):  # type: ignore[misc]
            framework_id = "second"

        with pytest.raises(ValueError, match="defines 2 concrete"):
            sole_parser_class("parse_pair.py", [First, Second])

    def test_no_candidate_raises(self) -> None:
        with pytest.raises(ValueError, match="defines 0 concrete"):
            sole_parser_class("parse_empty.py", [])


class TestTheStopRuleIsAnAssertion:
    """Step 6 of the previous version was prose an autonomous worker reads past."""

    def test_an_unexpected_framework_halts_the_run(self) -> None:
        report = RebuildReport(changed=["capec:125"])
        with pytest.raises(SystemExit, match="capec"):
            assert_expected_frameworks_only(report)

    def test_a_framework_that_did_not_move_halts_the_run(self) -> None:
        """A parser that silently no-ops leaves the previous artifact in place."""
        report = RebuildReport(changed=[f"{f}:x" for f in (
            "biml", "csa_ccm", "dsomm", "enisa", "etsi", "nist_800_63",
            "nist_ssdf", "owasp_proactive_controls", "owasp_top10_2021", "samm",
        )])
        with pytest.raises(SystemExit, match="wstg"):
            assert_expected_frameworks_only(report)

    def test_a_wrong_unchanged_count_halts_the_run(self) -> None:
        report = RebuildReport(
            changed=[f"{f}:x" for f in EXPECTED_CHANGED_FRAMEWORK_IDS],
            unchanged=EXPECTED_UNCHANGED_RECORDS - 1,
        )
        with pytest.raises(SystemExit, match=str(EXPECTED_UNCHANGED_RECORDS)):
            assert_expected_frameworks_only(report)

    def test_a_declared_record_passes_and_its_neighbour_does_not(self) -> None:
        """The exemption is keyed on the record, not on the framework.

        pdfminer.six moves two NIST AI 100-2 figure records. Exempting the
        framework would hide the other 64, which is how a narrow known issue
        becomes a blanket amnesty.
        """
        base = [f"{f}:x" for f in EXPECTED_CHANGED_FRAMEWORK_IDS]
        allowed = RebuildReport(
            changed=[*base, "nist_ai_100_2:2.1", "nist_ai_100_2:3.1"],
            unchanged=EXPECTED_UNCHANGED_RECORDS,
        )
        assert assert_expected_frameworks_only(allowed) is None

        blocked = RebuildReport(
            changed=[*base, "nist_ai_100_2:2.1", "nist_ai_100_2:4.1"],
            unchanged=EXPECTED_UNCHANGED_RECORDS,
        )
        with pytest.raises(SystemExit, match="nist_ai_100_2:4.1"):
            assert_expected_frameworks_only(blocked)

    def test_a_rename_outside_the_eleven_halts_the_run(self) -> None:
        """renamed carries keys too, and reading only three buckets misses them."""
        report = RebuildReport(
            renamed=[("capec:125", "capec:CAPEC-125")],
            changed=[f"{f}:x" for f in EXPECTED_CHANGED_FRAMEWORK_IDS],
            unchanged=EXPECTED_UNCHANGED_RECORDS,
        )
        with pytest.raises(SystemExit, match="capec"):
            assert_expected_frameworks_only(report)

    def test_the_expected_shape_passes(self) -> None:
        report = RebuildReport(
            changed=[f"{f}:x" for f in (
                "biml", "csa_ccm", "dsomm", "enisa", "etsi", "nist_800_63",
                "nist_ssdf", "owasp_proactive_controls", "owasp_top10_2021",
                "samm", "wstg",
            )],
            added=["owasp_llm_top10_2026:LLM01"],
            unchanged=EXPECTED_UNCHANGED_RECORDS,
        )
        assert assert_expected_frameworks_only(report) is None

    def test_the_measured_shape_passes(self) -> None:
        """The eleven reach the gate through `removed` and `added`, not `changed`.

        Every baseline key in the eleven carries a redundant framework prefix,
        so nothing lands in `changed`. A gate that only inspected `changed`
        would pass a run in which all eleven parsers produced nothing.
        """
        report = RebuildReport(
            removed=[f"{f}:{f}:x" for f in EXPECTED_CHANGED_FRAMEWORK_IDS],
            added=[f"{f}:y" for f in EXPECTED_CHANGED_FRAMEWORK_IDS],
            unchanged=EXPECTED_UNCHANGED_RECORDS,
        )
        assert assert_expected_frameworks_only(report) is None

    def test_a_removal_outside_the_eleven_halts_the_run(self) -> None:
        report = RebuildReport(
            removed=["cwe:CWE-79"],
            changed=[f"{f}:x" for f in EXPECTED_CHANGED_FRAMEWORK_IDS],
            unchanged=EXPECTED_UNCHANGED_RECORDS,
        )
        with pytest.raises(SystemExit, match="cwe"):
            assert_expected_frameworks_only(report)


class TestTheSnapshotIsARollback:
    """The overlay's four members are untracked, so git cannot restore them.

    `scripts/fetch_frameworks.py` has no iso_27001 entry at all [measured], so
    ISO's raw source is hand-staged and its output is re-derivable from no
    scripted path. ISO is the corpus's only high-prose fold.
    """

    def test_a_snapshot_restores_byte_for_byte(self, tmp_path: Path) -> None:
        source = tmp_path / "processed"
        source.mkdir()
        original = '{\n  "a": 1\n}\n'
        (source / "etsi.json").write_text(original, encoding="utf-8")

        snapshot = snapshot_processed(tmp_path / "snapshots",
                                      members=[source / "etsi.json"])
        (source / "etsi.json").write_text('{"a": 2}', encoding="utf-8")
        assert restore_snapshot(snapshot) == 1
        assert (source / "etsi.json").read_text(encoding="utf-8") == original

    def test_a_tampered_snapshot_refuses_to_restore(self, tmp_path: Path) -> None:
        source = tmp_path / "processed"
        source.mkdir()
        (source / "etsi.json").write_text("{}\n", encoding="utf-8")
        snapshot = snapshot_processed(tmp_path / "snapshots",
                                      members=[source / "etsi.json"])
        member = next(p for p in snapshot.rglob("etsi.json"))
        member.write_text("tampered", encoding="utf-8")
        with pytest.raises(ValueError, match="does not match its manifest"):
            restore_snapshot(snapshot)

    def test_the_same_inputs_land_in_the_same_directory(self, tmp_path: Path) -> None:
        """No clock read, so a second --commit cannot bury the pristine copy."""
        source = tmp_path / "processed"
        source.mkdir()
        (source / "etsi.json").write_text("{}\n", encoding="utf-8")
        first = snapshot_processed(tmp_path / "s", members=[source / "etsi.json"])
        second = snapshot_processed(tmp_path / "s", members=[source / "etsi.json"])
        assert first == second

    def test_the_irrecoverable_set_is_derived_from_the_licence_tiering(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A hand-written list leaves every new overlay member with no rollback.

        Rulings R4 and R10 added csa_ccm and dsomm after the brief named three
        files. Deriving the set is what makes the next addition automatic.

        Driven through a substituted tier rather than read off the real one, so
        it still fails on a checkout that holds none of the overlay files. That
        is every CI run, and an assertion that reduces to {} == {} there would
        pass a hardcoded list without complaint.
        """
        frameworks = tmp_path / "frameworks"
        frameworks.mkdir()
        for name in ("etsi", "iso_27001", "csa_ccm", "dsomm", "capec"):
            (frameworks / f"{name}.json").write_text("{}\n", encoding="utf-8")
        monkeypatch.setattr("scripts.rebuild_corpus.PROCESSED_FRAMEWORKS_DIR",
                            frameworks)
        monkeypatch.setattr("scripts.rebuild_corpus.PROCESSED_LICENSED_DIR",
                            tmp_path / "licensed")
        monkeypatch.setattr("scripts.rebuild_corpus.OVERLAY_FRAMEWORK_IDS",
                            frozenset({"etsi", "iso_27001", "csa_ccm", "dsomm"}))
        assert {path.stem for path in irrecoverable_members()} == {
            "etsi", "iso_27001", "csa_ccm", "dsomm",
        }

        # The next member added to the tier is covered without editing this file.
        monkeypatch.setattr("scripts.rebuild_corpus.OVERLAY_FRAMEWORK_IDS",
                            frozenset({"etsi", "iso_27001", "csa_ccm", "dsomm",
                                       "capec"}))
        assert {path.stem for path in irrecoverable_members()} == {
            "etsi", "iso_27001", "csa_ccm", "dsomm", "capec",
        }

    def test_the_tier_holds_the_four_members_the_rulings_left(self) -> None:
        assert OVERLAY_FRAMEWORK_IDS == {"etsi", "iso_27001", "csa_ccm", "dsomm"}
        present = {fid for fid in OVERLAY_FRAMEWORK_IDS
                   if (PROCESSED_FRAMEWORKS_DIR / f"{fid}.json").exists()}
        assert {path.stem for path in irrecoverable_members()
                if path.parent == PROCESSED_FRAMEWORKS_DIR} == present

    def test_the_production_call_demands_the_irrecoverable_set(self) -> None:
        """The default path is the one that can lose ISO, so it carries the demand.

        A caller naming its own members is snapshotting some other tree, and
        demanding this repository's overlay files of it would refuse every call
        that is not the production one.
        """
        assert resolve_required(None, None) == irrecoverable_members()
        assert resolve_required([Path("/x/capec.json")], None) == []
        assert resolve_required([Path("/x/capec.json")], [Path("/x/etsi.json")]) == [
            Path("/x/etsi.json")
        ]

    def test_a_snapshot_that_misses_an_overlay_member_is_refused(
        self, tmp_path: Path,
    ) -> None:
        """The one irreversible act in this task, refused rather than logged."""
        source = tmp_path / "processed"
        source.mkdir()
        (source / "capec.json").write_text("{}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="irrecoverable"):
            snapshot_processed(
                tmp_path / "snapshots",
                members=[source / "capec.json"],
                require=[PROCESSED_FRAMEWORKS_DIR / "etsi.json"],
            )


class TestTheRegeneratedBaselineAgreesWithTheCommittedOne:
    """Changing the instrument must not change the answer it already gave."""

    def test_description_only_hashes_are_kept_alongside_the_new_ones(self) -> None:
        """The legacy map stays in the file so this check never goes dark.

        The brief's version skipped once the file was regenerated, which
        retires the only evidence that the five-field digest was fitted to the
        same corpus the description-only one measured.
        """
        committed = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        legacy = committed["sha256_of_description"]
        assert len(legacy) == 4222
        assert set(legacy) == set(committed["digests"])
        assert committed["n_keys"] == 4222
        assert committed["n_records"] == 4261
        assert sum(len(v) for v in committed["digests"].values()) == 4261

    def test_nine_keys_hold_thirty_nine_extra_records(self) -> None:
        """All nine sit in the two frameworks this rebuild replaces. [measured]"""
        committed = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        colliding = {key: len(values)
                     for key, values in committed["digests"].items()
                     if len(values) > 1}
        assert len(colliding) == 9
        assert sum(n - 1 for n in colliding.values()) == 39
        assert {key.split(":", 1)[0] for key in colliding} == {"enisa", "etsi"}

    def test_the_baseline_holds_the_pre_rebuild_corpus_not_the_current_one(
        self,
    ) -> None:
        """The file is named pre_rebuild and has to stay that.

        Regenerating it from the corpus on disk, which the brief's Step 4 does,
        makes every subsequent diff empty and turns the stop rule into a
        tautology. The eleven's ids all carried a redundant framework prefix
        before the rebuild and none carries one after, so this separates the
        two corpora with no dependence on their text.
        """
        committed = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        for framework_id in EXPECTED_CHANGED_FRAMEWORK_IDS:
            keys = [key for key in committed["digests"]
                    if key.startswith(f"{framework_id}:")]
            assert keys, f"{framework_id} absent from the pre-rebuild baseline"
            assert all(key.startswith(f"{framework_id}:{framework_id}:")
                       for key in keys), (
                f"{framework_id} keys in the baseline carry no framework prefix, "
                f"so the baseline was regenerated from the rebuilt corpus"
            )

    def test_the_legacy_map_reproduces_from_the_stored_digests(self) -> None:
        """Both maps must describe the same key set at the same multiplicity."""
        committed = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        legacy = committed["sha256_of_description"]
        assert all(len(value) == 64 for value in legacy.values())
        assert all(int(value, 16) >= 0 for value in legacy.values())


class TestOnlyTheElevenMovedAgainstTheRealCorpus:
    """The task's whole claim, frozen so a later parser edit cannot undo it."""

    def test_every_framework_outside_the_eleven_still_matches_the_baseline(
        self,
    ) -> None:
        if not PROCESSED_FRAMEWORKS_DIR.exists():
            pytest.skip("no processed frameworks in this checkout")
        committed = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        live = build_baseline(corpus_from_framework_dir(PROCESSED_FRAMEWORKS_DIR))
        # A checkout without the overlay holds no iso_27001.json, and reporting
        # its 93 records as moved would make this test fail on CI for a licence
        # reason rather than a corpus one. Absences are allowed only for the
        # overlay tier, so a missing capec.json still turns this red.
        on_disk = {path.stem for path in PROCESSED_FRAMEWORKS_DIR.glob("*.json")}
        absent = {key.split(":", 1)[0] for key in committed["digests"]} - on_disk
        assert absent <= OVERLAY_FRAMEWORK_IDS, (
            f"{sorted(absent - OVERLAY_FRAMEWORK_IDS)} are in the baseline and "
            f"not on disk, so the corpus lost a framework"
        )

        checked = 0
        moved: list[str] = []
        for key, digests in sorted(committed["digests"].items()):
            framework_id = key.split(":", 1)[0]
            if framework_id in EXPECTED_CHANGED_FRAMEWORK_IDS or framework_id in absent:
                continue
            checked += len(digests)
            if live["digests"].get(key) != digests:
                moved.append(key)
        assert sorted(moved) == sorted(DECLARED_MOVED_KEYS), (
            f"{sorted(set(moved) - DECLARED_MOVED_KEYS)} moved and their "
            f"parsers were not touched by this plan"
        )
        # 3,786 with the overlay, 3,693 without it: iso_27001 is the only
        # overlay member outside the eleven and it holds 93 records. [measured]
        assert checked == (3786 if not absent else 3693)
        if not absent:
            assert checked - len(DECLARED_MOVED_KEYS) == EXPECTED_UNCHANGED_RECORDS

    def test_no_key_in_the_eleven_survives_the_rebuild(self) -> None:
        """All 436 carried a redundant framework prefix. [measured]"""
        if not PROCESSED_FRAMEWORKS_DIR.exists():
            pytest.skip("no processed frameworks in this checkout")
        committed = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        live = build_baseline(corpus_from_framework_dir(PROCESSED_FRAMEWORKS_DIR))
        survivors = [
            key for key in committed["digests"]
            if key.split(":", 1)[0] in EXPECTED_CHANGED_FRAMEWORK_IDS
            and key in live["digests"]
        ]
        assert survivors == []
        retired = [key for key in committed["digests"]
                   if key.split(":", 1)[0] in EXPECTED_CHANGED_FRAMEWORK_IDS]
        assert len(retired) == 436


def test_the_committed_stopword_list_reproduces_from_the_committed_corpus() -> None:
    """Catches the staleness directly rather than by remembering to rerun it.

    The list is applied to every control and hub text and hashed into every
    fold record. A list built for a corpus that no longer exists is invisible
    in the metrics and changes every one of them.
    """
    from scripts.build_stopwords import collect_documents
    from tract.stopwords import STOPWORDS_PATH, generate_stopwords

    committed = json.loads(STOPWORDS_PATH.read_text(encoding="utf-8"))
    documents, protected = collect_documents()
    words = generate_stopwords(
        documents,
        min_doc_freq=committed["min_doc_freq"],
        max_words=committed["max_words"],
        protect=protected,
    )
    assert sorted(words) == committed["stopwords"]
    assert len(documents) == committed["n_documents"]


def test_every_retired_published_id_is_named() -> None:
    """Published assignments that lose their control identity are recorded.

    The export path cannot see them: they carry review_status='ground_truth'
    and tract/export/canonical.py filters on 'accepted', so no changeset will
    ever mention them. This file is the only record.
    """
    retired = json.loads(
        (REPO_ROOT / "results/corpus/retired_control_ids.json")
        .read_text(encoding="utf-8")
    )
    assert retired["n_rows"] == len(retired["rows"])
    assert retired["n_rows"] == 341
    assert {row["review_status"] for row in retired["rows"]} == {"ground_truth"}
    # Only the eleven can retire an identity, because only their parsers moved.
    assert {row["framework_id"] for row in retired["rows"]} <= (
        EXPECTED_CHANGED_FRAMEWORK_IDS
    )
    assert all(
        set(row) == {"control_id", "framework_id", "hub_id", "id_lineage",
                     "review_status", "section_id"}
        for row in retired["rows"]
    )
    # An operator can remap the reshaped ones mechanically and cannot remap the
    # rest, so the split is the actionable part of the record.
    assert retired["n_rows_by_id_lineage"] == {"gone": 78, "id_reshaped": 263}
    assert all(row["id_lineage"] in {"prefix_only", "id_reshaped", "gone"}
               for row in retired["rows"])


def test_the_rebuild_diff_records_what_the_run_reported() -> None:
    """The artifact an operator reads instead of rerunning a 90-second parse."""
    diff = json.loads(
        (REPO_ROOT / "results/corpus/rebuild_diff.json").read_text(encoding="utf-8")
    )
    assert diff["unchanged"] == EXPECTED_UNCHANGED_RECORDS
    # A dropped record has no live digest, so it leaves through the removed
    # bucket rather than the changed one. Both are declared, and the split is
    # what tells a reader whether a control was rewritten or withdrawn.
    assert diff["changed"] == sorted(DECLARED_MOVED_KEYS - DECLARED_DROPPED_KEYS)
    assert DECLARED_DROPPED_KEYS <= set(diff["removed"])
    assert len(diff["removed"]) == 436 + len(DECLARED_DROPPED_KEYS)
    buckets = diff["removed_classification"]
    assert (len(buckets["prefix_only"]) + len(buckets["id_reshaped"])
            + len(buckets["gone"])) == 436 + len(DECLARED_DROPPED_KEYS)
    touched = {key.split(":", 1)[0]
               for key in diff["changed"] + diff["added"] + diff["removed"]}
    assert touched == (
        EXPECTED_CHANGED_FRAMEWORK_IDS
        | {"owasp_llm_top10_2026", "nist_ai_100_2", "nist_ai_rmf", "aiuc_1"}
    )
    # Nothing outside the eleven reproduces anything but its own bytes, and
    # the five whose source_files block moved carry 0 differing controls.
    assert diff["live_artifact_comparison"]["absent"] == []


def test_the_tracked_corpus_carries_no_overlay_prose() -> None:
    """Checked here as well as in the tree-wide gate, per the licence routing.

    merge_all_controls withholds prose for OVERLAY_FRAMEWORK_IDS and drops
    RESTRICTED_FRAMEWORK_IDS outright. A rebuild that wrote through either is
    the publication event, because `git push` is one-way.
    """
    from tract.config import PROCESSED_DIR, RESTRICTED_FRAMEWORK_IDS

    tracked = json.loads(
        (PROCESSED_DIR / "all_controls.json").read_text(encoding="utf-8")
    )
    present = {str(f["framework_id"]) for f in tracked["frameworks"]}
    assert not (present & RESTRICTED_FRAMEWORK_IDS)
    for framework in tracked["frameworks"]:
        if framework["framework_id"] not in OVERLAY_FRAMEWORK_IDS:
            continue
        for control in framework["controls"]:
            assert not control.get("full_text"), (
                f"{framework['framework_id']}:{control['control_id']} carries "
                f"full_text in the tracked corpus"
            )
            assert str(control.get("description") or "").strip() == str(
                control.get("title") or ""
            ).strip(), (
                f"{framework['framework_id']}:{control['control_id']} carries "
                f"prose in the tracked corpus"
            )


def test_the_snapshot_manifest_digest_is_over_the_restored_bytes(
    tmp_path: Path,
) -> None:
    """A manifest computed over re-serialised text would verify nothing."""
    source = tmp_path / "processed"
    source.mkdir()
    text = '{"a":   1}\n'
    (source / "etsi.json").write_text(text, encoding="utf-8")
    snapshot = snapshot_processed(tmp_path / "s", members=[source / "etsi.json"])
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert list(manifest["files"].values()) == [expected]
    restore_snapshot(snapshot)
    assert (source / "etsi.json").read_text(encoding="utf-8") == text
