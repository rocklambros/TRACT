"""Tests for training data quality pipeline."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from tract.config import (
    CONTESTED_RECOVERY_DEFAULT,
    PHASE1B_MIN_ANCHOR_TEXT_LENGTH,
)
from tract.text_selection import ProseIndex, canonical_framework
from tract.training.data_quality import (
    CURATED_PATH,
    FilterReport,
    QualityTier,
    TieredLink,
    assign_quality_tier,
    compute_data_hash,
    curated_link_filter_report,
    filter_training_links,
    fold_input_digests,
    link_key,
    save_training_links,
)

# An anchor comfortably over PHASE1B_MIN_ANCHOR_TEXT_LENGTH, invented rather
# than quoted so no fixture in this repository carries licensed source text.
ANCHOR = (
    "The service rejects a request whose parameters fall outside the declared "
    "schema, and records the rejection with the offending field named."
)


def _index_with(framework_name: str, control_id: str, title: str) -> ProseIndex:
    """A one-control index that resolves by id and by title."""
    return ProseIndex([{
        "framework_name": framework_name,
        "controls": [
            {"control_id": control_id, "title": title, "description": ANCHOR},
        ],
    }])


class TestQualityTierAssignment:
    """Test quality tier assignment logic."""

    def test_human_linked_traditional_is_t1(self) -> None:
        link = {
            "cre_id": "760-764",
            "cre_name": "Injection protection",
            "standard_name": "OWASP Top 10 2021",
            "link_type": "LinkedTo",
            "section_id": "A03",
            "section_name": "Injection prevention and input validation",
            "framework_id": "owasp_top10_2021",
        }
        assert assign_quality_tier(link, ANCHOR) == QualityTier.T1

    def test_auto_linked_is_t3(self) -> None:
        link = {
            "cre_id": "760-764",
            "cre_name": "Injection protection",
            "standard_name": "CAPEC",
            "link_type": "AutomaticallyLinkedTo",
            "section_id": "CAPEC-66",
            "section_name": "SQL Injection",
            "framework_id": "capec",
        }
        assert assign_quality_tier(link, ANCHOR) == QualityTier.T3

    def test_ai_framework_is_t1_ai(self) -> None:
        link = {
            "cre_id": "123-456",
            "cre_name": "Test",
            "standard_name": "MITRE ATLAS",
            "link_type": "LinkedTo",
            "section_id": "AML.T0001",
            "section_name": "Adversarial Perturbation",
            "framework_id": "mitre_atlas",
        }
        assert assign_quality_tier(link, ANCHOR) == QualityTier.T1_AI

    def test_an_unresolved_link_is_dropped(self) -> None:
        """The gate reads the anchor, and there is none."""
        link = {
            "cre_id": "111-222",
            "cre_name": "Test",
            "standard_name": "NIST 800-63",
            "link_type": "LinkedTo",
            "section_id": "5.1.4.2",
            "section_name": "5.1.4.2",
            "framework_id": "nist_800_63",
        }
        assert assign_quality_tier(link, None) == QualityTier.DROPPED

    def test_a_thin_anchor_is_dropped(self) -> None:
        link = {
            "cre_id": "333-444",
            "cre_name": "Test",
            "standard_name": "DSOMM",
            "link_type": "LinkedTo",
            "section_id": "D1",
            "section_name": "Process",
            "framework_id": "dsomm",
        }
        # The floor is exclusive, so pin both sides of it. "Do backup" is 9
        # characters and "Do backups" is 10, which is the shortest anchor the
        # gate admits.
        assert len("Do backup") == PHASE1B_MIN_ANCHOR_TEXT_LENGTH - 1
        assert assign_quality_tier(link, "Do backup") == QualityTier.DROPPED
        assert assign_quality_tier(link, "Do backups") == QualityTier.T1

    def test_surrounding_whitespace_does_not_pad_a_thin_anchor(self) -> None:
        """The floor reads the stripped anchor, not the raw one."""
        link = {
            "cre_id": "333-445",
            "standard_name": "DSOMM",
            "link_type": "LinkedTo",
            "section_id": "D2",
            "section_name": "Process",
            "framework_id": "dsomm",
        }
        assert assign_quality_tier(link, "   Backups   ") == QualityTier.DROPPED

    def test_a_formerly_denied_framework_is_kept_on_its_anchor(self) -> None:
        """This assertion is inverted from the one it replaces.

        The retired gate dropped every owasp_proactive_controls link by name,
        whatever text stood behind it. All 76 now train, because the parser
        landed and the gate reads what the parser produced.
        """
        link = {
            "cre_id": "555-666",
            "cre_name": "Test",
            "standard_name": "OWASP Proactive Controls",
            "link_type": "LinkedTo",
            "section_id": "C1",
            "section_name": "C1",
            "framework_id": "owasp_proactive_controls",
        }
        assert assign_quality_tier(link, ANCHOR) == QualityTier.T1

    def test_all_five_ai_frameworks_are_t1_ai(self) -> None:
        for name in [
            "MITRE ATLAS",
            "NIST AI 100-2",
            "OWASP AI Exchange",
            "OWASP Top10 for LLM",
            "OWASP Top10 for ML",
        ]:
            link = {
                "cre_id": "100-200",
                "cre_name": "Test",
                "standard_name": name,
                "link_type": "LinkedTo",
                "section_id": "X1",
                "section_name": "A descriptive section text",
                "framework_id": "test_fw",
            }
            assert assign_quality_tier(link, ANCHOR) == QualityTier.T1_AI, (
                f"Failed for {name}"
            )


class TestFilterTrainingLinks:
    """Test end-to-end link filtering."""

    def test_filters_dropped_links(self) -> None:
        links = [
            {
                "cre_id": "1",
                "cre_name": "A",
                "standard_name": "ASVS",
                "link_type": "LinkedTo",
                "section_id": "V1",
                "section_name": "Architecture Assessment",
                "framework_id": "asvs",
            },
            {
                "cre_id": "2",
                "cre_name": "B",
                "standard_name": "NIST 800-63",
                "link_type": "LinkedTo",
                "section_id": "5.1.4.2",
                "section_name": "5.1.4.2",
                "framework_id": "nist_800_63",
            },
        ]
        report = filter_training_links(
            links, _index_with("ASVS", "V1", "Architecture Assessment"),
        )
        assert len(report.kept) == 1
        assert report.kept[0].link["cre_id"] == "1"
        assert report.kept[0].tier == QualityTier.T1

    def test_preserves_all_ai_links(self) -> None:
        links = [
            {
                "cre_id": "1",
                "cre_name": "A",
                "standard_name": "MITRE ATLAS",
                "link_type": "LinkedTo",
                "section_id": "AML.T0001",
                "section_name": "Adversarial Perturbation",
                "framework_id": "mitre_atlas",
            },
        ]
        report = filter_training_links(
            links,
            _index_with("MITRE ATLAS", "AML.T0001", "Adversarial Perturbation"),
        )
        assert len(report.kept) == 1
        assert report.kept[0].tier == QualityTier.T1_AI

    def test_mixed_tiers_in_result(self) -> None:
        links = [
            {
                "cre_id": "1",
                "cre_name": "A",
                "standard_name": "ASVS",
                "link_type": "LinkedTo",
                "section_id": "V1",
                "section_name": "Architecture Assessment",
                "framework_id": "asvs",
            },
            {
                "cre_id": "2",
                "cre_name": "B",
                "standard_name": "CAPEC",
                "link_type": "AutomaticallyLinkedTo",
                "section_id": "CAPEC-66",
                "section_name": "SQL Injection Attack",
                "framework_id": "capec",
            },
            {
                "cre_id": "3",
                "cre_name": "C",
                "standard_name": "OWASP AI Exchange",
                "link_type": "LinkedTo",
                "section_id": "OAI-1",
                "section_name": "AI Supply Chain Integrity",
                "framework_id": "owasp_ai_exchange",
            },
        ]
        index = ProseIndex([
            {"framework_name": "ASVS", "controls": [
                {"control_id": "V1", "title": "Architecture Assessment",
                 "description": ANCHOR}]},
            {"framework_name": "CAPEC", "controls": [
                {"control_id": "CAPEC-66", "title": "SQL Injection Attack",
                 "description": ANCHOR}]},
            {"framework_name": "OWASP AI Exchange", "controls": [
                {"control_id": "OAI-1", "title": "AI Supply Chain Integrity",
                 "description": ANCHOR}]},
        ])
        report = filter_training_links(links, index)
        assert len(report.kept) == 3
        tiers = {r.tier for r in report.kept}
        assert tiers == {QualityTier.T1, QualityTier.T3, QualityTier.T1_AI}


class TestDataHash:
    """Test deterministic data hashing."""

    def test_hash_is_deterministic(self) -> None:
        data = [{"a": 1}, {"b": 2}]
        h1 = compute_data_hash(data)
        h2 = compute_data_hash(data)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_hash_changes_with_data(self) -> None:
        h1 = compute_data_hash([{"a": 1}])
        h2 = compute_data_hash([{"a": 2}])
        assert h1 != h2

    def test_hash_is_order_independent_for_keys(self) -> None:
        h1 = compute_data_hash([{"a": 1, "b": 2}])
        h2 = compute_data_hash([{"b": 2, "a": 1}])
        assert h1 == h2  # sort_keys=True makes this deterministic


def test_quality_tier_al_exists() -> None:
    from tract.training.data_quality import QualityTier
    assert QualityTier.AL.value == "AL"


class TestGatesTestTheAnchorNotTheTitle:
    """Both drops used to test section_name, which the model never sees."""

    def test_a_short_title_with_a_resolved_anchor_is_kept(self) -> None:
        """76 owasp_proactive_controls links look exactly like this one."""
        link = {
            "framework_id": "owasp_proactive_controls",
            "standard_name": "OWASP Proactive Controls",
            "section_id": "C6", "section_name": "C6",
            "link_type": "LinkedTo",
        }
        assert assign_quality_tier(link, ANCHOR) is QualityTier.T1

    def test_an_unresolved_link_is_dropped_however_long_its_title(self) -> None:
        """The nine wstg links this closes carry 11 and 12 character ids.

        Falling back to section_name would train "WSTG-BUSL-$$" against a real
        CRE hub, because section_name == section_id for all 118 wstg rows and
        the four bogus ids clear the ten-character floor. [measured]
        """
        for name in ("WSTG-BUSL-$$", "WSTG-INPV-00",
                     "Security of assets off-premises"):
            assert len(name) >= PHASE1B_MIN_ANCHOR_TEXT_LENGTH, name
            link = {
                "framework_id": "wstg", "standard_name": "OWASP WSTG",
                "section_id": name, "section_name": name,
                "link_type": "LinkedTo",
            }
            assert assign_quality_tier(link, None) is QualityTier.DROPPED, name

    def test_the_anchor_parameter_has_no_default(self) -> None:
        """A defaulted second parameter is how the ceiling study broke silently.

        tract/ceiling_study.py called assign_quality_tier(record) with one
        argument under a docstring promising it mirrored training. Give
        resolved_text a default and that call keeps compiling while the two
        pools diverge, and nothing raises.
        """
        parameter = inspect.signature(assign_quality_tier).parameters["resolved_text"]
        assert parameter.default is inspect.Parameter.empty

    def test_the_framework_deny_list_is_gone(self) -> None:
        import tract.config as config

        assert not hasattr(config, "PHASE1B_DROPPED_FRAMEWORKS")
        assert not hasattr(config, "PHASE1B_MIN_SECTION_TEXT_LENGTH")

    def test_filter_reports_each_drop_reason_separately(self) -> None:
        links = [
            {"framework_id": "owasp_proactive_controls",
             "standard_name": "OWASP Proactive Controls",
             "section_id": "C6", "section_name": "C6",
             "cre_id": "1", "link_type": "LinkedTo"},
            {"framework_id": "owasp_proactive_controls",
             "standard_name": "OWASP Proactive Controls",
             "section_id": "C9", "section_name": "C9",
             "cre_id": "2", "link_type": "LinkedTo"},
        ]
        index = _index_with(
            "OWASP Proactive Controls", "C6", "Use Secure Dependencies",
        )
        report = filter_training_links(links, index)
        assert len(report.kept) == 1
        assert report.dropped_unresolved == ["owasp_proactive_controls|C9|C9|2"]
        assert report.dropped_thin_anchor == []
        assert report.dropped_contested == []
        assert report.n_dropped == 1

    def test_a_thin_anchor_is_reported_apart_from_an_absent_one(self) -> None:
        """The two drops call for different responses, so they are two lists.

        An unresolved link means a parser or a join is missing. A thin anchor
        means the parser ran and the source really is that terse.

        full_text rather than description, because ProseIndex admits a short
        description only when it exceeds the title by PROSE_MIN_EXTRA_CHARS.
        A control terse enough to fail the anchor floor via its description is
        never indexed at all, which is why the live corpus reports sixteen
        unresolved drops and zero thin ones.
        """
        links = [
            {"framework_id": "dsomm", "standard_name": "DSOMM",
             "section_id": "D1", "section_name": "Backups",
             "cre_id": "1", "link_type": "AutomaticallyLinkedTo"},
        ]
        index = ProseIndex([{
            "framework_name": "DSOMM",
            "controls": [{"control_id": "D1", "title": "B",
                          "full_text": "Do backup"}],
        }])
        report = filter_training_links(links, index)
        assert report.kept == []
        assert report.dropped_unresolved == []
        assert report.dropped_thin_anchor == ["dsomm|D1|Backups|1"]

    def test_contested_recovery_is_a_lever_with_both_values_live(self) -> None:
        """capec alpha-1 is 0.181, so restoring its terse links is a choice.

        [measured, results/ceiling_study/panel_agreement.md]
        """
        index = _index_with("CAPEC", "125", "Flooding")
        link = {"framework_id": "capec", "standard_name": "CAPEC",
                "section_id": "125", "section_name": "Flooding",
                "cre_id": "1", "link_type": "LinkedTo"}
        on = filter_training_links([link], index, recover_contested=True)
        assert len(on.kept) == 1
        assert on.dropped_contested == []
        off = filter_training_links([link], index, recover_contested=False)
        assert off.kept == []
        assert off.dropped_contested == ["capec|125|Flooding|1"]

    def test_the_lever_leaves_a_long_titled_capec_link_alone(self) -> None:
        """recover_contested only ever moves links the title floor dropped.

        Without this, "exclude capec" and "exclude the terse capec links"
        would be the same switch, and reverting the recovery would take 1,755
        uncontested links with it.
        """
        index = _index_with(
            "CAPEC", "66", "SQL Injection Attack Against A Web Service",
        )
        link = {"framework_id": "capec", "standard_name": "CAPEC",
                "section_id": "66",
                "section_name": "SQL Injection Attack Against A Web Service",
                "cre_id": "1", "link_type": "AutomaticallyLinkedTo"}
        off = filter_training_links([link], index, recover_contested=False)
        assert len(off.kept) == 1
        assert off.dropped_contested == []


class TestTheFilterReportNamesTheCorpusItRead:
    """Provenance has to follow the index, never a second look at the disk."""

    def test_an_in_memory_index_names_no_corpus(self) -> None:
        """Reporting merged_corpus_path() here would be a false digest.

        It is the same defect run_single_fold shipped: a recorded hash for a
        file the run never opened. None is the honest answer for an index
        built from literals.
        """
        report = filter_training_links([], _index_with("ASVS", "V1", "T"))
        assert report.corpus_path is None
        assert report.corpus_sha256 is None

    def test_a_loaded_index_names_the_file_it_loaded(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus.json"
        corpus.write_text(json.dumps({"frameworks": [
            {"framework_id": "asvs", "framework_name": "ASVS", "controls": [
                {"control_id": "V1", "title": "Architecture",
                 "description": ANCHOR}]},
        ]}), encoding="utf-8")

        report = filter_training_links([], ProseIndex.load(corpus))
        assert report.corpus_path == str(corpus)
        assert report.corpus_sha256 == hashlib.sha256(
            corpus.read_bytes()
        ).hexdigest()

    def test_two_corpora_produce_two_digests(self, tmp_path: Path) -> None:
        """The property that was missing: two corpora, two recorded digests."""
        digests = set()
        for n, name in enumerate(("a.json", "b.json")):
            corpus = tmp_path / name
            corpus.write_text(json.dumps({"frameworks": [
                {"framework_id": "asvs", "framework_name": "ASVS", "controls": [
                    {"control_id": f"V{n}", "title": "Architecture",
                     "description": ANCHOR}]},
            ]}), encoding="utf-8")
            report = filter_training_links([], ProseIndex.load(corpus))
            digests.add(report.corpus_sha256)
        assert len(digests) == 2


class TestTrainingFileRecordsTheCorpusItRead:
    """hub_links_training.jsonl is a function of the corpus after this task.

    save_training_links previously recorded only the curated-links hash, so
    two runs over different corpora produced the same raw_hash. Nothing in the
    repository reads the file it writes, so these tests exercise the writer
    directly rather than asserting against a committed artifact that no run
    consumes.
    """

    def test_save_requires_the_corpus_digest(self) -> None:
        parameter = inspect.signature(save_training_links).parameters["corpus_sha256"]
        assert parameter.default is inspect.Parameter.empty

    def test_the_sidecar_names_the_corpus_and_counts_the_rows(
        self, tmp_path: Path,
    ) -> None:
        links = [
            TieredLink(link={"cre_id": "1", "framework_id": "asvs"},
                       tier=QualityTier.T1),
            TieredLink(link={"cre_id": "2", "framework_id": "asvs"},
                       tier=QualityTier.T3),
        ]
        out = tmp_path / "hub_links_training.jsonl"
        output_hash = save_training_links(links, "raw-hash", "corpus-hash", out)

        meta = json.loads(
            (tmp_path / "hub_links_training.meta.json").read_text(encoding="utf-8")
        )
        assert meta["corpus_sha256"] == "corpus-hash"
        assert meta["curated_links_sha256"] == "raw-hash"
        assert meta["output_sha256"] == output_hash
        assert meta["n_links"] == 2
        assert meta["n_links"] == len(
            [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
        )

    def test_two_corpora_do_not_share_a_sidecar_digest(
        self, tmp_path: Path,
    ) -> None:
        """Two runs over corpora that differ must be distinguishable.

        Before this change both runs recorded only raw_hash, which is a
        digest of the curated links and identical across the two.
        """
        links = [TieredLink(link={"cre_id": "1"}, tier=QualityTier.T1)]
        recorded = []
        for n, corpus_hash in enumerate(("corpus-a", "corpus-b")):
            out = tmp_path / f"run{n}.jsonl"
            save_training_links(links, "same-raw-hash", corpus_hash, out)
            meta = json.loads(
                (tmp_path / f"run{n}.meta.json").read_text(encoding="utf-8")
            )
            recorded.append((meta["curated_links_sha256"], meta["corpus_sha256"]))
        assert recorded[0][0] == recorded[1][0], "the curated hash is the same"
        assert recorded[0][1] != recorded[1][1], "the corpus hash is not"


class TestFoldProvenanceNamesTheCorpusTheRunRead:
    """run_single_fold recorded a digest for a file it had not opened."""

    def test_the_recorded_digest_follows_merged_corpus_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Hashing PROCESSED_DIR / "all_controls.json" passes nothing here.

        The overlay and the tracked corpus are different files, and the run
        reads whichever merged_corpus_path returns. Pointing that function at
        a known file and asserting the recorded digest matches it is what the
        old implementation could not do.
        """
        corpus = tmp_path / "overlay.json"
        corpus.write_text('{"frameworks": []}', encoding="utf-8")
        monkeypatch.setattr(
            "tract.text_selection.merged_corpus_path", lambda: corpus,
        )

        digests = fold_input_digests(
            with_prose=True, with_stopwords=False,
            with_framework_identity=False,
        )
        assert digests["all_controls_sha256"] == hashlib.sha256(
            corpus.read_bytes()
        ).hexdigest()
        assert digests["stopwords_sha256"] is None
        assert digests["framework_identity_sha256"] is None

    def test_an_arm_without_prose_records_no_corpus_digest(self) -> None:
        digests = fold_input_digests(
            with_prose=False, with_stopwords=False,
            with_framework_identity=False,
        )
        assert digests["all_controls_sha256"] is None
        assert digests["curated_links_sha256"] is not None

    def test_each_filter_arm_records_only_the_file_it_read(self) -> None:
        """Two arms, two artifacts, and neither may stand in for the other.

        A run with only the framework-identity arm on still holds a non-empty
        filter set, so a record keyed on "the set is non-empty" would name
        stopwords.json for a run that never opened it.
        """
        identity_only = fold_input_digests(
            with_prose=False, with_stopwords=False,
            with_framework_identity=True,
        )
        assert identity_only["stopwords_sha256"] is None
        assert identity_only["framework_identity_sha256"] is not None

        stopwords_only = fold_input_digests(
            with_prose=False, with_stopwords=True,
            with_framework_identity=False,
        )
        assert stopwords_only["stopwords_sha256"] is not None
        assert stopwords_only["framework_identity_sha256"] is None


class TestTheAnchorGateReachesItsDerivedCount:
    """4,389 of 4,405, and the sixteen exceptions named rather than counted."""

    # Every link the gate is expected to drop, keyed (framework_id,
    # section_id). Nine wstg ids appear in no file of the pinned archive and
    # one dsomm activity carries a placeholder statement that ProseIndex
    # refuses to index; the other six match no parsed control. Measured
    # against the 31-framework corpus after Tasks 3-13. [measured]
    EXPECTED_UNRESOLVED: frozenset[tuple[str, str]] = frozenset({
        ("wstg", "WSTG-BUSL-$$"), ("wstg", "WSTG-INPV-00"),
        ("wstg", "WSTG-APPE-D"), ("wstg", "WSTG-INFO-##"),
        ("nist_800_53", "SC-23(1)"), ("nist_800_53", "SC-23(3)"),
        ("iso_27001", "7.8"), ("iso_27001", "7.9"),
        ("nist_800_63", "are g"), ("cwe", "937"),
        ("dsomm", "7de0ae33-6538-45cd-8222-a1475647ba58"),
    })
    N_CURATED = 4405                   # [measured]
    EXPECTED_KEPT_FULL_CORPUS = 4389   # [measured] 4,405 less 16 unresolved
    CONTESTED_RECOVERED = 60           # capec 44 + cwe 16 [measured]

    def _report(self, *, recover_contested: bool = True) -> FilterReport:
        """Always explicit. The shipped default is asserted in one place only,
        TestTheShippedContestedDefault, so flipping it moves one test rather
        than every count in this class.
        """
        report, _ = curated_link_filter_report(recover_contested=recover_contested)
        return report

    def _curated(self) -> list[dict[str, str]]:
        return [
            json.loads(line)
            for line in CURATED_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_the_curated_file_is_the_one_this_plan_measured(self) -> None:
        """Every count below is derived from this base."""
        assert len(self._curated()) == self.N_CURATED

    def test_every_drop_is_one_this_plan_predicted(self) -> None:
        """Fails in both directions: an unexpected drop, or an unexpected keep.

        Restricted to frameworks this checkout's corpus can answer at all. A
        framework whose parser output has not been merged into the tracked
        corpus resolves none of its links, and that is a property of the
        checkout rather than of the gate.
        """
        report = self._report()
        answerable = ProseIndex.load().answerable_frameworks()
        by_id = {
            link["framework_id"]: canonical_framework(link.get("standard_name", ""))
            for link in self._curated()
        }
        surprises = sorted(
            key for key in report.dropped_unresolved
            if by_id.get(key.split("|")[0], "") in answerable
            and (key.split("|")[0], key.split("|")[1]) not in self.EXPECTED_UNRESOLVED
        )
        assert surprises == [], (
            "these links resolve to no parsed control and this plan did not "
            f"predict that: {surprises[:20]}"
        )
        assert report.dropped_thin_anchor == [], (
            "a control resolved to fewer than ten characters of text. No "
            "parser in Tasks 3-13 was expected to emit one, so this is a "
            f"parser defect, not a source limit: {report.dropped_thin_anchor}"
        )

    def test_the_count_matches_the_corpus_that_was_read(self) -> None:
        """4,389 needs all 31 frameworks. Derive, never hard-code one literal.

        merged_corpus_path returns the gitignored overlay when it exists and
        the tracked corpus otherwise, and the tracked corpus always exists, so
        an existence check never skips. The expectation is derived from which
        frameworks the corpus can answer, NOT from which framework records the
        corpus file contains: a record whose controls all restate their titles
        contributes no index key, and counting records instead scores this
        checkout's tracked corpus 4,259 against an actual 4,019. [measured]
        """
        curated = self._curated()
        report = self._report()
        answerable = ProseIndex.load().answerable_frameworks()

        unanswerable_links = 0
        predicted_drops = 0
        for link in curated:
            if canonical_framework(link.get("standard_name", "")) not in answerable:
                unanswerable_links += 1
            elif (link["framework_id"], link.get("section_id", "")) \
                    in self.EXPECTED_UNRESOLVED:
                predicted_drops += 1

        expected = len(curated) - unanswerable_links - predicted_drops
        assert len(report.kept) == expected, (
            f"{len(report.kept)} kept against {expected} expected, reading "
            f"{report.corpus_path} which answers {len(answerable)} frameworks"
        )

    def test_the_headline_count_is_what_the_arithmetic_gives(self) -> None:
        """Ties the 4,389 in the commit message to the sixteen named drops.

        Without this the literal and the drop set could drift apart and both
        would still look measured.
        """
        assert self.EXPECTED_KEPT_FULL_CORPUS == self.N_CURATED - 16
        answerable = ProseIndex.load().answerable_frameworks()
        curated = self._curated()
        if all(
            canonical_framework(link.get("standard_name", "")) in answerable
            for link in curated
        ):
            assert len(self._report().kept) == self.EXPECTED_KEPT_FULL_CORPUS

    def test_no_kept_link_trains_on_a_section_title(self) -> None:
        """This is the property the whole task exists to establish."""
        index = ProseIndex.load()
        report = self._report()
        titles = [
            tiered.link for tiered in report.kept
            if index.lookup(tiered.link.get("standard_name", ""),
                            tiered.link.get("section_id"),
                            tiered.link.get("section_name")) is None
        ]
        assert titles == []

    def test_the_contested_lever_moves_exactly_the_contested_links(self) -> None:
        full = self._report()
        without = self._report(recover_contested=False)
        assert len(full.kept) - len(without.kept) == self.CONTESTED_RECOVERED
        assert {key.split("|")[0] for key in without.dropped_contested} == {
            "capec", "cwe",
        }
        # cwe 937 is contested by definition and unresolved as well, so the
        # contested bucket holds one more link than the recovery is worth.
        assert len(without.dropped_contested) == self.CONTESTED_RECOVERED + 1

    def test_the_lever_does_not_touch_any_other_framework(self) -> None:
        full = self._report()
        without = self._report(recover_contested=False)
        moved = {
            link_key(t.link) for t in full.kept
        } - {link_key(t.link) for t in without.kept}
        assert {key.split("|")[0] for key in moved} == {"capec", "cwe"}


class TestTheShippedContestedDefault:
    """What every caller that passes nothing gets.

    Split out on purpose. Recovering the 44 capec and 16 cwe links the title
    floor dropped is a decision, not a repair: capec agreement with OpenCRE is
    alpha-1 = 0.181 [0.113, 0.277] on n=83, so those are the terse links from
    the least-agreed framework in the study. It ships as its own commit, and
    reverting that commit restores the constant below without disturbing the
    eleven other frameworks' 202 net recoveries. Every other assertion in this
    file passes recover_contested explicitly, so the flip moves one line here
    and nothing else.
    """

    def test_both_entry_points_expose_the_same_default(self) -> None:
        """One decision, two signatures. They cannot half-move."""
        inner = inspect.signature(
            filter_training_links
        ).parameters["recover_contested"].default
        outer = inspect.signature(
            curated_link_filter_report
        ).parameters["recover_contested"].default
        assert inner is outer is CONTESTED_RECOVERY_DEFAULT

    def test_the_shipped_decision_is_to_recover_the_contested_links(self) -> None:
        """The one line reverting the recovery commit flips back."""
        assert CONTESTED_RECOVERY_DEFAULT is True

    def test_a_caller_that_passes_nothing_gets_the_shipped_decision(self) -> None:
        index = _index_with("CAPEC", "125", "Flooding")
        link = {"framework_id": "capec", "standard_name": "CAPEC",
                "section_id": "125", "section_name": "Flooding",
                "cre_id": "1", "link_type": "LinkedTo"}
        kept = len(filter_training_links([link], index).kept)
        assert kept == (1 if CONTESTED_RECOVERY_DEFAULT else 0)


class TestCorpusIdentityPreflight:
    """A fresh clone trains on 8.4% fewer links and says nothing about it.

    `merged_corpus_path()` prefers the gitignored licensed overlay and falls
    back to the tracked corpus when it is absent. That is correct for a reader
    and wrong for a trainer: 341 of the 4,389 training links belong to the three
    overlay frameworks, so without the overlay they resolve to nothing and the
    run trains on 4,048 while reporting the same shape of output. It was four
    frameworks and 370 links until csa_ccm left the overlay on 2026-08-26.

    Existence cannot catch it, because both files exist. Only the digest can.
    """

    def _meta(self, tmp_path: Path, digest: str | None) -> Path:
        path = tmp_path / "hub_links_training.meta.json"
        body: dict[str, object] = {"n_links": 4389}
        if digest is not None:
            body["corpus_sha256"] = digest
        path.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
        return path

    def test_the_matching_corpus_passes_and_returns_its_digest(
        self, tmp_path: Path
    ) -> None:
        from tract.text_selection import merged_corpus_sha256
        from tract.training.data_quality import assert_corpus_matches_training_links

        actual = merged_corpus_sha256()
        assert assert_corpus_matches_training_links(
            self._meta(tmp_path, actual)
        ) == actual

    def test_a_different_corpus_is_refused_and_names_both_digests(
        self, tmp_path: Path
    ) -> None:
        from tract.text_selection import merged_corpus_sha256
        from tract.training.data_quality import (
            CorpusMismatchError,
            assert_corpus_matches_training_links,
        )

        wrong = "0" * 64
        with pytest.raises(CorpusMismatchError) as caught:
            assert_corpus_matches_training_links(self._meta(tmp_path, wrong))
        message = str(caught.value)
        assert wrong in message
        assert merged_corpus_sha256() in message
        assert "RUNNING_ELSEWHERE" in message

    def test_an_absent_sidecar_stops_rather_than_passing_quietly(
        self, tmp_path: Path
    ) -> None:
        from tract.training.data_quality import assert_corpus_matches_training_links

        with pytest.raises(FileNotFoundError):
            assert_corpus_matches_training_links(tmp_path / "absent.json")

    def test_a_sidecar_with_no_digest_is_refused(self, tmp_path: Path) -> None:
        """Predates the anchor gate, so it cannot say which corpus it used."""
        from tract.training.data_quality import (
            CorpusMismatchError,
            assert_corpus_matches_training_links,
        )

        with pytest.raises(CorpusMismatchError, match="records no corpus_sha256"):
            assert_corpus_matches_training_links(self._meta(tmp_path, None))


class TestRunFoldEnforcesTheCorpusGate:
    """The pod-side half of the corpus gate.

    The orchestrator refuses before it provisions, which is where the money is
    saved. This is the half that holds when a fold is launched some other way:
    by hand on a pod, by a resumed fleet, or by a future caller that does not
    go through runpod_parallel. The control has to live where training starts,
    not only where the convenient caller happens to be.
    """

    def test_a_partial_corpus_stops_the_fold_before_it_trains(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("torch", reason="needs the phase0 extra")
        pytest.importorskip("datasets", reason="needs the phase0 extra")

        from scripts.phase1b import run_fold
        from tract.training.data_quality import CorpusMismatchError

        trained: list[str] = []
        monkeypatch.setattr(
            run_fold, "run_single_fold",
            lambda *a, **k: trained.append("trained") or {},
        )

        def _mismatch() -> str:
            raise CorpusMismatchError("corpus digest differs from the recorded one")

        monkeypatch.setattr(
            run_fold, "assert_corpus_matches_training_links", _mismatch,
        )
        monkeypatch.setattr(
            "sys.argv",
            ["run_fold.py", "--framework", "MITRE ATLAS", "--config-name", "cfg"],
        )

        with pytest.raises(CorpusMismatchError):
            run_fold.main()

        assert trained == []
