"""Bridge links must actually reach the training corpus.

Checkpoint 2's top finding, reached independently by four perspectives:
`hub_links_bridge.jsonl` had exactly one reference in the repository and it was
the WRITE target. `bridge_training_records` and `merge_for_training` had zero
non-test callers. Every training entry point called
`load_and_filter_curated_links()`, which reads `CURATED_PATH` and nothing else.

So the documented flow -- build packet, annotate, import, retrain -- produced a
retrain byte-identical to every prior run. Under the strict firewall the AI
region stays 78/78 orphaned, the trained arm loses to its zero-shot, and the
round is written up as "human bridge links do not help" after the money is
spent. The measurement would never have contained a bridge link.

The machinery was already correct. The ML Engineer perspective pushed 25 real
NIST 800-53 controls through `bridge_training_records` ->
`filter_training_links` and got 25 kept, all tiered T2. It was simply not
connected to anything.

These tests connect it and keep it connected. The load-bearing one is
`test_a_bridge_file_changes_what_training_receives` -- it goes through the real
`load_and_filter_curated_links`, the function every training entry point calls,
rather than through the bridge helpers directly. A test that calls the helpers
stays green while the pipeline ignores them, which is exactly what happened.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from tract.bridge.links import BridgeLink
from tract.training.data_quality import (
    QualityTier,
    curated_link_filter_report,
    fold_input_digests,
    load_and_filter_curated_links,
)


def _digests(**kwargs: object) -> dict[str, str | None]:
    """fold_input_digests with the three required flags supplied."""
    return fold_input_digests(
        with_prose=True,
        with_stopwords=False,
        with_framework_identity=False,
        **kwargs,  # type: ignore[arg-type]
    )


def _real_targets() -> tuple[str, str]:
    """A hub id and a NIST 800-53 control id that resolve to real prose."""
    from tract.config import PROCESSED_DIR

    hierarchy = json.loads(
        (PROCESSED_DIR / "cre_hierarchy.json").read_text(encoding="utf-8")
    )
    controls = json.loads(
        (PROCESSED_DIR / "all_controls.json").read_text(encoding="utf-8")
    )
    framework = next(
        f for f in controls["frameworks"] if f["framework_id"] == "nist_800_53"
    )
    control = next(
        c
        for c in framework["controls"]
        if len((c.get("description") or "").strip()) > 200
    )
    return sorted(hierarchy["hubs"])[0], control["control_id"]


@pytest.fixture()
def bridge_file(tmp_path: Path) -> Path:
    """A small, real Tier-2 corpus: real hub ids, real NIST control ids."""
    from tract.config import PROCESSED_DIR

    hub, _ = _real_targets()
    controls = json.loads(
        (PROCESSED_DIR / "all_controls.json").read_text(encoding="utf-8")
    )
    framework = next(
        f for f in controls["frameworks"] if f["framework_id"] == "nist_800_53"
    )
    usable = [
        c
        for c in framework["controls"]
        if len((c.get("description") or "").strip()) > 200
    ][:5]
    assert len(usable) == 5, "fixture needs five controls with real prose"

    links = [
        BridgeLink(
            framework_id="nist_800_53",
            standard_name="NIST 800-53 v5",
            section_id=c["control_id"],
            section_name=c.get("title", ""),
            cre_id=hub,
            tier=2,
            annotator_id="test-annotator",
            created_at="2026-09-04T12:00:00Z",
            confidence=3,
            rationale="fixture",
        )
        for c in usable
    ]
    path = tmp_path / "hub_links_bridge.jsonl"
    path.write_text(
        "".join(json.dumps(asdict(link), sort_keys=True) + "\n" for link in links),
        encoding="utf-8",
    )
    return path


class TestTrainingConsumesBridgeLinks:
    def test_a_bridge_file_changes_what_training_receives(
        self, bridge_file: Path
    ) -> None:
        """The load-bearing assertion, through the real entry point.

        Not through bridge_training_records -- a test that calls the helpers
        directly is what stayed green while nothing called them.
        """
        without, _ = load_and_filter_curated_links()
        with_bridge, _ = load_and_filter_curated_links(bridge_path=bridge_file)
        assert len(with_bridge) > len(without), (
            "Adding a bridge corpus did not change the training links, so "
            "nothing consumes it and the round cannot affect any model."
        )

    def test_the_added_links_are_tiered_t2(self, bridge_file: Path) -> None:
        without, _ = load_and_filter_curated_links()
        with_bridge, _ = load_and_filter_curated_links(bridge_path=bridge_file)
        added = len(with_bridge) - len(without)
        t2 = [link for link in with_bridge if link.tier is QualityTier.T2]
        assert len(t2) == added == 5

    def test_no_bridge_path_leaves_the_corpus_exactly_as_before(self) -> None:
        """The default must not change for any existing caller."""
        a, hash_a = load_and_filter_curated_links()
        b, hash_b = load_and_filter_curated_links(bridge_path=None)
        assert len(a) == len(b)
        assert hash_a == hash_b
        assert not [link for link in a if link.tier is QualityTier.T2]

    def test_a_missing_bridge_file_raises_rather_than_training_without_it(
        self, tmp_path: Path
    ) -> None:
        """Silently training bridge-free would report a null as a result."""
        with pytest.raises(FileNotFoundError):
            load_and_filter_curated_links(bridge_path=tmp_path / "absent.jsonl")

    def test_bridge_links_still_face_the_anchor_gate(
        self, tmp_path: Path
    ) -> None:
        """A Tier-2 tag is not an exemption from the resolvable-anchor floor."""
        hub, _ = _real_targets()
        link = BridgeLink(
            framework_id="nist_800_53",
            standard_name="NIST 800-53 v5",
            section_id="NOT-A-REAL-CONTROL-ID",
            section_name="x",
            cre_id=hub,
            tier=2,
            annotator_id="a",
            created_at="2026-09-04T12:00:00Z",
            confidence=3,
            rationale="r",
        )
        path = tmp_path / "b.jsonl"
        path.write_text(json.dumps(asdict(link), sort_keys=True) + "\n", encoding="utf-8")

        without, _ = load_and_filter_curated_links()
        with_bridge, _ = load_and_filter_curated_links(bridge_path=path)
        assert len(with_bridge) == len(without), (
            "A bridge link with no resolvable anchor was kept; the tier tag "
            "must not bypass the gate every other link passes."
        )


class TestTheReportNamesWhatItLoaded:
    def test_the_report_counts_bridge_links_separately(
        self, bridge_file: Path
    ) -> None:
        report, _ = curated_link_filter_report(bridge_path=bridge_file)
        assert report.n_bridge == 5

    def test_it_is_zero_and_not_none_when_no_bridge_corpus_was_read(self) -> None:
        """Zero and "field absent" must stay distinguishable in an artifact."""
        report, _ = curated_link_filter_report()
        assert report.n_bridge == 0


class TestProvenanceRecordsWhichCorpusWasUsed:
    """Two runs with different bridge corpora must be distinguishable.

    Without this, `fold_result.json` files agree on git_sha, config and every
    input digest while disagreeing on the metric, and no artifact can say which
    corpus produced which number.
    """

    def test_the_digest_set_includes_a_bridge_entry(self) -> None:
        assert "bridge_links_sha256" in _digests()

    def test_it_is_none_when_no_bridge_corpus_is_present(self) -> None:
        assert _digests(bridge_path=None)["bridge_links_sha256"] is None

    def test_two_different_corpora_produce_different_digests(
        self, bridge_file: Path, tmp_path: Path
    ) -> None:
        other = tmp_path / "other.jsonl"
        other.write_text(
            bridge_file.read_text(encoding="utf-8").replace(
                '"confidence": 3', '"confidence": 2'
            ),
            encoding="utf-8",
        )
        a = _digests(bridge_path=bridge_file)["bridge_links_sha256"]
        b = _digests(bridge_path=other)["bridge_links_sha256"]
        assert a is not None and b is not None and a != b


class TestTheFlagReachesTheConfig:
    """Plumbing without a CLI flag is plumbing nothing can use.

    `TrainingConfig.bridge_links_path` reached `curated_link_filter_report`,
    `fold_input_digests` and `staleness`, and `run_fold.py` -- which is how
    `runpod_parallel` invokes a fold -- had no argument to set it. So every
    fold ran with `bridge_path=None` and Gate 2's treatment arm was not
    expressible. The wiring commit stopped one argument short.
    """

    def test_run_fold_accepts_bridge_links(self) -> None:
        """Read the source; do not import it.

        scripts/phase1b/run_fold.py imports torch, which is absent from CI's
        requirements. Importing it here made this test the only red in an
        otherwise green suite -- better than the collection error that once
        aborted 3,000 tests, and still avoidable. The property under test is
        that the flag exists and is forwarded, which the source answers.
        """
        from tract.config import PROJECT_ROOT

        source = (
            PROJECT_ROOT / "scripts" / "phase1b" / "run_fold.py"
        ).read_text(encoding="utf-8")
        assert '"--bridge-links"' in source, (
            "run_fold.py exposes no --bridge-links flag, so a fold cannot be "
            "told to use a bridge corpus and Gate 2's treatment arm is not "
            "expressible."
        )
        assert "bridge_links_path" in source, (
            "the flag exists but is not forwarded into TrainingConfig."
        )

    def test_the_config_field_survives_a_round_trip(self) -> None:
        from tract.training.config import TrainingConfig

        config = TrainingConfig(name="t", bridge_links_path="b.jsonl")
        assert config.to_dict()["bridge_links_path"] == "b.jsonl"

    def test_the_default_is_none_so_a_fold_is_bridge_free(self) -> None:
        """The comparator arm must be the default, not something to remember."""
        from tract.training.config import TrainingConfig

        assert TrainingConfig(name="t").bridge_links_path is None
