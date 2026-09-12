"""The bridge digest must be checked against the corpus the run actually read.

`tracked_inputs()` mapped `bridge_links_sha256` to a module constant,
`data/training/hub_links_bridge.jsonl`. That file does not exist, `_digest`
returns None for a missing path, and `check_result` skips the comparison when
current is None -- so the bridge staleness check was a no-op. It is the one
instrument that distinguishes two Gate 2 arms which agree on every other digest
and disagree on the metric.

Staging each arm's corpus AT the constant path does not fix it either: running
A0 then A1 leaves the file holding the last arm's bytes, and aggregating the
first arm afterwards trips the stale check on the honest run, whose only escape
is `--allow-stale` -- which this repository documents as "cannot be quoted as a
current measurement."

So the path is resolved from the fold record's own `config.bridge_links_path`,
and an absent file with a recorded digest is STALE rather than fresh.
"""

from __future__ import annotations

import json
from pathlib import Path

from tract.staleness import check_result, tracked_inputs


def _fold(
    tmp_path: Path,
    *,
    bridge_path: str | None,
    bridge_digest: str | None,
    name: str = "fold_result.json",
) -> Path:
    payload = {
        "config": {"name": "arm", "bridge_links_path": bridge_path},
        "inputs": {"bridge_links_sha256": bridge_digest},
    }
    out = tmp_path / name
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


class TestThePathComesFromTheRecord:
    def test_resolves_the_corpus_the_run_read(self, tmp_path: Path) -> None:
        corpus = tmp_path / "hub_links_bridge.r2.jsonl"
        corpus.write_text('{"a": 1}\n', encoding="utf-8")
        paths = tracked_inputs(
            {"config": {"bridge_links_path": str(corpus)}}
        )
        assert paths["bridge_links_sha256"] == corpus

    def test_two_arms_resolve_to_two_different_files(
        self, tmp_path: Path
    ) -> None:
        """The whole point: A0's record and A1's record must not share a path."""
        a1 = tracked_inputs(
            {"config": {"bridge_links_path": str(tmp_path / "r1.jsonl")}}
        )
        a2 = tracked_inputs(
            {"config": {"bridge_links_path": str(tmp_path / "r2.jsonl")}}
        )
        assert a1["bridge_links_sha256"] != a2["bridge_links_sha256"]

    def test_no_payload_falls_back_to_the_constant(self) -> None:
        """Callers that only want field NAMES keep working."""
        assert "bridge_links_sha256" in tracked_inputs()


class TestAMissingCorpusIsStaleNotFresh:
    def test_recorded_digest_with_absent_file_is_stale(
        self, tmp_path: Path
    ) -> None:
        result = _fold(
            tmp_path,
            bridge_path=str(tmp_path / "gone.jsonl"),
            bridge_digest="a" * 64,
        )
        status = check_result(result)
        assert status.is_stale
        fields = {s.field for s in status.stale}
        assert "bridge_links_sha256" in fields

    def test_the_stale_entry_says_the_file_is_missing(
        self, tmp_path: Path
    ) -> None:
        """An operator reading this must not think the bytes merely changed."""
        result = _fold(
            tmp_path,
            bridge_path=str(tmp_path / "gone.jsonl"),
            bridge_digest="a" * 64,
        )
        entry = next(
            s for s in check_result(result).stale
            if s.field == "bridge_links_sha256"
        )
        assert entry.current == "<absent>"

    def test_matching_digest_is_not_stale(self, tmp_path: Path) -> None:
        import hashlib

        corpus = tmp_path / "corpus.jsonl"
        body = b'{"cre_id": "010-108"}\n'
        corpus.write_bytes(body)
        result = _fold(
            tmp_path,
            bridge_path=str(corpus),
            bridge_digest=hashlib.sha256(body).hexdigest(),
        )
        status = check_result(result)
        assert "bridge_links_sha256" not in {s.field for s in status.stale}

    def test_changed_bytes_are_stale(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_bytes(b"changed\n")
        result = _fold(
            tmp_path, bridge_path=str(corpus), bridge_digest="b" * 64
        )
        assert check_result(result).is_stale


class TestABridgeFreeArmIsNotUnrecorded:
    """A0 deliberately used no corpus. That is a fact, not a missing field."""

    def test_null_path_and_null_digest_agree(self, tmp_path: Path) -> None:
        result = _fold(tmp_path, bridge_path=None, bridge_digest=None)
        status = check_result(result)
        assert not status.is_stale
        assert "bridge_links_sha256" not in status.unrecorded, (
            "A bridge-free arm records None deliberately. Counting it as "
            "unrecorded conflates 'this run used no corpus' with 'this record "
            "predates the field', and is_checkable then understates coverage."
        )

    def test_a_record_with_no_config_still_reports_unrecorded(
        self, tmp_path: Path
    ) -> None:
        """An old record that predates the field must stay distinguishable."""
        out = tmp_path / "old.json"
        out.write_text(json.dumps({"inputs": {}}), encoding="utf-8")
        assert "bridge_links_sha256" in check_result(out).unrecorded

    def test_a_path_without_a_digest_is_unrecorded(self, tmp_path: Path) -> None:
        """Trained WITH a corpus but recorded no digest: the dangerous case."""
        result = _fold(
            tmp_path, bridge_path=str(tmp_path / "c.jsonl"), bridge_digest=None
        )
        assert "bridge_links_sha256" in check_result(result).unrecorded
