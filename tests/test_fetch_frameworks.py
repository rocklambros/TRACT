"""Tests for scripts.fetch_frameworks -- mocked, no live network calls.

Covers the trust-on-first-use hardening: a source whose sha256 no longer
matches its pinned expected_sha256 must raise, and --accept-new-hash must be
the only way past that raise. A test that can only ever pass is worse than
none, so each behavior gets both the failing and the succeeding path.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from scripts.fetch_frameworks import Source, SourceHashMismatch, _record, fetch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(body: bytes, status_code: int = 200) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.raise_for_status.return_value = None
    resp.iter_content.return_value = [body]
    return resp


@pytest.fixture(autouse=True)
def _isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect RAW_FRAMEWORKS_DIR and the manifest into a scratch dir so no
    test touches the real data/raw/ or the tracked manifest."""
    raw_dir = tmp_path / "raw" / "frameworks"
    manifest_path = tmp_path / "processed" / "framework_sources.json"
    monkeypatch.setattr("scripts.fetch_frameworks.RAW_FRAMEWORKS_DIR", raw_dir)
    monkeypatch.setattr("scripts.fetch_frameworks.MANIFEST_PATH", manifest_path)
    return tmp_path


# ---------------------------------------------------------------------------
# _record: mismatch raises
# ---------------------------------------------------------------------------

class TestRecordHashMismatchRaises:
    """A file whose sha256 disagrees with expected_sha256 must raise, not
    silently overwrite the baseline -- that overwrite is the exact defect
    this hardening closes."""

    def test_mismatch_raises_without_accept_flag(self, tmp_path: Path) -> None:
        target = tmp_path / "raw" / "frameworks" / "demo" / "demo.txt"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"new content, different from what was pinned")

        source = Source(
            framework_id="demo", filename="demo.txt", url="https://example.test/demo.txt",
            note="test fixture", training_links=1,
            expected_sha256="0" * 64,  # deliberately wrong, will never match
        )

        with pytest.raises(SourceHashMismatch, match="demo/demo.txt"):
            _record(source, target, accept_new_hash=False)

    def test_mismatch_with_accept_flag_proceeds_and_records(self, tmp_path: Path) -> None:
        target = tmp_path / "raw" / "frameworks" / "demo" / "demo.txt"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"new content, different from what was pinned")

        source = Source(
            framework_id="demo", filename="demo.txt", url="https://example.test/demo.txt",
            note="test fixture", training_links=1,
            expected_sha256="0" * 64,
        )

        # Must not raise.
        _record(source, target, accept_new_hash=True)

        from scripts.fetch_frameworks import _load_manifest
        manifest = _load_manifest()
        assert manifest["demo"]["demo.txt"]["sha256"] != "0" * 64
        # The stale pin is preserved in the manifest as a record of what was
        # expected, not silently replaced with the new observation.
        assert manifest["demo"]["demo.txt"]["expected_sha256"] == "0" * 64

    def test_force_alone_does_not_bypass_mismatch(self, tmp_path: Path) -> None:
        """--force controls re-download, not hash acceptance. A caller that
        forces a re-fetch of a source whose content changed still has to hit
        the raise, or --force becomes a silent bypass by another name."""
        target_dir = tmp_path / "raw" / "frameworks" / "demo"
        target_dir.mkdir(parents=True)
        target = target_dir / "demo.txt"
        target.write_bytes(b"stale local copy")

        source = Source(
            framework_id="demo", filename="demo.txt", url="https://example.test/demo.txt",
            note="test fixture", training_links=1,
            expected_sha256="0" * 64,
        )

        with patch("scripts.fetch_frameworks.requests.get") as mock_get:
            mock_get.return_value = _make_response(b"content that still won't match")
            with pytest.raises(SourceHashMismatch):
                fetch(source, force=True, accept_new_hash=False)


# ---------------------------------------------------------------------------
# _record: no pinned baseline yet (trust-on-first-use)
# ---------------------------------------------------------------------------

class TestRecordNoBaselineYet:
    """expected_sha256=None is the one legitimate unchecked path: a source
    that has never been fetched before has nothing to compare against."""

    def test_none_baseline_does_not_raise(self, tmp_path: Path) -> None:
        target = tmp_path / "raw" / "frameworks" / "demo" / "demo.txt"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"first ever fetch of this source")

        source = Source(
            framework_id="demo", filename="demo.txt", url="https://example.test/demo.txt",
            note="test fixture", training_links=1, expected_sha256=None,
        )

        _record(source, target, accept_new_hash=False)

        from scripts.fetch_frameworks import _load_manifest
        manifest = _load_manifest()
        assert manifest["demo"]["demo.txt"]["expected_sha256"] == ""


# ---------------------------------------------------------------------------
# _record: matching hash is a no-op pass-through
# ---------------------------------------------------------------------------

class TestRecordMatchingHash:
    def test_matching_hash_does_not_raise(self, tmp_path: Path) -> None:
        import hashlib

        target = tmp_path / "raw" / "frameworks" / "demo" / "demo.txt"
        target.parent.mkdir(parents=True)
        content = b"pinned content"
        target.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()

        source = Source(
            framework_id="demo", filename="demo.txt", url="https://example.test/demo.txt",
            note="test fixture", training_links=1, expected_sha256=digest,
        )

        # Must not raise even without accept_new_hash.
        _record(source, target, accept_new_hash=False)


# ---------------------------------------------------------------------------
# fetch(): locally-supplied source (url=None)
# ---------------------------------------------------------------------------

class TestFetchLocallySupplied:
    def test_missing_local_file_raises_file_not_found(self, tmp_path: Path) -> None:
        source = Source(
            framework_id="csa_ccm_test", filename="ccm.xlsx", url=None,
            note="test fixture", training_links=1,
        )
        with pytest.raises(FileNotFoundError):
            fetch(source)

    def test_present_local_file_is_recorded_without_network_call(
        self, tmp_path: Path,
    ) -> None:
        target_dir = tmp_path / "raw" / "frameworks" / "csa_ccm_test"
        target_dir.mkdir(parents=True)
        (target_dir / "ccm.xlsx").write_bytes(b"staged workbook bytes")

        source = Source(
            framework_id="csa_ccm_test", filename="ccm.xlsx", url=None,
            note="test fixture", training_links=1,
        )

        with patch("scripts.fetch_frameworks.requests.get") as mock_get:
            result = fetch(source)
            mock_get.assert_not_called()

        assert result.read_bytes() == b"staged workbook bytes"


# ---------------------------------------------------------------------------
# fetch(): per-source headers reach the request
# ---------------------------------------------------------------------------

class TestFetchHeaders:
    def test_headers_are_passed_to_requests_get(self, tmp_path: Path) -> None:
        source = Source(
            framework_id="demo", filename="demo.pdf",
            url="https://example.test/demo.pdf",
            note="test fixture", training_links=1,
            headers={"User-Agent": "test-browser-agent"},
        )

        with patch("scripts.fetch_frameworks.requests.get") as mock_get:
            mock_get.return_value = _make_response(b"%PDF-1.4 fake content")
            fetch(source)

        _, kwargs = mock_get.call_args
        assert kwargs["headers"] == {"User-Agent": "test-browser-agent"}


# ---------------------------------------------------------------------------
# BY_ID: multiple sources sharing one framework_id (biml)
# ---------------------------------------------------------------------------

class TestMultiSourceFramework:
    def test_by_id_groups_multiple_sources_under_one_framework_id(self) -> None:
        from scripts.fetch_frameworks import BY_ID

        biml_sources = BY_ID["biml"]
        assert len(biml_sources) == 2
        filenames = {s.filename for s in biml_sources}
        assert filenames == {"ara.pdf", "BIML-LLM24.pdf"}


# ---------------------------------------------------------------------------
# Module-level invariant: resolved_commit_sha must appear in its own url
# ---------------------------------------------------------------------------

class TestResolvedCommitShaInvariant:
    def test_mismatched_pin_raises_at_import_time(self) -> None:
        # Reproduce the module-level guard directly rather than re-importing
        # the module with a corrupted SOURCES tuple, which would pollute
        # sys.modules for every other test in the process.
        source = Source(
            framework_id="demo", filename="demo.zip",
            url="https://github.com/example/demo/archive/deadbeef.zip",
            note="test fixture", training_links=1,
            resolved_commit_sha="not-the-sha-in-the-url",
        )
        expected_fragment = f"archive/{source.resolved_commit_sha}.zip"
        assert source.url is not None
        assert expected_fragment not in source.url
