"""Staleness must stay detectable, whatever the results happen to say today.

This project has published one figure that did not survive audit. The defence
is not that results never go stale, which they must after a corpus rebuild, but
that a reader can always tell. So the assertions here are about the RECORDING,
not about the current verdict: a suite that failed on staleness would be red for
the whole of any rebuild and would be silenced rather than heeded.
"""
from __future__ import annotations

import json
from pathlib import Path

from tract.staleness import (
    TRACKED_INPUTS,
    check_result,
    describe,
    scan,
)


def _result(path: Path, inputs: dict[str, str] | None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body: dict[str, object] = {"fold": "DEMO", "hit1": 0.5}
    if inputs is not None:
        body["inputs"] = inputs
    path.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
    return path


class TestEveryRecordedResultCanBeChecked:
    def test_every_fold_result_records_at_least_one_input_digest(self) -> None:
        """An unrecorded digest is worse than a stale one.

        A stale digest says the number is old. A missing digest says nothing,
        and cannot be told apart from a number that is current.
        """
        uncheckable = [s.result_path for s in scan() if not s.is_checkable]
        assert not uncheckable, (
            f"{uncheckable} record none of {sorted(TRACKED_INPUTS)}, so nobody "
            f"can tell whether the figures in them still describe their inputs."
        )

    def test_the_scan_finds_the_results_that_exist(self) -> None:
        """A scan that silently found nothing would pass the test above."""
        assert len(scan()) > 0


class TestTheCheckItself:
    def test_a_matching_digest_is_not_stale(self, tmp_path: Path) -> None:
        import hashlib

        digests = {
            field: hashlib.sha256(path.read_bytes()).hexdigest()
            for field, path in TRACKED_INPUTS.items()
            if path.exists()
        }
        status = check_result(_result(tmp_path / "fold_result.json", digests))
        assert not status.is_stale
        assert status.is_checkable

    def test_a_moved_input_is_reported_with_both_digests(
        self, tmp_path: Path
    ) -> None:
        wrong = {field: "0" * 64 for field in TRACKED_INPUTS}
        status = check_result(_result(tmp_path / "fold_result.json", wrong))
        assert status.is_stale
        assert len(status.stale) == sum(1 for p in TRACKED_INPUTS.values() if p.exists())
        for item in status.stale:
            assert item.recorded == "0" * 64
            assert item.current != item.recorded

    def test_a_result_with_no_inputs_block_is_uncheckable(
        self, tmp_path: Path
    ) -> None:
        status = check_result(_result(tmp_path / "fold_result.json", None))
        assert not status.is_checkable
        assert not status.is_stale

    def test_a_partially_recorded_result_is_still_checkable(
        self, tmp_path: Path
    ) -> None:
        """One digest is enough to catch a corpus that moved under a rerun."""
        status = check_result(
            _result(tmp_path / "fold_result.json", {"stopwords_sha256": "0" * 64})
        )
        assert status.is_checkable
        assert status.is_stale


class TestTheReportSaysWhatMoved:
    def test_it_names_the_file_and_both_digests(self, tmp_path: Path) -> None:
        status = check_result(
            _result(tmp_path / "fold_result.json", {"stopwords_sha256": "0" * 64})
        )
        text = describe([status])
        assert "stopwords_sha256" in text
        assert "data/processed/stopwords.json" in text
        assert "may not be quoted as a current measurement" in text

    def test_an_empty_scan_says_so_rather_than_reporting_success(self) -> None:
        assert describe([]) == "no fold results found"

    def test_a_clean_report_does_not_carry_the_quoting_warning(
        self, tmp_path: Path
    ) -> None:
        """The warning must mean something when it appears."""
        import hashlib

        digests = {
            field: hashlib.sha256(path.read_bytes()).hexdigest()
            for field, path in TRACKED_INPUTS.items()
            if path.exists()
        }
        text = describe([check_result(_result(tmp_path / "f.json", digests))])
        assert "may not be quoted" not in text
