"""`tract export` accepted four filter flags and ignored all of them.

`_cmd_export` called `export_crosswalk(db, output_path, fmt=fmt)`, whose
signature takes no filters at all. `--hub`, `--min-confidence` and `--status`
were read nowhere in the package, and `--framework` was honoured only on the
`--opencre` branch.

Measured before the fix: `tract export --format csv --framework mitre_atlas`
exited 0 and wrote **636 rows across 6 frameworks**, only 260 of them ATLAS. Two
runs with contradictory filters produced byte-identical output.

That is worse than a crash. A crash sends the user to the docs; silently wrong
output gets used. `--framework` is the CLI epilog's own worked example.

The `--status` default also mattered: the JSON path hardcoded
`review_status = 'accepted'` while the CSV path applied no status filter, so the
same flag meant different things depending on `--format`.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pytest

from tract.config import PHASE1C_CROSSWALK_DB_PATH
from tract.crosswalk.export import export_crosswalk


@pytest.fixture(scope="module")
def db() -> Path:
    if not PHASE1C_CROSSWALK_DB_PATH.is_file():
        pytest.skip(f"{PHASE1C_CROSSWALK_DB_PATH} absent")
    return PHASE1C_CROSSWALK_DB_PATH


@pytest.fixture(scope="module")
def a_framework(db: Path) -> str:
    """A framework id that is present but is not the only one."""
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT f.id, COUNT(*) FROM assignments a "
            "JOIN controls c ON a.control_id = c.id "
            "JOIN frameworks f ON c.framework_id = f.id "
            "GROUP BY f.id ORDER BY 2 DESC"
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) > 1, "one framework only; a filter test would be vacuous"
    return str(rows[0][0])


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestTheFilterActuallyFilters:
    def test_framework_filter_narrows_the_csv(
        self, db: Path, a_framework: str, tmp_path: Path
    ) -> None:
        """The load-bearing one. Unfiltered this returned every framework."""
        unfiltered = export_crosswalk(db, tmp_path / "all.csv", fmt="csv")
        filtered = export_crosswalk(
            db, tmp_path / "one.csv", fmt="csv", framework=a_framework
        )
        all_rows = _csv_rows(unfiltered)
        one_rows = _csv_rows(filtered)

        assert len(all_rows) > len(one_rows) > 0, (
            "the filter removed nothing, so it is being ignored"
        )
        assert len({row["framework"] for row in one_rows}) == 1

    def test_min_confidence_narrows_the_csv(
        self, db: Path, tmp_path: Path
    ) -> None:
        low = _csv_rows(
            export_crosswalk(db, tmp_path / "lo.csv", fmt="csv", min_confidence=0.0)
        )
        high = _csv_rows(
            export_crosswalk(db, tmp_path / "hi.csv", fmt="csv", min_confidence=0.95)
        )
        assert len(low) > len(high), "min_confidence removed nothing"
        for row in high:
            if row["confidence"]:
                assert float(row["confidence"]) >= 0.95

    def test_status_filter_narrows_the_csv(self, db: Path, tmp_path: Path) -> None:
        every = _csv_rows(
            export_crosswalk(db, tmp_path / "any.csv", fmt="csv", status="all")
        )
        accepted = _csv_rows(
            export_crosswalk(db, tmp_path / "acc.csv", fmt="csv", status="accepted")
        )
        assert len(every) >= len(accepted) > 0
        assert {row["review_status"] for row in accepted} == {"accepted"}

    def test_hub_filter_narrows_the_csv(self, db: Path, tmp_path: Path) -> None:
        rows = _csv_rows(export_crosswalk(db, tmp_path / "a.csv", fmt="csv"))
        hub = rows[0]["hub_id"]
        only = _csv_rows(
            export_crosswalk(db, tmp_path / "h.csv", fmt="csv", hub=hub)
        )
        assert 0 < len(only) < len(rows)
        assert {row["hub_id"] for row in only} == {hub}

    def test_two_different_filters_do_not_produce_identical_output(
        self, db: Path, a_framework: str, tmp_path: Path
    ) -> None:
        """The symptom as a user would meet it.

        Before the fix, contradictory filters produced byte-identical files.
        """
        one = export_crosswalk(
            db, tmp_path / "1.csv", fmt="csv", framework=a_framework
        )
        two = export_crosswalk(db, tmp_path / "2.csv", fmt="csv", min_confidence=0.99)
        assert one.read_bytes() != two.read_bytes()


class TestTheJsonPathFiltersToo:
    def test_framework_filter_narrows_the_json(
        self, db: Path, a_framework: str, tmp_path: Path
    ) -> None:
        every = json.loads(
            export_crosswalk(db, tmp_path / "a.json", fmt="json").read_text(
                encoding="utf-8"
            )
        )
        one = json.loads(
            export_crosswalk(
                db, tmp_path / "b.json", fmt="json", framework=a_framework
            ).read_text(encoding="utf-8")
        )
        assert len(one) == 1
        assert len(every) > 1

    def test_status_means_the_same_thing_in_both_formats(
        self, db: Path, tmp_path: Path
    ) -> None:
        """The JSON path hardcoded 'accepted'; the CSV path filtered nothing.

        So `--status` meant different things depending on `--format`, and
        neither honoured what the user asked for.
        """
        json_rows = json.loads(
            export_crosswalk(
                db, tmp_path / "a.json", fmt="json", status="accepted"
            ).read_text(encoding="utf-8")
        )
        csv_rows = _csv_rows(
            export_crosswalk(db, tmp_path / "a.csv", fmt="csv", status="accepted")
        )
        json_pairs = {
            (control, entry["hub_id"])
            for framework in json_rows.values()
            for control, entries in framework.items()
            for entry in entries
        }
        csv_pairs = {(row["control_id"], row["hub_id"]) for row in csv_rows}
        assert json_pairs == csv_pairs


class TestNothingIsSilentlyIgnored:
    def test_every_documented_filter_is_a_parameter(self) -> None:
        """A flag the CLI accepts and the exporter cannot see is the defect."""
        import inspect

        params = set(inspect.signature(export_crosswalk).parameters)
        for name in ("framework", "hub", "min_confidence", "status"):
            assert name in params, (
                f"`tract export --{name.replace('_', '-')}` is accepted by the "
                "parser but export_crosswalk has no such parameter, so it is "
                "silently ignored."
            )

    def test_the_cli_forwards_all_of_them(self) -> None:
        import inspect

        from tract import cli

        source = inspect.getsource(cli._cmd_export)
        for name in ("framework", "hub", "min_confidence", "status"):
            assert name in source, (
                f"_cmd_export never mentions {name}, so the flag is parsed and "
                "dropped."
            )
