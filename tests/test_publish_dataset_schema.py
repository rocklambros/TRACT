"""`tract publish-dataset` could not run against the database it ships.

The second instance of the `publish-hf` failure mode, found by a product-surface
audit written to look for exactly that class.

`_CROSSWALK_QUERY` selects `a.reviewer_notes` and `a.original_hub_id`. The
PUBLISHED, pinned `crosswalk.db` has neither -- its `assignments` table ends at
`created_at`. The local copy is byte-identical to the published artifact
(sha256 matches `TRACT_CROSSWALK_DB_SHA256`), so this is what every
`tract download` delivers.

`SCHEMA_SQL` declares both columns and `migrate_schema()` adds them, but
`migrate_schema` had exactly one caller -- `tract/review/import_review.py`, the
`review-import` path. Nothing on the publish path migrated, so a user who
downloaded the database and ran the documented command got an unhandled
`sqlite3.OperationalError` and a traceback. `--dry-run` and `--skip-upload` do
not help: the query runs before either flag is consulted.

It survived for the same reason `publish-hf` did. `publish-dataset` had parser
tests and no test that reached the real function.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from tract.config import PHASE1C_CROSSWALK_DB_PATH


@pytest.fixture()
def shipped_db(tmp_path: Path) -> Path:
    """A copy of the database as published, un-migrated."""
    src = PHASE1C_CROSSWALK_DB_PATH
    if not src.is_file():
        pytest.skip(f"{src} absent")
    dst = tmp_path / "crosswalk.db"
    shutil.copy(src, dst)
    return dst


def _columns(db_path: Path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {row[1] for row in conn.execute("PRAGMA table_info(assignments)")}
    finally:
        conn.close()


class TestTheShippedDatabaseIsMissingTheColumns:
    """Pin the premise, so the fix is not defended by an assumption."""

    def test_the_published_database_lacks_them(self, shipped_db: Path) -> None:
        cols = _columns(shipped_db)
        assert "reviewer_notes" not in cols
        assert "original_hub_id" not in cols

    def test_the_query_still_selects_them(self) -> None:
        from tract.dataset.bundle import _CROSSWALK_QUERY

        assert "reviewer_notes" in _CROSSWALK_QUERY
        assert "original_hub_id" in _CROSSWALK_QUERY


class TestBundleDatasetRunsAgainstTheShippedDatabase:
    def test_it_does_not_raise_on_a_missing_column(
        self, shipped_db: Path, tmp_path: Path
    ) -> None:
        """The load-bearing test: the real function, the real schema.

        Not a parser test. `publish-dataset` had four of those and they all
        passed while the command could not run.
        """
        from tract.config import PROCESSED_DIR, PROJECT_ROOT
        from tract.dataset.bundle import bundle_dataset

        stats = bundle_dataset(
            db_path=shipped_db,
            staging_dir=tmp_path / "staging",
            hierarchy_path=PROCESSED_DIR / "cre_hierarchy.json",
            hub_descriptions_path=PROCESSED_DIR / "hub_descriptions_reviewed.json",
            bridge_report_path=PROJECT_ROOT / "results" / "bridge" / "bridge_report.json",
            review_metrics_path=(
                PROJECT_ROOT / "results" / "review" / "review_metrics.json"
            ),
        )
        assert stats["total_rows"] > 0

    def test_it_migrates_the_copy_it_was_given(
        self, shipped_db: Path, tmp_path: Path
    ) -> None:
        """Migration is what makes the query legal; assert it actually ran."""
        from tract.config import PROCESSED_DIR, PROJECT_ROOT
        from tract.dataset.bundle import bundle_dataset

        assert "reviewer_notes" not in _columns(shipped_db)
        bundle_dataset(
            db_path=shipped_db,
            staging_dir=tmp_path / "staging",
            hierarchy_path=PROCESSED_DIR / "cre_hierarchy.json",
            hub_descriptions_path=PROCESSED_DIR / "hub_descriptions_reviewed.json",
            bridge_report_path=PROJECT_ROOT / "results" / "bridge" / "bridge_report.json",
            review_metrics_path=(
                PROJECT_ROOT / "results" / "review" / "review_metrics.json"
            ),
        )
        assert "reviewer_notes" in _columns(shipped_db)
        assert "original_hub_id" in _columns(shipped_db)

    def test_a_migrated_database_is_unchanged_by_a_second_run(
        self, shipped_db: Path, tmp_path: Path
    ) -> None:
        """Migration must be idempotent; publishing twice is normal."""
        from tract.crosswalk.schema import migrate_schema

        first = migrate_schema(shipped_db)
        second = migrate_schema(shipped_db)
        assert first, "the first migration should have added columns"
        assert not second, "the second added more, so it is not idempotent"
