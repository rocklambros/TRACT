"""Tests for ground truth import — section ID resolver strategies."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from tract.crosswalk.ground_truth import (
    ResolverResult,
    _backup_database,
    build_control_lookups,
    import_ground_truth,
    resolve_framework_links,
    resolve_section_id,
)
from tract.crosswalk.schema import create_database, get_connection

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def resolver_db(tmp_path: Path) -> Path:
    """Create a temporary DB with synthetic controls for resolver testing."""
    db_path = tmp_path / "resolver_test.db"
    create_database(db_path)
    conn = get_connection(db_path)
    try:
        # Insert frameworks
        for fw_id, fw_name in [
            ("mitre_atlas", "MITRE ATLAS"),
            ("asvs", "ASVS"),
            ("nist_800_53", "NIST 800-53"),
            ("nist_800_63", "NIST 800-63"),
            ("owasp_ai_exchange", "OWASP AI Exchange"),
            ("test_fw", "Test Framework"),
        ]:
            conn.execute(
                "INSERT INTO frameworks (id, name) VALUES (?, ?)",
                (fw_id, fw_name),
            )

        # Insert controls for each strategy test:
        controls = [
            # Strategy 1 (direct): section_id directly matches GT section_id
            ("mitre_atlas:AML.M0008", "mitre_atlas", "AML.M0008",
             "Adversarial ML Mitigation", None, None),

            # Strategy 2 (prefixed): GT section_id "V1.1.1" → prefixed "asvs:V1.1.1" matches
            ("asvs:asvs:V1.1.1", "asvs", "asvs:V1.1.1",
             "Verify secure SDLC", None, None),

            # Strategy 3 (title_exact): GT section_id matches title exactly
            ("nist_800_53:cm-2", "nist_800_53", "nist_800_53:cm-2-baseline-configuration",
             "CM-2 BASELINE CONFIGURATION", None, None),

            # Strategy 4 (title_case_insensitive): GT section_id matches title case-insensitively
            # Title is mixed case; GT will supply all-uppercase to trigger case-insensitive
            ("nist_800_63:5.1.1.2", "nist_800_63", "nist_800_63:5.1.1.2-memorized-secrets",
             "Memorized Secrets Verifier", None, None),

            # Extra nist_800_63 control for fixture resolution via prefixed strategy
            ("nist_800_63:nist_800_63:5.1.1.2", "nist_800_63", "nist_800_63:5.1.1.2",
             "Section 5.1.1.2", None, None),

            # Strategy 5 (normalized): GT section_id normalizes to same key as DB section_id
            # GT "aiprogram" → normalized "aiprogram", DB section_id "ai-program" → normalized "aiprogram"
            ("owasp_ai_exchange:ai-program", "owasp_ai_exchange", "ai-program",
             "AI Program Management", None, None),

            # Extra control for priority testing — has both section_id and title that could match
            ("test_fw:priority-ctrl", "test_fw", "EXACT-MATCH",
             "EXACT-MATCH", None, None),
        ]
        for ctrl in controls:
            conn.execute(
                "INSERT INTO controls (id, framework_id, section_id, title, description, full_text) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ctrl,
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


class TestBuildControlLookups:
    def test_builds_section_id_map(self, resolver_db: Path) -> None:
        conn = get_connection(resolver_db)
        try:
            sid_map, _, _ = build_control_lookups(conn, "mitre_atlas")
            assert "AML.M0008" in sid_map
            assert sid_map["AML.M0008"] == "mitre_atlas:AML.M0008"
        finally:
            conn.close()

    def test_builds_title_map_with_case_variants(self, resolver_db: Path) -> None:
        conn = get_connection(resolver_db)
        try:
            _, title_map, _ = build_control_lookups(conn, "nist_800_53")
            assert "CM-2 BASELINE CONFIGURATION" in title_map
            assert "cm-2 baseline configuration" in title_map
        finally:
            conn.close()

    def test_builds_normalized_map(self, resolver_db: Path) -> None:
        conn = get_connection(resolver_db)
        try:
            _, _, norm_map = build_control_lookups(conn, "owasp_ai_exchange")
            assert "aiprogram" in norm_map
            assert norm_map["aiprogram"] == "owasp_ai_exchange:ai-program"
        finally:
            conn.close()

    def test_empty_framework_returns_empty_maps(self, resolver_db: Path) -> None:
        conn = get_connection(resolver_db)
        try:
            sid_map, title_map, norm_map = build_control_lookups(conn, "nonexistent")
            assert sid_map == {}
            assert title_map == {}
            assert norm_map == {}
        finally:
            conn.close()


class TestResolveStrategyDirect:
    def test_direct_match(self, resolver_db: Path) -> None:
        conn = get_connection(resolver_db)
        try:
            sid_map, title_map, norm_map = build_control_lookups(conn, "mitre_atlas")
            ctrl_id, strategy = resolve_section_id(
                "AML.M0008", "mitre_atlas", sid_map, title_map, norm_map,
            )
            assert ctrl_id == "mitre_atlas:AML.M0008"
            assert strategy == "direct"
        finally:
            conn.close()


class TestResolveStrategyPrefixed:
    def test_prefixed_match(self, resolver_db: Path) -> None:
        conn = get_connection(resolver_db)
        try:
            sid_map, title_map, norm_map = build_control_lookups(conn, "asvs")
            ctrl_id, strategy = resolve_section_id(
                "V1.1.1", "asvs", sid_map, title_map, norm_map,
            )
            assert ctrl_id == "asvs:asvs:V1.1.1"
            assert strategy == "prefixed"
        finally:
            conn.close()


class TestResolveStrategyTitleExact:
    def test_title_exact_match(self, resolver_db: Path) -> None:
        conn = get_connection(resolver_db)
        try:
            sid_map, title_map, norm_map = build_control_lookups(conn, "nist_800_53")
            ctrl_id, strategy = resolve_section_id(
                "CM-2 BASELINE CONFIGURATION", "nist_800_53",
                sid_map, title_map, norm_map,
            )
            assert ctrl_id == "nist_800_53:cm-2"
            assert strategy == "title_exact"
        finally:
            conn.close()


class TestResolveStrategyTitleCaseInsensitive:
    def test_title_case_insensitive_match(self, resolver_db: Path) -> None:
        """GT "MEMORIZED SECRETS VERIFIER" doesn't match title "Memorized Secrets Verifier"
        exactly, but matches via .lower() → "memorized secrets verifier"."""
        conn = get_connection(resolver_db)
        try:
            sid_map, title_map, norm_map = build_control_lookups(conn, "nist_800_63")
            ctrl_id, strategy = resolve_section_id(
                "MEMORIZED SECRETS VERIFIER", "nist_800_63",
                sid_map, title_map, norm_map,
            )
            assert ctrl_id == "nist_800_63:5.1.1.2"
            assert strategy == "title_case_insensitive"
        finally:
            conn.close()


class TestResolveStrategyNormalized:
    def test_normalized_match(self, resolver_db: Path) -> None:
        conn = get_connection(resolver_db)
        try:
            sid_map, title_map, norm_map = build_control_lookups(conn, "owasp_ai_exchange")
            ctrl_id, strategy = resolve_section_id(
                "aiprogram", "owasp_ai_exchange", sid_map, title_map, norm_map,
            )
            assert ctrl_id == "owasp_ai_exchange:ai-program"
            assert strategy == "normalized"
        finally:
            conn.close()


class TestResolveUnresolvable:
    def test_unresolvable_returns_none(self, resolver_db: Path) -> None:
        conn = get_connection(resolver_db)
        try:
            sid_map, title_map, norm_map = build_control_lookups(conn, "mitre_atlas")
            ctrl_id, strategy = resolve_section_id(
                "NONEXISTENT-999", "mitre_atlas", sid_map, title_map, norm_map,
            )
            assert ctrl_id is None
            assert strategy == "unresolved"
        finally:
            conn.close()


class TestResolveStrategyPriority:
    def test_direct_beats_title(self, resolver_db: Path) -> None:
        """When section_id AND title both match, direct (strategy 1) wins."""
        conn = get_connection(resolver_db)
        try:
            sid_map, title_map, norm_map = build_control_lookups(conn, "test_fw")
            ctrl_id, strategy = resolve_section_id(
                "EXACT-MATCH", "test_fw", sid_map, title_map, norm_map,
            )
            assert ctrl_id == "test_fw:priority-ctrl"
            assert strategy == "direct"
        finally:
            conn.close()


class TestResolveFrameworkLinks:
    def test_resolves_all_links(self, resolver_db: Path) -> None:
        conn = get_connection(resolver_db)
        try:
            links = [
                {"section_id": "AML.M0008", "cre_id": "364-516", "link_type": "LinkedTo"},
            ]
            result = resolve_framework_links(conn, "mitre_atlas", links)
            assert isinstance(result, ResolverResult)
            assert len(result.resolved) == 1
            assert len(result.unresolved) == 0
            assert result.resolved["mitre_atlas:AML.M0008"] == "mitre_atlas:AML.M0008"
            assert result.strategy_counts["direct"] == 1
        finally:
            conn.close()

    def test_tracks_unresolved(self, resolver_db: Path) -> None:
        conn = get_connection(resolver_db)
        try:
            links = [
                {"section_id": "DOES-NOT-EXIST", "cre_id": "999-999", "link_type": "LinkedTo"},
            ]
            result = resolve_framework_links(conn, "mitre_atlas", links)
            assert len(result.resolved) == 0
            assert len(result.unresolved) == 1
            assert result.unresolved[0]["section_id"] == "DOES-NOT-EXIST"
        finally:
            conn.close()

    def test_mixed_resolved_and_unresolved(self, resolver_db: Path) -> None:
        conn = get_connection(resolver_db)
        try:
            links = [
                {"section_id": "AML.M0008", "cre_id": "364-516", "link_type": "LinkedTo"},
                {"section_id": "NONEXISTENT", "cre_id": "000-000", "link_type": "LinkedTo"},
            ]
            result = resolve_framework_links(conn, "mitre_atlas", links)
            assert len(result.resolved) == 1
            assert len(result.unresolved) == 1
        finally:
            conn.close()

    def test_strategy_counts_accumulated(self, resolver_db: Path) -> None:
        """Insert multiple controls that resolve via the same strategy."""
        conn = get_connection(resolver_db)
        try:
            conn.execute(
                "INSERT INTO controls (id, framework_id, section_id, title) "
                "VALUES (?, ?, ?, ?)",
                ("mitre_atlas:AML.M0009", "mitre_atlas", "AML.M0009", "Another Mitigation"),
            )
            conn.commit()
            links = [
                {"section_id": "AML.M0008", "cre_id": "364-516", "link_type": "LinkedTo"},
                {"section_id": "AML.M0009", "cre_id": "547-824", "link_type": "LinkedTo"},
            ]
            result = resolve_framework_links(conn, "mitre_atlas", links)
            assert result.strategy_counts.get("direct", 0) == 2
        finally:
            conn.close()

    def test_with_mini_fixture(self, resolver_db: Path) -> None:
        fixture_path = FIXTURES_DIR / "phase3_mini_hub_links.json"
        with open(fixture_path, encoding="utf-8") as f:
            hub_links = json.load(f)

        conn = get_connection(resolver_db)
        try:
            total_resolved = 0
            total_unresolved = 0
            for fw_id, links in hub_links.items():
                result = resolve_framework_links(conn, fw_id, links)
                total_resolved += len(result.resolved)
                total_unresolved += len(result.unresolved)

            assert total_resolved == 5
            assert total_unresolved == 0
        finally:
            conn.close()


class TestResolveEdgeCases:
    def test_empty_section_id_normalized_skipped(self) -> None:
        """Empty string after normalization should not match."""
        ctrl_id, strategy = resolve_section_id(
            "---", "test_fw", {}, {}, {},
        )
        assert ctrl_id is None
        assert strategy == "unresolved"

    def test_empty_links_list(self, resolver_db: Path) -> None:
        conn = get_connection(resolver_db)
        try:
            result = resolve_framework_links(conn, "mitre_atlas", [])
            assert result.resolved == {}
            assert result.unresolved == []
            assert result.strategy_counts == {}
        finally:
            conn.close()


# ── Import tests ──────────────────────────────────────────────────────────


@pytest.fixture()
def import_db(tmp_path: Path) -> Path:
    """Create a temporary DB with frameworks, controls, and hubs for import testing."""
    db_path = tmp_path / "import_test.db"
    create_database(db_path)
    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO frameworks (id, name, version, fetch_date, control_count) "
            "VALUES (?, ?, ?, ?, ?)",
            ("test_fw", "Test Framework", "1.0", "2024-01-01", 3),
        )
        conn.execute(
            "INSERT INTO frameworks (id, name, version, fetch_date, control_count) "
            "VALUES (?, ?, ?, ?, ?)",
            ("other_fw", "Other Framework", "1.0", "2024-01-01", 1),
        )
        controls = [
            ("test_fw:ctrl-1", "test_fw", "CTRL-1", "Control One", "desc1", "full1"),
            ("test_fw:ctrl-2", "test_fw", "CTRL-2", "Control Two", "desc2", "full2"),
            ("test_fw:ctrl-3", "test_fw", "CTRL-3", "Control Three", "desc3", "full3"),
            ("other_fw:ctrl-a", "other_fw", "CTRL-A", "Control A", "descA", "fullA"),
        ]
        for ctrl in controls:
            conn.execute(
                "INSERT INTO controls (id, framework_id, section_id, title, description, full_text) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ctrl,
            )
        hubs = [
            ("100-200", "Hub Alpha", "/security/alpha", None),
            ("200-300", "Hub Beta", "/security/beta", None),
            ("300-400", "Hub Gamma", "/security/gamma", None),
            ("400-500", "Hub Delta", "/security/delta", None),
        ]
        for hub in hubs:
            conn.execute(
                "INSERT INTO hubs (id, name, path, parent_id) VALUES (?, ?, ?, ?)",
                hub,
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _write_hub_links(path: Path, data: dict[str, list[dict[str, str]]]) -> None:
    """Write a hub_links JSON file for testing."""
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _make_link(
    section_id: str,
    cre_id: str,
    framework_id: str = "test_fw",
    link_type: str = "LinkedTo",
) -> dict[str, str]:
    return {
        "cre_id": cre_id,
        "cre_name": "Test CRE",
        "framework_id": framework_id,
        "link_type": link_type,
        "section_id": section_id,
        "section_name": section_id,
        "standard_name": "Test Framework",
    }


class TestImportGroundTruth:
    def test_basic_import(self, import_db: Path, tmp_path: Path) -> None:
        links_path = tmp_path / "hub_links.json"
        _write_hub_links(links_path, {
            "test_fw": [
                _make_link("CTRL-1", "100-200"),
                _make_link("CTRL-2", "200-300"),
            ],
        })
        summary = import_ground_truth(import_db, links_path)
        assert summary["imported"] == 2
        assert summary["skipped_duplicate"] == 0
        assert summary["unresolved"] == 0
        assert summary["dry_run"] is False

        conn = get_connection(import_db)
        try:
            count = conn.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]
            assert count == 2
        finally:
            conn.close()

    def test_dedup_skips_existing_assignment(self, import_db: Path, tmp_path: Path) -> None:
        conn = get_connection(import_db)
        try:
            conn.execute(
                "INSERT INTO assignments (control_id, hub_id, confidence, provenance, review_status) "
                "VALUES (?, ?, ?, ?, ?)",
                ("test_fw:ctrl-1", "100-200", 0.9, "active_learning_round_2", "accepted"),
            )
            conn.commit()
        finally:
            conn.close()

        links_path = tmp_path / "hub_links.json"
        _write_hub_links(links_path, {
            "test_fw": [
                _make_link("CTRL-1", "100-200"),
                _make_link("CTRL-2", "200-300"),
            ],
        })
        summary = import_ground_truth(import_db, links_path)
        assert summary["imported"] == 1
        assert summary["skipped_duplicate"] == 1

        conn = get_connection(import_db)
        try:
            count = conn.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]
            assert count == 2  # 1 existing + 1 new
        finally:
            conn.close()

    def test_source_link_id_stores_link_type(self, import_db: Path, tmp_path: Path) -> None:
        links_path = tmp_path / "hub_links.json"
        _write_hub_links(links_path, {
            "test_fw": [
                _make_link("CTRL-1", "100-200", link_type="LinkedTo"),
                _make_link("CTRL-2", "200-300", link_type="AutomaticallyLinkedTo"),
            ],
        })
        import_ground_truth(import_db, links_path)

        conn = get_connection(import_db)
        try:
            rows = conn.execute(
                "SELECT control_id, source_link_id FROM assignments ORDER BY control_id"
            ).fetchall()
            assert rows[0]["source_link_id"] == "LinkedTo"
            assert rows[1]["source_link_id"] == "AutomaticallyLinkedTo"
        finally:
            conn.close()

    def test_provenance_and_review_status(self, import_db: Path, tmp_path: Path) -> None:
        links_path = tmp_path / "hub_links.json"
        _write_hub_links(links_path, {
            "test_fw": [_make_link("CTRL-1", "100-200")],
        })
        import_ground_truth(import_db, links_path)

        conn = get_connection(import_db)
        try:
            row = conn.execute(
                "SELECT provenance, review_status, confidence FROM assignments"
            ).fetchone()
            assert row["provenance"] == "opencre_ground_truth"
            assert row["review_status"] == "ground_truth"
            assert row["confidence"] == pytest.approx(1.0)
        finally:
            conn.close()

    def test_backup_creation(self, import_db: Path, tmp_path: Path) -> None:
        links_path = tmp_path / "hub_links.json"
        _write_hub_links(links_path, {
            "test_fw": [_make_link("CTRL-1", "100-200")],
        })
        import_ground_truth(import_db, links_path)

        backups = list(import_db.parent.glob("*.backup.*"))
        assert len(backups) == 1
        assert backups[0].name.startswith("import_test.backup.")

    def test_dry_run_returns_counts_without_modifying_db(
        self, import_db: Path, tmp_path: Path,
    ) -> None:
        links_path = tmp_path / "hub_links.json"
        _write_hub_links(links_path, {
            "test_fw": [
                _make_link("CTRL-1", "100-200"),
                _make_link("CTRL-2", "200-300"),
            ],
        })
        summary = import_ground_truth(import_db, links_path, dry_run=True)
        assert summary["imported"] == 2
        assert summary["dry_run"] is True

        conn = get_connection(import_db)
        try:
            count = conn.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]
            assert count == 0
        finally:
            conn.close()

        backups = list(import_db.parent.glob("*.backup.*"))
        assert len(backups) == 0

    def test_per_framework_breakdown(self, import_db: Path, tmp_path: Path) -> None:
        links_path = tmp_path / "hub_links.json"
        _write_hub_links(links_path, {
            "test_fw": [
                _make_link("CTRL-1", "100-200"),
                _make_link("CTRL-2", "200-300"),
            ],
            "other_fw": [
                _make_link("CTRL-A", "300-400", framework_id="other_fw"),
            ],
        })
        summary = import_ground_truth(import_db, links_path)
        assert summary["imported"] == 3
        pf = summary["per_framework"]
        assert pf["test_fw"]["imported"] == 2
        assert pf["other_fw"]["imported"] == 1

    def test_unresolved_links_counted(self, import_db: Path, tmp_path: Path) -> None:
        links_path = tmp_path / "hub_links.json"
        _write_hub_links(links_path, {
            "test_fw": [
                _make_link("CTRL-1", "100-200"),
                _make_link("NONEXISTENT", "400-500"),
            ],
        })
        summary = import_ground_truth(import_db, links_path)
        assert summary["imported"] == 1
        assert summary["unresolved"] == 1
        assert summary["per_framework"]["test_fw"]["unresolved"] == 1

    def test_strategy_counts_aggregated(self, import_db: Path, tmp_path: Path) -> None:
        links_path = tmp_path / "hub_links.json"
        _write_hub_links(links_path, {
            "test_fw": [
                _make_link("CTRL-1", "100-200"),
                _make_link("CTRL-2", "200-300"),
            ],
        })
        summary = import_ground_truth(import_db, links_path)
        assert "direct" in summary["strategy_counts"]
        assert summary["strategy_counts"]["direct"] == 2


class TestBackupDatabase:
    def test_backup_creates_timestamped_copy(self, import_db: Path) -> None:
        backup_path = _backup_database(import_db)
        assert backup_path.exists()
        assert "backup" in backup_path.name
        assert backup_path.stat().st_size == import_db.stat().st_size

    def test_backup_preserves_content(self, import_db: Path) -> None:
        conn = get_connection(import_db)
        try:
            conn.execute(
                "INSERT INTO assignments (control_id, hub_id, confidence, provenance, review_status) "
                "VALUES (?, ?, ?, ?, ?)",
                ("test_fw:ctrl-1", "100-200", 0.5, "test", "pending"),
            )
            conn.commit()
        finally:
            conn.close()

        backup_path = _backup_database(import_db)

        backup_conn = sqlite3.connect(str(backup_path))
        try:
            count = backup_conn.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]
            assert count == 1
        finally:
            backup_conn.close()
