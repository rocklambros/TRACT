"""End-to-end test: crosswalk.db → filters → CSV → manifest."""
from __future__ import annotations

import csv
import json
from io import StringIO

import pytest

from tract.crosswalk.schema import create_database
from tract.crosswalk.store import (
    insert_assignments,
    insert_controls,
    insert_frameworks,
    insert_hubs,
)


@pytest.fixture
def e2e_db(tmp_path):
    """Realistic mini-DB with multiple frameworks and mixed assignments."""
    db_path = tmp_path / "e2e.db"
    create_database(db_path)

    insert_frameworks(db_path, [
        {"id": "csa_aicm", "name": "CSA AI Controls Matrix", "version": "1.0",
         "fetch_date": "2026-04-30", "control_count": 3},
        {"id": "mitre_atlas", "name": "MITRE ATLAS", "version": "1.0",
         "fetch_date": "2026-04-30", "control_count": 2},
    ])
    insert_hubs(db_path, [
        {"id": "217-168", "name": "Audit & accountability", "path": "R > Audit", "parent_id": None},
        {"id": "607-671", "name": "Protect against injection", "path": "R > Injection", "parent_id": None},
    ])
    insert_controls(db_path, [
        {"id": "csa_aicm:A&A-01", "framework_id": "csa_aicm", "section_id": "A&A-01",
         "title": "Audit Policy", "description": "Establish audit policies.", "full_text": None},
        {"id": "csa_aicm:A&A-02", "framework_id": "csa_aicm", "section_id": "A&A-02",
         "title": "Independent Assessments", "description": "Conduct assessments.", "full_text": None},
        {"id": "csa_aicm:A&A-03", "framework_id": "csa_aicm", "section_id": "A&A-03",
         "title": "Risk Planning", "description": "Risk-based plans.", "full_text": None},
        {"id": "mitre_atlas:AML.T0000", "framework_id": "mitre_atlas", "section_id": "AML.T0000",
         "title": "Search Databases", "description": "Search technical databases.", "full_text": None},
        {"id": "mitre_atlas:AML.M0015", "framework_id": "mitre_atlas", "section_id": "AML.M0015",
         "title": "Adversarial Input Detection", "description": "Detect adversarial inputs.", "full_text": None},
    ])
    insert_assignments(db_path, [
        {"control_id": "csa_aicm:A&A-01", "hub_id": "217-168", "confidence": 0.60, "in_conformal_set": 1,
         "is_ood": 0, "provenance": "active_learning_round_2", "source_link_id": None,
         "model_version": "v1", "review_status": "accepted"},
        {"control_id": "csa_aicm:A&A-02", "hub_id": "217-168", "confidence": 0.33, "in_conformal_set": 0,
         "is_ood": 0, "provenance": "active_learning_round_2", "source_link_id": None,
         "model_version": "v1", "review_status": "accepted"},
        {"control_id": "csa_aicm:A&A-03", "hub_id": "607-671", "confidence": 0.20, "in_conformal_set": 0,
         "is_ood": 0, "provenance": "active_learning_round_2", "source_link_id": None,
         "model_version": "v1", "review_status": "accepted"},
        {"control_id": "mitre_atlas:AML.T0000", "hub_id": "607-671", "confidence": 0.50, "in_conformal_set": 1,
         "is_ood": 0, "provenance": "active_learning_round_2", "source_link_id": None,
         "model_version": "v1", "review_status": "accepted"},
        {"control_id": "mitre_atlas:AML.M0015", "hub_id": "217-168", "confidence": 0.90, "in_conformal_set": 1,
         "is_ood": 0, "provenance": "ground_truth_T1-AI", "source_link_id": None,
         "model_version": "v1", "review_status": "ground_truth"},
    ])
    return db_path


class TestOpenCREExportE2E:
    def test_full_pipeline_csa(self, e2e_db, tmp_path) -> None:
        from tract.export.filters import query_exportable_assignments
        from tract.export.opencre_csv import generate_opencre_csv

        rows = query_exportable_assignments(
            e2e_db, confidence_floor=0.30, confidence_overrides={},
            framework_filter="csa_aicm",
        )
        assert len(rows) == 2

        csv_text = generate_opencre_csv(rows, "csa_aicm")
        reader = csv.DictReader(StringIO(csv_text))
        csv_rows = list(reader)
        assert len(csv_rows) == 2

        assert "CSA AI Controls Matrix|name" in reader.fieldnames
        assert csv_rows[0]["CRE 0"].startswith("217-168|")

    def test_full_pipeline_atlas_with_override(self, e2e_db, tmp_path) -> None:
        from tract.export.filters import query_exportable_assignments
        from tract.export.opencre_csv import generate_opencre_csv

        rows = query_exportable_assignments(
            e2e_db, confidence_floor=0.30,
            confidence_overrides={"mitre_atlas": 0.35},
            framework_filter="mitre_atlas",
        )
        assert len(rows) == 1
        assert rows[0]["section_id"] == "AML.T0000"

        csv_text = generate_opencre_csv(rows, "mitre_atlas")
        reader = csv.DictReader(StringIO(csv_text))
        csv_rows = list(reader)
        assert len(csv_rows) == 1
        assert "MITRE ATLAS|name" in reader.fieldnames

    def test_full_pipeline_writes_files(self, e2e_db, tmp_path) -> None:
        from tract.export.filters import compute_filter_stats, query_exportable_assignments
        from tract.export.manifest import build_manifest
        from tract.export.opencre_csv import write_opencre_csv
        from tract.io import atomic_write_json

        output_dir = tmp_path / "export"
        all_exported = []

        for fw_id in ["csa_aicm", "mitre_atlas"]:
            rows = query_exportable_assignments(
                e2e_db, confidence_floor=0.30,
                confidence_overrides={"mitre_atlas": 0.35},
                framework_filter=fw_id,
            )
            if rows:
                write_opencre_csv(rows, fw_id, output_dir)
                all_exported.extend(rows)

        assert (output_dir / "CSA_AI_Controls_Matrix.csv").exists()
        assert (output_dir / "MITRE_ATLAS.csv").exists()

        stats = compute_filter_stats(
            e2e_db, all_exported, 0.30, {"mitre_atlas": 0.35},
        )
        manifest = build_manifest(
            per_framework_stats=stats,
            confidence_floor=0.30,
            confidence_overrides={"mitre_atlas": 0.35},
            staleness_result={"status": "skipped", "upstream_hub_count": 0},
            model_adapter_hash="test_hash",
        )
        manifest_path = output_dir / "export_manifest.json"
        atomic_write_json(manifest, manifest_path)

        assert manifest_path.exists()
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert loaded["total_exported"] == 3
        assert loaded["confidence_floor"] == 0.30
