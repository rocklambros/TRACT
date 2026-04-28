"""Tests for traditional framework extraction from OpenCRE."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "phase1a_mini_cres.json"


@pytest.fixture
def mini_cres() -> list[dict]:
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["cres"]


class TestSlugify:

    def test_basic_slugify(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import slugify
        assert slugify("Hello World") == "hello-world"

    def test_special_chars(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import slugify
        assert slugify("CWE-79: Cross-Site Scripting") == "cwe-79-cross-site-scripting"

    def test_max_length(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import slugify
        long_name = "a" * 200
        result = slugify(long_name)
        assert len(result) <= 80

    def test_empty_raises(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import slugify
        with pytest.raises(ValueError, match="empty"):
            slugify("")

    def test_whitespace_only_raises(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import slugify
        with pytest.raises(ValueError, match="empty"):
            slugify("   ")


class TestFrameworkSlugValidation:

    def test_valid_slug(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import validate_framework_slug
        validate_framework_slug("capec")
        validate_framework_slug("nist_800_53")
        validate_framework_slug("owasp_cheat_sheets")

    def test_rejects_path_traversal(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import validate_framework_slug
        with pytest.raises(ValueError):
            validate_framework_slug("../etc/passwd")

    def test_rejects_uppercase(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import validate_framework_slug
        with pytest.raises(ValueError):
            validate_framework_slug("CAPEC")

    def test_rejects_empty(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import validate_framework_slug
        with pytest.raises(ValueError):
            validate_framework_slug("")

    def test_rejects_too_long(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import validate_framework_slug
        with pytest.raises(ValueError):
            validate_framework_slug("a" * 60)


class TestTripleKeyDedup:

    def test_dedup_same_section_different_cre(self, mini_cres: list[dict]) -> None:
        from scripts.phase1a.extract_traditional_frameworks import extract_framework_controls
        controls = extract_framework_controls(
            mini_cres,
            framework_names={"Framework Alpha"},
            framework_id="framework_alpha",
        )
        # Framework Alpha has sections A-1, A-2, A-3 across different CREs
        # Each should appear once in the output
        ids = [c.control_id for c in controls]
        assert len(ids) == len(set(ids))

    def test_empty_section_id_different_names_kept(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import extract_framework_controls
        cres = [
            {"doctype": "CRE", "id": "HUB-1", "name": "Hub 1", "links": [
                {"ltype": "Linked To", "document": {"doctype": "Standard", "name": "TestFW", "sectionID": "", "section": "Section A"}},
                {"ltype": "Linked To", "document": {"doctype": "Standard", "name": "TestFW", "sectionID": "", "section": "Section B"}},
            ]},
        ]
        controls = extract_framework_controls(cres, {"TestFW"}, "test_fw")
        assert len(controls) == 2

    def test_empty_section_id_same_name_deduped(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import extract_framework_controls
        cres = [
            {"doctype": "CRE", "id": "HUB-1", "name": "Hub 1", "links": [
                {"ltype": "Linked To", "document": {"doctype": "Standard", "name": "TestFW", "sectionID": "", "section": "Section A"}},
            ]},
            {"doctype": "CRE", "id": "HUB-2", "name": "Hub 2", "links": [
                {"ltype": "Linked To", "document": {"doctype": "Standard", "name": "TestFW", "sectionID": "", "section": "Section A"}},
            ]},
        ]
        controls = extract_framework_controls(cres, {"TestFW"}, "test_fw")
        assert len(controls) == 1


class TestBareIdHandling:

    def test_capec_bare_id(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import extract_framework_controls
        cres = [
            {"doctype": "CRE", "id": "HUB-1", "name": "Hub 1", "links": [
                {"ltype": "Automatically Linked To", "document": {
                    "doctype": "Standard", "name": "CAPEC", "sectionID": "184", "section": ""
                }},
            ]},
        ]
        controls = extract_framework_controls(cres, {"CAPEC"}, "capec")
        assert len(controls) == 1
        assert controls[0].control_id == "capec:184"
        assert controls[0].title == "CAPEC-184"

    def test_section_name_used_when_available(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import extract_framework_controls
        cres = [
            {"doctype": "CRE", "id": "HUB-1", "name": "Hub 1", "links": [
                {"ltype": "Linked To", "document": {
                    "doctype": "Standard", "name": "CAPEC",
                    "sectionID": "184",
                    "section": "Software Integrity Attack"
                }},
            ]},
        ]
        controls = extract_framework_controls(cres, {"CAPEC"}, "capec")
        assert controls[0].title == "Software Integrity Attack"

    def test_link_type_in_metadata(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import extract_framework_controls
        cres = [
            {"doctype": "CRE", "id": "HUB-1", "name": "Hub 1", "links": [
                {"ltype": "Automatically Linked To", "document": {
                    "doctype": "Standard", "name": "CAPEC", "sectionID": "184", "section": ""
                }},
            ]},
        ]
        controls = extract_framework_controls(cres, {"CAPEC"}, "capec")
        assert controls[0].metadata is not None
        assert controls[0].metadata["link_type"] == "AutomaticallyLinkedTo"


class TestEndToEndExtraction:
    """Integration tests against real OpenCRE data (skipped if not available)."""

    @pytest.fixture
    def real_cres(self) -> list[dict]:
        path = Path("data/raw/opencre/opencre_all_cres.json")
        if not path.exists():
            pytest.skip("OpenCRE data not available")
        import json
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data["cres"]

    def test_extracts_19_frameworks(self, real_cres: list[dict]) -> None:
        from tract.config import OPENCRE_EXTRACT_FRAMEWORK_IDS, OPENCRE_FRAMEWORK_ID_MAP
        from scripts.phase1a.extract_traditional_frameworks import extract_framework_controls
        from collections import defaultdict

        id_to_names: dict[str, set[str]] = defaultdict(set)
        for name, fwid in OPENCRE_FRAMEWORK_ID_MAP.items():
            id_to_names[fwid].add(name)

        extracted_count = 0
        for fw_id in sorted(OPENCRE_EXTRACT_FRAMEWORK_IDS):
            names = id_to_names.get(fw_id)
            if not names:
                continue
            controls = extract_framework_controls(real_cres, names, fw_id)
            if controls:
                extracted_count += 1
                assert len(controls) > 0, f"{fw_id} has zero controls"
                # Verify control_ids are unique
                ids = [c.control_id for c in controls]
                assert len(ids) == len(set(ids)), f"{fw_id} has duplicate control_ids"

        assert extracted_count == 19

    def test_capec_has_expected_controls(self, real_cres: list[dict]) -> None:
        from scripts.phase1a.extract_traditional_frameworks import extract_framework_controls
        controls = extract_framework_controls(real_cres, {"CAPEC"}, "capec")
        # CAPEC has 1799 links but many point to same section
        assert len(controls) > 100
        assert all(c.control_id.startswith("capec:") for c in controls)

    def test_all_controls_no_empty_descriptions(self, real_cres: list[dict]) -> None:
        from tract.config import OPENCRE_EXTRACT_FRAMEWORK_IDS, OPENCRE_FRAMEWORK_ID_MAP
        from scripts.phase1a.extract_traditional_frameworks import extract_framework_controls
        from collections import defaultdict

        id_to_names: dict[str, set[str]] = defaultdict(set)
        for name, fwid in OPENCRE_FRAMEWORK_ID_MAP.items():
            id_to_names[fwid].add(name)

        for fw_id in sorted(OPENCRE_EXTRACT_FRAMEWORK_IDS):
            names = id_to_names.get(fw_id)
            if not names:
                continue
            controls = extract_framework_controls(real_cres, names, fw_id)
            for c in controls:
                assert c.description, f"{fw_id}:{c.control_id} has empty description"
