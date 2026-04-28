"""Tests for parsers/extract_hub_links.py."""
from __future__ import annotations

import json
from pathlib import Path

from parsers.extract_hub_links import extract_links, normalize_framework_id


def test_normalize_framework_id() -> None:
    assert normalize_framework_id("CAPEC") == "capec"
    assert normalize_framework_id("NIST 800-53") == "nist_800_53"
    assert normalize_framework_id("NIST 800-53 v5") == "nist_800_53"
    assert normalize_framework_id("Cloud Controls Matrix") == "csa_ccm"
    assert normalize_framework_id("MITRE ATLAS") == "mitre_atlas"
    assert normalize_framework_id("OWASP Cheat Sheets") == "owasp_cheat_sheets"
    assert normalize_framework_id("DevSecOps Maturity Model (DSOMM)") == "dsomm"


def test_extract_links_filters_correctly(tmp_path: Path) -> None:
    mock_data = {
        "cres": [
            {
                "id": "001-01",
                "name": "Test CRE",
                "links": [
                    {
                        "ltype": "Linked To",
                        "document": {
                            "doctype": "Standard",
                            "name": "CAPEC",
                            "sectionID": "CAPEC-1",
                            "section": "Attack 1",
                        },
                    },
                    {
                        "ltype": "",
                        "document": {
                            "doctype": "CRE",
                            "name": "Child CRE",
                            "id": "002-01",
                        },
                    },
                    {
                        "ltype": "Automatically Linked To",
                        "document": {
                            "doctype": "Standard",
                            "name": "CWE",
                            "sectionID": "CWE-79",
                            "section": "XSS",
                        },
                    },
                    {
                        "ltype": "Linked To",
                        "document": {
                            "doctype": "Tool",
                            "name": "Some Tool",
                        },
                    },
                ],
            }
        ]
    }

    data_file = tmp_path / "opencre.json"
    data_file.write_text(json.dumps(mock_data), encoding="utf-8")

    links = extract_links(data_file)
    assert len(links) == 2
    types = {link.link_type for link in links}
    assert types == {"LinkedTo", "AutomaticallyLinkedTo"}
    assert all(link.cre_id == "001-01" for link in links)


def test_extract_links_real_data() -> None:
    from tract.config import RAW_OPENCRE_DIR

    opencre_path = RAW_OPENCRE_DIR / "opencre_all_cres.json"
    if not opencre_path.exists():
        import pytest
        pytest.skip("OpenCRE data not available")

    links = extract_links(opencre_path)
    assert len(links) >= 4000
    framework_ids = {link.framework_id for link in links}
    assert "capec" in framework_ids
    assert "cwe" in framework_ids
    assert "mitre_atlas" in framework_ids
