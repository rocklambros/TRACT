"""Tests for TRACT→OpenCRE framework name mapping."""
from __future__ import annotations

import pytest


class TestOpenCRENameMapping:
    def test_all_exportable_frameworks_have_mapping(self) -> None:
        from tract.export.opencre_names import TRACT_TO_OPENCRE_NAME
        exportable = ["mitre_atlas", "owasp_llm_top10", "nist_ai_600_1",
                       "csa_aicm", "eu_ai_act", "owasp_agentic_top10"]
        for fw_id in exportable:
            assert fw_id in TRACT_TO_OPENCRE_NAME, f"Missing mapping for {fw_id}"

    def test_known_names_are_exact(self) -> None:
        from tract.export.opencre_names import TRACT_TO_OPENCRE_NAME
        assert TRACT_TO_OPENCRE_NAME["mitre_atlas"] == "MITRE ATLAS"
        assert TRACT_TO_OPENCRE_NAME["owasp_llm_top10"] == "OWASP Top10 for LLM"
        assert TRACT_TO_OPENCRE_NAME["nist_ai_600_1"] == "NIST AI 600-1"
        assert TRACT_TO_OPENCRE_NAME["csa_aicm"] == "CSA AI Controls Matrix"
        assert TRACT_TO_OPENCRE_NAME["eu_ai_act"] == "EU AI Act"
        assert TRACT_TO_OPENCRE_NAME["owasp_agentic_top10"] == "OWASP Agentic AI Top 10"

    def test_get_opencre_name_raises_on_unknown(self) -> None:
        from tract.export.opencre_names import get_opencre_name
        with pytest.raises(KeyError, match="no_such_framework"):
            get_opencre_name("no_such_framework")

    def test_get_opencre_name_returns_correct_value(self) -> None:
        from tract.export.opencre_names import get_opencre_name
        assert get_opencre_name("mitre_atlas") == "MITRE ATLAS"


class TestHyperlinkTemplates:
    def test_all_exportable_frameworks_have_hyperlink_template(self) -> None:
        from tract.export.opencre_names import HYPERLINK_TEMPLATES
        exportable = ["mitre_atlas", "owasp_llm_top10", "nist_ai_600_1",
                       "csa_aicm", "eu_ai_act", "owasp_agentic_top10"]
        for fw_id in exportable:
            assert fw_id in HYPERLINK_TEMPLATES, f"Missing hyperlink template for {fw_id}"

    def test_build_hyperlink_mitre_atlas(self) -> None:
        from tract.export.opencre_names import build_hyperlink
        assert build_hyperlink("mitre_atlas", "AML.T0000") == "https://atlas.mitre.org/techniques/AML.T0000"

    def test_build_hyperlink_nist_ai_600_1(self) -> None:
        from tract.export.opencre_names import build_hyperlink
        assert build_hyperlink("nist_ai_600_1", "GAI-CBRN") == "https://airc.nist.gov/Docs/1"

    def test_build_hyperlink_unknown_framework_raises(self) -> None:
        from tract.export.opencre_names import build_hyperlink
        with pytest.raises(KeyError):
            build_hyperlink("unknown_fw", "X01")
