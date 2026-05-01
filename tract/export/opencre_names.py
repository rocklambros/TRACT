"""TRACT framework_id → OpenCRE FAMILY_NAME mapping.

CRITICAL (spec §4): A mismatch creates duplicate standard entries in
OpenCRE's database. These names are verified against OpenCRE's parser
source code (spreadsheet_mitre_atlas.py FAMILY_NAME, etc).
"""
from __future__ import annotations

from typing import Final

TRACT_TO_OPENCRE_NAME: Final[dict[str, str]] = {
    "mitre_atlas": "MITRE ATLAS",
    "owasp_llm_top10": "OWASP Top10 for LLM",
    "nist_ai_600_1": "NIST AI 600-1",
    "csa_aicm": "CSA AI Controls Matrix",
    "eu_ai_act": "EU AI Act",
    "owasp_agentic_top10": "OWASP Agentic AI Top 10",
}

HYPERLINK_TEMPLATES: Final[dict[str, str]] = {
    "mitre_atlas": "https://atlas.mitre.org/techniques/{section_id}",
    "owasp_llm_top10": "https://genai.owasp.org/llmrisk/{section_id}",
    "nist_ai_600_1": "https://airc.nist.gov/Docs/1",
    "csa_aicm": "https://cloudsecurityalliance.org/artifacts/ai-controls-matrix",
    "eu_ai_act": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689",
    "owasp_agentic_top10": "https://genai.owasp.org",
}


def get_opencre_name(framework_id: str) -> str:
    return TRACT_TO_OPENCRE_NAME[framework_id]


def build_hyperlink(framework_id: str, section_id: str) -> str:
    template = HYPERLINK_TEMPLATES[framework_id]
    return template.format(section_id=section_id)
