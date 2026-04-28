"""TRACT configuration — paths, constants, and framework metadata.

All magic numbers, paths, and external API settings live here.
Import from this module; never hardcode values in library code.
"""

from pathlib import Path
from typing import Final

# ── Project Paths ──────────────────────────────────────────────────────────

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
PROCESSED_FRAMEWORKS_DIR: Final[Path] = PROCESSED_DIR / "frameworks"
TRAINING_DIR: Final[Path] = DATA_DIR / "training"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
HUB_PROPOSALS_DIR: Final[Path] = PROJECT_ROOT / "hub_proposals"
PARSERS_DIR: Final[Path] = PROJECT_ROOT / "parsers"

# Raw framework subdirectories
RAW_OPENCRE_DIR: Final[Path] = RAW_DIR / "opencre"
RAW_FRAMEWORKS_DIR: Final[Path] = RAW_DIR / "frameworks"

# ── Text Processing ────────────────────────────────────────────────────────

DESCRIPTION_MAX_LENGTH: Final[int] = 2000

# Tolerance for expected-count validation (10% deviation triggers WARNING)
COUNT_TOLERANCE: Final[float] = 0.10

# ── OpenCRE API Settings ──────────────────────────────────────────────────

OPENCRE_API_BASE_URL: Final[str] = "https://opencre.org/rest/v1/all_cres"
OPENCRE_PER_PAGE: Final[int] = 50
OPENCRE_RETRY_MAX_ATTEMPTS: Final[int] = 5
OPENCRE_RETRY_INITIAL_DELAY_S: Final[float] = 1.0
OPENCRE_RETRY_BACKOFF_FACTOR: Final[float] = 2.0
OPENCRE_RETRY_MAX_DELAY_S: Final[float] = 30.0
OPENCRE_REQUEST_TIMEOUT_S: Final[int] = 30
OPENCRE_REQUEST_DELAY_S: Final[float] = 0.5

# ── Expected Control Counts ───────────────────────────────────────────────
# Source: PRD Section 4.2 — mapping-unit counts per framework.
# Used by BaseParser._check_expected_count to warn on deviation.

EXPECTED_COUNTS: Final[dict[str, int | None]] = {
    "csa_aicm": 243,
    "aiuc_1": 132,
    "mitre_atlas": 202,
    "cosai": 55,
    "nist_ai_rmf": 72,
    "nist_ai_600_1": 12,
    "owasp_ai_exchange": 88,
    "owasp_llm_top10": 10,
    "owasp_agentic_top10": 10,
    "eu_gpai_cop": 40,
    "owasp_dsgai": 21,
    "eu_ai_act": 100,
}

# ── OpenCRE Framework ID Map ─────────────────────────────────────────────
# Normalizes standard names as they appear in OpenCRE API responses
# to TRACT's canonical framework_id strings.

OPENCRE_FRAMEWORK_ID_MAP: Final[dict[str, str]] = {
    # AI frameworks
    "MITRE ATLAS": "mitre_atlas",
    "OWASP AI Exchange": "owasp_ai_exchange",
    "NIST AI 100-2": "nist_ai_100_2",
    "OWASP Top10 for LLM": "owasp_llm_top10",
    "OWASP Top10 for ML": "owasp_ml_top10",
    # Traditional frameworks (from OpenCRE)
    "CAPEC": "capec",
    "CWE": "cwe",
    "NIST 800-53": "nist_800_53",
    "ASVS": "asvs",
    "ISO 27001": "iso_27001",
    "DSOMM": "dsomm",
    "WSTG": "wstg",
    "OWASP Cheat Sheets": "owasp_cheat_sheets",
    "OWASP Proactive Controls": "owasp_proactive_controls",
    "ENISA": "enisa",
    "ETSI": "etsi",
    "SAMM": "samm",
    "Cloud Controls Matrix": "csa_ccm",
    "BIML": "biml",
    "OWASP Top 10 2021": "owasp_top10_2021",
    "NIST 800-63": "nist_800_63",
    "NIST SSDF": "nist_ssdf",
}
