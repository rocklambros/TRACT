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
    "owasp_ai_exchange": 54,
    "owasp_llm_top10": 10,
    "owasp_agentic_top10": 10,
    "eu_gpai_cop": 40,
    "owasp_dsgai": 21,
    "eu_ai_act": 126,
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
    # Traditional frameworks (from OpenCRE) — include alternate names
    "CAPEC": "capec",
    "CWE": "cwe",
    "NIST 800-53": "nist_800_53",
    "NIST SP 800-53": "nist_800_53",
    "NIST 800-53 v5": "nist_800_53",
    "ASVS": "asvs",
    "OWASP Application Security Verification Standard": "asvs",
    "ISO 27001": "iso_27001",
    "DSOMM": "dsomm",
    "DevSecOps Maturity Model (DSOMM)": "dsomm",
    "WSTG": "wstg",
    "OWASP Web Security Testing Guide": "wstg",
    "OWASP Web Security Testing Guide (WSTG)": "wstg",
    "OWASP Cheat Sheet Series": "owasp_cheat_sheets",
    "OWASP Cheat Sheets": "owasp_cheat_sheets",
    "OWASP Proactive Controls": "owasp_proactive_controls",
    "ENISA": "enisa",
    "ETSI": "etsi",
    "SAMM": "samm",
    "OWASP SAMM": "samm",
    "Cloud Controls Matrix": "csa_ccm",
    "BIML": "biml",
    "OWASP Top 10 2021": "owasp_top10_2021",
    "NIST 800-63": "nist_800_63",
    "NIST SSDF": "nist_ssdf",
}

# ── Phase 0: Zero-Shot Baseline Settings ─────────────────────────────────

PHASE0_BOOTSTRAP_N_RESAMPLES: Final[int] = 10_000
PHASE0_BOOTSTRAP_CI_LEVEL: Final[float] = 0.95
PHASE0_BOOTSTRAP_SEED: Final[int] = 42

PHASE0_GATE_A_OPUS_HIT5_THRESHOLD: Final[float] = 0.50
PHASE0_GATE_B_HIT1_GAP_THRESHOLD: Final[float] = 0.10

PHASE0_LLM_PROBE_MODEL: Final[str] = "claude-opus-4-20250514"
PHASE0_LLM_PROBE_MAX_CONCURRENT: Final[int] = 5
PHASE0_LLM_SHORTLIST_PER_BRANCH: Final[int] = 20
PHASE0_LLM_FINAL_TOP_K: Final[int] = 10

PHASE0_DESCRIPTION_PILOT_N_HUBS: Final[int] = 50

# ── Phase 1A: Data Infrastructure ───────────────────────────────────────

PHASE1A_DESCRIPTION_MODEL: Final[str] = "claude-opus-4-20250514"
PHASE1A_DESCRIPTION_TEMPERATURE: Final[float] = 0.0
PHASE1A_DESCRIPTION_MAX_TOKENS: Final[int] = 500
PHASE1A_DESCRIPTION_MAX_CONCURRENT: Final[int] = 5
PHASE1A_DESCRIPTION_SAVE_INTERVAL: Final[int] = 50
PHASE1A_DESCRIPTION_TIMEOUT_S: Final[int] = 60
PHASE1A_FRAMEWORK_SLUG_RE: Final[str] = r"^[a-z][a-z0-9_]{1,49}$"

# Framework IDs that have primary-source parsers (take precedence over OpenCRE extraction)
AI_PARSER_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({
    "aiuc_1", "cosai", "csa_aicm", "eu_ai_act", "eu_gpai_cop",
    "mitre_atlas", "nist_ai_600_1", "nist_ai_rmf",
    "owasp_agentic_top10", "owasp_ai_exchange", "owasp_dsgai", "owasp_llm_top10",
})

# OpenCRE framework IDs to extract (those WITHOUT primary-source parsers)
OPENCRE_EXTRACT_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset(
    set(OPENCRE_FRAMEWORK_ID_MAP.values()) - AI_PARSER_FRAMEWORK_IDS
)
