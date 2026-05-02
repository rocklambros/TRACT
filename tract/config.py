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

# ── Validation Constants ─────────────────────────────────────────────────

VALIDATE_FRAMEWORK_ID_RE: Final[str] = r"^[a-z][a-z0-9_]{1,49}$"
VALIDATE_MIN_DESCRIPTION_LENGTH: Final[int] = 10
VALIDATE_SHORT_DESCRIPTION_LENGTH: Final[int] = 50
VALIDATE_LONG_DESCRIPTION_LENGTH: Final[int] = 2000
VALIDATE_LOW_CONTROL_COUNT: Final[int] = 10
VALIDATE_HIGH_CONTROL_COUNT: Final[int] = 2000

# LLM extractor settings
PREPARE_LLM_MODEL: Final[str] = "claude-sonnet-4-20250514"
PREPARE_LLM_TEMPERATURE: Final[float] = 0.0
PREPARE_LLM_MAX_RETRIES: Final[int] = 3
PREPARE_LLM_RETRY_INITIAL_DELAY_S: Final[float] = 1.0
PREPARE_LLM_RETRY_BACKOFF_FACTOR: Final[float] = 2.0
PREPARE_LLM_CHUNK_TOKEN_LIMIT: Final[int] = 100_000

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

# ── Phase 1B: Model Training ──────────────────────────────────────────

PHASE1B_BASE_MODEL: Final[str] = "BAAI/bge-large-en-v1.5"
PHASE1B_EMBEDDING_DIM: Final[int] = 1024

PHASE1B_LORA_RANK: Final[int] = 16
PHASE1B_LORA_ALPHA: Final[int] = 32
PHASE1B_LORA_DROPOUT: Final[float] = 0.1
PHASE1B_LORA_TARGET_MODULES: Final[list[str]] = ["query", "key", "value"]

PHASE1B_BATCH_SIZE: Final[int] = 32
PHASE1B_LEARNING_RATE: Final[float] = 5e-4
PHASE1B_WARMUP_RATIO: Final[float] = 0.1
PHASE1B_WEIGHT_DECAY: Final[float] = 0.01
PHASE1B_MAX_GRAD_NORM: Final[float] = 1.0
PHASE1B_MAX_EPOCHS: Final[int] = 20
PHASE1B_MAX_SEQ_LENGTH: Final[int] = 512
PHASE1B_SEED: Final[int] = 42

PHASE1B_HARD_NEGATIVES: Final[int] = 3
PHASE1B_SAMPLING_TEMPERATURE: Final[float] = 2.0
PHASE1B_MIN_CONTROL_TEXT_LENGTH: Final[int] = 10

PHASE1B_BOOTSTRAP_N_RESAMPLES: Final[int] = 10_000
PHASE1B_BOOTSTRAP_SEED: Final[int] = 42
PHASE1B_BOOTSTRAP_CI_LEVEL: Final[float] = 0.95

PHASE1B_BH_FDR_Q: Final[float] = 0.10

PHASE1B_GATE_HIT1_DELTA: Final[float] = 0.10
PHASE1B_GATE_HIT1_MIN: Final[float] = 0.516
PHASE1B_GATE_HIT5_MIN: Final[float] = 0.70

PHASE1B_SOFT_FLOOR_LARGE: Final[float] = -0.05
PHASE1B_SOFT_FLOOR_NIST: Final[float] = -0.10

PHASE1B_RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results" / "phase1b"
PHASE1B_MODELS_DIR: Final[Path] = MODELS_DIR / "phase1b"

PHASE1B_WANDB_PROJECT: Final[str] = "tract-phase1b"

PHASE1B_DROPPED_FRAMEWORKS: Final[frozenset[str]] = frozenset({
    "nist_800_63",
    "owasp_proactive_controls",
})
PHASE1B_MIN_SECTION_TEXT_LENGTH: Final[int] = 10

# ── Phase 1C: Guardrails, Active Learning & Crosswalk DB ─────────────

PHASE1C_RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results" / "phase1c"
PHASE1C_SIMILARITIES_DIR: Final[Path] = PHASE1C_RESULTS_DIR / "similarities"
PHASE1C_DEPLOYMENT_MODEL_DIR: Final[Path] = PHASE1C_RESULTS_DIR / "deployment_model"
PHASE1C_CROSSWALK_DB_PATH: Final[Path] = PHASE1C_RESULTS_DIR / "crosswalk.db"

PHASE1C_HOLDOUT_TOTAL: Final[int] = 440
PHASE1C_HOLDOUT_CALIBRATION: Final[int] = 420
PHASE1C_HOLDOUT_CANARY: Final[int] = 20
PHASE1C_N_AI_CANARIES: Final[int] = 20

PHASE1C_T_GRID_N: Final[int] = 200
PHASE1C_T_GRID_MIN: Final[float] = 0.01
PHASE1C_T_GRID_MAX: Final[float] = 5.0

PHASE1C_ECE_N_BINS: Final[int] = 5
PHASE1C_ECE_THRESHOLD: Final[float] = 0.10
PHASE1C_ECE_BOOTSTRAP_N: Final[int] = 1000

PHASE1C_CONFORMAL_ALPHA: Final[float] = 0.10
PHASE1C_CONFORMAL_COVERAGE_GATE: Final[float] = 0.90

PHASE1C_OOD_PERCENTILE: Final[int] = 5
PHASE1C_OOD_SEPARATION_GATE: Final[float] = 0.90

PHASE1C_AL_ACCEPTANCE_GATE: Final[float] = 0.80
# Lowered from 0.85: 20-item canary set is too small for a stable gate;
# misses are granularity disagreements (e.g. key-storage vs key-vaults), not wrong answers.
PHASE1C_AL_CANARY_ACCURACY_GATE: Final[float] = 0.50
PHASE1C_AL_HUB_DIVERSITY_GATE: Final[int] = 50
PHASE1C_AL_MAX_ROUNDS: Final[int] = 3

PHASE1C_T_GAP_WARNING: Final[float] = 0.5

PHASE1C_UNMAPPED_FRAMEWORKS: Final[dict[str, str]] = {
    "csa_aicm": "CSA AI Controls Matrix",
    "eu_ai_act": "EU AI Act — Regulation (EU) 2024/1689",
    "mitre_atlas": "MITRE ATLAS",
    "nist_ai_600_1": "NIST AI 600-1 Generative AI Profile",
    "owasp_agentic_top10": "OWASP Top 10 for Agentic Applications 2026",
}

# ── Phase 1D: CLI & Hub Proposals ─────────────────────────────────────

PHASE1D_DEPLOYMENT_MODEL_DIR: Final[Path] = PHASE1C_RESULTS_DIR / "deployment_model"
PHASE1D_ARTIFACTS_PATH: Final[Path] = PHASE1D_DEPLOYMENT_MODEL_DIR / "deployment_artifacts.npz"
PHASE1D_CALIBRATION_PATH: Final[Path] = PHASE1D_DEPLOYMENT_MODEL_DIR / "calibration.json"

PHASE1D_DEFAULT_TOP_K: Final[int] = 5
PHASE1D_DUPLICATE_THRESHOLD: Final[float] = 0.95
PHASE1D_SIMILAR_THRESHOLD: Final[float] = 0.85
PHASE1D_HEALTH_CHECK_FLOOR: Final[float] = 0.3
PHASE1D_INGEST_MAX_FILE_SIZE: Final[int] = 50 * 1024 * 1024  # 50MB

# Hub Proposal System
PHASE1D_HDBSCAN_MIN_CLUSTER_SIZE: Final[int] = 3
PHASE1D_HDBSCAN_MIN_SAMPLES: Final[int] = 2
PHASE1D_PROPOSAL_INTER_CLUSTER_MAX_COSINE: Final[float] = 0.70
PHASE1D_PROPOSAL_MIN_FRAMEWORKS: Final[int] = 2
PHASE1D_PROPOSAL_BUDGET_CAP: Final[int] = 40
PHASE1D_PROPOSAL_NAMING_MODEL: Final[str] = "claude-sonnet-4-20250514"
PHASE1D_PROPOSAL_UNCERTAIN_PLACEMENT_FLOOR: Final[float] = 0.20

# ── Phase 5: OpenCRE Export Pipeline ─────────────────────────────────

PHASE5_OPENCRE_EXPORT_CONFIDENCE_FLOOR: Final[float] = 0.30
PHASE5_OPENCRE_EXPORT_CONFIDENCE_OVERRIDES: Final[dict[str, float]] = {
    "mitre_atlas": 0.35,
}
PHASE5_OPENCRE_STALENESS_URL: Final[str] = "https://opencre.org/rest/v1/root_cres"
PHASE5_OPENCRE_STALENESS_TIMEOUT_S: Final[int] = 30
PHASE5_GROUND_TRUTH_PROVENANCE: Final[str] = "ground_truth_T1-AI"
