"""TRACT configuration — paths, constants, and framework metadata.

All magic numbers, paths, and external API settings live here.
Import from this module; never hardcode values in library code.
"""

import re
from pathlib import Path
from typing import Final

# ── Project Paths ──────────────────────────────────────────────────────────

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
PROCESSED_FRAMEWORKS_DIR: Final[Path] = PROCESSED_DIR / "frameworks"
# Gitignored overlay for the merged corpus including licensed frameworks.
# See RESTRICTED_FRAMEWORK_IDS below and parsers/merge_all_controls.py.
PROCESSED_LICENSED_DIR: Final[Path] = PROCESSED_DIR / "licensed"
# Gitignored before/after pairs for repairs that move text across control ids.
# Gitignored because the pairs quote source text verbatim, and a restricted
# framework's audit file would carry licensed prose into git.
PROCESSED_REPAIR_AUDIT_DIR: Final[Path] = PROCESSED_DIR / "repair_audit"
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

# Below this a description is a section title rather than a control statement.
# Measured: the 12 synthesised frameworks have a 0% honest-prose rate at this
# threshold and every parser-backed framework has at least 76%.
HONEST_PROSE_MIN_CHARS: Final[int] = 60

# Marks a control whose source text is known to be incomplete. Set by a parser
# that can prove the damage but cannot repair it, so downstream readers can
# exclude the control instead of trusting a statement with a hole in it.
CONTROL_DAMAGED_METADATA_KEY: Final[str] = "damaged"
CONTROL_DAMAGE_REASON_METADATA_KEY: Final[str] = "damage_reason"
CONTROL_DAMAGED_METADATA_VALUE: Final[str] = "true"

# Stands in for text the source lost. Editorial on purpose: a reader must not
# be able to mistake a repaired statement for the standard's own wording.
CONTROL_ELISION_MARKER: Final[str] = "[...]"

# ── Licensed source text ──────────────────────────────────────────────────
# Frameworks whose source text is licensed such that redistribution under this
# repository's CC0 grant would assert rights the project does not hold. CC0 is
# not a disclaimer, it is an affirmative grant, so a licensed control statement
# inside any tracked artifact is a rights claim the project cannot make.
#
# Single source of truth on purpose. This list is read by
# parsers/merge_all_controls.py, which excludes these frameworks from the
# tracked all_controls.json, and by tests/test_licensed_text_not_tracked.py,
# which asserts nothing licensed reached git. A second copy in the test would
# drift, and a gate that disagrees with the writer is not a gate.
# Membership is decided by the source's own notice, quoted below, not by a
# judgement call about how much text ends up in an artifact.
#
#   iso_27001  ISO/IEC 27001:2022, cover page: "COPYRIGHT PROTECTED DOCUMENT
#              (c) ISO/IEC 2022 All rights reserved ... no part of this
#              publication may be reproduced ... without prior written
#              permission." The staged copy is a single-user store licence.
#   etsi       ETSI GR SAI 005 v1.1.1, page 2, Copyright Notification: no
#              reproduction in any medium except by written permission of
#              ETSI. (c) ETSI 2021, all rights reserved. Paraphrased on
#              purpose: the fingerprint gate covers that page too, so quoting
#              the notice at length here would trip it.
#
# csa_ccm is NOT here. Its notice reserves redistribution too, and the owner
# ruled on 2026-08-16 that CCM is redistributable for this project. That ruling
# is recorded in NOTICE and in the run ledger; do not reverse it here.
RESTRICTED_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({"etsi", "iso_27001"})

# Reproduction is permitted, but on terms a CC0 grant cannot carry. CC0 is not a
# disclaimer; it is an affirmative assertion that the publisher holds the rights
# and waives them, which is false for GPL-3.0 and for share-alike text. These
# frameworks' processed files route to the gitignored overlay exactly as the
# restricted ones do, and their ASSIGNMENTS stay tracked and published, because
# a mapping is a fact about two documents rather than a reproduction of either.
# Training reads the overlay, so this costs zero anchors. See rulings R4 to R6.
#
# Held two members on 2026-08-19, down from seven. What left, and why:
#
#   biml, samm, wstg, owasp_top10_2021, owasp_proactive_controls
#     All five are CC BY-SA. Seven other CC BY-SA frameworks are tracked and
#     published already: asvs, owasp_cheat_sheets, owasp_llm_top10,
#     owasp_ml_top10, owasp_agentic_top10, owasp_dsgai, owasp_llm_top10_2026.
#     Treating five of twelve differently is defensible on no reading of the
#     licence, and the split was an artifact of which files happened to be in
#     git when the tiers landed rather than of anything CC BY-SA says. With
#     LICENSES/ shipping the real texts, NOTICE carrying attribution and the
#     modification statement, and one licence declaration across the published
#     artifacts, section 3(a)'s attribution and notice obligations are
#     discharged as well as a mixed-source corpus allows. The obligation is to
#     attribute and to notice, not to withhold.
#
# What stayed, and why each is a different question from the five:
#
#   dsomm      GPL-3.0-only. Section 5's aggregation carve-out says inclusion
#              in an aggregate does not apply the License to THE OTHER PARTS of
#              the aggregate. It does not say the covered work stops being GPL,
#              so DSOMM's text is still conveyed under GPL-3.0 with section 4's
#              obligations attached. The carve-out also wants "a volume of a
#              storage or distribution medium", and data/processed/
#              all_controls.json is one document interleaving DSOMM with 28
#              other frameworks specifically so a trainer consumes them
#              jointly. CC BY-SA's share-alike attaches to the Adapted
#              Material; GPL's attaches to the whole work. That difference is
#              why the parity argument above carries the five and does not
#              reach here. Its prose has never been published, `git push` is
#              one-way, and training reads the overlay, so the anchor cost is
#              zero.
#
# What LEFT on 2026-08-26, and why:
#
#   csa_ccm    Removed on owner decision D1(b), 2026-08-26, recorded in
#              claudedocs/jetson-runpod-start.md and in NOTICE. The owner
#              re-affirmed the 2026-08-16 ruling that CSA material is
#              redistributable for this project, this time on an explicit
#              reading of the CSA membership terms rather than on an
#              unrecorded basis.
#
#              This is a CLASSIFICATION fix, not a gate change, and the
#              distinction is the whole point. Conditional membership meant
#              "prose withheld from git". Holding csa_ccm there while the
#              ruling said redistributable put an overlay framework outside
#              the fingerprint corpus -- see FINGERPRINT_EXCLUDED_FRAMEWORK_IDS
#              in tract/licensing.py, which had to defer it because 138 of the
#              243 TRACKED csa_aicm descriptions are byte-identical to a CCM
#              specification and fingerprinting CCM would have failed the
#              branch on tracked AICM text. One framework in the overlay with
#              no gate coverage was the only real hole in that gate, and
#              removing the misclassification closes it without weakening
#              anything: the corpus it covers is now exactly the corpus it is
#              supposed to cover, with no deferrals left.
#
#              What this does NOT do: it does not move dsomm, etsi or
#              iso_27001, and it does not trim any fingerprint corpus. Those
#              three are still withheld and still fingerprinted in full.
CONDITIONAL_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({
    "dsomm",                     # GPL-3.0-only
})

# What routes to the overlay. RESTRICTED_FRAMEWORK_IDS keeps its narrower
# meaning everywhere else: the fingerprint gate and the "must never appear in
# git in any form" rule.
OVERLAY_FRAMEWORK_IDS: Final[frozenset[str]] = (
    RESTRICTED_FRAMEWORK_IDS | CONDITIONAL_FRAMEWORK_IDS
)

# ── Prose licences, adjudicated ───────────────────────────────────────────
# Nine of the 32 sources state their terms in the publisher's own sentence
# rather than as an SPDX identifier. An identifier is a structured token this
# repository can act on: it names shippable terms, and both `_copyleft` and the
# LICENSES/ checks work off it. A sentence is not. "Reproduction authorised
# provided the source is acknowledged" and "Reproduction only by written
# permission" are the same shape to every automated check here and opposite in
# effect, and no substring test separates them reliably -- a heuristic over
# publisher prose fails silently in the PERMISSIVE direction, because a reader
# cannot tell "no match" from "no such source".
#
# So these two sets do not derive anything. They record what a human decided
# after reading the notice, and tests/test_framework_licenses.py refuses to let
# a prose-licenced framework exist in neither. That refusal is the check that
# was missing when csa_aicm reached 243 tracked control statements with no
# owner ruling: nothing failed, and nothing could have.
#
# The full notice for each is in FRAMEWORK_LICENSES above and in NOTICE. Only
# the deciding clause is summarised here, and ETSI's is paraphrased rather than
# quoted because the fingerprint gate covers its notice page.

# Redistribution is reserved by the publisher. Each must be withheld from git
# (overlay) or carry a recorded owner ruling. Enforced, not advisory.
REDISTRIBUTION_RESERVED_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({
    "csa_aicm",    # all rights reserved, no redistribution
    "csa_ccm",     # all rights reserved, no redistribution
    "etsi",        # reproduction only by written permission (paraphrased)
    "iso_27001",   # all rights reserved, single-user store licence
})

# Redistribution is permitted by the publisher's own words. Recorded so the
# check above can tell "adjudicated permissive" from "nobody has looked".
PROSE_LICENCE_ADJUDICATED_PERMISSIVE: Final[frozenset[str]] = frozenset({
    "enisa",        # "Reproduction authorised provided the source is acknowledged"
    "eu_ai_act",    # reuse permitted with attribution, per Commission Decision
    "eu_gpai_cop",  # "Published for public use"
    "nist_800_63",  # US Government work, not subject to domestic copyright
    "nist_ssdf",    # US Government work, not subject to domestic copyright
})

# Reserved by the publisher AND ruled redistributable for this project by the
# owner. The ruling is an entitlement the owner holds, NOT a property of the
# document, so a fork does not inherit it. See NOTICE for the basis and date.
#
#   csa_aicm, csa_ccm   Owner decision D1(b), 2026-08-26, on a reading of the
#                       CSA membership terms. Recorded in NOTICE and in
#                       claudedocs/jetson-runpod-start.md.
OWNER_RULED_REDISTRIBUTABLE: Final[frozenset[str]] = frozenset({
    "csa_aicm",
    "csa_ccm",
})

# ── Pretraining-contamination holdout ─────────────────────────────────────
# Frameworks parsed into data/processed/frameworks/ that must never reach a
# training roster, a LOFO fold roster, the curated link file, or the merged
# corpus a trainer reads.
#
#   owasp_llm_top10_2026  Published after BAAI/bge-large-en-v1.5 was trained,
#                         so it is the only corpus here that can separate an
#                         encoder mapping meaning from an encoder recalling
#                         text it saw in pretraining. Every other framework
#                         predates the checkpoint and cannot answer that
#                         question. Spec Part 1.6.
#
# Restricted and holdout are different properties. A restricted framework is
# one this repository may not redistribute; a holdout is one the model may not
# see. owasp_llm_top10_2026 is CC BY-SA 4.0 and freely redistributable, and it
# is still excluded from training, so the two lists stay separate.
#
# Wired, not decorative: parsers/merge_all_controls.py drops these before it
# builds either corpus, and tests/test_holdout_framework.py sweeps the roster
# constants, the link files, and the merge for any mention of them.
HOLDOUT_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({
    "owasp_llm_top10_2026",
})

# ── Third-party framework licences ────────────────────────────────────────
# Every framework whose content reaches data/processed/frameworks/, with the
# licence read off that framework's own staged artifact. SPDX identifiers where
# one applies, a short quotation of the source's own notice where none does,
# and "UNDETERMINED" where the staged artifact states no terms at all.
#
# UNDETERMINED is a real value, not a placeholder. Guessing a permissive
# licence for a source that never granted one is the mistake this table exists
# to make visible, and the entries below marked that way are listed in NOTICE
# as open questions rather than resolved ones.
#
# Wired, not decorative: tests/test_framework_licenses.py fails when a file
# appears in data/processed/frameworks/ without an entry here, and when an
# entry here is missing from NOTICE. A new ingest cannot skip the question.
UNDETERMINED_LICENSE: Final[str] = "UNDETERMINED"

# Adjudicated 2026-09-06 by the repository owner: nist_800_53, nist_ai_100_2,
# nist_ai_600_1 and nist_ai_rmf carry the same terms as nist_800_63 and
# nist_ssdf, which were already recorded. All six are NIST publications authored
# by US federal employees. The table had been internally inconsistent for one
# publisher, and the four unadjudicated entries blocked the Phase 2C annotator
# packet, whose default framework is nist_800_53.

FRAMEWORK_LICENSES: Final[dict[str, str]] = {
    "aiuc_1": UNDETERMINED_LICENSE,
    "asvs": "CC-BY-SA-4.0",
    "biml": "CC-BY-SA-3.0 AND CC-BY-SA-4.0",
    "capec": UNDETERMINED_LICENSE,
    "cosai": "CC-BY-4.0",
    "csa_aicm": (
        "Proprietary. (c) Cloud Security Alliance, all rights reserved. Personal "
        "non-commercial use, no redistribution, fair-use quotation with "
        "attribution."
    ),
    "csa_ccm": (
        "Proprietary. (c) Cloud Security Alliance, all rights reserved. Personal "
        "non-commercial use, no redistribution, fair-use quotation with "
        "attribution."
    ),
    "cwe": UNDETERMINED_LICENSE,
    "dsomm": "GPL-3.0-only",
    "enisa": (
        "(c) ENISA 2021. Reproduction authorised provided the source is "
        "acknowledged."
    ),
    "etsi": (
        "(c) ETSI 2021, all rights reserved. Reproduction only by written "
        "permission of ETSI."
    ),
    "eu_ai_act": (
        "(c) European Union. Reuse permitted with attribution per Commission "
        "Decision 2011/833/EU."
    ),
    "eu_gpai_cop": "(c) European Commission. Published for public use.",
    "iso_27001": (
        "Proprietary. (c) ISO/IEC 2022, all rights reserved. Single-user store "
        "licence, no reproduction without prior written permission."
    ),
    "mitre_atlas": "Apache-2.0",
    "nist_800_53": (
        "US Government work, not subject to copyright in the United States. "
        "Attribution appreciated by NIST."
    ),
    "nist_800_63": (
        "US Government work, not subject to copyright in the United States. "
        "Attribution appreciated by NIST."
    ),
    "nist_ai_100_2": (
        "US Government work, not subject to copyright in the United States. "
        "Attribution appreciated by NIST."
    ),
    "nist_ai_600_1": (
        "US Government work, not subject to copyright in the United States. "
        "Attribution appreciated by NIST."
    ),
    "nist_ai_rmf": (
        "US Government work, not subject to copyright in the United States. "
        "Attribution appreciated by NIST."
    ),
    "nist_ssdf": (
        "US Government work, not subject to copyright in the United States. "
        "Attribution appreciated by NIST."
    ),
    "owasp_agentic_top10": "CC-BY-SA-4.0",
    "owasp_ai_exchange": UNDETERMINED_LICENSE,
    "owasp_cheat_sheets": "CC-BY-SA-4.0",
    "owasp_dsgai": "CC-BY-SA-4.0",
    "owasp_llm_top10": "CC-BY-SA-4.0",
    # A separate framework from owasp_llm_top10 above, not a newer version of
    # it. The 2025 ids carry all 13 of OpenCRE's links for this standard, so
    # the two must never share a file. Licence read from the document's own
    # "License and Usage" block.
    "owasp_llm_top10_2026": "CC-BY-SA-4.0",
    "owasp_ml_top10": "CC-BY-SA-4.0",
    "owasp_proactive_controls": "CC-BY-SA-4.0",
    "owasp_top10_2021": "CC-BY-SA-4.0",
    "samm": "CC-BY-SA-4.0",
    "wstg": "CC-BY-SA-4.0",
}

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
# There is no table here on purpose. Each parser declares its own
# expected_count and expected_count_is_floor, and BaseParser.run() enforces
# them at write time.
#
# The table that used to live here duplicated those declarations and drifted
# from them: it held 54 for owasp_ai_exchange against the parser's 107, which
# made parsers/validate_all.py fail every run on a framework that was parsing
# correctly. A second copy of a number nobody updates is worse than no copy,
# because it turns a passing check into noise people learn to ignore.

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

# ── Framework name reconciliation ─────────────────────────────────────
# hub_links_curated.jsonl carries the standard_name OpenCRE uses; the parsed
# corpus carries the name the framework calls itself. Where they disagree, a
# control's prose cannot be joined to its own link and the pipeline silently
# falls back to the section title. Maps the link-side spelling to the
# control-side one. Keyed and compared case-insensitively after whitespace
# collapse; this table only needs the cases that differ by more than that.
FRAMEWORK_NAME_ALIASES: Final[dict[str, str]] = {
    # OpenCRE says "ISO 27001", the parser says "ISO/IEC 27001:2022 Annex A",
    # and without this line those two strings never meet: all 94 ISO links
    # resolved to nothing and fell back to their three-word section title,
    # while the parser's 93 controls at 0.967 prose sat unread. Measured, not
    # assumed. The prose fraction was checked and the reachability was not,
    # which is why tests/test_prose_reachability.py now checks the second one.
    "iso 27001": "ISO/IEC 27001:2022 Annex A",
    "nist 800-53 v5": "NIST 800-53",
    "devsecops maturity model (dsomm)": "DSOMM",
    "owasp web security testing guide (wstg)": "WSTG",
    # This maps to the 2025 edition and must keep doing so. There is
    # deliberately NO alias for the 2026 edition: an alias exists to let an
    # OpenCRE standard_name reach a parser's framework_name, and the 2026
    # edition is the pretraining-contamination holdout, which has no OpenCRE
    # links and must never acquire a path to one. See HOLDOUT_FRAMEWORK_IDS.
    "owasp top10 for llm": "OWASP Top 10 for LLM Applications 2025",
    "owasp top10 for ml": "OWASP Top10 for ML",
    "owasp top10 for agentic ai": "OWASP Top 10 for Agentic Applications 2026",
}

# A control's description counts as prose only when it carries meaningfully more
# than its own title. Nineteen frameworks arrive from OpenCRE with description
# set to the title verbatim, and those must not be mistaken for full text.
PROSE_MIN_EXTRA_CHARS: Final[int] = 20

# The encoder truncates at PHASE1B_MAX_SEQ_LENGTH tokens regardless, so text
# beyond this is discarded silently. Anchors are cut here instead, where it can
# be counted and reported. Roughly four characters per token for English.
#
# This matters unevenly: measured over the eval corpus, the two smallest folds
# were 100% over budget (OWASP Top10 for LLM median ~2,246 tokens, OWASP Top10
# for ML ~1,135) while MITRE ATLAS was 0% over. A truncation that varies by
# fold and by arm is a confound, not a detail.
#
# It is DERIVED from the token budget, not chosen independently. At 2000 it
# was the binding constraint rather than the encoder: only 7 of 147 eval
# anchors exceeded 512 tokens after this cut, while 51 exceeded it before.
# Raising the encoder's context without raising this does nothing at all,
# which is the trap a model swap walks into.
CHARS_PER_TOKEN: Final[int] = 4


def max_anchor_chars(max_seq_length: int = PHASE1B_MAX_SEQ_LENGTH) -> int:
    """Character budget matching a given token budget, with headroom.

    The margin keeps the character cut slightly LOOSER than the token limit so
    the tokenizer, not this heuristic, decides what is dropped -- and so the
    truncation counter reports the encoder's behaviour rather than its own.
    """
    return int(max_seq_length * CHARS_PER_TOKEN * 1.05)


MAX_ANCHOR_CHARS: Final[int] = max_anchor_chars()

# Long-context encoders, for the arm that tests whether the 512-token ceiling
# costs anything. BGE-large is BertModel with 512 absolute position
# embeddings, a hard architectural limit. These are alternatives whose context
# is 8k or more.
#
# Only models natively supported by the pinned transformers are listed.
# Alibaba-NLP/gte-large-en-v1.5 is the closest drop-in by hidden size but
# declares model_type "new" and needs trust_remote_code=True, which executes
# code from the repository on every training pod. That is not a trade worth
# making for a measured ceiling of ~2 hit@1 points.
# Attention projection names differ by architecture, and PEFT raises
# "Target modules not found" rather than attaching nothing -- but it raises
# AFTER SentenceTransformer has downloaded the encoder and run the full
# zero-shot GPU evaluation, so the crash costs a fold's setup on every pod.
# Resolved from AutoConfig.model_type in a pre-flight instead.
LORA_TARGET_MODULES_BY_ARCH: Final[dict[str, list[str]]] = {
    "bert": ["query", "key", "value"],
    "xlm-roberta": ["query", "key", "value"],
    "roberta": ["query", "key", "value"],
    "modernbert": ["Wqkv", "Wo"],
    "qwen3": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

LONG_CONTEXT_MODELS: Final[dict[str, int]] = {
    "Alibaba-NLP/gte-modernbert-base": 8192,
    "Qwen/Qwen3-Embedding-0.6B": 32768,
    "BAAI/bge-m3": 8192,
}

# Section headings that begin remediation guidance rather than description.
# BGE-large is BertModel with absolute position embeddings and exactly 512 of
# them, so the budget cannot be raised; the only question is which 512 tokens
# it spends. Measured over the corpus, 175 controls carry one of these and the
# median control puts 47% of its body after it. That tail says how to fix the
# problem, not which hub the control is about, and a mean-pooled bi-encoder
# dilutes toward generic security language when it is included.
#
# Deliberately conservative. "Controls" and "Risk Factors" are omitted because
# both occur in ordinary prose, and cutting on them would truncate description.
REMEDIATION_HEADINGS: Final[tuple[str, ...]] = (
    "How to Prevent",
    "Example Attack Scenarios",
    "Example Attack Scenario",
    "Prevention and Mitigation Strategies",
    "Countermeasures",
    "Remediation",
    "Mitigations",
    "Mitigation",
    "Prevention",
    "References",
)

PHASE1B_GATE_HIT1_DELTA: Final[float] = 0.10
PHASE1B_GATE_HIT1_MIN: Final[float] = 0.516
PHASE1B_GATE_HIT5_MIN: Final[float] = 0.70

# Preference order when two links collapse onto one anchor and only one pair can
# be kept. Lower wins. Held here, not in tract/training/data.py, because
# tract/ceiling_study.py needs the same order and must run without torch --
# importing it from there pulled in torch, sentence-transformers and datasets.
# It was previously duplicated for exactly that reason, and the duplicate then
# drifted: neither copy learned about T2, and the lookup defaults to 99, so a
# human-authored bridge link would have lost every contest to an automatic one.
TIER_PRIORITY: Final[dict[str, int]] = {
    "T1": 0,      # OpenCRE-curated, independently of TRACT
    "T1-AI": 1,   # human-curated AI framework link
    "T2": 2,      # Phase 2C bridge link: human-authored, one annotator
    "T3": 3,      # AutomaticallyLinkedTo
    "AL": 4,      # active-learning acceptance
}

# ── Phase 2C gates ────────────────────────────────────────────────────────
# docs/phase2c-preregistration.md Section 2, verbatim. Held here rather than
# read from the markdown so a mismatch between the two is a test failure
# instead of a reading. Every one of these existed only as prose until
# checkpoint 2 demonstrated that a one-control sheet mapping AC-1 onto all 78
# hub ids -- copied from the packet's own first column, confidence 1, rationale
# "." -- takes the orphan rate from 78/78 to 0/78 while violating three of them
# with no code objecting.
PHASE2C_GATE1_MAX_ORPHANS: Final[int] = 55
PHASE2C_GATE1_MIN_DEORPHANED: Final[int] = 23
# Q1: distinct controls that must contribute at least one accepted link.
# 23 links from one control is not a sweep.
PHASE2C_Q1_MIN_DISTINCT_CONTROLS: Final[int] = 40
# Q2: a control mapping to more than this many AI hubs is making a judgement
# about the region, not about the control.
PHASE2C_Q2_MAX_HUBS_PER_CONTROL: Final[int] = 6
# Q3: a link below this does not count toward Gate 1. Low-confidence links are
# data, not evidence.
PHASE2C_Q3_CONFIDENCE_FLOOR: Final[int] = 2
# Q4: fraction of controls that must be annotated by two people. Not a
# threshold on the agreement rate -- a requirement that the number exist.
PHASE2C_Q4_MIN_DOUBLE_ANNOTATED: Final[float] = 0.15

# Stamped into the `link_type` of every Phase 2C bridge link so that
# assign_quality_tier returns T2 rather than falling through to T1. Held here
# rather than in tract/bridge/ so tract/training/data_quality.py can read it
# without importing the bridge package. Deliberately not "AutomaticallyLinkedTo",
# which already denotes the deterministic CAPEC->CWE->CRE chain and tiers T3.
BRIDGE_LINK_TYPE: Final[str] = "BridgeCurated"

# results/phase1b/CAMPAIGN3.md Section 3, verbatim:
#   PASS iff P(true delta <= 0.10) < 0.05.
# Held here rather than inline because the alternative reading -- testing the
# lower bound of a two-sided 95% interval against the same threshold -- is the
# SAME rule at alpha 0.025, and the two were used interchangeably in three
# campaign write-ups. Naming the alpha makes the substitution visible.
PREREGISTERED_GATE_ALPHA: Final[float] = 0.05

PHASE1B_SOFT_FLOOR_LARGE: Final[float] = -0.05
PHASE1B_SOFT_FLOOR_NIST: Final[float] = -0.10

PHASE1B_RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results" / "phase1b"
PHASE1B_MODELS_DIR: Final[Path] = MODELS_DIR / "phase1b"

PHASE1B_WANDB_PROJECT: Final[str] = "tract-phase1b"

# The LOFO re-derivation gets its own project. It supersedes the tract-phase1b
# runs rather than extending them: the anchors changed from section titles to
# full control prose, four frameworks were added to the corpus, and the LoRA
# adapter now actually reaches the saved checkpoint. Charting the new folds
# beside the old ones in one project would put two incomparable experiments on
# a single axis, which is the mistake the published card already made once.
LOFO_WANDB_PROJECT: Final[str] = "tract-lofo-rederivation"
LOFO_WANDB_ENTITY: Final[str | None] = None

# A link is worth training on when the text the model sees is substantial.
# Both of the gates this replaces tested link["section_name"], a title the
# model never sees: a framework deny list naming nist_800_63 and
# owasp_proactive_controls, and a 10-character floor on the same field.
# Between them they dropped 278 of 4,405 curated links, 155 by the deny list
# and 123 by the floor, while admitting links that resolved to no control at
# all and trained on their title instead. [measured]
#
# The threshold is unchanged at 10 characters. Only the field it is applied to
# moved, from the title to the anchor the encoder is handed.
PHASE1B_MIN_ANCHOR_TEXT_LENGTH: Final[int] = 10

# Frameworks whose recovered links are a decision rather than a repair. The
# anchor gate restores 44 capec and 16 cwe links that the title floor dropped,
# and those are the terse ones ("UDP Ping", "Fuzzing", "Pharming"). The human
# ceiling study measured capec agreement with OpenCRE at alpha-1 = 0.181
# [0.113, 0.277] on n=83 [measured, results/ceiling_study/panel_agreement.md],
# so recovering its least-agreed stratum is not self-evidently progress.
# filter_training_links(recover_contested=False) is the lever the later
# training-mix decision needs, and it is not entangled with the eleven other
# frameworks' 202 recoveries.
CONTESTED_RECOVERY_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({
    "capec",
    "cwe",
})

# What a caller that passes nothing gets. Both entry points that expose the
# lever default to this one constant, so the shipped decision cannot half-move
# when only one signature is edited.
#
# True recovers the 60 contested links, taking training links to 4,389. It is
# its own commit, and reverting that commit restores this line to False
# without disturbing the eleven other frameworks' 202 net recoveries.
CONTESTED_RECOVERY_DEFAULT: Final[bool] = True

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
PHASE5_CANONICAL_EXPORT_DIR: Final[Path] = PROJECT_ROOT / "canonical_export"
# Default output of `tract export --opencre`. Named here rather than repeated
# as "./opencre_export" at two CLI call sites, so the gitignore gate and the
# CLI cannot disagree about which directory has to be ignored.
PHASE5_OPENCRE_EXPORT_DIR: Final[Path] = PROJECT_ROOT / "opencre_export"

# ── Phase 2B: Bridge Analysis ─────────────────────────────────────────

# All eight AI-security frameworks in the corpus, not just the five that
# rotate through the LOFO roster. ENISA, ETSI and BIML are AI/ML-security
# frameworks -- ENISA maps to "AI model performance validation" and "Anomalous
# AI input handling", BIML to "Data poisoning of train/finetune/augment" and
# "Supply-chain model poisoning" -- and listing only the rotating five made
# `classify_hubs` count them on the TRADITIONAL side. That is what produced the
# published claim of 60 "naturally bridged" hubs whose worked example was
# "Data poisoning (linked by both ATLAS and CWE)": MITRE ATLAS hubs and CWE
# hubs intersect in ZERO hubs, and the traditional side of all of those bridges
# came from ENISA (51), ETSI (28) and BIML (11) and from nothing else.
#
# Under this definition the AI and traditional hub regions are disjoint, which
# is what PRD.md:58 and docs/campaign2-results.md §14 have always recorded.
# Keep this set in step with AI_FRAMEWORK_NAMES; tests/test_ai_framework_sets.py
# asserts they describe the same eight frameworks.
BRIDGE_AI_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({
    "mitre_atlas", "owasp_ai_exchange", "nist_ai_100_2",
    "owasp_llm_top10", "owasp_ml_top10",
    "enisa", "etsi", "biml",
})
BRIDGE_TOP_K: Final[int] = 3
BRIDGE_LLM_MODEL: Final[str] = "claude-sonnet-4-20250514"
BRIDGE_LLM_TEMPERATURE: Final[float] = 0.0
BRIDGE_OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "results" / "bridge"
HIERARCHY_BRIDGE_VERSION: Final[str] = "1.1"

# ── Phase 2B: HuggingFace Publication ─────────────────────────────────

HF_DEFAULT_REPO_ID: Final[str] = "rockCO78/tract-cre-assignment"
HF_DATASET_REPO_ID: Final[str] = "rockCO78/tract-crosswalk-dataset"
HF_STAGING_DIR: Final[Path] = PROJECT_ROOT / "build" / "hf_repo"

HF_MODEL_FILES: Final[tuple[str, ...]] = (
    "model.safetensors", "config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
    "config_sentence_transformers.json", "modules.json",
    "sentence_bert_config.json", "1_Pooling/config.json",
)
HF_DEPLOY_FILES: Final[tuple[str, ...]] = (
    "deployment_artifacts.npz", "calibration.json",
)
HF_DATABASE_FILES: Final[tuple[str, ...]] = (
    "crosswalk.db",
)
HF_BASE_MODEL: Final[str] = "BAAI/bge-large-en-v1.5"
HF_SCAN_EXTENSIONS: Final[frozenset[str]] = frozenset({
    ".py", ".md", ".txt", ".yaml", ".yml", ".json",
})
HF_SECRET_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"hf_[a-zA-Z0-9]{20,}"),
    re.compile(r"wandb_[a-zA-Z0-9]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    # Any home directory, not two specific ones. These were pinned to /home/rock
    # and /Users/rock, the usernames of the original Jetson and its macOS
    # counterpart. This repo now lives under a different account, so a leaked
    # local path would have passed the pre-publication scan unnoticed. Matching
    # the shape of a home path rather than an enumerated list keeps the check
    # working wherever the repo is checked out.
    re.compile(r"/(?:home|Users)/[^/\s\"'`]+"),
    re.compile(r"^pass\s+\w+/\w+", re.MULTILINE),
    re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    re.compile(r"(HF_TOKEN|WANDB_API_KEY|ANTHROPIC_API_KEY)\s*="),
]

# ── Pinned deployment model (lazy auto-download) ──────────────────────
# Bump procedure (run on every HF model re-publish):
#   1. Push new artifacts to HF; note the new full commit SHA.
#   2. python scripts/recompute_model_pins.py <new_sha>   # prints the 5 constants
#   3. Replace the constants below with the printed values; commit.
#   The CI "model-pins" job recomputes these from HF and fails on drift.
TRACT_MODEL_PINNED_REVISION: Final[str] = "2d2095518428b4ae88566bad43e57c9b370eba0c"
TRACT_MODEL_SAFETENSORS_SHA256: Final[str] = (
    "c1f7b6d65c4440ea6b497a47de85898812ebc5efce63f608902de9a4fbe215cd")
TRACT_DEPLOYMENT_ARTIFACTS_SHA256: Final[str] = (
    "7e8b8f834db503118d75727675716471636f139ecb3b64fbd6bc96d6690122f7")
TRACT_CALIBRATION_SHA256: Final[str] = (
    "a49c532d7f8e4d42ff1e5208f68aabdd60d87feb5231c77fbc276c757edda88a")
TRACT_HIERARCHY_SHA256: Final[str] = (
    "8dc48bd397cf6ee455193a9768760258f235fb5519659915dccd733dcaa19738")

# sha256 keyed by the file's path within the snapshot, for download-time
# integrity. EVERY member of TRACT_MODEL_SNAPSHOT_ALLOW_PATTERNS needs an entry
# and _verify_pinned refuses a snapshot carrying a file it has no hash for.
#
# This used to name four: the weights and the three deployment artifacts. The
# other nine were downloaded and consumed unverified, and they are not inert
# padding -- SentenceTransformer imports the classes modules.json names and
# builds the model from config.json, so an altered pair changes every assignment
# the tool produces while the CLI reports a clean integrity check. Pinning the
# weights and leaving the file that decides which code loads them unpinned is
# the wrong half.
#
# Computed 2026-08-29 from the pinned revision; the original four re-verified
# unchanged at the same time.
TRACT_MODEL_PINNED_FILE_HASHES: Final[dict[str, str]] = {
    "model.safetensors": TRACT_MODEL_SAFETENSORS_SHA256,
    "deployment_artifacts.npz": TRACT_DEPLOYMENT_ARTIFACTS_SHA256,
    "calibration.json": TRACT_CALIBRATION_SHA256,
    "cre_hierarchy.json": TRACT_HIERARCHY_SHA256,
    "config.json": (
        "18614f5bf7d7912a48ff06cdf9717ec4f2394fe727278b7c821e895df16f19ff"),
    "tokenizer.json": (
        "91f1def9b9391fdabe028cd3f3fcc4efd34e5d1f08c3bf2de513ebb5911a1854"),
    "tokenizer_config.json": (
        "479a5afc56069a77cc24c74a1943501275664e8d493e215f5716a81ebc0e86db"),
    "special_tokens_map.json": (
        "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a"),
    "vocab.txt": (
        "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"),
    "config_sentence_transformers.json": (
        "fbb6db75971b7f9a254da06349b1f5fa7427666b4ae1170ff398a0c6594622ef"),
    "modules.json": (
        "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf"),
    "sentence_bert_config.json": (
        "65c2293c310f5476a8d1cbada277722c3e7ae5b5ddbaa2d2fe5d075f410d6d02"),
    "1_Pooling/config.json": (
        "31345c23e6f8196484977bf94e465bfe9859101fc4a89d53befad73f776014a9"),
}

TRACT_MODEL_SNAPSHOT_ALLOW_PATTERNS: Final[tuple[str, ...]] = (
    *HF_MODEL_FILES, *HF_DEPLOY_FILES, "cre_hierarchy.json",
)

# ── Pinned crosswalk dataset (`tract download`) ───────────────────────
# The model above is fetched at a pinned revision AND checked against recorded
# digests. crosswalk.db was fetched from a dataset repo's default branch with
# neither -- a deliberate deferral (the lazy-download plan said in as many
# words to leave the dataset download unchanged) that does not survive the
# reason the model pins exist. A HuggingFace tag is mutable on a dataset repo
# exactly as on a model repo, and crosswalk.db is the file every later `tract`
# query reads, so whoever can move that tag rewrites the answers.
#
# These ship UNSET, and UNSET is not "no opinion, carry on": the default-repo
# download REFUSES until a maintainer records both values. A check that waives
# itself when its constant is empty leaves the fetch as unverified as it was
# and says so nowhere. `tract download --model-only` and the lazy resolver
# behind `tract assign` are unaffected.
#
# The digest is NOT copied off whatever crosswalk.db happens to sit in
# results/phase1c/ on a developer's machine. That file's bytes have never been
# confirmed equal to the published artifact, and pinning to a local file would
# record the wrong thing with full confidence.
#
# Bump procedure, mirroring the model's:
#   1. Push crosswalk.db to the dataset repo; note that repo's full commit SHA.
#   2. python scripts/recompute_model_pins.py --dataset <that_sha>
#      (fetches the published file and prints both constants; it also compares
#      against any local copy so a divergence is seen before it is pinned)
#   3. Replace the two constants below with the printed values; commit.
#   tests/test_model_pins_consistency.py then holds them against the Hub in the
#   CI "model-pins" job, which is what keeps this pin from rotting the way the
#   original deferral did.
TRACT_PIN_UNSET: Final[str] = "UNSET"
# Verified 2026-08-29 against the published artifact, not against the local
# copy: fetched from the dataset repo at this revision into a temp directory and
# hashed there. It matches the crosswalk.db already on disk byte for byte, but
# the point is that the comparison was made -- pinning to a local digest nobody
# had checked against the published file would record the wrong thing with full
# confidence.
TRACT_DATASET_PINNED_REVISION: Final[str] = (
    "57930dcae45503956a1510ac72e3f57bef215764"
)
TRACT_CROSSWALK_DB_SHA256: Final[str] = (
    "e9ddba3596399ea48e17223519c73bd77e13c31218203de2029552e945356e29"
)

# Keyed by basename, matching TRACT_MODEL_PINNED_FILE_HASHES above, so a name
# added to HF_DATABASE_FILES without a digest beside it is a lookup miss the
# download path refuses on rather than a file that quietly skips the check.
TRACT_DATASET_PINNED_FILE_HASHES: Final[dict[str, str]] = {
    "crosswalk.db": TRACT_CROSSWALK_DB_SHA256,
}

# Environment overrides for the dataset fetch, named here so the CLI's refusal
# messages and the code that reads them cannot drift apart. Unlike the model's
# overrides, naming a different repo or revision does NOT by itself downgrade
# to revision-trust: the operator must either supply the digest they expect
# (TRACT_DATASET_SHA256) or say out loud that they want no check at all
# (TRACT_DATASET_ALLOW_UNVERIFIED=1). One environment variable must not be able
# to restore the unpinned fetch this pin exists to end.
TRACT_DATASET_REPO_ID_ENV: Final[str] = "TRACT_DATASET_REPO_ID"
TRACT_DATASET_REVISION_ENV: Final[str] = "TRACT_DATASET_REVISION"
TRACT_DATASET_SHA256_ENV: Final[str] = "TRACT_DATASET_SHA256"
TRACT_DATASET_ALLOW_UNVERIFIED_ENV: Final[str] = "TRACT_DATASET_ALLOW_UNVERIFIED"

# ── CLI exit codes (scriptable failure classes) ───────────────────────
EXIT_USER_ERROR: Final[int] = 2
EXIT_OFFLINE: Final[int] = 3
EXIT_INTEGRITY: Final[int] = 4
EXIT_MISSING_RUNTIME: Final[int] = 5

PHASE1B_TEXTAWARE_RESULTS_DIR: Final[Path] = (
    PROJECT_ROOT / "results" / "phase1b" / "phase1b_textaware"
)
PHASE1B_CORRECTED_METRICS_PATH: Final[Path] = (
    PROJECT_ROOT / "results" / "phase1b" / "phase1b_textaware" / "corrected_metrics.json"
)
PHASE1C_ECE_GATE_PATH: Final[Path] = (
    PHASE1C_RESULTS_DIR / "calibration" / "ece_gate.json"
)

# ── Phase 3: Crosswalk Dataset Publication ────────────────────────────

PHASE3_REVIEW_OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "results" / "review"
PHASE3_DATASET_STAGING_DIR: Final[Path] = PROJECT_ROOT / "build" / "dataset"
PHASE3_DATASET_REPO_ID: Final[str] = "rockCO78/tract-crosswalk-dataset"

PHASE3_CALIBRATION_SEED: Final[int] = 42
PHASE3_CALIBRATION_N_ITEMS: Final[int] = 20
PHASE3_CALIBRATION_EASY_N: Final[int] = 5
PHASE3_CALIBRATION_HARD_N: Final[int] = 5

PHASE3_TEXT_QUALITY_HIGH_THRESHOLD: Final[int] = 500
PHASE3_TEXT_QUALITY_LOW_THRESHOLD: Final[int] = 100

PHASE3_UNCOVERED_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({
    "aiuc_1", "cosai", "eu_gpai_cop", "nist_ai_rmf", "owasp_dsgai",
})

PHASE3_GT_PROVENANCE: Final[str] = "opencre_ground_truth"
PHASE3_MODEL_PROVENANCE: Final[str] = "model_prediction"

PHASE3_PROVENANCE_PRIORITY: Final[list[str]] = [
    "opencre_ground_truth",
    "ground_truth_T1-AI",
    "active_learning_round_2",
    "model_prediction",
]

# ── Part 0.1: Ceiling study (blind expert-agreement) ──────────────────
# Design doc: docs/superpowers/specs/2026-08-15-semantic-rebuild-design.md.
# The 13/20 hidden-calibration datum (PHASE3_CALIBRATION_N_ITEMS above) has a
# Wilson half-width of 0.193, too wide to gate anything. This study replaces
# it with n=250, powered to a half-width of 0.059 at alpha ~= 0.65.
CEILING_STUDY_DIR: Final[Path] = PROJECT_ROOT / "results" / "ceiling_study"
CEILING_STUDY_SEED: Final[int] = 42
CEILING_STUDY_N_ITEMS: Final[int] = 250
CEILING_STUDY_STRATUM_SIZE: Final[int] = 125
CEILING_STUDY_MAX_ACCEPTABLE_HUBS: Final[int] = 5

# Ruling R22. The 250 items a domain expert annotated by hand live in exactly
# one file, and that file is the study of record. build_ceiling_study() draws
# from the live curated-link pool, so it answers "what would a study drawn
# today look like", never "what did the owner score". Those two answers
# separated when Task 14 moved the link gates, and nothing said so.
CEILING_STUDY_PINNED_ITEMS: Final[Path] = CEILING_STUDY_DIR / "ceiling_items.json"
CEILING_STUDY_PROVENANCE_PATH: Final[Path] = (
    CEILING_STUDY_DIR / "ceiling_study_provenance.json"
)
# A fresh draw is a NEW study with its own name. It lands here, one
# subdirectory per name, so no new draw can share a path with the pinned
# artifact even by a typo.
CEILING_STUDY_NEW_DIR: Final[Path] = CEILING_STUDY_DIR / "studies"

# Only frameworks whose text is stable under the pending corpus rebuild are
# eligible. The other 15 curated frameworks either have a parser landing
# (biml, csa_ccm, dsomm, enisa, etsi, iso_27001, nist_800_63, nist_ssdf,
# owasp_proactive_controls, owasp_top10_2021, samm, wstg) or a source being
# re-pinned (asvs, owasp_cheat_sheets, owasp_ml_top10), and sampling their
# text now would draw items whose control_text will not match what the model
# is eventually trained and evaluated against.
CEILING_STUDY_VALIDATION_FRAMEWORKS: Final[tuple[str, ...]] = (
    "capec", "cwe", "nist_800_53",
)
CEILING_STUDY_TEST_FRAMEWORKS: Final[tuple[str, ...]] = (
    "mitre_atlas", "owasp_ai_exchange", "nist_ai_100_2", "owasp_llm_top10",
)

# The half-width the study was powered to (n=250 at alpha ~= 0.65, see the
# design doc table). The scorer reports this alongside the achieved
# half-width so a wider-than-planned result is visible rather than silent.
CEILING_STUDY_TARGET_HALF_WIDTH: Final[float] = 0.059

# ── LLM judge panel ──────────────────────────────────────────────────────
# Three model families answer the same blind annotation prompt the human
# answered, to separate "OpenCRE's CAPEC links are poor" from "one human's
# reading of CAPEC is idiosyncratic". Three families rather than three
# checkpoints of one family: correlated pretraining would make agreement
# between two judges evidence of shared lineage rather than of the label.
#
# Every candidate route is OpenAI-compatible, so one client covers all of
# them and the only per-route variables are the base URL, the model id
# spelling, and where the key lives. Ordered by preference: an aggregator
# reaches all three with one key, per-vendor keys are cheapest, and the
# HuggingFace router is the fallback that needs no new vendor relationship.
#
# The HF router was verified working on 2026-08-18 against all three models
# and is the only route with a credential already on this machine. It is
# last in preference because its free monthly allowance is small, not
# because it does not work.
PANEL_ROUTES: Final[dict[str, dict[str, str]]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "pass_entry": "openrouter/api-key",
        "env_var": "OPENROUTER_API_KEY",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.ai/v1",
        "pass_entry": "moonshot/api-key",
        "env_var": "MOONSHOT_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "pass_entry": "deepseek/api-key",
        "env_var": "DEEPSEEK_API_KEY",
    },
    "zhipu": {
        "base_url": "https://api.z.ai/api/paas/v4",
        "pass_entry": "zhipu/api-key",
        "env_var": "ZHIPU_API_KEY",
    },
    "hf_router": {
        "base_url": "https://router.huggingface.co/v1",
        "pass_entry": "huggingface/read-token",
        "env_var": "HF_TOKEN",
    },
}

# Per-route model id spellings. The same weights are called different things
# by different resellers, and sending the wrong spelling is a 404 rather
# than a silent substitution, which is the failure mode worth having.
PANEL_MODEL_IDS: Final[dict[str, dict[str, str]]] = {
    "moonshotai/Kimi-K3": {
        "openrouter": "moonshotai/kimi-k3",
        "moonshot": "kimi-k3",
        "hf_router": "moonshotai/Kimi-K3:fireworks-ai",
    },
    "z-ai/GLM-5.3": {
        "openrouter": "z-ai/glm-5.3",
        "zhipu": "glm-5.3",
    },
    "deepseek-ai/DeepSeek-V4-Pro": {
        "openrouter": "deepseek/deepseek-v4-pro",
        "deepseek": "deepseek-v4-pro",
        "hf_router": "deepseek-ai/DeepSeek-V4-Pro:fireworks-ai",
    },
    "meta-llama/Llama-4-Maverick": {
        "openrouter": "meta-llama/llama-4-maverick",
    },
    "x-ai/Grok-4.20": {
        "openrouter": "x-ai/grok-4.20",
    },
}

# OpenRouter load-balances one model id across several backends, and those
# backends differ in quantization. The same prompt served at fp4 and at bf16
# is not the same judge. Each model is pinned to one backend with fallbacks
# disabled, so a routing decision cannot silently become an experimental
# variable. Chosen for full context length first, then for the least lossy
# quantization available, then for first-party serving.
PANEL_OPENROUTER_PROVIDER_PIN: Final[dict[str, str]] = {
    # DeepInfra's bf16 endpoint is the only unquantized one and was the first
    # pin, but kimi-k3 is rate-limited upstream across OpenRouter's shared
    # pool and it never cleared. BaseTen's fp8 is the next best, and it has
    # the side benefit of matching the fp8 the other quantized members run
    # at, so precision is one fewer difference across the panel.
    "moonshotai/Kimi-K3": "baseten/fp8",
    "z-ai/GLM-5.3": "Z.AI",                  # fp8, first party, sole provider
    # DeepSeek's own endpoint 404s for this account at any token budget, so
    # the pin is the best third party: fp8 at 1M context, not the fp4 ones.
    "deepseek-ai/DeepSeek-V4-Pro": "streamlake",
    "meta-llama/Llama-4-Maverick": "DeepInfra",  # fp8 at 1M, others cap at 128k
    "x-ai/Grok-4.20": "xAI",                 # first party, sole provider
}

# Pinned when the route is the HF router, so the serving provider is not a
# free variable there either.
PANEL_PROVIDER: Final[str] = "fireworks-ai"

# Greedy decoding. The study measures where a model family lands, not the
# spread of its sampling distribution, and a judge that answers differently
# on re-run cannot be audited against the file committed alongside it.
PANEL_TEMPERATURE: Final[float] = 0.0

# Every panel member is a thinking model and reasoning tokens bill at the
# output rate. Applied to all five rather than only to the expensive ones,
# so deliberation budget stays constant across the panel.
#
# Sent as OpenRouter's unified `reasoning: {"effort": ...}` object. The
# OpenAI-style `reasoning_effort` string is NOT honoured on that route: a
# control call sent with `reasoning_effort="low"` spent all 3,000 of its
# allowed tokens on reasoning, while the same call with the object form
# spent 560. Sending the wrong spelling does not error, it just silently
# runs at full effort and bills for it.
PANEL_REASONING_EFFORT: Final[str] = "low"

# 25 items per request, matching the batch size the runbook specifies for
# the human study. All three models have a 1M-token context and could take
# all 250 items at once, so this is about answer quality across a long list
# and about comparability with the human protocol, not about context.
PANEL_BATCH_SIZE: Final[int] = 25

# Set by the tightest endpoint on the panel, not by preference. OpenRouter
# filters out any backend whose max_completion_tokens is below the requested
# max_tokens, and DeepInfra caps Kimi and Llama at 16,384. Asking for more
# does not raise a parameter error, it returns "No endpoints found" and
# looks like the model does not exist.
#
# Uniform across all five so deliberation budget is not a variable, and
# large enough for a 25-item batch: measured 2,475 completion tokens for 3
# items at the verbose end, and reasoning at "low" is bounded.
PANEL_MAX_TOKENS: Final[int] = 16384

PANEL_TIMEOUT_S: Final[int] = 900

# Tuned for the failure actually seen: an upstream provider 429 on a pinned
# backend, where fallbacks are deliberately disabled so there is nowhere
# else to go and the only cure is waiting. A 4-attempt, 4-second-base
# schedule gives up after about a minute, which is far too soon. This one
# spans roughly twenty minutes.
PANEL_MAX_RETRIES: Final[int] = 6
PANEL_RETRY_BASE_DELAY_S: Final[float] = 20.0

# Five, not three, and odd on purpose: an even panel can split 2-2 on
# exactly the contested CAPEC items the study turns on, and a tie there has
# no majority to compare the human against. Five distinct labs, three
# Chinese and two American, so that convergence cannot be dismissed as one
# training-data monoculture agreeing with itself.
PANEL_MODELS: Final[tuple[str, ...]] = (
    "moonshotai/Kimi-K3",
    "z-ai/GLM-5.3",
    "deepseek-ai/DeepSeek-V4-Pro",
    "meta-llama/Llama-4-Maverick",
    "x-ai/Grok-4.20",
)

# (input, output) USD per million tokens, per route, read from the
# provider's posted rate for the pinned backend. Used only for the dry-run
# estimate: the executed run records `usage.cost` returned by OpenRouter on
# every response, which is the amount actually billed.
PANEL_PRICING_USD_PER_MTOK: Final[dict[str, dict[str, tuple[float, float]]]] = {
    "moonshotai/Kimi-K3": {
        "openrouter": (2.85, 14.25),
        "moonshot": (3.00, 15.00),
        "hf_router": (3.00, 15.00),
    },
    "z-ai/GLM-5.3": {
        "openrouter": (1.40, 4.40),
        "zhipu": (1.40, 4.40),
    },
    "deepseek-ai/DeepSeek-V4-Pro": {
        "openrouter": (0.66, 1.98),
        "deepseek": (0.435, 0.87),
        "hf_router": (1.74, 3.48),
    },
    "meta-llama/Llama-4-Maverick": {
        "openrouter": (0.20, 0.80),
    },
    "x-ai/Grok-4.20": {
        "openrouter": (1.25, 2.50),
    },
}

# The probe framework for contamination. Published after every panel
# member's training cutoff, so memorised OpenCRE mappings cannot cover it.
PANEL_CONTAMINATION_PROBE_FRAMEWORK: Final[str] = "owasp_llm_top10"

# Written per fold, and the only artifact carrying the per-item hit@1
# indicators needed to micro-average across folds. It lives here rather than
# in tract.training.orchestrate because the RunPod orchestrator has to name
# this file to verify a collection, and it runs on a machine that has no
# training stack installed. Importing orchestrate there pulls torch and
# datasets into a path that must work without them.
FOLD_RESULT_FILENAME: Final[str] = "fold_result.json"
