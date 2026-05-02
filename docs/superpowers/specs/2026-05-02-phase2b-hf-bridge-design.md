# Phase 2B: AI/Traditional Bridge Analysis + HuggingFace Publication

> **For agentic workers:** This spec covers two workstreams with a hard dependency: bridge analysis MUST complete before HuggingFace publication begins. The published model artifact includes bridge results.

**Goal:** Identify conceptual bridges between AI-specific and traditional CRE hubs, then publish the complete TRACT model to HuggingFace with a merged full model, bundled inference data, and an AIBOM-compliant model card.

**PRD Sections:** 7.3 (HuggingFace Publication), 7.4 (AI/Traditional Security Bridge)

**Dependencies:** Phase 1C (deployment model, calibration, crosswalk DB), Phase 1D (hub proposals, OOD analysis)

**Hard dependency chain:** Bridge Analysis → HuggingFace Publication (publication gate checks for completed bridge_report.json)

---

## 1. Bridge Analysis

### 1.1 Problem Statement

The CRE ontology has 81 AI-specific hubs (linked from MITRE ATLAS, OWASP AI Exchange, NIST AI 100-2, OWASP LLM Top 10, OWASP ML Top 10) and 442 traditional hubs (linked from CAPEC, CWE, NIST 800-53, ASVS, etc.). 60 hubs are already linked from both domains — natural bridges. 21 hubs are AI-only. 382 are traditional-only.

Bridge analysis identifies conceptual overlaps between AI and traditional security hubs that aren't captured by existing framework links, then adds them as `Related` links in the CRE hierarchy.

### 1.2 Hub Classification

Hubs are classified by which frameworks link to them in `data/training/hub_links_by_framework.json`:

| Category | Count | Definition |
|---|---|---|
| AI-linked | 81 | Linked from at least one AI framework |
| Traditional-linked | 442 | Linked from at least one traditional framework |
| Both (known bridges) | 60 | Linked from both AI and traditional frameworks |
| AI-only | 21 | Linked from AI frameworks only, no traditional links |
| Traditional-only | 382 | Linked from traditional frameworks only, no AI links |

AI framework IDs: `mitre_atlas`, `owasp_ai_exchange`, `nist_ai_100_2`, `owasp_llm_top10`, `owasp_ml_top10`.

### 1.3 Threshold Calibration from Known Bridges

The 60 known bridge hubs serve as ground truth for calibrating the discovery threshold. Each hub has a single embedding in `deployment_artifacts.npz`. The calibration approach:

1. For each of the 60 known bridge hubs, find its nearest neighbor among the 382 traditional-only hubs (by cosine similarity of hub embeddings). This measures "how close do known bridge hubs get to the purely-traditional part of the embedding space?"
2. Compute the distribution of these 60 nearest-neighbor distances.
3. Set the discovery threshold at `mean - 1σ` of this distribution, floored at `BRIDGE_MIN_SIMILARITY` (0.50).
4. Document the empirical distribution (mean, std, min, max, quartiles) in bridge_report.json.

This gives a principled threshold grounded in how close known bridges get to traditional hubs, rather than an arbitrary cutoff.

### 1.4 Bridge Discovery Pipeline

```
deployment_artifacts.npz (522 hub embeddings, 1024-dim)
  + hub_links_by_framework.json (framework→hub mappings)
  + cre_hierarchy.json (hub structure)
  ↓
classify_hubs() → {ai_only: 21, trad_only: 382, known_bridges: 60}
  ↓
calibrate_threshold() → empirical threshold from 60 known bridges
  ↓
compute_bridge_similarities() → 21×382 cosine matrix
  ↓
filter_candidates() → ranked pairs above threshold
  ↓
generate_descriptions() → LLM bridge descriptions (Claude API, sanitize_text applied)
  ↓
bridge_candidates.json → expert review (accept/reject in JSON file)
  ↓
commit_bridges() → updated cre_hierarchy.json + bridge_report.json
```

### 1.5 Bridge Candidate JSON Format

```json
{
  "generated_at": "2026-05-02T...",
  "threshold": 0.682,
  "threshold_method": "known_bridge_nearest_neighbor_mean_minus_1std",
  "known_bridge_stats": {
    "count": 60,
    "mean_nearest_trad_similarity": 0.743,
    "std": 0.061,
    "min": 0.612,
    "max": 0.891
  },
  "candidates": [
    {
      "ai_hub_id": "062-850",
      "ai_hub_name": "Data poisoning of train/finetune/augmentation data",
      "trad_hub_id": "457-210",
      "trad_hub_name": "Input validation",
      "cosine_similarity": 0.734,
      "seed_evidence": {
        "shared_frameworks": [],
        "ai_controls_linked": 5,
        "trad_controls_linked": 12,
        "note": "shared_frameworks lists ENISA/ETSI/BIML if they link to both hubs; empty for most AI-only candidates since they lack traditional links by definition"
      },
      "description": "Both hubs address the integrity of data entering a system...",
      "status": "pending",
      "reviewer_notes": ""
    }
  ]
}
```

Review workflow: user edits `status` to `"accepted"` or `"rejected"`, optionally adds `reviewer_notes`. Then runs `tract bridge --commit`.

### 1.6 Hierarchy Update

Accepted bridges are added to `cre_hierarchy.json` as `Related` links between the two hubs (bidirectional). This does NOT change:
- Model weights (trained on control→hub assignments, not hub→hub)
- Hub embeddings (encoded from name + hierarchy path; Related links are lateral, not hierarchical)
- Calibration (T, thresholds fitted on model output distribution)
- deployment_artifacts.npz (pre-computed from the above)

The updated hierarchy is bundled with the HuggingFace publication, giving downstream users a richer ontology.

### 1.7 OOD Cross-Reference

Phase 1D identified 95 OOD controls (3.4%) that didn't map well to any existing hub. 5 hub proposals passed guardrails. Cross-reference OOD controls against bridge candidates: if OOD controls cluster near a bridge pair, this is supporting evidence that the bridge captures a real concept. Logged in bridge_report.json but does not affect accept/reject decisions (that's expert judgment).

---

## 2. HuggingFace Publication

### 2.1 Publication Gate

`publish_to_huggingface()` checks before proceeding:
1. `bridge_report.json` exists
2. All candidates in bridge_report.json have status `accepted` or `rejected` (no `pending`)
3. `cre_hierarchy.json` has been updated with accepted bridges

If any check fails, raise `ValueError` with specific message.

### 2.2 Model Merge

Merge LoRA adapters into base model to produce a standalone ~1.3GB model:

- **Input:** `results/phase1c/deployment_model/model/model/` (adapter_config.json + adapter_model.safetensors) + `BAAI/bge-large-en-v1.5` (downloaded from HuggingFace)
- **Process:** Load base model, load PEFT adapters, merge with `model.merge_and_unload()`, save as safetensors
- **Output:** Complete model directory with model.safetensors, config.json, tokenizer files, sentence_bert_config.json, modules.json, 1_Pooling/config.json
- **Verification:** No adapter_config.json in output (fully merged). Model produces near-identical embeddings to LoRA-loaded model (cosine similarity > 0.9999 for test inputs; minor float divergence from merge rounding is expected).

### 2.3 Inference Data Bundle

Files bundled alongside the model (all validated before copy):

| File | Source | Purpose |
|---|---|---|
| `hub_descriptions.json` | `data/processed/hub_descriptions_reviewed.json` | Hub semantic descriptions for display/search |
| `cre_hierarchy.json` | `data/processed/cre_hierarchy.json` (post-bridge) | Hub structure with bridge Related links |
| `calibration.json` | `results/phase1c/deployment_model/calibration.json` | T, thresholds, conformal quantile |
| `hub_ids.json` | Extracted from deployment_artifacts.npz | Ordered hub ID list matching model output dimensions |
| `bridge_report.json` | `results/bridge/bridge_report.json` | Bridge analysis summary and evidence |

Total additional size: ~5MB on top of ~1.3GB model.

### 2.4 Model Card (README.md)

AIBOM-compliant model card targeting 100/100 score. Content sourced from verified artifact paths:

**Model Description:**
- TRACT: Transitive Reconciliation and Assignment of CRE Taxonomies
- Maps security framework control text to OpenCRE hub positions via bi-encoder
- Assignment paradigm: `g(control_text) → CRE_position`, not pairwise
- 522 CRE hubs, 400 leaf hubs as label space

**Architecture:**
- Base: BAAI/bge-large-en-v1.5 (335M params)
- Fine-tuning: LoRA rank=16, alpha=32, dropout=0.1, target modules: query/key/value
- Training: MNRL contrastive loss, text-aware batch sampling, 20 epochs, batch=32, lr=5e-4, seed=42
- Training data: 4,237 links → 4,061 training pairs from 22 OpenCRE-linked frameworks

**Evaluation (LOFO cross-validation):**

| Fold | hit@1 | Zero-shot | Delta | n |
|---|---|---|---|---|
| MITRE ATLAS | 0.279 | 0.273 | +0.006 | 43 |
| NIST AI 100-2 | 0.429 | 0.107 | +0.322 | 28 |
| OWASP AI Exchange | 0.762 | 0.619 | +0.143 | 63 |
| OWASP LLM Top 10 | 0.333 | 0.333 | +0.000 | 6 |
| OWASP ML Top 10 | 0.714 | 0.429 | +0.285 | 7 |
| **Micro average** | **0.531** | **0.399** | **+0.132** | **147** |

Bootstrap CIs included from fold summary files.

**Calibration:**
- Temperature scaling: T=0.074
- ECE=0.079, CI [0.049, 0.111]
- OOD threshold=0.568 (96.7% separation rate)
- Conformal coverage: quantile=0.997

**Limitations (documented honestly):**
- ATLAS fold shows near-zero improvement (+0.006) — hub disambiguation is the failure mode
- ECE=0.079 indicates imperfect calibration; confidence scores are ordinal rankings, not true probabilities
- 35% of controls map to multiple hubs — predictions are multi-label by design
- Calibrated on 420 traditional framework holdout items; accuracy on AI-specific text may differ
- DeBERTa-v3-NLI completely fails for this task (hit@1=0.000) — NLI is not semantic similarity

**Ethical Considerations:**
- Not a replacement for expert judgment in compliance decisions
- Model predictions require human review before use in security assessments
- Active learning rounds used expert-reviewed predictions, not autonomous deployment

**Environmental Impact:**
- Training: H100 GPU via RunPod. Source actual hours from RunPod billing dashboard or WandB run metadata (total training time across LOFO folds + deployment model). Estimate CO2 using ML CO2 Impact calculator with US-West grid factor.
- Deployment: runs on Jetson Orin AGX (edge device, ~30W TDP)

**Usage snippet, citation (BibTeX), license (CC0 1.0)**

**Bridge Analysis Summary:** Included as a section — number of bridges found, methodology, link to bridge_report.json.

### 2.5 Standalone Scripts

**predict.py:**
- Loads merged model + bundled data from repo directory
- Takes control text as input, returns top-K hub predictions with calibrated confidence
- Dependencies: `sentence-transformers`, `torch`, `numpy` (no TRACT package required)
- Handles: single text and batch mode

**train.py:**
- Documents full reproduction pipeline
- Points to TRACT GitHub repo for training code
- Lists pinned requirements
- Includes data download instructions (OpenCRE API fetch)
- Includes exact training command with all hyperparameters

### 2.6 Security Scan

Automated scan before upload — regex patterns for:
- API keys: `sk-`, `hf_`, `key-`, `token-` followed by alphanumeric strings
- File paths containing usernames: `/home/rock`, `/Users/`
- Email addresses
- `pass ` commands (credential manager invocations)
- Environment variable references to secrets

Scan runs against ALL files in the staging directory. Any match = hard failure with file path and line number. No manual override — fix the source and re-run.

### 2.7 AIBOM Validation

Final step before upload:
1. Clone `GenAI-Security-Project/aibom-generator` to a temp directory
2. Run against the generated README.md
3. Report score and any missing fields
4. Target: 100/100

If the tool is unavailable or broken, log a warning and proceed (the model card was written to spec). Do not block publication on external tool availability.

---

## 3. CLI Commands

### 3.1 `tract bridge`

```
tract bridge [--output-dir DIR] [--threshold FLOAT] [--skip-descriptions]
```

- `--output-dir`: output directory for bridge_candidates.json and bridge_report.json (default: `results/bridge/`)
- `--threshold`: override auto-calibrated threshold (optional)
- `--skip-descriptions`: skip LLM description generation (for testing/dry runs)
- Prints: threshold used, number of candidates found, path to bridge_candidates.json
- Does NOT auto-commit bridges — requires explicit `--commit`

```
tract bridge --commit --candidates results/bridge/bridge_candidates.json
```

- Reads reviewed candidates file
- Validates all entries have non-pending status
- Accepted bridges → Related links in cre_hierarchy.json (atomic write)
- Writes bridge_report.json with full statistics
- Prints: N accepted, N rejected, hierarchy updated

### 3.2 `tract publish-hf`

```
tract publish-hf --repo-id rockCO78/tract-cre-assignment [--staging-dir DIR] [--dry-run] [--skip-upload]
```

- `--staging-dir`: local build directory (default: `build/hf_repo/`)
- `--dry-run`: build + security scan + AIBOM validation, no upload
- `--skip-upload`: build + security scan only
- Gate: checks bridge_report.json before proceeding
- HuggingFace token: `pass huggingface/token`
- Prints: staging dir path, security scan result, AIBOM score, upload status

---

## 4. Module Structure

```
tract/
  bridge/
    __init__.py          # run_bridge_analysis() orchestrator
    classify.py          # classify_hubs() — AI/trad/both split
    similarity.py        # compute_bridge_similarities(), calibrate_threshold()
    describe.py          # generate_bridge_descriptions() — LLM + sanitize_text
    review.py            # load_candidates(), commit_bridges() — JSON I/O, hierarchy update
  publish/
    __init__.py          # publish_to_huggingface() orchestrator, bridge gate
    merge.py             # merge_lora_adapters() — PEFT merge + safetensors export
    bundle.py            # bundle_inference_data() — copy/validate data files
    model_card.py        # generate_model_card() — templated README.md
    scripts.py           # write_predict_script(), write_train_script()
    security.py          # scan_for_secrets() — regex scan
```

---

## 5. Config Constants

Added to `tract/config.py`:

```python
# Bridge analysis
BRIDGE_AI_FRAMEWORK_IDS: frozenset[str] = frozenset({
    "mitre_atlas", "owasp_ai_exchange", "nist_ai_100_2",
    "owasp_llm_top10", "owasp_ml_top10",
})
BRIDGE_THRESHOLD_SIGMA: float = 1.0  # mean - N*std for threshold
BRIDGE_MIN_SIMILARITY: float = 0.50  # absolute floor regardless of calibration
BRIDGE_LLM_MODEL: str = "claude-sonnet-4-20250514"
BRIDGE_LLM_TEMPERATURE: float = 0.0
BRIDGE_OUTPUT_DIR: Path = Path("results/bridge")

# HuggingFace publication
HF_DEFAULT_REPO_ID: str = "rockCO78/tract-cre-assignment"
HF_STAGING_DIR: Path = Path("build/hf_repo")
HF_BASE_MODEL: str = "BAAI/bge-large-en-v1.5"
HF_SECRET_PATTERNS: list[str] = [
    r"sk-[a-zA-Z0-9]{20,}",
    r"hf_[a-zA-Z0-9]{20,}",
    r"/home/\w+",
    r"/Users/\w+",
    r"pass\s+\w+/\w+",
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
]
```

---

## 6. Testing Strategy

### Bridge Tests (`tests/test_bridge/`)

| Test file | Scope |
|---|---|
| `test_classify.py` | Hub classification counts. Edge cases: hub with zero links, hub in neither set. |
| `test_similarity.py` | Cosine matrix shape (21×382). Threshold calibration produces value in (0,1). Candidates sorted descending. Deterministic output. |
| `test_describe.py` | sanitize_text applied to descriptions. Non-empty output. API failure retry. |
| `test_review.py` | JSON round-trip. Commit rejects pending entries. Accepted → Related links. Rejected excluded. Atomic writes. |
| `test_bridge_integration.py` | End-to-end with synthetic fixture: 3 AI hubs, 5 trad hubs, 1 known bridge → candidates → commit → hierarchy updated. |

### Publish Tests (`tests/test_publish/`)

| Test file | Scope |
|---|---|
| `test_merge.py` | Merged model has model.safetensors, no adapter_config.json. |
| `test_bundle.py` | All required files copied. Missing file → ValueError. Hierarchy includes bridges. |
| `test_model_card.py` | All AIBOM sections present. Metrics match source. Limitations honest. No secrets. |
| `test_security.py` | Planted secrets detected. Clean dir passes. |
| `test_publish_integration.py` | End-to-end --dry-run passes. Bridge gate blocks without bridge_report.json. |

### Fixtures

- Synthetic hierarchy: ~10 hubs with fake 1024-dim embeddings, known bridge relationship
- All tests run without GPU, real model, or network access (mocked where needed)

---

## 7. Dependencies

**New pip dependencies:**
- `peft` — LoRA merge (already available for training, may need explicit addition)
- `huggingface_hub` — upload API

**External tools (not pip):**
- `GenAI-Security-Project/aibom-generator` — cloned at validation time, not a persistent dependency

**Existing dependencies used:**
- `sentence-transformers` — model loading
- `anthropic` — LLM bridge descriptions (optional, via `tract[llm]`)
- `numpy` — cosine similarity computation

---

## 8. Out of Scope

- **Model retraining with bridge signal.** Bridges are metadata additions to the hierarchy. The PRD's "retrain with bridge links" (Section 7.4, step 8) is a future experiment, not a prerequisite.
- **Web UI integration.** Phase 2A is independent.
- **Automated bridge discovery beyond hub embeddings.** No control-text-based bridge detection, no graph traversal — hub embedding similarity is the signal.
- **HuggingFace Datasets publication.** That's Phase 3 (published crosswalk dataset), not Phase 2B.
- **OpenCRE upstream submission of bridges.** That's Phase 5B.
