# Phase 2B: AI/Traditional Bridge Analysis + HuggingFace Publication

> **For agentic workers:** This spec covers two workstreams with a hard dependency: bridge analysis MUST complete before HuggingFace publication begins. The published model artifact includes bridge results.

**Goal:** Identify conceptual bridges between AI-specific and traditional CRE hubs, then publish the complete TRACT model to HuggingFace with a merged full model, bundled inference data, and an AIBOM-compliant model card.

**PRD Sections:** 7.3 (HuggingFace Publication), 7.4 (AI/Traditional Security Bridge)

**Dependencies:** Phase 1C (deployment model, calibration, crosswalk DB)

**Hard dependency chain:** Bridge Analysis → HuggingFace Publication (publication gate checks for completed bridge_report.json)

---

## 1. Bridge Analysis

### 1.1 Problem Statement

The CRE ontology has 81 AI-specific hubs (linked from MITRE ATLAS, OWASP AI Exchange, NIST AI 100-2, OWASP LLM Top 10, OWASP ML Top 10) and 442 traditional hubs (linked from CAPEC, CWE, NIST 800-53, ASVS, etc.). 60 hubs are already linked from both domains — natural bridges. 21 hubs are AI-only. 382 are traditional-only.

Bridge analysis identifies conceptual overlaps between AI and traditional security hubs that aren't captured by existing framework links, then records them as `related_hub_ids` in the CRE hierarchy (bidirectional).

### 1.2 Hub Classification

Hubs are classified by which frameworks link to them in `data/training/hub_links_by_framework.json`:

| Category | Count | Definition |
|---|---|---|
| AI-linked | 81 | Linked from at least one AI framework |
| Traditional-linked | 442 | Linked from at least one traditional framework |
| Both (naturally bridged) | 60 | Linked from both AI and traditional frameworks |
| AI-only | 21 | Linked from AI frameworks only, no traditional links |
| Traditional-only | 382 | Linked from traditional frameworks only, no AI links |
| Unlinked | 59 | Have embeddings in deployment_artifacts.npz but no framework links (55 non-leaf internal nodes, 4 leaves). Excluded from bridge analysis — logged in bridge_report.json as "unclassified" for transparency. |

AI framework IDs: `mitre_atlas`, `owasp_ai_exchange`, `nist_ai_100_2`, `owasp_llm_top10`, `owasp_ml_top10`.

Note: 522 hubs have embeddings total. 463 appear in hub_links_by_framework.json. The 59 unlinked hubs are mostly organizational grouping nodes without direct framework connections.

### 1.3 Similarity Computation (No Threshold)

Hub embeddings in `deployment_artifacts.npz` are 1024-dim unit vectors (all norms = 1.0), so cosine similarity = dot product. The bridge pipeline computes the full 21×382 cosine matrix and extracts the **top-K traditional matches per AI-only hub** (default K=3, configurable via `--top-k`).

No automated threshold is applied. Empirical analysis showed:
- Mann-Whitney U test (p=0.374): AI-only and naturally-bridged hub similarity distributions to traditional hubs are statistically indistinguishable. No threshold can separate "real bridges" from "non-bridges" using embedding similarity alone.
- Known bridges like "Data poisoning" → "Weakening training set backdoors" sit at cosine 0.51 — below any reasonable threshold.
- At threshold 0.55, only 10/21 AI-only hubs would get any match, missing obvious bridges like "Testing against direct prompt injection" → "Manual penetration testing" (0.54).

With K=3, the pipeline produces 63 candidates — reviewable by an expert in ~30 minutes. The similarity score is a **ranking signal for the expert**, not a classifier. The expert accepts or rejects each candidate based on domain knowledge.

### 1.4 Bridge Discovery Pipeline

```
deployment_artifacts.npz (522 hub embeddings, 1024-dim, unit-normalized)
  + hub_links_by_framework.json (framework→hub mappings)
  + cre_hierarchy.json (hub structure)
  ↓
classify_hubs() → {ai_only: 21, trad_only: 382, naturally_bridged: 60, unlinked: 59}
  ↓
compute_bridge_similarities() → 21×382 cosine matrix (= dot product, unit vectors)
  ↓
extract_top_k() → top-3 traditional matches per AI hub (63 candidates)
  ↓
generate_descriptions() → LLM bridge descriptions for candidates (Claude API, sanitize_text)
  + generate_negative_descriptions() → blinded negatives (bottom-1 per hub, 21 items)
  ↓
  (If LLM API fails: write candidates with empty descriptions. Re-run without --skip-descriptions to fill gaps.)
  ↓
bridge_candidates.json → expert review (accept/reject in JSON file)
  ↓
commit_bridges() → updated cre_hierarchy.json + bridge_report.json
```

Bridge analysis is hub-level ontology enrichment — not control→hub assignment. The core constraint (`g(control_text) → CRE_position`, no pairwise `f(A,B) → relationship`) applies to the assignment model, not to taxonomy curation.

### 1.5 Bridge Candidate JSON Format

```json
{
  "generated_at": "2026-05-02T...",
  "method": "top_k_per_ai_hub",
  "top_k": 3,
  "similarity_stats": {
    "matrix_shape": [21, 382],
    "mean": 0.162,
    "std": 0.106,
    "min": -0.149,
    "max": 0.774,
    "percentiles": {"25": 0.083, "50": 0.155, "75": 0.234, "90": 0.310, "95": 0.370, "99": 0.530}
  },
  "candidates": [
    {
      "ai_hub_id": "202-604",
      "ai_hub_name": "Human AI oversight",
      "trad_hub_id": "427-113",
      "trad_hub_name": "Security governance regarding people",
      "cosine_similarity": 0.774,
      "rank_for_ai_hub": 1,
      "seed_evidence": {
        "ai_controls_linked": 5,
        "trad_controls_linked": 12
      },
      "description": "Both hubs address human governance and oversight of systems...",
      "status": "pending",
      "reviewer_notes": ""
    }
  ],
  "negative_controls": [
    {
      "ai_hub_id": "202-604",
      "ai_hub_name": "Human AI oversight",
      "trad_hub_id": "xxx-yyy",
      "trad_hub_name": "...",
      "cosine_similarity": 0.012,
      "description": "...",
      "is_negative": true
    }
  ],
  "unclassified_leaf_hubs": ["hub-1", "hub-2", "hub-3", "hub-4"]
}
```

**Negative controls:** For each AI hub, the pipeline also generates a description for the **worst match** (bottom-1 of 382 traditional hubs). These 21 negative descriptions are presented alongside the 63 real candidates. If the expert can't distinguish negative descriptions from real candidates, the LLM descriptions add no filtering value.

**Review workflow:** User edits `status` to `"accepted"` or `"rejected"`, optionally adds `reviewer_notes`. Then runs `tract bridge --commit`. `commit_bridges()` validates: JSON parseable, all required keys present, `status ∈ {"accepted", "rejected"}`, all hub IDs exist in hierarchy. Rejects malformed input with specific error messages.

### 1.6 Hierarchy Update

Accepted bridges are stored in `cre_hierarchy.json` by adding `related_hub_ids` to each hub in the bridge pair (bidirectional). This requires a schema extension to `HubNode` in `tract/hierarchy.py`:

- Add field: `related_hub_ids: list[str] = Field(default_factory=list)`
- `HubNode` is `frozen=True` (Pydantic v2) — adding a new field with a default works; existing hierarchy files load without error because Pydantic v2 ignores unknown fields by default (no `extra="forbid"` set)
- Bump hierarchy version from `"1.0"` to `"1.1"`
- Update `validate_integrity()` to check that all `related_hub_ids` reference existing hubs

This does NOT change:
- Model weights (trained on control→hub assignments, not hub→hub)
- Hub embeddings (encoded from name + hierarchy path; related links are lateral, not hierarchical)
- Calibration (T, thresholds fitted on model output distribution)
- deployment_artifacts.npz (pre-computed from the above)

The updated hierarchy is bundled with the HuggingFace publication, giving downstream users a richer ontology. Bridge provenance (similarity scores, evidence, review decisions) lives in `bridge_report.json`, not the hierarchy — the hierarchy stores topology, not provenance.

---

## 2. HuggingFace Publication

### 2.1 Publication Gate

`publish_to_huggingface()` checks before proceeding:
1. `bridge_report.json` exists
2. All candidates in bridge_report.json have status `accepted` or `rejected` (no `pending`)
3. `cre_hierarchy.json` has been updated with accepted bridges (or no accepted bridges — zero accepted is valid)

If any check fails, raise `ValueError` with specific message. Zero accepted bridges is explicitly valid — the publication proceeds with bridge_report.json documenting the null result.

### 2.2 Model Merge

Merge LoRA adapters into base model to produce a standalone ~1.3GB model. The saved model is a **SentenceTransformer with a PEFT adapter overlay**, not a raw transformers model — the merge path must respect this.

- **Input:** `results/phase1c/deployment_model/model/model/` (SentenceTransformer directory with adapter_config.json + adapter_model.safetensors overlaying BAAI/bge-large-en-v1.5)
- **Process:**
  1. Load via `SentenceTransformer(model_dir)` — this auto-detects the PEFT adapter
  2. Access the inner PeftModel: `model[0].auto_model` (the `Transformer` module's `.auto_model` attribute)
  3. Merge: `model[0].auto_model = model[0].auto_model.merge_and_unload()`
  4. Save: `model.save(output_dir)` — preserves the full SentenceTransformer directory structure
- **Output directory structure** (differs from the flat adapter input layout):
  ```
  output_dir/
    0_Transformer/
      model.safetensors  (~1.3GB, fully merged weights)
      config.json
      tokenizer.json, tokenizer_config.json, vocab.txt, special_tokens_map.json
    1_Pooling/
      config.json
    2_Normalize/
    modules.json
    sentence_bert_config.json
    config_sentence_transformers.json
  ```
- **Verification:** No adapter_config.json in output (fully merged). Model produces near-identical embeddings to LoRA-loaded model (cosine similarity > 0.9999 for test inputs; minor float divergence from merge rounding is expected). Output loads correctly via `SentenceTransformer(output_dir)`.

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
- Training: H100 GPU via RunPod. `generate_model_card()` accepts `gpu_hours: float` parameter — hardcode actual hours from RunPod billing dashboard or WandB run metadata. Estimate CO2 using ML CO2 Impact calculator with US-West grid factor.
- Deployment: runs on Jetson Orin AGX (edge device, ~30W TDP)

**Evaluation addition — hit@any:** Because 35% of controls map to multiple hubs, hit@1 alone understates performance. Include a hit@any column in the LOFO table (whether the prediction matches ANY ground-truth hub for that control).

**Usage snippet, citation (BibTeX), license (MIT for model weights and code, with NOTICE attributing BAAI/bge-large-en-v1.5 base model; bundled data files are CC0 1.0)**

**Bridge Analysis Summary:** Included as a section — number of bridges found, methodology, link to bridge_report.json.

### 2.5 Standalone Scripts

**predict.py:**
- Loads merged model + bundled data from repo directory
- Takes control text as input, returns top-K hub predictions with calibrated confidence
- Dependencies: `sentence-transformers`, `torch`, `numpy` (no TRACT package required)
- Handles: single text and batch mode
- Inlines the following from TRACT's inference pipeline:
  - Simplified `sanitize_text()`: NFC normalization + whitespace stripping
  - Temperature scaling: `softmax(similarities / T)` (one line, T from calibration.json)
  - OOD flagging: `max_sim < ood_threshold` (threshold from calibration.json)
  - Does NOT include conformal prediction — reference TRACT repo for full pipeline

**train.py:**
- Documents full reproduction pipeline
- Acknowledges reproduction requires cloning the full TRACT repository (custom training procedures: text-aware batch sampling, joint temperature-scaled loss, active learning)
- Lists pinned requirements
- Includes data download instructions (OpenCRE API fetch)
- Includes exact training command with all hyperparameters
- References specific modules: `tract/training/`, active learning scripts, batch sampler

### 2.6 Security Scan

Automated scan before upload. Context-aware: scans Python scripts, model card (README.md), config files, and `reviewer_notes` fields in bridge_report.json. Does NOT scan JSON data files (hierarchy paths contain words like "home" that false-positive on path patterns).

**Patterns for scanned files:**
- API keys: `sk-[a-zA-Z0-9]{20,}`, `hf_[a-zA-Z0-9]{20,}`, `wandb_[a-zA-Z0-9]{10,}`
- Specific paths: `/home/rock`, `/Users/rock` (not broad `/home/\w+` which false-positives)
- Email addresses: standard email regex
- Credential manager: `^pass\s+\w+/\w+` (anchored to line start, avoids matching prose "pass through/to")
- AWS keys: `AKIA[0-9A-Z]{16}`
- Environment variable assignments: `(HF_TOKEN|WANDB_API_KEY|ANTHROPIC_API_KEY)\s*=`

**Structural checks:**
- No `.git/` directory in staging area
- No `adapter_config.json` in output (confirms merge completed)
- `reviewer_notes` fields in bridge_report.json scanned for PII (names, emails)

Any match = hard failure with file path and line number. No manual override — fix the source and re-run.

### 2.7 Upload

Use `huggingface_hub.HfApi.upload_folder()` to push the staging directory to the HuggingFace repo. This handles Git LFS automatically for files over 10MB (the ~1.3GB model.safetensors). HuggingFace token retrieved via `pass huggingface/token` and set as `HF_TOKEN` environment variable.

### 2.8 AIBOM Validation

Final step after upload (or during `--dry-run`):
1. Clone `GenAI-Security-Project/aibom-generator` to a temp directory
2. Run against the generated README.md
3. Report score and any missing fields
4. Target: 100/100

If the tool is unavailable or broken, log a warning and proceed (the model card was written to spec). Do not block publication on external tool availability.

---

## 3. CLI Commands

### 3.1 `tract bridge`

```
tract bridge [--output-dir DIR] [--top-k INT] [--skip-descriptions]
```

- `--output-dir`: output directory for bridge_candidates.json and bridge_report.json (default: `results/bridge/`)
- `--top-k`: number of traditional matches per AI hub (default: 3)
- `--skip-descriptions`: skip LLM description generation (for testing/dry runs). If skipped and re-run without this flag, fills in missing descriptions without regenerating existing ones.
- Prints: number of AI-only hubs, number of candidates generated, path to bridge_candidates.json
- Does NOT auto-commit bridges — requires explicit `--commit`

```
tract bridge --commit --candidates results/bridge/bridge_candidates.json
```

- Reads reviewed candidates file
- Validates: JSON parseable, all required keys present, `status ∈ {"accepted", "rejected"}`, all hub IDs exist in hierarchy
- Accepted bridges → `related_hub_ids` in cre_hierarchy.json (bidirectional, atomic write)
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
    classify.py          # classify_hubs() — AI/trad/both/unlinked split
    similarity.py        # compute_bridge_similarities(), extract_top_k()
    describe.py          # generate_bridge_descriptions(), generate_negative_descriptions() — LLM + sanitize_text
    review.py            # load_candidates(), validate_candidates(), commit_bridges() — JSON I/O, hierarchy update
  publish/
    __init__.py          # publish_to_huggingface() orchestrator, bridge gate
    merge.py             # merge_lora_adapters() — SentenceTransformer-aware PEFT merge
    bundle.py            # bundle_inference_data() — copy/validate data files
    model_card.py        # generate_model_card(gpu_hours: float) — templated README.md
    scripts.py           # write_predict_script(), write_train_script()
    security.py          # scan_for_secrets() — context-aware regex scan
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
BRIDGE_TOP_K: int = 3  # top-K traditional matches per AI hub
BRIDGE_LLM_MODEL: str = "claude-sonnet-4-20250514"
BRIDGE_LLM_TEMPERATURE: float = 0.0
BRIDGE_OUTPUT_DIR: Path = Path("results/bridge")

# HuggingFace publication
HF_DEFAULT_REPO_ID: str = "rockCO78/tract-cre-assignment"
HF_STAGING_DIR: Path = Path("build/hf_repo")
HF_BASE_MODEL: str = "BAAI/bge-large-en-v1.5"
HF_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"hf_[a-zA-Z0-9]{20,}"),
    re.compile(r"wandb_[a-zA-Z0-9]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"/home/rock"),
    re.compile(r"/Users/rock"),
    re.compile(r"^pass\s+\w+/\w+", re.MULTILINE),
    re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    re.compile(r"(HF_TOKEN|WANDB_API_KEY|ANTHROPIC_API_KEY)\s*="),
]
# Security scan targets: .py, .md, .json (config only), .txt, .yaml
# Excludes: JSON data files (hierarchy, descriptions) to avoid false positives
HF_SCAN_EXTENSIONS: frozenset[str] = frozenset({".py", ".md", ".txt", ".yaml", ".yml"})
```

---

## 6. Testing Strategy

### Bridge Tests (`tests/test_bridge/`)

| Test file | Scope |
|---|---|
| `test_classify.py` | Hub classification counts (ai_only, trad_only, naturally_bridged, unlinked). Edge cases: hub with zero links, hub in neither set, leaf vs non-leaf unlinked. |
| `test_similarity.py` | Cosine matrix shape (21×382). Top-K extraction returns K items per hub sorted descending. Deterministic output. Unit-vector dot product equals cosine. |
| `test_describe.py` | sanitize_text applied to descriptions. Non-empty output. API failure → empty description (not crash). Negative control descriptions generated for bottom-1 per hub. |
| `test_review.py` | JSON round-trip. Validation rejects: pending entries, unknown status values, non-existent hub IDs, malformed JSON. Accepted → `related_hub_ids` bidirectional. Rejected excluded. Atomic writes. Zero accepted = valid. |
| `test_bridge_integration.py` | End-to-end with synthetic fixture: 3 AI hubs, 5 trad hubs, 1 naturally bridged → top-3 candidates → commit → hierarchy updated with `related_hub_ids`. |

### Publish Tests (`tests/test_publish/`)

| Test file | Scope |
|---|---|
| `test_merge.py` | Merged model directory has `0_Transformer/model.safetensors`, `modules.json`, `1_Pooling/`, `2_Normalize/`. No adapter_config.json. Loads via `SentenceTransformer()`. |
| `test_bundle.py` | All required files copied. Missing file → ValueError. Hierarchy includes `related_hub_ids`. hub_ids.json ordering matches embeddings (`len(hub_ids) == hub_embeddings.shape[0]`). |
| `test_model_card.py` | All AIBOM sections present. Metrics match source. hit@any included. Limitations honest. No secrets. MIT license with NOTICE. |
| `test_security.py` | Planted secrets detected in .py/.md files. Clean dir passes. JSON data files with `/home` in hierarchy paths do NOT trigger. `reviewer_notes` with PII detected. |
| `test_publish_integration.py` | End-to-end --dry-run passes. Bridge gate blocks without bridge_report.json. Zero-bridge gate passes. |

### Fixtures

- Synthetic hierarchy: ~10 hubs with fake 1024-dim unit-normalized embeddings, known bridge relationship
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
