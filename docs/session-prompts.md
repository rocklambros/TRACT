# TRACT Session Prompts

Self-contained prompts for continuing the TRACT project in new Claude Code sessions. Each prompt bootstraps a fresh session with full context. Complete sequentially.

**Project state (2026-05-03):**
- Data Preparation: COMPLETE
- Phase 0 (Zero-Shot Baselines): COMPLETE — Gates A+B passed
- Phase 1A (Hierarchy, Descriptions, Ingestion): COMPLETE
- Phase 1B (Training Pipeline): COMPLETE — Gate 1 CLEAN PASS (hit@1=0.531, delta=+0.132)
- Phase 1C (Guardrails, Active Learning, Crosswalk DB): COMPLETE — 2 AL rounds converged, 636 assignments, 339 tests
- Phase 1D (CLI, Hub Proposals): COMPLETE — 8 CLI commands, hub proposal pipeline, 394 tests
- Phase 5A (Export Pipeline + Fork Import): COMPLETE — 411 assignments across 5 frameworks, coverage gaps report
- Framework Preparation Pipeline (PR #21): COMPLETE — `tract prepare` + `tract validate` + ingest integration, 553 tests total
- Phase 2B (Bridge + HF Publication): COMPLETE — 46 bridges accepted (5-round adversarial review), model published to huggingface.co/rockCO78/tract-cre-assignment, 654 tests
- ~~Phase 2A (Web UI):~~ CANCELLED — no web dashboard, CLI + API only
- Phase 3 (Published Crosswalk Dataset): NOT STARTED
- Phase 3B (Experimental Narrative Notebook): NOT STARTED
- Phase 4 (Secure API): NOT STARTED
- Phase 5B (OpenCRE Upstream Contribution): BLOCKED on Phase 3

**Key results driving all remaining work:**
- BGE-large-v1.5 + LoRA rank 16 + MNRL contrastive loss + text-aware batching
- Per-fold deltas vs zero-shot: NIST +0.322, ML +0.285, OWASP-X +0.143, ATLAS +0.006, LLM-10 +0.000
- 640 tests passing, 13 CLI subcommands, all code typed and validated
- CUDA determinism flags added to training loop
- 5-round adversarial review completed — all findings resolved
- Phase 5A: 411 assignments exported (conf ≥ 0.30, ATLAS ≥ 0.35), 192 controls excluded with reasons, all imported into local OpenCRE fork
- Framework Preparation Pipeline: CSV/Markdown/JSON/LLM extractors, 17-rule validation engine, ingest integration with calibration disclaimer + quality summary
- Phase 2B: bridge analysis (classify → cosine similarity → top-3 → LLM describe → review → commit) + HF publish (LoRA merge → bundle → model card → scripts → security scan → upload). 18 commits, 3,332 lines, 640 tests. 3-round adversarial review + 3 implementation reviews all passed.

---

## Prompt 1: Phase 1C — Guardrails, Active Learning & Crosswalk Database

```
TRACT Phase 1C: Implement guardrails (PRD 6.6), active learning loop (PRD 6.7), and crosswalk database (PRD 6.8).

Read PRD.md Sections 6.6–6.8. Read CLAUDE.md for code standards (especially "Fail loud, fail early" and "Defensive I/O").

## Current state — verify before starting
Run these checks and stop if any fail:
1. `python -m pytest tests/ -q` → 262 tests pass
2. `ls data/processed/cre_hierarchy.json data/processed/hub_descriptions.json data/processed/all_controls.json` → all exist
3. `ls results/phase1b/phase1b_textaware/` → 5 fold directories with predictions.json and metrics.json
4. `python -c "from tract.hierarchy import CREHierarchy; from tract.io import load_json; from tract.config import PROCESSED_DIR; h = CREHierarchy.model_validate(load_json(PROCESSED_DIR / 'cre_hierarchy.json')); print(f'{len(h.hubs)} hubs, {len(h.leaf_hub_ids())} leaves')"` → 457 hubs, ~400 leaves

## Phase 1B results to incorporate
The trained model exists at results/phase1b/phase1b_textaware/fold_*/model/. Key facts:
- Gate 1 PASS: micro hit@1=0.531 (delta=+0.132 over zero-shot 0.399)
- Per-fold: ATLAS=0.279(n=43), NIST=0.429(n=28), OWASP-X=0.762(n=63), LLM-10=0.333(n=6), ML-10=0.714(n=7)
- Zero-shot per-fold: ATLAS=0.273, NIST=0.107, OWASP-X=0.619, LLM-10=0.333, ML-10=0.429
- Model predictions for all 5 folds are in fold_*/predictions.json (control_text, ground_truth_hub_id, predicted_top10)
- Confidence scores: cosine similarities, NOT calibrated probabilities yet — calibration is part of THIS phase (6.6 Model Integrity)
- Training config: 20 epochs, batch=32, lr=5e-4, LoRA rank=16, seed=42

## Lessons from Phase 1B (incorporate these)
1. **Pre-register gate metrics.** Define success criteria BEFORE running experiments. Post-hoc metric substitution (e.g., switching micro→macro after seeing results) is not permitted. Document criteria in the spec.
2. **Report per-fold deltas, not just aggregates.** Aggregate hit@1=0.531 masked that ATLAS was flat. Per-fold delta vs zero-shot revealed the real picture.
3. **Traditional framework links are always in training by LOFO design.** When a CWE section maps to the same hub as an ATLAS eval item, this is the semantic bridge mechanism working as designed, NOT information leakage. Do not filter these out.
4. **Multi-hub text mappings are valid CRE graph structure.** 35% of control texts map to >1 hub. Handle at the batching/loss layer (HubAwareTemperatureSampler), never at the data layer (dropping examples).
5. **Adversarial review catches real errors.** R2 compared 4-fold FT vs 5-fold ZS (apples to oranges) — without cross-examination this would have become a false "model only helps one framework" conclusion.
6. **CUDA determinism flags are required.** Without torch.backends.cudnn.deterministic=True + CUBLAS_WORKSPACE_CONFIG, fp16 on H100 produces ±1 item variance on small folds.
7. **Hub disambiguation is harder than hub neighborhood identification.** ATLAS misses are 77% unrelated-subtree (model confused at top level), and regressions trade 1:1 with improvements (7 gained, 7 lost). The model reaches the right neighborhood but sometimes picks the wrong leaf.

## Scope — three components

### 6.6 Guardrails (5 categories, all concrete and testable)

**Data Integrity** (mostly exists — formalize):
- LOFO train/test split: already in tract/training/orchestrate.py. Promote split logic to a standalone tract/guardrails/data_integrity.py with explicit assertions.
- Mapping-level dedup: detect same control text appearing via different CRE link paths. Already handled by build_training_pairs() dedup on (text.lower().strip(), hub_id). Add a reporting function that surfaces dedup statistics.
- Auto-link provenance: already stored in tiered_links (QualityTier.T1 vs T1_AI vs T2 vs T3). Add provenance chain field (CAPEC→CWE→CRE path).
- Source version pinning: each framework tagged with document version + fetch date. Check data/processed/frameworks/*.json for existing metadata.

**Model Integrity**:
- Hub representation firewall: already in tract/training/firewall.py. Promote to tract/guardrails/model_integrity.py. Add programmatic assertion: for each fold, verify zero overlap between eval framework sections and hub representations.
- Confidence calibration: NEW. Implement temperature scaling (Platt scaling) on held-out validation predictions. Input: cosine similarities from predictions.json. Output: calibrated probabilities. Use tract/training/calibrate.py (stub exists). Fit on LOFO validation predictions, evaluate with reliability diagrams and ECE (Expected Calibration Error).
- Per-hub threshold tuner: NEW. For each hub, find the cosine similarity threshold that maximizes F1 on validation data. Freeze thresholds before test evaluation.

**Output Integrity**:
- Confidence threshold filter: predictions below calibrated threshold → flagged, not committed to crosswalk.db
- Conformal prediction wrapper: guarantee coverage ≥ 90% (the true hub is in the prediction set 90%+ of the time). Use split conformal prediction on LOFO validation data.
- Multi-hub disagreement detector: flag cases where top-2 predictions have similar confidence (delta < 0.05 calibrated probability)

**Adversarial Robustness**:
- Paraphrase probe test set: take 20 controls, generate 3 paraphrases each via LLM, verify same hub assignments. This is a test, not a training feature.
- OOD detector: max cosine similarity below threshold → flag as "no good hub match". Threshold from calibration.
- OOD → hub proposal pipeline trigger (connects to Phase 1D)

**Provenance Tracking**:
- Prediction log schema: control_text, predicted_hubs (with scores), model_version (git SHA + data hash + config name), timestamp
- Human review tracking: accept/reject/correct linked to prediction ID
- Training data lineage per example: source framework, link type, quality tier

### 6.7 Active Learning Loop
Target: 6 unmapped AI frameworks (AIUC-1: 132, CSA AICM: 243, CoSAI: 29, EU GPAI CoP: 32, OWASP Agentic: 10, NIST AI RMF: 72) = 518 controls total.
1. Load trained model (best fold or retrained on all 5 folds)
2. Run inference on all 518 unmapped controls → top-5 hub predictions with calibrated confidence
3. Export to review format: control_text | framework | predicted_hub_1 (conf) | ... | reviewer_decision
4. Expert reviews via CLI tool (tract review — implement minimal version here, full CLI in 1D)
5. Accepted predictions → training data with provenance="active_learning_round_1"
6. Retrain model on expanded dataset
7. Re-predict remaining unreviewed/rejected controls
8. Repeat until acceptance rate > 80%

### 6.8 Crosswalk Database
- SQLite: crosswalk.db
- Schema tables: frameworks, controls, hubs, assignments (control_id, hub_id, confidence, calibrated_probability, review_status, model_version, provenance), reviews (assignment_id, reviewer, decision, timestamp, notes)
- Populate from: (a) all 4,406 known CRE links (review_status="ground_truth"), (b) Phase 1B model predictions for mapped frameworks (review_status="model_predicted"), (c) active learning predictions (review_status="active_learning_round_N")
- Cross-framework relationships derived transitively: controls sharing a hub are equivalent, sharing a parent are related
- Export: JSON, CSV per framework or full matrix
- All writes via atomic transactions

## Approach
Use brainstorming → spec → plan → subagent-driven-development.

Key architectural decisions to make in brainstorming:
1. Should calibration fit on all 5 folds pooled, or per-fold? (Pooled is simpler; per-fold respects LOFO but has n=6 for LLM-10)
2. Should the active learning model be retrained on all 5 folds combined (no LOFO), or keep LOFO? Once we're predicting truly unmapped frameworks, LOFO is no longer needed.
3. Conformal prediction: use LOFO validation data or a separate calibration split?

Before finalizing the spec, run adversarial review: 3 critic agents (data scientist, ML engineer, security engineer) attack the spec, then a judge synthesizes. Focus on: calibration methodology soundness, active learning bias risks, database integrity under concurrent review.

## Success criteria
- All 5 guardrail categories pass automated test suite (one test class per category)
- Calibration: ECE < 0.05 on held-out validation data
- Conformal prediction: empirical coverage ≥ 0.90
- Active learning round 1: predictions generated for all 518 unmapped controls
- Crosswalk database: populated with all ground-truth links + model predictions
- All new code typed, tested, no bare exceptions

## Anti-patterns from Phase 1B — do NOT repeat
- Do not drop data to "simplify." Multi-hub mappings, multi-label predictions, low-confidence outputs — keep everything, handle complexity at the right layer.
- Do not substitute metrics post-hoc. If ECE < 0.05 is the calibration gate, don't switch to "calibration looks reasonable" after seeing results.
- Do not compare across different data subsets without accounting for the difference (R2's 4-fold vs 5-fold error).

Think deeply using the sequential-thinking MCP server. --ultrathink
```

---

## ~~Prompt 2: Phase 1D — CLI Tool & Hub Proposal System~~ ✅ COMPLETE

**Completed 2026-04-30.** PR #15 merged. 14 commits, 2,855 lines, 394 tests (387 unit + 7 integration).

**What was built:**
- `tract/cli.py` — 8 subcommands: assign, compare, ingest, export, hierarchy, propose-hubs, review-proposals, tutorial
- `tract/inference.py` — TRACTPredictor with calibrated confidence, conformal sets, OOD detection, duplicate detection
- `tract/compare.py` — SQL-backed cross-framework equivalence/gap analysis
- `tract/proposals/` — HDBSCAN clustering → 4-guardrail filter → LLM/placeholder naming → versioned JSON proposals
- Deployment artifacts: 522×1024 hub embeddings + 2,802×1024 control embeddings in single NPZ

**Key results:**
- 95 OOD controls (3.4%) → 8 HDBSCAN clusters → 5 proposals passing guardrails
- Calibration: T=0.074, OOD threshold=0.568, conformal quantile=0.997
- All CLI commands verified end-to-end with real deployment model on Jetson Orin AGX

---

## ~~Prompt 2.5: Framework Preparation Pipeline~~ ✅ COMPLETE

**Completed 2026-05-01.** PR #21 merged. 18 commits, 553 tests total (159 new).

**What was built:**
- `tract/prepare/` — ExtractorRegistry with Protocol-based extractors: CSV (BOM, TSV, flexible aliases), Markdown (4 ID patterns, heading-level auto-detect), JSON (passthrough/array/nested), LLM (Claude tool_use, optional dep)
- `tract/validate.py` — ValidationIssue frozen dataclass, 6 error + 11 warning rules
- `tract/cli.py` — now 11 subcommands (+prepare, +validate, +accept from PR #20)
- Ingest integration: validation gate, calibration disclaimer for prepare-sourced data, quality summary (mean_max_cosine_sim, ood_fraction, below_confidence_floor)
- `tract/sanitize.py` — added sanitize_control() for dict-based ingest pipeline
- `examples/` — sample_framework.csv, sample_framework.md, README.md

**Key results:**
- Full onboarding pipeline: `tract prepare` → `tract validate` → `tract ingest` → `tract accept`
- 17 validation rules catch data quality issues before expensive model inference
- Optional LLM dependencies via `pip install tract[llm]` with guarded imports

---

## ~~Prompt 3: Phase 2A — Dash Web UI~~ ❌ CANCELLED

**Decision (2026-05-02):** No web UI. TRACT is CLI + API only. All web UI references removed from PRD.

---

## ~~Prompt 4: Phase 2B — HuggingFace Publication & AI/Traditional Bridge~~ ✅ COMPLETE

**Code completed 2026-05-02.** PR #22 merged. 18 commits, 3,332 lines, 640 tests (134 new).
**Executed 2026-05-03.** Bridge analysis run, expert review completed, model published to HuggingFace.

**What was built:**
- `tract/bridge/` — 6 modules: types.py (TypedDicts), classify.py, similarity.py, describe.py, review.py, orchestrator
- `tract/publish/` — 6 modules: merge.py (LoRA merge + cosine verification), bundle.py, model_card.py, scripts.py, security.py, orchestrator
- `tract/cli.py` — 13 subcommands (+bridge, +publish-hf)
- `tract/hierarchy.py` — added `related_hub_ids` with bidirectional integrity validation
- `tract/config.py` — BRIDGE_*/HF_*/HIERARCHY_BRIDGE_VERSION constants
- `tract/bridge/types.py` — TypedDict structures: RawCandidate, BridgeCandidate, RawNegative, NegativeControl, SimilarityStats

**Bridge analysis results:**
- 21 AI-only hubs × 382 traditional-only hubs → 63 candidates (top-3 per AI hub)
- 5-round adversarial review of acceptance methodology (security architecture → methodology → implementation → cross-attack → convergence)
- 46 accepted, 17 rejected (13 below p99 threshold of 0.448, 4 specious MFA/OTP connections)
- 92 bidirectional related_hub_id edges added to hierarchy (version 1.0 → 1.1)
- Top bridge: "Human AI oversight" ↔ "Security governance regarding people" (cosine 0.774)

**HuggingFace publication:**
- Published to: huggingface.co/rockCO78/tract-cre-assignment
- 456-line model card (novice + expert audience): plain English overview, Quick Start, architecture diagram, LOFO evaluation explained, calibration walkthrough, bridge analysis summary, 3 detailed usage examples, glossary
- Merged model: 1.34GB (BGE-large + LoRA fully merged, no adapter needed)
- Bundled: predict.py, train.py, hub_embeddings.npy, hub_ids.json, cre_hierarchy.json (with bridges), calibration.json, bridge_report.json, hub_descriptions.json

**Key implementation fixes during execution:**
- merge.py: SentenceTransformer doesn't auto-detect PEFT adapter in flat directory layout. Added duck-typing fallback (`hasattr(inner, "merge_and_unload")`) with manual `PeftModel.from_pretrained` + `delattr(merged, "peft_config")` after merge
- merge.py: validate_merged_output updated to accept both flat layout (model.safetensors at root) and subdirectory layout (0_Transformer/)
- `__init__.py`: added `api.create_repo(exist_ok=True)` before upload_folder

**Reviews passed:** 3-round code review (spec compliance, code quality, final holistic) + 5-round adversarial review of bridge methodology. All approved. 654 tests passing.

---

## Prompt 5: Phase 3 — Published Human-Reviewed Crosswalk Dataset

```
TRACT Phase 3: Produce and publish the human-reviewed crosswalk dataset. PRD Section 8.

Read PRD.md Section 8. Read CLAUDE.md for code standards.

## Current state — verify before starting
1. Phase 2B COMPLETE: model published to huggingface.co/rockCO78/tract-cre-assignment
2. Bridge analysis COMPLETE: 46 bridges accepted, 17 rejected, hierarchy version 1.1
3. crosswalk.db at results/phase1c/crosswalk.db has 636 assignments across 6 AI frameworks:
   - MITRE ATLAS: 260, CSA AICM: 243, EU AI Act: 100, OWASP LLM Top 10: 13, OWASP Agentic: 10, NIST AI 600-1: 10
   - Provenance: active_learning_round_2 (558 accepted) + ground_truth_T1-AI (78)
   - 31 frameworks registered in DB, but only 6 have assignments
4. Active learning: 2 rounds completed and converged
5. Hub proposals: proposals.json exists at hub_proposals/round_1/ — review status needs verification
6. Bridge relationships in cre_hierarchy.json: 51 hubs have related_hub_ids (92 bidirectional edges)
7. 654 tests passing, 13 CLI subcommands
8. Calibration: T=0.074, OOD threshold=0.568, ECE=0.079

## What does NOT yet exist
- No predictions for the 16 traditional (non-AI) OpenCRE-linked frameworks in crosswalk.db
  (OWASP ASVS, NIST 800-53, CWE, CAPEC, etc. have 4,237 existing ground-truth links but
  these are NOT in crosswalk.db — they live in data/training/hub_links_by_framework.json)
- No review tooling for expert review workflow
- No dataset publication pipeline (HuggingFace Datasets card, Zenodo DOI)
- No formal review metrics tracking

## Key context
This phase has TWO parts:
1. **Engineering** (~2-3 sessions): Build review tooling, populate crosswalk.db with ALL framework assignments (ground truth + model predictions), build dataset publication pipeline
2. **Expert review** (~62+ hours, outside Claude Code): Human reviews all AI framework assignments. Traditional framework assignments from OpenCRE are ground truth and need no review.

The engineering work must happen first to give the expert a review interface.

## Data inventory for crosswalk population
| Source | Frameworks | Controls | Status | Action |
|--------|-----------|----------|--------|--------|
| OpenCRE ground truth | 16 traditional + 1 AI (OWASP ML) | ~4,237 links | In hub_links_by_framework.json | Import as ground_truth |
| Active learning round 2 | 6 AI frameworks | 636 assignments | In crosswalk.db | Already present |
| Phase 5A export | 5 AI frameworks | 411 assignments (conf ≥ 0.30) | In results/phase1c/exports/ | Subset of AL assignments |

Total target: ~4,873 assignments (4,237 ground truth + 636 AL predictions) across 22+ frameworks.

## Results from all previous phases
1. Model confidence per framework → prioritize review: start with highest-confidence frameworks
2. Active learning acceptance rates → frameworks with low acceptance need more careful review
3. Bridge hubs from Phase 2B → include 46 accepted bridge relationships in published dataset
4. Calibration (T=0.074, ECE=0.079) → show calibrated probabilities to reviewer to guide effort allocation
5. Per-hub prediction accuracy from LOFO → identify systematically difficult hubs (ATLAS hub disambiguation is the primary failure mode)
6. 5 hub proposals from Phase 1D → check if any were accepted and integrate

## Scope

### Step 1: Populate crosswalk.db with ALL frameworks
1. Import all 4,237 OpenCRE ground-truth links into crosswalk.db with review_status="ground_truth"
2. Verify no duplicates between ground truth and existing AL assignments
3. Register all 22+ frameworks with metadata (name, version, source, control count)
4. Result: crosswalk.db has complete coverage of all OpenCRE-linked frameworks

### Step 2: Review Tooling
1. `tract review` CLI command: export predictions grouped by framework → review spreadsheets
2. Each row: control_id, control_text, predicted_hub_1 (calibrated_conf), predicted_hub_2 (calibrated_conf), ..., reviewer_decision, reviewer_notes
3. Review interface: accept top prediction, select different hub, flag for discussion, mark "no good hub"
4. Track: total reviewed, acceptance rate, edit rate, rejection rate, time per control
5. Second-pass consistency check for edited/rejected controls
6. Inter-reviewer agreement metrics if multiple reviewers (Cohen's kappa)
7. Atomic review state: review progress survives crashes (write-to-temp + os.replace pattern)

### Step 3: Dataset Publication Pipeline
Published to HuggingFace Datasets AND Zenodo (for DOI):
- crosswalk_v1.0.jsonl — every control + hub assignment + confidence + review status
- framework_metadata.json — all framework descriptions, versions, sources
- cre_hierarchy_v1.1.json — hub ontology with bridge links at publication time
- hub_descriptions_v1.0.json — validated descriptions
- bridge_report.json — bridge analysis evidence and review decisions
- review_metrics.json — acceptance rates, agreement, reviewer effort
- README.md — dataset card (HuggingFace Datasets format, novice-friendly like model card)
- LICENSE — CC-BY-4.0

### Step 4: Expert Review (outside Claude Code)
1. Expert reviews all 636 AI framework assignments using review tooling
2. Ground truth assignments (4,237) are NOT reviewed — they are authoritative
3. Track review decisions and metrics
4. Freeze as crosswalk_reviewed_v1.0.jsonl

## Success criteria
- crosswalk.db contains assignments for all 22+ OpenCRE-linked frameworks
- Review tooling functional: export, review, track, atomic state
- 100% of AI framework predicted assignments reviewed by expert
- Published dataset on HuggingFace Datasets with DOI from Zenodo
- Review metrics documented (acceptance rate, edit rate, per-framework breakdown)
- Dataset card comprehensive and novice-friendly (matching model card quality)

## Anti-patterns — do NOT repeat
- Do not conflate model publication (Phase 2B, done) with dataset publication (this phase). Different artifacts, different HuggingFace repo types (model vs dataset).
- Do not skip ground truth import — the published crosswalk needs ALL frameworks, not just AI.
- Do not let review tooling complexity delay the expert review. Simple CSV export is acceptable as v1; fancy UI can come later.

Think deeply using the sequential-thinking MCP server. --ultrathink
```

---

## Prompt 6: Phase 3B — Experimental Narrative Notebook

```
TRACT Phase 3B: Create the publication-quality experimental narrative notebook. PRD Section 8B.

Read PRD.md Section 8B for full requirements including visualization standards and narrative structure.

## Current state — verify before starting
1. All training results available in results/phase0/, results/phase0r/, results/phase1b/
2. Model published to HuggingFace (Phase 2B complete)
3. Crosswalk reviewed and published (Phase 3 complete)
4. All raw data for visualizations accessible

## Key context
This is NOT a code dump. It is a narrative document following a problem → exploration → failure → insight → solution arc. Style reference: ai-security-framework-crosswalk/project1/COMP_4433_RockLambros_project1_crosswalk_eda.ipynb (128 cells, 82 markdown / 46 code, 24 figures).

## Concrete requirements
- ≥128 cells, ≥1.5:1 markdown-to-code ratio
- ≥24 figures, all numbered with titles (e.g., "Figure 4.2: Per-Framework hit@1 Across Base Models")
- Every figure: interpretation paragraph before (what to look for) and after (what it shows)
- "Plain English" blockquotes after every technical section
- Interactive visualizations with static fallbacks for PDF export
- Colorblind-accessible palettes (Okabe-Ito for categorical, single-hue sequential for counts)
- Reproducible: all cells run top-to-bottom with identical output
- Full notebook runs in < 10 minutes (pre-computed embeddings, not re-computed)

## Narrative structure (11 sections)
1. Introduction & Motivation — CRE as coordinate system, assignment paradigm
2. Data Landscape — 4,406 links, 22 frameworks, hub distribution
3. Phase 0 Baselines — zero-shot results, what worked/failed (DeBERTa=0.000)
4. Base Model Selection — BGE vs GTE vs DeBERTa, per-fold analysis
5. Contrastive Fine-Tuning — MNRL, LoRA, text-aware batching, loss curves
6. Ablation Analysis — what mattered, paired deltas with CIs
7. The Hub Firewall — LOFO, with/without firewall comparison
8. Final Results — best model vs all baselines, per-framework deep dive
9. Error Analysis — ATLAS item-level trades, hub disambiguation, attractor hubs
10. Calibration — temperature scaling, reliability diagrams, ECE
11. Conclusion & Next Steps

## Key stories to tell honestly
- DeBERTa-v3-NLI complete failure (hit@1=0.000) — why NLI ≠ semantic similarity for this task
- Multi-hub text correction — user insight about CRE graph structure, not noise
- ATLAS flat performance — model trades hits 1:1, hub disambiguation problem
- R2's methodology error — 4-fold vs 5-fold comparison, caught by adversarial cross-examination
- NIST AI massive improvement (0.107 → 0.429) — biggest single-fold gain

## Success criteria (PRD Section 11)
- ≥128 cells, ≥24 figures, full story arc
- All cells run top-to-bottom with identical output
- Interactive 3D/animated figures with static fallbacks
- Colorblind-accessible palettes throughout

Think deeply using the sequential-thinking MCP server. --ultrathink
```

---

## Prompt 7: Phase 4 — Secure API with Full Documentation

```
TRACT Phase 4: Build and deploy the secure API with full documentation. PRD Section 9.

Read PRD.md Section 9. Read CLAUDE.md for code standards (especially Security section).

## Current state — verify before starting
1. All Phase 1-3 complete: model trained, crosswalk reviewed, dataset published
2. CLI commands working (`tract assign`, `tract compare`, `tract ingest`, `tract export`, `tract hierarchy`)
3. Model published to HuggingFace (inference code available)
4. Crosswalk.db finalized with reviewed assignments

## Key principle
The API wraps the SAME business logic as the CLI. tract/ library functions are the shared layer. The API adds: HTTP transport, authentication, rate limiting, OpenAPI docs. No new business logic in the API layer.

## Results from all phases
1. Model inference latency from CLI → baseline for API <500ms target
2. crosswalk.db schema → API endpoints query this
3. Common query patterns from CLI usage → optimize API for actual patterns
4. Security patterns from all phases → apply consistently (sanitize_text, parameterized SQL, no eval/exec)
5. Calibrated confidence scores → API returns calibrated probabilities

## Scope

### API Endpoints (FastAPI)
| Method | Endpoint | Latency Target |
|--------|----------|---------------|
| POST | /v1/assign | <500ms (model inference) |
| GET | /v1/hub/{cre_id} | <50ms (DB lookup) |
| GET | /v1/compare?fw1=X&fw2=Y | <200ms (DB query) |
| GET | /v1/framework/{fw_id} | <100ms (DB query) |
| POST | /v1/ingest | <30s (validation + inference) |
| GET | /v1/hierarchy | <100ms (cached) |
| GET | /v1/search?q=... | <200ms (DB + optional inference) |
| GET | /v1/health | <10ms |

### Security (CRITICAL — this is internet-facing)
- API key authentication (issued per user/org, stored hashed)
- Rate limiting: 100/min for /assign (model inference), 1000/min for reads
- Input validation: max 2000 chars, sanitize_text() on all user input
- Parameterized SQL only — no string interpolation in queries
- HTTPS only (TLS termination at reverse proxy)
- No PII stored; prediction logs use hashed request IDs
- CORS configuration for web clients
- No eval/exec/shell=True anywhere in the API

### Documentation
- OpenAPI 3.0 spec auto-generated from FastAPI endpoint definitions
- Hosted Swagger UI at /docs
- Python SDK: pip install tract-client
- Usage examples in README

### Deployment
- Dockerfile + docker-compose.yml (API + SQLite + model weights)
- Model loaded once at startup, shared across requests
- Health check endpoint with model status
- Deployment guide: local, Docker, cloud

## Approach
Use brainstorming → spec → plan → subagent-driven-development.

Security review is MANDATORY for this phase — run a dedicated security review agent after implementation. Focus on: input validation completeness, SQL injection, authentication bypass, rate limit circumvention, information disclosure in error messages.

Adversarial review: 3 critics (API design, security, DevOps) attack the spec.

## Success criteria
- API latency < 500ms per /assign
- Complete OpenAPI spec with all endpoints documented
- Python SDK published and tested
- Docker deployment working end-to-end
- Zero security vulnerabilities in dedicated security review
- All endpoints tested: valid input, invalid input, auth failures, rate limit enforcement

Think deeply using the sequential-thinking MCP server. --ultrathink
```

---

## Prompt 8: Phase 5B — OpenCRE Upstream Contribution

```
TRACT Phase 5B: Submit validated TRACT outputs as contributions to the OpenCRE project. PRD Section 10.

Read PRD.md Section 10 (especially Phase 5B subsection). Read CLAUDE.md for code standards.

## Current state — verify before starting
1. Phase 3 complete: all assignments human-reviewed and published
2. Phase 5A infrastructure exists: `tract export --opencre` generates CSVs, coverage_gaps.json, export_manifest.json
3. Local OpenCRE fork at ~/github_projects/OpenCRE with pilot import (411 assignments from 2026-05-01)
4. `scripts/direct_opencre_import.py` working for fork validation
5. Hub proposals from Phase 1D available in hub_proposals/
6. Bridge mappings from Phase 2B available (if Phase 2B is complete)

## Phase 5A results to incorporate
1. Export pipeline tested: 411 assignments across 5 frameworks at conf ≥ 0.30 (ATLAS ≥ 0.35)
2. Coverage gaps: 192 controls not exported — 140 below confidence floor, 39 no assignment, 6 ground truth, 3 null confidence, 4 other
3. OWASP LLM Top 10 correctly excluded (ground truth training data)
4. MITRE ATLAS deduplication works: 28 pre-existing nodes matched, 100 new nodes created, zero duplicates
5. Import path: direct SQLAlchemy script (~3s) preferred over REST API (requires Redis/graph loading)

## Phase 3 review results to incorporate
1. Which assignments changed during human review? Regenerate CSVs with reviewed data.
2. Review acceptance rate per framework → confidence in contribution quality
3. Any new hub proposals accepted → include in contribution
4. Did human review reveal systematic model errors? Document in contribution notes.

## Scope

### Re-export with reviewed data
1. Re-run `tract export --opencre` after Phase 3 review is complete — CSVs now reflect human-reviewed assignments
2. Re-import into local fork to validate the reviewed data
3. Compare pilot (411 assignments) vs reviewed export — document what changed

### Upstream contribution preparation
1. Fork the official OpenCRE repo (or verify existing fork is up-to-date with upstream)
2. Create a feature branch per framework (e.g., `tract/add-csa-aicm`, `tract/add-eu-ai-act`)
3. Import each framework's CSV into the fork branch
4. Verify import via OpenCRE's own validation (run tests, check DB integrity)
5. Write PR descriptions with: framework name, control count, coverage stats, confidence distribution, link to TRACT project

### PR strategy
- One PR per framework (not one monolithic PR) — easier to review and merge independently
- Order: smallest first for easy review wins (NIST AI 600-1 → OWASP Agentic → EU AI Act → CSA AICM → MITRE ATLAS)
- MITRE ATLAS last because it modifies existing data (pre-existing upstream links)
- Include coverage_gaps.json as supplementary information for OpenCRE maintainers

### Hub proposals (if accepted in Phase 1D review)
- Format hub proposals in whatever format OpenCRE maintainers prefer (coordinate with them)
- Submit as separate PRs from framework assignments
- Include evidence: member controls, similarity scores, parent hub justification

## Open questions to resolve
1. Attribution format — discuss with OpenCRE maintainers before first PR
2. Does OpenCRE want the CSVs committed to their repo, or just the DB changes?
3. Should TRACT confidence scores be included as metadata in the OpenCRE link entries?

## Success criteria
- All human-reviewed assignments submitted as PRs to OpenCRE GitHub
- One PR per framework, each with clear description and stats
- Hub proposals submitted (if any accepted)
- Contribution tracking: record acceptance/rejection/modification per PR
- No data quality regressions (fork passes OpenCRE's own test suite)

Think deeply using the sequential-thinking MCP server. --ultrathink
```

---

## Accumulated Lessons Learned

These compound across phases. Each prompt above incorporates the relevant subset.

### Architecture
- Assignment paradigm only: g(control_text) → CRE_position. Never pairwise.
- CRE graph has legitimate multi-hop structure — controls map to multiple hubs (35%). Handle at batching/loss layer, not data layer.
- Hub firewall at LINK level, not text level. Substring matching gives false positives.
- Traditional framework links always in training by LOFO design — this is the semantic bridge, not leakage.

### ML Engineering
- Pre-register gate metrics before running experiments. No post-hoc substitution.
- Report per-fold deltas alongside aggregates — aggregates mask framework-specific behavior.
- CUDA determinism flags required: cudnn.deterministic=True, cudnn.benchmark=False, CUBLAS_WORKSPACE_CONFIG=:4096:8.
- DeBERTa-v3-NLI fails completely (hit@1=0.000). No classification heads, NLI models, or RoBERTa.
- Hierarchy paths help (+7.6%). Descriptions hurt zero-shot but may help trained models (ablation).
- Bootstrap CIs (10K resamples) and paired deltas for ALL model comparisons.

### Data
- CSA CCM (cloud, 29 CRE links) ≠ CSA AICM (AI, 243 controls). Never conflate.
- Auto-links are expert-quality (deterministic CAPEC→CWE→CRE chain). Treat as equivalent to human LinkedTo.
- data/raw/ is immutable. Parsers read raw/, write processed/.
- Text sanitization: null bytes → NFC normalize → strip HTML → fix ligatures → collapse whitespace → truncate.

### Engineering
- Typed, validated, tested, deterministic. Every function has a clear contract.
- Fail loud: raise ValueError with specific message. No bare except. No silent None returns.
- Atomic writes everywhere: tempfile + os.replace.
- Logging not print. DEBUG/INFO/WARNING/ERROR levels.
- No eval/exec/shell=True. No pickle. Credentials via `pass` manager.
- Constants in tract/config.py, not scattered across modules.

### OpenCRE Integration
- OpenCRE's REST API import path requires Flask + Redis + full graph loading — too heavyweight for local SQLite. Bypass with direct SQLAlchemy import.
- `dbNodeFromNode()` creates ORM objects WITHOUT the database primary key → `add_link()` crashes on `object_select()`. Query the ORM `Node` table directly to get objects with `id` set.
- MITRE ATLAS has pre-existing upstream data (44 nodes/links). Import must match existing nodes by name/section/section_id, not create duplicates.
- OpenCRE's `export_format_parser.parse_export_format()` is the canonical CSV parser. Target its format: `CRE 0` = `"hub_id|hub_name"` (pipe-delimited).
- Coverage gaps report is essential for "Top N" frameworks where users expect complete coverage. Always generate alongside CSVs.
- `INSECURE_REQUESTS=1` and `NO_LOGIN=1` env vars required for local OpenCRE Flask app.

### Framework Onboarding
- Front-load validation before expensive model inference. `tract validate` catches data quality issues in seconds; `tract ingest` takes minutes.
- Confidence scores from `tract prepare` are ordinal cosine similarities (T=0.074), not calibrated probabilities. Always include calibration disclaimer for prepare-sourced data.
- CSV extractor must handle BOM (utf-8-sig) — Excel exports include BOM by default. Use `encoding='utf-8-sig'` which handles both BOM and plain UTF-8.
- Flexible column aliases prevent user frustration: accept control_id/id/section_id, title/name, description/desc/text/body, etc.
- Protocol-based ExtractorRegistry pattern makes adding new extractors trivial — implement `.extract(path) -> list[Control]`.

### Publication & Bridge Analysis
- Cosine similarity = dot product when embeddings are unit-normalized. Verify with `all norms = 1.0000`.
- n_pairs in fold summaries is training pair count (~3891-3931), NOT eval item count. Use len(predictions.json) for eval counts.
- corrected_metrics.json must come from the TEXTAWARE run (deployment model), not PRIMARY (different model). Cross-model contamination silently produces wrong model card numbers.
- LoRA merge: verify cosine > 0.9999 between pre/post-merge embeddings. Merge can corrupt weights silently.
- Publication gate BEFORE upload: bridge report exists, no pending candidates, hierarchy version updated, accepted bridges in related_hub_ids. Prevents publishing incomplete work.
- TypedDict for domain objects shared across 5+ modules (BridgeCandidate has 8 fields). Field name typos become type errors instead of silent KeyErrors.
- allow_pickle=False on all np.load calls. hub_ids dtype is `<U7` (fixed unicode), pickle never needed.
- Token handling: `pass` → `HfApi(token=token)` → `del token` in finally. No os.environ leak.

### Process
- Adversarial review catches real methodology errors (R2's 4-fold vs 5-fold comparison, n_pairs vs eval count).
- Cross-examination between critics is essential — without it, false findings persist.
- Small test fixtures (9 CREs, 3 frameworks) catch real bugs.
- Two-stage review (spec compliance + code quality) is an effective quality gate.
- Marathon subagent runs (one subagent completing 10+ tasks) work when the plan is well-specified. The key enabler: complete code in every plan step, no placeholders.
- Three-round adversarial review convergence: findings shift from "will produce wrong results" to "will confuse the implementer" → safe to stop.
