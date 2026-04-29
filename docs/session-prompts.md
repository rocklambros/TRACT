# TRACT Phase-by-Phase Session Prompts

Standalone prompts for continuing the TRACT project in new Claude Code sessions. Each prompt is self-contained: paste it to start a session. Prompts are sequential — complete each before moving to the next.

**Current state (as of 2026-04-28):**
- Data preparation: COMPLETE (12 parsers, OpenCRE fetch, hub link extraction, all_controls.json)
- Phase 0 code: COMPLETE (4 experiments, RunPod infra, tests, security-hardened)
- Phase 0 execution: NOT YET RUN on GPUs

---

## Prompt 1: Phase 0 — Execute Experiments & Gate Evaluation

```
TRACT Phase 0: Execute zero-shot baseline experiments on RunPod and evaluate gate criteria.

Read PRD.md Section 5 and docs/superpowers/plans/2026-04-28-phase0-zero-shot-baselines.md for full context.

## What's already built
All experiment code is committed and tested (90 tests passing):
- scripts/phase0/common.py — shared infrastructure (hierarchy, LOFO, metrics, bootstrap CIs)
- scripts/phase0/exp1_embedding_baseline.py — BGE, GTE bi-encoders + DeBERTa cross-encoder
- scripts/phase0/exp2_llm_probe.py — Two-stage Opus probe (branch shortlist → final rank)
- scripts/phase0/exp3_hierarchy_paths.py — Path-enriched hub text comparison
- scripts/phase0/exp4_hub_descriptions.py — LLM description pilot (50 hubs)
- scripts/phase0/runpod_setup.sh — 3x A100 80GB provisioning with parallel phases
- scripts/phase0/run_summary.py — Aggregation and gate evaluation

## What to do
1. Review runpod_setup.sh and confirm the provisioning plan looks correct
2. Help me execute `./scripts/phase0/runpod_setup.sh all` — troubleshoot any issues
3. Once results are collected in results/phase0/, analyze them:
   - Parse exp1 results: which model (BGE vs GTE vs DeBERTa) performed best on hit@1, hit@5, MRR?
   - Parse exp2 results: what did Opus achieve? Is hit@5 > 0.50? (Gate A)
   - Parse exp3 results: do hierarchy paths help? Are paired deltas significant?
   - Parse exp4 results: do LLM descriptions improve the described-hub subset?
   - Run run_summary.py and interpret the gate table
4. Evaluate gate criteria (PRD Section 5):
   - Gate A: Opus hit@5 > 0.50 on all-198 track → task is feasible
   - Gate B: Best embedding hit@1 is at least 0.10 below Opus hit@1 → room for trained model
5. Write a results analysis document at docs/phase0-results.md with:
   - Summary table with all metrics and CIs
   - Per-fold breakdown (which frameworks are hardest?)
   - Gate pass/fail with evidence
   - Recommendations for Phase 1 model architecture based on what we learned
6. Commit everything

## Key context
- 198 AI evaluation items across 5 frameworks (MITRE ATLAS: 65, OWASP AI Exchange: 65, NIST AI 100-2: 45, OWASP LLM Top10: 13, OWASP ML Top10: 10)
- Two evaluation tracks: "full-text" (125 items with parsed descriptions) and "all-198" (73 use section_name fallback)
- RunPod pods write model-specific output files (e.g. exp1_embedding_baseline_bge.json) — collect() merges them
- ANTHROPIC_API_KEY needed for exp2 and exp4 (set env var or ensure `pass anthropic/api_key` works)

## If gates fail
If Gate A fails (Opus can't do this task): we need to reassess whether CRE hub assignment is viable at this granularity. Consider coarser labels (parent hubs instead of leaf hubs).
If Gate B fails (embeddings already match Opus): a trained model may not be worth the effort — consider shipping the best embedding model directly with confidence calibration.
Document the analysis either way.
```

---

## Prompt 2: Phase 1A — CRE Hierarchy, Hub Descriptions & Traditional Frameworks

```
/using-superpowers TRACT Phase 1A: Build the production CRE hierarchy, generate all 400 hub descriptions, and ingest 13 traditional frameworks from OpenCRE. PRD Sections 6.1, 6.2, 6.3.

Read PRD.md Sections 6.1-6.3 for full requirements. Read CLAUDE.md for code standards.

## Prerequisites — verify before starting
- Phase 0 experiments have been run and both gates passed (check docs/phase0-results.md)
- Phase 0 results inform this work: check which embedding model performed best, whether hierarchy paths helped, whether descriptions helped the pilot

## Phase 0 results to incorporate
Before designing, read results/phase0/ and docs/phase0-results.md to answer:
1. Did hierarchy paths improve hit@1/hit@5? (exp3 results) → Informs whether hierarchy_path should be a first-class feature in hub representations
2. Did descriptions improve the 50-hub pilot? (exp4 results) → Informs how aggressively to invest in description quality
3. Which embedding model performed best? → Informs which encoder to use for description quality validation (semantic similarity between generated description and linked standards)
4. Which LOFO folds were hardest? → Informs where to focus expert review effort

## Lessons learned from previous phases
- LLM output is untrusted input. All generated descriptions MUST be sanitized (_sanitize_text: strip null bytes, NFC normalize, enforce max length) before storage or use as model features.
- Constants belong in one place. Import from tract/config.py — do not duplicate values across modules.
- Async API clients must be closed in finally blocks. Use try/finally with await client.close() for all Anthropic API usage.
- API calls need timeouts. Use asyncio.wait_for() + client-level timeout + max_retries.
- Test with small representative fixtures first, then validate on real data.
- Security review as a parallel agent catches real bugs — run one after implementation.
- The hub firewall operates at the LINK level, not text level. Substring matching produces false positives. This matters for how hub descriptions interact with evaluation.

## Scope

### 6.1 Production CRE Hierarchy
We already have build_hierarchy() in scripts/phase0/common.py. For Phase 1, promote this to a proper tract/ module:
- tract/hierarchy.py — CREHierarchy class with full tree operations
- Queryable: parent lookup, child listing, path computation, branch enumeration, leaf detection
- Validated: no cycles, all leaves reachable from roots, depth matches expectations
- Versioned: include fetch timestamp, data hash
- Output: data/processed/cre_hierarchy.json (the "coordinate system")
- Tests: comprehensive hierarchy validation

### 6.2 Hub Description Generation (ALL 400 leaf hubs)
Phase 0 piloted 50 hubs. Now do all 400.
- Generate 2-3 sentence descriptions for every leaf hub
- Input per hub: name, hierarchy path, all linked standard section names
- Output: what it covers, what distinguishes it from siblings, scope boundary
- Expert review workflow: CLI-based review tool (accept/edit/reject per hub)
- Rate: ~5 min review per hub × 400 = ~33 hours (batch in sessions of 50)
- Store: data/processed/hub_descriptions.json with review status per hub
- The descriptions become part of the model's hub representation — quality matters

### 6.3 Traditional Framework Ingestion
13 traditional frameworks appear in OpenCRE but we haven't parsed them separately:
CAPEC (1799 links), CWE (613), OWASP Cheat Sheets (391), NIST 800-53 (300), ASVS (277), DSOMM (214), WSTG (118), ISO 27001 (94), NIST 800-63 (79), OWASP Proactive Controls (76), ENISA (68), NIST SSDF (46), ETSI (36), SAMM (30), Cloud Controls Matrix (29), BIML (21), OWASP Top 10 2021 (17)

These frameworks' control texts come from OpenCRE link metadata (section names, section IDs). No separate source files needed — extract from opencre_all_cres.json.
- Characterize mapping units per framework
- Output: one standardized JSON per framework in data/processed/frameworks/
- Update all_controls.json with all 22+ frameworks
- CSA Cloud Controls Matrix (CCM, 29 CRE links) is a DIFFERENT framework from CSA AI Controls Matrix (AICM, 243 controls). Never conflate them.

## Approach
Use the brainstorming → spec → plan → subagent-driven-development pipeline. Run a security review agent after implementation. Parallelize independent tasks. Target: comprehensive tests for every module. Before presenting the spec, run 5 competing agents with different competing personas to analyze and debate the spec and then a 6th judge to build concensus. They must debate with each other until the judge can decide. Then, a 7th agent acts as the brutal evaluator of the methodologies used in troubleshooting and diagnosing this fix and new approach. Challenge my assumptions and choices, expose blind spots, and name opportunity costs. If my reasoning is weak, dissect it and show why. Skip validation, flattery, and softening. Show reasoning before conclusions, then give a precise, prioritized plan for what to change. We need an optimal outcome once and for all. 

## Success criteria (PRD Section 11)
- CRE hierarchy built and validated: all 522 hubs, no cycles, all leaves reachable
- 400 hub descriptions generated (expert review is ongoing — start the process)
- 22 frameworks fully ingested with correct mapping units
- All tests passing, code typed and validated

Think deeply using the seqauential-thinking MCP server and save memories accordingly. --ultrathink
```
'''
Human review of data/processed/hub_descriptions.json

Review guide saved to docs/hub-description-review-guide.md. It covers:

  - What they're reviewing and why it matters
  - Three quality criteria to check per description (concrete definition, sibling distinction, scope boundary)
  - How to mark reviews — accept/edit/reject with exact JSON field instructions
  - Common problems to watch for (parent/child confusion, vague boundaries, factual errors)
  - Workflow tips — batch in sessions of 50, work by hierarchy path, save frequently
  - Concrete examples of accept, edit, and reject reviews using real descriptions from the file
  - How to check progress via validate_descriptions
  - How to look up linked standards in hub_links.jsonl when unsure



---

## Prompt 3: Phase 1B — CRE Hub Assignment Model

```
/using-superpowers TRACT Phase 1B: Combine what you said abouve about:

Phase 1B — What's Next                                                                                                                  
                                                                  
  The remaining Phase 1 sections (6.4–6.11) form the ML training pipeline, evaluation, and CLI. They naturally group into three waves:    
                                                                  
  Wave 1 — Model Training Core (6.4, 6.5, 6.11)                                                                                           
  - 6.4 Hub Assignment Model — Contrastive fine-tuning of BGE-large-v1.5 bi-encoder (Phase 0 best baseline). Training
  data: 4,406 standard-to-hub links. Hub representation = name + hierarchy path (descriptions as ablation experiment).                                   
  - 6.5 Hub Representation Firewall — When evaluating framework X, rebuild hub representations without X's linked sections. Hard
  requirement for honest LOFO eval.                                                                                                       
  - 6.11 Evaluation — Leave-one-framework-out cross-validation. hit@1, hit@5, MRR. Must beat Phase 0 baselines (Opus hit@1=0.465, BGE     
  hit@1=0.348).                                                                                                                      
                                                                                                                                          
  Wave 2 — Guardrails & Active Learning (6.6, 6.7)                
  - 6.6 Guardrails — Five categories: data integrity, model integrity, output integrity (conformal prediction, confidence thresholds),    
  adversarial robustness (paraphrase probes, OOD detection), provenance tracking.                                                         
  - 6.7 Active Learning Loop — Model predicts hubs for zero-coverage frameworks (AIUC-1, CSA AICM, CoSAI, etc.), expert reviews, retrain. 
  Target >80% acceptance rate.                                                                                                            
                                                                                                                                          
  Wave 3 — Product Layer (6.8, 6.9, 6.10)
  - 6.8 Crosswalk Database — SQLite with confidence scores, provenance, cross-framework relationship matrix.                              
  - 6.9 CLI Tool — tract assign, tract compare, tract ingest, tract export.                                                               
  - 6.10 Hub Proposal System — HDBSCAN clustering of OOD controls, 6-guardrail filter, review interface.                                  
                                                                                                                                          
  The big decision is whether to tackle this as one combined design or break it into sub-projects. Given the scope, I'd recommend two     
  specs: one for the training pipeline (6.4+6.5+6.11) and one for everything else (6.6–6.10). The model needs to exist before guardrails  
  and active learning make sense. 

  With the following prompt from session-prompts.md in this project:

Train the CRE hub assignment model. PRD Sections 6.4, 6.5, 6.11.

Read PRD.md Sections 6.4, 6.5, 6.11 for full requirements. Read CLAUDE.md for code standards (especially ML Engineering section).

## Prerequisites — verify before starting
- Phase 1A complete: cre_hierarchy.json exists, hub_descriptions.json exists (at least generated, review ongoing), all 22 frameworks ingested
- Phase 0 results analyzed: check docs/phase0-results.md for baseline numbers to beat

## Phase 0 & 1A results to incorporate
Before designing the model, extract these answers from previous results:
1. Best embedding model from Phase 0 → starting point for encoder selection
2. Whether hierarchy paths helped → include as feature or not
3. Whether descriptions helped → confirms description investment was worthwhile
4. Per-fold difficulty → expect smaller frameworks (LLM Top10: 10 items, ML Top10: 10 items) to have wide CIs
5. Opus ceiling → this is the target to approach (not necessarily beat — Opus has 6 calls per control)
6. Full-text vs all-198 gap → how much does rich control text matter vs section names?
7. Hub description quality from expert review → are descriptions ready for use?

## Lessons learned from previous phases
- Assignment paradigm only: g(control_text) → CRE_position. NEVER pairwise f(A,B) → relationship. If you find yourself comparing two controls directly, you're doing it wrong.
- Hub firewall is non-negotiable. When evaluating framework X, rebuild hub representations WITHOUT X's linked sections. Assert this programmatically in every evaluation path.
- LOFO is the only valid evaluation. Leave-one-framework-out. Never hold out random controls. Never use a frozen test set.
- No pairwise metrics. hit@1, hit@5, MRR, NDCG on hub assignment only.
- Bootstrap CIs (10,000 resamples) and paired deltas are essential for comparing models. Use them for every comparison.
- Auto-links (AutomaticallyLinkedTo) are expert-quality — treat as equivalent to human LinkedTo (penalty=0).
- Constants: import thresholds/seeds/paths from tract/config.py.
- Experiment tracking: every run logs data hash, hyperparameters, git SHA, seed, full metrics. Use WandB.
- Checkpoint discipline: save model + optimizer + scheduler + epoch + metrics. Never just weights.
- Deterministic: set seeds explicitly, sort keys in output, pin versions.
- Security: use safetensors for model weights (not pickle). Credentials via `pass` manager.

## Scope

### 6.4 Hub Assignment Model
- Architecture: contrastive fine-tuning of BGE-large-v1.5 bi-encoder (Phase 0 best baseline, hit@1=0.348, 0.424 with paths)
- Hub representation: hub name + hierarchy path, encoded as target embeddings. Descriptions as ablation experiment (Phase 0 showed they hurt zero-shot).
- Training data: 4,406 standard-to-hub links from OpenCRE (2,047 human + 2,359 expert-transitive)
- Training strategy: contrastive learning with hard negatives (sibling hubs). Transfer learning (all 4,406 vs AI-only 198 vs two-stage) is an ablation, not prescribed.
- Multi-label: median 1 hub/section but max 38. Per-hub similarity threshold tuning.
- Output: cosine similarity scores over 400 leaf hubs, calibrated to probabilities via temperature/Platt scaling.
- Phase 0 disproved: DeBERTa-v3-NLI (hit@1=0.000), classification heads, RoBERTa-large (old project only).

### 6.5 Hub Representation Firewall
Already implemented in scripts/phase0/common.py build_hub_texts(). Promote to tract/ module.
Must work at the LINK level: remove held-out framework's links, rebuild hub text, re-encode.
This is a build step, not a runtime hack — hub embeddings are pre-computed per fold.

### 6.11 Evaluation
- LOFO cross-validation on 5 AI frameworks
- Primary metrics: hit@1, hit@5, MRR, NDCG@10 with bootstrap CIs
- Compare against ALL Phase 0 baselines — trained model MUST exceed them
- Per-fold breakdown to identify hard frameworks
- Transfer learning A/B: compare with vs without traditional pre-training
- Full-text vs all-198 track analysis

## Approach
Use brainstorming → spec → plan → subagent-driven-development. This is the most complex ML engineering phase — break into smaller tasks: data pipeline, model architecture, training loop, evaluation harness, experiment runner. WandB integration from the start. Security review for credential handling and model serialization. Run a security review agent after implementation. Parallelize independent tasks. Target: comprehensive tests for every module. Before presenting the spec, run 5 competing agents with different competing personas to analyze and debate the spec and then a 6th judge to build concensus. They must debate with each other until the judge can decide. Then, a 7th agent acts as the brutal evaluator of the methodologies used in troubleshooting and diagnosing this fix and new approach. Challenge my assumptions and choices, expose blind spots, and name opportunity costs. If my reasoning is weak, dissect it and show why. Skip validation, flattery, and softening. Show reasoning before conclusions, then give a precise, prioritized plan for what to change. We need an optimal outcome once and for all. 

## Success criteria (PRD Section 11)
- Trained model hit@1 > Phase 0 embedding baseline by 0.10+
- Trained model hit@5 > 0.70
- All Phase 0 baselines exceeded with statistically significant margins
- Full experiment logs with data hash, hyperparams, git SHA
- Model checkpoint with optimizer + scheduler + metrics

Think deeply for this challenge using the sequential-thinking MCP

--ultrathink

```



---

## Prompt 4: Phase 1C — Guardrails, Active Learning & Crosswalk Database

```
TRACT Phase 1C: Implement all 5 guardrail categories, the active learning loop, and the crosswalk database. PRD Sections 6.6, 6.7, 6.8.

Read PRD.md Sections 6.6, 6.7, 6.8 for full requirements. Read CLAUDE.md for code standards.

## Prerequisites — verify before starting
- Phase 1B complete: trained model exists, exceeds Phase 0 baselines, evaluation results documented
- Hub descriptions reviewed (at least first batch — active learning needs the model)
- All 22 frameworks ingested

## Results from previous phases to incorporate
1. Model confidence distribution → informs confidence threshold for Output Integrity guardrails
2. Per-hub prediction accuracy → identifies hubs that need more training data (active learning targets)
3. OOD characteristics → which controls got low-confidence predictions? What do they look like?
4. Transfer learning results → did traditional pre-training help? Informs active learning strategy.
5. Multi-label distribution → how often does the model predict multiple hubs? Informs disagreement detector thresholds.

## Lessons learned from previous phases
- Validate at boundaries, trust internals. Every guardrail component validates its inputs and outputs at the edges.
- Fail loud: raise ValueError with specific messages, never return None to signal failure. No bare except.
- Defensive I/O: atomic writes for all database operations and file writes.
- Test-driven: write the failing test first for every guardrail. Guardrails that aren't tested don't exist.
- Security: no eval/exec/shell=True. All external data is untrusted. Sanitize text at ingestion boundaries.
- Logging not print. DEBUG for internal state, INFO for pipeline progress, WARNING for recoverable issues.
- Small representative test fixtures catch real bugs — build fixtures that exercise edge cases (hubs with 1 link, hubs with 100+ links, multi-label controls).

## Scope

### 6.6 Guardrails (5 categories, all concrete and testable)

**Data Integrity:**
- Framework-level train/test split logic (LOFO — already have this, promote to tract/ module)
- Mapping-level dedup (detect same control appearing via different CRE paths)
- Auto-link provenance chain stored per link
- Source version pinning (framework version + fetch date per record)

**Model Integrity:**
- Hub representation firewall (promote from common.py to tract/)
- Confidence calibration module (temperature/Platt scaling)
- Per-hub threshold tuner (optimize on validation, freeze before test)
- Transfer learning A/B comparison (already done in 1B, formalize the comparison)

**Output Integrity:**
- Confidence threshold filter (low-confidence → flagged, not committed)
- Conformal prediction wrapper (coverage ≥ 90%)
- Multi-hub disagreement detector (flag near-equal competing predictions)

**Adversarial Robustness:**
- Paraphrase probe test set (same controls reworded → same hub assignments)
- OOD detector (max confidence below threshold → flag as "no good hub match")
- OOD → hub proposal pipeline trigger

**Provenance Tracking:**
- Prediction log schema (control text, hub(s), scores, model version, timestamp)
- Human review tracking (accept/reject/correct linked to prediction ID)
- Training data lineage per example
- Model version tag (training data hash + hyperparameters)

### 6.7 Active Learning Loop
For 5 zero-coverage AI frameworks (AIUC-1: 132, CSA AICM: 243, CoSAI: 29, EU GPAI CoP: 32, OWASP Agentic: 10):
1. Model predicts top-K hub assignments
2. Expert reviews via CLI (accept/reject/correct)
3. Accepted predictions added to training data with provenance="active_learning_round_N"
4. Model retrained on expanded dataset
5. Repeat until expert acceptance rate > 80%
This is ~446 controls needing initial review.

### 6.8 Crosswalk Database
- SQLite database: crosswalk.db
- Schema: controls, hubs, assignments (with confidence + provenance), frameworks, reviews
- Cross-framework relationships derived transitively from shared hub assignments
- Every assignment traced to: model version, training data version, review status
- Export scripts: JSON, CSV per framework or full matrix

## Approach
Use brainstorming → spec → plan → subagent-driven-development. Break into: guardrail modules, active learning CLI, crosswalk DB schema + ORM, export pipeline. Security review for database operations (SQL injection prevention — use parameterized queries only).

## Success criteria (PRD Section 11)
- All 5 guardrail categories pass automated test suite
- Active learning: >80% expert acceptance rate by round 3
- Crosswalk database complete with all 22 frameworks
- Every prediction traceable to model version + training data version
```

---

## Prompt 5: Phase 1D — CLI Tool & Hub Proposal System

```
TRACT Phase 1D: Build the CLI tool and guardrailed hub proposal system. PRD Sections 6.9, 6.10.

Read PRD.md Sections 6.9, 6.10 for full requirements. Read CLAUDE.md for code standards.

## Prerequisites — verify before starting
- Phase 1C complete: all guardrails implemented, crosswalk.db populated, active learning started
- OOD detector from guardrails identifies controls that don't map to existing hubs

## Results from previous phases to incorporate
1. OOD detection results → how many controls are flagged as "no good hub"? This determines hub proposal volume.
2. Active learning acceptance rates → are reviewers finding that existing hubs are insufficient? Do they suggest new hubs?
3. Confidence calibration curves → verify the OOD threshold is well-calibrated before feeding proposals.
4. Crosswalk coverage → which framework pairs have the most/least overlap? CLI compare command should highlight this.

## Lessons learned from previous phases
- CLI tools should use argparse with clear subcommands. Follow the pattern in exp1-exp4 scripts.
- All file outputs use atomic writes (tempfile + os.replace).
- Deterministic: HDBSCAN clustering must produce identical results on re-run. Pin min_cluster_size, min_samples, and use a fixed metric.
- User-facing output goes through logger.info, not print (even in CLI scripts — the logging format is configurable).
- Validate all user inputs at the boundary. Control text max 2000 chars, sanitized.
- Security: the CLI will be used by security professionals. No eval/exec. Input text treated as untrusted.

## Scope

### 6.9 CLI Tool
Five core commands:

tract assign "control text here"
  → Top-5 hub assignments with confidence scores, hierarchy paths

tract compare --framework atlas --framework asvs
  → Relationship matrix: shared hubs (equivalent), parent/child (related), disjoint (gap)

tract ingest --file new_framework.json --template standard
  → Schema validation, model inference, predicted assignments for review

tract export --format csv --framework atlas
  → Full crosswalk table from crosswalk.db

tract hierarchy --hub CRE-236
  → Full hierarchy path, linked controls from all frameworks

Implementation: use click or argparse. Each command is a thin wrapper over tract/ library functions. The CLI layer handles I/O; business logic lives in tract/.

### 6.10 Guardrailed Hub Proposal System
When OOD detector flags controls with no good hub match:

1. HDBSCAN clustering on OOD control embeddings (deterministic, seeded)
2. Six-guardrail filter pipeline per proposed cluster:
   - Minimum evidence: 3+ controls from 2+ frameworks
   - Hierarchy constraint: must identify parent hub (no top-level creation)
   - Similarity ceiling: cosine < 0.85 to all existing hubs
   - Budget cap: ~40 proposals max per review cycle
   - Candidate queue writer (stores proposals with evidence)
   - Deterministic reproducibility (re-run = same clusters)
3. CLI review: `tract review-proposals` shows name, parent, controls, similarity scores
4. Accept/reject/edit per proposal
5. Accepted proposals inserted into cre_hierarchy.json as children

Output: hub_proposals/ directory with versioned rounds, acceptance records, hierarchy diffs.

## Approach
Use brainstorming → spec → plan → subagent-driven-development. Break into: CLI framework + commands, hub proposal clustering, 6-guardrail pipeline, review interface, hierarchy updater. Security review for input validation and proposal integrity.

## Success criteria (PRD Section 11)
- Hub proposal system functional end-to-end: OOD detect → cluster → propose → review
- CLI commands working with crosswalk.db
- All commands tested with representative inputs
```

---

## Prompt 6: Phase 2A — Dash Web UI & Framework Submission

```
TRACT Phase 2A: Build the Dash web UI and framework submission system. PRD Sections 7.1, 7.2.

Read PRD.md Sections 7.1, 7.2 for full requirements. Read CLAUDE.md for code standards.

NOTE: PRD says "Phase 2 will be planned in detail after Phase 1 ships and we have real model results." This prompt is a starting point — adapt based on actual Phase 1 outcomes.

## Prerequisites — verify before starting
- Phase 1 fully complete: model trained, crosswalk.db populated, CLI working, hub proposals functional
- All 22 frameworks ingested with hub assignments
- Active learning acceptance rate > 80%

## Phase 1 results to incorporate
1. Crosswalk database schema → Dash pages query this directly
2. Model inference latency → Control Search page needs sub-second response for live inference
3. Hub proposal volume → Ontology Browser should show proposed (pending) hubs differently from established ones
4. Framework coverage matrix → Framework Comparison page should highlight coverage gaps
5. Confidence distribution → Confidence Dashboard heatmap needs appropriate color scale based on actual score ranges
6. Which frameworks have the most/least CRE overlap → prioritize comparison page design for common use cases

## Lessons learned from previous phases
- Atomic writes for any data mutation through the web UI.
- Input sanitization on all user-provided text (control text search, framework upload). Use _sanitize_text pattern.
- All external data (uploaded frameworks) is untrusted. Validate schema before processing.
- Security: no eval/exec. Parameterized SQL queries only. CSRF protection on forms.
- Test the UI in a browser — type checking and test suites verify code correctness, not feature correctness.
- Async API client usage needs timeout + retry + cleanup (if any LLM features in UI).

## Scope

### 7.1 Dash Web UI (5 pages)
- Crosswalk Explorer: framework A → controls → CRE hub(s) → framework B controls at same hub
- Framework Comparison: 2 frameworks side-by-side with equivalent/related/gap breakdown
- Hub Ontology Browser: navigate CRE tree → click hub → linked controls, description, confidence
- Confidence Dashboard: heatmap (frameworks × hubs) colored by assignment confidence
- Control Search: paste text → live model inference → top-5 hubs + related controls

Tech: Dash + Plotly + dash-bootstrap-components (CYBORG theme). SQLite backend.

### 7.2 Framework Submission System
- framework_template.json: JSON Schema defining required fields
- tract validate --file new_framework.json: schema validation + duplicate detection (cosine > 0.95 flagged)
- Upload page in Dash: drag-and-drop JSON, validation results, model inference
- Review queue page: predicted assignments, accept/reject/correct per control
- framework_registry.json: versioned list of all frameworks with metadata

## Approach
Use brainstorming → spec → plan → subagent-driven-development. Design the UI pages as mockups first before implementing. Break into: Dash app skeleton, individual pages, framework submission pipeline, review queue. Security review for web application vulnerabilities (XSS, injection, CSRF).

## Success criteria (PRD Section 11)
- 5 pages deployed, all data sources connected
- New framework submission < 1 hour for standard-format frameworks
- UI tested in browser for golden path and edge cases
```

---

## Prompt 7: Phase 2B — HuggingFace Publication & AI/Traditional Bridge

```
TRACT Phase 2B: Publish model to HuggingFace and identify AI/traditional security bridges. PRD Sections 7.3, 7.4.

Read PRD.md Sections 7.3, 7.4 for full requirements. Read CLAUDE.md for code standards.

## Prerequisites — verify before starting
- Phase 2A complete: web UI functional, framework submission working
- Model finalized (no more retraining planned before publication)
- Hub descriptions fully reviewed by expert

## Phase 1 & 2A results to incorporate
1. Final model metrics (all LOFO folds) → goes into model card
2. Training data hash + hyperparameters → full reproducibility info for model card
3. Environmental impact (GPU hours, energy) → required for AIBOM compliance
4. Active learning rounds completed → documents training data provenance
5. Hub proposal outcomes → new hubs created affect the bridge analysis
6. Which traditional hubs have high-confidence AI framework assignments → seed candidates for bridges

## Lessons learned from previous phases
- CSA Cloud Controls Matrix (CCM, 29 CRE links, traditional) ≠ CSA AI Controls Matrix (AICM, 243 controls, AI). Verify framework identity when computing bridges.
- LLM-generated content (bridge descriptions) must be sanitized before storage.
- All API calls need timeout + retry + cleanup.
- Deterministic: bridge identification must be reproducible (cosine threshold, not random sampling).
- Expert review is the bottleneck. Design review workflows for efficiency (batch, sort by confidence).

## Scope

### 7.3 HuggingFace Model Publication
Published to huggingface.co/rockCO78/tract-cre-assignment:
- Model weights (safetensors format — NOT pickle)
- Bundled: hub_descriptions.json, cre_hierarchy.json
- Model card (AIBOM-compliant, target 100/100): description, intended use, architecture, training details, evaluation results, limitations, ethical considerations, environmental impact, usage snippet, citation
- predict.py: standalone inference script
- train.py: reproduction script with requirements and data download
- README.md: usage documentation

### 7.4 AI/Traditional Security Bridge
81 AI hubs + 441 traditional hubs. Find conceptual bridges:
1. Compute embedding similarity: each AI hub vs all traditional hubs. Flag cosine > 0.70.
2. Use ENISA (68 links), ETSI (36), BIML (21) as seed evidence — they appear on both AI and traditional hubs.
3. For each bridge candidate: LLM-generate bridge description explaining overlap.
4. Expert review: accept bridge, reject, or propose new parent hub.
5. Accepted bridges → new Related links in cre_hierarchy.json.
6. New parent hubs via guardrailed proposal system (6.10).
7. Model retrained with bridge links (if significant new training signal).
8. Output: bridge_report.json documenting all bridges with evidence.

## Approach
Use brainstorming → spec → plan → subagent-driven-development. HuggingFace publication and bridge identification are independent — can be parallelized. Security review for model publication (no secrets in model artifacts, no PII in training data).

## Success criteria (PRD Section 11)
- HuggingFace AIBOM score: 100/100
- At least 10 validated AI/traditional bridges
- bridge_report.json with full evidence and review status
```

---

## Prompt 8: Phase 3 — Published Human-Reviewed Crosswalk Dataset

```
TRACT Phase 3: Produce and publish the human-reviewed crosswalk dataset. PRD Section 8.

Read PRD.md Section 8 for full requirements. Read CLAUDE.md for code standards.

## Prerequisites — verify before starting
- All Phase 2 work complete: model published, bridges identified, web UI functional
- crosswalk.db contains predictions for all 22 frameworks
- Active learning rounds completed with >80% acceptance rate

## Results from all previous phases to incorporate
1. Model confidence per framework → prioritize review order (start with highest-confidence frameworks for quick wins)
2. Active learning acceptance rates per framework → frameworks with low acceptance need more careful review
3. Bridge hubs from Phase 2B → include bridge relationships in the published dataset
4. Hub proposals accepted → expanded hierarchy included in publication
5. Per-control confidence scores → reviewer can focus time on low-confidence predictions
6. Inter-framework overlap patterns → inform review batching strategy (review overlapping controls together)

## Lessons learned from previous phases
- Expert review is ~5 min/control. 749+ AI controls = ~62 hours minimum. Budget accordingly.
- Atomic writes for review state. If a review session crashes, no work is lost.
- Deterministic: review state serialization must be stable (sorted keys, consistent encoding).
- Provenance: every reviewed assignment links back to model version, training data version, reviewer ID, timestamp.
- Active learning showed which prediction patterns experts correct most → build review interface to surface these patterns.

## Scope

### Full Review Workflow
1. Export all model predictions from crosswalk.db (every control, all 22 frameworks)
2. Group by framework. Per framework, generate review spreadsheet:
   control_id | control_text | predicted_hub_1 (confidence) | predicted_hub_2 | ... | reviewer_decision | notes
3. Expert reviews one framework at a time:
   - Accept top prediction
   - Select different hub
   - Flag for discussion
   - Mark "no good hub" (→ hub proposal pipeline)
4. Track: total reviewed, acceptance rate, edit rate, rejection rate, time per control
5. Second-pass review for rejected/edited controls (consistency check)
6. Inter-reviewer agreement metrics if multiple reviewers (Cohen's kappa)
7. Freeze as crosswalk_reviewed_v1.0.jsonl

### Publication
Published dataset structure:
- crosswalk_v1.0.jsonl — every control + hub assignment + confidence + review status
- framework_metadata.json — 22 framework descriptions, versions, sources
- cre_hierarchy_v1.0.json — hub ontology at time of publication
- hub_descriptions_v1.0.json — validated hub descriptions
- review_metrics.json — acceptance rates, agreement metrics
- README.md — dataset card (HuggingFace Datasets format)
- LICENSE — Apache 2.0 or CC-BY-4.0

Publish to: HuggingFace Datasets AND Zenodo (for DOI).
Contribute back to OpenCRE: submit validated assignments for 5 zero-coverage AI frameworks as proposed LinkedTo links.

## Approach
Use brainstorming → spec → plan → subagent-driven-development. Break into: review interface tooling, review state management, publication formatting, HuggingFace/Zenodo upload pipeline.

## Success criteria (PRD Section 11)
- 100% of predicted assignments reviewed by expert
- Published dataset available on HuggingFace Datasets
- Review metrics documented
```

---

## Prompt 9: Phase 4 — Secure API with Documentation

```
TRACT Phase 4: Build and deploy the secure API with full documentation. PRD Section 9.

Read PRD.md Section 9 for full requirements. Read CLAUDE.md for code standards.

## Prerequisites — verify before starting
- Phase 3 complete: reviewed dataset published, crosswalk.db finalized
- Model published to HuggingFace (inference code available)
- CLI commands working (API wraps the same business logic)

## Results from all previous phases to incorporate
1. Model inference latency from Phase 2A Control Search page → baseline for API latency target (<500ms)
2. Crosswalk database schema → API endpoints query this
3. Common query patterns from web UI usage → optimize API for actual usage patterns
4. Framework submission workflow from Phase 2A → /v1/ingest endpoint mirrors this
5. Security patterns established throughout project → apply consistently to API

## Lessons learned from all previous phases
- Input sanitization: all user text max 2000 chars, null byte stripped, NFC normalized. Already have _sanitize_text — reuse it.
- No eval/exec/shell=True. Parameterized SQL only. No pickle (safetensors for models).
- Credentials via environment variables or pass manager. Never in code, config, or logs.
- API keys: don't log them, don't include in error messages, don't return in responses.
- Defensive I/O: timeouts on all operations, retry with backoff for downstream calls.
- Atomic writes for any state mutation.
- Structured logging (not print) with request IDs for traceability.
- Rate limiting must be per-key, not global — prevent one user from blocking others.

## Scope

### API Endpoints
| Method | Endpoint | Function |
|--------|----------|----------|
| POST | /v1/assign | Control text → top-K hub assignments with confidence |
| GET | /v1/hub/{cre_id} | Hub description, hierarchy path, linked controls |
| GET | /v1/compare?fw1=X&fw2=Y | Cross-framework relationship matrix |
| GET | /v1/framework/{fw_id} | All controls with assignments and confidence |
| POST | /v1/ingest | Framework JSON → validation + predicted assignments |
| GET | /v1/hierarchy | Full CRE hierarchy tree |
| GET | /v1/search?q=... | Text search across controls with hub assignments |
| GET | /v1/health | Health check |

### Security
- API key authentication (issued per user/org)
- Rate limiting: 100/min for /assign (model inference), 1000/min for reads
- Input validation: max 2000 chars, sanitized
- HTTPS only
- No PII stored; prediction logs anonymized
- CORS configuration for web clients

### Documentation
- OpenAPI 3.0 spec (auto-generated from endpoint definitions)
- Hosted Swagger UI at /docs
- Python SDK: pip install tract-client
- Usage examples in README

### Deployment
- Dockerfile + docker-compose.yml (API + SQLite + model weights)
- Health check endpoint
- Deployment guide: local, cloud, Docker

## Approach
Use brainstorming → spec → plan → subagent-driven-development. Break into: API framework (FastAPI recommended), endpoint implementations, auth/rate-limiting middleware, OpenAPI spec, Python SDK, Docker packaging, deployment guide. Security review is CRITICAL for this phase — it's the only internet-facing component.

## Success criteria (PRD Section 11)
- API latency < 500ms per single control assignment
- Complete OpenAPI spec
- Python SDK published
- Docker deployment working
- All security measures tested
```

---

## Cross-Phase Lessons Learned Reference

These compound over time. Each prompt above incorporates the relevant subset.

### From Data Preparation
- 12 parsers completed; EU AI Act HTML was hardest (15,530 lines EUR-Lex)
- OWASP AI Exchange ID matching requires normalization (lowercase, strip hyphens/underscores/spaces/slashes)
- NIST AI 600-1 may appear as "NIST AI 100-2" in OpenCRE — verify framework identity
- CSA CCM (cloud, 29 CRE links) ≠ CSA AICM (AI, 243 controls, zero CRE links) — ALWAYS verify
- Parser pattern: read raw/ → validate input → transform → validate output against Pydantic schema → write processed/
- Text sanitization: null bytes → NFC normalize → strip HTML → fix ligatures → collapse whitespace → truncate

### From Phase 0 Implementation
- Hub firewall operates at LINK level, not text level. Substring matching gives false positives.
- LOFO folds are uneven: ATLAS(65), AIE(65), NIST(45), LLM(13), ML(10)
- Two tracks: full-text(125 items) vs all-198(73 use section_name fallback)
- Bootstrap CIs (10K resamples, vectorized numpy) are fast — use everywhere
- Paired bootstrap deltas essential for model comparison on same data
- Cross-encoder is O(controls × hubs), bi-encoder is O(controls + hubs)
- LLM probe costs ~6 API calls/control × 198 = ~1,188 calls total
- RunPod parallel pods need model-specific output filenames to avoid overwrites
- Async clients: always try/finally with await client.close()
- API calls: asyncio.wait_for() + client timeout + max_retries
- Security reviews caught: shell injection, path traversal, input validation gaps, resource leaks
- Constants in one place (tract/config.py), not duplicated across modules
- Decide API boundaries upfront — private functions imported cross-module should be public
- Small test fixtures (9 CREs, 3 frameworks) catch real bugs
- Two-stage review (spec compliance + code quality) is an effective quality gate
