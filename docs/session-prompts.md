# TRACT Session Prompts

Self-contained prompts for continuing the TRACT project in new Claude Code sessions. Each prompt bootstraps a fresh session with full context. Complete sequentially.

**Project state (2026-04-30):**
- Data Preparation: COMPLETE
- Phase 0 (Zero-Shot Baselines): COMPLETE — Gates A+B passed
- Phase 1A (Hierarchy, Descriptions, Ingestion): COMPLETE
- Phase 1B (Training Pipeline): COMPLETE — Gate 1 CLEAN PASS (hit@1=0.531, delta=+0.132)
- Phase 1C (Guardrails, Active Learning, Crosswalk DB): COMPLETE — 2 AL rounds converged, 636 assignments, 339 tests
- Phase 1D (CLI, Hub Proposals): COMPLETE — 8 CLI commands, hub proposal pipeline, 394 tests
- Phase 2+ : NOT STARTED

**Key results driving all remaining work:**
- BGE-large-v1.5 + LoRA rank 16 + MNRL contrastive loss + text-aware batching
- Per-fold deltas vs zero-shot: NIST +0.322, ML +0.285, OWASP-X +0.143, ATLAS +0.006, LLM-10 +0.000
- 262 tests passing, all code typed and validated
- CUDA determinism flags added to training loop
- 5-round adversarial review completed — all findings resolved

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

## Prompt 3: Phase 2A — Next.js Web UI & Framework Submission (Vercel)

```
TRACT Phase 2A: Build the public web UI (PRD 7.1) and framework submission system (PRD 7.2).

Read PRD.md Sections 7.1, 7.2. Read CLAUDE.md for code standards.

NOTE: PRD specifies Dash but we are hosting publicly on Vercel, which requires Next.js.
Dash (Python/Flask) cannot run on Vercel's serverless platform — no persistent process,
no WebSockets, no local SQLite. The 5 pages and their functionality are unchanged;
only the tech stack changes.

## Architecture pivot: Dash → Next.js on Vercel

| PRD Spec (Dash)                  | Vercel Implementation              |
|----------------------------------|------------------------------------|
| Dash + Flask                     | Next.js 15 App Router              |
| Plotly (server-side callbacks)   | Plotly.js or Recharts (client-side) |
| dash-bootstrap-components        | Tailwind CSS + shadcn/ui           |
| SQLite crosswalk.db (local)      | Pre-exported static JSON (SSG)     |
| Live model inference in process  | External inference API (FastAPI)   |
| Docker deployment                | Vercel Git Integration (auto-deploy) |

**Why static-first:** Crosswalk data only changes after active learning rounds (weeks/months).
4 of 5 pages are pure data display. Only Control Search needs a live API call.

**Model inference:** BGE-large-v1.5 + LoRA (~1.3GB) cannot load in a Vercel function (10s timeout,
250MB limit). Run a FastAPI inference server externally (Jetson via Cloudflare Tunnel, or
Modal/RunPod serverless GPU). Next.js API route proxies to it.

## Monorepo structure

```
TRACT/
├── tract/                    # Python ML library (existing, unchanged)
├── scripts/
│   └── export_for_web.py     # NEW: crosswalk.db → JSON for Next.js
├── web/                      # NEW: Next.js app (Vercel root directory)
│   ├── package.json
│   ├── next.config.ts
│   ├── tailwind.config.ts
│   ├── app/                  # App Router pages
│   │   ├── layout.tsx
│   │   ├── page.tsx          # Landing / Crosswalk Explorer
│   │   ├── compare/page.tsx
│   │   ├── hierarchy/page.tsx
│   │   ├── dashboard/page.tsx
│   │   ├── search/page.tsx   # Control Search (calls inference API)
│   │   ├── submit/page.tsx   # Framework submission
│   │   └── api/
│   │       ├── search/route.ts    # Proxy to inference API
│   │       └── submit/route.ts    # Framework upload handler
│   ├── components/           # React components
│   ├── lib/                  # Data loading, types
│   └── public/data/          # Pre-exported JSON from crosswalk.db
└── vercel.json               # Points root to web/
```

## Vercel deployment setup

**Vercel GitHub app is already installed.** Deployment is automatic:
- Push to `main` → **production** deployment
- Push to PR branch → **preview** deployment (acts as dev/staging)

**Environments (dev + prod):**
- Production: `main` branch → custom domain (e.g., tract.rockcyber.com)
- Preview: PR branches → auto-generated *.vercel.app URLs (use for dev/staging)
- Environment variables scoped per environment in Vercel dashboard:
  - `NEXT_PUBLIC_INFERENCE_API_URL` (different per env)
  - `INFERENCE_API_KEY` (server-side only, for proxy route)

**No GitHub Actions needed for deployment** — Vercel handles it via Git integration.
Use GitHub Actions only for CI (pytest, lint, type-check on PRs).

**To create the Vercel project:**
1. Create `web/` with Next.js app
2. Push to GitHub
3. Import repo in Vercel dashboard → set root directory to `web/`
4. Set environment variables per environment
5. Add custom domain in Vercel dashboard

## Current state — verify before starting
1. `python -m pytest tests/ -q` → 394+ tests pass
2. `tract assign "test input"` → CLI works, returns hub assignments with calibrated confidence
3. `tract export --format json` → crosswalk.db populated and queryable (2,802 controls, 522 hubs)
4. `ls results/phase1c/deployment_model/deployment_artifacts.npz` → deployment artifacts exist
5. Phase 1 fully complete: model trained, crosswalk.db populated, CLI working, guardrails tested

## Phase 1 results to incorporate
1. Crosswalk.db → export to static JSON via `scripts/export_for_web.py` (frameworks, controls, assignments, hubs)
2. Model inference latency: cold start ~8s on Jetson Orin AGX → Control Search needs a pre-warmed inference endpoint
3. Calibrated confidence range: T_deploy=0.074, scores typically 0.05–0.85 → design heatmap scale accordingly
4. 95 OOD controls found, 5 hub proposals generated → Ontology Browser should show proposed (pending) hubs
5. 636 assignments across 22 frameworks → design Crosswalk Explorer for this scale
6. Duplicate detection thresholds: 0.95 (duplicate), 0.85 (similar) → Framework Submission uses these

## Lessons from Phase 1
1. **Test the UI in a browser.** Type checking verifies code, not features. Run `next dev` and use every page.
2. **All external data is untrusted.** Framework uploads need schema validation + sanitize_text() before processing.
3. **No eval/exec.** Parameterized queries only. Next.js API routes use typed inputs.
4. **Input sanitization.** Control text search uses the same sanitize_text() logic (port to TypeScript or call Python API).
5. **CSA CCM ≠ CSA AICM.** Verify framework identity in submission validation.

## Scope

### 7.1 Next.js Web UI — 5 pages + submission

| Page | Route | Data Source | Key Interaction |
|------|-------|------------|-----------------|
| Crosswalk Explorer | `/` | Static JSON | Framework dropdown → controls table → click → hub detail → related controls |
| Framework Comparison | `/compare` | Static JSON | Two dropdowns → equivalences/related/gaps table |
| Hub Ontology Browser | `/hierarchy` | Static JSON | Collapsible tree → click hub → detail panel with controls |
| Confidence Dashboard | `/dashboard` | Static JSON | Heatmap (frameworks × hubs) → click cell → prediction detail |
| Control Search | `/search` | Inference API | Text input → POST to /api/search → results with confidence bars |
| Framework Submission | `/submit` | Inference API | Upload JSON → validate → inference → review queue |

Tech: Next.js 15, React 19, Tailwind CSS, shadcn/ui, Plotly.js (heatmap), react-arborist (tree).

### Data export pipeline

`scripts/export_for_web.py` reads crosswalk.db and outputs:
- `web/public/data/frameworks.json` — framework metadata
- `web/public/data/assignments.json` — all accepted assignments with confidence
- `web/public/data/hierarchy.json` — CRE hub tree with descriptions
- `web/public/data/comparison_matrix.json` — pre-computed framework pair overlaps

Run locally after any data change, commit the JSON files, push → Vercel rebuilds.

### Inference API (separate from Vercel)

FastAPI endpoint on Jetson Orin AGX (or Modal/RunPod):
- `POST /predict` — text → top-K hub assignments with calibrated confidence
- `POST /find-duplicates` — text → duplicate/similar matches
- `POST /ingest` — framework JSON → batch predictions + duplicate detection
- Authentication: API key in header
- Exposed via Cloudflare Tunnel (free) for stable public URL

Next.js API route `/api/search` proxies to this, keeping the inference URL server-side.

## Approach
Use brainstorming → spec → plan → subagent-driven-development.

Design the UI layout as mockups first (use the visual companion). Each page is an independent
React component — can be developed and tested in parallel. Start with data export + static pages,
add inference API integration last.

Adversarial review: 3 critics (frontend/UX, security, data visualization) attack the spec.
Focus on: XSS in user text display, API route input validation, responsive layout, accessibility.

## Success criteria
- 5 pages + submission deployed on Vercel and publicly accessible
- Production and preview environments working
- Crosswalk Explorer loads in < 2 seconds (static data)
- Control Search returns results in < 5 seconds (including inference API call)
- Framework submission validates, runs inference, shows review UI
- All pages tested in browser: golden path + edge cases
- No XSS or injection vulnerabilities in any user input path
- Lighthouse performance score > 90 for static pages

Think deeply using the sequential-thinking MCP server. --ultrathink
```

---

## Prompt 4: Phase 2B — HuggingFace Publication & AI/Traditional Bridge

```
TRACT Phase 2B: Publish model to HuggingFace (PRD 7.3) and identify AI/traditional security bridges (PRD 7.4).

Read PRD.md Sections 7.3, 7.4. Read CLAUDE.md for code standards.

## Current state — verify before starting
1. Phase 2A complete: web UI functional, framework submission working
2. Model finalized (no more retraining planned before publication)
3. Hub descriptions fully expert-reviewed: `python -c "import json; d=json.load(open('data/processed/hub_descriptions_reviewed.json')); reviewed=[h for h in d.values() if h.get('review_status')=='accepted']; print(f'{len(reviewed)} reviewed')"`
4. Active learning rounds completed with acceptance rate documented

## Phase 1-2A results to incorporate
1. Final model metrics (all LOFO folds with CIs) → model card
2. Training config: base model, LoRA rank, epochs, batch size, learning rate, seed → model card
3. Data hash + git SHA from training metadata → reproducibility section
4. GPU hours and energy estimate → environmental impact for AIBOM
5. Active learning rounds and acceptance rates → training data provenance
6. Per-fold zero-shot vs fine-tuned comparison table → evaluation section
7. Calibration metrics (ECE, reliability diagrams) → model card limitations section

## Lessons from all previous phases
1. **CSA CCM ≠ CSA AICM.** Verify framework identity when computing bridges.
2. **Safetensors only.** No pickle for model weights. HuggingFace Hub uses safetensors by default.
3. **No secrets in artifacts.** Scan model repo for API keys, credentials, paths containing usernames.
4. **LLM-generated content must be sanitized.** Bridge descriptions go through sanitize_text() before storage.
5. **Deterministic bridge identification.** Cosine threshold, not random sampling. Same inputs = same bridges.

## Scope

### 7.3 HuggingFace Publication
Publish to huggingface.co/rockCO78/tract-cre-assignment:
- Model weights (safetensors, LoRA adapters + base model reference)
- Bundled data: hub_descriptions.json, cre_hierarchy.json
- Model card targeting AIBOM 100/100: description, intended use, architecture, training details, evaluation results (full LOFO table with CIs), limitations (ATLAS flat performance, calibration caveats), ethical considerations, environmental impact, usage code snippet, citation
- predict.py: standalone inference script
- train.py: reproduction script with pinned requirements

### 7.4 AI/Traditional Security Bridge
81 AI-specific CRE hubs + 441 traditional hubs. Find conceptual bridges:
1. Compute embedding similarity: each AI hub representation vs all traditional hub representations. Flag cosine > 0.70.
2. Use ENISA (68 links), ETSI (36), BIML (21) as seed evidence — they appear on both AI and traditional hubs.
3. For each candidate: LLM-generate bridge description explaining the conceptual overlap.
4. Expert review: accept bridge, reject, or propose new parent hub.
5. Accepted bridges → new Related links in cre_hierarchy.json.
6. Output: bridge_report.json with all candidates, evidence, review status.

## Approach
HuggingFace publication and bridge identification are independent — parallelize. Security review for model artifacts (no secrets, no PII).

Adversarial review: focus on model card completeness (does it document the ATLAS flat performance and calibration limitations honestly?) and bridge methodology soundness.

## Success criteria
- HuggingFace AIBOM score: 100/100
- Model card documents all known limitations honestly
- At least 10 validated AI/traditional bridges with evidence
- bridge_report.json with full evidence and review status
- No secrets or PII in any published artifact

Think deeply using the sequential-thinking MCP server. --ultrathink
```

---

## Prompt 5: Phase 3 — Published Human-Reviewed Crosswalk Dataset

```
TRACT Phase 3: Produce and publish the human-reviewed crosswalk dataset. PRD Section 8.

Read PRD.md Section 8. Read CLAUDE.md for code standards.

## Current state — verify before starting
1. All Phase 2 work complete: model published, bridges identified, web UI functional
2. crosswalk.db contains predictions for all 22 frameworks
3. Active learning rounds completed (acceptance rate documented)
4. Hub proposals reviewed and accepted proposals integrated

## Key context
This phase is primarily expert review, not engineering. The engineering work is building the review tooling and publication pipeline. The expert review itself (~62+ hours for 749 AI controls) happens outside Claude Code sessions.

## Results from all previous phases
1. Model confidence per framework → prioritize review: start with highest-confidence frameworks
2. Active learning acceptance rates → frameworks with low acceptance need more careful review
3. Bridge hubs from 2B → include bridge relationships in published dataset
4. Calibration reliability → show calibrated probabilities to reviewer to guide effort allocation
5. Per-hub prediction accuracy from LOFO → identify systematically difficult hubs

## Scope

### Review Tooling
1. Export all predictions from crosswalk.db grouped by framework → review spreadsheets (CSV or web interface)
2. Each row: control_id, control_text, predicted_hub_1 (calibrated_conf), predicted_hub_2 (calibrated_conf), ..., reviewer_decision, reviewer_notes
3. Review interface: accept top prediction, select different hub, flag for discussion, mark "no good hub"
4. Track: total reviewed, acceptance rate, edit rate, rejection rate, time per control
5. Second-pass consistency check for edited/rejected controls
6. Inter-reviewer agreement metrics if multiple reviewers (Cohen's kappa)
7. Freeze as crosswalk_reviewed_v1.0.jsonl

### Publication Pipeline
Published to HuggingFace Datasets AND Zenodo (for DOI):
- crosswalk_v1.0.jsonl — every control + hub assignment + confidence + review status
- framework_metadata.json — 22 framework descriptions, versions, sources
- cre_hierarchy_v1.0.json — hub ontology at publication time
- hub_descriptions_v1.0.json — validated descriptions
- review_metrics.json — acceptance rates, agreement, reviewer effort
- README.md — dataset card (HuggingFace Datasets format)
- LICENSE — Apache 2.0 or CC-BY-4.0

Contribute back to OpenCRE: submit validated assignments for 6 unmapped AI frameworks as proposed LinkedTo links.

### Atomic review state
Review progress must survive crashes. Write review state atomically after each decision. Use the same atomic write pattern as all other TRACT I/O (write to temp, os.replace).

## Success criteria
- 100% of predicted assignments reviewed by expert
- Published dataset on HuggingFace Datasets with DOI from Zenodo
- Review metrics documented (acceptance rate, edit rate, inter-reviewer agreement if applicable)
- OpenCRE contribution submitted for unmapped frameworks

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
3. Common query patterns from web UI usage → optimize API for actual patterns
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

### Process
- Adversarial review catches real methodology errors (R2's 4-fold vs 5-fold comparison).
- Cross-examination between critics is essential — without it, false findings persist.
- Small test fixtures (9 CREs, 3 frameworks) catch real bugs.
- Two-stage review (spec compliance + code quality) is an effective quality gate.
