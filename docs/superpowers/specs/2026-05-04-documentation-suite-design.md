# TRACT Documentation Suite — Design Spec

**Date:** 2026-05-04
**Scope:** Documentation only — no code changes. All work confined to `.md` files.
**Conflict safety:** Parallel session works on feature code (`tract/`, `scripts/`, `tests/`). This session touches only: `README.md` (new), `docs/*.md` (new), `CONTRIBUTING.md`, `examples/README.md`. Zero overlap on current file inventory. **Fragility note:** If the parallel session creates any `docs/*.md` file, modifies `CONTRIBUTING.md`, or adds/removes CLI subcommands, coordinate before committing.

## Goals

Create a comprehensive, layered documentation suite that takes readers from "what is this?" to expert usage. Three audiences served:

1. **Security/GRC practitioners** — understand the crosswalk, trust the assignments, add their frameworks
2. **ML/NLP researchers** — understand the assignment paradigm, model architecture, evaluation methodology
3. **Open-source developers** — contribute parsers, extend the CLI, integrate TRACT

Cross-domain bridge: assume readers are strong in one domain but may be unfamiliar with the other. Brief contextual explanations for cross-domain concepts; no fundamentals tutorials.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Documentation structure | Hub and Spokes | README as front door, deep-dive docs for each audience. Proportional to project scope. |
| Notebook relationship | Summarize and link | Key results inline (tables, metrics), notebook is authoritative deep-dive. GitHub visitors see evidence without opening .ipynb. |
| Audience level | Cross-domain bridging | Security person doesn't know contrastive fine-tuning; ML person doesn't know CRE hubs. Brief "What is X?" callouts. |
| Diagrams | Mermaid | GitHub renders natively. Version-controlled, diffable. Best expressiveness for TRACT's concepts. |
| Scope | All 7 documents | Complete suite in one pass. |

## Document Inventory

### 1. README.md (NEW) — ~250-300 lines

**Purpose:** The front door. 30-second understanding, 2-minute first run, 5-minute navigation to depth.

**Sections:**

1. **Title + badges** — Project name, one-line description. Badges: CC0 license, Python 3.11+, HuggingFace model link, HuggingFace dataset link. *(Note: test count is volatile — omit from badge or use CI-generated dynamic badge.)*

2. **What is TRACT?** — Two paragraphs: the problem (frameworks are siloed, manual crosswalking doesn't scale) and the solution (trained bi-encoder assigns controls to OpenCRE hubs, creating transitive crosswalks). Brief "What is OpenCRE?" callout.

3. **Key Results** — Compact table:
   - hit@1 = 0.537 (trained model, LOFO evaluation, +0.139 over zero-shot baseline)
   - 5,238 control-to-hub assignments across 31 frameworks (4,390 ground truth, 528 expert-reviewed, 320 model predictions)
   - 46 AI↔traditional bridge links proposed and accepted (of 63 candidates)
   - Model and dataset published to HuggingFace
   
   Link to notebook for full experiment narrative. *(Metrics sourced from `results/phase1b/phase1b_textaware/corrected_metrics.json` — verify at implementation time.)*

4. **How It Works — Mermaid diagram** — Assignment pipeline flowchart:
   ```
   Raw Framework → Parser/Prepare → Standardized JSON → Encoder → CRE Hub Assignment → Crosswalk DB → Export
   ```
   Annotated: "Assignment paradigm: g(control) → CRE hub. Never pairwise f(A,B) → relationship."

5. **Quick Start** — Two paths, clearly labeled:
   
   **Explore without model artifacts** (works on fresh clone):
   ```bash
   git clone https://github.com/rocklambros/TRACT.git
   cd TRACT
   pip install -e ".[dev]"
   tract prepare --file examples/sample_framework.csv --framework-id demo --name "Demo Framework"
   tract validate --file demo_prepared.json
   ```
   
   **Full assignment workflow** (requires deployed model — see docs/architecture.md):
   ```bash
   pip install -e ".[phase0]"
   tract tutorial          # checks prerequisites, shows guided walkthrough
   tract assign "Implement input validation for AI model training data"
   ```
   
   *Note: `tract assign` and `tract tutorial` require model artifacts from the Phase 1C pipeline. `tract prepare` and `tract validate` work immediately after install.*

6. **Framework Coverage** — Table of all 31 frameworks grouped by:
   - AI-specific (12): CSA AICM, MITRE ATLAS, OWASP AI Exchange, OWASP LLM Top 10, EU AI Act, etc.
   - Traditional (19): NIST 800-53, ASVS, CWE, CAPEC, ISO 27001, etc.
   
   Each with control count. Total: 2,802 controls.

7. **CLI Overview** — All 18 subcommands grouped by workflow stage:
   - **Explore:** `hierarchy`, `compare`, `tutorial`
   - **Prepare:** `prepare`, `validate`
   - **Assign:** `assign`, `ingest`, `accept`
   - **Review:** `review-export`, `review-validate`, `review-import`, `review-proposals`
   - **Analyze:** `bridge`, `propose-hubs`, `import-ground-truth`
   - **Export:** `export`
   - **Publish:** `publish-hf`, `publish-dataset`
   
   One-line description each. Link to `docs/cli-reference.md`. *(Grouping matches cli-reference.md exactly — all 18 commands accounted for.)*

8. **Project Architecture — Mermaid diagram** — Directory/module structure and data flow:
   ```
   data/raw/ → parsers/ → data/processed/ → tract/ → results/ → build/
   ```
   Brief text on key directories.

9. **Where to Go Next** — Signpost section:
   - Adding a new framework? → `docs/framework-guide.md`
   - How does the model work? → `docs/architecture.md`
   - CLI reference → `docs/cli-reference.md`
   - Glossary → `docs/glossary.md`
   - Contributing → `CONTRIBUTING.md`

10. **Published Artifacts** — Links:
    - Model: `huggingface.co/rockCO78/tract-cre-assignment`
    - Dataset: `huggingface.co/datasets/rockCO78/tract-crosswalk-dataset`
    - Experimental narrative: `tract_experimental_narrative.ipynb`

11. **License** — CC0 1.0 Universal, one line.

---

### 2. docs/architecture.md (NEW) — ~500-600 lines

**Purpose:** Deep technical explanation for ML researchers and curious security practitioners. Reviewable depth — a reader could critique the approach without running the notebook.

**Sections:**

1. **The Assignment Paradigm** — Core principle: `g(control_text) → CRE_position`. Why not pairwise: O(n^2) doesn't scale, and OpenCRE already provides the universal taxonomy. Mermaid diagram: control text → encoder → cosine similarity → ranked CRE hubs.

2. **The OpenCRE Hierarchy** — What CRE hubs are. Group → hub → linked standards. AutomaticallyLinkedTo = expert-quality (deterministic CAPEC→CWE→CRE chain). "What is OpenCRE?" box for ML readers.

3. **Data Landscape** — Three tiers:
   - Tier 1: 19 frameworks already linked to OpenCRE (training signal, 4,406 links)
   - Tier 2: 12 AI frameworks with primary-source parsers (inference targets)
   - Tier 3: New frameworks via `tract prepare`
   
   Mermaid diagram showing tier flow into training/inference.

4. **Phase 0: Zero-Shot Baselines** — Gate criteria and results table. All methods compared. Key findings: DeBERTa-v3-NLI total failure (hit@1=0.000), hierarchy paths help (+7.6%), descriptions hurt zero-shot. Link to `docs/phase0-results.md` and notebook Section 3. **Practitioner bridge:** "What this means — off-the-shelf similarity tools cannot reliably map controls to CRE hubs. A dedicated model is required, but it's feasible."

5. **Model Architecture & Training** — BGE-large-v1.5 + LoRA (rank 16, alpha 32, dropout 0.1). Contrastive fine-tuning with hard negatives (3 per positive, temperature=2.0). Hub representations = hierarchy path + LLM description. Hyperparameter table from `tract/config.py`. Training signal: known OpenCRE links as (control, hub) positive pairs.

6. **LOFO Evaluation** — Leave-One-Framework-Out. Mermaid `sequenceDiagram` with `loop` syntax (not flowchart — flowcharts lack native loop constructs). Hub firewall: rebuild hub representations WITHOUT the evaluation framework's linked sections. Metrics: hit@1, hit@5, MRR, NDCG@10 with 95% bootstrap CIs (10,000 resamples). **Practitioner bridge:** "Why you should trust these numbers — the model never sees the framework it's being tested on, and the CRE hub representations are rebuilt without that framework's linked sections. This is the strictest evaluation possible."

7. **Key Results** — hit@1=0.537 (micro-average, n=147), delta=+0.139 over zero-shot firewalled BGE baseline (0.399). Per-framework breakdown. Gate 1: clean pass, all folds non-negative. *(Source: `results/phase1b/phase1b_textaware/corrected_metrics.json` — verify at implementation time.)*

8. **Calibration & Confidence** — Temperature scaling (raw cosine → calibrated probability). ECE evaluation. Conformal prediction sets (90% coverage). OOD detection for novel controls.

9. **Bridge Analysis** — AI↔traditional connections through shared CRE hubs. 46/63 accepted. What bridges mean for practitioners. Example: MITRE ATLAS technique → same CRE hub as NIST 800-53 control.

10. **Limitations & Future Work** — 5 uncovered frameworks. Controls with no existing CRE hub (hub proposals address this). Granularity disagreements. Single-label assignment (some controls span multiple hubs).

---

### 3. docs/framework-guide.md (NEW) — ~300-350 lines

**Purpose:** End-to-end walkthrough from "I have a framework document" to "my controls are in the crosswalk." Two paths: LLM-assisted (most frameworks) and custom parser (structured sources).

**Sections:**

1. **Overview** — What "adding a framework" means. Your controls get standardized, assigned to CRE hubs, linked to all other frameworks through shared hubs. Mermaid lifecycle diagram: `Source Doc → Prepare/Parse → Validate → Ingest → Review → Accept → Export`.

2. **Path 1: LLM-Assisted Preparation** — For PDF, Markdown, CSV sources.
   - **Prerequisite note:** `tract prepare` requires an `ANTHROPIC_API_KEY` environment variable (it calls Claude Sonnet for control extraction). State this upfront.
   - `tract prepare` command walkthrough with options
   - What happens under the hood (Claude Sonnet chunking + extraction)
   - Output JSON schema explanation
   - Common issues (PDF table extraction, missing control IDs)
   - Reference to `examples/` sample files

3. **Path 2: Writing a Custom Parser** — For JSON, YAML, structured HTML.
   - `BaseParser` class overview (validation, count checking, sanitization)
   - Parser anatomy: `__init__`, `parse()`, output conformance
   - Annotated walkthrough of `parse_cosai.py` (simplest parser, 55 controls)
   - Testing pattern: fixture file + schema assertion

4. **Validation** — `tract validate` walkthrough. Schema conformance, control ID format, description lengths, duplicate detection, adversarial rules. Warnings vs. errors.

5. **Ingestion** — `tract ingest` walkthrough. Embedding, hub assignment, confidence calibration, review file output. Review file format explained.

6. **Review & Accept** — Human-in-the-loop. Reading the review file (control, proposed hub, confidence, alternatives). `tract accept` and what it means for the crosswalk.

7. **Verification** — Post-ingestion checks: `tract export`, `tract compare`, `tract hierarchy`.

8. **Tips & Gotchas** — Framework ID conventions, description quality, 2000-char cap, sanitization behavior.

---

### 4. docs/cli-reference.md (NEW) — ~400-500 lines

**Purpose:** Complete reference for all 18 CLI subcommands with examples and workflow context.

**Structure:**

1. **Installation & Setup** — Prerequisites, install command, verify with `tract --help`.

2. **Command Flow — Mermaid diagram** — Shows recommended order and which commands feed into which:
   ```
   prepare → validate → ingest → [review-export → review-validate → review-import] → accept → export
   assign (standalone)
   compare (any time)
   hierarchy (any time)
   bridge (after crosswalk populated)
   publish-hf / publish-dataset (release)
   ```

3. **Command Reference** — Each of the 18 subcommands:
   - **Synopsis:** `tract <command> [options]`
   - **Description:** What it does, when to use it
   - **Options table:** Flag, type, default, description
   - **Examples:** 2-3 practical examples with expected output snippets
   - **See also:** Related commands

   Grouped by workflow stage:
   - **Explore:** `tutorial`, `hierarchy`, `compare`
   - **Prepare:** `prepare`, `validate`
   - **Assign:** `assign`, `ingest`, `accept`
   - **Review:** `review-export`, `review-validate`, `review-import`, `review-proposals`
   - **Analyze:** `bridge`, `propose-hubs`, `import-ground-truth`
   - **Export:** `export`
   - **Publish:** `publish-hf`, `publish-dataset`

4. **Common Workflows** — 3-4 recipes:
   - "Add a framework from PDF to crosswalk" (prepare → validate → ingest → accept)
   - "Compare two frameworks" (compare)
   - "Export assignments for integration" (export with format options)
   - "Full publication pipeline" (publish-hf, publish-dataset) — **Note:** publish commands require `pass` password manager with `huggingface/token` entry, or `HF_TOKEN` environment variable. Use placeholder values in examples (`<repo-id>`, `<hours>`), never concrete infrastructure details.

---

### 5. docs/glossary.md (NEW) — ~150-200 lines

**Purpose:** Cross-domain term reference. Security people look up ML terms; ML people look up security terms; everyone looks up TRACT-specific terms.

**Structure:** Alphabetical entries, each with:
- **Term** (bold)
- One-line definition
- Context sentence showing how TRACT uses it

**Term categories (not visible grouping, just for coverage):**

- **Security terms:** control, crosswalk, CRE hub, framework, OpenCRE, security standard
- **ML terms:** bi-encoder, bootstrap CI, contrastive fine-tuning, cosine similarity, embedding, LoRA, temperature scaling
- **Evaluation terms:** ECE, hit@k, LOFO, MRR, NDCG, OOD detection, conformal prediction
- **TRACT-specific terms:** assignment paradigm, bridge, hub description, hub firewall, hub proposal, hub hierarchy path, mapping unit, tier (1/2/3)

---

### 6. CONTRIBUTING.md (UPDATE) — ~200-250 lines

**Purpose:** Enrich the existing file with architecture orientation and practical contribution guidance.

**Changes from current version:**

- **Add "Architecture Orientation"** section — Where does code go? `tract/` for core library, `parsers/` for framework parsers, `scripts/` for phase scripts, `tests/` for tests. How modules relate. Which files a typical contribution touches.

- **Add "Three Contribution Tracks"** section:
  - **New parser:** Write a parser + test + fixture. Easiest entry point.
  - **Core library:** Extend `tract/` modules. Higher bar — needs architecture understanding.
  - **Evaluation/analysis:** Scripts that analyze results. Medium complexity.

- **Add "Your First Contribution"** section — Step-by-step: fork, create a parser test fixture, write a 30-line parser, run tests, submit PR. Concrete enough that someone could do it in an afternoon.

- **Update stale details:**
  - Test count: verify at implementation time (currently 871, CLAUDE.md says 831 — stale)
  - mypy coverage: must match CI exactly: `mypy tract/ scripts/phase1a/ --strict` (do NOT broaden beyond what CI enforces)
  - Add ruff linting: `ruff check tract/ scripts/phase1a/ parsers/` (matches CI configuration)

- **Keep existing sections:** PR guidelines, code review, reporting issues, license. Refresh for accuracy.

---

### 7. examples/README.md (UPDATE) — ~100-150 lines

**Purpose:** Tutorial entry point with worked examples.

**Changes from current version:**

- **Expand from 30 lines to ~120 lines**

- **Three worked examples:**
  1. **CSV path:** `tract prepare` on `sample_framework.csv` → `tract validate` → examine output JSON
  2. **Markdown path:** Same flow with `sample_framework.md`
  3. **What happens next:** Shows `tract ingest` → review file → `tract accept` → `tract export` with expected output snippets (conceptual — actual ingest requires a deployed model)

- **Link to framework-guide.md** for the full story

- **Expected output snippets** for each step so readers can verify they got it right

---

## File Conflict Analysis

| File | This Session | Parallel Session | Risk |
|------|-------------|-----------------|------|
| `README.md` | CREATE | Won't touch | None |
| `docs/architecture.md` | CREATE | Won't touch | None |
| `docs/framework-guide.md` | CREATE | Won't touch | None |
| `docs/cli-reference.md` | CREATE | Won't touch | None |
| `docs/glossary.md` | CREATE | Won't touch | None |
| `CONTRIBUTING.md` | UPDATE | Won't touch | None |
| `examples/README.md` | UPDATE | Won't touch | None |
| `CLAUDE.md` | DO NOT TOUCH | May update | Avoided |
| `tract_experimental_narrative.ipynb` | DO NOT TOUCH | May touch | Avoided |
| `tract/**/*.py` | DO NOT TOUCH | Will touch | Avoided |
| `scripts/**/*.py` | DO NOT TOUCH | Will touch | Avoided |
| `tests/**/*.py` | DO NOT TOUCH | Will touch | Avoided |
| `pyproject.toml` | DO NOT TOUCH | May touch | Avoided |

## Content Sources

All documentation content will be derived from:
- Current codebase state (CLI help, config.py constants, module structure)
- Actual results files (e.g., `results/phase1b/phase1b_textaware/corrected_metrics.json`) — **not** from CLAUDE.md or PRD.md which may contain stale transcriptions
- `docs/phase0-results.md` (metrics tables — cross-check against `results/phase0/`)
- `CLAUDE.md` (project status, commands, standards — **subject to sanitization rule below**)
- `PRD.md` (framework inventory, phase descriptions)
- Git history (commit messages for timeline)
- HuggingFace published artifacts (model card, dataset card content)

No content will be fabricated. All metrics cited must be verified against the actual results files at implementation time.

## Content Sanitization Rule

When deriving content from CLAUDE.md or PRD.md, **never reproduce** the following in public documentation:
- References to `pass` password manager or specific credential store paths (`pass huggingface/token`, etc.)
- Home directory paths (`/home/rock/`, `/Users/rock/`)
- Infrastructure provider names (RunPod, specific GPU configurations)
- Credential patterns or API key environment variable names in prose (OK in "prerequisites" sections where needed for setup)
- Email addresses (link to SECURITY.md instead of embedding the address)

Use generic forms: "credential manager", "training infrastructure", "GPU cluster."

## Volatile Values

The following values are point-in-time snapshots. The implementer must verify each at implementation time and should avoid hardcoding them in badges or prominent locations where staleness is most visible:

| Value | Current | Source of truth | Staleness trigger |
|-------|---------|-----------------|-------------------|
| Test count | 871 | `python -m pytest tests/ --co -q` | Any test added/removed |
| Framework count | 31 | `data/processed/all_controls.json` | Framework added via ingest |
| Control count | 2,802 | `data/processed/all_controls.json` | Framework added via ingest |
| hit@1 | 0.537 | `results/phase1b/phase1b_textaware/corrected_metrics.json` | Model retrained |
| Assignment count | 5,238 | `build/dataset/crosswalk_v1.0.jsonl` | Dataset republished |
| Bridge count | 46/63 | `results/bridge/bridge_report.json` | Bridge analysis rerun |
| CLI subcommand count | 18 | `tract --help` | Subcommand added/removed |

When citing these in documentation, include the value inline (for readability) but do not make them the sole content of a badge or heading that would go stale silently.

## GitHub URL

The canonical GitHub URL is `https://github.com/rocklambros/TRACT` (confirmed from git remote). **Note:** Several code files (`tract/cli.py:1208`, `tract/publish/model_card.py`) reference `rockcyber/TRACT` — this is a known bug in the codebase. All documentation must use `rocklambros/TRACT` consistently.

## Implementation Order

1. `docs/glossary.md` — Foundation. Other docs link to it.
2. `docs/architecture.md` — Deepest content, links to glossary.
3. `docs/framework-guide.md` — Practical guide, links to glossary and CLI reference.
4. `docs/cli-reference.md` — Reference doc, links to framework guide for workflows.
5. `examples/README.md` — Tutorial entry, links to framework guide.
6. `CONTRIBUTING.md` — Update with architecture context.
7. `README.md` — Front door, written last because it links to all other docs.

README is last because it contains forward references to every other document. Writing it last ensures all link targets exist and content can be summarized accurately.

## Adversarial Review Log

This spec was subjected to a 4-round adversarial review (2026-05-04) attacking security architecture, methodology, and implementation/operations layers. Key corrections applied:
- hit@1 corrected from 0.531 → 0.537 (verified against results files; 0.531 was a transcription error in PRD.md)
- Delta corrected from +0.132 → +0.139
- Quick Start rewritten to work on fresh clone (prepare/validate path added)
- `propose-hubs` added to README CLI overview (was missing — 17/18 listed)
- Implementation order reversed to put README last (eliminated 7 forward references)
- Content sanitization rule added (CLAUDE.md contains operational secrets)
- Volatile values table added (no staleness mitigation existed)
- mypy paths aligned to CI (was incorrectly broadened)
- GitHub URL canonicalized to rocklambros/TRACT
- architecture.md practitioner bridge sections added (was 5:2 ML-skewed)
- LOFO diagram changed from flowchart to sequenceDiagram (native loop syntax)
- Dataset composition breakdown added (was misleadingly presented as all model output)
- API/credential prerequisites added to framework guide and CLI reference
