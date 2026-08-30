# TRACT — Translating Requirements Across CRE Trees

PRD.md is the master spec. All section numbers reference it. Read it before starting any phase.

## Role

You are a senior data scientist and ML engineer building production-grade research infrastructure. Write code that a skeptical reviewer would trust on first read: typed, validated, tested, deterministic. Prefer explicit over clever. Every function has a clear contract — what it accepts, what it returns, what it raises.

## Code Standards

**Type everything.** All function signatures fully typed. Use `TypedDict` or `@dataclass` for structured data — never bare dicts for domain objects. Return types always declared.

**Validate at boundaries.** Every parser validates its input schema before processing and its output schema after. Use jsonschema or pydantic for the standardized control schema (PRD Section 4.8). Interior functions trust their callers — no redundant checks.

**Fail loud, fail early.** `raise ValueError` with a specific message, never `return None` to signal failure. Never silently skip malformed records — log the exact record and raise. No bare `except:` — catch specific exceptions only.

**Deterministic reproducibility.** Set random seeds explicitly in every script that touches randomness. Parsers must produce byte-identical output on re-run. Sort keys in JSON output. Pin library versions in requirements.txt.

**No magic numbers.** Constants live in a single `tract/config.py` or at module top-level with ALL_CAPS names. Thresholds, paths, counts — all named and documented.

**Logging, not print.** Use `logging` module everywhere. DEBUG for internal state, INFO for pipeline progress, WARNING for recoverable issues, ERROR for failures. Never `print()` in library code.

**Defensive I/O.** All file writes use atomic write patterns (write to temp, then rename). All network calls use retry with exponential backoff and timeout. All file reads specify encoding='utf-8' explicitly.

**Test-driven.** Write the test first for any non-trivial function. Tests use fixtures, not hardcoded paths. Tests assert on structure and content, not just "no exception." Each parser has a test with a small representative fixture that validates output schema conformance.

## Security

- Never `eval()`, `exec()`, or `subprocess.shell=True`. No pickle for untrusted data — use safetensors for model weights.
- Sanitize all text fields: strip null bytes, normalize unicode (NFC), enforce max length before storage.
- No secrets in code or config files. Credentials via `pass` password manager only.
- API keys passed as environment variables at runtime, never hardcoded or logged.
- All external data (frameworks, OpenCRE API responses) treated as untrusted input — validate structure before processing.

## ML Engineering

- **Experiment tracking.** Every training run logs: data hash, hyperparameters, git SHA, seed, full metric suite. Use WandB.
- **Data versioning.** Hash raw data at fetch time. Store hash in processed output metadata. If hash changes, force re-processing.
- **Checkpoint discipline.** Save model + optimizer + scheduler + epoch + metrics. Never save just weights.
- **Evaluation honesty.** Hub firewall is non-negotiable — no information leakage from held-out framework into hub representations. Assert this programmatically.
- **Calibration.** Raw model outputs are cosine similarities, not probabilities. Always calibrate (temperature/Platt scaling) before reporting confidence scores.
- **Model architecture.** BGE-large-v1.5 bi-encoder with contrastive fine-tuning. Phase 0 proved: DeBERTa-v3-NLI fails completely (hit@1=0.000); hierarchy paths help (+7.6%); descriptions hurt zero-shot. Do not use classification heads, NLI models, or RoBERTa — these are old-project patterns.

## Standing Rules

**All inference and training runs on RunPod. Never locally.** This Mac is the
owner's daily driver and its resources are not available. That covers model
loading, embedding, fine-tuning, evaluation, and calibration sweeps. Writing
code, running unit tests that do not load a model, linting, and type checking
are fine locally. Anything that would allocate a model goes to a pod. CI runs
on GitHub runners, which is not this machine and is allowed.

**Consider all available prose, always.** Prefer a control's full text over its
title everywhere text is selected, for training anchors and eval items alike.
Falling back to `section_name` is a last resort, not a default, and any fallback
is logged and counted. `data/processed/all_controls.json` is the source of
prose; join to it before reaching for the title.

**Stop word removal.** Filter common low-information words from control and hub
text during processing so the model sees the distinctive terms. The list is
generated from this corpus rather than borrowed, so it reflects actual security
boilerplate ("shall", "system", "ensure") and not just English function words.
Implemented as a toggle and measured as an ablation arm, because removing
function words moves input off the distribution a contextual encoder was
pretrained on and that trade has to be shown, not assumed.

## Core Constraint

Assignment paradigm only: `g(control_text) -> CRE_position`. NEVER pairwise `f(A,B) -> relationship`. If you find yourself comparing two controls directly, you're doing it wrong — map each to CRE hubs independently.

## Things That Break If You Forget

- **CSA CCM ≠ CSA AICM.** Cloud Controls Matrix (traditional cloud, 29 CRE links) is a completely different framework from AI Controls Matrix (AI security, 243 controls, zero CRE links). Never conflate them.
- **Hub firewall.** When evaluating framework X, rebuild hub representations WITHOUT X's linked sections. No exceptions — this is what makes LOFO honest.
- **LOFO only.** Leave-one-framework-out cross-validation. Never hold out random controls. Never use a frozen test set.
- **No pairwise metrics.** hit@1, hit@5, MRR, NDCG@10 on hub assignment. Bootstrap CIs (10,000 resamples) for all comparisons. No F1 on pairwise tiers.
- **Auto-links are expert-quality.** AutomaticallyLinkedTo in OpenCRE = deterministic CAPEC→CWE→CRE transitive chain, NOT ML output. Treat as equivalent to human LinkedTo (penalty=0).
- **data/raw/ is immutable.** Never modify files after initial fetch. Parsers read raw/, write processed/.
- **Fresh OpenCRE fetch.** Always from `opencre.org/rest/v1/all_cres` (1-indexed, per_page=50, ~261 pages). Never copy from old project.

## Operational

- **Old project:** `~/github_projects/ai-security-framework-crosswalk/` — data source only, no runtime dependencies. Written relative to home on purpose: the absolute path here used to be the Jetson's, under a different account, and resolved on no other machine. Use `~` for any path under a home directory.
- **Credentials:** `pass` password manager (not .env). `pass huggingface/token`, `pass runpod/api-key`, `pass wandb/api-key`
- **OpenCRE API:** Paginated JSON. Retry with exponential backoff. Endpoint changed from /rest/v1/all to /rest/v1/all_cres (the old one returns HTML now).

## Cross-Session Memory (claude-mem)

This project uses the claude-mem plugin for persistent cross-session memory. Follow these rules strictly.

### Session Start

At the beginning of every session, search for recent project context before doing any work:
```
search(query="TRACT", project="TRACT", limit=10, orderBy="date_desc")
```
If the user references past work ("we already did X", "last time", "where were we"), use `mem-search` to find it. Never guess — search first, then `get_observations` for the relevant IDs.

### Code Exploration

For Python files over ~100 lines, prefer smart-explore over Read:
- `smart_search(query="...", path="./parsers")` to discover symbols across the codebase
- `smart_outline(file_path="...")` for file structure before reading
- `smart_unfold(file_path="...", symbol_name="...")` for specific functions

Use Read directly for: JSON data files, markdown, config files, and small Python files under 100 lines.

### What Gets Tracked Automatically

claude-mem records observations as you work. These are valuable for continuity:
- **Discoveries** (🔵): Data anomalies, API behavior, framework quirks, count mismatches
- **Decisions** (⚖️): Ambiguous data interpretation, parser design choices, schema decisions
- **Bug fixes** (🔴): Parsing failures, validation errors, data corruption
- **Features** (🟣): New parsers, pipeline stages, tooling additions
- **Changes** (✅): Commits, config updates, dependency additions

### When to Use Each Tool

| Situation | Tool |
|-----------|------|
| "What did we do last session?" | `mem-search` → search → get_observations |
| "Where is X defined in the codebase?" | `smart-explore` → smart_search |
| "Navigate a large parser file" | `smart-explore` → smart_outline → smart_unfold |
| Multi-step implementation task | `make-plan` to plan, `do` to execute with subagents |
| "Show me the project timeline" | `timeline-report` |
| "What do we know about framework X?" | `knowledge-agent` to build a queryable corpus |

### What NOT to Use Memory For

- Current session task tracking — use TodoWrite/tasks instead
- Code structure that can be derived from the codebase — use smart-explore
- Git history — use `git log` / `git blame`
- Things already documented in this file or PRD.md

## Project Status

- **Data Preparation:** COMPLETE
- **Phase 0 (Zero-Shot Baselines):** COMPLETE — Gates A+B passed
- **Phase 1A–1D:** COMPLETE — model trained, 11 CLI subcommands, hub proposals. **The Gate 1 headline is WITHDRAWN (2026-08-15)** — the hit@1 figure that stood here passed on the point estimate only, mixed two runs, and did not generalize. Do not quote a Phase 1 accuracy number from memory; PRD.md §6.4 carries the withdrawal.
- **Phase 2B (Bridge + HF Publication):** COMPLETE — 46/63 bridges accepted, model published to huggingface.co/rockCO78/tract-cre-assignment
- **Phase 3 (Crosswalk Dataset):** COMPLETE — 5,238 assignments across 31 frameworks, expert-reviewed, published to huggingface.co/datasets/rockCO78/tract-crosswalk-dataset
- **Phase 5A (Export Pipeline):** COMPLETE — 411 assignments imported into local OpenCRE fork
- **Phase 5B (Canonical Export):** COMPLETE — per-framework JSON snapshots + changesets for OpenCRE RFC
- **Framework Prep Pipeline:** COMPLETE — `tract prepare` + `tract validate` + ingest integration
- **Lazy Model Auto-Download:** COMPLETE — `tract assign` downloads the pinned model from HuggingFace on first use (sha256-verified, sentinel-gated), tolerates the published flat layout, and adds `tract --version`. Forces the PyTorch backend (`USE_TF=0` at import) to avoid a TensorFlow import deadlock in `sentence-transformers`. `assign --file` preserves input order and carries an `input_index`. Distinct exit codes: 2 user error, 3 offline, 4 integrity, 5 missing runtime.
- **2,944 tests** (`pytest tests/ -q --collect-only`, 2026-08-29 — this line said 920, then 2,722, then 2,832, each long after the suite had moved, so re-derive it rather than trust it), 20 CLI subcommands. 28 fail locally on `sentence-transformers==3.4.1`, which is outside the pinned set (3.2.0 / 5.7.0); `tests/test_st_compat.py` is the suite saying so. They pass in CI.
- **Campaign 2 (LOFO re-run, 2026-08-28):** COMPLETE. Three validation arms
  (n=1,265) then one held-out AI test round (n=147). Test-round micro hit@1
  delta **+0.1361 [+0.0476, +0.2245]** over paired zero-shot, absolute 0.5918.
  This clears the 0.10 gate on the **point estimate only** — `ci_low_pass` and
  `familywise_pass` both fail, and P(true delta ≤ 0.10) = 0.203. The clause
  designating the point estimate as the verdict was written after arm results
  existed, so it is not blind pre-registration. **No arm cleared Gate 1 on
  validation** and the primary arm A1 is significantly negative. The pre-declared
  agentic smoke test FAILED at 1–2 of 6. Read `docs/campaign2-results.md` before
  quoting anything from this campaign; it lists which of the campaign's own
  commit messages were later superseded. A3 is **not** the shipped model.
- **Campaign 2 amended (2026-08-29), two findings, both verified:**
  - **25% of the test gold was rewritten by TRACT's own link audit** and was
    disclosed nowhere. On the 110 audit-untouched items the delta is **+0.1000
    [0.000, 0.200], P(δ ≤ 0.10) = 0.531** — report it as a co-primary beside the
    pooled figure, never the pooled figure alone. `docs/campaign2-results.md`
    §13; reproduce with `scripts/analysis/audit_stratified_delta.py`.
  - **The domain-shortcut hypothesis is refuted.** Handing the zero-shot encoder
    the whole 78-hub AI region free — 444 of 522 candidates deleted — moves one
    item in 147 (+0.0068). The gain is not candidate-set narrowing. §14.
- **Campaign 3:** pre-registered in `results/phase1b/CAMPAIGN3.md`, **plus
  Amendment 1 (2026-08-30) which corrects three things in it** — read the
  amendment, not just the body. Binding numeric thresholds; **no synthetic data
  in training or evaluation**; the n=940 power table is **wrong** (the permitted
  frameworks hold 500 controls, so the ceiling is 721 and real power is 43–54%,
  not 63%).
  - **Owner decisions taken 2026-08-30:** the link audit was NOT model-informed
    (Tier 2, published artifacts are clean); fix the anchor truncation and
    **rebaseline**, retiring +0.1361 as a forward target; and fund curation.
  - **Anchor-budget rebaseline in progress** as arm `C3TEST`
    (`max_seq_length` 512 → 1024, batch stays 32). It is a **second run on the
    held-out test split** — sanctioned, costed, and recorded in Amendment 1 §1.5.
  - **Curation package:** `claudedocs/curation-package.md` (untracked) holds the
    recruiting persona and the annotator handbook. Generate the blind packet
    with `python -m scripts.build_curation_packet`. **Never send
    `results/review/hub_reference.md`** — 400 of its hub descriptions are
    LLM-written from the gold links and would make the round Tier 3.
- **Provisioning constraint:** while the licensed overlay is staged, pods are
  restricted to SECURE tier at create time (`_require_secure_cloud`), because
  `_rsync_to` ships `data/processed/licensed` to whichever host answers.
  `TRACT_RUNPOD_ALLOW_COMMUNITY=1` overrides and logs that it did.
- **`results/review/review_export.json` is Tier 3 and quarantined.** 898
  model-proposed, human-ratified items; agrees with independent OpenCRE gold on
  only 47 of 63 overlapping items. Never a gate denominator. See
  `results/review/PROVENANCE.md`.
- **No web UI.** TRACT is CLI + API only. No Dash dashboard.

## Commands

```bash
# Assign a control to CRE hubs — downloads the pinned model (~1.3 GB) on first use
tract assign "Implement input validation for AI model training data"
tract --version                                    # Package version + pinned model revision

# Optional: pre-fetch model + crosswalk.db (CI / airgapped; assign auto-downloads otherwise)
tract download                                     # Everything (model ~1.3 GB + crosswalk.db)
tract download --model-only                        # Model artifacts only

# Run all parsers
for f in parsers/parse_*.py; do python "$f"; done

# Validate processed output
python parsers/validate_all.py

# Run tests
python -m pytest tests/ -q

# Type check
mypy tract/ parsers/ scripts/phase1a/ scripts/phase1b/ scripts/phase0/runpod_provision.py --strict

# Bridge analysis (Phase 2B)
tract bridge --top-k 3                           # Generate bridge candidates
tract bridge --commit --candidates <path>         # Commit reviewed bridges to hierarchy

# HuggingFace publication (Phase 2B)
tract publish-hf --repo-id <repo> --dry-run       # Build staging dir without upload
tract publish-hf --repo-id <repo> --gpu-hours N   # Full publish

# Phase 3 — Review & Dataset Publication
tract import-ground-truth                          # Import OpenCRE ground truth into crosswalk.db
tract review-export                                # Export predictions for expert review
tract review-validate <path>                       # Validate reviewed JSON
tract review-import <path>                         # Apply review decisions to crosswalk.db
tract publish-dataset --repo-id <repo>             # Bundle and upload dataset to HuggingFace

# Phase 5B — Canonical Export (OpenCRE RFC)
tract export-canonical --dry-run                  # Preview what would be exported
tract export-canonical --framework csa_aicm       # Export single framework
tract export-canonical --with-embeddings           # Include .npz embedding files
```
