# TRACT — Transitive Reconciliation and Assignment of CRE Taxonomies

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

- **Old project:** `/home/rock/github_projects/ai-security-framework-crosswalk/` — data source only, no runtime dependencies
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
- **Phase 1A–1D:** COMPLETE — model trained (hit@1=0.531), 11 CLI subcommands, hub proposals
- **Phase 2B (Bridge + HF Publication):** COMPLETE — 46/63 bridges accepted, model published to huggingface.co/rockCO78/tract-cre-assignment
- **Phase 3 (Crosswalk Dataset):** COMPLETE — 5,238 assignments across 31 frameworks, expert-reviewed, published to huggingface.co/datasets/rockCO78/tract-crosswalk-dataset
- **Phase 5A (Export Pipeline):** COMPLETE — 411 assignments imported into local OpenCRE fork
- **Phase 5B (Canonical Export):** COMPLETE — per-framework JSON snapshots + changesets for OpenCRE RFC
- **Framework Prep Pipeline:** COMPLETE — `tract prepare` + `tract validate` + ingest integration
- **866 tests passing**, 19 CLI subcommands
- **No web UI.** TRACT is CLI + API only. No Dash dashboard.

## Commands

```bash
# Run all parsers
for f in parsers/parse_*.py; do python "$f"; done

# Validate processed output
python parsers/validate_all.py

# Run tests
python -m pytest tests/ -q

# Type check
mypy parsers/ scripts/ --strict

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
