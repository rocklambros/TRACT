# OpenCRE Export Pipeline — Design Spec

**Date:** 2026-04-30
**Author:** Rock Lambros
**Status:** Draft — pending approval
**PRD Sections:** 10 (Phase 5), 7.2 (Framework Submission)
**Adversarial Review:** 4 rounds, 12 agents, convergence at Round 4

---

## 1. Goal

Build a `tract export --opencre` command that generates OpenCRE-compatible CSV from TRACT's crosswalk.db. This is the "gives back to OpenCRE" promise from PRD Section 1 — TRACT's ML-assisted crosswalk assignments, reviewed through active learning, exported in a format OpenCRE can directly import.

**Not in scope:** Web UI (eliminated), OpenCRE codebase modifications, hub proposal ID assignment (requires OpenCRE maintainer coordination), full human review of all assignments (Phase 3).

## 2. Architecture

```
crosswalk.db ──→ staleness check ──→ filter pipeline ──→ CSV generator ──→ export.csv
                       │                    │                   │              │
                  fetch upstream        exclude:             CRE 0 only    manifest.json
                  CRE IDs, diff      - ground_truth          (no hierarchy
                  against TRACT      - confidence < floor     columns)
                                     - is_ood = 1
                                     - NULL confidence
```

**Development flow:** export CSV → import into local OpenCRE fork → verify → iterate
**Production flow:** export CSV → review in TRACT → submit directly to upstream OpenCRE

The fork at `~/github_projects/OpenCRE` is a development sandbox only. Production submissions skip the fork.

## 3. OpenCRE CSV Format

OpenCRE's canonical interchange format, parsed by `parse_export_format()` in `application/utils/external_project_parsers/parsers/export_format_parser.py`.

### Column structure

| Column | Format | Example |
|--------|--------|---------|
| `CRE 0` | `id\|name` (pipe-delimited) | `607-671\|Protect against JS or JSON injection attacks` |
| `StandardName\|name` | Section name/title | `Adversarial Input Detection` |
| `StandardName\|id` | Section ID | `AML.M0015` |
| `StandardName\|description` | Section description | `Detect and block adversarial inputs...` |
| `StandardName\|hyperlink` | URL to official source | `https://atlas.mitre.org/mitigations/AML.M0015` |

### Critical format decisions (from adversarial review)

**CRE 0 only — no hierarchy columns.** TRACT populates only `CRE 0` with the leaf hub. We are adding `LinkedTo` edges between standards and existing CRE hubs, not redefining the CRE hierarchy tree. The hierarchy already exists in OpenCRE's database.

**Verified safe (R4):** The parser's hierarchy state machine (`previous_cre`/`highest_cre`/`previous_index`) only fires when `previous_index < i` or `highest_index < i`. With all rows at `i=0`, neither condition is ever true after row 1. No spurious `Contains` links are created.

**One row per assignment.** A control mapped to multiple hubs produces multiple rows with the same standard columns but different `CRE 0` values. The parser deduplicates standards by `(name, id)` and accumulates `LinkedTo` links correctly.

**Rows sorted by `(hub_id, framework_name, section_id)`.** Deterministic output, not required by the parser but aids debugging and diffing.

## 4. Framework Name Mapping

**CRITICAL (R1-C7, R4-F2):** TRACT's internal framework IDs must map to OpenCRE's exact `FAMILY_NAME` strings. A mismatch creates duplicate standard entries in OpenCRE's database.

### Verified mapping table

| TRACT `framework_id` | OpenCRE name | Status | Verification source |
|---|---|---|---|
| `mitre_atlas` | `"MITRE ATLAS"` | EXISTS | `spreadsheet_mitre_atlas.py` → `FAMILY_NAME` |
| `owasp_llm_top10` | `"OWASP Top10 for LLM"` | EXISTS | OpenCRE parser `FAMILY_NAME` (note: "Top10" no space) |
| `nist_ai_600_1` | `"NIST AI 600-1"` | **NEW** | Not in OpenCRE. Different from `"NIST AI 100-2"`. |
| `csa_aicm` | `"CSA AI Controls Matrix"` | **NEW** | Not in OpenCRE |
| `eu_ai_act` | `"EU AI Act"` | **NEW** | Not in OpenCRE |
| `owasp_agentic_top10` | `"OWASP Agentic AI Top 10"` | **NEW** | Not in OpenCRE |

**Implementation:** Hardcoded dict in `tract/export/opencre_names.py`. `KeyError` on unknown framework — no silent fallthrough. See GitHub issue #16 for the ongoing process when adding new frameworks.

**New-to-OpenCRE frameworks** must include full metadata in the CSV (all 4 standard columns populated: `name`, `id`, `description`, `hyperlink`). OpenCRE creates stub entries from CSV; we ensure they're not skeletal.

## 5. Filter Pipeline

Assignments pass through these filters in order before export. Each filter logs what it excludes and why.

### 5.1 Ground Truth Exclusion (R1-C3)

**Filter:** `provenance != 'ground_truth_T1-AI'`

The 78 ground_truth_T1-AI assignments are OpenCRE's own `AutomaticallyLinkedTo` edges, imported into TRACT as training data. Exporting them back is circular — contributing data they already have.

**Removes:** 78 assignments. Remaining: 558.

### 5.2 NULL and OOD Exclusion

**Filter:** `confidence IS NOT NULL AND is_ood != 1`

NULL confidence means the assignment was never scored by the model. OOD-flagged assignments, by definition, don't fit existing hubs — contributing them contradicts the OOD signal.

**Removes:** ~7 NULL confidence records.

### 5.3 Confidence Floor (R1-C4, R4 validated)

**Global floor:** `confidence >= 0.30`

**Per-framework override:** `mitre_atlas >= 0.35`

**Empirical justification (R4):** 60% of human corrections and 57% of human rejections during active learning had confidence <0.30. The 0.30 floor captures the majority of known-bad predictions. Values are calibrated probabilities (temperature-scaled), not raw cosine similarities.

**ATLAS override rationale:** Model delta over zero-shot was +0.006 for ATLAS — essentially zero improvement. The 0.35 floor is more conservative than the global 0.30, filtering the weakest zero-shot guesses while keeping assignments where the model shows genuine confidence.

**Configuration:** `tract/config.py`

```python
OPENCRE_EXPORT_CONFIDENCE_FLOOR = 0.30
OPENCRE_EXPORT_CONFIDENCE_OVERRIDES = {"mitre_atlas": 0.35}
```

### 5.4 Expected surviving counts

| Framework | Before filters | After all filters | Status |
|-----------|---------------|-------------------|--------|
| CSA AICM | 243 | ~184 | NEW |
| MITRE ATLAS | 195 | ~120 | EXISTS |
| EU AI Act | 100 | ~84 | NEW |
| OWASP Agentic | 10 | ~8 | NEW |
| NIST AI 600-1 | 10 | ~7 | NEW |
| OWASP LLM | 0 (all ground_truth) | 0 | — |
| **Total** | **558** | **~403** | |

## 6. Pre-Export Staleness Check (R1-C1)

Before generating the CSV, fetch current CRE IDs from upstream OpenCRE and diff against TRACT's hub snapshot.

```
GET https://opencre.org/rest/v1/root_cres → collect all CRE IDs recursively
Diff against TRACT's hub IDs from crosswalk.db
```

**Outcomes:**
- All IDs match → proceed
- TRACT has IDs not in upstream → WARN (hub may have been removed/merged). Affected assignments excluded from export with logged reason.
- Upstream has IDs not in TRACT → INFO only (new hubs added, not our concern)

**File:** `tract/export/staleness.py`

**Note (R2 downgrade):** OpenCRE's ontology changes slowly (expert-curated). The `myopencre_parser` also fails loud on ID/name conflicts (`ValueError`), so stale IDs won't silently corrupt — they'll crash the import. This check is defense-in-depth, not the primary safety net.

## 7. Export Manifest (R2-N1)

Every export generates `export_manifest.json` alongside the CSV:

```json
{
  "tract_version": "1.0.0",
  "model_adapter_hash": "a3f2b1...",
  "confidence_floor": 0.30,
  "confidence_overrides": {"mitre_atlas": 0.35},
  "export_date": "2026-04-30T22:00:00Z",
  "tract_git_sha": "bf99bf1...",
  "staleness_check": {"status": "pass", "upstream_hub_count": 522},
  "per_framework": {
    "csa_aicm": {"exported": 184, "excluded_confidence": 59, "excluded_ood": 0},
    "mitre_atlas": {"exported": 120, "excluded_confidence": 75, "excluded_ood": 0},
    "eu_ai_act": {"exported": 84, "excluded_confidence": 16, "excluded_ood": 0},
    "owasp_agentic_top10": {"exported": 8, "excluded_confidence": 2, "excluded_ood": 0},
    "nist_ai_600_1": {"exported": 7, "excluded_confidence": 3, "excluded_ood": 0}
  },
  "total_exported": 403,
  "total_excluded": 155,
  "provenance": "active_learning_round_2, human-reviewed (reviewer: Rock Lambros)"
}
```

## 8. Import Path

### 8.1 Development (fork)

**API endpoint:** `POST /rest/v1/cre_csv_import` with `CRE_ALLOW_IMPORT=1` environment variable.

**NOT `from_spreadsheet`** — that path is hardwired to Google Sheets (`gspread.oauth()`). Confirmed in R1-C6, verified in R4.

**Fork initialization:** The fork has no database. Before first import, initialize with:
```bash
cd ~/github_projects/OpenCRE
CRE_ALLOW_IMPORT=1 python cre.py --upstream_sync --cache_file cre.db
```

**Import helper:** `scripts/opencre_import.sh` — starts fork app, uploads CSV via curl, reports result.

### 8.2 Production (upstream)

Submit the exported CSV and manifest as a PR to the OpenCRE project, or upload via their hosted API if they enable `CRE_ALLOW_IMPORT`. The approval gate is the upstream maintainers' PR review process.

### 8.3 Idempotency (R1-I1, R4-F1)

**No `--clean` flag.** The proposed `--clean` using `delete_nodes()` was rejected in R4 — it deletes ALL nodes for a standard name, including OpenCRE's own curated links. No provenance tag distinguishes TRACT imports from native data.

**Development idempotency:** Reinitialize the fork's DB from scratch (`--upstream_sync`), then re-import. The fork is disposable.

**Production idempotency:** OpenCRE's `myopencre_parser` handles duplicates via `DuplicateLinkException` — duplicate links are caught and skipped. Re-submitting the same CSV is safe (idempotent by exception handling).

## 9. Hub Proposals (Separate Pipeline)

TRACT's 5 hub proposals (from HDBSCAN clustering of OOD controls) cannot flow through the CSV pipeline — OpenCRE has no mechanism for proposing new CRE IDs. The `CRE.__post_init__` validator enforces `\d\d\d-\d\d\d` format; there is no draft/pending CRE concept.

**Hub proposal delivery:** A structured document (JSON + human-readable summary) submitted alongside the CSV contribution. Contains:
- Proposed hub name and description
- Member control IDs with framework attribution
- Nearest existing hub (ID, name, similarity)
- Suggested parent hub for hierarchy placement
- Evidence score

OpenCRE maintainers assign the CRE ID if they accept the proposal. Once assigned, member control assignments can flow through the normal CSV pipeline in a subsequent export.

**File:** `tract export --opencre-proposals` generates `hub_proposals_for_opencre.json`.

## 10. Verification (R1-M4, escalated to IMPORTANT)

### 10.1 Post-import verification script

After importing into the fork, `scripts/verify_opencre_import.py` queries the fork's API and compares against the export manifest:

```
For each framework in manifest:
  GET /rest/v1/standard/<opencre_name>
  Count LinkedTo CRE links in response
  Compare against manifest's "exported" count
  Report: match / mismatch / missing standard
```

### 10.2 Pilot batch strategy (R4-F3 corrected)

Do not import all ~403 assignments at once. Pilot sequence:

1. **NIST AI 600-1** (7 assignments) — smallest, new to OpenCRE, zero collision risk. Validates end-to-end pipeline.
2. **EU AI Act** (84 assignments) — medium batch, also new to OpenCRE. Validates scale.
3. **CSA AICM** (184 assignments) — largest new framework.
4. **OWASP Agentic** (8 assignments) — small, new.
5. **MITRE ATLAS** (120 assignments) — EXISTS in OpenCRE. This is the highest-risk import (could create duplicate links if section IDs don't match). Import last, after all new-framework pilots succeed.

**NOT OWASP LLM Top 10** — zero surviving assignments after ground truth exclusion (R4-F3).

## 11. CLI Interface

```bash
# Full export with all filters
tract export --opencre --output-dir ./opencre_export/

# Dry run — show what would be exported without writing files
tract export --opencre --dry-run

# Export specific framework only (for pilot batches)
tract export --opencre --framework nist_ai_600_1

# Export hub proposals document
tract export --opencre-proposals --output-dir ./opencre_export/
```

**Output files:**
- `opencre_export/<framework_name>.csv` — one CSV per framework (avoids multi-framework row ordering issues)
- `opencre_export/export_manifest.json` — provenance and counts
- `opencre_export/hub_proposals_for_opencre.json` — hub proposals (separate command)

## 12. File Structure

```
tract/
  export/
    __init__.py
    opencre_csv.py        # CSV generation from crosswalk.db
    opencre_names.py      # Framework name mapping table
    staleness.py          # Pre-export CRE ID staleness check
    manifest.py           # Export manifest generation
    filters.py            # Confidence floor, ground truth, OOD filters
  config.py               # Add OPENCRE_EXPORT_* constants
  cli.py                  # Add export subcommand

scripts/
  opencre_import.sh       # Fork import helper (start app + curl upload)
  verify_opencre_import.py # Post-import verification

tests/
  test_export/
    test_opencre_csv.py   # CSV format correctness
    test_filters.py       # Filter pipeline unit tests
    test_opencre_names.py # Name mapping completeness
    test_staleness.py     # Staleness check logic
    test_manifest.py      # Manifest generation
```

## 13. PRD Impact

### Eliminated
- **Phase 2A (Section 7.1):** Dash Web UI — eliminated entirely. OpenCRE provides the web view.

### Modified
- **Phase 2B (Section 7.2):** Framework Submission — CLI-only (`tract ingest` + `tract validate`), no web upload page.
- **Phase 4 (Section 9):** API scoped to inference-only (`/v1/assign`). Read endpoints (`/v1/hub`, `/v1/compare`, `/v1/framework`) duplicate OpenCRE's existing API — either kill them or explicitly document them as TRACT-local convenience endpoints.
- **Phase 5 (Section 10):** OpenCRE Upstream Contribution — this spec IS Phase 5. Export pipeline + pilot strategy + hub proposals.

### Unchanged
- **Phase 2C (Section 7.3):** HuggingFace publication — independent of export.
- **Phase 2D (Section 7.4):** AI/Traditional bridge — output feeds through same CSV export pipeline.
- **Phase 3 (Section 8):** Published dataset — independent of OpenCRE contribution.

## 14. Adversarial Review Summary

4 rounds, 12 specialized agents. Key corrections incorporated:

| Round | Key Finding | Design Response |
|-------|------------|-----------------|
| R1 | Ground truth circularity (78 records) | Excluded via provenance filter (§5.1) |
| R1 | No confidence floor | Global 0.30 + per-framework overrides (§5.3) |
| R1 | `from_spreadsheet` is Google Sheets only | Use API endpoint instead (§8.1) |
| R1 | Framework name mismatch | Verified mapping table with `KeyError` on unknown (§4) |
| R2 | Ontology drift risk | Pre-export staleness check (§6) |
| R2 | New frameworks need metadata | Full 4-column standard entries (§4) |
| R2 | No export manifest | `export_manifest.json` with provenance (§7) |
| R3 | CRE 0 only strategy | Verified safe — parser traced line-by-line (§3) |
| R3 | Fix ordering matters | Staleness → exclusions → filters → name mapping → CSV (§5) |
| R3 | Pilot batch strategy | Start small, new frameworks first, ATLAS last (§10.2) |
| R4 | `--clean` destroys upstream data | Dropped entirely, use fresh fork DB (§8.3) |
| R4 | OWASP LLM name wrong in mapping | Corrected to `"OWASP Top10 for LLM"` (§4) |
| R4 | OWASP LLM pilot dead (0 exports) | Switched pilot to NIST AI 600-1 (§10.2) |
| R4 | 0.30 floor empirically validated | 60% corrections, 57% rejections below 0.30 (§5.3) |
