# Canonical Export Design Specification

**Status:** Draft — pending user review before implementation
**Date:** 2026-05-04
**Scope:** TRACT canonical export format for OpenCRE RFC integration

## 1. Motivation

OpenCRE's [easier-importing RFC](https://github.com/OWASP/OpenCRE/blob/main/docs/designs/easier-importing.md) defines a 9-module architecture (A-I) for incremental, reviewable imports. The RFC explicitly defers the canonical schema to "Spike A2." TRACT proposes to be the pilot integration — defining the canonical schema that OpenCRE's Module B parser will consume.

This spec defines:
- A machine-readable canonical format for controls and CRE mappings
- A changeset format for incremental updates (RFC Module D)
- Version tracking infrastructure for diffing between export generations
- Backward compatibility with the existing CSV export pipeline

## 2. Canonical Schema

### 2.1 Core Models (Pydantic v2)

```python
class CanonicalControl(BaseModel):
    control_id: str          # format: "{framework_id}:{section_id}" (DB canonical format)
    framework_id: str        # e.g., "csa_aicm"
    section_id: str          # e.g., "AIS-02"
    title: str
    description: str
    hyperlink: str           # computed at export time via build_hyperlink(framework_id, section_id)

class CREMapping(BaseModel):
    control_id: str          # matches CanonicalControl.control_id
    hub_id: str              # OpenCRE hub ID, e.g., "004-517"
    hub_name: str            # human-readable, e.g., "Security requirements"
    confidence: float        # calibrated confidence score (T=0.074)
    rank: int                # 1-indexed position in top-k for this control (v1 data is all rank=1; field is forward-compatible for top-k>1)
    link_type: str = "TRACT_ML_PREDICTED"  # distinguishes from LinkedTo/AutomaticallyLinkedTo
    provenance: str          # "active_learning_round_2"
    model_version: str       # DERIVED from StandardSnapshot.model_adapter_hash at export time (not stored per-assignment in DB)

class StandardSnapshot(BaseModel):
    schema_version: str = "1.0"
    framework_id: str
    framework_name: str      # OpenCRE display name from opencre_names.py
    export_date: str         # ISO 8601 UTC
    content_hash: str        # SHA-256 of this snapshot's canonical JSON (see §2.2)
    tract_version: str       # git SHA at export time
    model_adapter_hash: str
    filter_policy: FilterPolicy
    controls: list[CanonicalControl]
    mappings: list[CREMapping]

class FilterPolicy(BaseModel):
    confidence_floor: float
    confidence_override: float | None  # per-framework override, if different
    excluded_ground_truth: bool = True
    excluded_ood: bool = True
    excluded_null_confidence: bool = True
    review_status_required: str = "accepted"
```

### 2.2 Content Hash

Deterministic hash for version comparison:

```python
content_hash = SHA256(json.dumps(
    snapshot.model_dump(exclude={"content_hash", "export_date"}),
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
))
```

The hash excludes `content_hash` (self-referential) and `export_date` (non-semantic). Two exports with identical controls, mappings, and filter policy produce the same hash regardless of when they run.

### 2.3 Control ID Format

**Canonical format: single colon** — `{framework_id}:{section_id}` (e.g., `csa_aicm:AIS-02`).

This matches the crosswalk.db schema, which is the source of truth for exports. The `::` separator in `deployment_artifacts.npz` is internal to the embedding pipeline (`t5_finalize_crosswalk.py:470`) and does not appear in canonical exports.

The embedding export (§5) normalizes `::` → `:` when cross-referencing artifact IDs against canonical control IDs.

## 3. Changeset Format

Changesets represent the diff between two export versions, enabling OpenCRE's incremental import (RFC Module D).

### 3.1 Operations

Six operations covering controls and mappings:

| Operation | Entity | Semantics |
|-----------|--------|-----------|
| `ADD_CONTROL` | CanonicalControl | New control not in prior version |
| `UPDATE_CONTROL` | CanonicalControl | Control metadata changed (title, description, hyperlink) |
| `DELETE_CONTROL` | control_id | Control removed from framework or filtered out |
| `ADD_MAPPING` | CREMapping | New control-to-hub assignment |
| `UPDATE_MAPPING` | CREMapping | Mapping metadata changed (confidence, rank, provenance, model_version) |
| `DELETE_MAPPING` | (control_id, hub_id) | Assignment removed or rejected |

### 3.2 Changeset Envelope

```python
class ChangesetEntry(BaseModel):
    operation: Literal[
        "ADD_CONTROL", "UPDATE_CONTROL", "DELETE_CONTROL",
        "ADD_MAPPING", "UPDATE_MAPPING", "DELETE_MAPPING",
    ]
    entity: CanonicalControl | CREMapping | None = None  # present for ADD/UPDATE
    before: CanonicalControl | CREMapping | None = None   # present for UPDATE (prior state)
    key: str | None = None  # present for DELETE: control_id or "control_id|hub_id"

class Changeset(BaseModel):
    schema_version: str = "1.0"
    framework_id: str
    from_version: str | None   # content_hash of base version (None for initial)
    to_version: str            # content_hash of target version
    export_date: str
    operations: list[ChangesetEntry]
    summary: ChangesetSummary
    impact: ImpactAnalysis      # downstream effect estimate (§7)

class ChangesetSummary(BaseModel):
    controls_added: int
    controls_updated: int
    controls_deleted: int
    mappings_added: int
    mappings_updated: int
    mappings_deleted: int
```

### 3.3 Diff Algorithm

1. Load current snapshot from crosswalk.db (applying filter policy from §4)
2. Load prior snapshot from `export_history` table (§6)
3. Sort both by `control_id` (controls) and `(control_id, hub_id)` (mappings)
4. Diff controls by `control_id` key: ADD if only in current, DELETE if only in prior, UPDATE if present in both but mutable fields differ
   - **Identity key:** `control_id`
   - **Mutable fields:** `title`, `description`, `hyperlink`
   - **Immutable fields (not diffed):** `framework_id`, `section_id` (derived from control_id)
5. Diff mappings by `(control_id, hub_id)` key: same logic
   - **Identity key:** `(control_id, hub_id)`
   - **Mutable fields:** `confidence`, `rank`, `provenance`, `model_version`
   - **Immutable fields (not diffed):** `hub_name` (OpenCRE display label — may change in OpenCRE but is not TRACT's mutation), `link_type`
6. For UPDATE operations, include `before` state for auditability

**Confidence drift:** Model retraining may change confidence scores for existing mappings without changing the assignment itself. These correctly appear as `UPDATE_MAPPING` operations. No epsilon filtering is applied in v1 — all confidence changes are surfaced. If changeset noise becomes a problem after future retraining, epsilon-based filtering can be added in v2.

### 3.4 Initial Export

If the `export_history` table is empty, the first export is automatically treated as initial — all entries are `ADD_CONTROL` and `ADD_MAPPING`. No `--initial` flag is needed.

## 4. Filter Policy

The canonical export applies the same filter pipeline as the existing CSV export (`tract/export/filters.py`), codified explicitly:

1. **Ground truth exclusion:** `provenance != 'ground_truth_T1-AI'`
   - *Rationale:* Ground truth entries originate from OpenCRE itself. Re-exporting them would create circular duplicates. 65 of 78 ground truth entries are MITRE ATLAS links already present in OpenCRE.
2. **NULL confidence exclusion:** `confidence IS NOT NULL`
3. **OOD exclusion:** `is_ood != 1`
4. **Review status:** `review_status = 'accepted'`
5. **Confidence floor:** Per-framework threshold (global default: 0.3, mitre_atlas override: 0.35)

After filtering, the current dataset yields 411 exportable assignments across 5 frameworks:
- csa_aicm: 184
- mitre_atlas: 128 (195 accepted, minus 67 below confidence floor of 0.35; 65 ground truth excluded separately)
- eu_ai_act: 84
- owasp_agentic_top10: 8
- nist_ai_600_1: 7

Note: `owasp_llm_top10` has 13 ground truth entries in the DB but zero ML predictions — it is entirely filtered out and does not appear in canonical exports.

The `FilterPolicy` model (§2.1) is embedded in each snapshot, making the filter configuration self-documenting and auditable.

## 5. Embedding Export

Ship pre-computed BGE-large-v1.5 embeddings alongside canonical JSON for downstream consumers who want similarity search without running inference.

### 5.1 Format

```
canonical_export/
  {framework_id}/
    snapshot.json          # StandardSnapshot
    changeset.json         # Changeset (if not initial)
    embeddings.npz         # optional, per-framework control embeddings
```

The `embeddings.npz` contains:
- `control_embeddings`: float32 array, shape (n_controls, 1024)
- `control_ids`: string array matching row order
- `hub_embeddings`: float32 array, shape (n_hubs, 1024)
- `hub_ids`: string array matching row order
- `model_adapter_hash`: string, must match snapshot's `model_adapter_hash`

### 5.2 ID Normalization

`deployment_artifacts.npz` uses `::` as the framework-control separator. The embedding export normalizes these to `:` to match canonical control IDs:

```python
canonical_id = artifact_id.replace("::", ":")
```

This normalization is validated by an assertion that all normalized IDs exist in the canonical snapshot's control set.

## 6. Export State Tracking

### 6.1 Schema

New table in the project's crosswalk database (`PHASE1C_CROSSWALK_DB_PATH` from `tract/config.py`, currently `results/phase1c/crosswalk.db`). The `export_history` table is created by the canonical export module on first run via `CREATE TABLE IF NOT EXISTS`:

```sql
CREATE TABLE IF NOT EXISTS export_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    framework_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    export_date TEXT NOT NULL DEFAULT (datetime('now')),
    snapshot_json TEXT NOT NULL,  -- full StandardSnapshot as JSON
    filter_policy_json TEXT NOT NULL,
    assignment_count INTEGER NOT NULL,
    control_count INTEGER NOT NULL,
    tract_version TEXT NOT NULL
);
CREATE INDEX idx_export_history_fw ON export_history(framework_id, export_date);
```

### 6.2 Workflow

1. Ensure `export_history` table exists (`CREATE TABLE IF NOT EXISTS`)
2. Build current snapshot from live DB data + filter policy
3. Query `export_history` for the most recent entry for this framework_id
4. If no prior entry exists → initial export (all ADDs)
5. If prior entry exists → deserialize `snapshot_json`, **verify its `content_hash` matches the stored hash** before diffing (guards against DB corruption or manual edits)
6. Diff current snapshot against prior snapshot
7. Write snapshot JSON + changeset JSON + optional embeddings to output directory
8. Insert new row into `export_history` (skipped if `--dry-run`)

### 6.3 Validation

Before serialization, every snapshot passes `StandardSnapshot.model_validate()`. This is enforced by the Pydantic schema — no separate validation step needed.

## 7. Impact Analysis

Each changeset includes an impact analysis section estimating downstream effects:

```python
class ImpactAnalysis(BaseModel):
    affected_hubs: list[str]       # hub_ids with changed mappings
    affected_frameworks: list[str]  # frameworks with changed controls
    co_mapped_changes: int         # controls that map to hubs shared with other frameworks
    scope: str                     # "minor" | "moderate" | "major"
```

Scope thresholds:
- **minor:** <10 operations, no DELETE_MAPPINGs
- **moderate:** 10-50 operations or any DELETE_MAPPINGs
- **major:** >50 operations or any DELETE_CONTROLs

The `ImpactAnalysis` is included in the `Changeset` model (added as a field alongside `summary`).

Note: `affected_frameworks` and `co_mapped_changes` require a cross-framework query — the export must check which other frameworks share hubs with the changed mappings. This is computed from the full `assignments` table, not just the framework being exported.

## 8. CLI Design

### 8.1 New Subcommand

```
tract export-canonical [--framework FW_ID] [--output-dir DIR]
                       [--with-embeddings] [--dry-run]
```

- `--framework`: Export single framework (default: all)
- `--output-dir`: Output directory (default: `./canonical_export`)
- `--with-embeddings`: Include `.npz` embedding files
- `--dry-run`: Show what would be exported (snapshot summary, changeset operations count) without writing files or updating `export_history`. Dry-run MUST NOT mutate state.

### 8.2 Module Structure

To avoid further bloating `cli.py` (960+ lines), the canonical export logic lives in `tract/export/canonical.py` with the CLI handler being a thin wrapper:

```
tract/export/
  canonical.py       # NEW: snapshot builder, differ, serializer
  canonical_schema.py  # NEW: Pydantic models (§2.1)
  filters.py         # EXISTING: reused for filter logic (canonical export extends query to include model_version derivation, rank computation, link_type defaulting)
  opencre_csv.py     # EXISTING: retained for backward compatibility
  opencre_names.py   # EXISTING: reused for framework_name lookup + build_hyperlink()
```

The canonical export builds its own query that wraps `filters.py`'s filter logic but selects additional fields. `filters.py` itself is not modified — the canonical query extends it.

## 9. Backward Compatibility

The existing `tract export-opencre` subcommand and CSV format are retained unchanged. The canonical export is a new, parallel path:

- `tract export-opencre` → per-framework CSVs (current behavior, no changes)
- `tract export-canonical` → per-framework JSON snapshots + changesets (new)

Both commands share the same filter pipeline (`filters.py`) and framework name mapping (`opencre_names.py`). The canonical format is a superset of the CSV — every field in the CSV exists in the canonical schema, plus confidence, provenance, link_type, and version metadata.

## 10. OpenCRE RFC Alignment

| RFC Module | TRACT Support | Notes |
|------------|--------------|-------|
| A: Canonical schema | **Full** | This spec defines the schema the RFC punted |
| B: Parser | **Deferred** | PR to OpenCRE after they approve the schema |
| C: Diff engine | **Full** | Changeset format with 6 operations |
| D: Review UI | **N/A** | OpenCRE's responsibility |
| E: Provenance | **Full** | `link_type`, `provenance`, `model_version` fields |
| F: Rollback | **Partial** | `before` state in UPDATE operations enables undo |
| G: Staleness | **Full** | Existing `tract/export/staleness.py` |
| H: Multi-source | **Full** | Per-framework snapshots, content hashing |
| I: Monitoring | **Partial** | Impact analysis; dashboards deferred |

## 11. Adversarial Review Summary

This design was subjected to a 4-round, 8-agent adversarial review (security architecture × 2, methodology × 2, implementation/operations × 2, empirical validation, cross-examination). Review converged at Round 4 — cross-examiner found zero new P0s; remaining findings are P2 (implementer clarity).

### Rounds 1-2 (design-level)

| Finding | Severity | Resolution |
|---------|----------|-----------|
| Filter policy undefined | P0 | §4 explicitly codifies 5-stage filter with rationale |
| Control ID `::` vs `:` mismatch | P1 | §2.3 defines canonical format; §5.2 adds normalization |
| Missing link_type field | P1 | §2.1 adds `link_type` with default `TRACT_ML_PREDICTED` |
| Multi-hub ranking lost | P1 | §2.1 adds `rank` field to CREMapping |
| Diff field specificity | P1 | §3.3 distinguishes identity keys from mutable fields |
| Pydantic validation not mandated | P1 | §6.3 mandates `model_validate()` |
| Version hash underspecified | P2 | §2.2 defines deterministic hash formula |
| No `--initial` flag needed | — | §3.4 auto-detects from empty export_history |
| Confidence drift floods changesets | DEFERRED | §3.3 documents as expected; epsilon filtering in v2 |

### Rounds 3-4 (data-model verification)

| Finding | Severity | Resolution |
|---------|----------|-----------|
| `model_version` NULL in DB — Pydantic crash | P0 | §2.1: derived from `model_adapter_hash` at export time |
| Per-framework counts wrong (3 of 5) | P1 | §4: corrected to match verified SQL query output |
| JSONL format undefined — untestable feature | P1 | §8.1: removed from v1 CLI |
| `--dry-run` state mutation ambiguous | P1 | §8.1 + §6.2: dry-run skips export_history writes |
| `hyperlink` computed, not stored | P2 | §2.1: documented as `build_hyperlink()` derivation |
| `rank` always 1 in v1 | P2 | §2.1: documented as forward-compatible |
| filters.py "reused unchanged" false | P2 | §8.2: clarified as extended, not modified |
| DB path ambiguous | P2 | §6.1: references config constant |
| content_hash verify on deser | P2 | §6.2: hash verified before diffing |
| ImpactAnalysis cross-fw query | P2 | §7: cross-framework dependency noted |
| export_history CREATE TABLE ownership | P2 | §6.1: canonical module creates on first run |

### Rejected findings

| Finding | Reason |
|---------|--------|
| Filter policy in hash → phantom changesets | Correct behavior: filter policy IS semantic identity |
| RFC Module E "Full" → "Partial" | Three provenance fields provide full traceability |
| OOD filter is a no-op | Guard rail for future runs |
| Snapshot retention policy needed | Non-issue at current scale (~400KB/export) |
