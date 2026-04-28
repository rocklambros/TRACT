# Phase 1A Design Spec: CRE Hierarchy, Hub Descriptions, Traditional Framework Ingestion

**Date:** 2026-04-28
**PRD Sections:** 6.1, 6.2, 6.3
**Prerequisites:** Phase 0 gates passed (Gate A: Opus hit@5=0.722 > 0.50; Gate B: gap=0.117 > 0.10)
**Scope:** Data infrastructure for Phase 1B model training

---

## 1. CRE Hierarchy Module (`tract/hierarchy.py`)

### 1.1 Purpose

Promote the Phase 0 `CREHierarchy` dataclass from `scripts/phase0/common.py` into a production Pydantic module at `tract/hierarchy.py`. This is the coordinate system — every downstream component depends on it.

### 1.2 Data Model

```python
class HubNode(BaseModel):
    hub_id: str
    name: str
    parent_id: str | None
    children_ids: list[str]
    depth: int
    branch_root_id: str
    hierarchy_path: str           # "Root > Parent > ... > Hub"
    is_leaf: bool
    sibling_hub_ids: list[str]    # same-parent hubs (hard-negative candidates)

class CREHierarchy(BaseModel):
    hubs: dict[str, HubNode]
    roots: list[str]              # sorted deterministically
    label_space: list[str]        # sorted leaf hub IDs (frozen, deterministic)
    fetch_timestamp: str          # ISO 8601 from opencre_all_cres.json
    data_hash: str                # SHA-256 of source file
    version: str                  # "1.0"
```

### 1.3 Construction

`CREHierarchy.from_opencre(cres: list[dict], fetch_timestamp: str, data_hash: str) -> CREHierarchy`

Build process (mirrors Phase 0 `build_hierarchy`):

1. Collect all CRE records (doctype=CRE) into hub dict
2. Build parent/child edges from "Contains" links
3. Compute roots (hubs with no parent), sorted by hub_id
4. BFS from roots to assign depth, branch_root_id
5. Compute `hierarchy_path` per hub during BFS
6. Compute `sibling_hub_ids` per hub (children of same parent, excluding self)
7. Compute `label_space`: sorted list of leaf hub IDs
8. Validate (see 1.5)

### 1.4 Orphan Policy

The current CRE data has 5 roots, all with children, 0 orphans (hubs with no parent and no children). The spec handles the general case:

- **Orphan hub**: a hub with no parent AND no children. Treated as a single-node branch. `branch_root_id = self`, `depth = 0`, `is_leaf = True`, included in `label_space`.
- If orphans are detected, log at WARNING level with hub IDs and names.

### 1.5 Validation

`CREHierarchy.validate_integrity() -> None` (raises `ValueError` on failure)

1. **No cycles**: every hub reachable from its branch root in ≤ max_depth steps
2. **All leaves reachable**: every leaf in `label_space` has a path to a root
3. **Depth consistency**: `depth[child] == depth[parent] + 1` for every edge
4. **No dangling references**: every `parent_id` and every entry in `children_ids` exists in `hubs`
5. **Label space determinism**: `label_space == sorted(label_space)` — always sorted by hub_id
6. **Expected counts**: 522 total hubs, 400 leaves, 5 roots (WARNING if different, not ERROR — CRE data may evolve)

### 1.6 Query Methods

- `leaf_hub_ids() -> list[str]` — returns `label_space`
- `parent(hub_id) -> HubNode | None`
- `children(hub_id) -> list[HubNode]`
- `siblings(hub_id) -> list[HubNode]` — same parent, excluding self
- `branch_hub_ids(root_id) -> list[str]` — all descendants including root
- `hierarchy_path(hub_id) -> str` — cached in HubNode
- `hub_by_name(name: str) -> HubNode | None` — case-insensitive lookup

### 1.7 Serialization

- `save(path: Path) -> None` — atomic write to JSON (via `tract.io.atomic_write_json`)
- `CREHierarchy.load(path: Path) -> CREHierarchy` — load + validate
- JSON has sorted keys for deterministic output
- Output path: `data/processed/cre_hierarchy.json`

### 1.8 Phase 0 Compatibility

Phase 0 code stays frozen. The new `tract.hierarchy.CREHierarchy` must produce identical `leaf_hub_ids()` output as the Phase 0 version when given the same input data. Enforced by an integration test:

```python
def test_phase0_leaf_hub_parity():
    """Phase 1A label_space must match Phase 0 leaf_hub_ids."""
    phase0_hierarchy = phase0_build_hierarchy(cres)
    phase1a_hierarchy = CREHierarchy.from_opencre(cres, ...)
    assert phase1a_hierarchy.label_space == sorted(phase0_hierarchy.leaf_hub_ids())
```

---

## 2. Hub Description Generation (`scripts/phase1a/generate_descriptions.py`)

### 2.1 Purpose

Generate 2-3 sentence descriptions for all 400 leaf hubs. These become part of the model's hub representation in Phase 1B. Phase 0 showed hierarchy paths help (+7.6% hit@1) but verbose LLM descriptions hurt zero-shot embeddings (BGE -0.019, GTE -0.147). The descriptions are for the *trained* model, not zero-shot — but this is an untested hypothesis, so generation quality and expert review are critical.

### 2.2 Generation Pipeline

```
Input per hub:
  - hub name
  - hierarchy_path (from CREHierarchy)
  - all linked standard section names (from hub_links.json, excluding held-out frameworks)
  - sibling hub names (from CREHierarchy.siblings())

LLM call:
  - Model: claude-opus-4-20250514
  - Temperature: 0.0 (deterministic)
  - Max tokens: 500
  - System prompt: fixed (see 2.3)

Output per hub:
  - description: str (2-3 sentences)
  - what_it_covers: first sentence
  - distinguishing_features: what separates it from siblings
  - scope_boundary: what it does NOT cover
```

### 2.3 Prompt Template

```
You are a cybersecurity taxonomy expert. Generate a precise 2-3 sentence description for a CRE (Common Requirements Enumeration) hub node.

Hub name: {name}
Hierarchy path: {hierarchy_path}
Sibling hubs (same parent): {sibling_names}
Linked standard sections: {linked_section_names}

Write a description that:
1. Defines what this hub covers in concrete terms
2. Distinguishes it from its sibling hubs
3. States the boundary of its scope (what it does NOT cover)

Be specific and technical. Do not use filler phrases. Every word must add information.
```

### 2.4 Determinism

- `temperature=0.0` pinned in config
- Seed parameter set if API supports it
- Same prompt + same model = same output (within API determinism guarantees)
- Pilot descriptions from Phase 0 (50 hubs) are re-generated, not reused — the prompt is different (adds siblings), and mixing sampling regimes is unsound

### 2.5 Sanitization

All generated descriptions pass through `tract.sanitize.sanitize_text()` before storage. This handles:
- Null bytes, NFC normalization, HTML stripping, ligature fixes, whitespace collapse
- Max length enforcement (2000 chars via `DESCRIPTION_MAX_LENGTH`)

Zero-width characters (U+200B, U+200C, U+200D, U+FEFF) are stripped as part of the sanitization pipeline. Add to `tract/sanitize.py`:

```python
_ZERO_WIDTH_RE = re.compile(r'[​‌‍﻿]')
```

### 2.6 Execution

- Async with `asyncio.gather(*tasks, return_exceptions=True)`
- Semaphore limits concurrency (5 concurrent, configurable)
- `asyncio.wait_for(call, timeout=60)` per request
- Anthropic client created once, closed in `finally` block
- Intermediate saves every 50 hubs (atomic write to `data/processed/hub_descriptions_partial.json`)
- On completion, atomic write to `data/processed/hub_descriptions.json`
- Resume support: check for existing partial file, skip already-generated hubs (validated by checking hub_id exists AND description is non-empty string)
- On `return_exceptions=True`: inspect each result, log failures with hub_id, continue with remaining hubs. Raise if >10% fail.

### 2.7 Output Schema

```python
class HubDescription(BaseModel):
    hub_id: str
    hub_name: str
    hierarchy_path: str
    description: str               # sanitized 2-3 sentence description
    model: str                     # "claude-opus-4-20250514"
    temperature: float             # 0.0
    generated_at: str              # ISO 8601
    review_status: Literal["pending", "accepted", "edited", "rejected"]
    reviewed_description: str | None  # expert-edited version, if any
    reviewer_notes: str | None

class HubDescriptionSet(BaseModel):
    descriptions: dict[str, HubDescription]  # hub_id -> description
    generation_model: str
    generation_timestamp: str
    data_hash: str                            # SHA-256 of cre_hierarchy.json used
    total_generated: int
    total_pending_review: int
```

### 2.8 Expert Review Workflow

The user opens `data/processed/hub_descriptions.json` in a JSON editor (VS Code, etc.) and for each hub:
- Sets `review_status` to `accepted`, `edited`, or `rejected`
- If `edited`, writes the corrected text to `reviewed_description`
- Optionally adds `reviewer_notes`

A validation script (`scripts/phase1a/validate_descriptions.py`) checks:
- All 400 hubs present
- No description is empty
- All `reviewed_description` values (when present) pass sanitization
- Reports: N accepted, N edited, N rejected, N pending

### 2.9 Cross-Reference Validation

After generation, validate that every hub_id in `hub_descriptions.json` exists in `cre_hierarchy.json` and vice versa (for leaf hubs). Raises `ValueError` on mismatch.

### 2.10 Cost Estimate

400 hubs × ~1500 input tokens × ~200 output tokens = ~600K input + ~80K output tokens.
At Opus pricing (~$15/M input, ~$75/M output): ~$9 input + ~$6 output = ~$15 total.

---

## 3. Traditional Framework Ingestion (`scripts/phase1a/extract_traditional_frameworks.py`)

### 3.1 Purpose

Extract control/section data for the 19 OpenCRE frameworks that lack primary-source parsers. These frameworks already have CRE links (training data) but we need their control text representations for model input.

### 3.2 Framework Inventory

OpenCRE contains 22 distinct frameworks with `LinkedTo` or `AutomaticallyLinkedTo` links. Of these, 3 already have primary-source AI parsers (MITRE ATLAS, OWASP AI Exchange, OWASP Top10 for LLM). The remaining 19 are extracted here:

| Framework | OpenCRE Name | framework_id | Links | Link Type |
|-----------|-------------|--------------|-------|-----------|
| CAPEC | CAPEC | capec | 1799 | AutomaticallyLinkedTo |
| CWE | CWE | cwe | 613 | AutomaticallyLinkedTo |
| OWASP Cheat Sheets | OWASP Cheat Sheets | owasp_cheat_sheets | 391 | LinkedTo |
| NIST 800-53 v5 | NIST 800-53 v5 | nist_800_53 | 300 | LinkedTo |
| ASVS | ASVS | asvs | 277 | LinkedTo |
| DSOMM | DevSecOps Maturity Model (DSOMM) | dsomm | 214 | LinkedTo |
| WSTG | OWASP Web Security Testing Guide (WSTG) | wstg | 118 | LinkedTo |
| ISO 27001 | ISO 27001 | iso_27001 | 94 | LinkedTo |
| NIST 800-63 | NIST 800-63 | nist_800_63 | 79 | LinkedTo |
| OWASP Proactive Controls | OWASP Proactive Controls | owasp_proactive_controls | 76 | LinkedTo |
| ENISA | ENISA | enisa | 68 | LinkedTo |
| NIST AI 100-2 | NIST AI 100-2 | nist_ai_100_2 | 45 | LinkedTo |
| NIST SSDF | NIST SSDF | nist_ssdf | 46 | LinkedTo |
| ETSI | ETSI | etsi | 36 | LinkedTo |
| SAMM | SAMM / OWASP SAMM | samm | 30 | LinkedTo |
| Cloud Controls Matrix | Cloud Controls Matrix | csa_ccm | 29 | LinkedTo |
| BIML | BIML | biml | 21 | LinkedTo |
| OWASP Top 10 2021 | OWASP Top 10 2021 | owasp_top10_2021 | 17 | LinkedTo |
| OWASP Top10 for ML | OWASP Top10 for ML | owasp_ml_top10 | 10 | LinkedTo |

**Note:** NIST AI 100-2 and OWASP Top10 for ML appear in OpenCRE with CRE links but have no primary-source parser. They are extracted here alongside the traditional frameworks.

**Critical:** CSA Cloud Controls Matrix (csa_ccm, 29 links) is NOT the same as CSA AI Controls Matrix (csa_aicm, 243 controls, 0 CRE links). Never conflate them.

### 3.3 Extraction Logic

For each framework, extract unique mapping units from OpenCRE standard links:

```python
for cre in all_cres:
    for link in cre["links"]:
        doc = link["document"]
        if doc["doctype"] == "Standard" and doc["name"] in framework_names:
            key = (doc["name"], doc.get("sectionID", ""), doc.get("section", ""))
            # dedup on triple key: (standard_name, section_id, section_name)
```

**Dedup key**: `(framework_id, section_id, section_name)` — using triple key to handle empty `section_id` values. Two sections with same `section_id=""` but different `section_name` are distinct controls.

### 3.4 Control Construction

Each unique mapping unit becomes a `Control`:

```python
Control(
    control_id=f"{framework_id}:{section_id or slugify(section_name)}",
    title=section_name,
    description=section_name,  # OpenCRE only has section names, no full text
    full_text=None,
    hierarchy_level=None,
    parent_id=None,
    parent_name=None,
    metadata={
        "opencre_standard_name": original_standard_name,
        "opencre_section_id": section_id,
        "link_type": link_type,       # "LinkedTo" or "AutomaticallyLinkedTo"
    },
)
```

### 3.5 Framework Slug Validation

All `framework_id` values validated against regex: `^[a-z][a-z0-9_]{1,49}$`. Rejects path traversal attempts and enforces consistent naming. The `control_id` prefix uses the validated `framework_id`.

For `section_id` used in `control_id` construction: if `section_id` is empty, generate a slug from `section_name` using `re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')[:80]`. Validate the resulting slug is non-empty.

### 3.6 Output Schema

Each framework produces a `FrameworkOutput` (reusing existing `tract.schema.FrameworkOutput`):

```python
FrameworkOutput(
    framework_id="capec",
    framework_name="CAPEC",
    version="opencre-2026-04-27",    # "opencre-{fetch_date}"
    source_url="https://opencre.org",
    fetched_date="2026-04-27T...",    # from opencre_all_cres.json fetch_timestamp
    mapping_unit_level="section",     # all traditional frameworks use "section"
    controls=[...],                   # deduped Control objects
)
```

### 3.7 Output Files

- Per-framework: `data/processed/frameworks/{framework_id}.json` (atomic write)
- File listing order: glob results sorted by filename for determinism
- Existing AI framework files (from primary-source parsers) are NOT overwritten — the script only writes the 19 framework files listed in §3.2
- The 3 overlapping AI frameworks (MITRE ATLAS, OWASP AI Exchange, OWASP Top10 for LLM) already have richer primary-source parser outputs — those take precedence in `all_controls.json`
- After all frameworks written, rebuild `data/processed/all_controls.json` containing one entry per `framework_id`. AI parser outputs take precedence over OpenCRE extractions for overlapping framework_ids
- **Framework count in all_controls.json**: up to 31 unique framework_ids (12 AI primary-source + 19 OpenCRE-extracted, no overlap by construction). Exact count depends on which AI parsers have been run.

### 3.8 CAPEC/CWE Bare-ID Handling

CAPEC and CWE sections in OpenCRE often have bare numeric IDs (e.g., `section_id="184"`) without descriptive `section_name`. For these:
- `control_id` uses the standard prefix: `capec:184`, `cwe:79`
- `title` and `description` use `section_name` if available, otherwise `"{FRAMEWORK}-{section_id}"` (e.g., "CAPEC-184")
- Log at INFO level the count of bare-ID controls per framework

### 3.9 Execution

Single script, sequential processing (no async needed — all data is local):

1. Load `opencre_all_cres.json`
2. For each framework in `OPENCRE_FRAMEWORK_ID_MAP`:
   a. Extract unique mapping units (triple-key dedup)
   b. Construct `Control` objects
   c. Build `FrameworkOutput`
   d. Validate with Pydantic
   e. Atomic write to `data/processed/frameworks/{framework_id}.json`
3. Aggregate into `all_controls.json` (AI parser outputs take precedence)
4. Log summary: framework_id, control count, link count per framework

### 3.10 Expected Counts

The script logs a WARNING if extracted control count deviates >10% from the link count for any framework (some links point to the same section, so control count ≤ link count). This is informational, not blocking.

---

## 4. Cross-Cutting Concerns

### 4.1 Configuration (`tract/config.py`)

New constants:

```python
# Phase 1A
PHASE1A_DESCRIPTION_MODEL: Final[str] = "claude-opus-4-20250514"
PHASE1A_DESCRIPTION_TEMPERATURE: Final[float] = 0.0
PHASE1A_DESCRIPTION_MAX_TOKENS: Final[int] = 500
PHASE1A_DESCRIPTION_MAX_CONCURRENT: Final[int] = 5
PHASE1A_DESCRIPTION_SAVE_INTERVAL: Final[int] = 50
PHASE1A_DESCRIPTION_TIMEOUT_S: Final[int] = 60
PHASE1A_FRAMEWORK_SLUG_RE: Final[str] = r"^[a-z][a-z0-9_]{1,49}$"
```

### 4.2 Sanitization Extension

Add zero-width character stripping to `tract/sanitize.py`, applied after NFC normalization:

```python
_ZERO_WIDTH_RE: re.Pattern[str] = re.compile(r'[​‌‍﻿]')

def _strip_zero_width(text: str) -> str:
    return _ZERO_WIDTH_RE.sub("", text)
```

Insert into the pipeline between `_normalize_unicode` and `strip_html`.

### 4.3 Dependencies

No new dependencies. Uses existing: `pydantic`, `anthropic`, `asyncio`, `hashlib`.

### 4.4 File Layout

```
tract/
  hierarchy.py          # CREHierarchy + HubNode models (NEW)
  sanitize.py           # add _strip_zero_width (MODIFY)
  config.py             # add Phase 1A constants (MODIFY)
  schema.py             # unchanged — reuse FrameworkOutput, Control

scripts/phase1a/
  build_hierarchy.py         # CLI: build + validate + save hierarchy (NEW)
  generate_descriptions.py   # CLI: generate all 400 descriptions (NEW)
  validate_descriptions.py   # CLI: validate review status (NEW)
  extract_traditional_frameworks.py  # CLI: extract 19 frameworks (NEW)

data/processed/
  cre_hierarchy.json         # output of build_hierarchy
  hub_descriptions.json      # output of generate_descriptions
  frameworks/
    capec.json               # one per traditional framework
    cwe.json
    ...
    all_controls.json        # unified, AI parsers take precedence

tests/
  test_hierarchy.py          # unit + integration tests (NEW)
  test_descriptions.py       # generation + validation tests (NEW)
  test_traditional_frameworks.py  # extraction tests (NEW)
```

### 4.5 Error Handling

- All scripts use `logging` (never `print`)
- Specific exceptions only (no bare `except:`)
- Failed description generations logged with hub_id, error type, and message
- Script exits non-zero if any validation fails

### 4.6 Testing Strategy

| Component | Test Type | What It Validates |
|-----------|-----------|-------------------|
| CREHierarchy.from_opencre | Unit (fixture) | Correct tree from minimal CRE data |
| CREHierarchy.validate_integrity | Unit | Catches cycles, dangling refs, wrong depths |
| Phase 0 parity | Integration | label_space matches Phase 0 leaf_hub_ids |
| HubDescription sanitization | Unit | Zero-width chars stripped, length enforced |
| Description prompt | Unit | Correct template rendering |
| Triple-key dedup | Unit | Empty section_id handled correctly |
| Framework slug validation | Unit | Rejects path traversal, accepts valid slugs |
| CAPEC bare-ID handling | Unit | Correct control_id and title generation |
| End-to-end extraction | Integration | 19 frameworks × correct control counts |
| all_controls.json merge | Integration | AI parsers take precedence, no duplicates |

---

## 5. Adversarial Review Findings (Incorporated)

The following issues from the 7-agent adversarial review are addressed in this spec:

### 5.1 Must-Fix (All 10 Addressed)

| # | Issue | Where Addressed |
|---|-------|-----------------|
| 1 | Orphan hub policy | §1.4 — explicit handling |
| 2 | Empty section_id dedup | §3.3 — triple key |
| 3 | Framework slug path traversal | §3.5 — regex validation |
| 4 | Pilot description re-sanitization | §2.4 — re-generate, don't reuse |
| 5 | Zero-width character stripping | §4.2 — sanitize.py extension |
| 6 | Label space determinism | §1.2 — sorted, persisted in model |
| 7 | asyncio.gather failure handling | §2.6 — return_exceptions + threshold |
| 8 | Intermediate writes during generation | §2.6 — every 50 hubs |
| 9 | Cross-reference validation | §2.9 — hierarchy ↔ descriptions check |
| 10 | Glob ordering determinism | §3.7 — sorted by filename |

### 5.2 Should-Fix (8 Addressed Where Appropriate)

| # | Issue | Disposition |
|---|-------|-------------|
| 1 | Canonical JSON hash | Addressed: data_hash uses SHA-256 of source file |
| 2 | Skip-existing validation | Addressed: §2.6 resume checks hub_id + non-empty |
| 3 | Dry-run/limit flags | Deferred to implementation plan (CLI args) |
| 4 | Link-type metadata | Addressed: §3.4 metadata includes link_type |
| 5 | CAPEC/CWE bare-ID flagging | Addressed: §3.8 |
| 6 | NIST 800-53 versioning | Addressed: §3.6 version="opencre-{date}" |
| 7 | section_id integration test | Addressed: §4.6 test table |
| 8 | related_ids / sibling adjacency | Addressed: §1.2 sibling_hub_ids on HubNode |

### 5.3 Brutal Evaluator Challenges (Accepted)

| Challenge | Response |
|-----------|----------|
| Near-synonym hub pairs need analysis | Pre-generation: cluster 400 leaf hub names by embedding similarity, flag pairs > 0.85 cosine. Report only — do not collapse hubs (CRE owns the taxonomy). |
| Temperature/seed not pinned | Fixed: §2.4 — temperature=0.0, seed if API supports |
| Hard-negative fields should be MUST | Accepted: §1.2 — sibling_hub_ids on HubNode |
| $30 on descriptions vs synthetic augmentation | Noted as Phase 1B consideration. Descriptions serve dual purpose (hub representation + augmentation source). Cost is ~$15, not $30. |
| Phase 0 ↔ Phase 1A integration test | Fixed: §1.8 — explicit parity test |
| Zero-width chars demoted to SHOULD | Rejected — cheap to implement, prevents a class of embedding-space bugs. Kept as MUST. |

---

## 6. Success Criteria

| Criterion | Metric |
|-----------|--------|
| CRE hierarchy built and validated | 522 hubs, 400 leaves, 5 roots, no cycles, all leaves reachable |
| Label space deterministic | `label_space` is sorted, byte-identical across runs |
| Phase 0 parity | Integration test passes |
| Hub descriptions generated | 400 descriptions, all sanitized, all non-empty |
| Expert review started | Validation script reports status counts |
| Traditional frameworks extracted | 19 frameworks with correct control counts |
| all_controls.json complete | 19 extracted + available AI parsers, AI takes precedence |
| All tests passing | pytest, mypy --strict |
| Zero-width sanitization | Added to pipeline with tests |

---

## 7. Non-Goals (Explicitly Out of Scope)

- Hub collapsing or taxonomy modification (CRE owns the taxonomy)
- Model training (Phase 1B)
- Active learning loop (Phase 1B, PRD §6.7)
- Hub representation firewall implementation (Phase 1B, PRD §6.5)
- Embedding-quality signal for descriptions (Phase 1B)
- Synthetic data augmentation (Phase 1B)
- Web UI for expert review (Phase 2)
