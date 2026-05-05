# TRACT — Translating Requirements Across CRE Trees

## Product Requirements Document

**Date:** 2026-05-03 (last updated)
**Author:** Rock Lambros
**Status:** Active — Phases 0–1D + 2B + 3 + 5A complete, Phase 3B/4/5B in queue

---

## 1. Vision

TRACT is a platform that uses OpenCRE's 522 CRE hub ontology as a universal coordinate system for security frameworks. Any security control text maps to a position in the CRE ontology via a trained assignment model. Cross-framework relationships are derived from those positions — not predicted pairwise. TRACT extends OpenCRE by making it easy to add new frameworks and bridge AI security with traditional security standards. This is a community tool that gives back to OpenCRE.

## 2. Problem Statement

Security professionals maintain dozens of overlapping frameworks (NIST 800-53, ISO 27001, MITRE ATLAS, OWASP AI Exchange, etc.) with no systematic way to determine which controls across frameworks address the same concept. Manual crosswalking is O(n^2) in frameworks and O(m^2) in controls. OpenCRE already provides an expert-curated ontology linking 22 standards through 522 hub concepts — but no ML-assisted tool exists to automatically assign new controls to this ontology and derive cross-framework relationships.

## 3. Core Architecture: The Assignment Paradigm

**Function:** `g(control_text) -> CRE_position`

- ONE function, ONE input (control text), ONE output (position in CRE space)
- Cross-framework relationships derived transitively: if controls A and B both map to CRE hub X with high confidence, they are equivalent
- NOT pairwise: no f(A,B) -> relationship. The pairwise approach scales as O(n^2) and cannot generalize to unseen frameworks
- CRE hierarchy encodes relationship strength: same hub = equivalent, parent/child = related, disjoint = unrelated

**New repo: `tract/`.** Clean room. Independent of the COMP 4433 ai-security-framework-crosswalk project. That project is reference material only — not a dependency.

**Data sourcing: raw primary sources, not the old graph.** The old project's processed data (nodes.json, edges.json) was built by a 1,934-line build_graph.py that mixes authoritative, expert, unvalidated, and suggestive confidence levels. While the node data is generally accurate (verified: AIUC-1 189 nodes matches raw source exactly), the edge data has variable quality and the processing pipeline wasn't designed for CRE-based work. This project ingests ALL framework data fresh from primary sources:

| Source | What We Take | What We Ignore |
|--------|-------------|----------------|
| Official framework repos (MITRE ATLAS, NIST, OWASP, CoSAI, etc.) | Raw standard documents | Old project's nodes.json |
| OpenCRE API (opencre.org/rest/v1/all_cres) | Live CRE hub data with all links | Old project's opencre_pairs.jsonl |
| Old project's expert_train.jsonl | Validation signal only (compare our hub assignments against expert tiers) | Not used as training data |
| Old project's framework source files (data/frameworks/) | Reference for format/structure | We re-download from official sources |

This avoids inheriting processing bugs, confidence-level ambiguity, or structural assumptions from the old pipeline.

## 4. Data Foundation

### 4.1 CRE Hub Ontology (The Coordinate System)

| Metric | Count | Source |
|--------|-------|--------|
| Total CRE hubs | 522 | opencre_all_cres.json |
| Parent/structural hubs | 122 | Hubs with Contains links to children |
| Leaf hubs (primary label space) | 400 | Hubs without Contains links |
| Structural parents with zero standard links | 55 | Category nodes — have children but no direct framework mappings |
| True orphan hubs (no children, no standard links) | 4 | Anomalies to investigate |
| Hierarchy depth | 3 levels typical | e.g., "Technical AI security controls > Secure AI inference > Prompt injection I/O handling" |

**Missing:** Zero hub descriptions exist. Hub names ARE semantically rich ("Data poisoning of train/finetune/augmentation data") but lack formal definitions.

**Mitigation:** LLM-generated descriptions validated by domain expert. ~400 leaf hubs at ~5 min review each = ~33 hours expert review.

**The AI/Traditional divide:** 81 CRE hubs are AI-specific (linked to ATLAS, AI Exchange, NIST AI 100-2, LLM/ML Top 10). 441 are traditional security (CAPEC, CWE, NIST 800-53, ASVS, etc.). Zero hubs currently bridge AI to traditional. Bridge standards (ENISA, ETSI, BIML) appear on AI hubs but aren't core traditional standards. Bridging is a Phase 2 deliverable.

### 4.2 Framework Mapping Units

Each framework has a distinct internal hierarchy. The "mapping unit" is the level that carries enough semantic content for CRE hub assignment — not too abstract (domains/functions), not too granular (sub-activities without context).

| Framework | ID | Hierarchy | Mapping Unit | Count | Secondary Units | Total Nodes |
|-----------|-----|-----------|-------------|-------|----------------|-------------|
| AIUC-1 | aiuc_1 | domain -> control (requirement) -> activity | activity | 132 | 51 controls (requirements) | 189 |
| CSA AICM | csa_aicm | domain -> control | control | 243 | -- | 261 |
| MITRE ATLAS | mitre_atlas | tactic -> technique/sub-technique + mitigation | technique | 167 | 35 mitigations | 218 |
| NIST AI RMF | nist_rmf | function -> subcategory | subcategory | 72 | -- | 76 |
| OWASP AI Exchange | owasp_ai_exchange | flat controls + risks | control | 54 | 34 risks | 88 |
| CoSAI Risk Map | cosai_rm | domain -> control + risk | control | 29 | 26 risks | 61 |
| EU GPAI CoP | eu_gpai_cop | commitment -> requirement -> measure | measure | 32 | 28 requirements | 70 |
| OWASP LLM Top 10 | owasp_llm | flat | risk | 10 | -- | 10 |
| OWASP Agentic Top 10 | owasp_agentic | flat | risk | 10 | -- | 10 |
| **AI framework totals** | | | | **749** | **174** | **983** |

**AIUC-1 note:** The 51 "controls" are testable requirements (the WHAT). The 132 "activities" are implementation-level controls at technical, operational, and legal levels (the HOW). Activities carry the semantic detail needed for CRE hub assignment. The standard itself makes this distinction — treating AIUC-1 as "51 controls" undersells it.

**CSA AICM note:** 243 controls across 18 domains. Each control contains specification text, implementation guidelines, auditing guidelines, and CAIQ questions. Controls are self-contained mapping units with rich semantic content.

**Dual-dimension frameworks:** OWASP AI Exchange, CoSAI, and MITRE ATLAS have both defensive units (controls/mitigations) and threat units (risks/techniques). Both dimensions are mapped, but defensive controls are the primary alignment surface with CRE hubs.

**Discrepancy to verify:** User reports AIUC-1 has 187 activities; graph data shows 132. Verify against the official AIUC-1 standard during Phase 1 ingestion.

13 additional traditional security frameworks in OpenCRE (CAPEC, CWE, NIST 800-53, ASVS, ISO 27001, DSOMM, WSTG, OWASP Cheat Sheets, OWASP Proactive Controls, ENISA, ETSI, SAMM, Cloud Controls Matrix, etc.) will be characterized during Phase 1 ingestion.

### 4.3 Training Data: Standard-to-Hub Links

| Link Type | Count | Source | Quality |
|-----------|-------|--------|---------|
| Linked To (human) | 2,047 | Expert-curated by OpenCRE maintainers | Gold standard |
| Automatically Linked To | 2,359 | Deterministic transitive inference: CAPEC -> Related_Weaknesses -> CWE -> human_Linked_To -> CRE | Expert-transitive (MITRE taxonomy chain) |
| Contains / Is Part Of | 517 each | CRE internal hierarchy | Structural |
| Related | 398 | Inter-CRE relationships | Structural |
| **Total standard-to-hub links** | **4,406** | | |

**Auto-link provenance (critical):** AutomaticallyLinkedTo links are NOT ML-generated. They follow a deterministic path through MITRE's expert CWE/CAPEC taxonomy. Code path: `capec_parser.py:link_capec_to_cwe_cre()` lines 39-55 — for each CAPEC, find Related_Weaknesses (CWEs), for each CWE find human-linked CREs, create transitive link. OpenCRE treats these as equivalent quality to human links (penalty=0 in gap_analysis.py line 19). These links have been human-validated after generation.

**AI-specific links:** 198 total (all human LinkedTo). Distribution: MITRE ATLAS (65), OWASP AI Exchange (65), NIST AI 100-2 (45), OWASP LLM Top 10 (13), OWASP ML Top 10 (10).

**Training density:** 4,406 links / 400 leaf hubs = ~11 examples/hub average. Distribution is highly skewed — some hubs have 100+ links, others have 1-2. AI hubs average ~2.4 examples each (198 / 81).

**Frameworks linked to CRE hubs (by link count):** CAPEC (1,799), CWE (613), OWASP Cheat Sheets (391), NIST 800-53 (300), ASVS (277), DSOMM (214), WSTG (118), ISO 27001 (94), NIST 800-63 (79), OWASP Proactive Controls (76), ENISA (68), MITRE ATLAS (65), OWASP AI Exchange (65), NIST SSDF (46), NIST AI 100-2 (45), ETSI (36), SAMM (30), Cloud Controls Matrix (29), BIML (21), OWASP Top 10 2021 (17), OWASP Top10 for LLM (13), OWASP Top10 for ML (10).

### 4.4 Known Data Gaps and Planned Mitigations

| Gap | Impact | Mitigation |
|-----|--------|------------|
| Zero hub descriptions | Model lacks target label semantics | LLM-generate + expert validate (Phase 1 Task 1) |
| 198 AI-specific links (~2.4/hub) | Thin training for AI hub assignment | Transfer learning from 4,208 traditional links via contrastive fine-tuning + active learning loop |
| 5 of 9 AI frameworks have zero CRE coverage (AIUC-1, CSA AICM, CoSAI, EU GPAI CoP, OWASP Agentic) | Cannot train directly on these | Model-predicted + expert-reviewed hub assignments via active learning loop |
| AI and traditional CRE hubs are disconnected | No cross-domain bridges | Phase 2: bridge identification using ENISA/ETSI/BIML as seeds |
| Hub granularity varies | "Data poisoning" = 1 hub but 5 standards describe different aspects | CRE hierarchy paths as features; hub proposal system for fine-grained concepts |
| No zero-shot baseline exists | Cannot evaluate improvement over naive approaches | Phase 0 gate: characterize baselines before building |
| 5,920 expert pairs have no CRE labels | Cannot directly evaluate hub assignment against expert ordinal judgments | Derive: if two controls share a hub, predict equivalent; compare to expert tier labels |

### 4.5 Raw Data Inventory

Every framework exists in the old project at `ai-security-framework-crosswalk/data/frameworks/`. Their formats vary widely — some are production-grade structured JSON, others are raw markdown extracted from PDFs. This table captures the current state.

| Framework | Old Project Path | Format | Size | Prep Needed |
|-----------|-----------------|--------|------|-------------|
| **AIUC-1** | `data/frameworks/aiuc-1/aiuc-1-standard.json` | Structured JSON (domains → controls → activities) | 108 KB | **None** — extract mapping units directly |
| **CSA AICM** | `data/frameworks/csa-aicm/csa_aicm.json` | Structured JSON (18 domains, 243 controls, 14 fields each: specification, implementation guidelines, auditing guidelines, CAIQ questions) | 5.0 MB | **None** — fully enriched, production-ready |
| **MITRE ATLAS** | `data/frameworks/mitre-atlas/ATLAS_compiled.json` | Structured JSON (ATT&CK-style: tactics → techniques → sub-techniques + mitigations) | 483 KB | **None** — extract techniques and mitigations directly |
| **CoSAI** | `data/frameworks/cosai/` | Structured YAML + JSON Schemas (risks.yaml, controls.yaml, components.yaml + 10 validation schemas) | 852 KB | **YAML→JSON conversion** — parse YAML, validate against schemas, output JSON |
| **NIST 800-53** | `data/frameworks/nist-800-53/` | OSCAL JSON (NIST standard machine-readable format) | 10 MB | **OSCAL extraction** — parse OSCAL structure, extract controls with enhancements |
| **NIST AI RMF** | `data/frameworks/nist-ai-rmf/nist_ai_rmf_1.0.md` | Markdown (PDF extraction with frontmatter) | 103 KB | **Markdown parsing** — regex-extract subcategories by ID pattern |
| **NIST AI 600-1** | `data/frameworks/nist-ai-600-1/` | Markdown | 192 KB | **Markdown parsing** — extract sections with risk/control IDs |
| **OWASP AI Exchange** | `data/frameworks/owasp-ai-exchange/` | Modularized Markdown (8 files, 13,770 lines total from Hugo source) | 884 KB | **Markdown parsing** — parse control IDs, threat descriptions, mitigations from Hugo-format MD |
| **OWASP LLM Top 10** | `data/frameworks/owasp-llm-top10/owasp_llm_top_10_2025.md` | Markdown (PDF extraction) | 91 KB | **Markdown parsing** — extract 10 risk entries with descriptions and mitigations |
| **OWASP Agentic Top 10** | `data/frameworks/owasp-agentic-top10/owasp_agentic_top10_2026.md` | Markdown (PDF extraction, Dec 2025) | 100 KB | **Markdown parsing** — extract 10 risk entries with descriptions and mitigations |
| **OWASP DSGAI** | `data/frameworks/owasp-dsgai/` | PDF + TXT extraction + MANIFEST.json (21 risk IDs, pattern `DSGAI(0[1-9]\|1[0-9]\|2[01])`) | 1.4 MB | **TXT parsing** — use manifest regex to extract risk entries from pdftotext output |
| **EU GPAI CoP** | `data/frameworks/eu-gpai-code-of-practice/` | Markdown (4 files: combined + 3 topic splits) | 268 KB | **Markdown parsing** — extract commitments → requirements → measures hierarchy |
| **EU AI Act** | `data/frameworks/eu-ai-act/` | HTML (EUR-Lex) + manifest | 1.3 MB | **HTML extraction** — parse articles and annexes from EUR-Lex HTML |
| **OpenCRE** | `data/opencre/opencre_all_cres.json` | Structured JSON (paginated API dump, 522 hubs with all links) | 4.4 MB | **Re-fetch from API** — do NOT copy old cache; fetch fresh from opencre.org/rest/v1/all_cres |

**CSA Cloud Controls Matrix (CCM) vs. CSA AI Controls Matrix (AICM):** These are COMPLETELY DIFFERENT frameworks. CCM is a traditional cloud security standard (cloudsecurityalliance.org/research/cloud-controls-matrix) with 29 existing CRE links. AICM is an AI security standard (cloudsecurityalliance.org/artifacts/ai-controls-matrix) with 243 controls and zero CRE links. CCM appears in OpenCRE as "Cloud Controls Matrix." AICM is one of our 9 target AI frameworks. Do not conflate them.

### 4.6 Per-Framework Data Preparation

Each framework's raw source must be parsed into the TRACT standardized control schema (Section 4.8) BEFORE Phase 0 or Phase 1 can begin. Grouped by complexity:

**Tier 1: Copy as-is (already structured JSON)**

These frameworks have structured JSON with clear control hierarchies. Copy the source file into `tract/data/raw/`, then write a thin parser that extracts fields into the standardized schema.

| Framework | Parser Input | Mapping Unit Field | ID Field | Text Field |
|-----------|-------------|-------------------|----------|------------|
| CSA AICM | `csa_aicm.json` → `controls[]` | Each object in controls array | `id` | `specification` (primary) + `implementation_guidelines` (secondary) |
| AIUC-1 | `aiuc-1-standard.json` → `domains[].controls[].activities[]` | Each activity object | Construct from domain + control + activity index | `description` |
| MITRE ATLAS | `ATLAS_compiled.json` → `techniques[]` + `mitigations[]` | Each technique/mitigation | `id` (e.g., `AML.T0001`) | `description` |
| NIST 800-53 | OSCAL JSON → `catalog.groups[].controls[]` | Each control + enhancement | `id` (e.g., `AC-1`) | `title` + `parts[name=statement].prose` |

**Tier 2: YAML → JSON conversion**

| Framework | Steps |
|-----------|-------|
| **CoSAI** | 1. Parse `risks.yaml` and `controls.yaml` with PyYAML. 2. Validate each entry against the co-located JSON schemas. 3. Extract risk IDs, control IDs, descriptions, and component mappings. 4. Output one standardized JSON per dimension (controls, risks). |

**Tier 3: Markdown → JSON parsing**

These are the bulk of the prep work. Each markdown file was extracted from a PDF and has a consistent internal structure that can be regex-parsed.

| Framework | Parsing Strategy |
|-----------|-----------------|
| **NIST AI RMF** | Regex for subcategory IDs (pattern: `GOVERN X.Y`, `MAP X.Y`, `MEASURE X.Y`, `MANAGE X.Y`). Extract ID, title, and body text between consecutive IDs. Parent = function name (GOVERN, MAP, MEASURE, MANAGE). |
| **NIST AI 600-1** | Regex for section headers with risk/control numbering. Similar structure to AI RMF but different ID pattern. |
| **OWASP AI Exchange** | Parse Hugo-format markdown. Controls have structured headers (`## Control: ...`). Extract control ID, threat description, and mitigation text. Handle modular file structure (8 files map to different control categories). |
| **OWASP LLM Top 10** | Regex for `## LLM0[1-9]|LLM10` headers. Extract risk name, description, common examples, prevention strategies. 10 entries total. |
| **OWASP Agentic Top 10** | Same structure as LLM Top 10. Regex for numbered risk headers. Extract risk name, description, examples, mitigations. 10 entries total. |
| **EU GPAI CoP** | Parse commitment → requirement → measure hierarchy from markdown headers. Measures are the mapping units. Extract measure ID, parent requirement, parent commitment, and measure text. |

**Tier 4: PDF/TXT → JSON extraction**

| Framework | Steps |
|-----------|-------|
| **OWASP DSGAI** | 1. Read `MANIFEST.json` for ID pattern regex (`DSGAI(0[1-9]\|1[0-9]\|2[01])`). 2. Parse `.txt` file (pdftotext output). 3. Use manifest regex to locate 21 risk entries. 4. Extract risk ID, title, description, and mitigation text between consecutive IDs. |

**Tier 5: HTML → JSON extraction**

| Framework | Steps |
|-----------|-------|
| **EU AI Act** | 1. Parse EUR-Lex HTML with BeautifulSoup. 2. Extract articles (Article 1-113) and annexes (Annex I-XIII) as separate entries. 3. Each article/annex = one mapping unit with ID, title, and full text. |

**Tier 6: Fresh API fetch (do NOT copy old cache)**

| Source | Steps |
|--------|-------|
| **OpenCRE** | 1. Paginate through `opencre.org/rest/v1/all_cres?per_page=50&page=N` (1-indexed, ~261 pages). 2. Retry with exponential backoff on failures. 3. Save raw API responses to `tract/data/raw/opencre/opencre_all_cres.json`. 4. Extract: 522 hubs, 4,406 standard→hub links, 517 Contains links, 398 Related links. 5. Version-pin with fetch timestamp. |

**Traditional frameworks (13 in OpenCRE):** CAPEC, CWE, NIST 800-53, ASVS, ISO 27001, DSOMM, WSTG, OWASP Cheat Sheets, OWASP Proactive Controls, ENISA, ETSI, SAMM, Cloud Controls Matrix. These are ingested FROM the OpenCRE API data during Phase 1 (Section 6.1) — their control texts come from the CRE link metadata. No separate source files needed.

### 4.7 TRACT Repository Structure

```
tract/
├── data/
│   ├── raw/                              # Original sources — NEVER modified after initial fetch
│   │   ├── opencre/
│   │   │   └── opencre_all_cres.json     # Fresh API fetch (Section 4.6, Tier 6)
│   │   └── frameworks/
│   │       ├── aiuc_1/                   # Copy of aiuc-1-standard.json
│   │       ├── csa_aicm/                 # Copy of csa_aicm.json
│   │       ├── mitre_atlas/              # Copy of ATLAS_compiled.json
│   │       ├── nist_800_53/              # Copy of OSCAL JSON
│   │       ├── nist_ai_rmf/             # Copy of nist_ai_rmf_1.0.md
│   │       ├── nist_ai_600_1/           # Copy of markdown
│   │       ├── owasp_ai_exchange/        # Copy of 8 markdown files
│   │       ├── owasp_llm_top10/          # Copy of markdown
│   │       ├── owasp_agentic_top10/      # Copy of markdown
│   │       ├── owasp_dsgai/              # Copy of TXT + MANIFEST.json
│   │       ├── cosai/                    # Copy of YAML + schemas
│   │       ├── eu_gpai_cop/              # Copy of markdown files
│   │       └── eu_ai_act/                # Copy of HTML + manifest
│   ├── processed/
│   │   ├── cre_hierarchy.json            # Phase 1, Task 6.1 output
│   │   ├── hub_descriptions.json         # Phase 1, Task 6.2 output
│   │   ├── frameworks/                   # One standardized JSON per framework (Section 4.8 schema)
│   │   │   ├── aiuc_1.json
│   │   │   ├── csa_aicm.json
│   │   │   ├── mitre_atlas.json
│   │   │   └── ...                       # All 22 frameworks
│   │   └── all_controls.json             # Unified file — all frameworks merged
│   └── training/
│       ├── hub_links.jsonl               # All 4,406 standard→hub links from OpenCRE
│       └── hub_links_by_framework.json   # Grouped by framework (for LOFO splits)
├── parsers/                              # One parser script per framework (Tier 1-5)
│   ├── parse_csa_aicm.py
│   ├── parse_aiuc_1.py
│   ├── parse_mitre_atlas.py
│   ├── parse_nist_800_53.py
│   ├── parse_cosai.py
│   ├── parse_nist_ai_rmf.py
│   ├── parse_owasp_ai_exchange.py
│   ├── parse_owasp_llm_top10.py
│   ├── parse_owasp_agentic_top10.py
│   ├── parse_owasp_dsgai.py
│   ├── parse_eu_gpai_cop.py
│   ├── parse_eu_ai_act.py
│   ├── fetch_opencre.py                  # API fetcher with retry/backoff
│   └── extract_hub_links.py              # Extracts training links from opencre_all_cres.json
├── models/                               # Trained model checkpoints
├── scripts/                              # CLI entry points
├── crosswalk.db                          # SQLite crosswalk database (Phase 1 output)
├── hub_proposals/                        # Versioned proposal rounds
└── tests/
```

**Storage rules:**
- `data/raw/` is immutable after initial population. Source files are copied/fetched once, then never touched. If a framework releases a new version, create a versioned subdirectory (e.g., `aiuc_1/v1.0/`, `aiuc_1/v2.0/`).
- `data/processed/` is regenerated from `data/raw/` by the parsers. Deleting and re-running parsers must produce identical output (deterministic).
- `data/training/` is generated from `data/processed/` during Phase 1. Contains only derived data, no raw content.
- `parsers/` scripts read from `data/raw/` and write to `data/processed/`. Each parser is standalone — no inter-parser dependencies.
- OpenCRE data is ALWAYS fetched fresh from the API, not copied from the old project. The old project's `opencre_all_cres.json` may be stale.

### 4.8 Standardized Control Schema

Every parser must output a JSON file conforming to this schema. This is the contract between data prep and all downstream Phase 1 tasks.

```json
{
  "framework_id": "csa_aicm",
  "framework_name": "CSA AI Controls Matrix",
  "version": "1.0",
  "source_url": "https://cloudsecurityalliance.org/artifacts/ai-controls-matrix",
  "fetched_date": "2026-04-27",
  "mapping_unit_level": "control",
  "controls": [
    {
      "control_id": "AICM-AIS-01",
      "title": "AI System Inventory",
      "description": "Organizations shall maintain a comprehensive inventory...",
      "full_text": "Organizations shall maintain a comprehensive inventory of all AI systems...",
      "hierarchy_level": "control",
      "parent_id": "AIS",
      "parent_name": "AI System Lifecycle Security",
      "metadata": {
        "control_type": "preventive",
        "domain": "AI System Lifecycle Security"
      }
    }
  ]
}
```

**Required fields:** `framework_id`, `framework_name`, `version`, `source_url`, `fetched_date`, `mapping_unit_level`, `controls[].control_id`, `controls[].title`, `controls[].description`.

**Optional fields:** `controls[].full_text` (when description is a summary and full text is longer), `controls[].hierarchy_level`, `controls[].parent_id`, `controls[].parent_name`, `controls[].metadata` (framework-specific fields).

**Text field rules:**
- `description` is the primary text used for CRE hub assignment — it must be the semantically richest representation of the control.
- For CSA AICM: `description` = `specification` field (the core requirement text).
- For MITRE ATLAS: `description` = `description` field from the technique/mitigation.
- For AIUC-1: `description` = activity description (not the parent control's requirement text).
- Strip HTML tags, normalize Unicode to ASCII where possible, remove boilerplate headers/footers.
- Maximum `description` length: 2000 characters. If longer, truncate with `full_text` preserving the complete version.

---

## 5. Phase 0: Zero-Shot Baseline Gate (1-2 days)

**Purpose:** Establish baselines BEFORE committing to model training. If zero-shot approaches already achieve acceptable performance, the trained model may not be needed. This is a go/no-go gate.

**Experiments:**

1. **Embedding similarity baseline:** Encode hub names (+ names of linked standards) and control texts with a pre-trained encoder (BGE-large-v1.5). Compute cosine similarity. Measure hit@1, hit@5, MRR for hub assignment on the 198 AI links using leave-one-framework-out.

2. **LLM probe:** Ask Claude/GPT-4 to assign CRE hubs to held-out controls given a list of hub names + hierarchy paths. Measure the same metrics. This establishes the ceiling for "how well can a general model do this task without fine-tuning?"

3. **Hierarchy-path features:** Test whether encoding full CRE hierarchy paths (e.g., "Technical AI security controls > Secure AI inference > Prompt injection I/O handling") as additional hub features improves retrieval over hub names alone.

4. **LLM hub description pilot:** Generate descriptions for 50 representative hubs. Measure whether adding descriptions to hub representations improves embedding similarity hit rates.

**Gate criteria:** Proceed to Phase 1 only if:
- (a) LLM probe hit@5 > 0.50 on held-out frameworks (the task is feasible), AND
- (b) hit@1 gap between LLM and embedding > 0.10 (room for a trained model to improve)

If both fail, revisit the architecture before investing further.

---

## 6. Phase 1: Core Model + Crosswalk Database + CLI

**Scope:** All 22 OpenCRE frameworks. Prove the assignment paradigm works. Ship a usable tool.

### 6.1 Build the CRE Hierarchy (The Coordinate System)
This is foundational — everything else depends on it.
- Fetch ALL CRE data fresh from OpenCRE API (opencre.org/rest/v1/all_cres), not from the old project's cache
- Parse the 522 hubs, 517 Contains/Is Part Of relationships, 398 Related links
- Build the full hierarchy tree: root hubs -> parent hubs -> leaf hubs (3 levels typical)
- Compute full hierarchy paths for each hub (e.g., "Technical AI security controls > Secure AI inference > Prompt injection I/O handling")
- Identify and tag: 122 parent hubs, 400 leaf hubs (the label space), 4 true orphans
- Store as a versioned, queryable data structure (not just a flat JSON dump)
- Validate: every leaf hub reachable from a root; no cycles; hierarchy depth matches expectations
- **Deliverable:** `cre_hierarchy.json` — the complete coordinate system, version-pinned

### 6.2 LLM Hub Description Generation
The CRE hubs have names but zero descriptions. We fix that here.
- Generate descriptions for ALL 400 leaf hubs using an LLM (Claude or GPT-4)
- **Input per hub:** hub name, full hierarchy path from 6.1, names of all linked standard sections (from the 4,406 links)
- **Output per hub:** 2-3 sentence description defining: (a) what the hub covers, (b) what distinguishes it from sibling hubs, (c) the boundary of its scope
- **Expert validation workflow:** Rock reviews each description. Accept / edit / reject. Edited descriptions are the gold standard.
- **Estimated effort:** ~33 hours for 400 hubs (~5 min each). Can be batched in sessions of 50.
- **Deliverable:** `hub_descriptions.json` — every leaf hub with a validated semantic description
- **Phase 0 finding:** LLM-generated descriptions hurt zero-shot embedding performance (BGE -0.019, GTE -0.147). However, descriptions may help a trained model that learns to use them — this is an experiment in Phase 1B. Descriptions are also valuable for CLI/web display and expert review regardless of model impact.

### 6.3 Framework Ingestion (All 22 — From Primary Sources)
Every framework re-ingested from its official source. NOT from the old project's nodes.json.
- Download/access each framework's official standard document
- Parse and identify the correct mapping unit per framework (see Section 4.2 for the 9 AI frameworks; traditional frameworks characterized during this task)
- Extract: control_id, title, description/specification text, hierarchy level, parent
- Normalize text (consistent encoding, no HTML artifacts, standard field names)
- Store with full provenance: source URL, document version, date accessed, page/section reference
- **Deliverable:** Per-framework JSON files + unified `all_controls.json` with 22 frameworks

**Primary sources for AI frameworks:**
| Framework | Official Source |
|-----------|---------------|
| AIUC-1 | AIUC standard document (JSON) |
| CSA AICM | CSA website / structured download |
| MITRE ATLAS | github.com/mitre-atlas/atlas-data |
| NIST AI RMF | NIST AI RMF 1.0 playbook |
| OWASP AI Exchange | owaspai.org Hugo source repo |
| CoSAI | cosai.dev YAML data |
| EU GPAI CoP | EU Trustworthy AI documentation |
| OWASP LLM Top 10 | OWASP official 2025 PDF/markdown |
| OWASP Agentic Top 10 | OWASP official 2026 PDF/markdown |

### 6.4 CRE Hub Assignment Model
- **Architecture:** Contrastive fine-tuning of BGE-large-v1.5 (335M params) — the best-performing zero-shot embedding model from Phase 0 (hit@1=0.348 baseline, 0.424 with hierarchy paths). Produces a bi-encoder that maps control text and hub representations into a shared embedding space. *(Phase 0 showed DeBERTa-v3-NLI completely fails for this task — hit@1=0.000 — confirming semantic similarity, not entailment, is the right approach.)*
- **Hub representation:** Hub name + hierarchy path (from 6.1), encoded as target embeddings. *(Phase 0 proved hierarchy paths help: +7.6% hit@1 for BGE, significant.)* Whether to include expert-reviewed descriptions (from 6.2) is an ablation experiment — Phase 0 showed descriptions hurt zero-shot embeddings, but a trained model may learn to use them.
- **Training data:** 4,406 standard-to-hub links (2,047 human + 2,359 expert-transitive), extracted fresh from OpenCRE API data. ~35% of control texts map to multiple CRE hubs — these are valid multi-neighborhood relationships in the CRE graph, not noise (see 6.4.1).
- **Training strategy:** Contrastive learning with MNRL (MultipleNegativesRankingLoss) and mined hard negatives (sibling hubs in the CRE tree). Text-aware batch sampling prevents MNRL false negatives from same-hub and same-text collisions (see 6.4.1). Whether to train on all 4,406 links vs. AI-specific 198 links vs. a two-stage transfer approach is an ablation experiment, not a prescribed architecture.
- **Output:** Cosine similarity scores over 400 leaf hubs, calibrated to probabilities via temperature/Platt scaling
- **Multi-label handling:** Median 1 hub/section but max 38; model returns ranked hub list with calibrated similarity scores. Per-hub similarity thresholds tuned on validation set.
- **Gate 1 criteria:** Micro-averaged (sample-weighted) hit@1 delta > 0.10 over zero-shot baseline. Gate metrics are pre-registered — post-hoc metric substitution is not permitted for gate decisions. Report per-fold delta, macro average, and worst-fold delta alongside micro as diagnostics. If any fold shows delta < 0 vs its own zero-shot baseline, that framework is flagged for investigation before deployment.
- **Phase 1B Gate 1 result:** PASS. Micro hit@1=0.531 (delta=+0.132 over zero-shot baseline 0.399). Per-fold deltas vs zero-shot: NIST +0.322, ML-10 +0.285, OWASP-X +0.143, ATLAS +0.006, LLM-10 +0.000. All folds non-negative. NIST fold determinism rerun with CUDA flags confirmed hit@1=0.429 (exact match, no variance). Training density varies across folds but no fold regresses vs its own zero-shot baseline.

#### 6.4.1 Multi-Hub Training Pairs and Batch Sampling
Controls legitimately map to multiple CRE hubs (multi-hop graph structure). These multi-hub text mappings are preserved as separate training pairs — never dropped or deduplicated across hubs. Deduplication occurs only within the same (text, hub) pair (case-insensitive), keeping the best quality tier.

MNRL uses in-batch examples as negatives, creating two collision risks:
1. **Hub collision:** Two examples targeting the same hub in a batch → false negative (pushing apart things that should be close)
2. **Text collision:** Same anchor text mapped to different hubs in a batch → contradictory gradients

The `HubAwareTemperatureSampler` prevents both collision types at the batching layer while preserving all training pairs. This is the correct approach — handle multi-label at the loss/batching layer, not by discarding data.

#### 6.4.2 Training Reproducibility
Training requires CUDA determinism flags for reproducible results: `torch.backends.cudnn.deterministic=True`, `torch.backends.cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Confirmed: NIST fold rerun with determinism flags produces identical metrics (hit@1=0.429, MRR=0.516, NDCG@10=0.557) — training is fully reproducible on H100 with fp16.

### 6.5 Hub Representation Firewall
When evaluating hub assignment for framework X, rebuild hub representations WITHOUT contributions from X's own linked sections. This means:
- Remove X's section names from the hub's linked-standards list
- Regenerate the hub's embedding without X's influence
- This is a HARD requirement for honest leave-one-framework-out evaluation
- Implemented as a build step, not a runtime hack

### 6.6 Guardrail Implementation ✅ COMPLETE
Build all 5 guardrail categories as concrete, testable components:

**Data Integrity guardrails (built during ingestion):**
- Framework-level train/test split logic (hold out entire frameworks, not random controls)
- Mapping-level dedup (detect same control appearing via different CRE paths)
- Auto-link provenance chain stored per link (CAPEC->CWE->CRE derivation path)
- Source version pinning (each framework tagged with document version + fetch date)

**Model Integrity guardrails (built with the model):**
- Hub representation firewall (6.5 above)
- Confidence calibration module (temperature/Platt scaling on validation set)
- Per-hub similarity threshold tuner (optimize on validation set, freeze before test evaluation)
- Transfer learning A/B test (compare model with vs. without traditional pre-training)

**Output Integrity guardrails (built with the inference pipeline):**
- Confidence threshold filter (low-confidence predictions flagged, not committed)
- Conformal prediction wrapper (guarantee coverage >= 90%)
- Multi-hub disagreement detector (flag near-equal competing hub predictions)

**Adversarial Robustness guardrails (built as test suite):**
- Paraphrase probe test set (same controls reworded, should get same hub assignments)
- OOD detector (max hub confidence below threshold -> flag as "no good hub match")
- OOD -> hub proposal pipeline trigger

**Provenance Tracking guardrails (built into every component):**
- Prediction log schema (control text, hub(s), scores, model version, timestamp)
- Human review tracking (accept/reject/correct linked to prediction ID)
- Training data lineage per example (source: OpenCRE link type + link ID)
- Model version tag (training data hash + hyperparameters)

### 6.7 Active Learning Loop ✅ COMPLETE (2 rounds, converged)
For frameworks with zero CRE coverage (AIUC-1, CSA AICM, CoSAI, EU GPAI CoP, OWASP Agentic) and any thin-coverage traditional frameworks:
1. Model predicts top-K hub assignments for each control
2. Expert reviews predictions (accept / reject / correct) via a review interface (can be CLI-based in Phase 1)
3. Accepted predictions added to training data with provenance="active_learning_round_N"
4. Model retrained on expanded dataset
5. Repeat until expert acceptance rate stabilizes (target: >80% accept rate)

### 6.8 Crosswalk Database ✅ COMPLETE
- SQLite database for local use, exportable to JSON/CSV
- Each control assigned to CRE hub(s) with confidence scores and provenance
- Cross-framework relationship matrix derived transitively from shared hub assignments
- Every assignment traced to: model version, training data version, expert review status
- **Deliverable:** `crosswalk.db` + export scripts

### 6.9 CLI Tool ✅ COMPLETE (11 subcommands, 553 tests)
```
tract assign "Implement rate limiting for API endpoints"
  -> CRE-236 (API security, 0.89), CRE-441 (Rate limiting, 0.72)

tract compare --framework atlas --framework asvs
  -> relationship matrix with confidence scores

tract prepare --file framework.csv --framework-id my_fw --name "My Framework"
  -> extracts controls from CSV/Markdown/JSON/unstructured → standardized FrameworkOutput JSON
  -> flexible column mapping, heading-level auto-detection, LLM fallback (--use-llm)

tract validate --file prepared_framework.json [--json]
  -> checks FrameworkOutput against 6 error rules + 11 warning rules
  -> schema conformance, duplicate IDs, short descriptions, non-NFC unicode, etc.

tract ingest --file new_framework.json --template standard
  -> runs validation gate, then predicted hub assignments for review
  -> appends calibration disclaimer + quality summary (mean_max_cosine_sim, ood_fraction)

tract accept --review-file review.json [--force]
  -> commits reviewed predictions to crosswalk.db

tract export --format csv --framework atlas
  -> full crosswalk table

tract export --opencre [--framework FW] [--skip-staleness] [--dry-run]
  -> per-framework OpenCRE-compatible CSVs + export_manifest.json + coverage_gaps.json

tract hierarchy --hub CRE-236
  -> full path: Root > Network Security > API Security > CRE-236
  -> linked controls from 5 frameworks

tract propose-hubs --min-controls 3 --min-frameworks 2
  -> HDBSCAN clustering of OOD controls → guardrailed hub proposals

tract review-proposals --round 1
  -> interactive CLI review of proposed hubs

tract tutorial
  -> guided walkthrough of TRACT features
```

#### Framework Preparation Pipeline (PR #21)
The `prepare` → `validate` → `ingest` pipeline enables onboarding new frameworks from arbitrary file formats:

| Extractor | Input | Key Features |
|-----------|-------|-------------|
| CSV | `.csv`, `.tsv` | Flexible column aliases (control_id/id/section_id, etc.), BOM handling (utf-8-sig) |
| Markdown | `.md` | 4 ID patterns, heading-level auto-detection, positional fallback IDs |
| JSON | `.json` | FrameworkOutput passthrough, top-level arrays, nested controls/items/data keys |
| LLM | any | Claude API tool_use extraction via `--use-llm` (requires `pip install tract[llm]`) |

Validation rules (6 errors + 11 warnings): schema_conformance, empty_description, duplicate_control_id, invalid_framework_id, null_bytes, zero_controls, short_description, long_description_no_full_text, problematic_control_id_chars, low_control_count, high_control_count, missing_optional_field, non_nfc_unicode, reference_only_description, title_description_redundancy, non_english_text.

### 6.10 Guardrailed Hub Proposal System ✅ COMPLETE
When the OOD detector (from 6.6 adversarial guardrails) flags controls that don't map to any existing hub, they feed into this system.

**Built as a Phase 1 CLI command:** `tract propose-hubs --min-controls 3 --min-frameworks 2`

**Components built:**
- **HDBSCAN clustering module:** Takes OOD control embeddings, clusters them. Deterministic (same inputs = same clusters).
- **6-guardrail filter pipeline:** Each proposed cluster passes through:
  1. Minimum evidence check (3+ controls from 2+ frameworks)
  2. Hierarchy constraint (must identify parent hub; no top-level creation)
  3. Similarity ceiling (cosine < 0.85 to all existing hubs)
  4. Budget cap (~40 proposals max per review cycle)
  5. Candidate queue writer (stores proposals with evidence for human review)
  6. Deterministic reproducibility check (re-run produces same clusters)
- **Review interface:** CLI-based for Phase 1 (`tract review-proposals`). Shows proposed hub name, parent hub, contributing controls, similarity scores. Expert accepts/rejects/edits.
- **Hierarchy update:** Accepted proposals inserted into `cre_hierarchy.json` as children of their parent hub. Model retrained with expanded label space.
- **Deliverable:** `hub_proposals/` directory with versioned proposal rounds, acceptance records, and hierarchy diffs

### 6.11 Evaluation Strategy
**Leave-one-framework-out cross-validation (LOFO):**
- For each of the 5 CRE-mapped AI frameworks: hold out entirely, train on remaining 4, predict hub assignments for held-out framework
- **Primary metrics (hub assignment quality):** hit@1, hit@5, MRR, NDCG@10 — matching Phase 0 metrics for direct comparison. Bootstrap CIs (10,000 resamples) and paired deltas for all model comparisons.
- **Metric reporting:** Primary gate metric is micro-averaged (sample-weighted) hit@1 delta. Additionally report: per-fold hit@1 with per-fold delta vs zero-shot baseline, macro average across folds, and worst-fold delta. Per-fold deltas surface framework-specific regression that aggregate metrics can mask.
- Compare against all Phase 0 baselines — trained model MUST exceed them
- No pairwise metrics as success criteria. The assignment paradigm is evaluated on assignment quality, period.

**LOFO training data design:** Traditional framework links (CWE, CAPEC, NIST 800-53, ASVS, etc.) remain in training for ALL folds by design. They provide the semantic bridge — the model learns hub neighborhoods from traditional frameworks and must generalize to unseen AI-framework text. Only the held-out AI framework's text and hub-representation contributions are removed (via the hub firewall, Section 6.5). When a traditional framework control maps to the same hub as a held-out eval item, this is the mechanism working as designed, not information leakage.

---

## 7. Phase 2: HuggingFace Publication + AI/Traditional Bridge

### 7.1 Framework Submission System ✅ COMPLETE (CLI pipeline)
**Deliverable:** CLI-based framework onboarding pipeline.

**Built (Phase 1):** `tract prepare` (multi-format extraction → FrameworkOutput JSON), `tract validate` (6 error + 11 warning rules), `tract ingest` (validation gate + model inference + calibration disclaimer), `tract accept` (commit reviewed predictions to DB). The full CLI pipeline for framework onboarding is operational.

**Future enhancements (not planned):**
- Duplicate detection: cosine similarity > 0.95 to existing controls flagged during validation (requires model embeddings)
- `framework_registry.json`: versioned list of all ingested frameworks with metadata, changelog, ingestion date

### 7.2 HuggingFace Model Publication ✅ COMPLETE
**Deliverable:** Published model repo at huggingface.co/rockCO78/tract-cre-assignment with:
- Merged model weights (~1.34GB, LoRA adapters merged into base BGE-large-v1.5 with cosine verification >0.9999)
- `hub_descriptions.json` and `cre_hierarchy.json` (v1.1 with bridge links) bundled with model
- Comprehensive model card: plain-English "What Is This?" section, architecture diagram, LOFO evaluation walkthrough, calibration explanation, bridge analysis summary, 3 usage examples, 10-term glossary — serves both novices and experts
- `predict.py`: standalone inference script — takes control text, returns hub assignments with calibrated confidence
- `train.py`: reproduction script with requirements.txt and data download instructions

### 7.3 AI/Traditional Security Bridge ✅ COMPLETE
**Deliverable:** Extended `cre_hierarchy.json` with bridge hubs + bridge validation report.

**Implementation (PR #22, 654 tests):**
1. Classify 522 CRE hubs by framework type → AI-only, traditional-only, both, unlinked
2. Compute full cosine similarity matrix between AI-only and traditional-only hub embeddings (dot product — all embeddings unit-normalized)
3. Extract top-3 traditional matches per AI hub (63 candidates total). No threshold — rank-based selection
4. LLM-generate bridge descriptions explaining the conceptual overlap (Claude, T=0.0). Also generate negative control descriptions (bottom-1 per AI hub)
5. Expert review: accept/reject each candidate via `tract bridge --commit --candidates <path>`
6. Accepted bridges become bidirectional `related_hub_ids` in `cre_hierarchy.json` (version bumped to "1.1")
7. Publication gate enforces bridge completion before HuggingFace upload (no pending candidates, hierarchy updated)
8. **Deliverables:** `bridge_candidates.json` (analysis output), `bridge_report.json` (after review), updated `cre_hierarchy.json`

**Execution results:**
- 5-round adversarial review: security architecture → methodology → implementation → cross-attack → convergence
- 46/63 candidates accepted, 17 rejected (13 below p99 threshold of 0.45, 4 specious MFA/OTP bridges)
- 51 hubs now have `related_hub_ids`, 92 total bidirectional edges in hierarchy v1.1
- Mann-Whitney U test confirms AI-only top-3 similarities significantly higher than naturally-bridged top-3 (p=0.0098)

---

## 8. Phase 3: Published Human-Reviewed Crosswalk Dataset ✅ COMPLETE

**Deliverable:** Versioned dataset published to HuggingFace Datasets AND Zenodo.

**Completed 2026-05-03.** PR #23 merged. 19 commits, 6,715 lines, 831 tests (278 new).

**Published to:** huggingface.co/datasets/rockCO78/tract-crosswalk-dataset

**Pipeline results:**
- 4,331 ground truth links imported from OpenCRE (57 duplicates skipped, 18 unresolvable OWASP AI Exchange controls)
- 320 model predictions generated for 5 uncovered AI frameworks via BGE-large-v1.5 inference
- 898 predictions exported for review (878 real + 20 hidden calibration items)
- Expert review: 680 accepted (77.4%), 196 reassigned (22.3%), 2 rejected (0.2%)
- Calibration quality: 65% agreement (13/20 calibration items matched ground truth)
- 5,238 deduplicated assignments published across 31 frameworks

**Review workflow (concrete):**
1. Export all hub assignments from `crosswalk.db` (636 model predictions across 6 AI frameworks) plus 4,406 ground-truth links from OpenCRE (22 traditional frameworks)
2. Group by framework. For each framework, generate a review spreadsheet (CSV or web interface): control_id, control_text, predicted_hub_1 (confidence), predicted_hub_2 (confidence), ..., reviewer_decision, reviewer_notes
3. Expert reviews one framework at a time. Per control: accept top prediction, select a different hub, flag for discussion, or mark "no good hub" (feeds into hub proposal system)
4. Track: total controls reviewed, acceptance rate, edit rate, rejection rate, avg time per control
5. Second-pass review for rejected/edited controls: check if edits are consistent across similar controls
6. Generate inter-reviewer agreement metrics (if multiple reviewers): Cohen's kappa or Krippendorff's alpha
7. Freeze reviewed assignments as `crosswalk_reviewed_v1.0.jsonl`

**Published dataset structure:**
```
tract-crosswalk-dataset/
  crosswalk_v1.0.jsonl          # 5,238 control-hub assignments (deduplicated)
  framework_metadata.json       # 31 framework descriptions, versions, sources
  cre_hierarchy_v1.1.json       # Hub ontology with bridge links at time of publication
  hub_descriptions_v1.0.json    # Validated hub descriptions
  review_metrics.json           # Acceptance rates, calibration quality, per-framework breakdown
  README.md                     # 16-section dataset card (novice + expert audience)
  bridge_report.json            # Bridge analysis evidence and review decisions
  zenodo_metadata.json          # Metadata for Zenodo DOI registration
  LICENSE                       # CC-BY-SA-4.0
```

**New CLI commands (5):**
- `tract import-ground-truth` — import OpenCRE ground truth into crosswalk.db
- `tract review-export` — generate review JSON with re-inference and calibration items
- `tract review-validate` — validate reviewed JSON before import
- `tract review-import` — apply expert review decisions (UPDATE-in-place)
- `tract publish-dataset` — bundle and upload to HuggingFace Hub

**Key implementation details:**
- Schema migration: `ALTER TABLE assignments ADD COLUMN reviewer_notes TEXT; ADD COLUMN original_hub_id TEXT`
- Re-inference at export: loads TRACTPredictor, runs predict_batch() for fresh calibrated confidence values
- Calibration items: 20 hidden GT items (negative IDs -1 to -20), stratified: 5 easy + 5 hard + 10 middle (seed=42)
- Provenance-priority dedup: opencre_ground_truth > ground_truth_T1-AI > active_learning_round_2 > model_prediction
- GT-confirmed exclusion: model predictions where ground truth already confirms the same (control_id, hub_id) pair are excluded from review

**Contribute back to OpenCRE:** For the 5 AI frameworks with zero CRE coverage (AIUC-1, CSA AICM, CoSAI, EU GPAI CoP, OWASP Agentic), submit validated hub assignments as proposed LinkedTo links to the OpenCRE project.

---

## 8B. Phase 3B: Experimental Narrative Notebook

**Deliverable:** A publication-quality Jupyter notebook (`notebooks/tract_experimental_narrative.ipynb`) that tells the complete story of TRACT's model development — from zero-shot baselines through base model selection, contrastive fine-tuning, ablation analysis, and final evaluation. This is not a code dump; it is a narrative document that a reader with no prior context can follow from start to finish.

**Style reference:** `ai-security-framework-crosswalk/project1/COMP_4433_RockLambros_project1_crosswalk_eda.ipynb` (128 cells, 82 markdown / 46 code, 24 figures). Match or exceed that notebook's standards:

- **Narrative-first structure.** More markdown than code (≥1.5:1 ratio). Every code cell has dense inline comments explaining what AND why. Every figure has a paragraph of interpretation before it (what to look for) and after it (what it shows).
- **"Plain English" callouts.** After every technical section, a `> **Plain English:**` blockquote explains what just happened in language accessible to a security professional who doesn't know ML.
- **Numbered figures with titles.** Every visualization is named (e.g., "Figure 4.2: Per-Framework hit@1 Across Base Models"). Figures are referenced by number in the narrative text.
- **Story arc.** The notebook follows a problem → exploration → failure → insight → solution arc. Each section builds on the previous. Dead ends and failures are documented honestly — they teach readers (and future contributors) what doesn't work and why.

**Narrative structure (tentative — adapts to actual results):**

| Section | Content | Key Figures |
|---------|---------|-------------|
| 1. Introduction & Motivation | The crosswalk problem, CRE as coordinate system, assignment paradigm | CRE hierarchy tree visualization, framework coverage heatmap |
| 2. Data Landscape | 4,406 training links, 31 frameworks, 2,802 controls, hub distribution | Link distribution by hub (long tail), multi-label frequency, framework size comparison |
| 3. Phase 0 Baselines | Zero-shot embedding + Opus results, what worked and what failed | Radar chart of per-fold metrics, BGE vs GTE vs DeBERTa comparison, hierarchy path impact |
| 4. Base Model Selection | 6 embedding models compared, per-fold complementarity analysis | Model comparison heatmap, per-framework performance matrix, embedding space UMAP projections |
| 5. Contrastive Fine-Tuning | Training approach, loss curves, hard negative strategy | Training loss curves, learning rate schedules, negative sampling analysis |
| 6. Ablation Analysis | Data scope, hub representation, hard negatives — what mattered | Ablation matrix heatmap, paired delta forest plots with CIs, interaction effects |
| 7. The Hub Firewall | How LOFO evaluation works, what happens without it (information leakage demo) | With-firewall vs without-firewall comparison, per-hub leakage magnitude |
| 8. Final Results | Best model vs all baselines vs Opus, per-framework deep dive | Bootstrap CI comparison chart, confusion analysis (which hubs are hardest), per-framework waterfall |
| 9. Error Analysis | Where the model fails and why, framework difficulty analysis | t-SNE/UMAP of misclassified controls, similarity distribution of errors vs correct predictions |
| 10. Calibration | Temperature scaling, reliability diagrams, ECE | Reliability diagram (before/after calibration), confidence histogram |
| 11. Human Review & Dataset Publication | Phase 3 review results, acceptance rates by framework, calibration quality | Per-framework acceptance rate bar chart, calibration agreement confusion matrix |
| 12. Using TRACT (CLI Tutorial) | Hands-on walkthrough of all 18 CLI commands across 3 workflows: single-control assignment, new framework onboarding, crosswalk exploration | Shell cells with real `!tract ...` output, annotated step-by-step |
| 13. Conclusion & Next Steps | Summary, limitations, future work | Summary metrics table, roadmap visualization |
| Appendix A | Experiment log — every run with hyperparameters and metrics | Full results table linked to WandB |
| Appendix B | Visual style guide — palette definitions, design principles with citations | Palette swatches, accessibility notes |

**Visualization standards:**

- **Innovative approaches where they earn their complexity.** 3D renderings of embedding spaces (Plotly 3D scatter with framework-colored points). Animated training progression (epoch-by-epoch embedding space evolution as HTML widget). Interactive UMAP projections where hovering reveals control text and predicted hub. Sankey diagrams showing the flow from frameworks → hubs.
- **Static fallbacks for every interactive element.** The notebook must render meaningfully as a static PDF/HTML export. Interactive widgets degrade to static snapshots.
- **Colorblind-accessible palettes.** Okabe-Ito for categorical data, single-hue sequential ramps for counts, diverging palettes centered at meaningful zero points. No rainbow or jet colormaps.
- **Publication-quality typography.** Consistent font sizes, axis labels on every plot, no default matplotlib titles. LaTeX-rendered equations where formulas appear.
- **Dark/light theme compatibility.** Use Plotly themes that work on both light and dark backgrounds.

**Technical requirements:**
- All data loaded from TRACT's canonical paths (`data/processed/`, `data/training/`, WandB API)
- Reproducible: running all cells from top to bottom produces identical output (seeded randomness, sorted keys)
- Dependencies: plotly, matplotlib, seaborn, umap-learn, scikit-learn (for t-SNE/UMAP), ipywidgets (for animations)
- Cell execution time: full notebook runs in < 10 minutes (pre-computed embeddings loaded, not re-computed)

---

## 9. Phase 4: Secure API with Full Documentation

**Deliverable:** Deployed API server + OpenAPI spec + Python SDK + Docker image.

**API endpoints:**
| Method | Endpoint | Input | Output |
|--------|----------|-------|--------|
| POST | /v1/assign | `{"control_text": "..."}` | `{"hubs": [{"cre_id": "CRE-236", "name": "...", "confidence": 0.89}, ...]}` |
| GET | /v1/hub/{cre_id} | CRE hub ID | Hub description, hierarchy path, linked controls from all frameworks |
| GET | /v1/compare?fw1=atlas&fw2=asvs | Two framework IDs | Crosswalk matrix: shared hubs, related hubs, gaps |
| GET | /v1/framework/{fw_id} | Framework ID | All controls with hub assignments and confidence |
| POST | /v1/ingest | Framework JSON (template format) | Validation result + predicted hub assignments |
| GET | /v1/hierarchy | -- | Full CRE hierarchy tree |
| GET | /v1/search?q=rate+limiting | Search query | Controls matching query with their hub assignments |

**Security:**
- API key authentication (issued per user/org)
- Rate limiting: 100 requests/minute for /assign (model inference), 1000/minute for reads
- Input validation: max control text length 2000 chars, sanitized against injection
- HTTPS only
- No PII stored; prediction logs anonymized

**Documentation:**
- OpenAPI 3.0 spec auto-generated from endpoint definitions
- Hosted Swagger UI at /docs
- Python SDK: `pip install tract-client`
  ```python
  from tract import Client
  client = Client(api_key="...")
  result = client.assign("Implement rate limiting for API endpoints")
  ```

**Deployment:**
- Dockerfile + docker-compose.yml (API server + SQLite + model weights)
- Deployment guide: local, AWS/GCP/Azure, Heroku
- Health check endpoint: GET /v1/health

---

## 10. Phase 5: OpenCRE Upstream Contribution

**Deliverable:** Validated TRACT outputs packaged and submitted as contributions to the OpenCRE project.

TRACT consumes the OpenCRE ontology as its coordinate system. Phase 5 closes the loop by contributing back: human-reviewed crosswalk assignments, validated hub proposals, and AI/traditional bridge mappings. This is the "gives back to OpenCRE" promise from Section 1.

### Phase 5A: Export Pipeline & Fork Validation ✅ COMPLETE

Built the full export-to-import pipeline and validated it against a local OpenCRE fork (~/github_projects/OpenCRE).

**Infrastructure built (PRs #17, #18, #19):**

| Component | Location | Purpose |
|-----------|----------|---------|
| Export filter pipeline | `tract/export/filters.py` | SQL+Python filters: ground truth exclusion, confidence floor, OOD, null confidence, review status |
| OpenCRE CSV generator | `tract/export/opencre_csv.py` | Per-framework CSVs targeting OpenCRE's `export_format_parser` |
| Coverage gaps report | `tract/export/gaps.py` | Per-control exclusion reasons for every unexported control |
| Export manifest | `tract/export/manifest.py` | Provenance: git SHA, model hash, filter stats, staleness check |
| Staleness check | `tract/export/staleness.py` | Pre-export verification against upstream OpenCRE API |
| CLI command | `tract export --opencre` | Full pipeline: filter → CSV → manifest → gaps report |
| Direct import script | `scripts/direct_opencre_import.py` | SQLAlchemy import bypassing Flask/Redis/graph loading (~3s) |
| REST API import script | `scripts/opencre_import.sh` | Flask-based import path (backup, slower) |

**Pilot results (2026-05-01):**

| Framework | Exported | Total | Coverage | Gaps |
|-----------|----------|-------|----------|------|
| CSA AI Controls Matrix | 184 | 243 | 75.7% | 59 below floor |
| MITRE ATLAS | 128 | 202 | 63.4% | 74 (64 below floor, 3 null, 7 no assignment) |
| EU AI Act | 84 | 126 | 66.7% | 42 (16 below floor, 26 no assignment) |
| OWASP Agentic AI Top 10 | 8 | 10 | 80.0% | 2 below floor (ASI05=0.115, ASI07=0.267) |
| NIST AI 600-1 | 7 | 12 | 58.3% | 5 (3 below floor, 2 no assignment) |
| OWASP LLM Top 10 | 0 | 10 | 0.0% | 10 (6 ground truth, 4 no assignment) — correct exclusion |
| **Total** | **411** | **603** | **68.2%** | **192 missing** |

All 411 assignments imported into local fork DB. MITRE ATLAS handled correctly: 100 new nodes created, 28 pre-existing upstream nodes matched without duplication (44 pre-existing links preserved).

**Confidence thresholds:** 0.30 global floor (temperature-scaled cosine similarity, T=0.074), 0.35 override for MITRE ATLAS. These are ~2.5× the F1-optimal threshold (0.121), reflecting a conservative policy for upstream contribution quality.

**Export format (resolved):** OpenCRE CSV with `CRE 0` column = `"hub_id|hub_name"` (pipe-delimited), standard columns = `StandardName|name`, `StandardName|id`, `StandardName|description`, `StandardName|hyperlink`. Consumed by OpenCRE's `export_format_parser.parse_export_format()`.

### Phase 5B: Upstream Contribution (unblocked — Phase 3 complete)

**Contribution scope:**

| Contribution | Source | Format | Status |
|-------------|--------|--------|--------|
| New framework→hub assignments | Phase 3 human-reviewed crosswalk | OpenCRE CSV (per framework) | ✅ Phase 3 complete |
| Hub proposals | Phase 1D HDBSCAN clustering of OOD controls | Proposed new CRE hubs with evidence | Proposals generated, needs upstream format |
| AI/traditional bridge mappings | Phase 2B bridge analysis | Cross-domain Related links | Not started |
| Confidence metadata | Phase 1C calibration | Per-assignment calibrated scores | Available now |

**Workflow:**

1. **Export:** ✅ `tract export --opencre` generates per-framework CSVs + manifest + coverage gaps
2. **Validate:** ✅ Confidence floor filters, ground truth exclusion, staleness check, coverage gaps report
3. **Package:** ✅ CSVs + `export_manifest.json` + `coverage_gaps.json` in `opencre_export/`
4. **Import to fork:** ✅ `scripts/direct_opencre_import.py` loads into local fork for validation
5. **Submit upstream:** PENDING — open PRs against OpenCRE GitHub after Phase 3 human review
6. **Track:** NOT YET — record acceptance/rejection by OpenCRE maintainers

**Contribution criteria (only submit high-quality data):**

- Assignments must be human-reviewed (Phase 3 complete) — fork import is a validation step, not an upstream submission
- Hub proposals must pass the guardrailed review process (Phase 1D)
- Calibrated confidence ≥ 0.30 (ATLAS ≥ 0.35) — tuned during Phase 5A pilot
- Ground truth frameworks excluded (OWASP LLM Top 10 — already in OpenCRE as training data)
- No assignments from frameworks already fully linked in OpenCRE (avoid contradicting existing expert curation)

**Open questions (remaining):**

- Attribution — how to credit TRACT's ML-assisted curation in OpenCRE's provenance tracking
- Contribution cadence — one-time bulk submission vs. incremental as new frameworks are ingested (initial plan: one-time per framework)
- Hub proposal format — OpenCRE's preferred format for proposing new hubs (not just linking to existing ones)

---

## 11. Guardrails and Safety

**All guardrails are implemented as concrete Phase 1 deliverables in Section 6.6.** The five categories (Data Integrity, Model Integrity, Output Integrity, Adversarial Robustness, Provenance Tracking) are built into the components that need them — ingestion pipeline, model training, inference pipeline, test suite, and logging — not bolted on after the fact. See Section 6.6 for the complete implementation plan.

---

## 12. Success Criteria

| Phase | Metric | Target |
|-------|--------|--------|
| Phase 0 | LLM probe hit@5 (LOFO) | > 0.50 (feasibility gate) |
| Phase 0 | Embedding baseline hit@1 (LOFO) | Establish numeric baseline |
| Phase 0 | LLM hub description pilot (50 hubs) | Expert acceptance rate > 80% |
| Phase 1 | Trained model hit@1 micro (LOFO) | > Phase 0 embedding baseline by 0.10+ (pre-registered, not substitutable) |
| Phase 1 | Per-fold delta vs zero-shot | No fold with delta < 0 (regression flag) |
| Phase 1 | Trained model hit@5 (LOFO) | > 0.70 |
| Phase 1 | Active learning expert acceptance rate | > 80% on 3rd round |
| Phase 1 | 22 frameworks fully ingested | 100% with correct mapping units |
| Phase 1 | 400 hub descriptions generated and reviewed | 100% |
| Phase 1 | CRE hierarchy built and validated | All 522 hubs, no cycles, all leaves reachable |
| Phase 1 | Guardrails implemented (5 categories) | All pass automated test suite |
| Phase 1 | Hub proposal system functional | End-to-end: OOD detect -> cluster -> propose -> review |
| Phase 1 | Crosswalk database complete | All 22 frameworks with hub assignments |
| Phase 2 | HuggingFace AIBOM score | 100/100 |
| Phase 2 | AI/Traditional bridge hubs identified | At least 10 validated bridges |
| Phase 3 | Human review coverage | ✅ 100% of predicted assignments reviewed (878/878) |
| Phase 3 | Published dataset | ✅ Published to huggingface.co/datasets/rockCO78/tract-crosswalk-dataset |
| Phase 3B | Experimental narrative notebook | ≥128 cells, ≥24 figures, full story arc from baselines to final model + CLI tutorial |
| Phase 3B | Notebook reproducibility | All cells run top-to-bottom with identical output |
| Phase 3B | Visualization quality | Interactive 3D/animated figures with static fallbacks, colorblind-accessible palettes |
| Phase 4 | API latency | < 500ms per single control assignment |
| Phase 4 | API documentation | Complete OpenAPI spec |
| Phase 5A | Export pipeline functional | ✅ `tract export --opencre` generates CSVs + manifest + coverage gaps |
| Phase 5A | Fork import validated | ✅ 411 assignments across 5 frameworks imported into local OpenCRE fork |
| Phase 5A | Coverage gaps report | ✅ Per-control exclusion reasons for all 192 unexported controls |
| Phase 5B | Upstream PRs submitted | Human-reviewed assignments submitted to OpenCRE GitHub (after Phase 3) |
| Phase 5B | Hub proposals submitted | Validated proposals submitted to OpenCRE project |
| Phase 5B | Contribution acceptance rate | Track acceptance/rejection by OpenCRE maintainers |

---

## 13. Non-Goals and Scope Boundaries

- **NOT rebuilding OpenCRE.** We extend it, contribute to it, build on top of it. The CRE ontology is maintained by the OpenCRE project; we consume and enrich it.
- **NOT a general NLI or semantic similarity system.** Narrowly scoped to security framework crosswalking via CRE hub assignment.
- **NOT replacing expert judgment.** Model outputs are predictions that require human validation for compliance decisions. The active learning loop is designed around expert review, not autonomous deployment.
- **NOT competing with the existing v_final pairwise classifier.** TRACT is a fundamentally different architecture (assignment vs. pairwise). The old project is a COMP 4433 academic deliverable; TRACT is a production community tool.
- **NOT a public hosted service** (like opencre.org). This is a tool for security professionals to run locally or self-host. Phase 4 API is for programmatic access, not SaaS.
- **Phase 1 does NOT include AI/traditional security bridging.** That's explicitly Phase 2.
- **No web UI.** TRACT is CLI-first. The API (Phase 4) provides programmatic access; there is no planned Dash/web dashboard.

---

## 14. Open Questions and Risks

| # | Question/Risk | Impact | Resolution Path |
|---|--------------|--------|-----------------|
| 1 | Is 198 AI-specific links enough for hub assignment? | May need more AI training data | Phase 0 gate + active learning loop adds data iteratively |
| 2 | Do traditional security links transfer to AI domain? | Transfer learning may not help if domains are too distinct | Phase 1B ablation: compare training on all 4,406 links vs. AI-only 198 links vs. two-stage transfer |
| 3 | Can 400 leaf hubs distinguish fine-grained relationships? | Granularity may be too coarse | Hub proposal system as escape valve; hierarchy paths add precision |
| 4 | How much expert review time is actually needed? | 33 hrs for descriptions + active learning review | Pilot with 50 hubs in Phase 0 to calibrate |
| 5 | AIUC-1 data shows 132 activities but user reports ~187 | May be missing data in graph | Verify against official AIUC-1 standard during Phase 1 ingestion |
| 6 | CRE ontology may evolve (hubs added/removed by OpenCRE maintainers) | Model needs to handle ontology changes | Version pin CRE data; rebuild on new releases |
| 7 | Hub link distribution is highly skewed | Some hubs have 100+ examples, others have 1-2 | Contrastive learning with hard negative mining (sibling hubs); hub-frequency-weighted sampling during training |
| 8 | Community adoption / "build it and they will come" risk | No users means no value | Phase 2 framework submission template lowers barrier; target specific user community (OWASP, MITRE) |
| 9 | Training data density drives per-fold performance | Model only improves frameworks with dense CRE linkage (OWASP-X: 63 eval items, ML-10: 7) and regresses on sparse ones (ATLAS: 43 items but 0.279 hit@1). Phase 1B adversarial review confirmed this is the #1 finding. | Phase 1B iteration: error analysis on ATLAS fold, investigate fold-aware training weight balancing, curriculum sampling, or hub-neighborhood augmentation |
| 10 | ~~fp16 non-determinism on H100~~ **RESOLVED** | CUDA determinism flags added. NIST fold rerun produces identical metrics (hit@1=0.429). The 0.393 discrepancy was a measurement artifact (different computation context), not training variance. | No further action needed. |

---

## 15. Technical Dependencies

| Dependency | Purpose | Phase |
|------------|---------|-------|
| opencre_all_cres.json | CRE hub ontology (522 hubs, 4,406 links, 22 frameworks) | Phase 0+ |
| 9 AI framework source files | Primary source data for mapping units | Phase 1 |
| 13 traditional framework data (from OpenCRE) | Additional training data + traditional security coverage | Phase 1 |
| ai-security-framework-crosswalk repo | Reference only: expert_train.jsonl (5,920 pairs), nodes.json (983 nodes), framework source files | Phase 0+ |
| GPU compute | Contrastive fine-tuning of best base model (single A100/H100 per run; multiple pods for parallel ablations) | Phase 1 |
| LLM API access (Claude or GPT-4) | Hub description generation, Phase 0 LLM probe | Phase 0-1 |
| transformers, sentence-transformers | Encoder fine-tuning and embedding | Phase 1+ |
| hdbscan | Guardrailed hub proposal clustering | Phase 1+ |
| huggingface_hub | Model and dataset publication | Phase 2-3 |

---

## Appendix: Stress Test Findings (Corrected)

The 4-critic adversarial review (2026-04-27) identified several issues, many of which turned out to be based on incorrect assumptions. Key corrections:

- **"40-50% of auto-links semantically wrong"** -- WRONG. Auto-links are deterministic transitive inferences through MITRE's CWE/CAPEC taxonomy, not ML output. Traced in code: capec_parser.py lines 39-55.
- **"Only 1,413 human-curated links"** -- WRONG. Total is 4,406 (2,047 human + 2,359 expert-transitive). Auto-links are expert-validated.
- **"59 phantom hubs"** -- MOSTLY WRONG. 55 are structural parent nodes with children. Only 4 are true orphans.
- **"hit@1=0.130 is zero-shot baseline"** -- WRONG TASK. That's crossref retrieval, not hub assignment. We have no hub assignment baseline yet (hence Phase 0).
- **Phase 0 gate requirement** -- VALID. Must establish baselines before committing to trained model.
- **Leave-one-framework-out evaluation** -- VALID. Adopted as primary evaluation strategy.
- **Hub description gap** -- VALID. Solved by LLM generation + expert validation.

These corrections are incorporated into the main PRD. The stress test was valuable for identifying Phase 0 as a gate, but its data quality conclusions were systematically wrong due to misunderstanding auto-link provenance.
