# TRACT — Transitive Reconciliation and Assignment of CRE Taxonomies

## Product Requirements Document

**Date:** 2026-04-27
**Author:** Rock Lambros
**Status:** Draft — pending approval

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
- **Training data:** 4,406 standard-to-hub links (2,047 human + 2,359 expert-transitive), extracted fresh from OpenCRE API data
- **Training strategy:** Contrastive learning with hard negatives (sibling hubs in the CRE tree are natural hard negatives). Whether to train on all 4,406 links vs. AI-specific 198 links vs. a two-stage transfer approach is an ablation experiment, not a prescribed architecture.
- **Output:** Cosine similarity scores over 400 leaf hubs, calibrated to probabilities via temperature/Platt scaling
- **Multi-label handling:** Median 1 hub/section but max 38; model returns ranked hub list with calibrated similarity scores. Per-hub similarity thresholds tuned on validation set.

### 6.5 Hub Representation Firewall
When evaluating hub assignment for framework X, rebuild hub representations WITHOUT contributions from X's own linked sections. This means:
- Remove X's section names from the hub's linked-standards list
- Regenerate the hub's embedding without X's influence
- This is a HARD requirement for honest leave-one-framework-out evaluation
- Implemented as a build step, not a runtime hack

### 6.6 Guardrail Implementation
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

### 6.7 Active Learning Loop
For frameworks with zero CRE coverage (AIUC-1, CSA AICM, CoSAI, EU GPAI CoP, OWASP Agentic) and any thin-coverage traditional frameworks:
1. Model predicts top-K hub assignments for each control
2. Expert reviews predictions (accept / reject / correct) via a review interface (can be CLI-based in Phase 1)
3. Accepted predictions added to training data with provenance="active_learning_round_N"
4. Model retrained on expanded dataset
5. Repeat until expert acceptance rate stabilizes (target: >80% accept rate)

### 6.8 Crosswalk Database
- SQLite database for local use, exportable to JSON/CSV
- Each control assigned to CRE hub(s) with confidence scores and provenance
- Cross-framework relationship matrix derived transitively from shared hub assignments
- Every assignment traced to: model version, training data version, expert review status
- **Deliverable:** `crosswalk.db` + export scripts

### 6.9 CLI Tool
```
tract assign "Implement rate limiting for API endpoints"
  -> CRE-236 (API security, 0.89), CRE-441 (Rate limiting, 0.72)

tract compare --framework atlas --framework asvs
  -> relationship matrix with confidence scores

tract ingest --file new_framework.json --template standard
  -> predicted hub assignments for review

tract export --format csv --framework atlas
  -> full crosswalk table

tract hierarchy --hub CRE-236
  -> full path: Root > Network Security > API Security > CRE-236
  -> linked controls from 5 frameworks
```

### 6.10 Guardrailed Hub Proposal System
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
- Compare against all Phase 0 baselines — trained model MUST exceed them
- No pairwise metrics as success criteria. The assignment paradigm is evaluated on assignment quality, period.

---

## 7. Phase 2: Web Platform + HuggingFace + AI/Traditional Bridge

**Note:** Phase 2 will be planned in detail after Phase 1 ships and we have real model results. The deliverables below define WHAT gets built; the implementation plan will be written when Phase 2 starts. Phases 2-4 are intentionally less granular than Phase 1 because their scope depends on Phase 1 outcomes.

### 7.1 Dash Web UI
**Deliverable:** Running Dash app with 5 pages, deployable locally or via Docker.

| Page | What It Does | Data Source |
|------|-------------|-------------|
| Crosswalk Explorer | Select framework A -> see all controls -> click control -> see CRE hub(s) -> see controls from framework B at the same hub | crosswalk.db (Phase 1) |
| Framework Comparison | Select 2 frameworks -> side-by-side table showing which controls share hubs (equivalent), which share parent hubs (related), which don't overlap (gap) | crosswalk.db derived relationships |
| Hub Ontology Browser | Navigate CRE hierarchy tree -> click hub -> see all linked controls from all 22 frameworks, confidence scores, description | cre_hierarchy.json + hub_descriptions.json + crosswalk.db |
| Confidence Dashboard | Heatmap: frameworks x hubs, colored by assignment confidence. Click a cell -> see the control text and model prediction details | crosswalk.db prediction logs |
| Control Search | Paste any control text -> live model inference -> show top-5 hub assignments with confidence + related controls from other frameworks | Trained model (Phase 1) + crosswalk.db |

**Tech stack:** Dash + Plotly + dash-bootstrap-components (CYBORG theme, matching Project 2). SQLite backend from Phase 1.

### 7.2 Framework Submission System
**Deliverable:** JSON schema template + upload CLI + web upload page + review queue.

**Concrete components:**
- `framework_template.json`: JSON Schema defining required fields (control_id, title, description, hierarchy_level, framework_name, version, source_url)
- `tract validate --file new_framework.json`: Schema validation + duplicate detection (cosine similarity > 0.95 to existing controls flagged)
- Upload page in Dash UI: drag-and-drop JSON, shows validation results, submits to model for hub assignment
- Review queue page: expert sees predicted assignments, accepts/rejects/corrects per control, batch approve
- `framework_registry.json`: versioned list of all ingested frameworks with metadata, changelog, ingestion date

### 7.3 HuggingFace Model Publication
**Deliverable:** Published model repo at huggingface.co/rockCO78/tract-cre-assignment with:
- Model weights (fine-tuned bi-encoder)
- `hub_descriptions.json` and `cre_hierarchy.json` bundled with model
- Model card (AIBOM-compliant, targeting 100/100): model description, intended use, architecture, training details, evaluation results, limitations, ethical considerations, environmental impact, usage code snippet, citation
- `predict.py`: standalone inference script — takes control text, returns hub assignments
- `train.py`: reproduction script with requirements.txt and data download instructions

### 7.4 AI/Traditional Security Bridge
**Deliverable:** Extended `cre_hierarchy.json` with bridge hubs + bridge validation report.

**Concrete process:**
1. Take the 81 AI hubs and 441 traditional hubs
2. For each AI hub, compute embedding similarity to all traditional hubs. Flag pairs with cosine > 0.70 as bridge candidates
3. Use ENISA (68 links), ETSI (36 links), BIML (21 links) as seed evidence — these frameworks appear on both AI and traditional hubs
4. For each bridge candidate pair: LLM-generate a bridge description explaining the conceptual overlap
5. Expert review: accept bridge, reject bridge, or propose a new parent hub that contains both
6. Accepted bridges become new Related links in `cre_hierarchy.json`
7. New parent hubs (if any) created via the guardrailed hub proposal system (6.10)
8. Model retrained with bridge links as additional training signal
9. **Deliverable:** `bridge_report.json` documenting all AI/traditional bridges with evidence and expert review status

---

## 8. Phase 3: Published Human-Reviewed Crosswalk Dataset

**Deliverable:** Versioned dataset published to HuggingFace Datasets AND Zenodo.

**Review workflow (concrete):**
1. Export all model-predicted hub assignments from `crosswalk.db` (every control from all 22 frameworks)
2. Group by framework. For each framework, generate a review spreadsheet (CSV or web interface): control_id, control_text, predicted_hub_1 (confidence), predicted_hub_2 (confidence), ..., reviewer_decision, reviewer_notes
3. Expert reviews one framework at a time. Per control: accept top prediction, select a different hub, flag for discussion, or mark "no good hub" (feeds into hub proposal system)
4. Track: total controls reviewed, acceptance rate, edit rate, rejection rate, avg time per control
5. Second-pass review for rejected/edited controls: check if edits are consistent across similar controls
6. Generate inter-reviewer agreement metrics (if multiple reviewers): Cohen's kappa or Krippendorff's alpha
7. Freeze reviewed assignments as `crosswalk_reviewed_v1.0.jsonl`

**Published dataset structure:**
```
tract-crosswalk-dataset/
  crosswalk_v1.0.jsonl          # Every control + hub assignment + confidence + review status
  framework_metadata.json       # 22 framework descriptions, versions, sources
  cre_hierarchy_v1.0.json       # Hub ontology at time of publication
  hub_descriptions_v1.0.json    # Validated hub descriptions
  review_metrics.json           # Acceptance rates, agreement metrics, reviewer effort
  README.md                     # Dataset card (HuggingFace Datasets format)
  LICENSE                       # Apache 2.0 or CC-BY-4.0
```

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
| 11. Conclusion & Next Steps | Summary, limitations, what Phase 1C needs | Summary metrics table, roadmap visualization |
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

## 10. Guardrails and Safety

**All guardrails are implemented as concrete Phase 1 deliverables in Section 6.6.** The five categories (Data Integrity, Model Integrity, Output Integrity, Adversarial Robustness, Provenance Tracking) are built into the components that need them — ingestion pipeline, model training, inference pipeline, test suite, and logging — not bolted on after the fact. See Section 6.6 for the complete implementation plan.

---

## 11. Success Criteria

| Phase | Metric | Target |
|-------|--------|--------|
| Phase 0 | LLM probe hit@5 (LOFO) | > 0.50 (feasibility gate) |
| Phase 0 | Embedding baseline hit@1 (LOFO) | Establish numeric baseline |
| Phase 0 | LLM hub description pilot (50 hubs) | Expert acceptance rate > 80% |
| Phase 1 | Trained model hit@1 (LOFO) | > Phase 0 embedding baseline by 0.10+ |
| Phase 1 | Trained model hit@5 (LOFO) | > 0.70 |
| Phase 1 | Active learning expert acceptance rate | > 80% on 3rd round |
| Phase 1 | 22 frameworks fully ingested | 100% with correct mapping units |
| Phase 1 | 400 hub descriptions generated and reviewed | 100% |
| Phase 1 | CRE hierarchy built and validated | All 522 hubs, no cycles, all leaves reachable |
| Phase 1 | Guardrails implemented (5 categories) | All pass automated test suite |
| Phase 1 | Hub proposal system functional | End-to-end: OOD detect -> cluster -> propose -> review |
| Phase 1 | Crosswalk database complete | All 22 frameworks with hub assignments |
| Phase 2 | Web UI functional | 5 pages deployed, all data sources connected |
| Phase 2 | New framework submission < 1 hour | For standard-format frameworks using template |
| Phase 2 | HuggingFace AIBOM score | 100/100 |
| Phase 2 | AI/Traditional bridge hubs identified | At least 10 validated bridges |
| Phase 3 | Human review coverage | 100% of predicted assignments reviewed |
| Phase 3 | Published dataset | Available on HuggingFace or Zenodo |
| Phase 3B | Experimental narrative notebook | ≥128 cells, ≥24 figures, full story arc from baselines to final model |
| Phase 3B | Notebook reproducibility | All cells run top-to-bottom with identical output |
| Phase 3B | Visualization quality | Interactive 3D/animated figures with static fallbacks, colorblind-accessible palettes |
| Phase 4 | API latency | < 500ms per single control assignment |
| Phase 4 | API documentation | Complete OpenAPI spec |

---

## 12. Non-Goals and Scope Boundaries

- **NOT rebuilding OpenCRE.** We extend it, contribute to it, build on top of it. The CRE ontology is maintained by the OpenCRE project; we consume and enrich it.
- **NOT a general NLI or semantic similarity system.** Narrowly scoped to security framework crosswalking via CRE hub assignment.
- **NOT replacing expert judgment.** Model outputs are predictions that require human validation for compliance decisions. The active learning loop is designed around expert review, not autonomous deployment.
- **NOT competing with the existing v_final pairwise classifier.** TRACT is a fundamentally different architecture (assignment vs. pairwise). The old project is a COMP 4433 academic deliverable; TRACT is a production community tool.
- **NOT a public hosted service** (like opencre.org). This is a tool for security professionals to run locally or self-host. Phase 4 API is for programmatic access, not SaaS.
- **Phase 1 does NOT include AI/traditional security bridging.** That's explicitly Phase 2.
- **Phase 1 does NOT require the web UI.** CLI-first; web comes in Phase 2.

---

## 13. Open Questions and Risks

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

---

## 14. Technical Dependencies

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
| dash, plotly | Web UI | Phase 2 |
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
