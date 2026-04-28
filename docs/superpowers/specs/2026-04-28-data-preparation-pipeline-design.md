# TRACT Data Preparation Pipeline — Design Spec

**Date:** 2026-04-28
**Scope:** PRD Sections 4.5–4.8 — all data preparation for Phase 0
**Status:** Draft — pending approval

---

## 1. Goal

Prepare all framework data and OpenCRE hub links so Phase 0 zero-shot baseline experiments can run cleanly. This means: 12 framework parsers producing standardized JSON, fresh OpenCRE data from the live API, 4,406 hub links extracted for LOFO training splits, and a merged all_controls.json.

## 2. Architecture

### 2.1 `tract/` Python Package

TRACT is a platform for adding future frameworks to the CRE ontology (PRD line 582: `pip install tract-client`). The shared infrastructure is the product, not a convenience. All parsers, scripts, and future phases import from this package.

```
tract/
├── __init__.py
├── config.py              # Paths, constants, expected counts, framework registry
├── schema.py              # Pydantic v2 models: Control, FrameworkOutput, HubLink
├── sanitize.py            # sanitize_text(), strip_html(), normalize_unicode()
├── io.py                  # atomic_write_json(), load_json(), deterministic serialization
└── parsers/
    ├── __init__.py
    └── base.py            # BaseParser ABC with Template Method pattern
```

### 2.2 BaseParser Contract

Every framework parser subclasses `BaseParser` and implements one method:

```python
from abc import ABC, abstractmethod
from pathlib import Path
from tract.schema import Control, FrameworkOutput

class BaseParser(ABC):
    # --- Metadata (declared by each subclass) ---
    framework_id: str           # e.g., "csa_aicm"
    framework_name: str         # e.g., "CSA AI Controls Matrix"
    version: str                # e.g., "1.0"
    source_url: str
    mapping_unit_level: str     # e.g., "control"
    expected_count: int | None  # e.g., 243 (None for uncharacterized new frameworks)

    def __init__(self, raw_dir: Path, output_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.output_dir = output_dir

    @abstractmethod
    def parse(self) -> list[Control]:
        """Framework-specific extraction logic. Only this method varies."""

    def run(self) -> FrameworkOutput:
        """Invariant pipeline: parse -> sanitize -> validate -> write.
        Sanitization and pydantic validation happen automatically.
        Subclasses NEVER override this method."""
        controls = self.parse()
        controls = [self._sanitize_control(c) for c in controls]
        output = FrameworkOutput(
            framework_id=self.framework_id,
            framework_name=self.framework_name,
            version=self.version,
            source_url=self.source_url,
            fetched_date=self._today(),
            mapping_unit_level=self.mapping_unit_level,
            controls=controls,
        )
        self._check_expected_count(output)
        self._write_atomic(output)
        return output
```

**Why Template Method:** 12 parsers follow the identical 5-step pipeline (read → parse → sanitize → validate → write). Only step 2 varies. The invariant steps must be guaranteed — a parser author cannot accidentally skip sanitization or atomic writes.

**Adding a new framework:** Subclass BaseParser, set metadata attributes, implement `parse()` returning `list[Control]`. ~50–150 lines per parser.

### 2.3 Pydantic v2 Schema Models

Single source of truth in `tract/schema.py`. Validates on construction — no separate validation pass needed.

```python
from pydantic import BaseModel, Field

class Control(BaseModel):
    control_id: str
    title: str
    description: str = Field(max_length=2000)
    full_text: str | None = None
    hierarchy_level: str | None = None
    parent_id: str | None = None
    parent_name: str | None = None
    metadata: dict[str, str | list[str]] | None = None

class FrameworkOutput(BaseModel):
    framework_id: str
    framework_name: str
    version: str
    source_url: str
    fetched_date: str
    mapping_unit_level: str
    controls: list[Control]

class HubLink(BaseModel):
    cre_id: str
    cre_name: str
    standard_name: str
    section_id: str
    section_name: str
    link_type: str              # "LinkedTo" or "AutomaticallyLinkedTo"
    framework_id: str           # Normalized ID for LOFO grouping
```

### 2.4 Text Sanitization Pipeline

`tract/sanitize.py` — every text field passes through this before pydantic validation:

1. Strip null bytes
2. Normalize Unicode to NFC
3. Decode HTML entities (`&sect;` → `§`, `&copy;` → `©`)
4. Strip HTML tags (for EU AI Act, defensively for all)
5. Normalize whitespace (collapse runs, strip leading/trailing)
6. Fix PDF extraction artifacts (broken hyphenation, ligatures fi→fi, ff→ff)
7. Enforce max length: if `len > 2000`, truncate to `description`, preserve full in `full_text`

### 2.5 Atomic I/O

`tract/io.py`:
- `atomic_write_json(data, path)`: Write to temp file in same directory, then `os.replace()` to target. Sorted keys, `ensure_ascii=False`, indent=2, trailing newline.
- `load_json(path)`: Read with `encoding='utf-8'` explicitly.
- All JSON output is deterministic: sorted keys, consistent indent, stable ordering.

## 3. Frameworks — 12 Parsers

### 3.1 Tier 1: Structured JSON (3 parsers)

| Parser | Input | Mapping Unit | Expected Count | ID Field | Text Field |
|--------|-------|-------------|----------------|----------|------------|
| `parse_csa_aicm.py` | `csa_aicm.json` → `controls[]` | control | 243 | `id` | `specification` |
| `parse_aiuc_1.py` | `aiuc-1-standard.json` → `domains[].controls[].activities[]` | activity | 132 | `id` (e.g., `A001.1`) | `description` |
| `parse_mitre_atlas.py` | `ATLAS_compiled.json` → `matrices[0].techniques[]` + `matrices[0].mitigations[]` | technique/mitigation | 202 (167+35) | `id` (e.g., `AML.T0000`) | `description` |

**CSA AICM notes:** 14 fields per control. `specification` is primary text. `implementation_guidelines` and `auditing_guidelines` go into `full_text` or `metadata`.

**AIUC-1 notes:** Activity IDs already exist in the source (`A001.1`, etc.). Activities have `category` (Core/Advanced) and `evidence_types` — store in `metadata`.

**MITRE ATLAS notes:** Data lives under `matrices[0]`. Techniques and mitigations are separate arrays. Both are mapping units. Distinguish via `hierarchy_level`: "technique" vs "mitigation".

### 3.2 Tier 2: YAML (1 parser)

| Parser | Input | Mapping Unit | Expected Count |
|--------|-------|-------------|----------------|
| `parse_cosai.py` | `risk-map/controls.yaml` + `risk-map/risks.yaml` | control + risk | 55 (29+26) |

Dual-dimension framework. Parse both YAML files. Validate against co-located JSON schemas in `risk-map/schemas/`. Controls and risks are separate mapping units with different `hierarchy_level` values.

### 3.3 Tier 3: Markdown (6 parsers)

| Parser | Input | Pattern | Expected Count |
|--------|-------|---------|----------------|
| `parse_nist_ai_rmf.py` | `nist_ai_rmf_1.0.md` | `GOVERN X.Y`, `MAP X.Y`, `MEASURE X.Y`, `MANAGE X.Y` | 72 |
| `parse_nist_ai_600_1.py` | `nist_ai_600_1.md` | Section headers `2.1`–`2.12` for 12 GAI risk categories | 12 |
| `parse_owasp_ai_exchange.py` | 7 `src_*.md` files | `#### #CONTROL_NAME` headers | 88 (54+34) |
| `parse_owasp_llm_top10.py` | `owasp_llm_top_10_2025.md` | `LLM0[1-9]:2025`, `LLM10:2025` | 10 |
| `parse_owasp_agentic_top10.py` | `owasp_agentic_top10_2026.md` | Map `##` headers to ASI01–ASI10 | 10 |
| `parse_eu_gpai_cop.py` | `gpai_code_of_practice_combined.md` | `### Measure N.X` headers | 32 |

**OWASP Agentic Top 10 ID mapping** (from user specification):

| Header | ID | Name |
|--------|----|------|
| (1st risk) | ASI01 | Agent Goal Hijack |
| (2nd risk) | ASI02 | Tool Misuse and Exploitation |
| (3rd risk) | ASI03 | Identity and Privilege Abuse |
| (4th risk) | ASI04 | Agentic Supply Chain Vulnerabilities |
| (5th risk) | ASI05 | Unexpected Code Execution (RCE) |
| (6th risk) | ASI06 | Memory & Context Poisoning |
| (7th risk) | ASI07 | Insecure Inter-Agent Communication |
| (8th risk) | ASI08 | Cascading Failures |
| (9th risk) | ASI09 | Human-Agent Trust Exploitation |
| (10th risk) | ASI10 | Rogue Agents |

**NIST AI 600-1 notes:** 12 GAI risk categories (CBRN, Confabulation, Dangerous Recommendations, Data Privacy, Environmental, Harmful Bias, Human-AI Configuration, Information Integrity, Information Security, Intellectual Property, Obscene Content, Value Chain). Mapping unit is the risk category (12 entries), NOT individual suggested actions within each category — the risk category carries the coherent semantic content needed for CRE hub assignment, while individual actions are implementation guidance that would lose context if extracted alone. Each risk category maps to GOVERN/MAP/MEASURE/MANAGE subcategories from the AI RMF — store those cross-references in `metadata`.

**OWASP AI Exchange notes:** 7 source files, Hugo-format markdown. Controls use `#### #IDENTIFIER` headers. File `src_ai_security_overview.md` and `owasp_ai_exchange.md` may be overview/index files — verify before parsing. Dual-dimension: controls + risks (threats).

**EU GPAI CoP notes:** Three chapters (Transparency, Copyright, Safety & Security) each with Commitments → Requirements → Measures. Measures are the mapping units (32 expected). Requirements (28) are secondary units — include as separate entries with `hierarchy_level: "requirement"`.

### 3.4 Tier 4: TXT (1 parser)

| Parser | Input | Pattern | Expected Count |
|--------|-------|---------|----------------|
| `parse_owasp_dsgai.py` | `MANIFEST.json` + `.txt` file | `DSGAI(0[1-9]\|1[0-9]\|2[01])` regex from manifest | 21 |

Read MANIFEST.json for metadata and ID regex. Parse `.txt` (pdftotext output). Use regex to locate 21 risk entries. Extract text between consecutive ID matches.

### 3.5 Tier 5: HTML (1 parser)

| Parser | Input | Pattern | Expected Count |
|--------|-------|---------|----------------|
| `parse_eu_ai_act.py` | `eu_ai_act_2024_1689.html` + `MANIFEST.json` | BeautifulSoup: articles + annexes | 100+ |

EUR-Lex HTML (XML 1.0, 15,530 lines). Parse with BeautifulSoup + lxml. Extract Articles 1–113 and Annexes I–XIII as separate mapping units. Each gets an ID (e.g., `AIA-Art1`, `AIA-AnnexI`), title, and full article text.

## 4. NIST 800-53 — Excluded

NIST 800-53 is already in OpenCRE with 300 links (PRD Section 4.3). Per PRD Section 4.6: "Traditional frameworks (13 in OpenCRE)... are ingested FROM the OpenCRE API data during Phase 1 — their control texts come from the CRE link metadata. No separate source files needed."

No parser. No raw data copy. Control texts come from OpenCRE API via `extract_hub_links.py`.

## 5. OpenCRE Fetch

`parsers/fetch_opencre.py` — standalone script (not a BaseParser subclass).

- **Endpoint:** `https://opencre.org/rest/v1/all_cres?per_page=50&page=N` (1-indexed)
- **Expected:** ~261 pages, ~522 CRE hubs
- **Retry:** Exponential backoff starting at 1s, max 30s, max 5 retries per page
- **Resumable:** Save individual page responses to `data/raw/opencre/pages/page_NNN.json`. On completion, merge into `data/raw/opencre/opencre_all_cres.json`. On restart, skip already-fetched pages.
- **Metadata:** Record fetch timestamp, total pages, total CREs in output file header.
- **Progress:** Log every 10 pages at INFO level.
- **Rate limiting:** 0.5s delay between successful requests to be respectful.
- **Output:** `data/raw/opencre/opencre_all_cres.json` — list of all CRE objects with embedded links.

## 6. Hub Link Extraction

`parsers/extract_hub_links.py` — reads `opencre_all_cres.json`, produces training data.

**Output 1:** `data/training/hub_links.jsonl` — one HubLink per line, all 4,406 standard-to-hub links.

**Output 2:** `data/training/hub_links_by_framework.json` — grouped by `framework_id` for LOFO splits:
```json
{
  "capec": [{"cre_id": "...", ...}, ...],
  "cwe": [...],
  "nist_800_53": [...],
  ...
}
```

**Link type filtering:**
- `LinkedTo` → include (human-curated, gold standard)
- `AutomaticallyLinkedTo` → include (expert-transitive, penalty=0)
- `Contains` / `Is Part Of` → exclude (structural, not training signal)
- `Related` → exclude (inter-CRE, not standard-to-hub)

**Framework ID normalization:** Map OpenCRE standard names to canonical IDs:
- "CAPEC" → `capec`
- "CWE" → `cwe`
- "NIST 800-53" → `nist_800_53`
- "ASVS" → `asvs`
- "Cloud Controls Matrix" → `csa_ccm` (NOT csa_aicm)
- "MITRE ATLAS" → `mitre_atlas`
- "OWASP AI Exchange" → `owasp_ai_exchange`
- etc.

**Validation:** Assert total link count is within 5% of 4,406. Log per-framework counts and compare to PRD Section 4.3 expectations.

## 7. Validation and Merge

### 7.1 `parsers/validate_all.py`

Discovers all `data/processed/frameworks/*.json` files. For each:
1. Load and validate against `FrameworkOutput` pydantic model
2. Check control count against parser's `expected_count` (within 10%)
3. Assert no empty `description` fields
4. Assert no duplicate `control_id` within a framework
5. Assert all required fields present

Reports pass/fail per framework with counts. Exit code 1 if any framework fails.

### 7.2 `parsers/merge_all_controls.py`

Reads all validated framework JSONs from `data/processed/frameworks/`. Merges into `data/processed/all_controls.json`:
```json
{
  "generated_date": "2026-04-28",
  "framework_count": 12,
  "total_controls": <sum>,
  "frameworks": [
    { <full FrameworkOutput for csa_aicm> },
    { <full FrameworkOutput for aiuc_1> },
    ...
  ]
}
```

Sorted by `framework_id` for determinism.

## 8. Project Setup

### 8.1 Dependencies (`requirements.txt`)

```
pydantic>=2.0,<3.0
pyyaml>=6.0
beautifulsoup4>=4.12
lxml>=5.0
requests>=2.31
```

### 8.2 `pyproject.toml`

Minimal project metadata. Package name: `tract`. Python >=3.11. Development dependencies: pytest, mypy, types-pyyaml, types-beautifulsoup4, types-requests.

### 8.3 `.gitignore`

```
data/raw/
models/
*.db
__pycache__/
.env
*.egg-info/
dist/
build/
.mypy_cache/
.pytest_cache/
```

## 9. Commit Sequence

| # | Commit | Contents |
|---|--------|----------|
| 1 | Repo scaffold | git init, .gitignore, directory tree, pyproject.toml, requirements.txt |
| 2 | `tract/` package foundation | config.py, schema.py, sanitize.py, io.py, base.py, `__init__.py` files |
| 3 | Raw framework files | Copy 12 framework sources from old project into data/raw/frameworks/ |
| 4 | OpenCRE fetch | fetch_opencre.py + data/raw/opencre/opencre_all_cres.json |
| 5 | Tier 1 parsers | CSA AICM, AIUC-1, MITRE ATLAS (validates BaseParser works) |
| 6 | Tier 2–3 parsers | CoSAI, NIST AI RMF, NIST AI 600-1, OWASP AI Exchange, LLM Top 10, Agentic Top 10, EU GPAI CoP |
| 7 | Tier 4–5 parsers | OWASP DSGAI, EU AI Act |
| 8 | Hub link extraction | extract_hub_links.py + hub_links.jsonl + hub_links_by_framework.json |
| 9 | Validation + merge | validate_all.py + merge_all_controls.py + all_controls.json |

## 10. Expected Control Counts (for validation)

| Framework | ID | Expected | Tolerance (10%) |
|-----------|----|----------|-----------------|
| CSA AICM | csa_aicm | 243 | 219–267 |
| AIUC-1 | aiuc_1 | 132 | 119–145 |
| MITRE ATLAS | mitre_atlas | 202 | 182–222 |
| CoSAI | cosai | 55 | 50–61 |
| NIST AI RMF | nist_ai_rmf | 72 | 65–79 |
| NIST AI 600-1 | nist_ai_600_1 | 12 | 11–13 |
| OWASP AI Exchange | owasp_ai_exchange | 88 | 79–97 |
| OWASP LLM Top 10 | owasp_llm_top10 | 10 | 9–11 |
| OWASP Agentic Top 10 | owasp_agentic_top10 | 10 | 9–11 |
| EU GPAI CoP | eu_gpai_cop | 32 | 29–35 |
| OWASP DSGAI | owasp_dsgai | 21 | 19–23 |
| EU AI Act | eu_ai_act | 100 | 90+ |
