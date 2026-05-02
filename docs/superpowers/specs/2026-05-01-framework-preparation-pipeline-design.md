# Framework Preparation Pipeline — Design Spec

**Date:** 2026-05-01
**Author:** Rock Lambros
**Status:** Draft — pending approval
**PRD Sections:** 4.8 (Standardized Schema), 6 (CLI Interface)
**Design Process:** 3-agent adversarial brainstorm + judge synthesis

---

## 1. Goal

Build `tract prepare` and `tract validate` commands that close the gap between "I have a raw framework document" and "I can run `tract ingest`." Today, getting a new framework into TRACT requires either writing a custom parser (12 exist) or hand-crafting a FrameworkOutput JSON file. Neither scales for ad-hoc new frameworks.

**The pipeline after this feature:**
```
raw document → tract prepare → prepared.json → tract validate → tract ingest → _review.json → tract accept → crosswalk.db → tract export
```

**Not in scope:** Modifying existing parsers, web UI, automatic framework discovery, bulk re-preparation of already-processed frameworks.

## 2. Architecture

```
tract prepare --file <path> [--llm] [--format <type>] --framework-id <id> ...
  │
  ├─ detect_format(path) → csv | markdown | json | unstructured
  │
  ├─ IF csv:    CsvExtractor.extract(path) → list[RawControl]
  ├─ IF markdown: MarkdownExtractor.extract(path) → list[RawControl]
  ├─ IF json:   JsonExtractor.extract(path) → list[RawControl]   (passthrough, minor reshaping)
  ├─ IF unstructured + --llm:
  │    ├─ read_document(path) → plain text (pdf: pdfplumber, html: bs4, other: raw read)
  │    ├─ llm_extract_controls(text, metadata) → list[RawControl]
  │    │    Claude API, tool_use structured output, temperature=0
  │    │    Saves raw LLM response to <framework_id>_llm_raw.json
  │    └─ For docs >100K tokens: chunk at heading boundaries, extract per chunk, deduplicate
  │
  ├─ sanitize_text() on every text field (description, title, full_text)
  ├─ Assemble FrameworkOutput from extracted controls + CLI metadata
  ├─ Run validation (same as `tract validate`)
  └─ Write prepared.json (atomic write)

tract validate --file <path>
  │
  ├─ Load JSON
  ├─ Run validation rules (§4)
  └─ Print report: errors (block) + warnings (inform)

tract ingest --file <path>  (modified: validation gate added)
  │
  ├─ Run validation rules — abort on errors
  ├─ (existing) Schema validation via Pydantic
  ├─ (existing) Model inference
  └─ (existing) Write _review.json
```

## 3. CLI Interface

### `tract prepare`

```bash
# Heuristic extraction from structured formats
tract prepare --file framework_controls.csv \
  --framework-id new_fw \
  --name "New Framework" \
  --version "1.0" \
  --source-url "https://example.com/framework" \
  --mapping-unit control

# LLM extraction from unstructured documents
tract prepare --file messy_framework.pdf --llm \
  --framework-id new_fw \
  --name "New Framework" \
  --version "1.0" \
  --source-url "https://example.com/framework" \
  --mapping-unit control

# Override output path (default: <input_stem>_prepared.json)
tract prepare --file raw.md --output prepared.json ...

# Override auto-detected format
tract prepare --file weird_extension.dat --format csv ...
```

**Required flags for all prepare calls:**
- `--file` — input file path
- `--framework-id` — slug matching `^[a-z][a-z0-9_]{1,49}$`
- `--name` — human-readable framework name
- `--version` — version string
- `--source-url` — official URL
- `--mapping-unit` — what each control represents (control, technique, risk, article, etc.)

**Optional flags:**
- `--llm` — use Claude API for extraction (required for unstructured formats without clear structure)
- `--format` — override auto-detected format (`csv`, `markdown`, `json`, `unstructured`)
- `--output` — override output file path
- `--heading-level` — for markdown extraction: heading depth to split on (default: auto-detect)

**Format auto-detection** uses file extension: `.csv`/`.tsv` → csv, `.md`/`.markdown` → markdown, `.json` → json. All other extensions (`.pdf`, `.html`, `.txt`, `.docx`) → unstructured (requires `--llm` or explicit `--format`). Override with `--format` when the extension is misleading.

### `tract validate`

```bash
tract validate --file prepared.json
tract validate --file prepared.json --json    # machine-readable output
```

Exits 0 if no errors (warnings OK), exits 1 if any errors found.

### `tract ingest` (modified)

No CLI changes. Internally adds a validation gate before model loading. If validation errors exist, prints them and exits 1. Warnings are printed but don't block.

## 4. Validation Rules

### Errors (block ingest, exit 1)

| Rule | Check | Message |
|------|-------|---------|
| Schema conformance | `FrameworkOutput.model_validate()` | Pydantic error details |
| Empty description | `len(description.strip()) < 10` | "Control {id}: description too short ({n} chars, min 10)" |
| Duplicate control_id | `len(ids) != len(set(ids))` | "Duplicate control_id: {id}" |
| Invalid framework_id | Not matching `^[a-z][a-z0-9_]{1,49}$` | "Invalid framework_id: {id}" |
| Null bytes | `\x00` in any text field | "Control {id}: null bytes in {field}" |
| Zero controls | `len(controls) == 0` | "No controls found" |

### Warnings (report, don't block)

| Rule | Check | Message |
|------|-------|---------|
| Short description | `len(description.strip()) < 50` | "Control {id}: description only {n} chars — may produce weak embeddings" |
| Long unsplit description | `len(description) > 2000` and no `full_text` | "Control {id}: description {n} chars without full_text split" |
| Problematic control_id chars | `:` or whitespace in control_id | "Control {id}: contains characters that may cause issues in DB keys" |
| Low control count | `len(controls) < 3` | "Only {n} controls — unusually low" |
| High control count | `len(controls) > 2000` | "{n} controls — unusually high, verify extraction" |
| Missing optional fields | `version` or `source_url` empty | "Missing {field} — recommended for traceability" |
| Non-NFC unicode | Text not in NFC form | "Control {id}: text contains non-NFC unicode (will be normalized)" |

## 5. Format Extractors

### CSV Extractor

Expects a CSV with a header row. Column mapping (case-insensitive, flexible):
- `control_id` or `id` → `control_id`
- `title` or `name` → `title`
- `description` or `desc` or `text` → `description`
- `full_text` (optional) → `full_text`

Uses Python's `csv.DictReader`. Skips rows where all mapped fields are empty.

### Markdown Extractor

Splits on heading patterns. Each heading becomes a control:
- Heading text → `title`
- Body text below the heading (until next same-or-higher-level heading) → `description`
- `control_id` extracted from heading if it matches a pattern like `ASI01:`, `CTRL-01 -`, `1.2.3` prefix. Falls back to positional ID (`CTRL-01`, `CTRL-02`, ...) if no pattern found.

Configurable heading level via `--heading-level` flag (default: auto-detect the most common heading level with body text).

### JSON Extractor

Handles two cases:
1. **Already FrameworkOutput** — passthrough (validates only, re-sanitizes text fields)
2. **Array of objects** — maps fields heuristically: looks for keys matching `id`/`control_id`/`section_id`, `name`/`title`, `description`/`desc`/`text`/`body`. If the top-level JSON has a `controls` or `items` or `data` array, uses that as the control list. Fails with a clear error if no recognizable structure is found — suggests `--llm` for complex nested formats.

### LLM Extractor (--llm)

Uses Claude API with `tool_use` structured output:

```python
tools = [{
    "name": "extract_controls",
    "description": "Extract security framework controls from the document",
    "input_schema": {
        "type": "object",
        "properties": {
            "controls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "control_id": {"type": "string"},
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "full_text": {"type": ["string", "null"]},
                    },
                    "required": ["control_id", "title", "description"]
                }
            }
        },
        "required": ["controls"]
    }
}]
```

- Model: `claude-sonnet-4-20250514` (cost-effective for extraction)
- Temperature: 0.0
- System prompt instructs exhaustive extraction — every control/requirement/technique in the document
- For documents >100K tokens: chunk at heading boundaries, extract per chunk, deduplicate by control_id (keep longer description)
- Raw API response saved to `<framework_id>_llm_raw.json` in the same directory as the output file, for audit
- API key from `pass anthropic/api-key` (existing credential pattern)

## 6. Module Structure

```
tract/
  prepare/
    __init__.py          — public API: prepare_framework()
    extract.py           — format detection, ExtractorRegistry
    csv_extractor.py     — CsvExtractor
    markdown_extractor.py — MarkdownExtractor
    json_extractor.py    — JsonExtractor
    llm_extractor.py     — LlmExtractor (Claude API integration)
  validate.py            — validate_framework() → list[ValidationIssue]
  cli.py                 — add prepare + validate subcommands, validation gate in ingest
```

`ValidationIssue` is a simple dataclass:
```python
@dataclass
class ValidationIssue:
    severity: Literal["error", "warning"]
    control_id: str | None  # None for framework-level issues
    rule: str               # machine-readable rule name
    message: str            # human-readable message
```

## 7. Sanitization

`tract prepare` runs `sanitize_text()` from `tract/sanitize.py` on every text field (title, description, full_text) during extraction. This is the same pipeline used by all existing parsers via `BaseParser._sanitize_control()`:

1. Strip null bytes
2. Unicode NFC normalization
3. Strip zero-width characters
4. HTML unescape + strip tags
5. Fix PDF ligatures
6. Fix broken hyphenation
7. Collapse whitespace
8. Enforce max length (description: 2000 chars, full_text: 50000 chars)

If description exceeds 2000 chars after sanitization, the full text is preserved in `full_text` and description is truncated — matching `sanitize_text(text, return_full=True)` behavior.

## 8. Reproducibility

- **Heuristic extractors** are fully deterministic. Same input → byte-identical output.
- **LLM extractor** is non-deterministic by nature. Mitigations:
  - `temperature=0` for maximum consistency
  - Raw LLM response saved as `_llm_raw.json` audit artifact
  - The prepared JSON (not the LLM call) is the source of truth — once generated, it's a deterministic artifact
  - Re-running `prepare --llm` may produce different output; the user should treat the prepared JSON as a snapshot to review and commit, not a reproducible build step

## 9. Error Handling

- **File not found / unreadable:** Exit 1 with message.
- **Format auto-detection fails:** Suggest using `--format` flag explicitly.
- **CSV with missing required columns:** Error listing which columns are missing and what was found.
- **Markdown with no extractable headings:** Error suggesting `--llm` for unstructured extraction.
- **LLM API failure:** Retry with exponential backoff (3 attempts). If all fail, exit 1 with error.
- **LLM produces zero controls:** Error — "LLM extraction returned no controls. The document may not contain a structured framework."
- **Validation errors during prepare:** `prepare` always writes the output JSON even if validation issues exist, so the user can inspect and fix manually. Issues are printed to stderr with severity labels. `validate` and `ingest` treat errors as blocking (exit 1); warnings are informational only.

## 10. Testing Strategy

- **CSV extractor:** Fixtures with various column naming conventions, empty rows, quoted fields
- **Markdown extractor:** Fixtures with different heading levels, ID patterns, nested structure
- **JSON extractor:** Fixtures for all three cases (FrameworkOutput, array, nested)
- **LLM extractor:** Mock Claude API responses, chunking logic, deduplication
- **Validation:** Test every rule (errors and warnings) with minimal fixtures
- **Integration:** `prepare → validate → ingest` round-trip with a small test framework
- **Sanitization:** Verify `sanitize_text()` is called on all fields (null bytes, unicode, length)
