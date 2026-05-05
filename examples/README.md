# TRACT Examples

Sample framework documents and step-by-step tutorials for getting started with TRACT.

## Prerequisites

```bash
pip install -e "."        # Core install (prepare, validate)
pip install -e ".[llm]"   # Add LLM-assisted extraction (optional)
```

## Files

| File | Format | Description |
|------|--------|-------------|
| `sample_framework.csv` | CSV | 5 sample controls with `control_id`, `title`, `description` columns |
| `sample_framework.md` | Markdown | Same 5 controls as `## ID: Title` headings with body descriptions |

## Tutorial 1: Prepare from CSV

TRACT auto-detects CSV columns named `control_id`, `title`, and `description`.

```bash
tract prepare \
  --file examples/sample_framework.csv \
  --framework-id example_csv \
  --name "Example CSV Framework"
```

**Expected output:**
```
INFO  Detected CSV format with columns: control_id, title, description
INFO  Parsed 5 controls from examples/sample_framework.csv
INFO  Wrote example_csv_prepared.json (5 controls)
```

Then validate:

```bash
tract validate --file example_csv_prepared.json
```

**Expected output:**
```
INFO  Validating example_csv_prepared.json
INFO  Schema: PASS
INFO  Control IDs: 5 unique, 0 duplicates
INFO  Descriptions: 5 valid (min 42 chars, max 187 chars)
✓ Validation passed with 0 errors, 0 warnings
```

## Tutorial 2: Prepare from Markdown

TRACT splits on heading boundaries and extracts the ID from the heading text.

```bash
tract prepare \
  --file examples/sample_framework.md \
  --framework-id example_md \
  --name "Example Markdown Framework"
```

Then validate:

```bash
tract validate --file example_md_prepared.json
```

## Tutorial 3: What Happens Next

After `prepare` and `validate`, the next steps require the deployed model (see [Architecture](../docs/architecture.md) for model details):

```bash
# Ingest: embed controls, assign to CRE hubs, generate review file
tract ingest --file example_csv_prepared.json

# The review file shows proposed assignments:
# {
#   "control_id": "EX-001",
#   "proposed_hub": "646-285",
#   "proposed_hub_name": "Input validation",
#   "confidence": 0.82,
#   "alternatives": [...]
# }

# After human review, commit to the crosswalk database
tract accept --review example_csv_review.json

# Export your framework's crosswalk assignments
tract export --framework example_csv --format csv

# See which frameworks share CRE hubs with yours
tract compare --framework example_csv --framework mitre_atlas
```

## Further Reading

- [Framework Guide](../docs/framework-guide.md) — Complete walkthrough with both LLM-assisted and custom parser paths
- [CLI Reference](../docs/cli-reference.md) — Full reference for all 18 commands
- [Glossary](../docs/glossary.md) — Cross-domain term definitions
