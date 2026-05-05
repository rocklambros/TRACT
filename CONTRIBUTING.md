# Contributing to TRACT

Thank you for your interest in contributing to TRACT (Transitive Reconciliation and Assignment of CRE Taxonomies).

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Install development dependencies: `pip install -e ".[dev,phase0]"`
4. Run tests: `pytest tests/ -q`
5. Run type checking: `mypy tract/ scripts/phase1a/ --strict`
6. Run linting: `ruff check tract/ scripts/phase1a/ parsers/`

## Architecture Orientation

TRACT is a CLI tool that assigns security framework controls to CRE hubs using a fine-tuned bi-encoder. Here's where code lives:

```
tract/                  # Core library
├── cli.py              # CLI entry point (18 subcommands)
├── config.py           # All constants, paths, thresholds
├── schema.py           # Pydantic models (Control, FrameworkOutput)
├── hierarchy.py        # CRE hub hierarchy operations
├── inference.py        # Model loading and prediction
├── sanitize.py         # Text sanitization pipeline
├── validate.py         # Framework validation rules
├── io.py               # Atomic file I/O utilities
├── descriptions.py     # Hub description generation
├── accept.py           # Review acceptance logic
├── compare.py          # Framework comparison via shared hubs
├── parsers/base.py     # BaseParser abstract class
├── training/           # Contrastive fine-tuning pipeline
├── calibration/        # Temperature scaling, conformal prediction, OOD
├── bridge/             # AI↔traditional hub bridge analysis
├── crosswalk/          # Crosswalk database (SQLite)
├── export/             # Export formatters (CSV, JSON, OpenCRE)
├── prepare/            # LLM-assisted framework preparation
├── publish/            # HuggingFace publication
├── dataset/            # Dataset bundling and publication
├── proposals/          # Hub proposal generation
├── review/             # Expert review pipeline
└── active_learning/    # Active learning loop

parsers/                # Framework source parsers (one per framework)
├── parse_csa_aicm.py
├── parse_mitre_atlas.py
├── ... (12 parsers total)
└── validate_all.py

scripts/                # Phase execution scripts
├── phase0/             # Zero-shot baseline experiments
├── phase1a/            # Data infrastructure
├── phase1b/            # Model training
├── phase1c/            # Guardrails, calibration, crosswalk
└── analysis/           # Post-hoc analysis scripts

tests/                  # Test suite
├── fixtures/           # Test data files
├── test_parse_*.py     # Parser tests
├── test_schema.py      # Schema validation tests
├── test_sanitize.py    # Sanitization tests
└── ... (871 tests total)

data/
├── raw/                # Immutable source data (never modify after fetch)
├── processed/          # Parser and pipeline output
└── training/           # Training data (hub links)
```

## Three Contribution Tracks

### 1. New Parser (Easiest Entry Point)

Add support for a new security framework. You'll write:
- A parser file (`parsers/parse_<framework_id>.py`) — typically 30–80 lines
- A test with fixture data (`tests/test_parse_<framework_id>.py`)
- Raw source data in `data/raw/frameworks/<framework_id>/`

See the [Framework Guide](docs/framework-guide.md) for a complete walkthrough with annotated code examples.

### 2. Core Library Enhancement

Extend or improve `tract/` modules. Higher bar — requires understanding the architecture:
- Read [Architecture](docs/architecture.md) for the assignment paradigm and model pipeline
- Read the [CLI Reference](docs/cli-reference.md) for how modules connect to user-facing commands
- Look at existing tests for the module you're modifying

### 3. Evaluation / Analysis

Add analysis scripts or improve evaluation methodology:
- Scripts go in `scripts/analysis/`
- Must produce deterministic output (set random seeds)
- Results go in `results/` (gitignored, not committed)

## Your First Contribution

A parser is the easiest way to contribute. Here's a concrete example:

**1. Create your test fixture** (`tests/fixtures/my_framework/controls.json`):
```json
{
  "controls": [
    {"id": "MF-001", "title": "Data Validation", "description": "Validate all inputs..."},
    {"id": "MF-002", "title": "Access Control", "description": "Restrict access to..."}
  ]
}
```

**2. Write the test** (`tests/test_parse_my_framework.py`):
```python
from pathlib import Path
from parsers.parse_my_framework import MyFrameworkParser

def test_parser_output(tmp_path):
    fixture_dir = Path("tests/fixtures/my_framework")
    parser = MyFrameworkParser(raw_dir=fixture_dir, output_dir=tmp_path)
    output = parser.run()

    assert output.framework_id == "my_framework"
    assert len(output.controls) == 2
    assert output.controls[0].control_id == "MF-001"
    assert all(c.description for c in output.controls)
```

**3. Run the test** (it should fail — you haven't written the parser yet):
```bash
pytest tests/test_parse_my_framework.py -v
```

**4. Write the parser** (`parsers/parse_my_framework.py`):
```python
from __future__ import annotations
import json
import logging
from tract.parsers.base import BaseParser
from tract.schema import Control

logger = logging.getLogger(__name__)

class MyFrameworkParser(BaseParser):
    framework_id = "my_framework"
    framework_name = "My Framework"
    version = "1.0"
    source_url = "https://example.com"
    mapping_unit_level = "control"
    expected_count = 2

    def parse(self) -> list[Control]:
        with open(self.raw_dir / "controls.json", encoding="utf-8") as f:
            data = json.load(f)
        return [
            Control(
                control_id=c["id"],
                title=c["title"],
                description=c["description"],
                hierarchy_level="control",
            )
            for c in data["controls"]
        ]

if __name__ == "__main__":
    MyFrameworkParser().run()
```

**5. Run the test again** (should pass):
```bash
pytest tests/test_parse_my_framework.py -v
```

**6. Run the full suite:**
```bash
pytest tests/ -q
mypy tract/ scripts/phase1a/ --strict
ruff check tract/ scripts/phase1a/ parsers/
```

## Development Standards

Read `CLAUDE.md` for the full coding standards. Key points:

- **Type everything.** All function signatures fully typed. Return types always declared.
- **Validate at boundaries.** Pydantic models for structured data. No bare dicts for domain objects.
- **Fail loud.** `raise ValueError` with a specific message. No bare `except:`. No silent failures.
- **Deterministic output.** Sorted JSON keys. Pinned random seeds. Byte-identical re-runs.
- **No magic numbers.** Constants in `tract/config.py` with `ALL_CAPS` names.
- **Logging, not print.** Use the `logging` module. Never `print()` in library code.
- **Atomic writes.** Use `tract.io.atomic_write_json()` for all file output.
- **Tests first.** Write the test before the implementation. Tests use fixtures, not hardcoded paths.

## Making Changes

1. Create a feature branch from `main`
2. Write tests for your changes
3. Run the full test suite: `pytest tests/ -q`
4. Run type checking: `mypy tract/ scripts/phase1a/ --strict`
5. Run linting: `ruff check tract/ scripts/phase1a/ parsers/`
6. Commit with a clear message describing what and why
7. Open a pull request

## Pull Request Guidelines

- Keep PRs focused — one feature or fix per PR
- Include tests for new functionality
- Ensure all CI checks pass (ruff, mypy, tests, pip-audit)
- Update documentation if behavior changes
- Do not include credentials, API keys, or secrets

## Code Review

All PRs require passing CI before merge. The maintainer will review for:

- Adherence to coding standards in `CLAUDE.md`
- Test coverage for new code paths
- Security considerations (sanitization, no injection vectors)
- Consistency with existing patterns

## Reporting Issues

- **Bugs**: Use the [Bug Report](https://github.com/rocklambros/TRACT/issues/new?template=bug_report.yml) template
- **Features**: Use the [Feature Request](https://github.com/rocklambros/TRACT/issues/new?template=feature_request.yml) template
- **Security**: See [SECURITY.md](SECURITY.md) — do not open public issues for vulnerabilities

## License

By contributing, you agree that your contributions will be licensed under CC0 1.0 Universal.
