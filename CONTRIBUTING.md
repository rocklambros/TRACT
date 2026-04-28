# Contributing to TRACT

Thank you for your interest in contributing to TRACT (Transitive Reconciliation and Assignment of CRE Taxonomies).

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Install dependencies: `pip install -r requirements.txt`
4. Run tests: `pytest tests/ -q`
5. Run type checking: `mypy tract/ scripts/phase1a/ --strict`

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
5. Commit with a clear message describing what and why
6. Open a pull request

## Pull Request Guidelines

- Keep PRs focused — one feature or fix per PR
- Include tests for new functionality
- Ensure all CI checks pass (lint, mypy, tests, audit)
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

## Project Structure

```
tract/           # Core library (config, hierarchy, descriptions, sanitize, schema, I/O)
scripts/         # Phase scripts (phase0/, phase1a/)
parsers/         # Framework source parsers
tests/           # Test suite with fixtures
data/raw/        # Immutable source data (never modify after fetch)
data/processed/  # Parser and pipeline output
data/training/   # Training data (hub links, etc.)
```

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
