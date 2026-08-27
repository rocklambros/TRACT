# TRACT Data Preparation Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build all data preparation infrastructure (PRD Sections 4.5–4.8) so Phase 0 zero-shot baseline experiments can run cleanly.

**Architecture:** `tract/` Python package with pydantic v2 schema models and a `BaseParser` ABC (Template Method pattern). 12 framework parsers subclass `BaseParser` and implement a single `parse()` method. OpenCRE data fetched fresh from the live API. Hub links extracted into LOFO-ready training splits.

**Tech Stack:** Python 3.11+, pydantic v2, PyYAML, BeautifulSoup4 + lxml, requests

**Spec:** `docs/superpowers/specs/2026-04-28-data-preparation-pipeline-design.md`

**Parallelization:** Tasks within each Phase are independent and can run as parallel subagents. Phases must be sequential.

---

## Phase A: Foundation (Sequential)

### Task 1: Repository Scaffold

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: directory tree for `data/`, `parsers/`, `models/`, `scripts/`, `hub_proposals/`, `tests/`

- [ ] **Step 1: Initialize git repo**

```bash
cd /home/rock/github_projects/TRACT
git init
```

- [ ] **Step 2: Create .gitignore**

Create `.gitignore`:

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
*.pyc
.ruff_cache/
```

- [ ] **Step 3: Create pyproject.toml**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "tract"
version = "0.1.0"
description = "Transitive Reconciliation and Assignment of CRE Taxonomies"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0,<3.0",
    "pyyaml>=6.0",
    "beautifulsoup4>=4.12",
    "lxml>=5.0",
    "requests>=2.31",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "mypy>=1.10",
    "types-PyYAML",
    "types-beautifulsoup4",
    "types-requests",
]

[tool.mypy]
strict = true
python_version = "3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Create requirements.txt**

Create `requirements.txt`:

```
pydantic>=2.0,<3.0
pyyaml>=6.0
beautifulsoup4>=4.12
lxml>=5.0
requests>=2.31
```

- [ ] **Step 5: Create directory structure**

```bash
mkdir -p data/raw/opencre/pages
mkdir -p data/raw/frameworks/{aiuc_1,csa_aicm,mitre_atlas,nist_ai_rmf,nist_ai_600_1,owasp_ai_exchange,owasp_llm_top10,owasp_agentic_top10,owasp_dsgai,cosai,eu_gpai_cop,eu_ai_act}
mkdir -p data/processed/frameworks
mkdir -p data/training
mkdir -p parsers
mkdir -p models
mkdir -p scripts
mkdir -p hub_proposals
mkdir -p tests/fixtures
mkdir -p tract/parsers
```

Create placeholder files to preserve empty directories in git:

```bash
touch data/processed/.gitkeep
touch data/processed/frameworks/.gitkeep
touch data/training/.gitkeep
touch models/.gitkeep
touch scripts/.gitkeep
touch hub_proposals/.gitkeep
```

- [ ] **Step 6: Install the package in editable mode**

```bash
pip install -e ".[dev]"
```

- [ ] **Step 7: Commit**

```bash
git add .gitignore pyproject.toml requirements.txt data/ parsers/ models/ scripts/ hub_proposals/ tests/ tract/ docs/
git commit -m "scaffold: init repo with directory structure and project config"
```

---

### Task 2: tract/ Package — Schema Models

**Files:**
- Create: `tract/__init__.py`
- Create: `tract/schema.py`
- Create: `tests/test_schema.py`

- [ ] **Step 1: Create tract/__init__.py**

Create `tract/__init__.py`:

```python
"""TRACT — Transitive Reconciliation and Assignment of CRE Taxonomies."""
```

- [ ] **Step 2: Write failing tests for schema models**

Create `tests/test_schema.py`:

```python
"""Tests for tract.schema pydantic models."""
import pytest
from tract.schema import Control, FrameworkOutput, HubLink


class TestControl:
    def test_valid_control(self) -> None:
        c = Control(
            control_id="AICM-AIS-01",
            title="AI System Inventory",
            description="Organizations shall maintain a comprehensive inventory.",
        )
        assert c.control_id == "AICM-AIS-01"
        assert c.title == "AI System Inventory"
        assert c.full_text is None
        assert c.metadata is None

    def test_control_with_all_fields(self) -> None:
        c = Control(
            control_id="A001.1",
            title="Define input data usage policies",
            description="Define and communicate input data usage policies.",
            full_text="Define and communicate input data usage policies including how customer data is used.",
            hierarchy_level="activity",
            parent_id="A001",
            parent_name="Establish input data policy",
            metadata={"category": "Core", "domain": "Data & Privacy"},
        )
        assert c.hierarchy_level == "activity"
        assert c.parent_id == "A001"
        assert c.metadata == {"category": "Core", "domain": "Data & Privacy"}

    def test_description_max_length_enforced(self) -> None:
        with pytest.raises(Exception):
            Control(
                control_id="X",
                title="X",
                description="A" * 2001,
            )

    def test_empty_control_id_rejected(self) -> None:
        with pytest.raises(Exception):
            Control(control_id="", title="X", description="X")

    def test_empty_description_rejected(self) -> None:
        with pytest.raises(Exception):
            Control(control_id="X", title="X", description="")


class TestFrameworkOutput:
    def test_valid_output(self) -> None:
        controls = [
            Control(control_id="C1", title="T1", description="D1"),
            Control(control_id="C2", title="T2", description="D2"),
        ]
        out = FrameworkOutput(
            framework_id="test_fw",
            framework_name="Test Framework",
            version="1.0",
            source_url="https://example.com",
            fetched_date="2026-04-28",
            mapping_unit_level="control",
            controls=controls,
        )
        assert out.framework_id == "test_fw"
        assert len(out.controls) == 2

    def test_serialization_roundtrip(self) -> None:
        controls = [Control(control_id="C1", title="T1", description="D1")]
        out = FrameworkOutput(
            framework_id="test_fw",
            framework_name="Test Framework",
            version="1.0",
            source_url="https://example.com",
            fetched_date="2026-04-28",
            mapping_unit_level="control",
            controls=controls,
        )
        data = out.model_dump()
        out2 = FrameworkOutput.model_validate(data)
        assert out == out2


class TestHubLink:
    def test_valid_hub_link(self) -> None:
        link = HubLink(
            cre_id="462-09",
            cre_name="Prompt injection I/O handling",
            standard_name="MITRE ATLAS",
            section_id="AML.T0051",
            section_name="LLM Prompt Injection",
            link_type="LinkedTo",
            framework_id="mitre_atlas",
        )
        assert link.link_type == "LinkedTo"
        assert link.framework_id == "mitre_atlas"
```

- [ ] **Step 3: Run tests — verify they fail**

```bash
python -m pytest tests/test_schema.py -v
```

Expected: `ModuleNotFoundError: No module named 'tract.schema'`

- [ ] **Step 4: Implement schema models**

Create `tract/schema.py`:

```python
"""Pydantic v2 models for the TRACT standardized control schema (PRD Section 4.8)."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Control(BaseModel):
    """A single control/risk/measure from a security framework."""

    control_id: str = Field(min_length=1)
    title: str
    description: str = Field(min_length=1, max_length=2000)
    full_text: str | None = None
    hierarchy_level: str | None = None
    parent_id: str | None = None
    parent_name: str | None = None
    metadata: dict[str, str | list[str]] | None = None

    model_config = {"str_strip_whitespace": True}


class FrameworkOutput(BaseModel):
    """Complete parsed output for one framework — the parser output contract."""

    framework_id: str = Field(min_length=1)
    framework_name: str
    version: str
    source_url: str
    fetched_date: str
    mapping_unit_level: str
    controls: list[Control]

    @field_validator("controls")
    @classmethod
    def controls_not_empty(cls, v: list[Control]) -> list[Control]:
        if not v:
            raise ValueError("controls list must not be empty")
        return v


class HubLink(BaseModel):
    """A single standard-to-CRE-hub link extracted from OpenCRE data."""

    cre_id: str
    cre_name: str
    standard_name: str
    section_id: str
    section_name: str
    link_type: str
    framework_id: str
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
python -m pytest tests/test_schema.py -v
```

Expected: All 7 tests PASS.

---

### Task 3: tract/ Package — Text Sanitization

**Files:**
- Create: `tract/sanitize.py`
- Create: `tests/test_sanitize.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_sanitize.py`:

```python
"""Tests for tract.sanitize text processing pipeline."""
import pytest
from tract.sanitize import sanitize_text, strip_html


class TestStripHtml:
    def test_removes_tags(self) -> None:
        assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"

    def test_decodes_entities(self) -> None:
        assert strip_html("&amp; &lt; &gt; &sect;") == "& < > §"

    def test_plain_text_unchanged(self) -> None:
        assert strip_html("no html here") == "no html here"

    def test_nested_tags(self) -> None:
        result = strip_html('<a href="url">link <em>text</em></a>')
        assert result == "link text"


class TestSanitizeText:
    def test_strips_null_bytes(self) -> None:
        assert sanitize_text("hello\x00world") == "hello world"

    def test_normalizes_unicode_nfc(self) -> None:
        decomposed = "café"
        result = sanitize_text(decomposed)
        assert result == "café"

    def test_collapses_whitespace(self) -> None:
        assert sanitize_text("hello   \n\t  world") == "hello world"

    def test_strips_leading_trailing(self) -> None:
        assert sanitize_text("  hello  ") == "hello"

    def test_fixes_pdf_ligatures(self) -> None:
        assert sanitize_text("ofﬁcial efﬀort") == "official effort"

    def test_fixes_broken_hyphenation(self) -> None:
        assert sanitize_text("imple-\nmentation") == "implementation"

    def test_truncation_returns_tuple(self) -> None:
        long_text = "A" * 2500
        desc, full = sanitize_text(long_text, max_length=2000, return_full=True)
        assert len(desc) <= 2000
        assert full == "A" * 2500

    def test_no_truncation_when_short(self) -> None:
        short = "Hello world"
        desc, full = sanitize_text(short, max_length=2000, return_full=True)
        assert desc == "Hello world"
        assert full is None

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            sanitize_text("")
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest tests/test_sanitize.py -v
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement sanitize.py**

Create `tract/sanitize.py`:

```python
"""Text sanitization pipeline for framework control text.

Every text field passes through this before pydantic validation.
Pipeline: null bytes -> unicode NFC -> HTML decode -> strip tags ->
whitespace normalize -> PDF artifact fix -> length enforcement.
"""
from __future__ import annotations

import html
import re
import unicodedata

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_HYPHENATION_RE = re.compile(r"(\w)-\n(\w)")

_LIGATURE_MAP: dict[str, str] = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
}


def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub("", text)
    return text


def sanitize_text(
    text: str,
    *,
    max_length: int = 2000,
    return_full: bool = False,
) -> str | tuple[str, str | None]:
    """Run the full sanitization pipeline on a text field.

    Args:
        text: Raw text to sanitize.
        max_length: Maximum length for the description field.
        return_full: If True, return (description, full_text_or_none) tuple.

    Returns:
        Sanitized text string, or (description, full_text) tuple if return_full=True.

    Raises:
        ValueError: If text is empty after sanitization.
    """
    text = text.replace("\x00", " ")

    text = unicodedata.normalize("NFC", text)

    text = html.unescape(text)
    text = _HTML_TAG_RE.sub("", text)

    for ligature, replacement in _LIGATURE_MAP.items():
        text = text.replace(ligature, replacement)

    text = _HYPHENATION_RE.sub(r"\1\2", text)

    text = _WHITESPACE_RE.sub(" ", text).strip()

    if not text:
        raise ValueError("Text is empty after sanitization")

    if return_full:
        if len(text) > max_length:
            full_text = text
            description = text[:max_length].rsplit(" ", 1)[0]
            return description, full_text
        return text, None

    if len(text) > max_length:
        text = text[:max_length].rsplit(" ", 1)[0]

    return text
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python -m pytest tests/test_sanitize.py -v
```

Expected: All 9 tests PASS.

---

### Task 4: tract/ Package — Atomic I/O

**Files:**
- Create: `tract/io.py`
- Create: `tests/test_io.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_io.py`:

```python
"""Tests for tract.io atomic file operations."""
import json
from pathlib import Path
import pytest
from tract.io import atomic_write_json, load_json


class TestAtomicWriteJson:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        target = tmp_path / "out.json"
        data = {"b": 2, "a": 1}
        atomic_write_json(data, target)
        assert target.exists()
        loaded = json.loads(target.read_text(encoding="utf-8"))
        assert loaded == {"a": 1, "b": 2}

    def test_sorted_keys(self, tmp_path: Path) -> None:
        target = tmp_path / "out.json"
        data = {"z": 1, "a": 2, "m": 3}
        atomic_write_json(data, target)
        text = target.read_text(encoding="utf-8")
        keys_order = [k for k in json.loads(text)]
        assert keys_order == ["a", "m", "z"]

    def test_trailing_newline(self, tmp_path: Path) -> None:
        target = tmp_path / "out.json"
        atomic_write_json({"x": 1}, target)
        text = target.read_text(encoding="utf-8")
        assert text.endswith("\n")

    def test_unicode_preserved(self, tmp_path: Path) -> None:
        target = tmp_path / "out.json"
        atomic_write_json({"text": "café"}, target)
        text = target.read_text(encoding="utf-8")
        assert "café" in text
        assert "\\u" not in text

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        target = tmp_path / "out.json"
        atomic_write_json({"v": 1}, target)
        atomic_write_json({"v": 2}, target)
        loaded = load_json(target)
        assert loaded["v"] == 2

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        target = tmp_path / "sub" / "dir" / "out.json"
        atomic_write_json({"x": 1}, target)
        assert target.exists()


class TestLoadJson:
    def test_loads_utf8(self, tmp_path: Path) -> None:
        target = tmp_path / "in.json"
        target.write_text('{"key": "café"}', encoding="utf-8")
        data = load_json(target)
        assert data["key"] == "café"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "nonexistent.json")
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest tests/test_io.py -v
```

- [ ] **Step 3: Implement io.py**

Create `tract/io.py`:

```python
"""Atomic file I/O for deterministic, crash-safe JSON output."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(data: Any, path: Path) -> None:
    """Write JSON atomically: temp file + os.replace().

    Output is deterministic: sorted keys, indent=2, ensure_ascii=False,
    trailing newline. Parent directories are created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    content = json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False) + "\n"

    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except BaseException:
        os.unlink(tmp_path)
        raise


def load_json(path: Path) -> Any:
    """Load JSON with explicit UTF-8 encoding."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python -m pytest tests/test_io.py -v
```

Expected: All 7 tests PASS.

---

### Task 5: tract/ Package — Config and BaseParser

**Files:**
- Create: `tract/config.py`
- Create: `tract/parsers/__init__.py`
- Create: `tract/parsers/base.py`
- Create: `tests/test_base_parser.py`

- [ ] **Step 1: Create config.py**

Create `tract/config.py`:

```python
"""Project-wide paths, constants, and framework registry."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TRAINING_DIR = DATA_DIR / "training"
RAW_FRAMEWORKS_DIR = RAW_DIR / "frameworks"
RAW_OPENCRE_DIR = RAW_DIR / "opencre"
PROCESSED_FRAMEWORKS_DIR = PROCESSED_DIR / "frameworks"

DESCRIPTION_MAX_LENGTH = 2000

OPENCRE_API_BASE = "https://opencre.org/rest/v1/all_cres"
OPENCRE_PER_PAGE = 50
OPENCRE_RETRY_MAX = 5
OPENCRE_RETRY_BASE_DELAY = 1.0
OPENCRE_RETRY_MAX_DELAY = 30.0
OPENCRE_REQUEST_DELAY = 0.5

EXPECTED_COUNTS: dict[str, int | None] = {
    "csa_aicm": 243,
    "aiuc_1": 132,
    "mitre_atlas": 202,
    "cosai": 55,
    "nist_ai_rmf": 72,
    "nist_ai_600_1": 12,
    "owasp_ai_exchange": 88,
    "owasp_llm_top10": 10,
    "owasp_agentic_top10": 10,
    "eu_gpai_cop": 40,
    "owasp_dsgai": 21,
    "eu_ai_act": 100,
}

COUNT_TOLERANCE = 0.10

OPENCRE_FRAMEWORK_ID_MAP: dict[str, str] = {
    "CAPEC": "capec",
    "CWE": "cwe",
    "OWASP Cheat Sheet Series": "owasp_cheat_sheets",
    "NIST 800-53": "nist_800_53",
    "NIST SP 800-53": "nist_800_53",
    "ASVS": "asvs",
    "OWASP Application Security Verification Standard": "asvs",
    "DSOMM": "dsomm",
    "WSTG": "wstg",
    "OWASP Web Security Testing Guide": "wstg",
    "OWASP Proactive Controls": "owasp_proactive_controls",
    "ENISA": "enisa",
    "ETSI": "etsi",
    "SAMM": "samm",
    "OWASP SAMM": "samm",
    "Cloud Controls Matrix": "csa_ccm",
    "BIML": "biml",
    "MITRE ATLAS": "mitre_atlas",
    "OWASP AI Exchange": "owasp_ai_exchange",
    "NIST AI 100-2": "nist_ai_100_2",
    "OWASP Top10 for LLM": "owasp_llm_top10",
    "OWASP Top10 for ML": "owasp_ml_top10",
    "ISO 27001": "iso_27001",
    "NIST 800-63": "nist_800_63",
    "NIST SSDF": "nist_ssdf",
    "OWASP Top 10 2021": "owasp_top10_2021",
}
```

- [ ] **Step 2: Write failing test for BaseParser**

Create `tests/test_base_parser.py`:

```python
"""Tests for tract.parsers.base BaseParser ABC."""
from pathlib import Path
import json
import pytest
from tract.parsers.base import BaseParser
from tract.schema import Control


class StubParser(BaseParser):
    framework_id = "test_fw"
    framework_name = "Test Framework"
    version = "1.0"
    source_url = "https://example.com"
    mapping_unit_level = "control"
    expected_count = 2

    def parse(self) -> list[Control]:
        return [
            Control(control_id="C1", title="First", description="First control."),
            Control(control_id="C2", title="Second", description="Second control."),
        ]


class TestBaseParser:
    def test_run_produces_valid_output(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        out_dir = tmp_path / "processed"
        out_dir.mkdir()

        parser = StubParser(raw_dir=raw_dir, output_dir=out_dir)
        result = parser.run()

        assert result.framework_id == "test_fw"
        assert len(result.controls) == 2
        assert result.controls[0].control_id == "C1"

    def test_run_writes_json_file(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        out_dir = tmp_path / "processed"
        out_dir.mkdir()

        parser = StubParser(raw_dir=raw_dir, output_dir=out_dir)
        parser.run()

        output_file = out_dir / "test_fw.json"
        assert output_file.exists()
        data = json.loads(output_file.read_text(encoding="utf-8"))
        assert data["framework_id"] == "test_fw"
        assert len(data["controls"]) == 2

    def test_sanitizes_text_fields(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        out_dir = tmp_path / "processed"
        out_dir.mkdir()

        class DirtyParser(BaseParser):
            framework_id = "dirty"
            framework_name = "Dirty"
            version = "1.0"
            source_url = "https://example.com"
            mapping_unit_level = "control"
            expected_count = 1

            def parse(self) -> list[Control]:
                return [Control(
                    control_id="D1",
                    title="Test",
                    description="Hello\x00world  with   spaces",
                )]

        parser = DirtyParser(raw_dir=raw_dir, output_dir=out_dir)
        result = parser.run()
        assert "\x00" not in result.controls[0].description
        assert "  " not in result.controls[0].description

    def test_count_mismatch_warns(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        out_dir = tmp_path / "processed"
        out_dir.mkdir()

        class WrongCountParser(BaseParser):
            framework_id = "wrong"
            framework_name = "Wrong"
            version = "1.0"
            source_url = "https://example.com"
            mapping_unit_level = "control"
            expected_count = 100

            def parse(self) -> list[Control]:
                return [Control(control_id="C1", title="T", description="D")]

        parser = WrongCountParser(raw_dir=raw_dir, output_dir=out_dir)
        import logging
        with caplog.at_level(logging.WARNING):
            parser.run()
        assert "expected 100" in caplog.text.lower() or "Expected 100" in caplog.text
```

- [ ] **Step 3: Run tests — verify they fail**

```bash
python -m pytest tests/test_base_parser.py -v
```

- [ ] **Step 4: Implement BaseParser**

Create `tract/parsers/__init__.py`:

```python
"""TRACT parser infrastructure."""
```

Create `tract/parsers/base.py`:

```python
"""BaseParser ABC — the Template Method contract for all framework parsers.

Every concrete parser subclasses BaseParser, sets metadata class attributes,
and implements parse() -> list[Control]. The invariant pipeline (sanitize,
validate, write) is handled by run() and must not be overridden.
"""
from __future__ import annotations

import datetime
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from tract.config import COUNT_TOLERANCE, DESCRIPTION_MAX_LENGTH
from tract.io import atomic_write_json
from tract.sanitize import sanitize_text
from tract.schema import Control, FrameworkOutput

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """Abstract base for all framework parsers."""

    framework_id: str
    framework_name: str
    version: str
    source_url: str
    mapping_unit_level: str
    expected_count: int | None = None

    def __init__(self, raw_dir: Path, output_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.output_dir = output_dir

    @abstractmethod
    def parse(self) -> list[Control]:
        """Extract controls from raw source files. Subclasses implement this."""

    def run(self) -> FrameworkOutput:
        """Invariant pipeline: parse -> sanitize -> validate -> write."""
        logger.info("Parsing %s from %s", self.framework_id, self.raw_dir)

        controls = self.parse()
        logger.info("Extracted %d controls from %s", len(controls), self.framework_id)

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

        output_path = self.output_dir / f"{self.framework_id}.json"
        atomic_write_json(output.model_dump(), output_path)
        logger.info("Wrote %s", output_path)

        return output

    def _sanitize_control(self, control: Control) -> Control:
        """Apply text sanitization to description and full_text fields."""
        desc_result = sanitize_text(
            control.description,
            max_length=DESCRIPTION_MAX_LENGTH,
            return_full=True,
        )
        assert isinstance(desc_result, tuple)
        description, full_text = desc_result

        existing_full = control.full_text
        if existing_full is not None:
            existing_full = sanitize_text(existing_full)
            assert isinstance(existing_full, str)
        elif full_text is not None:
            existing_full = full_text

        title = sanitize_text(control.title)
        assert isinstance(title, str)

        return control.model_copy(update={
            "description": description,
            "full_text": existing_full,
            "title": title,
        })

    def _check_expected_count(self, output: FrameworkOutput) -> None:
        """Warn if control count is outside expected tolerance."""
        if self.expected_count is None:
            return
        actual = len(output.controls)
        low = int(self.expected_count * (1 - COUNT_TOLERANCE))
        high = int(self.expected_count * (1 + COUNT_TOLERANCE))
        if not (low <= actual <= high):
            logger.warning(
                "%s: got %d controls, expected %d (tolerance %d–%d)",
                self.framework_id, actual, self.expected_count, low, high,
            )

    @staticmethod
    def _today() -> str:
        return datetime.date.today().isoformat()
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
python -m pytest tests/test_base_parser.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 6: Run all tests + type check**

```bash
python -m pytest tests/ -v
mypy tract/ --strict
```

- [ ] **Step 7: Commit**

```bash
git add tract/ tests/
git commit -m "feat: add tract/ package with schema, sanitize, io, config, and BaseParser"
```

---

## Phase B: Data Population (Parallel)

### Task 6: Copy Raw Framework Files

**Files:**
- Populate: `data/raw/frameworks/` (12 framework directories)

- [ ] **Step 1: Copy Tier 1 (structured JSON) sources**

```bash
OLD=/home/rock/github_projects/ai-security-framework-crosswalk/data/frameworks
RAW=/home/rock/github_projects/TRACT/data/raw/frameworks

cp "$OLD/csa-aicm/csa_aicm.json" "$RAW/csa_aicm/"
cp "$OLD/aiuc-1/aiuc-1-standard.json" "$RAW/aiuc_1/"
cp "$OLD/mitre-atlas/ATLAS_compiled.json" "$RAW/mitre_atlas/"
```

- [ ] **Step 2: Copy Tier 2 (YAML) sources**

```bash
cp "$OLD/cosai/risk-map/risks.yaml" "$RAW/cosai/"
cp "$OLD/cosai/risk-map/controls.yaml" "$RAW/cosai/"
cp "$OLD/cosai/risk-map/components.yaml" "$RAW/cosai/"
cp -r "$OLD/cosai/risk-map/schemas/" "$RAW/cosai/schemas/"
```

- [ ] **Step 3: Copy Tier 3 (markdown) sources**

```bash
cp "$OLD/nist-ai-rmf/nist_ai_rmf_1.0.md" "$RAW/nist_ai_rmf/"
cp "$OLD/nist-ai-600-1/nist_ai_600_1.md" "$RAW/nist_ai_600_1/"
cp "$OLD/owasp-ai-exchange/src_1_general_controls.md" "$RAW/owasp_ai_exchange/"
cp "$OLD/owasp-ai-exchange/src_2_threats_through_use.md" "$RAW/owasp_ai_exchange/"
cp "$OLD/owasp-ai-exchange/src_3_development_time_threats.md" "$RAW/owasp_ai_exchange/"
cp "$OLD/owasp-ai-exchange/src_4_runtime_application_security_threats.md" "$RAW/owasp_ai_exchange/"
cp "$OLD/owasp-llm-top10/owasp_llm_top_10_2025.md" "$RAW/owasp_llm_top10/"
cp "$OLD/owasp-agentic-top10/owasp_agentic_top10_2026.md" "$RAW/owasp_agentic_top10/"
cp "$OLD/eu-gpai-code-of-practice/gpai_code_of_practice_combined.md" "$RAW/eu_gpai_cop/"
```

- [ ] **Step 4: Copy Tier 4–5 (TXT, HTML) sources**

```bash
cp "$OLD/owasp-dsgai/MANIFEST.json" "$RAW/owasp_dsgai/"
cp "$OLD/owasp-dsgai/OWASP-GenAI-Data-Security-Risks-and-Mitigations-2026-v1.0.txt" "$RAW/owasp_dsgai/"
cp "$OLD/eu-ai-act/MANIFEST.json" "$RAW/eu_ai_act/"
cp "$OLD/eu-ai-act/eu_ai_act_2024_1689.html" "$RAW/eu_ai_act/"
```

- [ ] **Step 5: Verify file counts**

```bash
find data/raw/frameworks/ -type f | wc -l
```

Expected: ~22-25 files across 12 directories.

- [ ] **Step 6: Commit (raw data is in .gitignore — this is a no-op commit note)**

Raw data is in `.gitignore` so it won't be committed. This is correct per the spec — `data/raw/` is immutable local state. Log the copy for traceability:

```bash
echo "Raw framework files copied from ai-security-framework-crosswalk on $(date -I)" > data/raw/PROVENANCE.txt
git add data/raw/PROVENANCE.txt
git commit -m "data: document raw framework file provenance"
```

---

### Task 7: Fetch OpenCRE Data

**Files:**
- Create: `parsers/fetch_opencre.py`
- Create: `tests/test_fetch_opencre.py`

- [ ] **Step 1: Write fetch_opencre.py**

Create `parsers/fetch_opencre.py`:

```python
"""Fetch all CRE data from the OpenCRE API with retry and resumability.

Paginates through opencre.org/rest/v1/all_cres, saving each page individually
for resumability. Merges into a single opencre_all_cres.json on completion.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from tract.config import (
    OPENCRE_API_BASE,
    OPENCRE_PER_PAGE,
    OPENCRE_REQUEST_DELAY,
    OPENCRE_RETRY_BASE_DELAY,
    OPENCRE_RETRY_MAX,
    OPENCRE_RETRY_MAX_DELAY,
    RAW_OPENCRE_DIR,
)
from tract.io import atomic_write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PAGES_DIR = RAW_OPENCRE_DIR / "pages"


def fetch_page(page: int, session: requests.Session) -> list[dict[str, object]]:
    """Fetch a single page with exponential backoff retry."""
    url = f"{OPENCRE_API_BASE}?per_page={OPENCRE_PER_PAGE}&page={page}"
    delay = OPENCRE_RETRY_BASE_DELAY

    for attempt in range(1, OPENCRE_RETRY_MAX + 1):
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                raise ValueError(f"Page {page}: expected list, got {type(data).__name__}")
            return data
        except (requests.RequestException, ValueError) as e:
            if attempt == OPENCRE_RETRY_MAX:
                raise
            logger.warning("Page %d attempt %d failed: %s. Retrying in %.1fs", page, attempt, e, delay)
            time.sleep(delay)
            delay = min(delay * 2, OPENCRE_RETRY_MAX_DELAY)

    raise RuntimeError("unreachable")


def fetch_all() -> None:
    """Fetch all CRE pages, saving each individually for resumability."""
    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "TRACT/0.1.0 (security-framework-crosswalk research)"

    page = 1
    all_cres: list[dict[str, object]] = []
    empty_pages = 0

    while True:
        page_file = PAGES_DIR / f"page_{page:03d}.json"

        if page_file.exists():
            logger.debug("Page %d already fetched, loading from cache", page)
            with open(page_file, encoding="utf-8") as f:
                page_data = json.load(f)
        else:
            page_data = fetch_page(page, session)
            atomic_write_json(page_data, page_file)
            time.sleep(OPENCRE_REQUEST_DELAY)

        if not page_data:
            empty_pages += 1
            if empty_pages >= 2:
                logger.info("Two consecutive empty pages at page %d — done.", page)
                break
        else:
            empty_pages = 0
            all_cres.extend(page_data)

        if page % 10 == 0:
            logger.info("Fetched %d pages, %d CREs so far", page, len(all_cres))

        page += 1

    total_pages = page - 1
    logger.info("Fetch complete: %d pages, %d CREs", total_pages, len(all_cres))

    output = {
        "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_pages": total_pages,
        "total_cres": len(all_cres),
        "cres": all_cres,
    }
    output_path = RAW_OPENCRE_DIR / "opencre_all_cres.json"
    atomic_write_json(output, output_path)
    logger.info("Wrote %s (%d CREs)", output_path, len(all_cres))


if __name__ == "__main__":
    fetch_all()
```

- [ ] **Step 2: Write a basic test for the fetcher**

Create `tests/test_fetch_opencre.py`:

```python
"""Tests for parsers/fetch_opencre.py — uses mock to avoid live API calls."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_fetch_page_retry_on_failure() -> None:
    from parsers.fetch_opencre import fetch_page

    session = MagicMock()
    response_fail = MagicMock()
    response_fail.raise_for_status.side_effect = Exception("503")
    response_ok = MagicMock()
    response_ok.raise_for_status.return_value = None
    response_ok.json.return_value = [{"id": "123-45", "name": "Test CRE"}]

    session.get.side_effect = [response_fail, response_ok]

    with patch("parsers.fetch_opencre.time.sleep"):
        result = fetch_page(1, session)

    assert len(result) == 1
    assert result[0]["id"] == "123-45"


def test_fetch_all_writes_output(tmp_path: Path) -> None:
    from parsers import fetch_opencre

    page1_data = [{"id": "001-01", "name": "CRE1", "links": []}]
    page2_data: list[dict[str, object]] = []

    call_count = 0

    def mock_fetch_page(page: int, session: object) -> list[dict[str, object]]:
        nonlocal call_count
        call_count += 1
        if page == 1:
            return page1_data
        return page2_data

    with (
        patch.object(fetch_opencre, "fetch_page", side_effect=mock_fetch_page),
        patch.object(fetch_opencre, "PAGES_DIR", tmp_path / "pages"),
        patch.object(fetch_opencre, "RAW_OPENCRE_DIR", tmp_path),
        patch("parsers.fetch_opencre.time.sleep"),
    ):
        (tmp_path / "pages").mkdir()
        fetch_opencre.fetch_all()

    output = tmp_path / "opencre_all_cres.json"
    assert output.exists()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["total_cres"] == 1
    assert data["cres"][0]["id"] == "001-01"
```

- [ ] **Step 3: Run tests — verify they pass**

```bash
PYTHONPATH=. python -m pytest tests/test_fetch_opencre.py -v
```

- [ ] **Step 4: Run the actual fetch (live API call)**

```bash
python parsers/fetch_opencre.py
```

This takes ~5-10 minutes. Monitor progress via INFO log messages every 10 pages.

- [ ] **Step 5: Verify the fetch**

```bash
python3 -c "
import json
d = json.load(open('data/raw/opencre/opencre_all_cres.json'))
print(f'Total CREs: {d[\"total_cres\"]}')
print(f'Total pages: {d[\"total_pages\"]}')
print(f'Fetch timestamp: {d[\"fetch_timestamp\"]}')
"
```

Expected: ~522 CREs, ~261 pages.

- [ ] **Step 6: Commit**

```bash
git add parsers/fetch_opencre.py tests/test_fetch_opencre.py
git commit -m "feat: add OpenCRE API fetcher with retry and resumability"
```

---

## Phase C: Tier 1 Parsers (Parallel — 3 subagents)

Each parser task below is independent and can run as a separate subagent.

### Task 8: Parser — CSA AICM

**Files:**
- Create: `parsers/parse_csa_aicm.py`
- Create: `tests/test_parse_csa_aicm.py`

- [ ] **Step 1: Write test with fixture**

Create `tests/fixtures/csa_aicm_sample.json`:

```json
{
  "metadata": {"source": "https://cloudsecurityalliance.org/artifacts/ai-controls-matrix"},
  "domains": [{"id": "A&A", "name": "Audit & Assurance"}],
  "controls": [
    {
      "id": "A&A-01",
      "domain": "Audit & Assurance",
      "domain_full": "Audit & Assurance - A&A",
      "title": "Audit and Assurance Policy",
      "specification": "Establish, document, approve, communicate, apply, evaluate and maintain audit and assurance policies.",
      "control_type": "Cloud & AI Related",
      "ownership": {"GenAI OPS/PI": "Shared"},
      "architectural_relevance": ["Physical", "Network"],
      "lifecycle_relevance": {"Preparation": "Data collection"},
      "threat_categories": [],
      "cross_references": [],
      "implementation_guidelines": "Implement audit controls.",
      "auditing_guidelines": "Review audit logs.",
      "caiq_questions": ["Is there an audit policy?"]
    },
    {
      "id": "A&A-02",
      "domain": "Audit & Assurance",
      "domain_full": "Audit & Assurance - A&A",
      "title": "Independent Assessments",
      "specification": "Conduct independent assessments at planned intervals.",
      "control_type": "Cloud & AI Related",
      "ownership": {},
      "architectural_relevance": [],
      "lifecycle_relevance": {},
      "threat_categories": [],
      "cross_references": [],
      "implementation_guidelines": "Schedule assessments.",
      "auditing_guidelines": "Verify assessment results.",
      "caiq_questions": []
    }
  ]
}
```

Create `tests/test_parse_csa_aicm.py`:

```python
"""Tests for parsers/parse_csa_aicm.py."""
import json
from pathlib import Path

from parsers.parse_csa_aicm import CsaAicmParser


def test_parses_sample_fixture(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/csa_aicm_sample.json")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    import shutil
    shutil.copy(fixture, raw_dir / "csa_aicm.json")

    out_dir = tmp_path / "processed"
    out_dir.mkdir()

    parser = CsaAicmParser(raw_dir=raw_dir, output_dir=out_dir)
    result = parser.run()

    assert result.framework_id == "csa_aicm"
    assert len(result.controls) == 2
    assert result.controls[0].control_id == "A&A-01"
    assert result.controls[0].title == "Audit and Assurance Policy"
    assert "audit and assurance policies" in result.controls[0].description.lower()
    assert result.controls[0].parent_id == "A&A"
    assert result.controls[0].metadata is not None
    assert result.controls[0].metadata["control_type"] == "Cloud & AI Related"

    output_file = out_dir / "csa_aicm.json"
    assert output_file.exists()
```

- [ ] **Step 2: Run test — verify it fails**

```bash
PYTHONPATH=. python -m pytest tests/test_parse_csa_aicm.py -v
```

- [ ] **Step 3: Implement the parser**

Create `parsers/parse_csa_aicm.py`:

```python
"""Parser for CSA AI Controls Matrix (AICM) — Tier 1 structured JSON."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from tract.config import PROCESSED_FRAMEWORKS_DIR, RAW_FRAMEWORKS_DIR
from tract.io import load_json
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class CsaAicmParser(BaseParser):
    framework_id = "csa_aicm"
    framework_name = "CSA AI Controls Matrix"
    version = "1.0.3"
    source_url = "https://cloudsecurityalliance.org/artifacts/ai-controls-matrix"
    mapping_unit_level = "control"
    expected_count = 243

    def parse(self) -> list[Control]:
        data = load_json(self.raw_dir / "csa_aicm.json")
        controls: list[Control] = []

        for raw_ctrl in data["controls"]:
            domain_prefix = raw_ctrl["domain"].split(" - ")[-1] if " - " in raw_ctrl.get("domain_full", "") else raw_ctrl.get("domain", "")

            full_parts = [raw_ctrl.get("specification", "")]
            if raw_ctrl.get("implementation_guidelines"):
                full_parts.append(raw_ctrl["implementation_guidelines"])
            if raw_ctrl.get("auditing_guidelines"):
                full_parts.append(raw_ctrl["auditing_guidelines"])
            full_text = "\n\n".join(p for p in full_parts if p) if len(full_parts) > 1 else None

            controls.append(Control(
                control_id=raw_ctrl["id"],
                title=raw_ctrl["title"],
                description=raw_ctrl["specification"],
                full_text=full_text,
                hierarchy_level="control",
                parent_id=domain_prefix,
                parent_name=raw_ctrl.get("domain", ""),
                metadata={
                    "control_type": raw_ctrl.get("control_type", ""),
                    "domain": raw_ctrl.get("domain", ""),
                },
            ))

        return controls


if __name__ == "__main__":
    parser = CsaAicmParser(
        raw_dir=RAW_FRAMEWORKS_DIR / "csa_aicm",
        output_dir=PROCESSED_FRAMEWORKS_DIR,
    )
    parser.run()
```

- [ ] **Step 4: Run test — verify it passes**

```bash
PYTHONPATH=. python -m pytest tests/test_parse_csa_aicm.py -v
```

- [ ] **Step 5: Run parser on real data**

```bash
python parsers/parse_csa_aicm.py
python3 -c "import json; d=json.load(open('data/processed/frameworks/csa_aicm.json')); print(f'Controls: {len(d[\"controls\"])}')"
```

Expected: 243 controls.

---

### Task 9: Parser — AIUC-1

**Files:**
- Create: `parsers/parse_aiuc_1.py`
- Create: `tests/test_parse_aiuc_1.py`
- Create: `tests/fixtures/aiuc_1_sample.json`

- [ ] **Step 1: Write test with fixture**

Create `tests/fixtures/aiuc_1_sample.json`:

```json
{
  "standard": "AIUC-1",
  "version": "1.0",
  "url": "https://www.aiuc-1.com",
  "scraped_at": "2026-04-05",
  "domains": [
    {
      "id": "A",
      "name": "Data & Privacy",
      "url": "https://www.aiuc-1.com/domains/a",
      "description": "Data & Privacy domain",
      "controls": [
        {
          "id": "A001",
          "title": "Establish input data policy",
          "url": "https://www.aiuc-1.com/controls/a001",
          "classification": "critical",
          "type": "Directive",
          "frequency": "Annual",
          "description": "Ensure policies for input data usage.",
          "applicable_capabilities": ["NLP", "Vision"],
          "activities": [
            {
              "id": "A001.1",
              "description": "Define and communicate input data usage policies.",
              "category": "Core",
              "evidence_types": ["Terms of Service", "Privacy Policy"]
            },
            {
              "id": "A001.2",
              "description": "Implement technical controls to enforce data retention.",
              "category": "Advanced",
              "evidence_types": ["Automated deletion implementation"]
            }
          ],
          "keywords": ["data", "privacy"],
          "framework_references": []
        }
      ]
    }
  ]
}
```

Create `tests/test_parse_aiuc_1.py`:

```python
"""Tests for parsers/parse_aiuc_1.py."""
import shutil
from pathlib import Path

from parsers.parse_aiuc_1 import Aiuc1Parser


def test_parses_sample_fixture(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    shutil.copy("tests/fixtures/aiuc_1_sample.json", raw_dir / "aiuc-1-standard.json")

    out_dir = tmp_path / "processed"
    out_dir.mkdir()

    parser = Aiuc1Parser(raw_dir=raw_dir, output_dir=out_dir)
    result = parser.run()

    assert result.framework_id == "aiuc_1"
    assert len(result.controls) == 2
    assert result.controls[0].control_id == "A001.1"
    assert result.controls[0].parent_id == "A001"
    assert result.controls[0].parent_name == "Establish input data policy"
    assert result.controls[0].hierarchy_level == "activity"
    assert result.controls[0].metadata is not None
    assert result.controls[0].metadata["category"] == "Core"
    assert result.controls[0].metadata["domain"] == "Data & Privacy"
```

- [ ] **Step 2: Implement the parser**

Create `parsers/parse_aiuc_1.py`:

```python
"""Parser for AIUC-1 Standard — Tier 1 structured JSON."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from tract.config import PROCESSED_FRAMEWORKS_DIR, RAW_FRAMEWORKS_DIR
from tract.io import load_json
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class Aiuc1Parser(BaseParser):
    framework_id = "aiuc_1"
    framework_name = "AIUC-1 Standard"
    version = "1.0"
    source_url = "https://www.aiuc-1.com"
    mapping_unit_level = "activity"
    expected_count = 132

    def parse(self) -> list[Control]:
        data = load_json(self.raw_dir / "aiuc-1-standard.json")
        controls: list[Control] = []

        for domain in data["domains"]:
            domain_name = domain["name"]
            for ctrl in domain["controls"]:
                ctrl_id = ctrl["id"]
                ctrl_title = ctrl["title"]
                for activity in ctrl.get("activities", []):
                    controls.append(Control(
                        control_id=activity["id"],
                        title=ctrl_title,
                        description=activity["description"],
                        hierarchy_level="activity",
                        parent_id=ctrl_id,
                        parent_name=ctrl_title,
                        metadata={
                            "category": activity.get("category", ""),
                            "domain": domain_name,
                            "evidence_types": activity.get("evidence_types", []),
                        },
                    ))

        return controls


if __name__ == "__main__":
    parser = Aiuc1Parser(
        raw_dir=RAW_FRAMEWORKS_DIR / "aiuc_1",
        output_dir=PROCESSED_FRAMEWORKS_DIR,
    )
    parser.run()
```

- [ ] **Step 3: Run test + real data**

```bash
PYTHONPATH=. python -m pytest tests/test_parse_aiuc_1.py -v
python parsers/parse_aiuc_1.py
python3 -c "import json; d=json.load(open('data/processed/frameworks/aiuc_1.json')); print(f'Activities: {len(d[\"controls\"])}')"
```

Expected: 132 activities.

---

### Task 10: Parser — MITRE ATLAS

**Files:**
- Create: `parsers/parse_mitre_atlas.py`
- Create: `tests/test_parse_mitre_atlas.py`
- Create: `tests/fixtures/mitre_atlas_sample.json`

- [ ] **Step 1: Write test with fixture**

Create `tests/fixtures/mitre_atlas_sample.json`:

```json
{
  "id": "ATLAS",
  "name": "ATLAS",
  "version": "4.6.1",
  "matrices": [
    {
      "id": "ATLAS",
      "name": "ATLAS Matrix",
      "tactics": [{"id": "AML.TA0000", "name": "Reconnaissance"}],
      "techniques": [
        {
          "id": "AML.T0000",
          "name": "Search Open Technical Databases",
          "description": "Adversaries may search publicly available research.",
          "tactics": ["AML.TA0000"],
          "subtechniques": []
        },
        {
          "id": "AML.T0001",
          "name": "Acquire Infrastructure",
          "description": "Adversaries may buy or compromise infrastructure.",
          "tactics": ["AML.TA0000"],
          "subtechniques": [
            {
              "id": "AML.T0001.001",
              "name": "ML Development Workspaces",
              "description": "Acquire access to ML development workspaces."
            }
          ]
        }
      ],
      "mitigations": [
        {
          "id": "AML.M0000",
          "name": "Limit Adversary Knowledge",
          "description": "Limit public exposure of technical details."
        }
      ]
    }
  ],
  "case-studies": []
}
```

Create `tests/test_parse_mitre_atlas.py`:

```python
"""Tests for parsers/parse_mitre_atlas.py."""
import shutil
from pathlib import Path

from parsers.parse_mitre_atlas import MitreAtlasParser


def test_parses_sample_fixture(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    shutil.copy("tests/fixtures/mitre_atlas_sample.json", raw_dir / "ATLAS_compiled.json")

    out_dir = tmp_path / "processed"
    out_dir.mkdir()

    parser = MitreAtlasParser(raw_dir=raw_dir, output_dir=out_dir)
    result = parser.run()

    assert result.framework_id == "mitre_atlas"
    ids = [c.control_id for c in result.controls]
    assert "AML.T0000" in ids
    assert "AML.T0001" in ids
    assert "AML.T0001.001" in ids
    assert "AML.M0000" in ids
    assert len(result.controls) == 4

    tech = next(c for c in result.controls if c.control_id == "AML.T0000")
    assert tech.hierarchy_level == "technique"

    mit = next(c for c in result.controls if c.control_id == "AML.M0000")
    assert mit.hierarchy_level == "mitigation"

    sub = next(c for c in result.controls if c.control_id == "AML.T0001.001")
    assert sub.hierarchy_level == "sub-technique"
    assert sub.parent_id == "AML.T0001"
```

- [ ] **Step 2: Implement the parser**

Create `parsers/parse_mitre_atlas.py`:

```python
"""Parser for MITRE ATLAS — Tier 1 structured JSON."""
from __future__ import annotations

import logging
from pathlib import Path

from tract.config import PROCESSED_FRAMEWORKS_DIR, RAW_FRAMEWORKS_DIR
from tract.io import load_json
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class MitreAtlasParser(BaseParser):
    framework_id = "mitre_atlas"
    framework_name = "MITRE ATLAS"
    version = "4.6.1"
    source_url = "https://atlas.mitre.org"
    mapping_unit_level = "technique"
    expected_count = 202

    def parse(self) -> list[Control]:
        data = load_json(self.raw_dir / "ATLAS_compiled.json")
        matrix = data["matrices"][0]
        controls: list[Control] = []

        for tech in matrix["techniques"]:
            tactic_names = [t.get("name", t) if isinstance(t, dict) else t for t in tech.get("tactics", [])]
            controls.append(Control(
                control_id=tech["id"],
                title=tech["name"],
                description=tech["description"],
                hierarchy_level="technique",
                metadata={"tactics": tactic_names} if tactic_names else None,
            ))
            for sub in tech.get("subtechniques", []):
                controls.append(Control(
                    control_id=sub["id"],
                    title=sub["name"],
                    description=sub["description"],
                    hierarchy_level="sub-technique",
                    parent_id=tech["id"],
                    parent_name=tech["name"],
                ))

        for mit in matrix["mitigations"]:
            controls.append(Control(
                control_id=mit["id"],
                title=mit["name"],
                description=mit["description"],
                hierarchy_level="mitigation",
            ))

        return controls


if __name__ == "__main__":
    parser = MitreAtlasParser(
        raw_dir=RAW_FRAMEWORKS_DIR / "mitre_atlas",
        output_dir=PROCESSED_FRAMEWORKS_DIR,
    )
    parser.run()
```

- [ ] **Step 3: Run test + real data**

```bash
PYTHONPATH=. python -m pytest tests/test_parse_mitre_atlas.py -v
python parsers/parse_mitre_atlas.py
python3 -c "import json; d=json.load(open('data/processed/frameworks/mitre_atlas.json')); print(f'Techniques+mitigations: {len(d[\"controls\"])}')"
```

Expected: ~202 entries (167 techniques + sub-techniques + 35 mitigations).

- [ ] **Step 4: Commit all Tier 1 parsers**

```bash
git add parsers/parse_csa_aicm.py parsers/parse_aiuc_1.py parsers/parse_mitre_atlas.py tests/test_parse_*.py tests/fixtures/
git commit -m "feat: add Tier 1 parsers (CSA AICM, AIUC-1, MITRE ATLAS)"
```

---

## Phase D: Tier 2–3 Parsers (Parallel — 7 subagents)

### Task 11: Parser — CoSAI

**Files:**
- Create: `parsers/parse_cosai.py`
- Create: `tests/test_parse_cosai.py`

- [ ] **Step 1: Write test**

Create `tests/test_parse_cosai.py`:

```python
"""Tests for parsers/parse_cosai.py."""
from pathlib import Path
from parsers.parse_cosai import CosaiParser


def test_parses_real_data() -> None:
    from tract.config import RAW_FRAMEWORKS_DIR, PROCESSED_FRAMEWORKS_DIR
    raw_dir = RAW_FRAMEWORKS_DIR / "cosai"
    if not (raw_dir / "controls.yaml").exists():
        import pytest
        pytest.skip("Raw data not available")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        parser = CosaiParser(raw_dir=raw_dir, output_dir=out_dir)
        result = parser.run()

        assert result.framework_id == "cosai"
        assert len(result.controls) >= 50
        control_ids = [c.control_id for c in result.controls]
        levels = {c.hierarchy_level for c in result.controls}
        assert "control" in levels
        assert "risk" in levels
        assert all(cid for cid in control_ids)
```

- [ ] **Step 2: Implement the parser**

Create `parsers/parse_cosai.py`:

```python
"""Parser for CoSAI Risk Map — Tier 2 YAML -> JSON."""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

from tract.config import PROCESSED_FRAMEWORKS_DIR, RAW_FRAMEWORKS_DIR
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class CosaiParser(BaseParser):
    framework_id = "cosai"
    framework_name = "CoSAI Landscape of AI Security Risk Map"
    version = "1.0"
    source_url = "https://cosai.dev"
    mapping_unit_level = "control"
    expected_count = 55

    def parse(self) -> list[Control]:
        controls: list[Control] = []

        with open(self.raw_dir / "controls.yaml", encoding="utf-8") as f:
            ctrl_data = yaml.safe_load(f)
        for ctrl in ctrl_data.get("controls", []):
            desc_parts = ctrl.get("description", [])
            description = " ".join(desc_parts) if isinstance(desc_parts, list) else str(desc_parts)
            controls.append(Control(
                control_id=ctrl["id"],
                title=ctrl["title"],
                description=description,
                hierarchy_level="control",
                metadata={
                    "category": ctrl.get("category", ""),
                    "personas": ctrl.get("personas", []),
                    "components": ctrl.get("components", []),
                },
            ))

        with open(self.raw_dir / "risks.yaml", encoding="utf-8") as f:
            risk_data = yaml.safe_load(f)
        for risk in risk_data.get("risks", []):
            description = risk.get("shortDescription", "")
            long_desc = risk.get("longDescription", "")
            controls.append(Control(
                control_id=risk["id"],
                title=risk["title"],
                description=description,
                full_text=long_desc if long_desc else None,
                hierarchy_level="risk",
                metadata={
                    "category": risk.get("category", ""),
                    "controls": risk.get("controls", []),
                },
            ))

        return controls


if __name__ == "__main__":
    parser = CosaiParser(
        raw_dir=RAW_FRAMEWORKS_DIR / "cosai",
        output_dir=PROCESSED_FRAMEWORKS_DIR,
    )
    parser.run()
```

- [ ] **Step 3: Run test + real data**

```bash
PYTHONPATH=. python -m pytest tests/test_parse_cosai.py -v
python parsers/parse_cosai.py
```

---

### Task 12: Parser — NIST AI RMF

**Files:**
- Create: `parsers/parse_nist_ai_rmf.py`
- Create: `tests/test_parse_nist_ai_rmf.py`

- [ ] **Step 1: Implement the parser**

Create `parsers/parse_nist_ai_rmf.py`:

```python
"""Parser for NIST AI RMF 1.0 — Tier 3 markdown regex extraction."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from tract.config import PROCESSED_FRAMEWORKS_DIR, RAW_FRAMEWORKS_DIR
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SUBCATEGORY_RE = re.compile(
    r"\*\*(?P<func>GOVERN|MAP|MEASURE|MANAGE)\s+(?P<cat>\d+)\.(?P<sub>\d+)[:.]?\*\*\s*(?P<title>[^\n]*)",
    re.MULTILINE,
)

FUNCTION_NAMES = {
    "GOVERN": "Govern",
    "MAP": "Map",
    "MEASURE": "Measure",
    "MANAGE": "Manage",
}


class NistAiRmfParser(BaseParser):
    framework_id = "nist_ai_rmf"
    framework_name = "NIST AI Risk Management Framework"
    version = "1.0"
    source_url = "https://doi.org/10.6028/NIST.AI.100-1"
    mapping_unit_level = "subcategory"
    expected_count = 72

    def parse(self) -> list[Control]:
        text = (self.raw_dir / "nist_ai_rmf_1.0.md").read_text(encoding="utf-8")
        matches = list(SUBCATEGORY_RE.finditer(text))
        controls: list[Control] = []

        for i, m in enumerate(matches):
            func = m.group("func")
            cat = m.group("cat")
            sub = m.group("sub")
            title = m.group("title").strip().rstrip("*").strip()
            control_id = f"{func} {cat}.{sub}"

            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            body = re.sub(r"\*\*[A-Z]+ \d+:\*\*[^\n]*\n?", "", body).strip()

            if not title and body:
                title = body[:80].split("\n")[0]

            controls.append(Control(
                control_id=control_id,
                title=title if title else control_id,
                description=body[:2000] if body else title,
                full_text=body if len(body) > 2000 else None,
                hierarchy_level="subcategory",
                parent_id=f"{func} {cat}",
                parent_name=FUNCTION_NAMES.get(func, func),
                metadata={"function": func},
            ))

        return controls


if __name__ == "__main__":
    parser = NistAiRmfParser(
        raw_dir=RAW_FRAMEWORKS_DIR / "nist_ai_rmf",
        output_dir=PROCESSED_FRAMEWORKS_DIR,
    )
    parser.run()
```

- [ ] **Step 2: Write test and run**

Create `tests/test_parse_nist_ai_rmf.py`:

```python
"""Tests for parsers/parse_nist_ai_rmf.py."""
from pathlib import Path
from parsers.parse_nist_ai_rmf import NistAiRmfParser


def test_parses_real_data() -> None:
    from tract.config import RAW_FRAMEWORKS_DIR
    raw_dir = RAW_FRAMEWORKS_DIR / "nist_ai_rmf"
    if not (raw_dir / "nist_ai_rmf_1.0.md").exists():
        import pytest
        pytest.skip("Raw data not available")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        parser = NistAiRmfParser(raw_dir=raw_dir, output_dir=Path(td))
        result = parser.run()

        assert result.framework_id == "nist_ai_rmf"
        assert len(result.controls) >= 65
        funcs = {c.metadata["function"] for c in result.controls if c.metadata}
        assert funcs == {"GOVERN", "MAP", "MEASURE", "MANAGE"}
        assert all(c.description for c in result.controls)
```

```bash
PYTHONPATH=. python -m pytest tests/test_parse_nist_ai_rmf.py -v
python parsers/parse_nist_ai_rmf.py
```

---

### Task 13: Parser — NIST AI 600-1

**Files:**
- Create: `parsers/parse_nist_ai_600_1.py`
- Create: `tests/test_parse_nist_ai_600_1.py`

- [ ] **Step 1: Implement the parser**

Create `parsers/parse_nist_ai_600_1.py`:

```python
"""Parser for NIST AI 600-1 GenAI Profile — Tier 3 markdown extraction."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from tract.config import PROCESSED_FRAMEWORKS_DIR, RAW_FRAMEWORKS_DIR
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RISK_SECTION_RE = re.compile(
    r"\*\*2\.(?P<num>\d+)\.?\*\*\s+\*\*(?P<title>[^*]+)\*\*",
    re.MULTILINE,
)

GAI_RISK_IDS: dict[int, str] = {
    1: "GAI-CBRN",
    2: "GAI-CONFAB",
    3: "GAI-DANGEROUS",
    4: "GAI-PRIVACY",
    5: "GAI-ENVIRON",
    6: "GAI-BIAS",
    7: "GAI-HUMANAI",
    8: "GAI-INFOINTEG",
    9: "GAI-INFOSEC",
    10: "GAI-IP",
    11: "GAI-OBSCENE",
    12: "GAI-VALUECHAIN",
}


class NistAi600Parser(BaseParser):
    framework_id = "nist_ai_600_1"
    framework_name = "NIST AI 600-1 Generative AI Profile"
    version = "1.0"
    source_url = "https://doi.org/10.6028/NIST.AI.600-1"
    mapping_unit_level = "risk_category"
    expected_count = 12

    def parse(self) -> list[Control]:
        text = (self.raw_dir / "nist_ai_600_1.md").read_text(encoding="utf-8")
        matches = list(RISK_SECTION_RE.finditer(text))
        controls: list[Control] = []

        for i, m in enumerate(matches):
            num = int(m.group("num"))
            title = m.group("title").strip()
            risk_id = GAI_RISK_IDS.get(num, f"GAI-{num:02d}")

            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()

            rmf_refs = re.findall(r"(GOVERN|MAP|MEASURE|MANAGE)\s+\d+\.\d+", body)

            controls.append(Control(
                control_id=risk_id,
                title=title,
                description=body[:2000] if body else title,
                full_text=body if len(body) > 2000 else None,
                hierarchy_level="risk_category",
                metadata={"rmf_subcategories": list(set(rmf_refs))} if rmf_refs else None,
            ))

        return controls


if __name__ == "__main__":
    parser = NistAi600Parser(
        raw_dir=RAW_FRAMEWORKS_DIR / "nist_ai_600_1",
        output_dir=PROCESSED_FRAMEWORKS_DIR,
    )
    parser.run()
```

- [ ] **Step 2: Write test and run**

Create `tests/test_parse_nist_ai_600_1.py`:

```python
"""Tests for parsers/parse_nist_ai_600_1.py."""
from pathlib import Path
from parsers.parse_nist_ai_600_1 import NistAi600Parser


def test_parses_real_data() -> None:
    from tract.config import RAW_FRAMEWORKS_DIR
    raw_dir = RAW_FRAMEWORKS_DIR / "nist_ai_600_1"
    if not (raw_dir / "nist_ai_600_1.md").exists():
        import pytest
        pytest.skip("Raw data not available")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        parser = NistAi600Parser(raw_dir=raw_dir, output_dir=Path(td))
        result = parser.run()

        assert result.framework_id == "nist_ai_600_1"
        assert len(result.controls) == 12
        ids = {c.control_id for c in result.controls}
        assert "GAI-CBRN" in ids
        assert "GAI-CONFAB" in ids
        assert "GAI-VALUECHAIN" in ids
```

```bash
PYTHONPATH=. python -m pytest tests/test_parse_nist_ai_600_1.py -v
python parsers/parse_nist_ai_600_1.py
```

---

### Task 14: Parser — OWASP AI Exchange

**Files:**
- Create: `parsers/parse_owasp_ai_exchange.py`
- Create: `tests/test_parse_owasp_ai_exchange.py`

- [ ] **Step 1: Implement the parser**

Create `parsers/parse_owasp_ai_exchange.py`:

```python
"""Parser for OWASP AI Exchange — Tier 3 Hugo markdown extraction."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from tract.config import PROCESSED_FRAMEWORKS_DIR, RAW_FRAMEWORKS_DIR
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONTROL_HEADER_RE = re.compile(r"^####\s+#(?P<id>[A-Z][A-Z0-9 ]+)\s*$", re.MULTILINE)

SOURCE_FILES_WITH_CONTROLS = [
    "src_1_general_controls.md",
    "src_2_threats_through_use.md",
    "src_3_development_time_threats.md",
    "src_4_runtime_application_security_threats.md",
]

CATEGORY_MAP: dict[str, str] = {
    "src_1_general_controls.md": "General Controls",
    "src_2_threats_through_use.md": "Threats Through Use",
    "src_3_development_time_threats.md": "Development Time Threats",
    "src_4_runtime_application_security_threats.md": "Runtime Application Security Threats",
}


class OwaspAiExchangeParser(BaseParser):
    framework_id = "owasp_ai_exchange"
    framework_name = "OWASP AI Exchange"
    version = "2024"
    source_url = "https://owaspai.org"
    mapping_unit_level = "control"
    expected_count = 88

    def parse(self) -> list[Control]:
        controls: list[Control] = []

        for filename in SOURCE_FILES_WITH_CONTROLS:
            filepath = self.raw_dir / filename
            if not filepath.exists():
                logger.warning("Missing file: %s", filepath)
                continue

            text = filepath.read_text(encoding="utf-8")
            category = CATEGORY_MAP.get(filename, filename)
            matches = list(CONTROL_HEADER_RE.finditer(text))

            for i, m in enumerate(matches):
                raw_id = m.group("id").strip()
                control_id = raw_id.replace(" ", "_")

                start = m.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                body = text[start:end].strip()

                next_section = re.search(r"^#{1,3}\s+\d", body, re.MULTILINE)
                if next_section:
                    body = body[:next_section.start()].strip()

                title = raw_id.replace("_", " ").title()

                is_threat = any(
                    kw in filename
                    for kw in ["threats_through_use", "development_time", "runtime_application"]
                )
                level = "threat" if is_threat and not body.startswith("Control") else "control"

                controls.append(Control(
                    control_id=control_id,
                    title=title,
                    description=body[:2000] if body else title,
                    full_text=body if len(body) > 2000 else None,
                    hierarchy_level=level,
                    metadata={"category": category, "source_file": filename},
                ))

        return controls


if __name__ == "__main__":
    parser = OwaspAiExchangeParser(
        raw_dir=RAW_FRAMEWORKS_DIR / "owasp_ai_exchange",
        output_dir=PROCESSED_FRAMEWORKS_DIR,
    )
    parser.run()
```

- [ ] **Step 2: Write test and run**

Create `tests/test_parse_owasp_ai_exchange.py`:

```python
"""Tests for parsers/parse_owasp_ai_exchange.py."""
from pathlib import Path
from parsers.parse_owasp_ai_exchange import OwaspAiExchangeParser


def test_parses_real_data() -> None:
    from tract.config import RAW_FRAMEWORKS_DIR
    raw_dir = RAW_FRAMEWORKS_DIR / "owasp_ai_exchange"
    if not (raw_dir / "src_1_general_controls.md").exists():
        import pytest
        pytest.skip("Raw data not available")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        parser = OwaspAiExchangeParser(raw_dir=raw_dir, output_dir=Path(td))
        result = parser.run()

        assert result.framework_id == "owasp_ai_exchange"
        assert len(result.controls) >= 50
        ids = [c.control_id for c in result.controls]
        assert any("SEC" in cid for cid in ids)
        assert all(c.description for c in result.controls)
```

```bash
PYTHONPATH=. python -m pytest tests/test_parse_owasp_ai_exchange.py -v
python parsers/parse_owasp_ai_exchange.py
```

---

### Task 15: Parser — OWASP LLM Top 10

**Files:**
- Create: `parsers/parse_owasp_llm_top10.py`
- Create: `tests/test_parse_owasp_llm_top10.py`

- [ ] **Step 1: Implement the parser**

Create `parsers/parse_owasp_llm_top10.py`:

```python
"""Parser for OWASP Top 10 for LLM Applications 2025 — Tier 3 markdown."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from tract.config import PROCESSED_FRAMEWORKS_DIR, RAW_FRAMEWORKS_DIR
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LLM_HEADER_RE = re.compile(
    r"\*\*(?P<id>LLM(?:0[1-9]|10):2025)\s+(?P<title>[^*]+)\*\*",
    re.MULTILINE,
)


class OwaspLlmTop10Parser(BaseParser):
    framework_id = "owasp_llm_top10"
    framework_name = "OWASP Top 10 for LLM Applications 2025"
    version = "2025"
    source_url = "https://genai.owasp.org"
    mapping_unit_level = "risk"
    expected_count = 10

    def parse(self) -> list[Control]:
        text = (self.raw_dir / "owasp_llm_top_10_2025.md").read_text(encoding="utf-8")
        matches = list(LLM_HEADER_RE.finditer(text))
        controls: list[Control] = []

        for i, m in enumerate(matches):
            control_id = m.group("id").strip()
            title = m.group("title").strip().rstrip(".")

            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()

            controls.append(Control(
                control_id=control_id,
                title=title,
                description=body[:2000] if body else title,
                full_text=body if len(body) > 2000 else None,
                hierarchy_level="risk",
            ))

        return controls


if __name__ == "__main__":
    parser = OwaspLlmTop10Parser(
        raw_dir=RAW_FRAMEWORKS_DIR / "owasp_llm_top10",
        output_dir=PROCESSED_FRAMEWORKS_DIR,
    )
    parser.run()
```

- [ ] **Step 2: Write test and run**

Create `tests/test_parse_owasp_llm_top10.py`:

```python
"""Tests for parsers/parse_owasp_llm_top10.py."""
from pathlib import Path
from parsers.parse_owasp_llm_top10 import OwaspLlmTop10Parser


def test_parses_real_data() -> None:
    from tract.config import RAW_FRAMEWORKS_DIR
    raw_dir = RAW_FRAMEWORKS_DIR / "owasp_llm_top10"
    if not (raw_dir / "owasp_llm_top_10_2025.md").exists():
        import pytest
        pytest.skip("Raw data not available")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        parser = OwaspLlmTop10Parser(raw_dir=raw_dir, output_dir=Path(td))
        result = parser.run()

        assert result.framework_id == "owasp_llm_top10"
        assert len(result.controls) == 10
        ids = [c.control_id for c in result.controls]
        assert "LLM01:2025" in ids
        assert "LLM10:2025" in ids
```

```bash
PYTHONPATH=. python -m pytest tests/test_parse_owasp_llm_top10.py -v
python parsers/parse_owasp_llm_top10.py
```

---

### Task 16: Parser — OWASP Agentic Top 10

**Files:**
- Create: `parsers/parse_owasp_agentic_top10.py`
- Create: `tests/test_parse_owasp_agentic_top10.py`

- [ ] **Step 1: Implement the parser**

Create `parsers/parse_owasp_agentic_top10.py`:

```python
"""Parser for OWASP Top 10 for Agentic Applications 2026 — Tier 3 markdown."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from tract.config import PROCESSED_FRAMEWORKS_DIR, RAW_FRAMEWORKS_DIR
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ASI_MAPPING: list[tuple[str, str]] = [
    ("ASI01", "Agent Goal Hijack"),
    ("ASI02", "Tool Misuse and Exploitation"),
    ("ASI03", "Identity and Privilege Abuse"),
    ("ASI04", "Agentic Supply Chain Vulnerabilities"),
    ("ASI05", "Unexpected Code Execution (RCE)"),
    ("ASI06", "Memory & Context Poisoning"),
    ("ASI07", "Insecure Inter-Agent Communication"),
    ("ASI08", "Cascading Failures"),
    ("ASI09", "Human-Agent Trust Exploitation"),
    ("ASI10", "Rogue Agents"),
]

RISK_SECTION_RE = re.compile(r"^#{1,6}\s*\*{0,2}Description\*{0,2}\s*$", re.MULTILINE)


class OwaspAgenticTop10Parser(BaseParser):
    framework_id = "owasp_agentic_top10"
    framework_name = "OWASP Top 10 for Agentic Applications 2026"
    version = "2026"
    source_url = "https://genai.owasp.org"
    mapping_unit_level = "risk"
    expected_count = 10

    def parse(self) -> list[Control]:
        text = (self.raw_dir / "owasp_agentic_top10_2026.md").read_text(encoding="utf-8")
        desc_matches = list(RISK_SECTION_RE.finditer(text))
        controls: list[Control] = []

        for i, (asi_id, asi_name) in enumerate(ASI_MAPPING):
            if i >= len(desc_matches):
                logger.warning("Could not find Description section %d for %s", i + 1, asi_id)
                continue

            m = desc_matches[i]
            start = m.end()

            next_major = re.search(r"^#{1,4}\s+\*{0,2}(?:Common Examples|Prevention|Example Attack)", text[start:], re.MULTILINE)
            if next_major:
                body = text[start:start + next_major.start()].strip()
            elif i + 1 < len(desc_matches):
                body = text[start:desc_matches[i + 1].start()].strip()
            else:
                body = text[start:start + 3000].strip()

            controls.append(Control(
                control_id=asi_id,
                title=asi_name,
                description=body[:2000] if body else asi_name,
                full_text=body if len(body) > 2000 else None,
                hierarchy_level="risk",
            ))

        return controls


if __name__ == "__main__":
    parser = OwaspAgenticTop10Parser(
        raw_dir=RAW_FRAMEWORKS_DIR / "owasp_agentic_top10",
        output_dir=PROCESSED_FRAMEWORKS_DIR,
    )
    parser.run()
```

- [ ] **Step 2: Write test and run**

Create `tests/test_parse_owasp_agentic_top10.py`:

```python
"""Tests for parsers/parse_owasp_agentic_top10.py."""
from pathlib import Path
from parsers.parse_owasp_agentic_top10 import OwaspAgenticTop10Parser


def test_parses_real_data() -> None:
    from tract.config import RAW_FRAMEWORKS_DIR
    raw_dir = RAW_FRAMEWORKS_DIR / "owasp_agentic_top10"
    if not (raw_dir / "owasp_agentic_top10_2026.md").exists():
        import pytest
        pytest.skip("Raw data not available")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        parser = OwaspAgenticTop10Parser(raw_dir=raw_dir, output_dir=Path(td))
        result = parser.run()

        assert result.framework_id == "owasp_agentic_top10"
        assert len(result.controls) == 10
        ids = [c.control_id for c in result.controls]
        assert ids == ["ASI01", "ASI02", "ASI03", "ASI04", "ASI05",
                       "ASI06", "ASI07", "ASI08", "ASI09", "ASI10"]
        assert all(c.description for c in result.controls)
```

```bash
PYTHONPATH=. python -m pytest tests/test_parse_owasp_agentic_top10.py -v
python parsers/parse_owasp_agentic_top10.py
```

---

### Task 17: Parser — EU GPAI Code of Practice

**Files:**
- Create: `parsers/parse_eu_gpai_cop.py`
- Create: `tests/test_parse_eu_gpai_cop.py`

- [ ] **Step 1: Implement the parser**

Create `parsers/parse_eu_gpai_cop.py`:

```python
"""Parser for EU GPAI Code of Practice — Tier 3 markdown extraction."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from tract.config import PROCESSED_FRAMEWORKS_DIR, RAW_FRAMEWORKS_DIR
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHAPTER_RE = re.compile(r"^#\s+(?P<chapter>Transparency|Copyright|Safety and Security)\s+Chapter", re.MULTILINE)
COMMITMENT_RE = re.compile(r"^##\s+Commitment\s+(?P<num>\d+)\s+(?P<title>.+)$", re.MULTILINE)
MEASURE_RE = re.compile(r"^###\s+Measure\s+(?P<num>\d+\.\d+)\s+(?P<title>.+)$", re.MULTILINE)

CHAPTER_PREFIX: dict[str, str] = {
    "Transparency": "GPAI-T",
    "Copyright": "GPAI-C",
    "Safety and Security": "GPAI-SS",
}


class EuGpaiCopParser(BaseParser):
    framework_id = "eu_gpai_cop"
    framework_name = "EU GPAI Code of Practice"
    version = "2025"
    source_url = "https://digital-strategy.ec.europa.eu/en/policies/ai-pact"
    mapping_unit_level = "measure"
    expected_count = 40

    def parse(self) -> list[Control]:
        text = (self.raw_dir / "gpai_code_of_practice_combined.md").read_text(encoding="utf-8")
        controls: list[Control] = []

        chapters = list(CHAPTER_RE.finditer(text))
        chapter_ranges: list[tuple[str, int, int]] = []
        for i, cm in enumerate(chapters):
            ch_name = cm.group("chapter")
            start = cm.start()
            end = chapters[i + 1].start() if i + 1 < len(chapters) else len(text)
            chapter_ranges.append((ch_name, start, end))

        for ch_name, ch_start, ch_end in chapter_ranges:
            chapter_text = text[ch_start:ch_end]
            prefix = CHAPTER_PREFIX.get(ch_name, "GPAI")

            current_commitment = ""
            measures = list(MEASURE_RE.finditer(chapter_text))

            for i, m in enumerate(measures):
                measure_num = m.group("num")
                measure_title = m.group("title").strip()
                control_id = f"{prefix}-M{measure_num}"

                commitment_before = None
                for cm in COMMITMENT_RE.finditer(chapter_text[:m.start()]):
                    commitment_before = cm
                if commitment_before:
                    current_commitment = commitment_before.group("title").strip()

                start = m.end()
                end = measures[i + 1].start() if i + 1 < len(measures) else ch_end - ch_start
                body = chapter_text[start:end].strip()

                next_commitment = COMMITMENT_RE.search(chapter_text[start:])
                if next_commitment and start + next_commitment.start() < end:
                    body = chapter_text[start:start + next_commitment.start()].strip()

                controls.append(Control(
                    control_id=control_id,
                    title=measure_title,
                    description=body[:2000] if body else measure_title,
                    full_text=body if len(body) > 2000 else None,
                    hierarchy_level="measure",
                    parent_id=f"{prefix}-C{commitment_before.group('num')}" if commitment_before else None,
                    parent_name=current_commitment if current_commitment else None,
                    metadata={"chapter": ch_name},
                ))

        return controls


if __name__ == "__main__":
    parser = EuGpaiCopParser(
        raw_dir=RAW_FRAMEWORKS_DIR / "eu_gpai_cop",
        output_dir=PROCESSED_FRAMEWORKS_DIR,
    )
    parser.run()
```

- [ ] **Step 2: Write test and run**

Create `tests/test_parse_eu_gpai_cop.py`:

```python
"""Tests for parsers/parse_eu_gpai_cop.py."""
from pathlib import Path
from parsers.parse_eu_gpai_cop import EuGpaiCopParser


def test_parses_real_data() -> None:
    from tract.config import RAW_FRAMEWORKS_DIR
    raw_dir = RAW_FRAMEWORKS_DIR / "eu_gpai_cop"
    if not (raw_dir / "gpai_code_of_practice_combined.md").exists():
        import pytest
        pytest.skip("Raw data not available")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        parser = EuGpaiCopParser(raw_dir=raw_dir, output_dir=Path(td))
        result = parser.run()

        assert result.framework_id == "eu_gpai_cop"
        assert len(result.controls) >= 35
        chapters = {c.metadata["chapter"] for c in result.controls if c.metadata}
        assert "Transparency" in chapters
        assert "Copyright" in chapters
        assert "Safety and Security" in chapters
        assert all(c.description for c in result.controls)
```

```bash
PYTHONPATH=. python -m pytest tests/test_parse_eu_gpai_cop.py -v
python parsers/parse_eu_gpai_cop.py
```

- [ ] **Step 3: Commit all Tier 2–3 parsers**

```bash
git add parsers/parse_cosai.py parsers/parse_nist_ai_rmf.py parsers/parse_nist_ai_600_1.py parsers/parse_owasp_ai_exchange.py parsers/parse_owasp_llm_top10.py parsers/parse_owasp_agentic_top10.py parsers/parse_eu_gpai_cop.py tests/test_parse_cosai.py tests/test_parse_nist_ai_rmf.py tests/test_parse_nist_ai_600_1.py tests/test_parse_owasp_ai_exchange.py tests/test_parse_owasp_llm_top10.py tests/test_parse_owasp_agentic_top10.py tests/test_parse_eu_gpai_cop.py
git commit -m "feat: add Tier 2-3 parsers (CoSAI, NIST AI RMF, NIST AI 600-1, OWASP AI Exchange, LLM Top 10, Agentic Top 10, EU GPAI CoP)"
```

---

## Phase E: Tier 4–5 Parsers (Parallel — 2 subagents)

### Task 18: Parser — OWASP DSGAI

**Files:**
- Create: `parsers/parse_owasp_dsgai.py`
- Create: `tests/test_parse_owasp_dsgai.py`

- [ ] **Step 1: Implement the parser**

Create `parsers/parse_owasp_dsgai.py`:

```python
"""Parser for OWASP GenAI Data Security (DSGAI) — Tier 4 TXT extraction."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from tract.config import PROCESSED_FRAMEWORKS_DIR, RAW_FRAMEWORKS_DIR
from tract.io import load_json
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class OwaspDsgaiParser(BaseParser):
    framework_id = "owasp_dsgai"
    framework_name = "OWASP GenAI Data Security Risks and Mitigations"
    version = "1.0"
    source_url = "https://genai.owasp.org/"
    mapping_unit_level = "risk"
    expected_count = 21

    def parse(self) -> list[Control]:
        manifest = load_json(self.raw_dir / "MANIFEST.json")
        id_pattern = manifest["id_pattern"]
        id_re = re.compile(id_pattern)
        txt_filename = manifest["source_file"]

        text = (self.raw_dir / txt_filename).read_text(encoding="utf-8")
        matches = list(id_re.finditer(text))
        controls: list[Control] = []
        seen_ids: set[str] = set()

        for i, m in enumerate(matches):
            risk_id = m.group(0)
            if risk_id in seen_ids:
                continue
            seen_ids.add(risk_id)

            start = m.start()
            remaining_matches = [mm for mm in matches[i + 1:] if mm.group(0) not in seen_ids]
            end = remaining_matches[0].start() if remaining_matches else len(text)
            section = text[start:end].strip()

            lines = section.split("\n")
            title_line = lines[0] if lines else risk_id
            title = re.sub(id_pattern, "", title_line).strip().strip(":.– ")
            if not title:
                title = risk_id

            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else title

            controls.append(Control(
                control_id=risk_id,
                title=title,
                description=body[:2000] if body else title,
                full_text=body if len(body) > 2000 else None,
                hierarchy_level="risk",
            ))

        return controls


if __name__ == "__main__":
    parser = OwaspDsgaiParser(
        raw_dir=RAW_FRAMEWORKS_DIR / "owasp_dsgai",
        output_dir=PROCESSED_FRAMEWORKS_DIR,
    )
    parser.run()
```

- [ ] **Step 2: Write test and run**

Create `tests/test_parse_owasp_dsgai.py`:

```python
"""Tests for parsers/parse_owasp_dsgai.py."""
from pathlib import Path
from parsers.parse_owasp_dsgai import OwaspDsgaiParser


def test_parses_real_data() -> None:
    from tract.config import RAW_FRAMEWORKS_DIR
    raw_dir = RAW_FRAMEWORKS_DIR / "owasp_dsgai"
    if not (raw_dir / "MANIFEST.json").exists():
        import pytest
        pytest.skip("Raw data not available")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        parser = OwaspDsgaiParser(raw_dir=raw_dir, output_dir=Path(td))
        result = parser.run()

        assert result.framework_id == "owasp_dsgai"
        assert len(result.controls) == 21
        ids = [c.control_id for c in result.controls]
        assert "DSGAI01" in ids
        assert "DSGAI21" in ids
        assert all(c.description for c in result.controls)
```

```bash
PYTHONPATH=. python -m pytest tests/test_parse_owasp_dsgai.py -v
python parsers/parse_owasp_dsgai.py
```

---

### Task 19: Parser — EU AI Act

**Files:**
- Create: `parsers/parse_eu_ai_act.py`
- Create: `tests/test_parse_eu_ai_act.py`

- [ ] **Step 1: Implement the parser**

Create `parsers/parse_eu_ai_act.py`:

```python
"""Parser for EU AI Act — Tier 5 HTML extraction with BeautifulSoup."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from tract.config import PROCESSED_FRAMEWORKS_DIR, RAW_FRAMEWORKS_DIR
from tract.io import load_json
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARTICLE_RE = re.compile(r"^Article\s+(\d+)$")
ANNEX_RE = re.compile(r"^ANNEX\s+([IVXLCDM]+)$")


class EuAiActParser(BaseParser):
    framework_id = "eu_ai_act"
    framework_name = "EU AI Act — Regulation (EU) 2024/1689"
    version = "2024/1689"
    source_url = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689"
    mapping_unit_level = "article"
    expected_count = 100

    def parse(self) -> list[Control]:
        html_file = self.raw_dir / "eu_ai_act_2024_1689.html"
        soup = BeautifulSoup(html_file.read_text(encoding="utf-8"), "lxml")
        controls: list[Control] = []

        for el in soup.find_all(["p", "div", "span", "h1", "h2", "h3", "h4"]):
            text = el.get_text(strip=True)
            article_match = ARTICLE_RE.match(text)
            annex_match = ANNEX_RE.match(text)

            if article_match:
                art_num = int(article_match.group(1))
                control_id = f"AIA-Art{art_num}"
                title, body = self._extract_section_text(el)
                controls.append(Control(
                    control_id=control_id,
                    title=title if title else f"Article {art_num}",
                    description=body[:2000] if body else f"Article {art_num}",
                    full_text=body if len(body) > 2000 else None,
                    hierarchy_level="article",
                ))

            elif annex_match:
                annex_num = annex_match.group(1)
                control_id = f"AIA-Annex{annex_num}"
                title, body = self._extract_section_text(el)
                controls.append(Control(
                    control_id=control_id,
                    title=title if title else f"Annex {annex_num}",
                    description=body[:2000] if body else f"Annex {annex_num}",
                    full_text=body if len(body) > 2000 else None,
                    hierarchy_level="annex",
                ))

        seen: set[str] = set()
        deduped: list[Control] = []
        for c in controls:
            if c.control_id not in seen:
                seen.add(c.control_id)
                deduped.append(c)
        return deduped

    @staticmethod
    def _extract_section_text(header_el: Tag) -> tuple[str, str]:
        """Extract title and body text following an article/annex header."""
        parts: list[str] = []
        title = ""

        sibling = header_el.find_next_sibling()
        first = True
        while sibling:
            text = sibling.get_text(strip=True)
            if not text:
                sibling = sibling.find_next_sibling()
                continue
            if ARTICLE_RE.match(text) or ANNEX_RE.match(text):
                break
            if first:
                title = text[:200]
                first = False
            parts.append(text)
            sibling = sibling.find_next_sibling()

        body = "\n".join(parts)
        return title, body


if __name__ == "__main__":
    parser = EuAiActParser(
        raw_dir=RAW_FRAMEWORKS_DIR / "eu_ai_act",
        output_dir=PROCESSED_FRAMEWORKS_DIR,
    )
    parser.run()
```

- [ ] **Step 2: Write test and run**

Create `tests/test_parse_eu_ai_act.py`:

```python
"""Tests for parsers/parse_eu_ai_act.py."""
from pathlib import Path
from parsers.parse_eu_ai_act import EuAiActParser


def test_parses_real_data() -> None:
    from tract.config import RAW_FRAMEWORKS_DIR
    raw_dir = RAW_FRAMEWORKS_DIR / "eu_ai_act"
    if not (raw_dir / "eu_ai_act_2024_1689.html").exists():
        import pytest
        pytest.skip("Raw data not available")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        parser = EuAiActParser(raw_dir=raw_dir, output_dir=Path(td))
        result = parser.run()

        assert result.framework_id == "eu_ai_act"
        assert len(result.controls) >= 90
        ids = [c.control_id for c in result.controls]
        assert "AIA-Art1" in ids
        assert any("Annex" in cid for cid in ids)
        assert all(c.description for c in result.controls)
```

```bash
PYTHONPATH=. python -m pytest tests/test_parse_eu_ai_act.py -v
python parsers/parse_eu_ai_act.py
```

- [ ] **Step 3: Commit Tier 4–5 parsers**

```bash
git add parsers/parse_owasp_dsgai.py parsers/parse_eu_ai_act.py tests/test_parse_owasp_dsgai.py tests/test_parse_eu_ai_act.py
git commit -m "feat: add Tier 4-5 parsers (OWASP DSGAI, EU AI Act)"
```

---

## Phase F: Post-Processing (Sequential)

### Task 20: Hub Link Extraction

**Files:**
- Create: `parsers/extract_hub_links.py`
- Create: `tests/test_extract_hub_links.py`

- [ ] **Step 1: Implement extract_hub_links.py**

Create `parsers/extract_hub_links.py`:

```python
"""Extract standard-to-hub links from OpenCRE data for LOFO training splits."""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from tract.config import (
    OPENCRE_FRAMEWORK_ID_MAP,
    RAW_OPENCRE_DIR,
    TRAINING_DIR,
)
from tract.io import atomic_write_json, load_json
from tract.schema import HubLink

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRAINING_LINK_TYPES = {"LinkedTo", "Linked To", "AutomaticallyLinkedTo", "Automatically Linked To"}
EXCLUDE_LINK_TYPES = {"Contains", "Is Part Of", "Related"}


def normalize_framework_id(standard_name: str) -> str:
    """Map OpenCRE standard name to canonical framework ID."""
    if standard_name in OPENCRE_FRAMEWORK_ID_MAP:
        return OPENCRE_FRAMEWORK_ID_MAP[standard_name]
    slug = standard_name.lower().replace(" ", "_").replace("-", "_")
    logger.debug("Unknown standard '%s' -> '%s'", standard_name, slug)
    return slug


def extract_links(opencre_path: Path) -> list[HubLink]:
    """Extract all standard-to-hub links from the OpenCRE dump."""
    data = load_json(opencre_path)
    cres = data.get("cres", data) if isinstance(data, dict) else data
    links: list[HubLink] = []

    for cre in cres:
        cre_id = cre.get("id", "")
        cre_name = cre.get("name", "")

        for link in cre.get("links", []):
            link_type = link.get("type", "")
            if link_type in EXCLUDE_LINK_TYPES:
                continue
            if link_type not in TRAINING_LINK_TYPES:
                continue

            doc = link.get("document", {})
            standard_name = doc.get("name", "")
            if not standard_name:
                continue

            section_id = doc.get("sectionID", doc.get("section", ""))
            section_name = doc.get("section", doc.get("sectionID", ""))

            links.append(HubLink(
                cre_id=cre_id,
                cre_name=cre_name,
                standard_name=standard_name,
                section_id=str(section_id),
                section_name=str(section_name),
                link_type=link_type.replace(" ", ""),
                framework_id=normalize_framework_id(standard_name),
            ))

    return links


def main() -> None:
    opencre_path = RAW_OPENCRE_DIR / "opencre_all_cres.json"
    if not opencre_path.exists():
        raise FileNotFoundError(f"OpenCRE data not found at {opencre_path}. Run fetch_opencre.py first.")

    links = extract_links(opencre_path)
    logger.info("Extracted %d standard-to-hub links", len(links))

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = TRAINING_DIR / "hub_links.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for link in links:
            f.write(json.dumps(link.model_dump(), sort_keys=True, ensure_ascii=False) + "\n")
    logger.info("Wrote %s (%d links)", jsonl_path, len(links))

    by_framework: dict[str, list[dict[str, str]]] = defaultdict(list)
    for link in links:
        by_framework[link.framework_id].append(link.model_dump())

    grouped_path = TRAINING_DIR / "hub_links_by_framework.json"
    atomic_write_json(dict(by_framework), grouped_path)
    logger.info("Wrote %s (%d frameworks)", grouped_path, len(by_framework))

    for fw_id, fw_links in sorted(by_framework.items(), key=lambda x: -len(x[1])):
        logger.info("  %s: %d links", fw_id, len(fw_links))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write test**

Create `tests/test_extract_hub_links.py`:

```python
"""Tests for parsers/extract_hub_links.py."""
import json
from pathlib import Path

from parsers.extract_hub_links import extract_links, normalize_framework_id


def test_normalize_framework_id() -> None:
    assert normalize_framework_id("CAPEC") == "capec"
    assert normalize_framework_id("NIST 800-53") == "nist_800_53"
    assert normalize_framework_id("Cloud Controls Matrix") == "csa_ccm"
    assert normalize_framework_id("MITRE ATLAS") == "mitre_atlas"


def test_extract_links_filters_correctly(tmp_path: Path) -> None:
    mock_data = {
        "cres": [
            {
                "id": "001-01",
                "name": "Test CRE",
                "links": [
                    {
                        "type": "LinkedTo",
                        "document": {"name": "CAPEC", "sectionID": "CAPEC-1", "section": "Attack 1"},
                    },
                    {
                        "type": "Contains",
                        "document": {"name": "Child CRE", "sectionID": "002-01"},
                    },
                    {
                        "type": "AutomaticallyLinkedTo",
                        "document": {"name": "CWE", "sectionID": "CWE-79", "section": "XSS"},
                    },
                ],
            }
        ]
    }

    data_file = tmp_path / "opencre.json"
    data_file.write_text(json.dumps(mock_data), encoding="utf-8")

    links = extract_links(data_file)
    assert len(links) == 2
    types = {l.link_type for l in links}
    assert types == {"LinkedTo", "AutomaticallyLinkedTo"}
```

- [ ] **Step 3: Run test + real data**

```bash
PYTHONPATH=. python -m pytest tests/test_extract_hub_links.py -v
python parsers/extract_hub_links.py
```

Expected output: ~4,406 links, per-framework breakdown matching PRD Section 4.3.

- [ ] **Step 4: Commit**

```bash
git add parsers/extract_hub_links.py tests/test_extract_hub_links.py
git commit -m "feat: add hub link extraction from OpenCRE data for LOFO splits"
```

---

### Task 21: Validation and Merge

**Files:**
- Create: `parsers/validate_all.py`
- Create: `parsers/merge_all_controls.py`

- [ ] **Step 1: Implement validate_all.py**

Create `parsers/validate_all.py`:

```python
"""Cross-framework validation: schema conformance, counts, no empty fields."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from tract.config import EXPECTED_COUNTS, COUNT_TOLERANCE, PROCESSED_FRAMEWORKS_DIR
from tract.io import load_json
from tract.schema import FrameworkOutput

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def validate_framework(path: Path) -> list[str]:
    """Validate a single framework JSON. Returns list of error messages."""
    errors: list[str] = []
    fw_id = path.stem

    try:
        data = load_json(path)
        output = FrameworkOutput.model_validate(data)
    except Exception as e:
        return [f"{fw_id}: schema validation failed: {e}"]

    if output.framework_id != fw_id:
        errors.append(f"{fw_id}: framework_id mismatch (file={fw_id}, data={output.framework_id})")

    expected = EXPECTED_COUNTS.get(fw_id)
    if expected is not None:
        actual = len(output.controls)
        low = int(expected * (1 - COUNT_TOLERANCE))
        high = int(expected * (1 + COUNT_TOLERANCE))
        if not (low <= actual <= high):
            errors.append(f"{fw_id}: count {actual} outside expected {expected} (tolerance {low}-{high})")

    seen_ids: set[str] = set()
    for ctrl in output.controls:
        if not ctrl.description.strip():
            errors.append(f"{fw_id}: empty description for {ctrl.control_id}")
        if ctrl.control_id in seen_ids:
            errors.append(f"{fw_id}: duplicate control_id {ctrl.control_id}")
        seen_ids.add(ctrl.control_id)

    return errors


def main() -> None:
    framework_dir = PROCESSED_FRAMEWORKS_DIR
    files = sorted(framework_dir.glob("*.json"))

    if not files:
        logger.error("No framework files found in %s", framework_dir)
        sys.exit(1)

    total_errors: list[str] = []
    for path in files:
        errors = validate_framework(path)
        if errors:
            for e in errors:
                logger.error("FAIL: %s", e)
            total_errors.extend(errors)
        else:
            data = load_json(path)
            count = len(data.get("controls", []))
            logger.info("PASS: %s (%d controls)", path.stem, count)

    logger.info("Validated %d frameworks, %d errors", len(files), len(total_errors))
    if total_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Implement merge_all_controls.py**

Create `parsers/merge_all_controls.py`:

```python
"""Merge all validated framework JSONs into a single all_controls.json."""
from __future__ import annotations

import datetime
import logging
from pathlib import Path

from tract.config import PROCESSED_DIR, PROCESSED_FRAMEWORKS_DIR
from tract.io import atomic_write_json, load_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    files = sorted(PROCESSED_FRAMEWORKS_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No framework files in {PROCESSED_FRAMEWORKS_DIR}")

    frameworks: list[dict[str, object]] = []
    total_controls = 0

    for path in files:
        data = load_json(path)
        frameworks.append(data)
        count = len(data.get("controls", []))
        total_controls += count
        logger.info("Loaded %s: %d controls", path.stem, count)

    output = {
        "generated_date": datetime.date.today().isoformat(),
        "framework_count": len(frameworks),
        "total_controls": total_controls,
        "frameworks": frameworks,
    }

    output_path = PROCESSED_DIR / "all_controls.json"
    atomic_write_json(output, output_path)
    logger.info("Wrote %s: %d frameworks, %d total controls", output_path, len(frameworks), total_controls)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run validation + merge**

```bash
python parsers/validate_all.py
python parsers/merge_all_controls.py
python3 -c "import json; d=json.load(open('data/processed/all_controls.json')); print(f'Frameworks: {d[\"framework_count\"]}, Controls: {d[\"total_controls\"]}')"
```

Expected: 12 frameworks, ~850+ total controls.

- [ ] **Step 4: Run full test suite**

```bash
PYTHONPATH=. python -m pytest tests/ -v
```

- [ ] **Step 5: Commit**

```bash
git add parsers/validate_all.py parsers/merge_all_controls.py
git commit -m "feat: add cross-framework validation and all_controls.json merge"
```

---

## PRD Discrepancy Notes

1. **EU GPAI CoP count:** PRD says 32 measures; actual file has 40 across 3 chapters (Transparency: 3, Copyright: 5, Safety & Security: 32). PRD likely counted Safety chapter only. Plan uses 40.
2. **NIST AI 600-1 parser:** Missing from PRD Section 4.7 parser list but raw data is listed in Section 4.5. Included in this plan.
3. **NIST 800-53:** Excluded — already in OpenCRE with 300 links. Control texts come from API.
4. **OWASP AI Exchange:** Only 4 of 7 source files contain `#### #` control headers. Files `src_5_testing.md`, `src_6_privacy.md`, and `src_ai_security_overview.md` have zero controls and are overview/reference files.
