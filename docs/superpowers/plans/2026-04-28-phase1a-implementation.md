# Phase 1A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the three data infrastructure components for Phase 1B model training: production CRE hierarchy, hub descriptions for all 400 leaf hubs, and traditional framework ingestion from OpenCRE.

**Architecture:** Pydantic v2 models for the CRE hierarchy tree (tract/hierarchy.py), async Opus-based description generation with intermediate saves, and a single synchronous extraction script for 19 traditional frameworks from OpenCRE link metadata. All outputs are deterministic JSON with atomic writes. Cross-cutting: zero-width char sanitization extension, Phase 1A config constants, and integration tests for Phase 0 parity.

**Tech Stack:** Python 3.11, Pydantic v2, anthropic SDK (async), pytest, mypy --strict

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `tract/hierarchy.py` | HubNode + CREHierarchy Pydantic models with tree ops | CREATE |
| `tract/sanitize.py` | Add `_strip_zero_width` to pipeline | MODIFY (line 130) |
| `tract/config.py` | Add Phase 1A constants | MODIFY (append) |
| `scripts/phase1a/__init__.py` | Package marker | CREATE |
| `scripts/phase1a/build_hierarchy.py` | CLI: build + validate + save hierarchy | CREATE |
| `scripts/phase1a/generate_descriptions.py` | CLI: generate 400 hub descriptions | CREATE |
| `scripts/phase1a/validate_descriptions.py` | CLI: validate review status | CREATE |
| `scripts/phase1a/extract_traditional_frameworks.py` | CLI: extract 19 frameworks from OpenCRE | CREATE |
| `tests/fixtures/phase1a_mini_cres.json` | Fixture with orphan + deeper tree | CREATE |
| `tests/test_hierarchy.py` | Unit + integration tests for CREHierarchy | CREATE |
| `tests/test_sanitize.py` | Add zero-width char tests | MODIFY |
| `tests/test_descriptions.py` | Prompt rendering + schema tests | CREATE |
| `tests/test_traditional_frameworks.py` | Extraction + dedup + slug tests | CREATE |

---

### Task 1: Add Phase 1A constants to config.py

**Files:**
- Modify: `tract/config.py:102-117`

- [ ] **Step 1: Add Phase 1A constants after the Phase 0 section**

Append these constants to the end of `tract/config.py`:

```python
# ── Phase 1A: Data Infrastructure ───────────────────────────────────────

PHASE1A_DESCRIPTION_MODEL: Final[str] = "claude-opus-4-20250514"
PHASE1A_DESCRIPTION_TEMPERATURE: Final[float] = 0.0
PHASE1A_DESCRIPTION_MAX_TOKENS: Final[int] = 500
PHASE1A_DESCRIPTION_MAX_CONCURRENT: Final[int] = 5
PHASE1A_DESCRIPTION_SAVE_INTERVAL: Final[int] = 50
PHASE1A_DESCRIPTION_TIMEOUT_S: Final[int] = 60
PHASE1A_FRAMEWORK_SLUG_RE: Final[str] = r"^[a-z][a-z0-9_]{1,49}$"

# Framework IDs that have primary-source parsers (take precedence over OpenCRE extraction)
AI_PARSER_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({
    "aiuc_1", "cosai", "csa_aicm", "eu_ai_act", "eu_gpai_cop",
    "mitre_atlas", "nist_ai_600_1", "nist_ai_rmf",
    "owasp_agentic_top10", "owasp_ai_exchange", "owasp_dsgai", "owasp_llm_top10",
})

# OpenCRE framework IDs to extract (those WITHOUT primary-source parsers)
OPENCRE_EXTRACT_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset(
    set(OPENCRE_FRAMEWORK_ID_MAP.values()) - AI_PARSER_FRAMEWORK_IDS
)
```

- [ ] **Step 2: Verify config imports still work**

Run: `python -c "from tract.config import PHASE1A_DESCRIPTION_MODEL, OPENCRE_EXTRACT_FRAMEWORK_IDS; print(len(OPENCRE_EXTRACT_FRAMEWORK_IDS))"`
Expected: `19`

- [ ] **Step 3: Commit**

```bash
git add tract/config.py
git commit -m "feat: add Phase 1A constants to config"
```

---

### Task 2: Add zero-width character stripping to sanitize.py

**Files:**
- Modify: `tract/sanitize.py:40,128-131`
- Modify: `tests/test_sanitize.py`

- [ ] **Step 1: Write the failing tests**

Add these tests to `tests/test_sanitize.py` inside the `TestSanitizeText` class:

```python
def test_strips_zero_width_space(self) -> None:
    result = sanitize_text("hello​world")
    assert result == "helloworld"

def test_strips_zero_width_non_joiner(self) -> None:
    result = sanitize_text("hello‌world")
    assert result == "helloworld"

def test_strips_zero_width_joiner(self) -> None:
    result = sanitize_text("hello‍world")
    assert result == "helloworld"

def test_strips_bom(self) -> None:
    result = sanitize_text("﻿hello world")
    assert result == "hello world"

def test_strips_multiple_zero_width(self) -> None:
    result = sanitize_text("a​‌‍﻿b")
    assert result == "ab"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sanitize.py::TestSanitizeText::test_strips_zero_width_space -v`
Expected: FAIL (zero-width chars not yet stripped)

- [ ] **Step 3: Add the `_strip_zero_width` function to `tract/sanitize.py`**

Add after the `_WHITESPACE_RE` pattern (around line 41):

```python
# Matches zero-width characters that can poison embeddings
_ZERO_WIDTH_RE: re.Pattern[str] = re.compile(
    "[​‌‍﻿]"
)


def _strip_zero_width(text: str) -> str:
    """Remove zero-width characters that can shift embedding spaces."""
    return _ZERO_WIDTH_RE.sub("", text)
```

Then insert `_strip_zero_width` into the pipeline in `sanitize_text()`, after `_normalize_unicode` and before `strip_html`:

```python
    cleaned = _strip_null_bytes(cleaned)
    cleaned = _normalize_unicode(cleaned)
    cleaned = _strip_zero_width(cleaned)  # NEW
    cleaned = strip_html(cleaned)
```

Also update the module docstring at the top to reflect the new step:

```python
"""TRACT text sanitization pipeline.

Every text field passes through this pipeline before storage:
1. Strip null bytes
2. Unicode NFC normalization
3. Strip zero-width characters
4. HTML unescape + strip tags
5. Fix common PDF ligatures
6. Fix broken hyphenation from PDF line-wrapping
7. Collapse whitespace
8. Strip leading/trailing whitespace
```

- [ ] **Step 4: Run all sanitize tests to verify they pass**

Run: `python -m pytest tests/test_sanitize.py -v`
Expected: ALL PASS (including the 5 new tests and all 14 existing tests)

- [ ] **Step 5: Commit**

```bash
git add tract/sanitize.py tests/test_sanitize.py
git commit -m "feat: add zero-width character stripping to sanitization pipeline"
```

---

### Task 3: Create test fixture for CREHierarchy

**Files:**
- Create: `tests/fixtures/phase1a_mini_cres.json`

- [ ] **Step 1: Create a fixture with orphan hubs and a deeper tree**

This fixture has:
- 3 roots (ROOT-A, ROOT-B, ROOT-C)
- ROOT-A has 2 children (PAR-A1, PAR-A2), PAR-A1 has 2 leaf children (LEAF-A1a, LEAF-A1b) → depth 2
- ROOT-B has 2 leaf children (LEAF-B1, LEAF-B2) → depth 1
- ROOT-C has no children (orphan — root that is also a leaf)
- ORPHAN-D has no parent, no children (orphan — not a root in the traditional sense, but will be detected as a root)
- Total: 10 hubs, 4 roots (ROOT-A, ROOT-B, ROOT-C, ORPHAN-D), 7 leaves (LEAF-A1a, LEAF-A1b, PAR-A2, LEAF-B1, LEAF-B2, ROOT-C, ORPHAN-D)
- Includes Standard links for testing description prompt context

Write to `tests/fixtures/phase1a_mini_cres.json`:

```json
{
  "cres": [
    {
      "doctype": "CRE", "id": "ROOT-A", "name": "Root A", "tags": [],
      "links": [
        {"ltype": "Contains", "document": {"doctype": "CRE", "id": "PAR-A1", "name": "Parent A1", "tags": []}},
        {"ltype": "Contains", "document": {"doctype": "CRE", "id": "PAR-A2", "name": "Parent A2", "tags": []}}
      ]
    },
    {
      "doctype": "CRE", "id": "PAR-A1", "name": "Parent A1", "tags": [],
      "links": [
        {"ltype": "Contains", "document": {"doctype": "CRE", "id": "LEAF-A1a", "name": "Leaf A1a", "tags": []}},
        {"ltype": "Contains", "document": {"doctype": "CRE", "id": "LEAF-A1b", "name": "Leaf A1b", "tags": []}}
      ]
    },
    {
      "doctype": "CRE", "id": "PAR-A2", "name": "Parent A2", "tags": [], "links": [
        {"ltype": "Linked To", "document": {"doctype": "Standard", "name": "Framework Alpha", "sectionID": "A-1", "section": "Alpha Section 1"}}
      ]
    },
    {
      "doctype": "CRE", "id": "LEAF-A1a", "name": "Leaf A1a", "tags": [], "links": [
        {"ltype": "Linked To", "document": {"doctype": "Standard", "name": "Framework Alpha", "sectionID": "A-2", "section": "Alpha Section 2"}},
        {"ltype": "Linked To", "document": {"doctype": "Standard", "name": "Framework Beta", "sectionID": "B-1", "section": "Beta Section 1"}}
      ]
    },
    {
      "doctype": "CRE", "id": "LEAF-A1b", "name": "Leaf A1b", "tags": [], "links": [
        {"ltype": "Linked To", "document": {"doctype": "Standard", "name": "Framework Alpha", "sectionID": "A-3", "section": "Alpha Section 3"}}
      ]
    },
    {
      "doctype": "CRE", "id": "ROOT-B", "name": "Root B", "tags": [],
      "links": [
        {"ltype": "Contains", "document": {"doctype": "CRE", "id": "LEAF-B1", "name": "Leaf B1", "tags": []}},
        {"ltype": "Contains", "document": {"doctype": "CRE", "id": "LEAF-B2", "name": "Leaf B2", "tags": []}}
      ]
    },
    {
      "doctype": "CRE", "id": "LEAF-B1", "name": "Leaf B1", "tags": [], "links": [
        {"ltype": "Linked To", "document": {"doctype": "Standard", "name": "Framework Beta", "sectionID": "B-2", "section": "Beta Section 2"}}
      ]
    },
    {
      "doctype": "CRE", "id": "LEAF-B2", "name": "Leaf B2", "tags": [], "links": [
        {"ltype": "Automatically Linked To", "document": {"doctype": "Standard", "name": "Framework Gamma", "sectionID": "", "section": "Gamma Section 1"}}
      ]
    },
    {"doctype": "CRE", "id": "ROOT-C", "name": "Root C (Orphan Root)", "tags": [], "links": []},
    {"doctype": "CRE", "id": "ORPHAN-D", "name": "Orphan D", "tags": [], "links": []}
  ],
  "fetch_timestamp": "2026-04-28T12:00:00Z",
  "total_cres": 10,
  "total_pages": 1
}
```

- [ ] **Step 2: Verify the fixture is valid JSON**

Run: `python -c "import json; json.load(open('tests/fixtures/phase1a_mini_cres.json')); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/phase1a_mini_cres.json
git commit -m "test: add Phase 1A fixture with orphan hubs and deeper tree"
```

---

### Task 4: Implement HubNode and CREHierarchy models

**Files:**
- Create: `tract/hierarchy.py`
- Create: `tests/test_hierarchy.py`

- [ ] **Step 1: Write the failing tests for HubNode and CREHierarchy construction**

Create `tests/test_hierarchy.py`:

```python
"""Tests for tract.hierarchy — CRE hierarchy tree model."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "phase1a_mini_cres.json"


@pytest.fixture
def mini_cres_data() -> dict:
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def hierarchy(mini_cres_data: dict):
    from tract.hierarchy import CREHierarchy
    return CREHierarchy.from_opencre(
        cres=mini_cres_data["cres"],
        fetch_timestamp=mini_cres_data["fetch_timestamp"],
        data_hash="abc123",
    )


class TestCREHierarchyConstruction:

    def test_hub_count(self, hierarchy) -> None:
        assert len(hierarchy.hubs) == 10

    def test_roots_sorted(self, hierarchy) -> None:
        assert hierarchy.roots == sorted(hierarchy.roots)
        assert set(hierarchy.roots) == {"ORPHAN-D", "ROOT-A", "ROOT-B", "ROOT-C"}

    def test_label_space_contains_leaves_only(self, hierarchy) -> None:
        for hub_id in hierarchy.label_space:
            assert hierarchy.hubs[hub_id].is_leaf

    def test_label_space_sorted(self, hierarchy) -> None:
        assert hierarchy.label_space == sorted(hierarchy.label_space)

    def test_label_space_count(self, hierarchy) -> None:
        # LEAF-A1a, LEAF-A1b, PAR-A2, LEAF-B1, LEAF-B2, ROOT-C, ORPHAN-D = 7 leaves
        assert len(hierarchy.label_space) == 7

    def test_depth_root(self, hierarchy) -> None:
        assert hierarchy.hubs["ROOT-A"].depth == 0

    def test_depth_parent(self, hierarchy) -> None:
        assert hierarchy.hubs["PAR-A1"].depth == 1

    def test_depth_leaf(self, hierarchy) -> None:
        assert hierarchy.hubs["LEAF-A1a"].depth == 2

    def test_parent_id(self, hierarchy) -> None:
        assert hierarchy.hubs["LEAF-A1a"].parent_id == "PAR-A1"
        assert hierarchy.hubs["PAR-A1"].parent_id == "ROOT-A"
        assert hierarchy.hubs["ROOT-A"].parent_id is None

    def test_children_ids(self, hierarchy) -> None:
        assert set(hierarchy.hubs["PAR-A1"].children_ids) == {"LEAF-A1a", "LEAF-A1b"}
        assert hierarchy.hubs["LEAF-A1a"].children_ids == []

    def test_hierarchy_path(self, hierarchy) -> None:
        assert hierarchy.hubs["LEAF-A1a"].hierarchy_path == "Root A > Parent A1 > Leaf A1a"
        assert hierarchy.hubs["ROOT-A"].hierarchy_path == "Root A"

    def test_branch_root_id(self, hierarchy) -> None:
        assert hierarchy.hubs["LEAF-A1a"].branch_root_id == "ROOT-A"
        assert hierarchy.hubs["ROOT-A"].branch_root_id == "ROOT-A"

    def test_orphan_is_leaf_and_root(self, hierarchy) -> None:
        orphan = hierarchy.hubs["ORPHAN-D"]
        assert orphan.is_leaf
        assert orphan.parent_id is None
        assert orphan.children_ids == []
        assert orphan.depth == 0
        assert orphan.branch_root_id == "ORPHAN-D"
        assert "ORPHAN-D" in hierarchy.label_space

    def test_orphan_root_is_leaf(self, hierarchy) -> None:
        rootc = hierarchy.hubs["ROOT-C"]
        assert rootc.is_leaf
        assert rootc.parent_id is None
        assert "ROOT-C" in hierarchy.label_space

    def test_sibling_hub_ids(self, hierarchy) -> None:
        node = hierarchy.hubs["LEAF-A1a"]
        assert "LEAF-A1b" in node.sibling_hub_ids
        assert "LEAF-A1a" not in node.sibling_hub_ids

    def test_version_and_metadata(self, hierarchy) -> None:
        assert hierarchy.version == "1.0"
        assert hierarchy.data_hash == "abc123"
        assert hierarchy.fetch_timestamp == "2026-04-28T12:00:00Z"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_hierarchy.py -v`
Expected: FAIL (module `tract.hierarchy` does not exist)

- [ ] **Step 3: Implement `tract/hierarchy.py`**

Create `tract/hierarchy.py`:

```python
"""TRACT CRE hierarchy tree model.

Provides HubNode and CREHierarchy Pydantic models for the CRE taxonomy.
This is the coordinate system — every downstream component depends on it.
"""
from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tract.io import atomic_write_json, load_json

logger = logging.getLogger(__name__)


class HubNode(BaseModel):
    """A single hub in the CRE hierarchy tree."""

    model_config = ConfigDict(frozen=True)

    hub_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    parent_id: str | None = None
    children_ids: list[str] = Field(default_factory=list)
    depth: int = Field(..., ge=0)
    branch_root_id: str = Field(..., min_length=1)
    hierarchy_path: str = Field(..., min_length=1)
    is_leaf: bool
    sibling_hub_ids: list[str] = Field(default_factory=list)


class CREHierarchy(BaseModel):
    """The full CRE hub tree — the coordinate system for hub assignment."""

    model_config = ConfigDict(frozen=True)

    hubs: dict[str, HubNode]
    roots: list[str]
    label_space: list[str]
    fetch_timestamp: str = Field(..., min_length=1)
    data_hash: str = Field(..., min_length=1)
    version: str = "1.0"

    @classmethod
    def from_opencre(
        cls,
        cres: list[dict[str, Any]],
        fetch_timestamp: str,
        data_hash: str,
    ) -> CREHierarchy:
        """Build a CREHierarchy from raw OpenCRE CRE records."""
        hub_names: dict[str, str] = {}
        for cre in cres:
            if cre.get("doctype") != "CRE":
                continue
            cre_id = cre["id"]
            cre_name = cre["name"]
            if not cre_id or not cre_name:
                raise ValueError(f"CRE record missing id or name: {cre!r:.200}")
            hub_names[cre_id] = cre_name

        children_map: dict[str, list[str]] = {}
        parent_map: dict[str, str] = {}
        for cre in cres:
            if cre.get("doctype") != "CRE":
                continue
            cre_id = cre["id"]
            for link in cre.get("links", []):
                if link.get("ltype") != "Contains":
                    continue
                doc = link.get("document", {})
                if doc.get("doctype") != "CRE":
                    continue
                child_id = doc.get("id", "")
                if child_id in hub_names:
                    children_map.setdefault(cre_id, []).append(child_id)
                    parent_map[child_id] = cre_id

        roots = sorted(hid for hid in hub_names if hid not in parent_map)

        depth_map: dict[str, int] = {}
        branch_map: dict[str, str] = {}
        path_map: dict[str, str] = {}
        queue: deque[str] = deque()

        for root_id in roots:
            depth_map[root_id] = 0
            branch_map[root_id] = root_id
            path_map[root_id] = hub_names[root_id]
            queue.append(root_id)

        while queue:
            current = queue.popleft()
            for child_id in children_map.get(current, []):
                if child_id not in depth_map:
                    depth_map[child_id] = depth_map[current] + 1
                    branch_map[child_id] = branch_map[current]
                    path_map[child_id] = f"{path_map[current]} > {hub_names[child_id]}"
                    queue.append(child_id)

        # Detect orphans (hubs with no depth assigned — not reachable from any root)
        unreachable = set(hub_names.keys()) - set(depth_map.keys())
        if unreachable:
            logger.warning(
                "Found %d unreachable hubs (treating as orphan roots): %s",
                len(unreachable),
                sorted(unreachable),
            )
            for orphan_id in sorted(unreachable):
                depth_map[orphan_id] = 0
                branch_map[orphan_id] = orphan_id
                path_map[orphan_id] = hub_names[orphan_id]
                if orphan_id not in roots:
                    roots.append(orphan_id)
            roots = sorted(roots)

        # Compute sibling_hub_ids per hub
        sibling_map: dict[str, list[str]] = {}
        for hub_id in hub_names:
            pid = parent_map.get(hub_id)
            if pid is not None:
                sibling_map[hub_id] = sorted(
                    cid for cid in children_map.get(pid, []) if cid != hub_id
                )
            else:
                sibling_map[hub_id] = []

        # Detect orphans and log
        orphan_ids = [
            hid for hid in roots
            if hid not in children_map or not children_map[hid]
        ]
        if orphan_ids:
            logger.warning(
                "Orphan hubs (root + leaf): %s",
                [(oid, hub_names[oid]) for oid in orphan_ids],
            )

        # Build HubNode objects
        hub_nodes: dict[str, HubNode] = {}
        for hub_id, name in hub_names.items():
            kids = children_map.get(hub_id, [])
            hub_nodes[hub_id] = HubNode(
                hub_id=hub_id,
                name=name,
                parent_id=parent_map.get(hub_id),
                children_ids=sorted(kids),
                depth=depth_map[hub_id],
                branch_root_id=branch_map[hub_id],
                hierarchy_path=path_map[hub_id],
                is_leaf=len(kids) == 0,
                sibling_hub_ids=sibling_map[hub_id],
            )

        label_space = sorted(
            hid for hid, node in hub_nodes.items() if node.is_leaf
        )

        logger.info(
            "Built hierarchy: %d hubs, %d roots, %d leaves",
            len(hub_nodes), len(roots), len(label_space),
        )

        hierarchy = cls(
            hubs=hub_nodes,
            roots=roots,
            label_space=label_space,
            fetch_timestamp=fetch_timestamp,
            data_hash=data_hash,
            version="1.0",
        )
        hierarchy.validate_integrity()
        return hierarchy

    def validate_integrity(self) -> None:
        """Validate the hierarchy tree. Raises ValueError on failure."""
        # 1. No dangling references
        for hub_id, node in self.hubs.items():
            if node.parent_id is not None and node.parent_id not in self.hubs:
                raise ValueError(
                    f"Hub {hub_id} has dangling parent_id: {node.parent_id}"
                )
            for child_id in node.children_ids:
                if child_id not in self.hubs:
                    raise ValueError(
                        f"Hub {hub_id} has dangling child_id: {child_id}"
                    )

        # 2. Depth consistency
        for hub_id, node in self.hubs.items():
            if node.parent_id is not None:
                parent = self.hubs[node.parent_id]
                if node.depth != parent.depth + 1:
                    raise ValueError(
                        f"Hub {hub_id} depth={node.depth} but parent "
                        f"{node.parent_id} depth={parent.depth}"
                    )

        # 3. All leaves reachable from a root
        for leaf_id in self.label_space:
            current: str | None = leaf_id
            visited: set[str] = set()
            while current is not None:
                if current in visited:
                    raise ValueError(f"Cycle detected at hub {current}")
                visited.add(current)
                current = self.hubs[current].parent_id
            # current is None means we reached a root (no parent)
            top = leaf_id
            for v in visited:
                if self.hubs[v].parent_id is None:
                    top = v
                    break
            if top not in self.roots:
                raise ValueError(
                    f"Leaf {leaf_id} not reachable from any root"
                )

        # 4. Label space determinism
        if self.label_space != sorted(self.label_space):
            raise ValueError("label_space is not sorted")

        # 5. Expected counts (warnings only)
        if len(self.hubs) != 522:
            logger.warning(
                "Expected 522 hubs, got %d", len(self.hubs)
            )
        if len(self.label_space) != 400:
            logger.warning(
                "Expected 400 leaves, got %d", len(self.label_space)
            )
        if len(self.roots) != 5:
            logger.warning(
                "Expected 5 roots, got %d", len(self.roots)
            )

    # ── Query methods ──────────────────────────────────────────────────

    def leaf_hub_ids(self) -> list[str]:
        """Return the label space (sorted leaf hub IDs)."""
        return list(self.label_space)

    def get_parent(self, hub_id: str) -> HubNode | None:
        """Return the parent node, or None if root."""
        node = self.hubs[hub_id]
        if node.parent_id is None:
            return None
        return self.hubs[node.parent_id]

    def get_children(self, hub_id: str) -> list[HubNode]:
        """Return child nodes of a hub."""
        return [self.hubs[cid] for cid in self.hubs[hub_id].children_ids]

    def get_siblings(self, hub_id: str) -> list[HubNode]:
        """Return sibling nodes (same parent, excluding self)."""
        return [self.hubs[sid] for sid in self.hubs[hub_id].sibling_hub_ids]

    def get_branch_hub_ids(self, root_id: str) -> list[str]:
        """Return all hub IDs under a root (including the root)."""
        if root_id not in self.hubs:
            raise ValueError(f"Unknown hub ID: {root_id}")
        result: list[str] = []
        q: deque[str] = deque([root_id])
        while q:
            current = q.popleft()
            result.append(current)
            q.extend(self.hubs[current].children_ids)
        return result

    def get_hierarchy_path(self, hub_id: str) -> str:
        """Return the cached hierarchy path string."""
        return self.hubs[hub_id].hierarchy_path

    def hub_by_name(self, name: str) -> HubNode | None:
        """Case-insensitive hub lookup by name. Returns first match or None."""
        lower = name.lower()
        for node in self.hubs.values():
            if node.name.lower() == lower:
                return node
        return None

    # ── Serialization ──────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Atomically save hierarchy to JSON."""
        atomic_write_json(self.model_dump(), path)
        logger.info("Saved hierarchy to %s", path)

    @classmethod
    def load(cls, path: Path) -> CREHierarchy:
        """Load hierarchy from JSON and validate."""
        data = load_json(path)
        hierarchy = cls.model_validate(data)
        hierarchy.validate_integrity()
        return hierarchy
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_hierarchy.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run mypy on the new module**

Run: `python -m mypy tract/hierarchy.py --strict`
Expected: Success

- [ ] **Step 6: Commit**

```bash
git add tract/hierarchy.py tests/test_hierarchy.py
git commit -m "feat: add CREHierarchy Pydantic model with tree operations"
```

---

### Task 5: Add CREHierarchy validation and query method tests

**Files:**
- Modify: `tests/test_hierarchy.py`

- [ ] **Step 1: Add validation and query tests**

Append to `tests/test_hierarchy.py`:

```python
class TestCREHierarchyValidation:

    def test_validate_passes_on_good_data(self, hierarchy) -> None:
        hierarchy.validate_integrity()

    def test_detects_dangling_parent(self, mini_cres_data: dict) -> None:
        from tract.hierarchy import CREHierarchy
        cres = mini_cres_data["cres"]
        # Inject a hub with a bogus parent
        cres.append({
            "doctype": "CRE", "id": "BAD-HUB", "name": "Bad Hub",
            "tags": [], "links": [],
        })
        # Manually add a Contains link pointing to BAD-HUB from a nonexistent parent
        # The easier way: just build and then tamper with the model
        h = CREHierarchy.from_opencre(cres, "2026-01-01T00:00:00Z", "test")
        # Now create a corrupted version
        bad_node = h.hubs["BAD-HUB"].model_copy(update={"parent_id": "NONEXISTENT"})
        bad_hubs = dict(h.hubs)
        bad_hubs["BAD-HUB"] = bad_node
        bad_h = h.model_copy(update={"hubs": bad_hubs})
        with pytest.raises(ValueError, match="dangling parent_id"):
            bad_h.validate_integrity()

    def test_detects_unsorted_label_space(self, hierarchy) -> None:
        reversed_ls = list(reversed(hierarchy.label_space))
        bad_h = hierarchy.model_copy(update={"label_space": reversed_ls})
        with pytest.raises(ValueError, match="not sorted"):
            bad_h.validate_integrity()


class TestCREHierarchyQueries:

    def test_leaf_hub_ids_returns_label_space(self, hierarchy) -> None:
        assert hierarchy.leaf_hub_ids() == list(hierarchy.label_space)

    def test_get_parent(self, hierarchy) -> None:
        parent = hierarchy.get_parent("LEAF-A1a")
        assert parent is not None
        assert parent.hub_id == "PAR-A1"

    def test_get_parent_of_root(self, hierarchy) -> None:
        assert hierarchy.get_parent("ROOT-A") is None

    def test_get_children(self, hierarchy) -> None:
        children = hierarchy.get_children("PAR-A1")
        child_ids = {c.hub_id for c in children}
        assert child_ids == {"LEAF-A1a", "LEAF-A1b"}

    def test_get_children_of_leaf(self, hierarchy) -> None:
        assert hierarchy.get_children("LEAF-A1a") == []

    def test_get_siblings(self, hierarchy) -> None:
        siblings = hierarchy.get_siblings("LEAF-A1a")
        sib_ids = {s.hub_id for s in siblings}
        assert sib_ids == {"LEAF-A1b"}

    def test_get_branch_hub_ids(self, hierarchy) -> None:
        branch = hierarchy.get_branch_hub_ids("ROOT-A")
        assert "ROOT-A" in branch
        assert "LEAF-A1a" in branch
        assert "LEAF-B1" not in branch

    def test_get_hierarchy_path(self, hierarchy) -> None:
        assert hierarchy.get_hierarchy_path("LEAF-A1a") == "Root A > Parent A1 > Leaf A1a"

    def test_hub_by_name(self, hierarchy) -> None:
        node = hierarchy.hub_by_name("leaf a1a")
        assert node is not None
        assert node.hub_id == "LEAF-A1a"

    def test_hub_by_name_not_found(self, hierarchy) -> None:
        assert hierarchy.hub_by_name("nonexistent") is None
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `python -m pytest tests/test_hierarchy.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_hierarchy.py
git commit -m "test: add validation and query tests for CREHierarchy"
```

---

### Task 6: Add CREHierarchy serialization tests and Phase 0 parity test

**Files:**
- Modify: `tests/test_hierarchy.py`

- [ ] **Step 1: Add serialization round-trip test**

Append to `tests/test_hierarchy.py`:

```python
class TestCREHierarchySerialization:

    def test_save_and_load_roundtrip(self, hierarchy, tmp_path: Path) -> None:
        from tract.hierarchy import CREHierarchy
        out = tmp_path / "hierarchy.json"
        hierarchy.save(out)
        loaded = CREHierarchy.load(out)
        assert loaded.label_space == hierarchy.label_space
        assert loaded.roots == hierarchy.roots
        assert len(loaded.hubs) == len(hierarchy.hubs)
        for hub_id in hierarchy.hubs:
            assert loaded.hubs[hub_id] == hierarchy.hubs[hub_id]

    def test_load_validates(self, hierarchy, tmp_path: Path) -> None:
        import json
        from tract.hierarchy import CREHierarchy
        out = tmp_path / "bad.json"
        data = hierarchy.model_dump()
        data["label_space"] = list(reversed(data["label_space"]))
        with open(out, "w") as f:
            json.dump(data, f)
        with pytest.raises(ValueError, match="not sorted"):
            CREHierarchy.load(out)


class TestPhase0Parity:

    def test_leaf_hub_ids_match_phase0(self) -> None:
        """Phase 1A label_space must match Phase 0 leaf_hub_ids on real data."""
        import hashlib
        from scripts.phase0.common import build_hierarchy as phase0_build
        from tract.hierarchy import CREHierarchy

        opencre_path = Path("data/raw/opencre/opencre_all_cres.json")
        if not opencre_path.exists():
            pytest.skip("OpenCRE data not available")

        data = load_json(opencre_path)
        cres = data["cres"]
        ts = data.get("fetch_timestamp", "unknown")
        raw = opencre_path.read_bytes()
        data_hash = hashlib.sha256(raw).hexdigest()

        phase0_tree = phase0_build(cres)
        phase0_leaves = sorted(phase0_tree.leaf_hub_ids())

        phase1a_tree = CREHierarchy.from_opencre(cres, ts, data_hash)

        assert phase1a_tree.label_space == phase0_leaves
```

Add `from tract.io import load_json` to the imports at the top of the file if not already present.

- [ ] **Step 2: Run tests to verify they pass**

Run: `python -m pytest tests/test_hierarchy.py -v`
Expected: ALL PASS (Phase 0 parity test runs against real data if available, otherwise skips)

- [ ] **Step 3: Commit**

```bash
git add tests/test_hierarchy.py
git commit -m "test: add serialization roundtrip and Phase 0 parity tests"
```

---

### Task 7: Create build_hierarchy.py CLI script

**Files:**
- Create: `scripts/phase1a/__init__.py`
- Create: `scripts/phase1a/build_hierarchy.py`

- [ ] **Step 1: Create the package marker**

Create an empty `scripts/phase1a/__init__.py`:

```python
```

- [ ] **Step 2: Create the build script**

Create `scripts/phase1a/build_hierarchy.py`:

```python
"""Build and validate the production CRE hierarchy.

Reads OpenCRE data, constructs the CREHierarchy, validates integrity,
and writes to data/processed/cre_hierarchy.json.

Usage: python -m scripts.phase1a.build_hierarchy
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from tract.config import PROCESSED_DIR, RAW_OPENCRE_DIR
from tract.hierarchy import CREHierarchy
from tract.io import load_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    opencre_path = RAW_OPENCRE_DIR / "opencre_all_cres.json"
    if not opencre_path.exists():
        raise FileNotFoundError(
            f"OpenCRE data not found at {opencre_path}. "
            "Run scripts/fetch_opencre.py first."
        )

    logger.info("Loading OpenCRE data from %s", opencre_path)
    data = load_json(opencre_path)
    cres = data["cres"]
    fetch_timestamp = data.get("fetch_timestamp", "unknown")

    raw_bytes = opencre_path.read_bytes()
    data_hash = hashlib.sha256(raw_bytes).hexdigest()
    logger.info("Data hash: %s", data_hash)

    hierarchy = CREHierarchy.from_opencre(cres, fetch_timestamp, data_hash)

    output_path = PROCESSED_DIR / "cre_hierarchy.json"
    hierarchy.save(output_path)

    logger.info(
        "Hierarchy saved: %d hubs, %d roots, %d leaves, %d label_space",
        len(hierarchy.hubs),
        len(hierarchy.roots),
        len(hierarchy.label_space),
        len(hierarchy.label_space),
    )

    # Print branch summary
    for root_id in hierarchy.roots:
        branch = hierarchy.get_branch_hub_ids(root_id)
        root_name = hierarchy.hubs[root_id].name
        logger.info("  Branch '%s': %d hubs", root_name, len(branch))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the build script**

Run: `python -m scripts.phase1a.build_hierarchy`
Expected: Logs showing 522 hubs, 5 roots, 400 leaves. Creates `data/processed/cre_hierarchy.json`.

- [ ] **Step 4: Verify the output file**

Run: `python -c "from tract.hierarchy import CREHierarchy; h = CREHierarchy.load('data/processed/cre_hierarchy.json'); print(f'{len(h.hubs)} hubs, {len(h.roots)} roots, {len(h.label_space)} leaves')"`
Expected: `522 hubs, 5 roots, 400 leaves`

- [ ] **Step 5: Commit**

```bash
git add scripts/phase1a/__init__.py scripts/phase1a/build_hierarchy.py data/processed/cre_hierarchy.json
git commit -m "feat: add build_hierarchy CLI and generate cre_hierarchy.json"
```

---

### Task 8: Implement hub description schema models

**Files:**
- Create: `tract/descriptions.py`
- Create: `tests/test_descriptions.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_descriptions.py`:

```python
"""Tests for tract.descriptions — hub description models and prompt rendering."""
from __future__ import annotations

import pytest
from pydantic import ValidationError


class TestHubDescriptionModel:

    def test_valid_description(self) -> None:
        from tract.descriptions import HubDescription
        desc = HubDescription(
            hub_id="123-456",
            hub_name="Input validation",
            hierarchy_path="Root > Security > Input validation",
            description="Covers input validation controls.",
            model="claude-opus-4-20250514",
            temperature=0.0,
            generated_at="2026-04-28T12:00:00Z",
            review_status="pending",
            reviewed_description=None,
            reviewer_notes=None,
        )
        assert desc.hub_id == "123-456"
        assert desc.review_status == "pending"

    def test_rejects_empty_description(self) -> None:
        from tract.descriptions import HubDescription
        with pytest.raises(ValidationError):
            HubDescription(
                hub_id="123-456",
                hub_name="Test",
                hierarchy_path="Root > Test",
                description="",
                model="test",
                temperature=0.0,
                generated_at="2026-04-28T12:00:00Z",
                review_status="pending",
                reviewed_description=None,
                reviewer_notes=None,
            )

    def test_rejects_invalid_review_status(self) -> None:
        from tract.descriptions import HubDescription
        with pytest.raises(ValidationError):
            HubDescription(
                hub_id="123-456",
                hub_name="Test",
                hierarchy_path="Root > Test",
                description="Some description.",
                model="test",
                temperature=0.0,
                generated_at="2026-04-28T12:00:00Z",
                review_status="invalid_status",
                reviewed_description=None,
                reviewer_notes=None,
            )


class TestHubDescriptionSetModel:

    def test_valid_set(self) -> None:
        from tract.descriptions import HubDescription, HubDescriptionSet
        desc = HubDescription(
            hub_id="123-456",
            hub_name="Test",
            hierarchy_path="Root > Test",
            description="Test description.",
            model="claude-opus-4-20250514",
            temperature=0.0,
            generated_at="2026-04-28T12:00:00Z",
            review_status="pending",
            reviewed_description=None,
            reviewer_notes=None,
        )
        desc_set = HubDescriptionSet(
            descriptions={"123-456": desc},
            generation_model="claude-opus-4-20250514",
            generation_timestamp="2026-04-28T12:00:00Z",
            data_hash="abc123",
            total_generated=1,
            total_pending_review=1,
        )
        assert len(desc_set.descriptions) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_descriptions.py -v`
Expected: FAIL (module `tract.descriptions` does not exist)

- [ ] **Step 3: Implement `tract/descriptions.py`**

Create `tract/descriptions.py`:

```python
"""TRACT hub description models and prompt rendering.

Provides Pydantic models for hub descriptions and the prompt template
used to generate them via Opus.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HubDescription(BaseModel):
    """A generated description for a single CRE leaf hub."""

    model_config = ConfigDict(str_strip_whitespace=True)

    hub_id: str = Field(..., min_length=1)
    hub_name: str = Field(..., min_length=1)
    hierarchy_path: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    temperature: float
    generated_at: str = Field(..., min_length=1)
    review_status: Literal["pending", "accepted", "edited", "rejected"] = "pending"
    reviewed_description: str | None = None
    reviewer_notes: str | None = None


class HubDescriptionSet(BaseModel):
    """Container for all hub descriptions with generation metadata."""

    model_config = ConfigDict(str_strip_whitespace=True)

    descriptions: dict[str, HubDescription]
    generation_model: str = Field(..., min_length=1)
    generation_timestamp: str = Field(..., min_length=1)
    data_hash: str = Field(..., min_length=1)
    total_generated: int = Field(..., ge=0)
    total_pending_review: int = Field(..., ge=0)


DESCRIPTION_SYSTEM_PROMPT: str = (
    "You are a cybersecurity taxonomy expert. Generate a precise 2-3 "
    "sentence description for a CRE (Common Requirements Enumeration) "
    "hub node.\n\n"
    "Write a description that:\n"
    "1. Defines what this hub covers in concrete terms\n"
    "2. Distinguishes it from its sibling hubs\n"
    "3. States the boundary of its scope (what it does NOT cover)\n\n"
    "Be specific and technical. Do not use filler phrases. "
    "Every word must add information."
)


def build_description_prompt(
    hub_name: str,
    hierarchy_path: str,
    sibling_names: list[str],
    linked_section_names: list[str],
) -> str:
    """Build the user message for generating one hub description."""
    siblings_str = ", ".join(sibling_names[:20]) if sibling_names else "(none)"
    linked_str = ", ".join(linked_section_names[:50]) if linked_section_names else "(none)"

    return (
        f"Hub name: {hub_name}\n"
        f"Hierarchy path: {hierarchy_path}\n"
        f"Sibling hubs (same parent): {siblings_str}\n"
        f"Linked standard sections: {linked_str}"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_descriptions.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add tract/descriptions.py tests/test_descriptions.py
git commit -m "feat: add hub description Pydantic models and prompt template"
```

---

### Task 9: Add prompt rendering tests

**Files:**
- Modify: `tests/test_descriptions.py`

- [ ] **Step 1: Add prompt rendering tests**

Append to `tests/test_descriptions.py`:

```python
class TestBuildDescriptionPrompt:

    def test_includes_hub_name(self) -> None:
        from tract.descriptions import build_description_prompt
        prompt = build_description_prompt(
            hub_name="Input validation",
            hierarchy_path="Root > Security > Input validation",
            sibling_names=["Output encoding", "Parameterized queries"],
            linked_section_names=["CWE-20", "ASVS V5.1"],
        )
        assert "Input validation" in prompt
        assert "Root > Security > Input validation" in prompt

    def test_includes_siblings(self) -> None:
        from tract.descriptions import build_description_prompt
        prompt = build_description_prompt(
            hub_name="Test",
            hierarchy_path="Root > Test",
            sibling_names=["Sibling A", "Sibling B"],
            linked_section_names=[],
        )
        assert "Sibling A" in prompt
        assert "Sibling B" in prompt

    def test_includes_linked_sections(self) -> None:
        from tract.descriptions import build_description_prompt
        prompt = build_description_prompt(
            hub_name="Test",
            hierarchy_path="Root > Test",
            sibling_names=[],
            linked_section_names=["CWE-79", "OWASP A7"],
        )
        assert "CWE-79" in prompt
        assert "OWASP A7" in prompt

    def test_handles_empty_siblings(self) -> None:
        from tract.descriptions import build_description_prompt
        prompt = build_description_prompt(
            hub_name="Test",
            hierarchy_path="Root > Test",
            sibling_names=[],
            linked_section_names=["CWE-79"],
        )
        assert "(none)" in prompt

    def test_handles_empty_linked_sections(self) -> None:
        from tract.descriptions import build_description_prompt
        prompt = build_description_prompt(
            hub_name="Test",
            hierarchy_path="Root > Test",
            sibling_names=["A"],
            linked_section_names=[],
        )
        assert "(none)" in prompt.split("Linked standard sections:")[1]
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `python -m pytest tests/test_descriptions.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_descriptions.py
git commit -m "test: add prompt rendering tests for hub descriptions"
```

---

### Task 10: Implement generate_descriptions.py CLI script

**Files:**
- Create: `scripts/phase1a/generate_descriptions.py`

- [ ] **Step 1: Create the generation script**

Create `scripts/phase1a/generate_descriptions.py`:

```python
"""Generate hub descriptions for all 400 leaf hubs via Opus.

Usage:
    python -m scripts.phase1a.generate_descriptions
    python -m scripts.phase1a.generate_descriptions --limit 10  # generate first 10 only
    python -m scripts.phase1a.generate_descriptions --dry-run   # show what would be generated
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from tract.config import (
    PHASE1A_DESCRIPTION_MAX_CONCURRENT,
    PHASE1A_DESCRIPTION_MAX_TOKENS,
    PHASE1A_DESCRIPTION_MODEL,
    PHASE1A_DESCRIPTION_SAVE_INTERVAL,
    PHASE1A_DESCRIPTION_TEMPERATURE,
    PHASE1A_DESCRIPTION_TIMEOUT_S,
    PROCESSED_DIR,
    TRAINING_DIR,
)
from tract.descriptions import (
    DESCRIPTION_SYSTEM_PROMPT,
    HubDescription,
    HubDescriptionSet,
    build_description_prompt,
)
from tract.hierarchy import CREHierarchy
from tract.io import atomic_write_json, load_json
from tract.sanitize import sanitize_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PARTIAL_PATH = PROCESSED_DIR / "hub_descriptions_partial.json"
OUTPUT_PATH = PROCESSED_DIR / "hub_descriptions.json"


def _get_api_key() -> str:
    """Retrieve Anthropic API key from env or pass manager."""
    import os
    import subprocess

    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    result = subprocess.run(
        ["pass", "anthropic/api-key"],
        capture_output=True, text=True, timeout=10, check=True,
    )
    api_key = result.stdout.strip()
    if not api_key:
        raise RuntimeError("pass returned empty API key")
    return api_key


def _load_hub_links() -> dict[str, list[str]]:
    """Load hub links and return hub_id -> list of section names."""
    links_path = TRAINING_DIR / "hub_links.jsonl"
    if not links_path.exists():
        raise FileNotFoundError(
            f"Hub links not found at {links_path}. "
            "Run parsers/extract_hub_links.py first."
        )

    import json
    hub_sections: dict[str, list[str]] = defaultdict(list)
    with open(links_path, encoding="utf-8") as f:
        for line in f:
            link = json.loads(line)
            section = link.get("section_name") or link.get("section_id", "")
            if section:
                hub_sections[link["cre_id"]].append(section)

    for hub_id in hub_sections:
        hub_sections[hub_id] = sorted(set(hub_sections[hub_id]))

    return dict(hub_sections)


def _load_existing_descriptions() -> dict[str, HubDescription]:
    """Load any previously generated descriptions for resume support."""
    if PARTIAL_PATH.exists():
        logger.info("Found partial file, loading for resume: %s", PARTIAL_PATH)
        data = load_json(PARTIAL_PATH)
        desc_set = HubDescriptionSet.model_validate(data)
        valid = {
            hub_id: desc
            for hub_id, desc in desc_set.descriptions.items()
            if desc.description
        }
        logger.info("Resuming with %d existing descriptions", len(valid))
        return valid
    return {}


async def _generate_all(
    hierarchy: CREHierarchy,
    hub_sections: dict[str, list[str]],
    existing: dict[str, HubDescription],
    limit: int | None,
) -> dict[str, HubDescription]:
    """Generate descriptions for all leaf hubs not already in existing."""
    import anthropic

    api_key = _get_api_key()
    client = anthropic.AsyncAnthropic(
        api_key=api_key,
        max_retries=3,
        timeout=PHASE1A_DESCRIPTION_TIMEOUT_S + 30,
    )
    semaphore = asyncio.Semaphore(PHASE1A_DESCRIPTION_MAX_CONCURRENT)
    descriptions = dict(existing)

    hub_ids_to_generate = [
        hid for hid in hierarchy.label_space
        if hid not in existing
    ]
    if limit is not None:
        hub_ids_to_generate = hub_ids_to_generate[:limit]

    logger.info(
        "Generating descriptions: %d to generate, %d already done, %d total leaf hubs",
        len(hub_ids_to_generate), len(existing), len(hierarchy.label_space),
    )

    async def generate_one(hub_id: str) -> tuple[str, HubDescription | Exception]:
        node = hierarchy.hubs[hub_id]
        siblings = hierarchy.get_siblings(hub_id)
        sibling_names = [s.name for s in siblings]
        linked_sections = hub_sections.get(hub_id, [])

        prompt = build_description_prompt(
            hub_name=node.name,
            hierarchy_path=node.hierarchy_path,
            sibling_names=sibling_names,
            linked_section_names=linked_sections,
        )

        async with semaphore:
            try:
                response = await asyncio.wait_for(
                    client.messages.create(
                        model=PHASE1A_DESCRIPTION_MODEL,
                        max_tokens=PHASE1A_DESCRIPTION_MAX_TOKENS,
                        temperature=PHASE1A_DESCRIPTION_TEMPERATURE,
                        system=DESCRIPTION_SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": prompt}],
                    ),
                    timeout=PHASE1A_DESCRIPTION_TIMEOUT_S,
                )
                raw_text = response.content[0].text.strip()
                clean_text = sanitize_text(raw_text)

                desc = HubDescription(
                    hub_id=hub_id,
                    hub_name=node.name,
                    hierarchy_path=node.hierarchy_path,
                    description=clean_text,
                    model=PHASE1A_DESCRIPTION_MODEL,
                    temperature=PHASE1A_DESCRIPTION_TEMPERATURE,
                    generated_at=datetime.now(timezone.utc).isoformat(),
                    review_status="pending",
                    reviewed_description=None,
                    reviewer_notes=None,
                )
                return hub_id, desc
            except Exception as exc:
                logger.error("Failed to generate for %s (%s): %s", hub_id, node.name, exc)
                return hub_id, exc

    try:
        generated_count = 0
        tasks = [generate_one(hid) for hid in hub_ids_to_generate]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        failures: list[str] = []
        for hub_id, result in results:
            if isinstance(result, Exception):
                failures.append(hub_id)
                continue
            descriptions[hub_id] = result
            generated_count += 1
            if generated_count % PHASE1A_DESCRIPTION_SAVE_INTERVAL == 0:
                _save_partial(descriptions, hierarchy)
                logger.info("Intermediate save at %d descriptions", generated_count)

        if failures:
            fail_rate = len(failures) / len(hub_ids_to_generate)
            logger.error(
                "%d/%d generations failed (%.1f%%): %s",
                len(failures), len(hub_ids_to_generate),
                fail_rate * 100, failures[:10],
            )
            if fail_rate > 0.10:
                raise RuntimeError(
                    f"Failure rate {fail_rate:.1%} exceeds 10% threshold"
                )
    finally:
        await client.close()

    return descriptions


def _save_partial(
    descriptions: dict[str, HubDescription],
    hierarchy: CREHierarchy,
) -> None:
    """Save intermediate results."""
    desc_set = HubDescriptionSet(
        descriptions=descriptions,
        generation_model=PHASE1A_DESCRIPTION_MODEL,
        generation_timestamp=datetime.now(timezone.utc).isoformat(),
        data_hash=hierarchy.data_hash,
        total_generated=len(descriptions),
        total_pending_review=sum(
            1 for d in descriptions.values() if d.review_status == "pending"
        ),
    )
    atomic_write_json(desc_set.model_dump(), PARTIAL_PATH)


def _save_final(
    descriptions: dict[str, HubDescription],
    hierarchy: CREHierarchy,
) -> None:
    """Save final results and validate cross-references."""
    leaf_ids = set(hierarchy.label_space)
    desc_ids = set(descriptions.keys())
    missing = leaf_ids - desc_ids
    extra = desc_ids - leaf_ids

    if missing:
        logger.warning("Missing descriptions for %d leaf hubs: %s", len(missing), sorted(missing)[:10])
    if extra:
        raise ValueError(f"Descriptions exist for non-leaf hubs: {sorted(extra)[:10]}")

    desc_set = HubDescriptionSet(
        descriptions=descriptions,
        generation_model=PHASE1A_DESCRIPTION_MODEL,
        generation_timestamp=datetime.now(timezone.utc).isoformat(),
        data_hash=hierarchy.data_hash,
        total_generated=len(descriptions),
        total_pending_review=sum(
            1 for d in descriptions.values() if d.review_status == "pending"
        ),
    )
    atomic_write_json(desc_set.model_dump(), OUTPUT_PATH)
    logger.info("Saved %d descriptions to %s", len(descriptions), OUTPUT_PATH)

    # Clean up partial file
    if PARTIAL_PATH.exists():
        PARTIAL_PATH.unlink()
        logger.info("Removed partial file %s", PARTIAL_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hub descriptions")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of hubs to generate")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be generated without calling API")
    args = parser.parse_args()

    hierarchy_path = PROCESSED_DIR / "cre_hierarchy.json"
    if not hierarchy_path.exists():
        raise FileNotFoundError(
            f"Hierarchy not found at {hierarchy_path}. "
            "Run scripts/phase1a/build_hierarchy.py first."
        )

    hierarchy = CREHierarchy.load(hierarchy_path)
    hub_sections = _load_hub_links()
    existing = _load_existing_descriptions()

    to_generate = [hid for hid in hierarchy.label_space if hid not in existing]
    if args.limit:
        to_generate = to_generate[:args.limit]

    if args.dry_run:
        logger.info("DRY RUN: would generate %d descriptions", len(to_generate))
        for hid in to_generate[:5]:
            node = hierarchy.hubs[hid]
            logger.info("  %s: %s (%s)", hid, node.name, node.hierarchy_path)
        if len(to_generate) > 5:
            logger.info("  ... and %d more", len(to_generate) - 5)
        return

    t0 = time.monotonic()
    descriptions = asyncio.run(
        _generate_all(hierarchy, hub_sections, existing, args.limit)
    )
    elapsed = time.monotonic() - t0

    _save_final(descriptions, hierarchy)
    logger.info("Done in %.1fs (%.1fs per hub)", elapsed, elapsed / max(len(to_generate), 1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax and imports**

Run: `python -c "import scripts.phase1a.generate_descriptions; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Run dry-run to validate**

Run: `python -m scripts.phase1a.generate_descriptions --dry-run --limit 3`
Expected: Logs showing "DRY RUN: would generate 3 descriptions" with hub names

- [ ] **Step 4: Commit**

```bash
git add scripts/phase1a/generate_descriptions.py
git commit -m "feat: add generate_descriptions CLI for 400 leaf hub descriptions"
```

---

### Task 11: Implement validate_descriptions.py CLI script

**Files:**
- Create: `scripts/phase1a/validate_descriptions.py`

- [ ] **Step 1: Create the validation script**

Create `scripts/phase1a/validate_descriptions.py`:

```python
"""Validate hub description review status.

Reports counts of accepted, edited, rejected, and pending descriptions.
Validates that all reviewed_description values pass sanitization.

Usage: python -m scripts.phase1a.validate_descriptions
"""
from __future__ import annotations

import logging
import sys

from tract.config import PROCESSED_DIR
from tract.descriptions import HubDescriptionSet
from tract.hierarchy import CREHierarchy
from tract.io import load_json
from tract.sanitize import sanitize_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    hierarchy_path = PROCESSED_DIR / "cre_hierarchy.json"
    descriptions_path = PROCESSED_DIR / "hub_descriptions.json"

    if not hierarchy_path.exists():
        logger.error("Hierarchy not found at %s", hierarchy_path)
        return 1
    if not descriptions_path.exists():
        logger.error("Descriptions not found at %s", descriptions_path)
        return 1

    hierarchy = CREHierarchy.load(hierarchy_path)
    data = load_json(descriptions_path)
    desc_set = HubDescriptionSet.model_validate(data)

    errors: list[str] = []

    # Check all leaf hubs present
    leaf_ids = set(hierarchy.label_space)
    desc_ids = set(desc_set.descriptions.keys())
    missing = leaf_ids - desc_ids
    extra = desc_ids - leaf_ids

    if missing:
        errors.append(f"Missing descriptions for {len(missing)} leaf hubs: {sorted(missing)[:5]}")
    if extra:
        errors.append(f"Extra descriptions for {len(extra)} non-leaf hubs: {sorted(extra)[:5]}")

    # Check no empty descriptions
    empty = [hid for hid, d in desc_set.descriptions.items() if not d.description]
    if empty:
        errors.append(f"{len(empty)} hubs have empty descriptions: {empty[:5]}")

    # Validate reviewed_description sanitization
    sanitize_errors: list[str] = []
    for hub_id, desc in desc_set.descriptions.items():
        if desc.reviewed_description is not None:
            try:
                sanitize_text(desc.reviewed_description)
            except ValueError as e:
                sanitize_errors.append(f"{hub_id}: {e}")

    if sanitize_errors:
        errors.append(f"{len(sanitize_errors)} reviewed descriptions fail sanitization: {sanitize_errors[:3]}")

    # Count by status
    from collections import Counter
    status_counts = Counter(d.review_status for d in desc_set.descriptions.values())

    logger.info("=== Description Review Status ===")
    logger.info("  Total:    %d", len(desc_set.descriptions))
    logger.info("  Pending:  %d", status_counts.get("pending", 0))
    logger.info("  Accepted: %d", status_counts.get("accepted", 0))
    logger.info("  Edited:   %d", status_counts.get("edited", 0))
    logger.info("  Rejected: %d", status_counts.get("rejected", 0))

    if errors:
        for error in errors:
            logger.error("VALIDATION ERROR: %s", error)
        return 1

    logger.info("All validations passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import scripts.phase1a.validate_descriptions; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/phase1a/validate_descriptions.py
git commit -m "feat: add validate_descriptions CLI for review status reporting"
```

---

### Task 12: Implement traditional framework extraction — tests first

**Files:**
- Create: `tests/test_traditional_frameworks.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_traditional_frameworks.py`:

```python
"""Tests for traditional framework extraction from OpenCRE."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "phase1a_mini_cres.json"


@pytest.fixture
def mini_cres() -> list[dict]:
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["cres"]


class TestSlugify:

    def test_basic_slugify(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import slugify
        assert slugify("Hello World") == "hello-world"

    def test_special_chars(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import slugify
        assert slugify("CWE-79: Cross-Site Scripting") == "cwe-79-cross-site-scripting"

    def test_max_length(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import slugify
        long_name = "a" * 200
        result = slugify(long_name)
        assert len(result) <= 80

    def test_empty_raises(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import slugify
        with pytest.raises(ValueError, match="empty"):
            slugify("")

    def test_whitespace_only_raises(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import slugify
        with pytest.raises(ValueError, match="empty"):
            slugify("   ")


class TestFrameworkSlugValidation:

    def test_valid_slug(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import validate_framework_slug
        validate_framework_slug("capec")
        validate_framework_slug("nist_800_53")
        validate_framework_slug("owasp_cheat_sheets")

    def test_rejects_path_traversal(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import validate_framework_slug
        with pytest.raises(ValueError):
            validate_framework_slug("../etc/passwd")

    def test_rejects_uppercase(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import validate_framework_slug
        with pytest.raises(ValueError):
            validate_framework_slug("CAPEC")

    def test_rejects_empty(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import validate_framework_slug
        with pytest.raises(ValueError):
            validate_framework_slug("")

    def test_rejects_too_long(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import validate_framework_slug
        with pytest.raises(ValueError):
            validate_framework_slug("a" * 60)


class TestTripleKeyDedup:

    def test_dedup_same_section_different_cre(self, mini_cres: list[dict]) -> None:
        from scripts.phase1a.extract_traditional_frameworks import extract_framework_controls
        controls = extract_framework_controls(
            mini_cres,
            framework_names={"Framework Alpha"},
            framework_id="framework_alpha",
        )
        # Framework Alpha has sections ALPHA-1, ALPHA-2, ALPHA-3 across different CREs
        # Each should appear once in the output
        ids = [c.control_id for c in controls]
        assert len(ids) == len(set(ids))

    def test_empty_section_id_different_names_kept(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import extract_framework_controls
        cres = [
            {"doctype": "CRE", "id": "HUB-1", "name": "Hub 1", "links": [
                {"ltype": "Linked To", "document": {"doctype": "Standard", "name": "TestFW", "sectionID": "", "section": "Section A"}},
                {"ltype": "Linked To", "document": {"doctype": "Standard", "name": "TestFW", "sectionID": "", "section": "Section B"}},
            ]},
        ]
        controls = extract_framework_controls(cres, {"TestFW"}, "test_fw")
        assert len(controls) == 2

    def test_empty_section_id_same_name_deduped(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import extract_framework_controls
        cres = [
            {"doctype": "CRE", "id": "HUB-1", "name": "Hub 1", "links": [
                {"ltype": "Linked To", "document": {"doctype": "Standard", "name": "TestFW", "sectionID": "", "section": "Section A"}},
            ]},
            {"doctype": "CRE", "id": "HUB-2", "name": "Hub 2", "links": [
                {"ltype": "Linked To", "document": {"doctype": "Standard", "name": "TestFW", "sectionID": "", "section": "Section A"}},
            ]},
        ]
        controls = extract_framework_controls(cres, {"TestFW"}, "test_fw")
        assert len(controls) == 1


class TestBareIdHandling:

    def test_capec_bare_id(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import extract_framework_controls
        cres = [
            {"doctype": "CRE", "id": "HUB-1", "name": "Hub 1", "links": [
                {"ltype": "Automatically Linked To", "document": {
                    "doctype": "Standard", "name": "CAPEC", "sectionID": "184", "section": ""
                }},
            ]},
        ]
        controls = extract_framework_controls(cres, {"CAPEC"}, "capec")
        assert len(controls) == 1
        assert controls[0].control_id == "capec:184"
        assert controls[0].title == "CAPEC-184"

    def test_section_name_used_when_available(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import extract_framework_controls
        cres = [
            {"doctype": "CRE", "id": "HUB-1", "name": "Hub 1", "links": [
                {"ltype": "Linked To", "document": {
                    "doctype": "Standard", "name": "CAPEC",
                    "sectionID": "184",
                    "section": "Software Integrity Attack"
                }},
            ]},
        ]
        controls = extract_framework_controls(cres, {"CAPEC"}, "capec")
        assert controls[0].title == "Software Integrity Attack"

    def test_link_type_in_metadata(self) -> None:
        from scripts.phase1a.extract_traditional_frameworks import extract_framework_controls
        cres = [
            {"doctype": "CRE", "id": "HUB-1", "name": "Hub 1", "links": [
                {"ltype": "Automatically Linked To", "document": {
                    "doctype": "Standard", "name": "CAPEC", "sectionID": "184", "section": ""
                }},
            ]},
        ]
        controls = extract_framework_controls(cres, {"CAPEC"}, "capec")
        assert controls[0].metadata is not None
        assert controls[0].metadata["link_type"] == "AutomaticallyLinkedTo"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_traditional_frameworks.py -v`
Expected: FAIL (module does not exist)

- [ ] **Step 3: Commit**

```bash
git add tests/test_traditional_frameworks.py
git commit -m "test: add tests for traditional framework extraction"
```

---

### Task 13: Implement extract_traditional_frameworks.py

**Files:**
- Create: `scripts/phase1a/extract_traditional_frameworks.py`

- [ ] **Step 1: Implement the extraction script**

Create `scripts/phase1a/extract_traditional_frameworks.py`:

```python
"""Extract traditional framework controls from OpenCRE link metadata.

Extracts the 19 OpenCRE frameworks that lack primary-source parsers
and writes per-framework JSON + unified all_controls.json.

Usage: python -m scripts.phase1a.extract_traditional_frameworks
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tract.config import (
    AI_PARSER_FRAMEWORK_IDS,
    COUNT_TOLERANCE,
    OPENCRE_EXTRACT_FRAMEWORK_IDS,
    OPENCRE_FRAMEWORK_ID_MAP,
    PHASE1A_FRAMEWORK_SLUG_RE,
    PROCESSED_DIR,
    PROCESSED_FRAMEWORKS_DIR,
    RAW_OPENCRE_DIR,
)
from tract.io import atomic_write_json, load_json
from tract.sanitize import sanitize_text
from tract.schema import Control, FrameworkOutput

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(PHASE1A_FRAMEWORK_SLUG_RE)
_BARE_ID_FRAMEWORKS = {"CAPEC", "CWE"}


def validate_framework_slug(slug: str) -> None:
    """Validate a framework_id slug. Raises ValueError if invalid."""
    if not _SLUG_RE.match(slug):
        raise ValueError(
            f"Invalid framework slug: {slug!r} "
            f"(must match {PHASE1A_FRAMEWORK_SLUG_RE})"
        )


def slugify(name: str) -> str:
    """Generate a URL-safe slug from a section name."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:80]
    if not slug:
        raise ValueError(f"Slugified name is empty. Original: {name!r}")
    return slug


def _normalize_link_type(raw: str) -> str | None:
    """Normalize link type string. Returns None if not a training link."""
    lower = raw.replace(" ", "").lower()
    if lower == "linkedto":
        return "LinkedTo"
    if lower == "automaticallylinkedto":
        return "AutomaticallyLinkedTo"
    return None


def extract_framework_controls(
    cres: list[dict[str, Any]],
    framework_names: set[str],
    framework_id: str,
) -> list[Control]:
    """Extract unique controls for a framework from OpenCRE CRE records."""
    seen: set[tuple[str, str, str]] = set()
    controls: list[Control] = []
    bare_id_count = 0

    for cre in cres:
        if cre.get("doctype") != "CRE":
            continue
        for link in cre.get("links", []):
            doc = link.get("document", {})
            if doc.get("doctype") != "Standard":
                continue

            standard_name = doc.get("name", "")
            if standard_name not in framework_names:
                continue

            raw_ltype = link.get("ltype", link.get("type", ""))
            link_type = _normalize_link_type(raw_ltype)
            if link_type is None:
                continue

            section_id = str(doc.get("sectionID", "")).strip()
            section_name = str(doc.get("section", "")).strip()

            dedup_key = (framework_id, section_id, section_name)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            # Build control_id
            if section_id:
                control_id_suffix = section_id
            else:
                control_id_suffix = slugify(section_name)

            control_id = f"{framework_id}:{control_id_suffix}"

            # Build title
            if section_name:
                title = section_name
            elif section_id and standard_name in _BARE_ID_FRAMEWORKS:
                title = f"{standard_name}-{section_id}"
                bare_id_count += 1
            else:
                title = section_id or "(unknown)"

            try:
                description = sanitize_text(title)
            except ValueError:
                description = title if title else "(no description)"

            controls.append(Control(
                control_id=control_id,
                title=title,
                description=description,
                full_text=None,
                hierarchy_level=None,
                parent_id=None,
                parent_name=None,
                metadata={
                    "opencre_standard_name": standard_name,
                    "opencre_section_id": section_id,
                    "link_type": link_type,
                },
            ))

    if bare_id_count > 0:
        logger.info(
            "%s: %d bare-ID controls (no section name)",
            framework_id, bare_id_count,
        )

    controls.sort(key=lambda c: c.control_id)
    return controls


def _count_links_per_framework(
    cres: list[dict[str, Any]],
) -> dict[str, int]:
    """Count raw training links per framework_id."""
    counts: dict[str, int] = defaultdict(int)
    for cre in cres:
        if cre.get("doctype") != "CRE":
            continue
        for link in cre.get("links", []):
            doc = link.get("document", {})
            if doc.get("doctype") != "Standard":
                continue
            raw_ltype = link.get("ltype", link.get("type", ""))
            if _normalize_link_type(raw_ltype) is None:
                continue
            standard_name = doc.get("name", "")
            fw_id = OPENCRE_FRAMEWORK_ID_MAP.get(standard_name)
            if fw_id:
                counts[fw_id] += 1
    return dict(counts)


def _build_id_to_names_map() -> dict[str, set[str]]:
    """Build map: framework_id -> set of OpenCRE standard names."""
    id_to_names: dict[str, set[str]] = defaultdict(set)
    for name, fwid in OPENCRE_FRAMEWORK_ID_MAP.items():
        id_to_names[fwid].add(name)
    return dict(id_to_names)


def main() -> None:
    opencre_path = RAW_OPENCRE_DIR / "opencre_all_cres.json"
    if not opencre_path.exists():
        raise FileNotFoundError(f"OpenCRE data not found at {opencre_path}")

    data = load_json(opencre_path)
    cres = data["cres"]
    fetch_timestamp = data.get("fetch_timestamp", "unknown")

    # Parse date from fetch_timestamp for version string
    try:
        fetch_date = datetime.fromisoformat(fetch_timestamp).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        fetch_date = "unknown"

    link_counts = _count_links_per_framework(cres)
    id_to_names = _build_id_to_names_map()

    PROCESSED_FRAMEWORKS_DIR.mkdir(parents=True, exist_ok=True)

    extracted_frameworks: list[FrameworkOutput] = []

    for framework_id in sorted(OPENCRE_EXTRACT_FRAMEWORK_IDS):
        validate_framework_slug(framework_id)

        names = id_to_names.get(framework_id)
        if not names:
            logger.warning("No OpenCRE names mapped to framework_id=%s, skipping", framework_id)
            continue

        controls = extract_framework_controls(cres, names, framework_id)
        if not controls:
            logger.warning("No controls extracted for %s, skipping", framework_id)
            continue

        # Pick the shortest name as framework_name
        framework_name = min(names, key=len)

        fw_output = FrameworkOutput(
            framework_id=framework_id,
            framework_name=framework_name,
            version=f"opencre-{fetch_date}",
            source_url="https://opencre.org",
            fetched_date=fetch_timestamp,
            mapping_unit_level="section",
            controls=controls,
        )

        output_path = PROCESSED_FRAMEWORKS_DIR / f"{framework_id}.json"
        atomic_write_json(fw_output.model_dump(), output_path)

        raw_links = link_counts.get(framework_id, 0)
        deviation = abs(len(controls) - raw_links) / max(raw_links, 1)
        if deviation > COUNT_TOLERANCE:
            logger.warning(
                "%s: %d controls vs %d raw links (%.1f%% deviation)",
                framework_id, len(controls), raw_links, deviation * 100,
            )

        logger.info(
            "Extracted %s: %d controls (%d raw links)",
            framework_id, len(controls), raw_links,
        )
        extracted_frameworks.append(fw_output)

    # Build all_controls.json: AI parsers take precedence
    all_frameworks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    # First: load existing AI parser outputs
    for fw_path in sorted(PROCESSED_FRAMEWORKS_DIR.glob("*.json")):
        fw_id = fw_path.stem
        if fw_id in AI_PARSER_FRAMEWORK_IDS:
            fw_data = load_json(fw_path)
            all_frameworks.append(fw_data)
            seen_ids.add(fw_id)

    # Then: add extracted traditional frameworks
    for fw in extracted_frameworks:
        if fw.framework_id not in seen_ids:
            all_frameworks.append(fw.model_dump())
            seen_ids.add(fw.framework_id)

    all_frameworks.sort(key=lambda f: f["framework_id"])
    total_controls = sum(len(f["controls"]) for f in all_frameworks)

    all_controls = {
        "framework_count": len(all_frameworks),
        "total_controls": total_controls,
        "generated_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "frameworks": all_frameworks,
    }
    atomic_write_json(all_controls, PROCESSED_DIR / "all_controls.json")

    logger.info(
        "all_controls.json: %d frameworks, %d total controls",
        len(all_frameworks), total_controls,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `python -m pytest tests/test_traditional_frameworks.py -v`
Expected: ALL PASS

- [ ] **Step 3: Run mypy**

Run: `python -m mypy scripts/phase1a/extract_traditional_frameworks.py --strict`
Expected: Success (or minor issues to fix)

- [ ] **Step 4: Commit**

```bash
git add scripts/phase1a/extract_traditional_frameworks.py
git commit -m "feat: add traditional framework extraction from OpenCRE"
```

---

### Task 14: Add integration test for traditional framework extraction

**Files:**
- Modify: `tests/test_traditional_frameworks.py`

- [ ] **Step 1: Add integration tests**

Append to `tests/test_traditional_frameworks.py`:

```python
class TestEndToEndExtraction:
    """Integration tests against real OpenCRE data (skipped if not available)."""

    @pytest.fixture
    def real_cres(self) -> list[dict]:
        path = Path("data/raw/opencre/opencre_all_cres.json")
        if not path.exists():
            pytest.skip("OpenCRE data not available")
        import json
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data["cres"]

    def test_extracts_19_frameworks(self, real_cres: list[dict]) -> None:
        from tract.config import OPENCRE_EXTRACT_FRAMEWORK_IDS, OPENCRE_FRAMEWORK_ID_MAP
        from scripts.phase1a.extract_traditional_frameworks import extract_framework_controls
        from collections import defaultdict

        id_to_names: dict[str, set[str]] = defaultdict(set)
        for name, fwid in OPENCRE_FRAMEWORK_ID_MAP.items():
            id_to_names[fwid].add(name)

        extracted_count = 0
        for fw_id in sorted(OPENCRE_EXTRACT_FRAMEWORK_IDS):
            names = id_to_names.get(fw_id)
            if not names:
                continue
            controls = extract_framework_controls(real_cres, names, fw_id)
            if controls:
                extracted_count += 1
                assert len(controls) > 0, f"{fw_id} has zero controls"
                # Verify control_ids are unique
                ids = [c.control_id for c in controls]
                assert len(ids) == len(set(ids)), f"{fw_id} has duplicate control_ids"

        assert extracted_count == 19

    def test_capec_has_expected_controls(self, real_cres: list[dict]) -> None:
        from scripts.phase1a.extract_traditional_frameworks import extract_framework_controls
        controls = extract_framework_controls(real_cres, {"CAPEC"}, "capec")
        # CAPEC has 1799 links but many point to same section
        assert len(controls) > 100
        assert all(c.control_id.startswith("capec:") for c in controls)

    def test_all_controls_no_empty_descriptions(self, real_cres: list[dict]) -> None:
        from tract.config import OPENCRE_EXTRACT_FRAMEWORK_IDS, OPENCRE_FRAMEWORK_ID_MAP
        from scripts.phase1a.extract_traditional_frameworks import extract_framework_controls
        from collections import defaultdict

        id_to_names: dict[str, set[str]] = defaultdict(set)
        for name, fwid in OPENCRE_FRAMEWORK_ID_MAP.items():
            id_to_names[fwid].add(name)

        for fw_id in sorted(OPENCRE_EXTRACT_FRAMEWORK_IDS):
            names = id_to_names.get(fw_id)
            if not names:
                continue
            controls = extract_framework_controls(real_cres, names, fw_id)
            for c in controls:
                assert c.description, f"{fw_id}:{c.control_id} has empty description"
```

- [ ] **Step 2: Run integration tests**

Run: `python -m pytest tests/test_traditional_frameworks.py -v`
Expected: ALL PASS (integration tests run if real data available, skip otherwise)

- [ ] **Step 3: Commit**

```bash
git add tests/test_traditional_frameworks.py
git commit -m "test: add integration tests for traditional framework extraction"
```

---

### Task 15: Run the traditional framework extraction script

**Files:**
- (no new files — runs existing script)

- [ ] **Step 1: Run the extraction**

Run: `python -m scripts.phase1a.extract_traditional_frameworks`
Expected: Logs showing 19 frameworks extracted with control counts, then `all_controls.json` rebuilt

- [ ] **Step 2: Verify output files exist**

Run: `ls data/processed/frameworks/ | wc -l`
Expected: 31 (12 AI + 19 traditional)

Run: `python -c "from tract.io import load_json; d = load_json('data/processed/all_controls.json'); print(f'{d[\"framework_count\"]} frameworks, {d[\"total_controls\"]} controls')"`
Expected: `31 frameworks, NNNN controls` (total depends on dedup)

- [ ] **Step 3: Commit output files**

```bash
git add data/processed/frameworks/ data/processed/all_controls.json
git commit -m "feat: extract 19 traditional frameworks from OpenCRE"
```

---

### Task 16: Run the full test suite and type check

**Files:**
- (no new files — validation only)

- [ ] **Step 1: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: Run mypy strict on all new modules**

Run: `python -m mypy tract/hierarchy.py tract/descriptions.py scripts/phase1a/ --strict`
Expected: Success (or fix any type errors)

- [ ] **Step 3: Fix any type errors found**

If mypy reports issues, fix them in the relevant files and re-run until clean.

- [ ] **Step 4: Commit any type fixes**

```bash
git add -u
git commit -m "fix: resolve mypy strict type errors in Phase 1A modules"
```

---

### Task 17: Generate hub descriptions (API call — run when ready)

This task makes real API calls (~$15 cost). Run only when Tasks 1-16 are complete and all tests pass.

**Files:**
- (runs existing script)

- [ ] **Step 1: Run the generation script for all 400 hubs**

Run: `python -m scripts.phase1a.generate_descriptions`
Expected: ~15-20 minutes. Logs show progress with intermediate saves every 50 hubs. Final output at `data/processed/hub_descriptions.json`.

- [ ] **Step 2: Validate the generated descriptions**

Run: `python -m scripts.phase1a.validate_descriptions`
Expected: `All validations passed.` with 400 pending descriptions.

- [ ] **Step 3: Commit the generated descriptions**

```bash
git add data/processed/hub_descriptions.json
git commit -m "feat: generate descriptions for all 400 leaf hubs via Opus"
```

---

## Summary

| Task | Component | Dependencies |
|------|-----------|-------------|
| 1 | Config constants | — |
| 2 | Sanitize zero-width | — |
| 3 | Test fixture | — |
| 4 | CREHierarchy model | 1, 3 |
| 5 | Hierarchy validation tests | 4 |
| 6 | Serialization + Phase 0 parity | 4, 5 |
| 7 | build_hierarchy.py CLI | 4 |
| 8 | Description schema models | — |
| 9 | Prompt rendering tests | 8 |
| 10 | generate_descriptions.py CLI | 1, 2, 7, 8 |
| 11 | validate_descriptions.py CLI | 8, 7 |
| 12 | Extraction tests | 3 |
| 13 | extract_traditional_frameworks.py | 1, 12 |
| 14 | Extraction integration tests | 13 |
| 15 | Run extraction | 13 |
| 16 | Full test suite + mypy | 1-14 |
| 17 | Generate descriptions (API) | 10, 16 |

**Parallelizable groups:**
- Tasks 1, 2, 3, 8 can run in parallel (no dependencies on each other)
- Tasks 5, 6 depend on 4 but are independent of each other
- Tasks 12, 9 can run in parallel after their respective dependencies
- Tasks 10, 11, 13 are independent after their dependencies complete
