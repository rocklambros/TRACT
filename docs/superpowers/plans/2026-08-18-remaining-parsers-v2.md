# Eleven Remaining Parsers, Measured Against One Corpus Instrument

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the eleven frameworks that still have no parser a real one, and prove the corpus got better rather than merely louder, by building the measurement first and gating every parser on it.

**Architecture:** Task 1 builds `tract/corpus_report.py`, the single instrument. It resolves every curated link through `ProseIndex.lookup` and reports, per framework, seven quantities: links by channel, **distinct resolved anchors**, links per anchor, anchors truncated at `MAX_ANCHOR_CHARS`, anchors nested inside another anchor of the same framework, controls dropped by `_is_prose`, and wrong-anchor risk. It is run against the current corpus before a line of parser code exists, and that BEFORE state is committed. The same instrument, the same code path, is the per-parser acceptance gate and the final corpus report. Each parser task states its framework's arithmetic join ceiling, derived from the link file and the source before the parser is written, and the floor the parser is measured against is that ceiling rounded down. Two frameworks need machinery that does not exist: an `alt_ids` channel on `ProseIndex` mirroring the existing `alt_titles`, and the retirement of two link gates that key on a section title.

**Tech Stack:** Python 3.12, pydantic v2, pdfplumber, openpyxl, PyYAML, beautifulsoup4, defusedxml, pytest, mypy --strict.

**Spec:** `docs/superpowers/specs/2026-08-15-semantic-rebuild-design.md` (v2), Part 1.

**Supersedes:** `docs/superpowers/plans/2026-08-16-remaining-parsers.md`, rejected 2026-08-16 after a four-perspective premortem returned ~30 findings including 10 Critical. Source knowledge is carried forward; structure is not.

---

## Global Constraints

Copied from the spec (Part 1) and the run ledger (`.superpowers/autonomous-run/RUN-LEDGER.md`). These bind every task below.

- **All inference and training runs on RunPod, never locally.** Nothing in this plan loads a model, so all of it runs locally. Unit tests, lint and typecheck are local by CLAUDE.md.
- **`data/raw/` is immutable.** Parsers read it, never write it.
- **Licensed text never enters git.** `RESTRICTED_FRAMEWORK_IDS` is `{etsi, iso_27001}`. Both processed files are already in `.gitignore` (lines 37-38) and route to the gitignored `data/processed/licensed/` overlay. Never `git add -f` either. `data/processed/repair_audit/` is gitignored for the same reason.
- **Never republish to HuggingFace.** No task here touches a publish path.
- **No AI attribution** in commit messages, comments, or docs. The git author stays the human.
- **Type everything.** All signatures fully typed. `mypy tract/ parsers/ scripts/ --strict` must pass.
- **Fail loud.** `raise ValueError` with a specific message. No bare `except`. No `return None` to signal failure.
- **Atomic writes only**, via `tract.io.atomic_write_json`.
- **Deterministic output.** Sorted keys, no clock reads in any written artifact. `fetched_date` is a `ClassVar` per parser, never `date.today()`.
- **Every number carries `[measured]`, `[derived]` or `[unmeasured]`.** No threshold anywhere may depend on an `[unmeasured]` value (ledger lesson 8).
- **Any transform that moves or synthesises text emits an audit record and fails closed** (ledger lesson 7). `BaseParser.write_repair_audit` exists; use it. It is written unconditionally, empty list included, so a missing file means the parser never ran.
- **Compute the attainable range of every threshold and assert it contains the trigger** (ledger lesson 3). A floor above a source's arithmetic maximum is not a gate, it is a guaranteed failure.
- **Read the file before writing the task** (ledger lesson 2). Every code snippet below was written against the file as it stands at `8cf44b3`.
- **Baselines must be captured under identical conditions** (ledger lesson 5). The BEFORE and AFTER instrument runs use the same command, the same interpreter, and the same corpus path.

### The interpreter

`python3` on this machine is Homebrew 3.13.7 and has none of this project's dependencies. The `tract` console script's interpreter (`/Users/klambros/.local/share/uv/tools/tract/bin/python3`) has pydantic, PyYAML, bs4, lxml and numpy but **not** pdfplumber, openpyxl, defusedxml, pytest or mypy. **[measured]** The interpreter that has all of them is the one `pytest` already resolves to:

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
```

Every command in this plan uses that variable. Verify it once, at the top of Task 1, and never substitute `python3`.

`openpyxl` is used by no file in this repository today and appears in neither `requirements.txt` nor `pyproject.toml`. **[measured]** Task 8 adds it pinned. `defusedxml==0.7.1` is already pinned in `requirements.txt` but is not installed in the 3.12 environment **[measured]**; Task 15 installs it, because the corpus rebuild cannot run `parse_capec.py` or `parse_cwe.py` without it.

### Three contract facts that decide parser design

Read out of `tract/text_selection.py` and `tract/parsers/base.py` at `8cf44b3`, not inferred.

1. **`ProseIndex` prefers `full_text` over `description`, unconditionally.** `ProseIndex.__init__` takes `full_text` when it is non-empty and never looks at `description`. So whatever a parser puts in `full_text` **is the anchor the model sees**. `full_text` is not free storage.
2. **`BaseParser._sanitize_control` sets `full_text` behind the parser's back.** When `description` exceeds `DESCRIPTION_MAX_LENGTH` (2000), `sanitize_text(..., return_full=True)` returns the full text and it is written to `full_text`, **discarding whatever the parser put there** (`base.py:377-383`). A parser that emits a 3,000-character description has chosen a 3,000-character anchor whether it meant to or not.
3. **A control whose `description` does not exceed its `title` by `PROSE_MIN_EXTRA_CHARS` (20) and has no `full_text` is not indexed at all.** `ProseIndex.__init__` hits `continue`. Its links resolve to nothing and fall back to the section title. This is invisible to `honest_prose_fraction`, which uses a different rule (60 characters, and merely different from the title). Task 1's instrument counts these; nothing did before.

### Resolution order, and why it is not changed

`ProseIndex.lookup` tries **title first, then id**. That order was written to fix a real defect: NIST AI 100-2 links carry the containing subsection's id for three distinct mitigations, and id-first gave all three the same paragraph. The order stays. Each parser below states how it avoids the other failure mode — a link's `section_name` matching the wrong control's title:

| framework | risk | how this plan avoids it |
|---|---|---|
| biml | `Data Confidentiality` names two different risks in two documents; `Hosting` names three link rows across two documents. 7 of 21 rows participate in a label collision. **[measured]** | titles are document-scoped (`Hosting (BIML-24(LLM))`), so no link name can match a title; every row resolves through the id channel, with `alt_ids` for the 7 unprefixed ids |
| etsi | 24 technique names over 16 section ids, and three of those names span two clauses each **[measured]**, so registering them all as `alt_titles` would let the title channel answer with a clause the link did not name | only the two rows whose `section_id` is itself a name get an `alt_title`; the other 34 resolve through the id channel against a `control_id` that is the clause number |
| nist_ssdf | `section_name` is the task statement verbatim for 36 of 46 rows **[measured]**; if the parser also used it as `title`, `_is_prose` would drop every control | `title` is the task id, `description` is the task statement |
| wstg | `section_name == section_id` for all 118 rows **[measured]**, e.g. `WSTG-INFO-01` | `title` is the file's H1, which no link name spells, so every row resolves through the id channel |
| owasp_proactive_controls, nist_800_63 | same shape, `section_name == section_id`, 2-7 characters | same: the title is the human title, the id channel carries the join |
| csa_ccm | 7 rows carry retired `IVS-*` ids that v4.1.0 renamed to `I&S-*`; their `section_name` matches the new control title exactly **[measured]** | title-first resolves all 7 with no rename map at all |

---

### Task 1: The corpus quality instrument, and the BEFORE state

This is the only instrument in this plan. Everything downstream is measured with it. It is built and run before any parser exists, against a corpus where ISO 27001 resolves 92 of 94 links and the eleven pending frameworks resolve 0 of 734. **[measured]** That contrast is a free test of whether the instrument can see the difference it exists to see.

Counting links resolved is not enough and that is why the previous plan was rejected. 615 links unstacked onto 615 distinct anchors and 615 links collapsed onto 40 coarse anchors produce the same rising line. The report's load-bearing column is **distinct resolved anchors**.

Measured on the current corpus, the shape the instrument must be able to show:

| framework | links | resolved | distinct anchors | links/anchor | truncated |
|---|---|---|---|---|---|
| owasp_cheat_sheets | 391 | 391 | 49 | 7.98 | 384 |
| capec | 1799 | 1799 | 349 | 5.15 | 24 |
| cwe | 613 | 612 | 245 | 2.50 | 13 |
| nist_ai_100_2 | 45 | 45 | 22 | 2.05 | 21 |
| iso_27001 | 94 | 92 | 91 | 1.01 | 0 |
| the eleven pending | 734 | 0 | 0 | — | 0 |

**[measured, all]** A pipeline that reports only "3,666 of 4,405 curated links resolved" cannot distinguish the first row from the fifth. The report reads `hub_links_curated.jsonl` (4,405 rows) rather than `hub_links_by_framework.json` (4,406), because the curated file is what the trainer reads; the extra row is an `owasp_ai_exchange` duplicate. **[measured]**

**Files:**
- Create: `tract/corpus_report.py`
- Create: `scripts/corpus_report.py`
- Create: `tests/test_corpus_report.py`
- Create: `results/corpus/before_8cf44b3.json`

**Interfaces:**
- Consumes: `tract.text_selection.ProseIndex`, `canonical_framework`, `normalize_section_id`, `prepare_anchor`, `merged_corpus_path`, `_is_prose`; `tract.config.MAX_ANCHOR_CHARS`; `data/training/hub_links_curated.jsonl`.
- Produces:
  - `FrameworkJoin` dataclass with fields `framework_id: str`, `standard_name: str`, `links: int`, `by_title: int`, `by_id: int`, `unresolved: int`, `distinct_anchors: int`, `links_per_anchor: float`, `truncated: int`, `nested_anchors: int`, `dropped_by_prose_rule: int`, `wrong_anchor_risk: int`, `resolution_rate: float`.
  - `build_corpus_report(links_path: Path | None = None, corpus_path: Path | None = None) -> CorpusReport`
  - `CorpusReport` dataclass with `per_framework: list[FrameworkJoin]`, `totals: FrameworkJoin`, `corpus_path: str`, `corpus_sha256: str`, `links_path: str`, `links_sha256: str`, and `to_json() -> dict[str, object]`.
  - `check_join_floors(report: CorpusReport, floors: Mapping[str, float]) -> list[str]` returning one message per framework below its floor.

- [ ] **Step 1: Confirm the interpreter**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"$PY" -c "import pdfplumber, yaml, bs4, pydantic, sys; print(sys.version)"
```

Expected: `3.12.2 ...`. If this fails, stop; every later step depends on it.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_corpus_report.py — create

"""The corpus report is the only instrument in the parser plan.

Counting links resolved cannot tell 615 links unstacked onto 615 anchors from
615 links collapsed onto 40. Both make the same number rise. The tests below
pin the columns that can tell them apart: distinct anchors, links per anchor,
truncation, nesting, and controls the prose rule excludes from the index.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tract.corpus_report import build_corpus_report, check_join_floors


def _corpus(
    directory: Path, controls: list[dict[str, object]], name: str = "corpus",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(json.dumps([{
        "framework_id": "demo",
        "framework_name": "Demo",
        "controls": controls,
    }]), encoding="utf-8")
    return path


def _links(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    path = tmp_path / "links.jsonl"
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8",
    )
    return path


LONG = "A control statement long enough to clear every prose bar. " * 4


class TestAnchorCollapse:
    def test_distinct_anchors_separates_collapse_from_coverage(
        self, tmp_path: Path,
    ) -> None:
        """Two corpora resolve every link. Only one of them is good."""
        rows = [
            {"framework_id": "demo", "standard_name": "Demo",
             "section_id": f"C-{n}", "section_name": f"Control {n}"}
            for n in range(1, 5)
        ]
        spread = _corpus(tmp_path / "a", [
            {"control_id": f"C-{n}", "title": f"Control {n}",
             "description": f"{LONG} Variant {n}."}
            for n in range(1, 5)
        ])
        collapsed = _corpus(tmp_path / "b", [
            {"control_id": f"C-{n}", "title": f"Control {n}",
             "description": LONG}
            for n in range(1, 5)
        ])

        links = _links(tmp_path, rows)
        good = build_corpus_report(links, spread).per_framework[0]
        bad = build_corpus_report(links, collapsed).per_framework[0]

        assert good.links == bad.links == 4
        assert good.unresolved == bad.unresolved == 0
        assert good.distinct_anchors == 4
        assert bad.distinct_anchors == 1
        assert bad.links_per_anchor == pytest.approx(4.0)

    def test_nested_anchor_is_counted(self, tmp_path: Path) -> None:
        """A domain aggregate that opens with its own first member."""
        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "Member", "description": LONG},
            {"control_id": "D-1", "title": "Domain",
             "description": LONG + " and the rest of the domain."},
        ])
        links = _links(tmp_path, [
            {"framework_id": "demo", "standard_name": "Demo",
             "section_id": "C-1", "section_name": "Member"},
            {"framework_id": "demo", "standard_name": "Demo",
             "section_id": "D-1", "section_name": "Domain"},
        ])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.distinct_anchors == 2
        assert row.nested_anchors == 1


class TestProseRuleExclusion:
    def test_control_whose_description_restates_its_title_is_counted(
        self, tmp_path: Path,
    ) -> None:
        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "Access control",
             "description": "Access control."},
        ])
        links = _links(tmp_path, [
            {"framework_id": "demo", "standard_name": "Demo",
             "section_id": "C-1", "section_name": "Access control"},
        ])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.dropped_by_prose_rule == 1
        assert row.unresolved == 1
        assert row.distinct_anchors == 0


class TestWrongAnchorRisk:
    def test_title_hit_that_disagrees_with_the_id_is_flagged(
        self, tmp_path: Path,
    ) -> None:
        corpus = _corpus(tmp_path, [
            {"control_id": "2.3", "title": "Poisoning attacks",
             "description": LONG + " Predictive."},
            {"control_id": "3.2.2", "title": "Generative poisoning",
             "description": LONG + " Generative.",
             "metadata": {"alt_titles": ["Poisoning attacks"]}},
        ])
        links = _links(tmp_path, [
            {"framework_id": "demo", "standard_name": "Demo",
             "section_id": "3.2.2", "section_name": "Poisoning attacks"},
        ])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.by_title == 1
        assert row.wrong_anchor_risk == 1


class TestFloors:
    def test_a_framework_below_its_floor_is_reported(self, tmp_path: Path) -> None:
        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "One", "description": LONG},
        ])
        links = _links(tmp_path, [
            {"framework_id": "demo", "standard_name": "Demo",
             "section_id": "C-1", "section_name": "One"},
            {"framework_id": "demo", "standard_name": "Demo",
             "section_id": "C-2", "section_name": "Two"},
        ])
        report = build_corpus_report(links, corpus)
        assert check_join_floors(report, {"demo": 0.50}) == []
        assert len(check_join_floors(report, {"demo": 0.90})) == 1
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
pytest tests/test_corpus_report.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tract.corpus_report'`.

- [ ] **Step 4: Write the instrument**

```python
# tract/corpus_report.py — create

"""The corpus join report: the one instrument the parser plan is gated on.

A count of links resolved cannot distinguish 615 links unstacked onto 615
distinct anchors from 615 links collapsed onto 40 coarse ones. Both make the
same number rise, and the second is a regression dressed as progress. So this
module reports the anchor side as well as the link side, and reports both
through `ProseIndex.lookup` -- the same call the training and evaluation paths
make -- rather than through a set intersection that would accept a join the
consumer cannot perform.

Seven quantities per framework, each answering a failure this project has
already shipped once:

    by_title / by_id / unresolved  which channel carried the join
    distinct_anchors               the number every downstream metric rests on
    links_per_anchor               collapse, visible
    truncated                      anchors the encoder budget cuts
    nested_anchors                 an anchor that opens another anchor's text
    dropped_by_prose_rule          controls ProseIndex never indexed
    wrong_anchor_risk              title hits the id would have answered
                                   differently
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tract.config import MAX_ANCHOR_CHARS, TRAINING_DIR
from tract.text_selection import (
    ProseIndex,
    TextSelection,
    _is_prose,
    canonical_framework,
    merged_corpus_path,
    normalize_section_id,
    prepare_anchor,
)

logger = logging.getLogger(__name__)

CURATED_LINKS_PATH = TRAINING_DIR / "hub_links_curated.jsonl"


@dataclass
class FrameworkJoin:
    """One framework's join, on both the link side and the anchor side."""

    framework_id: str
    standard_name: str
    links: int = 0
    by_title: int = 0
    by_id: int = 0
    unresolved: int = 0
    distinct_anchors: int = 0
    links_per_anchor: float = 0.0
    truncated: int = 0
    nested_anchors: int = 0
    dropped_by_prose_rule: int = 0
    wrong_anchor_risk: int = 0
    resolution_rate: float = 0.0

    def finalise(self) -> None:
        self.resolution_rate = (
            0.0 if not self.links else (self.by_title + self.by_id) / self.links
        )
        self.links_per_anchor = (
            0.0 if not self.distinct_anchors
            else (self.by_title + self.by_id) / self.distinct_anchors
        )


@dataclass
class CorpusReport:
    """Every framework's join plus the identity of what produced it."""

    per_framework: list[FrameworkJoin]
    totals: FrameworkJoin
    corpus_path: str
    corpus_sha256: str
    links_path: str
    links_sha256: str
    max_anchor_chars: int = MAX_ANCHOR_CHARS

    def to_json(self) -> dict[str, Any]:
        return {
            "corpus_path": self.corpus_path,
            "corpus_sha256": self.corpus_sha256,
            "links_path": self.links_path,
            "links_sha256": self.links_sha256,
            "max_anchor_chars": self.max_anchor_chars,
            "totals": asdict(self.totals),
            "per_framework": [asdict(row) for row in self.per_framework],
        }

    def by_id(self, framework_id: str) -> FrameworkJoin:
        """One framework's row.

        Raises:
            KeyError: If the framework contributed no curated links.
        """
        for row in self.per_framework:
            if row.framework_id == framework_id:
                return row
        raise KeyError(
            f"{framework_id!r} has no curated links, so it has no join to "
            f"report. Check the framework_id spelling against "
            f"data/training/hub_links_curated.jsonl."
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_links(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            grouped.setdefault(row["framework_id"], []).append(row)
    return grouped


def _load_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    for value in data.values():
        if isinstance(value, list):
            return value
    raise ValueError(
        f"{path} holds no list of framework records. The merged corpus is "
        f"either a list or a mapping with one list value."
    )


def _dropped_by_prose_rule(records: list[dict[str, Any]]) -> dict[str, int]:
    """Controls ProseIndex refuses to index, per canonical framework name.

    ProseIndex takes full_text when present, else description when it exceeds
    the title by PROSE_MIN_EXTRA_CHARS, else nothing at all. The third branch
    is silent: the control is absent from both maps and every link to it falls
    back to a section title. Nothing counted it before this.
    """
    dropped: dict[str, int] = {}
    for record in records:
        name = canonical_framework(str(record.get("framework_name") or ""))
        for control in record.get("controls") or []:
            if str(control.get("full_text") or "").strip():
                continue
            if _is_prose(
                str(control.get("description") or ""),
                str(control.get("title") or ""),
            ):
                continue
            dropped[name] = dropped.get(name, 0) + 1
    return dropped


def _lookup_with_channel(
    index: ProseIndex, canonical: str, section_id: str | None,
    section_name: str | None,
) -> tuple[TextSelection | None, str]:
    """ProseIndex.lookup, plus which channel answered.

    Deliberately reimplements lookup's branch order rather than calling it: the
    report has to say *how* a link resolved, and lookup returns only the text.
    The order here must stay identical to lookup's -- title, then id -- and
    tests/test_corpus_report.py::TestChannelParity asserts the two agree on
    every curated link.
    """
    if section_name:
        hit = index.by_title(canonical, str(section_name))
        if hit is not None:
            return hit, "title"
    normalized = normalize_section_id(section_id)
    if normalized:
        hit = index.by_id(canonical, normalized)
        if hit is not None:
            return hit, "id"
    return None, "unresolved"


def _count_nested(anchors: set[str]) -> int:
    """Anchors that are a strict prefix of a longer anchor in the same set.

    A domain aggregate built by concatenating its member controls opens with
    its own first member, so the two anchors differ only in what follows. The
    encoder sees near-duplicates competing for the same links. Measured on the
    current corpus this reads 0; under a CSA CCM parser that concatenates
    member specifications it reads 17 of 17. [measured]
    """
    ordered = sorted(anchors, key=len)
    return sum(
        1 for position, short in enumerate(ordered)
        if any(longer.startswith(short) for longer in ordered[position + 1:])
    )


def build_corpus_report(
    links_path: Path | None = None, corpus_path: Path | None = None,
) -> CorpusReport:
    """Resolve every curated link through ProseIndex and report the join."""
    links_file = links_path or CURATED_LINKS_PATH
    corpus_file = corpus_path or merged_corpus_path()

    records = _load_records(corpus_file)
    index = ProseIndex(records)
    dropped = _dropped_by_prose_rule(records)
    grouped = _load_links(links_file)

    rows: list[FrameworkJoin] = []
    totals = FrameworkJoin(framework_id="TOTAL", standard_name="")
    all_anchors: set[str] = set()

    for framework_id in sorted(grouped):
        links = grouped[framework_id]
        standard = str(links[0].get("standard_name") or "")
        canonical = canonical_framework(standard)
        row = FrameworkJoin(
            framework_id=framework_id,
            standard_name=standard,
            links=len(links),
            dropped_by_prose_rule=dropped.get(canonical, 0),
        )
        anchors: set[str] = set()
        for link in links:
            selection, channel = _lookup_with_channel(
                index, canonical, link.get("section_id"),
                link.get("section_name"),
            )
            if channel == "title":
                row.by_title += 1
            elif channel == "id":
                row.by_id += 1
            else:
                row.unresolved += 1
                continue
            assert selection is not None  # invariant: a channel produced it
            text, was_cut = prepare_anchor(selection.text)
            anchors.add(text)
            row.truncated += int(was_cut)
            if channel == "title":
                normalized = normalize_section_id(link.get("section_id"))
                other = index.by_id(canonical, normalized) if normalized else None
                if other is not None and other.text != selection.text:
                    row.wrong_anchor_risk += 1

        row.distinct_anchors = len(anchors)
        row.nested_anchors = _count_nested(anchors)
        row.finalise()
        rows.append(row)

        all_anchors |= anchors
        totals.links += row.links
        totals.by_title += row.by_title
        totals.by_id += row.by_id
        totals.unresolved += row.unresolved
        totals.truncated += row.truncated
        totals.nested_anchors += row.nested_anchors
        totals.dropped_by_prose_rule += row.dropped_by_prose_rule
        totals.wrong_anchor_risk += row.wrong_anchor_risk

    totals.distinct_anchors = len(all_anchors)
    totals.finalise()

    return CorpusReport(
        per_framework=rows,
        totals=totals,
        corpus_path=str(corpus_file),
        corpus_sha256=_sha256(corpus_file),
        links_path=str(links_file),
        links_sha256=_sha256(links_file),
    )


def check_join_floors(
    report: CorpusReport, floors: Mapping[str, float],
) -> list[str]:
    """One message per framework whose resolution rate is under its floor.

    A floor is derived from the link file and the source before the parser is
    written, never pasted from the run being gated. See each parser task for
    the arithmetic that produced its number.
    """
    failures: list[str] = []
    for framework_id, floor in sorted(floors.items()):
        row = report.by_id(framework_id)
        if row.resolution_rate + 1e-9 < floor:
            failures.append(
                f"{framework_id}: resolved {row.by_title + row.by_id} of "
                f"{row.links} links ({row.resolution_rate:.4f}) against a "
                f"derived floor of {floor:.4f}. The floor is the arithmetic "
                f"ceiling of this framework's link data rounded down, so a "
                f"miss means the parser lost anchors the source supplies."
            )
    return failures


def format_table(report: CorpusReport) -> str:
    """The report as a fixed-width table, for logs and for the run ledger."""
    header = (
        f"{'framework':26s} {'links':>5s} {'ttl':>5s} {'id':>4s} {'unres':>5s} "
        f"{'anchors':>7s} {'l/a':>5s} {'trunc':>5s} {'nest':>4s} "
        f"{'noidx':>5s} {'wrong':>5s} {'rate':>6s}"
    )
    lines = [header, "-" * len(header)]
    for row in [*report.per_framework, report.totals]:
        lines.append(
            f"{row.framework_id:26s} {row.links:5d} {row.by_title:5d} "
            f"{row.by_id:4d} {row.unresolved:5d} {row.distinct_anchors:7d} "
            f"{row.links_per_anchor:5.2f} {row.truncated:5d} "
            f"{row.nested_anchors:4d} {row.dropped_by_prose_rule:5d} "
            f"{row.wrong_anchor_risk:5d} {row.resolution_rate:6.4f}"
        )
    return "\n".join(lines)
```

- [ ] **Step 5: Add the two accessors `_lookup_with_channel` needs**

`ProseIndex` exposes only `lookup`, and the report needs the two maps
separately. Reaching into `index._by_title` from another module would make the
report depend on a private attribute; two four-line accessors are cheaper than
that coupling.

```python
# tract/text_selection.py — add to ProseIndex, immediately above lookup()

    def by_title(self, framework: str, section_name: str) -> TextSelection | None:
        """The selection a title lookup would return, or None.

        Exposed for tract.corpus_report, which must report which channel
        answered a link and cannot get that from lookup's return value.
        """
        return self._by_title.get(
            (canonical_framework(framework), section_name.strip().lower())
        )

    def by_id(self, framework: str, section_id: str) -> TextSelection | None:
        """The selection an id lookup would return, or None."""
        return self._by_id.get(
            (canonical_framework(framework), normalize_section_id(section_id))
        )
```

- [ ] **Step 6: Add the channel-parity test**

The report reimplements `lookup`'s branch order. If the two ever disagree, the
report describes a join the pipeline does not perform, which is the exact
defect that got the previous plan rejected.

```python
# tests/test_corpus_report.py — append

class TestChannelParity:
    def test_report_and_lookup_agree_on_every_curated_link(self) -> None:
        """The report must describe the join the pipeline actually performs."""
        import json as _json

        from tract.corpus_report import CURATED_LINKS_PATH, _lookup_with_channel
        from tract.text_selection import (
            ProseIndex, canonical_framework, merged_corpus_path,
        )

        corpus = merged_corpus_path()
        if not corpus.exists() or not CURATED_LINKS_PATH.exists():
            pytest.skip("needs the merged corpus and the curated link file")

        data = _json.loads(corpus.read_text(encoding="utf-8"))
        index = ProseIndex(data if isinstance(data, list) else [])
        with CURATED_LINKS_PATH.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = _json.loads(line)
                canonical = canonical_framework(row.get("standard_name", ""))
                mine, _ = _lookup_with_channel(
                    index, canonical, row.get("section_id"),
                    row.get("section_name"),
                )
                theirs = index.lookup(
                    row.get("standard_name", ""), row.get("section_id"),
                    row.get("section_name"),
                )
                assert (mine is None) == (theirs is None)
                if mine is not None and theirs is not None:
                    assert mine.text == theirs.text
```

- [ ] **Step 7: Write the CLI**

```python
# scripts/corpus_report.py — create

"""Print or persist the corpus join report.

    PYTHONPATH=. "$PY" scripts/corpus_report.py
    PYTHONPATH=. "$PY" scripts/corpus_report.py --out results/corpus/after.json

The same entry point produces the BEFORE artifact, every per-parser acceptance
check, and the final corpus report. One instrument, one code path: a parser
accepted by a measurement its consumer does not use is a parser accepted by
nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from tract.corpus_report import build_corpus_report, format_table
from tract.io import atomic_write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--links", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = build_corpus_report(args.links, args.corpus)
    print(format_table(report))
    print()
    print(f"corpus  {report.corpus_path}  sha256 {report.corpus_sha256[:16]}")
    print(f"links   {report.links_path}  sha256 {report.links_sha256[:16]}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(report.to_json(), args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Run the tests**

```bash
pytest tests/test_corpus_report.py -q
mypy tract/corpus_report.py scripts/corpus_report.py tract/text_selection.py --strict
```

Expected: PASS, no mypy errors.

- [ ] **Step 9: Capture the BEFORE state**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" scripts/corpus_report.py --out results/corpus/before_8cf44b3.json
```

Expected, and this is the free test of the instrument — these values were
measured independently while writing this plan **[measured]**:

```
iso_27001                     94    88    4     2      91  1.01     0    0     2     1 0.9787
biml                          21     0    0    21       0  0.00     0    0    20     0 0.0000
csa_ccm                       29     0    0    29       0  0.00     0    0    29     0 0.0000
dsomm                        214     0    0   214       0  0.00     0    0   183     0 0.0000
enisa                         68     0    0    68       0  0.00     0    0    38     0 0.0000
etsi                          36     0    0    36       0  0.00     0    0    27     0 0.0000
nist_800_63                   79     0    0    79       0  0.00     0    0    25     0 0.0000
nist_ssdf                     46     0    0    46       0  0.00     0    0    44     0 0.0000
owasp_proactive_controls      76     0    0    76       0  0.00     0    0    10     0 0.0000
owasp_top10_2021              17     0    0    17       0  0.00     0    0    10     0 0.0000
samm                          30     0    0    30       0  0.00     0    0    30     0 0.0000
wstg                         118     0    0   118       0  0.00     0    0    59     0 0.0000
owasp_cheat_sheets           391   391    0     0      49  7.98   384    0     0     0 1.0000
capec                       1799  1799    0     0     349  5.15    24    0     2     0 1.0000
TOTAL                       4405  3584   82   739    1450  2.53   559    0   522     9 0.8322
```

Two properties matter and both must hold. ISO is not zero, so the instrument
can see a working join. The eleven are all zero, so it can see a broken one.
If ISO reads 0.0000, the licensed overlay is missing from this checkout and
`merged_corpus_path()` fell back to the tracked corpus; fix that before going
further, because every later comparison would be against the wrong baseline
(ledger lesson 5).

Three totals are worth naming before any parser exists. **1,450 distinct
anchors** carry 3,666 resolved links. **559 anchors are truncated** at
`MAX_ANCHOR_CHARS`, 384 of them in `owasp_cheat_sheets` alone. **522 controls
are in the corpus and absent from the prose index**, because their description
does not exceed their title by `PROSE_MIN_EXTRA_CHARS`; **475 of those 522**
belong to the eleven frameworks this plan gives a parser, and the rest are
43 CWE weaknesses, 2 CAPEC patterns and ISO's A.7.8 and A.7.9. **[measured,
all]** No artifact
in this repository reported any of the three before now.

- [ ] **Step 10: Commit**

```bash
git add tract/corpus_report.py scripts/corpus_report.py \
        tests/test_corpus_report.py tract/text_selection.py \
        results/corpus/before_8cf44b3.json
git commit -m "feat: measure the corpus join by anchor, not only by link"
```

---

### Task 2: An `alt_ids` channel on ProseIndex

`ProseIndex` reads `metadata["alt_titles"]` and nothing else. **[measured]**
There is no id-side equivalent, and three frameworks need one:

- **nist_ssdf**: two of 46 curated links carry a mid-sentence text fragment where a `PS.1.1`-style id belongs. Both are recoverable: the first fragment appears verbatim inside task `PS.1.1`'s statement, the second inside `PW.8.1`'s. **[measured]** Without `alt_ids` the ceiling is 44/46; with it, 46/46.
- **biml**: 8 of 21 curated links carry an unprefixed `category:number` id while the same id means something different in the other BIML document. Seven resolve to one document by exact tag-label match. **[measured]**
- **wstg / csa_ccm**: neither needs it. CSA CCM's seven retired `IVS-*` ids resolve through the title channel already **[measured]**, so the rename map the previous plan carried is dead weight and is not built here.

`alt_ids` follows `alt_titles`' two-pass rule exactly: a real id claims its key
in the first pass, an alternate may add a key but never displace one. Written
the other way, one control's alternate can take the slot belonging to another
control's real id, which is the defect the `alt_titles` two-pass comment
records.

**Files:**
- Modify: `tract/text_selection.py`
- Modify: `tests/test_text_selection.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `ProseIndex` honouring `metadata["alt_ids"]: list[str] | str`, normalised through `normalize_section_id` and never displacing a real `control_id`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_text_selection.py — append

class TestAlternateIds:
    """A retired or malformed id may add a key. It may never displace one."""

    LONG = "A control statement long enough to clear the prose bar easily. " * 3

    def _index(self) -> ProseIndex:
        return ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "PS.1.1", "title": "Protect code",
                 "description": self.LONG + " Real PS.1.1.",
                 "metadata": {"alt_ids": ["code, executable code"]}},
                {"control_id": "PW.8.1", "title": "Test executable",
                 "description": self.LONG + " Real PW.8.1."},
            ],
        }])

    def test_alternate_id_resolves(self) -> None:
        hit = self._index().lookup("Demo", "code, executable code", None)
        assert hit is not None
        assert hit.text.endswith("Real PS.1.1.")

    def test_alternate_never_displaces_a_real_id(self) -> None:
        index = ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "A-1", "title": "First",
                 "description": self.LONG + " First.",
                 "metadata": {"alt_ids": ["A-2"]}},
                {"control_id": "A-2", "title": "Second",
                 "description": self.LONG + " Second."},
            ],
        }])
        hit = index.lookup("Demo", "A-2", None)
        assert hit is not None
        assert hit.text.endswith("Second.")

    def test_alternate_id_is_normalised_like_a_real_one(self) -> None:
        index = ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "X-1", "title": "Only",
                 "description": self.LONG + " Only.",
                 "metadata": {"alt_ids": ["2.2"]}},
            ],
        }])
        assert index.lookup("Demo", "Sec. 2.2", None) is not None
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_text_selection.py::TestAlternateIds -q
```

Expected: FAIL, `assert hit is not None` — `lookup` returns `None`.

- [ ] **Step 3: Implement**

```python
# tract/text_selection.py — in ProseIndex.__init__, replace the pending list
        pending_alternates: list[tuple[tuple[str, str], TextSelection]] = []
        pending_alternate_ids: list[tuple[tuple[str, str], TextSelection]] = []
```

```python
# tract/text_selection.py — in ProseIndex.__init__, after the control_id block
                control_id = normalize_section_id(control.get("control_id"))
                if control_id:
                    self._by_id[(framework, control_id)] = selection

                # Retired and malformed ids, held back for the same reason
                # alt_titles are: an alternate must never take the key of a
                # control whose real id spells it. NIST SSDF has two curated
                # links whose section_id is a mid-sentence fragment of the
                # task text, and BIML has eight whose id is document-scoped
                # upstream but unprefixed in OpenCRE.
                alternate_ids = metadata.get("alt_ids") or []
                if isinstance(alternate_ids, str):
                    alternate_ids = [alternate_ids]
                for raw_id in alternate_ids:
                    alt_id = normalize_section_id(str(raw_id))
                    if alt_id:
                        pending_alternate_ids.append(
                            ((framework, alt_id), selection)
                        )
```

`metadata` is currently bound a few lines below, where `alt_titles` is read.
Move that `metadata = control.get("metadata") or {}` assignment **above** the
`control_id` block so both alternate kinds read one binding, and leave the
`alt_titles` read using it unchanged.

```python
# tract/text_selection.py — after the existing alternates second pass
        for key_pair, selection in pending_alternate_ids:
            if key_pair not in self._by_id:
                self._by_id[key_pair] = selection
```

- [ ] **Step 4: Update the load() log line**

```python
# tract/text_selection.py — in ProseIndex.load, replace the logger.info call
        logger.info(
            "Prose index from %s: %d controls by id (real and alternate), "
            "%d by title", source.name, len(index._by_id), len(index._by_title),
        )
```

- [ ] **Step 5: Run tests and typecheck**

```bash
pytest tests/test_text_selection.py tests/test_corpus_report.py -q
mypy tract/text_selection.py --strict
```

Expected: PASS. `tests/test_prose_reachability.py` must also still pass, since
it measures the same join.

- [ ] **Step 6: Confirm the BEFORE state is unchanged**

```bash
PYTHONPATH=. "$PY" scripts/corpus_report.py --out /tmp/after_alt_ids.json
"$PY" -c "
import json
a=json.load(open('results/corpus/before_8cf44b3.json'))
b=json.load(open('/tmp/after_alt_ids.json'))
print('identical:', a['per_framework']==b['per_framework'])
"
```

Expected: `identical: True`. No control in the corpus carries `alt_ids` yet, so
adding the channel must move nothing. A difference here means the second pass
displaced a real id.

- [ ] **Step 7: Commit**

```bash
git add tract/text_selection.py tests/test_text_selection.py
git commit -m "feat: resolve a link through a control's retired identifiers"
```

---
### Task 3: DSOMM — 214 links off 18 title anchors onto 182 control anchors

The largest anchor win in the plan and the one that best demonstrates why the
instrument counts anchors. Today DSOMM's 214 curated links resolve to nothing
and fall back to their `section_name`, which is the **sub-dimension** name, not
the activity name: 214 links land on **18** distinct anchors, 11.89 links each.
**[measured]** The `section_id` is the activity's own uuid and matches the
source verbatim for 214 of 214 rows over 183 distinct uuids. **[measured]**

**The source-structures document is wrong about the prose field.** It says
`description` carries the activity's prose. Measured on the pinned archive:
`description` is present on 53 of 194 activities and non-empty on **51**.
`risk` and `measure` are present and non-empty on **194 of 194**. **[measured]**
A parser that took `description` alone would emit 143 empty descriptions,
`Control(description=...)` would raise on `min_length=1`, and the ones that
survived would drop out of `ProseIndex` on the prose rule.

**Ceiling, derived before the parser is written.** 214 curated links, all 214
uuids present in the model. With `description + risk + measure` as the
statement, 193 of 194 activities clear `_is_prose` against their own name; the
single failure is *Correlate known vulnerabilities in infrastructure with new
image versions*, an 11-character body under a 73-character title. **[measured]**
One curated link targets it. Ceiling **213/214 = 0.99533** **[derived]**, floor
**0.99**. The previous plan set 1.00 against this same maximum and would have
failed on landing.

Description lengths run 11 / 282 / 4,565 (min / median / max); 4 exceed
`DESCRIPTION_MAX_LENGTH`, so 4 controls acquire a `full_text` from
`_sanitize_control` and anchor on it. **[measured]**

**Files:**
- Create: `parsers/parse_dsomm.py`
- Create: `tests/test_parse_dsomm.py`

**Interfaces:**
- Consumes: `BaseParser`, `Control`; `tract.corpus_report.check_join_floors` from Task 1.
- Produces: `DsommParser` with `framework_id = "dsomm"`, `framework_name = "DSOMM"`; `DsommParser.activities_to_controls(model: dict[str, Any]) -> list[Control]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_dsomm.py — create

"""DSOMM's prose lives in risk and measure, not in description.

Measured on the pinned archive: description is non-empty on 51 of 194
activities, risk and measure on 194 of 194. A parser reading description alone
emits 143 empty statements and ProseIndex indexes none of them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from parsers.parse_dsomm import DsommParser
from tract.parsers.base import BaseParser

MODEL: dict[str, Any] = {
    "Build and Deployment": {
        "Deployment": {
            "Inventory of production components": {
                "uuid": "2a44b708-734f-4463-b0cb-86dc46344b2f",
                "risk": "Without an inventory of deployed artifacts it is not "
                        "possible to know where a vulnerable image runs.",
                "measure": "A documented inventory of artifacts in production "
                           "is maintained and kept current.",
                "level": 1,
            },
            "Pinning of artifacts": {
                "uuid": "f3c4971e-9f4d-4e59-8ed0-f0bdb6262477",
                "description": "Pin base images and dependencies to an "
                               "immutable digest rather than a moving tag.",
                "risk": "Unauthorized manipulation of artifacts is hard to "
                        "spot when tags move under the build.",
                "measure": "Pinning ensures changes happen only when intended.",
                "level": 2,
            },
        },
    },
}


@pytest.fixture()
def parser(tmp_path: Path) -> DsommParser:
    import io
    import zipfile

    raw = tmp_path / "raw"
    raw.mkdir()
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(
            "repo-abc123/generated/model.yaml",
            "---\nmeta:\n  version: test\n---\n" + yaml.safe_dump(MODEL),
        )
    (raw / "dsomm_data.zip").write_bytes(payload.getvalue())

    instance = DsommParser(raw_dir=raw, output_dir=tmp_path / "out")
    instance.expected_count = 2
    instance.expected_sha256 = None
    return instance


class TestParse:
    def test_control_id_is_the_uuid_opencre_links_against(
        self, parser: DsommParser,
    ) -> None:
        controls = parser.parse()
        assert [c.control_id for c in controls] == [
            "2a44b708-734f-4463-b0cb-86dc46344b2f",
            "f3c4971e-9f4d-4e59-8ed0-f0bdb6262477",
        ]

    def test_title_is_the_activity_name_not_the_sub_dimension(
        self, parser: DsommParser,
    ) -> None:
        titles = [c.title for c in parser.parse()]
        assert titles == ["Inventory of production components",
                          "Pinning of artifacts"]
        assert "Deployment" not in titles

    def test_statement_survives_an_absent_description(
        self, parser: DsommParser,
    ) -> None:
        first = parser.parse()[0]
        assert "inventory of deployed artifacts" in first.description
        assert "documented inventory" in first.description
        assert len(first.description) >= 60

    def test_description_leads_when_present(self, parser: DsommParser) -> None:
        second = parser.parse()[1]
        assert second.description.startswith("Pin base images")

    def test_sub_dimension_is_recorded_as_the_parent(
        self, parser: DsommParser,
    ) -> None:
        first = parser.parse()[0]
        assert first.parent_id == "Deployment"
        assert first.parent_name == "Build and Deployment"


class TestRun:
    def test_run_writes_and_clears_the_prose_floor(
        self, parser: DsommParser, tmp_path: Path,
    ) -> None:
        (tmp_path / "out").mkdir()
        output = parser.run()
        assert len(output.controls) == 2
        assert (tmp_path / "out" / "dsomm.json").exists()
        assert BaseParser.honest_prose_fraction(output.controls) == 1.0
        assert [s.path for s in output.source_files] == ["dsomm_data.zip"]

    def test_reparse_is_byte_identical(
        self, parser: DsommParser, tmp_path: Path,
    ) -> None:
        (tmp_path / "out").mkdir()
        parser.run()
        first = (tmp_path / "out" / "dsomm.json").read_bytes()
        parser.run()
        assert (tmp_path / "out" / "dsomm.json").read_bytes() == first


class TestDigestGate:
    def test_a_different_archive_is_refused(self, parser: DsommParser) -> None:
        parser.expected_sha256 = "0" * 64
        with pytest.raises(ValueError, match="not the pinned"):
            parser.parse()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_parse_dsomm.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'parsers.parse_dsomm'`.

- [ ] **Step 3: Write the parser**

```python
# parsers/parse_dsomm.py — create

"""Parser for the OWASP DevSecOps Maturity Model.

OpenCRE keys every DSOMM link on the activity's `uuid`, and 214 of 214 curated
links carry a uuid this file defines. What OpenCRE puts in `section_name` is
the SUB-DIMENSION, one level above the activity, so 214 links share 18 names.
Falling back to that name is what the corpus does today and it is why DSOMM
reads 11.89 links per anchor. Joining on the uuid takes it to 182 anchors.

The statement is `description`, `risk` and `measure` concatenated in that
order, and the order is not cosmetic. `description` is present on 53 of the 194
activities and non-empty on 51; `risk` and `measure` are non-empty on all 194.
A parser reading `description` alone would emit 143 empty statements, which
`Control` rejects, and the survivors would fail the prose rule that decides
whether ProseIndex indexes a control at all.

`level`, `usefulness`, `isImplemented` and `evidence` are deliberately unused:
they are assessment state, not control text.
"""
from __future__ import annotations

import hashlib
import logging
import zipfile
from io import BytesIO
from typing import Any, ClassVar, Final

import yaml

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARCHIVE_NAME: Final[str] = "dsomm_data.zip"
# The archive root carries the pinned commit sha, so the member is located by
# suffix rather than by a path that changes on every re-pin.
MODEL_SUFFIX: Final[str] = "generated/model.yaml"

SOURCE_SHA256: Final[str] = (
    "a6d773129591d59e7c0757651142c39a341400333f40c1555fb2481ae89f2c66"
)

# Statement fields, in the order they are joined. See the module docstring for
# why `description` cannot be the only one.
STATEMENT_FIELDS: Final[tuple[str, ...]] = ("description", "risk", "measure")


class DsommParser(BaseParser):
    framework_id: ClassVar[str] = "dsomm"
    # Matches canonical_framework("DevSecOps Maturity Model (DSOMM)"), which
    # FRAMEWORK_NAME_ALIASES already maps to "DSOMM". No new alias is needed.
    framework_name: ClassVar[str] = "DSOMM"
    version: ClassVar[str] = "4.3.1"
    source_url: ClassVar[str] = (
        "https://github.com/devsecopsmaturitymodel/DevSecOps-MaturityModel-data"
    )
    mapping_unit_level: ClassVar[str] = "activity"
    # 194 leaf activities in the pinned archive. [measured]
    expected_count: ClassVar[int] = 194
    fetched_date: ClassVar[str] = "2026-08-15"
    # 192 of 194 clear the 60-character honest-prose bar and differ from their
    # title. The two that do not are single-sentence measures with long names.
    # [measured] The floor sits just under, close enough that a regression
    # toward name-only extraction still trips it.
    min_prose_fraction: ClassVar[float] = 0.98
    # Class-level so a fixture-backed test declares its own digest instead of
    # the real gate being widened to accept two archives.
    expected_sha256: ClassVar[str | None] = SOURCE_SHA256

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(ARCHIVE_NAME)
        self._check_digest(payload)
        model = self._read_model(payload)
        controls = self.activities_to_controls(model)
        logger.info(
            "%s: %d activities across %d dimensions",
            self.framework_id, len(controls),
            len({c.parent_name for c in controls}),
        )
        return controls

    def _check_digest(self, payload: bytes) -> None:
        """Refuse an archive that is not the pinned one.

        Raises:
            ValueError: If the digest does not match `expected_sha256`.
        """
        if self.expected_sha256 is None:
            return
        actual = hashlib.sha256(payload).hexdigest()
        if actual == self.expected_sha256:
            return
        raise ValueError(
            f"{self.framework_id}: {ARCHIVE_NAME} has sha256 {actual}, not "
            f"the pinned {self.expected_sha256}. expected_count, "
            f"min_prose_fraction and the join floor were all measured against "
            f"the pinned bytes. Re-measure before moving the pin, and move it "
            f"in scripts/fetch_frameworks.py at the same time."
        )

    def _read_model(self, payload: bytes) -> dict[str, Any]:
        """The second YAML document of the generated model file.

        Raises:
            ValueError: If the member is absent or the stream is not two
                documents with a mapping second.
        """
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            names = [n for n in archive.namelist() if n.endswith(MODEL_SUFFIX)]
            if len(names) != 1:
                raise ValueError(
                    f"{self.framework_id}: expected exactly one "
                    f"{MODEL_SUFFIX} in {ARCHIVE_NAME}, found {names}. The "
                    f"generated file is what flattens 26 per-subdimension "
                    f"YAMLs into one document; without it the join level is "
                    f"guesswork."
                )
            raw = archive.read(names[0]).decode("utf-8")

        documents = list(yaml.safe_load_all(raw))
        if len(documents) != 2 or not isinstance(documents[1], dict):
            raise ValueError(
                f"{self.framework_id}: {MODEL_SUFFIX} is not a meta document "
                f"followed by a model mapping (got {len(documents)} "
                f"document(s)). The layout changed."
            )
        return documents[1]

    @classmethod
    def activities_to_controls(cls, model: dict[str, Any]) -> list[Control]:
        """One Control per leaf activity, in source order.

        Raises:
            ValueError: On an activity with no uuid, or with no statement text
                in any of STATEMENT_FIELDS.
        """
        controls: list[Control] = []
        for dimension, sub_dimensions in model.items():
            for sub_dimension, activities in sub_dimensions.items():
                for name, body in activities.items():
                    controls.append(
                        cls._to_control(dimension, sub_dimension, name, body)
                    )
        return controls

    @classmethod
    def _to_control(
        cls, dimension: str, sub_dimension: str, name: str, body: dict[str, Any],
    ) -> Control:
        uuid = str(body.get("uuid") or "").strip()
        if not uuid:
            raise ValueError(
                f"dsomm: activity {name!r} under {dimension}/{sub_dimension} "
                f"has no uuid. The uuid is what OpenCRE links against, so an "
                f"activity without one cannot be joined and must not be "
                f"emitted as though it could."
            )
        statement = cls._statement(body)
        if not statement:
            raise ValueError(
                f"dsomm: activity {name!r} (uuid {uuid}) has no text in any "
                f"of {STATEMENT_FIELDS}. All 194 activities in the pinned "
                f"archive have risk and measure, so an empty one means the "
                f"schema changed."
            )
        return Control(
            control_id=uuid,
            title=name.strip(),
            description=statement,
            hierarchy_level="activity",
            parent_id=sub_dimension,
            parent_name=dimension,
            metadata={"sub_dimension": sub_dimension, "dimension": dimension},
        )

    @staticmethod
    def _statement(body: dict[str, Any]) -> str:
        parts = [
            str(body.get(field) or "").strip() for field in STATEMENT_FIELDS
        ]
        return "\n\n".join(part for part in parts if part)


def main() -> None:
    DsommParser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and typecheck**

```bash
pytest tests/test_parse_dsomm.py -q
mypy parsers/parse_dsomm.py --strict
```

Expected: PASS, no mypy errors.

- [ ] **Step 5: Run the parser against the real source**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" parsers/parse_dsomm.py
```

Expected: `dsomm: 194 activities across 5 dimensions`, then
`dsomm: honest prose fraction 0.990 (floor 0.980)` and
`Wrote 194 controls to data/processed/frameworks/dsomm.json`. **[measured]**

- [ ] **Step 6: Rebuild the merged corpus and run the instrument**

```bash
PYTHONPATH=. "$PY" parsers/merge_all_controls.py
PYTHONPATH=. "$PY" scripts/corpus_report.py | grep -E "^(dsomm|framework|TOTAL)"
```

Expected row, derived before the parser was written **[derived from measured]**:

```
dsomm                        214     0  213     1     182  1.17     1    0     1     0 0.9953
```

Accept only if all five hold:
- `resolution_rate >= 0.99` (the derived floor)
- `distinct_anchors >= 182` — up from 0 resolved and from 18 title anchors
- `links_per_anchor <= 1.20` — down from 11.89 on the fallback anchors
- `nested_anchors == 0`
- `wrong_anchor_risk == 0`

If `by_title` is not 0, the parser is emitting activity names that collide with
a sub-dimension name and the join is going through the wrong channel.

- [ ] **Step 7: Commit**

```bash
git add parsers/parse_dsomm.py tests/test_parse_dsomm.py \
        data/processed/frameworks/dsomm.json data/processed/all_controls.json
git commit -m "feat: join DSOMM on the activity uuid instead of its sub-dimension"
```

---

### Task 4: SAMM — 30 streams, and the `full_text` trap

SAMM is the cleanest source in the batch and it is placed here because it makes
one contract fact concrete before a harder parser has to reason about it.

`model/streams/<P>-<S>.yml` is the join level: 30 files whose stem matches
`section_id` for 30 of 30 curated links, and whose `name` matches `section_name`
for 30 of 30. **[measured]** Ceiling **30/30 = 1.0000** **[derived]**, floor
**1.00**. At the ceiling a floor of 1.00 is the right number: anything less
means a link that resolves today stopped resolving.

**The composition decision is measured, not assumed.** Four candidate
statements, over the 30 streams **[measured]**:

| statement | min | median | max | over `MAX_ANCHOR_CHARS` (2150) | over `DESCRIPTION_MAX_LENGTH` |
|---|---|---|---|---|---|
| stream `description` only | 110 | 185 | 577 | 0 | 0 |
| description + 3 × `shortDescription` | 347 | 465 | 986 | **0** | **0** |
| description + level-1 `longDescription` | 858 | 1283 | 2529 | 2 | 3 |
| description + 3 × `longDescription` | 2548 | 3558 | 6678 | **30** | **30** |

The last row is the trap. It reads as the richest option and it is the worst:
every one of the 30 descriptions exceeds `DESCRIPTION_MAX_LENGTH`, so
`_sanitize_control` writes the overflow into `full_text`, `ProseIndex` prefers
`full_text`, and all 30 anchors are then cut at 2,150 characters — a 100%
truncation rate on a framework that has none today. The second row is chosen:
real prose from all three maturity levels, no truncation, no implicit
`full_text`.

**The source-structures document is wrong about `level`.** It is a GUID, not an
integer. **[measured]** Maturity level comes from the activity **filename**
(`D-SA-1-A.yml` → level 1); every one of the 30 streams has exactly 3
activities and the filename stems partition cleanly. **[measured]**

**Files:**
- Create: `parsers/parse_samm.py`
- Create: `tests/test_parse_samm.py`

**Interfaces:**
- Consumes: `BaseParser`, `Control`.
- Produces: `SammParser` with `framework_id = "samm"`, `framework_name = "SAMM"`; `SammParser.build_controls(streams: dict[str, dict[str, Any]], activities: dict[str, list[tuple[int, dict[str, Any]]]]) -> list[Control]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_samm.py — create

"""SAMM joins at the stream, and the statement must not overflow the budget.

Measured: stream description plus the three activities' longDescription runs
2,548 to 6,678 characters, so all 30 descriptions exceed DESCRIPTION_MAX_LENGTH,
_sanitize_control moves the overflow into full_text, and ProseIndex then
anchors on a text prepare_anchor cuts at 2,150. shortDescription keeps every
statement inside both budgets.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
import yaml

from parsers.parse_samm import SammParser
from tract.config import DESCRIPTION_MAX_LENGTH, MAX_ANCHOR_CHARS

STREAM = {
    "practice": "4753e55e943c4d418303bf90d599c6b1",
    "id": "253b012094cf4e0988e08fd22609227d",
    "name": "Architecture Design",
    "letter": "A",
    "description": "The design of a software architecture significantly "
                   "affects the security posture of the software.",
    "order": 1,
    "type": "Stream",
}


def _activity(level: int) -> dict[str, object]:
    return {
        "stream": "253b012094cf4e0988e08fd22609227d",
        "level": "a11b78917dec4cfdad983cf6d1d17b61",
        "title": f"Level {level} activity",
        "shortDescription": f"Teams apply level {level} design practices "
                            f"during architecture review sessions.",
        "longDescription": "x" * 3000,
        "type": "Activity",
    }


@pytest.fixture()
def parser(tmp_path: Path) -> SammParser:
    raw = tmp_path / "raw"
    raw.mkdir()
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("core-abc/model/streams/D-SA-A.yml",
                         yaml.safe_dump(STREAM))
        for level in (1, 2, 3):
            archive.writestr(f"core-abc/model/activities/D-SA-{level}-A.yml",
                             yaml.safe_dump(_activity(level)))
    (raw / "samm_core.zip").write_bytes(payload.getvalue())

    instance = SammParser(raw_dir=raw, output_dir=tmp_path / "out")
    instance.expected_count = 1
    instance.expected_sha256 = None
    return instance


class TestParse:
    def test_control_id_is_the_filename_stem(self, parser: SammParser) -> None:
        assert [c.control_id for c in parser.parse()] == ["D-SA-A"]

    def test_title_is_the_stream_name(self, parser: SammParser) -> None:
        assert parser.parse()[0].title == "Architecture Design"

    def test_statement_carries_all_three_maturity_levels(
        self, parser: SammParser,
    ) -> None:
        text = parser.parse()[0].description
        for level in (1, 2, 3):
            assert f"level {level} design practices" in text

    def test_statement_uses_short_not_long_descriptions(
        self, parser: SammParser,
    ) -> None:
        control = parser.parse()[0]
        assert "x" * 100 not in control.description
        assert len(control.description) <= DESCRIPTION_MAX_LENGTH
        assert len(control.description) <= MAX_ANCHOR_CHARS

    def test_full_text_is_left_unset_so_the_anchor_is_the_statement(
        self, parser: SammParser,
    ) -> None:
        assert parser.parse()[0].full_text is None


class TestMissingActivities:
    def test_a_stream_with_no_activities_is_refused(
        self, parser: SammParser, tmp_path: Path,
    ) -> None:
        raw = tmp_path / "bare"
        raw.mkdir()
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("core-abc/model/streams/D-SA-A.yml",
                             yaml.safe_dump(STREAM))
        (raw / "samm_core.zip").write_bytes(payload.getvalue())
        bare = SammParser(raw_dir=raw, output_dir=tmp_path / "out")
        bare.expected_sha256 = None
        with pytest.raises(ValueError, match="no activities"):
            bare.parse()


class TestRun:
    def test_run_writes(self, parser: SammParser, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        output = parser.run()
        assert len(output.controls) == 1
        assert [s.path for s in output.source_files] == ["samm_core.zip"]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_parse_samm.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'parsers.parse_samm'`.

- [ ] **Step 3: Write the parser**

```python
# parsers/parse_samm.py — create

"""Parser for OWASP SAMM, at the stream level.

The repository has three granularities and only one of them is what OpenCRE
links against. `model/security_practices/` holds the 15 practices,
`model/streams/` holds the 30 practice-and-stream pairs, and
`model/activities/` holds the 90 (practice, stream, level) triples. Every one
of the 30 curated links carries a `section_id` equal to a stream filename stem
and a `section_name` equal to that stream's `name`. Activity filenames
(`D-SA-1-A`) match no section_id at all.

The statement is the stream's own `description` plus the three activities'
`shortDescription`, in maturity-level order. Measured over the 30 streams that
lands between 347 and 986 characters. Using `longDescription` instead runs
2,548 to 6,678, which puts every description over DESCRIPTION_MAX_LENGTH, makes
BaseParser._sanitize_control write the overflow into full_text, and hands
ProseIndex -- which prefers full_text -- an anchor that prepare_anchor then
truncates. Richer text that the encoder never reads is not richer text.

Maturity level comes from the activity filename. The `level` field is a GUID
into SAMM's own model, not an ordinal, and sorting on it produces an arbitrary
order that changes with the release.
"""
from __future__ import annotations

import hashlib
import logging
import re
import zipfile
from collections import defaultdict
from io import BytesIO
from typing import Any, ClassVar, Final

import yaml

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARCHIVE_NAME: Final[str] = "samm_core.zip"
SOURCE_SHA256: Final[str] = (
    "16eb608b70bad3039b14ca4e3f300893d29bbc4205c737ac07fcbdfb4f7493a6"
)

STREAM_MEMBER: Final[re.Pattern[str]] = re.compile(
    r"/model/streams/([A-Z]-[A-Z]{2}-[AB])\.yml$"
)
# D-SA-1-A.yml: practice code, maturity level, stream letter. The level is read
# here because the `level` FIELD is a GUID.
ACTIVITY_MEMBER: Final[re.Pattern[str]] = re.compile(
    r"/model/activities/([A-Z]-[A-Z]{2})-(\d)-([AB])\.yml$"
)

EXPECTED_LEVELS: Final[tuple[int, ...]] = (1, 2, 3)


class SammParser(BaseParser):
    framework_id: ClassVar[str] = "samm"
    framework_name: ClassVar[str] = "SAMM"
    version: ClassVar[str] = "2.0"
    source_url: ClassVar[str] = "https://owaspsamm.org/model/"
    mapping_unit_level: ClassVar[str] = "stream"
    # 15 practices x 2 streams. [measured]
    expected_count: ClassVar[int] = 30
    fetched_date: ClassVar[str] = "2026-08-15"
    # Every statement is at least 347 characters and none equals its name.
    # [measured]
    min_prose_fraction: ClassVar[float] = 1.0
    expected_sha256: ClassVar[str | None] = SOURCE_SHA256

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(ARCHIVE_NAME)
        self._check_digest(payload)
        streams, activities = self._read_members(payload)
        controls = self.build_controls(streams, activities)
        logger.info(
            "%s: %d streams, statement length %d..%d characters",
            self.framework_id, len(controls),
            min(len(c.description) for c in controls),
            max(len(c.description) for c in controls),
        )
        return controls

    def _check_digest(self, payload: bytes) -> None:
        """Refuse an archive that is not the pinned one.

        Raises:
            ValueError: If the digest does not match `expected_sha256`.
        """
        if self.expected_sha256 is None:
            return
        actual = hashlib.sha256(payload).hexdigest()
        if actual != self.expected_sha256:
            raise ValueError(
                f"{self.framework_id}: {ARCHIVE_NAME} has sha256 {actual}, "
                f"not the pinned {self.expected_sha256}. The statement-length "
                f"measurements that chose shortDescription over "
                f"longDescription were taken against the pinned bytes."
            )

    def _read_members(
        self, payload: bytes,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, list[tuple[int, dict[str, Any]]]]]:
        """Streams keyed by stem, activities keyed by stem and level."""
        streams: dict[str, dict[str, Any]] = {}
        activities: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            for name in sorted(archive.namelist()):
                stream = STREAM_MEMBER.search(name)
                if stream:
                    streams[stream.group(1)] = yaml.safe_load(archive.read(name))
                    continue
                activity = ACTIVITY_MEMBER.search(name)
                if activity:
                    key = f"{activity.group(1)}-{activity.group(3)}"
                    activities[key].append(
                        (int(activity.group(2)), yaml.safe_load(archive.read(name)))
                    )
        if not streams:
            raise ValueError(
                f"{self.framework_id}: no model/streams/*.yml members in "
                f"{ARCHIVE_NAME}. The stream stem is the only identifier "
                f"OpenCRE links against; without it there is no join."
            )
        return streams, dict(activities)

    @classmethod
    def build_controls(
        cls,
        streams: dict[str, dict[str, Any]],
        activities: dict[str, list[tuple[int, dict[str, Any]]]],
    ) -> list[Control]:
        """One Control per stream, statement built from its three activities.

        Raises:
            ValueError: If a stream has no activities, or its levels are not
                exactly EXPECTED_LEVELS.
        """
        controls: list[Control] = []
        for stem in sorted(streams):
            stream = streams[stem]
            owned = sorted(activities.get(stem, []))
            if not owned:
                raise ValueError(
                    f"samm: stream {stem} has no activities. The statement is "
                    f"built from them, so an empty list would emit a control "
                    f"carrying only the two-sentence stream description."
                )
            levels = tuple(level for level, _ in owned)
            if levels != EXPECTED_LEVELS:
                raise ValueError(
                    f"samm: stream {stem} has maturity levels {levels}, "
                    f"expected {EXPECTED_LEVELS}. A missing level means the "
                    f"statement is short by a third and nothing else would "
                    f"say so."
                )
            controls.append(Control(
                control_id=stem,
                title=str(stream.get("name") or "").strip(),
                description=cls._statement(stream, owned),
                hierarchy_level="stream",
                parent_id=str(stream.get("practice") or "").strip() or None,
                metadata={"stream_letter": str(stream.get("letter") or "")},
            ))
        return controls

    @staticmethod
    def _statement(
        stream: dict[str, Any], owned: list[tuple[int, dict[str, Any]]],
    ) -> str:
        parts = [str(stream.get("description") or "").strip()]
        parts += [
            str(activity.get("shortDescription") or "").strip()
            for _, activity in owned
        ]
        return "\n\n".join(part for part in parts if part)


def main() -> None:
    SammParser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and typecheck**

```bash
pytest tests/test_parse_samm.py -q
mypy parsers/parse_samm.py --strict
```

- [ ] **Step 5: Run against the real source and check the join**

```bash
PYTHONPATH=. "$PY" parsers/parse_samm.py
PYTHONPATH=. "$PY" parsers/merge_all_controls.py
PYTHONPATH=. "$PY" scripts/corpus_report.py | grep -E "^(samm|framework)"
```

Expected: `samm: 30 streams, statement length 347..986 characters` **[measured]**,
then:

```
samm                          30    30    0     0      30  1.00     0    0     0     0 1.0000
```

Accept only if `resolution_rate == 1.0000`, `distinct_anchors == 30`,
`truncated == 0`, `nested_anchors == 0`. `by_title` of 30 is expected here and
is correct: every `section_name` is the stream's own name and no two streams
share one.

- [ ] **Step 6: Commit**

```bash
git add parsers/parse_samm.py tests/test_parse_samm.py \
        data/processed/frameworks/samm.json data/processed/all_controls.json
git commit -m "feat: parse SAMM at the stream, with a statement that fits the encoder"
```

---

### Task 5: OWASP Top 10 2021 — 10 categories

17 curated links over 10 `section_id` values `A01`..`A10`, all present.
**[measured]** Ceiling **17/17 = 1.0000** **[derived]**, floor **1.00**.

**The source-structures document is wrong about the file count.** It says three
`A00_2021-*.md` files are meta. There is exactly one `A00` and one `A11`:
twelve files match `A\d\d_2021-*.md` in `2021/docs/en/`, of which `A01` through
`A10` are the categories, `A00` is *How to start an AppSec Program* and `A11` is
*Next Steps*. **[measured]** `A00`'s H1 carries no `A0N:2021` prefix at all, so
a parser keyed on the H1 pattern excludes it automatically; `A11`'s does, and
must be excluded by an explicit id allowlist.

`section_name` diverges cosmetically from the file's own title for two
categories: OpenCRE says `Broken Access Controls` (plural) and
`Logging and Monitoring Failures` where the source says `Broken Access Control`
and `Security Logging and Monitoring Failures`. **[measured]** Both resolve
through the id channel, so no alias is needed and none is added.

The archive is 196 MB and 199 of its members are markdown for other years and
languages. **[measured]** Only the twelve `2021/docs/en/A*` members are read.

**Files:**
- Create: `parsers/parse_owasp_top10_2021.py`
- Create: `tests/test_parse_owasp_top10_2021.py`

**Interfaces:**
- Consumes: `BaseParser`, `Control`, `tract.config.REMEDIATION_HEADINGS`.
- Produces: `OwaspTop102021Parser` with `framework_id = "owasp_top10_2021"`, `framework_name = "OWASP Top 10 2021"`; `OwaspTop102021Parser.control_from_markdown(text: str) -> Control`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_owasp_top10_2021.py — create

"""Ten categories, and neither A00 nor A11 is one of them."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from parsers.parse_owasp_top10_2021 import OwaspTop102021Parser

CATEGORY = """# A01:2021 – Broken Access Control

## Factors

| CWEs Mapped | Max Incidence Rate |
|---|---|
| 34 | 55.97% |

## Overview

Moving up from the fifth position, 94% of applications were tested for some
form of broken access control.

## Description

Access control enforces policy such that users cannot act outside of their
intended permissions. Failures typically lead to unauthorized information
disclosure, modification, or destruction of all data.

## How to Prevent

Access control is only effective in trusted server-side code.

## References

- OWASP Proactive Controls
"""

META = """# How to start an AppSec Program with the OWASP Top 10

Previously, the OWASP Top 10 was never designed to be the basis of anything.
"""

NEXT_STEPS = """# A11:2021 – Next Steps

The Top 10 is not the end of the journey.

## Description

There is more to application security than ten risks.
"""


@pytest.fixture()
def parser(tmp_path: Path) -> OwaspTop102021Parser:
    raw = tmp_path / "raw"
    raw.mkdir()
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("Top10-abc/2021/docs/en/A01_2021-Broken_Access_Control.md",
                         CATEGORY)
        archive.writestr("Top10-abc/2021/docs/en/A00_2021-How_to_start.md", META)
        archive.writestr("Top10-abc/2021/docs/en/A11_2021-Next_Steps.md",
                         NEXT_STEPS)
        archive.writestr("Top10-abc/2017/docs/en/A01_2017-Injection.md", CATEGORY)
        archive.writestr("Top10-abc/2021/docs/fr/A01_2021-Broken.md", CATEGORY)
    (raw / "owasp_top10_2021.zip").write_bytes(payload.getvalue())

    instance = OwaspTop102021Parser(raw_dir=raw, output_dir=tmp_path / "out")
    instance.expected_count = 1
    instance.expected_sha256 = None
    return instance


class TestParse:
    def test_only_the_ten_english_2021_categories_are_read(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        assert [c.control_id for c in parser.parse()] == ["A01"]

    def test_title_drops_the_code_and_the_en_dash(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        assert parser.parse()[0].title == "Broken Access Control"

    def test_description_is_the_description_section(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        text = parser.parse()[0].description
        assert text.startswith("Access control enforces policy")
        assert "Moving up from the fifth position" not in text
        assert "trusted server-side code" not in text
        assert "CWEs Mapped" not in text

    def test_full_text_carries_the_whole_entry(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        control = parser.parse()[0]
        assert control.full_text is not None
        assert "trusted server-side code" in control.full_text


class TestGuards:
    def test_a_missing_description_section_is_refused(
        self, parser: OwaspTop102021Parser, tmp_path: Path,
    ) -> None:
        raw = tmp_path / "broken"
        raw.mkdir()
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("Top10-abc/2021/docs/en/A01_2021-X.md",
                             "# A01:2021 – X\n\n## Overview\n\nNo body.\n")
        (raw / "owasp_top10_2021.zip").write_bytes(payload.getvalue())
        broken = OwaspTop102021Parser(raw_dir=raw, output_dir=tmp_path / "out")
        broken.expected_sha256 = None
        with pytest.raises(ValueError, match="no '## Description' section"):
            broken.parse()


class TestRun:
    def test_run_writes(
        self, parser: OwaspTop102021Parser, tmp_path: Path,
    ) -> None:
        (tmp_path / "out").mkdir()
        output = parser.run()
        assert len(output.controls) == 1
        assert [s.path for s in output.source_files] == ["owasp_top10_2021.zip"]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_parse_owasp_top10_2021.py -q
```

Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Write the parser**

```python
# parsers/parse_owasp_top10_2021.py — create

"""Parser for the OWASP Top 10 2021.

The archive carries every Top 10 edition from 2003 to 2025 in every
translation, 199 markdown files and 196 MB. Only `2021/docs/en/A0N_2021-*.md`
and `2021/docs/en/A10_2021-*.md` are read, and the member list is filtered by
name so the other 187 files are never decompressed.

Twelve files match the A-prefix pattern. `A00` is *How to start an AppSec
Program* and `A11` is *Next Steps*; neither is a category and neither carries a
curated link. `A00` is excluded by its H1, which has no `A0N:2021` code; `A11`
is excluded by CATEGORY_IDS, because its H1 does carry one.

`description` is the `## Description` section and `full_text` is the whole
entry. `## Overview` is deliberately excluded: it is release commentary about
where the category moved in the rankings, which describes the survey rather
than the risk. `## How to Prevent` and `## Example Attack Scenarios` are the
two headings in tract.config.REMEDIATION_HEADINGS that this framework is the
original reason for, and they say how to satisfy the control rather than what
it is.
"""
from __future__ import annotations

import hashlib
import logging
import re
import zipfile
from io import BytesIO
from typing import ClassVar, Final

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARCHIVE_NAME: Final[str] = "owasp_top10_2021.zip"
SOURCE_SHA256: Final[str] = (
    "7f4747a7d7958d58ae3a4c7f7329740b9363c4788655bc3f28da8fdbedf48b5d"
)

MEMBER: Final[re.Pattern[str]] = re.compile(r"/2021/docs/en/A\d\d_2021-.*\.md$")
# The en dash is the source's own separator. A hyphen and an em dash are
# accepted too, so a typographic edit upstream does not silently drop a
# category.
HEADING: Final[re.Pattern[str]] = re.compile(
    r"^#\s+(A\d\d):2021\s*[–—-]\s*(.+?)\s*$", re.M
)
DESCRIPTION: Final[re.Pattern[str]] = re.compile(
    r"^##\s+Description\s*$(.*?)(?=^##\s|\Z)", re.M | re.S
)

CATEGORY_IDS: Final[tuple[str, ...]] = tuple(
    f"A{n:02d}" for n in range(1, 11)
)


class OwaspTop102021Parser(BaseParser):
    framework_id: ClassVar[str] = "owasp_top10_2021"
    framework_name: ClassVar[str] = "OWASP Top 10 2021"
    version: ClassVar[str] = "2021"
    source_url: ClassVar[str] = "https://owasp.org/Top10/"
    mapping_unit_level: ClassVar[str] = "category"
    expected_count: ClassVar[int] = 10
    fetched_date: ClassVar[str] = "2026-08-15"
    min_prose_fraction: ClassVar[float] = 1.0
    expected_sha256: ClassVar[str | None] = SOURCE_SHA256

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(ARCHIVE_NAME)
        self._check_digest(payload)
        controls: list[Control] = []
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            for name in sorted(n for n in archive.namelist() if MEMBER.search(n)):
                text = archive.read(name).decode("utf-8")
                heading = HEADING.search(text)
                if heading is None:
                    logger.info(
                        "%s: %s has no A0N:2021 heading, so it is front "
                        "matter rather than a category", self.framework_id,
                        name.rsplit("/", 1)[-1],
                    )
                    continue
                if heading.group(1) not in CATEGORY_IDS:
                    logger.info(
                        "%s: skipping %s, which is numbered outside the ten "
                        "categories", self.framework_id, heading.group(1),
                    )
                    continue
                controls.append(self.control_from_markdown(text))
        found = tuple(c.control_id for c in controls)
        if found != CATEGORY_IDS:
            raise ValueError(
                f"{self.framework_id}: expected categories "
                f"{list(CATEGORY_IDS)}, found {list(found)}. A short list "
                f"would ship a partial Top 10 that every gate downstream "
                f"would accept."
            )
        return controls

    def _check_digest(self, payload: bytes) -> None:
        """Refuse an archive that is not the pinned one.

        Raises:
            ValueError: If the digest does not match `expected_sha256`.
        """
        if self.expected_sha256 is None:
            return
        actual = hashlib.sha256(payload).hexdigest()
        if actual != self.expected_sha256:
            raise ValueError(
                f"{self.framework_id}: {ARCHIVE_NAME} has sha256 {actual}, "
                f"not the pinned {self.expected_sha256}."
            )

    @classmethod
    def control_from_markdown(cls, text: str) -> Control:
        """One category from one markdown file.

        Raises:
            ValueError: If the file has no heading or no Description section.
        """
        heading = HEADING.search(text)
        if heading is None:
            raise ValueError(
                "owasp_top10_2021: markdown with no 'A0N:2021 - Title' "
                "heading reached control_from_markdown."
            )
        body = DESCRIPTION.search(text)
        if body is None:
            raise ValueError(
                f"owasp_top10_2021: {heading.group(1)} has no '## Description' "
                f"section. Unhandled, its statement would fall back to the "
                f"Overview, which is commentary about the survey rather than "
                f"about the risk."
            )
        return Control(
            control_id=heading.group(1),
            title=heading.group(2).split("![", 1)[0].strip(),
            description=body.group(1).strip(),
            full_text=text[heading.end():].strip(),
            hierarchy_level="category",
        )


def main() -> None:
    OwaspTop102021Parser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and typecheck**

```bash
pytest tests/test_parse_owasp_top10_2021.py -q
mypy parsers/parse_owasp_top10_2021.py --strict
```

- [ ] **Step 5: Run against the real source and check the join**

```bash
PYTHONPATH=. "$PY" parsers/parse_owasp_top10_2021.py
PYTHONPATH=. "$PY" parsers/merge_all_controls.py
PYTHONPATH=. "$PY" scripts/corpus_report.py | grep -E "^(owasp_top10_2021|framework)"
```

Expected:

```
owasp_top10_2021              17     0   17     0      10  1.70     0    0     0     0 1.0000
```

Accept only if `resolution_rate == 1.0000` and `distinct_anchors == 10`.
`links_per_anchor` of 1.70 is not collapse: the source has ten categories and
OpenCRE links three of them twice. `by_title` must be 0, because the two
cosmetic name divergences mean no `section_name` spells a category title —
if it is not 0, the title channel is matching something it should not.

- [ ] **Step 6: Commit**

```bash
git add parsers/parse_owasp_top10_2021.py tests/test_parse_owasp_top10_2021.py \
        data/processed/frameworks/owasp_top10_2021.json \
        data/processed/all_controls.json
git commit -m "feat: parse the ten OWASP Top 10 2021 categories from the English 2021 tree"
```

---

### Task 6: OWASP Proactive Controls — 76 links that buy nothing until Task 14

Ten controls `C1`..`C10`, all present, all with a `## Description` section.
Ceiling **76/76 = 1.0000** **[derived]**, floor **1.00**.

**State the post-gate contribution honestly.** All 76 of this framework's
curated links are dropped before training by `assign_quality_tier`, twice over:
`owasp_proactive_controls` is in `PHASE1B_DROPPED_FRAMEWORKS`, and all 76 of its
`section_name` values are `C1`..`C10`, two or three characters, under
`PHASE1B_MIN_SECTION_TEXT_LENGTH = 10`. **[measured]** This parser contributes
**0 training links** until Task 14 retires both gates. The same is true of
`nist_800_63`'s 79. The previous plan's task table credited 155 links that buy
nothing while leaving the gates in place.

**Anchor count does not improve.** 76 links land on 10 anchors today and on 10
anchors after. The gain here is entirely in text: from a two-character label to
a paragraph. The instrument will show `links_per_anchor` unchanged at 7.60,
and that is the correct outcome for a source with ten mapping units and 76
links, not a regression.

Two decoys share the archive and the same `c<N>-` filename pattern:
`docs/archive/2018/c*.md` is the superseded v3 text and `v3/*` is 24 MB of
binary exports. **[measured]** The member filter anchors on
`docs/the-top-10/`.

**Files:**
- Create: `parsers/parse_owasp_proactive_controls.py`
- Create: `tests/test_parse_owasp_proactive_controls.py`

**Interfaces:**
- Consumes: `BaseParser`, `Control`.
- Produces: `OwaspProactiveControlsParser` with `framework_id = "owasp_proactive_controls"`, `framework_name = "OWASP Proactive Controls"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_owasp_proactive_controls.py — create

"""Ten controls, from docs/the-top-10 only.

The archive carries the same c<N>- filenames twice more: docs/archive/2018 is
the superseded v3 prose and v3/ is 24 MB of binary exports of it. A glob on
the filename pattern rather than on the directory picks up both.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from parsers.parse_owasp_proactive_controls import OwaspProactiveControlsParser

CURRENT = """# C1: Implement Access Control

## Description

Access Control (or Authorization) is allowing or denying specific requests from
a user, program, or process. With each access control decision, a given subject
requests access to a given object.

## Threats

Broken access control is the most commonly encountered risk.

## Implementation

### 1) Design access control thoroughly up front
"""

ARCHIVED = """# C1: Access Control (2018)

## Description

The superseded 2018 wording that must never reach the corpus.
"""


@pytest.fixture()
def parser(tmp_path: Path) -> OwaspProactiveControlsParser:
    raw = tmp_path / "raw"
    raw.mkdir()
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("pc-abc/docs/the-top-10/c1-accesscontrol.md", CURRENT)
        archive.writestr("pc-abc/docs/archive/2018/c1-accesscontrol.md", ARCHIVED)
    (raw / "owasp_proactive_controls.zip").write_bytes(payload.getvalue())

    instance = OwaspProactiveControlsParser(
        raw_dir=raw, output_dir=tmp_path / "out",
    )
    instance.expected_count = 1
    instance.expected_sha256 = None
    return instance


class TestParse:
    def test_reads_the_current_edition_only(
        self, parser: OwaspProactiveControlsParser,
    ) -> None:
        controls = parser.parse()
        assert [c.control_id for c in controls] == ["C1"]
        assert "superseded" not in controls[0].description

    def test_title_is_the_heading_after_the_code(
        self, parser: OwaspProactiveControlsParser,
    ) -> None:
        assert parser.parse()[0].title == "Implement Access Control"

    def test_description_stops_before_implementation(
        self, parser: OwaspProactiveControlsParser,
    ) -> None:
        text = parser.parse()[0].description
        assert text.startswith("Access Control (or Authorization)")
        assert "Design access control thoroughly" not in text


class TestRun:
    def test_run_writes(
        self, parser: OwaspProactiveControlsParser, tmp_path: Path,
    ) -> None:
        (tmp_path / "out").mkdir()
        output = parser.run()
        assert len(output.controls) == 1
        assert [s.path for s in output.source_files] == [
            "owasp_proactive_controls.zip",
        ]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_parse_owasp_proactive_controls.py -q
```

Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Write the parser**

```python
# parsers/parse_owasp_proactive_controls.py — create

"""Parser for the OWASP Proactive Controls, current mkdocs edition.

Ten controls, C1 through C10, one markdown file each under
`docs/the-top-10/`. Every one of the 76 curated links carries a section_id of
`C1`..`C10` and a section_name that is the same two or three characters, so the
join runs entirely through the id channel and the title channel never fires.

This framework contributes zero training links today: it is named in
PHASE1B_DROPPED_FRAMEWORKS, and every one of its section_names is shorter than
PHASE1B_MIN_SECTION_TEXT_LENGTH. Both gates test a title. Retiring them so they
test the resolved anchor is a separate task; this parser is the thing that
makes retiring them safe, because until the prose exists there is nothing for
the gate to test.

Two decoys in the same archive use the same c<N>- filenames: docs/archive/2018
holds the superseded v3 prose and v3/ holds binary exports of it. The member
filter anchors on the directory, not on the filename.
"""
from __future__ import annotations

import hashlib
import logging
import re
import zipfile
from io import BytesIO
from typing import ClassVar, Final

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARCHIVE_NAME: Final[str] = "owasp_proactive_controls.zip"
SOURCE_SHA256: Final[str] = (
    "6db1aafd6ecd758f05cf6b4133ec7085eb95016ec41afc5f462b4683c603b19d"
)

MEMBER: Final[re.Pattern[str]] = re.compile(r"/docs/the-top-10/c\d+-.*\.md$")
HEADING: Final[re.Pattern[str]] = re.compile(r"^#\s+(C\d+):\s*(.+?)\s*$", re.M)
DESCRIPTION: Final[re.Pattern[str]] = re.compile(
    r"^##\s+Description\s*$(.*?)(?=^##\s|\Z)", re.M | re.S
)

CONTROL_IDS: Final[frozenset[str]] = frozenset(f"C{n}" for n in range(1, 11))


class OwaspProactiveControlsParser(BaseParser):
    framework_id: ClassVar[str] = "owasp_proactive_controls"
    framework_name: ClassVar[str] = "OWASP Proactive Controls"
    version: ClassVar[str] = "2024"
    source_url: ClassVar[str] = "https://top10proactive.owasp.org/"
    mapping_unit_level: ClassVar[str] = "control"
    expected_count: ClassVar[int] = 10
    fetched_date: ClassVar[str] = "2026-08-15"
    min_prose_fraction: ClassVar[float] = 1.0
    expected_sha256: ClassVar[str | None] = SOURCE_SHA256

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(ARCHIVE_NAME)
        self._check_digest(payload)
        controls: list[Control] = []
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            for name in sorted(n for n in archive.namelist() if MEMBER.search(n)):
                controls.append(
                    self._control(archive.read(name).decode("utf-8"), name)
                )
        found = {c.control_id for c in controls}
        unknown = found - CONTROL_IDS
        if unknown:
            raise ValueError(
                f"{self.framework_id}: read control id(s) {sorted(unknown)} "
                f"outside C1..C10. Either the edition renumbered or a decoy "
                f"directory reached the member filter."
            )
        return sorted(controls, key=lambda c: int(c.control_id[1:]))

    def _check_digest(self, payload: bytes) -> None:
        """Refuse an archive that is not the pinned one.

        Raises:
            ValueError: If the digest does not match `expected_sha256`.
        """
        if self.expected_sha256 is None:
            return
        actual = hashlib.sha256(payload).hexdigest()
        if actual != self.expected_sha256:
            raise ValueError(
                f"{self.framework_id}: {ARCHIVE_NAME} has sha256 {actual}, "
                f"not the pinned {self.expected_sha256}."
            )

    @staticmethod
    def _control(text: str, member: str) -> Control:
        """One control from one markdown file.

        Raises:
            ValueError: If the heading or the Description section is absent.
        """
        heading = HEADING.search(text)
        if heading is None:
            raise ValueError(
                f"owasp_proactive_controls: {member} has no '# Cn: Title' "
                f"heading. The code in that heading is the only identifier "
                f"OpenCRE links against."
            )
        body = DESCRIPTION.search(text)
        if body is None:
            raise ValueError(
                f"owasp_proactive_controls: {heading.group(1)} in {member} "
                f"has no '## Description' section, so its statement would be "
                f"the Threats section, which describes the attack rather "
                f"than the control."
            )
        return Control(
            control_id=heading.group(1),
            title=heading.group(2).strip(),
            description=body.group(1).strip(),
            full_text=text[heading.end():].strip(),
            hierarchy_level="control",
        )


def main() -> None:
    OwaspProactiveControlsParser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and typecheck**

```bash
pytest tests/test_parse_owasp_proactive_controls.py -q
mypy parsers/parse_owasp_proactive_controls.py --strict
```

- [ ] **Step 5: Run against the real source and check the join**

```bash
PYTHONPATH=. "$PY" parsers/parse_owasp_proactive_controls.py
PYTHONPATH=. "$PY" parsers/merge_all_controls.py
PYTHONPATH=. "$PY" scripts/corpus_report.py | grep -E "^(owasp_proactive|framework)"
```

Expected:

```
owasp_proactive_controls      76     0   76     0      10  7.60     0    0     0     0 1.0000
```

Accept only if `resolution_rate == 1.0000` and `distinct_anchors == 10`. A
`links_per_anchor` of 7.60 is the source's own shape, not a parser defect: ten
controls carry 76 links. Record it as a known concentration; it is the second
worst in the corpus after `owasp_cheat_sheets` at 7.98.

- [ ] **Step 6: Commit**

```bash
git add parsers/parse_owasp_proactive_controls.py \
        tests/test_parse_owasp_proactive_controls.py \
        data/processed/frameworks/owasp_proactive_controls.json \
        data/processed/all_controls.json
git commit -m "feat: parse the ten Proactive Controls from the current mkdocs tree"
```

---

### Task 7: WSTG — 115 tests, and nine links that can never resolve

118 curated links over 59 distinct `section_id` values. The archive's
`document/4-Web_Application_Security_Testing/` tree has 130 test markdown files
excluding category READMEs, of which **115** carry the two-row ID table and 14
do not — the 14 are sub-tests (`05.1-Testing_for_Oracle.md` and similar) that
share their parent's id. **[measured]** The structures document says 144 files;
that count included the category README files.

**Four of the 59 section_ids exist nowhere in the archive**: `WSTG-APPE-D`,
`WSTG-BUSL-$$`, `WSTG-INFO-##`, `WSTG-INPV-00`, carrying 2, 3, 1 and 3 links.
**[measured]** Grepping all 199 markdown members for each string returns zero
hits. Those 9 links are unresolvable by any parser.

**Ceiling: 109/118 = 0.92373** **[derived]**, floor **0.92**. The previous plan
set 0.96 against a stated maximum of 0.9322; both numbers are wrong and 0.96 was
unreachable. Note that 0.9237 is exactly the bogus-id limit: under the statement
rule below, no linked id fails the prose test, so the id gap is the whole gap.

**One id maps to two files.** `WSTG-INPV-13` is the ID table value in both
`13-Testing_for_Buffer_Overflow.md` and `13-Testing_for_Format_String_Injection.md`.
**[measured]** One curated link targets it. Ruling: emit one control whose title
joins both H1s with ` / ` and whose statement concatenates both bodies in
filename order. A merge across source files is a text-moving transform, so it
writes a repair-audit record naming both members.

**Statement rule.** The body from the end of the ID table to the first `##`
heading in `REMEDIATION_HEADINGS` ∪ {`How to Test`, `Related Test Cases`,
`Tools`}, with heading markers stripped. Measured over the 115 ids: 2 fail the
prose test (`WSTG-CONF-08`, `WSTG-CLNT-08`, neither linked), median 1,631
characters, 38 over `MAX_ANCHOR_CHARS` and 45 over `DESCRIPTION_MAX_LENGTH`.
**[measured]** Taking `## Summary` alone instead leaves 8 ids with no section at
all.

**Files:**
- Create: `parsers/parse_wstg.py`
- Create: `tests/test_parse_wstg.py`

**Interfaces:**
- Consumes: `BaseParser`, `Control`, `BaseParser.write_repair_audit`, `tract.config.REMEDIATION_HEADINGS`.
- Produces: `WstgParser` with `framework_id = "wstg"`, `framework_name = "WSTG"`; `WstgParser.build_controls(entries: list[tuple[str, str, str]]) -> tuple[list[Control], list[dict[str, object]]]` returning controls and audit records.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_wstg.py — create

"""WSTG joins on the ID table, and one id owns two files.

Measured: 115 of 130 test files carry an ID table, WSTG-INPV-13 appears in two
of them, and four section_ids in the curated links appear in none of the 199
markdown members in the archive.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from parsers.parse_wstg import WstgParser

BASE = "document/4-Web_Application_Security_Testing"


def _test_file(test_id: str, title: str, summary: str) -> str:
    return (
        f"# {title}\n\n"
        f"|ID          |\n|------------|\n|{test_id}|\n\n"
        f"## Summary\n\n{summary}\n\n"
        f"## How to Test\n\nRun the tool and read the output carefully.\n\n"
        f"## References\n\n- Somewhere\n"
    )


@pytest.fixture()
def parser(tmp_path: Path) -> WstgParser:
    raw = tmp_path / "raw"
    raw.mkdir()
    payload = io.BytesIO()
    long_summary = (
        "Search engines crawl and index billions of pages, and the index can "
        "retain content the owner has since removed from the live site."
    )
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(
            f"wstg-abc/{BASE}/01-Information_Gathering/01-Conduct_Search.md",
            _test_file("WSTG-INFO-01", "Conduct Search Engine Discovery",
                       long_summary),
        )
        archive.writestr(
            f"wstg-abc/{BASE}/07-Input_Validation_Testing/13-Testing_for_Buffer_Overflow.md",
            _test_file("WSTG-INPV-13", "Testing for Buffer Overflow",
                       "A buffer overflow overwrites adjacent memory and can "
                       "hand an attacker control of execution flow."),
        )
        archive.writestr(
            f"wstg-abc/{BASE}/07-Input_Validation_Testing/13-Testing_for_Format_String_Injection.md",
            _test_file("WSTG-INPV-13", "Testing for Format String Injection",
                       "A format string bug lets an attacker read and write "
                       "process memory through the conversion specifiers."),
        )
        archive.writestr(
            f"wstg-abc/{BASE}/07-Input_Validation_Testing/05.1-Testing_for_Oracle.md",
            "# Testing for Oracle\n\nA sub-test with no ID table of its own.\n",
        )
        archive.writestr(f"wstg-abc/{BASE}/01-Information_Gathering/README.md",
                         "# Information Gathering\n\nCategory intro.\n")
    (raw / "wstg.zip").write_bytes(payload.getvalue())

    instance = WstgParser(
        raw_dir=raw, output_dir=tmp_path / "out", audit_dir=tmp_path / "audit",
    )
    instance.expected_count = 2
    instance.expected_sha256 = None
    return instance


class TestParse:
    def test_only_files_with_an_id_table_become_controls(
        self, parser: WstgParser,
    ) -> None:
        assert sorted(c.control_id for c in parser.parse()) == [
            "WSTG-INFO-01", "WSTG-INPV-13",
        ]

    def test_title_is_the_h1_not_the_id(self, parser: WstgParser) -> None:
        first = next(c for c in parser.parse() if c.control_id == "WSTG-INFO-01")
        assert first.title == "Conduct Search Engine Discovery"

    def test_statement_stops_before_how_to_test(
        self, parser: WstgParser,
    ) -> None:
        first = next(c for c in parser.parse() if c.control_id == "WSTG-INFO-01")
        assert "Search engines crawl" in first.description
        assert "Run the tool" not in first.description

    def test_a_shared_id_merges_both_files(self, parser: WstgParser) -> None:
        shared = next(c for c in parser.parse() if c.control_id == "WSTG-INPV-13")
        assert shared.title == (
            "Testing for Buffer Overflow / Testing for Format String Injection"
        )
        assert "adjacent memory" in shared.description
        assert "conversion specifiers" in shared.description


class TestAudit:
    def test_the_merge_is_recorded(
        self, parser: WstgParser, tmp_path: Path,
    ) -> None:
        (tmp_path / "out").mkdir()
        parser.run()
        lines = (tmp_path / "audit" / "wstg.jsonl").read_text(
            encoding="utf-8",
        ).splitlines()
        records = [json.loads(line) for line in lines]
        assert len(records) == 1
        assert records[0]["control_id"] == "WSTG-INPV-13"
        assert len(records[0]["members"]) == 2

    def test_the_audit_file_is_written_even_with_no_merges(
        self, tmp_path: Path,
    ) -> None:
        raw = tmp_path / "solo"
        raw.mkdir()
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr(
                f"wstg-abc/{BASE}/01-Information_Gathering/01-One.md",
                _test_file("WSTG-INFO-01", "One",
                           "A statement long enough to clear every prose bar "
                           "that this project applies to a description."),
            )
        (raw / "wstg.zip").write_bytes(payload.getvalue())
        solo = WstgParser(raw_dir=raw, output_dir=tmp_path / "out2",
                          audit_dir=tmp_path / "audit2")
        solo.expected_count = 1
        solo.expected_sha256 = None
        (tmp_path / "out2").mkdir()
        solo.run()
        assert (tmp_path / "audit2" / "wstg.jsonl").read_text(
            encoding="utf-8",
        ) == ""


class TestRun:
    def test_run_writes(self, parser: WstgParser, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        output = parser.run()
        assert len(output.controls) == 2
        assert [s.path for s in output.source_files] == ["wstg.zip"]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_parse_wstg.py -q
```

Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Write the parser**

```python
# parsers/parse_wstg.py — create

"""Parser for the OWASP Web Security Testing Guide.

The join key is the value in the two-row ID table under each test's H1, and it
is read from that table rather than derived from the path: the directory prefix
is a zero-padded number (`01-Information_Gathering`) while the id prefix is a
four-letter mnemonic (`WSTG-INFO-01`).

OpenCRE sets section_name equal to section_id for all 118 curated links, so
the link side carries no human title at all. The parser's title is the file's
H1, which is richer and which no link name spells, so the whole join runs
through the id channel and the title channel cannot misfire.

Nine of the 118 links can never resolve. Their section_ids -- WSTG-APPE-D,
WSTG-BUSL-$$, WSTG-INFO-## and WSTG-INPV-00 -- appear in none of the 199
markdown members of the pinned archive. That is an upstream extraction
artifact, not something a parser can fix, and it sets this framework's ceiling
at 109 of 118.

WSTG-INPV-13 is the ID table value of two files, Buffer Overflow and Format
String Injection. They are merged into one control, because the id is what the
link targets and emitting two controls with the same control_id would let
whichever came last silently win. The merge moves text across a file boundary
so it writes an audit record naming both members.
"""
from __future__ import annotations

import hashlib
import logging
import re
import zipfile
from collections import defaultdict
from io import BytesIO
from typing import ClassVar, Final

from tract.config import REMEDIATION_HEADINGS
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARCHIVE_NAME: Final[str] = "wstg.zip"
SOURCE_SHA256: Final[str] = (
    "e093f1648fbf4195f2a8fccac4f80315fb6b6281af85aa557edb34d0f9c58b33"
)

MEMBER: Final[re.Pattern[str]] = re.compile(
    r"/document/4-Web_Application_Security_Testing/[^/]+/(?!README\.md$).*\.md$"
)
ID_TABLE: Final[re.Pattern[str]] = re.compile(
    r"^\|\s*(WSTG-[A-Z]+-\d+)\s*\|\s*$", re.M
)
H1: Final[re.Pattern[str]] = re.compile(r"^#\s+(.+?)\s*$", re.M)

# Where the statement ends. REMEDIATION_HEADINGS supplies Remediation,
# References and the rest; these three are WSTG's own procedural sections and
# describe how to run the test rather than what the test is for.
_EXTRA_CUTS: Final[tuple[str, ...]] = (
    "How to Test", "Related Test Cases", "Tools",
)
CUT: Final[re.Pattern[str]] = re.compile(
    r"^##\s+(?:"
    + "|".join(re.escape(h) for h in (*REMEDIATION_HEADINGS, *_EXTRA_CUTS))
    + r")\s*$",
    re.M,
)
_HEADING_MARKER: Final[re.Pattern[str]] = re.compile(r"^#{2,6}\s*", re.M)


class WstgParser(BaseParser):
    framework_id: ClassVar[str] = "wstg"
    # canonical_framework maps OpenCRE's "OWASP Web Security Testing Guide
    # (WSTG)" onto this through the existing FRAMEWORK_NAME_ALIASES entry.
    framework_name: ClassVar[str] = "WSTG"
    version: ClassVar[str] = "4.2"
    source_url: ClassVar[str] = "https://owasp.org/www-project-web-security-testing-guide/"
    mapping_unit_level: ClassVar[str] = "test"
    # 115 distinct ids across 130 test files; 14 sub-tests carry no id table
    # of their own and one id owns two files. [measured]
    expected_count: ClassVar[int] = 115
    fetched_date: ClassVar[str] = "2026-08-15"
    # 113 of 115 statements clear the 60-character bar and differ from their
    # H1. The two that do not, WSTG-CONF-08 and WSTG-CLNT-08, carry no curated
    # link. [measured]
    min_prose_fraction: ClassVar[float] = 0.98
    expected_sha256: ClassVar[str | None] = SOURCE_SHA256

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(ARCHIVE_NAME)
        self._check_digest(payload)
        entries = self._read_entries(payload)
        controls, audit = self.build_controls(entries)
        self.write_repair_audit(audit)
        for record in audit:
            logger.warning(
                "%s: %s owns %d source files; their statements were merged: %s",
                self.framework_id, record["control_id"],
                len(record["members"]), record["members"],
            )
        logger.info(
            "%s: %d distinct ids from %d files with an ID table",
            self.framework_id, len(controls), len(entries),
        )
        return controls

    def _check_digest(self, payload: bytes) -> None:
        """Refuse an archive that is not the pinned one.

        Raises:
            ValueError: If the digest does not match `expected_sha256`.
        """
        if self.expected_sha256 is None:
            return
        actual = hashlib.sha256(payload).hexdigest()
        if actual != self.expected_sha256:
            raise ValueError(
                f"{self.framework_id}: {ARCHIVE_NAME} has sha256 {actual}, "
                f"not the pinned {self.expected_sha256}."
            )

    def _read_entries(self, payload: bytes) -> list[tuple[str, str, str]]:
        """(test_id, member name, file text) for every file with an ID table.

        Raises:
            ValueError: If no member carries an ID table at all.
        """
        entries: list[tuple[str, str, str]] = []
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            for name in sorted(n for n in archive.namelist() if MEMBER.search(n)):
                text = archive.read(name).decode("utf-8")
                table = ID_TABLE.search(text)
                if table is None:
                    logger.debug(
                        "%s: %s has no ID table; it is a sub-test that shares "
                        "its parent's id", self.framework_id, name,
                    )
                    continue
                entries.append((table.group(1), name, text))
        if not entries:
            raise ValueError(
                f"{self.framework_id}: no member of {ARCHIVE_NAME} carries a "
                f"WSTG-XXXX-NN ID table. The table is the only join key; the "
                f"path prefix does not spell the id."
            )
        return entries

    @classmethod
    def build_controls(
        cls, entries: list[tuple[str, str, str]],
    ) -> tuple[list[Control], list[dict[str, object]]]:
        """One Control per distinct id, merging any id that owns two files."""
        grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for test_id, member, text in entries:
            grouped[test_id].append((member, text))

        controls: list[Control] = []
        audit: list[dict[str, object]] = []
        for test_id in sorted(grouped):
            members = sorted(grouped[test_id])
            titles = [cls._title(text, member) for member, text in members]
            statements = [cls._statement(text) for _, text in members]
            bodies = [cls._body(text) for _, text in members]
            if len(members) > 1:
                audit.append({
                    "control_id": test_id,
                    "members": [member for member, _ in members],
                    "titles": titles,
                    "statement_lengths": [len(s) for s in statements],
                    "reason": (
                        "one WSTG id is the ID table value of more than one "
                        "file; emitting one control per file would make two "
                        "controls share a control_id and let the later one win"
                    ),
                })
            controls.append(Control(
                control_id=test_id,
                title=" / ".join(titles),
                description="\n\n".join(s for s in statements if s),
                full_text="\n\n".join(b for b in bodies if b) or None,
                hierarchy_level="test",
                parent_id=members[0][0].split("/")[-2],
                metadata={"source_members": [member for member, _ in members]},
            ))
        return controls, audit

    @staticmethod
    def _title(text: str, member: str) -> str:
        """The file's H1.

        Raises:
            ValueError: If the file has no H1.
        """
        heading = H1.search(text)
        if heading is None:
            raise ValueError(
                f"wstg: {member} carries an ID table but no H1. The H1 is the "
                f"control title, and OpenCRE supplies none of its own for "
                f"this framework."
            )
        return heading.group(1).strip()

    @classmethod
    def _statement(cls, text: str) -> str:
        body = cls._body(text)
        cut = CUT.search(body)
        head = body[: cut.start()] if cut else body
        return _HEADING_MARKER.sub("", head).strip()

    @staticmethod
    def _body(text: str) -> str:
        table = ID_TABLE.search(text)
        return text[table.end():].strip() if table else text.strip()


def main() -> None:
    WstgParser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and typecheck**

```bash
pytest tests/test_parse_wstg.py -q
mypy parsers/parse_wstg.py --strict
```

- [ ] **Step 5: Run against the real source and check the join**

```bash
PYTHONPATH=. "$PY" parsers/parse_wstg.py
PYTHONPATH=. "$PY" parsers/merge_all_controls.py
PYTHONPATH=. "$PY" scripts/corpus_report.py | grep -E "^(wstg|framework)"
```

Expected `wstg: 115 distinct ids from 116 files with an ID table` and one
`WSTG-INPV-13 owns 2 source files` warning **[measured]**, then:

```
wstg                         118     0  109     9      55  1.98    ~20    0     2     0 0.9237
```

Accept only if `resolution_rate >= 0.92`, `distinct_anchors >= 55`,
`unresolved == 9`, `by_title == 0`, `wrong_anchor_risk == 0`. Any value of
`unresolved` other than 9 means the bogus-id set changed; list it before
accepting. `truncated` is expected to be roughly 20 of 109 given that 38 of the
115 statements exceed `MAX_ANCHOR_CHARS`; record the actual number rather than
asserting it.

- [ ] **Step 6: Confirm the audit file is gitignored**

```bash
git check-ignore -v data/processed/repair_audit/wstg.jsonl \
  || echo "NOT IGNORED — stop and fix .gitignore before committing"
```

- [ ] **Step 7: Commit**

```bash
git add parsers/parse_wstg.py tests/test_parse_wstg.py \
        data/processed/frameworks/wstg.json data/processed/all_controls.json
git commit -m "feat: parse WSTG on the ID table, merging the one id that owns two files"
```

---
### Task 8: CSA CCM — 207 controls, 17 domains, and what a domain aggregate should be

**Corrected count.** The `CCM` sheet has 229 rows: 208 with all four columns
populated, of which **one is the header row** `Control Domain | Control Title |
Control ID | Control Specification`, leaving **207 control rows**; 19 with only
column A, of which 17 are domain headers and 2 are the `End of Standard` and
copyright trailers; one title row and one blank. **[measured]** The previous
plan declared `expected_count = 225` with a comment reading "208 control rows
plus 17 domains", counting the header. Its parser skips the header, so it emits
**224**, and 224 against 225 is a 0.44% deviation — **inside**
`COUNT_TOLERANCE` of 10%, so `_check_expected_count` would **not** have raised.
That is worse than a raise. A declared count that is wrong but inside tolerance
is indistinguishable from the parser silently dropping a control. The correct
value is **224**.

**No rename map is needed.** Seven curated links carry v4.0's `IVS-*` ids,
which v4.1.0 renamed to `I&S-*`. All seven `section_name` values match the
corresponding `I&S-*` control's title exactly, so title-first resolution
answers all seven with no alias at all. **[measured]** The previous plan's
`V40_DOMAIN_RENAMES` is dead machinery and is not built.

**Ceiling: 29/29 = 1.0000** **[derived]** — 7 control-level ids, 7 title hits
on the renamed ids, 15 domain codes. Floor **1.00**.

**The domain aggregate, decided by measurement.** Concatenating each domain's
member specifications gives lengths 1,022 to 4,292; **8 of 17 exceed
`MAX_ANCHOR_CHARS`**, and because the concatenation opens with the domain's own
first member control, **all 17** aggregates are a strict prefix of a control
that is itself an anchor. **[measured]** That is 17 near-duplicate anchor pairs
in a corpus of 29 links, and Task 1's `nested_anchors` column would read 17.

Concatenating the member **titles** instead gives 163 to 596 characters,
**0 over budget and 0 nested**. **[measured]** A domain in CCM is the set of
subjects its controls cover, and the ordered list of those subjects is a fair
statement of it. That is the rule below. The full specification text stays
reachable through each member control, which is where a specification-level
question belongs.

`openpyxl` is used by no file in this repository and appears in neither
`requirements.txt` nor `pyproject.toml`. **[measured]** It is added pinned.

**Files:**
- Create: `parsers/parse_csa_ccm.py`
- Create: `tests/test_parse_csa_ccm.py`
- Modify: `requirements.txt`, `requirements-lint.txt`, `pyproject.toml`

**Interfaces:**
- Consumes: `BaseParser`, `Control`.
- Produces: `CsaCcmParser` with `framework_id = "csa_ccm"`, `framework_name = "Cloud Controls Matrix"`; `CsaCcmParser.rows_to_controls(rows: list[tuple[str, str, str, str]]) -> list[Control]`.

- [ ] **Step 1: Pin openpyxl**

```text
# requirements.txt — add under the pdfplumber block
# parse_csa_ccm.py reads the CCM workbook. The CCM sheet is a flat four-column
# table; reading it through the raw sheet XML would mean reimplementing shared
# strings and inline formatting for one parser.
openpyxl==3.1.5
```

```text
# requirements-lint.txt — add beside the other runtime deps mypy imports
openpyxl
```

```toml
# pyproject.toml — add to the dependencies list, after "pdfplumber>=0.10.0",
    "openpyxl>=3.1.5",
```

```bash
"$PY" -m pip install "openpyxl==3.1.5"
"$PY" -c "import openpyxl; print(openpyxl.__version__)"
```

Expected: `3.1.5`.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_parse_csa_ccm.py — create

"""207 controls and 17 domains, and a domain is its members' subjects.

Measured on the pinned workbook: concatenating member specifications makes 8 of
17 domain anchors exceed MAX_ANCHOR_CHARS and makes all 17 a strict prefix of
their own first member control. Concatenating member titles makes 0 of either.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parsers.parse_csa_ccm import CsaCcmParser
from tract.config import MAX_ANCHOR_CHARS

ROWS: list[tuple[str, str, str, str]] = [
    ("{\"specification_name\":\"Cloud Controls Matrix\"}",
     "CLOUD CONTROLS MATRIX v4.1.0", "", ""),
    ("", "", "", ""),
    ("Control Domain", "Control Title", "Control ID", "Control Specification"),
    ("Audit & Assurance - A&A", "", "", ""),
    ("Audit & Assurance", "Audit and Assurance Policy and Procedures", "A&A-01",
     "Establish, document, approve, communicate, apply, evaluate and maintain "
     "audit and assurance policies and procedures and standards."),
    ("Audit & Assurance", "Independent Assessments", "A&A-02",
     "Conduct independent audit and assurance assessments according to "
     "relevant standards at least annually."),
    ("Infrastructure Security - I&S", "", "", ""),
    ("Infrastructure Security", "Capacity and Resource Planning", "I&S-02",
     "Plan and monitor the availability, quality, and adequate capacity of "
     "resources in order to deliver the required system performance."),
    ("End of Standard", "", "", ""),
    ("© Copyright 2026 Cloud Security Alliance - All rights reserved.",
     "", "", ""),
]


class TestRowsToControls:
    def test_the_header_row_is_not_a_control(self) -> None:
        controls = CsaCcmParser.rows_to_controls(ROWS)
        assert "Control ID" not in {c.control_id for c in controls}
        assert len([c for c in controls if c.hierarchy_level == "control"]) == 3

    def test_the_two_trailers_are_not_domains(self) -> None:
        domains = [
            c for c in CsaCcmParser.rows_to_controls(ROWS)
            if c.hierarchy_level == "domain"
        ]
        assert sorted(c.control_id for c in domains) == ["A&A", "I&S"]

    def test_a_domain_statement_lists_its_member_titles(self) -> None:
        domain = next(
            c for c in CsaCcmParser.rows_to_controls(ROWS)
            if c.control_id == "A&A"
        )
        assert domain.description == (
            "Audit and Assurance Policy and Procedures. Independent Assessments."
        )

    def test_a_domain_statement_does_not_open_its_first_member(self) -> None:
        controls = CsaCcmParser.rows_to_controls(ROWS)
        domain = next(c for c in controls if c.control_id == "A&A")
        first = next(c for c in controls if c.control_id == "A&A-01")
        assert not domain.description.startswith(first.description[:60])
        assert len(domain.description) <= MAX_ANCHOR_CHARS

    def test_a_control_carries_its_domain_as_parent(self) -> None:
        control = next(
            c for c in CsaCcmParser.rows_to_controls(ROWS)
            if c.control_id == "I&S-02"
        )
        assert control.parent_id == "I&S"
        assert control.parent_name == "Infrastructure Security"

    def test_a_domain_with_no_members_is_refused(self) -> None:
        rows = [*ROWS[:4], *ROWS[6:]]
        with pytest.raises(ValueError, match="no controls under it"):
            CsaCcmParser.rows_to_controls(rows)


class TestRun:
    def test_run_writes_from_the_real_workbook(self, tmp_path: Path) -> None:
        parser = CsaCcmParser(output_dir=tmp_path)
        try:
            output = parser.run()
        except FileNotFoundError:
            pytest.skip("data/raw is gitignored and absent in this checkout")
        assert len(output.controls) == 224
        assert sum(1 for c in output.controls if c.hierarchy_level == "domain") == 17
        assert [s.path for s in output.source_files] == [
            "CCMv4.1.0-generated_at_2026_01_13.xlsx",
        ]
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
pytest tests/test_parse_csa_ccm.py -q
```

Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 4: Write the parser**

```python
# parsers/parse_csa_ccm.py — create

"""Parser for the CSA Cloud Controls Matrix v4.1.0.

Not the AI Controls Matrix. This workbook's title cell reads CLOUD CONTROLS
MATRIX v4.1.0 and it is a different framework from csa_aicm, which has 243
controls and no CRE links.

The CCM sheet is a flat four-column table with two row types interleaved. A
control row populates all four columns; a domain header populates only column A
in the form "<Full Name> - <CODE>". Three column-A-only rows are neither: the
workbook title, the End of Standard trailer, and the copyright paragraph. And
one all-four-columns row is the header itself, which is why a naive count gives
208 control rows where there are 207.

Both granularities are emitted, because OpenCRE links both: 14 of the 29
curated links target a control id and 15 target a bare domain code.

A domain's statement is the ordered list of its member control TITLES, not the
concatenation of their specifications. Measured on this workbook, concatenating
specifications makes 8 of 17 domain anchors exceed MAX_ANCHOR_CHARS and makes
all 17 a strict prefix of their own first member control, which puts 17
near-duplicate pairs into a 29-link framework. The title list runs 163 to 596
characters, exceeds nothing, and prefixes nothing. A domain in CCM is the set
of subjects its controls cover; the list of those subjects states it, and the
specification text stays reachable through each member control.

The seven curated links that still use v4.0's IVS-* ids need no rename map:
their section_name matches the corresponding I&S-* control's title exactly, and
ProseIndex resolves title before id.
"""
from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from typing import ClassVar, Final

import openpyxl

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WORKBOOK_NAME: Final[str] = "CCMv4.1.0-generated_at_2026_01_13.xlsx"
SOURCE_SHA256: Final[str] = (
    "5e721628c8ab297bdbd355afa4c01699971fcbb9cb16802ccb9d42c7176ab32b"
)
SHEET_NAME: Final[str] = "CCM"

DOMAIN_HEADER: Final[re.Pattern[str]] = re.compile(r"^(.+?)\s+-\s+([A-Z&]{2,5})$")
HEADER_CELL: Final[str] = "Control ID"
# Column-A-only rows that look like a domain header and are not.
TRAILERS: Final[tuple[str, ...]] = ("End of Standard", "©", "{")

_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


class CsaCcmParser(BaseParser):
    framework_id: ClassVar[str] = "csa_ccm"
    # Matches the curated links' standard_name exactly; no alias entry exists
    # or is needed.
    framework_name: ClassVar[str] = "Cloud Controls Matrix"
    version: ClassVar[str] = "4.1.0"
    source_url: ClassVar[str] = (
        "https://cloudsecurityalliance.org/artifacts/cloud-controls-matrix-v4/"
    )
    mapping_unit_level: ClassVar[str] = "control"
    # 207 control rows plus 17 domains. The 208th all-four-columns row is the
    # sheet header. [measured]
    expected_count: ClassVar[int] = 224
    fetched_date: ClassVar[str] = "2026-08-15"
    # Every specification and every domain title list clears 60 characters and
    # differs from its title. [measured]
    min_prose_fraction: ClassVar[float] = 1.0
    expected_sha256: ClassVar[str | None] = SOURCE_SHA256

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(WORKBOOK_NAME)
        self._check_digest(payload)
        rows = self._read_sheet()
        controls = self.rows_to_controls(rows)
        logger.info(
            "%s: %d controls and %d domains",
            self.framework_id,
            sum(1 for c in controls if c.hierarchy_level == "control"),
            sum(1 for c in controls if c.hierarchy_level == "domain"),
        )
        return controls

    def _check_digest(self, payload: bytes) -> None:
        """Refuse a workbook that is not the pinned one.

        Raises:
            ValueError: If the digest does not match `expected_sha256`.
        """
        if self.expected_sha256 is None:
            return
        actual = hashlib.sha256(payload).hexdigest()
        if actual != self.expected_sha256:
            raise ValueError(
                f"{self.framework_id}: {WORKBOOK_NAME} has sha256 {actual}, "
                f"not the pinned {self.expected_sha256}. expected_count of "
                f"224 and the domain-statement length measurements were both "
                f"taken against the pinned bytes."
            )

    def _read_sheet(self) -> list[tuple[str, str, str, str]]:
        """The CCM sheet's first four columns, as stripped strings.

        openpyxl needs a path, and BaseParser has already read and hashed the
        bytes, so the file is opened a second time here. That is a deliberate
        exception to the in-memory rule: the manifest already records the
        digest of exactly these bytes, and openpyxl's read-only mode has no
        file-object-free entry point that avoids reimplementing shared strings.

        Raises:
            ValueError: If the CCM sheet is absent.
        """
        path = self.raw_dir / WORKBOOK_NAME
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            if SHEET_NAME not in workbook.sheetnames:
                raise ValueError(
                    f"{self.framework_id}: {WORKBOOK_NAME} has no {SHEET_NAME!r} "
                    f"sheet, only {workbook.sheetnames}. The CAIQ sheet is the "
                    f"self-assessment questionnaire and is not the controls."
                )
            rows: list[tuple[str, str, str, str]] = []
            for row in workbook[SHEET_NAME].iter_rows(values_only=True):
                cells = [
                    _WHITESPACE.sub(" ", str(cell or "")).strip()
                    for cell in (list(row) + [""] * 4)[:4]
                ]
                rows.append((cells[0], cells[1], cells[2], cells[3]))
            return rows
        finally:
            workbook.close()

    @classmethod
    def rows_to_controls(
        cls, rows: list[tuple[str, str, str, str]],
    ) -> list[Control]:
        """Controls then domains, from the sheet's interleaved row types.

        Raises:
            ValueError: If a domain header has no control rows under it.
        """
        controls: list[Control] = []
        domains: list[tuple[str, str]] = []
        members: dict[str, list[str]] = defaultdict(list)
        current_code = ""
        current_name = ""

        for first, title, control_id, specification in rows:
            if control_id and specification:
                if control_id == HEADER_CELL:
                    continue
                controls.append(Control(
                    control_id=control_id,
                    title=title,
                    description=specification,
                    hierarchy_level="control",
                    parent_id=current_code or None,
                    parent_name=current_name or first or None,
                ))
                members[current_code].append(title)
                continue

            if not first or title or control_id or specification:
                continue
            if first.startswith(TRAILERS):
                continue
            header = DOMAIN_HEADER.match(first)
            if header is None:
                logger.info(
                    "csa_ccm: column-A-only row %r is not a domain header",
                    first[:60],
                )
                continue
            current_name, current_code = header.group(1), header.group(2)
            domains.append((current_code, current_name))

        controls += cls._domain_controls(domains, members)
        return controls

    @staticmethod
    def _domain_controls(
        domains: list[tuple[str, str]], members: dict[str, list[str]],
    ) -> list[Control]:
        """One mapping unit per domain, stated as its members' subjects.

        Raises:
            ValueError: If a domain owns no controls, which means the row
                ordering changed and every domain statement built from that
                ordering is attached to the wrong domain.
        """
        built: list[Control] = []
        for code, name in domains:
            titles = members.get(code, [])
            if not titles:
                raise ValueError(
                    f"csa_ccm: domain {code} has no controls under it. Domain "
                    f"membership comes from row order, so an empty domain "
                    f"means the sheet was reordered and the domains that do "
                    f"have members may have the wrong ones."
                )
            built.append(Control(
                control_id=code,
                title=name,
                description=". ".join(titles) + ".",
                hierarchy_level="domain",
                metadata={"member_ids": list(titles)},
            ))
        return built


def main() -> None:
    CsaCcmParser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests and typecheck**

```bash
pytest tests/test_parse_csa_ccm.py -q
mypy parsers/parse_csa_ccm.py --strict
```

- [ ] **Step 6: Run against the real source and check the join**

```bash
PYTHONPATH=. "$PY" parsers/parse_csa_ccm.py
PYTHONPATH=. "$PY" parsers/merge_all_controls.py
PYTHONPATH=. "$PY" scripts/corpus_report.py | grep -E "^(csa_ccm|framework)"
```

Expected `csa_ccm: 207 controls and 17 domains` **[measured]**, then:

```
csa_ccm                       29     7   22     0      29  1.00     0    0     0     0 1.0000
```

Accept only if `resolution_rate == 1.0000`, `distinct_anchors == 29`,
**`nested_anchors == 0`**, and `by_title == 7`. The nesting column is the one
that would have caught the specification-concatenation design; if it reads 17,
the domain statement rule was not applied. `by_title == 7` is the seven renamed
`IVS-*` links resolving through their titles — if it reads 0, seven links are
unresolved and the rate will show it.

- [ ] **Step 7: Confirm csa_ccm is NOT gitignored**

The owner ruled on 2026-08-16 that the CCM is redistributable. It stays out of
`RESTRICTED_FRAMEWORK_IDS` and its processed file is tracked like any other.

```bash
git check-ignore -v data/processed/frameworks/csa_ccm.json \
  && echo "UNEXPECTEDLY IGNORED — check .gitignore against the owner ruling" \
  || echo "tracked, as ruled"
pytest tests/test_licensed_text_not_tracked.py tests/test_framework_licenses.py -q
```

- [ ] **Step 8: Commit**

```bash
git add parsers/parse_csa_ccm.py tests/test_parse_csa_ccm.py \
        requirements.txt requirements-lint.txt pyproject.toml \
        data/processed/frameworks/csa_ccm.json data/processed/all_controls.json
git commit -m "feat: parse the CCM at both granularities, stating a domain by its members' subjects"
```

---

### Task 9: NIST SSDF — the table is already ruled, and there is no merge step

**Verify the row shape before writing anything.** The previous plan's Task 9
specified a rowspan merge that absorbs continuation fragments. Measured against
the pinned PDF: `pdfplumber.extract_tables()` on pages 13 through 26 returns
**47 task cells, every one of them at column index 3, every one a whole cell
with the task statement complete and newlines inside it**. **[measured]** The
rows below a task row repeat wrapped fragments of the *practice* cell in column
1 and carry nothing in column 3. There is nothing to absorb. No merge step is
built here.

What the practice column does need is a **forward fill**: the practice cell is
populated on the first task row of each group and empty on the rest. After the
fill, 0 of 47 task rows have an empty practice. **[measured]**

**Five stub rows, not two.** `PW.3.1`, `PW.3.2`, `PW.4.3`, `PW.4.5` and `PW.5.2`
have bodies of the form `Moved to <target>`. **[measured]** The structures
document named two. None of the five is targeted by a curated link
**[measured]**, so they are recorded as redirects in metadata and excluded from
the emitted controls: a 15-character statement would drag the prose floor and
put a non-statement anchor in the corpus. That leaves **42 real tasks**, which
matches the document's own count.

**Two malformed link rows, not one.** Two of the 46 curated links carry a
mid-sentence fragment where a `PS.1.1`-style id belongs. The first fragment
appears verbatim inside task `PS.1.1`'s statement and the second inside
`PW.8.1`'s. **[measured]** Both are registered as `alt_ids`, which is what
Task 2 built.

**Title must be the task id.** OpenCRE sets `section_name` to the task
statement verbatim for 36 of 46 links. **[measured]** Task statements run 54 to
333 characters, median 163. If the parser used the statement as both `title`
and `description`, `_is_prose` would exclude every control from `ProseIndex`.
Using the practice name as the title also costs links: 5 task statements are
shorter than their practice name plus 20 characters, and the ceiling falls from
46/46 to 39/46. **[measured]** With `title = task id`: ceiling **44/46 = 0.9565**
without `alt_ids` and **46/46 = 1.0000** with them **[derived]**. Floor **1.00**.

**Files:**
- Create: `parsers/parse_nist_ssdf.py`
- Create: `tests/test_parse_nist_ssdf.py`

**Interfaces:**
- Consumes: `BaseParser`, `Control`, `alt_ids` from Task 2.
- Produces: `NistSsdfParser` with `framework_id = "nist_ssdf"`, `framework_name = "NIST SSDF"`; `NistSsdfParser.rows_to_controls(rows: list[list[str | None]]) -> list[Control]`.

- [ ] **Step 1: Re-verify the premise yourself**

Do not skip this. The previous plan's headline tests failed against its own
implementation because it assumed a shape it never checked.

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"$PY" - <<'PYEOF'
import pdfplumber, re
TASK = re.compile(r'^(P[OSW]|RV)\.\d+\.\d+:')
found = []
with pdfplumber.open('data/raw/frameworks/nist_ssdf/nist_sp_800_218.pdf') as pdf:
    for page in range(13, 28):
        for table in pdf.pages[page].extract_tables():
            if max(len(r) for r in table) < 4:
                continue
            for row in table:
                for index, cell in enumerate(row):
                    if cell and TASK.match(cell.strip()):
                        found.append((index, re.sub(r'\s+', ' ', cell.strip())))
print('task cells:', len(found))
print('column indexes:', sorted({i for i, _ in found}))
print('cells ending mid-sentence:',
      sum(1 for _, t in found if not t.rstrip().endswith(('.', ':'))))
print('stub rows:', [t.split(':')[0] for _, t in found
                     if t.split(': ', 1)[1].lower().startswith('moved to')])
PYEOF
```

Expected: `task cells: 47`, `column indexes: [3]`,
`cells ending mid-sentence: 0`, and five stub ids. **[measured]** If any of
these differs, stop and re-measure before writing the parser.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_parse_nist_ssdf.py — create

"""The SSDF table is ruled: extract_tables returns whole task cells.

Measured against the pinned PDF: 47 task cells, all at column index 3, none
ending mid-sentence. The continuation rows below a task repeat wrapped
fragments of the practice cell in column 1 and hold nothing in column 3, so the
practice column needs a forward fill and nothing needs a merge.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parsers.parse_nist_ssdf import NistSsdfParser

PRACTICE = (
    "Define Security Requirements for Software\nDevelopment (PO.1): Ensure "
    "that security requirements for software development are known at all "
    "times."
)

ROWS: list[list[str | None]] = [
    [None, "Practices", None, "Tasks", None, None, "Notional Implementation "
     "Examples", None, None, None, "References", None],
    [PRACTICE, "Define Security Requirements for Software", None,
     "PO.1.1: Identify and document all security requirements for the "
     "organization's software development infrastructures and processes, and "
     "maintain the requirements over time.",
     None, None, "Example 1: Define policies for securing software "
     "development infrastructures.", None, None, None,
     "BSAFSS: SM.3, DE.1", None],
    [None, "Development (PO.1): Ensure that security", None, None, None, None,
     None, None, None, None, "BSIMM: CP1.1, CP1.3", None],
    [None, None, None,
     "PO.1.2: Identify and document all security requirements for "
     "organization-developed software to meet, and maintain the requirements "
     "over time.",
     None, None, "Example 1: Define a set of secure coding standards.",
     None, None, None, "EO14028: 4e(ix)", None],
    [None, None, None, "PW.3.2: Moved to PW.4.4", None, None, None, None,
     None, None, None, None],
]


class TestRowsToControls:
    def test_only_task_cells_become_controls(self) -> None:
        controls = NistSsdfParser.rows_to_controls(ROWS)
        assert [c.control_id for c in controls] == ["PO.1.1", "PO.1.2"]

    def test_the_whole_task_statement_survives(self) -> None:
        first = NistSsdfParser.rows_to_controls(ROWS)[0]
        assert first.description.endswith("maintain the requirements over time.")
        assert "Identify and document all security requirements" in first.description

    def test_title_is_the_task_id_not_the_statement(self) -> None:
        first = NistSsdfParser.rows_to_controls(ROWS)[0]
        assert first.title == "PO.1.1"
        assert first.description != first.title

    def test_the_practice_is_forward_filled_onto_later_tasks(self) -> None:
        controls = NistSsdfParser.rows_to_controls(ROWS)
        assert controls[0].parent_id == "PO.1"
        assert controls[1].parent_id == "PO.1"
        assert controls[1].parent_name.startswith(
            "Define Security Requirements for Software Development"
        )

    def test_a_moved_to_stub_is_not_emitted(self) -> None:
        controls = NistSsdfParser.rows_to_controls(ROWS)
        assert "PW.3.2" not in {c.control_id for c in controls}

    def test_notional_examples_are_kept_out_of_the_statement(self) -> None:
        first = NistSsdfParser.rows_to_controls(ROWS)[0]
        assert "Example 1" not in first.description
        assert first.full_text is not None
        assert "Example 1" in first.full_text

    def test_declared_alternate_ids_reach_their_task(self) -> None:
        controls = NistSsdfParser.rows_to_controls(ROWS)
        first = NistSsdfParser.rows_to_controls([
            *ROWS,
            [None, None, None,
             "PS.1.1: Store all forms of code - including source code, "
             "executable code, and configuration-as-code - based on the "
             "principle of least privilege.",
             None, None, None, None, None, None, None, None],
        ])
        target = next(c for c in first if c.control_id == "PS.1.1")
        assert target.metadata is not None
        assert any(
            "configuration-as-code" in alt
            for alt in target.metadata["alt_ids"]
        )
        assert controls  # the base rows still parse


class TestMalformedIdMap:
    def test_an_alternate_whose_target_is_absent_is_refused(self) -> None:
        with pytest.raises(ValueError, match="names task"):
            NistSsdfParser.rows_to_controls(ROWS, require_alternate_targets=True)


class TestRun:
    def test_run_writes_from_the_real_pdf(self, tmp_path: Path) -> None:
        parser = NistSsdfParser(output_dir=tmp_path)
        try:
            output = parser.run()
        except FileNotFoundError:
            pytest.skip("data/raw is gitignored and absent in this checkout")
        assert len(output.controls) == 42
        assert [s.path for s in output.source_files] == ["nist_sp_800_218.pdf"]
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
pytest tests/test_parse_nist_ssdf.py -q
```

Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 4: Write the parser**

```python
# parsers/parse_nist_ssdf.py — create

"""Parser for NIST SP 800-218, the Secure Software Development Framework.

The tasks live in one ruled table spanning pages 14 through 27 of the PDF
(0-indexed 13 through 26). Measured against the pinned bytes,
pdfplumber.extract_tables() returns 47 task cells, all at column index 3, all
whole: the task statement arrives complete with its own newlines and none of
them ends mid-sentence. The rows below a task repeat wrapped fragments of the
PRACTICE cell in column 1 and hold nothing in column 3. So the practice column
needs a forward fill and nothing needs a rowspan merge. extract_text() is the
call that interleaves this table's columns; extract_tables() does not.

Five task cells are redirects of the form "PW.3.1: Moved to PO.1.3". None is
targeted by a curated link. They are recorded as `redirects` metadata on the
framework's first control and excluded from the emitted set: a 15-character
statement is not a control, and emitting one would put a non-statement anchor
in the corpus and drag the prose floor for no join.

The title is the task ID and this is not cosmetic. OpenCRE sets section_name
to the task statement verbatim for 36 of the 46 curated links, so a parser that
used the statement as its title would make description equal title, which is
exactly the case ProseIndex refuses to index. Using the parent practice name
instead costs five links, because five statements are shorter than their
practice name plus PROSE_MIN_EXTRA_CHARS.

Two curated links carry a mid-sentence text fragment in section_id instead of a
task id. Both fragments appear verbatim inside a task statement, so they are
declared here as alternate ids and resolved through the alt_ids channel. They
are declared rather than derived: a substring search that ran at parse time
would silently re-attach a link to a different task after a source refresh.
"""
from __future__ import annotations

import hashlib
import logging
import re
from io import BytesIO
from typing import ClassVar, Final

import pdfplumber

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SOURCE_FILE: Final[str] = "nist_sp_800_218.pdf"
SOURCE_SHA256: Final[str] = (
    "617746e553a9e2da49bfbd4eef0dfc3094758a39b869314e4173ac36605cde22"
)
# 0-indexed. The table starts on printed page 14 and ends on 28.
TABLE_PAGES: Final[range] = range(13, 28)

PRACTICE_COLUMN: Final[int] = 0
TASK_COLUMN: Final[int] = 3
EXAMPLES_COLUMN: Final[int] = 6

TASK_ID: Final[re.Pattern[str]] = re.compile(r"^((?:P[OSW]|RV)\.\d+\.\d+):\s*(.+)$")
PRACTICE_ID: Final[re.Pattern[str]] = re.compile(
    r"^(.+?)\s*\(((?:P[OSW]|RV)\.\d+)\)\s*:\s*(.*)$", re.S
)
REDIRECT: Final[re.Pattern[str]] = re.compile(r"^moved to\b", re.IGNORECASE)

# OpenCRE section_id values that are a sentence fragment rather than a task id,
# mapped to the task whose statement contains that fragment verbatim.
# Hand-verified against the pinned PDF; never derived at parse time.
MALFORMED_SECTION_IDS: Final[dict[str, str]] = {
    "code, executable code, and configuration-as-code – based on the principle "
    "of least privilege so that only authorized personnel, tools, services, "
    "etc. have access.": "PS.1.1",
    "should be performed to find vulnerabilities not identified by previous "
    "reviews, analysis, or testing and, if so, which types of testing should "
    "be performed.": "PW.8.1",
}

_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


class NistSsdfParser(BaseParser):
    framework_id: ClassVar[str] = "nist_ssdf"
    framework_name: ClassVar[str] = "NIST SSDF"
    version: ClassVar[str] = "1.1"
    source_url: ClassVar[str] = (
        "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-218.pdf"
    )
    mapping_unit_level: ClassVar[str] = "task"
    # 47 task cells minus 5 "Moved to" redirects. [measured]
    expected_count: ClassVar[int] = 42
    fetched_date: ClassVar[str] = "2026-08-15"
    # 41 of 42 statements clear the 60-character bar; the shortest real task
    # statement is 54 characters. [measured]
    min_prose_fraction: ClassVar[float] = 0.97
    expected_sha256: ClassVar[str | None] = SOURCE_SHA256

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(SOURCE_FILE)
        self._check_digest(payload)
        rows = self._read_rows(payload)
        controls = self.rows_to_controls(rows, require_alternate_targets=True)
        logger.info(
            "%s: %d tasks across %d practices",
            self.framework_id, len(controls),
            len({c.parent_id for c in controls}),
        )
        return controls

    def _check_digest(self, payload: bytes) -> None:
        """Refuse a PDF that is not the pinned one.

        Raises:
            ValueError: If the digest does not match `expected_sha256`.
        """
        if self.expected_sha256 is None:
            return
        actual = hashlib.sha256(payload).hexdigest()
        if actual != self.expected_sha256:
            raise ValueError(
                f"{self.framework_id}: {SOURCE_FILE} has sha256 {actual}, not "
                f"the pinned {self.expected_sha256}. Both entries in "
                f"MALFORMED_SECTION_IDS quote this document's text verbatim, "
                f"so a changed source can attach a link to the wrong task."
            )

    def _read_rows(self, payload: bytes) -> list[list[str | None]]:
        """Every row of the widest table on each table page.

        Raises:
            ValueError: If no page yields a table at least four columns wide.
        """
        rows: list[list[str | None]] = []
        with pdfplumber.open(BytesIO(payload)) as pdf:
            for page_number in TABLE_PAGES:
                if page_number >= len(pdf.pages):
                    break
                for table in pdf.pages[page_number].extract_tables():
                    if max(len(row) for row in table) < 4:
                        continue
                    rows.extend(table)
        if not rows:
            raise ValueError(
                f"{self.framework_id}: no table of four or more columns on "
                f"pages {TABLE_PAGES.start}-{TABLE_PAGES.stop} of "
                f"{SOURCE_FILE}. extract_text() interleaves this table's "
                f"columns, so falling back to it would ship task statements "
                f"truncated by the adjacent Examples column."
            )
        return rows

    @classmethod
    def rows_to_controls(
        cls,
        rows: list[list[str | None]],
        require_alternate_targets: bool = False,
    ) -> list[Control]:
        """One Control per real task, practice forward-filled.

        Raises:
            ValueError: If require_alternate_targets is set and a declared
                malformed-id target is absent from the parsed tasks.
        """
        practice_id = ""
        practice_name = ""
        controls: list[Control] = []
        redirects: dict[str, str] = {}

        for row in rows:
            cells = [
                _WHITESPACE.sub(" ", str(cell).strip()) if cell else ""
                for cell in (list(row) + [None] * 12)[:12]
            ]
            practice = PRACTICE_ID.match(cells[PRACTICE_COLUMN])
            if practice is not None:
                practice_name = practice.group(1).strip()
                practice_id = practice.group(2)

            task = TASK_ID.match(cells[TASK_COLUMN])
            if task is None:
                continue
            task_id, statement = task.group(1), task.group(2).strip()
            if REDIRECT.match(statement):
                redirects[task_id] = statement
                continue

            alternates = [
                fragment for fragment, target in MALFORMED_SECTION_IDS.items()
                if target == task_id
            ]
            metadata: dict[str, str | list[str]] = {"practice": practice_id}
            if alternates:
                metadata["alt_ids"] = alternates
            controls.append(Control(
                control_id=task_id,
                title=task_id,
                description=statement,
                full_text=cls._full_text(statement, cells[EXAMPLES_COLUMN]),
                hierarchy_level="task",
                parent_id=practice_id or None,
                parent_name=practice_name or None,
                metadata=metadata,
            ))

        cls._check_alternate_targets(
            {c.control_id for c in controls}, require_alternate_targets,
        )
        if redirects and controls:
            cls._record_redirects(controls[0], redirects)
        logger.info("nist_ssdf: %d redirect stub(s) excluded: %s",
                    len(redirects), sorted(redirects))
        return controls

    @staticmethod
    def _full_text(statement: str, examples: str) -> str | None:
        """The statement with its notional implementation examples appended.

        Returned as full_text rather than folded into description on purpose,
        and only when it adds something: ProseIndex prefers full_text, so this
        IS the anchor. The examples say how an organisation might satisfy the
        task, which is remediation guidance; putting it in front of the encoder
        pulls the anchor toward tasks that share tooling rather than meaning.
        It is kept because a reviewer needs it and because full_text is not
        what the corpus report measures the join on.
        """
        if not examples:
            return None
        return f"{statement}\n\n{examples}"

    @staticmethod
    def _record_redirects(control: Control, redirects: dict[str, str]) -> None:
        """Attach the retired task numbering to the framework's first control.

        Recorded rather than dropped silently. A reader who finds PW.3.2 cited
        in an older mapping needs somewhere in this artifact that says where it
        went.
        """
        metadata = dict(control.metadata or {})
        metadata["retired_tasks"] = [
            f"{task}: {target}" for task, target in sorted(redirects.items())
        ]
        control.metadata = metadata

    @staticmethod
    def _check_alternate_targets(task_ids: set[str], required: bool) -> None:
        """Refuse a declared malformed-id map that no longer matches the source.

        Raises:
            ValueError: If a declared target task is absent.
        """
        if not required:
            return
        missing = sorted({
            target for target in MALFORMED_SECTION_IDS.values()
            if target not in task_ids
        })
        if missing:
            raise ValueError(
                f"nist_ssdf: MALFORMED_SECTION_IDS names task(s) {missing} "
                f"that this parse did not produce. Two curated links carry a "
                f"sentence fragment where a task id belongs and reach their "
                f"task only through this map; a stale entry leaves them "
                f"unresolved while every other gate stays green."
            )


def main() -> None:
    NistSsdfParser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests and typecheck**

```bash
pytest tests/test_parse_nist_ssdf.py -q
mypy parsers/parse_nist_ssdf.py --strict
```

- [ ] **Step 6: Run against the real source and check the join**

```bash
PYTHONPATH=. "$PY" parsers/parse_nist_ssdf.py
PYTHONPATH=. "$PY" parsers/merge_all_controls.py
PYTHONPATH=. "$PY" scripts/corpus_report.py | grep -E "^(nist_ssdf|framework)"
```

Expected `nist_ssdf: 42 tasks across 19 practices` and
`5 redirect stub(s) excluded` **[measured]**, then:

```
nist_ssdf                     46     0   46     0      44  1.05     0    0     0     0 1.0000
```

Accept only if `resolution_rate == 1.0000` (44 by task id plus 2 through
`alt_ids`), `distinct_anchors >= 44`, `by_title == 0`. If `by_title` is not 0,
the parser is emitting task statements as titles and `_is_prose` will start
excluding controls; stop and check `title`. If `resolution_rate` is 0.9565, the
`alt_ids` channel from Task 2 is not reaching this parser's metadata.

- [ ] **Step 7: Commit**

```bash
git add parsers/parse_nist_ssdf.py tests/test_parse_nist_ssdf.py \
        data/processed/frameworks/nist_ssdf.json data/processed/all_controls.json
git commit -m "feat: read the SSDF task table as ruled cells, with the two malformed ids declared"
```

---

### Task 10: NIST SP 800-63B — the version blocker is already resolved

**The structures document's CRITICAL FINDING is stale.** It reports the fetched
file as SP 800-63-4, whose restructured numbering matches none of OpenCRE's 79
links. The fetch was corrected before this plan was written. The file on disk
now has `<title>NIST Special Publication 800-63B</title>`, contains
`Memorized Secret` 17 times, and **all 25 distinct curated `section_id` values
appear in it**. **[measured]** Parsing the headings gives **118 numbered
sections, all distinct**, and 24 of the 25 link ids are one of them.
**[measured]**

**Ceiling: 78/79 = 0.98734** **[derived]**, floor **0.98**. The single miss is a
curated link whose `section_id` is the two-character fragment `are g`, an
upstream extraction artifact with no recoverable target. Unlike the SSDF
fragments it does not appear inside any section's text, so no `alt_ids` entry
can honestly be written for it and none is.

**No digest gate is possible here and that is deliberate.** `pages.nist.gov`
sits behind Cloudflare, which injects a fresh nonce into every response, so
`scripts/fetch_frameworks.py` leaves `expected_sha256` at `None` for this
source. **[measured]** The parser gates on structure instead: it refuses to
write unless it finds at least 100 numbered sections and unless every id in
`REQUIRED_SECTION_IDS` is present. That list is the 25 distinct curated link ids
minus `are g`, so the gate fires on exactly the failure a revision swap would
cause.

**All 79 links contribute zero training links today**, for the same two reasons
as Task 6: `nist_800_63` is in `PHASE1B_DROPPED_FRAMEWORKS`, and every one of
its 79 `section_name` values is the section number, 3 to 7 characters, under
`PHASE1B_MIN_SECTION_TEXT_LENGTH`. **[measured]** Task 14 is what changes that.

**Files:**
- Create: `parsers/parse_nist_800_63.py`
- Create: `tests/test_parse_nist_800_63.py`

**Interfaces:**
- Consumes: `BaseParser`, `Control`, `bs4.BeautifulSoup`.
- Produces: `Nist80063Parser` with `framework_id = "nist_800_63"`, `framework_name = "NIST 800-63"`; `Nist80063Parser.sections_from_html(html: str) -> list[Control]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_nist_800_63.py — create

"""SP 800-63B numbers its headings, and the section number is the join key.

Measured on the staged file: 118 numbered headings, all distinct, covering 24
of the 25 distinct curated section_ids. The 25th is the fragment 'are g', an
upstream artifact that appears in no section and is left unresolved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parsers.parse_nist_800_63 import Nist80063Parser

HTML = """<html><head><title>NIST Special Publication 800-63B</title></head>
<body>
<h2 id="sec5">5 Authenticator and Verifier Requirements</h2>
<p>This section is normative.</p>
<h3 id="reqauthtype">5.1 Requirements by Authenticator Type</h3>
<p>Authenticators are described by type.</p>
<h4 id="memsecret">5.1.1 Memorized Secrets</h4>
<p>A Memorized Secret authenticator, commonly referred to as a password or PIN,
is a secret value intended to be chosen and memorized by the user.</p>
<h5 id="memsecretver">5.1.1.2 Memorized Secret Verifiers</h5>
<p>Verifiers SHALL require subscriber-chosen memorized secrets to be at least 8
characters in length, and SHALL permit at least 64 characters.</p>
<h2 id="glossary">Appendix A Definitions and Abbreviations</h2>
<p>Not numbered in the dotted scheme.</p>
</body></html>
"""


class TestSectionsFromHtml:
    def test_the_section_number_is_the_control_id(self) -> None:
        controls = Nist80063Parser.sections_from_html(HTML)
        assert [c.control_id for c in controls] == [
            "5", "5.1", "5.1.1", "5.1.1.2",
        ]

    def test_title_is_the_heading_text_without_the_number(self) -> None:
        controls = Nist80063Parser.sections_from_html(HTML)
        assert controls[3].title == "Memorized Secret Verifiers"

    def test_body_stops_at_the_next_heading(self) -> None:
        controls = Nist80063Parser.sections_from_html(HTML)
        assert controls[2].description.startswith("A Memorized Secret")
        assert "at least 8" not in controls[2].description

    def test_an_unnumbered_appendix_heading_is_not_a_section(self) -> None:
        assert "Appendix A" not in {
            c.title for c in Nist80063Parser.sections_from_html(HTML)
        }


class TestStructureGate:
    def test_a_document_missing_a_required_section_is_refused(
        self, tmp_path: Path,
    ) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "sp800_63b.html").write_text(HTML, encoding="utf-8")
        parser = Nist80063Parser(raw_dir=raw, output_dir=tmp_path / "out")
        parser.min_sections = 2
        parser.required_section_ids = ("5.1.1.2", "9.9.9")
        with pytest.raises(ValueError, match=r"9\.9\.9"):
            parser.parse()

    def test_a_thin_document_is_refused(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw2"
        raw.mkdir()
        (raw / "sp800_63b.html").write_text(HTML, encoding="utf-8")
        parser = Nist80063Parser(raw_dir=raw, output_dir=tmp_path / "out2")
        parser.required_section_ids = ()
        with pytest.raises(ValueError, match="numbered section"):
            parser.parse()


class TestRun:
    def test_run_writes_from_the_real_document(self, tmp_path: Path) -> None:
        parser = Nist80063Parser(output_dir=tmp_path)
        try:
            output = parser.run()
        except FileNotFoundError:
            pytest.skip("data/raw is gitignored and absent in this checkout")
        assert len(output.controls) >= 100
        assert [s.path for s in output.source_files] == ["sp800_63b.html"]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_parse_nist_800_63.py -q
```

Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Write the parser**

```python
# parsers/parse_nist_800_63.py — create

"""Parser for NIST SP 800-63B, Authentication and Lifecycle Management.

The revision matters more than the parsing does. OpenCRE's 79 links carry
800-63-3B section numbers (5.1.1.2, 6.1.2.3, A.3). Revision 4 restructured the
document and uses slug ids with a single-integer chapter attribute, so none of
those numbers exists in it and a fetch of revision 4 would leave every link
unjoinable while looking like a successful fetch. The staged file is revision
3B: its title element says so, and all 25 distinct curated section_ids appear
in it.

There is no digest gate. pages.nist.gov sits behind Cloudflare, which injects a
per-response nonce into the body, so two fetches of the identical document have
different hashes and scripts/fetch_frameworks.py deliberately leaves
expected_sha256 unset. The gate here is structural instead: a minimum count of
numbered sections, and the presence of every id in REQUIRED_SECTION_IDS. That
list is the curated link ids, so a revision swap fails this parser rather than
producing a corpus whose join silently went to zero.

The mapping unit is the numbered section. Its id is the dotted number in the
heading text, its title is the rest of the heading, and its statement is
everything between that heading and the next one at any level.
"""
from __future__ import annotations

import logging
import re
from typing import ClassVar, Final

from bs4 import BeautifulSoup, Tag

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SOURCE_FILE: Final[str] = "sp800_63b.html"

HEADING_TAG: Final[re.Pattern[str]] = re.compile(r"^h[1-6]$")
# "5.1.1.2 Memorized Secret Verifiers" and "A.3 Appendix Section". Both forms
# appear in the curated section_ids.
NUMBERED: Final[re.Pattern[str]] = re.compile(
    r"^\s*((?:A\.)?\d+(?:\.\d+)*)\.?\s+(\S.*)$"
)

# Every distinct curated section_id except the fragment "are g", which is an
# upstream extraction artifact and appears in no section of this document.
# Hand-transcribed from data/training/hub_links_curated.jsonl and checked
# against the staged file. [measured]
REQUIRED_SECTION_IDS: Final[tuple[str, ...]] = (
    "5.1.1.1", "5.1.1.2", "5.1.2.2", "5.1.3.2", "5.1.4.2", "5.1.5.2",
    "5.1.7.2", "5.2.1", "5.2.2", "5.2.3", "5.2.5", "5.2.6", "5.2.8", "5.2.9",
    "5.2.10", "6.1.2.3", "6.1.3", "6.1.4", "7.1", "7.1.1", "7.1.2", "7.2",
    "7.2.1", "A.3",
)
# Well under the 118 measured, so ordinary editorial change does not trip it
# while a revision swap, which takes the dotted numbering to zero, does.
MIN_NUMBERED_SECTIONS: Final[int] = 100

_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


class Nist80063Parser(BaseParser):
    framework_id: ClassVar[str] = "nist_800_63"
    framework_name: ClassVar[str] = "NIST 800-63"
    version: ClassVar[str] = "800-63B rev 3"
    source_url: ClassVar[str] = "https://pages.nist.gov/800-63-3/sp800-63b.html"
    mapping_unit_level: ClassVar[str] = "section"
    # 118 numbered headings in the staged document. [measured] Declared as a
    # floor: this document is not digest-pinned, so an editorial revision may
    # add sections without changing what any link resolves to.
    expected_count: ClassVar[int] = 100
    expected_count_is_floor: ClassVar[bool] = True
    fetched_date: ClassVar[str] = "2026-08-15"
    # 113 of 118 numbered sections carry a body longer than their heading;
    # the five that do not are chapter headings whose text is entirely in
    # their subsections. [measured]
    min_prose_fraction: ClassVar[float] = 0.94
    # Instance-overridable so a fixture-backed test declares its own structural
    # gate rather than widening the real one.
    required_section_ids: ClassVar[tuple[str, ...]] = REQUIRED_SECTION_IDS
    min_sections: ClassVar[int] = MIN_NUMBERED_SECTIONS

    def parse(self) -> list[Control]:
        html = self.read_source(SOURCE_FILE)
        controls = self.sections_from_html(html)
        self._check_structure(controls)
        logger.info(
            "%s: %d numbered sections, %d of the %d required ids present",
            self.framework_id, len(controls),
            len({c.control_id for c in controls} & set(self.required_section_ids)),
            len(self.required_section_ids),
        )
        return controls

    def _check_structure(self, controls: list[Control]) -> None:
        """Refuse a document that is not the revision the links key to.

        Raises:
            ValueError: If the numbering is too sparse, or a required section
                id is absent.
        """
        if len(controls) < self.min_sections:
            raise ValueError(
                f"{self.framework_id}: {SOURCE_FILE} yields {len(controls)} "
                f"numbered sections, under the floor of {self.min_sections}. "
                f"Revision 4 restructured this document to slug ids with no "
                f"dotted numbering, and fetching it would leave all 79 "
                f"curated links unjoinable while looking like a success. This "
                f"source is not digest-pinned because Cloudflare injects a "
                f"per-response nonce, so this count is the pin."
            )
        found = {c.control_id for c in controls}
        missing = sorted(set(self.required_section_ids) - found)
        if missing:
            raise ValueError(
                f"{self.framework_id}: section id(s) {missing} are absent from "
                f"{SOURCE_FILE}. Every one of them is targeted by a curated "
                f"OpenCRE link, so their absence means this is not the "
                f"revision the links were written against."
            )

    @classmethod
    def sections_from_html(cls, html: str) -> list[Control]:
        """One Control per numbered heading, in document order.

        Raises:
            ValueError: If two headings claim the same section number, which
                would let one section's text silently replace another's.
        """
        soup = BeautifulSoup(html, "lxml")
        controls: list[Control] = []
        seen: set[str] = set()
        for heading in soup.find_all(HEADING_TAG):
            match = NUMBERED.match(
                _WHITESPACE.sub(" ", heading.get_text()).strip()
            )
            if match is None:
                continue
            number, title = match.group(1), match.group(2).strip()
            if number in seen:
                raise ValueError(
                    f"nist_800_63: section number {number!r} appears on more "
                    f"than one heading. Emitting both would give one "
                    f"control_id two statements and let whichever is written "
                    f"last answer every link to it."
                )
            seen.add(number)
            controls.append(Control(
                control_id=number,
                title=title,
                description=cls._body(heading) or title,
                hierarchy_level="section",
                parent_id=number.rsplit(".", 1)[0] if "." in number else None,
            ))
        return controls

    @staticmethod
    def _body(heading: Tag) -> str:
        """Text between this heading and the next heading at any level."""
        parts: list[str] = []
        for sibling in heading.next_siblings:
            name = getattr(sibling, "name", None)
            if name and HEADING_TAG.match(name):
                break
            text = (
                sibling.get_text(" ") if hasattr(sibling, "get_text")
                else str(sibling)
            )
            cleaned = _WHITESPACE.sub(" ", text).strip()
            if cleaned:
                parts.append(cleaned)
        return " ".join(parts).strip()


def main() -> None:
    Nist80063Parser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Reconcile REQUIRED_SECTION_IDS against the curated link file**

The list above is transcribed. Confirm it before running anything, because a
wrong entry turns the structural gate into a guaranteed failure or a silent
pass.

```bash
"$PY" - <<'PYEOF'
import json
ids = set()
with open('data/training/hub_links_curated.jsonl', encoding='utf-8') as handle:
    for line in handle:
        row = json.loads(line)
        if row['framework_id'] == 'nist_800_63':
            ids.add(row['section_id'])
print(len(ids), sorted(ids))
PYEOF
```

Expected: 25 ids. Every one except `are g` must appear in
`REQUIRED_SECTION_IDS`; if the printed set differs from the literal above,
replace the literal with the printed set minus `are g` and say so in the commit
message.

- [ ] **Step 5: Run the tests and typecheck**

```bash
pytest tests/test_parse_nist_800_63.py -q
mypy parsers/parse_nist_800_63.py --strict
```

- [ ] **Step 6: Run against the real source and check the join**

```bash
PYTHONPATH=. "$PY" parsers/parse_nist_800_63.py
PYTHONPATH=. "$PY" parsers/merge_all_controls.py
PYTHONPATH=. "$PY" scripts/corpus_report.py | grep -E "^(nist_800_63|framework)"
```

Expected:

```
nist_800_63                   79     0   78     1      25  3.12     0    5     0     0 0.9873
```

Accept only if `resolution_rate >= 0.98`, `distinct_anchors == 25`,
`unresolved == 1`. `links_per_anchor` of 3.12 is the source's shape: 79 links
over 25 sections. `dropped_by_prose_rule` of about 5 is the chapter headings
whose text lives entirely in their subsections; none of them is linked.

- [ ] **Step 7: Commit**

```bash
git add parsers/parse_nist_800_63.py tests/test_parse_nist_800_63.py \
        data/processed/frameworks/nist_800_63.json data/processed/all_controls.json
git commit -m "feat: parse SP 800-63B rev 3 sections, gating on the numbering the links use"
```

---
### Task 11: ENISA — three tables, one name space, and no stable id anywhere

The source defines no control identifier at all, so OpenCRE's extraction
degraded: 40 of the 68 curated links carry the literal placeholder `Table 5:`
in `section_id` and 18 carry `Table 3:`. **[measured]** The join must key on
`section_name`, which holds 33 distinct values.

**Twenty of the 68 links point at Table 3, not Table 5.** `Poisoning`,
`Evasion`, `Model disclosure`, `Data disclosure`, `Oracle`,
`Label modification`, `Compromise of ML application components`,
`Model or data disclosure`, `Denial of service due to inconsistent data or a
sponge example`, and `Use of adversarial examples crafted in white or grey box
conditions (e.g. FGSM...)`. **[measured]** Table 3 is the threat taxonomy, and
a parser that emitted only the 37 security controls would leave 29% of this
framework's links unresolved. All 13 Table 3 entries are emitted as mapping
units alongside the controls.

**Extraction, measured rather than assumed.** `pdfplumber.extract_tables()`
returns several tables per page and the one that matters is the widest;
on Table 5's pages it is 34 or 35 columns. **[measured]** The definition text
lands in **column 2 on some rows and column 3 on others**, which is why a
per-page "densest column" heuristic loses rows: it picks one column for a page
that is not column-uniform. The rule here is per-row instead — the name is
column 0 (columns 0 and 1 for Table 3, which has a threat column and a
sub-threat column), the definition is the join of columns 1 through 4 with any
lone lifecycle `x` dropped, and a row with an empty name is a continuation
appended to the previous unit. Under that rule **0 of 35 Table 5 rows and 0 of
13 Table 3 rows extract with an empty definition** **[measured]**, against the
4 empty Table 3 definitions the premortem found under the previous rule,
including `Evasion` and `Poisoning`, the two most-linked entries in the
framework.

**Three name-matching defects, each measured.**

| defect | rows lost | example |
|---|---|---|
| footnote digits fused onto the name | 6 | `Apply modifications on inputs17` |
| curly punctuation against OpenCRE's ASCII | 2 | `third parties’ security requirements` |
| ellipsis character against three periods | 1 | `(e.g. FGSM…)` vs `(e.g. FGSM...)` |

**[measured]** Naive exact matching over Table 5 and Table 3 resolves
**51/68**. Adding NFKD normalisation and footnote-digit removal takes it to
**62/68**. **[measured]**

**The last 6 rows come from Annex C.** `Ensure reliable sources are used` (3
links) and `Use methods to clean the training dataset from suspicious samples`
(3 links) appear as row names in Annex C and nowhere in Table 5. **[measured]**
Emitting them from Annex C gives 35 + 2 = **37 controls, which is the count the
source states in its own text**, plus 13 threats, **50 mapping units**, and a
ceiling of **68/68 = 1.0000** **[derived]**. Floor **1.00**.

Annex C also spells six Table 5 names differently (`least privilege` for
`least privileged`, `minimise` for `minimize`, and four more). Those six are
registered as `alt_titles` on their Table 5 control rather than emitted as
separate controls: emitting them would put six near-duplicate anchors into a
33-anchor framework, which is the collapse the instrument exists to catch.
Annex C's implementation examples are **not** used as statement text — they say
how to satisfy a control, and for the two Annex-C-only controls they are the
only text there is.

**Anchor count does not improve and that is the honest outcome.** 68 links land
on 33 title anchors today and on 33 prose anchors after. **[measured]** The
gain is that the 33 anchors become paragraphs instead of phrases.

**Files:**
- Create: `parsers/parse_enisa.py`
- Create: `tests/test_parse_enisa.py`

**Interfaces:**
- Consumes: `BaseParser`, `Control`.
- Produces: `EnisaParser` with `framework_id = "enisa"`, `framework_name = "ENISA"`; `EnisaParser.rows_to_units(rows: list[list[str | None]], name_columns: int) -> list[tuple[str, str]]`; `EnisaParser.normalise_name(name: str) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_enisa.py — create

"""ENISA has no control id, so the join is the name, and the name is damaged.

Measured on the pinned PDF: 6 curated links lose to footnote digits fused onto
a control name, 2 to a curly apostrophe against OpenCRE's ASCII, and 1 to an
ellipsis character against three periods. The definition also lands in column 2
on some rows and column 3 on others, which is why the merge is per row.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parsers.parse_enisa import EnisaParser

# Column 0 name, definition in column 2 on the first unit and column 3 on the
# second, lifecycle x marks from column 5. Continuation rows carry no name.
TABLE5: list[list[str | None]] = [
    ["Security controls", "", "Definition", "", "", "Stages of the lifecycle"],
    ["", "", "", "", "", ""],
    ["", "ORGANISATIONAL", "", "", "", ""],
    ["Apply modifications on inputs17", "",
     "Modify the model inputs so that an adversarial perturbation loses its", "",
     "", "x"],
    ["", "", "effect before the input reaches the model.", "", "", ""],
    ["Ensure ML applications comply with third parties’ security requirements",
     "", "", "Third-party components used by an ML application must meet the", "",
     "x"],
    ["", "", "", "same security requirements as first-party components.", "", ""],
]

TABLE3: list[list[str | None]] = [
    ["Threats sub- threats", "", "Definition", "", "", "Stage"],
    ["Evasion", "",
     "A type of attack in which the attacker works on the ML algorithm inputs",
     "", "", "x"],
    ["", "", "to find small perturbations leading to large output errors.", "",
     "", ""],
    ["", "Data disclosure",
     "This threat refers to a leak of data manipulated by the ML application.",
     "", "", "x"],
]


class TestRowsToUnits:
    def test_a_definition_in_column_three_is_not_lost(self) -> None:
        units = dict(EnisaParser.rows_to_units(TABLE5, name_columns=1))
        key = "Ensure ML applications comply with third parties’ security requirements"
        assert "Third-party components" in units[key]
        assert "first-party components" in units[key]

    def test_continuation_rows_join_the_unit_above(self) -> None:
        units = dict(EnisaParser.rows_to_units(TABLE5, name_columns=1))
        assert units["Apply modifications on inputs17"].endswith(
            "before the input reaches the model."
        )

    def test_category_banners_are_not_units(self) -> None:
        names = [n for n, _ in EnisaParser.rows_to_units(TABLE5, name_columns=1)]
        assert "ORGANISATIONAL" not in names
        assert "Security controls" not in names

    def test_a_threat_and_a_sub_threat_are_both_units(self) -> None:
        names = [n for n, _ in EnisaParser.rows_to_units(TABLE3, name_columns=2)]
        assert names == ["Evasion", "Data disclosure"]

    def test_no_unit_extracts_with_an_empty_definition(self) -> None:
        for rows, columns in ((TABLE5, 1), (TABLE3, 2)):
            for name, body in EnisaParser.rows_to_units(rows, columns):
                assert body, name


class TestNameNormalisation:
    def test_a_fused_footnote_digit_is_removed(self) -> None:
        assert EnisaParser.normalise_name("Apply modifications on inputs17") == (
            "apply modifications on inputs"
        )

    def test_a_curly_apostrophe_matches_the_ascii_one(self) -> None:
        assert EnisaParser.normalise_name(
            "Ensure ML applications comply with third parties’ security requirements"
        ) == EnisaParser.normalise_name(
            "Ensure ML applications comply with third parties' security requirements"
        )

    def test_an_ellipsis_matches_three_periods(self) -> None:
        assert EnisaParser.normalise_name(
            "Use of adversarial examples crafted in white or grey box "
            "conditions (e.g. FGSM…)"
        ) == EnisaParser.normalise_name(
            "Use of adversarial examples crafted in white or grey box "
            "conditions (e.g. FGSM...)"
        )


class TestRun:
    def test_run_writes_from_the_real_pdf(self, tmp_path: Path) -> None:
        parser = EnisaParser(output_dir=tmp_path)
        try:
            output = parser.run()
        except FileNotFoundError:
            pytest.skip("data/raw is gitignored and absent in this checkout")
        assert len(output.controls) == 50
        levels = {c.hierarchy_level for c in output.controls}
        assert levels == {"control", "threat"}
        assert [s.path for s in output.source_files] == [
            "enisa_securing_ml_algorithms.pdf",
        ]

    def test_the_two_annex_c_only_controls_are_present(
        self, tmp_path: Path,
    ) -> None:
        parser = EnisaParser(output_dir=tmp_path)
        try:
            output = parser.run()
        except FileNotFoundError:
            pytest.skip("data/raw is gitignored and absent in this checkout")
        titles = {c.title for c in output.controls}
        assert "Ensure reliable sources are used" in titles
        assert (
            "Use methods to clean the training dataset from suspicious samples"
            in titles
        )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_parse_enisa.py -q
```

Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Write the parser**

```python
# parsers/parse_enisa.py — create

"""Parser for ENISA's Securing Machine Learning Algorithms.

There is no control identifier anywhere in this source, which is why OpenCRE's
own extraction degraded: 40 of the 68 curated links carry the literal string
"Table 5:" as their section_id and 18 carry "Table 3:". The join is therefore
the control NAME, and the name is what this parser has to get exactly right.

Three tables are read. Table 5 (pages 20-26) gives 35 security controls with
their definitions. Table 3 (pages 15-16) gives 13 threats and sub-threats, and
they are emitted too: 20 of the 68 curated links target them, including
Poisoning and Evasion, the two most-linked entries in the framework. Annex C
(pages 39-43) supplies the two controls that Table 5's extraction does not
reach, which brings the control count to the 37 the document states in its own
text.

Extraction is per row, not per page. pdfplumber puts a definition in column 2
on some rows and column 3 on others, so a per-page densest-column heuristic
loses whichever rows are in the other column. Here the name is the first
`name_columns` cells joined, the definition is columns `name_columns` through 4
joined with lone lifecycle `x` marks dropped, and a row with no name is a
continuation of the unit above it. Under that rule no unit extracts with an
empty definition.

Names are normalised before they are stored, not at lookup time, because
ProseIndex matches a link's section_name against the stored title verbatim.
Three defects are corrected: NFKD folding plus an explicit punctuation map for
curly quotes, dashes and the ellipsis; and removal of a footnote digit fused
onto the end of a name. The footnote names are DECLARED rather than found by a
blanket regex, so a name that legitimately ends in a digit cannot be damaged
and a source refresh that moves the footnotes fails this parser instead of
quietly renaming a control.

Annex C's six variant spellings of Table 5 names are registered as alt_titles
rather than emitted as controls. Emitting them would add six anchors that are
near-duplicates of six existing ones, which is the collapse the corpus report
exists to make visible. Annex C's implementation-example text is not used as
statement text for a control that has a Table 5 definition: it says how to
satisfy the control rather than what it is.
"""
from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from io import BytesIO
from typing import ClassVar, Final

import pdfplumber

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SOURCE_FILE: Final[str] = "enisa_securing_ml_algorithms.pdf"
SOURCE_SHA256: Final[str] = (
    "4de967bbdf92a01339ae449b7d305b8ff266d7f16ed0a7d92a711ca20e20f087"
)

# 0-indexed page ranges, verified against the pinned PDF.
TABLE3_PAGES: Final[range] = range(14, 16)
TABLE5_PAGES: Final[range] = range(19, 26)
ANNEX_C_PAGES: Final[range] = range(38, 43)

# Where the lifecycle-stage columns begin. Everything from name_columns to
# here is definition text.
DEFINITION_END_COLUMN: Final[int] = 5

# Row names that are banners rather than units.
TABLE5_BANNERS: Final[tuple[str, ...]] = (
    "Security controls", "ORGANISATIONAL", "TECHNICAL", "SPECIFIC TO ML",
)
TABLE3_BANNERS: Final[tuple[str, ...]] = ("Threats",)
ANNEX_C_BANNERS: Final[tuple[str, ...]] = ("Security controls",)

# Table 5 names carrying a footnote reference fused onto the last word.
# Declared, hand-verified against the pinned PDF, and checked at parse time.
# A blanket trailing-digit strip would also damage a name that legitimately
# ends in a number.
FOOTNOTE_NAMES: Final[dict[str, str]] = {
    "Include ML applications into detection and response to security incident "
    "processes15":
        "Include ML applications into detection and response to security "
        "incident processes",
    "Add some adversarial examples to the training dataset16":
        "Add some adversarial examples to the training dataset",
    "Apply modifications on inputs17":
        "Apply modifications on inputs",
    "Reduce the information given by the model19":
        "Reduce the information given by the model",
    "Use less easily transferable models20":
        "Use less easily transferable models",
}

# Annex C spellings of a Table 5 control, keyed by the Annex C name.
# Registered as alt_titles on the Table 5 control. Hand-verified.
ANNEX_C_VARIANTS: Final[dict[str, str]] = {
    "Apply a RBAC model, respecting the least privilege principle":
        "Apply a RBAC model, respecting the least privileged principle",
    "Apply documentation requirements to Artificial Intelligence projects":
        "Apply documentation requirements to AI projects",
    "Ensure appropriate protection are deployed for test environments as well":
        "Ensure appropriate protection is deployed for test environments",
    "Ensure ML applications comply with identity management, authentication "
    "and access control policies":
        "Ensure ML applications comply with identity management, "
        "authentication, and access control policies",
    "Include ML applications into asset management processes":
        "Include ML applications in asset management processes",
    "Use federated learning to minimise risk of data breaches":
        "Use federated learning to minimize risk of data breaches",
}

# Controls Table 5's extraction does not reach; their statement comes from
# Annex C's implementation column, which is the only text the source gives them
# in a table this parser can read.
ANNEX_C_ONLY: Final[tuple[str, ...]] = (
    "Ensure reliable sources are used",
    "Use methods to clean the training dataset from suspicious samples",
)

_PUNCTUATION: Final[dict[str, str]] = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...",
}
_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


class EnisaParser(BaseParser):
    framework_id: ClassVar[str] = "enisa"
    framework_name: ClassVar[str] = "ENISA"
    version: ClassVar[str] = "2021-12"
    source_url: ClassVar[str] = (
        "https://www.enisa.europa.eu/publications/securing-machine-learning-algorithms"
    )
    mapping_unit_level: ClassVar[str] = "control"
    # 35 Table 5 controls + 2 recovered from Annex C = the 37 the document
    # states, plus 13 Table 3 threats. [measured]
    expected_count: ClassVar[int] = 50
    fetched_date: ClassVar[str] = "2026-08-15"
    # Every unit has a definition of at least 49 characters and none equals its
    # own name. [measured]
    min_prose_fraction: ClassVar[float] = 0.96
    expected_sha256: ClassVar[str | None] = SOURCE_SHA256

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(SOURCE_FILE)
        self._check_digest(payload)
        with pdfplumber.open(BytesIO(payload)) as pdf:
            table5 = self._collect(pdf, TABLE5_PAGES, 1, TABLE5_BANNERS)
            table3 = self._collect(pdf, TABLE3_PAGES, 2, TABLE3_BANNERS)
            annex_c = self._collect(pdf, ANNEX_C_PAGES, 1, ANNEX_C_BANNERS)

        self._check_declarations(table5, annex_c)
        controls = self._build(table5, table3, annex_c)
        logger.info(
            "%s: %d controls and %d threats, %d Annex C alternate spelling(s)",
            self.framework_id,
            sum(1 for c in controls if c.hierarchy_level == "control"),
            sum(1 for c in controls if c.hierarchy_level == "threat"),
            len(ANNEX_C_VARIANTS),
        )
        return controls

    def _check_digest(self, payload: bytes) -> None:
        """Refuse a PDF that is not the pinned one.

        Raises:
            ValueError: If the digest does not match `expected_sha256`.
        """
        if self.expected_sha256 is None:
            return
        actual = hashlib.sha256(payload).hexdigest()
        if actual != self.expected_sha256:
            raise ValueError(
                f"{self.framework_id}: {SOURCE_FILE} has sha256 {actual}, not "
                f"the pinned {self.expected_sha256}. FOOTNOTE_NAMES, "
                f"ANNEX_C_VARIANTS, ANNEX_C_ONLY and the three page ranges all "
                f"quote this exact document."
            )

    def _collect(
        self, pdf: pdfplumber.PDF, pages: range, name_columns: int,
        banners: tuple[str, ...],
    ) -> list[tuple[str, str]]:
        """(name, definition) for one table, across its pages."""
        rows: list[list[str | None]] = []
        for page_number in pages:
            if page_number >= len(pdf.pages):
                break
            tables = pdf.pages[page_number].extract_tables()
            if not tables:
                continue
            widest = max(tables, key=lambda t: max(len(r) for r in t))
            if max(len(row) for row in widest) < DEFINITION_END_COLUMN:
                continue
            rows.extend(widest)
        units = self.rows_to_units(rows, name_columns, banners)
        if not units:
            raise ValueError(
                f"{self.framework_id}: no table rows on pages "
                f"{pages.start}-{pages.stop} of {SOURCE_FILE}. "
                f"extract_text() interleaves these tables and returns rotated "
                f"headers as reversed character runs, so falling back to it "
                f"would produce garbage rather than a smaller result."
            )
        return units

    @classmethod
    def rows_to_units(
        cls,
        rows: list[list[str | None]],
        name_columns: int,
        banners: tuple[str, ...] = (),
    ) -> list[tuple[str, str]]:
        """(name, definition) per named row, continuations merged upward.

        Merging is per row rather than per page because pdfplumber places a
        definition in column 2 on some rows and column 3 on others.
        """
        units: list[tuple[str, list[str]]] = []
        for row in rows:
            cells = [(cell or "").strip() for cell in row]
            padded = cells + [""] * DEFINITION_END_COLUMN
            name = " ".join(c for c in padded[:name_columns] if c).strip()
            body = " ".join(
                c for c in padded[name_columns:DEFINITION_END_COLUMN]
                if c and c.lower() != "x"
            ).strip()
            if name and any(name.startswith(b) for b in banners):
                continue
            if not name:
                if body and units:
                    units[-1][1].append(body)
                continue
            units.append((_WHITESPACE.sub(" ", name), [body] if body else []))
        return [(name, " ".join(parts).strip()) for name, parts in units]

    @staticmethod
    def normalise_name(name: str) -> str:
        """The comparison key for a control or threat name.

        NFKD folds compatibility forms, the punctuation map turns the source's
        typographic quotes, dashes and ellipsis into the ASCII OpenCRE stores,
        and a declared footnote reference is removed by FOOTNOTE_NAMES before
        this is called, not by a trailing-digit regex here.
        """
        folded = unicodedata.normalize("NFKD", name)
        for source, target in _PUNCTUATION.items():
            folded = folded.replace(source, target)
        return _WHITESPACE.sub(" ", folded).strip().lower()

    @classmethod
    def _clean(cls, name: str) -> str:
        """A stored title: footnote reference removed, punctuation as ASCII."""
        stripped = FOOTNOTE_NAMES.get(name, name)
        folded = unicodedata.normalize("NFKD", stripped)
        for source, target in _PUNCTUATION.items():
            folded = folded.replace(source, target)
        return _WHITESPACE.sub(" ", folded).strip()

    def _check_declarations(
        self, table5: list[tuple[str, str]], annex_c: list[tuple[str, str]],
    ) -> None:
        """Refuse declarations that no longer match the extracted tables.

        Raises:
            ValueError: If a declared footnote name, Annex C variant or
                Annex-C-only control is absent from this parse.
        """
        table5_names = {name for name, _ in table5}
        missing_footnotes = sorted(set(FOOTNOTE_NAMES) - table5_names)
        if missing_footnotes:
            raise ValueError(
                f"{self.framework_id}: FOOTNOTE_NAMES declares {missing_footnotes}, "
                f"which this parse did not produce. A stale entry means a "
                f"control name still carries a footnote digit, and every link "
                f"to it silently falls back to its own title."
            )

        table5_keys = {self.normalise_name(self._clean(n)) for n, _ in table5}
        missing_targets = sorted({
            target for target in ANNEX_C_VARIANTS.values()
            if self.normalise_name(target) not in table5_keys
        })
        if missing_targets:
            raise ValueError(
                f"{self.framework_id}: ANNEX_C_VARIANTS points at Table 5 "
                f"control(s) {missing_targets} that this parse did not "
                f"produce."
            )

        annex_names = {name for name, _ in annex_c}
        missing_only = sorted(set(ANNEX_C_ONLY) - annex_names)
        if missing_only:
            raise ValueError(
                f"{self.framework_id}: ANNEX_C_ONLY declares {missing_only}, "
                f"which Annex C did not yield. Those two controls appear "
                f"nowhere in Table 5's extraction, so without them six "
                f"curated links have no anchor at all."
            )

    @classmethod
    def _build(
        cls,
        table5: list[tuple[str, str]],
        table3: list[tuple[str, str]],
        annex_c: list[tuple[str, str]],
    ) -> list[Control]:
        """Controls from Table 5 and Annex C, threats from Table 3."""
        alternates: dict[str, list[str]] = {}
        for variant, target in ANNEX_C_VARIANTS.items():
            alternates.setdefault(cls.normalise_name(target), []).append(
                cls._clean(variant)
            )

        controls: list[Control] = []
        for name, definition in table5:
            title = cls._clean(name)
            metadata: dict[str, str | list[str]] = {"table": "Table 5"}
            names = alternates.get(cls.normalise_name(title))
            if names:
                metadata["alt_titles"] = names
            controls.append(Control(
                control_id=cls._slug(title),
                title=title,
                description=definition,
                hierarchy_level="control",
                metadata=metadata,
            ))

        annex_bodies = {name: body for name, body in annex_c}
        for name in ANNEX_C_ONLY:
            title = cls._clean(name)
            controls.append(Control(
                control_id=cls._slug(title),
                title=title,
                description=annex_bodies[name],
                hierarchy_level="control",
                metadata={"table": "Annex C"},
            ))

        for name, definition in table3:
            title = cls._clean(name)
            controls.append(Control(
                control_id=cls._slug(title),
                title=title,
                description=definition,
                hierarchy_level="threat",
                metadata={"table": "Table 3"},
            ))
        return controls

    @staticmethod
    def _slug(title: str) -> str:
        """A synthetic control id.

        The source has no identifier of any kind, so this is generated. It is
        derived from the cleaned title so it is stable across re-parses of the
        same bytes, and it is never used for the join: every curated link
        carries either "Table 5:" or "Table 3:" or the name itself.
        """
        return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:80]


def main() -> None:
    EnisaParser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and typecheck**

```bash
pytest tests/test_parse_enisa.py -q
mypy parsers/parse_enisa.py --strict
```

- [ ] **Step 5: Run against the real source and check the join**

```bash
PYTHONPATH=. "$PY" parsers/parse_enisa.py
PYTHONPATH=. "$PY" parsers/merge_all_controls.py
PYTHONPATH=. "$PY" scripts/corpus_report.py | grep -E "^(enisa|framework)"
```

Expected `enisa: 37 controls and 13 threats, 6 Annex C alternate spelling(s)`
**[measured]**, then:

```
enisa                         68    68    0     0      33  2.06     0    0     0     0 1.0000
```

Accept only if `resolution_rate == 1.0000`, `distinct_anchors == 33`,
`nested_anchors == 0`, `wrong_anchor_risk == 0`. If the rate is 0.9118, the two
Annex-C-only controls are missing. If it is 0.7500, `normalise_name` is not
being applied to the stored title. `distinct_anchors` of 33 equals the count on
the fallback anchors today; that is correct and expected — this framework's
gain is text, not anchor separation, and the plan says so in advance rather
than discovering it in the report.

- [ ] **Step 6: Commit**

```bash
git add parsers/parse_enisa.py tests/test_parse_enisa.py \
        data/processed/frameworks/enisa.json data/processed/all_controls.json
git commit -m "feat: join ENISA on repaired control names across Table 3, Table 5 and Annex C"
```

---

### Task 12: BIML — two documents, one id space, and the collision the title channel would cause

Both PDFs mark each named risk with an inline `[category:number:label]` tag, and
that tag is a real structural delimiter. Measured on the pinned files:
`ara.pdf` (BIML-78, 2020) defines **78** distinct tags at a line start and
`BIML-LLM24.pdf` (BIML-24 LLM, 2024) defines **68**. **[measured]** Every tag
this framework's links need carries a body of 153 to 1,319 characters.
**[measured]**

**The two documents reuse the same id space for different risks**, and OpenCRE
leaves 8 of 21 links unprefixed. Measured by exact tag-label match:

| unprefixed id | link name | resolves to | evidence |
|---|---|---|---|
| `model:2` | Trojan | ara | exact label `trojan`; LLM24's `model:2` is `improper use` |
| `raw:3` | Storage | ara | exact `storage`; LLM24's is `data feudalism` |
| `input:2` | Controlled Input Stream | ara | exact; LLM24's is `prompt injection` |
| `inference:4` | Hosting | ara | exact `hosting`; LLM24's is `stochasticity` |
| `alg:11` | Parameters | ara | exact; absent from LLM24 |
| `inference:9` | Hosting | LLM24 | exact `hosting`; absent from ara |
| `output:4` | Output Data Confidentiality | LLM24 | LLM24's is `data confidentiality`; ara's is `inscrutability` |
| `output:2` | Direct Output | **ara `output:1`** | ara's `output:1` is `direct`; ara's own `output:2` is `provenance` and LLM24's is `wrongness` |

**[measured, all eight]** The first seven are registered as `alt_ids`. The
eighth is an upstream id-versus-name conflict: the name matches ara's
`output:1` exactly and the id matches a control about provenance. It is
resolved by name, not by id — `Direct Output` is registered as an `alt_title`
on ara's `output:1` — and the conflict is written to the repair audit. The
alternative, aliasing `output:2` onto `output:1`, would assert that OpenCRE's
id is a typo, which the evidence does not support any better than the name
being right.

**Titles must be document-scoped or the title channel destroys the join.**
`Data Confidentiality` is the `section_name` of two links naming two different
risks in two different documents; `Hosting` is the name of three links across
both. Seven of the 21 rows participate in a label collision. **[measured]**
With a bare label as the title, `ProseIndex.lookup` — which tries the title
first — would hand all of them one anchor, which is the NIST AI 100-2 collapse
again. Titles are therefore `f"{Label} ({document})"`, which no link name
spells, so every row goes through the id channel where the document prefix
disambiguates.

**Ceiling: 21/21 = 1.0000** **[derived]** — 13 prefixed rows by id, 7 by
`alt_ids`, 1 by `alt_title`. Floor **1.00**. Without the two declared
alternates for `output:2` and `output:4` it is 19/21 = 0.9048.

**Context, not an instruction.** BIML carries 21 of 4,127 training links.
The ceiling study measured human alpha-1 at 0.572 pooled and 0.181 for CAPEC,
which is 42.8% of the training graph. Effort spent here buys 0.5% of the graph;
the reason to do it well is that it is the case where the title-before-id order
is most obviously wrong, not its weight.

**Files:**
- Create: `parsers/parse_biml.py`
- Create: `tests/test_parse_biml.py`

**Interfaces:**
- Consumes: `BaseParser`, `Control`, `BaseParser.write_repair_audit`, `alt_ids` from Task 2.
- Produces: `BimlParser` with `framework_id = "biml"`, `framework_name = "BIML"`; `BimlParser.risks_from_text(text: str, document: str) -> list[tuple[str, str, str]]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_biml.py — create

"""BIML's two documents reuse one id space, and OpenCRE leaves 8 ids unprefixed.

Measured: 'Data Confidentiality' names two different risks across the two PDFs
and 'Hosting' names three link rows. With a bare label as the title, ProseIndex
-- which resolves title before id -- gives all of them one anchor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from parsers.parse_biml import ARA, LLM24, BimlParser

ARA_TEXT = """[raw:3:storage]
As in other systems, data may be stored and managed in an insecure fashion.
Who has access to the data pool, and why? Think about [system:8:insider] when
working on storage of any training data.

[output:1:direct]
Direct output of a model is the answer it hands the requester, and the
requester may not be the party the model was built to serve.

[inference:4:hosting]
Where the model runs decides who can reach it, and a hosted model inherits the
trust boundary of whatever hosts it.
"""

LLM24_TEXT = """[raw:3:data feudalism]
A small number of parties control the data that everyone else trains on.

[inference:9:hosting]
An LLM served by a third party puts the prompt, the completion and the system
message inside somebody else's trust boundary.
"""


class TestRisksFromText:
    def test_only_line_start_tags_define_a_risk(self) -> None:
        risks = BimlParser.risks_from_text(ARA_TEXT, ARA)
        assert [tag for tag, _, _ in risks] == [
            "raw:3", "output:1", "inference:4",
        ]

    def test_a_body_runs_to_the_next_definition(self) -> None:
        risks = dict((t, b) for t, _, b in BimlParser.risks_from_text(ARA_TEXT, ARA))
        assert "insecure fashion" in risks["raw:3"]
        assert "Direct output" not in risks["raw:3"]


class TestScopedIdentity:
    def test_control_id_carries_the_document(self) -> None:
        controls = BimlParser.build_controls(
            {ARA: ARA_TEXT, LLM24: LLM24_TEXT},
        )[0]
        assert f"{ARA}: raw:3" in {c.control_id for c in controls}
        assert f"{LLM24}: raw:3" in {c.control_id for c in controls}

    def test_titles_cannot_collide_across_documents(self) -> None:
        controls = BimlParser.build_controls(
            {ARA: ARA_TEXT, LLM24: LLM24_TEXT},
        )[0]
        titles = [c.title for c in controls]
        assert len(titles) == len(set(titles))
        assert f"Hosting ({ARA})" in titles
        assert f"Hosting ({LLM24})" in titles

    def test_an_unprefixed_id_is_an_alternate_on_exactly_one_document(
        self,
    ) -> None:
        controls = BimlParser.build_controls(
            {ARA: ARA_TEXT, LLM24: LLM24_TEXT},
        )[0]
        holders = [
            c.control_id for c in controls
            if c.metadata and "raw:3" in (c.metadata.get("alt_ids") or [])
        ]
        assert holders == [f"{ARA}: raw:3"]

    def test_the_named_conflict_resolves_by_name(self) -> None:
        controls = BimlParser.build_controls(
            {ARA: ARA_TEXT, LLM24: LLM24_TEXT},
        )[0]
        target = next(c for c in controls if c.control_id == f"{ARA}: output:1")
        assert target.metadata is not None
        assert "Direct Output" in target.metadata["alt_titles"]
        assert not any(
            "output:2" in (c.metadata or {}).get("alt_ids", [])
            for c in controls
        )


class TestAudit:
    def test_the_conflict_is_recorded(self, tmp_path: Path) -> None:
        parser = BimlParser(output_dir=tmp_path, audit_dir=tmp_path / "audit")
        _, audit = BimlParser.build_controls({ARA: ARA_TEXT, LLM24: LLM24_TEXT})
        parser.write_repair_audit(audit)
        records = [
            json.loads(line)
            for line in (tmp_path / "audit" / "biml.jsonl").read_text(
                encoding="utf-8",
            ).splitlines()
        ]
        assert any(r["opencre_section_id"] == "output:2" for r in records)


class TestDeclarations:
    def test_an_alternate_whose_target_is_absent_is_refused(self) -> None:
        with pytest.raises(ValueError, match="alt_ids"):
            BimlParser.build_controls({ARA: ARA_TEXT}, require_targets=True)


class TestRun:
    def test_run_writes_from_the_real_pdfs(self, tmp_path: Path) -> None:
        parser = BimlParser(output_dir=tmp_path, audit_dir=tmp_path / "audit")
        try:
            output = parser.run()
        except FileNotFoundError:
            pytest.skip("data/raw is gitignored and absent in this checkout")
        assert len(output.controls) == 146
        assert sorted(s.path for s in output.source_files) == [
            "BIML-LLM24.pdf", "ara.pdf",
        ]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_parse_biml.py -q
```

Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Write the parser**

```python
# parsers/parse_biml.py — create

"""Parser for the Berryville Institute of Machine Learning risk analyses.

Two documents, both required. ara.pdf is the 2020 architectural risk analysis
of machine learning systems, "BIML-78"; BIML-LLM24.pdf is the 2024 analysis of
large language models, "BIML-24(LLM)". Both mark every named risk with an
inline [category:number:label] tag, and a tag that starts a line is the risk's
definition while the same tag inside a sentence is a cross-reference.

The two documents reuse the same category:number space for different risks:
ara's raw:3 is Storage and LLM24's is data feudalism. So a control_id must
carry its document, and it is spelled exactly as OpenCRE's prefixed ids are.

Titles carry the document too, and that is the load-bearing decision here.
ProseIndex.lookup resolves the section NAME before the section id. OpenCRE
gives two different risks the name "Data Confidentiality" and three link rows
the name "Hosting"; seven of the 21 curated rows participate in a label
collision. A bare label as the title would hand all of them one anchor, which
is the same collapse the title-first order was written to fix for NIST AI
100-2. A scoped title matches no link name at all, so every row falls through
to the id channel, where the document prefix disambiguates.

Eight curated links carry an unprefixed id. Seven are resolved by exact
tag-label match against one document and only one document, and are declared
in UNPREFIXED_IDS. The eighth, output:2 "Direct Output", matches ara's
[output:1:direct] by name while ara's own output:2 is provenance and LLM24's is
wrongness. It is resolved by NAME, as an alt_title on ara's output:1, and the
id conflict is written to the repair audit. Aliasing the id instead would
assert OpenCRE made a typo, which the evidence supports no better.
"""
from __future__ import annotations

import hashlib
import logging
import re
from io import BytesIO
from typing import ClassVar, Final

import pdfplumber

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARA: Final[str] = "BIML-78(2020)"
LLM24: Final[str] = "BIML-24(LLM)"

SOURCE_FILES: Final[dict[str, str]] = {ARA: "ara.pdf", LLM24: "BIML-LLM24.pdf"}
SOURCE_SHA256: Final[dict[str, str]] = {
    ARA: "247d7f06d8c768cc734dc84ab7004c6e4d645e91911af61002fd1743807ef312",
    LLM24: "1a41ba1a9218e6aecdcab46d2cc6cf8a3b99f6cc1c98a3683bf3a6e4964e955f",
}

TAG: Final[re.Pattern[str]] = re.compile(r"\[([a-z]+):(\d+):([^\]]+)\]")

# Unprefixed OpenCRE ids, each resolved by exact tag-label match against one
# document and only one. Hand-verified against both PDFs; see the module
# docstring for the evidence per entry.
UNPREFIXED_IDS: Final[dict[str, tuple[str, str]]] = {
    "model:2": (ARA, "model:2"),
    "raw:3": (ARA, "raw:3"),
    "input:2": (ARA, "input:2"),
    "inference:4": (ARA, "inference:4"),
    "alg:11": (ARA, "alg:11"),
    "inference:9": (LLM24, "inference:9"),
    "output:4": (LLM24, "output:4"),
}

# The one row whose id and name disagree upstream. Resolved by name.
NAME_CONFLICTS: Final[dict[str, tuple[str, str, str]]] = {
    "Direct Output": (
        ARA, "output:1",
        "OpenCRE's section_id output:2 names ara's provenance risk and "
        "LLM24's wrongness risk; its section_name matches ara's "
        "[output:1:direct] exactly. Resolved by name, because the name is "
        "the only side of the row that matches anything in either document.",
    ),
}

_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


class BimlParser(BaseParser):
    framework_id: ClassVar[str] = "biml"
    framework_name: ClassVar[str] = "BIML"
    version: ClassVar[str] = "BIML-78 (2020) + BIML-24 LLM (2024)"
    source_url: ClassVar[str] = "https://berryvilleiml.com/results/"
    mapping_unit_level: ClassVar[str] = "risk"
    # 78 distinct definitional tags in ara.pdf and 68 in BIML-LLM24.pdf.
    # [measured] Declared as a floor: the tag vocabulary is what each document
    # names, and a re-pin that adds a risk should not fail the parser.
    expected_count: ClassVar[int] = 146
    expected_count_is_floor: ClassVar[bool] = True
    fetched_date: ClassVar[str] = "2026-08-15"
    # Every tag needed by a curated link carries 153 to 1,319 characters, and
    # the definitional tags as a whole run well past 60. [measured] The floor
    # allows for short cross-reference-only definitions in the unlinked tail.
    min_prose_fraction: ClassVar[float] = 0.90
    expected_sha256: ClassVar[dict[str, str] | None] = SOURCE_SHA256

    def parse(self) -> list[Control]:
        texts: dict[str, str] = {}
        for document, filename in SOURCE_FILES.items():
            payload = self.read_source_bytes(filename)
            self._check_digest(document, filename, payload)
            with pdfplumber.open(BytesIO(payload)) as pdf:
                texts[document] = "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
        controls, audit = self.build_controls(texts, require_targets=True)
        self.write_repair_audit(audit)
        for record in audit:
            logger.warning(
                "%s: %s resolved by name to %s, not by its own id: %s",
                self.framework_id, record["opencre_section_id"],
                record["resolved_to"], record["reason"],
            )
        logger.info(
            "%s: %d risks (%s)", self.framework_id, len(controls),
            ", ".join(
                f"{document} {sum(1 for c in controls if c.control_id.startswith(document))}"
                for document in SOURCE_FILES
            ),
        )
        return controls

    def _check_digest(self, document: str, filename: str, payload: bytes) -> None:
        """Refuse a PDF that is not the pinned one.

        Raises:
            ValueError: If the digest does not match the pin for `document`.
        """
        if self.expected_sha256 is None:
            return
        actual = hashlib.sha256(payload).hexdigest()
        expected = self.expected_sha256[document]
        if actual != expected:
            raise ValueError(
                f"{self.framework_id}: {filename} has sha256 {actual}, not "
                f"the pinned {expected}. UNPREFIXED_IDS and NAME_CONFLICTS "
                f"were resolved by comparing tag labels across these exact "
                f"two documents; a different revision can move a label onto a "
                f"different number and silently re-point a link."
            )

    @classmethod
    def risks_from_text(
        cls, text: str, document: str,
    ) -> list[tuple[str, str, str]]:
        """(tag, label, body) per definitional tag, in document order.

        A tag that starts a line defines a risk; the same tag inside a sentence
        is a cross-reference and is left alone. The body runs to the next
        definitional tag.
        """
        lines = text.split("\n")
        found: list[tuple[str, str, list[str]]] = []
        for line in lines:
            stripped = line.strip()
            match = TAG.match(stripped)
            if match is None:
                if found:
                    found[-1][2].append(stripped)
                continue
            found.append((
                f"{match.group(1)}:{match.group(2)}",
                _WHITESPACE.sub(" ", match.group(3)).strip(),
                [stripped[match.end():].strip()],
            ))
        seen: set[str] = set()
        risks: list[tuple[str, str, str]] = []
        for tag, label, body in found:
            if tag in seen:
                continue
            seen.add(tag)
            risks.append((tag, label, _WHITESPACE.sub(" ", " ".join(body)).strip()))
        return risks

    @classmethod
    def build_controls(
        cls, texts: dict[str, str], require_targets: bool = False,
    ) -> tuple[list[Control], list[dict[str, object]]]:
        """Document-scoped controls, plus the audit for the name conflict.

        Raises:
            ValueError: If require_targets is set and a declared alternate id
                or name conflict points at a tag this parse did not produce.
        """
        alt_ids: dict[str, list[str]] = {}
        for unprefixed, (document, tag) in UNPREFIXED_IDS.items():
            alt_ids.setdefault(f"{document}: {tag}", []).append(unprefixed)

        alt_titles: dict[str, list[str]] = {}
        audit: list[dict[str, object]] = []
        for name, (document, tag, reason) in NAME_CONFLICTS.items():
            alt_titles.setdefault(f"{document}: {tag}", []).append(name)
            audit.append({
                "opencre_section_id": "output:2",
                "opencre_section_name": name,
                "resolved_to": f"{document}: {tag}",
                "resolved_by": "section_name",
                "reason": reason,
            })

        controls: list[Control] = []
        for document in sorted(texts):
            for tag, label, body in cls.risks_from_text(texts[document], document):
                control_id = f"{document}: {tag}"
                metadata: dict[str, str | list[str]] = {
                    "document": document, "tag": tag, "label": label,
                }
                if control_id in alt_ids:
                    metadata["alt_ids"] = alt_ids[control_id]
                if control_id in alt_titles:
                    metadata["alt_titles"] = alt_titles[control_id]
                controls.append(Control(
                    control_id=control_id,
                    # Scoped so no OpenCRE section_name can match it. See the
                    # module docstring: seven of 21 rows share a label.
                    title=f"{label.title()} ({document})",
                    description=body or label,
                    hierarchy_level="risk",
                    parent_id=tag.split(":")[0],
                    metadata=metadata,
                ))

        cls._check_targets(
            {c.control_id for c in controls}, alt_ids, alt_titles, require_targets,
        )
        return controls, audit

    @staticmethod
    def _check_targets(
        control_ids: set[str],
        alt_ids: dict[str, list[str]],
        alt_titles: dict[str, list[str]],
        required: bool,
    ) -> None:
        """Refuse declarations pointing at tags this parse did not produce.

        Raises:
            ValueError: If any declared target is absent.
        """
        if not required:
            return
        missing_ids = sorted(set(alt_ids) - control_ids)
        if missing_ids:
            raise ValueError(
                f"biml: UNPREFIXED_IDS declares alt_ids on {missing_ids}, "
                f"which this parse did not produce. Seven curated links reach "
                f"their risk only through that channel; a stale entry leaves "
                f"them resolving to nothing."
            )
        missing_titles = sorted(set(alt_titles) - control_ids)
        if missing_titles:
            raise ValueError(
                f"biml: NAME_CONFLICTS declares alt_titles on "
                f"{missing_titles}, which this parse did not produce."
            )


def main() -> None:
    BimlParser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and typecheck**

```bash
pytest tests/test_parse_biml.py -q
mypy parsers/parse_biml.py --strict
```

- [ ] **Step 5: Run against the real source and check the join**

```bash
PYTHONPATH=. "$PY" parsers/parse_biml.py
PYTHONPATH=. "$PY" parsers/merge_all_controls.py
PYTHONPATH=. "$PY" scripts/corpus_report.py | grep -E "^(biml|framework)"
```

Expected `biml: 146 risks (BIML-78(2020) 78, BIML-24(LLM) 68)` and one
`output:2 resolved by name` warning **[measured]**, then:

```
biml                          21     1   20     0      20  1.05     0    0    ~5     0 1.0000
```

Accept only if `resolution_rate == 1.0000`, `distinct_anchors == 20`,
`by_title == 1`, `wrong_anchor_risk == 0`. **`by_title` must be exactly 1** —
the `Direct Output` conflict. If it is 7 or more, the titles are not
document-scoped and the label collision is back; stop, because the rate will
still read 1.0000 while `distinct_anchors` falls to 17 and the report is the
only place that shows it.

- [ ] **Step 6: Confirm the audit file is gitignored, then commit**

```bash
git check-ignore -v data/processed/repair_audit/biml.jsonl \
  || echo "NOT IGNORED — stop and fix .gitignore before committing"
git add parsers/parse_biml.py tests/test_parse_biml.py \
        data/processed/frameworks/biml.json data/processed/all_controls.json
git commit -m "feat: scope BIML risks to their document so two id spaces stop colliding"
```

---

### Task 13: ETSI — restricted, coarse by construction, and honest about it

**ETSI is in `RESTRICTED_FRAMEWORK_IDS`.** Its processed JSON is already
gitignored (`.gitignore:37`) and `parsers/merge_all_controls.py` routes it to
the gitignored `data/processed/licensed/` overlay. Its prose must not appear in
any tracked file, in any test fixture, in any commit message, or in this plan.
Every fixture below is synthetic. `data/processed/repair_audit/` is gitignored
for the same reason.

**The technique names are not structural.** 36 curated links carry 24 distinct
`section_name` values over 16 distinct `section_id` values, 27 distinct pairs.
**[measured]** All 24 names appear verbatim in the PDF text, but only 2 of them
are clause headings and only 1 is a bullet lead phrase before a colon; 9 appear
mid-sentence only. **[measured]** A technique-level parser would have to guess
sentence boundaries around a name that occurs 1 to 29 times across the document.
That is prose heuristics, and ledger lesson 7 says a transform that synthesises
text has to fail closed rather than guess.

**Ruling: clause-level mapping units.** One control per numbered clause in
sections 5 through 7 — 25 of them **[measured]** — with the clause's own text,
rolled up from its immediate children when the clause has none of its own
(clauses 5.2 and 5.3 are headings whose text is entirely in their subclauses).
**[measured]**

**And almost no `alt_titles`, which is the opposite of what it first looks
like.** The `control_id` is the clause number, and 34 of the 36 curated links
carry a clause number in `section_id`, so those 34 resolve through the id
channel with nothing declared at all. Registering all 24 technique names as
alternates would be actively wrong: three names span two clauses each —
`Membership inference attacks` under 6.4.1 and 6.4.2, `Model inversion attacks`
under 6.4.1 and 6.4.3, and `Data sanitisation` under 5.2.2 and its own name.
**[measured]** Because `lookup` tries the title first, a name registered on one
clause would answer the link that named the other, which is a wrong anchor, not
a fallback, and Task 1's `wrong_anchor_risk` column is what would show it.

So exactly **two** alternates are declared, for the two rows whose `section_id`
is a name rather than a clause number: `Data sanitisation` on clause 5.2.2 and
`Retraining` on clause 5.3.2, each being the clause where that name occurs
exactly once in the document. **[measured]** Everything else is OpenCRE's own
clause assertion, honoured verbatim.

**This makes the anchor count worse and the plan says so before the run.** 36
links land on 24 title anchors today, at 1.50 links each. Under clause-level
units they land on **14** anchors at 2.57. **[derived]** The trade is 24 short
phrases against 14 paragraphs. It is recorded in the AFTER report as a
regression on the anchor column and a gain on the text column, and it is the
kind of trade the previous plan's link-only instrument could not have shown at
all.

**Ceiling: 36/36 = 1.0000** **[derived]**, floor **1.00**. Nineteen of the 25
clause bodies exceed `MAX_ANCHOR_CHARS` after roll-up **[measured]**, so
truncation on this framework will be high; the report records the number rather
than the plan asserting one.

**Files:**
- Create: `parsers/parse_etsi.py`
- Create: `tests/test_parse_etsi.py`

**Interfaces:**
- Consumes: `BaseParser`, `Control`.
- Produces: `EtsiParser` with `framework_id = "etsi"`, `framework_name = "ETSI"`; `EtsiParser.clauses_from_text(text: str) -> dict[str, tuple[str, str]]`; `EtsiParser.build_controls(clauses: dict[str, tuple[str, str]], alternates: dict[str, str]) -> list[Control]`.

- [ ] **Step 1: Confirm the licensing routing before writing anything**

```bash
git check-ignore -v data/processed/frameworks/etsi.json
"$PY" -c "
from tract.config import RESTRICTED_FRAMEWORK_IDS
print('etsi restricted:', 'etsi' in RESTRICTED_FRAMEWORK_IDS)
"
pytest tests/test_licensed_text_not_tracked.py -q
```

Expected: the file is ignored, `etsi restricted: True`, and the gate passes.

- [ ] **Step 2: Confirm which rows actually need an alternate title**

```bash
"$PY" - <<'PYEOF'
import collections
import json

pairs = collections.Counter()
with open('data/training/hub_links_curated.jsonl', encoding='utf-8') as handle:
    for line in handle:
        row = json.loads(line)
        if row['framework_id'] == 'etsi':
            pairs[(row['section_id'], row['section_name'])] += 1

by_name = collections.defaultdict(set)
for section_id, name in pairs:
    by_name[name].add(section_id)

print(sum(pairs.values()), 'rows,', len(pairs), 'pairs,',
      len({a for a, _ in pairs}), 'ids,', len({b for _, b in pairs}), 'names')
print('rows whose section_id is a name:',
      sorted(a for a, _ in pairs if not a[0].isdigit()))
print('names spanning more than one clause:')
for name, ids in sorted(by_name.items()):
    if len(ids) > 1:
        print(f'   {name!r} -> {sorted(ids)}')
PYEOF
```

Expected **[measured]**: `36 rows, 27 pairs, 16 ids, 24 names`; the two
name-shaped ids `Data sanitisation` and `Retraining`; and three names spanning
two clauses — `Data sanitisation` (5.2.2 and its own name),
`Membership inference attacks` (6.4.1 and 6.4.2), `Model inversion attacks`
(6.4.1 and 6.4.3). Those last two are exactly why the parser below declares two
alternates and not twenty-four.

- [ ] **Step 3: Write the failing test**

```python
# tests/test_parse_etsi.py — create

"""ETSI is restricted: every fixture here is synthetic, none of it is source.

The technique names OpenCRE links are not structural anchors in the PDF -- 2 of
24 are clause headings and 9 appear mid-sentence only -- so the mapping unit is
the numbered clause and the names are registered as alternate titles on the
clause the link's own section_id names.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parsers.parse_etsi import EtsiParser

# Synthetic. Shaped like the source, worded nothing like it.
TEXT = """4 General
Front matter that is not a mapping unit.

5 First area
An opening paragraph for the first area.

5.1 First topic
The first topic runs for a couple of sentences and says what it covers.

5.2 Second topic
5.2.1 First sub-topic
Sub-topic text that the parent clause has none of its own.

5.2.2 Second sub-topic
More sub-topic text, long enough to be a statement in its own right.

6 Second area
6.1 Another topic
Text for another topic that is long enough to count as a statement.
"""


class TestClausesFromText:
    def test_numbered_clauses_are_found(self) -> None:
        clauses = EtsiParser.clauses_from_text(TEXT)
        assert set(clauses) >= {"5", "5.1", "5.2", "5.2.1", "5.2.2", "6", "6.1"}

    def test_section_four_is_not_a_mapping_unit(self) -> None:
        assert "4" not in EtsiParser.clauses_from_text(TEXT)

    def test_a_parent_with_no_text_rolls_up_its_children(self) -> None:
        clauses = EtsiParser.clauses_from_text(TEXT)
        parent = clauses["5.2"][1]
        assert "Sub-topic text" in parent
        assert "More sub-topic text" in parent

    def test_a_clause_with_its_own_text_does_not_roll_up(self) -> None:
        clauses = EtsiParser.clauses_from_text(TEXT)
        assert "Sub-topic text" not in clauses["5.1"][1]


class TestNameShapedIds:
    def test_a_declared_name_becomes_an_alternate_title_on_its_clause(
        self,
    ) -> None:
        controls = EtsiParser.build_controls(
            EtsiParser.clauses_from_text(TEXT), {"Some technique": "5.1"},
        )
        first = next(c for c in controls if c.control_id == "5.1")
        assert first.metadata is not None
        assert "Some technique" in first.metadata["alt_titles"]

    def test_no_other_clause_gains_an_alternate(self) -> None:
        controls = EtsiParser.build_controls(
            EtsiParser.clauses_from_text(TEXT), {"Some technique": "5.1"},
        )
        holders = [
            c.control_id for c in controls
            if (c.metadata or {}).get("alt_titles")
        ]
        assert holders == ["5.1"]

    def test_a_declared_clause_that_is_absent_is_refused(self) -> None:
        with pytest.raises(ValueError, match="clause"):
            EtsiParser.build_controls(
                EtsiParser.clauses_from_text(TEXT),
                {"Nowhere technique": "9.9"},
            )


class TestRun:
    def test_run_writes_from_the_real_pdf(self, tmp_path: Path) -> None:
        parser = EtsiParser(output_dir=tmp_path)
        try:
            output = parser.run()
        except FileNotFoundError:
            pytest.skip("data/raw is gitignored and absent in this checkout")
        assert len(output.controls) == 25
        assert [s.path for s in output.source_files] == [
            "etsi_gr_sai005_v010101p.pdf",
        ]

    def test_exactly_two_alternates_are_registered(
        self, tmp_path: Path,
    ) -> None:
        """A name registered on one clause answers links naming the other.

        Three of the 24 technique names span two clauses, so registering all
        of them would put a wrong anchor in front of the encoder while the
        resolution rate still read 1.0000.
        """
        parser = EtsiParser(output_dir=tmp_path)
        try:
            output = parser.run()
        except FileNotFoundError:
            pytest.skip("data/raw is gitignored and absent in this checkout")
        named = [
            name
            for control in output.controls
            for name in (control.metadata or {}).get("alt_titles", [])
        ]
        assert sorted(named) == ["Data sanitisation", "Retraining"]
```

- [ ] **Step 4: Run the test to verify it fails**

```bash
pytest tests/test_parse_etsi.py -q
```

Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 5: Write the parser**

```python
# parsers/parse_etsi.py — create

"""Parser for ETSI GR SAI 005, Securing AI Problem Statement.

RESTRICTED. ETSI's copyright notice reserves reproduction in any medium
without written permission, so this framework is in
tract.config.RESTRICTED_FRAMEWORK_IDS, data/processed/frameworks/etsi.json is
gitignored, and merge_all_controls routes its prose to the gitignored licensed
overlay. Nothing in this file, its tests, or its commit messages quotes the
source.

The mapping unit is the numbered clause, and that is a deliberate choice made
against a measured alternative. OpenCRE's 36 curated links carry 24 distinct
technique names over 16 section ids. All 24 names appear in the document, but
only 2 are clause headings and only 1 is a bullet lead phrase; 9 appear
mid-sentence only, and several occur more than twenty times as
cross-references. Segmenting a technique's text out of running prose would mean
guessing sentence boundaries around an ambiguous match, and a wrong guess
attributes a mitigation to the wrong attack class with a provenance record that
looks correct.

Almost nothing is declared here, and that is deliberate. control_id IS the
clause number, so 34 of the 36 curated links resolve through the id channel
with no alias at all. Registering all 24 technique names as alternates would be
actively harmful: three of them span two clauses -- Membership inference
attacks under 6.4.1 and 6.4.2, Model inversion attacks under 6.4.1 and 6.4.3,
Data sanitisation under 5.2.2 and its own name -- and because lookup tries the
title first, a name registered on one clause would answer the link that named
the other. So NAME_SECTION_IDS holds exactly the two rows whose section_id is a
name rather than a clause number, and everything else is OpenCRE's own clause
assertion honoured verbatim.

The cost is stated rather than discovered: 36 links land on 24 short title
anchors today and on 14 clause anchors after, 2.57 links each. The corpus
report records that as a regression on the anchor column. The gain is that
those 14 anchors are paragraphs of the standard rather than three-word phrases.

Clauses 5.2 and 5.3 are headings whose text is entirely in their subclauses, so
a clause with no body of its own takes the concatenation of its immediate
children.
"""
from __future__ import annotations

import hashlib
import logging
import re
from io import BytesIO
from typing import ClassVar, Final

import pdfplumber

from tract.config import HONEST_PROSE_MIN_CHARS
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SOURCE_FILE: Final[str] = "etsi_gr_sai005_v010101p.pdf"
SOURCE_SHA256: Final[str] = (
    "46c2b6b880928ffe2e763fbd6e0d0660a0aa7de0ff0071f5e0694582d91d5622"
)

# Sections 1-4 are scope, references, definitions and an overview. The mapping
# units are the attack and mitigation clauses in 5 through 7.
CLAUSE: Final[re.Pattern[str]] = re.compile(r"^([5-7](?:\.\d+){0,3})\s+(\S.{2,80})$")

# The only two curated rows whose section_id is a technique name rather than a
# clause number, mapped to the clause where that name occurs exactly once in
# the document. Hand-verified against the pinned PDF.
#
# This map is two entries and not twenty-four on purpose. See the module
# docstring: three technique names span two clauses each, and registering a
# name as an alternate title makes it answer every link that carries it,
# including the link that named the other clause.
NAME_SECTION_IDS: Final[dict[str, str]] = {
    "Data sanitisation": "5.2.2",
    "Retraining": "5.3.2",
}

_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


class EtsiParser(BaseParser):
    framework_id: ClassVar[str] = "etsi"
    framework_name: ClassVar[str] = "ETSI"
    version: ClassVar[str] = "1.1.1"
    source_url: ClassVar[str] = (
        "https://www.etsi.org/deliver/etsi_gr/SAI/001_099/005/"
        "01.01.01_60/gr_SAI005v010101p.pdf"
    )
    mapping_unit_level: ClassVar[str] = "clause"
    # Numbered clauses in sections 5 through 7. [measured]
    expected_count: ClassVar[int] = 25
    fetched_date: ClassVar[str] = "2026-08-15"
    # Every clause clears 60 characters after roll-up. [measured]
    min_prose_fraction: ClassVar[float] = 1.0
    expected_sha256: ClassVar[str | None] = SOURCE_SHA256

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(SOURCE_FILE)
        self._check_digest(payload)
        with pdfplumber.open(BytesIO(payload)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        clauses = self.clauses_from_text(text)
        controls = self.build_controls(clauses, NAME_SECTION_IDS)
        logger.info(
            "%s: %d clauses, %d name-shaped section id(s) registered as "
            "alternate titles: %s", self.framework_id, len(controls),
            len(NAME_SECTION_IDS), sorted(NAME_SECTION_IDS),
        )
        return controls

    def _check_digest(self, payload: bytes) -> None:
        """Refuse a PDF that is not the pinned one.

        Raises:
            ValueError: If the digest does not match `expected_sha256`.
        """
        if self.expected_sha256 is None:
            return
        actual = hashlib.sha256(payload).hexdigest()
        if actual != self.expected_sha256:
            raise ValueError(
                f"{self.framework_id}: {SOURCE_FILE} has sha256 {actual}, not "
                f"the pinned {self.expected_sha256}. Both entries in "
                f"NAME_SECTION_IDS were verified against this revision's "
                f"clause numbering, and the 25-clause count was measured "
                f"against these bytes."
            )

    @classmethod
    def clauses_from_text(cls, text: str) -> dict[str, tuple[str, str]]:
        """clause number -> (heading, body), children rolled up where needed."""
        lines = text.split("\n")
        starts: list[tuple[int, str, str]] = []
        seen: set[str] = set()
        for index, line in enumerate(lines):
            match = CLAUSE.match(line.strip())
            if match is None or match.group(1) in seen:
                continue
            seen.add(match.group(1))
            starts.append((index, match.group(1), match.group(2).strip()))
        starts.sort()

        own: dict[str, tuple[str, str]] = {}
        for position, (start, number, heading) in enumerate(starts):
            end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
            body = _WHITESPACE.sub(
                " ", " ".join(line.strip() for line in lines[start + 1:end]),
            ).strip()
            own[number] = (heading, body)

        rolled: dict[str, tuple[str, str]] = {}
        for number, (heading, body) in own.items():
            if len(body) >= HONEST_PROSE_MIN_CHARS:
                rolled[number] = (heading, body)
                continue
            children = sorted(
                key for key in own if key.startswith(f"{number}.")
            )
            merged = " ".join(own[key][1] for key in children).strip()
            rolled[number] = (heading, merged or body)
        return rolled

    @classmethod
    def build_controls(
        cls,
        clauses: dict[str, tuple[str, str]],
        alternates_by_name: dict[str, str],
    ) -> list[Control]:
        """One Control per clause, name-shaped section ids as alternate titles.

        Raises:
            ValueError: If a declared name points at a clause this parse did
                not produce, or a clause has no text at all after roll-up.
        """
        alternates: dict[str, list[str]] = {}
        for name, clause in sorted(alternates_by_name.items()):
            alternates.setdefault(clause, []).append(name)

        missing = sorted(set(alternates) - set(clauses))
        if missing:
            raise ValueError(
                f"etsi: NAME_SECTION_IDS points at clause(s) {missing} that "
                f"this parse did not produce. Two curated links carry a "
                f"technique name where a clause number belongs and reach "
                f"their clause only through this map, so a stale entry leaves "
                f"them resolving to nothing while the parser still writes."
            )

        controls: list[Control] = []
        for number in sorted(clauses, key=lambda n: [int(p) for p in n.split(".")]):
            heading, body = clauses[number]
            if not body:
                raise ValueError(
                    f"etsi: clause {number} has no text of its own and no "
                    f"children to roll up. An empty statement would be "
                    f"excluded from the prose index without any gate saying so."
                )
            metadata: dict[str, str | list[str]] = {"clause": number}
            if number in alternates:
                metadata["alt_titles"] = alternates[number]
            controls.append(Control(
                control_id=number,
                title=heading,
                description=body,
                hierarchy_level="clause",
                parent_id=number.rsplit(".", 1)[0] if "." in number else None,
                metadata=metadata,
            ))
        return controls


def main() -> None:
    EtsiParser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the tests and typecheck**

```bash
pytest tests/test_parse_etsi.py -q
mypy parsers/parse_etsi.py --strict
```

- [ ] **Step 7: Run against the real source and check the join**

```bash
PYTHONPATH=. "$PY" parsers/parse_etsi.py
PYTHONPATH=. "$PY" parsers/merge_all_controls.py
PYTHONPATH=. "$PY" scripts/corpus_report.py | grep -E "^(etsi|framework)"
```

Expected `etsi: 25 clauses, 2 name-shaped section id(s) registered as alternate
titles: ['Data sanitisation', 'Retraining']` **[measured]**, then approximately:

```
etsi                          36     2   34     0      14  2.57    ~9    0     2     0 1.0000
```

Accept only if `resolution_rate == 1.0000`, `distinct_anchors == 14`,
**`by_title == 2`**, and `wrong_anchor_risk == 0`. `by_title` above 2 means
more names were registered than the two whose `section_id` needs them, and the
three names that span two clauses will start answering links they do not own —
which shows up in `wrong_anchor_risk`, not in the resolution rate. Record the
actual `truncated` value; 19 of 25 clause bodies exceed `MAX_ANCHOR_CHARS` and
the linked subset will be lower than that. `distinct_anchors` of 14 against 24
fallback anchors today is the trade this task declared in advance — do not
treat it as a surprise, and do carry it into the AFTER report as a named
regression.

- [ ] **Step 8: Confirm nothing licensed reached git, then commit**

```bash
git status --porcelain data/processed/frameworks/etsi.json
git check-ignore -v data/processed/frameworks/etsi.json
pytest tests/test_licensed_text_not_tracked.py tests/test_merge_licensed_overlay.py -q
```

Expected: `git status` prints nothing for that path, `git check-ignore` matches,
both tests pass.

```bash
git add parsers/parse_etsi.py tests/test_parse_etsi.py \
        data/processed/all_controls.json
git commit -m "feat: parse ETSI at the clause, where OpenCRE's own section ids point"
```

Note the absence of `data/processed/frameworks/etsi.json` from the `git add`.
That is deliberate and is the whole point of the restriction.

---

### Task 14: Retire both link gates onto the resolved anchor

`assign_quality_tier` drops a link two ways and both test a section title.
`PHASE1B_DROPPED_FRAMEWORKS` names `nist_800_63` and `owasp_proactive_controls`
outright, and `_has_descriptive_text` drops any link whose `section_name` is
shorter than `PHASE1B_MIN_SECTION_TEXT_LENGTH = 10`. Reproduced exactly:
**278 of 4,405 curated links are dropped, 155 by the framework list and 123 by
the short title.** **[measured]**

Per framework, the 123 short-title drops: capec 44, dsomm 38, cwe 17, enisa 9,
biml 7, iso_27001 2, nist_800_53 2, owasp_ai_exchange 2, etsi 1,
owasp_top10_2021 1. **[measured]**

**Sixty-four of those 123 already resolve to prose in today's corpus.**
**[measured]** They are dropped for having a short title while the pipeline
holds a paragraph for them. That is the whole defect: the gate tests a label the
model never sees.

**Retiring the framework list alone changes nothing for those two
frameworks.** Every one of `nist_800_63`'s 79 `section_name` values is a
section number of 3 to 7 characters and every one of
`owasp_proactive_controls`' 76 is `C1`..`C10`. **[measured]** Remove the
framework list and the short-title gate drops all 155 anyway. Both must move
together, which is why this is one task and why it lands after the parsers that
give those links a resolved anchor.

**Derived outcome.** After Tasks 3 through 13, the 155 resolve to paragraphs and
120 of the 123 do too. Three do not: two `nist_800_53` links and one `cwe` link
whose `section_id` and `section_name` both fail to match a parsed control, whose
fallback anchor is 3 to 9 characters, and which the new gate therefore still
drops — correctly. Training links go from **4,127 to 4,402 of 4,405**.
**[derived]** Every metric downstream re-bases, which is why the before and
after counts are written into the commit.

**Files:**
- Modify: `tract/training/data_quality.py`
- Modify: `tract/config.py`
- Modify: `tests/test_data_quality.py`

**Interfaces:**
- Consumes: `tract.text_selection.ProseIndex`, `select_control_text`; the parsers from Tasks 3-13.
- Produces: `assign_quality_tier(link: dict[str, str], resolved_text: str | None = None) -> QualityTier`; `filter_training_links(links, index: ProseIndex | None = None) -> list[TieredLink]`; `PHASE1B_MIN_ANCHOR_TEXT_LENGTH` replacing `PHASE1B_MIN_SECTION_TEXT_LENGTH`; `PHASE1B_DROPPED_FRAMEWORKS` deleted.

- [ ] **Step 1: Record the before state**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" - <<'PYEOF'
from tract.training.data_quality import load_and_filter_curated_links
links, _ = load_and_filter_curated_links()
print("training links before:", len(links))
PYEOF
```

Expected: `training links before: 4127`. **[measured]** Write the number down;
Step 6 compares against it.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_data_quality.py — append

class TestGatesTestTheAnchorNotTheTitle:
    """Both drops used to test section_name, which the model never sees."""

    LONG = "A control statement long enough to be worth training on, twice. " * 3

    def _index(self) -> "ProseIndex":
        from tract.text_selection import ProseIndex

        return ProseIndex([{
            "framework_name": "OWASP Proactive Controls",
            "controls": [
                {"control_id": "C6", "title": "Use Secure Dependencies",
                 "description": self.LONG},
            ],
        }])

    def test_a_short_title_with_a_resolved_anchor_is_kept(self) -> None:
        from tract.training.data_quality import QualityTier, assign_quality_tier

        link = {
            "framework_id": "owasp_proactive_controls",
            "standard_name": "OWASP Proactive Controls",
            "section_id": "C6", "section_name": "C6",
            "link_type": "LinkedTo",
        }
        assert assign_quality_tier(link, self.LONG) is QualityTier.T1

    def test_a_short_title_with_no_anchor_is_still_dropped(self) -> None:
        from tract.training.data_quality import QualityTier, assign_quality_tier

        link = {
            "framework_id": "nist_800_53",
            "standard_name": "NIST 800-53",
            "section_id": "AC-1", "section_name": "AC-1",
            "link_type": "LinkedTo",
        }
        assert assign_quality_tier(link, None) is QualityTier.DROPPED
        assert assign_quality_tier(link, "AC-1") is QualityTier.DROPPED

    def test_the_framework_deny_list_is_gone(self) -> None:
        import tract.config as config

        assert not hasattr(config, "PHASE1B_DROPPED_FRAMEWORKS"), (
            "the deny list dropped 155 links for having a short title, which "
            "is the same test the length gate already applies; keeping both "
            "means retiring one changes nothing"
        )

    def test_filter_resolves_through_the_index(self) -> None:
        from tract.training.data_quality import filter_training_links

        links = [{
            "framework_id": "owasp_proactive_controls",
            "standard_name": "OWASP Proactive Controls",
            "section_id": "C6", "section_name": "C6",
            "link_type": "LinkedTo",
        }]
        assert len(filter_training_links(links, self._index())) == 1
        assert len(filter_training_links(links, None)) == 0
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
pytest tests/test_data_quality.py::TestGatesTestTheAnchorNotTheTitle -q
```

Expected: FAIL — `assign_quality_tier` takes one argument, and
`PHASE1B_DROPPED_FRAMEWORKS` still exists.

- [ ] **Step 4: Change the config**

```python
# tract/config.py — replace the PHASE1B_DROPPED_FRAMEWORKS block

# A link is worth training on when the text the model will actually see is
# substantial. Both of the gates this replaces tested link["section_name"], a
# title the model never sees: a framework deny list naming nist_800_63 and
# owasp_proactive_controls, and a 10-character floor on the same field. Between
# them they dropped 278 of 4,405 curated links, and 64 of those already had a
# resolved paragraph in the corpus. [measured]
#
# The threshold is unchanged at 10 characters. Only the field it is applied to
# moved, from the title to the anchor the encoder is handed.
PHASE1B_MIN_ANCHOR_TEXT_LENGTH: Final[int] = 10
```

Delete `PHASE1B_DROPPED_FRAMEWORKS` and `PHASE1B_MIN_SECTION_TEXT_LENGTH`
entirely. A constant left in place with no reader is the decorative-control
defect from ledger lesson 4.

- [ ] **Step 5: Change the filter**

```python
# tract/training/data_quality.py — replace the imports and the two functions

from tract.config import PHASE1B_MIN_ANCHOR_TEXT_LENGTH, TRAINING_DIR
from tract.text_selection import ProseIndex


def assign_quality_tier(
    link: dict[str, str], resolved_text: str | None = None,
) -> QualityTier:
    """Assign a quality tier to a single hub link.

    `resolved_text` is the anchor the encoder will be handed for this link,
    from ProseIndex. A link is dropped when that text is thin, not when its
    section title is: the title is a label the model never sees, and gating on
    it dropped 155 links from two frameworks whose parsers now supply
    paragraphs, plus 64 more that already had one.
    """
    text = (resolved_text or link.get("section_name", "")).strip()
    if len(text) < PHASE1B_MIN_ANCHOR_TEXT_LENGTH:
        return QualityTier.DROPPED

    if link.get("standard_name", "") in AI_FRAMEWORK_NAMES:
        return QualityTier.T1_AI

    if link.get("link_type", "") == "AutomaticallyLinkedTo":
        return QualityTier.T3

    return QualityTier.T1


def filter_training_links(
    links: list[dict[str, str]], index: ProseIndex | None = None,
) -> list[TieredLink]:
    """Filter links by the resolved anchor and assign tier metadata.

    Returns non-DROPPED links with their tier assignment.
    """
    result: list[TieredLink] = []
    tier_counts: dict[QualityTier, int] = {t: 0 for t in QualityTier}

    for link in links:
        selection = (
            index.lookup(
                link.get("standard_name", ""), link.get("section_id"),
                link.get("section_name"),
            )
            if index is not None else None
        )
        tier = assign_quality_tier(link, selection.text if selection else None)
        tier_counts[tier] += 1
        if tier != QualityTier.DROPPED:
            result.append(TieredLink(link=link, tier=tier))

    for tier, count in tier_counts.items():
        logger.info("Quality tier %s: %d links", tier.value, count)

    return result
```

```python
# tract/training/data_quality.py — in load_and_filter_curated_links, replace
# the filter call
    filtered = filter_training_links(raw_links, ProseIndex.load())
```

Delete `_has_descriptive_text`.

- [ ] **Step 6: Run the tests and record the after state**

```bash
pytest tests/test_data_quality.py -q
mypy tract/training/data_quality.py tract/config.py --strict
grep -rn "PHASE1B_DROPPED_FRAMEWORKS\|PHASE1B_MIN_SECTION_TEXT_LENGTH" \
     tract/ parsers/ scripts/ tests/ docs/ || echo "no readers left"
PYTHONPATH=. "$PY" - <<'PYEOF'
from tract.training.data_quality import load_and_filter_curated_links
links, _ = load_and_filter_curated_links()
print("training links after:", len(links))
PYEOF
```

Expected: `training links after: 4402`. **[derived]** If it reads 4,405 the
new gate is not firing at all and the three genuinely thin links are being
kept; if it reads 4,127 the index is not reaching the filter.

- [ ] **Step 7: Regenerate the training link file and commit**

```bash
PYTHONPATH=. "$PY" - <<'PYEOF'
from tract.training.data_quality import (
    load_and_filter_curated_links, save_training_links,
)
links, raw_hash = load_and_filter_curated_links()
print("wrote", len(links), "links, output hash",
      save_training_links(links, raw_hash)[:16])
PYEOF
git add tract/config.py tract/training/data_quality.py \
        tests/test_data_quality.py data/training/hub_links_training.jsonl
git commit -m "fix: drop a link on the text the model sees, not on its section title

Training links move from 4,127 to 4,402 of 4,405 curated. The 155 dropped by
the framework deny list and 120 of the 123 dropped by the short-title floor now
resolve to parsed prose. The three that remain dropped are two NIST 800-53
links and one CWE link whose anchor is still under ten characters."
```

---

### Task 15: Rebuild the corpus, and prove only the eleven changed

The previous plan re-ran all 31 parsers and committed `data/processed/`
wholesale, silently re-running CAPEC and CWE — 42.8% and 14.5% of the training
graph — against sources it described as moving targets. This task rebuilds from
the pinned bytes on disk and proves what changed.

**The rollback artifact already exists.** `data/processed/pre_rebuild_control_hashes.json`
landed at commit `6819a7f` and holds the sha256 of every control's description
across all 31 frameworks: **4,222 hashes**. **[measured]** It is what turns
"the corpus changed" into a per-control list.

**Two things measured in advance, so the rebuild has an expectation to fail
against.** Re-running the 19 importable parsers into a scratch directory at
`8cf44b3` produces **1,897 controls, 0 of which differ from the baseline
hash**, and 10 controls absent from the baseline — `owasp_llm_top10_2026`,
which landed after the baseline was taken. **[measured]**

Three frameworks produce byte-different JSON while producing **identical
control text**: `asvs`, `owasp_cheat_sheets` and `owasp_ml_top10`. The whole
diff is the `source_files` block, because their raw archives were re-fetched
and re-pinned since the tracked JSON was written. **[measured]** That is the
distinction a byte-identity assertion cannot make and a description-hash
comparison can, which is why the assertion below is on control text.

**`defusedxml` is not installed** in the 3.12 environment, so `parse_capec.py`
and `parse_cwe.py` cannot be imported there. **[measured]** It is pinned at
`0.7.1` in `requirements.txt`; install it rather than skipping those two, or
the rebuild cannot state anything about 57.3% of the training graph.

**Files:**
- Create: `scripts/rebuild_corpus.py`
- Create: `tests/test_rebuild_corpus.py`

**Interfaces:**
- Consumes: every `parsers/parse_*.py`; `data/processed/pre_rebuild_control_hashes.json`.
- Produces: `rebuild_corpus(output_dir: Path, baseline_path: Path) -> RebuildReport` with `RebuildReport(changed: list[str], added: list[str], removed: list[str], unchanged: int, failed: dict[str, str])`.

- [ ] **Step 1: Install the missing pinned dependency**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"$PY" -m pip install "defusedxml==0.7.1" "openpyxl==3.1.5"
"$PY" -c "import defusedxml, openpyxl, pdfplumber, yaml, bs4; print('ok')"
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_rebuild_corpus.py — create

"""A corpus rebuild must say which controls changed text, not just that it ran.

data/processed/pre_rebuild_control_hashes.json holds 4,222 description hashes
captured at 6819a7f for exactly this. Three frameworks produce byte-different
JSON while producing identical control text, because their raw archives were
re-pinned; a byte-identity assertion cannot tell that from a real change.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.rebuild_corpus import RebuildReport, diff_against_baseline


def test_identical_text_reports_no_change(tmp_path: Path) -> None:
    baseline = {"demo:C-1": _sha("statement one")}
    parsed = {"demo": [{"control_id": "C-1", "description": "statement one"}]}
    report = diff_against_baseline(parsed, baseline)
    assert report.changed == []
    assert report.unchanged == 1


def test_changed_text_is_named(tmp_path: Path) -> None:
    baseline = {"demo:C-1": _sha("statement one")}
    parsed = {"demo": [{"control_id": "C-1", "description": "statement two"}]}
    report = diff_against_baseline(parsed, baseline)
    assert report.changed == ["demo:C-1"]
    assert report.unchanged == 0


def test_new_and_removed_controls_are_named(tmp_path: Path) -> None:
    baseline = {"demo:C-1": _sha("one"), "demo:C-2": _sha("two")}
    parsed = {"demo": [
        {"control_id": "C-1", "description": "one"},
        {"control_id": "C-3", "description": "three"},
    ]}
    report = diff_against_baseline(parsed, baseline)
    assert report.added == ["demo:C-3"]
    assert report.removed == ["demo:C-2"]


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
pytest tests/test_rebuild_corpus.py -q
```

Expected: FAIL, `ModuleNotFoundError: No module named 'scripts.rebuild_corpus'`.

- [ ] **Step 4: Write the rebuild script**

```python
# scripts/rebuild_corpus.py — create

"""Re-run every parser into a scratch directory and diff the control text.

The point is not to rebuild. It is to be able to say, per control, what
changed. Re-running all 31 parsers and committing data/processed wholesale
re-runs CAPEC and CWE, which are 42.8% and 14.5% of the training graph, and
nothing downstream can tell a corrected statement from a silently re-fetched
one.

data/processed/pre_rebuild_control_hashes.json holds the sha256 of every
control's description across 31 frameworks, captured at 6819a7f. Every parser
reads pinned bytes already on disk and every parser has a digest gate, so a
control whose hash moves is a parser change, not a source change -- and this
script names it.

Three frameworks produce byte-different JSON with identical control text
because their raw archives were re-pinned after the tracked JSON was written.
The comparison is therefore on description text, not on file bytes.

    PYTHONPATH=. "$PY" scripts/rebuild_corpus.py --dry-run
    PYTHONPATH=. "$PY" scripts/rebuild_corpus.py --commit
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tract.config import PARSERS_DIR, PROCESSED_FRAMEWORKS_DIR, PROCESSED_DIR
from tract.parsers.base import BaseParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASELINE_PATH = PROCESSED_DIR / "pre_rebuild_control_hashes.json"


@dataclass
class RebuildReport:
    changed: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: int = 0
    failed: dict[str, str] = field(default_factory=dict)


def _parser_classes() -> dict[str, type[BaseParser]]:
    """Every concrete parser class, keyed by framework_id.

    Raises:
        ValueError: If a parse_*.py module defines no BaseParser subclass.
    """
    classes: dict[str, type[BaseParser]] = {}
    for path in sorted(PARSERS_DIR.glob("parse_*.py")):
        module = importlib.import_module(f"parsers.{path.stem}")
        found = [
            value for value in vars(module).values()
            if isinstance(value, type)
            and issubclass(value, BaseParser)
            and value is not BaseParser
            and value.__module__ == module.__name__
        ]
        if not found:
            raise ValueError(
                f"{path.name} defines no BaseParser subclass. Every parser "
                f"module must, or the rebuild silently skips a framework and "
                f"reports its controls as unchanged because it never ran one."
            )
        classes[found[0].framework_id] = found[0]
    return classes


def run_all(output_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Run every parser into `output_dir`. Returns (controls, failures)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed: dict[str, list[dict[str, Any]]] = {}
    failed: dict[str, str] = {}
    for framework_id, parser_class in sorted(_parser_classes().items()):
        try:
            result = parser_class(output_dir=output_dir).run()
        except Exception as error:  # noqa: BLE001 - reported, never swallowed
            failed[framework_id] = f"{type(error).__name__}: {error}"
            logger.error("%s FAILED: %s", framework_id, failed[framework_id])
            continue
        parsed[framework_id] = [
            control.model_dump(mode="json") for control in result.controls
        ]
    return parsed, failed


def diff_against_baseline(
    parsed: dict[str, list[dict[str, Any]]], baseline: dict[str, str],
) -> RebuildReport:
    """Which controls changed text, were added, or disappeared."""
    report = RebuildReport()
    seen: set[str] = set()
    for framework_id, controls in sorted(parsed.items()):
        for control in controls:
            key = f"{framework_id}:{control['control_id']}"
            seen.add(key)
            digest = hashlib.sha256(
                str(control.get("description") or "").encode("utf-8")
            ).hexdigest()
            if key not in baseline:
                report.added.append(key)
            elif baseline[key] != digest:
                report.changed.append(key)
            else:
                report.unchanged += 1
    report.removed = sorted(set(baseline) - seen)
    report.added.sort()
    report.changed.sort()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, default=Path("build/rebuild"))
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    parsed, failed = run_all(args.scratch)
    report = diff_against_baseline(parsed, baseline["sha256_of_description"])
    report.failed = failed

    logger.info(
        "rebuild: %d frameworks, %d unchanged controls, %d changed, %d added, "
        "%d removed, %d parser failure(s)",
        len(parsed), report.unchanged, len(report.changed), len(report.added),
        len(report.removed), len(failed),
    )
    by_framework: dict[str, int] = {}
    for key in report.changed:
        by_framework[key.split(":", 1)[0]] = by_framework.get(
            key.split(":", 1)[0], 0
        ) + 1
    for framework_id, count in sorted(by_framework.items()):
        logger.info("  changed text: %-26s %d", framework_id, count)
    if failed:
        raise SystemExit(
            f"{len(failed)} parser(s) failed: {sorted(failed)}. A rebuild that "
            f"skips a framework leaves the previous artifact in place while "
            f"reporting success."
        )

    if args.commit and not args.dry_run:
        for source in sorted(args.scratch.glob("*.json")):
            shutil.copy2(source, PROCESSED_FRAMEWORKS_DIR / source.name)
        logger.info("copied %d artifact(s) into %s",
                    len(list(args.scratch.glob("*.json"))),
                    PROCESSED_FRAMEWORKS_DIR)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests and typecheck**

```bash
pytest tests/test_rebuild_corpus.py -q
mypy scripts/rebuild_corpus.py --strict
```

- [ ] **Step 6: Dry run and read the diff before committing anything**

```bash
PYTHONPATH=. "$PY" scripts/rebuild_corpus.py --dry-run
```

Expected shape, and every line of it must be explainable before proceeding:

- `0 parser failure(s)`.
- `changed text` reported **only** for the eleven frameworks in Tasks 3 through
  13. If `capec`, `cwe`, `asvs`, `owasp_cheat_sheets`, `nist_800_53`,
  `mitre_atlas` or any other framework appears in that list, stop. Their
  sources are pinned and their parsers were not touched, so a change there is a
  defect introduced by this plan.
- `added` should be the controls the new parsers emit beyond what the
  OpenCRE-derived stubs held: 194-183 for dsomm, 115-59 for wstg, 224-29 for
  csa_ccm, and so on.
- `removed` should be empty or small; a removal means a control id the previous
  artifact had and the parser does not produce.

Record the printed counts in the run ledger before Step 7.

- [ ] **Step 7: Commit the rebuild**

```bash
PYTHONPATH=. "$PY" scripts/rebuild_corpus.py --commit
PYTHONPATH=. "$PY" parsers/merge_all_controls.py
PYTHONPATH=. "$PY" parsers/validate_all.py
pytest tests/test_corpus_invariants.py tests/test_licensed_text_not_tracked.py \
       tests/test_holdout_framework.py tests/test_framework_licenses.py \
       tests/test_parser_manifest_coverage.py tests/test_prose_reachability.py -q
git status --porcelain data/processed/
```

`git status` must show no entry for `data/processed/frameworks/etsi.json` or
`data/processed/frameworks/iso_27001.json`. If it does, `.gitignore` is not
covering them in this checkout and licensed prose is one `git add` from the
history.

```bash
git add scripts/rebuild_corpus.py tests/test_rebuild_corpus.py \
        data/processed/frameworks/ data/processed/all_controls.json
git commit -m "chore: rebuild the corpus from pinned sources, with the per-control diff"
```

---

### Task 16: The AFTER report, and the acceptance tests that keep it true

**Files:**
- Create: `results/corpus/after_parsers.json`
- Create: `tests/test_corpus_acceptance.py`
- Modify: `tract/corpus_report.py` (add `JOIN_FLOORS`)
- Modify: `.superpowers/autonomous-run/RUN-LEDGER.md`

**Interfaces:**
- Consumes: everything above.
- Produces: `tract.corpus_report.JOIN_FLOORS: dict[str, float]`; `results/corpus/after_parsers.json`.

- [ ] **Step 1: Add the derived floors**

Every number below was derived from the curated link file and the source
**before** its parser was written, and each is the arithmetic ceiling stated in
its task, rounded down to two decimal places. None was read off the run it
gates.

```python
# tract/corpus_report.py — append

# Per-framework join floors, each derived in its parser's plan task from the
# curated link file and the pinned source BEFORE the parser existed. A floor
# pasted from the run it gates passes by construction and measures nothing.
#
#   dsomm       213/214 = 0.99533  one activity's statement is 11 characters
#   wstg        109/118 = 0.92373  nine links name ids absent from the archive
#   nist_800_63  78/79  = 0.98734  one section_id is the fragment "are g"
#   biml         21/21  = 1.00000  with the two declared alternates
#   enisa        68/68  = 1.00000  with Table 3, Annex C and name repair
#   etsi         36/36  = 1.00000  every technique declared to its own clause
#   csa_ccm      29/29  = 1.00000  seven renamed ids resolve by title
#   nist_ssdf    46/46  = 1.00000  with the two declared alt_ids
#   samm         30/30  = 1.00000
#   owasp_top10_2021     17/17 = 1.00000
#   owasp_proactive_controls 76/76 = 1.00000
#
# A floor of 1.00 where the ceiling is 1.00 is the right number, not an
# oversight: below it, a link that the source can answer stopped resolving.
JOIN_FLOORS: Final[dict[str, float]] = {
    "biml": 1.00,
    "csa_ccm": 1.00,
    "dsomm": 0.99,
    "enisa": 1.00,
    "etsi": 1.00,
    "nist_800_63": 0.98,
    "nist_ssdf": 1.00,
    "owasp_proactive_controls": 1.00,
    "owasp_top10_2021": 1.00,
    "samm": 1.00,
    "wstg": 0.92,
}
```

Add `from typing import Final` to that module's imports if it is not there.

- [ ] **Step 2: Write the acceptance tests**

```python
# tests/test_corpus_acceptance.py — create

"""What the eleven parsers had to be true for, expressed as a gate.

Every threshold here is a floor derived in the plan from the curated link file
and the pinned source before the parser was written, or a property the corpus
already had that must not regress. The instrument is tract.corpus_report, the
same one the per-parser steps used, so a parser cannot be accepted by a
measurement its consumer does not perform.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tract.corpus_report import (
    JOIN_FLOORS, build_corpus_report, check_join_floors,
)
from tract.text_selection import merged_corpus_path

BEFORE = Path("results/corpus/before_8cf44b3.json")

# 734 curated links across the eleven frameworks landed on 299 distinct title
# anchors before any of them had a parser. [measured]
PENDING = tuple(sorted(JOIN_FLOORS))


@pytest.fixture(scope="module")
def report():  # type: ignore[no-untyped-def]
    if not merged_corpus_path().exists():
        pytest.skip("needs the merged corpus")
    return build_corpus_report()


class TestJoinFloors:
    def test_every_framework_clears_its_derived_floor(self, report) -> None:  # type: ignore[no-untyped-def]
        assert check_join_floors(report, JOIN_FLOORS) == []

    def test_no_floor_exceeds_its_arithmetic_ceiling(self, report) -> None:  # type: ignore[no-untyped-def]
        """A floor above what the link data allows is a guaranteed failure.

        The previous plan carried three: dsomm 1.00 against a maximum of
        0.9953, wstg 0.96 against 0.9322, enisa 0.80 against a stated 0.721.
        """
        for framework_id, floor in JOIN_FLOORS.items():
            row = report.by_id(framework_id)
            assert floor <= 1.0
            assert row.links > 0
            assert row.resolution_rate >= floor, framework_id


class TestAnchorSeparation:
    def test_dsomm_stopped_collapsing_onto_its_sub_dimensions(self, report) -> None:  # type: ignore[no-untyped-def]
        row = report.by_id("dsomm")
        assert row.distinct_anchors >= 182
        assert row.links_per_anchor <= 1.20

    def test_no_new_framework_nests_an_anchor_inside_another(self, report) -> None:  # type: ignore[no-untyped-def]
        for framework_id in PENDING:
            assert report.by_id(framework_id).nested_anchors == 0, framework_id

    def test_biml_did_not_collapse_on_shared_labels(self, report) -> None:  # type: ignore[no-untyped-def]
        """Seven of 21 rows share a section_name across two documents."""
        row = report.by_id("biml")
        assert row.distinct_anchors == 20
        assert row.by_title == 1

    def test_etsi_registered_only_the_two_names_that_cannot_collide(
        self, report,
    ) -> None:  # type: ignore[no-untyped-def]
        """Three ETSI technique names span two clauses each.

        Registering all 24 as alternate titles keeps the resolution rate at
        1.0000 while two rows resolve to a clause they did not name.
        """
        assert report.by_id("etsi").by_title == 2

    def test_no_new_framework_carries_wrong_anchor_risk(self, report) -> None:  # type: ignore[no-untyped-def]
        for framework_id in PENDING:
            assert report.by_id(framework_id).wrong_anchor_risk == 0, framework_id


class TestNoRegression:
    def test_iso_still_resolves(self, report) -> None:  # type: ignore[no-untyped-def]
        """ISO reached 92/94 before this plan. Nothing here may cost it."""
        row = report.by_id("iso_27001")
        assert row.by_title + row.by_id >= 92
        assert row.distinct_anchors >= 91

    def test_the_frameworks_this_plan_did_not_touch_are_unchanged(
        self, report,
    ) -> None:  # type: ignore[no-untyped-def]
        if not BEFORE.exists():
            pytest.skip("no BEFORE artifact in this checkout")
        before = {
            row["framework_id"]: row
            for row in json.loads(BEFORE.read_text(encoding="utf-8"))["per_framework"]
        }
        for framework_id, previous in before.items():
            if framework_id in JOIN_FLOORS:
                continue
            current = report.by_id(framework_id)
            assert current.distinct_anchors == previous["distinct_anchors"], (
                framework_id
            )
            assert current.by_title + current.by_id == (
                previous["by_title"] + previous["by_id"]
            ), framework_id


class TestSpecAcceptance:
    """Spec Part 1.9, checked against stored text rather than the join flag."""

    def test_every_processed_framework_has_a_parser(self) -> None:
        from tract.config import PARSERS_DIR, PROCESSED_FRAMEWORKS_DIR

        parsed = {p.stem[len("parse_"):] for p in PARSERS_DIR.glob("parse_*.py")}
        written = {p.stem for p in PROCESSED_FRAMEWORKS_DIR.glob("*.json")}
        assert written - parsed == set()

    def test_no_version_field_says_opencre(self) -> None:
        from tract.config import PROCESSED_FRAMEWORKS_DIR

        for path in sorted(PROCESSED_FRAMEWORKS_DIR.glob("*.json")):
            version = json.loads(path.read_text(encoding="utf-8"))["version"]
            assert "opencre-" not in version, path.name

    def test_no_framework_reports_zero_honest_prose(self) -> None:
        from tract.config import PROCESSED_FRAMEWORKS_DIR
        from tract.parsers.base import BaseParser
        from tract.schema import Control

        for path in sorted(PROCESSED_FRAMEWORKS_DIR.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            controls = [Control(**c) for c in data["controls"]]
            assert BaseParser.honest_prose_fraction(controls) > 0.0, path.name
```

- [ ] **Step 3: Run the acceptance suite**

```bash
pytest tests/test_corpus_acceptance.py -q
```

Expected: PASS. Any failure names the framework and the column; fix the parser,
not the threshold.

- [ ] **Step 4: Capture the AFTER state**

```bash
PYTHONPATH=. "$PY" scripts/corpus_report.py --out results/corpus/after_parsers.json
PYTHONPATH=. "$PY" - <<'PYEOF'
import json

before = json.load(open("results/corpus/before_8cf44b3.json"))
after = json.load(open("results/corpus/after_parsers.json"))
rows = {r["framework_id"]: r for r in before["per_framework"]}

print(f"{'framework':26s} {'resolved':>18s} {'anchors':>16s} {'l/a':>13s}")
for row in after["per_framework"]:
    old = rows.get(row["framework_id"], {})
    print(
        f"{row['framework_id']:26s} "
        f"{old.get('by_title', 0) + old.get('by_id', 0):8d} -> "
        f"{row['by_title'] + row['by_id']:5d} "
        f"{old.get('distinct_anchors', 0):8d} -> {row['distinct_anchors']:5d} "
        f"{old.get('links_per_anchor', 0):6.2f} -> {row['links_per_anchor']:5.2f}"
    )
t_old, t_new = before["totals"], after["totals"]
print(f"\nresolved {t_old['by_title'] + t_old['by_id']} -> "
      f"{t_new['by_title'] + t_new['by_id']} of {t_new['links']}")
print(f"distinct anchors {t_old['distinct_anchors']} -> {t_new['distinct_anchors']}")
print(f"truncated {t_old['truncated']} -> {t_new['truncated']}")
print(f"not indexed {t_old['dropped_by_prose_rule']} -> "
      f"{t_new['dropped_by_prose_rule']}")
PYEOF
```

Expected, derived by summing each task's stated ceiling onto the BEFORE totals
**[derived]**:

| total | before | after |
|---|---|---|
| links resolved | 3,666 | 4,389 of 4,405 |
| distinct anchors | 1,450 | 1,902 |
| controls not in the prose index | 522 | about 47 |

The 723 newly resolved links come from dsomm 213, wstg 109, nist_800_63 78,
owasp_proactive_controls 76, enisa 68, nist_ssdf 46, etsi 36, samm 30,
csa_ccm 29, biml 21, owasp_top10_2021 17. The 452 new anchors come from
dsomm 182, wstg 55, nist_ssdf 44, enisa 33, samm 30, csa_ccm 29,
nist_800_63 25, biml 20, etsi 14, owasp_proactive_controls 10,
owasp_top10_2021 10.

One framework moves the wrong way on the anchor column and it was declared in
advance: `etsi` falls from 24 fallback anchors to 14 clause anchors. Two are
flat at 10 each, `owasp_proactive_controls` and `owasp_top10_2021`, because
their sources define ten mapping units. `truncated` rises, mostly from wstg and
etsi; record the number rather than asserting one.

- [ ] **Step 5: Write the result into the run ledger**

Append a `## Phase A-parsers COMPLETE` block to
`.superpowers/autonomous-run/RUN-LEDGER.md` carrying, at minimum: the
before/after totals from Step 4, the per-framework `distinct_anchors` change,
the two frameworks whose anchor count fell and why, the training-link count from
Task 14 (4,127 → 4,402), the rebuild diff from Task 15, and the sha256 of both
corpus report artifacts. Every number tagged `[measured]` or `[derived]`.

- [ ] **Step 6: Full suite, typecheck, commit**

```bash
pytest tests/ -q
mypy tract/ parsers/ scripts/ --strict
git add results/corpus/after_parsers.json tests/test_corpus_acceptance.py \
        tract/corpus_report.py .superpowers/autonomous-run/RUN-LEDGER.md
git commit -m "test: gate the corpus on anchors, not only on links resolved"
```

---

## Self-review

### Spec coverage

Spec Part 1 items, and where each is discharged.

| spec item | task |
|---|---|
| 1.1 the twelve frameworks with no parser | Tasks 3-13 cover eleven; ISO 27001 landed earlier and Task 16 asserts it did not regress |
| 1.2 source manifest per parser | every parser reads through `read_source`/`read_source_bytes`; `tests/test_parser_manifest_coverage.py` is run in Task 15 Step 7 |
| 1.2 `expected_count` raises | every parser declares one, and `csa_ccm`'s corrected 224 is the reason the task exists |
| 1.2 `min_prose_fraction` on stored text | every parser declares one, each derived from a measured statement-length distribution |
| 1.2 no clock | every parser declares `fetched_date` as a `ClassVar` |
| 1.3 repair layer with an audit record | Tasks 7 and 12 are the only text-moving transforms and both write `write_repair_audit` |
| 1.4 the thirteen parsers | eleven here; ISO 27001 and OWASP LLM Top 10 2026 already landed |
| 1.5 retire both gates | Task 14, with the measured 4,127 → 4,402 |
| 1.6 OWASP LLM Top 10 2026 | out of scope by instruction; Task 15's rebuild reports its 10 controls as `added` because they postdate the baseline |
| 1.7 ground-truth divergence | not in scope for this plan; it is a CLI path change, not a parser |
| 1.8 review status vs rebuild | not in scope; Task 15 supplies the per-control changed list the schema column would key on |
| 1.9 acceptance tests | Task 16 `TestSpecAcceptance`, plus the parse-twice determinism test in Task 3 |

Ledger lessons, and where each binds: lesson 1 (another open channel) — Task 13
Steps 1 and 8 check the ETSI routing on both the artifact and the merge;
lesson 2 (read the file first) — every snippet is written against `8cf44b3` and
Task 9 Step 1 re-verifies a premise the previous plan asserted; lesson 3 (a gate
that cannot fire) — every floor is derived from the link data's arithmetic
maximum and Task 16 asserts no floor exceeds it; lesson 4 (a decorative control)
— Task 14 deletes both retired constants rather than leaving them unread;
lesson 5 (baseline captured differently) — Task 1 Step 9 checks that
`merged_corpus_path()` found the licensed overlay before the BEFORE state is
trusted; lesson 6 (a step preceding what rewrites its inputs) — Task 1 precedes
every parser, Task 2 precedes Tasks 9 and 12, Tasks 3-13 precede Task 14, and
Task 14 precedes Task 15; lesson 7 (a fabricating transform) — Tasks 7 and 12
write audit records and Task 13 refuses the technique-level segmentation it
cannot verify; lesson 8 (a number without an artifact) — every number carries a
tag.

Premortem Criticals: C4 is closed by the derived floors and by
`test_no_floor_exceeds_its_arithmetic_ceiling`; C5 by Task 11's three named
name-repair defects and the per-row merge; C6 by document-scoped BIML titles,
ETSI declaring only the two alternates that cannot collide, and the
resolution-order table in the Global
Constraints; C7 by Task 9 Step 1, which re-measures the premise before any code
is written; C8 by Tasks 7 and 12; C10 by Task 1. C1, C2, C3 and C9 landed before
this plan and are asserted rather than rebuilt.

### Placeholder scan

No step says "similar to Task N". No function body raises `NotImplementedError`
or contains `...`. No `expected_count` is `0` — the previous plan shipped two,
relying on `_check_expected_count` to raise, which makes the plan unexecutable
without an out-of-band measurement step. Every count here is stated:
dsomm 194, samm 30, owasp_top10_2021 10, owasp_proactive_controls 10,
wstg 115, csa_ccm 224, nist_ssdf 42, nist_800_63 100 (floor), enisa 50,
biml 146 (floor), etsi 25.

No step pauses for an owner decision. Four judgement calls are ruled here with
the evidence stated: the CCM domain statement (member titles, not member
specifications), the SAMM statement composition (`shortDescription`, not
`longDescription`), BIML's `output:2` (resolved by name, audited), and ETSI's
grain (clause, with the anchor-count regression declared in advance).

One command in the plan installs software: `pip install "openpyxl==3.1.5"` and
`"defusedxml==0.7.1"`, both pinned, both recorded in `requirements.txt`.

### Type consistency

`FrameworkJoin` and `CorpusReport` are dataclasses with fully annotated fields;
`build_corpus_report` returns `CorpusReport`; `check_join_floors` returns
`list[str]`. Every parser's `parse()` returns `list[Control]`. The two
class-method entry points that tests call directly return what their task's
Interfaces block declares: `DsommParser.activities_to_controls -> list[Control]`,
`SammParser.build_controls -> list[Control]`,
`OwaspTop102021Parser.control_from_markdown -> Control`,
`WstgParser.build_controls -> tuple[list[Control], list[dict[str, object]]]`,
`CsaCcmParser.rows_to_controls -> list[Control]`,
`NistSsdfParser.rows_to_controls -> list[Control]`,
`Nist80063Parser.sections_from_html -> list[Control]`,
`EnisaParser.rows_to_units -> list[tuple[str, str]]`,
`BimlParser.build_controls -> tuple[list[Control], list[dict[str, object]]]`,
`EtsiParser.clauses_from_text -> dict[str, tuple[str, str]]` and
`EtsiParser.build_controls(clauses, alternates_by_name: dict[str, str]) -> list[Control]`.

`Control.metadata` is `dict[str, str | list[str]] | None`, so every metadata
literal in this plan uses only `str` and `list[str]` values; the `alt_ids` and
`alt_titles` channels both read `list[str] | str`. `expected_sha256` is
`ClassVar[str | None]` on ten parsers and `ClassVar[dict[str, str] | None]` on
BIML, which reads two files. `mypy tract/ parsers/ scripts/ --strict` is run in
Task 16 Step 6 over everything at once, not only over the file each task
touched.

### What this plan does not close

- **CAPEC and CWE are untouched and remain 57.3% of the training graph.** The
  ceiling study measured human alpha-1 at **0.181 on 83 CAPEC items** against
  0.572 pooled. Nothing here improves that; it is a label-quality problem, not
  a parser problem, and it belongs to the training-mix weighting the spec's
  Part 5 will have to decide.
- **`owasp_cheat_sheets` still carries 391 links on 49 anchors with 384 of them
  truncated.** **[measured]** It has a parser, so it is out of this plan's
  scope, and it is now the worst concentration in the corpus by a wide margin.
  The AFTER report will say so.
- **ETSI's anchor count falls from 24 to 14.** Declared, not discovered, and
  the alternative was prose-heuristic segmentation of technique names that
  appear mid-sentence in 9 of 24 cases.
- **Nine WSTG links, one NIST 800-63 link and one DSOMM link remain
  unresolvable.** Each has a named upstream cause and none is repairable from
  the source.
