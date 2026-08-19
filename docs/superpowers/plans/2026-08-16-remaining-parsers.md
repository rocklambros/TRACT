# The Eleven Remaining Framework Parsers

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the eleven remaining OpenCRE-synthesised framework files with real parsers reading real primary sources, then prove it by turning the three xfail-marked corpus invariants green and reporting, per framework, how many of OpenCRE's links now resolve to prose.

**Architecture:** Each parser subclasses `BaseParser`, reads every input through `read_source`/`read_source_bytes`, and declares `expected_count`, `min_prose_fraction`, and `fetched_date`. Three sources need machinery that does not exist yet: a rowspan merge in `tract/parsers/repair.py` for the two PDF table sources, an `alt_ids` channel on `ProseIndex` mirroring the existing `alt_titles` channel for the three frameworks whose OpenCRE anchors use retired identifiers, and a link-resolution report that measures the join instead of assuming it.

**Tech Stack:** Python 3.13, pydantic v2, pyyaml, beautifulsoup4, lxml, pdfplumber, openpyxl, pytest, mypy --strict.

**Spec:** `docs/superpowers/specs/2026-08-15-semantic-rebuild-design.md` (v2), Part 1.4, Part 1.5, Part 1.9.

**Predecessor:** `docs/superpowers/plans/2026-08-15-parser-contract-and-iso.md` landed the contract, the repair layer, and the ISO parser. This plan is its Task 9 work item.

**Source structures:** `.superpowers/autonomous-run/source-structures.md`. Where this plan's measurements disagree with that document, this plan wins and says so at the point of disagreement.

---

## Global Constraints

Copied verbatim from the assignment. Every task inherits all of them.

- Every parser subclasses `BaseParser` and MUST read inputs through `read_source`/`read_source_bytes`. A direct file open is invisible to the source manifest. `run()` raises on an empty manifest.
- Every parser declares: `framework_id`, `framework_name`, `version`, `source_url`, `mapping_unit_level`, `expected_count`, `fetched_date` (from `data/processed/framework_sources.json`, never the clock), and `min_prose_fraction`.
- `expected_count_is_floor = True` only where the source genuinely grows between releases.
- Repairs come from `tract/parsers/repair.py` with declared two-sided expected counts. Any NEW repair must return a count and emit an audit record if it moves or synthesizes text.
- `data/raw/` is immutable. `data/processed/frameworks/*.json` is tracked EXCEPT restricted frameworks in `RESTRICTED_FRAMEWORK_IDS` (tract/config.py).
- **csa_ccm licensing is RESOLVED by owner decision, 2026-08-16: the CCM is redistributable.** `csa_ccm` does not enter `RESTRICTED_FRAMEWORK_IDS`, which keeps ISO 27001 as its only member. `data/processed/frameworks/csa_ccm.json` is an ordinary tracked artifact, its prose may reach the tracked `all_controls.json`, and there is no gitignore entry and no licensed-overlay routing for it. This supersedes the "measure and stop for an owner decision" instruction the original brief carried. What survives from that instruction is a correctness question rather than a licensing one, and it lives in Task 13: the staged workbook is v4.1.0 while OpenCRE's 29 links may be keyed to v4.0, so the id overlap is measured and reported.
- Type everything, mypy --strict. Fail loud. No AI attribution. No em dashes or semicolons in comments or commit messages. Tests run with bare `pytest`.
- **No task step pauses for a decision.** Where a judgment is needed the task states the default, states the evidence that would overturn it, and keeps going. A plan that stops halfway is not executable.

Inherited from the predecessor plan and CLAUDE.md:

- **All inference and training runs on RunPod, never locally.** Nothing in this plan loads a model, so all of it runs locally.
- **Atomic writes only**, via `tract.io.atomic_write_json`. `BaseParser.run()` already does this.
- **Deterministic output.** Sorted keys, no clock reads in any written artifact. Re-parsing the same bytes must produce the same bytes.
- Run tests with `pytest` (resolves to Python 3.12 on this machine). `python3 -m pytest` fails with "No module named pytest". The same interpreter must be used for the measurement snippets in this plan: `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`. The homebrew `python3` on PATH is 3.13 and has none of the dependencies installed. Every `python3 - <<'PY'` block below assumes the 3.12 interpreter, so export it once per shell: `export PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`.
- **Consider all available prose, always.** Every parser below anchors on the source's own control statement, never on its title.

## Every number in this plan is labelled

`[measured]` means it was produced by running code against the pinned source on 2026-08-16 while writing this plan, and the snippet that produced it is in the task. `[declared]` means the source states it about itself. `[to measure]` means the implementer produces it and writes it into the parser, and the task says exactly how.

## Task order

By measured training-link value, which is what each parser buys. The two enabling tasks that several parsers depend on are inserted immediately before their first consumer, and the instrument that measures the outcome lands before the first parser so its baseline is captured under identical conditions.

| # | task | links | notes |
|---|---|---|---|
| 1 | Pin openpyxl, promote pdfplumber | - | unblocks tasks 9 and 14 |
| 2 | Link-resolution report and the BEFORE baseline | - | the instrument tasks 18 and 19 re-run |
| 3 | DSOMM | 176 | zip, YAML, composite prose |
| 4 | WSTG | 118 | zip, 130 markdown files, tombstones |
| 5 | NIST SP 800-63B | 79 | HTML, currently dropped |
| 6 | OWASP Proactive Controls | 76 | zip, decoy directories, currently dropped |
| 7 | `repair.merge_spanned_rows` | - | unblocks tasks 8 and 9 |
| 8 | ENISA | 68 | PDF tables, name-only join, hardest |
| 9 | NIST SSDF | 46 | PDF tables, retired-task stubs |
| 10 | ETSI | 35 | PDF prose, coarse section grain |
| 11 | SAMM | 30 | zip, YAML, two granularities |
| 12 | `ProseIndex.alt_ids` | - | unblocks tasks 13, 14, and fixes task 4's tombstones |
| 13 | CSA CCM | 29 | XLSX, two granularities, a version-drift measurement |
| 14 | BIML | 21 | two PDFs, colliding ids |
| 15 | OWASP Top 10 2021 | 17 | 196 MB archive, narrow extract |
| 16 | Corpus rebuild, merge, and the three xfail invariants | - | the completion signal |
| 17 | Link counts AFTER, and the Part 1.5 projection | - | records both numbers |
| 18 | Per-framework join-rate gate | - | a parser whose output nothing joins is a failure |

---

### Task 1: Pin the two dependencies the new parsers need

Four of the eleven parsers read PDFs and one reads an XLSX. `pdfplumber` is pinned in `requirements.txt` but sits in the `llm` extra of `pyproject.toml`, so an install from the distributable metadata does not get it. `openpyxl` is in neither file and is only present on this machine by accident.

The installed `pdfplumber` on this machine is 0.11.4 while `requirements.txt` pins 0.11.10. **[measured]** Table extraction results can differ between pdfplumber versions, and three tasks below declare exact counts measured against extracted tables, so the environment has to match the pin before those counts mean anything.

**Files:**
- Modify: `requirements.txt`
- Modify: `pyproject.toml`
- Test: `tests/test_parser_dependencies.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: an environment where `import openpyxl` and `import pdfplumber` both succeed at the pinned versions.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parser_dependencies.py — create

"""The parser dependencies must be declared, not merely installed.

pdfplumber sat in the llm extra of pyproject.toml while requirements.txt
pinned it, so an install from the distributable metadata did not get it and
four parsers in this plan need it. openpyxl was in neither file and was
present on the author's machine by accident.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_requirements_pins_both_parser_dependencies() -> None:
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "openpyxl==" in text, "the CSA CCM parser reads an XLSX workbook"
    assert "pdfplumber==" in text, "four parsers in Plan 1b read PDFs"


def test_pyproject_declares_both_as_core_dependencies() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    core = text.split("[project.optional-dependencies]")[0]
    assert "openpyxl" in core
    assert "pdfplumber" in core, (
        "pdfplumber moved out of the llm extra when it stopped being optional"
    )


def test_both_import_at_the_pinned_versions() -> None:
    import openpyxl
    import pdfplumber

    assert openpyxl.__version__.startswith("3.1.")
    assert pdfplumber.__version__ == "0.11.10", (
        "table extraction results are version sensitive and three parsers "
        "declare exact counts measured against extracted tables"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_parser_dependencies.py -v`
Expected: FAIL on the pyproject assertion and on the pdfplumber version assertion (installed 0.11.4).

- [ ] **Step 3: Add the pin**

```
# requirements.txt — add below the pdfplumber pin
# The CSA CCM source is an XLSX workbook. Read-only, values-only, so none of
# openpyxl's formula or chart surface is reachable from parser input.
openpyxl==3.1.5
```

Change the `pdfplumber` comment in `requirements.txt` to name all five readers:

```
# parse_nist_ai_100_2.py, parse_enisa.py, parse_nist_ssdf.py, parse_etsi.py
# and parse_biml.py read PDFs. Pinned like everything else here: pyproject.toml
# carries floors for the distributable, this file carries the exact
# environment. Table extraction results are version sensitive.
pdfplumber==0.11.10
```

- [ ] **Step 4: Promote both in pyproject.toml**

```toml
# pyproject.toml — [project] dependencies, append
    "pdfplumber>=0.11.10",
    "openpyxl>=3.1.5,<4",
```

Remove `"pdfplumber>=0.10.0",` from the `llm` extra. It is no longer optional.

```toml
# pyproject.toml — [[tool.mypy.overrides]] module list, add
    # openpyxl ships no py.typed marker. Same posture as pdfplumber: the
    # import is unchecked, the code around it is not.
    "openpyxl.*",
```

- [ ] **Step 5: Install the pinned versions and re-run**

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pip install \
    'pdfplumber==0.11.10' 'openpyxl==3.1.5'
pytest tests/test_parser_dependencies.py -v
mypy tract/ parsers/ --strict
```

Expected: PASS, no mypy errors.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pyproject.toml tests/test_parser_dependencies.py
git commit -m "build: pin openpyxl and promote pdfplumber out of the llm extra"
```

---

### Task 2: The link-resolution report, and the BEFORE baseline

Nothing currently measures whether a parsed control is reachable from the link that needed it. A parser can emit 194 clean controls that join to nothing, and every gate in `BaseParser.run()` passes. This task lands the instrument first so its baseline is captured under identical conditions, which is the Plan 1 lesson that a baseline measured a different way masks regressions.

The report answers three questions per framework: how many curated links exist, how many survive today's two quality gates, and how many resolve through `ProseIndex` to real prose and by which channel.

**Files:**
- Create: `scripts/phase1b/report_link_resolution.py`
- Create: `tests/test_report_link_resolution.py`
- Create: `data/processed/link_resolution_before.json` (generated, tracked)

**Interfaces:**
- Consumes: `tract.text_selection.ProseIndex`, `tract.training.data_quality.assign_quality_tier`.
- Produces:
  - `FrameworkResolution` TypedDict with keys `framework_id`, `standard_name`, `curated`, `kept_by_gates`, `resolved_by_title`, `resolved_by_id`, `unresolved`, `unresolved_section_ids`.
  - `resolve_links(links: list[dict[str, str]], index: ProseIndex) -> dict[str, FrameworkResolution]`
  - `main(argv: list[str] | None = None) -> int`, writing the report to `--output`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report_link_resolution.py — create

"""The report that says whether a parser's output is reachable.

A parser can emit 194 clean controls that join to nothing and every gate in
BaseParser.run() passes. Counting links is not the same as counting links
that resolve, and only the second number says the parse was useful.
"""

from __future__ import annotations

from typing import Any

from scripts.phase1b.report_link_resolution import resolve_links
from tract.text_selection import ProseIndex


def _index() -> ProseIndex:
    records: list[dict[str, Any]] = [{
        "framework_name": "DSOMM",
        "controls": [
            {
                "control_id": "2a44b708",
                "title": "Inventory of production components",
                "description": (
                    "A documented inventory of artifacts in production, such "
                    "as container images, is maintained and kept current by "
                    "the team that owns the deployment."
                ),
            },
        ],
    }]
    return ProseIndex(records)


class TestResolveLinks:
    def test_counts_a_link_that_resolves_by_id(self) -> None:
        links = [{
            "framework_id": "dsomm", "standard_name": "DSOMM",
            "section_id": "2a44b708", "section_name": "Deployment",
            "link_type": "AutomaticallyLinkedTo",
        }]
        report = resolve_links(links, _index())

        assert report["dsomm"]["resolved_by_id"] == 1
        assert report["dsomm"]["resolved_by_title"] == 0
        assert report["dsomm"]["unresolved"] == 0

    def test_counts_a_link_that_resolves_by_title(self) -> None:
        links = [{
            "framework_id": "dsomm", "standard_name": "DSOMM",
            "section_id": "no-such-id",
            "section_name": "Inventory of production components",
            "link_type": "LinkedTo",
        }]
        report = resolve_links(links, _index())

        assert report["dsomm"]["resolved_by_title"] == 1
        assert report["dsomm"]["resolved_by_id"] == 0

    def test_names_the_section_ids_that_resolve_to_nothing(self) -> None:
        links = [{
            "framework_id": "dsomm", "standard_name": "DSOMM",
            "section_id": "are g", "section_name": "are g",
            "link_type": "LinkedTo",
        }]
        report = resolve_links(links, _index())

        assert report["dsomm"]["unresolved"] == 1
        assert report["dsomm"]["unresolved_section_ids"] == ["are g"]

    def test_records_what_todays_quality_gates_keep(self) -> None:
        """kept_by_gates is the 4,127 number, per framework.

        Both gates test section_name, a title. A bare "C1" is two characters
        and dies on the short-title gate even though its control has prose.
        """
        links = [
            {
                "framework_id": "owasp_proactive_controls",
                "standard_name": "OWASP Proactive Controls",
                "section_id": "C1", "section_name": "C1",
                "link_type": "LinkedTo",
            },
            {
                "framework_id": "dsomm", "standard_name": "DSOMM",
                "section_id": "2a44b708",
                "section_name": "Inventory of production components",
                "link_type": "AutomaticallyLinkedTo",
            },
        ]
        report = resolve_links(links, _index())

        assert report["owasp_proactive_controls"]["kept_by_gates"] == 0
        assert report["dsomm"]["kept_by_gates"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_report_link_resolution.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.phase1b.report_link_resolution'`.

- [ ] **Step 3: Implement the report**

```python
# scripts/phase1b/report_link_resolution.py — create

"""Measure whether each curated OpenCRE link reaches real control prose.

Three numbers per framework, because three different things can be true at
once and only the third one says a parser was useful:

  curated        every link the curated file carries for this framework
  kept_by_gates  what PHASE1B_DROPPED_FRAMEWORKS and the short-title gate
                 leave behind, which is the 4,127 training links
  resolved       what ProseIndex can turn into a control statement, split by
                 the channel that resolved it

Spec Part 1.5 retires both gates in favour of the resolved anchor, so this
script produces the evidence that retirement rests on. It does not retire
anything itself.

Run before and after the corpus rebuild, with the same code, so the two
numbers are comparable. A baseline captured a different way is the Plan 1
defect that masked nine regressions.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Final, TypedDict

from tract.config import PROCESSED_DIR, PROCESSED_LICENSED_DIR, TRAINING_DIR
from tract.io import atomic_write_json
from tract.text_selection import ProseIndex
from tract.training.data_quality import QualityTier, assign_quality_tier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CURATED_PATH: Final[Path] = TRAINING_DIR / "hub_links_curated.jsonl"
MERGED_FILENAME: Final[str] = "all_controls.json"


class FrameworkResolution(TypedDict):
    """One framework's link accounting."""

    framework_id: str
    standard_name: str
    curated: int
    kept_by_gates: int
    resolved_by_title: int
    resolved_by_id: int
    unresolved: int
    unresolved_section_ids: list[str]


def _blank(framework_id: str, standard_name: str) -> FrameworkResolution:
    return FrameworkResolution(
        framework_id=framework_id,
        standard_name=standard_name,
        curated=0,
        kept_by_gates=0,
        resolved_by_title=0,
        resolved_by_id=0,
        unresolved=0,
        unresolved_section_ids=[],
    )


def resolve_links(
    links: list[dict[str, str]], index: ProseIndex,
) -> dict[str, FrameworkResolution]:
    """Account for every link, by framework and by resolution channel.

    Channels are probed separately rather than read off ProseIndex.lookup,
    which returns the text and not the path it took. Title first, matching
    lookup's own order, so the reported channel is the one lookup would use.
    """
    report: dict[str, FrameworkResolution] = {}
    unresolved: dict[str, set[str]] = defaultdict(set)

    for link in links:
        framework_id = link.get("framework_id", "")
        standard = link.get("standard_name", "")
        row = report.setdefault(framework_id, _blank(framework_id, standard))
        row["curated"] += 1
        if assign_quality_tier(link) is not QualityTier.DROPPED:
            row["kept_by_gates"] += 1

        section_id = link.get("section_id", "")
        section_name = link.get("section_name", "")
        if index.lookup(standard, None, section_name) is not None:
            row["resolved_by_title"] += 1
        elif index.lookup(standard, section_id, None) is not None:
            row["resolved_by_id"] += 1
        else:
            row["unresolved"] += 1
            unresolved[framework_id].add(section_id)

    for framework_id, ids in unresolved.items():
        report[framework_id]["unresolved_section_ids"] = sorted(ids)
    return report


def _load_links(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _corpus_path() -> Path:
    """Prefer the licensed overlay, matching merge_all_controls' read order.

    The tracked corpus excludes restricted frameworks, so a report built from
    it would score ISO and any other restricted framework as unresolvable and
    blame the parser for the licence.
    """
    overlay = PROCESSED_LICENSED_DIR / MERGED_FILENAME
    return overlay if overlay.exists() else PROCESSED_DIR / MERGED_FILENAME


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to write the JSON report")
    parser.add_argument("--links", type=Path, default=CURATED_PATH)
    args = parser.parse_args(argv)

    corpus = _corpus_path()
    index = ProseIndex.load(corpus)
    links = _load_links(args.links)
    report = resolve_links(links, index)

    # Written out key by key rather than by comprehension. A TypedDict cannot
    # be indexed by a variable under mypy --strict, and an ignore comment here
    # would hide a real key typo behind a silenced error.
    rows = list(report.values())
    totals = {
        "curated": sum(row["curated"] for row in rows),
        "kept_by_gates": sum(row["kept_by_gates"] for row in rows),
        "resolved_by_title": sum(row["resolved_by_title"] for row in rows),
        "resolved_by_id": sum(row["resolved_by_id"] for row in rows),
        "unresolved": sum(row["unresolved"] for row in rows),
    }
    payload = {
        # Name and a flag, never the absolute path. The report is tracked and
        # an absolute path makes its bytes differ per machine.
        "corpus": corpus.name,
        "licensed_overlay": corpus.parent.name == "licensed",
        "frameworks": {k: report[k] for k in sorted(report)},
        "totals": totals,
    }
    atomic_write_json(payload, args.output)

    logger.info(
        "curated %d, kept by gates %d, resolved %d (title %d, id %d), "
        "unresolved %d",
        totals["curated"], totals["kept_by_gates"],
        totals["resolved_by_title"] + totals["resolved_by_id"],
        totals["resolved_by_title"], totals["resolved_by_id"],
        totals["unresolved"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_report_link_resolution.py -v
mypy scripts/phase1b/report_link_resolution.py --strict
```

Expected: PASS, no mypy errors.

- [ ] **Step 5: Capture the BEFORE baseline**

```bash
python3 -m scripts.phase1b.report_link_resolution \
    --output data/processed/link_resolution_before.json
python3 - <<'PY'
import json, pathlib
d = json.loads(pathlib.Path("data/processed/link_resolution_before.json").read_text())
print("TOTALS", json.dumps(d["totals"], indent=2))
for fw in ("dsomm", "wstg", "nist_800_63", "owasp_proactive_controls", "enisa",
           "nist_ssdf", "etsi", "samm", "csa_ccm", "biml", "owasp_top10_2021"):
    r = d["frameworks"][fw]
    print(f"{fw:26s} curated={r['curated']:4d} kept={r['kept_by_gates']:4d} "
          f"title={r['resolved_by_title']:4d} id={r['resolved_by_id']:4d} "
          f"unresolved={r['unresolved']:4d}")
PY
```

Expected `totals["kept_by_gates"] == 4127`. **[measured]** If it is not 4,127, stop: the curated file or the gate constants moved since this plan was written, and every downstream count in it is stale.

Record the printed block verbatim in the commit message body. This is the BEFORE half of Task 17.

- [ ] **Step 6: Commit**

```bash
git add scripts/phase1b/report_link_resolution.py \
        tests/test_report_link_resolution.py \
        data/processed/link_resolution_before.json
git commit -m "feat: measure how many curated links reach real control prose"
```

---

### Task 3: DSOMM, 176 links

The highest-value single parser in this plan. The join key is a GUID that matches OpenCRE's `section_id` for **183 of 183** distinct link ids. **[measured]**

Two corrections to `source-structures.md`. First, it says `description` is "multi-paragraph prose". Measured across the 194 activities: `description` is present on **51** and absent on **143**, while `risk` and `measure` are present on **194** each. **[measured]** A parser anchoring on `description` alone would emit 143 controls with no text at all and fail pydantic's `min_length=1`. The prose is the concatenation of `description`, `risk`, and `measure`, which clears 60 characters on **192 of 194**. **[measured]** Second, it says the generated file flattens 26 sub-area files. There are **19** sub-dimensions. **[measured]**

`section_name` in the link data is the sub-dimension, not the activity title, and **no sub-dimension name collides with any activity title** **[measured]**, so the title channel resolves nothing here and the whole join rides on the GUID.

**Files:**
- Create: `parsers/parse_dsomm.py`
- Create: `tests/test_parse_dsomm.py`

**Interfaces:**
- Consumes: `BaseParser`, `Control`.
- Produces: `DsommParser` with `framework_id = "dsomm"`, and `DsommParser.compose_prose(body: Mapping[str, object]) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_dsomm.py — create

"""Tests for the DSOMM parser.

The fixture is a synthetic three-activity model with the same shape as the
real generated/model.yaml, including the case that matters most: an activity
with no description field at all, which is 143 of the real 194.
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path
from typing import ClassVar

import pytest

from parsers.parse_dsomm import DsommParser

MODEL_YAML = """\
meta:
  version: 4.3.1
  released: "2026-06-05"
  publisher: https://example.invalid/dsomm
---
Build and Deployment:
  Deployment:
    Inventory of production components:
      uuid: 2a44b708-734f-4463-b0cb-86dc46344b2f
      description: |
        A documented inventory is kept for every artifact in production.
      risk: An artifact carrying a critical vulnerability cannot be located.
      measure: Maintain an inventory of container images and their versions.
      level: 1
    Pinning of artifacts:
      uuid: f3c4971e-9f4d-4e59-8ed0-f0bdb6262477
      risk: Unauthorized manipulation of artifacts might be difficult to spot.
      measure: Pinning of artifacts ensures changes happen only when intended.
      level: 2
Culture and Organization:
  Design:
    Conduction of simple threat modeling:
      uuid: 0b74b2f4-2a1f-4a53-9c6b-2c4f9e6f4a11
      risk: Design flaws are found late, when they are expensive to correct.
      measure: Run a lightweight threat model on every significant change.
      level: 1
"""


class SampleDsommParser(DsommParser):
    """The parser with the fixture's counts rather than the source's."""

    expected_count: ClassVar[int] = 3
    min_prose_fraction: ClassVar[float] = 1.0


@pytest.fixture
def parser(tmp_path: Path) -> DsommParser:
    raw = tmp_path / "raw"
    raw.mkdir()
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("repo-abc123/generated/model.yaml", MODEL_YAML)
        archive.writestr("repo-abc123/README.md", "not the model")
    (raw / "dsomm_data.zip").write_bytes(buffer.getvalue())
    out = tmp_path / "out"
    out.mkdir()
    return SampleDsommParser(raw_dir=raw, output_dir=out)


class TestDsommParser:
    def test_control_id_is_the_bare_uuid(self, parser: DsommParser) -> None:
        """OpenCRE's section_id is the bare GUID.

        The synthesised file this replaces used "dsomm:<uuid>", which cannot
        match a link and is why the id channel resolved nothing.
        """
        ids = {c.control_id for c in parser.parse()}
        assert "2a44b708-734f-4463-b0cb-86dc46344b2f" in ids
        assert not any(cid.startswith("dsomm:") for cid in ids)

    def test_title_is_the_activity_name(self, parser: DsommParser) -> None:
        titles = {c.title for c in parser.parse()}
        assert "Inventory of production components" in titles

    def test_an_activity_without_a_description_still_carries_prose(
        self, parser: DsommParser,
    ) -> None:
        """143 of the real 194 activities have no description field."""
        controls = {c.control_id: c for c in parser.parse()}
        pinning = controls["f3c4971e-9f4d-4e59-8ed0-f0bdb6262477"]

        assert "Unauthorized manipulation" in pinning.description
        assert "only when intended" in pinning.description
        assert len(pinning.description) >= 60

    def test_the_subdimension_is_recorded_as_the_parent(
        self, parser: DsommParser,
    ) -> None:
        """The link's section_name is the sub-dimension, not the title.

        It resolves nothing on its own and must not be written into title,
        which would collapse dozens of activities onto one anchor.
        """
        controls = {c.control_id: c for c in parser.parse()}
        activity = controls["2a44b708-734f-4463-b0cb-86dc46344b2f"]

        assert activity.parent_name == "Deployment"
        assert activity.metadata is not None
        assert activity.metadata["dimension"] == "Build and Deployment"

    def test_a_version_that_does_not_match_the_declaration_raises(
        self, tmp_path: Path,
    ) -> None:
        """The pin is on a commit, so the model version cannot drift silently."""
        raw = tmp_path / "raw"
        raw.mkdir()
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "repo-abc123/generated/model.yaml",
                MODEL_YAML.replace("4.3.1", "9.9.9"),
            )
        (raw / "dsomm_data.zip").write_bytes(buffer.getvalue())
        out = tmp_path / "out"
        out.mkdir()

        with pytest.raises(ValueError, match="9.9.9"):
            SampleDsommParser(raw_dir=raw, output_dir=out).parse()

    def test_reads_the_archive_through_the_recording_reader(
        self, parser: DsommParser,
    ) -> None:
        parser.parse()
        assert "dsomm_data.zip" in parser._source_files


class TestComposeProse:
    def test_joins_the_three_fields_in_a_fixed_order(self) -> None:
        composed = DsommParser.compose_prose({
            "description": "The what.", "risk": "The why.",
            "measure": "The how.",
        })
        assert composed == "The what. The why. The how."

    def test_skips_absent_and_blank_fields(self) -> None:
        composed = DsommParser.compose_prose({
            "risk": "The why.", "measure": "  ", "description": None,
        })
        assert composed == "The why."

    def test_raises_when_no_field_carries_text(self) -> None:
        with pytest.raises(ValueError, match="no prose"):
            DsommParser.compose_prose({"level": 1})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_parse_dsomm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'parsers.parse_dsomm'`.

- [ ] **Step 3: Implement the parser**

```python
# parsers/parse_dsomm.py — create

"""Parser for the DevSecOps Maturity Model (DSOMM).

176 curated training links, the largest single recovery in this plan, and the
cleanest join in it: OpenCRE's section_id is the activity's own uuid and
matches 183 of 183 distinct link ids.

The prose is NOT the description field. Measured across the 194 activities in
the pinned commit, description is present on 51 and absent on 143, while risk
and measure are present on all 194. A parser anchoring on description alone
emits 143 controls with no text. The anchor is description plus risk plus
measure, which clears 60 characters on 192 of 194.

The link's section_name is the sub-dimension ("Deployment", "Process"), one
level above the activity, and dozens of distinct activities share one. It is
recorded as parent_name and never as title, because a title collision would
collapse them onto a single anchor.

Source: https://github.com/devsecopsmaturitymodel/DevSecOps-MaturityModel-data
"""
from __future__ import annotations

import logging
import zipfile
from collections.abc import Mapping
from io import BytesIO
from typing import ClassVar, Final, cast

import yaml

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARCHIVE_NAME: Final[str] = "dsomm_data.zip"
# The generated file already flattens the 19 per-sub-dimension YAML sources
# into one document, so this is the same data without a 19 file join.
MODEL_MEMBER_SUFFIX: Final[str] = "generated/model.yaml"
# The archive is downloaded from GitHub and CLAUDE.md treats framework data as
# untrusted input, so the member is bounded before it is read into memory.
# The real generated/model.yaml is 341,065 bytes.
MAX_MEMBER_BYTES: Final[int] = 2_000_000
# Order is fixed and meaningful. description says what the activity is, risk
# says why it matters, measure says what to do. Reordering them changes every
# anchor in the corpus and would not be visible in any count.
PROSE_FIELDS: Final[tuple[str, ...]] = ("description", "risk", "measure")


class DsommParser(BaseParser):
    framework_id: ClassVar[str] = "dsomm"
    # Must match what canonical_framework() resolves the link's standard_name
    # to. FRAMEWORK_NAME_ALIASES maps "devsecops maturity model (dsomm)" to
    # "DSOMM", so any other spelling here silently breaks every join.
    framework_name: ClassVar[str] = "DSOMM"
    version: ClassVar[str] = "4.3.1"
    source_url: ClassVar[str] = (
        "https://github.com/devsecopsmaturitymodel/DevSecOps-MaturityModel-data"
    )
    mapping_unit_level: ClassVar[str] = "activity"
    expected_count: ClassVar[int] = 194
    fetched_date: ClassVar[str] = "2026-08-15"
    # Measured 192 of 194 at or above HONEST_PROSE_MIN_CHARS. The two short
    # ones are genuinely one clause of risk plus one of measure.
    min_prose_fraction: ClassVar[float] = 0.98

    @staticmethod
    def compose_prose(body: Mapping[str, object]) -> str:
        """Join the activity's prose fields into one control statement.

        Raises:
            ValueError: If no field carries text. An activity with neither a
                risk nor a measure is a source defect, not a control, and a
                silent skip would drop it from the count check as well.
        """
        parts = [
            str(body[field]).strip()
            for field in PROSE_FIELDS
            if isinstance(body.get(field), str) and str(body[field]).strip()
        ]
        if not parts:
            raise ValueError(
                f"activity carries no prose in any of {PROSE_FIELDS}: "
                f"{sorted(body)}"
            )
        return " ".join(parts)

    def parse(self) -> list[Control]:
        model = self._load_model()
        controls: list[Control] = []
        for dimension, subdimensions in sorted(model.items()):
            for subdimension, activities in sorted(subdimensions.items()):
                for title, body in activities.items():
                    controls.append(
                        self._to_control(dimension, subdimension, title, body)
                    )
        logger.info("%s: parsed %d activities", self.framework_id, len(controls))
        return controls

    def _to_control(
        self,
        dimension: str,
        subdimension: str,
        title: str,
        body: Mapping[str, object],
    ) -> Control:
        uuid = body.get("uuid")
        if not isinstance(uuid, str) or not uuid.strip():
            raise ValueError(
                f"{self.framework_id}: activity {title!r} under "
                f"{dimension}/{subdimension} has no uuid. The uuid is the "
                f"OpenCRE join key, so an activity without one is unlinkable "
                f"and the source changed shape."
            )
        return Control(
            control_id=uuid.strip(),
            title=title.strip(),
            description=self.compose_prose(body),
            parent_id=subdimension,
            parent_name=subdimension,
            hierarchy_level="activity",
            metadata={"dimension": dimension},
        )

    def _load_model(self) -> dict[str, dict[str, dict[str, dict[str, object]]]]:
        """Read the one generated model document out of the archive."""
        payload = self.read_source_bytes(ARCHIVE_NAME)
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            names = [
                n for n in archive.namelist() if n.endswith(MODEL_MEMBER_SUFFIX)
            ]
            if len(names) != 1:
                raise ValueError(
                    f"{self.framework_id}: expected exactly one "
                    f"{MODEL_MEMBER_SUFFIX} member, found {len(names)}: "
                    f"{names}. The archive layout changed."
                )
            info = archive.getinfo(names[0])
            if info.file_size > MAX_MEMBER_BYTES:
                raise ValueError(
                    f"{names[0]}: declares {info.file_size} bytes, over the "
                    f"{MAX_MEMBER_BYTES} byte cap"
                )
            with archive.open(names[0]) as handle:
                raw = handle.read(MAX_MEMBER_BYTES + 1)
        if len(raw) > MAX_MEMBER_BYTES:
            raise ValueError(
                f"{names[0]}: expanded past the {MAX_MEMBER_BYTES} byte cap"
            )

        documents = list(yaml.safe_load_all(raw.decode("utf-8")))
        if len(documents) != 2:
            raise ValueError(
                f"{self.framework_id}: expected a meta document and a model "
                f"document, found {len(documents)}"
            )
        meta, model = documents
        self._check_version(meta)
        if not isinstance(model, dict):
            raise ValueError(
                f"{self.framework_id}: the model document is a "
                f"{type(model).__name__}, expected a mapping of dimensions"
            )
        return cast(
            "dict[str, dict[str, dict[str, dict[str, object]]]]", model
        )

    def _check_version(self, meta: object) -> None:
        """Refuse a model whose declared version is not the pinned one.

        The archive is pinned to a commit, so this can only fire when someone
        re-pins without updating the parser. That is exactly when a silent
        pass would ship 4.4.0 content under a 4.3.1 label.
        """
        declared = ""
        if isinstance(meta, dict):
            block = meta.get("meta")
            if isinstance(block, dict):
                declared = str(block.get("version", "")).strip()
        if declared != self.version:
            raise ValueError(
                f"{self.framework_id}: the model declares version "
                f"{declared!r} against the parser's {self.version!r}. Re-read "
                f"the changed activities before moving the literal."
            )


def main() -> None:
    DsommParser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_parse_dsomm.py -v
mypy parsers/parse_dsomm.py --strict
```

Expected: PASS, no mypy errors.

- [ ] **Step 5: Run against the real source and check the join**

```bash
python3 parsers/parse_dsomm.py
python3 - <<'PY'
import json, pathlib
d = json.loads(pathlib.Path("data/processed/frameworks/dsomm.json").read_text())
links = json.loads(
    pathlib.Path("data/training/hub_links_by_framework.json").read_text()
)["dsomm"]
ids = {c["control_id"] for c in d["controls"]}
link_ids = {l["section_id"] for l in links}
print("controls:", len(d["controls"]))
print("id join :", len(link_ids & ids), "of", len(link_ids))
print("prose   :", sum(1 for c in d["controls"] if len(c["description"]) >= 60),
      "of", len(d["controls"]))
PY
```

Expected: `controls: 194`, `id join : 183 of 183`, `prose : 192 of 194`. **[measured]** Any other id-join number means the uuid extraction or the archive pin changed, and the parser is wrong before the counts are.

- [ ] **Step 6: Commit**

```bash
git add parsers/parse_dsomm.py tests/test_parse_dsomm.py \
        data/processed/frameworks/dsomm.json
git commit -m "feat: parse DSOMM activities from the pinned generated model"
```

---

### Task 4: WSTG, 118 links

130 markdown test files under `document/4-Web_Application_Security_Testing/`, of which **108 are real tests with a `## Summary` section and 8 are tombstones** whose whole body is a redirect. **[measured]** `source-structures.md` says 144 files and does not mention the tombstones. 144 counts the 14 category `README.md` files, which are not controls.

Every one of the 108 summaries clears 60 characters, minimum 170. **[measured]** The prose floor is 1.0.

Four of the eight tombstones are linked by OpenCRE: `WSTG-ATHN-01`, `WSTG-ERRH-02`, `WSTG-INPV-03`, `WSTG-INPV-13`. **[measured]** Five of the eight carry a machine-readable successor in a markdown reference definition, `[merged]: # (WSTG-CRYP-03)`, so three of those four resolve deterministically. `WSTG-INPV-13` says only "This content has been removed" and has no successor, so it stays unresolved and is reported rather than guessed.

`WSTG-APPE-D` is linked and lives in `document/6-Appendix/D-Encoded_Injection.md`, which carries no ID table. The six appendix documents get ids derived from their filename letter. Three OpenCRE ids are extraction artifacts and no section is invented for them: `WSTG-BUSL-$$`, `WSTG-INFO-##`, `WSTG-INPV-00`. **[measured]**

The tombstone redirects need the `alt_ids` channel from Task 12. This task lands the redirect resolution and the metadata, and Task 12 makes `ProseIndex` read it.

**Files:**
- Create: `parsers/parse_wstg.py`
- Create: `tests/test_parse_wstg.py`

**Interfaces:**
- Consumes: `BaseParser`, `Control`.
- Produces: `WstgParser` with `framework_id = "wstg"`; `WstgParser.extract_test_id(text: str) -> str | None`; `WstgParser.extract_successor(text: str) -> str | None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_wstg.py — create

"""Tests for the OWASP WSTG parser.

The fixture is a synthetic four-member archive covering the three shapes that
matter: a real test with an ID table and a Summary, a tombstone carrying a
machine-readable successor, and an appendix with no ID table at all.
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path
from typing import ClassVar

import pytest

from parsers.parse_wstg import WstgParser

TEST_DOC = """\
# Conduct Search Engine Discovery Reconnaissance

|ID          |
|------------|
|WSTG-INFO-01|

## Summary

Search engines crawl and cache content that an application never intended to
publish, so a tester should look for indexed material before touching the
application itself.

## How to Test

Use a search engine operator to scope results to the target domain.
"""

SECOND_DOC = """\
# Testing for Weak Encryption

|ID          |
|------------|
|WSTG-CRYP-03|

## Summary

Applications that negotiate obsolete ciphers or accept short keys expose
transported data to an attacker positioned between client and server.
"""

TOMBSTONE_DOC = """\
# Testing for Credentials Transported over an Encrypted Channel

|ID          |
|------------|
|WSTG-ATHN-01|

This content has been merged into: [Testing for Weak Encryption](../09-Testing_for_Weak_Encryption.md).

[merged]: # (WSTG-CRYP-03)
"""

REMOVED_DOC = """\
# Testing for Buffer Overflow

|ID          |
|------------|
|WSTG-INPV-13|

This content has been removed.
"""

APPENDIX_DOC = """\
# Encoded Injection

## Background

Character encoding maps characters and symbols to a standard byte format, and
an application that decodes twice can be induced to execute what it believed
it had already sanitised.
"""

PREFIX = "wstg-abc123/document"


class SampleWstgParser(WstgParser):
    """The parser with the fixture's counts rather than the source's."""

    expected_count: ClassVar[int] = 3
    min_prose_fraction: ClassVar[float] = 1.0


@pytest.fixture
def parser(tmp_path: Path) -> WstgParser:
    raw = tmp_path / "raw"
    raw.mkdir()
    buffer = BytesIO()
    base = f"{PREFIX}/4-Web_Application_Security_Testing"
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{base}/01-Information_Gathering/01-Conduct.md", TEST_DOC)
        archive.writestr(f"{base}/09-Cryptography/03-Weak_Encryption.md", SECOND_DOC)
        archive.writestr(f"{base}/04-Authentication_Testing/01-Creds.md", TOMBSTONE_DOC)
        archive.writestr(f"{base}/07-Input_Validation_Testing/13-Buffer.md", REMOVED_DOC)
        archive.writestr(f"{base}/01-Information_Gathering/README.md", "# Intro\n")
        archive.writestr(f"{PREFIX}/6-Appendix/D-Encoded_Injection.md", APPENDIX_DOC)
        archive.writestr(f"{PREFIX}/6-Appendix/README.md", "# Appendix\n")
        archive.writestr(f"{PREFIX}/0-Foreword/README.md", "# Foreword\n")
    (raw / "wstg.zip").write_bytes(buffer.getvalue())
    out = tmp_path / "out"
    out.mkdir()
    return SampleWstgParser(raw_dir=raw, output_dir=out)


class TestWstgParser:
    def test_emits_one_control_per_real_test(self, parser: WstgParser) -> None:
        ids = {c.control_id for c in parser.parse()}
        assert "WSTG-INFO-01" in ids
        assert "WSTG-CRYP-03" in ids

    def test_skips_category_readmes_and_front_matter(
        self, parser: WstgParser,
    ) -> None:
        titles = {c.title for c in parser.parse()}
        assert "Intro" not in titles
        assert "Foreword" not in titles

    def test_a_tombstone_is_not_a_control(self, parser: WstgParser) -> None:
        """The redirect body would pass the length test and say nothing."""
        ids = {c.control_id for c in parser.parse()}
        assert "WSTG-ATHN-01" not in ids
        assert "WSTG-INPV-13" not in ids

    def test_a_tombstone_id_survives_as_an_alternate_on_its_successor(
        self, parser: WstgParser,
    ) -> None:
        """Three of the four linked tombstones name their successor.

        Dropping the id outright would drop the link with it.
        """
        controls = {c.control_id: c for c in parser.parse()}
        successor = controls["WSTG-CRYP-03"]

        assert successor.metadata is not None
        assert successor.metadata["alt_ids"] == ["WSTG-ATHN-01"]

    def test_a_removed_test_with_no_successor_is_reported_not_guessed(
        self, parser: WstgParser, caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        with caplog.at_level(logging.WARNING, logger="parsers.parse_wstg"):
            parser.parse()

        assert any(
            "WSTG-INPV-13" in record.getMessage() and "no successor" in
            record.getMessage() for record in caplog.records
        )

    def test_the_appendix_id_is_derived_from_the_filename_letter(
        self, parser: WstgParser,
    ) -> None:
        """WSTG-APPE-D is linked and the appendix carries no ID table."""
        controls = {c.control_id: c for c in parser.parse()}

        assert "WSTG-APPE-D" in controls
        assert controls["WSTG-APPE-D"].title == "Encoded Injection"
        assert "Character encoding" in controls["WSTG-APPE-D"].description

    def test_the_title_is_the_h1_not_the_id(self, parser: WstgParser) -> None:
        """OpenCRE carries the id in both section fields.

        The document's own H1 is richer and is what the anchor should say.
        """
        controls = {c.control_id: c for c in parser.parse()}
        assert controls["WSTG-INFO-01"].title == (
            "Conduct Search Engine Discovery Reconnaissance"
        )

    def test_the_description_is_the_summary_section(
        self, parser: WstgParser,
    ) -> None:
        controls = {c.control_id: c for c in parser.parse()}
        description = controls["WSTG-INFO-01"].description

        assert description.startswith("Search engines crawl")
        assert "How to Test" not in description
        assert "search engine operator" not in description


class TestIdExtraction:
    def test_reads_the_id_out_of_the_two_row_table(self) -> None:
        assert WstgParser.extract_test_id(TEST_DOC) == "WSTG-INFO-01"

    def test_returns_none_when_the_table_is_absent(self) -> None:
        assert WstgParser.extract_test_id(APPENDIX_DOC) is None

    def test_does_not_derive_the_id_from_the_directory_number(self) -> None:
        """The folder is 01-Information_Gathering, the id prefix is INFO.

        Deriving one from the other silently mislabels every category whose
        mnemonic does not match its ordinal.
        """
        assert WstgParser.extract_test_id(SECOND_DOC) == "WSTG-CRYP-03"


class TestSuccessorExtraction:
    def test_reads_the_merged_reference_definition(self) -> None:
        assert WstgParser.extract_successor(TOMBSTONE_DOC) == "WSTG-CRYP-03"

    def test_returns_none_for_removed_content(self) -> None:
        assert WstgParser.extract_successor(REMOVED_DOC) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_parse_wstg.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'parsers.parse_wstg'`.

- [ ] **Step 3: Implement the parser**

```python
# parsers/parse_wstg.py — create

"""Parser for the OWASP Web Security Testing Guide.

118 curated training links, all of them anchored today on a bare test id
because OpenCRE carries the id in both section_id and section_name and the
document's real title never reached the corpus.

Measured against the pinned commit: 130 markdown files under
document/4-Web_Application_Security_Testing/ excluding the 14 category
READMEs, of which 108 are real tests carrying a Summary and 8 are tombstones
whose whole body is a redirect. All 108 summaries clear 60 characters, the
shortest at 170.

Four tombstones are linked. Five of the eight name their successor in a
markdown reference definition, "[merged]: # (WSTG-CRYP-03)", which is a
deterministic redirect rather than a guess, so those ids move to the successor
as alternates. WSTG-INPV-13 says only that the content was removed and is
logged at WARNING rather than resolved.

The six appendix documents carry no ID table. WSTG-APPE-D is linked, so ids
are derived from the filename letter, which is the same convention OpenCRE
used.

Source: https://github.com/OWASP/wstg
"""
from __future__ import annotations

import logging
import re
import zipfile
from io import BytesIO
from typing import ClassVar, Final

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARCHIVE_NAME: Final[str] = "wstg.zip"
TESTS_DIR: Final[str] = "/document/4-Web_Application_Security_Testing/"
APPENDIX_DIR: Final[str] = "/document/6-Appendix/"
# The archive is 15.6 MB of downloaded content and CLAUDE.md treats framework
# data as untrusted input. The largest real test document is 27 KB.
MAX_MEMBER_BYTES: Final[int] = 200_000

# The id sits alone in a two row table under the H1. Anchored to a whole line
# so a mention of an id inside prose cannot be mistaken for the declaration.
_ID_TABLE: Final[re.Pattern[str]] = re.compile(
    r"^\|\s*(WSTG-[A-Z]+-\d+)\s*\|\s*$", re.MULTILINE,
)
# The tombstone's successor, in a markdown reference definition the site
# generator reads. Machine readable, which is what makes the redirect a
# resolution rather than an inference from the prose around it.
_MERGED: Final[re.Pattern[str]] = re.compile(
    r"^\[merged\]:\s*#\s*\((WSTG-[A-Z]+-\d+)\)", re.MULTILINE,
)
_H1: Final[re.Pattern[str]] = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_SUMMARY: Final[re.Pattern[str]] = re.compile(
    r"^##\s+Summary\s*$(.*?)(?=^##\s)", re.MULTILINE | re.DOTALL,
)
_ANY_H2: Final[re.Pattern[str]] = re.compile(r"^##\s+", re.MULTILINE)
# "D-Encoded_Injection.md" -> "D".
_APPENDIX_LETTER: Final[re.Pattern[str]] = re.compile(r"^([A-Z])-")
_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


class WstgParser(BaseParser):
    framework_id: ClassVar[str] = "wstg"
    # FRAMEWORK_NAME_ALIASES maps "owasp web security testing guide (wstg)" to
    # this exact string.
    framework_name: ClassVar[str] = "WSTG"
    version: ClassVar[str] = "95ce6cfe5d463bbde88aa52b3171b123a1ea9ada"
    source_url: ClassVar[str] = "https://github.com/OWASP/wstg"
    mapping_unit_level: ClassVar[str] = "test"
    # 108 tests plus 6 appendix documents. Exact, not a floor: the archive is
    # pinned to a commit, so this cannot move without a re-pin.
    expected_count: ClassVar[int] = 114
    fetched_date: ClassVar[str] = "2026-08-15"
    # Every one of the 108 summaries and all 6 appendix bodies clear the bar.
    min_prose_fraction: ClassVar[float] = 1.0

    @staticmethod
    def extract_test_id(text: str) -> str | None:
        """The declared test id, or None when the document carries no table."""
        match = _ID_TABLE.search(text)
        return match.group(1) if match else None

    @staticmethod
    def extract_successor(text: str) -> str | None:
        """The id a tombstone redirects to, or None when it names none."""
        match = _MERGED.search(text)
        return match.group(1) if match else None

    def parse(self) -> list[Control]:
        members = self._read_members()
        tests, tombstones = self._partition(members)
        controls = [self._to_control(name, text) for name, text in tests]
        controls += self._appendix_controls(members)
        self._apply_redirects(controls, tombstones)
        logger.info(
            "%s: %d tests, %d appendix documents, %d tombstones",
            self.framework_id, len(tests), len(controls) - len(tests),
            len(tombstones),
        )
        return controls

    def _read_members(self) -> dict[str, str]:
        """Every markdown member under the test and appendix trees."""
        payload = self.read_source_bytes(ARCHIVE_NAME)
        members: dict[str, str] = {}
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            for name in sorted(archive.namelist()):
                if not name.endswith(".md"):
                    continue
                if TESTS_DIR not in name and APPENDIX_DIR not in name:
                    continue
                if name.endswith("README.md"):
                    continue
                info = archive.getinfo(name)
                if info.file_size > MAX_MEMBER_BYTES:
                    raise ValueError(
                        f"{name}: declares {info.file_size} bytes, over the "
                        f"{MAX_MEMBER_BYTES} byte cap"
                    )
                with archive.open(name) as handle:
                    raw = handle.read(MAX_MEMBER_BYTES + 1)
                if len(raw) > MAX_MEMBER_BYTES:
                    raise ValueError(
                        f"{name}: expanded past the {MAX_MEMBER_BYTES} byte cap"
                    )
                members[name] = raw.decode("utf-8")
        if not members:
            raise ValueError(
                f"{self.framework_id}: no markdown members matched "
                f"{TESTS_DIR} or {APPENDIX_DIR}. The archive layout changed."
            )
        return members

    def _partition(
        self, members: dict[str, str],
    ) -> tuple[list[tuple[str, str]], dict[str, str | None]]:
        """Split the test tree into real tests and tombstones.

        A tombstone carries an id table and no section headings at all. That
        is the structural difference, not a phrase in the prose, so a reworded
        redirect notice does not turn one into a control.
        """
        tests: list[tuple[str, str]] = []
        tombstones: dict[str, str | None] = {}
        for name, text in members.items():
            if TESTS_DIR not in name:
                continue
            test_id = self.extract_test_id(text)
            if test_id is None:
                logger.info("%s: no id table, skipped", name)
                continue
            if _ANY_H2.search(text):
                tests.append((name, text))
                continue
            successor = self.extract_successor(text)
            tombstones[test_id] = successor
            if successor is None:
                logger.warning(
                    "%s: %s is a tombstone with no successor. Any OpenCRE "
                    "link to it stays unresolved rather than being pointed at "
                    "a document nothing in the source names.",
                    self.framework_id, test_id,
                )
        return tests, tombstones

    def _to_control(self, name: str, text: str) -> Control:
        test_id = self.extract_test_id(text)
        if test_id is None:
            raise ValueError(f"{name}: partitioned as a test with no id")
        heading = _H1.search(text)
        if heading is None:
            raise ValueError(f"{name}: no H1 heading, so the test has no title")
        summary = _SUMMARY.search(text)
        if summary is None:
            raise ValueError(
                f"{name}: no Summary section. Every real test carries one, so "
                f"either the document layout changed or this is a tombstone "
                f"that reached the wrong branch."
            )
        return Control(
            control_id=test_id,
            title=_WHITESPACE.sub(" ", heading.group(1)).strip(),
            description=_WHITESPACE.sub(" ", summary.group(1)).strip(),
        )

    def _appendix_controls(self, members: dict[str, str]) -> list[Control]:
        """One control per appendix document, id derived from the filename.

        WSTG-APPE-D is a real OpenCRE link and the appendix carries no id
        table, so the letter in the filename is the only identifier the source
        offers. That is the same convention OpenCRE's own id uses.
        """
        controls: list[Control] = []
        for name, text in members.items():
            if APPENDIX_DIR not in name:
                continue
            letter = _APPENDIX_LETTER.match(name.rsplit("/", 1)[-1])
            if letter is None:
                raise ValueError(
                    f"{name}: appendix filename does not start with a letter "
                    f"and a hyphen, so no id can be derived from it"
                )
            heading = _H1.search(text)
            if heading is None:
                raise ValueError(f"{name}: no H1 heading")
            body = _H1.sub("", text, count=1).strip()
            controls.append(Control(
                control_id=f"WSTG-APPE-{letter.group(1)}",
                title=_WHITESPACE.sub(" ", heading.group(1)).strip(),
                description=_WHITESPACE.sub(" ", body).strip(),
            ))
        return controls

    def _apply_redirects(
        self, controls: list[Control], tombstones: dict[str, str | None],
    ) -> None:
        """Attach each retired id to the control that absorbed its content.

        OpenCRE links four of the eight tombstones. Dropping the id with the
        document drops the link with it, and pointing the link at a document
        the source does not name would be a guess, so only the ids with a
        machine readable successor move.
        """
        by_id = {c.control_id: c for c in controls}
        for retired, successor in sorted(tombstones.items()):
            if successor is None:
                continue
            target = by_id.get(successor)
            if target is None:
                raise ValueError(
                    f"{self.framework_id}: {retired} redirects to {successor}, "
                    f"which is not among the parsed tests. The source's own "
                    f"redirect is dangling and must be read before this "
                    f"parser can be trusted."
                )
            metadata = dict(target.metadata or {})
            existing = metadata.get("alt_ids")
            alternates = list(existing) if isinstance(existing, list) else []
            alternates.append(retired)
            metadata["alt_ids"] = sorted(alternates)
            target.metadata = metadata
            logger.info(
                "%s: %s retired into %s, id kept as an alternate",
                self.framework_id, retired, successor,
            )


def main() -> None:
    WstgParser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_parse_wstg.py -v
mypy parsers/parse_wstg.py --strict
```

Expected: PASS, no mypy errors.

- [ ] **Step 5: Run against the real source and check the join**

```bash
python3 parsers/parse_wstg.py
python3 - <<'PY'
import json, pathlib
d = json.loads(pathlib.Path("data/processed/frameworks/wstg.json").read_text())
links = json.loads(
    pathlib.Path("data/training/hub_links_by_framework.json").read_text()
)["wstg"]
ids = {c["control_id"] for c in d["controls"]}
alt = {a for c in d["controls"] for a in (c.get("metadata") or {}).get("alt_ids", [])}
link_ids = {l["section_id"] for l in links}
print("controls:", len(d["controls"]))
print("id join :", len(link_ids & ids), "of", len(link_ids))
print("with alternates:", len(link_ids & (ids | alt)), "of", len(link_ids))
print("unresolved:", sorted(link_ids - ids - alt))
PY
```

Expected: `controls: 114`, `id join : 56 of 59`, `with alternates: 59 of 59` after Task 12 lands the reader, and the unresolved list containing only `WSTG-BUSL-$$`, `WSTG-INFO-##`, `WSTG-INPV-00` and `WSTG-INPV-13`. **[measured]** Before the alternates are counted the raw id join is 55 of 59 plus `WSTG-APPE-D`.

If any of the three `$$`/`##`/`00` placeholders resolves, a section was invented for an OpenCRE extraction artifact. Stop and remove it.

- [ ] **Step 6: Commit**

```bash
git add parsers/parse_wstg.py tests/test_parse_wstg.py \
        data/processed/frameworks/wstg.json
git commit -m "feat: parse WSTG tests and appendices with tombstone redirects"
```

---

### Task 5: NIST SP 800-63B, 79 links

Currently dropped by `PHASE1B_DROPPED_FRAMEWORKS`, so all 79 links contribute nothing today. The fetch already corrected the revision: the file on disk is **revision 3B** (`sp800_63b.html`, 215,987 bytes), and 24 of the 25 distinct OpenCRE section ids exist as headings in it. **[measured]** The one miss is `are g`, an artifact in OpenCRE's own extraction, and no section is invented for it.

`source-structures.md` describes revision 4 and is stale for this framework. Commit `ad22799` replaced the source. The revision-4 structure it documents, with slug ids and single-integer `data-section` attributes, does not apply. Revision 3B uses `<h2>` through `<h6>` headings whose text opens with the dotted section number.

Measured against the file on disk: **118 numbered headings**, of which 3 are pure containers with an empty body (`5.1`, `5.2`, `6.1.2`) and 2 have a 28 character informative note (`8`, `11`). None of the 5 is linked. Skipping the 3 empty containers leaves **115 controls, 113 of which clear 60 characters**. **[measured]**

**Files:**
- Create: `parsers/parse_nist_800_63.py`
- Create: `tests/test_parse_nist_800_63.py`

**Interfaces:**
- Consumes: `BaseParser`, `Control`.
- Produces: `Nist80063Parser` with `framework_id = "nist_800_63"`; `Nist80063Parser.extract_sections(html: str) -> list[tuple[str, str, str]]` returning `(number, title, body)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_nist_800_63.py — create

"""Tests for the NIST SP 800-63B parser.

The fixture is a synthetic page with the same heading and body structure as
the real revision 3B document, including a container heading whose own body is
empty. Three such headings exist in the source and pydantic rejects an empty
description, so the parser has to decide about them rather than crash.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from parsers.parse_nist_800_63 import Nist80063Parser

PAGE = """\
<html><body><div>
<h1 id="nist-special-publication-800-63b">NIST Special Publication 800-63B</h1>
<h2 id="table-of-contents">Table of Contents</h2>
<p>Front matter that carries no section number.</p>
<h2 id="5-authenticator-and-verifier-requirements">5 Authenticator and Verifier Requirements</h2>
<p>This section is normative and states requirements for authenticators used
at each authenticator assurance level defined in this document.</p>
<h3 id="51-requirements-by-authenticator-type">5.1 Requirements by Authenticator Type</h3>
<h4 id="511-memorized-secrets">5.1.1 Memorized Secrets</h4>
<p>A memorized secret is chosen by the subscriber at enrollment and is
typically a password or a personal identification number.</p>
<h5 id="5112-memorized-secret-verifiers">5.1.1.2 Memorized Secret Verifiers</h5>
<p>Verifiers shall require memorized secrets to be at least eight characters
long and shall compare the prospective secret against a list of values known
to be commonly used, expected, or compromised.</p>
<h2 id="a3-complexity">A.3 Complexity</h2>
<p>Composition rules impose a burden on the subscriber that is out of
proportion to the security benefit they deliver.</p>
</div></body></html>
"""


class SampleNist80063Parser(Nist80063Parser):
    """The parser with the fixture's counts rather than the source's."""

    expected_count: ClassVar[int] = 4
    min_prose_fraction: ClassVar[float] = 1.0


@pytest.fixture
def parser(tmp_path: Path) -> Nist80063Parser:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "sp800_63b.html").write_text(PAGE, encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    return SampleNist80063Parser(raw_dir=raw, output_dir=out)


class TestNist80063Parser:
    def test_control_id_is_the_bare_dotted_number(
        self, parser: Nist80063Parser,
    ) -> None:
        """OpenCRE's section_id is "5.1.1.2", not a slug and not a prefix."""
        ids = {c.control_id for c in parser.parse()}
        assert "5.1.1.2" in ids
        assert "A.3" in ids
        assert not any(cid.startswith("nist_800_63:") for cid in ids)

    def test_title_excludes_the_number(self, parser: Nist80063Parser) -> None:
        controls = {c.control_id: c for c in parser.parse()}
        assert controls["5.1.1.2"].title == "Memorized Secret Verifiers"

    def test_body_stops_at_the_next_heading(
        self, parser: Nist80063Parser,
    ) -> None:
        controls = {c.control_id: c for c in parser.parse()}
        assert controls["5.1.1"].description.startswith("A memorized secret")
        assert "Verifiers shall require" not in controls["5.1.1"].description

    def test_an_unnumbered_heading_is_not_a_section(
        self, parser: Nist80063Parser,
    ) -> None:
        titles = {c.title for c in parser.parse()}
        assert "Table of Contents" not in titles

    def test_a_container_heading_with_no_body_is_skipped_and_logged(
        self, parser: Nist80063Parser, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """5.1, 5.2 and 6.1.2 are containers in the real document.

        None is linked, and pydantic rejects an empty description, so they are
        dropped with a named log line rather than padded with their title.
        """
        import logging

        with caplog.at_level(logging.INFO, logger="parsers.parse_nist_800_63"):
            controls = parser.parse()

        assert "5.1" not in {c.control_id for c in controls}
        assert any(
            "5.1" in record.getMessage() and "no body" in record.getMessage()
            for record in caplog.records
        )

    def test_reads_the_page_through_the_recording_reader(
        self, parser: Nist80063Parser,
    ) -> None:
        parser.parse()
        assert "sp800_63b.html" in parser._source_files


class TestExtractSections:
    def test_returns_number_title_and_body_in_document_order(self) -> None:
        sections = Nist80063Parser.extract_sections(PAGE)
        numbers = [number for number, _, _ in sections]

        assert numbers == ["5", "5.1", "5.1.1", "5.1.1.2", "A.3"]

    def test_an_appendix_letter_counts_as_a_number(self) -> None:
        sections = dict((n, t) for n, t, _ in Nist80063Parser.extract_sections(PAGE))
        assert sections["A.3"] == "Complexity"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_parse_nist_800_63.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'parsers.parse_nist_800_63'`.

- [ ] **Step 3: Implement the parser**

```python
# parsers/parse_nist_800_63.py — create

"""Parser for NIST SP 800-63B revision 3, Digital Identity Guidelines.

79 curated links, none of which contribute anything today: the framework sits
in PHASE1B_DROPPED_FRAMEWORKS because it had no primary source, and its
section names are bare numbers that the short title gate drops anyway.

REVISION 3, NOT 4, AND THE DIFFERENCE IS LOAD BEARING. OpenCRE's 25 distinct
section ids are revision 3 numbering. Measured: revision 3B contains 24 of the
25 and revision 4B contains none of them, because revision 4 renumbered the
document and renamed memorized secrets to passwords. The one miss is "are g",
an artifact in OpenCRE's own extraction rather than a section, and no heading
is invented for it.

Measured against the file on disk: 118 numbered headings, of which 3 are pure
containers with no body of their own (5.1, 5.2, 6.1.2) and none of those 3 is
linked. They are dropped with a named log line, leaving 115 controls of which
113 clear 60 characters.

The source hash is deliberately unpinned upstream: pages.nist.gov sits behind
Cloudflare, which injects a per response nonce into the body, so two fetches of
the identical document differ. The source manifest still records what this run
read, which is the honest thing for it to say.

Source: https://pages.nist.gov/800-63-3/sp800-63b.html
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

# "5.1.1.2 Memorized Secret Verifiers" and "A.3 Complexity". The leading token
# is a chapter digit or an appendix letter, then any number of dotted levels.
# An optional trailing period on the number is tolerated because the source is
# inconsistent about it.
_NUMBERED: Final[re.Pattern[str]] = re.compile(
    r"^((?:\d+|[A-C])(?:\.\d+)*)\.?\s+(.+)$",
)
_HEADING: Final[re.Pattern[str]] = re.compile(r"^h[1-6]$")
_SECTION_HEADING: Final[re.Pattern[str]] = re.compile(r"^h[2-6]$")
_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


class Nist80063Parser(BaseParser):
    framework_id: ClassVar[str] = "nist_800_63"
    # Matches the link's standard_name exactly. There is no alias entry for
    # this framework, so the two strings have to agree character for character.
    framework_name: ClassVar[str] = "NIST 800-63"
    version: ClassVar[str] = "SP 800-63B-3"
    source_url: ClassVar[str] = "https://pages.nist.gov/800-63-3/sp800-63b.html"
    mapping_unit_level: ClassVar[str] = "section"
    expected_count: ClassVar[int] = 115
    fetched_date: ClassVar[str] = "2026-08-15"
    # 113 of 115. The two short ones are section 8 and section 11, each a one
    # line informative note, and neither is linked.
    min_prose_fraction: ClassVar[float] = 0.97

    @staticmethod
    def extract_sections(html: str) -> list[tuple[str, str, str]]:
        """Pull (number, title, body) for every numbered heading, in order.

        The body runs from the heading to the next heading of any level, which
        is why a container heading yields an empty string rather than its
        children's text. Rolling children up would give 5.1 the same text as
        5.1.1 and hand two distinct OpenCRE anchors one anchor text, which is
        the collapse the eval corpus dedupe cannot see.
        """
        soup = BeautifulSoup(html, "lxml")
        sections: list[tuple[str, str, str]] = []
        for heading in soup.find_all(_SECTION_HEADING):
            text = _WHITESPACE.sub(" ", heading.get_text(" ", strip=True)).strip()
            match = _NUMBERED.match(text)
            if match is None:
                continue
            parts: list[str] = []
            for sibling in heading.next_siblings:
                if not isinstance(sibling, Tag):
                    continue
                if _HEADING.match(sibling.name):
                    break
                parts.append(sibling.get_text(" ", strip=True))
            body = _WHITESPACE.sub(" ", " ".join(parts)).strip()
            sections.append((match.group(1), match.group(2).strip(), body))
        return sections

    def parse(self) -> list[Control]:
        sections = self.extract_sections(self.read_source(SOURCE_FILE))
        if not sections:
            raise ValueError(
                f"{self.framework_id}: no numbered headings matched. The page "
                f"layout changed, or the fetched revision is 4, whose headings "
                f"carry slug ids and no dotted numbers. Re-check {SOURCE_FILE}."
            )

        controls: list[Control] = []
        skipped: list[str] = []
        for number, title, body in sections:
            if not body:
                skipped.append(number)
                continue
            controls.append(Control(
                control_id=number,
                title=title,
                description=body,
            ))
        logger.info(
            "%s: %d numbered sections, %d skipped for no body of their own: %s",
            self.framework_id, len(sections), len(skipped),
            ", ".join(skipped) or "none",
        )
        self._check_duplicate_ids(controls)
        return controls

    def _check_duplicate_ids(self, controls: list[Control]) -> None:
        """Refuse two sections claiming the same number.

        The id is the entire join for this framework, so a duplicate silently
        hands one OpenCRE link whichever section the index saw first.
        """
        seen: set[str] = set()
        duplicates = sorted(
            c.control_id for c in controls
            if c.control_id in seen or seen.add(c.control_id)  # type: ignore[func-returns-value]
        )
        if duplicates:
            raise ValueError(
                f"{self.framework_id}: duplicate section numbers {duplicates}. "
                f"The id is the entire join for this framework."
            )


def main() -> None:
    Nist80063Parser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_parse_nist_800_63.py -v
mypy parsers/parse_nist_800_63.py --strict
```

Expected: PASS, no mypy errors. If mypy objects to the `seen.add` idiom in `_check_duplicate_ids`, replace it with an explicit loop rather than widening the ignore.

- [ ] **Step 5: Run against the real source and check the join**

```bash
python3 parsers/parse_nist_800_63.py
python3 - <<'PY'
import json, pathlib
d = json.loads(pathlib.Path("data/processed/frameworks/nist_800_63.json").read_text())
links = json.loads(
    pathlib.Path("data/training/hub_links_by_framework.json").read_text()
)["nist_800_63"]
ids = {c["control_id"] for c in d["controls"]}
link_ids = {l["section_id"] for l in links}
print("controls:", len(d["controls"]))
print("id join :", len(link_ids & ids), "of", len(link_ids))
print("unresolved:", sorted(link_ids - ids))
print("prose   :", sum(1 for c in d["controls"] if len(c["description"]) >= 60),
      "of", len(d["controls"]))
PY
```

Expected: `controls: 115`, `id join : 24 of 25`, `unresolved: ['are g']`, `prose : 113 of 115`. **[measured]** If `are g` resolves, a section was invented for an OpenCRE artifact.

Note that the framework stays in `PHASE1B_DROPPED_FRAMEWORKS` after this task. Removing it is spec Part 1.5 and is deferred, and Task 17 measures what removing it would buy.

- [ ] **Step 6: Commit**

```bash
git add parsers/parse_nist_800_63.py tests/test_parse_nist_800_63.py \
        data/processed/frameworks/nist_800_63.json
git commit -m "feat: parse NIST SP 800-63B revision 3 sections from the fetched page"
```

---

### Task 6: OWASP Proactive Controls, 76 links

The second framework currently in `PHASE1B_DROPPED_FRAMEWORKS`. Ten controls carrying 76 links, the densest ratio in the corpus at 7.6 links per control.

Measured: `docs/the-top-10/` holds 11 markdown files, 10 controls plus `index.md`. Every `## Description` body clears 60 characters, the shortest at 530 and the longest at 5,745. **[measured]** All ten ids `C1` through `C10` join. **[measured]** Every link's `section_name` is the bare id, two characters, which is why the short-title gate drops all 76 today.

The archive carries two decoys with the same control numbering: `docs/archive/2018/c*.md` is a v3 snapshot with different prose, and `v3/` holds 24 MB of PDF, DOCX and PPTX exports. A glob on `**/c[0-9]*-*.md` picks up the archive. The parser anchors on the exact directory.

**Files:**
- Create: `parsers/parse_owasp_proactive_controls.py`
- Create: `tests/test_parse_owasp_proactive_controls.py`

**Interfaces:**
- Consumes: `BaseParser`, `Control`.
- Produces: `OwaspProactiveControlsParser` with `framework_id = "owasp_proactive_controls"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_owasp_proactive_controls.py — create

"""Tests for the OWASP Proactive Controls parser.

The fixture carries the decoy the archive really contains: an archived 2018
copy of the same control number with different prose, sitting one directory
away from the current one.
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path
from typing import ClassVar

import pytest

from parsers.parse_owasp_proactive_controls import OwaspProactiveControlsParser

CURRENT_C1 = """\
# C1: Implement Access Control

## Description

Access control, or authorization, is allowing or denying specific requests
from a user, program, or process. With each access control decision, a given
subject requests access to a given object.

## Threats

An attacker acting outside their intended permissions.
"""

CURRENT_C10 = """\
# C10: Stop Server Side Request Forgery

## Description

Server side request forgery occurs when an application fetches a remote
resource at a location the user supplies, without validating that the
location is one the application intended to reach.

## Threats

An attacker reaching an internal service through the application.
"""

ARCHIVED_C1 = """\
# C1: Define Security Requirements

## Description

This is the 2018 edition of the first control and its prose does not match
the current one at all, which is exactly why it must not be parsed.
"""

PREFIX = "www-project-proactive-controls-abc123"


class SampleProactiveParser(OwaspProactiveControlsParser):
    """The parser with the fixture's counts rather than the source's."""

    expected_count: ClassVar[int] = 2
    min_prose_fraction: ClassVar[float] = 1.0


@pytest.fixture
def parser(tmp_path: Path) -> OwaspProactiveControlsParser:
    raw = tmp_path / "raw"
    raw.mkdir()
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{PREFIX}/docs/the-top-10/c1-accesscontrol.md", CURRENT_C1)
        archive.writestr(
            f"{PREFIX}/docs/the-top-10/c10-stop-server-side-request-forgery.md",
            CURRENT_C10,
        )
        archive.writestr(f"{PREFIX}/docs/the-top-10/index.md", "# Introduction\n")
        archive.writestr(f"{PREFIX}/docs/archive/2018/c1-security-requirements.md",
                         ARCHIVED_C1)
        archive.writestr(f"{PREFIX}/v3/OWASP_Top_10_Proactive_Controls.pdf", "binary")
    (raw / "owasp_proactive_controls.zip").write_bytes(buffer.getvalue())
    out = tmp_path / "out"
    out.mkdir()
    return SampleProactiveParser(raw_dir=raw, output_dir=out)


class TestProactiveControlsParser:
    def test_control_id_is_the_bare_c_token(
        self, parser: OwaspProactiveControlsParser,
    ) -> None:
        ids = sorted(c.control_id for c in parser.parse())
        assert ids == ["C1", "C10"]

    def test_title_is_the_h1_text_after_the_colon(
        self, parser: OwaspProactiveControlsParser,
    ) -> None:
        controls = {c.control_id: c for c in parser.parse()}
        assert controls["C1"].title == "Implement Access Control"

    def test_description_is_the_description_section_only(
        self, parser: OwaspProactiveControlsParser,
    ) -> None:
        controls = {c.control_id: c for c in parser.parse()}
        description = controls["C1"].description

        assert description.startswith("Access control, or authorization")
        assert "acting outside their intended permissions" not in description

    def test_the_archived_2018_edition_is_not_parsed(
        self, parser: OwaspProactiveControlsParser,
    ) -> None:
        """docs/archive/2018/ reuses C1 through C10 with different prose."""
        controls = {c.control_id: c for c in parser.parse()}
        assert controls["C1"].title != "Define Security Requirements"

    def test_the_index_page_is_not_a_control(
        self, parser: OwaspProactiveControlsParser,
    ) -> None:
        titles = {c.title for c in parser.parse()}
        assert "Introduction" not in titles

    def test_reads_the_archive_through_the_recording_reader(
        self, parser: OwaspProactiveControlsParser,
    ) -> None:
        parser.parse()
        assert "owasp_proactive_controls.zip" in parser._source_files
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_parse_owasp_proactive_controls.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the parser**

```python
# parsers/parse_owasp_proactive_controls.py — create

"""Parser for the OWASP Proactive Controls.

76 curated links across ten controls, the densest ratio in the corpus, and
none of them reaches training today: the framework sits in
PHASE1B_DROPPED_FRAMEWORKS and every link's section_name is the bare id,
two characters, which the short title gate drops on its own.

Measured against the pinned commit: ten controls under docs/the-top-10/, every
Description section clearing 60 characters with the shortest at 530.

The archive carries two decoys with the same C1 through C10 numbering. A v3
snapshot of the 2018 edition sits under docs/archive/2018/ with different
prose, and 24 MB of PDF, DOCX and PPTX exports sit under v3/. Members are
selected by exact directory rather than by a filename glob, which would match
the archive copy first for half the controls.

Source: https://github.com/OWASP/www-project-proactive-controls
"""
from __future__ import annotations

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
# The current mkdocs content. Anchored exactly, not globbed: docs/archive/2018/
# holds the same control numbers with the prior edition's prose.
CONTROLS_DIR: Final[str] = "/docs/the-top-10/"
MAX_MEMBER_BYTES: Final[int] = 200_000

# "# C1: Implement Access Control".
_H1: Final[re.Pattern[str]] = re.compile(
    r"^#\s+(C\d{1,2}):\s*(.+?)\s*$", re.MULTILINE,
)
_DESCRIPTION: Final[re.Pattern[str]] = re.compile(
    r"^##\s+Description\s*$(.*?)(?=^##\s)", re.MULTILINE | re.DOTALL,
)
_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


class OwaspProactiveControlsParser(BaseParser):
    framework_id: ClassVar[str] = "owasp_proactive_controls"
    framework_name: ClassVar[str] = "OWASP Proactive Controls"
    version: ClassVar[str] = "4f5cb1081b4253bbccb314ef7855a1430fec8571"
    source_url: ClassVar[str] = (
        "https://github.com/OWASP/www-project-proactive-controls"
    )
    mapping_unit_level: ClassVar[str] = "control"
    expected_count: ClassVar[int] = 10
    fetched_date: ClassVar[str] = "2026-08-15"
    min_prose_fraction: ClassVar[float] = 1.0

    def parse(self) -> list[Control]:
        controls: list[Control] = []
        payload = self.read_source_bytes(ARCHIVE_NAME)
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            for name in sorted(archive.namelist()):
                if CONTROLS_DIR not in name or not name.endswith(".md"):
                    continue
                text = self._read_member(archive, name)
                heading = _H1.search(text)
                if heading is None:
                    logger.info("%s: no C-numbered H1, skipped", name)
                    continue
                body = _DESCRIPTION.search(text)
                if body is None:
                    raise ValueError(
                        f"{name}: has a C-numbered H1 and no Description "
                        f"section. Every current control carries one, so the "
                        f"document layout changed."
                    )
                controls.append(Control(
                    control_id=heading.group(1),
                    title=heading.group(2).strip(),
                    description=_WHITESPACE.sub(" ", body.group(1)).strip(),
                ))
        if not controls:
            raise ValueError(
                f"{self.framework_id}: no members matched {CONTROLS_DIR}. The "
                f"archive layout changed, or the pin moved to a commit before "
                f"the mkdocs restructure."
            )
        logger.info("%s: parsed %d controls", self.framework_id, len(controls))
        return controls

    @staticmethod
    def _read_member(archive: zipfile.ZipFile, name: str) -> str:
        """Read one bounded member. The archive is downloaded, so untrusted."""
        info = archive.getinfo(name)
        if info.file_size > MAX_MEMBER_BYTES:
            raise ValueError(
                f"{name}: declares {info.file_size} bytes, over the "
                f"{MAX_MEMBER_BYTES} byte cap"
            )
        with archive.open(name) as handle:
            raw = handle.read(MAX_MEMBER_BYTES + 1)
        if len(raw) > MAX_MEMBER_BYTES:
            raise ValueError(
                f"{name}: expanded past the {MAX_MEMBER_BYTES} byte cap"
            )
        return raw.decode("utf-8")


def main() -> None:
    OwaspProactiveControlsParser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_parse_owasp_proactive_controls.py -v
mypy parsers/parse_owasp_proactive_controls.py --strict
```

Expected: PASS, no mypy errors.

- [ ] **Step 5: Run against the real source and check the join**

```bash
python3 parsers/parse_owasp_proactive_controls.py
python3 - <<'PY'
import json, pathlib
d = json.loads(pathlib.Path(
    "data/processed/frameworks/owasp_proactive_controls.json").read_text())
links = json.loads(
    pathlib.Path("data/training/hub_links_by_framework.json").read_text()
)["owasp_proactive_controls"]
ids = {c["control_id"] for c in d["controls"]}
link_ids = {l["section_id"] for l in links}
print("controls:", len(d["controls"]))
print("id join :", len(link_ids & ids), "of", len(link_ids))
print("shortest description:", min(len(c["description"]) for c in d["controls"]))
titles = {c["title"] for c in d["controls"]}
print("archived title present:", "Define Security Requirements" in titles)
PY
```

Expected: `controls: 10`, `id join : 10 of 10`, `shortest description: 530`, `archived title present: False`. **[measured]**

- [ ] **Step 6: Commit**

```bash
git add parsers/parse_owasp_proactive_controls.py \
        tests/test_parse_owasp_proactive_controls.py \
        data/processed/frameworks/owasp_proactive_controls.json
git commit -m "feat: parse the current OWASP Proactive Controls, not the 2018 archive"
```

---

### Task 7: A counted rowspan merge in the repair layer

ENISA and NIST SSDF are both PDF tables whose cells span many extracted rows. `pdfplumber.extract_tables()` returns one visual line per row, so a control's definition arrives as twenty rows with only the definition column populated, and the row that names the control is the row above them.

This is the same shape twice, so it is one repair rather than two private helpers. It goes in `tract/parsers/repair.py` with a count, like every other repair, so a parser can declare how many merges its source needs and notice when the source stops needing them.

**Files:**
- Modify: `tract/parsers/repair.py`
- Modify: `tests/test_repair.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `merge_spanned_rows(rows: list[list[str]], key_column: int, text_columns: tuple[int, ...]) -> tuple[list[tuple[str, str]], int]` returning `[(key, joined_text)]` and the number of continuation rows absorbed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_repair.py — append

from tract.parsers.repair import merge_spanned_rows


class TestMergeSpannedRows:
    def test_absorbs_continuation_rows_into_the_row_that_named_the_key(
        self,
    ) -> None:
        rows = [
            ["Apply a RBAC model", "", "", "Define access rights management"],
            ["", "", "", "using a RBAC model respecting the least"],
            ["", "", "", "privileged principle."],
            ["Build explainable models", "", "", "Favour models whose decisions"],
            ["", "", "", "can be explained to a reviewer."],
        ]
        merged, absorbed = merge_spanned_rows(rows, key_column=0, text_columns=(3,))

        assert absorbed == 3
        assert merged == [
            ("Apply a RBAC model",
             "Define access rights management using a RBAC model respecting "
             "the least privileged principle."),
            ("Build explainable models",
             "Favour models whose decisions can be explained to a reviewer."),
        ]

    def test_reads_several_text_columns_left_to_right(self) -> None:
        """SSDF puts the task in column 3 and the examples in 6 or 7."""
        rows = [
            ["", "", "", "PO.1.1: Identify requirements.", "", "",
             "Example 1: Define policies."],
            ["", "", "", "", "", "", "Example 2: Review them."],
        ]
        merged, absorbed = merge_spanned_rows(
            rows, key_column=3, text_columns=(3, 6),
        )

        assert absorbed == 1
        assert merged == [(
            "PO.1.1: Identify requirements.",
            "PO.1.1: Identify requirements. Example 1: Define policies. "
            "Example 2: Review them.",
        )]

    def test_drops_leading_rows_that_have_no_key_yet(self) -> None:
        """A header row carries text and belongs to no record."""
        rows = [
            ["", "", "", "Definition"],
            ["Apply a RBAC model", "", "", "Define access rights management."],
        ]
        merged, absorbed = merge_spanned_rows(rows, key_column=0, text_columns=(3,))

        assert absorbed == 0
        assert merged == [
            ("Apply a RBAC model", "Define access rights management."),
        ]

    def test_normalises_whitespace_and_cell_internal_newlines(self) -> None:
        rows = [["Apply a RBAC\nmodel", "", "", "  Define   access\nrights.  "]]
        merged, _ = merge_spanned_rows(rows, key_column=0, text_columns=(3,))

        assert merged == [("Apply a RBAC model", "Define access rights.")]

    def test_a_short_row_is_padded_rather_than_raising(self) -> None:
        """pdfplumber returns ragged rows when a page ends mid table."""
        rows = [
            ["Apply a RBAC model", "", "", "Define access rights management."],
            ["", ""],
        ]
        merged, absorbed = merge_spanned_rows(rows, key_column=0, text_columns=(3,))

        assert absorbed == 0
        assert merged == [
            ("Apply a RBAC model", "Define access rights management."),
        ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_repair.py::TestMergeSpannedRows -v`
Expected: FAIL with `ImportError: cannot import name 'merge_spanned_rows'`.

- [ ] **Step 3: Implement**

```python
# tract/parsers/repair.py — append

def merge_spanned_rows(
    rows: list[list[str]],
    key_column: int,
    text_columns: tuple[int, ...],
) -> tuple[list[tuple[str, str]], int]:
    """Rebuild records from a PDF table whose cells span many extracted rows.

    pdfplumber.extract_tables returns one visual line per row, so a cell that
    wraps over twenty lines arrives as twenty rows with only that column
    populated and the key column empty. A record therefore starts at the row
    that populates *key_column* and continues until the next one does.

    Returns (records, absorbed) where records is [(key, joined_text)] in table
    order and absorbed counts the continuation rows folded into a predecessor.
    The count is what a parser declares and checks: a source refresh that
    changes the wrap points changes this number, and a repair that stops
    absorbing anything means the extraction shape moved under it.

    Rows before the first key are dropped. They are column headers, which
    belong to no record, and attaching them to the first record would prepend
    the word "Definition" to a control statement.
    """
    records: list[tuple[str, list[str]]] = []
    absorbed = 0
    for row in rows:
        padded = list(row) + [""] * (max((key_column, *text_columns)) + 1 - len(row))
        key = _WHITESPACE_RUN.sub(" ", (padded[key_column] or "").replace("\n", " ")).strip()
        pieces = [
            _WHITESPACE_RUN.sub(" ", (padded[column] or "").replace("\n", " ")).strip()
            for column in text_columns
        ]
        text = " ".join(piece for piece in pieces if piece)

        if key:
            records.append((key, [text] if text else []))
            continue
        if not records:
            continue
        if text:
            records[-1][1].append(text)
            absorbed += 1
    return [(key, " ".join(parts)) for key, parts in records], absorbed
```

Add the shared pattern near the other module-level patterns, above `fix_hyphen_breaks`:

```python
# tract/parsers/repair.py — add beside the other compiled patterns

# PDF cells carry hard newlines from the source layout, and extract_tables
# preserves them. Collapsed here rather than in each caller so every table
# derived record is normalised the same way.
_WHITESPACE_RUN: Final[re.Pattern[str]] = re.compile(r"\s+")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_repair.py -v
mypy tract/parsers/repair.py --strict
```

Expected: PASS, no mypy errors, and the pre-existing repair tests still pass.

- [ ] **Step 5: Commit**

```bash
git add tract/parsers/repair.py tests/test_repair.py
git commit -m "feat: rebuild records from PDF table cells that span extracted rows"
```

---

### Task 8: ENISA, 68 links

The hardest source in the plan and the fifth most valuable. It has no stable control identifier of any kind, so OpenCRE fell back to the literal strings `Table 3:` and `Table 5:` as section ids and the real name survives only in `section_name`. **The join is by name.**

Three tables carry the mapping units, and `source-structures.md` documents only one of them.

| block | pages | what it holds | measured |
|---|---|---|---|
| Table 3 | 15 to 16 | threats and sub-threats, with definitions | 13 named rows, 39 then 37 columns |
| Table 5 | 20 to 26 | security controls, with definitions | 35 named rows, 35 then 34 columns |
| Annex C | 39 to 43 | the same controls with implementation guidance | 35 named rows, 5 or 7 columns |

All **[measured]**. Ten of the 33 distinct OpenCRE link names are threats from Table 3, not controls: `evasion`, `poisoning`, `oracle`, `label modification`, `data disclosure`, `model disclosure`, `model or data disclosure`, `compromise of ml application components`, `denial of service due to inconsistent data or a sponge example`, and `use of adversarial examples crafted in white or grey box conditions e g fgsm`. **[measured]** A parser that emits only Table 5 leaves those ten unresolvable, which is why threats are mapping units here.

**The definition column index is not fixed.** On the 35-column Table 5 pages it is 3, on the 34-column pages it is 2, and in Annex C it is 2 on three pages and 4 on the other two. **[measured]** A parser hardcoding column 3 recovers definitions for 18 of 35 controls and looks like the source is thin. The column is chosen by density instead.

Two spelling variants block an exact name join: Table 5 writes "least privileged principle" where Annex C writes "least privilege principle", and Annex C writes "minimise" where OpenCRE writes "minimize". **[measured]** Both spellings are emitted as alternates rather than picking a winner.

**Files:**
- Create: `parsers/parse_enisa.py`
- Create: `tests/test_parse_enisa.py`

**Interfaces:**
- Consumes: `merge_spanned_rows` from Task 7.
- Produces:
  - `EnisaParser` with `framework_id = "enisa"`.
  - `EnisaParser.densest_column(rows: list[list[str]], exclude: tuple[int, ...]) -> int`
  - `EnisaParser.slug(name: str) -> str`
  - `EnisaParser.collapse_key_columns(rows: list[list[str]], columns: tuple[int, ...]) -> list[list[str]]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_enisa.py — create

"""Tests for the ENISA parser.

Every unit under test is pure and takes already extracted table rows. The PDF
boundary is one method that calls pdfplumber and nothing else, so the logic
that decides which column is the definition and how rows merge is testable
without shipping a 2.2 MB PDF into tests/fixtures.
"""

from __future__ import annotations

import pytest

from parsers.parse_enisa import EnisaParser

TABLE5_35_COLUMNS = [
    ["Security controls", "", "Definition"] + [""] * 32,
    ["", "ORGANISATIONAL", ""] + [""] * 32,
    ["Apply a RBAC model,\nrespecting the least privileged\nprinciple", "", "",
     "Define access rights management using a"] + [""] * 31,
    ["", "", "", "RBAC model respecting the least privileged"] + [""] * 31,
    ["", "", "", "principle across every component."] + [""] * 31,
    ["Build explainable models", "", "",
     "Favour a model whose decisions can be explained"] + [""] * 31,
    ["", "", "", "to a reviewer without access to the weights."] + [""] * 31,
]

TABLE5_34_COLUMNS = [
    ["Control all data used by the ML model", "",
     "Check the integrity and the provenance of every"] + [""] * 31,
    ["", "", "dataset the model consumes."] + [""] * 31,
]

ANNEXC_ROWS = [
    ["Security controls", "Examples for operational implementation", "", "",
     "References", "", ""],
    ["Apply a RBAC model,\nrespecting the least privilege\nprinciple", "",
     "The NIST 800-53 and the ISO 27001/2 provide several points:", "",
     "ISO 27001/2", "", ""],
    ["", "", "- Manage access permissions and authorisations.", "", "", "", ""],
]

TABLE3_ROWS = [
    ["Threats | sub-\nthreats", "", "Definition"] + [""] * 36,
    ["", "Data disclosure",
     "This threat refers to a leak of data manipulated by ML algorithms."]
    + [""] * 36,
    ["Compromise of\nML application\ncomponents", "",
     "This threat refers to the compromise of a component or developing"]
    + [""] * 36,
    ["", "", "tool of the ML application."] + [""] * 36,
]


class TestDensestColumn:
    def test_picks_column_three_on_a_thirty_five_column_page(self) -> None:
        assert EnisaParser.densest_column(TABLE5_35_COLUMNS, exclude=(0, 1, 2)) == 3

    def test_picks_column_two_on_a_thirty_four_column_page(self) -> None:
        """Hardcoding 3 recovers 18 of 35 definitions and looks like thin text."""
        assert EnisaParser.densest_column(TABLE5_34_COLUMNS, exclude=(0, 1)) == 2

    def test_raises_when_no_column_carries_text(self) -> None:
        with pytest.raises(ValueError, match="no text column"):
            EnisaParser.densest_column([["", "", ""]], exclude=(0,))


class TestSlug:
    def test_is_stable_lowercase_and_punctuation_free(self) -> None:
        assert EnisaParser.slug("Apply a RBAC model, respecting the least "
                                "privileged principle") == (
            "apply-a-rbac-model-respecting-the-least-privileged-principle"
        )

    def test_collapses_runs_and_strips_edges(self) -> None:
        assert EnisaParser.slug("  Ensure ML applications comply  ") == (
            "ensure-ml-applications-comply"
        )


class TestCollapseKeyColumns:
    def test_takes_the_first_populated_key_column(self) -> None:
        """Table 3 puts a threat in column 0 and a sub-threat in column 1."""
        collapsed = EnisaParser.collapse_key_columns(TABLE3_ROWS, columns=(0, 1))

        keys = [row[0] for row in collapsed]
        assert keys[1] == "Data disclosure"
        assert keys[2].startswith("Compromise of")


class TestTableExtraction:
    def test_a_control_absorbs_its_wrapped_definition(self) -> None:
        records = EnisaParser.records_from_table(
            TABLE5_35_COLUMNS, key_column=0, exclude=(0, 1, 2),
        )
        assert records[0] == (
            "Apply a RBAC model, respecting the least privileged principle",
            "Define access rights management using a RBAC model respecting "
            "the least privileged principle across every component.",
        )

    def test_the_header_row_is_not_a_record(self) -> None:
        records = EnisaParser.records_from_table(
            TABLE5_35_COLUMNS, key_column=0, exclude=(0, 1, 2),
        )
        assert "Security controls" not in {key for key, _ in records}

    def test_a_category_row_is_not_a_record(self) -> None:
        """ORGANISATIONAL sits in column 1 and names a group, not a control."""
        records = EnisaParser.records_from_table(
            TABLE5_35_COLUMNS, key_column=0, exclude=(0, 1, 2),
        )
        assert "ORGANISATIONAL" not in {key for key, _ in records}


class TestControlAssembly:
    def test_annex_c_guidance_is_appended_to_the_table_five_definition(
        self,
    ) -> None:
        controls = EnisaParser.assemble(
            table5=[("Apply a RBAC model, respecting the least privileged "
                     "principle", "Define access rights management.")],
            annexc=[("Apply a RBAC model, respecting the least privilege "
                     "principle", "The NIST 800-53 provides several points.")],
            table3=[],
        )
        assert len(controls) == 1
        assert controls[0].description == (
            "Define access rights management. The NIST 800-53 provides "
            "several points."
        )

    def test_a_spelling_variant_becomes_an_alternate_title(self) -> None:
        """Table 5 says privileged, Annex C says privilege, OpenCRE says both."""
        controls = EnisaParser.assemble(
            table5=[("Apply a RBAC model, respecting the least privileged "
                     "principle", "Define access rights management.")],
            annexc=[("Apply a RBAC model, respecting the least privilege "
                     "principle", "The NIST 800-53 provides several points.")],
            table3=[],
        )
        assert controls[0].metadata is not None
        assert controls[0].metadata["alt_titles"] == [
            "Apply a RBAC model, respecting the least privilege principle"
        ]

    def test_a_threat_is_a_mapping_unit_marked_as_one(self) -> None:
        """Ten of the 33 distinct OpenCRE names are threats, not controls."""
        controls = EnisaParser.assemble(
            table5=[], annexc=[],
            table3=[("Data disclosure", "This threat refers to a leak of data "
                     "manipulated by ML algorithms and is hard to detect.")],
        )
        assert controls[0].control_id == "threat-data-disclosure"
        assert controls[0].metadata is not None
        assert controls[0].metadata["record_type"] == "threat"

    def test_an_annex_c_only_control_is_still_emitted(self) -> None:
        controls = EnisaParser.assemble(
            table5=[],
            annexc=[("Enlarge the training dataset",
                     "Collect further representative samples before training.")],
            table3=[],
        )
        assert [c.control_id for c in controls] == ["enlarge-the-training-dataset"]

    def test_a_record_with_no_text_at_all_raises(self) -> None:
        with pytest.raises(ValueError, match="no definition"):
            EnisaParser.assemble(table5=[("Oracle", "")], annexc=[], table3=[])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_parse_enisa.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'parsers.parse_enisa'`.

- [ ] **Step 3: Implement the parser**

```python
# parsers/parse_enisa.py — create

"""Parser for ENISA, Securing Machine Learning Algorithms (December 2021).

68 curated links and no stable control identifier anywhere in the source.
OpenCRE fell back to the literal strings "Table 3:" and "Table 5:" as section
ids, so the id channel cannot resolve anything and the join is entirely by
name. Control ids here are slugs this parser synthesises, which is honest
because the source offers nothing else, and they are stable because the slug
is a pure function of the name.

Three blocks carry mapping units, measured against the pinned PDF:

  Table 3, pages 15 to 16   13 named threats and sub-threats with definitions
  Table 5, pages 20 to 26   35 named security controls with definitions
  Annex C, pages 39 to 43   the same controls with implementation guidance

Threats are mapping units, not context. Ten of the 33 distinct OpenCRE link
names are Table 3 threats, and a parser that emits only Table 5 leaves every
one of them unresolvable.

The definition column index is not fixed. On the 35 column Table 5 pages it is
3, on the 34 column pages it is 2, and in Annex C it is 2 on three pages and 4
on two. Hardcoding 3 recovers definitions for 18 of 35 controls and reads like
a thin source rather than a wrong column, so the column is chosen by density.

Source: https://www.enisa.europa.eu/publications/securing-machine-learning-algorithms
"""
from __future__ import annotations

import logging
import re
from io import BytesIO
from typing import ClassVar, Final

import pdfplumber

from tract.parsers.base import BaseParser
from tract.parsers.repair import merge_spanned_rows
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PDF_NAME: Final[str] = "enisa_securing_ml_algorithms.pdf"

# One indexed, inclusive, as printed in the document. Pinned by sha256, so
# these cannot drift without the fetch raising first. Each range is checked
# structurally as well, because a page number is a weaker claim than a shape.
TABLE3_PAGES: Final[tuple[int, int]] = (15, 16)
TABLE5_PAGES: Final[tuple[int, int]] = (20, 26)
ANNEXC_PAGES: Final[tuple[int, int]] = (39, 43)

# Table 3 and Table 5 carry the ten lifecycle stage columns as rotated
# headers, which pdfplumber returns as reversed character runs. A table that
# wide is the marker for the right block, and a narrow one means the page
# range moved.
WIDE_TABLE_MIN_COLUMNS: Final[int] = 30
# Annex C is a plain five to seven column table.
ANNEXC_MIN_COLUMNS: Final[int] = 5

# Table 5 groups controls under three all caps category rows in column 1.
CATEGORY_ROWS: Final[frozenset[str]] = frozenset({
    "ORGANISATIONAL", "TECHNICAL", "SPECIFIC ML",
})
HEADER_KEYS: Final[frozenset[str]] = frozenset({
    "Security controls", "Definition", "Threats | sub- threats",
    "Threats | sub-threats",
})

_NON_SLUG: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")
_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


class EnisaParser(BaseParser):
    framework_id: ClassVar[str] = "enisa"
    framework_name: ClassVar[str] = "ENISA"
    version: ClassVar[str] = "December 2021"
    source_url: ClassVar[str] = (
        "https://www.enisa.europa.eu/publications/"
        "securing-machine-learning-algorithms"
    )
    mapping_unit_level: ClassVar[str] = "control"
    # Measure and declare in Step 5. The document states 37 security controls
    # and Table 3 adds its threats on top.
    expected_count: ClassVar[int] = 0
    fetched_date: ClassVar[str] = "2026-08-15"
    min_prose_fraction: ClassVar[float] = 0.90

    @staticmethod
    def slug(name: str) -> str:
        """Stable synthetic id for a source that has none.

        A pure function of the name, so re-parsing the same bytes gives the
        same ids and a reviewer can derive one by hand.
        """
        return _NON_SLUG.sub("-", _WHITESPACE.sub(" ", name).strip().lower()).strip("-")

    @staticmethod
    def densest_column(rows: list[list[str]], exclude: tuple[int, ...]) -> int:
        """Index of the column carrying text on the most rows.

        The definition column moves between pages of the same table, so it is
        found rather than declared. The lifecycle stage columns hold a single
        "x" per cell, which is excluded so a table with ten of them cannot
        outvote the one column holding the prose.
        """
        counts: dict[int, int] = {}
        for row in rows:
            for index, cell in enumerate(row):
                if index in exclude:
                    continue
                text = _WHITESPACE.sub(" ", (cell or "").replace("\n", " ")).strip()
                if text and text != "x":
                    counts[index] = counts.get(index, 0) + 1
        if not counts:
            raise ValueError(
                "no text column: every cell outside the excluded columns is "
                "empty or a lifecycle marker. The page range is wrong."
            )
        return max(sorted(counts), key=lambda index: counts[index])

    @staticmethod
    def collapse_key_columns(
        rows: list[list[str]], columns: tuple[int, ...],
    ) -> list[list[str]]:
        """Move the first populated key column into column 0.

        Table 3 puts a top level threat in column 0 and its sub-threats in
        column 1, and both are mapping units OpenCRE links by name.
        """
        collapsed: list[list[str]] = []
        for row in rows:
            key = ""
            for index in columns:
                if index < len(row):
                    text = _WHITESPACE.sub(
                        " ", (row[index] or "").replace("\n", " "),
                    ).strip()
                    if text:
                        key = text
                        break
            collapsed.append([key] + list(row[1:]))
        return collapsed

    @classmethod
    def records_from_table(
        cls, rows: list[list[str]], key_column: int, exclude: tuple[int, ...],
    ) -> list[tuple[str, str]]:
        """(name, text) pairs from one extracted table, wrapped cells merged."""
        text_column = cls.densest_column(rows, exclude=exclude)
        merged, absorbed = merge_spanned_rows(
            rows, key_column=key_column, text_columns=(text_column,),
        )
        logger.info(
            "table with %d columns: text column %d, %d continuation rows "
            "absorbed", len(rows[0]) if rows else 0, text_column, absorbed,
        )
        return [
            (key, text) for key, text in merged
            if key not in HEADER_KEYS and key not in CATEGORY_ROWS
        ]

    @classmethod
    def assemble(
        cls,
        table5: list[tuple[str, str]],
        annexc: list[tuple[str, str]],
        table3: list[tuple[str, str]],
    ) -> list[Control]:
        """One control per distinct name, joined across the three blocks.

        Table 5 supplies the definition, Annex C the implementation guidance,
        and the two spell two control names differently. Neither spelling is
        preferred: the Table 5 form is the title and the Annex C form is an
        alternate, so an OpenCRE link carrying either resolves.
        """
        guidance = {cls.slug(name): (name, text) for name, text in annexc}
        controls: list[Control] = []
        used: set[str] = set()

        for name, definition in table5:
            key = cls.slug(name)
            used.add(key)
            alternate, extra = guidance.get(key, ("", ""))
            description = " ".join(part for part in (definition, extra) if part)
            if not description:
                raise ValueError(
                    f"enisa: {name!r} has no definition in Table 5 and no "
                    f"guidance in Annex C. The column choice is wrong or the "
                    f"page range moved."
                )
            metadata: dict[str, str | list[str]] = {"record_type": "control"}
            if alternate and alternate.strip() != name.strip():
                metadata["alt_titles"] = [alternate.strip()]
            controls.append(Control(
                control_id=key, title=name.strip(),
                description=description, metadata=metadata,
            ))

        for key, (name, text) in sorted(guidance.items()):
            if key in used or not text:
                continue
            controls.append(Control(
                control_id=key, title=name.strip(), description=text,
                metadata={"record_type": "control"},
            ))

        for name, definition in table3:
            if not definition:
                logger.info("enisa: threat %r has no definition, skipped", name)
                continue
            controls.append(Control(
                control_id=f"threat-{cls.slug(name)}",
                title=name.strip(), description=definition,
                metadata={"record_type": "threat"},
            ))
        return controls

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(PDF_NAME)
        with pdfplumber.open(BytesIO(payload)) as pdf:
            table3_rows = self._widest_tables(pdf, TABLE3_PAGES, WIDE_TABLE_MIN_COLUMNS)
            table5_rows = self._widest_tables(pdf, TABLE5_PAGES, WIDE_TABLE_MIN_COLUMNS)
            annexc_rows = self._widest_tables(pdf, ANNEXC_PAGES, ANNEXC_MIN_COLUMNS)

        table3: list[tuple[str, str]] = []
        for rows in table3_rows:
            table3 += self.records_from_table(
                self.collapse_key_columns(rows, columns=(0, 1)),
                key_column=0, exclude=(0, 1),
            )
        table5: list[tuple[str, str]] = []
        for rows in table5_rows:
            table5 += self.records_from_table(rows, key_column=0, exclude=(0, 1))
        annexc: list[tuple[str, str]] = []
        for rows in annexc_rows:
            annexc += self.records_from_table(rows, key_column=0, exclude=(0, 1))

        controls = self.assemble(table5=table5, annexc=annexc, table3=table3)
        logger.info(
            "%s: %d controls from Table 5, %d guidance rows from Annex C, "
            "%d threats from Table 3, %d mapping units",
            self.framework_id, len(table5), len(annexc), len(table3),
            len(controls),
        )
        return controls

    def _widest_tables(
        self, pdf: pdfplumber.PDF, pages: tuple[int, int], min_columns: int,
    ) -> list[list[list[str]]]:
        """The one real table per page in an inclusive one indexed range.

        pdfplumber finds a dozen spurious single column tables on these pages,
        one per wrapped paragraph, so the widest table on the page is the one
        that matters and the width floor is what says the range is still right.
        """
        collected: list[list[list[str]]] = []
        for number in range(pages[0], pages[1] + 1):
            page = pdf.pages[number - 1]
            tables = [t for t in page.extract_tables() if t]
            if not tables:
                continue
            widest = max(tables, key=lambda t: len(t[0]))
            if len(widest[0]) < min_columns:
                continue
            collected.append([[cell or "" for cell in row] for row in widest])
        if not collected:
            raise ValueError(
                f"{self.framework_id}: no table of at least {min_columns} "
                f"columns on pages {pages[0]} to {pages[1]}. The page range "
                f"moved and every count below it is wrong."
            )
        return collected


def main() -> None:
    EnisaParser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_parse_enisa.py -v
mypy parsers/parse_enisa.py --strict
```

Expected: PASS, no mypy errors.

- [ ] **Step 5: Measure the counts, then declare them**

`expected_count` is `0` in the code above, which `BaseParser._check_expected_count` raises on. That is deliberate: the number is measured here and written in, never guessed.

```bash
python3 - <<'PY'
from parsers.parse_enisa import EnisaParser
from tract.parsers.base import BaseParser

controls = EnisaParser().parse()
threats = [c for c in controls if (c.metadata or {}).get("record_type") == "threat"]
print("mapping units :", len(controls))
print("  controls    :", len(controls) - len(threats))
print("  threats     :", len(threats))
print("prose fraction:", BaseParser.honest_prose_fraction(controls))
short = [c.control_id for c in controls if len(c.description) < 60]
print("under 60 chars:", short)
PY
```

Write the printed `mapping units` into `expected_count` and a floor just under the printed prose fraction into `min_prose_fraction`.

The document states **37 security controls** **[declared]**. If the control count is far from 37, the column choice or the page range is wrong. A count near 35 is expected, because Table 5's first and last pages each merge one control into a neighbour under pdfplumber's default table settings, and that is the known residual: record it in the module docstring rather than tuning the extractor until the number matches.

If the prose fraction is below 0.90, the densest-column choice failed on at least one page. Print the chosen column per page from the log lines before changing anything.

- [ ] **Step 6: Run against the real source and check the join**

```bash
python3 parsers/parse_enisa.py
python3 - <<'PY'
import json, pathlib, re
d = json.loads(pathlib.Path("data/processed/frameworks/enisa.json").read_text())
links = json.loads(
    pathlib.Path("data/training/hub_links_by_framework.json").read_text()
)["enisa"]

def key(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())

names = {key(c["title"]) for c in d["controls"]}
for c in d["controls"]:
    for alt in (c.get("metadata") or {}).get("alt_titles", []):
        names.add(key(alt))
link_names = {key(l["section_name"]) for l in links}
print("controls   :", len(d["controls"]))
print("name join  :", len(link_names & names), "of", len(link_names))
print("unresolved :", sorted(link_names - names))
PY
```

Expected: at least **28 of 33** names joining. **[to measure]** The exact-name baseline before this parser is 16 of 33 against Table 5 alone and 20 of 33 against Annex C alone. **[measured]** Anything at or below 20 means the three blocks are not being unioned.

Record the unresolved list in the commit message. Every entry is either a spelling variant to add as an alternate or a name that genuinely appears in no table, and the difference matters to Task 18.

- [ ] **Step 7: Commit**

```bash
git add parsers/parse_enisa.py tests/test_parse_enisa.py \
        data/processed/frameworks/enisa.json
git commit -m "feat: parse ENISA controls and threats from three PDF tables"
```

---

### Task 9: NIST SSDF, 46 links

The second `extract_tables()` source, and much easier than ENISA because a real identifier exists in the document.

Measured against the pinned PDF: `extract_tables()` returns 12-column rows, the task id sits in column 3 in the form `PO.1.1: <statement>`, and **47 distinct task ids** appear. **[measured]** Five of the 47 are retirement stubs whose whole statement is `Moved to PW.4.4` or similar, and **none of the five is linked**. **[measured]** Dropping them leaves **42 tasks, 41 of which clear 60 characters**; the short one is `RV.2.2` at 54 characters and it is linked. **[measured]**

OpenCRE carries 46 link rows over **44 distinct section ids, 42 of which join**. The two misses are mid-sentence text fragments rather than ids, an artifact of OpenCRE's own extraction that `source-structures.md` already documents for `PS.1.1`. **[measured]** They are reported and never repaired into an id.

The Notional Implementation Examples column is deliberately not read. Its index moves between 6 and 7 within one page **[measured]**, so its row alignment cannot be verified without a second merge pass, and attaching an example to the wrong task is the misattribution failure the repair layer exists to prevent. The task statement alone is a full control statement, so the standing "prefer prose over titles" rule is already satisfied.

**Files:**
- Create: `parsers/parse_nist_ssdf.py`
- Create: `tests/test_parse_nist_ssdf.py`

**Interfaces:**
- Consumes: `merge_spanned_rows` from Task 7.
- Produces: `NistSsdfParser` with `framework_id = "nist_ssdf"`; `NistSsdfParser.tasks_from_rows(rows: list[list[str]]) -> list[tuple[str, str]]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_nist_ssdf.py — create

"""Tests for the NIST SSDF parser.

The unit under test takes already extracted table rows. The PDF boundary is
one method that calls pdfplumber and nothing else.
"""

from __future__ import annotations

import pytest

from parsers.parse_nist_ssdf import NistSsdfParser

ROWS = [
    ["Define Security Requirements for Software Development (PO.1)",
     "Define Security Requirements for Software", "",
     "PO.1.1: Identify and document all security requirements for the",
     "", "", "Example 1: Define policies for securing software", "", "", "",
     "BSAFSS: SM.3", ""],
    ["", "", "",
     "organization's software development infrastructures and processes.",
     "", "", "Example 2: Review the policies annually.", "", "", "", "", ""],
    ["", "", "",
     "PO.1.2: Identify and document all security requirements for "
     "organization-developed software to meet.",
     "", "", "", "Example 1: Define policies for the architecture.", "", "",
     "BSAFSS: SC.1-1", ""],
    ["", "", "", "PW.3.2: Moved to PW.4.4", "", "", "", "", "", "", "", ""],
]


class TestTasksFromRows:
    def test_reads_the_task_id_and_statement_out_of_column_three(self) -> None:
        tasks = dict(NistSsdfParser.tasks_from_rows(ROWS))

        assert tasks["PO.1.1"] == (
            "Identify and document all security requirements for the "
            "organization's software development infrastructures and processes."
        )

    def test_a_statement_wrapped_over_two_rows_is_rejoined(self) -> None:
        tasks = dict(NistSsdfParser.tasks_from_rows(ROWS))
        assert "infrastructures and processes." in tasks["PO.1.1"]

    def test_the_examples_column_is_not_read(self) -> None:
        """Its index moves between 6 and 7 inside one page.

        Attaching an example to the wrong task is a fabricated requirement
        with a plausible provenance record, which is the whole reason the
        repair layer exists.
        """
        tasks = dict(NistSsdfParser.tasks_from_rows(ROWS))
        assert "Example 1" not in tasks["PO.1.1"]
        assert "Example 1" not in tasks["PO.1.2"]

    def test_a_retirement_stub_is_dropped(self) -> None:
        """Five tasks were renumbered in place and none of the five is linked."""
        tasks = dict(NistSsdfParser.tasks_from_rows(ROWS))
        assert "PW.3.2" not in tasks

    def test_the_practice_group_column_does_not_become_a_task(self) -> None:
        tasks = dict(NistSsdfParser.tasks_from_rows(ROWS))
        assert all(key.count(".") == 2 for key in tasks)


class TestControls:
    def test_the_title_is_the_task_id_because_the_source_has_no_short_title(
        self,
    ) -> None:
        """OpenCRE's section_name for this framework is the statement itself.

        The source carries no separate short title, so inventing one would put
        text in the anchor that appears nowhere in the standard.
        """
        controls = NistSsdfParser.to_controls([
            ("PO.1.1", "Identify and document all security requirements for "
                       "the organization's development infrastructure."),
        ])
        assert controls[0].control_id == "PO.1.1"
        assert controls[0].title == "PO.1.1"
        assert controls[0].description.startswith("Identify and document")

    def test_the_statement_is_recorded_as_an_alternate_title(self) -> None:
        """The link's section_name is the statement, so the title channel
        resolves when the id channel does not."""
        controls = NistSsdfParser.to_controls([
            ("PO.1.1", "Identify and document all security requirements for "
                       "the organization's development infrastructure."),
        ])
        assert controls[0].metadata is not None
        assert controls[0].metadata["alt_titles"] == [
            "Identify and document all security requirements for the "
            "organization's development infrastructure."
        ]

    def test_a_duplicate_task_id_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            NistSsdfParser.to_controls([
                ("PO.1.1", "First statement that is long enough to count."),
                ("PO.1.1", "Second statement that is long enough to count."),
            ])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_parse_nist_ssdf.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'parsers.parse_nist_ssdf'`.

- [ ] **Step 3: Implement the parser**

```python
# parsers/parse_nist_ssdf.py — create

"""Parser for NIST SP 800-218, the Secure Software Development Framework 1.1.

46 curated links anchored today on OpenCRE's section_name, which for this
framework is the task statement itself, wrapped mid-sentence by the PDF and in
two cases split across two link rows.

extract_text() is unusable here. It interleaves the Task column with the
adjacent Notional Implementation Examples column, so a statement arrives
truncated with an example spliced into the middle of it. extract_tables()
returns real columns and the task id sits in column 3.

Measured against the pinned PDF: 47 distinct task ids, of which 5 are
retirement stubs whose whole statement is "Moved to PW.4.4" or similar. None
of the 5 is linked, so they are dropped rather than resolved, which keeps the
parser from asserting a requirement the standard retired. That leaves 42
tasks, 41 of them clearing 60 characters. The short one is RV.2.2 at 54
characters and it is a complete sentence.

The Examples column is deliberately unread. Its index moves between 6 and 7
inside a single page, so its row alignment cannot be verified without a second
merge pass, and an example attached to the wrong task is a fabricated
requirement carrying a plausible provenance record.

Source: https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-218.pdf
"""
from __future__ import annotations

import logging
import re
from io import BytesIO
from typing import ClassVar, Final

import pdfplumber

from tract.parsers.base import BaseParser
from tract.parsers.repair import merge_spanned_rows
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PDF_NAME: Final[str] = "nist_sp_800_218.pdf"
# The task column in the 12 column table extract_tables returns. Column 0
# holds the practice group, which spans many task rows and also opens with a
# practice id, so a search across all columns would emit practices as tasks.
TASK_COLUMN: Final[int] = 3
TABLE_COLUMNS: Final[int] = 12

# "PO.1.1: Identify and document ...". Four practice groups: PO, PS, PW, RV.
_TASK: Final[re.Pattern[str]] = re.compile(
    r"^(?P<id>(?:P[OSW]|RV)\.\d+\.\d+):\s*(?P<statement>.*)$", re.DOTALL,
)
# "Moved to PW.4.4", "Moved to PW.4.1 and PW.4.4", "Moved to PW.5.1 as example".
_RETIRED: Final[re.Pattern[str]] = re.compile(r"^Moved to\b")
_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


class NistSsdfParser(BaseParser):
    framework_id: ClassVar[str] = "nist_ssdf"
    framework_name: ClassVar[str] = "NIST SSDF"
    version: ClassVar[str] = "1.1"
    source_url: ClassVar[str] = (
        "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-218.pdf"
    )
    mapping_unit_level: ClassVar[str] = "task"
    expected_count: ClassVar[int] = 42
    fetched_date: ClassVar[str] = "2026-08-15"
    # 41 of 42. RV.2.2 is a complete 54 character sentence.
    min_prose_fraction: ClassVar[float] = 0.97

    @staticmethod
    def tasks_from_rows(rows: list[list[str]]) -> list[tuple[str, str]]:
        """(task_id, statement) for every live task in one extracted table.

        A statement that wraps carries its continuation in later rows with the
        task column empty, which is what merge_spanned_rows rebuilds. Only the
        task column is read, so a continuation row's example text cannot enter
        a statement.
        """
        merged, absorbed = merge_spanned_rows(
            rows, key_column=TASK_COLUMN, text_columns=(TASK_COLUMN,),
        )
        logger.debug("absorbed %d continuation rows", absorbed)

        tasks: list[tuple[str, str]] = []
        for key, text in merged:
            match = _TASK.match(key)
            if match is None:
                continue
            statement = _WHITESPACE.sub(" ", text).strip()
            # merge_spanned_rows returns the key row's own text first, so the
            # statement still carries its "PO.1.1: " prefix. Strip it once.
            prefixed = _TASK.match(statement)
            if prefixed is not None:
                statement = _WHITESPACE.sub(
                    " ", prefixed.group("statement"),
                ).strip()
            if _RETIRED.match(statement):
                logger.info(
                    "nist_ssdf: %s is a retirement stub (%r), dropped",
                    match.group("id"), statement,
                )
                continue
            tasks.append((match.group("id"), statement))
        return tasks

    @staticmethod
    def to_controls(tasks: list[tuple[str, str]]) -> list[Control]:
        """One control per task.

        The title is the task id. The source carries no short title, the task
        statement doubles as both, and inventing a title would put text in the
        anchor that appears nowhere in the standard. The statement is recorded
        as an alternate so the title channel resolves too, since OpenCRE's
        section_name for this framework is the statement.
        """
        seen: set[str] = set()
        controls: list[Control] = []
        for task_id, statement in tasks:
            if task_id in seen:
                raise ValueError(
                    f"nist_ssdf: duplicate task id {task_id}. The same task "
                    f"was extracted twice, which means a page boundary was "
                    f"read twice and one of the two statements is truncated."
                )
            seen.add(task_id)
            controls.append(Control(
                control_id=task_id,
                title=task_id,
                description=statement,
                metadata={"alt_titles": [statement]},
            ))
        return controls

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(PDF_NAME)
        tasks: list[tuple[str, str]] = []
        with pdfplumber.open(BytesIO(payload)) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables():
                    if not table or len(table[0]) != TABLE_COLUMNS:
                        continue
                    rows = [[cell or "" for cell in row] for row in table]
                    tasks += self.tasks_from_rows(rows)
        if not tasks:
            raise ValueError(
                f"{self.framework_id}: no {TABLE_COLUMNS} column tables "
                f"yielded a task id. extract_tables returned a different "
                f"shape, which changes every count in this parser."
            )
        controls = self.to_controls(tasks)
        logger.info("%s: parsed %d live tasks", self.framework_id, len(controls))
        return controls


def main() -> None:
    NistSsdfParser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_parse_nist_ssdf.py -v
mypy parsers/parse_nist_ssdf.py --strict
```

Expected: PASS, no mypy errors.

- [ ] **Step 5: Run against the real source and check the join**

```bash
python3 parsers/parse_nist_ssdf.py
python3 - <<'PY'
import json, pathlib
d = json.loads(pathlib.Path("data/processed/frameworks/nist_ssdf.json").read_text())
links = json.loads(
    pathlib.Path("data/training/hub_links_by_framework.json").read_text()
)["nist_ssdf"]
ids = {c["control_id"] for c in d["controls"]}
link_ids = {l["section_id"] for l in links}
print("controls   :", len(d["controls"]))
print("id join    :", len(link_ids & ids), "of", len(link_ids))
for missing in sorted(link_ids - ids):
    print("unresolved :", missing[:90])
print("prose      :", sum(1 for c in d["controls"] if len(c["description"]) >= 60),
      "of", len(d["controls"]))
PY
```

Expected: `controls : 42`, `id join : 42 of 44`, `prose : 41 of 42`, and the two unresolved entries being sentence fragments rather than ids. **[measured]** If either unresolved entry looks like a task id, a live task was dropped as a stub.

- [ ] **Step 6: Commit**

```bash
git add parsers/parse_nist_ssdf.py tests/test_parse_nist_ssdf.py \
        data/processed/frameworks/nist_ssdf.json
git commit -m "feat: parse NIST SSDF tasks from the extracted practice table"
```

---

### Task 10: ETSI GR SAI 005, 35 links

Prose, not tables, so `extract_text()` is the right tool here and `extract_tables()` is not.

The grain decision is made here rather than discovered during parsing. `section_id` alone does not uniquely identify a mapping unit: `6.2.3` carries three distinct `section_name` values, one per mitigation technique, and the techniques are named only in running prose with no structural marker. **Section-level is the grain.** Every technique sharing a section id gets that section's text, which is the coarse option `source-structures.md` calls option (a), and the reason is that option (b) is a prose heuristic dressed up as a parser.

That choice costs nothing measurable on the join, because `ProseIndex.lookup` falls through to the id channel: **14 of the 16 distinct section ids exist as document sections**. **[measured]** The two that do not are `Data sanitisation` and `Retraining`, rows where OpenCRE's extraction put the technique name in both fields. Both are resolved from evidence rather than guessed:

- `Data sanitisation` maps to **5.2.2**, because a sibling link carries `section_id = 5.2.2` with `section_name = Data sanitisation`. **[measured]**
- `Retraining` maps to **5.3.2**, because its CRE is `854-183 Benign fine-tuning and pruning` and the only other link to that same CRE is `section_id = 5.3.2`. **[measured]**

One id and name disagree and are left alone: `section_id = 5.2` carries `section_name = Backdoor attacks` while the document's 5.2 is "Mitigating poisoning attacks". That is OpenCRE's numbering, not the document's, and re-deriving it is out of scope. It is reported.

Measured document facts: 28 numbered sections under chapters 4 through 7, of which 8 are containers with no text of their own. **[measured]** Two of the 8, `5.2` and `5.3`, are linked, so a container's description is its descendants' text concatenated. Page furniture repeats on every page as `ETSI GR SAI 005 V1.1.1 (2021-03)`, a bare `ETSI`, and a bare page number, and the change-history table after `Annex A:` would otherwise be swallowed by section 7.

**Files:**
- Create: `parsers/parse_etsi.py`
- Create: `tests/test_parse_etsi.py`

**Interfaces:**
- Consumes: `strip_page_furniture` from `tract/parsers/repair.py`.
- Produces: `EtsiParser` with `framework_id = "etsi"`; `EtsiParser.extract_sections(text: str) -> list[tuple[str, str, str]]`; `EtsiParser.fill_containers(sections) -> list[tuple[str, str, str]]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_etsi.py — create

"""Tests for the ETSI GR SAI 005 parser.

The fixture is a synthetic document with the same shapes as the real one: page
furniture on every page, container headings with no text of their own, and the
Annex A boundary that ends the last section.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from parsers.parse_etsi import EtsiParser

DOCUMENT = """\
ETSI GR SAI 005 V1.1.1 (2021-03)
14
1 Scope
The present document describes mitigations against threats to machine learning.
4 Overview
4.1 Machine learning models workflow
A machine learning workflow moves from data collection through to monitoring.
ETSI
15
5 Mitigations against training attacks
5.1 Introduction
Training attacks include poisoning attacks and backdoor attacks.
5.2 Mitigating poisoning attacks
5.2.1 Overview
Poisoning corrupts the training set so the model learns the attacker's rule.
5.2.2 Model enhancement mitigations against poisoning attacks
Data sanitisation removes suspicious samples before training begins.
Retraining on a cleaned dataset restores the model's intended behaviour.
5.3 Mitigating backdoor attacks
5.3.2 Model enhancement mitigations against backdoor attacks
Fine-pruning removes the neurons a backdoor trigger activates.
7 Conclusion
Mitigations must be selected against the attack the deployment actually faces.
Annex A:
Change History
"""


class SampleEtsiParser(EtsiParser):
    """The parser with the fixture's counts rather than the source's."""

    expected_count: ClassVar[int] = 8
    min_prose_fraction: ClassVar[float] = 0.5


@pytest.fixture
def parser(tmp_path: object) -> EtsiParser:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "etsi_gr_sai005_v010101p.pdf").write_bytes(b"%PDF-1.4 placeholder")
    out = tmp_path / "out"
    out.mkdir()
    return SampleEtsiParser(raw_dir=raw, output_dir=out)


class TestExtractSections:
    def test_takes_only_chapters_four_through_seven(self) -> None:
        numbers = [n for n, _, _ in EtsiParser.extract_sections(DOCUMENT)]

        assert "1" not in numbers
        assert numbers[0] == "4"
        assert numbers[-1] == "7"

    def test_page_furniture_never_becomes_a_section(self) -> None:
        titles = [t for _, t, _ in EtsiParser.extract_sections(DOCUMENT)]
        assert not any("ETSI GR SAI" in title for title in titles)

    def test_the_body_stops_at_annex_a(self) -> None:
        sections = {n: b for n, _, b in EtsiParser.extract_sections(DOCUMENT)}
        assert "Change History" not in sections["7"]
        assert sections["7"].startswith("Mitigations must be selected")

    def test_a_container_has_no_text_of_its_own(self) -> None:
        sections = {n: b for n, _, b in EtsiParser.extract_sections(DOCUMENT)}
        assert sections["5.2"] == ""

    def test_the_title_excludes_the_number(self) -> None:
        sections = {n: t for n, t, _ in EtsiParser.extract_sections(DOCUMENT)}
        assert sections["5.2.2"] == (
            "Model enhancement mitigations against poisoning attacks"
        )


class TestFillContainers:
    def test_a_container_takes_its_descendants_text(self) -> None:
        """5.2 and 5.3 are both linked and both empty in the source.

        Dropping them drops three links. Rolling descendants up is the
        section's own content, not synthesis.
        """
        filled = {
            n: b for n, _, b in
            EtsiParser.fill_containers(EtsiParser.extract_sections(DOCUMENT))
        }
        assert "Poisoning corrupts the training set" in filled["5.2"]
        assert "Data sanitisation removes suspicious samples" in filled["5.2"]

    def test_a_leaf_is_untouched(self) -> None:
        filled = {
            n: b for n, _, b in
            EtsiParser.fill_containers(EtsiParser.extract_sections(DOCUMENT))
        }
        assert filled["5.2.1"] == (
            "Poisoning corrupts the training set so the model learns the "
            "attacker's rule."
        )

    def test_a_container_that_is_still_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="no text"):
            EtsiParser.fill_containers([("6", "Mitigations", "")])


class TestNameOnlyAnchors:
    def test_each_anchor_becomes_an_alternate_title_on_its_section(
        self, parser: EtsiParser,
    ) -> None:
        controls = {
            c.control_id: c for c in
            parser.to_controls(
                parser.fill_containers(parser.extract_sections(DOCUMENT))
            )
        }
        assert controls["5.2.2"].metadata is not None
        assert controls["5.2.2"].metadata["alt_titles"] == ["Data sanitisation"]
        assert controls["5.3.2"].metadata is not None
        assert controls["5.3.2"].metadata["alt_titles"] == ["Retraining"]

    def test_an_anchor_whose_phrase_is_absent_raises(
        self, parser: EtsiParser,
    ) -> None:
        """The alias is asserted against the section's own text.

        A mapping carried forward after a source refresh that moved the
        technique would otherwise point a link at a section that no longer
        discusses it.
        """
        sections = [("5.2.2", "Model enhancement mitigations",
                     "This section no longer mentions the technique at all.")]
        with pytest.raises(ValueError, match="Data sanitisation"):
            parser.to_controls(sections)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_parse_etsi.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'parsers.parse_etsi'`.

- [ ] **Step 3: Implement the parser**

```python
# parsers/parse_etsi.py — create

"""Parser for ETSI GR SAI 005 V1.1.1, Securing AI Problem Statement.

35 curated links. The grain is the numbered section, decided here rather than
discovered while parsing.

section_id does not uniquely identify a mapping unit: 6.2.3 carries three
distinct section_name values, one per mitigation technique, and the techniques
are named only in running prose with no heading, no numbering and no other
structural marker. Segmenting them would be a prose heuristic presented as a
parse. Every technique sharing a section id therefore gets that section's text,
and ProseIndex resolves those links through the id channel.

Measured: 14 of the 16 distinct OpenCRE section ids exist as document sections.
The two that do not are rows where OpenCRE put a technique name in both
fields, and both are resolved from evidence in the link data rather than by
reading the prose:

  Data sanitisation -> 5.2.2, because a sibling link carries section_id 5.2.2
                       with section_name "Data sanitisation"
  Retraining        -> 5.3.2, because its CRE is 854-183 and the only other
                       link to that CRE carries section_id 5.3.2

Each alias is asserted against the section's own text before it is emitted, so
a source refresh that moves a technique fails loudly instead of pointing a link
at a section that no longer discusses it.

One id and name disagree and are left alone. section_id 5.2 carries
section_name "Backdoor attacks" while the document's 5.2 is "Mitigating
poisoning attacks". That is OpenCRE's numbering rather than the document's and
re-deriving it is out of scope for a parser.

Source: https://www.etsi.org/deliver/etsi_gr/SAI/001_099/005/01.01.01_60/gr_SAI005v010101p.pdf
"""
from __future__ import annotations

import logging
import re
from io import BytesIO
from typing import ClassVar, Final

import pdfplumber

from tract.parsers.base import BaseParser
from tract.parsers.repair import strip_page_furniture
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PDF_NAME: Final[str] = "etsi_gr_sai005_v010101p.pdf"

# Repeats on every page and would otherwise be read as a numbered heading,
# because the page number and the document identifier share a line.
PAGE_FURNITURE: Final[tuple[str, ...]] = (
    r"^\s*\d*\s*ETSI GR SAI 005 V1\.1\.1 \(2021-03\)\s*$",
    r"^\s*ETSI\s*$",
    r"^\s*\d{1,3}\s*$",
)

# Chapters 1 to 3 are scope, references and definitions. The mapping units
# start at 4 and the last is 7. Anchoring the first token to 4 through 7 also
# excludes the street address on the title page, which matches a bare heading
# pattern otherwise.
_HEADING: Final[re.Pattern[str]] = re.compile(
    r"^([4-7](?:\.\d+){0,2})\s+([A-Z][^\n]{2,90})$", re.MULTILINE,
)
# The change history table follows this line and would otherwise be read as
# the body of section 7.
_END_MARKER: Final[re.Pattern[str]] = re.compile(r"^Annex A:\s*$", re.MULTILINE)

# OpenCRE anchors that carry a technique name in both section fields. Each
# maps to the section the link data itself points at, never to the section a
# reader thinks discusses the technique.
NAME_ONLY_ANCHORS: Final[dict[str, str]] = {
    "Data sanitisation": "5.2.2",
    "Retraining": "5.3.2",
}
_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


class EtsiParser(BaseParser):
    framework_id: ClassVar[str] = "etsi"
    framework_name: ClassVar[str] = "ETSI"
    version: ClassVar[str] = "V1.1.1"
    source_url: ClassVar[str] = (
        "https://www.etsi.org/deliver/etsi_gr/SAI/001_099/005/"
        "01.01.01_60/gr_SAI005v010101p.pdf"
    )
    mapping_unit_level: ClassVar[str] = "section"
    expected_count: ClassVar[int] = 28
    fetched_date: ClassVar[str] = "2026-08-15"
    # Measure and confirm in Step 5. Every container is filled from its
    # descendants, so the only sections under the bar are genuinely short ones.
    min_prose_fraction: ClassVar[float] = 0.95

    @staticmethod
    def extract_sections(text: str) -> list[tuple[str, str, str]]:
        """(number, title, own_body) for chapters 4 through 7, in order.

        own_body is the text between this heading and the next one, so a
        container yields an empty string. fill_containers decides what to do
        about that, which keeps the two concerns separable and testable.
        """
        lines, dropped = strip_page_furniture(text.splitlines(), PAGE_FURNITURE)
        logger.info("dropped %d page furniture lines", dropped)
        cleaned = "\n".join(lines)

        end = _END_MARKER.search(cleaned)
        if end is not None:
            cleaned = cleaned[: end.start()]

        matches = list(_HEADING.finditer(cleaned))
        sections: list[tuple[str, str, str]] = []
        for index, match in enumerate(matches):
            stop = (
                matches[index + 1].start() if index + 1 < len(matches)
                else len(cleaned)
            )
            body = _WHITESPACE.sub(" ", cleaned[match.end():stop]).strip()
            sections.append((match.group(1), match.group(2).strip(), body))
        return sections

    @staticmethod
    def fill_containers(
        sections: list[tuple[str, str, str]],
    ) -> list[tuple[str, str, str]]:
        """Give a container heading its descendants' text.

        5.2 and 5.3 are both linked and both empty in the source. Dropping
        them drops three links. The text is the section's own content one
        level down, which is aggregation rather than synthesis, and the
        alternative of restating the title is the exact defect the prose floor
        exists to catch.
        """
        filled: list[tuple[str, str, str]] = []
        for number, title, body in sections:
            if body:
                filled.append((number, title, body))
                continue
            prefix = f"{number}."
            children = [
                child_body for child_number, _, child_body in sections
                if child_number.startswith(prefix) and child_body
            ]
            if not children:
                raise ValueError(
                    f"etsi: section {number} {title!r} has no text and no "
                    f"descendant with text. The heading pattern matched "
                    f"something that is not a section."
                )
            filled.append((number, title, " ".join(children)))
        return filled

    def to_controls(
        self, sections: list[tuple[str, str, str]],
    ) -> list[Control]:
        """One control per section, with the name-only anchors attached."""
        aliases: dict[str, list[str]] = {}
        for name, number in sorted(NAME_ONLY_ANCHORS.items()):
            aliases.setdefault(number, []).append(name)

        by_number = {number: body for number, _, body in sections}
        for number, names in aliases.items():
            body = by_number.get(number, "")
            for name in names:
                if name.lower() not in body.lower():
                    raise ValueError(
                        f"{self.framework_id}: the OpenCRE anchor {name!r} is "
                        f"mapped to section {number}, whose text does not "
                        f"contain the phrase. Re-derive the mapping from the "
                        f"link data before changing NAME_ONLY_ANCHORS."
                    )

        controls: list[Control] = []
        for number, title, body in sections:
            metadata: dict[str, str | list[str]] | None = None
            if number in aliases:
                metadata = {"alt_titles": sorted(aliases[number])}
            controls.append(Control(
                control_id=number, title=title, description=body,
                metadata=metadata,
            ))
        return controls

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(PDF_NAME)
        with pdfplumber.open(BytesIO(payload)) as pdf:
            text = "\n".join((page.extract_text() or "") for page in pdf.pages)
        sections = self.extract_sections(text)
        if not sections:
            raise ValueError(
                f"{self.framework_id}: no numbered headings in chapters 4 to "
                f"7. Extraction returned a different shape and every count in "
                f"this parser is wrong."
            )
        controls = self.to_controls(self.fill_containers(sections))
        logger.info("%s: parsed %d sections", self.framework_id, len(controls))
        return controls


def main() -> None:
    EtsiParser().run()


if __name__ == "__main__":
    main()
```

The fixture in the test writes a placeholder byte string rather than a real PDF, so the tests exercise `extract_sections`, `fill_containers` and `to_controls` directly and never call `parse()`. That is the same boundary split the ENISA and SSDF tasks use.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_parse_etsi.py -v
mypy parsers/parse_etsi.py --strict
```

Expected: PASS, no mypy errors.

- [ ] **Step 5: Run against the real source and check the join**

```bash
python3 parsers/parse_etsi.py
python3 - <<'PY'
import json, pathlib
d = json.loads(pathlib.Path("data/processed/frameworks/etsi.json").read_text())
links = json.loads(
    pathlib.Path("data/training/hub_links_by_framework.json").read_text()
)["etsi"]
ids = {c["control_id"] for c in d["controls"]}
alt = {a.lower() for c in d["controls"]
       for a in (c.get("metadata") or {}).get("alt_titles", [])}
link_ids = {l["section_id"] for l in links}
print("controls  :", len(d["controls"]))
print("id join   :", len(link_ids & ids), "of", len(link_ids))
print("via alias :", sorted(i for i in link_ids - ids if i.lower() in alt))
print("unresolved:", sorted(i for i in link_ids - ids if i.lower() not in alt))
print("prose     :", sum(1 for c in d["controls"] if len(c["description"]) >= 60),
      "of", len(d["controls"]))
PY
```

Expected: `controls : 28`, `id join : 14 of 16`, `via alias : ['Data sanitisation', 'Retraining']`, `unresolved: []`. **[measured]** Adjust `min_prose_fraction` to just under the measured prose ratio if it lands below 0.95, and say in the docstring which sections are short.

- [ ] **Step 6: Commit**

```bash
git add parsers/parse_etsi.py tests/test_parse_etsi.py \
        data/processed/frameworks/etsi.json
git commit -m "feat: parse ETSI SAI 005 at section grain with evidence-backed aliases"
```

---

### Task 11: SAMM, 30 links

The cleanest source in the plan. 30 stream files whose filename stem is exactly OpenCRE's `section_id` and whose `name` field is exactly its `section_name`, joining **30 of 30**. **[measured]**

The trap is granularity, and `source-structures.md` states it correctly: the repository has 15 practices, 30 streams, and 90 activities, and only the stream filename matches an OpenCRE id. `D-SA-1-A` is not `D-SA-A`.

Stream descriptions already clear the prose bar on all 30, minimum 110 characters. **[measured]** They are also short, a median of 185 characters, while each stream's three activities carry a `longDescription` running 642 to 2,001 characters each. **[measured]** The standing rule is to prefer the fullest available prose, so the description is the stream description followed by the three activity long descriptions in maturity-level order. `BaseParser._sanitize_control` truncates the stored `description` at `DESCRIPTION_MAX_LENGTH` and preserves the whole thing in `full_text`, so nothing is lost.

Activities join to their stream through the activity's `stream` field, which holds the stream's own `id` GUID. Those GUIDs are internal cross references and are never the control id.

**Files:**
- Create: `parsers/parse_samm.py`
- Create: `tests/test_parse_samm.py`

**Interfaces:**
- Consumes: `BaseParser`, `Control`.
- Produces: `SammParser` with `framework_id = "samm"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_samm.py — create

"""Tests for the OWASP SAMM parser.

The fixture carries all three granularities the repository has, because
picking the wrong one is the single way this parser fails: only the stream
filename stem matches an OpenCRE section id.
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path
from typing import ClassVar

import pytest

from parsers.parse_samm import SammParser

STREAM = """\
practice: 4753e55e943c4d418303bf90d599c6b1
id: 253b012094cf4e0988e08fd22609227d
name: Architecture Design
letter: A
description: The design of a software architecture can significantly impact the
  security posture of software, and the use of good security practices will
  improve the overall design.
order: 1
type: Stream
"""

ACTIVITY_ONE = """\
stream: 253b012094cf4e0988e08fd22609227d
id: aaa1
title: Basic architecture design review
shortDescription: Review the design informally.
longDescription: |
  Teams review the design of the system against a short checklist of security
  properties before the first release of a component.
maturity: 1
type: Activity
"""

ACTIVITY_TWO = """\
stream: 253b012094cf4e0988e08fd22609227d
id: aaa2
title: Structured architecture design review
shortDescription: Review the design against a reference architecture.
longDescription: |
  Reviews are performed against a documented reference architecture and the
  findings are tracked to closure.
maturity: 2
type: Activity
"""

PRACTICE = """\
id: 4753e55e943c4d418303bf90d599c6b1
name: Secure Architecture
type: Practice
"""

PREFIX = "core-abc123/model"


class SampleSammParser(SammParser):
    """The parser with the fixture's counts rather than the source's."""

    expected_count: ClassVar[int] = 1
    min_prose_fraction: ClassVar[float] = 1.0


@pytest.fixture
def parser(tmp_path: Path) -> SammParser:
    raw = tmp_path / "raw"
    raw.mkdir()
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{PREFIX}/streams/D-SA-A.yml", STREAM)
        archive.writestr(f"{PREFIX}/activities/D-SA-2-A.yml", ACTIVITY_TWO)
        archive.writestr(f"{PREFIX}/activities/D-SA-1-A.yml", ACTIVITY_ONE)
        archive.writestr(f"{PREFIX}/security_practices/D-Secure-Architecture.yml",
                         PRACTICE)
    (raw / "samm_core.zip").write_bytes(buffer.getvalue())
    out = tmp_path / "out"
    out.mkdir()
    return SampleSammParser(raw_dir=raw, output_dir=out)


class TestSammParser:
    def test_control_id_is_the_stream_filename_stem(
        self, parser: SammParser,
    ) -> None:
        assert [c.control_id for c in parser.parse()] == ["D-SA-A"]

    def test_neither_practices_nor_activities_become_controls(
        self, parser: SammParser,
    ) -> None:
        """D-SA-1-A is not D-SA-A and matches no OpenCRE id."""
        ids = {c.control_id for c in parser.parse()}
        assert "D-SA-1-A" not in ids
        assert "D-Secure-Architecture" not in ids

    def test_the_internal_guid_is_never_the_control_id(
        self, parser: SammParser,
    ) -> None:
        ids = {c.control_id for c in parser.parse()}
        assert "253b012094cf4e0988e08fd22609227d" not in ids

    def test_title_is_the_stream_name(self, parser: SammParser) -> None:
        assert parser.parse()[0].title == "Architecture Design"

    def test_activity_prose_follows_the_stream_description(
        self, parser: SammParser,
    ) -> None:
        description = parser.parse()[0].description

        assert description.startswith("The design of a software architecture")
        assert "short checklist of security properties" in description
        assert "documented reference architecture" in description

    def test_activities_are_ordered_by_maturity_not_by_filename(
        self, parser: SammParser,
    ) -> None:
        """The archive lists D-SA-2-A before D-SA-1-A.

        Reading in archive order would put level 2 prose ahead of level 1 and
        change the anchor without changing any count.
        """
        description = parser.parse()[0].description
        assert description.index("short checklist") < description.index(
            "documented reference architecture"
        )

    def test_an_activity_pointing_at_an_unknown_stream_raises(
        self, tmp_path: Path,
    ) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(f"{PREFIX}/streams/D-SA-A.yml", STREAM)
            archive.writestr(
                f"{PREFIX}/activities/D-XX-1-A.yml",
                ACTIVITY_ONE.replace("253b012094cf4e0988e08fd22609227d", "zzz"),
            )
        (raw / "samm_core.zip").write_bytes(buffer.getvalue())
        out = tmp_path / "out"
        out.mkdir()

        with pytest.raises(ValueError, match="unknown stream"):
            SampleSammParser(raw_dir=raw, output_dir=out).parse()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_parse_samm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'parsers.parse_samm'`.

- [ ] **Step 3: Implement the parser**

```python
# parsers/parse_samm.py — create

"""Parser for OWASP SAMM, the Software Assurance Maturity Model.

30 curated links, joining 30 of 30 on the stream filename stem.

The repository holds three granularities and only one of them is the OpenCRE
join level:

  model/security_practices/  15 practices, filenames like D-Secure-Architecture
  model/streams/             30 streams, filenames like D-SA-A  <- the join
  model/activities/          90 activities, filenames like D-SA-1-A

D-SA-1-A is not D-SA-A and matches no OpenCRE id, and the GUIDs in the id
fields are internal cross references rather than anything OpenCRE links.

The stream description clears the prose bar on all 30, minimum 110 characters,
but its median is 185 while each of the stream's three activities carries a
longDescription of 642 to 2,001 characters. The standing rule is to prefer the
fullest prose available, so the anchor is the stream description followed by
the three activity long descriptions in maturity order. BaseParser truncates
the stored description at DESCRIPTION_MAX_LENGTH and keeps the whole thing in
full_text, so nothing is discarded.

The archive's default branch is develop. owaspsamm/core has no master branch.

Source: https://github.com/owaspsamm/core
"""
from __future__ import annotations

import logging
import zipfile
from io import BytesIO
from pathlib import PurePosixPath
from typing import ClassVar, Final

import yaml

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARCHIVE_NAME: Final[str] = "samm_core.zip"
STREAMS_DIR: Final[str] = "/model/streams/"
ACTIVITIES_DIR: Final[str] = "/model/activities/"
MAX_MEMBER_BYTES: Final[int] = 100_000


class SammParser(BaseParser):
    framework_id: ClassVar[str] = "samm"
    framework_name: ClassVar[str] = "SAMM"
    version: ClassVar[str] = "bc2b5474ab248effbc357c389bec372b0f5e200f"
    source_url: ClassVar[str] = "https://github.com/owaspsamm/core"
    mapping_unit_level: ClassVar[str] = "stream"
    expected_count: ClassVar[int] = 30
    fetched_date: ClassVar[str] = "2026-08-15"
    min_prose_fraction: ClassVar[float] = 1.0

    def parse(self) -> list[Control]:
        streams, activities = self._load()
        by_guid = {
            str(body.get("id", "")): stem for stem, body in streams.items()
        }

        prose: dict[str, list[tuple[int, str]]] = {}
        for stem, body in sorted(activities.items()):
            guid = str(body.get("stream", ""))
            target = by_guid.get(guid)
            if target is None:
                raise ValueError(
                    f"{self.framework_id}: activity {stem} points at unknown "
                    f"stream {guid!r}. The model's internal references are "
                    f"inconsistent and the anchors built from them would be "
                    f"silently incomplete."
                )
            text = str(body.get("longDescription") or "").strip()
            if not text:
                continue
            maturity = int(body.get("maturity", 0))
            prose.setdefault(target, []).append((maturity, text))

        controls: list[Control] = []
        for stem, body in sorted(streams.items()):
            description = str(body.get("description") or "").strip()
            if not description:
                raise ValueError(
                    f"{self.framework_id}: stream {stem} has no description"
                )
            name = str(body.get("name") or "").strip()
            if not name:
                raise ValueError(
                    f"{self.framework_id}: stream {stem} has no name, which is "
                    f"what OpenCRE carries as section_name"
                )
            # Maturity order, not archive order. Reading in archive order puts
            # level 2 prose ahead of level 1 and changes the anchor without
            # changing any count.
            parts = [description] + [
                text for _, text in sorted(prose.get(stem, []))
            ]
            controls.append(Control(
                control_id=stem,
                title=name,
                description=" ".join(parts),
                hierarchy_level="stream",
            ))
        logger.info("%s: parsed %d streams", self.framework_id, len(controls))
        return controls

    def _load(
        self,
    ) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
        """Streams and activities, keyed by filename stem."""
        streams: dict[str, dict[str, object]] = {}
        activities: dict[str, dict[str, object]] = {}
        payload = self.read_source_bytes(ARCHIVE_NAME)
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            for name in sorted(archive.namelist()):
                if not name.endswith(".yml"):
                    continue
                if STREAMS_DIR in name:
                    target = streams
                elif ACTIVITIES_DIR in name:
                    target = activities
                else:
                    continue
                info = archive.getinfo(name)
                if info.file_size > MAX_MEMBER_BYTES:
                    raise ValueError(
                        f"{name}: declares {info.file_size} bytes, over the "
                        f"{MAX_MEMBER_BYTES} byte cap"
                    )
                with archive.open(name) as handle:
                    raw = handle.read(MAX_MEMBER_BYTES + 1)
                if len(raw) > MAX_MEMBER_BYTES:
                    raise ValueError(
                        f"{name}: expanded past the {MAX_MEMBER_BYTES} byte cap"
                    )
                document = yaml.safe_load(raw.decode("utf-8"))
                if not isinstance(document, dict):
                    raise ValueError(
                        f"{name}: parsed as {type(document).__name__}, "
                        f"expected a mapping"
                    )
                target[PurePosixPath(name).stem] = document
        if not streams:
            raise ValueError(
                f"{self.framework_id}: no members under {STREAMS_DIR}. The "
                f"archive layout changed."
            )
        return streams, activities


def main() -> None:
    SammParser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_parse_samm.py -v
mypy parsers/parse_samm.py --strict
```

Expected: PASS, no mypy errors.

- [ ] **Step 5: Run against the real source and check the join**

```bash
python3 parsers/parse_samm.py
python3 - <<'PY'
import json, pathlib
d = json.loads(pathlib.Path("data/processed/frameworks/samm.json").read_text())
links = json.loads(
    pathlib.Path("data/training/hub_links_by_framework.json").read_text()
)["samm"]
ids = {c["control_id"] for c in d["controls"]}
titles = {c["title"].strip().lower() for c in d["controls"]}
link_ids = {l["section_id"] for l in links}
link_names = {l["section_name"].strip().lower() for l in links}
print("controls   :", len(d["controls"]))
print("id join    :", len(link_ids & ids), "of", len(link_ids))
print("title join :", len(link_names & titles), "of", len(link_names))
print("median len :", sorted(len(c["description"]) for c in d["controls"])[15])
PY
```

Expected: `controls : 30`, `id join : 30 of 30`, `title join : 30 of 30`. **[measured]** The median description should be several thousand characters rather than 185, which is the check that the activity prose actually attached.

- [ ] **Step 6: Commit**

```bash
git add parsers/parse_samm.py tests/test_parse_samm.py \
        data/processed/frameworks/samm.json
git commit -m "feat: parse SAMM streams with their activity prose attached"
```

---

### Task 12: An `alt_ids` channel on ProseIndex

`ProseIndex` already carries `alt_titles`, where a control declares extra names it should answer to and an alternate may add a key but never displace a real one. Three frameworks in this plan need the same thing for identifiers, and today there is nowhere to put one.

- **WSTG**: three of the four linked tombstones name their successor in the source, and Task 4 already writes the retired id into `metadata["alt_ids"]`. Nothing reads it.
- **CSA CCM**: OpenCRE's 29 links include seven `IVS-*` ids from v4.0. The workbook is v4.1.0, where that domain was renamed and the controls are `I&S-*` with identical titles and identical numbering. **[measured]**
- **BIML**: eight of the twenty anchors predate OpenCRE's document-prefix convention and carry bare ids like `raw:3`, while the same bare id means something different in the other document.

The `alt_titles` second-pass rule is copied exactly, and for the same reason: NIST AI 100-2 section 2.3's alternate once claimed the key belonging to section 3.2.2's real title, and the eval item resolved to the wrong chapter.

**Files:**
- Modify: `tract/text_selection.py`
- Modify: `tests/test_text_selection.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ProseIndex` reading `control["metadata"]["alt_ids"]` into `_by_id`, alternates applied in a second pass.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_text_selection.py — append

class TestAlternateIds:
    """A control may answer to identifiers the source retired.

    Three frameworks need it. WSTG redirects four linked tombstone ids to
    their successors, CSA CCM's OpenCRE links carry seven v4.0 IVS ids against
    a v4.1.0 workbook that renamed them to I&S, and eight BIML anchors predate
    the document prefix convention.
    """

    @staticmethod
    def _records() -> list[dict[str, object]]:
        return [{
            "framework_name": "WSTG",
            "controls": [
                {
                    "control_id": "WSTG-CRYP-03",
                    "title": "Testing for Weak Encryption",
                    "description": (
                        "Applications that negotiate obsolete ciphers expose "
                        "transported data to an attacker in the path."
                    ),
                    "metadata": {"alt_ids": ["WSTG-ATHN-01"]},
                },
                {
                    "control_id": "WSTG-ATHN-02",
                    "title": "Testing for Default Credentials",
                    "description": (
                        "Shipped credentials that survive installation give an "
                        "attacker an account before the first login attempt."
                    ),
                },
            ],
        }]

    def test_a_retired_id_resolves_to_its_successor(self) -> None:
        index = ProseIndex(self._records())
        hit = index.lookup("WSTG", "WSTG-ATHN-01", None)

        assert hit is not None
        assert "obsolete ciphers" in hit.text

    def test_an_alternate_never_displaces_a_real_id(self) -> None:
        """The failure this mirrors already happened on alt_titles."""
        records = self._records()
        records[0]["controls"][0]["metadata"] = {"alt_ids": ["WSTG-ATHN-02"]}
        index = ProseIndex(records)
        hit = index.lookup("WSTG", "WSTG-ATHN-02", None)

        assert hit is not None
        assert "Shipped credentials" in hit.text

    def test_a_single_string_is_accepted_like_alt_titles(self) -> None:
        records = self._records()
        records[0]["controls"][0]["metadata"] = {"alt_ids": "WSTG-ATHN-01"}
        index = ProseIndex(records)

        assert index.lookup("WSTG", "WSTG-ATHN-01", None) is not None

    def test_the_prose_gate_still_applies_to_an_alternate(self) -> None:
        """A control whose description restates its title indexes nothing.

        An alternate must not become a back door around that.
        """
        records: list[dict[str, object]] = [{
            "framework_name": "WSTG",
            "controls": [{
                "control_id": "WSTG-CRYP-03",
                "title": "Testing for Weak Encryption",
                "description": "Testing for Weak Encryption",
                "metadata": {"alt_ids": ["WSTG-ATHN-01"]},
            }],
        }]
        index = ProseIndex(records)

        assert index.lookup("WSTG", "WSTG-ATHN-01", None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_text_selection.py::TestAlternateIds -v`
Expected: FAIL on the first test, which returns None.

- [ ] **Step 3: Implement**

```python
# tract/text_selection.py — in ProseIndex.__init__, beside pending_alternates

        pending_alternate_ids: list[tuple[tuple[str, str], TextSelection]] = []
```

```python
# tract/text_selection.py — in the per-control loop, after the control_id block

                # Retired identifiers the source itself redirects. WSTG names
                # a tombstone's successor in the document, CSA CCM's workbook
                # renamed a whole domain between the release OpenCRE linked and
                # the one on disk, and eight BIML anchors predate the document
                # prefix convention. Held back for the same reason alternate
                # titles are: an alternate may add a key, never take one.
                alternate_ids = metadata.get("alt_ids") or []
                if isinstance(alternate_ids, str):
                    alternate_ids = [alternate_ids]
                for alternate in alternate_ids:
                    alternate_key = normalize_section_id(str(alternate))
                    if alternate_key:
                        pending_alternate_ids.append(
                            ((framework, alternate_key), selection)
                        )
```

`metadata` is read a few lines below the `control_id` block today. Move the
`metadata = control.get("metadata") or {}` assignment above the `control_id`
block so both alternate kinds read the same local, rather than reading it
twice.

```python
# tract/text_selection.py — beside the existing second pass over alternates

        # Second pass: an alternate id may add a key, never displace a real one.
        for key_pair, selection in pending_alternate_ids:
            if key_pair not in self._by_id:
                self._by_id[key_pair] = selection
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_text_selection.py -v
mypy tract/text_selection.py --strict
```

Expected: PASS, including every pre-existing test in the file.

- [ ] **Step 5: Confirm WSTG's tombstone redirects now resolve**

```bash
python3 - <<'PY'
import json, pathlib
from tract.text_selection import ProseIndex

data = json.loads(
    pathlib.Path("data/processed/frameworks/wstg.json").read_text())
index = ProseIndex([data])
for retired in ("WSTG-ATHN-01", "WSTG-ERRH-02", "WSTG-INPV-03", "WSTG-INPV-13"):
    hit = index.lookup("WSTG", retired, None)
    print(f"{retired}: {'resolved' if hit else 'unresolved'}")
PY
```

Expected: the first three resolved, `WSTG-INPV-13` unresolved. **[measured]** That last one is content the source removed with no successor, and resolving it would mean pointing a link at a document nothing in the source names.

- [ ] **Step 6: Commit**

```bash
git add tract/text_selection.py tests/test_text_selection.py
git commit -m "feat: let a control answer to identifiers its source retired"
```

---

### Task 13: CSA CCM, 29 links

Licensing is settled by owner decision on 2026-08-16: the CCM is redistributable, `csa_ccm` stays out of `RESTRICTED_FRAMEWORK_IDS`, and its processed file is tracked like any other. Nothing in this task gitignores anything.

The open question is version drift, and it is real. The staged workbook is **v4.1.0** while OpenCRE's links were made against v4.0. Measured against the workbook on disk, the 29 distinct link ids split three ways:

- **15 domain short codes** that match a domain header row: `A&A BCR CCC CEK DCS DSP GRC HRS IAM IPY LOG SEF STA TVM UEM`. **[measured]**
- **7 control ids** that match directly, all `AIS-01` through `AIS-07`. **[measured]**
- **7 control ids that do not exist in v4.1.0**, all in the retired `IVS` domain: `IVS-01 IVS-02 IVS-04 IVS-05 IVS-06 IVS-08 IVS-09`. **[measured]**

The `IVS` gap is a rename, not a deletion. v4.1.0 carries an `I&S` domain, "Infrastructure Security", with nine controls at identical numbering and identical titles: OpenCRE's `IVS-01` is named "Infrastructure and Virtualization Security Policy and Procedures" and v4.1.0's `I&S-01` carries that exact title, and the same holds for all seven. **[measured]** The rename is declared as a constant and applied through the `alt_ids` channel from Task 12, which takes the id join from 22 of 29 to **29 of 29**.

The sheet interleaves two row types. **208 control rows** with all four columns populated, and **19 rows with only column A**, of which 17 are domain headers formatted `<Full Name> - <CODE>` and two are trailers, `End of Standard` and a copyright paragraph. **[measured]** `source-structures.md` says 207 control rows. The measured number is 208.

Domain-level links need a domain-level mapping unit, and a domain header row carries no text of its own. Its description is the concatenation of its member controls' specifications, in sheet order, which is the domain's own content and is recorded as such in metadata. The alternative is a title-only control, which is the exact defect the prose floor exists to catch.

Control specifications clear 60 characters on **205 of 208**. **[measured]**

**Files:**
- Create: `parsers/parse_csa_ccm.py`
- Create: `tests/test_parse_csa_ccm.py`

**Interfaces:**
- Consumes: `alt_ids` from Task 12.
- Produces: `CsaCcmParser` with `framework_id = "csa_ccm"`; `CsaCcmParser.rows_to_controls(rows: list[tuple[str, str, str, str]]) -> list[Control]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_csa_ccm.py — create

"""Tests for the CSA Cloud Controls Matrix parser.

The unit under test takes already read sheet rows, so the openpyxl boundary is
one method. The fixture carries every row type the real sheet has: a metadata
row, a header row, domain headers, controls, and the two trailer rows.

Not to be confused with csa_aicm, the AI Controls Matrix, which is a different
framework with zero CRE links.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from parsers.parse_csa_ccm import CsaCcmParser

ROWS: list[tuple[str, str, str, str]] = [
    ('{"specification_version":"4.1.0"}', "CLOUD CONTROLS MATRIX v4.1.0", "", ""),
    ("", "", "", ""),
    ("Control Domain", "Control Title", "Control ID", "Control Specification"),
    ("Audit & Assurance - A&A", "", "", ""),
    ("Audit & Assurance", "Audit and Assurance Policy and Procedures", "A&A-01",
     "Establish, document, approve, communicate, apply, evaluate and maintain\n"
     "audit and assurance policies and procedures and standards."),
    ("Audit & Assurance", "Independent Assessments", "A&A-02",
     "Conduct independent audit and assurance assessments according to\n"
     "relevant standards at least annually."),
    ("Infrastructure Security - I&S", "", "", ""),
    ("Infrastructure Security",
     "Infrastructure and Virtualization Security Policy and Procedures",
     "I&S-01",
     "Establish, document, approve, communicate, apply, evaluate and maintain\n"
     "infrastructure and virtualization security policies and procedures."),
    ("End of Standard", "", "", ""),
    ("© Copyright 2026 Cloud Security Alliance - All rights reserved.", "", "", ""),
]


class SampleCsaCcmParser(CsaCcmParser):
    """The parser with the fixture's counts rather than the workbook's."""

    expected_count: ClassVar[int] = 5
    min_prose_fraction: ClassVar[float] = 1.0


class TestRowsToControls:
    def test_emits_one_control_per_populated_row(self) -> None:
        ids = [c.control_id for c in CsaCcmParser.rows_to_controls(ROWS)]
        assert "A&A-01" in ids
        assert "A&A-02" in ids
        assert "I&S-01" in ids

    def test_emits_one_control_per_domain(self) -> None:
        """15 of the 29 links are domain level, not control level."""
        ids = [c.control_id for c in CsaCcmParser.rows_to_controls(ROWS)]
        assert "A&A" in ids
        assert "I&S" in ids

    def test_the_domain_title_is_the_full_name_without_the_code(self) -> None:
        controls = {c.control_id: c for c in CsaCcmParser.rows_to_controls(ROWS)}
        assert controls["A&A"].title == "Audit & Assurance"

    def test_a_domain_description_aggregates_its_controls(self) -> None:
        """A domain header row carries no text of its own.

        Restating the title instead is the defect the prose floor exists for.
        """
        controls = {c.control_id: c for c in CsaCcmParser.rows_to_controls(ROWS)}
        domain = controls["A&A"]

        assert "audit and assurance policies" in domain.description
        assert "independent audit and assurance assessments" in domain.description
        assert domain.metadata is not None
        assert domain.metadata["aggregated_from"] == ["A&A-01", "A&A-02"]

    def test_the_trailer_rows_are_not_domains(self) -> None:
        ids = {c.control_id for c in CsaCcmParser.rows_to_controls(ROWS)}
        assert "End of Standard" not in ids
        assert not any(cid.startswith("©") for cid in ids)

    def test_the_header_row_is_not_a_control(self) -> None:
        ids = {c.control_id for c in CsaCcmParser.rows_to_controls(ROWS)}
        assert "Control ID" not in ids

    def test_cell_internal_newlines_are_collapsed(self) -> None:
        controls = {c.control_id: c for c in CsaCcmParser.rows_to_controls(ROWS)}
        assert "\n" not in controls["A&A-01"].description
        assert "maintain audit and assurance" in controls["A&A-01"].description

    def test_the_parent_domain_is_recorded(self) -> None:
        controls = {c.control_id: c for c in CsaCcmParser.rows_to_controls(ROWS)}
        assert controls["A&A-01"].parent_id == "A&A"
        assert controls["A&A-01"].parent_name == "Audit & Assurance"


class TestVersionRename:
    def test_a_renamed_control_answers_to_its_v40_id(self) -> None:
        """OpenCRE's 7 IVS ids are v4.0 numbering for v4.1.0's I&S domain.

        Same numbers, same titles, different domain code.
        """
        controls = {c.control_id: c for c in CsaCcmParser.rows_to_controls(ROWS)}
        assert controls["I&S-01"].metadata is not None
        assert controls["I&S-01"].metadata["alt_ids"] == ["IVS-01"]

    def test_a_renamed_domain_answers_to_its_v40_code(self) -> None:
        controls = {c.control_id: c for c in CsaCcmParser.rows_to_controls(ROWS)}
        assert controls["I&S"].metadata is not None
        assert controls["I&S"].metadata["alt_ids"] == ["IVS"]

    def test_a_domain_with_no_rename_carries_no_alternates(self) -> None:
        controls = {c.control_id: c for c in CsaCcmParser.rows_to_controls(ROWS)}
        assert "alt_ids" not in (controls["A&A"].metadata or {})

    def test_a_rename_target_that_is_absent_raises(self) -> None:
        """The rename map is checked against the workbook, not assumed."""
        rows = [row for row in ROWS if not row[0].startswith("Infrastructure")]
        with pytest.raises(ValueError, match="I&S"):
            CsaCcmParser.rows_to_controls(rows, require_rename_targets=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_parse_csa_ccm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'parsers.parse_csa_ccm'`.

- [ ] **Step 3: Implement the parser**

```python
# parsers/parse_csa_ccm.py — create

"""Parser for the CSA Cloud Controls Matrix v4.1.0.

NOT the AI Controls Matrix. csa_aicm is a different framework with 243
controls and zero CRE links, and conflating the two is a documented way to
break this project.

29 curated links across two granularities, which is why this parser emits two
kinds of mapping unit. Measured against the staged workbook, the 29 distinct
link ids are 15 domain short codes, 7 control ids that match directly, and 7
control ids in a domain v4.1.0 renamed.

The rename is not a deletion. OpenCRE's links were made against v4.0, whose
Infrastructure & Virtualization Security domain (IVS) became Infrastructure
Security (I&S) in v4.1.0 with identical control numbering and identical
control titles. OpenCRE's IVS-01 is named "Infrastructure and Virtualization
Security Policy and Procedures" and v4.1.0's I&S-01 carries that exact title,
and the same holds for all seven. The rename is declared below and applied
through the alt_ids channel, which takes the id join from 22 of 29 to 29 of 29.

A domain header row carries no text of its own, so a domain's description is
its member controls' specifications concatenated in sheet order. That is the
domain's own content rather than invented text, and it is recorded in metadata
as aggregated_from. The alternative is a title-only control, which is exactly
what the prose floor exists to refuse.

The workbook is registration gated at cloudsecurityalliance.org and cannot be
fetched by script. It is staged on disk and verified by sha256 in
data/processed/framework_sources.json.

Source: https://cloudsecurityalliance.org/artifacts/cloud-controls-matrix-v4/
"""
from __future__ import annotations

import logging
import re
from io import BytesIO
from typing import ClassVar, Final

import openpyxl

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WORKBOOK_NAME: Final[str] = "CCMv4.1.0-generated_at_2026_01_13.xlsx"
# The CCM sheet holds the controls. CAIQ duplicates the structure as a self
# assessment questionnaire, and Scope Applicability (Mappings) is a stub that
# reads "This dataset is not available yet".
SHEET_NAME: Final[str] = "CCM"
EXPECTED_COLUMNS: Final[int] = 4

# "Cryptography, Encryption & Key Management - CEK". The code is the suffix
# and is what OpenCRE carries as a domain level section_id.
_DOMAIN_HEADER: Final[re.Pattern[str]] = re.compile(r"^(.+?)\s+-\s+([A-Z&]{2,4})$")
# Column A only rows that are not domains.
TRAILER_PREFIXES: Final[tuple[str, ...]] = ("End of Standard", "© Copyright")
HEADER_KEY: Final[str] = "Control Domain"

# Domain codes OpenCRE links under their v4.0 spelling. Key is the v4.1.0
# code, value is the v4.0 code the links carry. Control ids inside a renamed
# domain keep their number, so IVS-04 is I&S-04.
V40_DOMAIN_RENAMES: Final[dict[str, str]] = {"I&S": "IVS"}

_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


class CsaCcmParser(BaseParser):
    framework_id: ClassVar[str] = "csa_ccm"
    # Matches the link's standard_name exactly. There is no alias entry.
    framework_name: ClassVar[str] = "Cloud Controls Matrix"
    version: ClassVar[str] = "4.1.0"
    source_url: ClassVar[str] = (
        "https://cloudsecurityalliance.org/artifacts/cloud-controls-matrix-v4/"
    )
    mapping_unit_level: ClassVar[str] = "control"
    # 208 control rows plus 17 domains.
    expected_count: ClassVar[int] = 225
    fetched_date: ClassVar[str] = "2026-08-15"
    # 205 of 208 specifications clear the bar and every domain aggregate does.
    min_prose_fraction: ClassVar[float] = 0.98

    @classmethod
    def rows_to_controls(
        cls,
        rows: list[tuple[str, str, str, str]],
        require_rename_targets: bool = False,
    ) -> list[Control]:
        """Controls and domains from the CCM sheet's interleaved row types.

        A control row populates all four columns. A domain header populates
        only column A, in the form "<Full Name> - <CODE>". Two column A only
        rows are trailers and are neither.
        """
        controls: list[Control] = []
        domains: list[tuple[str, str]] = []
        members: dict[str, list[tuple[str, str]]] = {}
        current = ""

        for row in rows:
            cells = [_WHITESPACE.sub(" ", str(cell or "")).strip() for cell in row]
            first, title, control_id, specification = (cells + [""] * 4)[:4]

            if control_id and specification:
                if control_id == "Control ID":
                    continue
                controls.append(Control(
                    control_id=control_id,
                    title=title,
                    description=specification,
                    parent_id=current,
                    parent_name=first,
                    hierarchy_level="control",
                    metadata=cls._alternates(control_id),
                ))
                members.setdefault(current, []).append(
                    (control_id, specification)
                )
                continue

            if not first or title or control_id or specification:
                continue
            if first == HEADER_KEY or first.startswith(TRAILER_PREFIXES):
                continue
            header = _DOMAIN_HEADER.match(first)
            if header is None:
                logger.info("csa_ccm: column A only row %r is not a domain", first)
                continue
            current = header.group(2)
            domains.append((current, header.group(1)))

        controls += cls._domain_controls(domains, members)
        cls._check_renames(
            {c.control_id for c in controls}, require_rename_targets,
        )
        return controls

    @classmethod
    def _domain_controls(
        cls,
        domains: list[tuple[str, str]],
        members: dict[str, list[tuple[str, str]]],
    ) -> list[Control]:
        """One mapping unit per domain, described by its own controls."""
        built: list[Control] = []
        for code, name in domains:
            owned = members.get(code, [])
            if not owned:
                raise ValueError(
                    f"csa_ccm: domain {code} has no controls under it. Row "
                    f"ordering changed, and every domain description built "
                    f"from that ordering is wrong."
                )
            metadata: dict[str, str | list[str]] = {
                "aggregated_from": [control_id for control_id, _ in owned],
            }
            alternates = cls._alternates(code)
            if alternates:
                metadata.update(alternates)
            built.append(Control(
                control_id=code,
                title=name,
                description=" ".join(text for _, text in owned),
                hierarchy_level="domain",
                metadata=metadata,
            ))
        return built

    @staticmethod
    def _alternates(identifier: str) -> dict[str, str | list[str]] | None:
        """The v4.0 spelling of an id whose domain was renamed, if any."""
        for current, retired in V40_DOMAIN_RENAMES.items():
            if identifier == current:
                return {"alt_ids": [retired]}
            if identifier.startswith(f"{current}-"):
                return {"alt_ids": [identifier.replace(current, retired, 1)]}
        return None

    @staticmethod
    def _check_renames(identifiers: set[str], required: bool) -> None:
        """Refuse a rename map that no longer matches the workbook.

        A stale entry is worse than none. It would leave the seven v4.0 links
        unresolved while the parser reports that it handled them.
        """
        missing = sorted(
            current for current in V40_DOMAIN_RENAMES if current not in identifiers
        )
        if missing and required:
            raise ValueError(
                f"csa_ccm: V40_DOMAIN_RENAMES names domain(s) {missing} that "
                f"this workbook does not contain. Re-derive the rename from "
                f"the release that changed it."
            )
        if missing:
            logger.warning(
                "csa_ccm: rename target(s) %s absent from this input", missing,
            )

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(WORKBOOK_NAME)
        workbook = openpyxl.load_workbook(
            BytesIO(payload), read_only=True, data_only=True,
        )
        try:
            if SHEET_NAME not in workbook.sheetnames:
                raise ValueError(
                    f"{self.framework_id}: no {SHEET_NAME!r} sheet. Found "
                    f"{workbook.sheetnames}."
                )
            sheet = workbook[SHEET_NAME]
            if sheet.max_column != EXPECTED_COLUMNS:
                raise ValueError(
                    f"{self.framework_id}: the {SHEET_NAME} sheet has "
                    f"{sheet.max_column} columns, expected {EXPECTED_COLUMNS}. "
                    f"The workbook layout changed."
                )
            rows = [
                tuple(str(cell or "") for cell in row)[:EXPECTED_COLUMNS]
                for row in sheet.iter_rows(values_only=True)
            ]
        finally:
            workbook.close()

        controls = self.rows_to_controls(
            [(r[0], r[1], r[2], r[3]) for r in rows],
            require_rename_targets=True,
        )
        logger.info(
            "%s: %d mapping units from %d sheet rows",
            self.framework_id, len(controls), len(rows),
        )
        return controls


def main() -> None:
    CsaCcmParser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_parse_csa_ccm.py -v
mypy parsers/parse_csa_ccm.py --strict
```

Expected: PASS, no mypy errors.

- [ ] **Step 5: Measure the version overlap and report it**

This is the step the original brief asked for, minus the stop. The number is recorded whatever it says.

```bash
python3 parsers/parse_csa_ccm.py
python3 - <<'PY'
import json, pathlib
d = json.loads(pathlib.Path("data/processed/frameworks/csa_ccm.json").read_text())
links = json.loads(
    pathlib.Path("data/training/hub_links_by_framework.json").read_text()
)["csa_ccm"]

ids = {c["control_id"] for c in d["controls"]}
alt = {a for c in d["controls"]
       for a in (c.get("metadata") or {}).get("alt_ids", [])}
link_ids = {l["section_id"] for l in links}

print("mapping units      :", len(d["controls"]))
print("direct id join     :", len(link_ids & ids), "of", len(link_ids))
print("with v4.0 renames  :", len(link_ids & (ids | alt)), "of", len(link_ids))
print("still unresolved   :", sorted(link_ids - ids - alt))
print("v4.0-only ids seen  :", sorted(link_ids & alt))
PY
```

Expected: `mapping units : 225`, `direct id join : 22 of 29`, `with v4.0 renames : 29 of 29`, `still unresolved : []`, and the seven `IVS-*` ids listed as v4.0-only. **[measured]**

**Report this in the commit message body, with the numbers.** The finding is that OpenCRE's CCM links are keyed to v4.0 and that exactly one domain rename separates the two releases for the linked subset. If `still unresolved` is non-empty, the rename map is incomplete and the remaining ids name the domain to add: extend `V40_DOMAIN_RENAMES` only when the v4.1.0 control's title matches the link's `section_name`, and record the pair in the commit message.

- [ ] **Step 6: Confirm the file is tracked, not gitignored**

```bash
git check-ignore -v data/processed/frameworks/csa_ccm.json || echo "not ignored"
python3 -c "
from tract.config import RESTRICTED_FRAMEWORK_IDS
assert RESTRICTED_FRAMEWORK_IDS == frozenset({'iso_27001'}), RESTRICTED_FRAMEWORK_IDS
print('RESTRICTED_FRAMEWORK_IDS unchanged:', sorted(RESTRICTED_FRAMEWORK_IDS))
"
pytest tests/test_licensed_text_not_tracked.py -v
```

Expected: `not ignored`, the assertion passing, and the licensing tests still green. The owner decision is that the CCM is redistributable, so this file goes into git with its prose.

- [ ] **Step 7: Commit**

```bash
git add parsers/parse_csa_ccm.py tests/test_parse_csa_ccm.py \
        data/processed/frameworks/csa_ccm.json
git commit -m "feat: parse CSA CCM v4.1.0 controls and domains"
```

---

### Task 14: BIML, 21 links

Both PDFs are required and neither alone covers OpenCRE's anchors. `ara.pdf` is BIML-78 (2020), 78 distinct risk tags, and `BIML-LLM24.pdf` is BIML-24(LLM), 71 distinct risk tags. **[measured]**

Both documents mark each named risk with an inline `[category:number:label]` tag and follow it with the risk's definition, and **both reuse the same category vocabulary with different meanings**. `ara.pdf`'s `raw:3` is "storage" and `BIML-LLM24.pdf`'s `raw:3` is "data feudalism". Document provenance is therefore part of the identifier, which is exactly what OpenCRE's own prefix convention does.

Twenty distinct anchors over 21 link rows. Eight carry `BIML-78(2020): `, four carry `BIML-24(LLM): `, and eight are unprefixed legacy anchors. The eight legacy ones are resolved by the `alt_ids` channel from Task 12 against a declared map, and that map is copied from the fetch phase's evidence:

| legacy id | document | evidence |
|---|---|---|
| `alg:11` | ara | `[alg:11:parameters]` in ara only |
| `inference:4` | ara | `[inference:4:hosting]` in ara only |
| `input:2` | ara | `[input:2:controlled input stream]` in ara only |
| `model:2` | ara | `[model:2:Trojan]` in ara only |
| `raw:3` | ara | `[raw:3:storage]` in ara, "data feudalism" in LLM24 |
| `inference:9` | LLM24 | `[inference:9:hosting]` in LLM24, absent from ara |
| `output:4` | LLM24 | `[output:4:data confidentiality]` in LLM24, "inscrutability" in ara |
| `output:2` | **unresolved** | ara has `[output:1:direct]` and `[output:2:provenance]`, LLM24 has `output:2` as "wrongness". No exact match either way. |

All **[measured]**. `output:2`, which OpenCRE names "Direct Output", **is flagged and never guessed**. Assigning it to `ara`'s `output:1` on name proximity would attach a risk definition to an anchor the source does not support, and assigning it to either document's real `output:2` attaches the wrong definition outright.

Two extraction hazards. A tag is used both to define a risk and to cross-reference one, as in `[data:4:storage] As in [raw:3:storage], data may be stored...`, so the segment after a cross-reference belongs to a different risk. The rule is to take, per id, the longest following segment above a floor, which selects the definition and rejects the reference. A risk is also restated in an "Associated controls" section, which is a second real segment and shorter than the definition.

**Files:**
- Create: `parsers/parse_biml.py`
- Create: `tests/test_parse_biml.py`

**Interfaces:**
- Consumes: `alt_ids` from Task 12.
- Produces: `BimlParser` with `framework_id = "biml"`; `BimlParser.risks_from_text(text: str, prefix: str) -> list[Control]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_biml.py — create

"""Tests for the BIML parser.

Both PDFs use inline [category:number:label] tags and reuse the same category
vocabulary with different meanings, so the unit under test takes extracted
text plus the document prefix and never opens a PDF.
"""

from __future__ import annotations

import pytest

from parsers.parse_biml import BimlParser

ARA_TEXT = """\
Risks in the raw data stage follow.
[raw:3:storage] Data are stored and managed in an insecure fashion. Who has
access to the data pool, and why? Access controls can help mitigate this risk,
but such controls are not feasible when public data sources are in use.
[data:4:storage] As in [raw:3:storage], data may be stored and managed in an
insecure fashion, and the same access questions apply to the assembled set.
[model:2:Trojan] Model transfer leads to the possibility that what is being
reused may be a Trojaned or otherwise damaged version of the model sought out.
Associated controls follow. The labels refer to the original risks above.
[raw:3:storage] Encrypt the data pool at rest.
"""

LLM_TEXT = """\
[raw:3:data feudalism] A small number of organizations control the data that
every downstream model is trained on, and their curation choices propagate.
[inference:9:hosting] Many LLM systems run on hosted, remote servers, and
those machines need protection against both ordinary and model-level attacks.
"""


class TestRisksFromText:
    def test_a_tag_becomes_a_control_with_the_document_prefix(self) -> None:
        controls = {c.control_id: c for c in
                    BimlParser.risks_from_text(ARA_TEXT, prefix="BIML-78(2020): ")}
        assert "BIML-78(2020): raw:3" in controls

    def test_the_title_is_the_tag_label_title_cased(self) -> None:
        controls = {c.control_id: c for c in
                    BimlParser.risks_from_text(ARA_TEXT, prefix="BIML-78(2020): ")}
        assert controls["BIML-78(2020): raw:3"].title == "Storage"

    def test_the_definition_is_the_segment_that_follows_the_tag(self) -> None:
        controls = {c.control_id: c for c in
                    BimlParser.risks_from_text(ARA_TEXT, prefix="BIML-78(2020): ")}
        description = controls["BIML-78(2020): raw:3"].description

        assert description.startswith("Data are stored")
        assert "public data sources" in description

    def test_a_cross_reference_does_not_steal_a_definition(self) -> None:
        """[data:4:storage] As in [raw:3:storage], data may be stored ...

        The text after the inline reference belongs to data:4. Taking the
        longest segment per id selects the real definition instead.
        """
        controls = {c.control_id: c for c in
                    BimlParser.risks_from_text(ARA_TEXT, prefix="BIML-78(2020): ")}
        assert "the same access questions apply" not in (
            controls["BIML-78(2020): raw:3"].description
        )
        assert "the same access questions apply" in (
            controls["BIML-78(2020): data:4"].description
        )

    def test_the_associated_controls_restatement_does_not_win(self) -> None:
        controls = {c.control_id: c for c in
                    BimlParser.risks_from_text(ARA_TEXT, prefix="BIML-78(2020): ")}
        assert controls["BIML-78(2020): raw:3"].description != (
            "Encrypt the data pool at rest."
        )

    def test_the_same_bare_id_means_different_things_in_each_document(
        self,
    ) -> None:
        ara = {c.control_id: c for c in
               BimlParser.risks_from_text(ARA_TEXT, prefix="BIML-78(2020): ")}
        llm = {c.control_id: c for c in
               BimlParser.risks_from_text(LLM_TEXT, prefix="BIML-24(LLM): ")}

        assert ara["BIML-78(2020): raw:3"].title == "Storage"
        assert llm["BIML-24(LLM): raw:3"].title == "Data Feudalism"

    def test_a_segment_under_the_floor_is_not_a_definition(self) -> None:
        controls = BimlParser.risks_from_text(
            "[raw:9:tiny] Short. [raw:10:next] " + "Long enough. " * 12,
            prefix="BIML-78(2020): ",
        )
        assert "BIML-78(2020): raw:9" not in {c.control_id for c in controls}


class TestLegacyAnchors:
    def test_an_ara_legacy_anchor_becomes_an_alternate_id(self) -> None:
        controls = {c.control_id: c for c in
                    BimlParser.risks_from_text(ARA_TEXT, prefix="BIML-78(2020): ")}
        assert controls["BIML-78(2020): raw:3"].metadata is not None
        assert controls["BIML-78(2020): raw:3"].metadata["alt_ids"] == ["raw:3"]

    def test_a_legacy_anchor_is_not_added_to_the_wrong_document(self) -> None:
        """LLM24's raw:3 is data feudalism, so the legacy raw:3 is not its id."""
        controls = {c.control_id: c for c in
                    BimlParser.risks_from_text(LLM_TEXT, prefix="BIML-24(LLM): ")}
        assert "alt_ids" not in (controls["BIML-24(LLM): raw:3"].metadata or {})

    def test_the_llm_legacy_anchor_attaches_to_the_llm_document(self) -> None:
        controls = {c.control_id: c for c in
                    BimlParser.risks_from_text(LLM_TEXT, prefix="BIML-24(LLM): ")}
        assert controls["BIML-24(LLM): inference:9"].metadata is not None
        assert controls["BIML-24(LLM): inference:9"].metadata["alt_ids"] == [
            "inference:9"
        ]

    def test_the_ambiguous_anchor_is_assigned_to_neither_document(self) -> None:
        """output:2 is "Direct Output" in OpenCRE and matches nothing.

        ara has output:1 "direct" and output:2 "provenance", LLM24 has
        output:2 "wrongness". Every assignment is wrong, so none is made.
        """
        from parsers.parse_biml import UNRESOLVED_ANCHORS, LEGACY_ANCHORS

        assert "output:2" in UNRESOLVED_ANCHORS
        assert "output:2" not in LEGACY_ANCHORS

    def test_every_legacy_anchor_names_a_real_document(self) -> None:
        from parsers.parse_biml import ARA_PREFIX, LEGACY_ANCHORS, LLM_PREFIX

        assert set(LEGACY_ANCHORS.values()) == {ARA_PREFIX, LLM_PREFIX}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_parse_biml.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'parsers.parse_biml'`.

- [ ] **Step 3: Implement the parser**

```python
# parsers/parse_biml.py — create

"""Parser for the Berryville Institute of Machine Learning risk analyses.

Two documents, both required. ara.pdf is BIML-78 (2020) with 78 distinct risk
tags, and BIML-LLM24.pdf is BIML-24(LLM) with 71. Neither alone covers
OpenCRE's 20 distinct anchors.

Both documents mark a named risk with an inline [category:number:label] tag
and follow it with the risk's definition, and both reuse the same category
vocabulary with different meanings. ara's raw:3 is storage and LLM24's raw:3
is data feudalism. Document provenance is therefore part of the identifier,
which is what OpenCRE's own prefix convention does and what this parser
copies.

Eight of the 20 anchors predate that convention and carry a bare id. Seven are
assigned to a document on measured evidence, recorded in LEGACY_ANCHORS below,
and reach their control through the alt_ids channel. The eighth, output:2, is
named "Direct Output" by OpenCRE and matches nothing: ara has output:1 "direct"
and output:2 "provenance", LLM24 has output:2 "wrongness". Assigning it on name
proximity would attach a definition the source does not support, so it stays in
UNRESOLVED_ANCHORS and is logged.

Two extraction hazards. A tag is used both to define a risk and to
cross-reference one, as in "[data:4:storage] As in [raw:3:storage], data may
be stored ...", so the text after a cross-reference belongs to a different
risk. A risk is also restated briefly in an Associated controls section. Taking
the longest segment per id above a length floor selects the definition in both
cases.

Sources: https://berryvilleiml.com/results/ara.pdf
         https://berryvilleiml.com/results/BIML-LLM24.pdf
"""
from __future__ import annotations

import logging
import re
from io import BytesIO
from typing import ClassVar, Final

import pdfplumber

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARA_PDF: Final[str] = "ara.pdf"
LLM_PDF: Final[str] = "BIML-LLM24.pdf"
# OpenCRE's own prefixes, character for character. A different spelling here
# breaks 12 of the 20 anchors and nothing would notice.
ARA_PREFIX: Final[str] = "BIML-78(2020): "
LLM_PREFIX: Final[str] = "BIML-24(LLM): "

# "[raw:3:storage]". The label runs to the closing bracket and may contain
# spaces.
_TAG: Final[re.Pattern[str]] = re.compile(r"\[([a-z]+):(\d+):([^\]]+)\]")
# Below this a segment is a cross-reference or a one line restatement rather
# than a risk definition. The shortest real definition in either document runs
# well past this.
MIN_DEFINITION_CHARS: Final[int] = 120

# Bare anchors that predate OpenCRE's prefix convention, each assigned to the
# document whose tag label matches OpenCRE's section_name exactly.
LEGACY_ANCHORS: Final[dict[str, str]] = {
    "alg:11": ARA_PREFIX,
    "inference:4": ARA_PREFIX,
    "input:2": ARA_PREFIX,
    "model:2": ARA_PREFIX,
    "raw:3": ARA_PREFIX,
    "inference:9": LLM_PREFIX,
    "output:4": LLM_PREFIX,
}
# Anchors with no defensible assignment. NEVER move one of these into
# LEGACY_ANCHORS on name proximity. output:2 is OpenCRE's "Direct Output" and
# ara's output:1 is "direct", one off in the number, while both documents'
# real output:2 means something else entirely. Attaching either is a wrong
# risk definition carrying a correct looking id.
UNRESOLVED_ANCHORS: Final[frozenset[str]] = frozenset({"output:2"})

_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


class BimlParser(BaseParser):
    framework_id: ClassVar[str] = "biml"
    framework_name: ClassVar[str] = "BIML"
    version: ClassVar[str] = "BIML-78 v1.0 and BIML-24(LLM) v1.0"
    source_url: ClassVar[str] = "https://berryvilleiml.com/results/"
    mapping_unit_level: ClassVar[str] = "risk"
    # Measure and declare in Step 5. 78 and 71 distinct tags exist, and the
    # ones whose longest segment falls under the floor are cross references
    # rather than definitions.
    expected_count: ClassVar[int] = 0
    fetched_date: ClassVar[str] = "2026-08-15"
    min_prose_fraction: ClassVar[float] = 1.0

    @staticmethod
    def risks_from_text(text: str, prefix: str) -> list[Control]:
        """One control per risk tag that carries a definition.

        Segments are the spans between consecutive tags. Per id the longest
        segment wins, which rejects both the cross-reference case, where the
        span after the tag belongs to a neighbouring risk, and the Associated
        controls restatement, which is a real but much shorter span.
        """
        flat = _WHITESPACE.sub(" ", text)
        matches = list(_TAG.finditer(flat))
        best: dict[str, tuple[str, str]] = {}
        for index, match in enumerate(matches):
            identifier = f"{match.group(1)}:{match.group(2)}"
            label = match.group(3).strip()
            stop = (
                matches[index + 1].start() if index + 1 < len(matches)
                else len(flat)
            )
            segment = flat[match.end():stop].strip()
            if len(segment) < MIN_DEFINITION_CHARS:
                continue
            current = best.get(identifier)
            if current is None or len(segment) > len(current[1]):
                best[identifier] = (label, segment)

        controls: list[Control] = []
        for identifier, (label, segment) in sorted(best.items()):
            metadata: dict[str, str | list[str]] | None = None
            if LEGACY_ANCHORS.get(identifier) == prefix:
                metadata = {"alt_ids": [identifier]}
            controls.append(Control(
                control_id=f"{prefix}{identifier}",
                title=label.title(),
                description=segment,
                metadata=metadata,
            ))
        return controls

    def parse(self) -> list[Control]:
        controls: list[Control] = []
        for name, prefix in ((ARA_PDF, ARA_PREFIX), (LLM_PDF, LLM_PREFIX)):
            text = self._extract(name)
            found = self.risks_from_text(text, prefix=prefix)
            if not found:
                raise ValueError(
                    f"{self.framework_id}: no risk tags with a definition in "
                    f"{name}. The bracket tag convention is what makes this "
                    f"source parseable at all."
                )
            logger.info("%s: %d risks from %s", self.framework_id, len(found), name)
            controls += found

        for anchor in sorted(UNRESOLVED_ANCHORS):
            logger.warning(
                "%s: the legacy anchor %r has no defensible document "
                "assignment and stays unresolved. OpenCRE names it "
                "differently from both documents' tag at that number, so any "
                "assignment attaches a definition the source does not support.",
                self.framework_id, anchor,
            )
        self._check_unique(controls)
        return controls

    def _extract(self, name: str) -> str:
        payload = self.read_source_bytes(name)
        with pdfplumber.open(BytesIO(payload)) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages)

    def _check_unique(self, controls: list[Control]) -> None:
        """The prefix is what keeps the two documents' ids apart.

        A duplicate means a prefix was dropped, and the second document's risk
        would silently overwrite the first's in every index downstream.
        """
        seen: set[str] = set()
        for control in controls:
            if control.control_id in seen:
                raise ValueError(
                    f"{self.framework_id}: duplicate id {control.control_id}. "
                    f"Both documents reuse the same category vocabulary, so "
                    f"the document prefix is what keeps them apart."
                )
            seen.add(control.control_id)


def main() -> None:
    BimlParser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_parse_biml.py -v
mypy parsers/parse_biml.py --strict
```

Expected: PASS, no mypy errors.

- [ ] **Step 5: Measure the count, then declare it**

```bash
python3 - <<'PY'
from parsers.parse_biml import ARA_PREFIX, BimlParser, LLM_PREFIX
from tract.parsers.base import BaseParser

parser = BimlParser()
controls = parser.parse()
ara = [c for c in controls if c.control_id.startswith(ARA_PREFIX)]
llm = [c for c in controls if c.control_id.startswith(LLM_PREFIX)]
print("risks total:", len(controls))
print("  ara      :", len(ara), "of 78 tags")
print("  llm24    :", len(llm), "of 71 tags")
print("prose      :", BaseParser.honest_prose_fraction(controls))
PY
```

Write the printed total into `expected_count`. A count well under 149 is expected and correct, because a tag used only as a cross-reference carries no definition of its own. A count of exactly 149 means the length floor is not filtering anything and cross-references are being emitted as risks.

- [ ] **Step 6: Run against the real sources and check the join**

```bash
python3 parsers/parse_biml.py
python3 - <<'PY'
import json, pathlib
d = json.loads(pathlib.Path("data/processed/frameworks/biml.json").read_text())
links = json.loads(
    pathlib.Path("data/training/hub_links_by_framework.json").read_text()
)["biml"]
ids = {c["control_id"] for c in d["controls"]}
alt = {a for c in d["controls"]
       for a in (c.get("metadata") or {}).get("alt_ids", [])}
link_ids = {l["section_id"] for l in links}
print("risks       :", len(d["controls"]))
print("prefixed    :", len(link_ids & ids), "of", len(link_ids))
print("with legacy :", len(link_ids & (ids | alt)), "of", len(link_ids))
print("unresolved  :", sorted(link_ids - ids - alt))
PY
```

Expected: `prefixed : 12 of 20`, `with legacy : 19 of 20`, `unresolved : ['output:2']`. **[measured]** If `output:2` resolves, someone moved it out of `UNRESOLVED_ANCHORS` and the parser is now asserting a risk definition the source does not support.

- [ ] **Step 7: Commit**

```bash
git add parsers/parse_biml.py tests/test_parse_biml.py \
        data/processed/frameworks/biml.json
git commit -m "feat: parse both BIML risk analyses with document-scoped ids"
```

---

### Task 15: OWASP Top 10 2021, 17 links

The smallest recovery and the largest archive. `owasp_top10_2021.zip` is 196,415,531 bytes because the repository carries every historical edition from 2003 to 2025, every translation of each, and the PDF and PPTX exports. Only `2021/docs/en/` matters, and members are selected by name from the central directory rather than extracted wholesale.

Measured: `2021/docs/en/` holds 18 markdown files, of which **10 are categories** matching `A01_2021` through `A10_2021`. **[measured]** The other eight are front matter, and `source-structures.md` counts only three of them: there are three `A00_2021*` files plus `A11_2021-Next_Steps.md`, `0x00_2021-introduction.md`, `0x00_2021-notice.md`, `0x01_2021-about-owasp.md` and `index.md`. `A11` is the trap, because a pattern of `A\d\d_2021` matches it and Next Steps is not a category.

All ten `## Description` bodies clear 60 characters, from 582 to 2,944. **[measured]** All ten ids join. **[measured]**

Three of the ten link names differ cosmetically from the document's own title: OpenCRE says "Broken Access Controls" against "Broken Access Control", "Logging and Monitoring Failures" against "Security Logging and Monitoring Failures", and "Server Side Request Forgery (SSRF)" against "Server-Side Request Forgery (SSRF)". **[measured]** The document's title wins, per the standing preference for source prose over a secondary label, and OpenCRE's spelling is recorded as an alternate so the title channel resolves as well as the id channel.

The H1 carries a trailing image, `# A01:2021 – Broken Access Control    ![icon](assets/...)`, and the separator is an en dash rather than a hyphen.

**Files:**
- Create: `parsers/parse_owasp_top10_2021.py`
- Create: `tests/test_parse_owasp_top10_2021.py`

**Interfaces:**
- Consumes: `BaseParser`, `Control`.
- Produces: `OwaspTop102021Parser` with `framework_id = "owasp_top10_2021"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_owasp_top10_2021.py — create

"""Tests for the OWASP Top 10 2021 parser.

The fixture carries the two decoys the real archive has: an A00 front matter
file and A11_2021-Next_Steps.md, which any A-digit-digit pattern matches and
which is not a category.
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path
from typing import ClassVar

import pytest

from parsers.parse_owasp_top10_2021 import OwaspTop102021Parser

A01 = """\
# A01:2021 – Broken Access Control    ![icon](assets/TOP_10_Icons.png){: style="height:80px"}

## Factors

| CWEs Mapped | Max Incidence Rate |
|:-----------:|:------------------:|
| 34          | 55.97%             |

## Overview

Moving up from the fifth position, 94% of applications were tested for some
form of broken access control.

## Description

Access control enforces policy such that users cannot act outside of their
intended permissions. Failures typically lead to unauthorized information
disclosure, modification, or destruction of all data.

## How to Prevent

Deny by default, except for public resources.
"""

A09 = """\
# A09:2021 – Security Logging and Monitoring Failures    ![icon](assets/i.png)

## Description

Logging and monitoring failures, coupled with missing or ineffective
integration with incident response, allow attackers to persist and pivot to
more systems without detection.

## How to Prevent

Ensure all login and access control failures can be logged.
"""

A11 = """\
# A11:2021 – Next Steps

## Description

The Top 10 is not the end of the list, and four further categories nearly
made the cut this cycle.
"""

A00 = """\
# Introduction

## Welcome to the OWASP Top 10 2021
"""

PREFIX = "Top10-abc123/2021/docs/en"


class SampleTop10Parser(OwaspTop102021Parser):
    """The parser with the fixture's counts rather than the source's."""

    expected_count: ClassVar[int] = 2
    min_prose_fraction: ClassVar[float] = 1.0


@pytest.fixture
def parser(tmp_path: Path) -> OwaspTop102021Parser:
    raw = tmp_path / "raw"
    raw.mkdir()
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{PREFIX}/A01_2021-Broken_Access_Control.md", A01)
        archive.writestr(
            f"{PREFIX}/A09_2021-Security_Logging_and_Monitoring_Failures.md", A09)
        archive.writestr(f"{PREFIX}/A11_2021-Next_Steps.md", A11)
        archive.writestr(f"{PREFIX}/A00_2021_Introduction.md", A00)
        archive.writestr("Top10-abc123/2021/docs/fr/A01_2021-Controle.md", A01)
        archive.writestr("Top10-abc123/archives/2017/A1_2017-Injection.md", A01)
    (raw / "owasp_top10_2021.zip").write_bytes(buffer.getvalue())
    out = tmp_path / "out"
    out.mkdir()
    return SampleTop10Parser(raw_dir=raw, output_dir=out)


class TestTop102021Parser:
    def test_control_id_is_the_bare_category_token(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        assert sorted(c.control_id for c in parser.parse()) == ["A01", "A09"]

    def test_next_steps_is_not_a_category(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        """A11 matches any A-digit-digit pattern and is not one of the ten."""
        assert "A11" not in {c.control_id for c in parser.parse()}

    def test_front_matter_is_not_a_category(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        assert "A00" not in {c.control_id for c in parser.parse()}

    def test_other_languages_and_editions_are_not_read(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        """The archive is 196 MB of translations and historical editions."""
        assert len(parser.parse()) == 2

    def test_the_title_drops_the_number_and_the_icon(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        controls = {c.control_id: c for c in parser.parse()}
        assert controls["A01"].title == "Broken Access Control"

    def test_the_description_is_the_description_section_only(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        controls = {c.control_id: c for c in parser.parse()}
        description = controls["A01"].description

        assert description.startswith("Access control enforces policy")
        assert "Moving up from the fifth position" not in description
        assert "Deny by default" not in description
        assert "CWEs Mapped" not in description

    def test_opencres_spelling_is_kept_as_an_alternate_title(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        """OpenCRE says "Broken Access Controls", the document says singular."""
        controls = {c.control_id: c for c in parser.parse()}
        assert controls["A01"].metadata is not None
        assert controls["A01"].metadata["alt_titles"] == ["Broken Access Controls"]

    def test_a_category_with_no_alternate_carries_no_alt_titles(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        controls = {c.control_id: c for c in parser.parse()}
        assert controls["A09"].metadata is not None
        assert controls["A09"].metadata["alt_titles"] == [
            "Logging and Monitoring Failures"
        ]

    def test_reads_the_archive_through_the_recording_reader(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        parser.parse()
        assert "owasp_top10_2021.zip" in parser._source_files
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_parse_owasp_top10_2021.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the parser**

```python
# parsers/parse_owasp_top10_2021.py — create

"""Parser for the OWASP Top 10 2021.

17 curated links across ten categories, and the largest archive in this plan
by a wide margin: 196 MB, because the repository carries every edition from
2003 to 2025, every translation of each, and the PDF and PPTX exports. Members
are selected by name from the central directory, so nothing outside
2021/docs/en/ is ever read into memory.

Measured: 18 markdown files under 2021/docs/en/, of which 10 are categories.
The rest are front matter, and one of them is a trap. A11_2021-Next_Steps.md
matches any A-digit-digit pattern and is not a category, so the pattern is
anchored to A01 through A10 rather than to two digits.

Three of OpenCRE's ten names differ cosmetically from the document's own
title. The document's title wins, per the standing preference for source prose
over a secondary label, and OpenCRE's spelling is recorded as an alternate so
both the title channel and the id channel resolve.

REMEDIATION_HEADINGS in tract/config.py lists "How to Prevent" and "Example
Attack Scenarios" because of this framework. The description here is the
Description section only, so that cut never has to fire on these anchors.

Source: https://github.com/OWASP/Top10
"""
from __future__ import annotations

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
# English 2021 content only. Anchored, because 2021/docs/<lang>/ carries the
# same filenames in a dozen languages and archives/ carries every prior year.
CONTENT_DIR: Final[str] = "/2021/docs/en/"
# A01 through A10 and nothing else. A11_2021-Next_Steps.md is not a category
# and A00_2021* are front matter.
_MEMBER: Final[re.Pattern[str]] = re.compile(r"/A(0[1-9]|10)_2021[-_]")
MAX_MEMBER_BYTES: Final[int] = 200_000

# "# A01:2021 – Broken Access Control    ![icon](assets/...)". The separator is
# an en dash, not a hyphen, and the trailing image is dropped by stopping the
# title at the first exclamation-bracket or brace.
_H1: Final[re.Pattern[str]] = re.compile(
    r"^#\s+(A\d{2}):2021\s*[–—-]\s*(.+?)\s*(?:!\[|\{|$)",
    re.MULTILINE,
)
_DESCRIPTION: Final[re.Pattern[str]] = re.compile(
    r"^##\s+Description\s*$(.*?)(?=^##\s)", re.MULTILINE | re.DOTALL,
)
_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")

# OpenCRE's spelling where it differs from the document's own title. Recorded
# as alternates rather than adopted, so the anchor text stays the source's.
OPENCRE_NAMES: Final[dict[str, str]] = {
    "A01": "Broken Access Controls",
    "A09": "Logging and Monitoring Failures",
    "A10": "Server Side Request Forgery (SSRF)",
}


class OwaspTop102021Parser(BaseParser):
    framework_id: ClassVar[str] = "owasp_top10_2021"
    framework_name: ClassVar[str] = "OWASP Top 10 2021"
    version: ClassVar[str] = "2021"
    source_url: ClassVar[str] = "https://github.com/OWASP/Top10"
    mapping_unit_level: ClassVar[str] = "category"
    expected_count: ClassVar[int] = 10
    fetched_date: ClassVar[str] = "2026-08-15"
    min_prose_fraction: ClassVar[float] = 1.0

    def parse(self) -> list[Control]:
        controls: list[Control] = []
        payload = self.read_source_bytes(ARCHIVE_NAME)
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            for name in sorted(archive.namelist()):
                if CONTENT_DIR not in name or not name.endswith(".md"):
                    continue
                if not _MEMBER.search(name):
                    continue
                text = self._read_member(archive, name)
                heading = _H1.search(text)
                if heading is None:
                    raise ValueError(
                        f"{name}: matched the category pattern and carries no "
                        f"A0N:2021 heading. The filename and the document "
                        f"disagree, so one of them changed."
                    )
                body = _DESCRIPTION.search(text)
                if body is None:
                    raise ValueError(
                        f"{name}: no Description section. Every category "
                        f"carries one."
                    )
                category = heading.group(1)
                alternate = OPENCRE_NAMES.get(category)
                controls.append(Control(
                    control_id=category,
                    title=_WHITESPACE.sub(" ", heading.group(2)).strip(),
                    description=_WHITESPACE.sub(" ", body.group(1)).strip(),
                    metadata=(
                        {"alt_titles": [alternate]} if alternate else None
                    ),
                ))
        if not controls:
            raise ValueError(
                f"{self.framework_id}: no members matched {CONTENT_DIR} with "
                f"an A01 to A10 filename. The archive layout changed."
            )
        logger.info("%s: parsed %d categories", self.framework_id, len(controls))
        return controls

    @staticmethod
    def _read_member(archive: zipfile.ZipFile, name: str) -> str:
        info = archive.getinfo(name)
        if info.file_size > MAX_MEMBER_BYTES:
            raise ValueError(
                f"{name}: declares {info.file_size} bytes, over the "
                f"{MAX_MEMBER_BYTES} byte cap"
            )
        with archive.open(name) as handle:
            raw = handle.read(MAX_MEMBER_BYTES + 1)
        if len(raw) > MAX_MEMBER_BYTES:
            raise ValueError(
                f"{name}: expanded past the {MAX_MEMBER_BYTES} byte cap"
            )
        return raw.decode("utf-8")


def main() -> None:
    OwaspTop102021Parser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_parse_owasp_top10_2021.py -v
mypy parsers/parse_owasp_top10_2021.py --strict
```

Expected: PASS, no mypy errors.

- [ ] **Step 5: Run against the real source and check the join**

```bash
python3 parsers/parse_owasp_top10_2021.py
python3 - <<'PY'
import json, pathlib
d = json.loads(pathlib.Path(
    "data/processed/frameworks/owasp_top10_2021.json").read_text())
links = json.loads(
    pathlib.Path("data/training/hub_links_by_framework.json").read_text()
)["owasp_top10_2021"]
ids = {c["control_id"] for c in d["controls"]}
titles = {c["title"].strip().lower() for c in d["controls"]}
titles |= {a.strip().lower() for c in d["controls"]
           for a in (c.get("metadata") or {}).get("alt_titles", [])}
link_ids = {l["section_id"] for l in links}
link_names = {l["section_name"].strip().lower() for l in links}
print("controls   :", len(d["controls"]))
print("id join    :", len(link_ids & ids), "of", len(link_ids))
print("title join :", len(link_names & titles), "of", len(link_names))
print("names missed:", sorted(link_names - titles))
PY
```

Expected: `controls : 10`, `id join : 10 of 10`, `title join : 10 of 10`, `names missed: []`. **[measured]** A missed name is an `OPENCRE_NAMES` entry that needs adding, and the entry is the link's spelling rather than a rewrite of the document's title.

- [ ] **Step 6: Commit**

```bash
git add parsers/parse_owasp_top10_2021.py \
        tests/test_parse_owasp_top10_2021.py \
        data/processed/frameworks/owasp_top10_2021.json
git commit -m "feat: parse the ten OWASP Top 10 2021 categories from the English source"
```

---

### Task 16: The corpus rebuild, and the three invariants turning green

`tests/test_corpus_invariants.py` carries a class-level `pytest.mark.xfail` marked "11 frameworks await parsers, tracked in Plan 1b". This task removes it. That is the completion signal for the whole plan, and it cannot be faked: the three tests read the processed directory and each of the eleven has to be a real parser's output for all three to pass.

**Files:**
- Modify: `tests/test_corpus_invariants.py`
- Regenerate: `data/processed/frameworks/*.json`, `data/processed/all_controls.json`, `data/processed/licensed/all_controls.json`

**Interfaces:**
- Consumes: Tasks 3 through 15.
- Produces: a corpus where every processed framework has a parser.

- [ ] **Step 1: Confirm every framework file now has a parser**

```bash
python3 - <<'PY'
import pathlib
processed = {p.stem for p in pathlib.Path("data/processed/frameworks").glob("*.json")}
parsers = {p.stem[len("parse_"):] for p in pathlib.Path("parsers").glob("parse_*.py")}
orphans = sorted(processed - parsers)
print("processed:", len(processed), "parsers:", len(parsers))
print("orphans  :", orphans)
PY
```

Expected: `orphans : []`. If anything is listed, the parser for it did not land or its filename does not match its `framework_id`.

- [ ] **Step 2: Rebuild the whole corpus from raw**

Order matters. Every parser writes its own file, then the merge reads all of them.

```bash
for f in parsers/parse_*.py; do
    echo "== $f"
    python3 "$f" || { echo "FAILED: $f"; break; }
done
python3 parsers/merge_all_controls.py
python3 parsers/validate_all.py
```

Expected: every parser writing without raising, the merge writing both the tracked corpus and the gitignored licensed overlay, and `validate_all.py` reporting no errors.

A parser that raises here is the point of the gates. Read the message before changing any number: a count that moved means the source moved, and a prose fraction below the floor means the parse regressed toward titles.

- [ ] **Step 3: Confirm the rebuild is byte-identical on a second run**

```bash
sha256sum data/processed/frameworks/*.json > /tmp/first.txt
for f in parsers/parse_*.py; do python3 "$f" >/dev/null 2>&1; done
sha256sum data/processed/frameworks/*.json > /tmp/second.txt
diff /tmp/first.txt /tmp/second.txt && echo "byte identical"
```

Expected: `byte identical`. A difference means a parser reads the clock or iterates an unordered structure, both of which break the fold-level corpus hash comparison.

- [ ] **Step 4: Run the invariants while they are still marked xfail**

```bash
pytest tests/test_corpus_invariants.py -v
```

Expected: **XPASS** on all three. That is the signal the marker can go. If any is XFAIL, read which one:

- `test_every_framework_file_has_a_parser` failing means Step 1 was skipped.
- `test_no_framework_is_entirely_titles` failing names the framework whose parser emits titles, which should have been impossible past its own prose floor.
- `test_no_framework_carries_a_synthesised_version_string` failing means a parser declares a `version` beginning `opencre-`, which no parser in this plan does.

- [ ] **Step 5: Remove the marker**

```python
# tests/test_corpus_invariants.py — delete these five lines from the class body

    # Plan 1 lands the contract and ISO. The remaining 11 title-only
    # frameworks are Plan 1b, and these tests stay red until then. That is
    # deliberate: a skipped invariant is a forgotten invariant.
    pytestmark = pytest.mark.xfail(
        reason="11 frameworks await parsers, tracked in Plan 1b",
        strict=False,
    )
```

Replace the class docstring, which still describes the state this plan removed:

```python
# tests/test_corpus_invariants.py — replace the module docstring

"""Corpus-level invariants that would have caught the synthesised frameworks.

12 of 31 processed frameworks once had no parser, no generator anywhere in
this repository, and a description that was a byte copy of the title for all
568 of their controls. All 12 now have parsers, so these tests are live rather
than expected failures, and a new synthesised framework fails them the day it
lands.
"""
```

The `import pytest` line stays. `pytest.mark.skipif` above the class still uses it.

- [ ] **Step 6: Run the full suite and typecheck**

```bash
pytest tests/ -q
mypy tract/ parsers/ scripts/phase1a/ scripts/phase1b/ \
     scripts/phase0/runpod_provision.py --strict
```

Expected: all green with no xfail block remaining, and no mypy errors. `tests/test_licensed_text_not_tracked.py` must still pass: ISO's file stays untracked and CSA CCM's is tracked by owner decision.

- [ ] **Step 7: Commit**

```bash
git add tests/test_corpus_invariants.py data/processed/
git commit -m "test: every processed framework now has a parser, invariants live"
```

---

### Task 17: The link counts, after

Task 2 recorded the before. This runs the same script against the rebuilt corpus and records the after, so the two numbers are comparable rather than merely both present.

Spec Part 1.5 retires `PHASE1B_DROPPED_FRAMEWORKS` and `PHASE1B_MIN_SECTION_TEXT_LENGTH` in favour of the resolved anchor, and it lands beside the evaluation contract rather than here. **This task does not retire either gate.** It measures what retiring them would now yield, which is the evidence that decision needs.

**Files:**
- Create: `data/processed/link_resolution_after.json` (generated, tracked)
- Create: `docs/link_recovery_2026-08-16.md`

**Interfaces:**
- Consumes: `scripts/phase1b/report_link_resolution.py` from Task 2, the rebuilt corpus from Task 16.
- Produces: a tracked comparison document with both numbers.

- [ ] **Step 1: Capture the after**

```bash
python3 -m scripts.phase1b.report_link_resolution \
    --output data/processed/link_resolution_after.json
```

- [ ] **Step 2: Diff the two and record the three numbers that matter**

```bash
python3 - <<'PY'
import json, pathlib

before = json.loads(pathlib.Path(
    "data/processed/link_resolution_before.json").read_text())
after = json.loads(pathlib.Path(
    "data/processed/link_resolution_after.json").read_text())

def resolved(report):
    totals = report["totals"]
    return int(totals["resolved_by_title"]) + int(totals["resolved_by_id"])

print("curated links            :", before["totals"]["curated"])
print("kept by gates, before    :", before["totals"]["kept_by_gates"])
print("kept by gates, after     :", after["totals"]["kept_by_gates"])
print("resolved to prose, before:", resolved(before))
print("resolved to prose, after :", resolved(after))
print()
print(f"{'framework':28s} {'before':>7s} {'after':>7s} {'delta':>7s}")
for fw in sorted(after["frameworks"]):
    b = after["frameworks"][fw]
    a_res = int(b["resolved_by_title"]) + int(b["resolved_by_id"])
    prev = before["frameworks"].get(fw, {})
    b_res = int(prev.get("resolved_by_title", 0)) + int(prev.get("resolved_by_id", 0))
    if a_res != b_res:
        print(f"{fw:28s} {b_res:7d} {a_res:7d} {a_res - b_res:+7d}")
PY
```

Three numbers come out of this and all three go in the document:

1. **`kept_by_gates` is unchanged at 4,127** in both runs. **[measured for before]** The gates test `section_name`, a title in the link record, and no parser touches link records. A change here means someone edited the gate constants, which is not this plan's work.
2. **`resolved to prose` rises**, and the per-framework delta lines say by how much and where.
3. **278 links sit outside the gates today**: 4,405 curated minus 4,127 kept. **[measured]** 155 are dropped by the framework list, all of them `nist_800_63` (79) and `owasp_proactive_controls` (76), and 123 by the short-title rule, concentrated in capec 44, dsomm 38 and cwe 17.

- [ ] **Step 3: Measure what retiring the gates would now recover**

```bash
python3 - <<'PY'
import json, pathlib

after = json.loads(pathlib.Path(
    "data/processed/link_resolution_after.json").read_text())

recoverable = 0
rows = []
for fw, row in sorted(after["frameworks"].items()):
    resolved = int(row["resolved_by_title"]) + int(row["resolved_by_id"])
    beyond = resolved - int(row["kept_by_gates"])
    if beyond > 0:
        rows.append((fw, int(row["kept_by_gates"]), resolved, beyond))
        recoverable += beyond

print(f"{'framework':28s} {'kept':>6s} {'resolved':>9s} {'recoverable':>12s}")
for fw, kept, resolved, beyond in rows:
    print(f"{fw:28s} {kept:6d} {resolved:9d} {beyond:12d}")
print()
print("training links today                :", after["totals"]["kept_by_gates"])
print("recoverable by moving the gate      :", recoverable)
print("training links after gate retirement:",
      int(after["totals"]["kept_by_gates"]) + recoverable)
PY
```

The last line is the number spec Part 1.5 needs. The upper bound is 4,405, because a link cannot be recovered twice, and any figure above that means the report is double counting.

- [ ] **Step 4: Write the comparison document**

Create `docs/link_recovery_2026-08-16.md` with, at minimum: the two totals blocks from Step 2 verbatim, the per-framework delta table, the recoverable table from Step 3, and one paragraph naming which of the 278 currently-dropped links now resolve and which still do not. Label every number `[measured]` and name the script and the two JSON artifacts that produced it.

Do not restate the projected total anywhere else in the repository. One number, one place, one artifact behind it.

- [ ] **Step 5: Commit**

```bash
git add data/processed/link_resolution_after.json \
        docs/link_recovery_2026-08-16.md
git commit -m "docs: record link resolution before and after the corpus rebuild"
```

---

### Task 18: The per-framework join-rate gate

A parser that emits 194 clean controls none of which join is a failure, and until now nothing detected it. Every gate in `BaseParser.run()` measures the parser against its own declarations, and none of them measures it against the links that needed it.

This task turns the Task 2 report into a test. Each of the eleven declares the join rate it achieved, measured in its own task, and the test fails when the achieved rate falls below it.

**Files:**
- Create: `tests/test_link_join_rates.py`

**Interfaces:**
- Consumes: `resolve_links` from Task 2, the rebuilt corpus from Task 16.
- Produces: nothing consumed later.

- [ ] **Step 1: Write the test**

```python
# tests/test_link_join_rates.py — create

"""A parser whose output nothing joins is a failure.

Every gate in BaseParser.run() measures a parser against its own declarations.
None measures it against the links that needed it, so a parser can emit 194
clean controls that resolve nothing and pass every check it has.

Each floor below is the rate measured when that parser landed, minus nothing.
A floor is not a target: it is the number that was true, so a drop is a
regression and a rise is a change worth reading. Move one only with the
measurement that justifies it in the commit message.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.phase1b.report_link_resolution import (
    CURATED_PATH,
    FrameworkResolution,
    resolve_links,
)
from tract.config import PROCESSED_DIR, PROCESSED_LICENSED_DIR
from tract.text_selection import ProseIndex

# framework_id -> (minimum resolved links, minimum resolved fraction).
# Both are recorded because a count alone hides a framework whose link total
# changed, and a fraction alone hides one that lost half its links.
JOIN_FLOORS: dict[str, tuple[int, float]] = {
    "dsomm": (214, 1.00),
    "wstg": (114, 0.96),
    "nist_800_63": (78, 0.98),
    "owasp_proactive_controls": (76, 1.00),
    "enisa": (55, 0.80),
    "nist_ssdf": (44, 0.95),
    "etsi": (36, 1.00),
    "samm": (30, 1.00),
    "csa_ccm": (29, 1.00),
    "biml": (20, 0.95),
    "owasp_top10_2021": (17, 1.00),
}


def _corpus() -> Path:
    overlay = PROCESSED_LICENSED_DIR / "all_controls.json"
    return overlay if overlay.exists() else PROCESSED_DIR / "all_controls.json"


@pytest.fixture(scope="module")
def report() -> dict[str, FrameworkResolution]:
    links = [
        json.loads(line)
        for line in CURATED_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return resolve_links(links, ProseIndex.load(_corpus()))


@pytest.mark.skipif(
    not (PROCESSED_DIR / "all_controls.json").exists(),
    reason="no merged corpus in this checkout",
)
@pytest.mark.parametrize("framework_id", sorted(JOIN_FLOORS))
def test_each_new_framework_meets_its_measured_join_rate(
    framework_id: str, report: dict[str, FrameworkResolution],
) -> None:
    row = report.get(framework_id)
    assert row is not None, (
        f"{framework_id} has no curated links at all, which means the "
        f"framework_id in the link file and in the parser disagree"
    )

    resolved = row["resolved_by_title"] + row["resolved_by_id"]
    minimum_count, minimum_fraction = JOIN_FLOORS[framework_id]
    fraction = resolved / row["curated"] if row["curated"] else 0.0

    assert resolved >= minimum_count, (
        f"{framework_id}: {resolved} of {row['curated']} links resolve, "
        f"below the measured floor of {minimum_count}. Unresolved section "
        f"ids: {row['unresolved_section_ids'][:10]}"
    )
    assert fraction >= minimum_fraction, (
        f"{framework_id}: {fraction:.3f} of links resolve, below the measured "
        f"floor of {minimum_fraction:.3f}"
    )


@pytest.mark.skipif(
    not (PROCESSED_DIR / "all_controls.json").exists(),
    reason="no merged corpus in this checkout",
)
def test_no_framework_resolves_nothing(
    report: dict[str, FrameworkResolution],
) -> None:
    """The invariant that needs no per-framework number to be useful."""
    dead = sorted(
        framework_id for framework_id, row in report.items()
        if row["curated"] and not (row["resolved_by_title"] + row["resolved_by_id"])
    )
    assert not dead, (
        f"{dead} carry curated links and resolve none of them. Either the "
        f"parser's framework_name disagrees with the link's standard_name, or "
        f"its control_id shape disagrees with the link's section_id."
    )
```

- [ ] **Step 2: Run it and reconcile the floors with reality**

```bash
pytest tests/test_link_join_rates.py -v
```

The floors above are derived from the per-framework measurements in Tasks 3 through 15 and are not themselves measured end to end. Reconcile rather than assume:

```bash
python3 - <<'PY'
import json, pathlib
after = json.loads(pathlib.Path(
    "data/processed/link_resolution_after.json").read_text())
for fw in ("dsomm", "wstg", "nist_800_63", "owasp_proactive_controls", "enisa",
           "nist_ssdf", "etsi", "samm", "csa_ccm", "biml", "owasp_top10_2021"):
    row = after["frameworks"][fw]
    resolved = row["resolved_by_title"] + row["resolved_by_id"]
    fraction = resolved / row["curated"] if row["curated"] else 0.0
    print(f'    "{fw}": ({resolved}, {fraction:.2f}),   # of {row["curated"]}')
PY
```

Paste the printed block into `JOIN_FLOORS`, rounding each fraction **down** to two places. A floor set above the measured value fails on the next run for no reason, and one set well below stops being a gate. If any framework prints a fraction under 0.80, do not lower the floor to accommodate it: read `unresolved_section_ids` for that framework and fix the parser or record in its docstring why those specific anchors cannot resolve.

- [ ] **Step 3: Re-run and typecheck**

```bash
pytest tests/test_link_join_rates.py -v
mypy scripts/phase1b/report_link_resolution.py --strict
```

Expected: PASS on all twelve tests.

- [ ] **Step 4: Run the full suite one last time**

```bash
pytest tests/ -q
mypy tract/ parsers/ scripts/phase1a/ scripts/phase1b/ \
     scripts/phase0/runpod_provision.py --strict
```

Expected: all green, no xfail, no mypy errors.

- [ ] **Step 5: Commit**

```bash
git add tests/test_link_join_rates.py
git commit -m "test: gate each new framework on the join rate it actually achieved"
```

---

## Self-review notes

**Spec coverage.** Part 1.4 items 2 through 4 are Tasks 3 through 15, the eleven parsers. Part 1.9's first three acceptance tests are Task 16, which removes the xfail marker that made them decorative. Part 1.9's "parse twice, bytes identical" is Task 16 Step 3, scoped to re-parsing rather than re-fetching because `data/raw/` is gitignored. Part 1.5 is Tasks 2, 17 and 18: this plan measures what retiring both gates would recover and does not retire them, because retirement re-bases every metric and belongs beside the evaluation contract.

**Not in scope, with the reason.** Part 1.6, the OWASP LLM Top 10 2026 contamination control, is blocked on the three-way licence conflict the spec names at 1.4 and is not one of the eleven. Part 1.8, review status surviving a corpus rebuild, touches the crosswalk schema. Both stay where the predecessor plan left them.

**Placeholder scan.** Two parsers ship `expected_count = 0`, ENISA in Task 8 and BIML in Task 14. That is deliberate and it is not a placeholder: `BaseParser._check_expected_count` raises on a zero, so neither parser can write until its task's measurement step replaces the literal. Every other number in every parser is measured and stated with the snippet that produced it. `JOIN_FLOORS` in Task 18 is derived from the per-parser measurements and Step 2 of that task reconciles it against the end-to-end report rather than trusting the derivation. No task step says "similar to Task N", no function raises `NotImplementedError`, and no comment says "add appropriate error handling".

**Type consistency.** Every parser declares its class attributes as `ClassVar` with explicit types, matching `parse_iso_27001.py` rather than the older bare-assignment parsers. Three helpers return typed tuples that their callers destructure: `merge_spanned_rows -> tuple[list[tuple[str, str]], int]`, `EnisaParser.records_from_table -> list[tuple[str, str]]`, and `EtsiParser.extract_sections -> list[tuple[str, str, str]]`. `FrameworkResolution` is a `TypedDict` so the report's keys are checked at the call sites in Tasks 17 and 18. `Control.metadata` is `dict[str, str | list[str]] | None`, which is why every `alt_ids` and `alt_titles` value is a list and every `record_type` is a string. `DsommParser._load_model` is the only `cast` in the plan, and it is there because `yaml.safe_load_all` returns `Any` and the isinstance check above it is what makes the cast honest.

**Where the source structures document was insufficient.** Named at the point of use in each task and collected here: DSOMM's prose fields, WSTG's file counts and tombstones, NIST 800-63's revision, ENISA's three blocks and moving column index, NIST SSDF's retirement stubs, ETSI's container sections, CSA CCM's row count and version drift, and OWASP Top 10's A11 trap.

**Ordering.** Task 12 lands `alt_ids` after Task 4 writes it and before Tasks 13 and 14 depend on it, so Task 4's own verification step is the one place in the plan that forward-references a later task. That is called out in Task 4 Step 5 rather than left for the implementer to discover. Task 2 lands the measurement instrument before the first parser so its baseline is captured under identical conditions, which is the Plan 1 lesson that a baseline measured a different way masked nine regressions. Task 7 lands the shared rowspan merge before its two consumers. Task 16 rewrites every processed artifact, so it precedes Tasks 17 and 18, both of which read the rebuilt corpus.
