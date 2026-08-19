### Task 1: The corpus quality instrument, and the BEFORE state

This is the only instrument in this plan. Everything downstream is measured with it, so it is built
and run before any parser exists. Version 2 of this task shipped an instrument with four defects,
each of which the premortem reproduced against source. All four are fixed here.

**Defect 1: the load-bearing column was the wrong column.** Version 2 reported the eleven pending
frameworks at `distinct_anchors == 0` and read the gain as `+452`. The trainer does not get zero
anchors for those links. `select_control_text` falls back to `(section_name or section_id)` when
`ProseIndex.lookup` returns nothing, so the eleven's 734 curated links already land on **299 distinct
fallback anchors** today: dsomm 18, wstg 59, nist_ssdf 44, enisa 33, samm 30, csa_ccm 29,
nist_800_63 25, biml 17, etsi 24, owasp_proactive_controls 10, owasp_top10_2021 10. **[measured,
orchestrator during adjudication, reproduced digit for digit by this task against
`data/processed/licensed/all_controls.json` sha256 in the artifact]** So `FrameworkJoin` gains a
`fallback_anchors` column and the report shows both sides.

**Defect 2: `nested_anchors` counted strict prefixes only.** An ETSI clause 5.2 rolled up over 5.2.2
nests undetected whenever the child is not sorted first. The column changes to containment, and the
old strict-prefix count survives as `contained_anchors` for continuity. Separately, two anchors that
share a 2,150-character prefix truncate to the identical string and merge into one set member, which
suppresses `nested_anchors` and quietly lowers `distinct_anchors` with no diagnostic. That is why
`distinct_anchors_pre_truncation` exists.

**Defect 3: `wrong_anchor_risk` could only fire in the title branch.** Nine of the eleven are
engineered to resolve through the id channel, so `== 0` was unfailable for them, while the one
framework where it does fire (csa_ccm, whose `IPY` link reaches control IPY-01's title rather than
the IPY domain) would halt a healthy run. The counter is redesigned into three detectors, two of
which reach the id branch, and this task states the attainable range per framework so nobody asserts
zero where zero is guaranteed.

**Defect 4: the channel-parity test was vacuous.** It built `ProseIndex(data if isinstance(data,
list) else [])`. Both `data/processed/all_controls.json` and `data/processed/licensed/all_controls.json`
are **dicts** with keys `[framework_count, frameworks, generated_date, total_controls]` **[measured,
this task]**, so the index was built from `[]` and all 4,405 assertions reduced to `True == True`.
The named guard for "the exact defect that got the previous plan rejected" asserted nothing. It now
uses the dict-aware loader.

#### What the honest gain is

Counting links resolved is not enough, and that is why the previous plan was rejected. 615 links
unstacked onto 615 distinct anchors and 615 links collapsed onto 40 coarse anchors produce the same
rising line. The report's load-bearing pair is `distinct_anchors` **beside** `fallback_anchors`.

| | the eleven's distinct anchors | source |
|---|---|---|
| BEFORE, as version 2 reported it | 0 | wrong, `distinct_anchors` alone |
| BEFORE, what the trainer gets | **299** | **[measured]** `fallback_anchors` |
| AFTER, summed from the eleven parser tasks' own predictions | **451 or more** | **[derived]** dsomm ≥182, wstg ≥55, nist_ssdf ≥44, enisa 33, samm 30, csa_ccm 29, nist_800_63 25, biml 19, etsi 14, proactive 10, top10 10 |
| honest delta | **+152 or more** | **[derived]** |

`+452` is the AFTER figure compared against a BEFORE of zero. It must never enter the run ledger.
The figure the ledger records is `299 -> 451+`, delta `+152 or more`, corpus-wide `1,749 -> 1,901+`.

Seven of the eleven parsers move the anchor count by **exactly zero**: csa_ccm 29 to 29, enisa 33 to
33, nist_800_63 25 to 25, nist_ssdf 44 to 44, samm 30 to 30, owasp_proactive_controls 10 to 10,
owasp_top10_2021 10 to 10. **[derived from the measured 299 against each parser task's own stated
prediction]** ETSI goes 24 to 14, a **loss** of ten anchors, because 36 links collapse onto 14 clause
anchors instead of 24 section names. The entire count gain comes from three frameworks: dsomm 18 to
182 or more, wstg 59 to 55 or more, biml 17 to 19.

If the count is flat or negative for eight of eleven, the parsers have to be worth something else,
and they are: **text quality**. A fallback anchor is a phrase. A parsed anchor is a paragraph. That
is a real change to what the encoder reads and version 2 had no column for it. This task adds the
four `anchor_source_*` columns and records `anchor_chars` per link, and the BEFORE median fallback
anchor length is the baseline the AFTER is read against:

| framework | median fallback anchor chars, BEFORE | mean |
|---|---|---|
| owasp_proactive_controls | 2.0 | 2.1 |
| nist_800_63 | 7.0 | 6.1 |
| wstg | 12.0 | 12.0 |
| biml | 14.0 | 14.5 |
| samm | 19.5 | 19.9 |
| etsi | 20.0 | 19.7 |
| dsomm | 21.5 | 18.8 |
| owasp_top10_2021 | 31.0 | 28.8 |
| csa_ccm | 37.0 | 37.1 |
| enisa | 44.0 | 45.1 |
| nist_ssdf | 156.5 | 155.9 |

**[measured, all, this task, over all 734 links including resolved-to-nothing rows]** Two of those
say the quiet part out loud. `owasp_proactive_controls` trains on a two-character anchor. `wstg`
trains on twelve characters of identifier. `nist_ssdf` at 156 characters is the one framework whose
fallback is already a sentence, which is exactly why its parser must not also use `section_name` as
the title.

#### The BEFORE state, measured

| framework | links | resolved | distinct anchors | fallback anchors | links/anchor | truncated |
|---|---|---|---|---|---|---|
| owasp_cheat_sheets | 391 | 391 | 49 | 0 | 7.98 | 384 |
| capec | 1799 | 1799 | 349 | 0 | 5.15 | 24 |
| cwe | 613 | 612 | 245 | 1 | 2.50 | 13 |
| nist_ai_100_2 | 45 | 45 | 22 | 0 | 2.05 | 21 |
| iso_27001 | 94 | 92 | 91 | 2 | 1.01 | 0 |
| the eleven pending | 734 | 0 | 0 | **299** | — | 0 |

**[measured, all]** The report reads `hub_links_curated.jsonl` (4,405 rows) rather than
`hub_links_by_framework.json` (4,406), because the curated file is what the trainer reads and the
extra row is an `owasp_ai_exchange` duplicate. **[measured]**

#### Two more numbers version 2 got wrong

`dropped_by_prose_rule` summed only frameworks carrying curated links, so it read 522 where the
corpus holds **558** un-indexed controls. The 36 invisible ones are NIST AI Risk Management
Framework 25, AIUC-1 Standard 10, CoSAI Landscape of AI Security Risk Map 1. **[measured, Data
Scientist, reproduced by this task]** The totals row now sums the whole corpus census, not the
link-bearing subset.

`wrong_anchor_risk` read 9 corpus-wide under the title-only detector. Under the three detectors it
reads **40**: asvs 1, cwe 2, iso_27001 5, nist_ai_100_2 20, owasp_ai_exchange 11, owasp_ml_top10 1.
**[measured, this task]** The 31 new flags are not noise. Four of them are ISO control titles the
parser wrote without spaces (`'Addressinginformationsecurity within supplier agreements'` for A.5.20,
`'Redundancyofinformation processing facilities'` for A.8.14), two are CWE-598 renamed upstream from
`'Use of GET Request Method With Sensitive Query Strings'` to `'Use of HTTP Request With Sensitive
Query String'`, and twelve are NIST AI 100-2 links where one coarse section id such as `Sec. 3.3.2`
answers for two differently named items. **[measured, this task, all five quoted strings read out of
the corpus]** Nothing in this repository reported any of them before now.

**Files:**
- Create: `tract/corpus_report.py`
- Create: `scripts/corpus_report.py`
- Create: `tests/test_corpus_report.py`
- Create: `results/corpus/before.json`
- Create: `results/corpus/link_resolution_before.jsonl`
- Modify: `.gitignore`
- Modify: `tract/text_selection.py`

**Interfaces:**
- Consumes: `tract.text_selection.ProseIndex`, `TextSelection`, `canonical_framework`,
  `normalize_section_id`, `prepare_anchor`, `strip_markup`, `merged_corpus_path`, `_is_prose`;
  `tract.config.MAX_ANCHOR_CHARS`, `PROJECT_ROOT`, `TRAINING_DIR`, `RESTRICTED_FRAMEWORK_IDS`;
  `tract.io.atomic_write_json`, `atomic_write_text`; `data/training/hub_links_curated.jsonl`.
- Produces:
  - `FrameworkJoin` with the exact field set in the v3 contract Rule 1, all twenty-two fields.
  - `LinkResolution`, one per curated link, digest and length only, no anchor text.
  - `CorpusReport` with `per_framework`, `totals`, `corpus_path`, `corpus_sha256`,
    `corpus_framework_count`, `links_path`, `links_sha256`, `max_anchor_chars`, `resolution_rows`,
    `to_json()` and `by_id()`.
  - `build_corpus_report(links_path: Path | None = None, corpus_path: Path | None = None) -> CorpusReport`
  - `check_join_floors(report: CorpusReport, floors: Mapping[str, float]) -> list[str]`
  - `floors_for_report(report, floors, restricted=RESTRICTED_FRAMEWORK_IDS) -> tuple[dict[str, float], list[str]]`
  - `write_link_resolution(report: CorpusReport, path: Path) -> None`
  - `wrong_anchor_applicable(report: CorpusReport) -> dict[str, int]`
  - `format_table(report: CorpusReport) -> str`
  - `JOIN_CEILINGS`, `JOIN_FLOORS`, `CORPUS_EVIDENCE_DIR`, `FULL_CORPUS_FRAMEWORK_COUNT`,
    `TEXT_ORIGIN_METADATA_KEY`, `SYNTHETIC_TEXT_ORIGIN`.
  - `ProseIndex.by_title`, `ProseIndex.by_id`.

**Invalidates:**
- The plan's own headline `+452 distinct anchors`, in the Architecture paragraph at line 7 and
  wherever Task 16 copies it into `.superpowers/autonomous-run/RUN-LEDGER.md`. The ledger figure is
  `299 -> 451+`.
- Task 16 Step 1. `JOIN_FLOORS` is committed here, before any parser exists, per contract Rule 6.
  Task 16 imports it and must not redefine it.
- Task 16's `BEFORE = Path("results/corpus/before_8cf44b3.json")`. The artifact is
  `tract.corpus_report.CORPUS_EVIDENCE_DIR / "before.json"`, anchored to `PROJECT_ROOT`, and the
  `pytest.skip("no BEFORE artifact in this checkout")` beneath it is deleted per Ruling R3.
- Task 16's `test_no_new_framework_nests_an_anchor_inside_another`. Under containment, csa_ccm's
  domain aggregates are expected to nest (the plan's own note says 17 of 17), so `nested_anchors == 0`
  for every pending framework is no longer the right assertion.
- Task 16's `test_no_new_framework_carries_wrong_anchor_risk`. See Step 11 for the attainable range
  per framework and what to assert instead.
- Nothing at runtime. No training, evaluation or publication artifact reads this module yet.

- [ ] **Step 1: Confirm the interpreter and the shape of the corpus**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"$PY" -c "import pdfplumber, yaml, bs4, pydantic, defusedxml, openpyxl, sys; print(sys.version)"
"$PY" -c "
import json
for p in ['data/processed/all_controls.json', 'data/processed/licensed/all_controls.json']:
    d = json.load(open(p, encoding='utf-8'))
    print(p, type(d).__name__, sorted(d) if isinstance(d, dict) else len(d))
"
```

Expected:

```
3.12.2 ...
data/processed/all_controls.json dict ['framework_count', 'frameworks', 'generated_date', 'total_controls']
data/processed/licensed/all_controls.json dict ['framework_count', 'frameworks', 'generated_date', 'total_controls']
```

**[measured]** If the first command fails, stop, every later step depends on it. If either file
prints `list`, stop as well: the loader below prefers the `frameworks` key and a shape change means
the corpus writer changed under this plan.

- [ ] **Step 2: Make the evidence directory tracked by design**

`.gitignore:3` is `results/`, so `git add results/corpus/before.json` exits 1 and stages nothing for
that path. Global Constraints forbid `git add -f`, and forcing ignored paths into git is how licensed
text escaped four times. The fix is a negation, and the form matters: `results/` excludes the
directory itself, git never descends into it, and a `!results/corpus/**` negation underneath is
inert. `results/**` excludes the contents and leaves the negation reachable. Both forms were run
against this repository with `git check-ignore -v`. **[measured]**

```bash
# .gitignore — replace line 3, which currently reads "results/"
```

```
# Evidence artifacts are the record a later reader audits, so results/corpus/
# is tracked by design. "results/" alone excludes the directory, git never
# descends into it, and any negation underneath is inert; "results/**"
# excludes the contents and leaves the negations reachable. Verified with
# git check-ignore -v.
results/**
!results/corpus/
!results/corpus/**
```

Contract Rule 2 also lists `!results/ceiling_study/` and `!results/ceiling_study/**`. That negation
is **not** added here. 73 files under `results/` are already tracked, 69 of them in
`results/ceiling_study/`, and un-ignoring the tree newly exposes exactly four untracked files:
`answers_llm_proxy.json`, `ceiling_answers_LLM_PROXY.json`, `LLM_PROXY_report.md`,
`LLM_PROXY_score_report.txt`. **[measured, this task]** No task in this plan reviews those four, and
this plan's whole-tree licence gate is red on purpose. Whichever task first commits a ceiling-study
artifact owns that negation and owns reviewing those four files first.

Verify:

```bash
git check-ignore -v results/corpus/before.json || echo "not ignored, which is the goal"
git check-ignore -v results/phase1b/anything.json
git status --porcelain results/ | head
```

Expected: the first prints `not ignored, which is the goal`. The second still reports
`.gitignore:N:results/**`. The third lists nothing new beyond the files already modified in this
checkout. A tracked file is never affected by `.gitignore`, so the 73 already-tracked paths do not
move.

- [ ] **Step 3: Write the failing tests**

```python
# tests/test_corpus_report.py — create

"""The corpus report is the only instrument in the parser plan.

Counting links resolved cannot tell 615 links unstacked onto 615 anchors from
615 links collapsed onto 40. Both make the same number rise. The tests below
pin the columns that can tell them apart: distinct anchors against the
fallback anchors the trainer already gets, links per anchor, truncation,
nesting by containment, controls the prose rule excludes from the index, and
the three wrong-anchor detectors.

Every path here is anchored to PROJECT_ROOT. A test that resolves a relative
path passes or fails on the directory pytest started in.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tract.config import PROJECT_ROOT
from tract.corpus_report import (
    JOIN_CEILINGS,
    JOIN_FLOORS,
    build_corpus_report,
    check_join_floors,
)

TRACKED_CORPUS = PROJECT_ROOT / "data" / "processed" / "all_controls.json"

LONG = "A control statement long enough to clear every prose bar. " * 4


def _corpus(
    directory: Path,
    controls: list[dict[str, object]],
    name: str = "corpus",
) -> Path:
    """A corpus in the shape the real files use: a dict, not a list.

    Version 2's parity test assumed a list and silently indexed nothing. The
    fixtures use the real shape so the loader is exercised the way production
    exercises it.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    payload = {
        "framework_count": 1,
        "frameworks": [
            {
                "framework_id": "demo",
                "framework_name": "Demo",
                "controls": controls,
            }
        ],
        "generated_date": "2026-01-01",
        "total_controls": len(controls),
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _corpus_as_list(
    directory: Path, controls: list[dict[str, object]], name: str = "legacy",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(
        json.dumps(
            [{
                "framework_id": "demo",
                "framework_name": "Demo",
                "controls": controls,
            }],
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _links(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    path = tmp_path / "links.jsonl"
    path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
        encoding="utf-8",
    )
    return path


def _row(section_id: str, section_name: str, cre_id: str = "1-1") -> dict[str, str]:
    return {
        "framework_id": "demo",
        "standard_name": "Demo",
        "section_id": section_id,
        "section_name": section_name,
        "cre_id": cre_id,
        "link_type": "LinkedTo",
    }


class TestCorpusShape:
    """The fact that made version 2's parity test assert nothing."""

    def test_the_tracked_corpus_is_a_dict(self) -> None:
        data = json.loads(TRACKED_CORPUS.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert sorted(data) == [
            "framework_count", "frameworks", "generated_date", "total_controls",
        ]

    def test_the_loader_reads_both_shapes(self, tmp_path: Path) -> None:
        controls = [
            {"control_id": "C-1", "title": "One", "description": LONG},
        ]
        links = _links(tmp_path, [_row("C-1", "One")])
        as_dict = build_corpus_report(links, _corpus(tmp_path / "d", controls))
        as_list = build_corpus_report(
            links, _corpus_as_list(tmp_path / "l", controls),
        )
        assert as_dict.per_framework[0].by_title == 1
        assert as_list.per_framework[0].by_title == 1
        assert as_dict.corpus_framework_count == 1

    def test_a_corpus_with_no_records_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"total_controls": 0}), encoding="utf-8")
        with pytest.raises(ValueError, match="no list of framework records"):
            build_corpus_report(_links(tmp_path, [_row("C-1", "One")]), path)


class TestAnchorCollapse:
    def test_distinct_anchors_separates_collapse_from_coverage(
        self, tmp_path: Path,
    ) -> None:
        """Two corpora resolve every link. Only one of them is good."""
        rows = [_row(f"C-{n}", f"Control {n}") for n in range(1, 5)]
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

    def test_nesting_is_containment_not_a_prefix(self, tmp_path: Path) -> None:
        """ETSI clause 5.2 rolled up over 5.2.2 does not start the parent's text.

        Version 2 counted strict prefixes, so this case read 0. The child here
        sits in the middle of the parent, which is what a clause rollup looks
        like when the parent opens with its own lead paragraph.
        """
        child = LONG + " The child clause statement."
        parent = "Lead-in paragraph for the parent clause. " + child + " Tail."
        corpus = _corpus(tmp_path, [
            {"control_id": "5.2.2", "title": "Child", "description": child},
            {"control_id": "5.2", "title": "Parent", "description": parent},
        ])
        links = _links(tmp_path, [_row("5.2.2", "Child"), _row("5.2", "Parent")])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.distinct_anchors == 2
        assert row.nested_anchors == 1
        assert row.contained_anchors == 0

    def test_contained_keeps_the_strict_prefix_count(self, tmp_path: Path) -> None:
        """A domain aggregate that opens with its own first member."""
        member = LONG + " Member statement."
        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "Member", "description": member},
            {"control_id": "D-1", "title": "Domain",
             "description": member + " And the rest of the domain."},
        ])
        links = _links(tmp_path, [_row("C-1", "Member"), _row("D-1", "Domain")])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.distinct_anchors == 2
        assert row.contained_anchors == 1
        assert row.nested_anchors == 1

    def test_truncation_can_merge_two_anchors_into_one(
        self, tmp_path: Path,
    ) -> None:
        """Two anchors sharing a long prefix collapse after MAX_ANCHOR_CHARS.

        distinct_anchors falls with no other column moving, which is how the
        collapse hid. distinct_anchors_pre_truncation is the witness.
        """
        from tract.config import MAX_ANCHOR_CHARS

        shared = "Shared opening text. " * (MAX_ANCHOR_CHARS // 21 + 2)
        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "One", "description": shared + " Tail A."},
            {"control_id": "C-2", "title": "Two", "description": shared + " Tail B."},
        ])
        links = _links(tmp_path, [_row("C-1", "One"), _row("C-2", "Two")])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.by_title == 2
        assert row.truncated == 2
        assert row.distinct_anchors == 1
        assert row.distinct_anchors_pre_truncation == 2


class TestFallbackAnchors:
    def test_unresolved_links_still_give_the_trainer_an_anchor(
        self, tmp_path: Path,
    ) -> None:
        """The column that turns +452 into +152.

        select_control_text falls back to section_name, so a framework with
        zero resolved links does not train on zero anchors.
        """
        corpus = _corpus(tmp_path, [])
        links = _links(tmp_path, [
            _row("C-1", "Access control policy"),
            _row("C-2", "Access control policy"),
            _row("C-3", "Cryptographic key management"),
        ])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.unresolved == 3
        assert row.distinct_anchors == 0
        assert row.fallback_anchors == 2

    def test_a_link_with_no_name_falls_back_to_its_id(self, tmp_path: Path) -> None:
        corpus = _corpus(tmp_path, [])
        links = _links(tmp_path, [_row("C-1", "")])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.fallback_anchors == 1


class TestProseRuleExclusion:
    def test_control_whose_description_restates_its_title_is_counted(
        self, tmp_path: Path,
    ) -> None:
        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "Access control",
             "description": "Access control."},
        ])
        links = _links(tmp_path, [_row("C-1", "Access control")])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.dropped_by_prose_rule == 1
        assert row.unresolved == 1
        assert row.distinct_anchors == 0
        assert row.fallback_anchors == 1

    def test_the_total_counts_frameworks_with_no_curated_links(
        self, tmp_path: Path,
    ) -> None:
        """522 was the link-bearing subset. The corpus holds 558."""
        path = tmp_path / "corpus.json"
        path.write_text(json.dumps({
            "framework_count": 2,
            "frameworks": [
                {"framework_id": "demo", "framework_name": "Demo",
                 "controls": [{"control_id": "C-1", "title": "A",
                               "description": "A."}]},
                {"framework_id": "silent", "framework_name": "Silent",
                 "controls": [{"control_id": "S-1", "title": "B",
                               "description": "B."}]},
            ],
            "generated_date": "2026-01-01",
            "total_controls": 2,
        }, sort_keys=True), encoding="utf-8")
        report = build_corpus_report(_links(tmp_path, [_row("C-1", "A")]), path)
        assert report.by_id("demo").dropped_by_prose_rule == 1
        assert report.totals.dropped_by_prose_rule == 2


class TestAnchorSource:
    def test_the_four_sources_partition_the_resolved_links(
        self, tmp_path: Path,
    ) -> None:
        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "Full", "description": "short",
             "full_text": LONG + " From full text."},
            {"control_id": "C-2", "title": "Described", "description": LONG},
            {"control_id": "C-3", "title": "Title restated as full text",
             "description": "short",
             "full_text": "Title restated as full text"},
            {"control_id": "C-4", "title": "Built", "description": LONG + " Built.",
             "metadata": {"text_origin": "synthetic"}},
        ])
        links = _links(tmp_path, [
            _row("C-1", "Full"), _row("C-2", "Described"),
            _row("C-3", "Title restated as full text"), _row("C-4", "Built"),
        ])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.anchor_source_full_text == 1
        assert row.anchor_source_description == 1
        assert row.anchor_source_title == 1
        assert row.anchor_source_synthetic == 1
        assert (
            row.anchor_source_full_text
            + row.anchor_source_description
            + row.anchor_source_title
            + row.anchor_source_synthetic
        ) == row.by_title + row.by_id


class TestHubSide:
    def test_hub_concentration_is_reported(self, tmp_path: Path) -> None:
        corpus = _corpus(tmp_path, [
            {"control_id": f"C-{n}", "title": f"T{n}",
             "description": f"{LONG} {n}."} for n in range(1, 5)
        ])
        links = _links(tmp_path, [
            _row("C-1", "T1", "hub-a"), _row("C-2", "T2", "hub-a"),
            _row("C-3", "T3", "hub-a"), _row("C-4", "T4", "hub-b"),
        ])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.distinct_hubs == 2
        assert row.links_per_hub == pytest.approx(2.0)


class TestWrongAnchorRisk:
    """Three detectors. Two of them reach the id branch, which is the fix."""

    def test_detector_a_title_hit_that_disagrees_with_the_id(
        self, tmp_path: Path,
    ) -> None:
        corpus = _corpus(tmp_path, [
            {"control_id": "2.3", "title": "Poisoning attacks",
             "description": LONG + " Predictive."},
            {"control_id": "3.2.2", "title": "Generative poisoning",
             "description": LONG + " Generative.",
             "metadata": {"alt_titles": ["Poisoning attacks"]}},
        ])
        links = _links(tmp_path, [_row("3.2.2", "Poisoning attacks")])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.by_title == 1
        assert row.wrong_anchor_risk == 1

    def test_detector_a_does_not_fire_when_the_channels_agree(
        self, tmp_path: Path,
    ) -> None:
        corpus = _corpus(tmp_path, [
            {"control_id": "2.3", "title": "Poisoning attacks",
             "description": LONG + " Predictive."},
        ])
        links = _links(tmp_path, [_row("2.3", "Poisoning attacks")])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.by_title == 1
        assert row.wrong_anchor_risk == 0

    def test_detector_b_id_hit_whose_control_does_not_carry_the_name(
        self, tmp_path: Path,
    ) -> None:
        """The id branch, where version 2 was blind."""
        corpus = _corpus(tmp_path, [
            {"control_id": "IPY", "title": "Interoperability and portability",
             "description": LONG + " Domain."},
        ])
        links = _links(tmp_path, [_row("IPY", "Data centre power redundancy")])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.by_id == 1
        assert row.wrong_anchor_risk == 1

    def test_detector_b_does_not_fire_on_an_identifier_shaped_name(
        self, tmp_path: Path,
    ) -> None:
        """wstg and owasp_proactive_controls have section_name == section_id.

        Those links make no independent claim, so nothing is checked and
        nothing is flagged. This is why their attainable range is zero, and it
        is asserted rather than assumed.
        """
        corpus = _corpus(tmp_path, [
            {"control_id": "WSTG-INFO-01", "title": "Information gathering",
             "description": LONG + " Gathering."},
        ])
        links = _links(tmp_path, [_row("WSTG-INFO-01", "WSTG-INFO-01")])
        report = build_corpus_report(links, corpus)
        assert report.per_framework[0].by_id == 1
        assert report.per_framework[0].wrong_anchor_risk == 0
        assert all(
            not entry.wrong_anchor_checked for entry in report.resolution_rows
        )

    def test_detector_c_a_parent_id_and_a_child_id_reach_one_paragraph(
        self, tmp_path: Path,
    ) -> None:
        """The NIST AI 100-2 failure that put title first in the lookup order.

        Both controls exist and both carry the same paragraph, which is what a
        parser produces when a subsection's text is copied to each of the three
        mitigations it contains. Section names equal section ids here, so
        detector B cannot apply and the flag is C's alone.
        """
        shared = LONG + " Mitigations subsection."
        corpus = _corpus(tmp_path, [
            {"control_id": "3.3.2", "title": "Mitigations",
             "description": shared},
            {"control_id": "3.3.2.1", "title": "Adversarial training",
             "description": shared},
        ])
        links = _links(tmp_path, [_row("3.3.2", "3.3.2"), _row("3.3.2.1", "3.3.2.1")])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.by_id == 2
        assert row.distinct_anchors == 1
        assert row.wrong_anchor_risk == 2

    def test_the_applicable_denominator_is_reported(self, tmp_path: Path) -> None:
        """A zero over a zero denominator proves nothing, so it is countable."""
        from tract.corpus_report import wrong_anchor_applicable

        corpus = _corpus(tmp_path, [
            {"control_id": "WSTG-INFO-01", "title": "Information gathering",
             "description": LONG + " Gathering."},
        ])
        links = _links(tmp_path, [_row("WSTG-INFO-01", "WSTG-INFO-01")])
        assert wrong_anchor_applicable(build_corpus_report(links, corpus)) == {
            "demo": 0,
        }


class TestFloors:
    def test_a_framework_below_its_floor_is_reported(self, tmp_path: Path) -> None:
        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "One", "description": LONG},
        ])
        links = _links(tmp_path, [_row("C-1", "One"), _row("C-2", "Two")])
        report = build_corpus_report(links, corpus)
        assert check_join_floors(report, {"demo": 0.50}) == []
        assert len(check_join_floors(report, {"demo": 0.90})) == 1

    def test_a_floor_for_an_absent_framework_raises(self, tmp_path: Path) -> None:
        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "One", "description": LONG},
        ])
        links = _links(tmp_path, [_row("C-1", "One")])
        report = build_corpus_report(links, corpus)
        with pytest.raises(KeyError, match="no curated links"):
            check_join_floors(report, {"absent": 0.50})

    def test_the_restricted_group_is_dropped_by_name_not_by_deletion(
        self, tmp_path: Path,
    ) -> None:
        """Rule 7. CI has 29 frameworks, the overlay has 31."""
        from tract.corpus_report import floors_for_report

        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "One", "description": LONG},
        ])
        links = _links(tmp_path, [_row("C-1", "One")])
        report = build_corpus_report(links, corpus)
        applicable, skipped = floors_for_report(
            report, {"demo": 0.50, "etsi": 1.00}, frozenset({"etsi"}),
        )
        assert applicable == {"demo": 0.50}
        assert skipped == ["etsi"]


class TestDerivedFloors:
    """The criterion is committed before the run it gates, and it is checkable."""

    def test_every_floor_has_a_ceiling(self) -> None:
        assert sorted(JOIN_FLOORS) == sorted(JOIN_CEILINGS)

    def test_no_floor_exceeds_its_arithmetic_ceiling(self) -> None:
        """Not `floor <= 1.0`, which is true of every literal in the dict.

        The previous plan carried three impossible floors: dsomm 1.00 against a
        maximum of 0.9953, wstg 0.96 against 0.9322, enisa 0.80 against 0.721.
        """
        for framework_id, floor in JOIN_FLOORS.items():
            ceiling = JOIN_CEILINGS[framework_id]
            assert 0.0 < floor <= ceiling, framework_id

    def test_each_floor_is_its_ceiling_rounded_down(self) -> None:
        for framework_id, ceiling in JOIN_CEILINGS.items():
            expected = math.floor(round(ceiling * 100, 6)) / 100
            assert JOIN_FLOORS[framework_id] == pytest.approx(expected), (
                framework_id
            )

    def test_the_floors_cover_the_eleven_pending_frameworks(self) -> None:
        assert {
            "biml", "csa_ccm", "dsomm", "enisa", "etsi", "nist_800_63",
            "nist_ssdf", "owasp_proactive_controls", "owasp_top10_2021",
            "samm", "wstg",
        } <= set(JOIN_FLOORS)
```

- [ ] **Step 4: Run the tests to verify they fail**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"$PY" -m pytest tests/test_corpus_report.py -q
```

Expected: collection error, `ModuleNotFoundError: No module named 'tract.corpus_report'`.

- [ ] **Step 5: Add the two accessors the report needs**

`ProseIndex` exposes only `lookup`, and the report has to say which channel answered. Reaching into
`index._by_title` from another module would couple the report to a private attribute.

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

- [ ] **Step 6: Write the instrument**

```python
# tract/corpus_report.py — create

"""The corpus join report: the one instrument the parser plan is gated on.

A count of links resolved cannot distinguish 615 links unstacked onto 615
distinct anchors from 615 links collapsed onto 40 coarse ones. Both make the
same number rise, and the second is a regression dressed as progress. So this
module reports the anchor side as well as the link side, and reports both
through the same lookup order the training and evaluation paths use, rather
than through a set intersection that would accept a join the consumer cannot
perform.

Two anchor columns exist because the first version of this instrument reported
a three-times-overstated gain. `distinct_anchors` counts the anchors a resolved
link reaches. `fallback_anchors` counts the distinct section names the trainer
already gets for links that resolve to nothing, because `select_control_text`
falls back to `section_name` rather than failing. A framework with 734
unresolved links is not a framework with zero anchors, and reporting it that
way turned a +152 gain into a +452 headline.

Columns, and the failure each one answers:

    by_title / by_id / unresolved   which channel carried the join
    fallback_anchors                what the trainer gets without a join
    distinct_anchors                the number every downstream metric rests on
    distinct_anchors_pre_truncation two anchors merging into one after the cut
    links_per_anchor                collapse, visible
    truncated                       anchors the encoder budget cuts
    nested_anchors                  an anchor contained in another anchor
    contained_anchors               the stricter prefix-only form, for continuity
    dropped_by_prose_rule           controls ProseIndex never indexed, corpus-wide
    wrong_anchor_risk               three detectors, two of them id-side
    anchor_source_*                 what kind of text the anchor is
    distinct_hubs / links_per_hub   hub-side concentration, for a later
                                    agreement study
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from tract.config import (
    MAX_ANCHOR_CHARS,
    PROJECT_ROOT,
    RESTRICTED_FRAMEWORK_IDS,
    TRAINING_DIR,
)
from tract.io import atomic_write_text
from tract.text_selection import (
    ProseIndex,
    TextSelection,
    _is_prose,
    canonical_framework,
    merged_corpus_path,
    normalize_section_id,
    prepare_anchor,
    strip_markup,
)

logger = logging.getLogger(__name__)

CURATED_LINKS_PATH: Final[Path] = TRAINING_DIR / "hub_links_curated.jsonl"

# Evidence, not results. Tracked by design through the .gitignore negations
# this task adds, and anchored to PROJECT_ROOT so a reader never depends on the
# directory pytest happened to start in.
CORPUS_EVIDENCE_DIR: Final[Path] = PROJECT_ROOT / "results" / "corpus"

# The tracked corpus carries 29 frameworks, the licensed overlay 31. [measured]
# A report built from fewer than the full set cannot assert the restricted
# rows, and gating on file existence never skips because the tracked file
# always exists. Task 15 owns updating this if the rebuild changes the census.
FULL_CORPUS_FRAMEWORK_COUNT: Final[int] = 31

# A parser that assembles an anchor out of several source fragments marks it,
# so the report can separate parser-written text from publisher-written text.
# Absent means the publisher wrote it.
TEXT_ORIGIN_METADATA_KEY: Final[str] = "text_origin"
SYNTHETIC_TEXT_ORIGIN: Final[str] = "synthetic"

_WHITESPACE = re.compile(r"\s+")

# Separators that make one identifier a parent of another: "5.2" of "5.2.2",
# "WSTG-INFO" of "WSTG-INFO-01", "IPY" of "IPY-01".
_ID_SEPARATORS: Final[tuple[str, ...]] = (".", "-", "_", ":", " ")


def _fold(text: str | None) -> str:
    """Whitespace-collapsed, case-folded form, for comparing a name to a title."""
    return _WHITESPACE.sub(" ", (text or "").strip()).casefold()


@dataclass(frozen=True)
class ControlFacts:
    """What the report needs about a control that TextSelection does not carry."""

    title: str
    origin: str


@dataclass
class LinkResolution:
    """One curated link, resolved, carrying no anchor text.

    Digest and length only. The file this serialises to is tracked for every
    framework including the licence-restricted ones, so it must hold nothing a
    publisher reserves. `section_id` and `section_name` come from
    hub_links_curated.jsonl, which is already tracked.
    """

    framework_id: str
    section_id: str
    section_name: str
    cre_id: str
    link_type: str
    channel: str
    anchor_source: str
    anchor_sha256: str
    anchor_chars: int
    truncated: bool
    wrong_anchor: bool
    wrong_anchor_checked: bool

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrameworkJoin:
    """One framework's join, on both the link side and the anchor side."""

    framework_id: str
    standard_name: str
    links: int = 0
    by_title: int = 0
    by_id: int = 0
    unresolved: int = 0
    fallback_anchors: int = 0
    distinct_anchors: int = 0
    distinct_anchors_pre_truncation: int = 0
    links_per_anchor: float = 0.0
    truncated: int = 0
    nested_anchors: int = 0
    contained_anchors: int = 0
    dropped_by_prose_rule: int = 0
    wrong_anchor_risk: int = 0
    anchor_source_full_text: int = 0
    anchor_source_description: int = 0
    anchor_source_title: int = 0
    anchor_source_synthetic: int = 0
    distinct_hubs: int = 0
    links_per_hub: float = 0.0
    resolution_rate: float = 0.0

    def finalise(self) -> None:
        resolved = self.by_title + self.by_id
        self.resolution_rate = 0.0 if not self.links else resolved / self.links
        self.links_per_anchor = (
            0.0 if not self.distinct_anchors else resolved / self.distinct_anchors
        )
        # All links, not only the resolved ones: hub concentration is a
        # property of the curated link file and must not move when a parser
        # lands.
        self.links_per_hub = (
            0.0 if not self.distinct_hubs else self.links / self.distinct_hubs
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
    corpus_framework_count: int = 0
    max_anchor_chars: int = MAX_ANCHOR_CHARS
    # Serialised to the JSONL rather than into to_json(): 4,405 rows would
    # dominate the summary artifact a reader opens first.
    resolution_rows: list[LinkResolution] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "corpus_path": self.corpus_path,
            "corpus_sha256": self.corpus_sha256,
            "corpus_framework_count": self.corpus_framework_count,
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
    """The framework records, from either corpus shape.

    Both real corpus files are mappings keyed
    [framework_count, frameworks, generated_date, total_controls]. [measured]
    Preferring the named key rather than "the first list value" means a new
    top-level list cannot silently take over.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        records = data.get("frameworks")
        if isinstance(records, list):
            return records
        for value in data.values():
            if isinstance(value, list):
                return value
    raise ValueError(
        f"{path} holds no list of framework records. The merged corpus is "
        f"either a list or a mapping carrying one under 'frameworks'."
    )


def _control_facts(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], ControlFacts], dict[str, int]]:
    """Title and text origin per indexed anchor, plus the prose-rule census.

    Keyed on the selection text rather than on the control id, because
    TextSelection carries no back-reference and duplicating ProseIndex's key
    logic here would be a second implementation to keep in step. Two controls
    with byte-identical text collapse to one entry, and the first in corpus
    order wins, and their anchors are indistinguishable to the encoder anyway.

    The census counts every framework in the corpus, including those with no
    curated links. Summing only the link-bearing subset read 522 where the
    corpus holds 558.
    """
    facts: dict[tuple[str, str], ControlFacts] = {}
    dropped: dict[str, int] = {}
    for record in records:
        framework = canonical_framework(str(record.get("framework_name") or ""))
        for control in record.get("controls") or []:
            title = str(control.get("title") or "")
            description = str(control.get("description") or "")
            full_text = str(control.get("full_text") or "")
            if full_text.strip():
                text = full_text.strip()
            elif _is_prose(description, title):
                text = description.strip()
            else:
                dropped[framework] = dropped.get(framework, 0) + 1
                continue
            metadata = control.get("metadata") or {}
            facts.setdefault(
                (framework, text),
                ControlFacts(
                    title=title,
                    origin=str(
                        metadata.get(TEXT_ORIGIN_METADATA_KEY) or "source"
                    ),
                ),
            )
    return facts, dropped


def _lookup_with_channel(
    index: ProseIndex,
    canonical: str,
    section_id: str | None,
    section_name: str | None,
) -> tuple[TextSelection | None, str]:
    """ProseIndex.lookup, plus which channel answered.

    Deliberately reimplements lookup's branch order rather than calling it: the
    report has to say *how* a link resolved, and lookup returns only the text.
    The order here must stay identical to lookup's, title then id, and
    tests/test_corpus_report.py::TestChannelParity asserts the two agree on
    every curated link in the real corpus.
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


def _fallback_anchor(section_id: str | None, section_name: str | None) -> str:
    """What select_control_text hands the trainer when the index misses.

    Same expression and same normalisation as the fallback branch of
    select_control_text, so the count here is the count the trainer sees.
    """
    fallback = (section_name or section_id or "").strip()
    if not fallback:
        return ""
    text, _ = prepare_anchor(strip_markup(fallback))
    return text


def _classify_anchor(
    selection: TextSelection, anchor: str, facts: ControlFacts | None,
) -> str:
    """Which of the four kinds of text this anchor is.

    Parser-assembled text wins over everything, because its provenance is the
    parser rather than the publisher. A stored anchor that only restates the
    control's own title is reported as `title` even though it arrived through
    full_text or description, since that is what the encoder reads and the
    prose rule cannot see it once a parser writes full_text.
    """
    if facts is not None and facts.origin == SYNTHETIC_TEXT_ORIGIN:
        return "synthetic"
    if facts is not None and facts.title and _fold(anchor) == _fold(facts.title):
        return "title"
    return selection.source


def _count_nested(anchors: set[str]) -> int:
    """Anchors contained anywhere inside a longer anchor of the same framework.

    Containment, not a strict prefix. An ETSI clause 5.2 that rolls up 5.2.2
    opens with its own lead paragraph, so the child sits in the middle of the
    parent and a prefix test reads 0. Measured on the current corpus this
    column reads 0 for every framework. [measured]
    """
    ordered = sorted(anchors, key=lambda item: (len(item), item))
    return sum(
        1
        for position, short in enumerate(ordered)
        if any(short in longer for longer in ordered[position + 1:])
    )


def _count_contained(anchors: set[str]) -> int:
    """The strict-prefix count the first version of this module reported.

    Kept so a reader comparing two runs across this change can separate the
    definition change from a corpus change.
    """
    ordered = sorted(anchors, key=lambda item: (len(item), item))
    return sum(
        1
        for position, short in enumerate(ordered)
        if any(longer.startswith(short) for longer in ordered[position + 1:])
    )


def _is_ancestor_id(parent: str, child: str) -> bool:
    """Whether one normalised section id is a parent of another."""
    if not parent or not child or len(child) <= len(parent):
        return False
    if not child.startswith(parent):
        return False
    return child[len(parent)] in _ID_SEPARATORS


@dataclass
class _Resolved:
    """One link that reached an anchor, with everything a detector needs."""

    link: Mapping[str, Any]
    normalized_id: str
    channel: str
    anchor: str
    raw_text: str
    truncated: bool
    anchor_source: str
    control_title: str


def _wrong_anchor(
    index: ProseIndex,
    canonical: str,
    entry: _Resolved,
    by_id_anchor: Mapping[str, str],
) -> tuple[bool, bool]:
    """Whether this link's anchor is suspect, and whether anything checked it.

    Three detectors, because the first version had one and it lived entirely
    inside the title branch. Nine of the eleven frameworks this plan adds are
    engineered to resolve through the id channel, so a title-only detector made
    `wrong_anchor_risk == 0` unfailable for them.

    A: the title channel answered and the id channel would have answered
       differently. The curator wrote both, and they disagree.
    B: the id channel answered, the link also carried a name that says
       something the id does not, and the control the id reached does not carry
       that name anywhere in its title.
    C: a coarser id and a finer id in the same framework reached the same
       paragraph. This is the NIST AI 100-2 failure that put title first in the
       lookup order, and neither A nor B can see it.

    The second return value is the denominator. A framework whose links carry
    `section_name == section_id` and no ancestor relations has zero applicable
    checks, so a zero in this column proves nothing about it, and
    wrong_anchor_applicable() makes that legible instead of leaving it implied.
    """
    name = str(entry.link.get("section_name") or "")
    checked = False
    flagged = False

    if entry.channel == "title":
        if entry.normalized_id:
            checked = True
            other = index.by_id(canonical, entry.normalized_id)
            if other is not None and prepare_anchor(other.text)[0] != entry.anchor:
                flagged = True
        return flagged, checked

    if name and _fold(name) != _fold(entry.normalized_id):
        checked = True
        title = _fold(entry.control_title)
        if title and _fold(name) not in title and title not in _fold(name):
            flagged = True

    for other_id, other_anchor in by_id_anchor.items():
        if other_id == entry.normalized_id:
            continue
        if _is_ancestor_id(other_id, entry.normalized_id) or _is_ancestor_id(
            entry.normalized_id, other_id
        ):
            checked = True
            if other_anchor == entry.anchor:
                flagged = True
    return flagged, checked


def build_corpus_report(
    links_path: Path | None = None, corpus_path: Path | None = None,
) -> CorpusReport:
    """Resolve every curated link through ProseIndex and report the join."""
    links_file = links_path or CURATED_LINKS_PATH
    corpus_file = corpus_path or merged_corpus_path()

    records = _load_records(corpus_file)
    index = ProseIndex(records)
    facts, dropped = _control_facts(records)
    grouped = _load_links(links_file)

    rows: list[FrameworkJoin] = []
    resolution_rows: list[LinkResolution] = []
    totals = FrameworkJoin(framework_id="TOTAL", standard_name="")
    all_anchors: set[str] = set()
    all_pre_truncation: set[str] = set()
    all_fallbacks: set[str] = set()
    all_hubs: set[str] = set()

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
        pre_truncation: set[str] = set()
        fallbacks: set[str] = set()
        hubs: set[str] = set()
        resolved: list[_Resolved] = []
        unresolved_rows: list[tuple[Mapping[str, Any], str]] = []

        for link in links:
            hubs.add(str(link.get("cre_id") or ""))
            selection, channel = _lookup_with_channel(
                index, canonical, link.get("section_id"), link.get("section_name"),
            )
            normalized = normalize_section_id(link.get("section_id"))
            if selection is None:
                row.unresolved += 1
                fallback = _fallback_anchor(
                    link.get("section_id"), link.get("section_name"),
                )
                if fallback:
                    fallbacks.add(fallback)
                unresolved_rows.append((link, fallback))
                continue

            anchor, was_cut = prepare_anchor(selection.text)
            control = facts.get((canonical, selection.text))
            resolved.append(
                _Resolved(
                    link=link,
                    normalized_id=normalized,
                    channel=channel,
                    anchor=anchor,
                    raw_text=selection.text,
                    truncated=was_cut,
                    anchor_source=_classify_anchor(selection, anchor, control),
                    control_title=control.title if control is not None else "",
                )
            )
            anchors.add(anchor)
            pre_truncation.add(selection.text)
            row.truncated += int(was_cut)
            if channel == "title":
                row.by_title += 1
            else:
                row.by_id += 1

        by_id_anchor: dict[str, str] = {}
        for entry in resolved:
            if entry.channel == "id" and entry.normalized_id:
                by_id_anchor.setdefault(entry.normalized_id, entry.anchor)

        for entry in resolved:
            flagged, checked = _wrong_anchor(index, canonical, entry, by_id_anchor)
            row.wrong_anchor_risk += int(flagged)
            if entry.anchor_source == "full_text":
                row.anchor_source_full_text += 1
            elif entry.anchor_source == "description":
                row.anchor_source_description += 1
            elif entry.anchor_source == SYNTHETIC_TEXT_ORIGIN:
                row.anchor_source_synthetic += 1
            else:
                row.anchor_source_title += 1
            resolution_rows.append(
                LinkResolution(
                    framework_id=framework_id,
                    section_id=str(entry.link.get("section_id") or ""),
                    section_name=str(entry.link.get("section_name") or ""),
                    cre_id=str(entry.link.get("cre_id") or ""),
                    link_type=str(entry.link.get("link_type") or ""),
                    channel=entry.channel,
                    anchor_source=entry.anchor_source,
                    anchor_sha256=hashlib.sha256(
                        entry.anchor.encode("utf-8")
                    ).hexdigest(),
                    anchor_chars=len(entry.anchor),
                    truncated=entry.truncated,
                    wrong_anchor=flagged,
                    wrong_anchor_checked=checked,
                )
            )

        # Unresolved links carry the fallback anchor the trainer receives, so
        # the BEFORE file holds the text-quality baseline the AFTER is read
        # against. Without these rows the JSONL would describe only the links
        # that already work.
        for missed, fallback in unresolved_rows:
            resolution_rows.append(
                LinkResolution(
                    framework_id=framework_id,
                    section_id=str(missed.get("section_id") or ""),
                    section_name=str(missed.get("section_name") or ""),
                    cre_id=str(missed.get("cre_id") or ""),
                    link_type=str(missed.get("link_type") or ""),
                    channel="unresolved",
                    anchor_source="title",
                    anchor_sha256=hashlib.sha256(
                        fallback.encode("utf-8")
                    ).hexdigest(),
                    anchor_chars=len(fallback),
                    truncated=False,
                    wrong_anchor=False,
                    wrong_anchor_checked=False,
                )
            )

        row.distinct_anchors = len(anchors)
        row.distinct_anchors_pre_truncation = len(pre_truncation)
        row.fallback_anchors = len(fallbacks)
        row.distinct_hubs = len(hubs)
        row.nested_anchors = _count_nested(anchors)
        row.contained_anchors = _count_contained(anchors)
        row.finalise()
        rows.append(row)

        all_anchors |= anchors
        all_pre_truncation |= pre_truncation
        all_fallbacks |= fallbacks
        all_hubs |= hubs
        totals.links += row.links
        totals.by_title += row.by_title
        totals.by_id += row.by_id
        totals.unresolved += row.unresolved
        totals.truncated += row.truncated
        totals.nested_anchors += row.nested_anchors
        totals.contained_anchors += row.contained_anchors
        totals.wrong_anchor_risk += row.wrong_anchor_risk
        totals.anchor_source_full_text += row.anchor_source_full_text
        totals.anchor_source_description += row.anchor_source_description
        totals.anchor_source_title += row.anchor_source_title
        totals.anchor_source_synthetic += row.anchor_source_synthetic

    # The census covers every framework in the corpus, including the ones with
    # no curated links and therefore no row above.
    totals.dropped_by_prose_rule = sum(dropped.values())
    totals.distinct_anchors = len(all_anchors)
    totals.distinct_anchors_pre_truncation = len(all_pre_truncation)
    totals.fallback_anchors = len(all_fallbacks)
    totals.distinct_hubs = len(all_hubs)
    totals.finalise()

    logger.info(
        "Corpus join: %d links, %d resolved, %d distinct anchors, %d fallback "
        "anchors, %d controls outside the prose index, over %d frameworks",
        totals.links, totals.by_title + totals.by_id, totals.distinct_anchors,
        totals.fallback_anchors, totals.dropped_by_prose_rule, len(records),
    )

    return CorpusReport(
        per_framework=rows,
        totals=totals,
        corpus_path=str(corpus_file),
        corpus_sha256=_sha256(corpus_file),
        corpus_framework_count=len(records),
        links_path=str(links_file),
        links_sha256=_sha256(links_file),
        resolution_rows=resolution_rows,
    )


def write_link_resolution(report: CorpusReport, path: Path) -> None:
    """One row per curated link, digests only, safe to track for any framework.

    This is what a later label-agreement study needs to sample the frameworks
    this plan re-weights: which channel carried each link, what kind of text it
    reached, how long that text was, and whether a detector questioned it. The
    premortem's answer to "does the plan create that artifact" was no.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        "".join(
            json.dumps(entry.to_json(), sort_keys=True) + "\n"
            for entry in report.resolution_rows
        ),
        path,
    )
    logger.info("wrote %d link resolutions to %s", len(report.resolution_rows), path)


def wrong_anchor_applicable(report: CorpusReport) -> dict[str, int]:
    """Per framework, how many links a wrong-anchor detector could fire on.

    `wrong_anchor_risk == 0` over a denominator of 0 is a fact about the link
    file, not about the parser. Reporting the denominator is what stops that
    zero from being read as a pass.
    """
    counts: dict[str, int] = {row.framework_id: 0 for row in report.per_framework}
    for entry in report.resolution_rows:
        if entry.wrong_anchor_checked:
            counts[entry.framework_id] = counts.get(entry.framework_id, 0) + 1
    return counts


def check_join_floors(
    report: CorpusReport, floors: Mapping[str, float],
) -> list[str]:
    """One message per framework whose resolution rate is under its floor.

    A floor is derived from the link file and the source before the parser is
    written, never pasted from the run being gated. See JOIN_CEILINGS for the
    arithmetic that produced each one.

    Raises:
        KeyError: If a floor names a framework with no curated links. Silently
            skipping it would retire the gate.
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


def floors_for_report(
    report: CorpusReport,
    floors: Mapping[str, float],
    restricted: frozenset[str] = RESTRICTED_FRAMEWORK_IDS,
) -> tuple[dict[str, float], list[str]]:
    """The floors this corpus can carry, and the named group it cannot.

    CI has no licensed overlay, so its corpus holds 29 frameworks against the
    overlay's 31, and every restricted row would read 0.0000 and hard-fail.
    Gating on file existence never skips, because the tracked corpus always
    exists. This gates on content instead, and returns the skipped group by
    name so the reason is stated rather than implied. Never delete a floor to
    make CI green: that retires the only gate on a parser nobody can inspect.

    The Rule 3 author widens the default to OVERLAY_FRAMEWORK_IDS when text
    routing moves the conditional frameworks into the overlay.
    """
    if report.corpus_framework_count >= FULL_CORPUS_FRAMEWORK_COUNT:
        return dict(floors), []
    applicable = {k: v for k, v in floors.items() if k not in restricted}
    skipped = sorted(k for k in floors if k in restricted)
    if skipped:
        logger.warning(
            "corpus has %d frameworks against %d in the full set, so the "
            "licensed overlay is absent from this checkout and these floors "
            "cannot be asserted: %s",
            report.corpus_framework_count, FULL_CORPUS_FRAMEWORK_COUNT,
            ", ".join(skipped),
        )
    return applicable, skipped


def format_table(report: CorpusReport) -> str:
    """The report as a fixed-width table, for logs and for the run ledger."""
    header = (
        f"{'framework':26s} {'links':>5s} {'ttl':>5s} {'id':>4s} {'unres':>5s} "
        f"{'fb':>4s} {'anch':>5s} {'pre':>5s} {'l/a':>5s} {'trunc':>5s} "
        f"{'nest':>4s} {'cont':>4s} {'noidx':>5s} {'wrong':>5s} "
        f"{'ftxt':>5s} {'desc':>5s} {'titl':>4s} {'synt':>4s} "
        f"{'hubs':>5s} {'l/h':>5s} {'rate':>6s}"
    )
    lines = [header, "-" * len(header)]
    for row in [*report.per_framework, report.totals]:
        lines.append(
            f"{row.framework_id:26s} {row.links:5d} {row.by_title:5d} "
            f"{row.by_id:4d} {row.unresolved:5d} {row.fallback_anchors:4d} "
            f"{row.distinct_anchors:5d} {row.distinct_anchors_pre_truncation:5d} "
            f"{row.links_per_anchor:5.2f} {row.truncated:5d} "
            f"{row.nested_anchors:4d} {row.contained_anchors:4d} "
            f"{row.dropped_by_prose_rule:5d} {row.wrong_anchor_risk:5d} "
            f"{row.anchor_source_full_text:5d} {row.anchor_source_description:5d} "
            f"{row.anchor_source_title:4d} {row.anchor_source_synthetic:4d} "
            f"{row.distinct_hubs:5d} {row.links_per_hub:5.2f} "
            f"{row.resolution_rate:6.4f}"
        )
    return "\n".join(lines)
```

- [ ] **Step 7: Commit the floors, before any parser exists**

The criterion cannot move in the same commit as the result it gates. Version 2 put `JOIN_FLOORS` in
Task 16 alongside the report it gates, and the plan file itself is gitignored (`.gitignore:25`), so a
floor edited mid-run would leave no diff anywhere. Both dicts land here, in tracked code, in the same
commit as the instrument and before a line of parser code exists.

```python
# tract/corpus_report.py — append

# Per-framework join ceilings, each derived from the curated link file and the
# pinned source in that framework's own plan task, BEFORE its parser existed.
# Written as the fraction so a transcription error is visible. The eleven
# pending frameworks resolve 0 of 734 links today, so none of these was read
# off the run it gates.
#
#   dsomm        213/214  one activity's statement is 11 characters
#   wstg         109/118  nine links name ids absent from the archive
#   nist_800_63   78/79   one section_id is the fragment "are g"
#   biml          21/21   with the two declared alternates
#   enisa         68/68   with Table 3, Annex C and name repair
#   etsi          36/36   every technique declared to its own clause
#   csa_ccm       29/29   seven renamed ids resolve by title
#   nist_ssdf     46/46   with the two declared alt_ids
#   samm          30/30
#   owasp_top10_2021          17/17
#   owasp_proactive_controls  76/76
#
# The eleven below the fold already resolve today, and their ceilings are the
# rates measured on the BEFORE corpus. They are here as a regression gate: the
# rebuild in Task 15 must not cost them. Each miss is known and named.
#   cwe          612/613  CWE-937 was withdrawn upstream
#   nist_800_53  298/300  SC-23(1) and SC-23(3) were withdrawn
#   iso_27001     92/94   A.7.8 and A.7.9 are shorter than their own titles
#                         plus PROSE_MIN_EXTRA_CHARS, so ProseIndex excludes
#                         them on purpose
JOIN_CEILINGS: Final[Mapping[str, float]] = {
    "asvs": 277 / 277,
    "biml": 21 / 21,
    "capec": 1799 / 1799,
    "csa_ccm": 29 / 29,
    "cwe": 612 / 613,
    "dsomm": 213 / 214,
    "enisa": 68 / 68,
    "etsi": 36 / 36,
    "iso_27001": 92 / 94,
    "mitre_atlas": 65 / 65,
    "nist_800_53": 298 / 300,
    "nist_800_63": 78 / 79,
    "nist_ai_100_2": 45 / 45,
    "nist_ssdf": 46 / 46,
    "owasp_ai_exchange": 64 / 64,
    "owasp_cheat_sheets": 391 / 391,
    "owasp_llm_top10": 13 / 13,
    "owasp_ml_top10": 10 / 10,
    "owasp_proactive_controls": 76 / 76,
    "owasp_top10_2021": 17 / 17,
    "samm": 30 / 30,
    "wstg": 109 / 118,
}

# Each ceiling rounded down to two decimals, which
# tests/test_corpus_report.py::TestDerivedFloors asserts rather than trusts.
# A floor of 1.00 where the ceiling is 1.00 is the right number, not an
# oversight: below it, a link the source can answer stopped resolving.
JOIN_FLOORS: Final[Mapping[str, float]] = {
    "asvs": 1.00,
    "biml": 1.00,
    "capec": 1.00,
    "csa_ccm": 1.00,
    "cwe": 0.99,
    "dsomm": 0.99,
    "enisa": 1.00,
    "etsi": 1.00,
    "iso_27001": 0.97,
    "mitre_atlas": 1.00,
    "nist_800_53": 0.99,
    "nist_800_63": 0.98,
    "nist_ai_100_2": 1.00,
    "nist_ssdf": 1.00,
    "owasp_ai_exchange": 1.00,
    "owasp_cheat_sheets": 1.00,
    "owasp_llm_top10": 1.00,
    "owasp_ml_top10": 1.00,
    "owasp_proactive_controls": 1.00,
    "owasp_top10_2021": 1.00,
    "samm": 1.00,
    "wstg": 0.92,
}
```

- [ ] **Step 8: Add the channel-parity test, with the loader that reads a dict**

The report reimplements `lookup`'s branch order. If the two ever disagree, the report describes a
join the pipeline does not perform, which is the defect that got the previous plan rejected. Version
2's test built `ProseIndex([])` and asserted `True == True` 4,405 times. There is no skip: parity is
a property of the code, both input files are tracked or always present, and a missing baseline is a
failure rather than a pass (Ruling R3).

```python
# tests/test_corpus_report.py — append

class TestChannelParity:
    def test_report_and_lookup_agree_on_every_curated_link(self) -> None:
        """The report must describe the join the pipeline performs."""
        from tract.corpus_report import (
            CURATED_LINKS_PATH, _load_records, _lookup_with_channel,
        )
        from tract.text_selection import (
            ProseIndex, canonical_framework, merged_corpus_path,
        )

        corpus = merged_corpus_path()
        records = _load_records(corpus)
        # The bug this test exists to prevent: an empty index agrees with
        # itself on everything.
        assert records, f"{corpus} produced no framework records"
        index = ProseIndex(records)
        assert len(index) > 1000, (
            f"prose index holds {len(index)} controls; a near-empty index "
            f"makes every assertion below vacuous"
        )

        compared = 0
        with CURATED_LINKS_PATH.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                canonical = canonical_framework(row.get("standard_name", ""))
                mine, _ = _lookup_with_channel(
                    index, canonical, row.get("section_id"),
                    row.get("section_name"),
                )
                theirs = index.lookup(
                    row.get("standard_name", ""), row.get("section_id"),
                    row.get("section_name"),
                )
                assert (mine is None) == (theirs is None), row
                if mine is not None and theirs is not None:
                    assert mine.text == theirs.text, row
                compared += 1
        assert compared == 4405, compared

    def test_the_report_resolves_a_useful_number_of_links(self) -> None:
        """A second guard against an index that silently held nothing."""
        report = build_corpus_report()
        resolved = report.totals.by_title + report.totals.by_id
        assert report.totals.links == 4405
        assert resolved >= 3600, resolved
        assert report.corpus_framework_count >= 29
```

- [ ] **Step 9: Write the CLI**

```python
# scripts/corpus_report.py — create

"""Print or persist the corpus join report.

    PYTHONPATH=. "$PY" scripts/corpus_report.py
    PYTHONPATH=. "$PY" scripts/corpus_report.py --tag before
    PYTHONPATH=. "$PY" scripts/corpus_report.py --out results/corpus/scratch.json

The same entry point produces the BEFORE artifact, every per-parser acceptance
check, and the final corpus report. One instrument, one code path: a parser
accepted by a measurement its consumer does not use is a parser accepted by
nothing.

--tag writes the pair a later reader needs together: results/corpus/<tag>.json
for the summary and results/corpus/link_resolution_<tag>.jsonl for the per-link
record. Both paths are anchored to PROJECT_ROOT, so the working directory does
not decide where evidence lands.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tract.corpus_report import (
    CORPUS_EVIDENCE_DIR,
    build_corpus_report,
    format_table,
    write_link_resolution,
    wrong_anchor_applicable,
)
from tract.io import atomic_write_json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--links", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--tag", type=str, default=None,
        help="write results/corpus/<tag>.json and link_resolution_<tag>.jsonl",
    )
    args = parser.parse_args()

    report = build_corpus_report(args.links, args.corpus)
    print(format_table(report))
    print()
    print(f"corpus  {report.corpus_path}  sha256 {report.corpus_sha256[:16]}")
    print(f"links   {report.links_path}  sha256 {report.links_sha256[:16]}")
    print(f"frameworks in corpus  {report.corpus_framework_count}")
    print()
    print("wrong-anchor checks applicable, per framework:")
    for framework_id, applicable in sorted(wrong_anchor_applicable(report).items()):
        risk = report.by_id(framework_id).wrong_anchor_risk
        note = "" if applicable else "   (blind: no detector applies)"
        print(f"  {framework_id:26s} {risk:4d} of {applicable:5d}{note}")

    if args.out is not None:
        atomic_write_json(report.to_json(), args.out)
        print(f"wrote {args.out}")

    if args.tag is not None:
        summary = CORPUS_EVIDENCE_DIR / f"{args.tag}.json"
        detail = CORPUS_EVIDENCE_DIR / f"link_resolution_{args.tag}.jsonl"
        atomic_write_json(report.to_json(), summary)
        write_link_resolution(report, detail)
        print(f"wrote {summary}")
        print(f"wrote {detail}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 10: Run the tests and the typecheck**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"$PY" -m pytest tests/test_corpus_report.py -q
"$PY" -m mypy tract/corpus_report.py scripts/corpus_report.py \
    tract/text_selection.py --strict
```

Expected: all tests pass, no mypy errors. If `TestChannelParity` fails on `compared == 4405`, the
curated link file changed and every number in this task needs re-measuring before going further.

- [ ] **Step 11: Record the attainable range of `wrong_anchor_risk`, per framework**

A gate that cannot fail is a defect. Before any parser lands, measure the denominator so nobody
writes `assert wrong_anchor_risk == 0` where zero is guaranteed.

```bash
PYTHONPATH=. "$PY" - <<'PYEOF'
import json
import re
from tract.config import TRAINING_DIR
from tract.text_selection import normalize_section_id

SEP = (".", "-", "_", ":", " ")
WS = re.compile(r"\s+")


def fold(text):
    return WS.sub(" ", (text or "").strip()).casefold()


def ancestor(parent, child):
    return (
        bool(parent) and bool(child) and len(child) > len(parent)
        and child.startswith(parent) and child[len(parent)] in SEP
    )


grouped = {}
with (TRAINING_DIR / "hub_links_curated.jsonl").open(encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            row = json.loads(line)
            grouped.setdefault(row["framework_id"], []).append(row)

eleven = [
    "biml", "csa_ccm", "dsomm", "enisa", "etsi", "nist_800_63", "nist_ssdf",
    "owasp_proactive_controls", "owasp_top10_2021", "samm", "wstg",
]
print(f"{'framework':26s}{'links':>6}{'B max':>7}{'C max':>7}")
for framework_id in eleven:
    links = grouped[framework_id]
    ids = [normalize_section_id(link.get("section_id")) for link in links]
    unique = sorted({i for i in ids if i})
    related = {
        i for i in unique
        if any(o != i and (ancestor(o, i) or ancestor(i, o)) for o in unique)
    }
    b = sum(
        1 for link in links
        if link.get("section_name")
        and fold(link["section_name"])
        != fold(normalize_section_id(link.get("section_id")))
    )
    c = sum(1 for i in ids if i in related)
    print(f"{framework_id:26s}{len(links):6d}{b:7d}{c:7d}")
PYEOF
```

Expected, and these are the numbers to write into the run ledger **[measured, this task]**:

```
framework                  links  B max  C max
biml                          21     21      0
csa_ccm                       29     29      0
dsomm                        214    214      3
enisa                         68     58      0
etsi                          36     34      8
nist_800_63                   79      0     14
nist_ssdf                     46     44      0
owasp_proactive_controls      76      0      0
owasp_top10_2021              17     17      0
samm                          30     30      0
wstg                         118      0      0
```

Read it this way. Detector A applies only to links that resolve through the title channel, and nine
of the eleven are designed to resolve through the id channel, so A contributes nothing for them.
Detectors B and C are the id-side ones. **`owasp_proactive_controls` and `wstg` have an attainable
maximum of zero on all three detectors**, because their `section_name` equals their `section_id` (so
B never applies) and their link ids carry no ancestor relations (so C never applies). For those two,
`wrong_anchor_risk == 0` is unfailable and asserting it measures nothing. Every other framework has a
real range and `csa_ccm` is expected to fire: its `IPY` link resolves to control IPY-01's title
rather than the IPY domain. **[measured, ML Engineer]**

What Task 16 asserts instead of `== 0`:

- For every pending framework, `wrong_anchor_risk` is either 0 or every flagged link appears in a
  committed adjudication list. The flags are in
  `results/corpus/link_resolution_after.jsonl` as `"wrong_anchor": true`, so an unadjudicated new
  flag is a diff, not a judgment call.
- For `owasp_proactive_controls` and `wstg`, record the blindness by name in the ledger rather than
  recording a pass.

- [ ] **Step 12: Capture the BEFORE state**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" scripts/corpus_report.py --tag before
```

Expected, and this is the free test of the instrument. These values were measured independently
while writing this task **[measured, all]**:

```
framework                  links   ttl   id unres   fb  anch   pre   l/a trunc nest cont noidx wrong  ftxt  desc titl synt  hubs   l/h   rate
asvs                         277   270    7     0    0   277   277  1.00     0    0    0     0     1     0   277    0    0   277  1.00 1.0000
biml                          21     0    0    21   17     0     0  0.00     0    0    0    20     0     0     0    0    0    11  1.91 0.0000
capec                       1799  1799    0     0    0   349   349  5.15    24    0    0     2     0    24  1775    0    0   194  9.27 1.0000
csa_ccm                       29     0    0    29   29     0     0  0.00     0    0    0    29     0     0     0    0    0    27  1.07 0.0000
cwe                          613   594   18     1    1   245   245  2.50    13    0    0    43     2    13   599    0    0   268  2.29 0.9984
dsomm                        214     0    0   214   18     0     0  0.00     0    0    0   183     0     0     0    0    0    24  8.92 0.0000
enisa                         68     0    0    68   33     0     0  0.00     0    0    0    38     0     0     0    0    0    56  1.21 0.0000
etsi                          36     0    0    36   24     0     0  0.00     0    0    0    27     0     0     0    0    0    29  1.24 0.0000
iso_27001                     94    88    4     2    2    91    91  1.01     0    0    0     2     5     0    92    0    0    47  2.00 0.9787
mitre_atlas                   65    56    9     0    0    43    43  1.51     0    0    0     0     0     0    65    0    0    41  1.59 1.0000
nist_800_53                  300   298    0     2    2   298   298  1.00    63    0    0     0     0    70   228    0    0    66  4.55 0.9933
nist_800_63                   79     0    0    79   25     0     0  0.00     0    0    0    25     0     0     0    0    0    70  1.13 0.0000
nist_ai_100_2                 45    29   16     0    0    22    22  2.05    21    0    0     0    20    24    21    0    0    27  1.67 1.0000
nist_ssdf                     46     0    0    46   44     0     0  0.00     0    0    0    44     0     0     0    0    0    28  1.64 0.0000
owasp_ai_exchange             64    38   26     0    0    63    63  1.02    31    0    0     0    11    35    29    0    0    64  1.00 1.0000
owasp_cheat_sheets           391   391    0     0    0    49    49  7.98   384    0    0     0     0   384     7    0    0   180  2.17 1.0000
owasp_llm_top10               13    12    1     0    0     6     6  2.17    13    0    0     0     0    13     0    0    0    10  1.30 1.0000
owasp_ml_top10                10     9    1     0    0     7     7  1.43    10    0    0     0     1    10     0    0    0     7  1.43 1.0000
owasp_proactive_controls      76     0    0    76   10     0     0  0.00     0    0    0    10     0     0     0    0    0    73  1.04 0.0000
owasp_top10_2021              17     0    0    17   10     0     0  0.00     0    0    0    10     0     0     0    0    0    16  1.06 0.0000
samm                          30     0    0    30   30     0     0  0.00     0    0    0    30     0     0     0    0    0    22  1.36 0.0000
wstg                         118     0    0   118   59     0     0  0.00     0    0    0    59     0     0     0    0    0   112  1.05 0.0000
TOTAL                       4405  3584   82   739  304  1450  1450  2.53   559    0    0   558    40   573  3093    0    0   458  9.62 0.8322

frameworks in corpus  31
```

Three properties must hold before going further.

1. **`frameworks in corpus` reads 31.** If it reads 29, the licensed overlay is missing from this
   checkout and `merged_corpus_path()` fell back to the tracked corpus. Fix that first: every later
   comparison would be against the wrong baseline (ledger lesson 5), and ISO is the corpus's only
   0.967-prose fold.
2. **ISO is not zero.** `iso_27001` resolves 92 of 94 at 0.9787, so the instrument can see a working
   join.
3. **The eleven are all zero on `anch` and non-zero on `fb`.** Zero resolved anchors, 299 fallback
   anchors between them, so the instrument can see a broken join and can also see what the trainer
   gets in spite of it.

Five totals are worth naming before any parser exists. **1,450 distinct anchors** carry 3,666
resolved links. **304 fallback anchors** carry the other 739, and **299 of the 304 belong to the
eleven**. **559 anchors are truncated** at `MAX_ANCHOR_CHARS`, 384 of them in `owasp_cheat_sheets`
alone. **558 controls are in the corpus and absent from the prose index**, because their description
does not exceed their title by `PROSE_MIN_EXTRA_CHARS`: 475 in the eleven, 43 CWE weaknesses, 2 CAPEC
patterns, ISO's A.7.8 and A.7.9, and 36 in three frameworks that carry no curated links at all (NIST
AI Risk Management Framework 25, AIUC-1 Standard 10, CoSAI 1). **`nest` and `cont` both read 0
corpus-wide**, so the containment change costs nothing today and exists for what the parsers will
build. **[measured, all]** No artifact in this repository reported any of the five before now.

- [ ] **Step 13: Confirm the evidence is stageable without `-f`**

```bash
git check-ignore -v results/corpus/before.json \
    || echo "before.json: not ignored"
git check-ignore -v results/corpus/link_resolution_before.jsonl \
    || echo "link_resolution_before.jsonl: not ignored"
"$PY" -c "
import json, pathlib
rows = [
    json.loads(line)
    for line in pathlib.Path(
        'results/corpus/link_resolution_before.jsonl'
    ).read_text(encoding='utf-8').splitlines()
]
print('rows', len(rows))
print('keys', sorted(rows[0]))
assert len(rows) == 4405, len(rows)
assert not any('anchor_text' in r for r in rows)
assert all(len(r['anchor_sha256']) == 64 for r in rows)
print('no anchor text in the file')
"
```

Expected: both paths report `not ignored`, `rows 4405`, twelve keys, and `no anchor text in the
file`. The JSONL carries digests and lengths only, which is what makes it safe to track for ETSI and
ISO alongside everything else.

- [ ] **Step 14: Commit**

```bash
git add tract/corpus_report.py scripts/corpus_report.py \
        tests/test_corpus_report.py tract/text_selection.py .gitignore \
        results/corpus/before.json \
        results/corpus/link_resolution_before.jsonl
git status --porcelain --short
git commit -m "feat: measure the corpus join by anchor and by fallback, not only by link"
```

`git status --porcelain --short` before the commit is not decoration. `git add` on an ignored path
exits 1 while still staging the other paths on the same command line, so a partially staged commit
looks successful. Confirm all seven paths are staged before committing.

---

### Task 2: An `alt_ids` channel on ProseIndex

`ProseIndex` reads `metadata["alt_titles"]` and nothing else. **[measured]** There is no id-side
equivalent, and three frameworks need one:

- **nist_ssdf**: two of 46 curated links carry a mid-sentence text fragment where a `PS.1.1`-style id
  belongs. Both are recoverable: the first fragment appears verbatim inside task `PS.1.1`'s
  statement, the second inside `PW.8.1`'s. **[measured]** Without `alt_ids` the ceiling is 44/46.
  With it, 46/46.
- **biml**: 8 of 21 curated links carry an unprefixed `category:number` id while the same id means
  something different in the other BIML document. Seven resolve to one document by exact tag-label
  match. **[measured]**
- **wstg / csa_ccm**: neither needs it. CSA CCM's seven retired `IVS-*` ids resolve through the title
  channel already **[measured]**, so the rename map the previous plan carried is dead weight and is
  not built here.

#### The two-pass rule, stated correctly

Version 2 said `alt_ids` "follows `alt_titles`' two-pass rule exactly". The **second** passes match.
The **first** passes do not, and the difference decides what the tests below have to pin.

| | real entries | alternate entries |
|---|---|---|
| `_by_title` | **first writer wins**: `if key and (framework, key) not in self._by_title` | first writer wins, and never displaces a real title |
| `_by_id` | **last writer wins**, unguarded: `self._by_id[(framework, control_id)] = selection` | this task: first writer wins, and never displaces a real id |

**[measured, read out of `tract/text_selection.py` at lines 270-300]** So the second pass carries the
whole guarantee on the id side, and the tests must cover two collisions version 2 never mentioned:
two controls declaring the **same real id** (last one silently wins today) and two controls declaring
the **same alternate id** (first one wins after this task).

Real-id behaviour does not change here. Switching `_by_id` to first-writer-wins would move the join
and invalidate the BEFORE artifact captured one commit ago, which is a different decision with a
different owner. The current corpus carries **zero** real `control_id` collisions among indexed
controls, so the asymmetry is latent rather than live. It carries **82** real title collisions, all
in AIUC-1 Standard 73, NIST AI 100-2 7, EU AI Act 1 and NIST AI RMF 1, where first-writer-wins
silently drops the later control's title. **[measured, this task]** Both facts are pinned below so a
later change to either rule turns a test red.

#### The regression check version 2 shipped was vacuous

Version 2's Step 6 diffed the BEFORE artifact and expected `identical: True`, calling that proof the
new channel is safe. Zero controls in the corpus carry `alt_ids` today **[measured]**, so that check
verifies the metadata-binding move and says nothing at all about the channel. It is kept, because the
binding move is a real code change that could break `alt_titles`, and a second check is added that
exercises the channel end to end on real data: seed the one CWE link that does not resolve today as
an `alt_id` and watch cwe go 612/613 to 613/613. That link is `section_id == "937"`. Control 937 is
in the corpus, an obsolete CWE **category** whose 85-character description does not exceed its
76-character title by `PROSE_MIN_EXTRA_CHARS`, so `ProseIndex` never indexes it and no channel can
reach it. **[measured, this task]**

#### ISO's 92 of 94 becomes an assertion

`92/94` appears nowhere in any test. It lives in a comment at `tests/test_prose_reachability.py:52`,
and `iso_27001` is deliberately absent from that file's `PARSER_BACKED_WITH_LINKS` set because its
prose is restricted. **[measured]** So the one framework whose join this task could break has no
guard. This task adds one, gated on corpus content rather than on file existence per Rule 7.

**Files:**
- Modify: `tract/text_selection.py`
- Modify: `tests/test_text_selection.py`
- Modify: `tests/test_corpus_report.py`

**Interfaces:**
- Consumes: `tract.corpus_report.build_corpus_report`, `CORPUS_EVIDENCE_DIR`,
  `FULL_CORPUS_FRAMEWORK_COUNT` from Task 1.
- Produces: `ProseIndex` honouring `metadata["alt_ids"]: list[str] | str`, normalised through
  `normalize_section_id`, never displacing a real `control_id`, and counting both collision kinds.

**Invalidates:**
- Nothing at runtime. No control in the corpus carries `alt_ids` yet **[measured]**, and Step 6
  asserts the report is byte-identical, so training data, evaluation corpora and every published
  artifact are untouched.
- `results/corpus/before.json` and `results/corpus/link_resolution_before.jsonl` are re-verified
  rather than regenerated. If they move, the second pass displaced a real id and the change is wrong.
- `ProseIndex.load`'s log line, which stops meaning "controls with a real id".
- The comment at `tests/test_prose_reachability.py:52`. It stays as documentation and stops being the
  only place the number exists.

- [ ] **Step 1: Write the failing tests**

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
        """The alternate is declared first, in corpus order, on purpose."""
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

    def test_alternate_never_displaces_a_real_id_declared_earlier(self) -> None:
        """Order must not matter. The second pass is what carries this."""
        index = ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "A-2", "title": "Second",
                 "description": self.LONG + " Second."},
                {"control_id": "A-1", "title": "First",
                 "description": self.LONG + " First.",
                 "metadata": {"alt_ids": ["A-2"]}},
            ],
        }])
        hit = index.lookup("Demo", "A-2", None)
        assert hit is not None
        assert hit.text.endswith("Second.")

    def test_two_alternates_claiming_one_key_keep_the_first(self) -> None:
        """Alternates are first-writer-wins among themselves, and counted."""
        index = ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "B-1", "title": "One",
                 "description": self.LONG + " One.",
                 "metadata": {"alt_ids": ["shared"]}},
                {"control_id": "B-2", "title": "Two",
                 "description": self.LONG + " Two.",
                 "metadata": {"alt_ids": ["shared"]}},
            ],
        }])
        hit = index.lookup("Demo", "shared", None)
        assert hit is not None
        assert hit.text.endswith("One.")
        assert index.alternate_id_collisions == 1

    def test_two_real_ids_claiming_one_key_keep_the_last(self) -> None:
        """Real ids are last-writer-wins and unguarded. Pinned, not fixed.

        Changing this would move the join measured in Task 1's BEFORE
        artifact. The current corpus carries zero such collisions among
        indexed controls. [measured]
        """
        index = ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "C-1", "title": "One",
                 "description": self.LONG + " One."},
                {"control_id": "C-1", "title": "Two",
                 "description": self.LONG + " Two."},
            ],
        }])
        hit = index.lookup("Demo", "C-1", None)
        assert hit is not None
        assert hit.text.endswith("Two.")
        assert index.real_id_collisions == 1

    def test_two_real_titles_claiming_one_key_keep_the_first(self) -> None:
        """The other half of the asymmetry, pinned for the same reason.

        82 of these exist in the corpus today: AIUC-1 Standard 73,
        NIST AI 100-2 7, EU AI Act 1, NIST AI RMF 1. [measured]
        """
        index = ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "D-1", "title": "Same title",
                 "description": self.LONG + " One."},
                {"control_id": "D-2", "title": "Same title",
                 "description": self.LONG + " Two."},
            ],
        }])
        hit = index.lookup("Demo", None, "Same title")
        assert hit is not None
        assert hit.text.endswith("One.")

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

    def test_a_bare_string_alternate_is_accepted(self) -> None:
        index = ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "Y-1", "title": "Only",
                 "description": self.LONG + " Only.",
                 "metadata": {"alt_ids": "Y-legacy"}},
            ],
        }])
        assert index.lookup("Demo", "Y-legacy", None) is not None

    def test_an_empty_alternate_is_ignored(self) -> None:
        index = ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "Z-1", "title": "Only",
                 "description": self.LONG + " Only.",
                 "metadata": {"alt_ids": ["", "   "]}},
            ],
        }])
        assert len(index) == 1
        assert index.alternate_id_collisions == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"$PY" -m pytest tests/test_text_selection.py::TestAlternateIds -q
```

Expected: FAIL. `test_alternate_id_resolves` fails on `assert hit is not None`, and the collision
tests fail with `AttributeError: 'ProseIndex' object has no attribute
'alternate_id_collisions'`.

- [ ] **Step 3: Implement**

```python
# tract/text_selection.py — in ProseIndex.__init__, replace the pending list
        pending_alternates: list[tuple[tuple[str, str], TextSelection]] = []
        pending_alternate_ids: list[tuple[tuple[str, str], TextSelection]] = []
        # Two controls can claim one key on either side. Neither case raises,
        # because the corpus is a fact rather than an input this class
        # validates, but neither is silent either: an unreported collision is
        # a control that vanished from the join with no column to see it in.
        self.real_id_collisions = 0
        self.alternate_id_collisions = 0
```

```python
# tract/text_selection.py — in ProseIndex.__init__, replace the control_id block
                # metadata is bound here rather than below, because both kinds
                # of alternate read it and one binding cannot drift from the
                # other. The alt_titles read below is unchanged and uses it.
                metadata = control.get("metadata") or {}

                control_id = normalize_section_id(control.get("control_id"))
                if control_id:
                    # Last writer wins, deliberately unchanged: the join
                    # measured in results/corpus/before.json depends on it.
                    if (framework, control_id) in self._by_id:
                        self.real_id_collisions += 1
                        logger.warning(
                            "Two %s controls claim id %r; the later one wins "
                            "and the earlier is unreachable by id.",
                            framework, control_id,
                        )
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

The existing `metadata = control.get("metadata") or {}` assignment a few lines below, where
`alt_titles` is read, is deleted by this edit. The `alternates = metadata.get("alt_titles") or []`
line beneath it is unchanged and now reads the binding above.

```python
# tract/text_selection.py — after the existing alternates second pass
        # Alternates are applied after every real id is in place, so a real id
        # always wins regardless of corpus order. Among themselves the first
        # writer wins, matching alt_titles, and a loser is counted rather than
        # dropped in silence.
        claimed_by_alternate: set[tuple[str, str]] = set()
        for key_pair, selection in pending_alternate_ids:
            if key_pair in self._by_id:
                if key_pair in claimed_by_alternate:
                    self.alternate_id_collisions += 1
                    logger.warning(
                        "Two %s controls declare alternate id %r; the first "
                        "one keeps the key.", key_pair[0], key_pair[1],
                    )
                continue
            self._by_id[key_pair] = selection
            claimed_by_alternate.add(key_pair)
```

- [ ] **Step 4: Update the load() log line**

```python
# tract/text_selection.py — in ProseIndex.load, replace the logger.info call
        logger.info(
            "Prose index from %s: %d controls by id (real and alternate), "
            "%d by title, %d real id collisions, %d alternate id collisions",
            source.name, len(index._by_id), len(index._by_title),
            index.real_id_collisions, index.alternate_id_collisions,
        )
```

- [ ] **Step 5: Prove the channel works end to end on the real corpus**

Version 2 had no test that exercised `alt_ids` against real data, because no control carries the
field. This one seeds it, on a framework that is in the **tracked** corpus so CI can run it. `cwe` is
the right subject: its single unresolved curated link is `section_id == "937"`, CWE-937 having been
withdrawn upstream. **[measured]**

```python
# tests/test_corpus_report.py — append

class TestAlternateIdsAgainstTheRealCorpus:
    """The new channel, exercised on the corpus rather than on a fixture.

    cwe resolves 612 of 613 curated links. The miss is section_id "937", an
    obsolete CWE category whose description is shorter than its title plus
    PROSE_MIN_EXTRA_CHARS, so ProseIndex never indexes it. [measured]
    Attaching "937" as an alt_id to any indexed CWE control closes the gap,
    which is a property of the channel rather than of that control.
    """

    def _cwe_only(self, tmp_path: Path) -> tuple[Path, Path]:
        from tract.corpus_report import CURATED_LINKS_PATH, _load_records
        from tract.text_selection import merged_corpus_path

        records = [
            record for record in _load_records(merged_corpus_path())
            if record.get("framework_name") == "CWE"
        ]
        assert len(records) == 1, "expected exactly one CWE record"
        corpus = tmp_path / "cwe.json"
        corpus.write_text(
            json.dumps(
                {"framework_count": 1, "frameworks": records,
                 "generated_date": "2026-01-01",
                 "total_controls": len(records[0]["controls"])},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        rows = [
            line for line in
            CURATED_LINKS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line)["framework_id"] == "cwe"
        ]
        links = tmp_path / "cwe.jsonl"
        links.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return corpus, links

    def test_the_unresolved_cwe_link_stays_unresolved_without_an_alt_id(
        self, tmp_path: Path,
    ) -> None:
        corpus, links = self._cwe_only(tmp_path)
        row = build_corpus_report(links, corpus).by_id("cwe")
        assert row.links == 613
        assert row.unresolved == 1
        assert row.by_title + row.by_id == 612

    def test_an_alt_id_closes_it(self, tmp_path: Path) -> None:
        corpus, links = self._cwe_only(tmp_path)
        data = json.loads(corpus.read_text(encoding="utf-8"))
        controls = data["frameworks"][0]["controls"]
        target = next(
            c for c in controls
            if str(c.get("full_text") or "").strip()
            or len(str(c.get("description") or "").strip())
            > len(str(c.get("title") or "").strip()) + 20
        )
        metadata = dict(target.get("metadata") or {})
        metadata["alt_ids"] = ["937"]
        target["metadata"] = metadata
        corpus.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")

        row = build_corpus_report(links, corpus).by_id("cwe")
        assert row.unresolved == 0
        assert row.by_id == 19
        assert row.resolution_rate == pytest.approx(1.0)

    def test_an_alt_id_cannot_take_a_live_cwe_id(self, tmp_path: Path) -> None:
        """The guarantee, on real data: 79 is a real CWE and must not move."""
        corpus, links = self._cwe_only(tmp_path)
        data = json.loads(corpus.read_text(encoding="utf-8"))
        controls = data["frameworks"][0]["controls"]
        real = next(c for c in controls if str(c["control_id"]) == "79")
        other = next(
            c for c in controls
            if str(c["control_id"]) != "79"
            and (str(c.get("full_text") or "").strip()
                 or len(str(c.get("description") or "").strip())
                 > len(str(c.get("title") or "").strip()) + 20)
        )
        metadata = dict(other.get("metadata") or {})
        metadata["alt_ids"] = ["79"]
        other["metadata"] = metadata
        corpus.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")

        from tract.corpus_report import _load_records
        from tract.text_selection import ProseIndex

        index = ProseIndex(_load_records(corpus))
        hit = index.by_id("CWE", "79")
        assert hit is not None
        expected = str(real.get("full_text") or "").strip() or str(
            real["description"]
        ).strip()
        assert hit.text == expected


class TestIsoStillResolves:
    """92 of 94 lived only in a comment. Now it is a gate.

    Skipped as a named group when the licensed overlay is absent, per Rule 7,
    because gating on file existence never skips: the tracked corpus always
    exists and the restricted rows would hard-fail in CI on data that cannot
    legally be there.
    """

    def test_iso_resolves_92_of_94_with_91_distinct_anchors(self) -> None:
        from tract.corpus_report import FULL_CORPUS_FRAMEWORK_COUNT

        report = build_corpus_report()
        if report.corpus_framework_count < FULL_CORPUS_FRAMEWORK_COUNT:
            pytest.skip(
                f"corpus has {report.corpus_framework_count} frameworks "
                f"against {FULL_CORPUS_FRAMEWORK_COUNT} in the full set, so "
                f"the licensed overlay is absent from this checkout and the "
                f"restricted rows cannot be asserted"
            )
        row = report.by_id("iso_27001")
        assert row.links == 94
        assert row.by_title + row.by_id == 92
        assert row.distinct_anchors == 91
        assert row.dropped_by_prose_rule == 2
```

- [ ] **Step 6: Run the tests and the typecheck**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"$PY" -m pytest tests/test_text_selection.py tests/test_corpus_report.py \
    tests/test_prose_reachability.py -q
"$PY" -m mypy tract/text_selection.py tract/corpus_report.py --strict
```

Expected: PASS. `tests/test_prose_reachability.py` measures the same join and must not move.

- [ ] **Step 7: Confirm the BEFORE state is unchanged**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" scripts/corpus_report.py --out /tmp/after_alt_ids.json
"$PY" - <<'PYEOF'
import json
from pathlib import Path

before = json.loads(
    Path("results/corpus/before.json").read_text(encoding="utf-8")
)
after = json.loads(Path("/tmp/after_alt_ids.json").read_text(encoding="utf-8"))
print("per_framework identical:", before["per_framework"] == after["per_framework"])
print("totals identical:", before["totals"] == after["totals"])
print("corpus sha identical:", before["corpus_sha256"] == after["corpus_sha256"])
assert before["per_framework"] == after["per_framework"]
assert before["totals"] == after["totals"]
PYEOF
```

Expected: three `True` lines. No control in the corpus carries `alt_ids` yet **[measured]**, so
adding the channel must move nothing. A difference here means the second pass displaced a real id,
and the guarantee this task exists to provide is broken.

This check alone is not proof the channel works. It verifies the metadata-binding move and the second
pass. Step 5 is what proves the channel, by seeding a real `alt_id` on a real framework and watching
`cwe` go 612 to 613.

- [ ] **Step 8: Commit**

```bash
git add tract/text_selection.py tests/test_text_selection.py \
        tests/test_corpus_report.py
git status --porcelain --short
git commit -m "feat: resolve a link through a control's retired identifiers"
```

---
