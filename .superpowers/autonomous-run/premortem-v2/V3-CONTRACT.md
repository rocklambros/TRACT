# Plan v3 — the interface contract every author must honour

Set by the orchestrator after adjudicating the premortem. Parallel authors own disjoint task
ranges and CANNOT negotiate these. Copy the exact names and types.

## Rule 0 — what you must NOT touch

The premortem independently reproduced the parser bodies in Tasks 3-13 and found them **correct**:
ENISA 68/68 end to end, csa_ccm 207+17=224, DSOMM 194 activities / 183 uuids, WSTG 109/118 with all
four bogus ids confirmed absent, ETSI 16 section ids → 14 clause anchors, and every parser's
`framework_name` bridging correctly to `canonical_framework(standard_name)`.

**Do not rewrite a parser body.** You are fixing instruments, gates, predictions and ordering.
Where a task's *acceptance prediction* is wrong, change the prediction, not the parser.

## Rule 1 — `FrameworkJoin` gains eight fields; nothing is removed

```python
@dataclass
class FrameworkJoin:
    """One framework's join, on both the link side and the anchor side."""

    framework_id: str
    standard_name: str
    links: int = 0
    by_title: int = 0
    by_id: int = 0
    unresolved: int = 0
    fallback_anchors: int = 0              # NEW distinct section_name anchors the trainer gets
                                           # TODAY for links the index misses. Without this the
                                           # BEFORE state reads 0 and the gain is 3x overstated.
    distinct_anchors: int = 0
    distinct_anchors_pre_truncation: int = 0   # NEW detects two anchors collapsing to one string
                                               # after MAX_ANCHOR_CHARS
    links_per_anchor: float = 0.0
    truncated: int = 0
    nested_anchors: int = 0                # CHANGED: containment, not strict prefix
    contained_anchors: int = 0             # NEW: the old strict-prefix count, kept for continuity
    dropped_by_prose_rule: int = 0
    wrong_anchor_risk: int = 0
    anchor_source_full_text: int = 0       # NEW four text-quality columns. Sum == by_title + by_id.
    anchor_source_description: int = 0
    anchor_source_title: int = 0
    anchor_source_synthetic: int = 0       # parser-synthesised text (csa_ccm domain aggregates,
                                           # WSTG merges, ENISA Annex-C fallbacks)
    distinct_hubs: int = 0                 # NEW hub side, so a later agreement study can sample
    links_per_hub: float = 0.0
    resolution_rate: float = 0.0
```

`CorpusReport` gains one field:

```python
    corpus_framework_count: int = 0        # NEW so a gate can assert the restricted frameworks
                                           # were present when the report was built
```

## Rule 2 — the evidence artifacts are tracked by design, never by `-f`

`.gitignore:3` is `results/`, so `git add results/corpus/before.json` exits 1 and stages nothing
for that path (REPRODUCED: it does stage the other paths on the same command line and returns 1).
Global Constraints forbid `git add -f` and that constraint stands — forcing ignored paths into git
is how licensed text escaped four times.

**Fix: add negations to `.gitignore` immediately after `results/`.**

```
results/*
!results/corpus/
!results/ceiling_study/
```

**CORRECTED 2026-08-19.** My first draft of this rule said `results/` plus `!results/corpus/**`.
That does not work and the Task 16 author caught it. Reproduced on git 2.50.1: an excluded
*directory* is never descended into, so no negation beneath it can rescue a file. `results/`
stages nothing; `results/*` plus `!results/corpus/` stages `results/corpus/a.json`. Use the
second form and verify with `git check-ignore -v` in the same step that writes the artifact.

Then every `git add` of an evidence artifact is an ordinary add. Anchor every path in tests to
`REPO_ROOT`, never a relative path. **Delete `pytest.skip("no BEFORE artifact in this checkout")`** —
a missing baseline is a failure, not a pass (Ruling R3).

## Rule 3 — three licence tiers (Ruling R4/R5/R6)

In `tract/config.py`:

```python
RESTRICTED_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({"etsi", "iso_27001"})
# Reproduction permitted, but on terms a CC0 grant cannot carry. Text routes to the gitignored
# overlay exactly as RESTRICTED does; ASSIGNMENTS stay tracked and published, because a mapping is
# a fact about two documents rather than a reproduction of either. Training reads the overlay, so
# this costs zero anchors.
CONDITIONAL_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({
    "dsomm",                    # GPL-3.0-only
    "biml",                     # CC-BY-SA-3.0 AND CC-BY-SA-4.0
    "samm", "wstg",             # CC-BY-SA-4.0
    "owasp_top10_2021", "owasp_proactive_controls",   # CC-BY-SA-4.0
    "csa_ccm",                  # all rights reserved, no redistribution — see Ruling R5
})
OVERLAY_FRAMEWORK_IDS: Final[frozenset[str]] = RESTRICTED_FRAMEWORK_IDS | CONDITIONAL_FRAMEWORK_IDS
```

Everything that today branches on `RESTRICTED_FRAMEWORK_IDS` for *text routing* branches on
`OVERLAY_FRAMEWORK_IDS` instead. `RESTRICTED_FRAMEWORK_IDS` keeps its current meaning everywhere
else (the fingerprint gate, the "must never appear in git at all" rule).

`csa_aicm` is tracked today with real prose under the identical CSA notice. It is covered by no
ruling. Do not change it in this plan; record it in "What this plan does not close".

## Rule 4 — the `invalidates` column is mandatory (ledger lesson 6, fourth recurrence)

Every task gets an `**Invalidates:**` line naming every artifact it makes stale. At minimum:

| task | invalidates |
|---|---|
| 14 (training links) | `hub_links_training.jsonl` consumers; the ceiling study's pool mirror in `tract/ceiling_study.py` |
| 15 (corpus rebuild) | `data/processed/stopwords.json` (13 consumers), `all_controls.json`, every `data_hash` recorded before it |

**Task 15 MUST regenerate `scripts/build_stopwords.py` output and commit it.** `stopwords.json` is
derived from the corpus, committed, applied to every control and hub text, and hashed into run
metadata by `tract/training/orchestrate.py`. The rebuild adds ~2,300 controls of new prose; without
regeneration every post-rebuild metric uses a stopword list built for a corpus that no longer exists.

## Rule 5 — numbers that are wrong in the plan

| plan says | truth | source |
|---|---|---|
| `+452 distinct anchors` | **+153** (1,749 → 1,902). The eleven's 734 links already land on **299** fallback anchors today | orchestrator measured: dsomm 18, wstg 59, nist_ssdf 44, enisa 33, samm 30, csa_ccm 29, nist_800_63 25, biml 17, etsi 24, proactive 10, top10 10 |
| training links `4,127 → 4,402` | **4,401**. A fourth link falls under the 10-char floor: `nist_800_63` `section_name == 'are g'` (5 chars) | Data Scientist, enumerated over 16 candidates |
| `dropped_by_prose_rule` total 522 | **558**. The total sums only frameworks carrying curated links, so NIST AI RMF 25, AIUC-1 10, CoSAI 1 are invisible | Data Scientist |
| csa_ccm `by_title == 7` | **~26 of 29.** 15 links target a bare domain code whose `section_name` is a descriptive domain title, and `lookup` tries title first | orchestrator measured the curated links directly |
| owasp_top10_2021 `by_title == 0` | **11 of 17.** 7 of 10 names match the source H1 exactly; only A01/A09/A10 diverge | ML Engineer |
| biml `distinct_anchors == 20` | **19.** `inference:9` appears both prefixed and unprefixed and `UNPREFIXED_IDS` routes both to the same control. 21 links / 19 anchors = 1.105 | ML Engineer |
| nist_ssdf premise check: 47 cells, columns [3], 0 mid-sentence | the snippet as written scans ALL columns and yields **54 / [3, 4] / 14**. Filtering to column 3 gives the stated 47 | ML Engineer, ran the snippet verbatim |

Every corrected number carries `[measured]` and names who measured it.

## Rule 6 — a gate that cannot fail is a defect (ledger lesson 9, new)

For every assertion, compute the attainable range in BOTH directions. Reject any assertion that
passes by construction. Known offenders to fix:

- `assert floor <= 1.0` — tautological against literals three lines above.
- `assert report.by_id(f).wrong_anchor_risk == 0` — the counter increments only in the `title`
  branch, and nine of eleven are engineered to resolve by id, so the maximum attainable value is 0.
- `assert BaseParser.honest_prose_fraction(controls) > 0.0` — ONE prose control in csa_ccm's 224
  gives 0.0045 and PASSES. Compare against the parser's declared `min_prose_fraction` instead.
- The channel-parity test in Task 1 builds `ProseIndex(data if isinstance(data, list) else [])`.
  **Both corpus files are dicts** with keys `[framework_count, frameworks, generated_date,
  total_controls]`, so the index is built from `[]` and all 4,405 assertions are `True == True`.
  Use the dict-aware loader (`ProseIndex.load()` / `corpus_report._load_records`).

`JOIN_FLOORS` must be **committed in Task 1, before any parser exists**, so the criterion cannot
move in the same commit as the result it gates. The plan file itself is gitignored (`.gitignore:25`),
so a floor edited mid-run leaves no diff — the floors must live in tracked code.

## Rule 7 — CI cannot see the overlay, and that must fail loudly rather than red-permanently

The tracked corpus has **29** frameworks (no etsi, no iso_27001); the overlay has **31**.
`merged_corpus_path()` returns the tracked file whenever the overlay is absent, and the tracked file
always exists — so a `if not merged_corpus_path().exists(): pytest.skip(...)` fixture **never skips**
and four assertions hard-fail in CI on data that cannot legally be there.

Gate on content, not on existence:

```python
report = build_corpus_report(...)
if report.corpus_framework_count < len(EXPECTED_FRAMEWORKS):
    pytest.skip(
        f"corpus has {report.corpus_framework_count} frameworks; the licensed overlay is "
        "absent from this checkout, so the restricted rows cannot be asserted"
    )
```

Restricted-framework rows skip **as a named group with the reason stated**; every other row still
asserts. Never delete or relax a floor to make CI green — that retires the only gate on a parser
nobody can inspect.

## Rule 8 — the rebuild must be reversible and must diff the field that matters

1. **Snapshot before `shutil.copy2`.** `data/processed/frameworks/etsi.json`,
   `iso_27001.json` and `licensed/all_controls.json` are untracked, so `git checkout` cannot
   recover them, and `scripts/fetch_frameworks.py` has **no `iso_27001` entry** — ISO is not
   re-derivable from any scripted path. Copy all three (plus the whole processed dir) to a
   timestamped scratch directory first, and add a `--restore` path.
2. **Use `tract.io.atomic_write_json`**, not `shutil.copy2` — the plan's own Global Constraint.
3. **Pass `audit_dir`** in `run_all`, or a `--dry-run` writes repair audits into the real
   `data/processed/repair_audit/`.
4. **Hash more than `description`.** `ProseIndex` prefers `full_text` unconditionally, so the diff
   as written can re-point every link and report `0 changed`. Hash a tuple of
   `(description, full_text, title, sorted(alt_ids), sorted(alt_titles))`.
5. **Expect ~113 `removed` entries and classify them.** Five parsers change the control_id shape:
   `nist_800_63` `5-1-1-1`→`5.1.1.1`, `wstg` `wstg-appe-d`→`WSTG-APPE-D`,
   `owasp_proactive_controls` `c1`→`C1`, `enisa` `Table 3:`→slug, `csa_ccm` `IVS-*`→`I&S-*`.
   A rename is not a loss. Emit a `renamed` bucket keyed on matching content hash, and make
   `removed` mean "content gone", so the operator's stop rule is actionable.
6. **Turn the prose stop rule into an assertion.** Task 15 Step 6 says "if capec, cwe, asvs …
   appears in that list, stop" — that is prose an autonomous worker reads past. It must be
   `raise SystemExit` on any changed framework outside the eleven. A control whose only enforcement
   is an instruction is decorative (ledger lesson 4).

## Rule 9 — record what a later agreement study will need

Task 1 also writes `results/corpus/link_resolution_<tag>.jsonl`, one row per curated link:

```json
{"framework_id": "...", "section_id": "...", "section_name": "...", "cre_id": "...",
 "link_type": "...", "channel": "title|id|unresolved", "anchor_source": "full_text|description|title|synthetic",
 "anchor_sha256": "...", "anchor_chars": 0, "truncated": false}
```

No anchor text — the file must be safe to track for overlay frameworks too. This is the minimum an
agreement study needs to sample the frameworks this plan re-weights, and the premortem's answer to
"does the plan create that artifact" was **no**.
