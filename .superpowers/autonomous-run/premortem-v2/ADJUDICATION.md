# Premortem v2 — adjudication (round 1, four agents, six lenses)

Artifact: `docs/superpowers/plans/2026-08-18-remaining-parsers-v2.md` (16 tasks, 111 steps, 6,987 lines)
Agents: Data Scientist; ML Engineer; Security Architect + MLOps/SRE; Red Teamer + Governance/Risk.
~38 raw findings in. This file is the merge, the cross-attack, and the verdict.

**Verdict: plan v2 is not executable as written.** Not because the parsers are wrong — they were
independently reproduced and are largely correct — but because the *instrument* is wrong in three
ways, three acceptance gates halt a healthy run, and the headline metric is 3x overstated.
Remediation is surgical, not a rewrite: Tasks 3-13's parser bodies survive nearly intact.

---

## Verified by me, against source, not accepted on report

| # | claim | verdict |
|---|---|---|
| V1 | Corpus JSON is a **dict**, so Task 1's channel-parity test builds `ProseIndex([])` and all 4,405 assertions reduce to `True == True` | **CONFIRMED** — both files have keys `[framework_count, frameworks, generated_date, total_controls]` |
| V2 | The eleven frameworks' 734 links already land on **299** distinct fallback anchors today | **CONFIRMED to the digit** — dsomm 18, wstg 59, nist_ssdf 44, enisa 33, samm 30, csa_ccm 29, nist_800_63 25, biml 17, etsi 24, proactive 10, top10 10 |
| V3 | ETSI's `CLAUSE` regex captures the running page header for clauses 5, 6, 7 | **CONFIRMED** — all three headings read `'ETSI GR SAI 005 V1.1.1 (2021-03)'`; 25 distinct numbers still match so `expected_count` passes |
| V4 | 15 of 29 `csa_ccm` links target a bare domain code whose `section_name` is a descriptive title | **CONFIRMED** — plan asserts `by_title == 7`; `'IPY' \| 'Interoperability and portability policy and procedures'` is control IPY-01's title, not the IPY domain's |
| V5 | `pre_rebuild_control_hashes.json` is digests only | **CONFIRMED** — 4,222 entries, every value 64-hex, zero recoverable text. It is a detector, not a rollback artifact |
| V6 | Licence *class* is unmodelled | **CONFIRMED** — `dsomm=GPL-3.0-only`; biml/samm/wstg/top10/proactive = CC-BY-SA; `csa_ccm` = "all rights reserved... no redistribution". Plan mentions `CC-BY-SA` and `GPL-3.0` **zero** times in 6,987 lines |
| V7 | `invalidates` column absent; stopwords goes stale | **CONFIRMED** — `invalidates`/`build_stopwords`/`stopwords` all grep to **0** in the plan; stopwords has 13 consumers incl. `training/data.py`, `firewall.py`, `orchestrate.py` |
| V8 | Tracked corpus has 29 frameworks, overlay 31; CI acceptance suite hard-fails on etsi/iso | **CONFIRMED** — etsi and iso_27001 are the only two untracked per-framework files |
| V9 | `honest_prose_fraction(...) > 0.0` is vacuous | **CONFIRMED** — it returns a ratio; 1 prose control in csa_ccm's 224 = 0.0045 > 0 PASSES |

## Closed by measurement during adjudication (findings retired, not deferred)

**C-A. CAPEC/CWE were never test-rebuilt (Sec M3, Gov #3 residual) — CLOSED.**
Installed the already-declared `defusedxml==0.7.1`, imported both parsers, ran them into scratch:
`capec 558/558 match, 0 mismatch` · `cwe 1331/1331 match, 0 mismatch`. defusedxml rejects neither XML.
Pre-measured rebuild coverage 45% → 89.7%.

**C-B. The 250-item ceiling study is at risk from the rebuild (Gov #3) — CLOSED, and this one
mattered most.** Three pieces from three sources combine: the Data Scientist measured that **zero**
of the 250 ceiling items fall in the eleven frameworks and the validation roster moves only 1.6%
(MDE 0.0400 → 0.0397); Governance measured the residual as capec+cwe = **111 of 250 items**; and
C-A above proves capec+cwe reproduce byte-identically. The owner's 250 irreplaceable judgments are
safe. No agent could reach this alone.

**C-C. openpyxl parses the registration-walled CCM workbook unhardened (Sec S3) — CLOSED.**
Same install flipped `openpyxl DEFUSEDXML: False → True`.

---

## Cross-attack log (where agents corrected or completed each other)

**X1 — `git add` on an ignored path. Two agents wrong on mechanism, right on consequence.**
Sec M2 and Gov #2 both asserted git add is atomic so Task 1's whole commit is empty. I reproduced it:
`git add real.py results/corpus/before.json` → **exit 1, `real.py` staged**. Git stages the legal
paths and refuses only the ignored one. So the instrument IS committed; the BEFORE artifact is not,
and `pytest.skip("no BEFORE artifact")` then reports green forever on the 20 untouched frameworks.
**Gov #2's proposed fix — `git add -f` — is rejected.** Global Constraints forbid it and forcing
ignored paths into git is precisely how licensed text escaped before. Correct fix: the corpus
reports are evidence, not results; write them to a tracked-by-design path, anchor to `REPO_ROOT`,
and delete the skip.

**X2 — `wrong_anchor_risk`: three agents said it cannot fire, a fourth found it firing.**
Red Team #1, DS #7 and Sec M5 independently established the counter increments only in the `title`
branch, and 9 of 11 frameworks are engineered to resolve by id — so `== 0` is unfailable. Red Team
bounded the residual honestly (~20 hand-declared entries, not 723 links). Then ML Eng #3 measured an
actual wrong anchor: csa_ccm `IPY`. Merged verdict, which no single agent held: **the column is
simultaneously blind on nine frameworks and will halt the run on the one where it fires.**

**X3 — the anchor-gain finding survives its own best counterargument in a stronger form.**
DS #1's counter is that the plan does argue text quality over count for enisa/proactive/top10. True.
But `distinct_anchors` is named "the report's load-bearing column", +452 is what enters the ledger,
and there is no text-quality column at all. The finding converts from "the number is wrong" to
"**the instrument gates on the wrong column**", which is worse and is the actual defect.

**X4 — Red Team #5 (the panel cannot audit the corpus it reads) partly self-refuted.**
Its cheap version is dead: the agent measured CAPEC ceiling items at median 554 chars with only 2 at
the cap and 0 duplicate texts, while owasp_ai_exchange has 25 of 54 at the cap and scores 0.981.
Truncation and collapse explain nothing. Hypothesis (c) — shared corpus corruption — survives only
as a cheap probe to run, not as a blocker. Demoted to the residual list with a concrete test.

---

## Surviving findings, ranked by expected cost reduction

### P0 — the instrument is wrong; everything downstream is measured against it (Task 1)
1. **Channel-parity test is vacuous** (V1). The plan's named guard for "the exact defect that got the
   previous plan rejected" asserts nothing. Fix: use `ProseIndex.load()` / the dict-aware loader.
2. **Load-bearing column is wrong** (V2). Add a `fallback_anchors` column and record
   `TextSelection.source` per link. Report the honest delta (+153) and a text-quality column beside it.
   Ledger the corrected figure; the +452 must never enter the run record.
3. **Diff is blind to the anchor field** (ML Eng #2). `diff_against_baseline` hashes `description`
   only, while `ProseIndex` prefers `full_text` unconditionally — set by wstg, top10, proactive,
   nist_ssdf and by `_sanitize_control` over 2,000 chars. Hash `full_text`, `title`, `alt_ids`,
   `alt_titles` too, or the rebuild can re-point every link and report `0 changed`.
4. **`nested_anchors` counts strict prefixes only** (DS #7), so ETSI 5.2 vs 5.2.2 nests undetected;
   and two anchors sharing a 2,150-char prefix truncate to one string, suppressing the column and
   breaking `distinct_anchors == 14` with no diagnostic.

### P0 — gates that halt a healthy run
5. **csa_ccm `by_title == 7`** → measured 26 (V4). **owasp_top10_2021 `by_title == 0`** → 11 of 17.
   **biml `distinct_anchors == 20`** → 19 (`inference:9` appears prefixed and unprefixed).
   The plan orders a hard stop on each. An executor either burns a day or edits the assertion —
   the self-calibrating-gate defect the plan claims to have closed.
6. **CI acceptance suite hard-fails on etsi/iso** (V8). The fixture never skips because the tracked
   corpus always exists. Predictable repair is deleting the ETSI floor, retiring the only gate on a
   restricted parser nobody can inspect. This is Ruling R3 inverted.

### P0 — fabrication that passes every gate
7. **ETSI clauses 5/6/7 capture the page header** (V3); clause 7 carries ~22.6 KB of front matter,
   bibliography and TOC as one control's statement, all three sharing one title so `_by_title`
   collapses them. Inert today (nothing links to bare 5/6/7) and live the moment the semantic
   rebuild reads the full corpus. ETSI is the only text-mode PDF parser with no page-furniture guard,
   and its output is gitignored so no reviewer will ever see it in a diff.

### P1 — destructive and irreversible
8. **No backup before the one-way `shutil.copy2`** (V5). etsi.json, iso_27001.json and
   licensed/all_controls.json are overwritten, untracked, and ISO has no scripted re-fetch path at
   all. ISO is the corpus's only 0.967-prose fold. Trivial fix; unrecoverable if skipped.
   Also violates the plan's own Global Constraint "atomic writes only, via `tract.io.atomic_write_json`",
   and `run_all` never passes `audit_dir`, so a `--dry-run` writes into the real audit directory.
9. **Licence tiering** (V6). See rulings R4/R5/R6 in `round1-security-mlops.md`. Least reversible
   action in the plan: `git push` is the publication event.

### P1 — provenance and ordering
10. **`invalidates` column absent** (V7); `stopwords.json` goes stale across 13 consumers and is
    hashed into run metadata as if current. Lesson 6 recurring a fourth time.
11. **Task 14 makes `hub_links_training.jsonl` a function of the corpus but records only the
    curated-links hash**, and Task 15 then rewrites the corpus. Two runs over different corpora
    report the same `data_hash`. Also `ceiling_study.py`'s `assign_quality_tier` call silently stops
    mirroring training when Task 14 adds a defaulted parameter — the docstring becomes false.
12. **≥113 spurious `removed` entries** from control_id shape changes across 5 parsers
    (`5-1-1-1`→`5.1.1.1`, `wstg-appe-d`→`WSTG-APPE-D`, `c1`→`C1`, enisa slugs, `IVS-*`→`I&S-*`),
    with no rule separating a rename from a lost control. Plus 39 baseline records lost to key
    collision (all in enisa/etsi, all with distinct text) and **63 published rows** pointing at ids
    the rebuild retires.

### P1 — training-data quality, which is what the owner actually cares about
13. **DSOMM: 214 links onto 24 hubs** (8.92 links/hub, second only to CAPEC's 9.27), one hub
    absorbing 54. All `AutomaticallyLinkedTo`/T3, never human-reviewed, agreement unmeasured, and the
    plan multiplies its training weight **6.1x** (35 → 214 unique pairs). Its provenance is *not*
    the CAPEC→CWE→CRE chain CLAUDE.md asserts for auto-links: only 26 of 183 uuids overlap the CREs
    reachable through their declared references, so `AutomaticallyLinkedTo` is ≥2 provenance classes
    collapsed into one tier.
14. **Task 14 restores all 44 contested CAPEC links** while the self-review states "CAPEC and CWE are
    untouched". Given α₁ = 0.181 for CAPEC, recovering its terse links (`'UDP Ping'`, `'Fuzzing'`) is
    not self-evidently progress. Split the commit by framework so the Part 5 weighting decision has a lever.
15. **9 WSTG links reach the trainer with a literal id as the anchor** — `"WSTG-BUSL-$$"`,
    `"WSTG-INPV-00"`, `"WSTG-APPE-D"`, `"WSTG-INFO-##"` — because Task 14's new gate falls back to
    `section_name`, the exact field its commit message says it is moving away from, and those names
    clear the 10-char floor.
16. **15 of 29 csa_ccm links anchor on a semicolon-joined list of member titles**, which
    `honest_prose_fraction` counts as prose and no column distinguishes from a normative statement.
    Against CLAUDE.md's standing rule that title fallback is "a last resort, not a default".

### P2 — test and tooling integrity
17. **Task 11's ENISA tests fail against Task 11's implementation** (`rows_to_units` called without
    `banners`). This is v1 Critical C7 recurring at a new task.
18. **Task 9's mandated premise check cannot produce its stated output**: the snippet scans all
    columns, giving 54 cells / columns [3,4] / 14 mid-sentence against the stated 47 / [3] / 0.
    Separately, every `[measured]` PDF number is a function of an unpinned extractor —
    `requirements.txt` pins `pdfplumber==0.11.10`, the mandated interpreter has **0.11.4**.
19. **5 of 11 parsers never reach `parse()` outside a skip** in CI (csa_ccm, nist_ssdf, enisa, biml,
    etsi — the two PDFs, the XLSX and the multi-document pair, i.e. the most fragile extraction).
    Ruling R3's defect class verbatim.
20. **`mypy --strict` fails at Task 8**: openpyxl ships no `py.typed`, `types-openpyxl` is not added,
    no override exists, and the pyproject line the plan specifies lands in `optional-dependencies.llm`
    rather than `dependencies`. Three-way pin drift on openpyxl and defusedxml; CI installs `-e .`
    so it resolves the floor, not the pin.
21. **`COUNT_TOLERANCE = 0.10` lets owasp_proactive_controls ship 9 of 10 silently** (`0.1 <= 0.10`),
    and its 10 controls carry 7.6 links each. Only 2 of 11 parsers have a completeness check that
    beats the band.
22. **Task 16's floors live in a gitignored file** (`.gitignore:25`), so a floor edited down mid-run
    leaves no diff, and `JOIN_FLOORS` lands in the same commit as the report it gates. Criterion and
    PASS can land in *zero* commits — stricter than the recorded `gate-preregistration-is-retrospective`.
23. **Repair audits store `statement_lengths`, not the before/after text** the base-class docstring
    promises a reviewer needs, and `data/processed/repair_audit/` is gitignored so no reviewer on
    another machine and no CI job can read it.
24. **`dropped_by_prose_rule` reads 522 where the corpus holds 558** un-indexed controls
    (NIST AI RMF 25, AIUC-1 10, CoSAI 1 are invisible). **`4,402` should be `4,401`** — a fourth link
    (`nist_800_63` `'are g'`, 5 chars) falls under the 10-char floor. That wrong number is hard-coded
    into a commit message and copied into the ledger.

---

## Residual risk, carried not fixed

- **Source content integrity.** Six upstream sources accept community PRs; SHA pins prove bytes did
  not change in transit and say nothing about authenticity. `nist_800_63` is deliberately unpinned
  (Cloudflare bot token) and Task 14 promotes it from 0 to 79 training links. `--accept-new-hash` is
  an alert with **no stated adjudication rule** — an alert nobody knows how to answer gets approved.
  A grep of the plan returns 0 for `malicious`, `tamper`, `supply chain`, `untrusted`, `quarantine`.
- **Hypothesis (c) for the CAPEC panel result** (Red Team #5, demoted per X4): both the human and all
  five models read the same corpus text, so shared corruption converges both readings while OpenCRE
  looks like the outlier. Cheap probe: re-render a sample of contested items from raw source and
  re-ask two panel members. Worth doing before the CAPEC finding reweights 42.8% of training.
- **The false "human-reviewed" claim is in the generator, not the artifact.** `dataset/card.py:98,137`
  and `bundle.py:216-219` regenerate "each individually reviewed by a cybersecurity domain expert" on
  every publish; the erratum lives only in `README.md`. The next publish overwrites the correction and
  destroys the `#erratum-2026-08-15` anchor the README links to. Held only by the standing
  republication ban. **Fix the generators before any publish, not after.**

---

## Post-adjudication verifications by the orchestrator

- **V10. The 63 dangling published rows are exact.** `build/dataset/crosswalk_v1.0.jsonl` holds
  5,238 rows; **56** carry an `enisa:...Table 3:/Table 5:` control_id and **7** carry a retired
  `csa_ccm:...IVS-0*` id. Both id shapes are dissolved by the rebuild. Governance finding #8 stands
  as measured.
- **V11. The erratum anchor was live-broken.** `README.md:48` links to
  `#erratum-2026-08-15` on the model card; `tract/publish/model_card.py` contained **no** erratum
  text, so the next `publish-hf` would have regenerated the card without it. Fixed at `a82680b`
  with four tests, one of which pins the exact heading string that produces the anchor.
- **V12. The false review claim was in the generator, not the artifact.** `dataset/card.py` headline
  called the whole 5,238-row crosswalk human-reviewed. Fixed at `5fa2c75`; all four new tests were
  verified to FAIL against the old wording before being accepted, so they are not vacuous
  (ledger lesson 9 applied to my own fix).

---

## Ruling R7 — the tracked ceiling study stays tracked; the licence record is the defect

The Task 16 author flagged `results/ceiling_study/` as a possible fifth licensed-text channel.
Measured: `ceiling_items.json` IS tracked (24 files under that directory are) and carries 250
`control_text` fields, median 764 characters.

Provenance of those 250 items and their recorded licence:

| framework | items | FRAMEWORK_LICENSES |
|---|---|---|
| capec | 83 | UNDETERMINED |
| owasp_ai_exchange | 54 | UNDETERMINED |
| mitre_atlas | 43 | Apache-2.0 |
| cwe | 28 | UNDETERMINED |
| nist_ai_100_2 | 22 | UNDETERMINED |
| nist_800_53 | 14 | UNDETERMINED |
| owasp_llm_top10 | 6 | CC-BY-SA-4.0 |

**The good news is real and worth stating: zero items come from a RESTRICTED framework.** No ETSI,
no ISO 27001. The fingerprint gate did its job on the channel it was built for.

**Ruling: keep it tracked.** The 250 owner annotations key on `item_index` alone, so removing the
items file makes the single most expensive asset in the project unreproducible and unauditable.
That cost is certain; the licence risk is not. CAPEC and CWE are MITRE works, NIST 800-53 and
NIST AI 100-2 are US Government works not subject to domestic copyright, and MITRE ATLAS is
Apache-2.0. The 6 CC-BY-SA-4.0 items are attributable under NOTICE.

**What IS a defect, and it is not the tracking:** 201 of 250 items come from frameworks whose
licence nobody ever determined. `UNDETERMINED` is a record of absent work, not a finding of
permissiveness, and this run has now twice reasoned about exposure using a field that five of
seven relevant frameworks do not populate. Determining those five is a task, not a footnote.
Cost if this ruling is wrong: 6 CC-BY-SA items sit in a CC0 file, attributable and removable in one
commit, against a study that would otherwise be unreproducible.
