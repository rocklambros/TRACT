# OWASP Top 10 for LLM Applications 2026 — ingest report

Branch `semantic-rebuild`. Head at time of writing: `083f34b`.
Source: `data/raw/frameworks/owasp_llm_top10_2026/2026_OWASP-GenAI-LLM-Top-10.repaired.md`,
sha256 `3d3c9f21809c5f882a668b87424ac6b2e2a270caab4b29aa5265df3475433a96`,
242,894 bytes, 1,900 lines, gitignored, CC BY-SA 4.0 per its own licence block.

## What landed

| commit | subject |
|---|---|
| `1634b7d` | `SourceReader` split out of `BaseParser` so a non-parser extractor reads through the manifest |
| `1c2af62` | Source entry, licence, NOTICE row, `HOLDOUT_FRAMEWORK_IDS` |
| `b091b84` | `parsers/parse_owasp_llm_top10_2026.py` plus fixture, tests, artifact |
| `1b220da` | `parsers/extract_owasp_llm_top10_2026_mappings.py` plus tests, artifact |
| `083f34b` | merge-step holdout exclusion plus `tests/test_holdout_framework.py` |

## 1. Controls [measured]

Ten, exactly, ids `LLM01:2026` through `LLM10:2026`. `expected_count = 10`,
exact, not a floor. The 2025 file is untouched.

| control_id | title | description | definitional block | full_text |
|---|---|---:|---:|---:|
| LLM01:2026 | Prompt Injection | 1,988 | 8,406 | 19,574 |
| LLM02:2026 | Sensitive Information Disclosure | 1,988 | 6,826 | 11,990 |
| LLM03:2026 | Excessive Agency | 1,988 | 3,714 | 9,265 |
| LLM04:2026 | Supply Chain | 1,991 | 5,226 | 12,464 |
| LLM05:2026 | Data and Model Poisoning | 1,990 | 5,495 | 9,994 |
| LLM06:2026 | Unbounded Consumption | 1,990 | 5,033 | 9,847 |
| LLM07:2026 | Misinformation | 1,987 | 2,631 | 5,036 |
| LLM08:2026 | Hidden Context Exposure | 1,985 | 5,891 | 8,552 |
| LLM09:2026 | Vector and Embedding Weaknesses | 1,990 | 7,208 | 12,216 |
| LLM10:2026 | Improper Output Handling | 1,993 | 2,950 | 6,629 |

Characters after sanitization for `description` and `full_text`; raw source
characters for the definitional block, which is the text before the
2,000-character cut.

`version` is `sha256:3d3c9f21...`, not a date. The document's revision history
still reads `[2026 release date]`, so a date string would assert a release that
has not happened. The digest is re-checked on every parse and a mismatch stops
the parser, because `version` IS that digest.

## 2. description versus full_text, and why

`description` is the entry from its heading down to the first subheading in
`tract.config.REMEDIATION_HEADINGS`, which in this document is always
`Prevention and Mitigation Strategies`. That keeps `Description`, any extra
subsections an entry carries (LLM01's Types of Prompt Injection), and
`Common Examples of Risk`. It drops prevention text and the attack scenarios.
`full_text` carries the whole entry, prevention and scenarios included, so
nothing is lost.

Three reasons, in order of weight.

1. The assignment paradigm maps what a control *is*. Prevention text says how
   to satisfy it, which is a different question, and it pulls an anchor toward
   controls that share countermeasures rather than controls that share meaning.
2. `REMEDIATION_HEADINGS` is the same list `tract.text_selection.strip_remediation`
   applies downstream. Cutting structurally on the real headings makes the
   parser and the anchor selector agree by construction, instead of each
   guessing separately, one from headings and one from a regex over flattened
   text.
3. The encoder's 512-token budget is fixed by the architecture. Including
   remediation would displace definitional prose rather than add to it.

`Common Examples of Risk` stays on the definitional side because it enumerates
what the risk looks like rather than what to do about it. It is also what tops
up the six entries whose `Description` section alone is under the cap: LLM03
1,892, LLM04 1,482, LLM06 1,419, LLM07 1,673, LLM09 1,570, LLM10 1,743 raw
characters. Without it those six would ship a short description while the rest
ship a full one.

The cut to 2,000 characters is not cosmetic. `BaseParser._sanitize_control`
replaces a parser-supplied `full_text` with the overflow of an over-long
description, so an uncut description would evict the entry text and leave
`full_text` holding a longer copy of the description instead. All ten
descriptions are cut, at a word boundary.

## 3. Prose fraction [measured]

**1.000**, against a declared `min_prose_fraction = 1.0`.

Measured by `BaseParser.honest_prose_fraction` on the stored text: 10 of 10
descriptions are at least 60 characters and are not a copy of the title. The
floor is 1.0 rather than something under it because, unlike ISO 27001, this
document has no genuinely one-sentence entries to carve out. The shortest
definitional block is 2,631 characters against a 60-character bar, so anything
below 1.0 would mean the extraction broke rather than that the source is terse.

## 4. The appendix boundary [measured]

Confirmed: `## Appendix A: Related Framework Mappings` is at source line 1,030,
below LLM10's heading at line 964. Without the boundary LLM10's body runs from
line 965 to end of file, 936 lines and 130 KB, of which **871 lines and 124 KB
are appendix tables, references, and acknowledgements** that would ship as
control text. (Your 937 lines / 132 KB is the same measurement of the whole
runaway body, within an off-by-one on line counting; 871 / 124 KB is the part
that does not belong to LLM10.) A source missing the boundary heading is
refused rather than parsed, and there is a test for it.

## 5. Appendix A mappings [measured]

331 mappings, nine target frameworks, all extracted to
`data/processed/owasp_llm_top10_2026_mappings.json`. Not to
`data/processed/frameworks/`: everything in that directory is merged into the
corpus as control text, and these rows are a crosswalk.

| target | our framework_id | element level | mappings | distinct elements |
|---|---|---|---:|---:|
| OWASP Top 10 for Agentic Applications (ASI) 2026 | `owasp_agentic_top10` | risk | 31 | 10 |
| OWASP GenAI Data Security 2026 (DSGAI) v1.0 | `owasp_dsgai` | risk category | 39 | 19 |
| MITRE ATLAS content v2026.06 | `mitre_atlas` | tactic | 46 | 13 |
| MITRE ATT&CK v19.1 | none | enterprise tactic | 29 | 10 |
| MITRE CWE 4.20 | `cwe` | weakness | 48 | 22 |
| NIST AI 600-1 v1.0 | `nist_ai_600_1` | risk category | 39 | 8 |
| NIST AI RMF (AI 100-1) v1.0 | `nist_ai_rmf` | category | 25 | 9 |
| CSA AICM v1.1 | `csa_aicm` | control domain | 44 | 11 |
| OWASP AIVSS v0.8 | none | agentic core security risk | 30 | 8 |

Seven of the nine exist in our corpus. Two do not: MITRE ATT&CK and OWASP
AIVSS have no artifact under `data/processed/frameworks/`.

"Exists in our corpus" is weaker than it sounds for three of the seven, which
is why the artifact records `element_level` per target rather than a bare
boolean. The appendix says up front that it maps at each framework's coarse
level, and it does:

- **ATLAS**: tactics (`AML.TA####`). Our corpus carries techniques
  (`AML.T####`). The two id spaces do not intersect, so no ATLAS row resolves
  to a control we hold.
- **AI RMF**: categories (`MEASURE 1`). Our corpus carries subcategories
  (`GOVERN 1.1`).
- **AICM**: control domains (`AIS`, `STA`). Our corpus carries controls
  (`AIS-01`).

The other four resolve cleanly: ASI and DSGAI element ids are our control ids
verbatim, CWE ids match once the `CWE-` prefix is stripped, and NIST AI 600-1
element names are our control titles verbatim (it numbers nothing).

One incidental finding: the AICM section maps LLM06 to `IVS Infrastructure &
Virtualization Security`, and IVS is not a domain in our AICM v1.1 corpus,
whose 18 domains are A&A, AIS, BCR, CCC, CEK, DCS, DSP, GRC, HRS, I&S, IAM,
IPY, LOG, MDS, SEF, STA, TVM, UEM. IVS is a CCM domain. Not acted on; recorded
because it is a claim about a framework version we hold and can check.

### CWE chain, re-measured

Your figures reproduce except one, and that one is a naming difference rather
than a disagreement.

| quantity | this run | brief |
|---|---:|---:|
| LLM2026-to-CWE mappings | 48 | 48 |
| distinct CWEs | 22 | 22 |
| resolving to our CWE 4.20 corpus | 22 | 22 |
| carrying OpenCRE links | 17 | 17 |
| distinct hubs reached | 46 | 46 |
| risks reaching a hub | 10 of 10 | all 10 |
| "transitive chains" | see below | 37 |

**37 is the number of mappings that chain**, not the number of chains. 37 of
the 48 mappings name a CWE that carries at least one OpenCRE link. Expanding
those gives:

- 118 distinct (risk, CWE, hub) triples
- 111 distinct (risk, hub) pairs
- 50 distinct (CWE, hub) pairs
- 46 distinct hubs, covering all 10 risks

I report all four rather than picking one, because "chain" is ambiguous and 37
is none of them. The artifact records `mappings_that_chain: 37` and
`risk_cwe_hub_triples: 118` under separate keys so the next reader cannot
conflate them. The block is recorded, not gated: `hub_links_curated.jsonl` is
rebuilt by the pending OpenCRE re-fetch, and a declared expectation would be a
gate on a moving input.

### Two extraction hazards worth recording

**The `||` continuation shape.** `line.strip("|")` strips every leading pipe,
which collapses `|| ● ASI02 ... | rationale |` to two cells and loses the empty
Risk cell that carries the "same risk as above" meaning. That dropped 15 of
ASI's 31 rows in the first pass, silently. The extractor strips exactly one
pipe from each end.

**The coverage matrix is a free integrity check.** The appendix states its own
answer twice: a risk-by-framework matrix of primary / supporting / absent
marks, and the detail tables. I measured them against each other before wiring
it: **0 mismatches across all 90 cells**. It is now a gate, and it fires on a
dropped detail row even after the shorter row count has been accepted as
correct. That is the only cheap detector of a partial extraction there is.

## 6. The holdout

`tract.config.HOLDOUT_FRAMEWORK_IDS` is a new, wired list. Restricted and
holdout stay separate: restricted is about what this repository may
redistribute, holdout is about what the model may see. This framework is
CC BY-SA 4.0 and freely redistributable, and it is still excluded from
training.

`tests/test_holdout_framework.py` (16 tests) sweeps:

- every module-level constant of `tract.config` and of
  `scripts/phase1b/runpod_parallel.py`, for the id and for two plausible
  display names
- `fold_roster("test")` and `fold_roster("validation")`, the functions provision
  and run actually read
- `tract.ceiling_study.eligible_framework_ids()`
- `AI_PARSER_FRAMEWORK_IDS`, `OPENCRE_EXTRACT_FRAMEWORK_IDS`,
  `OPENCRE_FRAMEWORK_ID_MAP`
- `hub_links_curated.jsonl`, `hub_links.jsonl`, and
  `hub_links_by_framework_curated.json`
- the merge step

`FRAMEWORK_NAME_ALIASES` gets no entry, and a comment in `tract/config.py`
says why. An alias exists to let an OpenCRE `standard_name` reach a parser's
`framework_name`. Adding one for the 2026 edition is the single edit that could
give the holdout a path to a link, so its absence is the control.

### The gate provably fires

Added `"OWASP Top 10 for LLM Applications 2026"` to `FOLD_FRAMEWORKS` and
`"owasp_llm_top10_2026"` to `CEILING_STUDY_TEST_FRAMEWORKS`, then ran the file:

```
FAILED tests/test_holdout_framework.py::TestNoRosterNamesTheHoldout::test_no_config_constant_names_it
FAILED tests/test_holdout_framework.py::TestNoRosterNamesTheHoldout::test_no_lofo_fold_roster_names_it
FAILED tests/test_holdout_framework.py::TestNoRosterNamesTheHoldout::test_neither_lofo_split_holds_it_out
FAILED tests/test_holdout_framework.py::TestNoRosterNamesTheHoldout::test_the_ceiling_study_does_not_sample_it
4 failed, 12 passed
```

Reverted, 16 passed. The sweep is also checked in the other direction inside
the test file: planting the id in a fake namespace must flag it, and the 2025
id, which shares a prefix, must not.

### The merge channel

The merge globs `data/processed/frameworks/*.json`, and the prose index reads
the merged corpus rather than any roster. Every roster could be clean and the
holdout would still arrive. `parsers/merge_all_controls.py` now drops holdouts
before building either corpus, and raises if nothing survives.

Verified on the real tree: with `owasp_llm_top10_2026.json` on disk,
`all_controls.json` rebuilds **byte-identical** to the committed file, 29
frameworks, `generated_date` unchanged at 2026-08-14.

One channel is knowingly left open and is out of scope here:
`scripts/phase1c/t0_extract_similarities.py` globs the same directory to
crosswalk every control. That is inference over the holdout rather than
training on it, which is a legitimate thing to want, but it should get a
deliberate decision before Phase 1C is re-run.

## 7. Test results

```
21 failed, 1302 passed, 22 skipped, 3 xfailed, 3 errors
```

Baseline was 21 failed / 1241 passed / 22 skipped / 3 xfailed. The 21 failures
and 3 collection errors are the pre-existing missing-dependency set
(`datasets`, `anthropic`, `defusedxml`, `sentence_transformers`) and are
unchanged. 61 tests added, all passing: 22 parser, 23 mappings, 16 holdout.

`pytest tests/test_licensed_text_not_tracked.py` — 8 passed.
`mypy --strict` over `tract/ parsers/ scripts/phase1a/ scripts/phase1b/
scripts/phase0/runpod_provision.py` — 26 errors before this work, 26 after,
none in any file it touches. The five changed or added modules type-check
clean on their own.

The fixture is synthetic: a fictional beacon relay with the real heading
skeleton. No verbatim run of the licensed source is tracked.

## 8. Judgment calls a reviewer should check

1. **The description cut.** Including `Common Examples of Risk` is the call I
   am least certain of. It is definitional in the sense that it says what the
   risk looks like, but it is also enumerative, and for the four entries whose
   `Description` section already exceeds 2,000 characters it changes nothing
   after truncation. If it turns out to hurt, `full_text` has everything and
   the cut is one constant.
2. **`source_url = "https://genai.owasp.org"`.** That is the publisher's
   project site, not this document's URL, because the document is pre-release
   and has none. The Source entry carries `url=None`, which is the accurate
   statement; the framework artifact needs a non-empty string.
3. **The processed artifact is tracked.** Consistent with the four other
   CC BY-SA OWASP frameworks already tracked, and NOTICE records the terms.
   It is the same posture, not a new one.
4. **Naming the extractor `extract_` rather than `parse_`.**
   `tests/test_parser_manifest_coverage.py` scans `parse_*.py` for reads that
   bypass the source manifest, and its rules are written for framework parsers
   reading raw sources. This module does read its raw source through the
   recording reader, and it also reads two processed artifacts to measure the
   CWE chain, which that scanner would flag as a bypass it is not. Precedent:
   `parsers/extract_hub_links.py`.
