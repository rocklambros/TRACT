# Prose floors for the nineteen undeclared parsers

Closes the gate gap where `BaseParser.min_prose_fraction` defaulted to `0.0`
and a parser that never overrode it had the prose gate switched off entirely.

## What was found

Every claim below was re-derived on this branch rather than carried forward.
Measurement drove each parser's `parse()` through `_sanitize_control()` and
`BaseParser.honest_prose_fraction()`, and then a second pass drove the full
`run()` into a scratch directory so the real gate produced the verdict.

**Correction to the brief.** Thirteen parsers already declared a floor, not
eleven: `biml`, `csa_ccm`, `dsomm`, `enisa`, `etsi`, `iso_27001`,
`nist_800_63`, `nist_ssdf`, `owasp_llm_top10_2026`,
`owasp_proactive_controls`, `owasp_top10_2021`, `samm`, `wstg`. The nineteen
undeclared parsers and their measured fractions matched the brief exactly, to
four decimal places, on all nineteen.

The committed artifact and a fresh parse agreed on every framework, so no
artifact is stale and the floors are set from what `run()` gates on.

## Floors declared

| framework | controls | measured | floor | fires at |
|---|---|---|---|---|
| asvs | 278 | 1.0000 | 1.00 | 277/278 |
| eu_ai_act | 126 | 1.0000 | 1.00 | 125/126 |
| eu_gpai_cop | 40 | 1.0000 | 1.00 | 39/40 |
| mitre_atlas | 202 | 1.0000 | 1.00 | 201/202 |
| nist_800_53 | 300 | 1.0000 | 1.00 | 299/300 |
| nist_ai_100_2 | 66 | 1.0000 | 1.00 | 65/66 |
| nist_ai_600_1 | 12 | 1.0000 | 1.00 | 11/12 |
| owasp_agentic_top10 | 10 | 1.0000 | 1.00 | 9/10 |
| owasp_ai_exchange | 107 | 1.0000 | 1.00 | 106/107 |
| owasp_cheat_sheets | 120 | 1.0000 | 1.00 | 119/120 |
| owasp_dsgai | 21 | 1.0000 | 1.00 | 20/21 |
| owasp_llm_top10 | 10 | 1.0000 | 1.00 | 9/10 |
| owasp_ml_top10 | 10 | 1.0000 | 1.00 | 9/10 |
| capec | 558 | 0.9964 | 0.99 | 552/558 |
| cwe | 1331 | 0.9925 | 0.99 | 1317/1331 |
| csa_aicm | 243 | 0.9753 | 0.97 | 235/243 |
| cosai | 55 | 0.9636 | 0.96 | 52/55 |
| aiuc_1 | 132 | 0.8333 | 0.83 | 109/132 |
| nist_ai_rmf | 72 | 0.7639 | 0.76 | 54/72 |

All 32 parsers then ran end to end through `run()` and wrote. The 32 artifacts
are byte-identical to the committed ones, so the floors changed no output.

## Judgement call 1: the thirteen at 1.0000

**Set 1.00.** Four reasons.

The convention already says so. Rounding 1.0000 down to two places gives 1.00,
and eight parsers that measure 1.0000 today already declare 1.0. Splitting the
fleet so half the wholly-prose sources get 1.00 and half get 0.99 makes the
number unreadable.

The asymmetry favours the stop. A false stop costs one build and one look at a
diff, and it names the framework, the measured fraction and the floor. A missed
regression puts title-only controls into training data and onto HuggingFace,
which is the failure this gate was built for.

A weaker floor buys almost nothing on these sets. Seven of the thirteen hold
126 controls or fewer, where losing one lands between 0.9750 and 0.9000 and
trips 0.99 or 0.95 anyway. Giving even one control of slack at n=10 needs a
floor of 0.90, and a 10% floor is not a gate.

The counter-argument, stated rather than hidden: two decimal places does not
give one uniform strictness. At 1,331 CWE weaknesses the 0.99 floor already
grants four losses before it fires, and at 558 CAPEC patterns the same. That is
a resolution limit of the convention, not a policy choice, and each affected
parser's comment records its own slack.

**When a 1.00 floor fires**, run the parser and list the controls whose
description is shorter than `HONEST_PROSE_MIN_CHARS` or byte-equal to their
title. Two things can have happened. If the source genuinely added a terse
control, lower the floor to the newly measured value rounded down, in the same
commit that moves `version` and `fetched_date`, so the artifact and its floor
never disagree. If the parser lost text, fix the parser and leave the floor
alone. Never relax a floor without naming the control that forced it. That
guidance lives once in `tract/parsers/base.py` next to the attribute.

`mitre_atlas` has the thinnest margin in the group: `AML.M0004`'s description
is 62 characters against a 60-character threshold, so an upstream copy edit can
trip it. Its comment says so.

## Judgement call 2: the two outliers

**`nist_ai_rmf` at 0.7639 is a parser defect, not a terse source.**
`SUBCATEGORY_RE` captures the title as `[^\n]*`, which stops at the first hard
line wrap in the markdown. Every NIST AI RMF subcategory is one sentence, so
the title takes the first line and the description takes the remainder. 67 of
the 72 descriptions open with a lowercase continuation of their own title.
`GOVERN 1.1` is title `"Legal and regulatory requirements involving AI"` and
description `"are understood, managed, and documented."`. `MEASURE 2.11` splits
inside a markdown token: title ends `"...as identified in the **MAP"` and the
description begins `"function - are evaluated..."`. The 17 that miss the
60-character threshold are the ones whose tail happens to be short, which makes
0.7639 a measure of where the converter wrapped lines rather than of how much
prose the source carries.

Not fixed here, per the brief. The floor is still declared at 0.76, because it
holds today's state against a further regression. Repairing the split will
raise the attainable value toward 1.0, and that repair needs its own change,
its own re-measurement, and a floor raised in the same commit. The parser
comment carries the whole finding.

**`aiuc_1` at 0.8333 is a terse source, and the parser drops nothing.** Every
activity record in `aiuc-1-standard.json` holds exactly four keys (`id`,
`description`, `category`, `evidence_types`), checked exhaustively across all
132 records, and `parse()` copies `description` verbatim. Source description
length runs 27 to 216 characters with a median of 76, and 22 are under 60 in
the source itself. Two of the 22 are `"RETIRED - merged into ..."` tombstones,
which is a separate data-quality question about retired activities remaining in
the corpus and not a prose-fraction defect. The floor at 0.83 is honest.

## Judgement call 3: csa_aicm

Floor set at 0.97 and content untouched. The six misses are one-line
specifications of 39 to 58 characters (`AIS-11`, `DSP-04`, `IAM-07`, `I&S-05`,
`I&S-08`, `STA-06`). The comment states that the floor is independent of the
open licensing question, which decides whether the text may ship rather than
whether the parser is reading it.

## The test

`tests/test_prose_floor_declarations.py`, five tests, reading the declared
class attribute with `ast` rather than by importing. It needs no `data/raw/`
and no parse toolchain, so it passes in a fresh clone, which is the same reason
`tests/test_prose_reachability.py` reads parsers statically. The parser list is
derived from `PARSERS_DIR.glob("parse_*.py")` and the class list from the
inheritance graph inside each module, transitively, so a shared intermediate
base is handled.

1. `test_the_scan_covers_the_parser_fleet` — at least 30 parser modules exist,
   so the other four cannot go quiet.
2. `test_every_parser_module_defines_a_parser_class` — no `parse_*.py` is
   invisible to the scan.
3. `test_the_inherited_default_is_still_the_off_position` — pins
   `BaseParser.min_prose_fraction == 0.0`, so "declares a floor" cannot stop
   meaning "the gate is on".
4. `test_every_parser_declares_a_floor_above_the_default` — the gap itself.
5. `test_no_parser_declares_a_floor_it_can_never_meet` — no floor above 1.0,
   which `honest_prose_fraction` can never reach.

The floor must be a numeric literal. A value hidden behind a module constant
raises a specific `ValueError` naming the class, because the number exists to
be read next to the parser it governs.

## Mutation results

Run with `PYTHONDONTWRITEBYTECODE=1` against a pristine snapshot restored
before and after each mutation. Digests verified equal to pristine afterward,
and no mutant file remained.

| # | mutation | result |
|---|---|---|
| M0 | pristine | survived, as expected |
| M1 | delete `asvs`'s declaration | KILLED (test 4) |
| M2 | set `cwe`'s floor to 0.0 | KILLED (test 4) |
| M3 | set `mitre_atlas`'s floor to 1.5 | KILLED (test 5) |
| M4 | move the base default to 0.5 | KILLED (test 3) |
| M5 | add a new parser with no floor | KILLED (test 4) |
| M6 | add a `parse_*.py` with no parser class | KILLED (test 2) |
| M7 | move the floor to a non-parser class in the same file | KILLED (test 4) |
| M8 | hide the floor behind a module constant | KILLED (tests 4 and 5) |
| M9 | inherit a floor from a local intermediate base | survived, as intended |
| M10 | shrink the parser tree to two modules | KILLED (test 1) |

M5 is the headline case from the brief. M7 is the discriminating one: a
grep-based implementation survives it because the string is still in the file,
and the AST implementation kills it. M9 is a false-positive control rather than
a mutation, confirming legitimate inheritance is not reported as ungated. M10
ran the test functions against a patched `PARSERS_DIR` outside the repo.

No survivors, so no defect exposed by a survivor.

## Side effect that had to be fixed

Three fixture tests started failing once the floors went in, because their
fixtures carry toy one-line descriptions:
`aiuc_1` at 0.000, `csa_aicm` at 0.500, `mitre_atlas` at 0.000.

The fixtures were repaired rather than the floors relaxed, and
`min_prose_fraction` was deliberately not overridden on the `SampleXParser`
subclasses. The repo's own precedent argues for this: those subclasses override
`expected_count` with a docstring saying the count gate "is real and must stay
real, so the test declares what this input contains instead of asking the gate
to look the other way". `expected_count` is a property of sample size, so
overriding it is honest. `min_prose_fraction` is a property of the text, so
overriding it would be the gate looking the other way.

`aiuc_1` and `csa_aicm` fixture text is now verbatim source. `mitre_atlas`
fixture text was extended to full statements in the ATLAS voice, keeping the
existing ids and structure that the test asserts on. Each `SampleXParser`
docstring now records why the fixture has to carry prose.

## Verification

- Baseline before the change: 9 failed, 2216 passed, 24 skipped, 3 xpassed.
- After: 9 failed, 2251 passed, 24 skipped, 3 xpassed. The nine are the same
  nine model-loading failures (`test_training_loop.py` and
  `test_publish_merge.py`, `ModuleNotFoundError: datasets`). The rest of the
  gain beyond my five tests comes from concurrent work on the branch.
- `ruff check` clean over the CI scope and the new test.
- `mypy --strict` clean over the CI scope (155 source files) and over the new
  test file.
- All 32 parsers run end to end and write; artifacts byte-identical.

## Concerns

1. **`nist_ai_rmf` needs a follow-up.** The title/description split is wrong on
   67 of 72 subcategories, which means the training anchors for this framework
   are half-sentences. The 0.76 floor documents the state, it does not repair
   it. This is a corpus-quality defect independent of the gate.
2. **Two decimal places is coarse on the large catalogs.** `cwe` and `capec`
   can lose four descriptions each before their floors fire. If the fleet ever
   wants per-control resolution, the floor has to become a count rather than a
   fraction, which is a bigger change than this one.
3. **`aiuc_1` carries two RETIRED tombstones** (`E007.1`, `E014.1`) as live
   controls. They depress the prose fraction and, more importantly, they are
   retired guidance sitting in a corpus that trains an assignment model.
4. **The 1.00 floors will fire on ordinary upstream churn** for
   `mitre_atlas` in particular. That is the accepted trade, and the runbook for
   it lives in `tract/parsers/base.py`.
