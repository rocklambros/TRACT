# NIST AI RMF line-wrap repair and AIUC-1 withdrawal notices

Two published data-quality defects found while setting prose floors on the
nineteen undeclared parsers. Neither framework contributes a curated OpenCRE
link, so neither touches training. Both reach the published crosswalk dataset,
so the repository is now correct for the next publish. The published bundle
under `build/` is untouched.

## Corrections to the brief

Every figure below was re-derived on this branch. Three claims in the brief did
not survive.

**"Five of the 72 are not split, and joining those would duplicate text."**
False. All 72 NIST AI RMF subcategories are wrapped. The brief's 67 counts
descriptions that open on a lowercase letter, which is the right count for that
signal and the wrong count for the split. The other five continue with `AI`,
`(both` or a capitalised word, so they are split just as thoroughly:

    GOVERN 2.2  title 'The organization's personnel and partners receive'
                desc  'AI risk management training to enable them to ...'
    MAP 5.1     title 'Likelihood and magnitude of each identified impact'
                desc  '(both potentially beneficial and harmful) based ...'

**"MEASURE 2.11 additionally splits inside a `**MAP**` marker."** True, and it
was not the only extra damage. The description ran from the wrap to the next
subcategory marker, so it also carried whatever the converter emitted in
between. `GOVERN 1.4` shipped `Continued on next page`, `Page 22`, the running
header `NIST AI 100-1 AI RMF 1.0`, the repeated table caption and the column
header row. `MEASURE 2.13` shipped the text of the `MEASURE 3` category cell.
`MEASURE 2.12` needed a second block for a different reason: the converter left
a stray blank line inside the sentence.

**"The measured prose fraction of 0.7639."** Confirmed at 55/72 on the old
parse. The repaired parse measures 72/72.

## Defect 1: nist_ai_rmf

The source is a PDF converted to markdown. Tables 1 through 4 render each
subcategory as one table cell, hard-wrapped, delimited by a blank line. The old
`SUBCATEGORY_RE` captured the title as `[^\n]*`, which stops at the first wrap.

The repair reads the cell rather than the line. It scans the blocks between one
subcategory marker and the next, keeps blocks until one ends a sentence, and
refuses any block that opens a new structural element. Two cells need the second
block, `MAP 1.4` and `MEASURE 2.12`, and both are identified by the same signal:
a first block that does not close on a period. Neither gets a rule of its own.

`BLOCK_STOP` is the load-bearing guard. Without it an unterminated cell walks
through the page furniture and closes on the period of a table caption or a
category cell, producing a grammatical sentence assembled from two table
columns that clears every count and prose gate. The parser raises instead.

**Title.** The subcategory identifier, `GOVERN 1.1`. The source gives
subcategories no title: the table has two columns, and the subcategory column
holds a bare statement under an identifier. Any title would be a truncation of
the statement, which is the defect being removed.
`parsers/parse_nist_ssdf.py` took the same route for the same reason.

**Audit.** Every one of the 72 rejoins writes a `write_repair_audit` record
carrying `before` as the two halves the wrap created and `after` as the
rejoined statement, both as text. `source_blocks` is 2 for the two cells the
converter also split with a blank line and 1 for the rest. `text_origin` is not
set: rejoining a converter's wrap restores the publisher's sentence.

| | before | after |
|---|---|---|
| controls | 72 | 72 |
| honest prose fraction | 0.7639 (55/72) | 1.0000 (72/72) |
| declared floor | 0.76 | 1.00 |
| descriptions opening lowercase | 67 | 0 |
| descriptions carrying page furniture | 6 | 0 |

The floor of 1.0 has one character of margin. `MAP 1.5`, "Organizational risk
tolerances are determined and documented.", is 61 characters against
`HONEST_PROSE_MIN_CHARS` of 60. `tests/test_parse_nist_ai_rmf.py` pins that
margin so a change in sanitisation names the control instead of surfacing as an
unexplained floor failure inside `run()`.

## Defect 2: aiuc_1

`E007.1` and `E014.1` carried `RETIRED - merged into E004.` and
`RETIRED - merged into E017.`. Their parent controls carry the matching notice
at control level. They are dropped, and `expected_count` moves 132 to 130.

Ruling R17 settled the shape for WSTG: a notice that names a successor does not
ship as a control, and the retired id becomes an `alt_ids` entry on the
successor. The first half applies here unchanged. The second does not, for two
reasons that were checked rather than assumed.

**The successor is named at the wrong level.** E007 and E014 name E004 and
E017, which are controls, while this parser's unit is the activity. E004 has
two activities and E017 has three, and the source nowhere says which absorbed
E007.1 or E014.1. An `alt_ids` entry would have to pick one, which asserts an
equivalence the publisher did not state.

**Nothing would read it.** AIUC-1 carries none of the 4,405 curated OpenCRE
links, measured over `results/corpus/link_resolution_after_parsers.jsonl`. No
curated link targets `E007.1`, `E014.1`, `E007` or `E014`. The alias channel
exists to let a curated link reach prose.

Keeping them with a damaged marker was the third option and is worse. The
marker takes a control out of the prose ratio, not out of the artifact, so both
redirect notices would still publish as crosswalk rows.

Each dropped activity writes an audit record carrying the notice text, the
retired control's own statement and the successor's statement.

| | before | after |
|---|---|---|
| controls | 132 | 130 |
| honest prose fraction | 0.8333 (110/132) | 0.8462 (110/130) |
| declared floor | 0.83 | 0.84 |
| shipped statements that are redirects | 2 | 0 |

The drop refuses to remove anything real. A notice that names no successor, one
that names an id the standard does not issue, one that names a retired
successor, a live activity under a retired control, and a notice under a live
control each raise a `ValueError` naming the record.

## Corpus ledger

`scripts/rebuild_corpus.py` gates every record outside the eleven rebuilt
frameworks against the pre-rebuild baseline, so both repairs had to be declared
there. `REPAIRED_PARSER_MOVED_KEYS` names the 72 nist_ai_rmf keys one at a time.
`DECLARED_DROPPED_KEYS` names the two aiuc_1 keys separately, because a record
with no live digest leaves through the diff report's removed bucket rather than
its changed bucket, and that split tells a reader whether a control was
rewritten or withdrawn. `EXPECTED_UNCHANGED_RECORDS` moves 3784 to 3710.
`results/corpus/rebuild_diff.json` is regenerated from a `--dry-run` into the
scratchpad, so no processed artifact was committed by that run.

## Verification

`mypy --strict` clean over the CI target list, 155 source files.

Full suite 2353 passed, 9 failed, 23 skipped, 3 xpassed. The 9 are the
environmental model-loading failures, `datasets` and `sentence-transformers`
absent by policy. The baseline before this work was 2290 passed and 12 failed,
where three of the twelve were a concurrent agent's in-flight
`tests/test_framework_identity.py` and are now green.

`tests/test_parse_nist_ai_rmf.py` 26 tests, `tests/test_parse_aiuc_1.py` 18.

## Mutation testing

25 mutants against a pristine snapshot, `PYTHONDONTWRITEBYTECODE=1`, restored
before and after each one. Final result: no survivors. Four survived the first
pass and each is recorded below with what it exposed.

| id | mutation | verdict |
|---|---|---|
| M1 | blind concatenation, keep every block in the segment | killed |
| M2 | first block only, stop at the first blank line | killed |
| M3 | drop the BLOCK_STOP guard and absorb until a period | killed on the second pass |
| M4 | keep the first source line as the title | killed |
| M5 | rejoin without a separator | killed |
| M6 | rmf audit records carry lengths instead of text | killed |
| M7 | never emit an rmf audit record | killed |
| M8 | unanchor the subcategory marker | killed |
| M9 | return a partial statement instead of raising | killed |
| M10 | leave the rmf prose floor at the pre-repair 0.76 | killed |
| M11 | rejoin without the hyphenation fix | killed |
| M12 | treat a blank block as continuable rather than a cell boundary | killed on the second pass |
| M13 | ship the withdrawal notices as controls | killed |
| M14 | unanchor the retirement marker in the pattern and at every call site | killed on the second pass |
| M15 | accept a successor the standard does not issue | killed |
| M16 | follow a redirect onto a retired successor | killed |
| M17 | drop a live activity along with its retired control | killed |
| M18 | skip a notice on a live control instead of raising | killed |
| M19 | aiuc audit record carries lengths instead of text | killed |
| M20 | leave the aiuc count at the pre-drop 132 | killed on the second pass |
| M21 | leave the aiuc prose floor at the pre-drop 0.83 | killed |
| M22 | accept a notice that names no successor | killed |
| M23 | under-declare the subcategory count | killed |
| M24 | match the retirement marker case-sensitively only | killed |
| M25 | let the cell scan close on any character, not a sentence | killed |

### Survivors and what they exposed

**M3 exposed a decorative test.** The `UNTERMINATED` fixture ended on page
furniture that happens not to close on a period, so the scan stopped by running
out of input and raised for the wrong reason. `BLOCK_STOP` was never exercised.
`UNTERMINATED_BEFORE_CATEGORY` puts a category cell after the unterminated one.
That cell ends on a period, so without the guard the scan absorbs it and ships
a statement built from two table columns with no gate to notice.

**M12 exposed an untested decision.** Treating a blank block as a cell boundary
rather than skipping it is a real choice about how wide a gap still counts as
inside a cell. The source uses exactly one blank line inside `MEASURE 2.12` and
never two. `UNTERMINATED_ACROSS_A_WIDE_GAP` pins the conservative side: the
parser refuses rather than assembling across a double blank line.

**M14 was an equivalent mutant twice before it was a real one.** Removing `^`
from the `RETIRED` pattern changes nothing while the call site uses `re.match`,
and switching `match` to `search` changes nothing while the pattern carries
`^`. Both defences are redundant with each other, which is why either edit
alone is unobservable. The plausible-wrong implementation needs both, and that
form is killed.

**M20 exposed a gate nothing exercised.** Every fixture-based test overrides
`expected_count`, and the real-source tests call `parse()` rather than `run()`,
so no test read the declaration. Both parsers now assert that the declared
count equals what the real source yields.

**A test found a defect before any mutant did.**
`test_a_live_activity_under_a_retired_control_raises` failed against the first
implementation: a retired control carrying a live activity dropped that
activity's real statement silently. The parser now raises and names the record.

## Concerns

`data/processed/all_controls.json` is now stale against both regenerated
per-framework artifacts. It is deliberately not committed, per the brief, and a
later step owns it.

The prose floor of 1.0 for nist_ai_rmf rests on `MAP 1.5` clearing
`HONEST_PROSE_MIN_CHARS` by one character. The margin is pinned by a test that
names the control, so the failure is readable, but a change to that constant or
to whitespace handling will fire the floor.
