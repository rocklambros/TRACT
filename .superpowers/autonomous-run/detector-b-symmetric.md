# Detector B: the predicate is symmetric now

Ruling R11 retired detector B where a link file's `section_name` sits at a
COARSER level than its `section_id`. The mirror was uncovered, and ETSI is where
it showed: 32 flags against a pre-registered budget of 1.

## What changed

`tract/corpus_report.py`

- `FINE_NAME_RATIO: Final[float] = 0.85` beside `COARSE_NAME_RATIO`.
- `_coarse_name_frameworks` / `coarse_name_frameworks` renamed to
  `_name_level_mismatch` / `name_level_mismatch_frameworks`. The old name
  described half the predicate. Membership is now
  `ratio >= COARSE_NAME_RATIO or ratio <= FINE_NAME_RATIO`.
- `DETECTOR_B_INAPPLICABLE` grew from `{dsomm}` to
  `{dsomm, enisa, etsi, nist_ai_100_2}`, with a per-member reason.
- `JOIN_WRONG_ANCHOR_BUDGET` untouched.

`tests/test_corpus_report.py`, `tests/test_parse_enisa.py`,
`tests/test_parse_nist_ssdf.py` follow.

## Why 0.85

Measured `distinct(section_id) / distinct(section_name)` over all 22 frameworks
carrying curated links, 2026-08-19:

```
dsomm         183/18  10.1667   names COARSER, R11
biml           20/17   1.1765   <- top of the 1:1 cluster
...16 more between 0.9773 and 1.1765...
iso_27001      92/93   0.9892
mitre_atlas    43/44   0.9773   <- floor of the 1:1 cluster
nist_ai_100_2  20/28   0.7143   names FINER, uncovered
etsi           16/24   0.6667   names FINER, uncovered
enisa          10/33   0.3030   names FINER, uncovered
```

Nothing at all sits between 0.7143 and 0.9773. 0.85 is near the midpoint of that
gap.

| side | nearest value | headroom |
|---|---|---|
| below | `nist_ai_100_2` 20/28 = 0.7143 | 0.1357 |
| above | `mitre_atlas` 43/44 = 0.9773 | 0.1273 |

Crossing 0.85 takes four more distinct ids or five fewer distinct names for
`nist_ai_100_2`, and seven more distinct names or six fewer distinct ids for
`mitre_atlas`. Neither a name repair nor a handful of new links moves a
framework across it.

Two corrections to ruling R21 as written in the ledger. It records the 1:1
cluster as bottoming out at 0.99. The measured floor is `mitre_atlas` at 0.9773,
which is the tighter of the two sides, so the headroom above is 0.1273 and not
the 0.14 the ruling implies. The ruling's summary table omits `mitre_atlas`
entirely; it has 65 links and belongs in it.

The tidy choice, `1 / COARSE_NAME_RATIO = 0.5`, is empirically wrong: `etsi` at
0.6667 and `nist_ai_100_2` at 0.7143 both carry the property and both sit above
it, so 0.5 would leave the defect uncovered. A test asserts that, because it is
the change a later reader is most likely to make on symmetry grounds alone.

## The decomposition, verified rather than trusted

Task 13 reported that detectors A and C give 1 of 9 for ETSI. Reproduced by
running `build_corpus_report()` twice against the full 31-framework overlay with
only the membership set varied:

```
                          B on            B off
etsi              32 of 36        ->   1 of 9
nist_ai_100_2     20 of 45        ->   8 of 29
enisa              0 of 68        ->   0 of 68
dsomm              0 of  3        ->   0 of  3
```

ETSI's surviving flag is the `6.3.1` link whose `section_name` is clause 6.3's
heading, on the title channel, which is the row
`JOIN_WRONG_ANCHOR_BUDGET["etsi"] = 1` was registered for. The denominator of 9
is five title-channel checks and four id-channel ones, so the 1 is not a zero
over nothing. The pre-registered budget is met, not moved.

All eight of `nist_ai_100_2`'s survivors are detector A. The twelve that left
were id-channel detector B flags.

`enisa` does not move at all. Every enisa link resolves through the title
channel, so detector B never ran for it. It is declared anyway: the property
belongs to the link file, and membership that turned on whether a channel
happened to fire would be membership nobody could predict from tracked inputs.

## Tests

+11 net over the baseline. `2071 passed` against `2060`, the same 9 environmental
model-loading failures, 24 skipped, 3 xpassed.

New:

- the equality ratchet, unchanged in shape, now over the symmetric predicate
- the shaped-fixture helper checked for the cardinalities it claims
- the real link file split into three named groups
- both thresholds sitting in a gap, with the nearest value on each side asserted
- the reciprocal 0.5 shown to miss `etsi` and `nist_ai_100_2`
- boundary at and past each threshold, in both directions
- the middle of the range asserted non-member at three points
- detector B skipped for a fine-name member, A and C still firing for one
- the real `etsi`, `nist_ai_100_2` and `enisa` rows

The three real-row tests build from `data/processed/frameworks/<id>.json` rather
than the shared `all_controls.json`, per ruling R15. The shared file is modified
in the working tree and lags at HEAD, so a test reading it asserts state no
commit carries. Confirmed the per-framework join reproduces the full-overlay
figures exactly. `etsi` is restricted, so its artifact is gitignored and that one
test skips with a stated reason in a checkout without licensed text; the
predicate tests assert `etsi`'s membership from the tracked link file, so the
exemption stays gated everywhere.

## Mutations

15 written, all died, in a full local run and in a `git archive HEAD` checkout
carrying only tracked files. Zero survivors.

```
M1  FINE_NAME_RATIO = 0.5                        reciprocal of the coarse one
M2  FINE_NAME_RATIO = 0.95                       loose, eats the headroom above
M3  FINE_NAME_RATIO = 0.70                       fitted to admit etsi only
M4  ratio <  FINE_NAME_RATIO                     strict at the boundary
M5  drop the coarse direction
M6  drop the fine direction                      the pre-R21 predicate
M7  or -> and
M8  ratio >= FINE_NAME_RATIO                     comparison flipped
M9  DETECTOR_B_INAPPLICABLE drops enisa
M10 DETECTOR_B_INAPPLICABLE adds nist_ssdf        1:1, saturates B honestly
M11 exempt the whole wrong-anchor column         C switched off with B
M12 runtime membership inverted
M13 COARSE_NAME_RATIO = 1.2                      eats the headroom below biml
M14 remove the empty-name guard                  nameless files become members
M15 the shaped-fixture helper made to lie
```

M4 is killed only by the fine boundary-at case, which is why that case is
written: `17/20 == 0.85` exactly in IEEE double, verified. M13 is killed only by
the gap test, which is what the gap test is for.

## Committed baselines

`git status --porcelain results/corpus/` is empty before and after.
`results/corpus/before.json` and `results/corpus/link_resolution_before.jsonl`
are untouched, as they must be: none of these frameworks resolved through a
parser in the BEFORE state.

## Concerns for whoever picks this up

1. `parsers/parse_etsi.py` and `parsers/parse_nist_ssdf.py` carry docstrings
   naming `COARSE_NAME_RATIO` and `coarse_name_frameworks()`. The ETSI one now
   states the opposite of the truth ("COARSE_NAME_RATIO cannot route it to
   DETECTOR_B_INAPPLICABLE"), and the SSDF one names a function that no longer
   exists. `parsers/` was out of scope for this task. Both need one edit.

2. `nist_ai_100_2` reports 8 wrong of 29 and has no `JOIN_WRONG_ANCHOR_BUDGET`
   entry. The comment above that mapping says a framework absent from it must
   report zero and that Task 16 asserts `by_title == 0` instead. `nist_ai_100_2`
   has `by_title == 29`, so neither reading holds. Its owner has to register a
   figure or explain the eight. The eight are real title-channel disagreements,
   not detector-B noise.

3. Four tests fail in a tracked-files-only checkout, all of them at pristine
   HEAD and none of them caused by this change:
   `test_the_real_dsomm_row_reports_zero_over_a_live_denominator`,
   `TestUnmovedCorpusGuard::test_the_committed_baseline_is_what_the_guard_now_refuses`,
   `test_parse_csa_ccm.py::TestRealWorkbook::test_the_written_artifact_is_not_tracked`,
   `test_prose_reachability.py::test_every_parser_backed_framework_resolves_its_links`.
   The common cause is `data/processed/all_controls.json` and several
   `data/processed/frameworks/*.json` being uncommitted while HEAD's copies lag
   the parsers that landed. CI is red on this branch today for that reason. Task
   15 owns it. The dsomm test in particular would be fixed by the same
   per-framework construction the three new real-row tests use.

4. `FULL_CORPUS_FRAMEWORK_COUNT` is no longer referenced by
   `tests/test_corpus_report.py`. It is still used by `require_full_corpus` and
   `floors_for_report`, so nothing is dead.
