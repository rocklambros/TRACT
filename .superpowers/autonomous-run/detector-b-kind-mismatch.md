# R19 — detector B and the mismatch of KIND

Status: complete. Branch `semantic-rebuild`.

## The defect

Detector B compares a link's `section_name` against the title of the control its
`section_id` reached. R11 (`COARSE_NAME_RATIO`) and R21 (`FINE_NAME_RATIO`) both
derive from `distinct(section_id) / distinct(section_name)`, so both measure
GRANULARITY. Neither can see two labels that sit at the same granularity and are
different SORTS of thing.

`nist_ssdf` is that case. Its ids and names are 1:1 at 44 each, so the count
ratio reads exactly 1.0 and sits between the two thresholds. Its `section_name`
is the task STATEMENT and `parse_nist_ssdf` titles each task by its own id, so B
held a 156-character sentence against `"PO.1.1"` and flagged **44 of 44**
applicable checks. The anchors were never wrong: all 46 resolved links reach a
`full_text` task statement.

## The third predicate

`NAME_KIND_RATIO = 7.0`, on `median len(_fold(section_name)) / median
len(_fold(title))`, kept as its own number rather than folded into the count
ratio because the two measure different properties.

Re-derived rather than trusted, and it reproduces the owner's table. Measured
over the 22 link-bearing frameworks on the population B compares: links with a
name, an id-channel hit, a non-empty title, and a name differing from its own id
(B's own guard).

| framework | median name | median title | ratio |
|---|---|---|---|
| nist_ssdf | 156.5 | 6.0 | **26.0833** |
| nist_ai_100_2 | 20.0 | 11.0 | 1.8182 |
| mitre_atlas | 27.0 | 22.0 | 1.2273 |
| csa_ccm | 38.5 | 32.0 | 1.2031 |
| asvs | 158.0 | 157.0 | 1.0064 |
| owasp_top10_2021 | 31.0 | 34.0 | 0.9118 |
| dsomm | 21.0 | 30.0 | 0.7000 |
| etsi | 20.0 | 50.0 | 0.4000 |
| biml | 14.5 | 38.0 | 0.3816 |
| wstg, nist_800_63, owasp_proactive_controls | — | — | no comparable links |

### Headroom

Nothing sits between 1.8182 and 26.0833.

- nearest value above: `nist_ssdf` 156.5/6.0 = 26.0833, headroom **19.0833**
- nearest value below: `nist_ai_100_2` 20.0/11.0 = 1.8182, headroom **5.1818**

7.0 is the GEOMETRIC midpoint of that gap rounded to a whole number,
`sqrt(1.8182 * 26.0833) = 6.8865`. Geometric rather than arithmetic because the
quantity is a ratio, so equal headroom means equal multiples: nist_ai_100_2's
names must grow from 20 folded characters to 77 (x3.85), and nist_ssdf's titles
from 6 to 23 (x3.73).

### Folding

Both sides go through `_fold`, which collapses whitespace and case-folds. That
is the exact string detector B compares and the case-insensitivity `ProseIndex`
keys titles with. Measuring the raw form would describe a comparison the
detector does not make, which is the defect already fixed once in
`_name_level_mismatch`.

### asvs stays unflagged

158.0/157.0 = 1.0064. Its names are long AND its titles are long, so both sides
are the same kind of label and B works on all 277 links. asvs's names are LONGER
than nist_ssdf's, so any predicate keyed on absolute name length orders the two
the wrong way round. The property is shape MISMATCH, not absolute length, and a
test asserts exactly that.

## One-sided, not symmetric

The mirror looks like the same defect and is not one. Measured over the
population B compares, `wstg` (raw 0.3243), `nist_800_63` (0.2692) and
`owasp_proactive_controls` (0.0625) have **no comparable links at all**. Their
`section_name` IS their `section_id` on every link, so B's own
`_fold(name) != _fold(normalized_id)` guard retires it before any comparison
happens. Their zero wrong anchors comes from that guard, not from a shape
property. Declaring them would switch off a detector already inert and grow the
exemption set without covering anything. Both keep reporting 0 of 0 and both
stay out of the declared set.

The low end also holds a framework where B WORKS. `biml` sits at 0.3816 over 20
comparable links and reports 0 against a pre-registered budget of 0, less than
0.06 below wstg. A symmetric threshold low enough to spare biml catches only the
three already-inert frameworks; one high enough to catch wstg retires biml's
working detector. Neither trade is worth making.

## The ratchet

`DETECTOR_B_INAPPLICABLE` equals `detector_b_inapplicable_frameworks()`, the
union of all three derived predicates, asserted exactly, with each direction
also asserted alone so a later weakening to a subset check loses neither.

Final set: `{dsomm, enisa, etsi, nist_ai_100_2, nist_ssdf}`.
`name_level_mismatch_frameworks()` returns the first four,
`name_kind_mismatch_frameworks()` returns `{nist_ssdf}`, and a test pins both
exactly so neither predicate can quietly do the other's work.

The kind predicate needs TITLES, which live in the corpus rather than in the
link file, so a checkout without the licensed overlay cannot measure the four
`OVERLAY_FRAMEWORK_IDS` members. None of them holds the property when the
overlay IS present, so the union is identical in both checkouts, and that is
asserted where the overlay exists and skipped as a NAMED group where it does not.

## nist_ssdf after the change

**0 of 0**, down from 44 of 44. That denominator is honest rather than
convenient, and it is stated rather than implied. Detectors A and C both still
run and neither has a candidate to reach:

- `by_title == 0`, so A never enters its branch.
- The 44 normalised ids are leaf tasks at one depth with **no ancestor pair
  among them** (asserted directly, attainable range [0, 44*43]), so C finds no
  candidate.

`scripts/corpus_report.py` already prints it as `0 of 0 (blind: no detector
applies)`, alongside `wstg` and `owasp_proactive_controls`, which is a
precedented and legible state. Trading 44 of 44 for 0 of 0 removes a false
positive and adds no certification, and `wrong_anchor_applicable()` is what
stops the zero reading as a pass.

`tests/test_corpus_acceptance.py::UNBUDGETED_WRONG_ANCHOR_EXPOSURE` held
`{"nist_ssdf": 44}` with the note "a repair must delete this entry rather than
lower it". Deleted. The mapping stays as the registration point for the next
exposure, and the test that checked the pinned cause now checks the exemption's
cause instead.

## Pre-registered budget

`JOIN_WRONG_ANCHOR_BUDGET = {csa_ccm: 1, etsi: 1, biml: 0}` untouched. All three
still meet it over live denominators of 29, 9 and 21.

## Baselines

`results/corpus/before.json` and `results/corpus/link_resolution_before.jsonl`
did not move. `git status --porcelain` on both is empty.

`results/corpus/after_parsers.json` was regenerated. The live corpus digest
matches the digest that artifact recorded (`b2514469...`), so
`require_unmoved_corpus` passed and this was a reproduction rather than a
re-baseline. The diff is exactly two lines: nist_ssdf 44 -> 0 and the total
74 -> 30, plus 44 rows in the companion JSONL flipping `wrong_anchor` and
`wrong_anchor_checked` to false.

## Tests

- Local, overlay present: **2358 passed, 9 failed, 23 skipped, 3 xpassed**. All
  9 failures are model-loading (`datasets` / `sentence-transformers` absent) and
  were failing before this change. Baseline at start of task was 2314 passed /
  13 failed, the extra 4 being concurrent agents' work since fixed.
- Fresh clone, no `data/raw` and no gitignored overlay: the affected modules run
  **158 passed, 10 skipped**, every skip named with its reason. 16 of my 17 new
  tests run live there, including the ratchet. Only
  `test_the_overlay_frameworks_do_not_hold_the_kind_property` skips, as a named
  group listing `['csa_ccm', 'dsomm', 'etsi', 'iso_27001']`.
- `mypy --strict` on the CI target list: clean, 156 source files.

## Mutation testing

21 real mutations plus one no-op control, run with `PYTHONDONTWRITEBYTECODE=1`
against a pristine snapshot restored before AND after each one, in two isolated
clones so the working tree was never mutated. **All 21 killed in BOTH modes.**

| id | mutation | verdict |
|---|---|---|
| M01 | threshold 27.0, above nist_ssdf | KILLED |
| M02 | threshold 1.5, under nist_ai_100_2 | KILLED |
| M03 | threshold 1.15, under csa_ccm and mitre_atlas | KILLED |
| M04 | `>` instead of `>=` at the boundary | KILLED |
| M05 | symmetric predicate instead of one-sided | KILLED |
| M06 | symmetric with a mirror threshold catching biml | KILLED |
| M07 | drop detector B's id-equality guard | KILLED |
| M08 | case-sensitive id-equality guard | KILLED |
| M09 | raw name length instead of folded | KILLED |
| M10 | drop the empty-population guard | KILLED |
| M11 | absolute name length instead of shape mismatch | KILLED |
| M12 | title channel instead of the id channel | KILLED |
| M13 | union accessor forgets the count predicate | KILLED |
| M14 | declared set loses nist_ssdf | KILLED |
| M15 | declared set gains asvs | KILLED |
| M16 | runtime ignores the exemption set | KILLED |
| M17 | exempt the whole column instead of detector B | KILLED |
| M18 | key the corpus on framework_id not the canonical name | KILLED |
| M19a | subset ratchet + declared without the property | KILLED |
| M19b | subset ratchet + property acquired undeclared | KILLED |
| M21 | mean instead of median | KILLED |
| M20 | no-op control, must survive | SURVIVED (as designed) |

### Survivors on the first pass, and the defects they exposed

Two mutations survived the first run, both exposing real gaps that are now closed.

1. **M21, mean instead of median.** Nothing separated the two statistics,
   because every synthetic fixture in the class carried a single link and mean
   equals median there, and no real framework changes membership either way.
   Robustness to outliers is the whole reason the statistic is a median: a
   framework with nine clean links and one enormous `section_name` would be
   exempted by the mean and not by the median. Added
   `test_the_ratio_is_a_median_so_a_few_outliers_cannot_carry_it`, nine links at
   10/10 and one at 1000/10, where the median ratio is 1.0 and the mean ratio is
   10.9. M21 now dies.

2. **M19, ratchet weakened from `==` to `<=`.** The subset check still catches a
   framework declared without the property and silently loses the other
   direction. Split the ratchet into the equality plus both directions asserted
   individually with their own messages, and confirmed with two compound
   mutations (M19a, M19b) that each direction is load-bearing on its own.

A third defect was found by inspection rather than by a mutation and is worth
recording, because it is the class the brief warned about:
`test_the_shaped_fixture_has_the_lengths_it_claims` originally rebuilt the
fixture inline rather than reading it out of the builder the boundary tests
call, so it checked its own copy and left the builder unguarded. Rewritten to
call `_shaped_pair`.

## Concerns

1. **nist_ssdf's denominator is 0.** The task asked me to confirm an honest
   denominator "rather than zero". It is zero, and that is the honest answer:
   nist_ssdf's link file offers nothing any of the three detectors can check.
   A detector that fires on 100% of its checks and one that has no checks both
   certify nothing, and the difference is that 0 of 0 is legible through
   `wrong_anchor_applicable()` while 44 of 44 was a false positive polluting the
   totals. `wstg` and `owasp_proactive_controls` are already in this state, so it
   is precedented, not novel. If the project wants nist_ssdf genuinely covered,
   the lever is `parse_nist_ssdf` giving its controls real titles, which would
   collapse the kind ratio and re-arm B. The acceptance test now fails if that
   happens without the exemption being re-derived.

2. **The kind predicate depends on the corpus, the other two do not.** The union
   is checkout-stable today and that is asserted, but a restricted framework
   could in principle acquire the kind property where CI cannot see it. The
   runtime is unaffected, since `build_corpus_report` reads the declared constant
   rather than re-deriving, so the exposure is limited to the ratchet test on a
   machine without the overlay.

3. **`results/corpus/after_parsers.json` regeneration** was a legitimate
   reproduction on an unmoved corpus, but it does mean a tracked artifact moved
   in this commit. The two-line diff is stated above so a reviewer can confirm
   nothing else rode along.
