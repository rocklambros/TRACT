# Semantic rebuild: assigning controls to OpenCRE hubs by meaning

Status: v2, rewritten 2026-08-15 after a six-perspective adversarial premortem.
Owner: rock@rockcyber.com
Branch: `semantic-rebuild`
Supersedes: v1 of this file, same date. v1's Problem statement, leading
hypothesis and graph design were each inverted by evidence the premortem
recomputed. The disposition of all 36 findings is in the appendix.

## Evidence standard

v1 claimed "every number in this spec was measured against committed artifacts
or source during design." **That was false**, and it is the claim that let the
other errors through. It is replaced with a per-number provenance rule:

- **[measured]** — recomputed from a committed artifact during this design pass.
- **[derived]** — arithmetic over measured values, shown inline.
- **[unmeasured]** — stated as an open quantity with the measurement that would
  close it. No threshold anywhere in this document may depend on an
  **[unmeasured]** value.

A number with no tag is an error in this document. The char-TF-IDF figure
`0.2212` that v1 leaned on is **[unmeasured]**: no TF-IDF implementation exists
anywhere in this repository, and no artifact under `results/` contains that
value. See Part 0.4.

---

# Problem

The PRD names semantics throughout and the system does something else. Three
measured facts, and one v1 got backwards.

**Fine-tuning is not beating the untrained model.** On the 1,265-control
validation roster, macro delta `-0.0004` at `p = 0.98` for arm A1 and `+0.015`
at `p = 0.82` for A2. On OWASP ASVS, 277 items, zero-shot BGE scores `0.556`
and the fine-tuned model scores `0.282`. **[measured]**

That pooled figure hides severe heterogeneity. Cochran Q = 79.8 on 4 df,
I² = 95%, and leave-one-fold-out swings the macro delta from `-0.049` to
`+0.068`, an 11.7-point range against a design that claims to detect 3.5.
On the 872 items whose anchor resolved through the prose path, the delta is
`-0.055` with a 95% paired-bootstrap CI of `[-0.088, -0.022]`, which excludes
zero. **[measured]** "Net zero at p = 0.98" is the p-value of an average no
fold supports, and every "vs zero-shot" comparison in Part 5 inherits the same
pooling unless it is stratified.

**Rank-1 errors are concentrated near the gold hub, not far from it.** v1
asserted the opposite and built Part 2.6 on it. Against a uniform-random
baseline conditioned on the observed gold distribution, over 3,217 scored items
and 2,491 errors:

| error class | observed | random | lift |
|---|---|---|---|
| sibling of gold | 8.3% | 0.9% | **9.3x** |
| ancestor or descendant | 9.2% | 1.0% | **9.5x** |
| same top branch | 39.5% | 36.2% | 1.1x |
| different branch | 45.7% | 61.9% | **0.7x** |

**[measured]** Cross-branch errors are *under*-represented relative to chance.
The model selects the branch better than random. Sibling and ancestor confusion
are the massively over-represented classes. v1 compared 8.3% against 45.7%
instead of comparing each against its own base rate, concluded that the
negative-mining budget was aimed at the wrong 8.3%, and proposed re-aiming it.
**That conclusion is withdrawn.** The existing sibling-and-cousin miner is
aimed at the right region. What is wrong with it is its quality, not its
target: 436 of 12,381 negative slots are the empty string and 99 links get no
negative at all. **[measured]**

Caveat on the 3,217: it is not 3,217 independent observations. The 147-item
test roster appears in four arms and the 1,265-item validation roster in two,
so roughly 1,511 distinct items are counted one to four times with correlated
errors, and 85% of pooled errors come from two arms of one configuration.
**[measured]** The lift ratios are stable across that pooling because they are
within-item comparisons, but the absolute error count is not a sample size.

**The model separates from lexical matching at hit@10 and not at hit@1.** Its
apparent advantage on the published test roster is 79% lexical echo. This is
carried from a prior analysis and is **[unmeasured]** in this pass. Part 0.4
schedules the measurement.

---

# Goal and non-goals

Build a system that assigns a control to a CRE hub because the two mean the
same thing, measure it in a way that would detect the failure above, and
**know when to stop**.

Non-goals, each explicit:

- **Reproducing or beating `0.531`.** Withdrawn, see Part 3.3. Its eval set has
  changed for independent reasons. Nothing here is comparable to it.
- **Rescuing the fine-tuned model.** If enrichment does not beat zero-shot, the
  deliverable is zero-shot plus better targets plus human review. That is a
  legitimate outcome, not a failure, and Part 0 says so in advance.
- **Optimising hit@1 as the primary metric.** See Part 0.2.

---

# Part 0 — Ceiling, operative metric, and stopping conditions

This part is first because it gates everything after it, and because v1 had no
equivalent. Every branch in v1 routed to more work and none said stop.

## 0.1 The ceiling has never been measured

The only ceiling evidence in the repository is the Phase 3 hidden-calibration
result: the expert reviewer agreed with OpenCRE ground truth on **13 of 20**
items. **[measured]** Wilson 95% CI `[0.433, 0.819]`, half-width **0.193**.
**[derived]**

That interval is wider than every effect this programme is trying to detect. It
is unusable as a gate and it is the only ceiling number that exists.

It also carries an uncomfortable implication that no document in this project
has confronted. If two qualified annotators agree on hub assignment at roughly
0.65, then a model at 0.53 was already near ceiling, and "the model does
literal matching" is partly a statement about label noise rather than about the
model. That possibility must be closed before spending on architecture.

**Deliverable: a 250-item blind agreement study.** Reuses the Phase 3
machinery exactly (`tract review-export`, hidden calibration items,
`tract review-import`). 250 items drawn from the validation roster
distribution, ground truth hidden, reviewed blind. Powered to a Wilson
half-width of **0.059** at α ≈ 0.65. **[derived]**

| n | half-width at α ≈ 0.65 |
|---|---|
| 100 | 0.092 |
| 150 | 0.075 |
| **250** | **0.059** |
| 364 | 0.049 |

It measures two quantities, not one:

- **α₁** — the annotator picks the same single hub as OpenCRE. Ceiling on hit@1.
- **α₅** — the OpenCRE gold is inside the annotator's acceptable set of up to
  five. Ceiling on hit@5.

No GPU. It is the highest-value measurement available to this project and it
has never been run.

## 0.2 The operative metric is hit@5, not hit@1

The downstream use is contributing crosswalk mappings to OpenCRE **with expert
review in the loop**. Phase 3 reviewed 878 predictions before publication. The
model's job in that workflow is to put the right hub into a short list a human
scans, not to be right at rank 1.

The measured data already says this is where the model is strong. Prose reaches
hit@10 `0.803` against title's `0.782` on the test roster while losing at
rank 1. **[measured]** A review UI showing five candidates is served better by
the representation with the better neighbourhood.

**Primary metric: hit@5, multi-label and single-gold reported together.**
Secondary: hit@1, which still governs the unreviewed `tract assign` path.
Reporting hit@1 alone was a mismatch between the metric and the product.

## 0.3 Three stopping gates

Each names an **action**, not a conclusion. v1's Risk 4 said "a refutation is a
finding, not a failure," which is exactly how a stop gate gets dissolved. A
conclusion is arguable; an action is auditable.

**S1 — Ceiling stop (success-shaped).**
Trigger: the best configuration's hit@5 point estimate falls inside the Wilson
CI of α₅.
Action: stop architecture work. Publish the model, the corpus and the graph.
Write the ceiling result as the headline finding. The task is solved to the
limit of its labels and further hit@5 work is measuring annotator noise.

**S2 — No-lift stop.**
Trigger: after Parts 1 through 4, no configuration's hit@5 delta over zero-shot
BGE has a Šidák-corrected CI lower bound above zero on the validation roster.
Action: stop the training programme. Ship zero-shot BGE plus enriched targets
plus review. Deprecate the fine-tuned checkpoint and repoint the `tract assign`
default. Publish the negative result.

**S3 — Paradigm stop.**
Trigger: both of (a) the panel judges a lexical baseline semantically
equivalent to the encoder on the Part 0.4 comparison, and (b) recall@50 on
validation does not exceed zero-shot hit@5 on the same roster, meaning L1
destroys more than L2 can recover.
Action: stop the encoder programme entirely. Close the RunPod account. Write
the negative-result report. **Scope limit:** S3 terminates the encoder and the
bake-off. It does not terminate the corpus rebuild, the hub graph, or the
published crosswalk, each of which stands on its own.

**Cannot-decide outcome.** If the CI on the trigger quantity is wider than the
gap being tested, the gate returns *cannot decide* and names the measurement
that would resolve it. A gate with only pass and fail forces a verdict the data
does not support, which is how `verdicts_agree: false` became `PASS`.

## 0.4 Premortem of the stopping condition

The user asked for this explicitly. Seven ways a stop gate fails, each with the
mitigation that is now part of the design.

| # | failure | mitigation |
|---|---|---|
| F1 | **The trigger is unreachable.** Exact precedent in this repo: the budget gate at `runpod_parallel.py:557-561` computes a worst case that maxes at $780 against a $2,000 threshold, so it can never fire. **[measured]** | For each gate, compute the attainable range of the trigger variable and assert the threshold lies inside it. Written as a test, not a note. |
| F2 | **It fires and is re-litigated.** v1's own Risk 4 supplies the escape hatch. | Gates name actions, not conclusions. See 0.3. |
| F3 | **Measured on an underpowered roster.** The 147-item test roster resolves nothing. | Every gate is evaluated on the 1,265-item validation roster or the 250-item ceiling set. Never the test roster. |
| F4 | **The ceiling estimate is itself unreliable.** n=20 gives ±0.193. | n=250 gives ±0.059, reported with every gate evaluation, and the cannot-decide branch fires when it is too wide. |
| F5 | **Perverse incentive not to measure.** A measurement that could trigger a stop gets deferred. | The stop-triggering measurements are sequenced first, and the pre-registration commit fixes the measurement schedule, not only the thresholds. |
| F6 | **It stops the wrong thing.** Refuting the encoder does not refute the corpus. | Each gate carries an explicit scope limit naming what it does not terminate. |
| F7 | **Nobody is accountable.** One owner, one agent, both author and reviewer. | The gate verdict is computed by code into a committed artifact before any prose interpretation, in the same shape as `gate_decision`. |

**The char-TF-IDF baseline must be built before S3 can be evaluated.** It does
not exist: no `TfidfVectorizer`, no `analyzer="char"`, nothing under `tract/`,
`scripts/`, `parsers/` or `tests/`. **[measured]** The `0.2212` v1 quoted has no
artifact behind it. Building it is a small, deterministic, CPU-only task and it
is a prerequisite of the S3 trigger, so it is a Part 0 deliverable rather than
an assumption.

---

# Part 1 — Corpus and parser contracts

## 1.1 What the corpus is

`data/processed/all_controls.json`: 4,261 controls across 31 frameworks, 3,634
(85.3%) with honest prose, defined as a description of at least 60 characters
that differs from the title. **[measured]**

| | frameworks | controls | prose |
|---|---|---|---|
| parser-backed | 19 | 3,693 | 76.4% to 100% |
| no parser | 12 | 568 | 0.0%, every one |

The twelve are `biml csa_ccm dsomm enisa etsi iso_27001 nist_800_63 nist_ssdf
owasp_proactive_controls owasp_top10_2021 samm wstg`. They contribute **615 of
the 4,127 training links (14.9%)**. **[measured]** Three properties:

1. Self-labelled: all twelve carry `version: "opencre-2026-04-28"`.
2. **No generator for them exists in this repository.** Their JSON files are
   unreproducible artifacts that `parsers/validate_all.py` validates as though
   a parser wrote them.
3. 19 of 31 frameworks have no `EXPECTED_COUNTS` entry, including all twelve.

**Anchor counts depend on how an anchor is identified, and v1 used the wrong
identity.** Keyed on the link tuple `(framework, section_id, section_name)`
there are 1,839 anchors of which 581 are multi-hub. Keyed on the **resolved
anchor text the model actually sees**, which is what the pipeline dedupes on,
there are **1,683 of which 572 are multi-hub**. **[measured, both]** Every
downstream count in v1 that derived from the tuple keying is restated against
the resolved-text keying in this document.

Provenance is self-contradictory. `framework_sources.json` records 7 sources;
`data/raw/PROVENANCE.txt` lists 13 and asserts NIST AI 100-2 and OWASP ML
Top 10 "never had a raw source," while both are on disk and sha256-recorded.
`data/raw/frameworks/` holds `nist_800_53` and `nist-800-53`, two copies of one
OSCAL catalog. **[measured]**

## 1.2 Parser contract

| defect | consequence | fix |
|---|---|---|
| nothing records what was read | `framework_sources.json` drifted to 7 of 19 | source manifest: `{path, sha256, bytes}` per input on `FrameworkOutput` |
| `_check_expected_count` warns and returns | a parser losing half its controls still writes | raise outside `COUNT_TOLERANCE`, with a written-down per-parser opt-out |
| nothing checks prose | the 568-control, 615-link poisoning | per-parser `min_prose_fraction`, checked at write time. **Not** the existing `prose_fraction`, see below |
| `fetched_date=self._today()` at `base.py:158` | output not byte-identical across days | `fetched_date` from the source manifest |
| **second clock** at `merge_all_controls.py:30` | `all_controls.json` carries `generated_date`, and its sha256 is what fold records pin and `orchestrate.py:519-532` compares across folds | remove it; the merged artifact's date comes from the newest input manifest |
| no repair stage | damaged source text reaches the encoder as tokens | `tract/parsers/repair.py`, below |
| **`expected_sha256` is absent** | `fetch_frameworks.py:151-170` warns on upstream change and then overwrites the baseline, so `verify()` can never detect it; three of seven sources point at a mutable branch head | checked-in `expected_sha256` per source; `--force` alone cannot re-baseline, a separate `--accept-new-hash` can |

**`min_prose_fraction` must not reuse the existing `prose_fraction`.** That
field records whether `ProseIndex.lookup` hit, not whether the model saw prose.
`fold_NIST_800-53_v5` records `prose_fraction: 0.0` with
`by_source: {title: 300}` while its 300 anchors have a **median length of 1,143
characters** of real OSCAL control text; `fold_ISO_27001` records the same 0.0
with a median anchor of **28** characters. **[measured]** A write gate keyed on
that field would pass thin text and block rich text. The new check measures the
stored `description` against the stored `title` directly.

`framework_sources.json` becomes generated. `PROVENANCE.txt` is **retained**,
not deleted as v1 proposed: it is the only artifact stating an *expectation*
rather than an *observation*, and `base.py:78-83` routes its own error message
to it. It becomes the human-authored expectation file that `expected_sha256` is
drawn from.

## 1.3 Repair layer

`tract/parsers/repair.py` holds named transforms a parser opts into. Each
returns an application count, and `run()` refuses to write when a repair fires
on more rows than declared.

ISO 27001, across its 93 Annex A rows **[measured]**:

| damage | rows | example |
|---|---|---|
| hyphenation breaks | 29, of which 17 corrupt the title | `secu - rity`, `seg - regated` |
| run-together tokens of 20+ chars | 22 | `Rulesfortheacceptableuseandproceduresforhandling...` |
| cell bleed across rows | 4 pairs | 5.6 ends `"...professional"`, 5.7 begins `"associations. Control ..."` |

Bleed pairs are exactly 5.6→5.7, 5.17→5.18, 7.5→7.6, 7.8→7.9. Repair moves the
fragment back at the `Control` keyword boundary.

**A count is not a diff.** The guard cannot distinguish four correct repairs
from four wrong ones, and a repair that moves text *between control IDs* can
misattribute a normative statement. Every repair therefore emits a
before/after pair into a gitignored audit file, and the ISO parser test asserts
the 4 bleed repairs against hand-checked expected output.

The join survives the title damage: ISO links match `control_id` on
`section_id` for **92 of 92**, while raw titles match only 71 of 93.
**[measured]**

## 1.4 The thirteen parsers

Order by what each one teaches:

1. **ISO 27001** — source in hand, worst damage, proves the repair layer.
2. **DSOMM** — 176 links from 13 short generic anchors at 13.5 links each, all
   `AutomaticallyLinkedTo`. Worst: `Infrastructure Hardening` (25),
   `Education and Guidance` (22), `Monitoring` (18). **[measured]**
3. **Clean GitHub markdown** — WSTG (118), SAMM (30), Top 10 2021 (16),
   Proactive Controls, CSA CCM (29). CSA CCM confirmed freely redistributable.
4. **PDF-derived** — NIST SSDF (46), ENISA (59), ETSI (35), BIML (14),
   NIST 800-63.
5. **OWASP LLM Top 10 2026** — see 1.6.

**Eleven of the thirteen have no raw source on disk.** `data/raw/frameworks/`
contains only `iso_27001` among them. **[measured]** A fetch step is a
prerequisite of the parser step and appears in the sequencing table; v1 omitted
it.

### ISO licensing, resolved

`LICENSE` is CC0 1.0 Universal, `.gitignore` covered only `data/raw/`, and 40
files under `data/processed/` were tracked. **[measured]** CC0 is an
affirmative grant asserting the publisher holds the rights and waives them, and
`git push` is the publication event that fires before any publish-path filter.
v1 scoped the ISO control to the publish path and left the repository open.

**Resolved 2026-08-15 (owner decision):** `data/processed/frameworks/iso_27001.json`
is gitignored and untracked. The tracked version held 93/93 title-only stubs,
so no licensed prose ever entered git history and nothing needs purging.
Enforcement is `tests/test_licensed_text_not_tracked.py`, which asserts the
file is not tracked and that a tracked `all_controls.json` carries no
restricted-license prose. A gitignored `data/processed/licensed/` overlay
carries the full merged corpus for training.

**Open, and it blocks the OWASP 2026 parser rather than publication:** the repo
is CC0 while `tract/dataset/bundle.py:221` publishes the dataset as
CC-BY-SA-4.0, and OWASP 2026 is CC BY-SA 4.0 whose ShareAlike and attribution
terms CC0 waives. Three license postures for one corpus. Reconcile before that
parser lands.

## 1.5 Retiring both gates

Both `PHASE1B_DROPPED_FRAMEWORKS` and `PHASE1B_MIN_SECTION_TEXT_LENGTH = 10`
test `link["section_name"]`, a title. Reproduced exactly: 278 links dropped,
155 by framework list and 123 by short title, concentrated in capec 44,
dsomm 38, cwe 17. **[measured]** The gate moves to the resolved anchor.
Recovers all 278, and re-bases every metric, so it lands in one commit with
counts recorded before and after.

## 1.6 OWASP LLM Top 10 2026 as a contamination control

Never parsed, never trained on, never in any fold, and it stays that way. Not a
LOFO fold: ten units cannot resolve anything and macro averaging would give
them a fifth of the headline. Not a replacement for the 2025 entry: OpenCRE's
13 links are keyed to `LLM0x:2025` and 2026 is a permutation with one rename.
It lands as `owasp_llm_top10_2026`.

Value: a pretraining-contamination control (BGE-large-v1.5 predates it), a
blind set for the panel, and an independent check on OpenCRE coherence. The
answer key is Appendix A **[measured]**: 48 expert LLM2026→CWE mappings over 22
distinct CWEs, all 22 resolving to our CWE 4.20 corpus, 17 carrying OpenCRE
links, giving **37 transitive chains over 46 hubs covering all 10 risks**.

Parser must stop at `## Appendix A` or LLM10 swallows 937 lines. The appendix
parses as a separate mapping artifact, never as control text.

## 1.7 Ground-truth divergence

`tract/cli.py:1662` feeds `import-ground-truth` the uncurated
`hub_links_by_framework.json` (4,406) while training reads the curated file
(4,405, filtered to 4,127). The curated grouped file sits unused beside it.
One path change plus a test that both readers agree on a count. **[measured]**

## 1.8 Review status does not survive a corpus rebuild

`review_status` is a column on the assignment; control text lives in a joined
table. `tract/crosswalk/export.py:42` gates publication on that flag alone.
Part 1 rewrites control text for 568 controls that were **titles at review
time** — a reviewer judging a three-word label did not evaluate the assignment
the rebuild will publish.

Fix: store `reviewed_control_sha256` beside `review_status`. On ingest, null the
review status of any assignment whose control text hash changed. The 568
title-reviewed controls get a re-review budget line in the sequencing table.

## 1.9 Acceptance tests

- Every file in `data/processed/frameworks/` has a parser. Orphans fail.
- No framework reports 0% honest prose, measured on stored text not on the
  join-path flag.
- No `version` field contains `opencre-`.
- Parse twice, bytes identical. **Scoped:** this asserts re-parsing the same
  input bytes, not re-fetching. `data/raw/` is gitignored so CI cannot run it;
  it is a local gate and the spec says so rather than implying CI coverage.
- Recorded `by_source` matches the actual anchor-length distribution.
- Both link readers agree on edge count.
- `owasp_llm_top10_2026` appears in no training or fold roster.
- No restricted-license prose in any tracked artifact.

---

# Part 2 — Target enrichment

## 2.1 Plumbing

**[measured]** `build_firewalled_hub_text` produces `f"{path} | {name}"`.

- `name` is the last path segment for **522 of 522** hubs; the duplicate is a
  median 23.2% of the target string.
- The five root targets are literally `X | X`. 24 training links point at roots.
- `children_ids` populated for 122 hubs, `related_hub_ids` 51, `sibling_hub_ids`
  498 with median 5.
- Sibling targets share a median content-token Jaccard of 0.48.
- Branch sizes are lopsided: `Technical application security controls` holds
  361 of 522 (69.2%).
- **Correction to v1:** `label_space` is **not** dead code. It is consumed at
  `generate_descriptions.py:129,137,260,306`, `validate_descriptions.py:44`,
  and `scripts/phase0/common.py`. It is unused in the *training* path only, and
  that path is not the whole repo. `hierarchy.py:246` is `logger.warning`, not
  an assertion as v1 stated.
- Fold artifacts still do not record the hub-text configuration:
  `include_description` appears in zero files under `results/`. Fixed before
  the bake-off or the bake-off yields unattributable numbers.

## 2.2 The existing descriptions leak

`tract/descriptions.py:56` builds the prompt from hub name, path, siblings, and
`linked_section_names`, up to 50, with no framework exclusion. **[measured]**

| held-out fold | leaf hubs affected | contains a held-out section name verbatim |
|---|---|---|
| owasp_ai_exchange | 58 | 21 |
| mitre_atlas | 34 | 2 |
| nist_ai_100_2 | 22 | 6 |
| owasp_llm_top10 | 7 | 2 |
| owasp_ml_top10 | 4 | 1 |
| **total** | **125** | **32** |

The substring firewall catches 32 of 125; the other 93 are paraphrase. The
artifact stays valid for the deployed model and is marked deployment-only.

## 2.3 What already exists on disk

**[measured]** 400 descriptions covering the 400 leaves exactly. Median 775
characters, range 544 to 1,164. `claude-opus-4-20250514` at temperature 0.0.
382 accepted, 18 human-edited. Only 75 of 400 repeat the hub name. **390 of 400
carry an explicit exclusion clause** and 71 name the sibling owning the
excluded scope. None of it has ever been used.

## 2.4 The three artifacts, and what blocks them

- **`hub_semantics.json`** — regenerated for all 522 from name, path, siblings,
  children. Firewall-safe by construction. Adds the 122 internal nodes, which
  replaces the `X | X` root targets with content.
- **`hub_contrast.json`** — exclusion clauses. Feeds negative mining only.
- **`hub_evidence/<excluded_framework>.json`** — per-fold, firewall-bound.

**Three blockers v1 missed:**

1. **The generator refuses non-leaf descriptions.**
   `generate_descriptions.py:267` raises `ValueError` on exactly what 2.4 asks
   for. **[measured]** The generator must be changed before the 122 internal
   nodes can exist.
2. **`hub_contrast.json` is derived from the contaminated 400.** Its honest
   `derived_from` is every linking framework, so the Part 3.6 assertion rejects
   it on every fold — while Part 2.6 lists it as a negative source. Resolution:
   contrast is generated from `hub_semantics.json` with the structure-only
   prompt, **not extracted** from the 400. `derived_from` is a computed
   transitive union of inputs, never a declared literal.
3. **Model change confounds the 4.5 comparison.** X was written by Opus via the
   Anthropic API. A RunPod H100 does not serve Opus. Either the regeneration
   uses the same API (and an Anthropic key must not go to a rented host, the
   failure already fixed for the HF write token in `6c3ae46`), or the generator
   changes and "the only variable is the prompt" is false. **Resolution:**
   regenerate via the API from the operator's machine, which is a description
   generation call and not a model load, so it does not violate the RunPod rule.

## 2.5 The label space stays at 522

**[measured]** Of 1,683 resolved-text anchors, **294 (17.5%)** have no leaf in
their gold set, 23 are root-only, and 25 mix granularity. Only 9 of 572
multi-hub anchors have two golds in an ancestor relation, so multi-hub mapping
here is lateral. A leaves-only space would strand 17.5% of anchors and require
inventing ground truth for **310** links. (v1 reported 389/21.2% and 447 using
the tuple keying.)

**Ancestor-consistent training signal, defined.** v1 named this and never
specified it, while `data.py:404-418` keys its batch-collision guard on exact
`hub_id`, so a parent and its child in one batch make each an explicit negative
for the other. Given 294 anchors gold on internal nodes, that is the normal
case. The definition: extend the collision guard so a hub and any ancestor or
descendant of it cannot both appear as positives in one batch, and exclude
ancestor/descendant pairs from negative sampling. This is a sampler change, not
a loss change, and it is testable without a GPU.

## 2.6 Negatives, re-aimed at what the baselines actually show

**v1's re-aiming is withdrawn.** Siblings and ancestors are 9.3x and 9.5x over
chance; cross-branch is 0.7x, better than chance. The miner's target region was
right. Three changes, all quality rather than aim:

1. **Delete the empty-string negatives.** 436 of 12,381 slots. `MNRL` treats
   every negative column as a document to rank against, so the model learns the
   empty string is a plausible wrong answer. Drop the column, do not pad it.
2. **Add ancestor and descendant negatives**, currently absent from a miner
   that takes siblings then cousins, against a 9.5x-over-chance error class.
3. **Exclude co-occurring hubs from negative sampling** — with the Part 4.1
   caveat that this is the change most at risk of leaking the answer key, so it
   is gated on the disjointness test there.

---

# Part 3 — Leakage and evaluation

## 3.1 Hypotheses for the ASVS result

**H1, lexical shortcut.** v1 filed this "weakened" on a quartile table. That
verdict is withdrawn: the overlap covariate ranges 0.091 to 0.786 with only 8
of 277 items below 0.20, so there is **no low-overlap stratum** in which to
falsify a shortcut hypothesis. **[measured]** Overlap significantly predicts
zero-shot correctness (point-biserial r = 0.167, p = 0.005) and does not
predict trained correctness (r = 0.042, p = 0.49), which is the signature H1
predicts. **H1 is live and is currently the best-supported hypothesis.**

**H2, genuine skill destroyed.** Live.

**H3, training-distribution capture.** Refuted. Predicted hubs carry a median
of 5 training links against 6 for correct hubs. **[measured]**

**H4, representational collapse.** **Demoted from v1's "leading candidate" to
unsupported.** The 0.52 ratio is a denominator artifact: **[measured]**

| fold | n | distinct gold | distinct pred | ratio | modal share |
|---|---|---|---|---|---|
| ASVS | 277 | **277** | 145 | 0.52 | **0.036** |
| CAPEC | 349 | 53 | 129 | 2.43 | 0.049 |
| CWE | 246 | 115 | 131 | 1.14 | 0.049 |
| ISO 27001 | 93 | 47 | 54 | 1.15 | 0.065 |
| NIST 800-53 | 300 | 66 | 116 | 1.76 | 0.060 |

ASVS is the only fold with a unique gold per item, so its denominator is fixed
by construction. Its numerator is in line with the other folds and it has the
**lowest** prediction concentration of the five, the opposite of collapse.
Under uniform random guessing the expected distinct-prediction count on ASVS is
`522·(1 − (1 − 1/522)^277) = 215` **[derived]**, so 145 is below random and
worth a null-model test, but it is not the leading hypothesis and it no longer
gates the bake-off.

## 3.2 Two rosters

| roster | folds | n |
|---|---|---|
| `lofo_*` test | ATLAS 43, NIST AI 100-2 28, OWASP AIX 63, OWASP LLM **6**, OWASP ML **7** | **147** |
| `c2_*` validation | ASVS 277, CAPEC 349, CWE 246, ISO 27001 93, NIST 800-53 300 | **1,265** |

The 6- and 7-item folds merge into one 13-item OWASP fold.

## 3.3 The gate, and what the record says

**[measured]** `results/phase1b/lofo_title_only/aggregate_metrics.json`:

```
threshold 0.1 | micro_delta 0.1293 | ci_low 0.0408
point_estimate_pass true | ci_low_pass false | verdicts_agree false
```

Uncommitted additions: `n_configurations: 4`, `selection_optimistic: true`,
`ci_low_familywise: 0.0204`, `familywise_pass: false`. `PRD.md:378` carried
only the point estimate and **spliced two runs**: its 0.531 headline is the
re-derivation while its per-fold deltas belong to the original run and imply
0.537. Both are withdrawn, with errata live.

**Campaign 2 pre-registered five arms and ran two.** `CAMPAIGN2.md` fixes
A1–A5; only A1, A2 and a canary exist on disk. **[measured]** A5 was
`title-only` on validation — the one comparison Part 5 needs. Its absence is
recorded here, and re-running it is a Part 5 prerequisite, not an optional
extra.

## 3.4 hit@1 is multi-label credit

**[measured]** Scored against the single recorded gold instead of
`valid_hub_ids`: CAPEC `0.175 → 0.032`, CWE `0.256 → 0.126`. ASVS, ISO and NIST
unchanged. On CAPEC, multi-label credit supplies 82% of reported hit@1.
**[derived]** Both numbers get reported, and Part 0.2 makes hit@5 primary.

## 3.5 Echo stratification

**[measured]** NIST 800-53: 23% echo, 0.426 vs **0.091** non-echo on 232 items.
ISO: 14% echo, 0.923 vs 0.300. ASVS: 5% echo, 0.267 vs 0.282.

## 3.6 Provenance firewall

The substring check is real since `6548703` and has poison tests, but it
catches only verbatim copies, and it has two total-bypass conditions:
`firewall.py:147` skips controls under 5 characters and `:162` exempts a
control from **every** hub check when its text is a substring of any hub name.

**The provenance check must be computed, not declared.** v1's `derived_from`
would be written by the same code that assembles the prompt, so one bug
produces the leak and the clean attestation together. Requirements:

- `derived_from` is a **computed transitive union** of the input records'
  `derived_from`, never a literal.
- Each artifact carries a sha256 over its concatenated inputs; the consumer
  recomputes and compares.
- Artifacts generated on a pod are verified against a pod-emitted manifest
  after `_rsync_from`, before entering `data/processed/`.
- The `:162` exemption narrows to exact equality with a hub name.
- Substring scanning is retained as a second line, with its bypass conditions
  documented as known limits rather than treated as coverage.

## 3.7 Evaluation changes

Provenance firewall as above. Minimum fold size. Zero-shot as a mandatory
column. Dual hit@1 and hit@5. Echo and depth stratification. **Per-fold
heterogeneity reported with every pooled estimate** (Cochran Q and I²), because
a pooled delta over I² = 95% is not a summary. Gate pre-registered in its own
commit.

**The MDE must be restated.** SD of the paired per-item delta is 0.4921 on
n = 1,265, so SE = 0.01384. **[measured]** At 80% power: **[derived]**

| regime | MDE |
|---|---|
| one-sided, uncorrected α = 0.05 | 0.034 |
| two-sided, uncorrected | 0.039 |
| **two-sided, Šidák k = 12** | **0.051** |
| macro, two-sided, Šidák k = 12 | 0.057 |

v1's "near 3.5 points" is the one-sided uncorrected figure, which is the one
its own familywise requirement forbids. The design is sized at **0.051**.

**Pre-registration needs an enforcement point.** `n_configurations` is an
argparse flag defaulting to 1, read at aggregate time, and
`orchestrate.py:788` calls `gate_decision` with the default. **[measured]**
Nothing reads a pre-registration file. Fix: `aggregate` refuses to run without
a pre-registration path whose declared arm count matches the arms found on
disk, and `check_publication_gate` refuses to build a card whose fold records
show `verdicts_agree: false`.

## 3.8 Published numbers

Errata live: model card `379e2e2`, dataset card `88b963b`. Repo commit
`4cb876a` annotates `README.md` and `PRD.md:378` with original text retained.

**A second correction was required the same day.** The withdrawal note claimed
"every assignment in the published crosswalk was reviewed by a human expert."
That is false: most rows are imported OpenCRE ground truth, and the reviewed
portion is 878 model predictions judged by **one** reviewer with 13/20
calibration agreement. **[measured]** Corrected on the dataset card at
`57930dc` and in `README.md`, original claim quoted. The lesson recorded here:
when a load-bearing claim is withdrawn, re-derive what the artifact now asserts
rather than reaching for the nearest surviving claim.

**The fine-tuned model is still the `tract assign` default** (`config.py:405`)
while the evidence says it may not beat its own base model. Interim action:
repoint the default to the base encoder or make the fine-tuned pin opt-in with
a stated reason. This is a Part 7 deliverable, not a rebuild outcome.

---

# Part 4 — The hub graph

## 4.1 The circularity that has to be resolved first

`build_evaluation_corpus` forms `valid_hub_ids` by unioning hubs across items
sharing a control text. The co-occurrence graph unions hubs across anchors.
**These are the same relation over the same anchors.** A graph-conditioned
configuration would be handed the metric's answer key, and the confirmation run
would not catch it because the AI folds contribute almost no edges.

**Gate, before any graph feature enters a model:** measure the overlap between
graph edges and multi-label-credit pairs on the validation roster. **[unmeasured]**

- If overlap is low, graph features proceed.
- If overlap is high, either graph features are excluded from any configuration
  selected on multi-label hit@k, or selection moves to **single-gold** hit@5,
  which is immune by construction.

Single-gold selection is the cleaner resolution and is the default unless the
overlap test clears the graph.

## 4.2 The graph is thin because we made it thin

**[measured]** Three code paths discard it: `hierarchy.py:76` consumes only
`ltype == "Contains"`; `extract_hub_links.py:44` skips non-Standard doctypes so
CRE-to-CRE links never reach an artifact; `related_hub_ids` is written only by
`tract/bridge/review.py`. Current graph: **517 Contains + 46 bridges = 563
edges over 522 nodes.**

Recoverable lateral edges, keyed on the tuple identity **[measured]**: 1,449 at
w≥2, 903 at w≥3, 595 cross-reference, 18 overlapping. Keyed on resolved text
the co-occurrence counts are 1,412 and 886. Both are reported because the
keying choice is a design decision, not a discrepancy.

## 4.3 LOFO retention

**[measured]** All five AI folds retain 99.9% to 100%. ASVS, NIST 800-53 and
ISO retain 100%, CWE 97.2%, and **CAPEC retains 42.9%** (41.4% under
resolved-text keying), losing 828 edges. Any graph-based method underperforms
on CAPEC for reasons unrelated to the method. Declared in the pre-registration;
CAPEC reported separately.

## 4.4 `hub_graph.json`

| edge type | count | `derived_from` | LOFO |
|---|---|---|---|
| `contains` | 517 | CRE-native | always |
| `cooccurrence` | 1,412 at w≥2 | frameworks producing it | drops per fold |
| `crossref` | 595 | structure-only regeneration | always |
| `bridge` | 46 | **all frameworks** | **drops every fold** |
| `opencre_related` | **[unmeasured]** | CRE-native | always, pending fetch |

`bridge` edges came from `tract bridge`, which ran the trained model over every
framework, so they are contaminated by construction.

## 4.5 The regeneration and edge-set comparison

The clean set ships regardless. Four corrections to v1's design:

1. **Power statement required.** X has roughly 657 lateral cross-reference
   edges over 308 source hubs, mean 1.64 per hub. Exposure is constant within a
   cluster, so the design effect is roughly 1.6 and the detectable odds ratio
   at 80% power is about **1.3 per SD of exposure**. **[measured/derived]** The
   read-out states that bound. An interval covering 1 means *underpowered or
   null*, and the two are not the same.
2. **Exposure is identity, not dose.** Contamination is per-fold: a hub with 3
   OWASP AI Exchange links is fully contaminated for that fold. The test is
   run per-fold on the contaminated set, not on total link count.
3. **Cluster-robust intervals**, since exposure is constant within source hub.
4. **"Only the prompt changes" is false** and the design says so. See 2.4
   blocker 3.

---

# Part 5 — Architecture bake-off

## 5.1 Prerequisites, all blocking

1. **Part 0 stop gates evaluated.** S2 and S3 can end this part before it runs.
2. **Arm A5** (title-only on validation) run, closing the gap that forced v1 to
   pick an anchor representation from the test roster.
3. **The 4.1 circularity gate** resolved.
4. **`hub_evidence/` and `hub_contrast.json` generated.** v1's sequencing table
   produced neither, so six of twelve cells had no input. The generation step
   is costed on the **validation** roster, not the 147-pair test roster.
5. **L2 fully specified** — base checkpoint, objective, loss, candidate-list
   size K, and hyperparameters. v1's entire L2 specification was a three-item
   list of architecture families, which cannot support a pre-registered
   familywise correction over cells that are not yet defined.

## 5.2 The anchor decision is unresolved

v1 chose prose for all 12 configurations. Paired on the same 147 items
**[measured]**:

| k | title | prose | diff | McNemar p | paired bootstrap 95% CI |
|---|---|---|---|---|---|
| 1 | 0.517 | 0.422 | **-0.095** | 0.059 | **[-0.191, -0.007]** |
| 5 | 0.728 | 0.707 | -0.020 | 0.728 | [-0.095, +0.054] |
| 10 | 0.782 | 0.803 | **+0.020** | 0.690 | [-0.048, +0.088] |

The hit@10 advantage is 3 net items with a CI four times its width. The only
interval excluding zero favours **title**, at rank 1. v1 selected on the
weakest, least significant difference, on the roster it forbids for selection.
**Decision deferred to arm A5 on the validation roster.**

## 5.3 The ceiling is unmeasured and v1 computed it from the wrong model

hit@k, single-gold **[measured]**:

| roster | arm | hit@1 | hit@5 | hit@10 |
|---|---|---|---|---|
| validation, n=1,265 | c2_A1 | 0.163 | 0.349 | 0.464 |
| test, n=147 | title-only | 0.517 | 0.728 | 0.782 |
| test, n=147 | prose+sw | 0.422 | 0.707 | 0.803 |

v1's "a perfect reranker over the top 10 reaches 0.464" uses the **trained**
model's candidate list, while Part 5 makes zero-shot the comparator. Zero-shot
multi-label hit@5 on ASVS is **0.866** against the trained model's 0.534.
**[measured]** recall@50 is measured for **both** models.

CAPEC hit@10 is 0.232 on 349 items, 42.8% of training links. If recall@50 does
not move it, no reranker will.

## 5.4 The matrix

Three L2 architectures by four target representations, 12 cells, 5 folds.
**Selection on the 1,265-item validation roster. The 147-item test roster is
touched once, by the winner.** Zero-shot BGE is the comparator in every cell.
Sized at MDE 0.051, Šidák k = 12.

**Cost is [unmeasured] and v1's figure was wrong.** The "$90 and ~6 GPU-hours"
was copied from `CAMPAIGN2.md`, which costed **30** folds. Measured
`elapsed_s` across the committed validation folds averages 1,417s, so 60 folds
of that shape is roughly **23.6 GPU-hours** **[derived]**, and L2 adds a second
training stage per cell. Fold cost already varies more than 2x by encoder
(3,404s Qwen vs 1,671s BGE on the same fold). A costed estimate is produced
after L2 is specified, before any fleet launches.

## 5.5 Panel

Two load-bearing checks: whether a lexical baseline wins on semantic
correctness, and whether OpenCRE's own links are coherent. Both feed S3.

**The panel is a dependency, not an instrument.** Every model gets a resolved
commit sha in an `EncoderSpec`-style record, `trust_remote_code=False`
explicitly with any model that will not load under it dropped, and
`panel_revision` in the evidence cache key is the concatenation of those shas
rather than a hand-typed label. **The orchestrator cannot currently provision
the panel:** it needs 6 to 8 H100s and `create_pod` hardcodes `gpu_count=1`
with only `22/tcp` exposed. **[measured]** A multi-GPU provisioning path is a
prerequisite, and the "one pod, under $100" line in v1 was wrong.

---

# Part 6 — PRD amendment

## 6.1 The spec is a root cause

`§6.5` scopes the firewall to removing section names from the linked-standards
list; descriptions were generated from those same names, outside that scope.
`§12` pre-registers a gate with no statement of point estimate versus interval.
`§6.11` fixes LOFO to the five AI frameworks, so the underpowered evaluation is
specified rather than accidental.

## 6.2 Amendments

`§3` gains the L1/L2/L3 pipeline and 6.3 below. `§4.5`/`§4.6` go to 33
frameworks and record that 12 of 22 were never parsed. `§4.8` gains the source
manifest, `min_prose_fraction`, and the repair layer. `§6.1` gains
`hub_graph.json`. `§6.2` forbids framework text in the description prompt.
`§6.4` gains the target ladder and the restated gate. `§6.5` becomes the
provenance firewall. `§6.11` gains the roster split, dual metrics and
stratification. `§12` gains the interval criterion, the familywise correction,
and **the Part 0 stop gates**. New `§16` holds the experiment ledger and the
no-unregistered-numbers rule. Superseded claims keep original text with dated
annotations.

## 6.3 Why the L2 cross-encoder is not the forbidden pairwise pattern

`§3` says `g(control_text) -> CRE_position`, never `f(A,B)`. An L2
cross-encoder takes two strings, and a reader pattern-matching on the signature
will delete it.

**The test:** if you removed every framework except the one being mapped, would
the function still have a second argument? For the cross-encoder yes, because
the 522 hubs are CRE-native. For pairwise no.

**Scaling.** Pairwise over the corpus is `4,261 × 4,260 / 2 = 9,075,930`
comparisons; assignment is `4,261 × 522 = 2,224,242`, and `4,261 × 50 = 213,050`
after L1. **[derived]** The point is growth, not the constant: onboarding a
100-control framework costs `100 × 4,261` pairwise **and rises with every
framework added**, against a fixed `100 × 522` by assignment.

**Generalization, the stronger reason.** Pairwise needs
`(control_A, control_B, label)` supervision that does not exist for an unmapped
framework. The 4,127 OpenCRE links are `(control, hub)`. L2 trains on the same
links in the same shape and introduces no new supervision requirement.

**The invariant:**

> No function in the assignment path may accept two framework-derived texts.
> The second argument of any scoring function is a CRE-native representation.
> Framework text appears in the first argument only.

Enforced by a test on the type signature, which is the provenance firewall
applied at the function boundary.

**What this does not license.** Deriving that A and B are equivalent because
both map to hub X remains correct. Scoring A against B to decide it remains
forbidden. A reranker conditioning on other frameworks' controls mapped to the
same hub is the forbidden pattern in a hub-shaped hat and is not covered here.

## 6.4 Out of scope

The assignment paradigm, LOFO-only, no pairwise metrics, auto-links as
expert-quality, `data/raw/` immutability, CLI and API only. Completed phases 2
through 5B get their outputs rebuilt; their designs stand — **except** the
review-status question in 1.8 and the serving path in Part 7, both of which are
now in scope.

---

# Part 7 — Serving path

v1 had no equivalent and this is why a bake-off winner could not ship.

`tract/inference.py:216-224` scores by cosine against a frozen
`hub_embeddings` matrix, calibrates with `t_deploy`, thresholds OOD on
`max_sim`, and builds conformal sets from `conformal_quantile`. **All three
constants were fit on raw bi-encoder cosines.** An L2 score is not a cosine, so
calibration, OOD detection and conformal coverage are all undefined under a
reranker, and CLAUDE.md's "always calibrate before reporting confidence" is
violated by construction. `deployment_artifacts.npz` holds precomputed hub
embeddings, so a two-stage pipeline has no artifact to load.

Deliverables: a two-stage inference path, recalibration on L2 scores, an OOD
criterion defined on the reranker, conformal coverage re-derived, and the
`tract assign` default question from 3.8. Chartered as the **fifth**
implementation plan.

---

# Sequencing

Each row states what it gates **and what it invalidates**, which v1 omitted for
steps 3 through 6.

| # | step | pod | trains | gates | invalidates |
|---|---|---|---|---|---|
| 0a | fetch the 11 missing raw sources | CPU | no | every parser | — |
| 0b | build the char-TF-IDF baseline | CPU | no | S3 | — |
| 1 | corpus rebuild, 13 parsers | CPU | no | everything downstream | **the ceiling study, the graph, and every eval item; nothing that reads the anchor set may precede it** |
| 2 | fresh OpenCRE fetch, all CRE-to-CRE relations | CPU | no | `opencre_related` | the hierarchy, so it precedes 4 and 5 |
| 3 | **250-item blind ceiling study** | none | no | S1, and the meaning of every metric | — |
| 4 | build `hub_graph.json` + the 4.1 circularity gate | CPU | no | any graph feature | — |
| 5 | regenerate 522 descriptions, contrast, evidence | API | no | T1, T2, T3 | — |
| 6 | recall@50 both models, ASVS diagnostic, arm A5 | fleet | A5 only | S3, the anchor decision | — |
| 7 | panel first pass | multi-GPU | no | S3, and can end the encoder path | — |
| 8 | 12-config bake-off | fleet | yes | selects one winner | — |
| 9 | winner on the test roster, once | 1 | no | final | — |

**Ordering correction, twice over.** v1 ran the graph build and the description
regeneration before the corpus rebuild, which rewrites the anchor set both
depend on. v2's first draft then made the same class of error one section over
by scheduling the ceiling study at 0b: that study draws 250 items from the
validation roster, and step 1 turns 615 title anchors into prose and restores
278 dropped links, so a study run first would measure a roster that ceases to
exist. It is the single most expensive step in owner time, so running it twice
is the worst outcome available. It now sits at step 3, after both the corpus
rebuild and the OpenCRE re-fetch.

The general rule this keeps violating, stated once so it can be checked: **no
step may precede anything that rewrites its inputs.** The `invalidates` column
exists to make that checkable, and every future amendment must fill it in.

Steps 0a through 5 involve no GPU training. Steps 0b, 3, 6 and 7 can each end
the programme under a Part 0 gate.

# Budget and placement

Roughly $1,850 of $2,000 remains. **Every cost figure in v1 was wrong or
uncosted and none is restated here as fact.** The bake-off is **[unmeasured]**
pending L2 specification; the measured fold time implies roughly 23.6 GPU-hours
for 60 folds of the current shape, which is 4x v1's claim. The panel is
**[unmeasured]** and needs 6 to 8 H100s that the orchestrator cannot yet
provision.

Three infrastructure defects must be fixed before any fleet launches:

- **`reap` cannot see a validation-split pod.** Its orphan sweep is built from
  `POD_CONFIGS`, which uses test-roster names, while validation pods are named
  `tract-p1b-val-*`. **[measured]** The recovery command reports all clear
  while five H100s bill.
- **No cumulative spend accounting**, and the per-provision gate is
  unreachable: worst case maxes at $780 against a $2,000 threshold.
  **[measured]**
- **No fold-level resume.** A roster guard rejects any `--folds` subset, so one
  failed fold costs a full five-pod re-run, and `_rsync_from` without
  `--delete` can silently mix two runs into one aggregate.

Everything that loads a model runs on RunPod. Description regeneration is an
API call from the operator's machine, which loads no model locally and keeps an
Anthropic key off rented hosts.

# Determinism

Two clocks, not one: `base.py:158` and `merge_all_controls.py:30`. Both removed
in favour of manifest dates. Seeds set explicitly. Graph extractor deterministic
and sorted output. Every hub-side artifact carries a computed `derived_from`.
ML stack on the pinned, digest-locked environment from the 2026-08-14 spec.
Panel weights pinned by commit sha.

**Scope of the byte-identical claim:** re-parsing the same input bytes, not
re-fetching. `data/raw/` is gitignored so CI cannot run that test.

# Risks

1. **The ceiling may be low.** If α₅ comes back near 0.65, much of what this
   project has called model failure is label noise, and S1 fires early. This is
   the most likely single outcome to change the programme. It is cheap in money
   and expensive in owner time, which is why it runs as early as it correctly
   can, at step 3, immediately after the corpus rebuild that determines which
   items it would be judging.
2. **Recall@50 may be low**, in which case Part 5 is revised rather than
   executed.
3. **CAPEC may be unreachable at any k.** 42.8% of training links, hit@10 of
   0.232.
4. **The graph may be inseparable from the answer key**, in which case
   selection moves to single-gold hit@5 and the graph's value drops to
   negatives only.
5. **The panel may refute the encoder paradigm.** That fires S3, whose action
   is stated, so it cannot be re-filed as a finding.
6. **The corpus rebuild re-bases every metric.** Deliberate, handled by
   before/after counts in one commit.
7. **Scope.** Seven parts. Mitigated by five implementation plans with
   checkpoints.

# Open items

1. Three license postures for one corpus: repo CC0, dataset CC-BY-SA-4.0,
   OWASP 2026 CC BY-SA. Reconcile before the OWASP 2026 parser lands.
2. `opencre_related` yield unknown until the fetch runs.
3. The 4.1 overlap number is unmeasured and gates Part 4.
4. L2 specification, which gates both the bake-off cost and its
   pre-registration.
5. This spec is gitignored at `.gitignore:25`, as is `results/`. A
   pre-registration that is not in version control is a note, not a
   pre-registration. `CAMPAIGN2.md` shows the `git add -f` precedent.

# Appendix — premortem disposition

Six perspectives, 36 findings, none dropped below Plausible. Accepted and
incorporated: the baseline-corrected error taxonomy, the H4 demotion, the
graph/answer-key circularity, the missing artifact generation steps, the
serving-path gap, the CC0/ISO exposure, the AIBOM pin, `reap` naming, the
budget-gate unreachability, fold resume, the MDE restatement, the anchor
re-keying, the `label_space` correction, the second clock, the char-TF-IDF
absence, the truncated Campaign 2, the anchor-decision reversal, review status
surviving the rebuild, panel pinning and provisioning, source-provenance TOFU,
`PROVENANCE.txt` retention, and the dataset PII scan gap.

**Tail-risk, retained rather than dropped.** Prompt injection through
`llm_extractor.py:182`, where an external document is the entire user turn.
Scored Unlikely: exploitation requires landing a hostile PR in an OWASP or
MITRE repository against a research crosswalk with no money attached. Trigger
that would raise it: any framework source accepting community contributions
without maintainer review, or any move to auto-ingest.
