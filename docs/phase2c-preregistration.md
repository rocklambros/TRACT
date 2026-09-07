# Phase 2C pre-registration — traditional-control → AI-hub bridge links

**Committed 2026-09-04, before a single control has been read by an annotator.**

This document is binding. It exists because this project has recorded three
passes that were later withdrawn or amended, and in at least one of them the
clause designating the verdict was written after the arm results existed. The
gates below are declared before the packet is sent, and the numbers they are
measured against are all reproducible from committed artifacts today.

---

## 1. What Phase 2C is, and the fact that motivates it

The AI and traditional hub regions are **disjoint**.

| region | hubs |
|---|---|
| AI-only | **78** |
| traditional-only | **380** |
| naturally bridged | **0** |

Measured over the 8-framework AI region (`BRIDGE_AI_FRAMEWORK_IDS`) against
`data/training/hub_links_by_framework_curated.json`. Reproduce with:

```bash
python -m scripts.analysis.orphan_rate
# AI hubs with no traditional supervision: 78 of 78 (100.0%)
```

So a model trained on the curated set never sees a traditional control
positioned against an AI hub. The PRD's bridging capability has no supervision
behind it, and under the strict all-AI firewall — every AI framework held out —
there is no trainable task at all, because all 78 gold hubs are orphaned.

Phase 2C buys that supervision by human annotation. The links are **Tier 2**
under `results/phase1b/CAMPAIGN3.md` §2: independently human-authored, with no
model output shown to the annotator.

### A note on the AI-framework definition, because the wrong one looks right

`scripts/phase0/common.AI_FRAMEWORK_NAMES` is the **five-framework LOFO eval
roster**. `BRIDGE_AI_FRAMEWORK_IDS` is the **eight-framework AI region**, adding
ENISA, ETSI and BIML. Measured both ways:

| definition | AI hubs | apparently bridged |
|---|---|---|
| 5-framework roster | 73 | **57** |
| 8-framework region | 78 | **0** |

The narrow reading suggests most AI hubs already have traditional supervision,
which would make this phase largely unnecessary. It is wrong: all 57 apparent
bridges are supplied by ENISA (51), ETSI (28) or BIML (11) — every one an AI
framework. `tests/test_orphan_rate.py` pins both numbers so a change of constant
is a deliberate act rather than a silent rebaseline.

---

## 2. Gate 1 — free. Graph arithmetic, no model, no pods.

**PASS iff the strict-firewall orphan rate falls from 78/78 to ≤ 55/78** — at
least **23** AI hubs given real non-AI supervision.

Reachable only because the annotator sheet carries **all 78 hubs, unranked**
(D1). Under the superseded 20-hub scope a flawless annotator reached 20 and
failed: a link carries one `cre_id`, so 20 hubs is a hard ceiling below the
threshold. That scoping also ranked by "eval weight" — appearances as gold in
the held-out split — which is a selection rule derived from the test set.

### Four quality conditions, all pre-registered, all checkable without a model

Gate 1 counts hubs, so it is gameable by volume. All four must hold:

| | condition | rationale |
|---|---|---|
| Q1 | **≥ 40 distinct NIST 800-53 controls** contribute ≥ 1 accepted link | 23 links from one control is not a sweep |
| Q2 | **≤ 6 AI hubs per control** | a control mapping to a third of the AI region is a judgement about the region, not the control |
| Q3 | **confidence ≥ 2** on the 1–3 scale for a link to count toward Gate 1 | low-confidence links are data, not evidence |
| Q4 | **≥ 15% of controls double-annotated**, agreement rate reported | not a threshold — a number that must exist and be published |

**All five are computed by `scripts/analysis/gate1_report.py`**, whose verdict
is their conjunction. Thresholds are named constants in `tract/config.py`
(`PHASE2C_GATE1_MAX_ORPHANS` and siblings) and a test asserts each appears in
this document, so the gate and the pre-registration cannot drift apart.

Q3 is applied **before** the orphan count, not reported beside it: a link this
document calls "data, not evidence" must not de-orphan a hub. Q2 is judged on
what was **submitted** rather than on what survives Q3 — otherwise a sheet
mapping one control onto all 78 hubs passes Q2 by having every link filtered
out. Q4's agreement is reported as *not measured*, never as 1.0, when no control
was worked by two people.

`scripts/analysis/orphan_rate.py` is the raw graph arithmetic and **is not the
gate**; it counts every link at any confidence, and warns as much when given
`--bridge`.

**Q4 produces this project's first human–human agreement number.** No such
measurement exists today. `results/ceiling_study/panel_agreement.md` reports
**one** human annotator against five LLM judges — its own text says "the single
human annotator" — as raw agreement, not chance-corrected. Any figure quoted as
prior human-human agreement is a misreading of that file; see the correction in
`claudedocs/curation-package.md` §1.6.

---

## 3. Gate 2 — one retrain, ~$40, on ENISA + BIML

**Restated 2026-09-04 after premortem checkpoint 2. The previous version is
kept below, because it was fatally broken in two ways and a reader who finds it
elsewhere needs to know why it was replaced.**

> **PASS iff the fold-stratified paired bootstrap gives `ci_low > 0`** on the
> hit@1 delta between a bridge-trained arm and a bridge-free arm, both retrained
> under the **strict all-AI firewall** (all eight AI frameworks held out), scored
> on the **ENISA + BIML** items.

Four choices, each forced by a measurement:

**Scored on ENISA + BIML, not the traditional validation split.** The
validation split is 1,265 items over 344 distinct gold hubs, and **zero** of
those hubs are AI hubs. Bridge links add training positives on AI hubs, so the
previous Gate 2 could not have detected them at any effect size. ENISA + BIML
give **50 items** (33 + 17) over **32 distinct gold hubs, all 32 AI-only** —
exactly the population bridges are bought to supervise. Neither framework
appears in the 147-item test split, so **no test-split draw is spent**. ETSI is
the third free AI framework and is excluded: its prose is licensed.

**Against a bridge-free arm, not zero-shot.** Zero-shot is not the
counterfactual for "do bridges carry signal" — it is the counterfactual for "does
training help at all", which is a different and already-answered question. The
comparator is the same retrain with `bridge_links_path=None`, identical seed,
identical firewall.

**`ci_low > 0`, not a point estimate.** The previous criterion — *"the trained
arm beats its own paired zero-shot"* — is a bare sign test. Measured on this
project's three committed Campaign 2 validation arms, **with zero bridge
links**, it is satisfied on **11 of 15 fold-tests (0.73)**. §3 of CAMPAIGN3.md
already condemns exactly this standard: *"Campaign 2 passed on the point
estimate alone … that is the failure this threshold exists to prevent."*

**And the power is stated in advance, because n=50 is small.** Simulated at the
observed discordant rate of 0.30, 400 trials:

| true delta | power |
|---|---|
| 0.10 | 0.22 |
| 0.15 | 0.50 |
| 0.20 | 0.75 |
| 0.25 | 0.96 |

The MDE at 80% power is **δ ≈ 0.21**. So **a FAIL is weak evidence of absence
and must be reported as such.** Reading a FAIL at this n as "bridges do not
help" is the error this table exists to forestall.

### What the previous Gate 2 was, and why it is gone

It required the τ leave-one-fold-out swing to stay under 0.15 — **a criterion
pre-registered to fail.** Bridge links add training supervision, not evaluation
folds, so fold sizes 63/30/11/4/2 are unchanged by construction and the measured
swing is 0.3702 → **0.0000** (`docs/campaign3-audit-mechanism.md`
§6e-corrected). τ is not what Phase 2C changes.

Checkpoint 1 replaced that with a validation-split criterion, which is the
version A2 above refutes: it scores a population containing no AI hub. Both are
recorded rather than deleted.

### Implementation status, stated plainly

**The strict all-AI firewall has no implementation.** `run_single_fold` takes
`held_out_framework: str`, singular, and `run_fold.py` exposes `--framework`
with one value. Holding out eight frameworks at once requires
`held_out_frameworks: frozenset[str]` and does not exist today.

**Bridge links now reach training** (`60475db`) through
`TrainingConfig.bridge_links_path`, and the corpus is pinned in
`fold_input_digests` as `bridge_links_sha256`, so a bridge run and a
bridge-free one are distinguishable in their artifacts. Before that commit they
were not: nothing consumed the bridge corpus at all.

**Stage 2 funds only if both gates pass.**

## 4. Explicitly not a gate

- **Whether the primary delta moves.** Judging the round by the result it
  produced is the outcome-switching this project has done three times.
- **The test split is not scored in Phase 2C at any point.**

---

## 5. What annotators must not see, and what cannot be hidden

**Revised 2026-09-04 after premortem checkpoint 2, which found the previous
version asserting a control that does not exist.**

### The instruction, which is the part that works

Annotators are told, in the handbook and in the attestation they sign:

> Do not read the TRACT repository at `github.com/rocklambros/TRACT` — including
> its issues, documentation, `data/` and `results/` directories — for the
> duration of your work. Do not consult `opencre.org`.

The previous version forbade opencre.org and never mentioned this repository.
That is the gap that matters: **the repository is public**, and it tracks
`results/ceiling_study/hub_reference.md` (whose 400 LLM-written hub
descriptions make any label that saw them Tier 3), the curated gold links
themselves, `results/review/review_export.json`, and
`results/bridge/bridge_report.json`. An OWASP volunteer recruited to a project
naturally looks that project up. One search reaches all of it, and the
attestation then asks them to affirm they *"did not seek out any existing or
predicted mapping"* — an affirmation the project made structurally hard to keep
and never warned them about.

### The numeric targets cannot be hidden, and this document no longer claims they are

The previous version stated that §2's and §3's numeric targets were *"not
disclosed to annotators"*. They are in this file, which is public. They are
also, since the Gate 1 implementation landed, named constants in
`tract/config.py` — `PHASE2C_GATE1_MAX_ORPHANS`,
`PHASE2C_Q1_MIN_DISTINCT_CONTROLS` and the rest.

Moving them out of this markdown would hide nothing while the code that
computes them is readable, and removing them from the code would undo the fix
that made Gate 1 a gate rather than a paragraph. **So the claim is withdrawn
rather than defended.** The blinding rests on the instruction and the
attestation, which are honest controls, and not on the targets being secret,
which was never true.

### What follows for the round

A round is Tier 2 on the strength of the instruction being followed. That is
weaker than a round where the answer key is unreachable, and the difference
should be stated wherever the corpus is described. If a stronger claim is
wanted, the answer key has to move — making the repository private for the
duration, or relocating `hub_reference.md` and the review/bridge exports — and
that is an owner decision this document records rather than makes.

## 6. Annotator exclusions

Beyond `claudedocs/curation-package.md` §1.3's exclusion of anyone who has worked
on TRACT: **anyone who authored or maintains the framework being mapped**, and
anyone who authored any of the eight AI frameworks in the corpus. With an OWASP
volunteer pool this is the expected case rather than an edge case — a
framework's own author recalls the intended mapping rather than judging it.
Screening asks about both.

---

## 7. What was known on the day this was written

Recorded so a later reader can tell what was and was not in view.

- Orphan rate 78/78; regions disjoint; no free non-model source of
  traditional→AI links exists (0 of 3,734 candidate links).
- The Phase 2B bridge set (46 accepted hub→hub pairs) is **Tier 3** and may not
  seed this round in any form — not as a shortlist, not as `related_hub_ids` in
  an annotator sheet (`results/bridge/PROVENANCE.md`).
- Bridge links do not change the 147-item evaluation corpus; verified
  byte-identical (`tests/test_bridge_links.py`).
- `hub_rep_format="path+name+standards"` would create a route by which bridge
  links reach evaluation **that the hub firewall does not catch**. It is
  currently unreachable and must stay so; see design §4.4 and
  `tests/test_standards_format_bridge_exposure.py`.
- Campaign 3's own primary gate remains **FAIL**, and the C3TEST rebaseline did
  not move it. Phase 2C is not a retry of that gate and does not report on it.

---

## 8. Reproducing every number in this document

```bash
# The baseline, and the two AI-framework definitions
python -m scripts.analysis.orphan_rate                          # 78 of 78
USE_TF=0 python -m pytest tests/test_orphan_rate.py -q

# Gate 1, in full. Exits non-zero on FAIL.
python -m scripts.analysis.gate1_report data/training/hub_links_bridge.jsonl
USE_TF=0 python -m pytest tests/test_gate1_report.py -q

# Bridge links reach training, and do not reach the eval corpus
USE_TF=0 python -m pytest tests/test_bridge_reaches_training.py -q
USE_TF=0 python -m pytest tests/test_bridge_links.py -q

# The packet is model-free and licence-guarded
USE_TF=0 python -m pytest tests/test_bridge_packet.py -q
USE_TF=0 python -m pytest tests/test_external_redistribution_guard.py -q
```
