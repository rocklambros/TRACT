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

Q3's scale is enforced at import (`scripts/import_bridge_links.py`,
`CONFIDENCE_MIN`/`CONFIDENCE_MAX` = 1/3, `GATE1_CONFIDENCE_FLOOR` = 2).

**Q4 produces this project's first human–human agreement number.** No such
measurement exists today. `results/ceiling_study/panel_agreement.md` reports
**one** human annotator against five LLM judges — its own text says "the single
human annotator" — as raw agreement, not chance-corrected. Any figure quoted as
prior human-human agreement is a misreading of that file; see the correction in
`claudedocs/curation-package.md` §1.6.

---

## 3. Gate 2 — one retrain, ~$40, on the VALIDATION split

> Retrain under the **strict all-AI firewall** — all eight AI frameworks held
> out — and measure on the **validation** split only. **PASS if the trained arm
> beats its own paired zero-shot on the held-out framework.**

That is the one question bridges can answer: before them the strict firewall
orphans 78/78 gold hubs and there is no trainable task.

**This gate was corrected after premortem checkpoint 1** and the original is
recorded here rather than deleted. It required the τ leave-one-fold-out swing to
stay under 0.15 — a criterion **pre-registered to fail**. Bridge links add
training supervision, not evaluation folds, so fold sizes 63/30/11/4/2 are
unchanged by construction and the measured swing is 0.3702 → **0.0000**
(`docs/campaign3-audit-mechanism.md` §6e-corrected). τ is not what Phase 2C
changes, so it is not what Phase 2C is gated on. The original also scored the
**test** split, spending a draw Amendment 1 §1.5 calls near-exhausted.

**Stage 2 funds only if both gates pass.**

---

## 4. Explicitly not a gate

- **Whether the primary delta moves.** Judging the round by the result it
  produced is the outcome-switching this project has done three times.
- **The test split is not scored in Phase 2C at any point.**

---

## 5. Not disclosed to annotators

The numeric targets in §2 and §3. A population generating the data should not be
told the quota it is generating against.

Also withheld, and for a stronger reason: **`results/ceiling_study/hub_reference.md`
must never be sent.** 400 of its 522 hub descriptions were written by an LLM
conditioned on the existing gold links. Sending it makes every label Tier 3 and
the round unusable for either gate. Generate the clean sheet with
`python -m scripts.build_bridge_packet <out_dir>`.

---

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
python -m scripts.analysis.orphan_rate                      # 78 of 78
USE_TF=0 python -m pytest tests/test_orphan_rate.py -q      # both definitions
USE_TF=0 python -m pytest tests/test_bridge_links.py -q     # corpus unchanged
USE_TF=0 python -m pytest tests/test_bridge_packet.py -q    # packet is model-free
```
