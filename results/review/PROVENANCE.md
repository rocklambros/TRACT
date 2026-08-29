# Provenance of the files in this directory

Read this before reusing anything here. One of these files is quarantined.

## `review_export.json` — **TIER 3. NOT GROUND TRUTH. DO NOT TRAIN OR EVALUATE ON IT.**

898 assignments that a **TRACT model proposed** and a human then **ratified in
the model's presence**. 693 of them (77.2%) were accepted exactly as the model
proposed; 203 were reassigned, 2 rejected.

Under the Campaign 3 provenance tiers a label produced by, or ratified in the
presence of, a model is **Tier 3** and may not sit in a gate denominator at any
ratio. Not at 10%, not at 1%. A metric computed on these labels is measuring the
model against its own earlier opinion with a human in the loop as an amplifier,
not as an independent check.

### Why this file in particular is dangerous

It is the single largest pool of ready-made `(framework, section, hub)` triples
in the repository — 898 of them, spanning 11 frameworks — and the project's
stated problem is that the AI eval set has only 147 items. It is roughly ten
minutes of work from becoming training or eval labels, and the honest
alternative is about 25 expert-hours of blind curation. That asymmetry is the
whole risk.

**215 of the 898 items belong to Campaign 2 test-split frameworks** — 211
`mitre_atlas`, 4 `owasp_llm_top10`. Contamination there would put model-derived
labels in the denominator of the number the project reports.

### This is measured, not asserted

63 of the 898 items land on a `(framework, section)` that OpenCRE independently
curated. Comparing the pipeline's final decision against that gold:

| | count |
|---|---|
| agrees with OpenCRE | 47 (74.6%) |
| disagrees | 16 |
| — of which the reviewer marked `accepted` (ratified the model over the gold) | 12 |
| — of which the reviewer reassigned to a third hub, also disagreeing | 4 |

**A quarter of the time this pipeline produces a different answer from the
curated taxonomy, and on 12 of those items a human confirmed the model rather
than the gold.** That 74.6% is the measured cost of admitting these labels.

### What the file IS good for

Prioritising human attention, estimating review effort, and choosing what to
curate. It is a worklist. It is not evidence.

### Enforcement

`tests/test_tier3_quarantine.py` fails if any review decision here appears in
`data/training/hub_links_curated.jsonl` without tracing to either the pre-review
baseline (`hub_links.jsonl`, 2026-04-28) or a documented audit correction. It
also pins the file's shape, so if it is ever regenerated the numbers above must
be re-derived rather than inherited.

---

## The other three files are ordinary artifacts

- **`hub_reference.json`** — the CRE hub tree as shown to reviewers. Derived from
  `cre_hierarchy.json`, no model output.
- **`reviewer_guide.md`** — instructions. Prose.
- **`review_metrics.json`** — aggregate rates over the review round. Summary
  statistics *about* the Tier-3 file; safe to read, and note that any quality
  number in it inherits the 74.6% caveat above.

---

## Related

- `docs/campaign2-results.md` §13 — the AI link audit, a *different* provenance
  problem in the same corpus (Tier 2, human relabel, 25% of the test gold).
- `docs/campaign3-premortem.md` — where this quarantine came from.
