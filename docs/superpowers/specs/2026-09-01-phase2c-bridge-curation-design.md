# Phase 2C — traditional→AI bridge curation

**Date:** 2026-09-01
**Status:** design, approved in outline; tooling not yet built
**Supersedes nothing.** Phase 2B remains as recorded; §6 explains why it did not
deliver this.

---

## 1. The problem, measured

TRACT's Vision (PRD.md §1) states the product "extends OpenCRE by making it easy
to add new frameworks **and bridge AI security with traditional security
standards**." Measured on `hub_links_curated.jsonl`:

| | count |
|---|---|
| hubs linked by an AI-security framework | 78 |
| hubs linked by a traditional framework | 380 |
| **hubs linked by both** | **0** |

The intersection is empty. Half the stated product is undelivered, and PRD.md:58
has recorded it as such since the beginning: *"Zero hubs currently bridge AI to
traditional. Bridging is a Phase 2 deliverable."*

The same fact is what makes the LOFO claim unsound. Holding out all eight AI
frameworks — the firewall that would make "generalises to an unseen framework"
literally true — orphans **78 of 78** AI gold hubs, because every AI hub is
supervised exclusively by AI frameworks. The evaluation problem and the product
problem are the same problem.

### 1.1 Why the planned mitigation cannot work

PRD.md:112 proposes *"Phase 2: bridge identification using ENISA/ETSI/BIML as
seeds."* Those three are AI-security frameworks — PRD.md:58 says so itself — and
their hubs are **91.9% redundant** with the existing AI hubs (57 of 62 already
incumbent). The planned seeds are on the AI side of the divide.

### 1.2 Why Phase 2B did not deliver it

Phase 2B is marked COMPLETE and produced 46 accepted bridges. They are:

- **hub→hub** `related_hub_ids` edges, not framework→hub links, so they create
  no training positives and cannot close the leak;
- **model-proposed** by cosine top-k and ratified in the model's presence —
  Tier 3 under `results/phase1b/CAMPAIGN3.md` §2, unusable as supervision;
- computed over **21** AI-only hubs under the superseded five-framework
  definition, against **78** under the corrected one (83 is the PRE-AUDIT figure, from `hub_links_by_framework.json`; the curated set gives 78).

### 1.3 No shortcut exists (spike, 2026-09-01)

Before designing an annotation round, two cheaper sources were tested:

- **Published cross-references.** MITRE ATLAS carries 83 distinct ATT&CK
  technique references, OWASP LLM Top10 12, OWASP AI Exchange 8. ATT&CK is not
  one of OpenCRE's 22 linked frameworks, so these do not reach a CRE hub
  directly. A transitive ATT&CK→CAPEC→CRE chain would bridge *AI controls onto
  traditional hubs* — the opposite of the direction the leak needs.
- **AI-relevant traditional controls.** Searching every traditional framework's
  link section names for machine-learning vocabulary returns **0 of 3,734**.

There is no free, non-model source of traditional→AI links in this corpus. The
human round is load-bearing.

## 2. The key property that makes this cheap

`build_evaluation_corpus` filters to `AI_FRAMEWORK_NAMES`. A NIST 800-53 link
pointing at an AI hub is therefore **training supervision that never becomes an
evaluation item.** Verified:

```
baseline                          eval corpus = 147 items, degree(342-641) = 6
+ one NIST 800-53 -> AI-hub link  eval corpus = 147 items, degree(342-641) = 7
```

Consequences, and they are the reason this design was chosen over the
alternatives considered in `docs/campaign3-premortem-round1.md`:

- **The eval corpus does not change.** The 147 items are byte-identical, so no
  composition shift and no re-derivation of what the denominator contains.

  **CORRECTED: this is not the same as spending no draw.** An earlier version of
  this line claimed it was. `results/phase1b/CAMPAIGN3.md` Amendment 1 §1.5
  defines the cost by *scoring*, not by changing: *"The 147 AI items have now
  been scored twice by the same recipe family. A third run would be much harder
  to justify."* Any gate that scores the split spends a draw whatever the items
  are. D4 is now written so that Gate 2 does not touch the test split at all.
- **No composition shift.** The denominator does not move, so the estimator does
  not have to price a change in eval-set makeup.
- **Every prior run stays comparable.** Only the training corpus changes.

The cost is one retrain.

## 3. Design decisions

### D1 — Purpose: product and evaluation. The sheet carries all 78 AI hubs.

**CORRECTED 2026-09-01 after premortem checkpoint 1. The original version of
this decision was wrong twice.**

It said *"Stage 1 targets the top 20 AI hubs by eval weight."* Two independent
perspectives found the same fatal arithmetic: a link carries one `cre_id`, so a
20-hub sheet can de-orphan at most 20 hubs, and Gate 1 (D4) requires **23**.
A flawless annotator accepting every hub on the sheet would still fail the gate.
The design terminated its own funding path, and the failure would have read as
*"the domains are less connected than the product assumes"* — a substantive
conclusion manufactured by a counting error.

Worse, the ranking itself was a **selection rule derived from the held-out test
split**: "eval weight" counts how often a hub appears as gold in the 147-item
corpus that Gate 2 then scores. Choosing which hubs receive training supervision
by counting test-set answers is the leakage shape that withdrew two previous
campaigns, re-entering through the sampling frame rather than the corpus.

> **The sheet carries all 78 AI hubs, unranked.** Cost is bounded by the 300
> NIST 800-53 controls, not by the hub count — the annotator reads a control and
> scans a hub list, so 78 rows grouped by branch costs little more than 20. No
> hub is privileged, no test-set statistic informs the packet, and Gate 1's
> ceiling becomes 78 rather than 20.

Recorded because it was measured and is no longer used: 73 of 78 AI hubs carry
test gold, the top 20 carry 50% of eval-item slots and the top 40 carry 76%
(median 2, max 7). That concentration is why the ranking looked attractive.

### D2 — Task: framework-scoped sweep, NIST 800-53 first

The annotator holds a **traditional framework** fixed and walks it against the AI
hub list, rather than holding an AI hub fixed and searching 4,155 traditional
links.

Rejected alternatives:

- *Hub-anchored, unguided.* The CRE hierarchy gives no usable neighbourhood: only
  **1 of 78** AI hubs has a traditional sibling and **1 of 78** has a traditional
  parent, though 4 of 5 branch roots are shared. So the annotator would be
  searching a branch, not a shortlist.
- *Hub-anchored with a lexical shortlist.* Deterministic and defensibly Tier 2,
  but it biases toward lexical overlap, which is precisely what the zero-shot arm
  is good at. It would tend to produce links that flatter the baseline, and
  nothing downstream could detect that.

NIST 800-53 first because PRD.md:58 names it a core traditional standard, it is
300 links, and a controls specialist can judge *"does AC-3 Access Enforcement
relate to Access control to AI inference?"* without searching anything.

### D3 — Storage: separate file, tier-tagged, proposed upstream

Bridge links land in **`data/training/hub_links_bridge.jsonl`**, merged at
training time, never merged into the evaluation build. The Tier 1 / Tier 2
boundary becomes a *file* boundary rather than a field.

This is deliberate. `tests/test_tier3_quarantine.py` shows this project can
enforce a file boundary; the link audit shows it cannot reliably enforce a field
one — 56 rewritten labels sat inside `hub_links_curated.jsonl` undisclosed for
months.

Accepted links are additionally **proposed upstream to OpenCRE**. They are
genuinely new OpenCRE links that fix a gap in OpenCRE's own ontology, the PRD's
vision says TRACT "gives back to OpenCRE," and upstream acceptance is the only
honest route by which this data ever becomes Tier 1.

### D4 — Checkpoint, pre-registered before any annotation

Two gates, in order. **Both are declared here, before a single control has been
read.**

**Gate 1 — free, graph arithmetic, no model, no pods.**
Strict-firewall orphan rate must fall from **78/78** to **≤ 55/78**: at least 23
AI hubs given real non-AI supervision. Reachable now that the sheet carries all
78 hubs (D1); under the superseded 20-hub scope it was not.

Gate 1 counts hubs, so it is gameable by volume. Four quality conditions, all
pre-registered here and all checkable by the importer without a model:

- **≥ 40 distinct NIST 800-53 controls** contribute at least one accepted link.
  Twenty-three links from one control is not a sweep.
- **≤ 6 AI hubs per control.** A control that maps to a third of the AI region
  is a judgement about the region, not the control.
- **Confidence ≥ 2** on a 1–3 scale for a link to count toward Gate 1.
- **≥ 15% of controls double-annotated**, with the agreement rate reported. Not
  a threshold — a number that must exist and be published with the result.

**Gate 2 — one retrain, ~$40, on the VALIDATION split.**

**CORRECTED after premortem checkpoint 1.** The original Gate 2 had two defects.

It required *"leave-one-fold-out on the τ estimator must not swing it by more
than 0.15"* — a criterion **pre-registered to fail**. Bridge links add training
supervision, not evaluation folds, so the fold sizes 63/30/11/4/2 are unchanged
by construction, and `campaign3-audit-mechanism.md` §6e-corrected measures the
LOFO swing at 0.3702 → 0.0000. This document says so itself, forty lines away.
τ is not what Phase 2C changes, so it is not what Phase 2C is gated on.

And it scored the **test** split, which spends a draw Amendment 1 §1.5 calls
near-exhausted, while §2 claimed no draw was spent.

> **Gate 2, restated.** Retrain under the **strict all-AI firewall** — every one
> of the eight AI frameworks held out — and measure on the **validation** split
> only. Pass if the trained arm beats its own paired zero-shot on the held-out
> framework. That is the one question bridges can answer: before them the strict
> firewall orphans 78/78 gold hubs and there is no trainable task at all.

**Stage 2 funds only if both pass.**

Explicitly **not** a gate, and not computed: whether the primary delta moves.
The test split is not scored in Phase 2C at any point.

**Not disclosed to annotators:** the numeric targets above. A population
generating the data should not be told the quota it is generating against.

Explicitly **not** a gate: whether the primary delta moves. Judging the round by
the result it produced is the outcome-switching this project has done three
times.

## 4. Architecture

```
                    (human, offline)
  AI hub sheet  ─┐
  NIST 800-53   ─┴─►  annotator  ──►  filled CSV
  control sheet                            │
                                           ▼
                              scripts/import_bridge_links.py
                                           │  validates, tiers, hashes
                                           ▼
                        data/training/hub_links_bridge.jsonl   (Tier 2)
                                           │
        ┌──────────────────────────────────┴───────────────┐
        ▼                                                  ▼
  training merge                                    orphan_rate check
  load_curated_links() + load_bridge_links()        (Gate 1, free)
        │                                                  │
        ▼                                                  ▼
   retrain (Gate 2)                          docs/phase2c-results.md
        │
        ▼
  evaluation corpus  ── UNCHANGED, 147 items
```

### 4.1 Files

| path | responsibility |
|---|---|
| `scripts/build_bridge_packet.py` | Emit the annotator packet: AI hub sheet (**all 78, unranked** — A1 removed the top-N-by-eval-weight scoping as both unpassable and test-set-derived) + NIST 800-53 control sheet. Model-free by construction. |
| `tract/bridge/links.py` | `load_bridge_links()`, `merge_bridge_links()`, the `BridgeLink` type, tier tagging |
| `scripts/import_bridge_links.py` | Validate a filled CSV, refuse malformed/duplicate/unknown ids, write `hub_links_bridge.jsonl` atomically |
| `scripts/analysis/orphan_rate.py` | Gate 1: strict-firewall orphan rate, before and after |
| `data/training/hub_links_bridge.jsonl` | The Tier-2 bridge corpus (created by the import) |
| `docs/phase2c-preregistration.md` | D4's gates, committed before annotation begins |

### 4.2 Provenance guards

Non-negotiable, each enforced by a test:

1. The packet contains **no model output** — no similarity, rank, prediction,
   candidate or suggestion column, and no `related_hub_ids` (which holds Phase
   2B's 46 model-proposed edges).
2. The packet contains **no licensed prose**. `build_control_sheet` currently has
   no restricted-framework allowlist; NIST 800-53 is unrestricted, but the guard
   is added anyway because `--frameworks etsi` is one flag away.
3. Bridge links never enter `hub_links_curated.jsonl` and never enter the
   evaluation corpus.
4. Every bridge link carries `tier: 2`, an annotator id, and a timestamp.

### 4.3 Annotator exclusions

`claudedocs/curation-package.md` §1.3 excludes anyone who has worked on TRACT.
It does **not** exclude anyone who worked on the framework being mapped. With an
OWASP volunteer pool that is the expected case, not an edge case: a framework's
own author recalls the intended mapping rather than judging it. Screening adds a
disclosure question covering both the AI frameworks in the corpus and NIST
800-53.

### 4.4 Constraint: `hub_rep_format="path+name+standards"` stays unreachable

Bridge links create a route by which this phase's own supervision reaches its
own evaluation, and **the hub firewall does not close it.**

`build_firewalled_hub_text(include_standards=True)` appends the standard
sections linked to a hub, dropping only those of the held-out framework. Once
traditional controls link to AI hubs, an AI hub carries NIST 800-53 sections.
Hold out MITRE ATLAS and score it: those sections are not ATLAS, so the filter
keeps them and `assert_firewall` passes. Nothing raises, and bridge-derived
text is sitting in the representations the fold is scored against.

The firewall is not broken. It is scoped to held-out-framework leakage, which
is a different property from "the supervision under test must not reach the
evaluation". Only the first one has an assertion behind it.

This is prospective, not live: no caller supplies `standard_sections`, so
`run_single_fold` raises on the format and the declared A3 ablation arm in
`scripts/phase1b/ablation.py` has never run. **Phase 2C must not change that.**
Enabling the standards format requires a bridge-exclusion rule designed first —
hub text at evaluation time must not incorporate sections that arrived over a
bridge link, because those links are the thing being measured.

`tests/test_standards_format_bridge_exposure.py` holds the line: it fails the
day a caller supplies `standard_sections`, and it demonstrates the clean
firewall pass rather than describing it.

## 5. Testing

Every claim above that a test can hold, a test holds:

- the eval corpus is byte-identical before and after a bridge link is added
- bridge links never appear in `build_evaluation_corpus` output
- the packet carries no model-derived column and no `related_hub_ids`
- the packet carries no control text from a restricted framework
- the importer refuses unknown hub ids, unknown control ids, duplicates, and
  rows whose verdict field is absent
- orphan rate is computed identically to `campaign3-audit-mechanism.md` §6g
- Gate 1 and Gate 2 thresholds are read from the pre-registration, not inlined

## 6. Risks

- **The round produces too few links to clear Gate 1.** Most likely failure. It
  is also a real result: it would say the two domains are less connected than the
  product assumes, which is worth knowing before more is spent.
- **Annotator familiarity bias.** Mitigated by §4.3, not eliminated.
- **Lexical-overlap bias by another route.** A controls specialist may gravitate
  to hubs whose names echo the control's wording. Detectable after the fact by
  comparing accepted links against zero-shot similarity — a diagnostic, never a
  filter, and computed only after the round closes.
- **Upstream rejection.** OpenCRE may decline the links. That does not invalidate
  them for TRACT's training corpus; it only keeps them Tier 2.

## 7. Out of scope

- Re-running Phase 2B's hub→hub bridge analysis over the corrected 83 AI-only
  hubs. Worth doing, separate deliverable, does not close the leak.
- Any change to `AI_FRAMEWORK_NAMES` or the LOFO roster. The premortem showed the
  roster rotation buys 5 genuinely new hubs and spends two test-split draws;
  it is not part of this design.
- Adopting `docs/campaign3-amendment2-draft.md`. It stands withdrawn.
