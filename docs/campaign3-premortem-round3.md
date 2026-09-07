# Adversarial premortem, checkpoint 2 — Phase 2C tooling and pre-registration

Six perspectives (Red Teamer, Data Scientist, ML Engineer, Security Architect,
MLOps/SRE, Governance/Risk), independent contexts, run 2026-09-04 against
`58a63cb` on `campaign3-premortem-fixes`. No perspective was skipped: the
annotator packet, the importer and the licensed-prose boundary gave Security
and MLOps the surface they lacked at checkpoint 1.

> **Verdict: the Phase 2C tooling is individually sound and collectively inert.
> Nothing consumes bridge links, so the round cannot affect any model. Gate 2
> measures a split containing zero AI hubs, so it could not detect them if it
> did. Gate 1's four quality conditions have no implementation, so it can be
> cleared in fifteen minutes of copy-paste. Three of these were found
> independently by four or more perspectives.**

Everything below was re-measured by the orchestrator or reproduced by the
perspective that raised it. Confidence bands follow
`reference/confidence-and-drop-rule.md`; a band is raised only where
perspectives brought genuinely different evidence, not where they ran the same
grep.

---

## A. Confirmed — Critical

### A1 — Bridge links have no training consumer. The phase is inert.
**Impact: Critical. Posterior: Confirmed** (prior Likely; raised by an
execution result, not by agreement).

Found independently by Red Teamer (R3), ML Engineer (F1), Governance (W3) and
MLOps (F3). Four perspectives is normally a correlated-evidence penalty, since
all four ran the same grep — `hub_links_bridge.jsonl` has exactly one
reference in the repository and it is the *write* target, and
`bridge_training_records` / `merge_for_training` have zero non-test callers.

What raises the band is the ML Engineer's separate check: they pushed 25 real
NIST 800-53 controls through `bridge_training_records` →
`filter_training_links(ProseIndex.load())` and got `kept: 25, tiers:
Counter({'T2': 25})`. The machinery works. It is simply not wired to anything.
Every training entry point calls `load_and_filter_curated_links()`, which
resolves to `CURATED_PATH` and nothing else.

**Failure mode.** The operator runs the documented flow — build packet,
annotate, import, retrain — and the retrain reads the identical corpus it read
before. Under the strict firewall the AI region stays 78/78 orphaned, the
trained arm loses to its zero-shot, and the round is written up as *"human
bridge links do not help"*. The money is spent before anything reveals the
measurement never contained a bridge link.

**No test can go red on this.** `test_bridge_links.py` calls
`bridge_training_records` directly, so it stays green while the pipeline
ignores it.

### A2 — Gate 2 scores a population with zero AI hubs
**Impact: Critical. Posterior: Confirmed** (direct enumeration; zero, not
"few").

The Data Scientist enumerated the split the pre-registration binds Gate 2 to:

```
VALIDATION split: 1,265 items, 344 distinct gold hubs
  gold hubs that are AI hubs:          0
  ANY valid_hub_id that is an AI hub:  0
TEST split (AI):    147 items, 71 gold hubs, all inside the 78-hub AI region
```

Bridge links add training positives on AI hubs. No validation item's answer is
an AI hub, and §4 forbids scoring the test split at any point. The round's
product is never scored.

### A3 — Gate 2's criterion passes ~73% of the time with no bridge links at all
**Impact: Critical. Posterior: Confirmed** (measured on this project's own
committed arms).

The criterion is *"the trained arm beats its own paired zero-shot on the
held-out framework"* — a bare sign test, no threshold, no interval. On the
three committed Campaign 2 validation arms, **with zero bridge links**, it is
satisfied on **11 of 15 fold-tests (0.73)**.

Governance reached the same conclusion from a different direction (W1): this is
a point-estimate criterion, and `CAMPAIGN3.md` §3 already condemns exactly that
standard in writing — *"Campaign 2 passed on the point estimate alone with
P(δ ≤ 0.10) = 0.203, and that is the failure this threshold exists to
prevent."* Phase 2C reintroduced the standard the campaign withdrew a headline
over.

**And the criterion is ambiguous.** "The held-out framework" is singular
against a five-fold split. All-five, any-one, pooled, and one-named-fold are
four different rules whose verdicts today span roughly 0 to 1. Nothing selects
one, so the reading gets picked after the numbers land.

### A4 — Gate 1's quality conditions are prose; Q3's floor is dead code
**Impact: Critical. Posterior: Confirmed** (demonstrated three ways).

Red Teamer, Data Scientist (D5), Governance (W4) and MLOps (F6) all found it,
and two of them demonstrated it independently. `GATE1_CONFIDENCE_FLOOR` has one
reader in the entire repository — a test asserting the constant lies inside its
own scale. `orphan_rate` adds every bridge link with no confidence predicate.
Q1, Q2 and Q4 have no implementation anywhere.

The Red Teamer's demonstration is the one to keep. A sheet mapping **one**
control (`AC-1`, "Policy and Procedures") onto all 78 hub ids — copied from the
first column of the packet the volunteer was handed, confidence 1, rationale
`"."` — imports cleanly and takes the orphan rate from 78/78 to **0/78**. It
violates Q1, Q2 and Q3 simultaneously and no code objects.

### A5 — The curation packet redistributes proprietary prose. **FIXED this checkpoint.**
**Impact: Critical. Posterior: Confirmed, remediated.**

`build_curation_packet` had no licence check, defaults to targets including
`csa_aicm` (recorded "Proprietary … no redistribution"), and the handbook
instructs the owner to run it before mailing sheets to an outside annotator.
`build_bridge_packet` did check, on `OVERLAY_FRAMEWORK_IDS` — the git-tracking
tier, which omits both CSA frameworks. The constant that governs the actual
question, `REDISTRIBUTION_RESERVED_FRAMEWORK_IDS`, was read by neither.

Fixed in `ead1e12` with a shared three-state guard. Recorded here because the
reasoning error is worth keeping: the bridge packet's comment argues correctly
that redistribution needs a wider set than RESTRICTED, and then reaches for the
wrong wider set.

### A6 — CI executed 15 tests of ~3,150. **FIXED this checkpoint.**
**Impact: Critical. Posterior: Confirmed, remediated.**

MLOps reproduced it end-to-end with a counterfactual (2,937 pass with the two
offending files ignored); the orchestrator confirmed it independently from the
live CI logs. Two test files added *this session* imported torch transitively,
raised at collection, and pytest aborts the whole run on a collection error
regardless of `-x`.

This is checkpoint 1's finding A2 recurring, in the session that documented A2,
caused by the tests written to close other findings. Fixed in `fc9ba1a`,
including a collection floor so the next occurrence fails with a message naming
the cause.

### A7 — Tier-3 links are live in the OpenCRE export path right now
**Impact: Critical. Posterior: Confirmed** (measured on the committed database).

Independent of Phase 2C. Governance queried `results/phase1c/crosswalk.db`:
551 rows with `provenance='active_learning_round_2'` and non-null confidence
pass **every** clause of the export filter in `tract/export/filters.py`. Against
`PRD.md:465-467`: *"nothing model-derived is downstream in the published model,
the published dataset, the OpenCRE fork import, or the Phase 5B export."*

Round 1 parked this as a tail risk with the trigger *"any move toward RFC
submission"*. Phase 2C's design makes upstream proposal part of the round, so
the trigger has fired.

---

## B. Confirmed — High

### B1 — The blinding regime is asserted against a public repository
**Posterior: Confirmed** (visibility and tracking both verified).

The repo is public. Tracked and reachable: `results/ceiling_study/hub_reference.md`
(the 400 LLM-written descriptions the pre-registration says *"must never be
sent … makes every label Tier 3"*), the curated gold links, `review_export.json`,
`bridge_report.json`, and the pre-registration itself — whose §5 declares the
numeric targets *"not disclosed to annotators"* while they sit on a public URL.

The handbook forbids exactly one source: opencre.org. It never mentions the
TRACT repository. An OWASP volunteer who looks the project up finds all of it,
then signs an attestation affirming they *"did not seek out any existing or
predicted mapping"*.

### B2 — The handbook describes a different round, and `NONE` aborts the import
**Posterior: Confirmed** (Red Teamer R5, Governance V2, executed).

Three schemas that do not agree: the handbook promises a 522-hub reference and
four `ANSWER_` columns; `build_bridge_packet` emits 78 hubs and no answer
columns at all; the importer demands `control_id, cre_id, confidence,
rationale`. The handbook's Part 2 says *"hand this part over as-is."*

Worse, the handbook instructs that `NONE` is *"a real, correct, expected
answer"*, and one `NONE` row rejects the entire sheet and writes nothing. The
cheapest fix under deadline is `grep -v NONE` — and `NONE` rows are the negative
evidence that the two domains are less connected than the product assumes,
which the design itself names as the most likely and most informative outcome.
Stripping them biases the round toward Gate 1 passing, at the operator's
keyboard.

### B3 — A second annotator's import destroys the first
**Posterior: Confirmed** (Red Teamer R6, MLOps F5, both executed).

`atomic_write_text` replaces. Q4 requires ≥15% double annotation, the importer
takes one `--annotator-id` per call, and the natural operator action is to run
it once per returned sheet against the default output. The result reports a
**passing** Gate 1 on the last sheet only, with an identical success message
both times. No agreement computation exists anywhere in the repository.

### B4 — The importer does not sanitise, and writes into a tracked directory
**Posterior: Confirmed** (executed against a hostile sheet).

Six payloads accepted verbatim: formula injection (`=HYPERLINK(...)`,
`@SUM(1+1)*cmd|...`), null bytes and ANSI escapes, bidi overrides, a 130,000
character field. `tract/sanitize.py` exists, is imported by 18 other modules,
and is not called here. CLAUDE.md's standing rule requires it.

The default output is `data/training/hub_links_bridge.jsonl`, a tracked
directory, and every record carries a named annotator plus their free text.

Mitigating, and worth recording: `rationale` does not reach the model —
`bridge_training_records` carries only ids and names. This is a human-channel
and repo-integrity problem, not training-data poisoning.

### B5 — CAMPAIGN3 §1.3's "frozen" partition is the retired key-based one
**Posterior: Confirmed** (ML Engineer F3 and Data Scientist D9, reproduced
exactly and independently).

The published table says echo 56 / non-echo 91 and `+0.1538 [+0.0440,+0.2637]`.
That reproduces **exactly** under the key-based partition retired in `3a41d9b`
earlier today. The current index-based partition gives **54 / 93**.

The verdict survives — both clear the floor — but the compounding defect is
mine: `tests/test_echo_partition_identity.py` hardcodes `FROZEN_NON_ECHO = 91`
and uses it as a `!=` guard. If someone wires the frozen partition into the
aggregate, which is the event that class exists to catch, `n_non_echo` becomes
93, `93 != 91` passes, and the tripwire never fires.

### B6 — `preregistered_pass` is uncorrected for selection
**Posterior: Confirmed** (measured).

Mine, from earlier today. `gate_decision` builds a Šidák-corrected interval and
then derives the pre-registered verdict from the **uncorrected** pair. Run at
`n_configurations=1` and `16`, the verdict is identical while
`ci_low_familywise` moves 0.0612 → 0.0136. The comment forty lines above prices
the exposure at 73% under 16 configurations, and the bound verdict sits outside
the correction.

Second half: `gate_decision` has no notion of the Tier-1 audit-untouched
stratum, which `CAMPAIGN3.md` makes the primary denominator. Pooled gives
`P(δ≤0.10)=0.157`; the pre-registered primary is **0.535**.

### B7 — The volunteer throughput figure is fabricated, and the budget rests on it
**Posterior: Confirmed** (verified against the cited source).

The handbook presents 1–3 minutes per item as *"from the project's own
annotation study"*. The source says: *"**Nothing here is timed**, this is only
so you can plan the session."* No timing field exists in the answers file.

The same document, in §1.6, forbids exactly this for agreement: *"Publishing a
target nobody measured invites … an annotator managed against a fabricated
threshold."* Then Part 2 tells the annotator *"much faster than that means
keyword matching."*

### B8 — Committed documents contradict each other on settled numbers
**Posterior: Confirmed** (Governance C1–C7, each quoted).

- `results/bridge/PROVENANCE.md` attributes 78 vs 83 to the framework
  definition. It is a raw-vs-curated **file** difference under the same
  eight-framework definition; the audit is the cause.
- The Phase 2C design spec still carries 83 as "the corrected one".
- **The implementation plan still instructs a future executor to write the
  retracted τ Gate 2 and the fatal `top_n_hubs=20` packet scope.** This project
  executes plans with an agent; one re-run reverts both of checkpoint 1's fatal
  design fixes.
- `PRD.md`, the declared master spec, says Campaign 3 *"Nothing has run"* — the
  C3TEST FAIL is five days old — and still presents the retired `+0.1743/n=109`.
- `CLAUDE.md` says the rebaseline is "in progress", points at the wrong packet
  generator, and calls the Tier-3 quarantined pool "expert-reviewed" sixty
  lines above the quarantine notice.

---

## C. Confirmed — Medium

| | finding | source |
|---|---|---|
| C1 | The order-dependence fix is incomplete twice: `_build_fold_index_matrix` in `evaluate.py` — the function the gate uses — has the same defect (measured spread 0.0162 in p, enough to make a `< 0.05` verdict undetermined in [0.043, 0.057]); and `_stratum_rng`'s payload is sensitive to row order, not just call order | DS D8, MLE F7 |
| C2 | The echo index-identity guard is a **length** check, blind to reordering. Mutation-proven: rotating `apply_prose_to_corpus` by one leaves 13 tests green. And all three item-exactness tests pass `prose_index=None`, so the prose branch never executes — they validate a 32-item partition, not the binding 54 | MLE F4 |
| C3 | `import_bridge_links` validates `cre_id` against all 522 hubs, not the 78 on the sheet. A restatement of an existing OpenCRE link enters training a second time as T2 — the same edge at two tiers | RT R8 |
| C4 | The importer silently drops unrecognised CSV columns, while the JSONL loader rejects unknown fields for exactly that reason | RT R7 |
| C5 | CSA AICM can be laundered into "traditional" supervision: `csa_aicm` and `csa_ccm` share 203 of 243 control ids, so an AICM sheet imports cleanly under `--framework-id csa_ccm` and the AICM origin is unrecoverable | RT R4 |
| C6 | The packet's 78 hubs **are** the eval gold hub set by construction — 71 of 71 eval gold hubs are on the sheet. Removing the top-20 ranking left the membership rule, which is the same statistic thresholded at ≥1 | RT R9 |
| C7 | No lineage: no packet manifest, no source digest on a link, `created_at` accepts any string | MLOps F7 |
| C8 | Gate 2's "~$40" is enforced nowhere. `runpod_retrain` has no `max_usd_per_hour`, no `try/finally` teardown, and its pod name is outside the reaper's sweep | MLOps F4 |
| C9 | No consent or attribution statement: `annotator_id` is required, written into a public repo, and proposed upstream, while the attestation grants the annotator nothing and tells them none of it | Gov V4 |
| C10 | The handbook hands the annotator the answer for an entire stratum (bias/fairness → `NONE`, with the confidence and rationale pre-written) | Gov V5 |

---

## D. Dropped, with reason

| finding | band | reason |
|---|---|---|
| Duplicate records load without complaint (RT R10) | Plausible, Low | Gate 1 counts hubs and is immune; the damage is double-weighted training positives and an inflated link count |
| `csv.Error` is not a `ValueError` (Sec S5) | Likely, Low | Real contract break, no security consequence; fix alongside sanitisation |
| `TestTheBridgeSetHasNotReachedTheGoldFile` cannot fail alone (MLE F9) | Likely, Low | Documents a mechanism; not detection. Do not count as coverage |
| The packet value-scan's "guards the guard" test re-implements the guard (MLE F8) | Likely, Low | The two copies disagree on the skip condition; extract the helper |

**Tail risk, parked with trigger:** the `TRACT_RUNPOD_ALLOW_COMMUNITY=1`
override defeats the sole control keeping ISO 27001 prose off a third-party
host, and leaves only a log line — no entry in the pod state file or run ledger
(Sec S6). *Trigger to raise: any pod run while the licensed overlay is staged.*

---

## E. Convergence

Not converged. This round produced **seven Critical findings**, four of them
found independently by four or more perspectives, and two of the seven are
defects introduced by this session's own fixes.

| round | found | introduced |
|---|---|---|
| 1 | 9 defects in the analysis | — |
| Phase R | fixed 6 | 4 new, 3 inside the corrections |
| checkpoint 1 | those 4 + 2 fatal design errors | — |
| **checkpoint 2** | **7 Critical + 8 High** | **2 of the 7 are this session's** |

The pattern named at checkpoint 1 — *"prose invariants that no test holds"* —
has moved up a layer and now governs the pre-registration itself. Gate 1's four
quality conditions, Gate 2's firewall, the `~$40` ceiling, the blinding regime
and the annotator exclusions are all prose. The design spec asserts as a tested
property that *"Gate 1 and Gate 2 thresholds are read from the pre-registration,
not inlined"*; no module mentions the file.

---

## F. Remediation, ordered by cost reduction per unit of effort

**Before any packet is generated:**

1. **Wire bridge links into training** (A1). Merge inside
   `curated_link_filter_report`, with a test asserting a bridge-staged run
   yields ≥1 `QualityTier.T2` in `report.kept`. Until this exists everything
   else is a gate on a corpus nothing consumes.
2. **Restate Gate 2, or delete it** (A2, A3). It must score AI items and it
   must carry an interval. The bridge-free strict-firewall retrain, not
   zero-shot, is the counterfactual for "do bridges carry signal".
3. **Implement Gate 1** (A4). One `gate1_report.py` computing the orphan
   reduction *and* Q1–Q4, filtering on the confidence floor, refusing to print
   a verdict if any condition is uncomputed.
4. **Decide the blinding** (B1). Either tell annotators not to read the
   repository and move the numeric targets out of a public file, or drop the
   blinding claim. Do not assert a control that does not exist.
5. **Write a Phase 2C handbook** (B2), against the packet that exists, with
   `NONE` as a first-class importable value.

**Before any sheet comes back:** sanitise and stop tracking the output (B4);
per-annotator corpora plus an agreement computation (B3).

**Owner decisions, not code:** paid or volunteer (Gov V3); adjudicate the four
UNDETERMINED NIST licences; whether to publish the corpus regardless of verdict
(Gov W8); whether Tier-3 links may reach the OpenCRE RFC (A7).

**Documents:** the plan file's superseded Gate 2 and `top_n_hubs=20` are the
most dangerous, because they are executable (B8).

---

## G. Residual risk

Two things this round could not test.

The premortem remains commissioned, run and adjudicated by the same party.
`.github/CODEOWNERS` is one name, no commit on this branch is signed, the
pre-registration is not on `main`, and the branch is force-pushable — so the
document's central claim, *"committed before a single control has been read"*,
rests on a push event no artifact records. Checkpoint 1 costed the cheapest
external check at ~2 hours; it has not been done, and this round adds a second
candidate: hand an outside reader Gate 2's criterion and the 11-of-15 table.

And the round's own value is untested. As gated, the modal outcome is a FAIL
that publishes nothing, and the design itself names it *"most likely"*. Nothing
commits to publishing the corpus or the agreement number either way, so a
volunteer's work currently has no guaranteed deliverable.

---

## H. Remediation status, 2026-09-06

Recorded here rather than in a commit message, because the open column is what
the next reader needs and it is the column that goes stale silently.

### Closed

| | finding | how |
|---|---|---|
| A1 | bridge links had no training consumer | merged in `curated_link_filter_report`, pinned by `bridge_links_sha256`, reachable via `run_fold --bridge-links` |
| A2/A3 | Gate 2 scored a split with zero AI hubs, on a criterion a bridge-free arm passes 11/15 | restated on ENISA + BIML (50 items, 32 AI-only gold hubs), `ci_low > 0` against a bridge-free arm, power table stated in advance |
| A4 | Gate 1's four conditions were prose | `scripts/analysis/gate1_report.py`; the 15-minute attack sheet now fails all five |
| A5 | curation packet redistributed proprietary CSA prose | shared three-state guard on `REDISTRIBUTION_RESERVED`, hub sheet included |
| A6 | CI ran 15 tests of 3,150 | lazy imports plus a collection floor; CI green on 3.11 and 3.12 |
| A7 | Tier-3 rows live in the OpenCRE export | filter is an allowlist; PRD claim corrected to what the export carries |
| B1 | blinding asserted against a public repo | annotators told not to read it; the unachievable half of the claim withdrawn |
| B3/B4 | second import destroyed the first; no sanitisation | refuses without `--replace`; sanitised, with formula and bidi payloads refused |
| B7 | fabricated throughput figure | marked a planning guess; the source says nothing was timed |
| B8 | five documents disagreeing | reconciled, with the executable plan file banner-marked |
| — | licence guard bypassable by case or whitespace | normalised; unknown ids raise |
| — | Gate 1 unpassable under the prescribed workflow (Q4 structurally 0) | reads a directory of per-annotator corpora; Q2 keyed per annotator |
| — | formula guard checked raw text, stored sanitised text | checks run on the stored result; five bypasses pinned |

### Open

| | finding | why it is still open |
|---|---|---|
| B2 | the handbook describes a different round | needs a Phase 2C handbook written against the packet that exists. `NONE` is now importable, which was the sharp edge |
| B5 | CAMPAIGN3 §1.3's table is the retired key partition (56/91, not 54/93) | restating it changes a published side-condition figure; wants the owner's eye |
| B6 | `preregistered_pass` uncorrected for selection, and no Tier-1 stratum | changes a gate's arithmetic; belongs with a run, not a merge |
| C1 | `_build_fold_index_matrix` has the order dependence the probe fixed | moves every published interval; belongs with a rebaseline |
| C2 | the echo index guard is a length check, blind to reordering | mutation-proven; cheap, but touches the frozen partition |
| C5 | CSA AICM launderable as `csa_ccm` (203 of 243 shared control ids) | needs a packet-manifest digest on each link |
| C6 | the packet's 78 hubs **are** the eval gold hub set | a disclosure, not a code fix |
| C7/C8 | no packet lineage; `~$40` enforced nowhere | operational, before the round rather than before the merge |
| C9/C10 | no consent or attribution statement; the handbook gives away a stratum | lands with B2 |
| — | `results/bridge/` 46 Tier-3 edges still in `cre_hierarchy.json` with no per-edge marker | owner decision, deferred since round 1 |

**None of the open items blocks the merge.** Every one of them blocks *running
the round*, which is the correct place for that line to sit: the branch ships
tooling and a pre-registration, and the round has not started.
