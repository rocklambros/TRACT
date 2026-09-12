# Phase 2C Gate 2 — execution plan

**Status: binding once Amendment 1 to `docs/phase2c-preregistration.md` is
committed. Supersedes the draft reviewed at `66d9287`.**

**Deliverable, as stated by the owner 2026-09-11: the bridge corpus, and whether
it helps the model.** Not the `NONE` rate, not a claim about how connected the
two domains are.

The draft this replaces was taken apart by a six-perspective adversarial
premortem; the round is `docs/phase2c-premortem-gate2.md`. Three of its findings
refuted the draft by measurement rather than argument, and the design below is
rebuilt around them rather than patched.

---

## 1. Why the draft was replaced

Three numbers, all re-derived from committed artifacts before renting anything.

**The treatment could not reach the measurement.** Under the strict all-AI
firewall the round-2 corpus supervises 18 of 50 eval items, **all 18 ENISA**.
BIML — 17 items, 34% of the denominator — has zero exposure at the counting floor
and could only ever have contributed noise.

```
eval: 50 items (ENISA 33, BIML 17); 32 ground_truth hubs; 56 valid_hub_ids (SCORED)
round 1 conf>=2: 16/50 exposed   {'ENISA': 16}
round 2 conf>=2: 18/50 exposed   {'ENISA': 18}
```

The draft's "32 AI-only gold hubs" was also the wrong denominator:
`evaluate.py:355-365` credits hit@1 against `valid_hub_ids`, of which there are
**56**.

**The comparator was degenerate.** Under the strict firewall a bridge-free arm
has **zero** training positives for all 56 scored hubs, because the AI and
traditional hub regions are disjoint. A PASS would have licensed only "some
supervision beats none."

**The interval did not contain the dominant variance.** `paired_bootstrap_delta`
resamples items, not training draws. Measured run-to-run discordance between two
same-arm runs in this repository is 19.1% per item, fold-drift SD 13.5pp, at
which the nominal 2.5% false-PASS rate is really **13.4%**.

Separately: the draft picked round 1 as primary on the **link/no-link** κ, the
one statistic this project has documented as moved by the round-1 instruction
wording, while hub agreement — which decides whether a link is *correct* —
favours round 2 (0.8909 on n=55 vs 0.8846 on n=26). And the draft's stated
ground, "zero of the 106 additions is confidence 3", is blind to the 86 retained
links, of which **seven moved 2→3**, growing the confident population 16 → 22.

## 2. The corpus, decided on the ground that survives

**The shipped corpus is round 2, the union.** Round 1 is strictly nested inside
it — every round-1 link present, identical hub, none withdrawn — so the union
loses nothing and carries both volunteers' full contribution. Round membership is
a per-link field, so any round-1-only analysis stays reproducible.

Three reasons, in the order they bind:

1. **Test the artifact you ship.** The deliverable is a corpus. Publishing the
   union while gating on a subset would measure something nobody receives.
2. **The clean agreement statistic favours it**, and the contaminated one is not
   admissible under `docs/phase2c-results.md:117-120`.
3. **The statistical case for either is void anyway** — the two corpora differ by
   2 eval items of exposure, so the choice was never going to be decided by
   power.

There is no round-1 arm. The draft's A2−A1 contrast is cut: 83 of the 91 counting
additions land on hubs round 1 already covered, only 3 hubs are new and 2 of
those are scored, so the contrast buys **2 eval items** while carrying the sum of
two training-draw variances. At this n it is a guaranteed null whose only
readable meaning is the claim `docs/phase2c-results.md` §4 forbids.

## 3. The evaluation

**ENISA + BIML + ETSI — 74 items, 62 scored hubs.**

ETSI is added because it raises exposure from 18/50 to 27/74 **and** makes the
exposed stratum span two frameworks instead of one:

```
+ETSI r2 conf>=2: exposed 27/74  {'ENISA': 18, 'ETSI': 9}
```

ETSI's gold is audit-untouched Tier 1 (`data/training/ai_link_audit.csv` touches
only OWASP-AIX, ATLAS, NIST-AI, LLM-T10, ML-T10), so the stratification rule of
`CAMPAIGN3.md` §2 is satisfied by construction, as it is for ENISA and BIML.

**ETSI requires a redaction fix first, and does not ship without it.** ETSI is in
`RESTRICTED_FRAMEWORK_IDS`; `orchestrate.py:371` writes verbatim `control_text`
into `predictions.json`; `.gitignore:43` negates `results/phase1b/**/*.json` back
into tracking. Adding ETSI without the fix would commit ETSI control statements
to a CC0 repository. Step 5.2 replaces the field with `control_text_sha256`,
which is what the arm-vs-arm comparison actually needs.

### Strata, fixed before any model runs

Computed from the committed corpus and committed as
`results/phase2c/gate2_strata.json` before the first pod:

| stratum | n | definition |
|---|---|---|
| **exposed** | 27 | ≥1 hub in `valid_hub_ids` receives a bridge positive |
| **unexposed** | 47 | no hub in `valid_hub_ids` receives any bridge positive |
| *of which BIML* | 17 | **negative control** — zero exposure by construction |

The partition is a property of the corpus and the gold links only. It cannot move
with the arm being measured, which is what `CAMPAIGN3.md` §3 requires of a
binding partition.

## 4. Arms

| arm | corpus | purpose |
|---|---|---|
| **A0** | none | bridge-free comparator |
| **A0′** | none, seed+1 | noise floor — the only thing that says whether a delta is readable |
| **A1** | round 2 union, confidence ≥ 2 | the treatment |
| **A0R** | random placebo, size- and framework-matched to A1 | isolates *judgment* from *the arrival of any AI-hub positive at all* |

Four retrains. The pre-registration budgets one at ~$40; this is four at ~$160,
recorded in Amendment 1 with its justification rather than absorbed silently.

A0R is what makes a PASS mean something. Without it, A1 − A0 answers "does any
supervision beat none" on a comparator with zero positives for every scored hub.
A0R draws the same number of NIST 800-53 controls onto the same AI hubs under a
declared seed, so A1 − A0R is the annotators' judgment with the supervision
volume held constant.

## 5. Criterion

**Primary — difference-in-differences, `ci_low > 0`:**

```
DiD = delta_exposed − delta_unexposed        (A1 vs A0)
```

fold-stratified paired bootstrap over the three framework strata, seed 42,
`threshold = 0.0`, `n_configurations = 2`.

DiD rather than the pooled delta because it is **both better calibrated and more
powerful** here. Under the global null with fold drift σ=0.10 the pooled contrast
false-passes at 0.086 and the DiD at 0.016; at a true 50% conversion on exposed
items the pooled contrast passes 0.363 and the DiD 0.419. Global training-draw
drift moves both strata together and cancels in the difference — which is exactly
the variance the pooled interval cannot see.

It is also the mechanism test the deliverable needs: a positive pooled delta with
a **flat** DiD says the gain was churn or a global training-mix effect, not the
bridge links.

**Noise-floor guard.** Compute the same DiD statistic for A0′ vs A0, where the
true effect is zero by construction. If `|DiD(A0′, A0)| ≥ DiD(A1, A0)`, the
instrument cannot resolve the effect and the round returns **NO VERDICT**, not a
FAIL.

**Negative control.** If `delta_BIML ≥ delta_exposed`, the movement is not
attributable to bridges and the round returns NO VERDICT.

**Secondary — `A1 − A0R`**, same statistic, reported with its interval.

**Descriptive, never a verdict:** pooled `A1 − A0` over all 74 items; per-stratum
deltas; per-framework deltas.

### Outcome table — every combination decided in advance

| primary DiD(A1,A0) | secondary A1−A0R | verdict | what may be claimed |
|---|---|---|---|
| `ci_low > 0` | `ci_low > 0` | **PASS** | The corpus helps, and the annotators' judgment is what helps. |
| `ci_low > 0` | not | **PASS (weak)** | Supervision on these hubs helps; this round does not show the judgment beat a random assignment. |
| not | any | **FAIL** | Not shown to help at this power. A null here is weak evidence of absence and must be reported as such. |
| noise-floor guard fails | any | **NO VERDICT** | The instrument could not resolve an effect of this size. Report descriptively. |
| negative control fails | any | **NO VERDICT** | Movement is not attributable to the corpus. |

No arm may be re-selected after the numbers land. No metric substitution. A
PASS on the secondary does not rescue a failed primary.

### What a PASS does not license

The annotator hub sheet was built from `BRIDGE_AI_FRAMEWORK_IDS`, which includes
ENISA, BIML and ETSI — so the candidate universe was constructed from the same
curated links that supply the eval gold (checkpoint-2 item C6, still open). A
PASS licenses *"bridges aimed at this hub region help on this hub region"*, not
*"bridges generalize."* A future round should rebuild the sheet with the eval
frameworks excluded.

## 6. Sequence

Nothing in 5.x is optional and nothing after 5.x starts until 5.x is green.

**5.0 — Commitments that must precede an irreversible step.**
   1. Close the withdrawal window in writing with both annotators, since training
      puts their contribution into weights that cannot be un-trained. The
      handbook promised withdrawal "before publication"; this is the last moment
      that promise can be kept.
   2. Commit the publication artifact — de-identified corpus projection plus
      manifest and both Gate 1 reports — **before** the run, so no verdict can
      shelve it.
   3. Commit Amendment 1.

**5.1 — Corpus.** `scripts/build_gate2_corpus.py`: merge the round directory,
apply the confidence ≥ 2 floor on the *training* path (it currently exists only
in the gate reporter), declare and apply the dedup rule, sort deterministically,
emit a committed manifest with counts and digests.

**5.2 — Artifacts.** `predictions.json` redaction for restricted frameworks;
`ARM_DEFINING_KEYS` += `bridge_links_path` and the exclusion set; campaign label
carries the bridge arm; staleness resolves the bridge path from the fold record;
launch lock keyed on `config_name`.

**5.3 — Firewall.** `held_out_frameworks: frozenset[str]` through the six
signatures; the converse assertion that the exclusion removed exactly the links
it should have; `run_fold` refuses `--framework ENISA|BIML|ETSI` under `--split
validation` by name, because that path runs today and silently leaves 52 of 56
gold hubs supervised.

**5.4 — Comparison.** `scripts/analysis/gate2_delta.py`: loads two arms, asserts
item alignment by `control_text_sha256`, asserts the arms differ *only* in
`bridge_links_sha256`, computes DiD and pooled with `threshold=0.0`. Not
`gate_decision` — that function is hardwired to a zero-shot baseline this gate
forbids, at a threshold this gate does not use.

**5.5 — Bootstrap.** `_build_fold_index_matrix` per-fold independent
`SeedSequence` children, plus a determinism test: the same data with folds
permuted must give a bit-identical `ci_low`.

**5.6 — Money and data handling.** `try/finally` teardown; `max_usd_per_hour`;
wall-clock deadline; pod name inside the reaper's sweep; `require_secure_cloud`
true when a bridge corpus is present, not only when the licensed overlay is;
rsync excludes for `data/training/bridge*`, `claudedocs`, and `results` wholesale.

**5.7 — Dry run, local, no model.** Confirm 74 items, 62 scored hubs, the 27/47
partition, and that the exclusion removed exactly the expected links.

**5.8 — Run.** One pod, four arms, per-arm output directories.

**5.9 — Score, report, tear down.**

## 7. What will be reported

The DiD and its interval for the primary; the noise floor; the negative control;
the secondary against the placebo; the pooled delta as a descriptive figure; the
power the design had; and an explicit statement that a null at this exposure is
weak evidence of absence.

**The corpus, its manifest, the agreement figures and both Gate 1 reports are
published regardless of the verdict, and are committed before the run so that
this sentence cannot quietly fail to happen.**

A null on any contrast here is uninformative about the link rate, the `NONE`
rate, or the connectedness of the two hub regions. `docs/phase2c-results.md` §4
governs those questions and is not superseded by any result in this round.
