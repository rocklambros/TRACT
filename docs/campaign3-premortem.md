# Campaign 3 — adversarial premortem, and what it found

Run 2026-08-29 against `main @ 9f46715`, before any Campaign 3 money was spent.
Six perspectives attacked the coordinator's own recommendations independently in
clean contexts, then one orchestrator pass cross-examined them. 945k tokens, 7
agents, zero errors.

Every load-bearing number below was recomputed in the main session against the
repository, not accepted from an agent. Where an agent's figure did not
reproduce, that is recorded rather than quietly corrected.

---

## What was under attack

Five recommendations. The load-bearing one, R1, claimed the highest-value next
action was a ~$40 three-arm retrain rather than a curation program, because the
framework-hub graph splits into two disjoint components and a model could score
on the AI test split by learning "AI text → answer in the AI region" instead of
by mapping meaning.

R1 committed in advance to four falsifiers. That commitment is what made the
rest of this useful.

## Outcome, in one line each

| recommendation | outcome |
|---|---|
| R1 — firewall experiment first | **Premise survived, design deleted, hypothesis then refuted by measurement** |
| R2 — synthetic data bounded by label provenance | Survived; Tier-1 definition found factually false |
| R3 — pooled gate + non-echo side condition | Survived as a report, **fails as a decision** — no numeric threshold |
| R4 — curation order | Survived, but conflicts with R3 in a way neither noticed |
| R5 — three open items | Two fixed this session; one is an owner decision |

## The finding nobody was looking for

The premortem was aimed at Campaign 3's plan. It found something in Campaign 2:
`data/training/audit_corrections_log.json` rewrites **56 AI gold labels**, all
inside the four test-split frameworks, covering **25% of the 147-item test
corpus**, disclosed in no markdown file and covered by no test.

Removing those items drops the headline delta from **+0.1361 to +0.1000** —
exactly the gate value, `P(δ ≤ 0.10) = 0.531`. Full treatment in
`docs/campaign2-results.md` §13; reproducible via
`scripts/analysis/audit_stratified_delta.py`.

This is the strongest argument for running the exercise at all. It was found by
an agent told to work backward from "this failed", not by the campaign that
depended on it.

## What the measurement then did to the premortem itself

R1's hypothesis was **refuted** by a $0.27 probe (§14 of the Campaign 2
results). Handing the zero-shot encoder the entire 78-hub AI region for free —
deleting 444 of 522 candidates — moves exactly **one item in 147**.

The instructive part is how both estimates got there:

| estimate | claimed oracle gain | measured |
|---|---|---|
| Data Scientist (95-hub text-defined region) | +0.041 | — |
| Orchestrator cross-attack (78-hub link-defined region) | +0.095 to +0.155 | — |
| **Actual, paired Qwen baseline, 78-hub region** | — | **+0.0068** |

The orchestrator's cross-attack **inverted** the Data Scientist's conclusion,
calling it "refuted", and was **further from the truth than the finding it
overturned**. It reached that number from the BGE zero-shot standing in for the
Qwen baseline that actually produced the delta — a substitution it named, in its
own residual-risk section, as "the weakest link in this entire
cross-examination."

Two lessons, and the second is the one worth keeping:

1. A cross-attack moves beliefs in both directions, and can move them
   confidently in the wrong one. Adjudication is not automatically better than
   the findings it adjudicates.
2. **The agent flagged the exact weakness that made it wrong, and that flag did
   not stop it concluding.** Naming a limitation is not the same as being
   bounded by one. The only thing that settled this was spending $0.27 on the
   measurement.

## What survived and was acted on

- **Both R1 arms deleted before any spend.** F2 does not remove the domain cue:
  after rotating ENISA/ETSI/BIML out, the AI/general intersection is *still
  exactly zero* and 73 of 78 AI hubs remain AI-only supervised, while 40–49% of
  in-domain training supervision disappears. F3 cannot remove it either: a bare
  `\bAI\b` regex over the exact ranked hub text matches **78/78** AI hubs and
  **0/380** general ones. Both verified in the main session.
- **The disjointness is not a tautology.** Enumerating connected components of
  the framework-hub graph with *no* AI/general labels supplied yields exactly
  two: 380/14 and 78/8. R1's falsifier #1 answered in R1's favour — the premise
  was right even though the hypothesis was wrong.
- **F19 fixed.** Calibration items were identifiable three ways at once.
  `tests/test_review_calibration_blind.py`.
- **CSV formula injection fixed**, with the remedy split by column: prose is
  neutralised, identifiers are refused. Verified against the local OpenCRE fork
  — `parse_export_format()` contains zero occurrences of `eval`, `exec` or any
  formula handling, so escaping is safe for prose there, and corrupting for keys.

## Still open

1. **R3 has no numeric threshold.** It pre-registers a report, not a decision —
   verbatim the defect that withdrew Campaign 2. No Campaign 3 arm should run
   before `results/phase1b/CAMPAIGN3.md` commits to one.
2. **R4.1 conflicts with R3.** 53 of 147 test anchors (36%) are truncated at the
   2,150-char cap. Fixing text selection changes the eval anchors and makes any
   Campaign 3 number non-comparable to +0.1361. Both documents want their own
   thing and neither noticed. Owner decision.
3. **The audit's provenance.** The CSV carries no prediction, score or ranking
   column, so this is a human relabel — Tier 2, not model-seeded. Whether any
   model output was *visible* to the reviewer is not answerable from the
   artifact. One question to the owner, no compute, and it should be answered
   before the OpenCRE RFC cites these links.
4. **The forge blob.** Licensed ISO text reachable via `refs/pull/73/head`,
   removable only by a GitHub Support request. No perspective attacked it and I
   did not verify it. It is the one item here that gets worse with external
   attention, and the RFC submission is exactly that — so the Support request
   belongs *before* the RFC, not after.
