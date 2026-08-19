# TRACT autonomous run ledger

Owner authorization: 2026-08-15, explicit, unattended, RunPod spend approved.
Standing instruction: most correct and comprehensive approach regardless of
cost, refactoring, or complexity. No shortcuts. Premortem every spec, plan and
implementation. Carry lessons forward into downstream phases before executing
them.

Spec: docs/superpowers/specs/2026-08-15-semantic-rebuild-design.md (v2)
Branch: semantic-rebuild

## Read this first after a compaction

Trust this file and `git log` over recollection. A phase with a `COMPLETE` line
is done. Resume at the first phase without one.

## Standing constraints that must not be violated unattended

1. **Licensed text never enters git.** Gate: `pytest
   tests/test_licensed_text_not_tracked.py`. It is GREEN as of 963da76. ISO and
   CSA CCM raw files live in gitignored `data/raw/`; ISO processed output and
   the merged licensed corpus live in gitignored paths.
2. **Never republish to HuggingFace.** The owner approved RunPod spend, not
   publication. The model card and dataset card carry errata; new numbers reach
   them only after a pre-registered gate AND explicit approval.
3. **All model loading on RunPod, never locally.**
4. **No AI attribution** anywhere. Author stays the human.
5. **Never `git push --force`.** Denied at the harness rule layer anyway.
6. **Tear down every pod.** Verify via REST API after each phase. `reap` could
   not see validation-split pods before the Plan 1 fix wave; re-verify.

## Budget

~$1,850 of $2,000 remained at run start. Record spend per phase below.

## The one thing this run cannot do

Spec Part 0.1 requires a **250-item blind human agreement study** to establish
the ceiling alpha-1 / alpha-5. It needs the owner personally making 250 expert
judgments. It gates stop condition S1 and, per the spec, "the meaning of every
metric."

Decision for this run: do BOTH of
- **Proxy ceiling**, measurable now: run the LLM panel as an independent second
  annotator against OpenCRE ground truth on the same 250-item sample. This
  yields a defensible ceiling ESTIMATE. It is not the human ceiling and every
  artifact that cites it must say so.
- **Prepare the human study** so it is one command away on the owner's return:
  export the 250-item blind review file via the Phase 3 machinery.

Any conclusion that depends on the ceiling is reported against the proxy with
its interval, and flagged as provisional until the human study runs.

## Phases

| # | phase | gate before executing | status |
|---|---|---|---|
| A | Plan 1b: fetch 10 sources + 11 remaining parsers | premortem the plan | pending |
| B | OpenCRE re-fetch, full CRE-to-CRE graph | premortem | pending |
| C | Ceiling: proxy study + human study prep | premortem | pending |
| D | hub_graph.json + the 4.1 circularity gate | premortem | pending |
| E | Target artifacts: semantics, contrast, evidence | premortem | pending |
| F | Diagnostics: char-TF-IDF, recall@50, ASVS, arm A5 | premortem | pending |
| G | Panel first pass, the two load-bearing checks | premortem | pending |
| H | 12-config bake-off | premortem + pre-registration commit | pending |
| I | Serving path (spec Part 7) | premortem | pending |
| J | PRD amendment, docs, PR | premortem | pending |

Stop conditions S1, S2, S3 (spec Part 0.3) can end phases G through I. If one
fires, record the action taken, do not re-litigate it as "a finding."

## Lessons carried forward from Plan 1

These are the defect classes this run has already produced. Every downstream
plan is checked against them before execution.

1. **Guarding one channel while another stays open.** Licensed text escaped
   three times: publish path, merge step, test fixtures. Ask of every control:
   what is the OTHER way this data reaches the outside?
2. **Specifying a change to a component nobody opened.** `label_space` was
   called dead code while live in the description generator;
   `generate_descriptions.py` raises on exactly what the plan asked for. Read
   the file before writing the task.
3. **A gate that cannot fire.** The budget check maxes at $780 against a $2,000
   threshold. `MAX_RUN_TOGETHER_REPAIRS` sat at 10x the actual. Compute the
   attainable range of every threshold and assert it contains the trigger.
4. **A control that is decorative.** `prose_floor_deviation_reason` looked like
   a gate and was read nowhere. If a field looks like a control, wire it or
   delete it.
5. **Comparing against a baseline captured differently.** "Identical failure
   set" was measured in a worktree lacking `data/`, which masked 9 regressions.
   Baselines must be captured under identical conditions.
6. **A step that precedes what rewrites its inputs.** Three occurrences across
   two spec versions. The `invalidates` column is now mandatory.
7. **Transformation that fabricates.** The bleed repair invented a normative
   statement that passed every gate. Any transform that moves or synthesizes
   text must emit an audit record and fail closed when it cannot verify.
8. **Asserting a number without an artifact.** `0.2212` had no source. Every
   number carries [measured] / [derived] / [unmeasured].

## Log

**Ruling R1: Phase A splits into A-fetch then A-parsers, and only the parser
plan gets a full six-perspective premortem.**
Lesson 2 says do not specify a change to a component nobody opened. I cannot
write 11 parser specifications without seeing the 11 source documents, so the
fetch must complete first and the parser plan is written against real
structure. The fetch is one file plus source definitions; its risk is
concentrated in supply-chain hardening, which the Plan 1 premortem already
diagnosed precisely, so it gets a targeted review rather than a six-lens
premortem. The parser plan, where the real risk lives, gets the full treatment.
*Cost if wrong:* a defect in the fetch script surfaces during fetch, cheaply.

2026-08-15 Licensing gate GREEN (963da76). 0 tracked files quote ISO
statements, verified independently. ISO parser unchanged: 93 controls, 0.967
prose, repairs 4/32/10.
2026-08-15 CSA CCM v4.1.0 staged, sha256 5e721628..., 569961 bytes, gitignored.
Note: OpenCRE's 29 CCM links may key to v4.0; verify during the parser phase.

**Phase A-fetch COMPLETE** (863504e + 800-63 correction).
11 frameworks staged. Hardening landed: expected_sha256 raises on drift,
--accept-new-hash is the only re-baseline path, GitHub sources pinned to commit
SHAs, per-source headers (ETSI needs a browser UA).

Findings that change the parser plan:
- **nist_800_63 was the wrong revision.** My brief passed the 800-63-4 URL from
  a memory note. Measured 0/25 id coverage; revision 3B gives 24/25. Fixed.
  Lesson 8 again: a URL carried from a note is an [unmeasured] claim.
- **BIML needs BOTH PDFs.** Measured 21 links / 20 distinct ids / 17 names, not
  the 14/12 my brief asserted. 8 ids prefixed BIML-78(2020) match ara.pdf, 4
  prefixed BIML-24(LLM) match BIML-LLM24.pdf, 8 unprefixed resolved by tag
  position, 1 ("output:2") is an imperfect match and must be flagged, not guessed.
- **ENISA has no stable control ids.** 68 links, only 10 distinct ids and several
  are "Table 3:" placeholders; 33 distinct names. The join must key on name.
- **ENISA and NIST SSDF need pdfplumber extract_tables(), not extract_text().**
- samm's default branch is `develop`, not `master`.
- nist_800_63 stays unpinned: Cloudflare injects a per-response nonce, so a pin
  would make --accept-new-hash routine rather than an alert.

**Owner decision 2026-08-16: CSA CCM IS redistributable.**
Given explicitly, with a standing instruction not to stop and ask. Consequences:
- csa_ccm does NOT join RESTRICTED_FRAMEWORK_IDS. ISO 27001 remains the only
  member.
- `data/processed/frameworks/csa_ccm.json` is a normal tracked artifact and its
  prose may reach the tracked all_controls.json and the published dataset.
- The licensed-text gate still scans the whole tree; it targets ISO statements
  only, so no change is needed there.
- The CCM v4.1.0 vs OpenCRE-v4.0 id-overlap measurement is still required. That
  is a correctness question, not a licensing one, and it stands.

**Standing instruction reaffirmed: do not stop to ask. Rule and record.**

## Premortem of Plan 1c — two Criticals, both verified by me, both blocking

**Ruling R2: licensing. Three actions, none needing the owner, all reversible.**
Verified from the staged archives and PDFs: DSOMM is GPL-3.0; WSTG, SAMM and
OWASP Proactive Controls are CC BY-SA 4.0; ETSI's PDF says "No part may be
reproduced ... except as authorized by written permission of ETSI"; ENISA
requires acknowledgement. The plan would have committed all of their normative
prose to a public CC0 repository.

Also verified: this is PRE-EXISTING, not created by the plan. owasp_cheat_sheets
is CC BY-SA 4.0 and its processed JSON is already tracked, one of 31 tracked
framework artifacts. The CC0 file has always been over-broad.

1. **ETSI joins RESTRICTED_FRAMEWORK_IDS.** Its notice is unambiguous and needs
   no judgment. Processed output goes to the gitignored overlay like ISO.
2. **Per-source `license` field** on Source and in the manifest, populated from
   each source's own LICENSE. Records what is currently invisible.
3. **NOTICE file** stating third-party framework content retains its original
   licence and the CC0 dedication covers only TRACT's original contributions.
   This makes the EXISTING state honest rather than changing posture. Flagged
   for owner legal review on return; it does not block.
*Cost if wrong:* ETSI restricted when it need not be, recoverable with one line.
The NOTICE is strictly more accurate than the status quo either way.

**Ruling R3: the licensed-text gate is inert in CI and every fresh clone.**
Reproduced: move the gitignored ISO source aside and the tree-wide test reports
"2 passed, 1 skipped". CI has no fetch step, so data/raw is never present there.
The control I committed as the fix for the third escape only ever ran on one
laptop. This is ledger lesson 3 landing on the licensing control itself.
Fix: the gate stores SALTED HASHES of normalised statement n-grams in a tracked
file, so it can detect verbatim licensed text without containing any. Works
with no data/raw. A skip becomes a failure, not a pass.
*Cost if wrong:* a hash set needs regenerating when a source is re-pinned.

## Plan 1c premortem: 4 reports, ~30 findings, 10 Critical. PLAN REJECTED.

I am not executing this plan. The defects are structural, not patchable.
Consolidated must-fix list, all verified by the reporting agent against real
sources, several re-verified by me:

CRITICAL
C1 ISO resolves 0/94 links. SHIPPED defect: framework_name never matches
   standard_name and no alias bridges them. Prose was measured, reachability
   never was. Fix routed to the licensing agent.
C2 Four sources conflict with CC0 (ETSI all-rights-reserved, DSOMM GPL-3.0,
   WSTG/SAMM/ProactiveControls CC BY-SA). Pre-existing, widened by the plan.
C3 The tree-wide licensed-text gate skips in CI and every fresh clone.
C4 The join gate self-calibrates (floors pasted from the run being gated) AND
   three floors exceed the arithmetic maximum: dsomm max 0.9953 vs floor 1.00,
   wstg max 0.9322 vs 0.96, enisa max 0.721 vs 0.80. Both decorative and
   unreachable at once.
C5 ENISA resolves 49/68, not 55. Three causes the plan never names: footnote
   digits fused to control names, curly punctuation vs OpenCRE ASCII, and 4 of
   13 Table 3 threats extracting with empty definitions, including Evasion and
   Poisoning, the two most-linked.
C6 ProseIndex resolves TITLE BEFORE ID, which defeats BIML's document-scoped
   ids (7 rows collide on shared labels) and hands 28 of ETSI's 36 links a
   shared anchor. This reintroduces the exact NIST AI 100-2 collapse that the
   title-first order was written to fix.
C7 Task 9's premise is false. The SSDF table is fully ruled; extract_tables
   returns whole cells and absorbed=0. Task 9's two headline tests FAIL against
   Task 7's implementation.
C8 Task 7 adds a text-moving repair with no audit record, violating the plan's
   own Global Constraint and lesson 7.
C9 5,238 published assignments' review claim is falsified by the rebuild.
   Cheap half CAPTURED at this commit; durable fix needs the schema column.
C10 The plan cannot tell whether the corpus got better or worse. Its only
   instrument counts links, not anchors, truncation, or eval-item identity.

HIGH: csa_ccm expected_count off by one (header row counted, would raise);
CCM domain aggregates exceed MAX_ANCHOR_CHARS for 8 of 17 and each is a prefix
of its own member control; Task 16 rebuilds all 31 frameworks including CAPEC
and CWE with no byte-identical assertion; no rollback or snapshot; the deferred
278-link gate retirement puts the ceiling study on a roster that changes again;
two different join instruments with the permissive one used as the acceptance
gate; the task table mixes pre- and post-gate counts so 155 credited links buy
nothing; four parsers have no test that instantiates them; 36 `python3` calls
against an interpreter the plan itself documents as having no dependencies.

**Gap caught by the owner 2026-08-16: OWASP LLM Top 10 2026 was never ingested.**
The owner supplied it, spec Part 1.6 specifies it as the pretraining-
contamination control, and the Plan 1c author deferred it as "blocked on the
three-way licence conflict". That block is stale: the licensing model landed in
443b0c7 records CC BY-SA 4.0 like any other source, and NOTICE covers it.

Source now staged: data/raw/frameworks/owasp_llm_top10_2026/,
sha256 3d3c9f21809c5f882a668b87424ac6b2e2a270caab4b29aa5265df3475433a96,
gitignored. CC BY-SA 4.0 per its own licence block.

It is NOT a normal parser task and the rewritten plan must say so:
- separate framework id `owasp_llm_top10_2026`, never overwriting the 2025 file
  whose LLM0x:2025 ids carry all 13 OpenCRE links
- held out of EVERY training roster and EVERY fold roster, asserted by test
- parser stops at "## Appendix A" or LLM10 swallows 937 lines of back matter
- Appendix A parses as a SEPARATE mapping artifact: 48 expert LLM2026-to-CWE
  mappings over 22 CWEs, all 22 resolving to CWE 4.20, 17 carrying OpenCRE
  links, giving 37 transitive chains over 46 hubs covering all 10 risks
- version pins to the sha256, not a date: the revision history still reads
  "[2026 release date]", so this is pre-release

## PHASE C COMPLETE 2026-08-18: the ceiling is measured. It reframes the project.

Owner completed all 250 items. 173 high confidence, 71 medium, 6 low, 84 with
notes. Half-width 0.058, inside the 0.059 design target.

```
                 alpha-1                    alpha-5
pooled    0.572 [0.510, 0.632]     0.660 [0.599, 0.716]
validation 0.296 [0.223, 0.381]     0.408 [0.326, 0.496]
test       0.848 [0.775, 0.900]     0.912 [0.849, 0.950]
```

Per framework, alpha-1: capec 0.181 (n=83), cwe 0.464 (n=28),
nist_800_53 0.643 (n=14), mitre_atlas 0.721 (n=43), nist_ai_100_2 0.773 (n=22),
owasp_llm_top10 0.833 (n=6), owasp_ai_exchange 0.981 (n=54).

**The corpus is two populations, not one.** A 55-point alpha-1 gap between
strata whose intervals do not come close to overlapping.

**CAPEC's ground truth is barely agreed.** alpha-1 0.181 means a domain expert
and OpenCRE's curators pick the same best hub for a CAPEC attack pattern fewer
than one time in five. CAPEC is 42.8% of the training graph. Nearly half the
training signal carries labels an expert mostly disagrees with.

**S1 does NOT fire.** Like-for-like on the three frameworks the validation
stratum sampled (capec+cwe+nist_800_53, n=895 model items):
model hit@1 0.103, hit@5 0.267, hit@10 0.379 against ceiling alpha-1 0.296 and
alpha-5 0.408. The model sits at 35% of the alpha-1 ceiling and 65% of alpha-5.
Headroom is real but far smaller than anyone assumed.
CAUTION RECORDED: my first read used hit@5 0.349 from the full validation
roster, which includes ASVS and ISO and is a different population from what the
ceiling sampled. That would have made the model look like it was at 86% of
ceiling and nearly triggered S1. Lesson 5 again, caught before it mattered.

**This explains "fine-tuning is net zero" without needing a model defect.**
Training is dominated by a framework whose labels an expert agrees with 18% of
the time. A model that fit those labels well would be fitting noise.

Implications the spec must absorb before Part 5:
- Label agreement is now a measurable per-framework property. It belongs in the
  training mix as a weight, not as an assumption of uniform quality.
- Reporting one pooled metric over two populations this different is the same
  pooling error the I2=95% heterogeneity finding already flagged.
- The AI roster has both good labels (alpha-1 0.848) and real headroom
  (model 0.517). That is where model work pays.
- owasp_ai_exchange at alpha-1 0.981 is the cleanest label set in the corpus.
- cwe alpha-1 equals alpha-5 at 0.464: the acceptable-set never helped, so its
  disagreements are not near-misses.

## Correction to a standing constraint I have been getting wrong

**The interpreter I prescribed in every dispatch is the wrong one.** Measured:
```
/Users/klambros/.local/share/uv/tools/tract/bin/python3   pytest=n pdfplumber=n openpyxl=n mypy=n
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3   all present
```
Bare `pytest` already resolves to the Framework 3.12 interpreter. Every future
dispatch uses:
  PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
Agents worked around the bad instruction rather than failing on it, which is
why it survived this long.

## Plan v2 landed and corrected 13 numbers, several of them mine

Notable corrections to what I passed on from the premortem:
- **csa_ccm 224 vs 225 does NOT raise.** 0.44% deviation against a 10%
  COUNT_TOLERANCE, so it passes silently. I told the plan author it would
  raise. Passing silently is the worse failure and the plan now handles it.
- **ENISA's ceiling is 68/68**, not 0.721. The premortem's 49/68 measured naive
  matching, not what a correct parser can achieve with NFKD and footnote
  repair. Naive exact 51/68, normalised 62/68, full repair 68/68.
- **Retiring PHASE1B_DROPPED_FRAMEWORKS alone recovers nothing.** All 155
  section_names in nist_800_63 and owasp_proactive_controls are ALSO under 10
  chars, so both gates must retire together or neither does.
- **615 links is the twelve including ISO.** The eleven pending contribute 523
  post-gate, 734 pre-gate.
- **ETSI alt_titles would create wrong-anchor risk**: three technique names span
  two clauses each, so only 2 alternates are safe, not 24.
- BEFORE state measured: 3,666 of 4,405 curated links resolve, 1,450 distinct
  anchors, 559 truncated, 522 controls never indexed.

**Live defect found in passing:** asvs, owasp_cheat_sheets and owasp_ml_top10
produce byte-different JSON today at IDENTICAL control text, because their
archives were re-fetched and re-pinned after the tracked JSON was written. The
entire diff is `source_files`. Task 15 therefore asserts on description hashes,
not file bytes. Also `defusedxml` is pinned but not installed, so parse_capec
and parse_cwe cannot be imported at all.

## PANEL COMPLETE 2026-08-18. CAPEC's labels are the problem, not the annotator.

Five models, five labs, 250/250 items each, $7.02 total. Backends pinned and
recorded. Head 91af54a.

```
model              alpha-1 pooled          alpha-1 CAPEC
Kimi K3            0.588 [0.526,0.647]     0.217
GLM-5.3            0.576 [0.514,0.636]     0.229
DeepSeek V4 Pro    0.580 [0.518,0.640]     0.253
Grok 4.20          0.484 [0.422,0.546]     0.183
Llama 4 Maverick   0.256 [0.206,0.314]     0.060   <- weak judge, see below
human              0.572 [0.510,0.632]     0.181
```

**The question the panel existed to answer is answered.** CAPEC contingency
over 83 items, verified independently by me:
```
human + panel majority agree, OpenCRE differs   44
all three differ                                23
both match OpenCRE                              15
panel matches OpenCRE, human does not            1   <- the decisive cell
```
Leave-one-out: the 44 cell ranges 43-47, reverse cell 0-4. No single model
carries it.

Human-to-panel-majority agreement is 0.780 pooled and 0.711 on CAPEC, against
human-to-OpenCRE on CAPEC of 0.181. Two blind readings agree with each other
four times as often as either agrees with the published label.

Under the hypothesis "the annotator's reading is idiosyncratic", the 44 cell
should be near zero: five unrelated labs have no route to one person's private
confusion, and the reverse cell is 1. **That hypothesis is refuted.** CAPEC's
alpha-1 of 0.181 is a statement about OpenCRE's CAPEC links, which are 42.8% of
the training graph and are AutomaticallyLinkedTo, meaning derived transitively
and never human-reviewed.

**Caveats that must travel with this finding:**
- Llama 4 Maverick is a weak judge, not merely a low scorer: 17 invented hub
  ids, alpha-5 only +0.016 over alpha-1. It is in the majority vote and the
  headline survives dropping it (47/4).
- Contamination probe: 1,250 closed-book recall attempts produced 105 emitted
  ids and ZERO correct against 1/522 chance; an exposure control asking for 50
  hubs by id produced 2 answers, both wrong. High confidence that id-level
  memorisation is not inflating the result. LOW-TO-MODERATE confidence that
  contamination is absent generally: CAPEC is heavily represented on the public
  web, so topic-level familiarity could push the panel toward the same reading
  the human reached, and the probe cannot bound that channel.
- Three of five models are Chinese frontier labs and may share lineage. The
  2 non-CN models (Grok, Llama) both sit lower, which is worth noting rather
  than explaining away.

**Three API behaviours that cost real money to discover**, now in PANEL_README:
`reasoning_effort` as a bare string is silently ignored by OpenRouter and runs
at full effort; `max_tokens` above a pinned backend's cap returns
"No endpoints found", which reads like the model not existing; and
`finish_reason: "length"` returns non-empty text that parses to nothing and
would have silently dropped 25 items from the denominator.

**Strategic consequence for the rebuild.** Label agreement is now a measured
per-framework property. Training that weights CAPEC at 42.8% is weighting the
least-agreed labels in the corpus most heavily. This belongs in the spec before
Part 5, and it is a data decision, not an architecture one.
