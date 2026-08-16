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
