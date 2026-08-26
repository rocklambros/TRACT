# v3 author contradictions, and the orchestrator's resolution of each

All four v3 authors were told to report anything in the contract they found wrong. All four did,
with measurements. Where an author corrected me, the correction is adopted and the contract is
amended. This file is the record; `V3-CONTRACT.md` carries the amendments inline.

---

## Contract defects the authors found in MY work

### D1 — Rule 2's gitignore snippet was broken. Adopted, with a third form chosen.
I wrote `results/` plus `!results/corpus/**`. Two authors independently reported it stages nothing,
and gave DIFFERENT working forms (`results/*` vs `results/**`). I settled it by measurement on
git 2.50.1 rather than picking a side:

```
results/*  + !results/corpus/ + !results/ceiling_study/   -> both files staged, results/junk/ excluded
results/** + !results/corpus/ + !results/ceiling_study/   -> identical
results/** + dir AND glob negations                       -> identical
```

All three work. Only my original was broken, because git never descends into an excluded
*directory*, so no negation beneath it can rescue a file. **Standardised on `results/*` plus
directory negations** as the simplest form that says what it means. Both authors were right about
their own form; neither needs to change.

### D2 — Rule 5 was internally inconsistent by one anchor. Adopted.
I corrected biml to 19 in one row and separately carried `1,749 -> 1,902 = +153`, but 1,902 requires
biml at 20. With biml at 19 the arithmetic is `1,749 -> 1,901 = +152`, and because dsomm >= 182 and
wstg >= 55 are floors rather than exact counts, the honest form is **"+152 or more"**. Adopted.
This is the same defect class the premortem found in the plan, committed by me in the fix for it.

### D3 — Rule 1's `anchor_source_title` had no producer. Adopted, and it creates a cross-task duty.
`ProseIndex` never returns a `title` selection, so as specified the column could only ever read 0 —
a Rule 6 violation in my own contract. The Task 1 author redefined it as "the stored anchor restates
the control's own title" (fires when a parser writes `full_text == title`), and defined
`metadata["text_origin"] == "synthetic"` as the marker for `anchor_source_synthetic`.

**This is now a binding interface, and the parser-task author must honour it.** csa_ccm's 17 domain
aggregates, WSTG's merges and ENISA's Annex-C fallbacks must set `metadata["text_origin"] =
"synthetic"` or the column silently reads zero and the synthetic-text arm becomes invisible again,
which is the defect it was added to expose.

### D4 — Rule 7 referenced `EXPECTED_FRAMEWORKS`, which does not exist. Adopted.
Replaced with a measured `FULL_CORPUS_FRAMEWORK_COUNT = 31` (tracked 29, overlay 31) and a
`floors_for_report()` that returns the skipped group by name.

### D5 — Rule 1 could not satisfy Rule 6 for `wrong_anchor_risk`. Adopted, scope widened.
A bare count has no denominator, so no attainable range can be stated. The Task 1 author added
`resolution_rows` to `CorpusReport` and two booleans to the Rule 9 JSONL row
(`wrong_anchor`, `wrong_anchor_checked`), both text-free. Contract said "gains one field"; it now
gains two. Approved.

---

## Measurement corrections the authors made to the brief

Each was re-derived by the author rather than accepted from me. All adopted.

| I said | truth | why it matters |
|---|---|---|
| CWE recovers 17 links, reaching 613 | **612.** 613 is CWE's total; `937` resolves to no parsed control, so 16 are recovered | a gate asserting 613 would fail on a healthy run |
| training links reach 4,401 | **4,389** once `resolved_text is not None` is required. 4,401 is the loose-gate figure | see D6 below |
| `removed` will be ~113 | **exactly 111**: wstg 59, nist_800_63 25, owasp_proactive_controls 10, enisa 10, csa_ccm 7 | an exact number is a gate; "~113" is not |
| the corpus holds 4,271 records | **4,261**. The 39 shadowed records figure was right | |
| 9 WSTG links train on a literal id | **12 links** train on a title: 9 wstg + 2 iso_27001 + 1 dsomm | I undercounted the harm |
| the 7 published CCM ids are `csa_ccm:IVS-*` | they are `csa_ccm:csa_ccm:IVS-*`, double-prefixed | a matcher keyed on my form finds nothing |

### D6 — my brief contained two requirements that cannot both hold. Author's call adopted.
HIGH-9 (require `resolved_text is not None`, so `"WSTG-BUSL-$$"` cannot be an anchor) and HIGH-10
(training links reach 4,401) are incompatible: the strict gate drops 12 more links, giving **4,389**.
The author shipped the strict gate and stated both numbers. **That is the right call** — the whole
point of HIGH-9 is that a punctuation-bearing identifier must not train, and 12 fewer links is the
price. I set the target at 4,389 and retire 4,401.

### D7 — `renamed` will be 0 for all 111, and that is not a bug.
The eleven's baseline records are OpenCRE stubs where `description == title` and `full_text` is
absent, so no prose control can content-match one. The bucket is built anyway per Rule 8.5, and the
operator's actionable rule became framework-membership plus an exact `unchanged` count. Sound.

---

## New defects the authors found that were in NEITHER the premortem NOR my brief

### N1 — the fold metadata records the wrong corpus. Live provenance bug.
`tract/training/orchestrate.py:348` hashes `PROCESSED_DIR / "all_controls.json"`, the **tracked**
29-framework corpus. But `ProseIndex.load()` reads `merged_corpus_path()`, which prefers the
**31-framework overlay**. VERIFIED by the orchestrator. So the recorded `all_controls_sha256`
describes a file the run did not read, and `merged_corpus_path`'s own docstring claim — "A run that
used it and a run that did not are distinguishable: the fold metadata records the corpus sha256" —
is false. Two runs 92 links apart record the same digest.
**Left to Task 14 Step 6** (the author wrote `merged_corpus_sha256()` there) rather than fixed here,
to avoid a conflicting edit. Recorded so it is not lost if the plan stalls. No training run may be
launched before it lands.

### N2 — the prose gate is off for 19 of 21 parsers.
Only `iso_27001` (0.96) and `owasp_llm_top10_2026` (1.0) declare `min_prose_fraction`; the other 19
inherit the `0.0` default. VERIFIED by the orchestrator. Retrofitting 19 floors means measuring each
parser's true prose fraction and defending a number for it, which is its own plan. The Task 16
author's ratchet (hold declared floors, freeze the unfloored count at 19 so it cannot grow) is the
proportionate call and is adopted. **Recorded as a follow-up plan, not as "does not close" filler.**

### N3 — `test_no_version_field_says_opencre` is red TODAY on exactly 11 files.
All read `opencre-2026-04-28`, and those 11 are precisely this plan's eleven. All eleven report
`honest_prose_fraction` of exactly 0.0000. The plan's premise is confirmed from a direction nobody
aimed at: these frameworks are stubs, and the test that would have said so was already failing.

### N4 — pdfplumber is pinned at 0.11.10 and installed at 0.11.4.
So every `[measured]` PDF figure in the plan came from a build CI will not reproduce. Two authors
found this independently. The premise checks now assert the version.

### N5 — the ceiling-study gitignore negation exposes four unreviewed untracked files.
`answers_llm_proxy.json`, `ceiling_answers_LLM_PROXY.json`, `LLM_PROXY_report.md`,
`LLM_PROXY_score_report.txt`. **Ruling: these stay untracked for now.** The runbook is explicit that
an LLM pre-pass is a proxy and "a proxy labelled as a ceiling is the same category of error as the
withdrawn accuracy figure". Committing four files with `LLM_PROXY` in the name beside the real
250-item human study invites exactly that misreading. Whichever task first commits a ceiling-study
artifact must add them explicitly, with a header stating they are a proxy, or leave them out.

---

## Ruling R8 — the licence tier is a "do not make it worse" rule, not a legal distinction

The Task 1 implementer found that my `CONDITIONAL_FRAMEWORK_IDS` is incoherent as a licence
argument. Verified: **13** frameworks are copyleft under `FRAMEWORK_LICENSES`; my tier lists 7.
The 7 I left out are asvs, owasp_agentic_top10, owasp_cheat_sheets, owasp_dsgai, owasp_llm_top10,
owasp_llm_top10_2026 and owasp_ml_top10, carrying **691 curated links** (asvs 277,
owasp_cheat_sheets 391, owasp_llm_top10 13, owasp_ml_top10 10, three with zero).

If CC-BY-SA text cannot sit in a CC0 repository, all 13 must move. If NOTICE cures it, only
GPL-3.0 dsomm needs the overlay and I over-restricted five. Six-of-thirteen is defensible on
neither reading, and I should say so plainly rather than let the constant imply a legal finding.

NOTICE is stronger than I assumed when I wrote R4. It states: "The CC0 dedication does not, and
cannot, cover third-party framework content ... those terms continue to apply to that framework's
text wherever it appears in this repository or in artifacts built from it", and it names every
framework with its licence and upstream URL. That substantially discharges CC-BY-SA's attribution
limb. Share-alike is the open question, and GPL-3.0 is the genuinely contestable case, which is
why dsomm belongs in the overlay under any reading.

**Ruling, stated for what it is.** The tier is drawn on PUBLICATION STATE, not on licence class:

- Text this plan is about to write, which has never been published: route to the overlay. It is
  reversible in one direction and not the other, it costs zero training anchors, and all seven
  files are pure stubs today (measured: 0 prose controls across all of them), so nothing is lost.
- Text already tracked and already published under NOTICE: leave it, ratchet against growth, and
  let the owner decide. Moving it now un-publishes nothing, moves 691 links out of the tracked
  corpus, and entangles this plan with a decision that is not its own.

I am NOT claiming asvs and wstg are legally different. They are not. I am claiming that an
unattended run should not enlarge an exposure it cannot evaluate, and should not resolve a
published-artifact question while the owner is away.

**Owner decision, the sharper version.** The conflict is between the repository's CC0 declaration
and its bundled content. There are two levers, and only one of them is about the data:
  (a) remove the content: move all 13 to the overlay, 691 links leave the tracked corpus;
  (b) fix the declaration: state in LICENSE, as NOTICE already does, that CC0 covers TRACT's own
      contributions and that bundled third-party text retains its terms.
(b) is cheaper, changes no metric, and is what NOTICE already argues. It is a licensing call,
not an engineering one, which is why it is yours.

### Concern 2, from the same implementer, was worse than reported and is FIXED
All seven conditional files were **tracked**, and git applies no ignore rule to a tracked path.
The seven new `.gitignore` lines were therefore inert and the whole tier was decorative:
`git check-ignore` reported every one of them unignored. Untracked with `git rm --cached` (files
kept on disk), and `test_no_overlay_framework_is_still_tracked` now asserts it. Verified
non-vacuous in both directions: re-adding dsomm to the index fails the test, removing it passes.
The implementer's suggestion that "the first parser task to rewrite one owns the `git rm --cached`"
is exactly the decorative-control shape of ledger lesson 4, so it was done now instead.
