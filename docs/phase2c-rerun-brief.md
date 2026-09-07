# Phase 2C — re-run brief

**Part 2 is what the annotator receives. Part 1 is for the coordinator, and it
contains the reason for the re-run, which the annotator must not be told.**

---

# Part 1 — for the coordinator

## Why this exists

The first round produced 87% and 85% `NONE`. That may be correct — the AI and
traditional hub regions are disjoint by construction, and the design named "too
few links" as the most likely outcome. The link rate is also sharply
domain-structured (48% in System & Information Integrity, 0% in Physical &
Environmental and Personnel), which is the shape of judgement, not of a nudge.

But the first handbook nudged toward `NONE` four times, and one of those
asserted the answer: *"Most NIST 800-53 controls are about traditional IT
security and have no AI-specific hub."* That is the experimenter telling the
annotator what the distribution should look like. The effect cannot be measured
from round one, because there was no arm without that sentence.

This re-run removes it and measures what changes.

## What this design can and cannot establish

**It cannot produce an independent second measurement.** The same people are
re-reading controls they have already judged. Anchoring is unavoidable; they
will remember some.

**That contamination runs toward the null,** which is what makes the round
usable. Anchoring makes an annotator repeat their earlier answer. So:

- If answers barely move, the biased sentence was probably not driving the
  `NONE` rate. Anchoring predicts that outcome too, so it is *weak* evidence —
  report it as consistent-with, not as proof.
- If answers move materially, the sentence was doing work **despite** anchoring
  pushing the other way. That is a lower bound on the effect, and it is strong.

Say exactly this in the write-up. Do not report a null here as "the instructions
were fine."

## The whole 300, not a sample

**Decision, 2026-09-07: both annotators re-run all 300 controls.** It is a
bigger ask, and it is the better design.

A sample has to be balanced, because a sheet of only the controls someone marked
`NONE` says "you said `NONE` too often" as plainly as a sentence would — the
request itself would carry the nudge the re-run exists to remove. Balancing
solves that but leaves a residue: the composition is a choice, and a choice is
something to defend.

**The complete set carries no composition signal at all.** There is nothing to
infer from "all of them". It also gives a like-for-like comparison on the same
denominator as round 1, which makes the analysis a paired one — every control
has both annotators' round-1 and round-2 answers — rather than a comparison
across differently-drawn populations.

`scripts/build_rerun_sample.py` remains for a future round where a full re-run
is too large an ask. It is not used here. If you do use it: it writes
`sample_composition.json` beside the CSVs, recording how many rows came from
each side. **Do NOT send that file** — it states the hypothesis in one line,
and a balanced sample's whole value is that the annotator cannot infer it. Keep
it as the record of what the sample was.

## Blinding rules

1. **Do not tell them why.** Not the hypothesis, not that the wording changed,
   not that we are looking at `NONE` rates. Part 2 says only that the
   instructions were revised.
2. **Do not show them their previous answers.** The sheet arrives blank — reuse
   the original packet, `packets/phase2c-nist_800_53/`, which is already blank.
3. **Same pseudonyms, separate inbox.** `vol-01`, `vol-02`, returned to
   `~/tract-inbox/phase2c-r2/` so round 1 stays intact.

## Generating and importing

Send the original packet again — it is already blank:

```bash
ls packets/phase2c-nist_800_53/   # ai_hubs.csv  controls.csv  annotate.csv
```

Import as `vol-01-r2`, `vol-02-r2` so the first round stays intact and the two
are comparable:

```bash
python -m scripts.import_bridge_links ~/tract-inbox/phase2c-r2/vol-01.csv \
  --framework-id nist_800_53 --annotator-id vol-01-r2 \
  --created-at <iso-timestamp>
```

Keep round-2 corpora in a **separate directory** from round 1. Gate 1 across a
mixed directory would count one person twice and inflate Q4.

---

# Part 2 — for the annotator

**Hand this part over as-is, with the sample.**

## Run this again, but this time

Run this again, but this time with one instruction changed.

Same 300 controls, same three files. We are asking for the whole set rather than
a subset, because a subset would have to be chosen and any choice we made would
tell you something about what we are looking at. Asking for all of them tells
you nothing, which is the point.

We have revised the guidance below. The task is the same: read a control, decide
which AI-security hub it belongs to, or record that none does. What has changed
is that some earlier wording told you what to expect before you had read
anything, and that was our error. It has been removed.

**Your previous answers are not shown, and we are not asking you to reproduce
them or to change them.** If your judgement on a control is the same as before,
record the same answer. If it is different, record the different one. Both are
useful. There is no answer we are hoping for, and there is nothing to correct.

You will recognise some of these controls. That is expected and it is fine.

## What you have been given

The same three files as before:

| file | what to do with it |
|---|---|
| `ai_hubs.csv` | the 78 hubs you may choose from |
| `controls.csv` | reference copy |
| `annotate.csv` | **your worksheet** — fill `cre_id`, `confidence`, `rationale` |

Same request as before: please do not read
`github.com/rocklambros/TRACT` or `opencre.org` while you work.

## The decision rules, revised

**1. Read the control's text, not its title.** Unchanged.

**2. Ask what the control is *for*, then find the hub with that purpose.** Not
the hub with matching words. Two controls can share no vocabulary and mean the
same thing; the same word can appear in hubs that have nothing to do with each
other.

**3. `NONE` and a hub are both ordinary answers.** Record whichever your reading
of the control supports. Neither is a failure, neither is a better outcome, and
we are not counting one against the other.

*What changed:* the earlier version told you that most of these controls would
have no matching hub. That was a claim about the answer, made before you had
read anything, and you should not have been given it. Whether it is true is what
your judgement decides, not ours.

**4. One hub per control.** If a control seems to fit many hubs at once, you may
be describing the whole AI region rather than that control — but if two hubs
genuinely fit, pick the one that fits best.

**5. Use the confidence scale honestly.**

| | means |
|---|---|
| `3` | I am confident this control belongs to this hub. |
| `2` | Defensible, but I can see an argument against it. |
| `1` | A guess. I want it recorded, but do not build on it. |

`1` is safe to use and useful. A low-confidence link is kept as data and
excluded from the headline count, which is what you would want.

**6. If you are agonising, record `1`, note the doubt, and move on.** Do not go
back and revise earlier rows to match later ones.

## The rationale still matters

One sentence. It is what makes a link reviewable by someone who was not there,
and on a `NONE` it is what tells us whether the region genuinely has no home for
that control or you simply could not find one.

Mechanical, because the import checks it: don't begin a rationale with `=`, `+`,
`-` or `@`, and keep it under 2,000 characters.

## Pace

There is no rate you are expected to hit; nothing about this task has been
timed. This is the full 300 again, which is a real ask on top of what you have
already given — work in blocks, stop before you are tired, and tell us if it is
too much rather than rushing it. A partial return is more useful than a hurried
complete one.

## What happens to your work

As before: a public Tier-2 corpus under your chosen pseudonym, credit on the
OpenCRE proposal, your rationales published alongside the links, and the right
to withdraw before publication.

## Questions

Ask. If anything in the revised rules reads as pointing you toward a particular
answer, tell us — that is the exact problem this revision is trying to fix, and
we would rather hear it now than find it in the results.
