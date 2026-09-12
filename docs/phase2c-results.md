# Phase 2C — results

Two annotators, 300 NIST SP 800-53 controls, two rounds. **Both rounds pass
Gate 1.** The headline finding is not the pass; it is that the round-1 `NONE`
rate was not a stable property of the task, and the instruction wording moved it.

---

## 1. Gate 1, both rounds

| | round 1 | round 2 |
|---|---|---|
| links imported | 86 | 192 |
| counting (confidence ≥ 2) | 80 | 173 |
| **orphans** | **78 → 50** (28 de-orphaned) | **78 → 47** (31 de-orphaned) |
| Q1 distinct controls (≥ 40) | 54 | 118 |
| Q2 max hubs per control (≤ 6) | 1 | 1 |
| Q3 above the floor | 80 of 86 | 173 of 192 |
| Q4 double-annotated (≥ 15%) | 48.2% | 46.6% |
| **Cohen's κ, link decision** | **0.597** | **0.508** |
| **verdict** | **PASS** | **PASS** |

Reproduce:

```bash
python -m scripts.analysis.gate1_report data/training/bridge
python -m scripts.analysis.gate1_report data/training/bridge-r2
```

---

## 2. The round-1 `NONE` rate was not stable

Round 1's handbook told annotators, before they had read a single control, that
*"most NIST 800-53 controls are about traditional IT security and have no
AI-specific hub."* That is the experimenter stating the answer distribution.
Round 2 removed it and re-ran the same 300 controls with the same two people.

| | round 1 linked | round 2 linked |
|---|---|---|
| vol-01 | 40 (13.3%) | **122 (40.7%)** |
| vol-02 | 46 (15.3%) | **70 (23.3%)** |

Paired, per control (exact McNemar on the link/no-link decision):

```
vol-01   NONE -> link  82      link -> NONE   0
vol-02   NONE -> link  24      link -> NONE   0
pooled   NONE -> link 106      link -> NONE   0     p < 0.0001
```

**Not one control moved the other way, and not one hub choice changed.**

### What this establishes, and what it does not

**It establishes that round 1's 87% / 85% `NONE` rate cannot be cited as
evidence that the two domains are disconnected.** The same people, on the same
controls, produced 59% and 77% under different framing. Whatever the true rate
is, round 1 did not measure it.

**It does not establish the size of the instruction effect**, for three reasons,
all of which point the same way — toward the measured movement being an
overstatement of what the sentence alone did.

1. **Round 2 was not independent re-judgement.** vol-02 returned **43 of 46
   byte-identical rationales** and 45 of 46 identical confidences: they worked
   from their round-1 sheet, which they of course still had. vol-01 rewrote
   every rationale (0 of 40 identical) yet chose the same hub 40 of 40 times —
   either genuine re-judgement with high self-consistency, or reconstruction
   from memory. The two behaved differently and only one is clearly a re-run.

2. **A demand characteristic was introduced by the correction itself.** Round
   2's rule 3 said the earlier sentence "was our error" and "you should not have
   been given it". A reasonable annotator reads that as *we think you said
   `NONE` too often*. That cannot be separated from the effect of removing the
   sentence, and it was avoidable — the revision could have been silent.

3. **Zero reversals is not what independent re-judgement looks like.** Genuine
   re-judgement produces noise in both directions. A strictly one-directional
   result is the signature of addition to an existing answer set.

---

## 3. The confident signal was already found in round 1

The most informative number in the comparison. Of the **106 links added in round
2**:

| confidence | vol-01 new | vol-02 new |
|---|---|---|
| 3 — confident | **0** | **0** |
| 2 — defensible | 76 | 15 |
| 1 — a guess | 6 | 9 |

**Not a single added link is confidence 3.** Every confident mapping either
annotator would make was already in round 1. What the instruction moved is the
marginal, hedged band.

That reading is corroborated by agreement falling as links rose: **κ 0.597 →
0.508**, with controls where exactly one annotator linked going from 28 to 63.
More links, less agreement. The added band is genuinely ambiguous territory, not
signal that round 1 suppressed.

---

## 4. What should be reported from this round

**Report:** 31 AI hubs given traditional supervision by two independent
annotators; Gate 1 passed on all five criteria in both rounds; the project's
first human–human agreement measurement, **κ ≈ 0.51–0.60** on the link decision,
which is *moderate*.

**Do not report:** the round-1 `NONE` rate as a property of the corpus, or the
round-2 link rate as one either. Two framings produced two answers and neither
is established as the right one.

**Do not report** κ without saying which denominator. On this data the
defensible figures range from 0.42 to 0.89 depending on whether controls only
one annotator linked are counted, and whether `NONE`–`NONE` agreements are.
`gate1_report` prints all of them for this reason.

---

## 5. An operational incident, recorded

Round 1's `~/tract-inbox/phase2c/vol-02.csv` **was overwritten by the round-2
file** — identical sha256, later mtime. The baseline survived only because it
had already been imported, and `data/training/bridge/vol-02.jsonl` plus its
sidecar reconstruct the round-1 decision exactly (46 links + 254 `NONE` = 300
reviewed).

`scripts/analysis/rerun_delta.py --round1-corpus` now reads the baseline from
the imported corpus rather than a filled CSV, because the corpus is the durable
record and an inbox file is transient. The first run of the comparison, before
this, silently reported vol-02 as "70 → 70, no movement" — a null produced by
reading one file against itself.

---

## 6. If there is a round 3

The instruction effect is real in direction and unmeasured in size. The design
that would measure it is a **split arm on a framework nobody has annotated
yet**: half the annotators get the neutral wording, half get nothing about
expected distribution at all, and neither has a prior answer to anchor to.

Anything run again on NIST 800-53 with these two people measures memory.
