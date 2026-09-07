# Phase 2C — annotator handbook

**Part 2 of this document is what a volunteer receives. Part 1 is for whoever
coordinates the round.** Hand over Part 2 and the packet; nothing else.

This is a *different task* from the Campaign 3 curation round, and it needs its
own handbook for that reason. `claudedocs/curation-package.md` describes mapping
**AI** controls across **522** hubs with a `hub_reference_sheet.csv`. This round
maps **traditional** controls (NIST 800-53) onto the **78** AI hubs, using a
packet from `scripts/build_bridge_packet.py`. Sending the wrong handbook with
the right packet, or the reverse, wastes the volunteer's time and produces a
sheet that will not import.

---

# Part 1 — for the coordinator

## 1.1 What this round buys

The AI and traditional hub regions are **disjoint**: 78 AI hubs, 380 traditional
hubs, and the intersection is exactly **0**. Nothing in the corpus positions a
traditional control against an AI hub, so the model has never seen that
relationship. This round is the first supervision for it.

Reproduce the premise before recruiting anyone:

```bash
python -m scripts.analysis.orphan_rate
# AI hubs with no traditional supervision: 78 of 78 (100.0%)
```

The binding gates are in `docs/phase2c-preregistration.md`, committed before any
control was read. **Do not tell annotators the numeric targets** — a population
generating data should not be told the quota. §5 of that document is honest
about which parts of the blinding actually hold.

## 1.2 It is a volunteer round

No rates, no invoicing, no attestation-before-payment. What a volunteer receives
in exchange is stated in Part 2 and must be honoured:

- **Named credit** on the published corpus, or a pseudonym if they prefer.
- **Credit on the OpenCRE upstream proposal**, where accepted links are proposed.
- **A right to withdraw** their contribution before publication.

Their `annotator_id` is recorded on every link they produce. Tell them that
before they start, not after.

## 1.3 Who to exclude

Beyond anyone who has worked on TRACT:

- **Anyone who authored or maintains NIST 800-53**, or any of the eight AI
  frameworks in the corpus (MITRE ATLAS, NIST AI 100-2, OWASP AI Exchange,
  OWASP Top 10 for LLM, OWASP Top 10 for ML, ENISA, ETSI, BIML). A framework's
  own author recalls the intended mapping rather than judging it. With an OWASP
  pool this is the expected case, not an edge case — ask directly.

## 1.4 How long to ask for

**These are planning figures. Nothing has been timed.** No per-item rate has
ever been measured on this task, and an earlier handbook presented one as
measured, citing a source that says the opposite. Do not manage anyone against
a number nobody produced.

NIST 800-53 has ~300 controls in the packet. Ask for what a volunteer can give,
tell them the pilot alone is useful, and let them stop.

## 1.5 Generating and sending the packet

```bash
python -m scripts.build_bridge_packet ~/tract-packets/phase2c-nist_800_53 \
  --framework-id nist_800_53
```

**Where to write it.** By default, outside the repository working tree — the
annotator gets the packet, not the repo, and a packet inside the tree is one
`git add -A` from being committed.

The exception is a packet you deliberately intend to distribute, which is
committed under `packets/` so a coordinator on another machine can `git clone`
rather than rebuild. `packets/phase2c-nist_800_53/` is one: NIST 800-53's
licence was checked before it went in. **That is a property of NIST 800-53, not
of packets** — a `csa_aicm`, `csa_ccm`, `etsi`, `iso_27001` or `dsomm` packet
may not be committed, and the fingerprint gate will not catch it because it
fingerprints only the last three. `tests/test_packet_manifest.py` is the guard
that will. See `packets/README.md`.

If you have not deliberately checked the licence, write it outside the tree.

No `--allow-undetermined` is needed: NIST 800-53's licence was adjudicated
2026-09-06 as a US Government work not subject to copyright. If the command
refuses a framework, it is right; do not work around it.

Three files, **all three and nothing else**. Measured from the generated
packet, 2026-09-07:

| file | size | what it is |
|---|---|---|
| `ai_hubs.csv` | 17 KB | all **78** AI hubs — `hub_id, hub_name, hierarchy_path, branch` |
| `controls.csv` | 452 KB | read-only reference of the **300** controls |
| `annotate.csv` | 453 KB | **the sheet they fill** — the same 300 controls with empty answer columns |

A fourth file, `manifest.json`, is written beside them. **Keep it; do not send
it.** It records the framework, the build time, the git SHA of the tree that
built the packet, and a sha256 of each CSV — so a sheet that comes back can be
tied to the exact bytes that went out. Without it, if the hub roster changes
between builds, nothing records which 78 hubs a given annotator actually saw.

The 78 hubs sit in four branches, which is worth knowing when you brief someone:

| hubs | branch |
|---|---|
| 46 | Technical application security controls |
| 24 | Cross-cutting concerns |
| 5 | Development processes for security |
| 3 | Governance processes for security |

Control text runs 290-5,508 characters, median 1,212, and **none is truncated**.
That took a fix: `all_controls.json` caps `description` at 2,000 characters, so
58 of the 300 controls used to arrive cut mid-word - one ended *"Procedures can
be documente"*. The builder now takes the longer of `description` and
`full_text`, recovering about 58 KB of prose. If you ever see a control ending
mid-word, stop and report it; the annotator is being asked to judge a control
they cannot read.

**Never send `results/ceiling_study/hub_reference.md`.** 400 of its hub
descriptions were written by an LLM conditioned on the existing gold links.
Sending it makes every label Tier 3 and the round unusable for either gate.

The packet builder refuses frameworks whose prose may not be redistributed. If
it raises, it is right — do not work around it.

## 1.6 Receiving the work back

```bash
python -m scripts.import_bridge_links filled.csv \
  --output data/training/hub_links_bridge.<annotator_id>.jsonl \
  --framework-id nist_800_53 \
  --annotator-id <annotator_id> \
  --created-at 2026-09-07T12:00:00Z
```

**One file per annotator.** The importer refuses to overwrite, and the gate
reads a directory of them. Writing two annotators to one path used to destroy
the first silently.

Then:

```bash
python -m scripts.analysis.gate1_report data/training/
```

That reports the orphan reduction **and** all four quality conditions, and exits
non-zero on FAIL. Do not use `orphan_rate --bridge` for a verdict — it is the
raw arithmetic, counts every link at any confidence, and will pass a sheet that
violates three conditions.

## 1.7 Two annotators on an overlap

Q4 requires at least 15% of controls annotated by two people, and the agreement
rate reported. **This produces the project's first human–human agreement
number** — no such measurement exists today. Any figure you may have seen quoted
as prior human–human agreement is a misreading of an LLM-judge study.

Set no expectation for what it should be. Report it with its interval.

---

# Part 2 — for the annotator

**Hand this part over as-is, with the packet.**

## Welcome, and what this job is

Security frameworks describe the same ideas in different words. NIST 800-53
says *"The information system enforces approved authorizations for logical
access"*; an AI security framework might have a hub called *"AI access
control"*. A shared taxonomy — OpenCRE — gives each idea a **hub**, and your
job is to say which hub, if any, a control belongs to.

You are working on a specific and narrow question: **which of 78 AI-security
hubs does this traditional security control belong to?** Most of the time the
answer will be *none*, and that is the single most important thing to know
before you start.

## What you have been given

| file | what to do with it |
|---|---|
| `ai_hubs.csv` | **Read this first.** All 78 hubs you may choose from, with their place in the tree. |
| `controls.csv` | Reference copy of the 300 controls. You do not need to edit it. |
| `annotate.csv` | **This is your worksheet.** 300 rows. Fill the last three columns. |

A hub row looks like this:

```
hub_id   hub_name                              branch
010-108  Obscuring confidence in AI output     Technical application security controls
011-087  Testing against membership inference  Development processes for security
012-625  Indirect prompt injection             Cross-cutting concerns
```

The `branch` tells you roughly where a hub lives. Most of the 78 (46 of them)
are technical application controls; 24 are cross-cutting; the remaining 8 are
about development or governance process.

`annotate.csv` has one row per control. The first three columns are the
control — its id, title and full text. The last three are yours:

- **`cre_id`** — the hub id from `ai_hubs.csv`, or `NONE`.
- **`confidence`** — `1`, `2` or `3`. Nothing else.
- **`rationale`** — one sentence on why.

Fill them in place and send the file back. Do not add columns, rename headers,
or reorder rows; the import will refuse the file and it comes back to you.

## Please do not look TRACT up

For the duration of your work, please **do not read**:

- `github.com/rocklambros/TRACT` — including its issues, docs, `data/` and
  `results/` directories
- `opencre.org`

That repository is public and contains existing mappings and generated hub
descriptions. If you read them, your answers stop being independent, which is
the entire value of what you are doing. We are asking rather than preventing,
and we would rather you tell us you looked than not.

## The decision rules

**1. Read the control's text, not its title.** Titles are compressed to the
point of ambiguity. The text is in your sheet for this reason.

**2. Ask what the control is *for*, then find the hub with that purpose.** Not
the hub with matching words. "Boundary protection" and "network segmentation"
share no words and may be the same idea; "model" appears in hubs that have
nothing to do with each other.

A worked example from your own sheet. `AC-3 Access Enforcement` reads *"Enforce
approved authorizations for logical access to information and system resources
in accordance with applicable access control policies."* Ask what it is for:
making sure only permitted actors reach a resource. Now look for an AI hub with
that purpose. If one is genuinely about enforcing authorization on an AI
system's resources, that is your answer. If the nearest candidate is a hub about
something like obscuring model confidence, that shares no purpose with AC-3 and
the answer is `NONE`. Do not map it because both happen to mention access.

**3. `NONE` is a real, correct and expected answer.** Most NIST 800-53 controls
are about traditional IT security and have no AI-specific hub. Writing `NONE`
is not a failure and not a gap in your work — it is a finding, it is recorded,
and it counts. **Do not stretch to find a hub.** A forced mapping is worse than
no mapping, because someone downstream will trust it.

**4. One hub per control, and at most a few controls per hub.** If a control
seems to belong to six or seven hubs, you are probably describing the whole AI
region rather than that control. Pick the best one, or write `NONE`.

**5. Use the confidence scale honestly.**

| | means |
|---|---|
| `3` | I am confident this control belongs to this hub. |
| `2` | Defensible, but I can see an argument against it. |
| `1` | A guess. I want it recorded, but do not build on it. |

`1` is useful and safe to use. Low-confidence links are kept as data and
excluded from the headline count, which is exactly what you would want.

**6. If you are agonising, record `1`, note the doubt in the rationale, and move
on.** Do not go back and revise earlier rows to be consistent with later ones —
that turns independent judgements into one drifting judgement.

## The rationale matters

One sentence is enough. It is what makes a disputed link reviewable later, and
it is the difference between a corpus someone can check and a list of ids.

Write what a reader would need to disagree with you: *"Both are about enforcing
approved authorizations at a boundary"* is useful. *"Seems related"* is not.

A few mechanical notes, because the import checks them: don't begin a rationale
with `=`, `+`, `-` or `@` (spreadsheets treat those as formulas), and keep it
under 2,000 characters.

## Pace

Work in blocks with real breaks and stop well before you feel tired — quality
falls off before you notice. If you find yourself reaching for a hub because it
is familiar rather than because it fits, stop for the day.

There is no rate you are expected to hit. Nobody has measured one.

## What happens to your work

- Your answers become a public dataset of Tier-2 (independently human-authored)
  links, with your **name or chosen pseudonym** recorded on each one.
- Accepted links are proposed upstream to **OpenCRE**, with credit.
- You may **withdraw your contribution** at any point before publication. Say so
  and it is removed.
- Your rationales are published alongside the links.

If any of that is not what you expected, say so before you start rather than
after.

## Questions

Ask. A question about the task is worth more than a guess, and if the packet or
these instructions are unclear that is our error and we want to know.
