# R23 — framework-identity tokens, and the asymmetry that put "owasp" in the stop word list

## The defect

`data/processed/stopwords.json` is derived by document frequency. "owasp" cleared
the 0.05 threshold because OWASP spans ten of the corpus's 29 frameworks; "cwe"
misses it despite a larger raw count, because CWE's occurrences concentrate
inside one framework's own documents. So the stop word arm scrubbed OWASP
anchors of their publisher token and left thirteen other frameworks theirs.

That is not a stop word question. Under leave-one-framework-out it is a
per-fold inconsistency: the fold holding out OWASP saw anchors with the giveaway
removed while every other fold did not, and no metric in the repository reports
it. The project's core constraint is `g(control_text) -> CRE_position`, a
semantic mapping, and Task 13 already removed 656 characters of ETSI document
identifier from anchors on the same reasoning.

## The criterion

A token is a framework-identity token when all three hold.

1. **It is a component of a framework's machine id** (`csa_ccm`, `nist_ssdf`),
   not merely a word in the human title.
2. **It is written in capitals in most of its occurrences**, measured over
   markup-stripped control text.
3. **It appears nowhere in hub data** — hub names, hierarchy paths, or hub
   descriptions.

### What was tried and rejected

**Every token in a framework id or name, occurring in no hub.** 42 tokens. Too
broad, as the ruling states: admits `matrix` (from "Cloud Controls Matrix"),
`regulation`, `profile`, `landscape`, `practice`, `cloud`, `framework`,
`mitigations`, `generative`, `map`, `standard`. Stripping "regulation" from
every control because one framework is named a regulation removes real signal.

**Machine id only, occurring in no hub.** Better — drops all 17 title-only
words — and still admits nine ordinary words that happen to sit in an id:
`act`, `agentic`, `cheat`, `cop`, `exchange`, `proactive`, `sheets`, `top`,
`controls`.

**Concentration: occurrences overwhelmingly inside the frameworks that bear the
name.** Rejected on measurement. It produces a *fresh* asymmetry rather than
closing the existing one. Measured fraction of document hits inside the bearing
frameworks:

| admitted | | rejected | |
|---|---|---|---|
| biml | 1.000 | ccm | 0.000 |
| capec | 0.926 | enisa | 0.000 |
| csa | 0.789 | wstg | 0.000 |
| cwe | 0.671 | samm | 0.000 |
| owasp | 0.538 | ssdf | 0.000 |
| | | mitre | 0.128 |
| | | nist | 0.189 |
| | | asvs | 0.216 |

`ccm` scores 0.000 because CSA AICM cites the CCM and CSA CCM does not cite
itself. Any threshold splits the acronyms into two arbitrary groups, which is
the defect wearing a different name.

**Acronym shape by casing, measured on markup-stripped text.** Chosen. The
separation is clean and the threshold sits in an empty band:

- lowest admitted: `mitre` 0.690, then `owasp` 0.918, `atlas` 0.947, everything
  else ≥ 0.996
- highest rejected non-hub id token: `top` 0.014, then `agentic`, `act`,
  `cheat`, `cop`, `exchange`, `proactive`, `sheets` all at 0.000

The majority rule (0.5) sits in the middle of the gap rather than on an edge.

Markup stripping is load-bearing. Counting the raw corpus field puts `mitre` at
0.152 and `owasp` at 0.737, because `cwe.mitre.org` and `owaspai.org` contribute
lowercase spellings the encoder never reads — `strip_markup` removes URLs before
the anchor is built. Without it, MITRE would have kept its acronym while OWASP
lost its, which is the original defect reproduced by a different route.

### Why each gate stays

Gate 1 bounds the candidate universe. Without it, gates 2 and 3 admit **1,137**
capitalised non-hub tokens, `jwt`, `siem`, `cve` and `fips` among them. A
control that has lost the token "JWT" has lost its meaning.

Gate 2 is the stable separator. Gate 3 catches `act`, `cheat` and `top` today
only because 400 generated hub descriptions happen to use those words, and this
repository carries a `hub_descriptions_reviewed.pre_regen.json` proving
descriptions get regenerated. Measured with hub vocabulary narrowed to names and
paths, gate 2 is the only thing rejecting six of them.

Gate 3 is the protection and the only gate keeping `nist`, `ai`, `llm` and `ml`
out. `nist` appears in **no** hub name and **no** hierarchy path — only in two
hub descriptions — which is why `load_hub_vocabulary` reads them and is
deliberately wider than the vocabulary the stop word builder protects.

One honest correction, forced by mutation M1: on this corpus, widening gate 1 to
the human titles changes the output not at all, because every title-only word
fails gate 2 anyway. Gate 1's contribution is the universe, not the survivors,
and the candidate universe is now pinned by its own test rather than left to
that accident.

## The derived set — 18 tokens

```
aicm  asvs  atlas  biml  capec  ccm  csa  cwe  dsgai
enisa eu    gpai   mitre owasp  rmf  samm ssdf wstg
```

Rejected as hub vocabulary (11): `act ai cheat controls exchange llm ml nist
proactive sheets top`. Rejected as not capitalised (1): `agentic`. Rejected as
absent from the corpus (4): `aiuc cop cosai dsomm`.

The absent four are the set's one asymmetry, and it is measurable rather than a
judgement: each occurs zero times in control text, so excluding it removes
nothing. A test asserts the count is still zero.

## The toggle

`TrainingConfig.use_framework_identity_filter`, **defaulting to `False`** —
which preserves current behaviour exactly. Nothing strips these until a
measurement says it helps. Carried as its own ablation arm rather than folded
into `use_stopword_filter`, because the two answer different questions:
boilerplate removal is about information density, this is about a learnable
shortcut. Arm label `fwid` (`prose-fwid`, `prose-stopwords-fwid`), CLI flag
`--framework-identity` on both `run_fold.py` and `runpod_parallel.py`, and
`use_framework_identity_filter` added to `ARM_DEFINING_KEYS` so two arms cannot
average into one number.

## What moved and what did not

**Moved.** `data/processed/stopwords.json`: 81 words to 80. The single change is
the removal of `owasp`; nothing was added and `n_documents` is unchanged at
5,149. This affects the `lofo_prose_stopwords` arm only. The default path
(`use_stopword_filter=False`) is byte-for-byte unaffected, the published model
is unaffected, and `merged_corpus_sha256` does not read this file.

**Added.** `data/processed/framework_identity_tokens.json`, and it is inert
until the new toggle is on.

**Not moved.** `all_controls.json`, `cre_hierarchy.json`, the curated links, the
corpus digest, every committed metric.

The identity artifact is derived from the **tracked** corpus, not
`merged_corpus_path()`. Two reasons: the committed set must reproduce from a
fresh clone, which is what the Jetson has, and a set derived from the licensed
overlay would carry `etsi` and per-token counts measured over restricted prose
into a CC0 repository. The resulting gap is closed at run time rather than left
to trust — `run_single_fold` asserts symmetry against the corpus the fold
actually read, and on a checkout holding the overlay the identity arm refuses to
start and names `etsi`. Verified.

## The ratchet

`assert_identity_symmetry` refuses a filter set that strips one framework's name
and keeps another's. The eligible set is derived independently of the three
gates: `self_acronym` takes the first token of the human name when it is
capitalised, else the first component of the machine id. Eleven frameworks name
themselves in the corpus; `nist` is the one guarded by hub vocabulary; `aiuc`,
`cosai` and `dsomm` never appear.

It fires in three places: `scripts/build_stopwords.py` checks all four arm
combinations before writing either artifact, `run_single_fold` checks the arm it
is about to run, and the test suite checks the committed pair.

It fails on the world before this change. Given the pre-fix stop word list, it
raises and names both sides.

## Tests

43 new in `tests/test_framework_identity.py`, plus provenance and arm-label
coverage extended in `tests/test_data_quality.py` and
`tests/test_fold_aggregation.py`. Suite: **2,351 passed**, 9 environmental
failures (all model-loading, `datasets` absent locally), unchanged from the
2,216-passed baseline apart from added tests. `mypy --strict` clean on the CI
target set, `ruff check` clean.

## Mutation testing

21 mutants, run with `PYTHONDONTWRITEBYTECODE=1` against a pristine snapshot
restored before and after each.

| # | mutation | result |
|---|---|---|
| M1 | gate 1 widened to the human title | **SURVIVED**, then killed |
| M2 | gate 1 removed, every corpus token a candidate | killed |
| M3 | casing threshold dropped to 0.001 | killed |
| M4 | casing gate removed | killed |
| M5 | hub protection removed | killed |
| M6 | hub vocabulary stops reading descriptions | killed |
| M7 | casing measured on the raw field, URLs included | killed |
| M8 | symmetry check is a no-op | killed |
| M9 | symmetry drops the protection direction | killed |
| M10 | symmetry accepts a partial strip | killed |
| M11 | `self_acronym` takes the first capital anywhere | killed |
| M12 | `filter_set` returns `frozenset()` instead of `None` | killed |
| M13 | identity toggle defaults to on | killed |
| M14 | identity digest keyed on the stop word flag | killed |
| M15 | build script stops protecting identity tokens | killed |
| M16 | frequency list regenerated without protection | killed by the build guard |
| M17 | empty derivation returned instead of raised | killed |
| M18 | a hub word added to the committed set | killed |
| M19 | one acronym dropped from the committed set | killed |
| M20 | `owasp` put back into the committed stop word list | killed |
| M21 | `MIN_TOKEN_LENGTH` raised, dropping `eu` | killed |

### The survivor and the defect it exposed

**M1** widened gate 1 to include human-title tokens. Every test passed. The
derived set is unchanged, because on this corpus every title-only word fails
gate 2 anyway — so no test that looks only at the output could see the widening.
The gate would have become silently wrong, and the first framework named
something like "OWASP TOP" would have paid for it.

Fixed by pinning the candidate universe rather than the survivors:
`test_the_candidate_universe_is_the_machine_ids_and_nothing_else` reconstructs
what the derivation considered from the four partitioning buckets in
`IdentityDerivation` and asserts it equals the union of the machine-id
components — 34 tokens, against 51 for the titles and five figures for the whole
vocabulary. M1 and M2 both die on it. The module docstring was corrected in the
same pass: it claimed each gate was load-bearing for the output, and gate 1 is
load-bearing for the universe.

## Concerns

1. **The stop word arm's committed metrics are now stale.**
   `results/phase1b/lofo_prose_stopwords/` was produced with `owasp` in the
   list. That arm's numbers describe a filter set that no longer exists. The
   delta is one token out of 81 and the arm was already reported as not
   separating statistically, so this is a re-run-when-convenient rather than a
   correction, but it should not be quoted as current.

2. **The identity arm cannot run on a licensed checkout without a local
   rebuild.** By design and it fails loudly with the remedy in the message, but
   it is a manual step and the rebuilt artifact must not be committed.

3. **`assert_identity_symmetry` recomputes surface counts on every call**, about
   1.2 seconds over the tracked corpus. Called once per fold, so roughly 36
   seconds across a 30-fold campaign. Left unoptimised: the cost is noise beside
   a GPU fold and a cache would be another thing to invalidate.

4. **Nothing here has been measured for accuracy.** The set is derived and the
   toggle exists; whether stripping publisher acronyms helps, hurts or does
   nothing is an open ablation. The default is off precisely because that
   question is unanswered.

5. **`eu` is in the set.** It is a jurisdiction marker rather than a publisher
   acronym, and it clears all three gates cleanly (0.998 capitals, in two
   framework ids, absent from hub data). Defensible under the shortcut argument
   — a fold holding out the EU AI Act can be answered from the token — but it is
   the one member of the 18 where a reviewer could reasonably disagree.
