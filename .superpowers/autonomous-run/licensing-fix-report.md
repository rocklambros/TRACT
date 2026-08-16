# Licensing and reachability fix report

Branch `semantic-rebuild`. Three commits on top of 62afd39.

| commit | subject |
|---|---|
| 92d0584 | record every framework's licence and stop redistributing ETSI's text |
| 7365354 | make the licensed-text gate run in CI and on a fresh clone |
| 6781ace | reach ISO 27001's prose, which the pipeline had never once read |

## CRITICAL 1 — the licensed-text gate no longer skips

`tract/licensing.py` holds the parameters, the normalisation and the lookup.
`scripts/build_licensed_fingerprints.py` rebuilds
`tests/fixtures/licensed_text_fingerprints.json` from the staged sources and
supports `--check` to report drift without writing.
`tests/test_licensed_text_not_tracked.py` runs off the tracked fixture and
never touches `data/raw/`.

Verified by moving both restricted sources aside: 7 passed, 1 skipped, and the
one skip is the redundant end-to-end check against the real source, not the
tree scan. Verified by moving the fixture aside: 1 failed, 4 errors, no skips.

### Fingerprint parameters

| parameter | value | why |
|---|---|---|
| n-gram length | 12 words | measured, see below |
| stride | 1 word, both sides | a quotation at any offset produces a matching window |
| digest | SHA-256, first 32 hex characters | 128 bits, collision probability ~3e-28 over ~7e6 candidate windows against ~1.1e4 stored |
| salt | `tract-licensed-fingerprint-v1`, public, recorded in the file | blocks a generic precomputed table, not a targeted attack |
| entries | 11,472 (ETSI 10,778, ISO 706) | per text unit, never across the join between two |

Measured over the 440 tracked text files against ISO Annex A statements:

| n | hits | assessment |
|---|---|---|
| 8 | 6 | all false positives (CSA AICM's own text, an OpenCRE export CSV, the ISO parser's docstring) |
| 10 | 5 | 4 false positives, led by CSA AICM HRS-10 sharing a 10-word NDA run with ISO A.6.6 |
| 12 | 1 | a real partial quotation, no false positives |
| 14 | 0 | too long, the real quotation walks through |

Twelve is the shortest window at which no independently authored document in
this corpus collides. On the sensitivity side, ISO Annex A statements run a
17-word median, so a 12-word window trips on a quotation of roughly 70% of a
median statement and 74 of the 93 statements contribute at least one window.
The 19 statements shorter than 12 words cannot be fingerprinted at any useful
length. Raising n to make a known overlap pass would be the "gate that cannot
fire" defect, so the tree was fixed instead.

ETSI's bibliography is excluded from its fingerprints. It is a list of other
people's paper titles, and NIST AI 100-2 and the OWASP AI Exchange cite the
same papers, so including it flagged four tracked files for reproducing an
author list. That is not an ETSI quotation. The contents page and the running
page header are excluded for the same reason of not being authored content.

### Three real escapes the gate found on its first run, all fixed

1. `tests/fixtures/iso_27001_sample.md` reproduced twelve consecutive words of
   ISO A.7.10 inside what is otherwise a synthetic "ACME" fixture. Reworded.
   No test asserted on that text.
2. `.superpowers/autonomous-run/source-structures.md` carried a 40-word
   verbatim excerpt of ETSI clause 6.2.3 as a "Verbatim sample". Replaced with
   a structural sketch that carries no ETSI wording and serves the same
   purpose for the parser author.
3. The comment in `tract/config.py` naming ETSI's copyright notice quoted it at
   length. Paraphrased.

### Tests

`test_the_gate_fires_on_a_planted_quotation` extends the real fingerprint set
by one entry, the hash of an invented sentence, writes that sentence into a
file and requires the scanner to find it. Real normalisation, real hashing,
real lookup, no licensed text and no `data/raw/`.
`TestTheFingerprintFileCarriesNoText` asserts the property structurally: the
schema has no free-text field, every fingerprint matches `^[0-9a-f]{32}$`, and
every metadata value is checked against its declared shape.
`test_every_restricted_framework_has_a_gitignore_line` closes the gap where
`git rm --cached` without a `.gitignore` entry lets the next parser run put the
file straight back.

## CRITICAL 2 — the licence model

ETSI joined `RESTRICTED_FRAMEWORK_IDS`. `data/processed/frameworks/etsi.json`
is untracked and gitignored, the merged corpus dropped from 30 frameworks /
4,168 controls to 29 / 4,141, and the gitignored overlay carries 31 / 4,261
with both restricted frameworks. The merge step was already driven by the set
rather than by a hardcoded id, verified by a new test that stages every member
instead of one.

`Source` gained a required `license` field with no default, so omitting it is a
`TypeError`. The field reaches the committed manifest. `FRAMEWORK_LICENSES` in
`tract/config.py` is the single copy for frameworks arriving by other routes,
`NOTICE` is generated from it, and tests fail when the two disagree, when a
framework artifact has no entry, or when the manifest loses one.

`NOTICE` added at the repository root, referenced from `README.md` and from a
clearly delimited scope note appended after the CC0 legal code in `LICENSE`.
The CC0 text itself is unmodified.

### Per-source licence table

| framework_id | licence | evidence |
|---|---|---|
| aiuc_1 | UNDETERMINED | staged JSON states no terms |
| asvs | CC-BY-SA-4.0 | archive `LICENSE.md` |
| biml | CC-BY-SA-3.0 AND CC-BY-SA-4.0 | `ara.pdf` p1 says 3.0, `BIML-LLM24.pdf` says 4.0 |
| capec | UNDETERMINED | XML carries no notice |
| cosai | CC-BY-4.0 | staged `SOURCE_MANIFEST.md` |
| csa_aicm | Proprietary, CSA, no redistribution | xlsx notice |
| csa_ccm | Proprietary, CSA, no redistribution | xlsx notice |
| cwe | UNDETERMINED | XML carries no notice |
| dsomm | GPL-3.0-only | archive `LICENSE` |
| enisa | ENISA 2021, reproduction with acknowledgement | PDF copyright notice |
| etsi | ETSI 2021, all rights reserved | PDF page 2 |
| eu_ai_act | European Union, Commission Decision 2011/833/EU | staged `MANIFEST.json` |
| eu_gpai_cop | European Commission, published for public use | staged front matter |
| iso_27001 | Proprietary, ISO/IEC 2022, single-user store licence | PDF cover page |
| mitre_atlas | Apache-2.0 | `atlas-data.json` metadata |
| nist_800_53 | UNDETERMINED | OSCAL catalog states no terms |
| nist_800_63 | US Government work, not subject to US copyright | HTML front matter |
| nist_ai_100_2 | UNDETERMINED | PDF points at an external policy page only |
| nist_ai_600_1 | UNDETERMINED | same |
| nist_ai_rmf | UNDETERMINED | staged markdown states no terms |
| nist_ssdf | US Government work, not subject to US copyright | PDF front matter |
| owasp_agentic_top10 | CC-BY-SA-4.0 | staged markdown |
| owasp_ai_exchange | UNDETERMINED | markdown names licences of third-party tools, not its own |
| owasp_cheat_sheets | CC-BY-SA-4.0 | archive `LICENSE.md`, SPDX line |
| owasp_dsgai | CC-BY-SA-4.0 | staged PDF text and `MANIFEST.json` |
| owasp_llm_top10 | CC-BY-SA-4.0 | staged markdown |
| owasp_ml_top10 | CC-BY-SA-4.0 | archive `LICENSE` |
| owasp_proactive_controls | CC-BY-SA-4.0 | archive `LICENSE` |
| owasp_top10_2021 | CC-BY-SA-4.0 | archive `LICENSE` |
| samm | CC-BY-SA-4.0 | archive `license.txt` |
| wstg | CC-BY-SA-4.0 | archive `LICENSE` |

Seven UNDETERMINED: aiuc_1, capec, cwe, nist_800_53, nist_ai_100_2,
nist_ai_600_1, nist_ai_rmf, plus owasp_ai_exchange. Each publisher states
terms somewhere outside the artifact this project downloaded. Recording them
means reading the publisher's own page, not inferring from a sibling artifact.

### Two facts the owner should see, neither acted on here

CSA CCM's v4.1.0 notice reads "the Cloud Controls Matrix v4.1.0 may not be
redistributed". The owner's 2026-08-16 ruling stands and CCM is not restricted.
The notice is recorded verbatim in `FRAMEWORK_LICENSES` and `NOTICE` so the
tension is visible rather than buried.

CSA AICM carries the identical notice and `data/processed/frameworks/csa_aicm.json`
is tracked today with 243 controls of full prose. It was tracked before this
branch and no ruling covers it. Posture unchanged, flagged for a decision.

## CRITICAL 3 — ISO 27001 reachability

`"iso 27001": "ISO/IEC 27001:2022 Annex A"` added to `FRAMEWORK_NAME_ALIASES`.

Measured after the fix: **92 of 94**, not 94 of 94. The two misses are A.7.8
and A.7.9, whose descriptions are 48 and 35 characters against titles of 29 and
31, below the `PROSE_MIN_EXTRA_CHARS` threshold of title + 20. `ProseIndex`
excludes them on purpose, the same way it excludes every OpenCRE description
that restates its title. That is the prose rule working, not a second defect.

The alias alone would have changed nothing on the run path. Every
`ProseIndex.load()` call site used the default path, the tracked
`all_controls.json`, which excludes every restricted framework by construction,
so ISO's prose is only ever in the gitignored overlay.
`parsers/merge_all_controls.py` has documented "prefer the licensed overlay
when it is present" since the overlay was introduced and nothing implemented
it. `merged_corpus_path()` in `tract/text_selection.py` now implements it and
`ProseIndex.load` resolves its default through it. This is the one change here
that was not in the brief, made because reporting 92/94 while the run path
still saw 0/94 would have been a number without an artifact behind it.

Full reachability, licensed overlay, after the fix:

```
capec 1799/1799   cwe 612/613       owasp_cheat_sheets 391/391
nist_800_53 298/300   asvs 277/277  mitre_atlas 65/65
owasp_ai_exchange 64/64   nist_ai_100_2 45/45
owasp_llm_top10 13/13   owasp_ml_top10 10/10   iso_27001 92/94
biml, csa_ccm, dsomm, enisa, etsi, nist_800_63, nist_ssdf,
owasp_proactive_controls, owasp_top10_2021, samm, wstg all 0, no parser yet
```

`tests/test_prose_reachability.py` has four tests. The naming check reads each
parser's `framework_name` with `ast`, so it needs neither the ML stack nor a
built corpus, and it fails on a clean checkout when an alias is missing. The
end-to-end check requires every parser-backed framework to resolve at least 90%
of its curated links, against a measured worst case of 0.979 and a broken join
of 0.000, and it is scoped to frameworks with a `parse_*.py` so the eleven
awaiting parsers do not make it permanently red. Both go red when the alias is
removed, confirmed by patching it out and re-running.

`tests/test_text_selection.py::test_fallbacks_are_counted_per_framework` moved
to the canonical key. `select_control_text` has always recorded
`canonical_framework(framework)`, so the stats key travelled with the alias.
The old assertion passed only because ISO had no alias.

## Test results

```
21 failed, 1241 passed, 22 skipped, 3 xfailed, 3 collection errors
```

Same 21 failures as the stated baseline, every one a missing dependency in this
environment: `datasets` (9), `defusedxml` (8), `hdbscan` (4), `anthropic` (2),
`peft` (1). The 3 collection errors are the same cause. Passing count rose from
1,177 to 1,241.

`mypy --strict` clean on every touched file. `ruff check` clean on every
touched file. The 26 pre-existing mypy errors in the full CI scope are all in
files this work did not touch and are missing-stub errors in this environment.

## What a reviewer should push back on

The 90% reachability floor is a threshold with one measurement behind it per
framework. If a corpus rebuild moves a framework from 1.000 to 0.85 for a real
reason, the test says "broken join" when the cause is elsewhere.

`merged_corpus_path()` makes a training run's inputs depend on whether a
gitignored file exists locally. That is the overlay's documented intent and the
fold metadata records the corpus sha256, so the difference is auditable, but it
is a real behaviour change on the run path and it was not requested.

Seven UNDETERMINED entries are honest and useless. They will stay UNDETERMINED
until someone records terms from each publisher's page, and nothing here forces
that to happen.
