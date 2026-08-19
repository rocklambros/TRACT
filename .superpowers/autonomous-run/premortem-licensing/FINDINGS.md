# Licensing premortem — findings and the orchestrator's plan

Run before implementing owner ruling R8 option (b), per directive 2 (premortem major decisions
BEFORE implementing them). One agent, adversarial, read-only. It found things more important than
the question it was asked, which is the point.

## Verdict on my proposed implementation

I proposed four limbs. Three survive, one does not.

| limb | verdict |
|---|---|
| 1. Strengthen the LICENSE scope note | survives, but insufficient alone. See F9. |
| 2. Dissolve CONDITIONAL for the six copyleft members | **survives for FIVE, fails for DSOMM.** See F10. |
| 3. Track csa_ccm on the owner's ruling | **narrow it.** See F7. |
| 4. Keep RESTRICTED = {etsi, iso_27001} | survives. No attack found. |

## Verified by me against source, not accepted on report

| # | finding | verdict |
|---|---|---|
| F1 | Four contradictory licence declarations across the published artifacts | **CONFIRMED.** `dataset/card.py:17` = `cc-by-sa-4.0`; `publish/model_card.py:110` = `mit`; `dataset/bundle.py:223` = `CC-BY-SA-4.0`; `pyproject.toml` has NO license key. NOTICE is not shipped in the bundle. |
| F3 | The CONDITIONAL tier already leaks through the merge | **CONFIRMED.** `parsers/merge_all_controls.py:139,142` filter on `RESTRICTED_FRAMEWORK_IDS` only. The moment a parser writes DSOMM or WSTG prose, `git add` puts it in the tracked `all_controls.json` and no test fires. This is verbatim the defect that file's own docstring says was fixed for RESTRICTED. |
| F4 | `csa_aicm` is a larger exposure than the whole copyleft debate | **CONFIRMED.** 243 controls, description min/median/max 39/176/485, TRACKED, in neither tier, licence reads "all rights reserved... no redistribution". Structural cause: `_copyleft()` selects on the substrings `GPL` and `CC-BY-SA`, so a source that reserves rights outright matches neither and produces no tier membership at all. The model detects share-alike and is blind to the stricter posture. |
| F5 | `tract export-canonical` writes full control text with zero licence filtering | **CONFIRMED.** `grep -c` for RESTRICTED/licens/OVERLAY in `tract/export/canonical.py` and `canonical_schema.py` returns **0** for both, and `canonical_export/` is NOT gitignored. |

Findings I did not independently re-derive but which cite clause text directly: F2 (GPL-3.0 §4
requires shipping a copy of the licence; the repo ships none, and TRACT's sanitisation,
2000-char truncation, stopword removal and elision marker are unstated modifications under
§5(a) and CC BY-SA §3(a)(1)(B)), F9 (a scope note appended to LICENSE either leaves GitHub
reporting plain `CC0-1.0`, in which case nobody parses the note, or drops the detector below
threshold, in which case the repo loses its licence identity; testable in ten minutes), F10.

## The DSOMM argument, which is why limb 2 only half survives

GPL-3.0 §5's aggregation carve-out says inclusion in an aggregate does not apply the License to
**the other parts** of the aggregate. It does not say the covered work stops being GPL. So even on
the most favourable reading, the DSOMM portion is still conveyed under GPL-3.0 and still carries
§4's obligations.

And the carve-out requires "a compilation ... in or on a volume of a storage or distribution
medium." `data/processed/all_controls.json` is one JSON document interleaving DSOMM's activity
descriptions with 28 other frameworks. "Separate and independent" is hard to argue for a merged
corpus that exists specifically so a trainer consumes the parts jointly.

CC BY-SA's share-alike attaches to the Adapted Material. GPL's attaches to the whole work. That is
why the parity argument that carries the five CC BY-SA members does not reach DSOMM.

**Decision: DSOMM stays in the overlay.** One framework, prose never published, `git push` is
one-way, zero anchor cost because training reads the overlay, and §5(c) is the only clause on the
table that reaches TRACT's own code. R8 option (b) does not disturb that reasoning.

## Plan, in dependency order

**L1 — licensing infrastructure. Must land BEFORE any parser writes copyleft prose into a tracked file.**
1. `LICENSES/` directory carrying the actual texts: GPL-3.0-only, CC-BY-SA-4.0, CC-BY-SA-3.0, CC-BY-4.0, Apache-2.0, CC0-1.0. This is the single cheapest item that converts the position from asserted to discharged, and it is what SPDX scanners, Scancode, FOSSA and ClearlyDefined actually parse.
2. NOTICE gains a modification statement naming the four transforms, the basis of the csa_ccm ruling rather than only its existence, and a csa_aicm entry.
3. Fix all four declarations. HuggingFace supports `license: other` with `license_name` and `license_link`. Ship NOTICE inside both published bundles.
4. `pyproject.toml` gains a `license` field.

**L2 — leak fixes. Critical, independent of every judgment call above.**
5. `merge_all_controls.py` filters on `OVERLAY_FRAMEWORK_IDS`, and `test_merged_corpus_carries_no_restricted_prose` widens to match.
6. `canonical_export/` gitignored, and `tract/export/canonical.py` filters by licence tier.

**L3 — the tier itself.**
7. Dissolve CONDITIONAL for the five CC BY-SA members. Keep DSOMM. Order matters: remove the gitignore lines, `git add` the artifacts, THEN edit the constants, THEN the tests, or a fresh CI checkout reports the five as stale registry entries pointing at the wrong problem.
8. csa_ccm narrows to titles until the ruling's basis is recorded. Strictly narrower than the owner's ruling and reverses nothing.
9. The copyleft gate is not deleted. Its demand changes from "copyleft implies overlay" to "copyleft implies a NOTICE row, a shipped licence text, a per-artifact licence field, and a modification statement." Deleting it would remove the only automated check on the next copyleft ingest, which is the silence the test file was written to end.

**L4 — measure and escalate.**
10. Settle F9 by measurement: push the modified LICENSE to a scratch repo and read the GitHub API's `.license.spdx_id`.
11. Escalate csa_aicm to the owner as its own question. It is bigger than the one I asked.

## Invariant to the whole decision, recorded so nobody thinks (b) resolved it
GPL and CC BY-SA prose train the model under (a), under (b) and under the tier alike, because
training reads the overlay in every case. The weights ship as `license: mit`. Whether model weights
are a derivative of their training data is unsettled and no cheap fix exists.

---

## F9 settled by measurement, not by a scratch repo

The premortem said a scope note appended to LICENSE lands in one of two states, and that I could
not know which without testing. I settled it read-only:

```
git show main:LICENSE | tail   ->  ends at CC0 clause (d). NO scope note.
gh api repos/rocklambros/TRACT ->  {"license": "CC0-1.0", "name": "Creative Commons Zero v1.0 Universal"}
```

So the scope note exists ONLY on `semantic-rebuild` and has never been through GitHub's detector.
`main` carries pure canonical CC0 and detects cleanly today. That is positive evidence for exactly
one of the two branches and none for the other.

**Ruling R9: `LICENSE` returns to pure canonical CC0. The scope moves to where humans and tools
both actually look.**

- `LICENSE` becomes byte-identical to canonical CC0-1.0. Measured to detect correctly.
- `NOTICE` carries the scope in full. It already says it better than the appended note did: "The
  CC0 dedication does not, and cannot, cover third-party framework content ... those terms continue
  to apply to that framework's text wherever it appears in this repository or in artifacts built
  from it."
- `README` leads its licence section with the scope, so a human who reads one file sees it.
- `LICENSES/` plus per-artifact licence fields carry it to Scancode, FOSSA, ClearlyDefined and
  `reuse lint`, which is the only channel automated consumers actually parse.

Why this over keeping the note in LICENSE: a LICENSE file that is not the licence text is a strange
artifact, the detector outcome is unmeasured in that direction, and the failure mode is the
repository losing its licence identity entirely. The note's CONTENT is not lost, it moves to three
places that each reach a different reader. This is the premortem's third option, chosen because the
measurement supports it rather than because it is cheaper.
