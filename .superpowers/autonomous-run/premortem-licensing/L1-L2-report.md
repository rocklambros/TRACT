# L1 and L2 — implementation report

Branch `semantic-rebuild`. Five commits, oldest first.

| commit | scope |
|---|---|
| `217df5b` | L1.1 `LICENSES/` |
| `a525e06` | L1.2 modification statement, L1.3 CSA entries |
| `95ab3b9` | L1.4 one licence declaration, L1.5 ship it in both bundles |
| `64ac86c` | L2.1 merge filter |
| `a3ce055` | L2.2 canonical export |

`tract/config.py` was not touched in any of them. `CONDITIONAL_FRAMEWORK_IDS`,
`RESTRICTED_FRAMEWORK_IDS` and `OVERLAY_FRAMEWORK_IDS` membership is untouched
and left for L3. `tract/text_selection.py` was not touched.

## L1.1 — LICENSES/

Derived from `tract.config.FRAMEWORK_LICENSES`, not from the brief's list. The
derivation is `tract.licensing.spdx_identifiers`: a recorded licence is an SPDX
expression when every whitespace-separated token is an operator or matches the
SPDX identifier grammar, and is not the `UNDETERMINED` sentinel. Prose
reservations fail because English carries commas and parentheses.

Five identifiers come out of the registry:

| identifier | declared by |
|---|---|
| `Apache-2.0` | mitre_atlas |
| `CC-BY-4.0` | cosai |
| `CC-BY-SA-3.0` | biml |
| `CC-BY-SA-4.0` | asvs, biml, owasp_agentic_top10, owasp_cheat_sheets, owasp_dsgai, owasp_llm_top10, owasp_llm_top10_2026, owasp_ml_top10, owasp_proactive_controls, owasp_top10_2021, samm, wstg |
| `GPL-3.0-only` | dsomm |

Plus `CC0-1.0`, which is not in `FRAMEWORK_LICENSES` because that registry
records third-party terms only. It is shipped because it is the repository's
own licence and REUSE, Scancode and ClearlyDefined read the project licence
from `LICENSES/` too. Six files total.

**MIT is not shipped.** The only MIT declaration anywhere in the tree was
`tract/publish/model_card.py:110`, and L1.4 removes it. A licence text nothing
declares makes the inventory wrong in the permissive direction, which is the
same class of error as the CC0 over-claim, so the test fails on a stray file.

### No text shipped, and why (17 sources)

Eight state no terms in the staged artifact and are recorded `UNDETERMINED`:
`aiuc_1`, `capec`, `cwe`, `nist_800_53`, `nist_ai_100_2`, `nist_ai_600_1`,
`nist_ai_rmf`, `owasp_ai_exchange`.

Nine carry a publisher's own prose notice: `csa_aicm`, `csa_ccm`, `enisa`,
`etsi`, `eu_ai_act`, `eu_gpai_cop`, `iso_27001`, `nist_800_63`, `nist_ssdf`.

Nothing was invented for either group. The set is asserted by equality, so
resolving an `UNDETERMINED` or ingesting a new prose source turns the suite red
until NOTICE says what changed.

### Fetch provenance

All six fetched from the publisher's own canonical plain text, not from a
mirror or a paraphrase. `LICENSES/CC0-1.0.txt` was verified byte-identical to
the CC0 legal code already carried in the repository's `LICENSE`.

| file | source | sha256 (first 12) |
|---|---|---|
| `Apache-2.0.txt` | `https://www.apache.org/licenses/LICENSE-2.0.txt` | `cfc7749b96f6` |
| `CC-BY-4.0.txt` | `https://creativecommons.org/licenses/by/4.0/legalcode.txt` | `9ba9550ad484` |
| `CC-BY-SA-3.0.txt` | `https://creativecommons.org/licenses/by-sa/3.0/legalcode.txt` | `3f941b3b89cf` |
| `CC-BY-SA-4.0.txt` | `https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt` | `28a9529c7d0b` |
| `CC0-1.0.txt` | `https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt` | `a2010f343487` |
| `GPL-3.0-only.txt` | `https://www.gnu.org/licenses/gpl-3.0.txt` | `3972dc9744f6` |

No fetch failed. Nothing was paraphrased.

## L1.2 — modification statement

Named from the code, not from the brief's summary. NOTICE gains a
`Modifications to framework text` section covering, in order:

At storage, `tract/sanitize.py::sanitize_text`, applied to title, description
and full text by `sanitize_control`:

1. null bytes (U+0000) replaced with a space (`_strip_null_bytes`)
2. NFC normalisation (`_normalize_unicode`)
3. zero-width removal, U+200B U+200C U+200D U+FEFF (`_strip_zero_width`)
4. HTML unescape then tag strip (`strip_html`)
5. PDF ligatures ﬀ ﬁ ﬂ ﬃ ﬄ folded to ASCII (`_fix_ligatures`)
6. line-broken hyphenation rejoined (`_fix_hyphenation`)
7. whitespace runs collapsed, then stripped (`_collapse_whitespace`)
8. word-boundary truncation at `tract.config.DESCRIPTION_MAX_LENGTH` (2000)
   for descriptions, 500 for titles, 50,000 for full text; the untruncated
   sanitised string is preserved in `full_text`

At training and evaluation, `tract/text_selection.py`:

9. `strip_markup`, then a second NFC normalisation and null-byte strip in
   `prepare_anchor`
10. stop word removal by `tract/stopwords.py::filter_stopwords` against the
    78-word corpus-derived list at `data/processed/stopwords.json`
11. truncation to `tract.config.MAX_ANCHOR_CHARS` (2150)

Plus `tract.config.CONTROL_ELISION_MARKER` (`[...]`) inserted at the point of
loss where a source is damaged, with the `damaged` metadata key.
`parsers/parse_iso_27001.py` is the only parser that does this today.

The three bounds and the marker are read from `tract.config` by
`tests/test_notice_completeness.py`, so raising a bound without updating NOTICE
fails. Verified: bumping `DESCRIPTION_MAX_LENGTH` to 2500 turns it red, and so
does changing `CONTROL_ELISION_MARKER`.

## L1.3 — the two CSA entries

**csa_aicm.** Facts stated, not resolved. Measured 2026-08-19: 243 controls
tracked in git at `data/processed/frameworks/csa_aicm.json`, description length
39 / 176 / 485 characters (min / median / max), which is control prose and not
titles. Notice reserves redistribution outright. In no tier. Structural cause
named: `tests/test_framework_licenses.py::_copyleft` selects on the substrings
`GPL` and `CC-BY-SA`, so a publisher reserving rights outright produces no tier
membership. Flagged as an owner question.

A test asserts the framework is still untiered and still tracked, so it fires
the moment L3 resolves it and forces NOTICE to be rewritten rather than going
stale. Verified: adding `csa_aicm` to `CONDITIONAL_FRAMEWORK_IDS` turns it red.

**csa_ccm basis.** NOTICE now records that the basis is not on record and names
the three candidates that differ materially in scope and in whether they
transfer to a fork: written agreement, membership, fair-use judgment. The test
guarding it says in its own docstring to delete it on the commit that records
the basis.

## L1.4 — one licence declaration

All three published declarations now read from
`tract.licensing.published_license_frontmatter()`:

```
license: other
license_name: tract-mixed-sources
license_link: NOTICE
```

- `tract/dataset/card.py` — was `license: cc-by-sa-4.0`
- `tract/publish/model_card.py` — was `license: mit`
- `tract/dataset/bundle.py` zenodo record — was `"license": "CC-BY-SA-4.0"`,
  now carries `license`, `license_name` and `license_link` from the same
  constants
- `pyproject.toml` — was absent, now `license = "CC0-1.0"` plus
  `license-files = ["LICENSE", "NOTICE", "LICENSES/*.txt"]`

A single identifier IS correct for `pyproject.toml`, and that is a different
claim from the other three: the wheel ships `tract*` and nothing else, so no
framework text reaches it. The test asserts that premise
(`tool.setuptools.packages.find.include == ["tract*"]`) alongside the value, so
widening the wheel's contents turns it red.

`license: other` is derived rather than asserted. One test proves the corpus
carries more than one SPDX identifier and at least one prose reservation; a
second asserts the published id is `other`. If the corpus ever narrows to a
single set of terms the first fails and the decision is retaken.

Both cards' bodies were rewritten too. The dataset card's `## License` section
had a full CC BY-SA 4.0 grant in prose; the model card's said "MIT License for
model weights and code". Both now state that no single licence covers the
artifact, point at NOTICE and `LICENSES/`, and say which claim was withdrawn
and why. The model card keeps the MIT reference for the base model, because
that part is true and its terms travel with the weights.

The dataset bundle's `LICENSE` was a hand-typed summary of the CC BY-SA 4.0
deed. It is now a byte copy of the repository's `LICENSE` (CC0 legal code plus
scope note), which removes both the over-claim and the paraphrase.

## L1.5 — NOTICE ships inside both bundles

`tract.licensing.copy_licensing_files(staging_dir)` copies `LICENSE`, `NOTICE`
and `LICENSES/` and is called from `bundle_dataset()` and from
`publish_to_huggingface()`. It validates every precondition before it writes
anything, because a partial licence record in a staging directory is worse than
none: the publish path continues past it and uploads an artifact that looks
complete.

Tests: the dataset bundle must contain `NOTICE`, byte-equal to the repository's;
the shipped NOTICE must still carry the modification statement; the shipped
`LICENSES/` must have exactly the repository's file set; the bundle `LICENSE`
must be the CC0 dedication and must not contain the string `CC BY-SA 4.0`; the
model staging directory must contain all three. Verified: deleting either call
site turns the suite red.

## L2.1 — merge filter

`parsers/merge_all_controls.py` split on `RESTRICTED_FRAMEWORK_IDS` alone.
Widened to `OVERLAY_FRAMEWORK_IDS`, and `test_merged_corpus_carries_no_restricted_prose`
renamed to `test_merged_corpus_carries_no_unpublishable_prose` and widened to
match.

**Semantics, and a measured correction to the brief.** The brief said widening
is "a no-op on current data". Under a wholesale exclusion it is not: the seven
conditional frameworks are currently *inside* the tracked `all_controls.json`,
contributing 341 controls, and dropping them would take the tracked artifact
from 29 frameworks / 4141 controls to 22 / 3800 and shift every downstream
count. So the filter withholds prose rather than dropping frameworks:

- restricted frameworks are excluded outright, unchanged
- overlay frameworks keep `control_id`, `title` and structure, and have
  `description` reduced to the title and `full_text` cleared

This is also the only reading under which the widened gate is not vacuous. If
the framework were absent entirely, the gate's loop body would never execute
and the assertion could only pass, which the brief forbids. The gate now
asserts it inspected something.

Measured before the change: all 341 tracked overlay controls already have
`description == title` and no `full_text`. Rebuilding the tracked corpus with
the new filter reproduces the committed `data/processed/all_controls.json` byte
for byte, sha256 `7106642cb3a79953...`, 29 frameworks / 4141 controls. A
control with nothing to redact is passed through as the same object, so the
byte-identity is by construction rather than by argument.

`full_text` is filtered alongside `description`, because `sanitize_control`
moves anything over `DESCRIPTION_MAX_LENGTH` into `full_text` and a filter on
`description` alone would publish the longest statements in the corpus.

The overlay-writing condition widened too: the overlay is written when any
overlay member is on disk, not only a restricted one.

**Proof it is a real gate.** `tests/test_merge_licensed_overlay.py` stages a
conditional framework carrying prose. Reverting the filter to
`RESTRICTED_FRAMEWORK_IDS` turns four of those tests red. Planting one DSOMM
prose description in the tracked `all_controls.json` turns the tree-wide gate
red. Both directions confirmed.

## L2.2 — canonical export

`canonical_export/` added to `.gitignore`, asserted through `git check-ignore`
rather than by reading the file, because a line shadowed by a later negation
ignores nothing.

`tract/export/canonical.py` gains `exportable_description(framework_id,
description)`, applied in `build_snapshot`. Keyed on the framework, not on the
text: a check that judged "is this string too much of the standard" would need
a threshold, and a threshold gets raised until the gate stops firing. Empty
`framework_id` raises `ValueError` rather than defaulting to publishable.

**Decision: withheld, not omitted.** An overlay framework exports its section
identifiers, its titles, its hyperlinks and its CRE mappings in full, and
exports a standing sentence in place of the publisher's control text. Two
grounds. The mapping is the deliverable an OpenCRE RFC is asking for, and
omitting these frameworks would withhold TRACT's own CC0 contribution in order
to protect somebody else's text, which serves nobody. And OpenCRE already
publishes these section identifiers and names, which is the same reasoning that
already keeps titles in the tracked corpus.

The placeholder explains itself and names the licence, rather than being an
empty string. A recipient holding only `snapshot.json` has to distinguish a
withheld statement from an absent one, and an empty field says the second
thing. Example: `[Control text withheld. dsomm is published under GPL-3.0-only,
which TRACT cannot sublicense. Section identifier, title and CRE mapping are
unaffected. See NOTICE at https://github.com/rocklambros/TRACT.]`

No field was added to the export schema. `compute_content_hash` dumps every
field, so a new one would make every snapshot already in `export_history` fail
its integrity check on load with a hard `ValueError`. The withholding is
reported per framework in `export_canonical`'s return value instead, which a
`--dry-run` prints before anything is sent.

Four directions verified red: no filter, filter over every framework, an empty
placeholder, and a removed gitignore line.

## Verification

Baseline before this work: 13 failed, 1428 passed, 22 skipped, 3 xfailed.

A concurrent task owns `tract/text_selection.py` and edited it mid-run,
producing 11 unrelated failures in `tests/test_text_selection.py`. My final run
excludes that file so the result is attributable: **13 failed, 1413 passed, 22
skipped, 3 xfailed**, failure set byte-identical to baseline (the same 13
model-loading paths). Before the concurrent edits landed, a full run including
that file gave 13 failed / 1459 passed with the identical failure set.

`mypy --strict` over `tract/ parsers/ scripts/phase1a/ scripts/phase1b/
scripts/phase0/runpod_provision.py`: 26 errors, error set diffed byte-identical
against `a525e06` in a scratch worktree. All 26 are the pre-existing
`huggingface_hub` missing-stubs class. `tests/test_merge_licensed_overlay.py`
is `--strict` clean; `tests/test_canonical_export.py` holds at its pre-existing
30 errors, none added.

`results/corpus/before.json` and `results/corpus/link_resolution_before.jsonl`
are unmodified.

Every assertion added in this work was exercised in both directions. The
perturbation probes and their outcomes are listed under each section above. One
assertion was strengthened after failing that test: the licence-text phrase
check passed on a CC-BY-4.0 file truncated to 20 lines, because its marker is
the title line, so a sha256 pin with recorded fetch provenance was added
alongside it.

## Concerns

1. **`opencre_export/` has the same defect as `canonical_export/` had, and is
   out of my scope.** `tract export` writes a `<Standard>|description` column
   with no licence filter, its default output directory is `./opencre_export`,
   and that directory is not gitignored and has 7 tracked files today. None of
   the 5 exported frameworks is currently in `OVERLAY_FRAMEWORK_IDS`, so
   nothing has leaked, but `tract export --framework dsomm && git add -A` would
   stage GPL-3.0 text. It is F5 in a sibling path. It needs the same two-part
   fix.

2. **`csa_aicm` is unresolved and its text is in git now.** 243 controls at a
   176-character median description, tracked, under a notice reserving
   redistribution. NOTICE states it and a test holds the statement true, but
   nothing removes the exposure. `opencre_export/CSA_AI_Controls_Matrix.csv` is
   tracked and carries those descriptions. Owner decision.

3. **`pyproject.toml` now needs setuptools >= 77 to install without build
   isolation.** The PEP 639 `license = "CC0-1.0"` string form is rejected by
   the setuptools 75.1.0 currently installed on this machine, confirmed by
   running its config validator. CI is unaffected: it uses `pip install -e .`
   with build isolation, which honours the existing `setuptools>=82.0.1` build
   requirement. Anyone using `--no-build-isolation` on an old environment gets
   a clear error naming `project.license`. The legacy table form was not used
   because setuptools deprecated it in 77.0.0 with removal scheduled before
   today's date, and the pinned build backend is 82.

4. **The SPDX parse has a known residual.** A future single-word prose licence
   such as `Proprietary` would be read as an identifier and demand a licence
   text that cannot exist. It fails loudly naming the framework, which is the
   right direction, but it is a failure mode worth knowing about before
   somebody records a one-word licence.

5. **`test_notice_records_that_the_ccm_ruling_basis_is_missing` is a
   placeholder with a deletion instruction in its docstring.** It fails only if
   the paragraph is removed. That is deliberate and it is the weakest assertion
   added here. It should be deleted on the commit that records the basis.

6. **The tier derivation is still substring-based and still blind.**
   `_copyleft` matches `GPL` and `CC-BY-SA`. Fixing it means touching tier
   membership, which is L3. Until then, any new source whose publisher reserves
   rights outright joins the tracked corpus with nothing firing, exactly as
   `csa_aicm` did.
