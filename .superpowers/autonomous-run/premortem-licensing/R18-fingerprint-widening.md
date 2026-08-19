# R18 — widen the licensed-text fingerprint corpus to the overlay tier

**Status: LANDED for DSOMM. CSA CCM deferred with a named trigger.**

Owner: TRACT. Date: 2026-08-19. Branch: `semantic-rebuild`.

Three commits:

| commit | what |
|---|---|
| `b43ff2ba` | the halt report, kept below as the record of why stage 1 was needed |
| `4931997446489e41e21becca5e5f5add1091ef0b` | stage 1, redact CCM and DSOMM text from six tracked documents |
| `a9602e90b97c5d52ff65b19310ad925c43aa4fe5` | stage 2, gate DSOMM, defer CSA CCM |

## Outcome

- Fingerprint corpus **11,472 to 21,158**. `dsomm` 10,374, `etsi` 10,778,
  `iso_27001` 706. `csa_ccm` measured at 2,900 and held out.
- Residual after redaction, swept at **n=7, n=9 and n=12**: zero in the
  documentation channel at every width, and zero for DSOMM anywhere in the tree.
  The six remaining CCM hits are all `csa_aicm`-derived and are the deferral.
- Suite **1,814 passed, 13 failed**, failure list byte-identical to the
  pre-existing environmental 13. The gate file went 8 to 15 tests. The rest of
  the count delta is the concurrent SSDF parser task, which committed at
  `0b1cdbf`.
- `mypy --strict` clean on both touched modules.

## Sub-threshold sweep, the part no gate would have caught

Sweeping at n=7 found four more real DSOMM fragments of 7 to 9 words in three
plan documents, below the 12-word window and therefore invisible to the gate
forever. They were redacted anyway. The window is a false-positive control, not
a licence boundary, and text four words under the alarm is not thereby licensed
to anyone.

The same sweep is what confirms n=12 is right: at n=9 the only remaining DSOMM
"hits" tree-wide are shared OWASP cheat-sheet URLs and one SAMM sentence DSOMM
cites, all of which vanish at n=12. Collisions below, coverage above.

## What the coordinator's message got wrong, or did not know

1. **"Six named so far ... there may be more."** There were more, but not more
   files. The same six files carried four additional DSOMM fragments beneath the
   gate's window. A re-scan at n=12 after redacting the visible blocks would have
   reported clean and left them in place.
2. **The `A&A-02` statement was quoted in full**, not partially: its real
   specification is 13 words and all 13 were present in four documents. That is
   the number that makes the `NGRAM_WORDS` standing rule concrete, since n=14
   clears it.
3. **`test_the_recorded_deferrals_match_the_code` was a tautology as first
   written.** `LicensedFingerprints.load` already raises on a deferral mismatch,
   so an assertion against the loaded object could never fail. Mutation M7 found
   it: blanking the field killed the suite through the loader while the
   assertion under test never ran. This is the same shape as the defect the brief
   cited, two guards sharing one failure path. Split into a raw-JSON check and a
   direct loader test, both now independently reachable.
4. **The stage-1 redaction target list was slightly off.** `RUN-LEDGER.md`'s hit
   was not a fixture block but a 14-word sample quoted inside prose that was
   describing the AICM exposure. Quoting the text to describe the leak reproduced
   the leak.

## Mutation list

Every mutation was run with `PYTHONDONTWRITEBYTECODE=1` against a pristine
snapshot, restored before and after. Final diff against the snapshot is empty,
confirming no mutant survived in the tree.

| # | mutation | killed by |
|---|---|---|
| M1 | `fingerprinted_framework_ids` forgets to subtract exclusions | `test_every_framework_in_scope_contributed_fingerprints`, `[csa_ccm]` positive control |
| M2 | scope reverted to `RESTRICTED_FRAMEWORK_IDS`, the old narrow gate | schema test, in-scope coverage test |
| M3 | `NGRAM_WORDS` 12 to 14, the forbidden "clear a hit" move | `test_the_ngram_window_has_not_been_widened` plus all four positive controls |
| M4 | deferred `csa_ccm` extractor deleted | `test_every_overlay_framework_has_an_extractor` |
| M5 | DSOMM extractor narrowed to `description` only | `[dsomm]` positive control, via the n-gram count check |
| M6 | `dsomm` added to the deferral set | schema test |
| M7 | `deferred_framework_ids` blanked in the fixture | **survivor of a sort**: killed by the loader, not by the test meant to check it. Exposed the tautology in item 3 above. |
| M7b | field blanked AND loader guard disabled | `test_the_recorded_deferrals_match_the_code` and the loader test, after the fix |
| M8 | `dsomm`'s `.gitignore` line removed | `test_every_overlay_framework_has_a_gitignore_line`, which the widening bought |
| M9 | every overlay framework deferred, emptying the gate | `test_the_gate_still_covers_something` |
| M10 | loader deferral guard disabled, fixture left correct | loader test only, confirming the guard is exercised rather than always-on |
| M11 | 563 words of real DSOMM text planted in a **tracked** file | `test_no_verbatim_licensed_statement_anywhere_in_the_tree`. The end-to-end proof. Probe was unstaged and deleted, verified, and the gate returned green. |

One survivor, M7, and it exposed a real test defect rather than a code defect.

## Deferral, and what reverses it

`csa_ccm` is held out because 138 of 243 tracked `csa_aicm` descriptions are
byte-identical to a CCM specification under the same control id. The trigger is
the AICM ruling, either way:

- AICM prose leaves the tracked tree, `csa_ccm` gates cleanly and comes off
  `FINGERPRINT_EXCLUDED_FRAMEWORK_IDS`.
- AICM prose stays, and that constant is the record of why `csa_ccm` cannot join.

The extractor is registered and measured at full coverage, and
`test_every_overlay_framework_has_an_extractor` keeps it that way, so the
reversal is a decision rather than a code project.

---

# Original halt report, preserved

**Status: HALTED at requirement 4. R18 is NOT landed. No behaviour changed.**

Owner: TRACT. Date: 2026-08-19. Branch: `semantic-rebuild`.

The ruling asked for the fingerprint corpus to move from `RESTRICTED_FRAMEWORK_IDS`
(`etsi`, `iso_27001`) to `OVERLAY_FRAMEWORK_IDS` (adds `csa_ccm`, `dsomm`), and it
carried a stop condition:

> Then confirm the gate is clean on the tree as it stands. **If it is NOT clean,
> STOP and report before changing anything** — that would mean CCM or DSOMM text
> is already tracked, which is a finding, not something to fix quietly.

The tree is not clean. The extractors were built, the widened corpus was generated
into a scratch path, and the scan was run against the tracked tree. It found **12
tracked files reproducing 12 or more consecutive words of CSA CCM or OWASP DSOMM
source text.** Per the stop condition, nothing was committed and both edited source
files were reverted byte-for-byte to `HEAD`. The machinery is preserved as a patch
at the end of this document.

---

## What was built, and what it measured

Extractors were added for both frameworks, reading from the same file, sheet,
member and fields their parsers read, with the locating constants imported from
`parsers/parse_csa_ccm.py` and `parsers/parse_dsomm.py` rather than restated.

| framework | source under `data/raw/frameworks/` | units | words | n-grams |
|---|---|---|---|---|
| `csa_ccm` | `CCMv4.1.0-generated_at_2026_01_13.xlsx` | 207 specs | 5,091 | **2,900** |
| `dsomm` | `dsomm_data.zip` → `generated/model.yaml` | 194 activities | 12,103 | **10,374** |
| `etsi` | `etsi_gr_sai005_v010101p.pdf` | 1 | 10,457 | 10,778 (unchanged) |
| `iso_27001` | `ISO_IEC_27001_2022_en.md` | 93 statements | 1,687 | 706 (unchanged) |

Total unique fingerprints **11,472 → 23,935**. The 823-gram shortfall against the
column sum is entirely within-document repetition: **cross-framework overlap between
all four sets is zero**, so every hit below attributes to exactly one publisher.

The `source_sha256` each extractor recorded matches the digest its parser pins
independently (`5e7216…` for the CCM workbook, `a6d773…` for the DSOMM archive),
which is a free cross-check that both read the intended bytes.

### Requirement 5 — the title false-positive hazard, handled

Both extractors take the statement channel only, the same way `_iso_27001_statements`
does, and for the same measured reason.

- **CCM**: column D (`Control Specification`) only. Columns A and B (`Control Domain`,
  `Control Title`) are excluded. All 29 CCM section names reach this project through
  OpenCRE's public link dump and are already tracked in `data/training/hub_links*`,
  and the merge keeps overlay frameworks' titles in the tracked corpus on purpose.
  The header row is located and asserted to appear exactly once before column D is
  read, because swapped columns still yield 207 rows with titles where the
  specifications belong and nothing downstream can see it.
- **CCM domain rows contribute nothing.** A CCM domain has no text of its own; the
  statement the corpus carries for one is a list of member titles that
  `parse_csa_ccm.py` assembles and marks `synthetic`. Fingerprinting it would flag
  TRACT's own output as a CSA quotation.
- **DSOMM**: `description`, `risk`, `measure`, joined in the parser's order. The
  activity name (the YAML key, which is the control title), the dimension and the
  sub-dimension are excluded. OpenCRE puts the sub-dimension in `section_name` for
  all 214 DSOMM links, so those names are already tracked.
- DSOMM is joined rather than fingerprinted per field, unlike the per-statement units
  elsewhere. The join is what the parser writes into one `description`, so a window
  spanning the seam between `risk` and `measure` is text somebody could quote from
  the corpus, not an artifact of concatenating unrelated records.

---

## Finding 1 — CCM and DSOMM text escaped into tracked project documentation

This is a **fifth channel**, distinct from the four already on record. It is not the
parser output and not the merged corpus. It is briefing documentation that transcribes
raw sources under an explicit `**Verbatim sample**` heading so downstream implementers
know what the source looks like.

`.superpowers/autonomous-run/source-structures.md` is the origin. It carries:

- A block headed `**Verbatim sample** (row 4, first real control)` that reproduces
  row 4 of the registration-walled CCM workbook in full — the four column labels plus
  the **complete 30-word `A&A-01` Control Specification**. The column labels
  (`Control Domain`, `Control Title`, `Control ID`, `Control Specification`) are the
  XLSX's own, so this is a transcription of the workbook, not of any downstream file.
- A block headed `**Verbatim sample** (one full Activity)` reproducing a DSOMM
  activity's raw `risk` and `measure` fields, **27 consecutive words**, GPL-3.0-only.

From there it propagated into the plan and ledger documents that quote it:

| tracked file | publisher | quoted controls | longest run |
|---|---|---|---|
| `.superpowers/autonomous-run/source-structures.md` | CCM + DSOMM | `A&A-01` (30/30 words) | 30 words CCM, 27 DSOMM |
| `.superpowers/autonomous-run/premortem-v2/v3-tasks-05-08-09-11-12-13.md` | CCM | `A&A-01`, `A&A-02`, `I&S-02`, `I&S-09` | 19 words |
| `docs/superpowers/plans/2026-08-19-remaining-parsers-v3.md` | CCM | `A&A-01`, `A&A-02`, `I&S-02`, `I&S-09` | 19 words |
| `docs/superpowers/plans/2026-08-18-remaining-parsers-v2.md` | CCM | `A&A-01`, `A&A-02`, `I&S-02` | 19 words |
| `docs/superpowers/plans/2026-08-16-remaining-parsers.md` | CCM | `A&A-01`, `A&A-02` | 16 words |
| `.superpowers/autonomous-run/RUN-LEDGER.md` | CCM | `A&A-01` | 13 words |

`A&A-02` is quoted in full: its specification is 13 words and 13 words are present.

This is the exact scenario the ruling was written from. The Task 8 brief that asked an
implementer to quote real CCM specification text into a tracked CC0 fixture almost
certainly drew that text from `source-structures.md`, which is where the verbatim CCM
row lives. The implementer invented strings instead and the leak did not widen, but
the source of the request was itself already a leak.

**Not fixed here, per the stop condition.** Six files need their verbatim blocks
replaced with structural descriptions or synthetic wording that reproduces the same
shape. `RUN-LEDGER.md` and the three plan documents are other tasks' artifacts.

---

## Finding 2 — `csa_ccm` cannot be gated independently of the escalated `csa_aicm` question

The ruling said not to add `csa_aicm`, because its 243 control statements are
deliberately tracked pending an owner decision, and gating it would turn the branch
red on an unanswered question. That exclusion does not hold, because **the CCM
fingerprints gate AICM's prose by proxy.**

Measured against `data/processed/frameworks/csa_aicm.json`: **138 of 243 tracked AICM
control descriptions are byte-identical, after `normalise_for_fingerprint`, to a CCM
control specification carrying the same control id** (`A&A-01`, `A&A-02`, `A&A-04`,
`A&A-06`, `AIS-02`, `AIS-03`, `AIS-05`, `AIS-06`, and 130 more). CSA built AICM on
CCM's control set and reused CCM's text verbatim for the domains that are not
AI-specific.

So gating `csa_ccm` puts these six tracked AICM-derived artifacts into the failure list:

| tracked file | matching windows |
|---|---|
| `data/processed/frameworks/csa_aicm.json` | 4,898 |
| `data/processed/all_controls.json` | 4,898 |
| `results/review/review_export.json` | 4,898 |
| `opencre_export/CSA_AI_Controls_Matrix.csv` | 1,787 |
| `data/canary_items_for_labeling.json` | 80 |
| `tests/fixtures/csa_aicm_sample.json` | 1 |

None of these mentions `csa_ccm`. All are AICM artifacts. The finding is that the
escalated question is larger than it was framed: it is not "may TRACT track AICM
prose", it is "may TRACT track AICM prose, **57% of which is CCM prose**". The two
frameworks' licence notices in `FRAMEWORK_LICENSES` are already identical, word for
word, which is consistent with them being one rights holder's one reservation.

---

## What has to happen before R18 can land

Ordered, because the second depends on the first.

1. **Clean Finding 1.** Replace the verbatim CCM and DSOMM blocks in the six
   documentation files with structural descriptions. This is independent of any owner
   ruling: neither publisher's terms permit it and nobody has argued they do.
2. **Rule on Finding 2.** Either AICM prose comes out of the tracked tree, in which
   case `csa_ccm` gates cleanly, or AICM prose stays, in which case `csa_ccm` cannot
   be added to the fingerprint corpus without the branch going permanently red and
   the exclusion has to be recorded against `csa_ccm` as well as `csa_aicm`, naming
   the 138-control overlap as the reason.
3. **Then apply the patch below and regenerate.** `dsomm` is blocked only by step 1
   and could land ahead of step 2 if the owner wants partial coverage sooner. It is
   worth saying plainly that partial coverage leaves open exactly the CCM hole this
   ruling was written to close.

A note on what NOT to do. Raising `NGRAM_WORDS` would make every number above go away.
`A&A-02` is 13 words, so `n=14` clears it, and the longest documentation run is 30
words, so `n=31` clears everything. `tract/licensing.py` already says it: "Raising this
number to make a known overlap pass would be the 'gate that cannot fire' defect. If the
tree collides at 12, fix the tree." The tree collides at 12.

---

## Verification performed

- Baseline re-measured, not carried forward: **13 failed, 1,762 passed, 22 skipped,
  3 xfailed** with the three ignores. All 13 failures are `ModuleNotFoundError` on
  training dependencies and are environmental.
- After the revert, `git diff` against `HEAD` for `tract/licensing.py` and
  `scripts/build_licensed_fingerprints.py` is empty. No other file was touched.
  `parsers/`, `data/processed/all_controls.json` and `results/corpus/` were not
  written to.
- No model was loaded. `data/raw/` was opened read-only.
- The scan was run through `tract.licensing.fingerprint_ngrams` and
  `LicensedFingerprints.first_hit`, never a hand-rolled n-gram loop, per the ruling's
  warning about the `f"{salt}:"` prefix and `normalise_for_fingerprint`.

### Mutation testing

Not performed. Mutation testing validates tests, and no test was written — the run
halted at requirement 4 before the test phase. The ruling's bar applies to the
assertions that would have shipped, and it carries forward to whoever resumes.

The one falsifiability check that *was* run belongs to the finding rather than to a
test: cross-framework fingerprint overlap was measured at **zero across all four
sets**, which is what makes the per-file attribution above sound. Without it, a CCM
hit could have been an ETSI hit wearing a different label.

---

## Preserved machinery

Reconstruct with `git apply` from the repository root. It is inert on its own: it
widens `build()`'s scope but does not regenerate
`tests/fixtures/licensed_text_fingerprints.json`, and it does not touch
`tests/test_licensed_text_not_tracked.py`, so applying it alone changes no gate.
Do not regenerate the fixture until steps 1 and 2 above are done.

Still to be written when it resumes, and not in this patch:

- `tests/test_licensed_text_not_tracked.py` must read `fingerprinted_framework_ids()`
  in place of `RESTRICTED_FRAMEWORK_IDS` in both the metadata-field assertion and the
  per-framework-coverage assertion.
- The real-source positive control should be parametrized over every fingerprinted
  framework and should call the generator's own extractors. The existing
  `test_real_fingerprints_flag_a_real_statement` reimplements the ISO Annex A row
  parse by hand, so the test's copy and the generator's can drift apart without
  anything noticing. That is a live defect in the current gate, found while reading
  it for this ruling, and worth fixing whether or not R18 lands.

```diff
diff --git a/scripts/build_licensed_fingerprints.py b/scripts/build_licensed_fingerprints.py
index 0c00b93..1e4d5be 100644
--- a/scripts/build_licensed_fingerprints.py
+++ b/scripts/build_licensed_fingerprints.py
@@ -8,13 +8,19 @@ Writes `tests/fixtures/licensed_text_fingerprints.json`, the committed input to
 source out of gitignored `data/raw/`, so on a fresh clone and in CI it skipped
 and the skip reported green.
 
-RE-PIN THE FINGERPRINTS WHENEVER A RESTRICTED SOURCE CHANGES. Every entry is
-derived from one document's bytes, and the sha256 of that document is recorded
-alongside the entry. Re-fetching ISO/IEC 27001 or ETSI GR SAI 005, correcting a
-parse, or adding a framework to RESTRICTED_FRAMEWORK_IDS all require running
-this script on a checkout that holds the sources under data/raw/, then
-committing the regenerated file. `--check` reports drift without writing, so a
-reviewer can tell a stale fingerprint file from a current one.
+Scope is `tract.licensing.fingerprinted_framework_ids()`: every framework that
+routes to the gitignored overlay, less a named and justified exclusion. It was
+RESTRICTED_FRAMEWORK_IDS, which covered ISO/IEC 27001 and ETSI and left CSA CCM
+and DSOMM contributing nothing, so a quotation from either would have walked
+past the gate that exists to catch exactly that.
+
+RE-PIN THE FINGERPRINTS WHENEVER A SOURCE CHANGES. Every entry is derived from
+one document's bytes, and the sha256 of that document is recorded alongside the
+entry. Re-fetching a source, correcting a parse, or moving a framework into the
+overlay tier all require running this script on a checkout that holds the
+sources under data/raw/, then committing the regenerated file. `--check`
+reports drift without writing, so a reviewer can tell a stale fingerprint file
+from a current one.
 
 The output carries hashes only. There is no free-text field in the schema, and
 tests/test_licensed_text_not_tracked.py asserts that, so this script cannot
@@ -33,7 +39,7 @@ from collections.abc import Callable
 from pathlib import Path
 from typing import Any, Final
 
-from tract.config import RAW_FRAMEWORKS_DIR, RESTRICTED_FRAMEWORK_IDS
+from tract.config import RAW_FRAMEWORKS_DIR
 from tract.io import atomic_write_json
 from tract.licensing import (
     FINGERPRINT_ALGORITHM,
@@ -44,6 +50,7 @@ from tract.licensing import (
     NGRAM_WORDS,
     LicensedFingerprints,
     fingerprint_ngrams,
+    fingerprinted_framework_ids,
 )
 
 logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
@@ -157,10 +164,179 @@ def _etsi_normative_text(path: Path) -> list[str]:
     return ["\n".join(kept)]
 
 
-# One extractor per restricted framework. Keyed on framework_id so adding a
-# framework to RESTRICTED_FRAMEWORK_IDS without deciding what its statement
-# text is raises here instead of producing a fingerprint file with a hole.
+# Cell values arrive with the workbook's own line breaks and runs of spaces.
+# The parser collapses both before it builds a Control, so collapsing them here
+# too keeps the fingerprinted wording identical to the wording that could leak.
+_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")
+
+
+def _check_pinned_name(path: Path, expected: str, owner: str) -> None:
+    """Refuse a source file whose parser has since been re-pinned elsewhere.
+
+    The filenames in _EXTRACTORS and the ones the parsers pin are two records
+    of one fact. A re-pin that moves only the parser leaves this script reading
+    a superseded document, and the fingerprints would then describe text that
+    no longer reaches the corpus.
+
+    Raises:
+        ValueError: the staged filename is not the one *owner* pins.
+    """
+    if path.name == expected:
+        return
+    raise ValueError(
+        f"{path.name} is not {expected!r}, which {owner} pins. Fingerprints "
+        f"built from a superseded document describe text that no longer "
+        f"reaches the corpus. Update _EXTRACTORS and re-measure together."
+    )
+
+
+def _csa_ccm_specifications(path: Path) -> list[str]:
+    """CSA CCM v4.1.0 control specifications, every other column excluded.
+
+    Read from the same workbook, the same sheet and the same column
+    parsers/parse_csa_ccm.py reads, with the constants imported from it rather
+    than restated, so a re-pin of the workbook moves both together.
+
+    The Control Title and Control Domain columns are left out for the reason
+    ISO's titles are: OpenCRE's public link dump carries all 29 CCM section
+    names, they are already tracked in data/training/hub_links*, and the merge
+    keeps overlay frameworks' titles in the tracked corpus on purpose. The
+    normative text at issue is the specification.
+
+    Domain rows contribute nothing here. A CCM domain has no text of its own,
+    and the statement the corpus carries for one is a list of member titles
+    that parse_csa_ccm.py assembled and marked synthetic. Fingerprinting
+    TRACT's own assembled text would flag TRACT's own output as a CSA
+    quotation.
+
+    Raises:
+        ValueError: the header row is absent or reordered, or no specification
+            rows matched. Column order is the whole assumption: swapped columns
+            still yield 207 rows, with titles where the specifications should
+            be, and every count check downstream still passes.
+    """
+    import openpyxl  # imported here so the gate never needs it, only the build
+
+    from parsers.parse_csa_ccm import EXPECTED_HEADER, SHEET_NAME, WORKBOOK_NAME
+
+    _check_pinned_name(path, WORKBOOK_NAME, "parsers/parse_csa_ccm.py")
+    specification_column = EXPECTED_HEADER.index("Control Specification")
+    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
+    try:
+        if SHEET_NAME not in workbook.sheetnames:
+            raise ValueError(
+                f"{path}: no {SHEET_NAME!r} sheet, only "
+                f"{workbook.sheetnames}. The CAIQ sheet is the "
+                f"self-assessment questionnaire and is not the controls."
+            )
+        rows = [
+            tuple(
+                "" if cell is None else _WHITESPACE.sub(" ", str(cell)).strip()
+                for cell in (list(row) + [None] * 4)[:4]
+            )
+            for row in workbook[SHEET_NAME].iter_rows(values_only=True)
+        ]
+    finally:
+        workbook.close()
+
+    headers = sum(1 for row in rows if row == EXPECTED_HEADER)
+    if headers != 1:
+        raise ValueError(
+            f"{path}: the {SHEET_NAME} sheet carries {headers} header rows "
+            f"equal to {list(EXPECTED_HEADER)}, expected 1. Without that "
+            f"anchor there is nothing to say column "
+            f"{specification_column} still holds specifications, and "
+            f"fingerprinting the title column produces false positives on "
+            f"tracked files that carry nothing licensed."
+        )
+
+    statements = [
+        row[specification_column]
+        for row in rows
+        if row != EXPECTED_HEADER and row[2] and row[specification_column]
+    ]
+    if not statements:
+        raise ValueError(
+            f"{path}: no control specification rows matched. The sheet changed "
+            f"shape and the fingerprints would be silently empty."
+        )
+    return statements
+
+
+def _dsomm_statements(path: Path) -> list[str]:
+    """OWASP DSOMM activity statements, activity and dimension names excluded.
+
+    Read from the archive member and the field set parsers/parse_dsomm.py
+    reads, joined in the same order, so what is fingerprinted is the text that
+    would leak if the parsed corpus reached git.
+
+    Joined rather than fingerprinted per field, unlike the per-statement units
+    elsewhere in this file. The join is what the parser writes into one
+    `description`, so a window spanning the seam between `risk` and `measure`
+    is text somebody could quote from the corpus, not an artifact of
+    concatenating two unrelated records.
+
+    The activity name is the YAML key and is the control's title. It is left
+    out for the same reason ISO's and CCM's titles are: `dimension` and
+    `sub_dimension` are what OpenCRE's public link dump puts in `section_name`
+    for all 214 DSOMM links, and they are already tracked.
+
+    Raises:
+        ValueError: the model member is absent or ambiguous, the stream is not
+            a meta document followed by a model mapping, or no activity
+            produced text.
+    """
+    import zipfile
+    from io import BytesIO
+
+    import yaml
+
+    from parsers.parse_dsomm import ARCHIVE_NAME, MODEL_SUFFIX, STATEMENT_FIELDS
+
+    _check_pinned_name(path, ARCHIVE_NAME, "parsers/parse_dsomm.py")
+    with zipfile.ZipFile(BytesIO(path.read_bytes())) as archive:
+        names = [n for n in archive.namelist() if n.endswith(MODEL_SUFFIX)]
+        if len(names) != 1:
+            raise ValueError(
+                f"{path}: expected exactly one {MODEL_SUFFIX} member, found "
+                f"{names}. The generated file is what flattens the "
+                f"per-subdimension YAMLs into one document."
+            )
+        raw = archive.read(names[0]).decode("utf-8")
+
+    documents = list(yaml.safe_load_all(raw))
+    if len(documents) != 2 or not isinstance(documents[1], dict):
+        raise ValueError(
+            f"{path}: {MODEL_SUFFIX} is not a meta document followed by a "
+            f"model mapping (got {len(documents)} document(s)). The layout "
+            f"changed and the fingerprints would be silently empty."
+        )
+
+    statements: list[str] = []
+    for sub_dimensions in documents[1].values():
+        for activities in sub_dimensions.values():
+            for body in activities.values():
+                parts = [
+                    str(body.get(field) or "").strip()
+                    for field in STATEMENT_FIELDS
+                ]
+                statement = "\n\n".join(part for part in parts if part)
+                if statement:
+                    statements.append(statement)
+    if not statements:
+        raise ValueError(
+            f"{path}: no activity carried text in any of {STATEMENT_FIELDS}. "
+            f"The schema changed and the fingerprints would be silently empty."
+        )
+    return statements
+
+
+# One extractor per fingerprinted framework. Keyed on framework_id so adding a
+# framework to the overlay tier without deciding what its statement text is
+# raises here instead of producing a fingerprint file with a hole.
 _EXTRACTORS: Final[dict[str, tuple[str, Callable[[Path], list[str]]]]] = {
+    "csa_ccm": ("CCMv4.1.0-generated_at_2026_01_13.xlsx", _csa_ccm_specifications),
+    "dsomm": ("dsomm_data.zip", _dsomm_statements),
     "iso_27001": ("ISO_IEC_27001_2022_en.md", _iso_27001_statements),
     "etsi": ("etsi_gr_sai005_v010101p.pdf", _etsi_normative_text),
 }
@@ -175,23 +351,26 @@ def _sha256(path: Path) -> str:
 
 
 def build() -> dict[str, Any]:
-    """Read every restricted source and return the fingerprint document.
+    """Read every fingerprinted source and return the fingerprint document.
 
     Raises:
-        KeyError: a framework is restricted but has no extractor here.
-        FileNotFoundError: a restricted source is not staged under data/raw/.
+        KeyError: a framework is in scope but has no extractor here.
+        FileNotFoundError: a source in scope is not staged under data/raw/.
+        ValueError: an excluded framework id names no known framework.
     """
-    missing_extractors = sorted(set(RESTRICTED_FRAMEWORK_IDS) - set(_EXTRACTORS))
+    in_scope = fingerprinted_framework_ids()
+    missing_extractors = sorted(in_scope - set(_EXTRACTORS))
     if missing_extractors:
         raise KeyError(
-            f"No fingerprint extractor for restricted framework(s) "
-            f"{missing_extractors}. Add one to _EXTRACTORS naming the staged "
-            f"document and the function that isolates its statement text."
+            f"No fingerprint extractor for framework(s) {missing_extractors}. "
+            f"Add one to _EXTRACTORS naming the staged document and the "
+            f"function that isolates its statement text, or name the framework "
+            f"in FINGERPRINT_EXCLUDED_FRAMEWORK_IDS with a reason."
         )
 
     documents: list[dict[str, Any]] = []
     fingerprints: set[str] = set()
-    for framework_id in sorted(RESTRICTED_FRAMEWORK_IDS):
+    for framework_id in sorted(in_scope):
         filename, extract = _EXTRACTORS[framework_id]
         path = RAW_FRAMEWORKS_DIR / framework_id / filename
         if not path.exists():
diff --git a/tract/licensing.py b/tract/licensing.py
index 752e51f..14ba122 100644
--- a/tract/licensing.py
+++ b/tract/licensing.py
@@ -335,6 +335,50 @@ FINGERPRINT_PATH: Final[Path] = (
     PROJECT_ROOT / "tests" / "fixtures" / "licensed_text_fingerprints.json"
 )
 
+# ── Which frameworks the fingerprint corpus covers ────────────────────────
+#
+# The corpus used to cover RESTRICTED_FRAMEWORK_IDS, so it held etsi and
+# iso_27001 and nothing else. dsomm and csa_ccm route to the same gitignored
+# overlay for the same reason -- their publishers' terms are ones a CC0 grant
+# cannot carry -- and neither contributed a single n-gram. A task brief then
+# asked for a tracked test fixture quoting real CCM specification text, the
+# implementer invented the strings instead of copying them, and had they not,
+# nothing here would have fired. A control whose enforcement is "someone reads
+# carefully" is not a control.
+#
+# Derived from OVERLAY_FRAMEWORK_IDS rather than listed, so a framework moved
+# into the overlay tier later is either fingerprinted or names itself in the
+# KeyError that scripts/build_licensed_fingerprints.py raises for a framework
+# with no extractor. Neither outcome is silence.
+#
+# csa_aicm is excluded, and the exclusion is a decision rather than an
+# oversight. Its 243 control statements are deliberately TRACKED today, at a
+# 176-character median, pending an owner ruling on whether CSA's notice permits
+# that. Fingerprinting it would turn the branch red on a question nobody has
+# answered, and a red gate that everyone agrees to ignore is worse than no
+# gate. It sits outside OVERLAY_FRAMEWORK_IDS as of 2026-08-19, so subtracting
+# it changes nothing today; the entry is here so that a ruling which moves it
+# into the overlay does not switch this gate on as a side effect.
+FINGERPRINT_EXCLUDED_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({"csa_aicm"})
+
+
+def fingerprinted_framework_ids() -> frozenset[str]:
+    """Every framework the tracked fingerprint corpus must cover.
+
+    Raises:
+        ValueError: an excluded id names no framework this project knows. A
+            typo would silently exclude nothing and read as though it excluded
+            something, which is the shape of a gate that quietly widened.
+    """
+    unknown = FINGERPRINT_EXCLUDED_FRAMEWORK_IDS - set(FRAMEWORK_LICENSES)
+    if unknown:
+        raise ValueError(
+            f"FINGERPRINT_EXCLUDED_FRAMEWORK_IDS names {sorted(unknown)}, "
+            f"which is not in FRAMEWORK_LICENSES. An exclusion that matches no "
+            f"framework excludes nothing while reading as a live decision."
+        )
+    return frozenset(OVERLAY_FRAMEWORK_IDS) - FINGERPRINT_EXCLUDED_FRAMEWORK_IDS
+
 # The generated file's schema. The gate asserts these are the ONLY keys
 # present, which is what makes "this file contains no licensed text" a checked
 # property rather than a claim: there is no free-text field to hide prose in.
```
