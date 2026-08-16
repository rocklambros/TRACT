# Primary source structures — 10 newly fetched frameworks + csa_ccm

Written after fetching all sources via `scripts/fetch_frameworks.py --all`.
Every file, size, and hash below is what is actually on disk at
`data/raw/frameworks/<framework_id>/` as of 2026-08-15. Cross-referenced
against `data/training/hub_links_by_framework.json` for each framework's
OpenCRE section_id / section_name shape. This is the input to the parser
implementation plan — a parser author should not need to re-open any of
these sources to know their shape.

Sample byte counts and hashes are recorded in
`data/processed/framework_sources.json`; not repeated in full here except
where load-bearing.

---

## dsomm — DevSecOps Maturity Model

**File**: `dsomm_data.zip` (198,672 bytes), GitHub archive of
`devsecopsmaturitymodel/DevSecOps-MaturityModel-data` pinned to commit
`ca6e5174aed85a7bdbb845cb7431fec21c224d06` (branch `main`).

**Relevant inner path**: `generated/model.yaml` (341,065 bytes uncompressed).
This is a single generated file that already flattens the per-subarea YAML
sources under `src/assets/YAML/default/<Dimension>/<SubDimension>.yaml`
(26 files) into one document. Use the generated file; it is the same data
without a 26-file join.

**Format**: YAML, two documents in one stream (`---` separated): doc 0 is a
`meta` block (`version: 4.3.1`, `released: "2026-06-05"`), doc 1 is the model
itself, nested three levels: `Dimension -> SubDimension -> Activity name ->
{uuid, description, risk, measure, assessment, level, difficultyOfImplementation,
usefulness, implementation, references, isImplemented, evidence, comments, tags}`.

Five top-level Dimensions: `Build and Deployment`, `Culture and Organization`,
`Implementation`, `Information Gathering`, `Test and Verification`.

**Delimiter**: each leaf Activity is a YAML mapping key at the third nesting
level. 194 leaf activities total across the file (counted by summing
`len(activities)` per SubDimension).

**Field mapping**:
- `control_id` = the `uuid` field (a GUID). **This is the OpenCRE join key.**
  `hub_links_by_framework.json["dsomm"][i]["section_id"]` is this same GUID
  verbatim (e.g. `2a44b708-734f-4463-b0cb-86dc46344b2f`).
- `title` = the Activity's own dict key (e.g. `"Inventory of production
  components"`). Not a separate field — it's the map key one level above
  `uuid`.
- `description` = the `description` field, multi-paragraph prose (markdown-ish,
  uses `*emphasis*` and bullet lists in some entries).
- **`section_name` in the link data is NOT the Activity title** — it's the
  SubDimension name (e.g. `"Deployment"`, `"Design"`, `"Process"`), one level
  above the Activity. Many distinct Activities/uuids share the same
  section_name. A parser matching purely on section_name would collapse
  dozens of distinct controls into one.

**Verbatim sample** (one full Activity):
```yaml
Inventory of production components:
  uuid: 2a44b708-734f-4463-b0cb-86dc46344b2f
  description: |
    ... [full text confirmed present under Build and Deployment > Deployment] ...
  risk: In case a vulnerability of severity high or critical exists, it needs
    to be known where an artifacts (e.g. container image) with that vulnerability
    is deployed.
  measure: A documented inventory of artifacts in production like container images...
  level: 1
  references:
    openCRE:
    - https://www.opencre.org/node/standard/DevSecOps%20Maturity%20Model%20%28DSOMM%29/section/Deployment/subsection/Inventory%20of%20production%20components
```
(Trimmed for length; the full record also carries `assessment`,
`difficultyOfImplementation`, `usefulness`, `implementation` (a list of named
tool links), `isImplemented`, `evidence`, `comments`, `tags`.)

**Count**: 194 leaf activities in the source. `hub_links_by_framework.json`
references 214 link rows against 183 distinct uuids (91 activities are
targeted by more than one CRE, 11 activities are unreferenced). Coverage:
183/194 = 94%.

**Parsing difficulty**: low. The `references.openCRE` field on many entries
is a literal OpenCRE URL that embeds the SubDimension and Activity name —
useful as an independent cross-check of the join, not needed for the join
itself since uuid already matches directly. Note: this repo (`-data`) is
distinct from the DSOMM web app repo; do not confuse the two if searching
GitHub manually.

---

## wstg — OWASP Web Security Testing Guide

**File**: `wstg.zip` (15,620,594 bytes), GitHub archive of `OWASP/wstg` pinned
to commit `95ce6cfe5d463bbde88aa52b3171b123a1ea9ada` (branch `master`).

**Relevant inner path**: `document/4-Web_Application_Security_Testing/<NN>-<Category>/<NN>-<TestName>.md`.
144 markdown test files across categories (Information_Gathering,
Configuration_and_Deployment_Management_Testing, Identity_Management_Testing,
Authentication_Testing, Authorization_Testing, Session_Management_Testing,
Input_Validation_Testing, Error_Handling, Cryptography, Business_Logic_Testing,
Client_Side_Testing, API_Testing). Non-test front matter (Foreword, About,
Introduction, Testing Framework) lives under `document/0-Foreword/` through
`document/3-The_OWASP_Testing_Framework/` — skip these, no test IDs there.
Each category directory also has an `images/` subdirectory (skip, binary)
and its own `README.md` (category intro, skip — not a control).

**Format**: markdown, one file per test.

**Delimiter**: one file per test case. Each file opens with an H1 title,
then an ID table, then `## Summary`, `## Test Objectives`, `## How to Test`,
`## Related Test Cases`, `## Remediation`, `## Tools`, `## References`
sections (not every file has every section).

**Field mapping**:
- `control_id` = the value in the two-row markdown table right under the H1:
  ```
  |ID          |
  |------------|
  |WSTG-INFO-01|
  ```
  This exact string (`WSTG-INFO-01`, `WSTG-CRYP-04`, etc.) is
  `hub_links_by_framework.json["wstg"][i]["section_id"]` **and**
  `section_name` verbatim — OpenCRE uses the test ID as both fields for this
  framework, there is no separate human title captured on the link side (the
  real human title is the file's H1, e.g. "Conduct Search Engine Discovery
  Reconnaissance for Information Leakage" — richer than what OpenCRE carries).
- `title` = H1 heading text (strip the leading `# `).
- `description` = the `## Summary` section body (text between `## Summary`
  and the next `##` heading).

**Verbatim sample** (`document/4-Web_Application_Security_Testing/01-Information_Gathering/01-Conduct_Search_Engine_Discovery_Reconnaissance_for_Information_Leakage.md`):
```markdown
# Conduct Search Engine Discovery Reconnaissance for Information Leakage

|ID          |
|------------|
|WSTG-INFO-01|

## Summary

In order for search engines to work, computer programs (or `robots`) regularly
fetch data (referred to as [crawling](https://en.wikipedia.org/wiki/Web_crawler))
from billions of pages on the web. ...
```

**Count**: 144 test markdown files under `document/4-.../`. OpenCRE's wstg
links reference 118 link rows. Distinct WSTG-XXXX-NN ids referenced were not
separately counted here but the id space is a subset of the 144 files by
construction (OpenCRE's ids are a controlled vocabulary matching WSTG's own
numbering).

**Parsing difficulty**: low. Only real gotcha: category folder names are
zero-padded two-digit prefixes (`01-Information_Gathering`) but the file's
own ID prefix (`WSTG-INFO-01`) uses a four-letter mnemonic, not the folder
number — do not derive the ID from the path, read it from the table.

---

## samm — OWASP SAMM (Software Assurance Maturity Model)

**File**: `samm_core.zip` (306,681 bytes), GitHub archive of
`owaspsamm/core` pinned to commit `bc2b5474ab248effbc357c389bec372b0f5e200f`.

**Branch deviation**: the assignment said branch `master`. `owaspsamm/core`
has no `master` branch — `git ls-remote` / the GitHub branches API list
`develop, main, feature-activity-dependencies, markdown/*, release/*` but no
`master`. The repository's own `default_branch` per the GitHub API is
`develop`, which is what this is pinned to. Flagged in the Source's `note`
field in the fetch script.

**Relevant inner path**: `model/streams/<Practice>-<Stream>.yml` — **this is
the OpenCRE join level**, not `model/activities/`.

The repo has three granularities and it matters which one is used:
- `model/security_practices/<D|G|I|V>-<Name>.yml` (15 files) — the 15
  top-level Practices (e.g. `D-Secure-Architecture.yml`).
- `model/streams/<PP>-<S>-<L>.yml` where PP=2-letter practice code, S=stream
  letter (A/B), L is absent here — **30 files** (15 practices × 2 streams).
  Filename stem exactly matches OpenCRE's section_id (e.g. `D-SA-A`).
- `model/activities/<PP>-<S>-<Level>-<Stream>.yml` — 90 files, one per
  (practice, stream, maturity level 1-3) triple, e.g. `D-SA-1-A.yml`,
  `D-SA-2-A.yml`, `D-SA-3-A.yml`. These carry the richest prose
  (`longDescription`, multi-paragraph) but their filenames do **not** match
  any OpenCRE section_id directly — `D-SA-1-A` is not `D-SA-A`.

**Format**: YAML, one file per record, all types share `id`/`title-or-name`/
`description` shaped fields but with different key names per type (streams
use `name`+`description`; activities use `title`+`shortDescription`+
`longDescription`).

**Field mapping** (for the `model/streams/` join level):
- `control_id` = filename stem, e.g. `D-SA-A`. Confirmed identical to
  `hub_links_by_framework.json["samm"][i]["section_id"]`.
- `title` = the `name:` field (e.g. `Architecture Design`). Confirmed
  identical to `section_name` in the link data.
- `description` = the `description:` field — a single paragraph, fairly
  short (2-3 sentences). For richer prose, join to the three
  `model/activities/<control_id-with-level-inserted>.yml` files via their
  `stream:` GUID field (which equals the stream's own `id:` GUID) and
  concatenate `longDescription` from levels 1-3.

**Verbatim sample** (`model/streams/D-SA-A.yml`):
```yaml
practice: 4753e55e943c4d418303bf90d599c6b1
id: 253b012094cf4e0988e08fd22609227d
name: Architecture Design
letter: A
description: The design of a software architecture can significantly impact
  the security posture of software, and the use of good security practices
  will improve the overall design.
order: 1
type: Stream
```

**Count**: 30 stream files. OpenCRE's samm links: 30 link rows (1:1, no
duplication observed in the sample checked).

**Parsing difficulty**: low once the three-granularity trap is known. The
GUIDs (`id:` fields) are internal cross-references between the three file
types (stream -> practice, activity -> stream) and are NOT what OpenCRE
links against — do not use them as control_id.

---

## owasp_top10_2021 — OWASP Top 10 2021

**File**: `owasp_top10_2021.zip` (196,415,531 bytes — the largest source in
this batch by a wide margin), GitHub archive of `OWASP/Top10` pinned to
commit `66ebc4798d2ca72973967a20264bdeb70dcf0a13` (branch `master`).

**Why it's 196 MB**: the repo carries every historical Top 10 edition
(2003, 2004, 2007, 2010, 2013, 2017, 2021, 2025) and every language
translation of each, plus PDFs, PPTX decks, and image assets, under
`archives/`. Only `2021/docs/en/` is relevant to `owasp_top10_2021`.

**Relevant inner path**: `2021/docs/en/A0{1..9}_2021-<Name>.md` and
`2021/docs/en/A10_2021-Server-Side_Request_Forgery_(SSRF).md` — 10 files, one
per category. (Three more `A00_2021-*.md` files are intro/meta content, not
categories — skip.) Do not read `2021/docs/<other-lang>/` or any other year
directory.

**Format**: markdown, one file per Top 10 category.

**Delimiter**: one file per category, H1 title includes the code
(`# A01:2021 – Broken Access Control`), then `## Factors` (a markdown stats
table — CWEs mapped, incidence rate, etc., not control text), `## Overview`,
`## Description`, `## How to Prevent`, `## Example Attack Scenarios`,
`## References`, `## List of Mapped CWEs`.

**Field mapping**:
- `control_id` = the leading `A0N` / `A10` token in the filename and in the
  H1 (e.g. `A01`). Matches `hub_links_by_framework.json["owasp_top10_2021"]`
  `section_id` exactly (values seen: `A01`...`A10`).
- `title` = H1 text after the colon/en-dash (e.g. `Broken Access Control`).
  Note: `section_name` in the link data sometimes differs cosmetically from
  the file's own title — e.g. link data has `"Broken Access Controls"`
  (plural) and `"Logging and Monitoring Failures"` where the file's actual
  title is `"Security Logging and Monitoring Failures"`. Prefer the file's
  own title over `section_name`, per project policy on preferring full
  source prose over any secondary label.
- `description` = `## Description` section body. `REMEDIATION_HEADINGS` in
  `tract/config.py` already lists `"How to Prevent"`, `"Example Attack
  Scenarios"`, `"Example Attack Scenario"` as cut points — this framework is
  exactly why those exist; both headings appear verbatim in every category
  file here.

**Verbatim sample** (`2021/docs/en/A01_2021-Broken_Access_Control.md`, opening):
```markdown
# A01:2021 – Broken Access Control    ![icon](assets/...){: style="height:80px;width:80px" align="right"}

## Factors

| CWEs Mapped | Max Incidence Rate | ... |
|:-----------:|:-------------------:|...|
| 34          | 55.97%              | ... |

## Overview

Moving up from the fifth position, 94% of applications were tested for
some form of broken access control with the average incidence rate of
3.81%, and has the most occurrences in the contributed dataset with over
318k. ...

## Description

Access control enforces policy such that users cannot act outside of
their intended permissions. Failures typically lead to unauthorized
information disclosure, modification, or destruction of all data or
performing a business function outside the user's limits. ...
```

**Count**: 10 real categories (A01-A10). OpenCRE's link rows: 17 (some
categories linked more than once — A07, A08, A09 each appear twice in the
sample checked).

**Parsing difficulty**: low for the target content, but the archive is huge
and a naive "extract everything" will pull ~196 MB of irrelevant translations
and historical editions. Extract only `2021/docs/en/A0*` and `2021/docs/en/A10*`
members from the zip by name; do not extract the whole archive.

---

## owasp_proactive_controls — OWASP Proactive Controls

**File**: `owasp_proactive_controls.zip` (24,237,020 bytes), GitHub archive
of `OWASP/www-project-proactive-controls` pinned to commit
`4f5cb1081b4253bbccb314ef7855a1430fec8571` (branch `master`).

**Relevant inner path**: `docs/the-top-10/c{1..10}-<slug>.md` — 10 files,
this is the current (post-restructure) mkdocs site content, distinct from
`docs/archive/2018/c{1..10}-*.md` (an older v3 snapshot with different
prose — do not use) and from the `v3/` directory at repo root (binary
PDF/DOCX/PPTX exports of the 2018 edition in multiple languages — skip
entirely, ~24 MB of the 24 MB total is these binaries).

**Format**: markdown, one file per control.

**Delimiter**: one file per control. `# C{n}: <Title>` H1, then
`## Description`, `## Threats`, `## Implementation` (with `### N) <subheading>`
sub-sections).

**Field mapping**:
- `control_id` = the `Cn` token in the filename and H1 (e.g. `C1`). Matches
  `hub_links_by_framework.json["owasp_proactive_controls"]["section_id"]`
  exactly (`C1` through `C10`).
- `title` = H1 text after the colon (e.g. `Implement Access Control`).
- `description` = `## Description` section body.

**Verbatim sample** (`docs/the-top-10/c1-accesscontrol.md`, opening):
```markdown
# C1: Implement Access Control

## Description

Access Control (or Authorization) is allowing or denying specific requests
from a user, program, or process. With each access control decision, a
given subject requests access to a given object. Access control is the
process that considers the defined policy and determines if a given
subject is allowed to access a given object.

Access control also involves the act of granting and revoking those privileges.
...
```

**Count**: 10 controls (C1-C10). OpenCRE's link rows: 76.

**Parsing difficulty**: low for the target files, but two decoys exist in
the same archive (`docs/archive/2018/c*.md` and `v3/*`) with the same
control numbering and could be mistaken for the current version if a parser
globs `**/c[0-9]*-*.md` instead of anchoring to `docs/the-top-10/`.

---

## enisa — ENISA Securing Machine Learning Algorithms

**File**: `enisa_securing_ml_algorithms.pdf` (2,225,507 bytes, matches the
task's stated size exactly), single PDF, 70 pages, published December 2021.
Not from GitHub, no commit pin applicable — the URL is a direct file link on
enisa.europa.eu and was verified byte-stable across two independent fetches.

**Structure**: prose report, not a flat control list. Section 4
("SECURITY CONTROLS") states "we ... came up with a list of **37 security
controls**" and presents them in **Table 5** (pages 18-25), grouped into
three categories (`ORGANISATIONAL`, `TECHNICAL`, `SPECIFIC TO ML`). The same
37 controls reappear, in a *different* (alphabetical) order, in **Annex C**
("IMPLEMENTING SECURITY CONTROLS", pages 38+) with an "Examples for
operational implementation" column and a References column, and again in
**Annex B** (threat-to-control mapping, not control text).

**Why this is hard**: Table 5 and Annex C are genuinely multi-column PDF
tables with rotated/vertical lifecycle-stage column headers ("Data
Collection", "Data Cleaning", ... 10 stages). `pdfplumber.extract_text()`
interleaves the columns into garbled single lines (rotated headers come out
as reversed character sequences, e.g. `no itcelloC ataD` = "Data Collection"
read backwards). `pdfplumber.extract_table()` / `.extract_tables()` is
required, not plain text extraction, and even then the merged/spanning
cells (one control's definition spans several visual rows) come back
None-padded and need a merge step.

**No stable control ID exists in the source at all.** Controls are
identified only by their free-text name (e.g. "Apply a RBAC model,
respecting the least privilege principle"). This is exactly why OpenCRE's
own extraction is degraded for this framework:
`hub_links_by_framework.json["enisa"]` has `section_id == "Table 5:"` (a
literal, non-unique placeholder) for the majority of rows, with the real
control name only surviving in `section_name`. A handful of rows have
`section_id == section_name` (both set to the control's own name — a
slightly better OpenCRE extraction outcome for those specific rows, but
still not a stable numeric ID). **A parser must join on control NAME text
(fuzzy/exact string match against the 37 names), not on section_id.**

**Field mapping**:
- `control_id`: none in the source; synthesize one (e.g. slugify the control
  name, or number 1-37 in Table 5's own order) since none exists upstream.
- `title` = the short control-name phrase (left column of Table 5 / Annex C).
- `description` = the longer definition paragraph from Table 5, optionally
  concatenated with the matching "Examples for operational implementation"
  text from Annex C (joined by matching the control name string across both
  tables, since no id links them either).

**Verbatim sample** (from Table 5, page 18, one control's Table 5 entry and
its Annex C counterpart):
```
Table 5 (definition):
"Apply a RBAC (Role Based Access Control) model, respecting the least
privileged principle" — Define access rights management using a RBAC
model respecting the least privileged principle. This should cover all
components of the ML model (e.g. host infrastructures) and allow ...
It is notable that the roles to be included also concern the end user.
For example: the end user who can submit inputs to the model should not
be able to have access to its configuration.

Annex C (implementation guidance, same control):
Apply a RBAC model, respecting the least privilege principle
The NIST 800-53 and the ISO 27001/2 provides several points:
- Manage access permissions and authorisations, incorporating the
  principles of least privilege and separation of duties
- Manage the identity of the users (Couple lifecycle management
  processes and procurement processes etc.)
References: ISO 27001/2, NIST 800-53, 162
```

**Count**: 37 controls (explicitly stated in the source text). OpenCRE's
link rows: 68 (many controls linked to multiple CREs; some rows share the
generic `"Table 5:"` id).

**Parsing difficulty**: HIGH. Needs `pdfplumber.extract_tables()` (not
`extract_text()`), a rowspan-merge step, and name-string matching instead of
ID matching since no stable ID exists upstream. Expect this to be the
hardest of the 10 to parse cleanly.

---

## nist_ssdf — NIST SP 800-218 (Secure Software Development Framework)

**File**: `nist_sp_800_218.pdf` (739,891 bytes), single PDF, 36 pages.
Direct nvlpubs.nist.gov link, verified byte-stable across two fetches
despite being served from a Cloudflare-fronted domain (Cloudflare's
challenge-platform script injection, unlike `pages.nist.gov`, did not
appear in this PDF response — see the nist_800_63 entry below for the case
where it does matter).

**Structure**: one large nested table spanning pages ~13-33: columns
`Practices | Tasks | Notional Implementation Examples | References`. Four
top-level Practice groups: `PO` (Prepare the Organization), `PS` (Protect
Software), `PW` (Produce Well-Secured Software), `RV` (Respond to
Vulnerabilities). Each Practice (e.g. `PO.1`) contains several numbered
Tasks (`PO.1.1`, `PO.1.2`, ...).

**Why plain text extraction is unreliable**: same problem class as ENISA —
`extract_text()` interleaves the Task description and its "Example 1: ..."
column into one run-on line (e.g. `"PO.1.2: Identify and document all
security Example 1: Define policies that specify risk-based software
architecture..."` — the task description is truncated mid-sentence by the
adjacent column's text). `pdfplumber.extract_tables()` on the affected
pages recovers real columns (verified: page 13 yields a 12-column,
64-row table object) but the source table has vertically-merged cells (one
Practice-group cell spans many Task rows), so extracted rows need a
forward-fill / rowspan-merge pass — the same pattern as ENISA's Table 5.

**Field mapping**:
- `control_id` = the `XX.N.N` Task ID (e.g. `PO.1.2`, `PW.4.4`). Matches
  `hub_links_by_framework.json["nist_ssdf"]["section_id"]` exactly for the
  well-formed rows.
- `title`/`description`: OpenCRE's own `section_name` for this framework IS
  the Task description text itself (there is no separate short title in the
  source — the Task statement doubles as both).
- Some Tasks have been renumbered/retired in-place, e.g. `PW.3.2: Moved to
  PW.4.4` and `PW.4.3: Moved to PW.1.3` appear as their own rows — a parser
  must either skip these stub rows or resolve the redirect.

**One malformed OpenCRE row observed**: a `hub_links_by_framework.json`
entry has `section_id` set to a mid-sentence text fragment
(`'code, executable code, and configuration-as-code – based on the
principle of least privilege so that only authorized personnel, tools,
services, etc. have access.'`) rather than a valid `PS.1.1`-style id — this
is an upstream OpenCRE extraction artifact (the real id is `PS.1.1`,
recoverable from the immediately preceding row's split description), not
something this fetch can fix. Flag rather than silently drop.

**Verbatim sample** (table cell content, page 13, `PO.1.1`):
```
Practice group cell (spans multiple rows): "Define Security Requirements
for Software Development (PO.1): Ensure that security requirements for
software development are known at all times so that they can be taken
into account throughout the SDLC ..."

Task cell: "PO.1.1: Identify and document all security requirements for
the organization's software development infrastructures and processes,
and maintain the requirements over time."

Notional Implementation Examples cell: "Example 1: Define policies for
securing software development infrastructures and their components,
including development endpoints, throughout the SDLC and maintaining
that security. Example 2: ... Example 3: ... Example 4: ..."

References cell: "BSAFSS: SM.3, DE.1, IA.1, IA.2" / "BSIMM: CP1.1, CP1.3,
SR1.1, SR2.2, SE1.2, SE2.6" / "EO14028: 4e(ix)" / "IEC62443: SM-7, SM-9"
```

**Count**: NIST SP 800-218 defines ~42 Tasks across 4 Practice groups (not
independently recounted cell-by-cell here; the `XX.N.N` ids visible in a
plain-text grep span PO.1.1 through RV.x.x). OpenCRE's link rows: 46.

**Parsing difficulty**: MEDIUM-HIGH. Needs `extract_tables()` plus a
rowspan-merge pass, same as ENISA. Unlike ENISA, a real stable ID (`XX.N.N`)
exists in-source, so the join key problem ENISA has does not apply here.

---

## nist_800_63 — NIST SP 800-63 (Digital Identity Guidelines)

**File**: `sp800_63b.html` (353,757 bytes), single HTML page from
`pages.nist.gov/800-63-4/sp800-63b.html`.

**CRITICAL FINDING — version mismatch, not a parsing difficulty but a
correctness blocker**: the fetched document is **NIST SP 800-63-4**
(finalized 2025; the page's own references section cites itself as
`NIST SP 800-63-4`, DOI `10.6028/NIST.SP.800-63-4`, and its changelog says
verbatim *"Changes the name of memorized secrets to passwords"*).
`hub_links_by_framework.json["nist_800_63"]`'s 79 `section_id` values
(`5.1.1.2`, `6.1.2.3`, `A.3`, `5.2.5`, etc.) are **SP 800-63-3B section
numbers** (the prior revision — e.g. old §5.1.1.2 "Memorized Secret
Verifiers" is well-known 800-63-3B numbering). **These numbers do not
exist anywhere in the fetched 800-63-4 document** — confirmed by grepping
the raw HTML for every one of the sampled section numbers (`5.1.1.2`,
`6.1.2.3`) with zero hits. 800-63-4 restructured its content entirely: no
`data-section` heading carries a dotted multi-level number at all, headings
use slug `id` attributes instead (e.g. `id="passwordver"`,
`id="throttle"`) with only a single-integer `data-section="N"` chapter
attribute, and even the *concept* named at old §5.1.1.2 has been renamed
(memorized secret -> password) and almost certainly moved to a different
chapter (search hit for "Memorized Secret" only in the definitions/glossary
and changelog, not as a heading).

**Implication for "restores 79 dropped links"**: fetching *some* SP
800-63 document, as the URL literally instructed, does not by itself
restore those 79 links, because this specific document's structure does
not contain the identifiers the links reference. A parser has two honest
options: (a) fetch NIST SP 800-63-3B specifically (the archived prior
revision, which does carry §5.1.1.2-style numbering) instead of/in addition
to this file, or (b) treat this as a framework where OpenCRE's links need
re-derivation against the new structure rather than direct id matching.
Neither is a parsing-code fix; this needs a decision before a parser is
written, not during it.

**Deliberately unpinned**: `pages.nist.gov` sits behind Cloudflare, which
injects a per-response random bot-challenge token
(`window.__CF$cv$params={r:'<nonce>',t:'<base64 timestamp>'}`) into the
HTML `<body>` on every single fetch. Two fetches of this exact URL made
roughly a minute apart during this run produced two different sha256
hashes with a one-line diff confined to that injected script tag (see
`scripts/fetch_frameworks.py`'s Source note for `nist_800_63` — verified,
not assumed). `expected_sha256` was deliberately left `None` for this
source in the script; pinning it would make `--accept-new-hash` routine
rather than an alert.

**Format** (of the document as it actually exists, independent of the
numbering mismatch above): HTML5, semantic headings `<h1>`-`<h4>` each with
an `id` slug and a `data-section="N"` attribute giving only the top-level
chapter number.

**Count**: not independently recountable against OpenCRE's numbering since
the numbering schemes don't overlap; the document has on the order of 100+
headings across ~9 chapters.

**Parsing difficulty**: not a parsing-code problem — a framework-version
problem that blocks the join before parsing logic is relevant.

---

## etsi — ETSI GR SAI 005 (Securing AI Problem Statement)

**File**: `etsi_gr_sai005_v010101p.pdf` (1,035,313 bytes, matches task's
stated size exactly), single PDF, 31 pages, v1.1.1 (2021-03).

**Header requirement confirmed operationally**: requests against this URL
without the browser `User-Agent` header were NOT observed to 403 during
this run's testing window (both curl's default UA and Python `requests`'
default UA returned 200) — but the header is applied regardless per the
task's instruction, since ETSI's edge behavior toward non-browser UAs is
documented as inconsistent/load-balancer-dependent and the header is a
zero-cost mitigation. The Source carries the exact browser UA string given.

**Structure**: prose report organized as numbered sections (`5.1`, `5.2.2`,
`6.2.2`, `6.2.3`, `6.3.2`, `6.3.3`, `6.4.3`, ...) each covering one attack
class (poisoning, evasion, model stealing) and its mitigations. Within a
numbered subsection, individual mitigation **techniques are named only in
running prose** (bolded/emphasized lead phrases in bullet points or topic
sentences, e.g. "Adversarial example detections investigate input
samples...", "• Ensemble: for classifiers, ensemble methods are
proposed..."), not as their own markdown-style headings.

**Why this is hard, same shape as ENISA**: `hub_links_by_framework.json["etsi"]`
shows one numbered section_id shared by multiple distinct section_names —
e.g. `"6.2.3"` maps to three different techniques (`"Adversarial example
detection"`, `"Input restoration"`, `"Ensemble"`) in the sampled rows.
`section_id` alone does not uniquely identify a control; `section_name`
(the technique name) is the true differentiator, but that name is not a
structural anchor in the PDF — it's the first few words of a paragraph or
bullet. One row (`"Data sanitisation"`) has `section_id == section_name`,
same OpenCRE-extraction-fallback pattern seen in ENISA and BIML's unprefixed
ids.

**Field mapping**:
- `control_id` = numbered section (e.g. `6.2.3`) — coarse, shared by
  multiple techniques.
- `title` = technique name, must be extracted from prose (bullet lead
  phrase before the first colon, or the noun phrase starting a topic
  sentence) — no structural markup delimits it.
- `description`: two honest options — (a) use the entire numbered
  subsection's text for every technique sharing that section_id (accepts
  the coarse grain, same fallback pattern as ENISA's Table-5-bucket rows),
  or (b) attempt paragraph/bullet-level segmentation by technique name,
  which is fragile prose-heuristic work, not structural parsing.

**Verbatim sample** (section 6.2.3, excerpt):
```
6.2.3 Model-agnostic mitigations against evasion attacks
Model-agnostic mitigations aim at detecting adversarial examples, restoring
input samples or restoring model output. With the focus on inference data
samples, adversarial example detection and input restoration are two
approaches.
Adversarial example detections investigate input samples and tell if they
are manipulated. Here are some existing methods:
• Input transformation: for image classifiers, image transformations, such
  as rotation and shifting, are proposed to detect adversarial examples...
• Statistics: to detect adversarial examples, some research seek for proper
  statistics...
```

**Count**: not independently recountable as a flat list — the source has no
enumerated "N controls" statement the way ENISA does. OpenCRE's link rows:
36, spanning roughly a dozen distinct section numbers and ~20 distinct
technique names in the full set (not exhaustively enumerated here).

**Parsing difficulty**: HIGH, same class as ENISA — no clean per-control
delimiter exists in-source; the choice between coarse (subsection-level) and
fine (technique-level) grain has to be made deliberately, not discovered
during parsing.

---

## biml — Berryville Institute of Machine Learning

**Two files, both required — no single PDF covers OpenCRE's biml anchors:**

1. `ara.pdf` (848,055 bytes) — *An Architectural Risk Analysis of Machine
   Learning Systems: Toward More Secure Machine Learning*, v1.0, January
   2020, 42 pages. This is "BIML-78" (the document itself says "we
   identified 78 risks, referred to as the BIML-78").
2. `BIML-LLM24.pdf` (875,190 bytes) — *An Architectural Risk Analysis of
   Large Language Models: Applied Machine Learning Security*, v1.0, January
   24 2024, 28 pages, "81 LLM risks". This is "BIML-24(LLM)".

Both fetched from `berryvilleiml.com/results/*.pdf`, which 307-redirects to
`berryvilleiml.com/docs/*.pdf` — `requests`/curl follow this automatically
for GET, no special handling needed.

**Determination method** (task required evidence, not a guess): extracted
full text from both PDFs and located every `[category:number:name]` inline
tag both documents use throughout their own text (e.g. `[raw:3:storage]`,
`[model:2:Trojan]`) to identify which document defines which risk.

**Discrepancy from the task's stated figures**: the task described "14
links over 12 distinct anchors." The actual current content of
`data/training/hub_links_by_framework.json["biml"]` (also cross-checked
against `hub_links_curated.jsonl`, identical) has **21 link rows across 20
distinct section_ids and 17 distinct section_names** — reported as
measured, not reconciled to the task's figures, since I could not find a
21->14/20->12 reduction rule that produces the stated numbers.

**Anchor breakdown** (all 20 distinct section_ids, evidence-matched):
- 8 ids carry an explicit `"BIML-78(2020): "` prefix (e.g. `"BIML-78(2020):
  data:1"` -> *Data Poisoning*). All 8 confirmed present in `ara.pdf` by
  exact or near-exact label match on the bracketed tag (e.g.
  `[data:1:poisoning]`).
- 4 ids carry an explicit `"BIML-24(LLM): "` prefix. All 4 confirmed present
  in `BIML-LLM24.pdf` (e.g. `"BIML-24(LLM): raw:5"` -> *Data
  Confidentiality* matches `[raw:5:data confidentiality]` in that PDF
  exactly).
- 8 ids carry **no prefix** at all (e.g. `alg:11`, `inference:4`, `input:2`,
  `model:2`, `output:2`, `output:4`, `raw:3`, and `inference:9` appearing
  twice with different CRE targets). These are legacy links that predate
  OpenCRE's prefix convention (plausibly created before `BIML-LLM24.pdf`
  existed, since both documents reuse the same category vocabulary — `raw`,
  `data`, `alg`, `model`, `evaluation`, `inference`, `input`, `output`,
  `system` — and only need disambiguation once a second document exists).
  Content-matched by comparing each id's tag label in both PDFs:
  - `alg:11`, `inference:4`, `input:2`, `model:2`, `raw:3` — exact-label
    match ONLY in `ara.pdf` (e.g. `[alg:11:parameters]`, `[raw:3:storage]`);
    the same numeric id exists in `BIML-LLM24.pdf` too but with an unrelated
    label there (e.g. `BIML-LLM24.pdf`'s `raw:3` is "data feudalism", not
    "Storage") — confirms these 5 belong to `ara.pdf`.
  - `inference:9`, `output:4` — exact-label match ONLY in `BIML-LLM24.pdf`
    (`[inference:9:hosting]`, `[output:4:data confidentiality]`); `ara.pdf`
    has no `inference:9` tag at all, and its `output:4` is "inscrutability"
    (unrelated) — confirms these 2 belong to `BIML-LLM24.pdf`.
  - `output:2` ("Direct Output") — **ambiguous**, does not exact-match
    either document. `ara.pdf` has `[output:1:direct]` (name matches,
    number is off by one) and its own `output:2` is "provenance"
    (unrelated); `BIML-LLM24.pdf`'s `output:2` is "wrongness" (unrelated).
    Best guess is `ara.pdf` by name proximity, but flagged as not a clean
    match — this one row may need manual resolution or acceptance as
    "damaged" per `CONTROL_DAMAGED_METADATA_KEY` in `tract/config.py`.

**Net**: `ara.pdf` (BIML-78/2020) covers 8 prefixed + 6 confirmed-unprefixed
+ 1 probable = 15 of 20 anchors. `BIML-LLM24.pdf` (BIML-24/LLM) covers 4
prefixed + 2 confirmed-unprefixed = 6 of 20. (15+6=21 > 20 because
`output:2` is counted as probable-ara, not double-counted as confirmed.)

**Format**: both PDFs use identical internal structure — running prose with
inline `[category:number:label]` tags marking each named risk, organized
under top-level pipeline-stage headings (raw data, data sets/data,
algorithm/alg, model, evaluation, inference, input, output, system).

**Field mapping**:
- `control_id`: synthesize from the bracket tag (`category:number`, e.g.
  `raw:3`) plus which document it came from — this is exactly what OpenCRE's
  prefix convention does, and a parser should probably do the same rather
  than assume ids are globally unique across both documents (they are
  provably NOT, see `raw:5`, `raw:3`, `inference:9`, `output:2`,`output:4`
  collisions above).
- `title` = the bracket tag's trailing label, capitalized (e.g.
  `poisoning` -> matches OpenCRE's `"Data Poisoning"`).
- `description` = the paragraph(s) following the tag until the next
  `[category:number:label]` tag or heading.

**Verbatim sample** (`ara.pdf`, risk `[raw:3:storage]`):
```
[raw:3:storage]
As in [raw:3:storage], data may be stored and managed in an insecure
fashion. Who has access to the data pool, and why? Think about
[system:8:insider] when working on storage.
```

**Count**: `ara.pdf` states 78 risks (BIML-78); `BIML-LLM24.pdf` states 81
risks. OpenCRE only links a small fraction of either (20 distinct anchors
combined) — most of both documents' risk catalogs are unlinked.

**Parsing difficulty**: MEDIUM. The bracket-tag convention is a genuine
structural delimiter (better than ENISA/ETSI's prose-only approach), but
the cross-document id collision (same `category:number` meaning different
things in each document) means a parser MUST track document provenance as
part of the id, and the `output:2` anchor has no clean resolution.

---

## csa_ccm — Cloud Controls Matrix v4.1.0

**File**: `CCMv4.1.0-generated_at_2026_01_13.xlsx` (569,961 bytes),
locally-staged (registration-gated at cloudsecurityalliance.org, not
fetchable by script — confirmed already present with the exact sha256
given in the assignment: `5e721628c8ab297bdbd355afa4c01699971fcbb9cb16802ccb9d42c7176ab32b`).

**Not to be confused with csa_aicm** (AI Controls Matrix) per CLAUDE.md —
verified: this workbook's title cell literally reads `"CLOUD CONTROLS
MATRIX v4.1.0"`, a completely different framework from AICM.

**Format**: XLSX, 8 sheets: `Introduction`, `CCM`, `Implementation
Guidelines`, `Auditing Guidelines`, `Scope Applicability (Mappings)`,
`CAIQ`, `Acknowledgments`, `Change Log`. **Use the `CCM` sheet** — the
`Scope Applicability (Mappings)` sheet is a stub (`"This dataset is not
available yet"`); `CAIQ` (304 rows) duplicates `CCM`'s structure but is the
self-assessment questionnaire variant, not the controls themselves.

**Delimiter**: the `CCM` sheet is a flat 4-column table (`Control Domain |
Control Title | Control ID | Control Specification`), 229 rows total.
Two row types interleaved:
- **Domain header rows**: only column A populated, formatted as
  `"<Full Domain Name> - <SHORT CODE>"` (e.g. `"Cryptography, Encryption &
  Key Management - CEK"`). 17 of these (one per CCM domain), plus 2 junk
  trailer rows (`"End of Standard"` and a copyright notice paragraph — both
  also only-column-A, must be filtered same as domain rows but are not
  actual domains).
- **Control rows**: all 4 columns populated. 207 of these.

**Field mapping**:
- `control_id` = column C (e.g. `A&A-01`). Matches
  `hub_links_by_framework.json["csa_ccm"]["section_id"]` for the
  control-level links.
- `title` = column B.
- `description` = column D (`Control Specification`, multi-paragraph,
  `\n`-separated within the cell).
- **Domain-level links exist too**: `hub_links_by_framework.json["csa_ccm"]`
  includes rows where `section_id` is just the domain short code (`CEK`,
  `STA`, `A&A`, `GRC` observed in the sample) rather than a control id.
  These join against the domain-header row's column A by splitting on
  `" - "` and taking the suffix (e.g. `"Cryptography, Encryption & Key
  Management - CEK"` -> `CEK`). A parser needs both granularities, same
  problem class as SAMM (stream vs. activity) and DSOMM (subdimension vs.
  activity).

**Verbatim sample** (row 4, first real control):
```
Control Domain: Audit & Assurance
Control Title:  Audit and Assurance Policy and Procedures
Control ID:     A&A-01
Control Specification: Establish, document, approve, communicate, apply,
evaluate and maintain audit and assurance policies and procedures and
standards. Review and update the policies and procedures at least
annually, or upon significant changes.
```

**Count**: 207 control rows, 17 domains. OpenCRE's link rows: 29 (mix of
control-level and domain-level ids, per CLAUDE.md's "29 CRE links" figure —
confirmed by direct count of `hub_links_curated.jsonl` rows with
`framework_id == "csa_ccm"`).

**Parsing difficulty**: LOW. Cleanest source in this batch — a real
tabular format, no PDF extraction, no OCR-adjacent table-merging. Only
gotcha is filtering the 2 non-domain junk rows (`End of Standard`, the
copyright paragraph) out of the 19 only-column-A rows, and splitting the
domain name to get the short code.

---

## Summary table

| framework_id | file(s) | type | controls (source) | OpenCRE link rows | control_id source | parse difficulty |
|---|---|---|---|---|---|---|
| dsomm | dsomm_data.zip -> generated/model.yaml | YAML | 194 activities | 214 | `uuid` field | low |
| wstg | wstg.zip -> document/4-.../*.md | markdown | 144 files | 118 | ID table in each file | low |
| samm | samm_core.zip -> model/streams/*.yml | YAML | 30 streams | 30 | filename stem | low |
| owasp_top10_2021 | owasp_top10_2021.zip -> 2021/docs/en/A0*.md | markdown | 10 categories | 17 | filename/H1 prefix | low (huge archive, narrow extract) |
| owasp_proactive_controls | owasp_proactive_controls.zip -> docs/the-top-10/*.md | markdown | 10 controls | 76 | filename/H1 prefix | low (decoy dirs in same zip) |
| enisa | enisa_securing_ml_algorithms.pdf | PDF (70pp) | 37 controls, no stable id | 68 | none — synthesize | HIGH |
| nist_ssdf | nist_sp_800_218.pdf | PDF (36pp) | ~42 tasks | 46 | `XX.N.N` task id | medium-high |
| nist_800_63 | sp800_63b.html | HTML | n/a — wrong revision | 79 | version mismatch, blocks join | blocked, not a parse problem |
| etsi | etsi_gr_sai005_v010101p.pdf | PDF (31pp) | not enumerable | 36 | numbered section, coarse | HIGH |
| biml | ara.pdf + BIML-LLM24.pdf | PDF x2 (42pp+28pp) | 78 + 81 risks | 21 (20 distinct ids) | bracket tag, cross-doc collisions | medium |
| csa_ccm | CCMv4.1.0-*.xlsx | XLSX | 207 controls, 17 domains | 29 | `Control ID` column | low |
