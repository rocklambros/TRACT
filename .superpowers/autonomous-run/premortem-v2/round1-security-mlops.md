# Premortem v2 — round 1, Security Architect + MLOps/SRE lens
Agent: a732bf8a3391a4246. Artifact: docs/superpowers/plans/2026-08-18-remaining-parsers-v2.md
Returned 10 findings. Orchestrator verification below each.

## S1 Licence CLASS vs licence STATUS (High)
RESTRICTED_FRAMEWORK_IDS = {etsi, iso_27001} models "publisher forbids reproduction".
It does not model copyleft. Of the 11 frameworks v2 adds: dsomm=GPL-3.0-only;
biml=CC-BY-SA-3.0 AND 4.0; samm/wstg/owasp_top10_2021/owasp_proactive_controls=CC-BY-SA-4.0.
Today those are title stubs (dsomm description len min/med/max = 5/21/32). Task 3 replaces
them with 182 statement anchors inside a TRACKED, CC0-declared artifact.
VERIFIED: grep -c "CC-BY-SA" plan = 0; grep -c "GPL-3.0" plan = 0.
Counter the agent itself raised: CC-BY-SA is satisfied by attribution + notice, which NOTICE
supplies. That narrows it to GPL-3.0 (dsomm) alone. One is enough.

## S2 Acceptance suite will red CI, and the cheap repair kills the licensed-text gate (High)
merged_corpus_path() prefers the gitignored overlay. Tracked corpus = 29 frameworks,
iso_27001:False etsi:False. Overlay = 31. In CI there is no overlay, so the fixture does
NOT skip; it builds a report where etsi/iso resolve zero, and the plan's
assert check_join_floors(...) == [] fails against "etsi": 1.00.
ci.yml runs pytest -x; test_corpus_acceptance sorts before test_licensed_text_not_tracked.
VERIFIED: etsi.json and iso_27001.json are the ONLY two untracked per-framework files.

## S3 CCM XLSX parsed with openpyxl DEFUSEDXML=False; defusedxml install deferred to Task 15 (Med-High)
VERIFIED: defusedxml absent from the 3.12 env (ModuleNotFoundError). openpyxl gates its own
hardening on that import. Task 8 parses a registration-walled, hand-staged workbook in the
window the plan itself creates. Also three-way pin drift: ==3.1.5 in requirements.txt,
bare in requirements-lint.txt, >=3.1.5 in pyproject.toml, and ci.yml installs via -e .

## S4 Eleven external sources gated on transport only; two not even that (Med)
5 GitHub sources pinned to commit SHA + expected_sha256 — proves bytes did not change in
transit, says nothing about content authenticity. nist_800_63 deliberately unpinned
(Cloudflare bot token); etsi fetched with a spoofed browser UA. --accept-new-hash is an
alert with no stated adjudication rule.

## S5 Two of three TestSpecAcceptance checks glob files that are gitignored (Med)
The two frameworks under the strictest licence get the LEAST checking. ETSI's
min_prose_fraction=1.0 is enforced on exactly one laptop.

## M1 No backup before the one-way copy; 3 artifacts unrecoverable (CRITICAL)
pre_rebuild_control_hashes.json schema = [generated_from, n_controls, sha256_of_description].
Digests only, zero text — a detector, not a rollback artifact. Task 15's only mutation is
shutil.copy2 over PROCESSED_FRAMEWORKS_DIR with no snapshot and no --restore.
For 29 frameworks git checkout recovers. For etsi.json, iso_27001.json and
licensed/all_controls.json it does not, and fetch_frameworks.py has NO iso_27001 entry,
so ISO is not re-derivable from any scripted path. ISO is the corpus's only 0.967-prose fold.

## M2 BEFORE artifact cannot be committed (High) — MECHANISM CORRECTED
Agent claimed git add is atomic so the whole Task 1 commit is empty. REPRODUCED AND FALSE:
  git add real.py results/corpus/before.json  ->  EXIT=1, staged: real.py
git stages the legal paths and refuses only the ignored one. So corpus_report.py IS committed.
What is true and still load-bearing: (a) exit 1 aborts any runner that checks exit codes,
mid-task, after a partial add; (b) the BEFORE artifact is never committed, so
  if not BEFORE.exists(): pytest.skip(...)
skips forever and the 20-untouched-framework regression test reports green having run nothing.
(c) BEFORE path is relative, not REPO_ROOT-anchored, unlike the rest of the suite.
VERIFIED: .gitignore:3 = results/

## M3 CAPEC+CWE never test-rebuilt; 57.3% of the training graph (High)
Plan pre-measures 19 of 21 parsers = 1,897 of 4,222 controls = 45%. The 2 excluded are
capec (42.8% of the graph) and cwe (14.5%), both blocked on the same missing defusedxml.
Task 15 installs it and immediately runs them for the first time ever in this env.
defusedxml is a REJECTING parser: a DOCTYPE raises rather than parses.

## M4 The mandatory `invalidates` column is absent; stopwords.json goes stale (High)
VERIFIED: grep -c invalidates plan = 0; build_stopwords = 0; stopwords = 0.
RUN-LEDGER.md:93 "The invalidates column is now mandatory." Spec:945 "no step may precede
anything that rewrites its inputs." The rebuild adds ~2,300 controls of new prose;
stopwords.json is derived from the corpus, committed, and applied to every control and hub
text. orchestrate.py hashes the stale file into run metadata as if current.
VERIFIED 13 consumers incl. tract/training/data.py, firewall.py, orchestrate.py,
text_selection.py, phase1b/run_fold.py.
This is lesson-6 recurring a FOURTH time.

## M5 Task 16 = 3 real gates + 6 that pass by construction; floors pre-registered in an invisible file (Med-High)
Real: dsomm anti-collapse pair, biml distinct_anchors==20/by_title==1, etsi by_title==2.
Vacuous: `assert floor <= 1.0` (tautological); `wrong_anchor_risk == 0` (counter only
increments in the title branch, and 9 of 11 are engineered to resolve via id, so unfailable);
`honest_prose_fraction(controls) > 0.0` (ONE prose control in csa_ccm's 224 = 0.0045 > 0 PASS)
— compares against zero instead of each parser's declared min_prose_fraction.
VERIFIED: plan is gitignored (.gitignore:25), git log on it is empty. A floor edited down
mid-run leaves no diff. Reproduces `gate-preregistration-is-retrospective` in a STRICTER
form: criterion and PASS can land in ZERO commits.

---

# Orchestrator resolutions taken during round 1

## S3 CLOSED by installation (not by plan amendment)
`pip install defusedxml==0.7.1` into the 3.12 env — this is the pin ALREADY declared in
requirements.txt, so it brings the env into compliance rather than adding a dependency.
  BEFORE: openpyxl DEFUSEDXML = False
  AFTER:  openpyxl DEFUSEDXML = True  | LXML = True
The CCM workbook will now be parsed with entity hardening ON. Task 15 Step 1's install
becomes a no-op; Task 8 no longer opens an unhardened window.
RESIDUAL, still a plan amendment: three-way pin drift stands —
requirements.txt `openpyxl==3.1.5`, requirements-lint.txt bare, pyproject `>=3.1.5`,
and ci.yml installs via `-e .` so CI resolves the FLOOR, not the pin. Task 8 must pin
pyproject to ==3.1.5. Same defect shape for defusedxml: pyproject says `>=0.7.1`.

## M3 CLOSED by measurement (the measurement the plan omitted)
Ran both parsers into a scratch dir and diffed against the 4,222-hash baseline:
  capec: REBUILT n=558  match=558  mismatch=0 new=0   (honest prose 0.996)
  cwe:   REBUILT n=1331 match=1331 mismatch=0 new=0   (honest prose 0.992)
defusedxml does NOT reject either XML. 1,889 of 1,889 reproduce byte-identically.
Pre-measured rebuild coverage rises 1,897/4,222 (45%) -> 3,786/4,222 (89.7%).
The finding was a genuine hole in the plan's evidence; the risk behind it is now retired
BEFORE eleven parsers get written, which is the whole point of running the premortem early.

## M2 mechanism corrected — see above. Fix is NOT `git add -f`.
Forcing an ignored artifact into git is how licensed text escaped before. Correct fix:
the BEFORE/AFTER corpus reports are evidence, not results — write them to a path that is
tracked by design, anchor it to REPO_ROOT, and drop the pytest.skip that turns a missing
baseline into a green test.

---

# Rulings (orchestrator, unattended run)

## R4 — Three licence tiers, not two. Conditional text defaults to the overlay.
VERIFIED FRAMEWORK_LICENSES:
  dsomm                    GPL-3.0-only
  biml                     CC-BY-SA-3.0 AND CC-BY-SA-4.0
  samm/wstg/owasp_top10_2021/owasp_proactive_controls   CC-BY-SA-4.0
  csa_ccm    "Proprietary. (c) CSA, all rights reserved. ... no redistribution"
  dsomm today: 183 controls, description len min/med/max = 5/21/32, ZERO over 200 chars.
  Task 3 would replace those stubs with 182 statement anchors in a TRACKED CC0 artifact.

Ruling: the binary RESTRICTED set becomes three tiers.
  RESTRICTED   (no reproduction at all):  etsi, iso_27001
  CONDITIONAL  (reproduction permitted, but attaches terms CC0 cannot carry):
               dsomm, biml, samm, wstg, owasp_top10_2021, owasp_proactive_controls, csa_ccm
  PERMISSIVE   everything else
CONDITIONAL text lives in the gitignored overlay exactly as ISO does. ASSIGNMENTS
(control_id -> hub) stay tracked and published: a mapping is a fact about two documents,
not a reproduction of either. Training reads the overlay, so model quality is untouched —
this costs zero anchors.

Why: the mechanism already exists and is tested (overlay + fingerprint gate + ISO proved it
end to end). Reversibility decides the default. Overlay -> tracked is a one-constant change.
Tracked -> published on HuggingFace is NOT reversible; a CC0 grant is an affirmative
assertion that the publisher holds the rights and waives them, which is false for GPL text.
Cost if wrong: the public corpus ships assignments without prose for 7 frameworks, and a
future owner flips a constant to change that.

## R5 — csa_ccm goes in CONDITIONAL despite the standing ruling, and this is the item to
## review first on return.
The owner ruled "we can redistribute csa ccm. don't stop to ask me" and reaffirmed it. That
ruling is honored in the sense that it does not block: csa_ccm is parsed, its 224 controls
train the model, and its assignments publish. What I am NOT doing unattended is writing
"all rights reserved, no redistribution" text into a CC0-declared file and pushing it to
HuggingFace while the owner is away. The owner may hold a CSA agreement I cannot see; if so
this is one constant to move and nothing else changes.
Cost if wrong: csa_ccm prose is absent from the public corpus until someone moves it.

## R6 — DSOMM/OWASP/BIML were never ruled on by anyone. R4 is the first ruling on them.
Distinct from R5: the owner has never seen the GPL-3.0 fact. Not a reversal of any decision.
