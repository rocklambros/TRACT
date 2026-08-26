"""What the eleven parsers had to be true for, expressed as a gate.

The instrument is tract.corpus_report, the same one every per-parser step used,
so a parser cannot be accepted by a measurement its consumer does not perform.

Three rules govern every assertion in this file.

**Rule 1, state the attainable range in both directions.** The suite this
replaces had nine terminal assertions and six could only ever return one value:
`floor <= 1.0` against literals three lines above, `wrong_anchor_risk == 0` on
frameworks engineered to resolve entirely through the id channel where the
counter increments only in the title branch, and `honest_prose_fraction > 0.0`
against a ratio, where one prose control in csa_ccm's 224 gives 0.0045 and
passes. A gate that cannot fail reports green having measured nothing. Every
assertion below carries a comment naming what it can read at the top and at the
bottom of its range, and the tautological half of the v2 floor check is gone
rather than restated: `0.0 < floor <= 1.0` against a dict of literals guards a
floor that is unreachably high and says nothing at all about a floor that cannot
be missed. `test_no_floor_leaves_more_than_one_percent_of_its_links_spendable`
is the half that was missing.

**Rule 2, no assertion may be silenced by a licence.** The tracked corpus holds
29 frameworks and the licensed overlay holds 31, `merged_corpus_path()` falls
back to the tracked file, and the tracked file always exists, so
`if not merged_corpus_path().exists(): pytest.skip(...)` never skips and four
assertions hard-failed in CI on text that cannot legally be in a fresh clone.
The predictable repair is deleting the ETSI floor, which retires the only gate
on a restricted parser nobody can inspect. Here the corpus-dependent tests admit
exactly two framework censuses, skip the overlay rows as a named group with the
reason stated, keep every other row asserting, and fail on any third census. All
eleven are asserted separately against the committed AFTER artifact, which is
tracked, carries counts and digests and no anchor text, and needs no corpus.

**Rule 3, separate a pre-registered criterion from a ratchet, and say which is
which.** `JOIN_FLOORS` and `JOIN_WRONG_ANCHOR_BUDGET` were derived from the
curated link file and the pinned source before any parser existed and committed
in Task 1, so a threshold cannot move in the same commit as the result it gates.
Those are criteria. `results/corpus/before.json` is a tracked measurement of the
corpus as it stood before this plan, so a comparison against it is also a
criterion. Everything else in this file that pins a measured value is a ratchet:
it stops a number moving without a human looking, and it certifies nothing about
whether the number is right. Each one says so at the point of use.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from tract.config import (
    OVERLAY_FRAMEWORK_IDS,
    PARSERS_DIR,
    PROCESSED_FRAMEWORKS_DIR,
    PROJECT_ROOT,
    RESTRICTED_FRAMEWORK_IDS,
)
from tract.corpus_report import (
    DETECTOR_B_INAPPLICABLE,
    FULL_CORPUS_FRAMEWORK_COUNT,
    JOIN_FLOORS,
    JOIN_WRONG_ANCHOR_BUDGET,
    TRACKED_CORPUS_FRAMEWORK_COUNT,
    CorpusReport,
    FrameworkJoin,
    build_corpus_report,
    check_join_floors,
    wrong_anchor_applicable,
)

# tract.config names the repository root PROJECT_ROOT. Every path below is
# anchored to it rather than to the working directory, because pytest can be
# invoked from anywhere and a relative path that misses turns an assertion into
# a skip.
REPO_ROOT: Path = PROJECT_ROOT

BEFORE_PATH: Path = REPO_ROOT / "results" / "corpus" / "before.json"
AFTER_PATH: Path = REPO_ROOT / "results" / "corpus" / "after_parsers.json"

# Rows that a fresh clone cannot assert, because their processed text routes to
# the gitignored overlay under CONDITIONAL_FRAMEWORK_IDS or
# RESTRICTED_FRAMEWORK_IDS. Named, not silent.
OVERLAY: frozenset[str] = OVERLAY_FRAMEWORK_IDS

_MIN_PROSE = re.compile(
    r"^\s*min_prose_fraction:\s*ClassVar\[float\]\s*=\s*([0-9]*\.?[0-9]+)\s*$",
    re.MULTILINE,
)

# A floor is a gate only if a plausible loss can cross it. This is the ceiling
# on how many resolved links a framework may lose before check_join_floors
# fires, as a fraction of its link count. At 1% the widest gap in the corpus is
# cwe, which may lose 5 of 613 links against an allowance of 6.
MAX_SPENDABLE_LINK_FRACTION: float = 0.01

# Anchors a parser assembled rather than copied, per framework, measured on the
# AFTER corpus. A RATCHET, not a criterion: it certifies nothing about whether
# synthesising the text was right, and it stops a parser starting to synthesise
# without a reviewer seeing it. `honest_prose_fraction` counts a synthetic
# statement as prose and no column separates it from a publisher's, so this is
# the only place the corpus says how much of its text the project wrote.
#
#   csa_ccm  14  domain aggregates built from member control titles
#   etsi      4  clause roll-ups
#   samm     30  every SAMM anchor, composed from shortDescription
#   wstg      1  one archived page rebuilt from its own headings
SYNTHETIC_ANCHOR_BUDGET: dict[str, int] = {
    "csa_ccm": 14,
    "etsi": 4,
    "samm": 30,
    "wstg": 1,
}

# Wrong-anchor flags outside JOIN_WRONG_ANCHOR_BUDGET on a framework this plan
# gave a parser to. EMPTY, and it is empty because the one entry it carried was
# repaired rather than lowered, which is what the entry's own note required.
#
# The entry was nist_ssdf at 44 of 44 applicable checks. MEASURED CAUSE:
# parse_nist_ssdf titles each task by its own id, so every control's title reads
# "PO.1.1", while the curated link file's section_name holds the full task
# statement. Detector B asks whether the link's name appears in the resolved
# control's title, and an identifier can never contain a sentence, so B fired on
# every id-channel link it reached. The anchors themselves were always correct:
# all 46 resolved links reach a full_text task statement.
#
# Ruling R19 closed it in the instrument, where it belonged. R11 and R21 both
# read distinct(section_id) / distinct(section_name), so both see GRANULARITY,
# and nist_ssdf reads exactly 1.0000 there. R19 added a third derived predicate
# on the KIND of label, median len(section_name) over median len(title), and
# nist_ssdf reads 26.08 against a threshold of 7.0. The row now reads 0 of 0:
# detectors A and C still run for it and neither has a candidate to reach.
#
# The mapping stays as the registration point for the next exposure. A new
# framework flagged outside the budget gets measured and recorded here with its
# cause, not assumed away.
UNBUDGETED_WRONG_ANCHOR_EXPOSURE: dict[str, int] = {}

# Parsers inheriting BaseParser.min_prose_fraction = 0.0, which no output can
# miss. A RATCHET at the count measured when this suite landed, so the number
# cannot grow. Raising the 19 needs a measured statement-length distribution per
# framework and is separate work, in flight elsewhere, so the comparison stays
# one-sided on purpose: a workstream driving the count down must not have to
# edit this file to stay green, and a new parser shipping without a floor must.
UNFLOORED_PARSER_RATCHET: int = 19


def _load(path: Path) -> dict[str, Any]:
    """A committed corpus report. A missing one is a failure, never a skip.

    The v2 suite guarded this read with `pytest.skip("no BEFORE artifact in
    this checkout")` while `.gitignore` excluded `results/` outright, so the
    skip would have fired on every machine forever and the only test protecting
    the untouched frameworks would have reported green having run nothing.
    """
    if not path.exists():
        raise AssertionError(
            f"{path.relative_to(REPO_ROOT)} is missing. It is committed "
            f"evidence, not an optional local file. Regenerate it with "
            f"scripts/corpus_report.py and confirm `git check-ignore` exits 1 "
            f"for it. Never `git add -f`."
        )
    report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return report


def _rows(report_json: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["framework_id"]: row for row in report_json["per_framework"]}


def _pending() -> tuple[str, ...]:
    """The frameworks this plan gave a parser to, read off the BEFORE artifact.

    Derived rather than listed. Every one of the eleven resolved 0 of its links
    before this plan, because none of them had a parser and the corpus held no
    text for them to join against. Reading the set out of tracked evidence means
    the suite cannot be pointed at a shorter list by editing this file, and a
    twelfth framework in the same state would join the gate automatically.

    JOIN_FLOORS is the wrong source: it now covers all 22 link-bearing
    frameworks, because the untouched eleven acquired regression floors, so
    `tuple(sorted(JOIN_FLOORS))` would put asvs and capec in the pending set.
    """
    return tuple(
        sorted(
            row["framework_id"]
            for row in _load(BEFORE_PATH)["per_framework"]
            if row["by_title"] + row["by_id"] == 0
        )
    )


PENDING: tuple[str, ...] = _pending()


def _parsed_framework_ids() -> frozenset[str]:
    """Framework ids with a parser module, which is the tracked census."""
    parsed = frozenset(
        path.stem[len("parse_"):] for path in PARSERS_DIR.glob("parse_*.py")
    )
    assert parsed, (
        f"no parse_*.py under {PARSERS_DIR}. The suite would otherwise derive "
        f"an expected census of zero and pass on an empty repository."
    )
    return parsed


def _on_disk_framework_ids() -> frozenset[str]:
    """Framework ids with processed output readable here."""
    if not PROCESSED_FRAMEWORKS_DIR.exists():
        return frozenset()
    return frozenset(path.stem for path in PROCESSED_FRAMEWORKS_DIR.glob("*.json"))


def _expected_framework_ids() -> frozenset[str]:
    """Framework ids the corpus must cover, from what this checkout can see.

    The union of the parser modules, what is on disk, and the overlay set. An
    overlay framework's per-framework JSON is absent from a fresh clone
    entirely, so the glob alone reads 28 where a checkout holding the overlay
    reads 32. The parser modules are in the union because a census taken only
    from the glob lets a framework leave every check below by having its
    processed file deleted, which is the hole the partition docstring claims to
    close. Derived rather than hard-coded so it does not rot the next time a
    framework lands.
    """
    on_disk = _on_disk_framework_ids()
    assert on_disk, (
        f"no framework JSON under {PROCESSED_FRAMEWORKS_DIR}. The suite would "
        f"otherwise derive an expected count of zero and pass on an empty "
        f"corpus."
    )
    return frozenset(_parsed_framework_ids() | on_disk | OVERLAY)


@pytest.fixture(scope="module")
def live() -> CorpusReport:
    """The report built from whatever corpus this checkout holds."""
    return build_corpus_report()


@pytest.fixture(scope="module")
def overlay_present(live: CorpusReport) -> bool:
    """Whether the licensed overlay is readable here.

    Three outcomes, not two. The full census and the tracked census are both
    legal. Anything else means the corpus is short by frameworks no licence
    explains, which is a red build rather than a skip.
    """
    count = live.corpus_framework_count
    if count == FULL_CORPUS_FRAMEWORK_COUNT:
        return True
    if count == TRACKED_CORPUS_FRAMEWORK_COUNT:
        return False
    raise AssertionError(
        f"the corpus reports {count} frameworks. Only "
        f"{FULL_CORPUS_FRAMEWORK_COUNT} (with the licensed overlay) and "
        f"{TRACKED_CORPUS_FRAMEWORK_COUNT} (a fresh clone or CI, without it) "
        f"are explainable. Any other census means frameworks are missing for a "
        f"reason that is not a licence, and skipping here would hide it. "
        f"Overlay set: {sorted(OVERLAY)}."
    )


def _assertable(overlay_present: bool) -> tuple[str, ...]:
    """The pending frameworks whose live rows can be asserted here."""
    if overlay_present:
        return PENDING
    return tuple(f for f in PENDING if f not in OVERLAY)


def _assertable_floors(overlay_present: bool) -> tuple[str, ...]:
    """Every framework carrying a floor, less the ones no licence permits here."""
    return tuple(
        f
        for f in sorted(JOIN_FLOORS)
        if overlay_present or f not in OVERLAY
    )


def _spendable_links(row: FrameworkJoin) -> int:
    """Resolved links this framework may lose before its floor fires.

    Mirrors check_join_floors' comparison, epsilon included, so the two agree on
    the boundary case rather than differing by a rounding decision.
    """
    floor = JOIN_FLOORS[row.framework_id]
    resolved = row.by_title + row.by_id
    spent = 0
    while spent < resolved and (
        (resolved - spent - 1) / row.links + 1e-9 >= floor
    ):
        spent += 1
    return spent


class TestTheSuiteCanActuallyRun:
    """Positive controls. Without these the file below can go quiet."""

    def test_the_before_artifact_names_the_eleven_frameworks_this_plan_parsed(
        self,
    ) -> None:
        # Attainable [0, 22], the number of link-bearing rows in the BEFORE
        # artifact. It reads 0 if the baseline is replaced by a post-parser
        # capture and 22 if it is replaced by an empty corpus, and both are the
        # failure that silently empties every loop below.
        assert len(PENDING) == 11, sorted(PENDING)
        # Every pending framework must carry a floor, or its parser is
        # ungated. Attainable [0, 11] missing.
        assert sorted(set(PENDING) - set(JOIN_FLOORS)) == []

    def test_the_two_corpus_censuses_differ_by_the_restricted_tier(self) -> None:
        """The overlay decision has to be readable from the census alone.

        Two of the three overlay frameworks are DROPPED from the tracked corpus
        and one is kept with its prose withheld, so the census gap is the
        restricted tier rather than the overlay. A change to either tier that
        did not move these constants would make the fixture above unable to
        tell a licence from a missing parser.
        """
        # Attainable: any integer. Reads 2 today and fails in both directions,
        # upward if a framework is dropped without joining the restricted tier
        # and downward if a restricted framework starts being tracked.
        assert FULL_CORPUS_FRAMEWORK_COUNT - TRACKED_CORPUS_FRAMEWORK_COUNT == len(
            RESTRICTED_FRAMEWORK_IDS
        )

    def test_the_assertable_set_is_eleven_locally_and_nine_in_ci(
        self, overlay_present: bool
    ) -> None:
        """CI must still gate something real.

        Two of the eleven route to the overlay -- dsomm and etsi -- so nine
        assert in a fresh clone. If a licence reclassification empties that
        set, this file measures nothing in CI and TestCommittedAfterReport
        becomes the only gate left.

        Nine since 2026-08-26, up from eight. csa_ccm left the overlay on owner
        decision D1(b), so its prose is tracked and CI can assert it like any
        other framework. A reclassification moved this number in the widening
        direction for once; the assertion is here so either direction has to be
        re-derived rather than absorbed.
        """
        # Attainable [0, 11]. Fails downward when a framework joins the overlay
        # and upward when one leaves it, and both changes need this number
        # re-derived rather than the gate quietly widening or narrowing.
        assert len(_assertable(overlay_present)) == (11 if overlay_present else 9)

    def test_the_silent_group_is_exactly_the_frameworks_the_licence_breaks(
        self, live: CorpusReport, overlay_present: bool
    ) -> None:
        """A framework may never join the silent group for any other reason.

        Comparing the skipped set against OVERLAY is a tautology, because
        `_assertable` is defined by subtracting OVERLAY. The check has to run
        against what the corpus actually does: a row is legitimately silent
        only when the missing overlay genuinely collapsed it, and a row that
        still resolves without the overlay has no business being skipped.
        """
        collapsed = {
            framework_id
            for framework_id in PENDING
            if live.by_id(framework_id).by_title + live.by_id(framework_id).by_id == 0
        }
        expected = set() if overlay_present else set(PENDING) & set(OVERLAY)
        # Attainable: any subset of the eleven. Reads empty locally and
        # {csa_ccm, dsomm, etsi} in CI. Fails upward when a framework collapses
        # for a reason no licence explains, and downward when a framework is
        # skipped under a licence it does not need.
        assert collapsed == expected, sorted(collapsed ^ expected)


class TestJoinFloors:
    def test_every_assertable_framework_clears_its_derived_floor(
        self, live: CorpusReport, overlay_present: bool
    ) -> None:
        assertable = _assertable_floors(overlay_present)
        floors = {f: JOIN_FLOORS[f] for f in assertable}
        failures = check_join_floors(live, floors)
        # Attainable [0, 22] messages. The eleven pending frameworks resolved 0
        # of 734 links before their parsers, so this returned 11 messages then
        # and returns 0 now, and any parser losing an anchor its source supplies
        # puts a message back.
        assert failures == [], failures
        # Positive control against a collapsed floor set. Attainable [0, 22].
        assert len(floors) == (22 if overlay_present else 18), sorted(floors)

    def test_no_floor_leaves_more_than_one_percent_of_its_links_spendable(
        self, live: CorpusReport, overlay_present: bool
    ) -> None:
        """The half of the v2 check that was missing.

        `assert floor <= 1.0` guards a floor set unreachably high. Nothing
        guarded a floor set so low that no realistic loss could cross it, and a
        floor of 0.0 passes that check on every framework. This measures the
        gap directly: how many resolved links a parser could lose and still
        report green.
        """
        too_loose: list[str] = []
        zero_slack = 0
        assertable = _assertable_floors(overlay_present)
        for framework_id in assertable:
            row = live.by_id(framework_id)
            spendable = _spendable_links(row)
            allowance = int(row.links * MAX_SPENDABLE_LINK_FRACTION)
            # Attainable [0, resolved] per framework. Reads 0 for 19 of the 22
            # today and 5 for cwe, whose 0.99 floor covers a link withdrawn
            # upstream. A floor dropped to 0.0 reads `resolved` and fails here.
            if spendable > allowance:
                too_loose.append(
                    f"{framework_id}: may lose {spendable} of "
                    f"{row.by_title + row.by_id} resolved links before its "
                    f"floor of {JOIN_FLOORS[framework_id]:.2f} fires, against "
                    f"an allowance of {allowance}"
                )
            zero_slack += int(spendable == 0)
        assert too_loose == [], too_loose
        # Non-vacuity. Most floors sit at their framework's arithmetic ceiling,
        # where a single lost link fails the gate. Attainable [0, 22]. Reads 19
        # locally and 16 in CI, and falls as floors are loosened.
        assert zero_slack * 2 >= len(assertable), (
            f"only {zero_slack} of {len(assertable)} floors sit where one lost "
            f"link fails them"
        )


class TestAnchorSeparation:
    def test_dsomm_stopped_collapsing_onto_its_sub_dimensions(
        self, live: CorpusReport, overlay_present: bool
    ) -> None:
        if "dsomm" not in _assertable(overlay_present):
            pytest.skip("dsomm is GPL-3.0-only and routes to the overlay")
        row = live.by_id("dsomm")
        # Attainable: distinct_anchors [0, 213] against 213 resolvable links,
        # links_per_anchor [1.0, 213.0]. Before the parser dsomm's 214 links
        # landed on 18 fallback anchors, which is 11.9 links per anchor, so
        # both assertions failed then and both can fail again.
        assert row.distinct_anchors >= 182, row.distinct_anchors
        assert row.links_per_anchor <= 1.20, row.links_per_anchor

    def test_biml_resolves_by_title_exactly_where_it_declared_an_alternate(
        self, live: CorpusReport, overlay_present: bool
    ) -> None:
        """Seven of 21 rows share a section_name across two documents.

        The title channel carries exactly the links whose OpenCRE spelling the
        parser declared as an alt_title, so the two artifacts are held equal
        rather than each pinned to a literal. Ruling R20 took that count from
        one to three.
        """
        if "biml" not in _assertable(overlay_present):
            pytest.skip("biml has no processed JSON in this checkout")
        row = live.by_id("biml")
        declared = _alt_title_count("biml")
        # Attainable [0, 21] on both sides. Reads 3 and 3. Dropping an alternate
        # moves both together, and an alternate that stops answering moves only
        # by_title, which is the drift this holds.
        assert declared == 3, declared
        assert row.by_title == declared, (row.by_title, declared)
        # 19, not the 20 the v2 plan asserted: `inference:9` appears prefixed
        # and unprefixed and both route to the same control. 19 is the
        # arithmetic maximum over 21 links, so this fails downward only, and
        # downward is the collapse being gated.
        assert row.distinct_anchors == 19, row.distinct_anchors

    def test_etsi_registered_only_the_names_that_cannot_collide(
        self, live: CorpusReport, overlay_present: bool
    ) -> None:
        """Three ETSI technique names span two clauses each.

        Registering all 24 as alternate titles keeps the resolution rate at
        1.0000 while two rows resolve to a clause they did not name, so the rate
        cannot see this and by_title can.
        """
        if "etsi" not in _assertable(overlay_present):
            pytest.skip(
                "etsi is restricted, reproduction only by written permission, "
                "and its processed text is absent from this checkout"
            )
        row = live.by_id("etsi")
        # Attainable [0, 36] across 36 links. Reads 5: two links answered by a
        # declared alternate plus three naming a clause heading verbatim. It
        # fails downward when an alternate is dropped and upward when one is
        # over-registered, and 24 is what registering every technique reads.
        assert row.by_title == 5, row.by_title
        # Attainable [0, 25]. Two alternates, which is the count the parser
        # derived from the clause-collision analysis.
        assert _alt_title_count("etsi") == 2

    def test_no_pending_framework_nests_an_anchor_inside_another(
        self, live: CorpusReport, overlay_present: bool
    ) -> None:
        for framework_id in _assertable(overlay_present):
            row = live.by_id(framework_id)
            # nested_anchors is containment, not strict prefix, so ETSI 5.2
            # inside 5.2.2 is visible. Attainable [0, distinct_anchors], which
            # is up to 182 on dsomm. Reads 0 everywhere and fails upward the
            # moment a parser emits a roll-up beside its own children.
            assert row.nested_anchors == 0, (
                f"{framework_id}: {row.nested_anchors} nested anchors"
            )

    def test_the_budgeted_wrong_anchor_counts_hold_and_a_detector_ran(
        self, live: CorpusReport, overlay_present: bool
    ) -> None:
        """The counter increments only where a detector applies.

        JOIN_WRONG_ANCHOR_BUDGET, committed in Task 1, holds one entry per
        framework whose task predicted the title channel would answer, each
        derived from that task's pre-parser premise check. Asserting `== 0`
        instead would halt a healthy run on csa_ccm, where `IPY` carries a
        section_name that is control IPY-01's title rather than the IPY domain's
        name, so title-first correctly answers with IPY-01.
        """
        applicable = wrong_anchor_applicable(live)
        checked = 0
        for framework_id, budget in sorted(JOIN_WRONG_ANCHOR_BUDGET.items()):
            if framework_id not in _assertable(overlay_present):
                continue
            row = live.by_id(framework_id)
            # Non-vacuity: a detector must have run, or the budget guards a
            # branch that never executes. Attainable [0, links]. Reads 29, 9
            # and 21 for csa_ccm, etsi and biml.
            assert applicable[framework_id] > 0, (
                f"{framework_id} has a wrong-anchor budget and not one link "
                f"reached an applicable detector, so the budget guards nothing"
            )
            # Attainable [0, applicable]. Reads 1, 1 and 0 against budgets of
            # 1, 1 and 0, so two of the three have zero headroom and the third
            # fails on any flag at all.
            assert row.wrong_anchor_risk <= budget, (
                f"{framework_id}: {row.wrong_anchor_risk} wrong anchors "
                f"against a pre-registered budget of {budget} over "
                f"{applicable[framework_id]} applicable checks"
            )
            checked += 1
        # Attainable [0, 3]. Reads 3 locally and 1 in CI, where csa_ccm and
        # etsi route to the overlay.
        assert checked == (3 if overlay_present else 1), checked

    def test_the_unbudgeted_wrong_anchor_exposure_is_named_and_pinned(
        self, live: CorpusReport, overlay_present: bool
    ) -> None:
        """No pending framework may acquire an unexamined wrong anchor.

        A framework outside JOIN_WRONG_ANCHOR_BUDGET must read zero, with one
        declared exception whose cause was measured and is recorded above.
        """
        for framework_id in _assertable(overlay_present):
            if framework_id in JOIN_WRONG_ANCHOR_BUDGET:
                continue
            row = live.by_id(framework_id)
            expected = UNBUDGETED_WRONG_ANCHOR_EXPOSURE.get(framework_id, 0)
            # Attainable [0, resolved], up to 213 on dsomm. All eight read 0
            # with no headroom since ruling R19 took nist_ssdf's 44 to 0, so a
            # single new flag anywhere fails this.
            assert row.wrong_anchor_risk == expected, (
                f"{framework_id}: {row.wrong_anchor_risk} wrong anchors "
                f"against {expected}. A framework outside "
                f"JOIN_WRONG_ANCHOR_BUDGET has no pre-registered allowance, so "
                f"measure the cause and register it in Task 1 rather than "
                f"widening this gate."
            )

    def test_the_nist_ssdf_kind_exemption_still_has_the_cause_it_rests_on(
        self,
    ) -> None:
        """The exemption's stated cause is checked, not asserted once and left.

        Ruling R19 exempts nist_ssdf from detector B because its section_name is
        a task statement and the title its id reaches is a task identifier. If
        parse_nist_ssdf ever gives its controls real titles, B stops comparing a
        sentence against an identifier, the kind ratio collapses toward 1.0, and
        the declared exemption stops being explained by the reason recorded for
        it. This is the assertion that catches that before the exemption goes on
        silently suppressing a detector that would now work.
        """
        assert "nist_ssdf" in DETECTOR_B_INAPPLICABLE
        controls = _controls("nist_ssdf")
        identifiers = sum(
            1
            for control in controls
            if control["title"].strip() == control["control_id"].strip()
        )
        # Attainable [0, 42]. Reads 42 of 42. Any real title lands here and
        # forces the exemption to be re-derived instead of inherited.
        assert identifiers == len(controls), (
            f"{identifiers} of {len(controls)} nist_ssdf controls are titled by "
            f"their own id. The R19 kind exemption above rests on all of them "
            f"being so."
        )

    def test_the_untouched_frameworks_did_not_gain_wrong_anchor_exposure(
        self, live: CorpusReport, overlay_present: bool
    ) -> None:
        """The rebuild rewrote every framework's text, not only the eleven's."""
        before = _rows(_load(BEFORE_PATH))
        live_baselines = 0
        for framework_id, previous in sorted(before.items()):
            if framework_id in PENDING:
                continue
            if not overlay_present and framework_id in OVERLAY:
                continue
            row = live.by_id(framework_id)
            baseline = previous["wrong_anchor_risk"]
            # Attainable [0, resolved], up to 1,799 on capec. Ten of the eleven
            # read exactly their baseline, so a single new flag fails. Ruling
            # R21 took nist_ai_100_2 from 20 to 8, which is the one row with
            # headroom and the one this cannot catch a regression inside.
            assert row.wrong_anchor_risk <= baseline, (
                f"{framework_id}: {row.wrong_anchor_risk} wrong anchors "
                f"against {baseline} before this plan"
            )
            live_baselines += int(baseline > 0)
        # Non-vacuity: `0 <= 0` on every row would pass having compared
        # nothing. Attainable [0, 11]. Reads 6 locally and 5 in CI.
        assert live_baselines >= 5, (
            f"only {live_baselines} untouched frameworks carried a non-zero "
            f"wrong-anchor baseline, so this comparison is close to vacuous"
        )


class TestTextQuality:
    """The column the v2 plan never had.

    distinct_anchors was named the load-bearing column and it is the wrong one
    for seven of eleven parsers, whose anchor count does not move at all. What
    moves for all eleven is where the anchor text comes from.
    """

    def test_the_pending_frameworks_stopped_anchoring_on_fallback_titles(
        self, live: CorpusReport, overlay_present: bool
    ) -> None:
        before = _rows(_load(BEFORE_PATH))
        for framework_id in _assertable(overlay_present):
            row = live.by_id(framework_id)
            baseline = before[framework_id]["fallback_anchors"]
            # A fallback anchor is a distinct section_name the trainer gets for
            # a link the prose index missed. Attainable [0, baseline] on the
            # passing side and [baseline, links] on the failing side. The
            # eleven summed to 299 before the parsers and sum to 6 now, and
            # every one of them failed this before its parser existed.
            assert baseline > 0, (
                f"{framework_id} had no fallback anchors in the BEFORE state, "
                f"so this comparison would be vacuous. Check that the BEFORE "
                f"artifact was captured with the fallback_anchors column."
            )
            assert row.fallback_anchors < baseline, (
                f"{framework_id}: {row.fallback_anchors} fallback anchors "
                f"against {baseline} before the parser"
            )

    def test_no_anchor_in_the_corpus_restates_its_own_control_title(
        self, live: CorpusReport
    ) -> None:
        """The gate the v2 plan meant to write.

        Its `anchor_source_full_text + anchor_source_description > 0` fails on
        samm, whose 30 anchors are all parser-assembled, and passes on a
        framework where one link in 200 reaches a statement. The column that
        answers the real question is anchor_source_title: an anchor a parser
        wrote into full_text that only restates the control's own title reads
        as a title here even though the prose rule cannot see it.
        """
        offenders = [
            f"{row.framework_id}: {row.anchor_source_title}"
            for row in live.per_framework
            if row.anchor_source_title
        ]
        # Attainable [0, 4405] across the corpus. Reads 0. A parser that stores
        # a title where a statement belongs lands here for every link it
        # answers, which is up to 1,799 on capec.
        assert offenders == [], offenders

    def test_parser_assembled_anchors_are_declared_per_framework(
        self, live: CorpusReport, overlay_present: bool
    ) -> None:
        """A ratchet on how much of the corpus the project wrote itself."""
        actual = {
            row.framework_id: row.anchor_source_synthetic
            for row in live.per_framework
            if row.anchor_source_synthetic
            and (overlay_present or row.framework_id not in OVERLAY)
        }
        expected = {
            framework_id: count
            for framework_id, count in SYNTHETIC_ANCHOR_BUDGET.items()
            if overlay_present or framework_id not in OVERLAY
        }
        # Attainable [0, resolved] per framework and an arbitrary key set.
        # Reads {csa_ccm 14, etsi 4, samm 30, wstg 1} locally and {samm 30,
        # wstg 1} in CI. Fails on a new synthesising parser, on a count moving
        # either way, and on a parser that stops marking its own assembly.
        assert actual == expected, (actual, expected)

    def test_the_anchor_source_columns_account_for_every_resolved_link(
        self, live: CorpusReport
    ) -> None:
        """An instrument check rather than a parser check, and cheap."""
        for row in [*live.per_framework, live.totals]:
            # Attainable: any pair of integers. A classifier that grew a fifth
            # bucket without a column reads short here.
            assert (
                row.anchor_source_full_text
                + row.anchor_source_description
                + row.anchor_source_title
                + row.anchor_source_synthetic
            ) == row.by_title + row.by_id, row.framework_id


class TestNoRegression:
    def test_iso_still_resolves(
        self, live: CorpusReport, overlay_present: bool
    ) -> None:
        """ISO reached 92 of 94 before this plan. Nothing here may cost it."""
        if not overlay_present:
            pytest.skip(
                "iso_27001 is restricted, single-user store licence, no "
                "reproduction without prior written permission, and its "
                "processed text is absent from this checkout. The v2 suite "
                "asserted >= 92 here and got 0 in CI."
            )
        row = live.by_id("iso_27001")
        # Attainable [0, 94] on both. ISO is the corpus's only high-prose fold,
        # so a regression here is the most expensive one available, and the
        # rebuild in Task 15 rewrote its text along with everything else.
        assert row.by_title + row.by_id >= 92, row.by_title + row.by_id
        assert row.distinct_anchors >= 91, row.distinct_anchors

    def test_the_frameworks_this_plan_did_not_touch_are_unchanged(
        self, live: CorpusReport, overlay_present: bool
    ) -> None:
        before = _rows(_load(BEFORE_PATH))
        checked = 0
        for framework_id, previous in sorted(before.items()):
            if framework_id in PENDING:
                continue
            if not overlay_present and framework_id in OVERLAY:
                continue
            current = live.by_id(framework_id)
            assert current.distinct_anchors == previous["distinct_anchors"], (
                framework_id
            )
            assert current.by_title + current.by_id == (
                previous["by_title"] + previous["by_id"]
            ), framework_id
            checked += 1
        # Eleven untouched frameworks, of which iso_27001 is the only overlay
        # member, so 10 in CI and 11 locally. Without this the loop could
        # iterate zero times and report green, which is exactly what the
        # deleted skip did. Attainable [0, 11].
        assert checked == (11 if overlay_present else 10), (
            f"only {checked} untouched frameworks compared"
        )


class TestCommittedAfterReport:
    """All eleven, gated off tracked evidence, on every machine.

    The live-corpus tests above cannot assert three of the eleven in CI, because
    those frameworks' text routes to the gitignored overlay. This class closes
    that hole without weakening a floor: the AFTER artifact is tracked, carries
    counts and digests and no anchor text, and is produced by a separate command
    from the floors it is checked against.
    """

    def test_the_after_artifact_was_captured_with_the_full_corpus(self) -> None:
        after = _load(AFTER_PATH)
        # Attainable: any non-negative integer. This is the assertion that
        # stops the AFTER state being captured on the tracked corpus and then
        # certified as covering the restricted frameworks.
        assert after["corpus_framework_count"] == FULL_CORPUS_FRAMEWORK_COUNT, (
            f"the AFTER report covers {after['corpus_framework_count']} "
            f"frameworks, not {FULL_CORPUS_FRAMEWORK_COUNT}. It must be "
            f"captured on a checkout holding the licensed overlay, with the "
            f"same command and the same interpreter as the BEFORE state."
        )
        assert len(after["corpus_sha256"]) == 64
        assert after["corpus_sha256"] != _load(BEFORE_PATH)["corpus_sha256"], (
            "the AFTER report names the same corpus as the BEFORE report, so "
            "nothing was rebuilt between them"
        )
        # The link file did not move, so every delta between the two artifacts
        # is a corpus change. Attainable: two 64-character strings, equal or
        # not. This fails if a later task edits the curated links and recaptures
        # only one side.
        assert after["links_sha256"] == _load(BEFORE_PATH)["links_sha256"]

    def test_the_after_artifact_clears_every_floor_including_the_licensed_ones(
        self,
    ) -> None:
        rows = _rows(_load(AFTER_PATH))
        misses: list[str] = []
        for framework_id, floor in sorted(JOIN_FLOORS.items()):
            row = rows[framework_id]
            if row["resolution_rate"] + 1e-9 < floor:
                misses.append(
                    f"{framework_id} {row['resolution_rate']:.4f} < {floor:.4f}"
                )
        # Attainable [0, 22] misses. This is the only place etsi's floor of 1.00
        # is asserted on a machine without the ETSI source, and deleting it is
        # the repair the v2 suite invited.
        assert misses == [], misses
        # Positive control against an artifact holding a truncated row set.
        assert len(rows) == 22, sorted(rows)

    def test_the_after_artifact_records_what_the_eleven_could_not_do_before(
        self,
    ) -> None:
        """The headline, gated rather than narrated.

        The eleven resolved 0 links and reached 0 anchors before their parsers,
        so every figure here is a change from zero and none of it can be read
        off the BEFORE artifact.
        """
        after = _rows(_load(AFTER_PATH))
        before = _rows(_load(BEFORE_PATH))
        resolved = sum(
            after[f]["by_title"] + after[f]["by_id"] for f in PENDING
        )
        statements = sum(
            after[f]["anchor_source_full_text"]
            + after[f]["anchor_source_description"]
            for f in PENDING
        )
        fallback_before = sum(before[f]["fallback_anchors"] for f in PENDING)
        fallback_after = sum(after[f]["fallback_anchors"] for f in PENDING)
        # Attainable [0, 734], the eleven frameworks' link count. Reads 723.
        assert resolved == 723, resolved
        # Attainable [0, 723]. Reads 674, with the remaining 49 parser-assembled
        # and pinned in SYNTHETIC_ANCHOR_BUDGET.
        assert statements == 674, statements
        # Attainable [0, 734] on both sides. Reads 299 and 6.
        assert fallback_before == 299, fallback_before
        assert fallback_after == 6, fallback_after

    def test_the_after_artifact_carries_no_free_text(self) -> None:
        """It is tracked, so it must be safe to track.

        Structural rather than heuristic. The report is counts, ratios, ids and
        digests. A string field longer than a framework id is somewhere prose
        could sit unnoticed, and once tracked this file is also scanned by
        tests/test_licensed_text_not_tracked.py.
        """
        after = _load(AFTER_PATH)
        long_strings: list[tuple[str, int]] = []

        def walk(node: object, path: str) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(value, f"{path}.{key}")
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    walk(value, f"{path}[{index}]")
            elif isinstance(node, str) and len(node) > 128:
                long_strings.append((path, len(node)))

        walk(after, "$")
        # Attainable [0, n]. The longest legitimate string is the corpus path.
        assert long_strings == [], long_strings

    def test_the_after_artifact_matches_the_live_report_where_both_can_see(
        self, live: CorpusReport, overlay_present: bool
    ) -> None:
        """The committed artifact must not be hand-edited.

        Without this the class above degrades into trusting a JSON file that a
        worker under a red build could open in an editor.
        """
        rows = _rows(_load(AFTER_PATH))
        compared = 0
        for framework_id, row in sorted(rows.items()):
            if not overlay_present and framework_id in OVERLAY:
                continue
            current = live.by_id(framework_id)
            assert current.by_title == row["by_title"], framework_id
            assert current.by_id == row["by_id"], framework_id
            assert current.distinct_anchors == row["distinct_anchors"], framework_id
            assert current.fallback_anchors == row["fallback_anchors"], framework_id
            assert current.wrong_anchor_risk == row["wrong_anchor_risk"], framework_id
            compared += 1
        # Attainable [0, 22]. Reads 22 locally and 18 in CI. A collapsed
        # comparison is how this goes quiet.
        assert compared == (22 if overlay_present else 18), (
            f"only {compared} rows cross-checked"
        )


class TestSpecAcceptance:
    """Spec Part 1.9, checked against stored text rather than the join flag.

    Both surviving checks in the v2 suite globbed PROCESSED_FRAMEWORKS_DIR, and
    the overlay frameworks' JSON files are absent from a fresh clone, so the
    glob returned 28 of 32 and the frameworks under the strictest licence got
    the least checking. These loops are driven from the union of the glob and
    OVERLAY_FRAMEWORK_IDS, and an absent file is recorded by name rather than
    skipped past.
    """

    def _partition(self) -> tuple[list[Path], list[str]]:
        present: list[Path] = []
        absent: list[str] = []
        for framework_id in sorted(_expected_framework_ids()):
            path = PROCESSED_FRAMEWORKS_DIR / f"{framework_id}.json"
            if path.exists():
                present.append(path)
            else:
                absent.append(framework_id)
        # An absent file is legal only under a licence. Anything else is a
        # missing parser output and must be loud.
        unexplained = [f for f in absent if f not in OVERLAY]
        assert not unexplained, (
            f"{unexplained} have no processed JSON and no licence that "
            f"explains the absence. A framework cannot leave these checks by "
            f"going missing."
        )
        return present, absent

    def test_every_processed_framework_has_a_parser(self) -> None:
        parsed = _parsed_framework_ids()
        # Deliberately NOT _expected_framework_ids(), which now includes the
        # parser stems and would make this read `parsed - parsed`. The claim is
        # that stored output traces back to a parser, so the left side is what
        # is stored plus what a licence hides.
        stored = _on_disk_framework_ids() | OVERLAY
        # Attainable [0, 32] unparsed ids. Read 11 before this plan, reads 0
        # now, and a processed file arriving without a parser module lands here.
        assert sorted(stored - parsed) == []
        # Positive control against an empty glob, which would make the line
        # above read `set() - set() == set()`. Attainable [0, n]. Reads 32.
        assert len(parsed) == len(_expected_framework_ids()), sorted(
            _expected_framework_ids() - parsed
        )

    def test_no_version_field_says_opencre(self) -> None:
        present, absent = self._partition()
        offenders = [
            path.name
            for path in present
            if "opencre-" in str(
                json.loads(path.read_text(encoding="utf-8"))["version"]
            )
        ]
        # Attainable [0, 32]. Read exactly 11 before this plan, and they were
        # exactly the eleven frameworks whose controls came out of the OpenCRE
        # link rows rather than out of a source document.
        assert offenders == [], offenders
        assert set(absent) <= OVERLAY, sorted(set(absent) - OVERLAY)
        # Attainable [0, 32]. Reads 32 locally and 28 in CI.
        assert len(present) == (32 if not absent else 28), len(present)

    def test_every_framework_meets_its_parsers_declared_prose_floor(self) -> None:
        """The gate with teeth, in place of a comparison against zero.

        honest_prose_fraction returns a ratio, so `> 0.0` passes on one prose
        control in csa_ccm's 224. The declared floor is read out of the parser
        source with a regex rather than by importing, matching the convention in
        tests/test_parser_manifest_coverage.py, so a parser whose extraction
        dependency is missing is still covered.
        """
        from tract.parsers.base import BaseParser
        from tract.schema import Control

        present, absent = self._partition()
        offenders: list[str] = []
        unfloored: list[str] = []
        floors_held = 0
        for path in present:
            source_file = PARSERS_DIR / f"parse_{path.stem}.py"
            assert source_file.exists(), (
                f"{path.stem} has processed output and no parser module"
            )
            match = _MIN_PROSE.search(source_file.read_text(encoding="utf-8"))
            if match is None:
                unfloored.append(path.stem)
                continue
            floor = float(match.group(1))
            # A declared 0.0 is the inherited default wearing a costume.
            assert floor > 0.0, f"parse_{path.stem}.py declares a floor of 0.0"
            data = json.loads(path.read_text(encoding="utf-8"))
            controls = [Control(**c) for c in data["controls"]]
            fraction = BaseParser.honest_prose_fraction(controls)
            floors_held += 1
            if fraction + 1e-9 < floor:
                offenders.append(f"{path.stem}: {fraction:.4f} < {floor:.4f}")

        # Attainable [0, floors_held]. Every one of the eleven declares 0.95 to
        # 1.00 and every one reported honest_prose_fraction 0.0000 before its
        # parser, so this was red for eleven frameworks and goes red again on
        # any parser storing titles where statements belong.
        assert offenders == [], offenders

        missing = sorted(set(PENDING) & set(unfloored))
        assert missing == [], (
            f"{missing} have a parser and no declared min_prose_fraction. "
            f"Every one of the eleven declares one, so a missing floor means "
            f"the declaration was dropped in implementation."
        )

        # Ratchet, not a refactor. Attainable [0, len(present)]. 19 of the 21
        # parsers that predate this plan inherit the 0.0 default, and this fails
        # upward the moment a new parser ships without a floor.
        assert len(unfloored) <= UNFLOORED_PARSER_RATCHET, sorted(unfloored)

        # Positive control against a collapsed file list. Attainable
        # [0, len(present)]. Read 13 locally and 9 in CI when this landed, and
        # the bound is one-sided for the same reason the ratchet above is: the
        # count rises as the unfloored 19 are retrofitted. len(present) below
        # carries the two-sided check on the file list itself.
        assert floors_held >= (13 if not absent else 9), (
            f"only {floors_held} declared floors were read"
        )
        assert set(absent) <= OVERLAY, sorted(set(absent) - OVERLAY)

    def test_restricted_frameworks_are_named_when_they_cannot_be_checked(
        self,
    ) -> None:
        """The exemption is stated, not inferred.

        RESTRICTED_FRAMEWORK_IDS is imported here so that a framework leaving
        the restricted tier stops being exempt in the same commit.
        """
        _, absent = self._partition()
        for framework_id in sorted(absent):
            assert framework_id in OVERLAY, framework_id
        # ETSI declares min_prose_fraction = 1.0, which is the strictest floor
        # in the plan, and in a fresh clone nothing checks it. Recording the
        # names is the minimum honest reporting of that hole. Attainable
        # [0, 4] absent ids.
        assert len(absent) in (0, len(OVERLAY)), sorted(absent)
        if absent:
            print(
                f"unchecked under licence: {sorted(absent)} "
                f"(restricted: {sorted(RESTRICTED_FRAMEWORK_IDS)})"
            )


def _controls(framework_id: str) -> list[dict[str, Any]]:
    """One framework's processed controls.

    Raises:
        AssertionError: If the file is absent. Callers check a tracked
            framework, so absence means the corpus is broken rather than
            licensed away.
    """
    path = PROCESSED_FRAMEWORKS_DIR / f"{framework_id}.json"
    assert path.exists(), (
        f"{framework_id} has no processed JSON and is not under a licence that "
        f"explains it"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    controls: list[dict[str, Any]] = data["controls"]
    return controls


def _alt_title_count(framework_id: str) -> int:
    """Alternate titles a parser declared, across the framework's controls."""
    total = 0
    for control in _controls(framework_id):
        metadata = control.get("metadata") or {}
        alternates = metadata.get("alt_titles")
        if isinstance(alternates, list):
            total += len(alternates)
        elif isinstance(alternates, str):
            total += 1
    return total
