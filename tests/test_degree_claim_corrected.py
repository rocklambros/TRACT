"""The audit's link-degree statistic was computed on the post-audit graph.

`docs/campaign2-results.md` §13, `scripts/analysis/audit_stratified_delta.py`
and `PRD.md` all carried the same figure: "49 of 56 corrections relocate gold
from a sparsely-linked hub to a densely-linked one (median link degree
3.0 -> 7.5)". Degree was counted over `hub_links_curated.jsonl`, which is the
file the corrections had already been applied to. Because 56 corrections
collapse onto 26 destination hubs, each destination is credited with the
corrections that landed on it -- and each source is drained by them.

Recomputed on the pre-audit graph the direction REVERSES: median 4.0 -> 3.0,
and 20 of 56 move to a higher-degree hub, not 49.

`audit_stratified_delta.py` disclosed the contamination and priced it at "+1
per correction". That understates it: 56/26 is roughly 2.15 corrections per
destination hub, and the sources lose an edge each at the same time.

This module pins the arithmetic and requires every document that states the
figure to state the corrected one. It follows the pattern of
`tests/test_audit_disclosure.py`: the project's failure mode is a true-looking
number surviving in prose after the thing it described has moved.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_LOG = PROJECT_ROOT / "data" / "training" / "audit_corrections_log.json"

# Documents that state the degree movement and must state it correctly.
DOCS_ASSERTING_THE_FIGURE = (
    PROJECT_ROOT / "docs" / "campaign2-results.md",
    PROJECT_ROOT / "scripts" / "analysis" / "audit_stratified_delta.py",
    PROJECT_ROOT / "PRD.md",
)

# The superseded figures. Any document still asserting one of these without
# also carrying the correction is stating a number the arithmetic refutes.
SUPERSEDED_FRAGMENTS = ("49 of 56", "49 of the 56", "3.0 → 7.5", "3.0 -> 7.5")


def _degrees() -> tuple[Counter[str], Counter[str], list[dict[str, str]]]:
    """Return (post-audit degree, pre-audit degree, corrections)."""
    from scripts.phase0.common import load_curated_links

    log = json.loads(AUDIT_LOG.read_text(encoding="utf-8"))
    corrections = log["corrections"]
    post: Counter[str] = Counter(link.cre_id for link in load_curated_links())
    pre = Counter(post)
    for c in corrections:
        pre[c["new_cre_id"]] -= 1
        pre[c["old_cre_id"]] += 1
    return post, pre, corrections


pytestmark = pytest.mark.skipif(
    not AUDIT_LOG.is_file(),
    reason="audit_corrections_log.json is not present on this checkout",
)


class TestDegreeArithmetic:

    def test_post_audit_degree_reproduces_the_published_figure(self) -> None:
        """The published number is reproducible — it is just measured wrong."""
        import statistics

        post, _, corrections = _degrees()
        old = [post[c["old_cre_id"]] for c in corrections]
        new = [post[c["new_cre_id"]] for c in corrections]
        assert statistics.median(old) == 3.0
        assert statistics.median(new) == 7.5
        assert sum(1 for o, n in zip(old, new) if n > o) == 49

    def test_pre_audit_degree_reverses_the_direction(self) -> None:
        import statistics

        _, pre, corrections = _degrees()
        old = [pre[c["old_cre_id"]] for c in corrections]
        new = [pre[c["new_cre_id"]] for c in corrections]
        assert statistics.median(old) == 4.0
        assert statistics.median(new) == 3.0
        moved_up = sum(1 for o, n in zip(old, new) if n > o)
        assert moved_up == 20, (
            f"{moved_up}/56 corrections move to a higher-degree hub on the "
            "pre-audit graph, against 49/56 on the post-audit graph. If this "
            "number has changed, the corpus moved and every document quoting "
            "it needs re-deriving."
        )

    def test_the_bias_is_not_one_edge_per_correction(self) -> None:
        """audit_stratified_delta.py priced the contamination at +1 per row."""
        _, _, corrections = _degrees()
        destinations = {c["new_cre_id"] for c in corrections}
        assert len(corrections) == 56
        assert len(destinations) == 26
        # ~2.15 corrections land on the average destination hub, so crediting
        # each destination with only one is an under-count of the inflation.
        assert len(corrections) / len(destinations) > 2.0


class TestDocumentsCarryTheCorrection:

    @pytest.mark.parametrize("path", DOCS_ASSERTING_THE_FIGURE, ids=lambda p: p.name)
    def test_superseded_figure_is_marked_where_it_appears(self, path: Path) -> None:
        if not path.is_file():
            pytest.skip(f"{path.name} not present")
        text = path.read_text(encoding="utf-8")
        stated = [f for f in SUPERSEDED_FRAGMENTS if f in text]
        if not stated:
            return
        lowered = text.lower()
        assert "20 of 56" in text or "20/56" in text, (
            f"{path.name} states {stated} but never gives the corrected "
            "figure (20 of 56 on the pre-audit graph). A document that quotes "
            "the superseded number without the correction is asserting it."
        )
        assert "pre-audit" in lowered, (
            f"{path.name} states {stated} without explaining that degree was "
            "counted on the post-audit graph."
        )
