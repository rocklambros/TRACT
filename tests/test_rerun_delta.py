"""The round-2 comparison must be paired, and must not overstate a null.

Both rounds cover the same 300 controls, so every control has a before and an
after. That makes the comparison paired, and the right test on a paired binary
decision is McNemar's -- which looks only at the controls where someone changed
their mind.

A raw before/after rate difference would be dominated by the ~250 controls
nobody was ever going to link, which is the same negative-class inflation that
made Q4's first agreement figure misleading.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.analysis.rerun_delta import _mcnemar_exact, compare

FIELDS = ["control_id", "control_title", "control_text",
          "cre_id", "confidence", "rationale"]


def _sheet(directory: Path, annotator: str, answers: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{annotator}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for control, hub in answers.items():
            writer.writerow({
                "control_id": control, "control_title": "t", "control_text": "x",
                "cre_id": hub, "confidence": "3", "rationale": "r",
            })


class TestMcNemarItself:
    def test_symmetric_discordance_is_not_significant(self) -> None:
        assert _mcnemar_exact(10, 10) == pytest.approx(1.0)

    def test_one_sided_movement_is_significant(self) -> None:
        p = _mcnemar_exact(12, 0)
        assert p is not None and p < 0.001

    def test_no_discordant_pairs_returns_none(self) -> None:
        """Not 1.0. Nobody changed their mind, so the test has no input."""
        assert _mcnemar_exact(0, 0) is None

    def test_it_ignores_the_concordant_majority(self) -> None:
        """The whole reason for McNemar here.

        250 controls nobody moved on must not dilute the signal from the few
        who did -- that is the negative-class inflation this project has
        already been bitten by once.
        """
        assert _mcnemar_exact(8, 1) == _mcnemar_exact(8, 1)
        few = _mcnemar_exact(8, 1)
        assert few is not None and few < 0.05


class TestTheComparisonIsPaired:
    def test_movement_toward_linking_is_detected(self, tmp_path: Path) -> None:
        r1 = {f"AC-{i}": "NONE" for i in range(20)}
        r2 = dict(r1)
        for i in range(6):
            r2[f"AC-{i}"] = "010-108"
        _sheet(tmp_path / "r1", "vol-01", r1)
        _sheet(tmp_path / "r2", "vol-01", r2)

        d = compare(tmp_path / "r1", tmp_path / "r2", "vol-01")
        assert d["none_to_link"] == 6
        assert d["link_to_none"] == 0
        assert d["mcnemar_p"] is not None and d["mcnemar_p"] < 0.05
        assert "toward linking" in d["direction"]

    def test_movement_toward_none_is_reported_as_against_the_hypothesis(
        self, tmp_path: Path
    ) -> None:
        r1 = {f"AC-{i}": "010-108" for i in range(20)}
        r2 = dict(r1)
        for i in range(6):
            r2[f"AC-{i}"] = "NONE"
        _sheet(tmp_path / "r1", "vol-01", r1)
        _sheet(tmp_path / "r2", "vol-01", r2)

        d = compare(tmp_path / "r1", tmp_path / "r2", "vol-01")
        assert d["link_to_none"] == 6
        assert "against the hypothesis" in d["direction"]

    def test_a_hub_change_is_not_counted_as_a_link_decision(
        self, tmp_path: Path
    ) -> None:
        """Changing WHICH hub is a different question from whether to link."""
        r1 = {"AC-1": "010-108", "AC-2": "NONE"}
        r2 = {"AC-1": "011-087", "AC-2": "NONE"}
        _sheet(tmp_path / "r1", "vol-01", r1)
        _sheet(tmp_path / "r2", "vol-01", r2)

        d = compare(tmp_path / "r1", tmp_path / "r2", "vol-01")
        assert d["hub_changed"] == 1
        assert d["none_to_link"] == d["link_to_none"] == 0
        assert d["mcnemar_p"] is None

    def test_no_movement_reports_no_p_value_rather_than_one(
        self, tmp_path: Path
    ) -> None:
        """A null must not arrive dressed as a measurement."""
        answers = {f"AC-{i}": "NONE" for i in range(20)}
        _sheet(tmp_path / "r1", "vol-01", answers)
        _sheet(tmp_path / "r2", "vol-01", answers)

        d = compare(tmp_path / "r1", tmp_path / "r2", "vol-01")
        assert d["mcnemar_p"] is None
        assert d["unchanged"] == 20

    def test_only_controls_in_both_rounds_are_compared(
        self, tmp_path: Path
    ) -> None:
        _sheet(tmp_path / "r1", "vol-01", {"AC-1": "NONE", "AC-2": "NONE"})
        _sheet(tmp_path / "r2", "vol-01", {"AC-2": "010-108", "AC-3": "010-108"})
        d = compare(tmp_path / "r1", tmp_path / "r2", "vol-01")
        assert d["n_common"] == 1

    def test_disjoint_rounds_raise_rather_than_report_nothing(
        self, tmp_path: Path
    ) -> None:
        _sheet(tmp_path / "r1", "vol-01", {"AC-1": "NONE"})
        _sheet(tmp_path / "r2", "vol-01", {"ZZ-9": "NONE"})
        with pytest.raises(ValueError, match="share no control ids"):
            compare(tmp_path / "r1", tmp_path / "r2", "vol-01")
