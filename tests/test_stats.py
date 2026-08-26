"""Tests for the Wilson score interval used by the ceiling study scorer."""
from __future__ import annotations

import pytest

from tract.stats import wilson_interval


class TestWilsonInterval:
    def test_matches_the_phase3_13_of_20_datum(self) -> None:
        """The design doc's reference figure: 13/20, Wilson 95% [0.433, 0.819]."""
        result = wilson_interval(13, 20)
        assert result.point == pytest.approx(0.65)
        assert result.lower == pytest.approx(0.433, abs=0.001)
        assert result.upper == pytest.approx(0.819, abs=0.001)

    def test_n_250_at_alpha_065_half_width_matches_design_target(self) -> None:
        """Design doc table: n=250 at alpha ~= 0.65 gives half-width 0.059."""
        result = wilson_interval(round(0.65 * 250), 250)
        assert result.half_width == pytest.approx(0.059, abs=0.001)

    def test_bounds_stay_within_unit_interval(self) -> None:
        for successes, n in ((0, 10), (10, 10), (1, 1), (0, 1)):
            result = wilson_interval(successes, n)
            assert 0.0 <= result.lower <= result.upper <= 1.0

    def test_point_estimate_is_the_raw_proportion(self) -> None:
        result = wilson_interval(3, 4)
        assert result.point == pytest.approx(0.75)

    def test_larger_n_gives_narrower_interval_at_same_proportion(self) -> None:
        small = wilson_interval(65, 100)
        large = wilson_interval(650, 1000)
        assert large.half_width < small.half_width

    def test_zero_n_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            wilson_interval(0, 0)

    def test_negative_n_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            wilson_interval(0, -5)

    def test_successes_exceeding_n_raises(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            wilson_interval(11, 10)

    def test_negative_successes_raises(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            wilson_interval(-1, 10)
