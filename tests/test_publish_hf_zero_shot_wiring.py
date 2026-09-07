"""`tract publish-hf` could not run at all.

`_load_fold_results` raises unless it is given `zero_shot_path`:

    "No zero-shot baseline available: pass zero_shot_path pointing at the
     paired per-fold baseline for THIS campaign. The card's Zero-shot and Delta
     columns are computed from it, and a hardcoded fallback publishes a
     comparison that was never run."

and `_cmd_publish_hf` called it with two arguments. The guard is right -- it
replaced five hardcoded constants that published a comparison nobody measured --
but the caller could never satisfy it, so the documented republish command had
no working path. The comment inside the raise even records that the measured
branch was unreachable, and the caller was left unfixed.

The baseline cannot be defaulted. Which per-fold zero-shot run pairs with a
given campaign is exactly the thing the guard exists to make explicit, and
`results/phase1b/zero_shot_firewalled_baseline/aggregate_metrics.json` carries
no campaign identifier -- only `model`, `hub_format` and `firewalled` -- so
nothing in the repo can infer it. The operator names it.
"""
from __future__ import annotations


import pytest


class TestTheFlagExists:

    def test_publish_hf_accepts_a_zero_shot_results_path(self) -> None:
        from tract.cli import build_parser
        args = build_parser().parse_args([
            "publish-hf", "--repo-id", "x/y",
            "--zero-shot-results", "results/phase1b/zs/aggregate_metrics.json",
        ])
        assert args.zero_shot_results.endswith("aggregate_metrics.json")

    def test_it_is_required(self) -> None:
        from tract.cli import build_parser
        with pytest.raises(SystemExit):
            build_parser().parse_args(["publish-hf", "--repo-id", "x/y"])

    def test_the_help_says_why(self) -> None:
        from tract.cli import build_parser
        action = next(
            a for a in build_parser()._subparsers._group_actions[0]  # type: ignore[union-attr]
            .choices["publish-hf"]._actions
            if a.dest == "zero_shot_results"
        )
        assert "paired" in (action.help or "").lower()


class TestTheWiringReachesTheLoader:

    def test_the_handler_forwards_the_path(self) -> None:
        import inspect
        from tract import cli
        source = inspect.getsource(cli._cmd_publish_hf)
        assert "zero_shot_results" in source, (
            "_cmd_publish_hf still calls _load_fold_results without the "
            "baseline, so publish-hf raises before it can upload anything"
        )

    def test_the_loader_still_refuses_without_one(self, tmp_path) -> None:
        # The guard must survive the wiring fix: a caller that omits it, or
        # points at a file that is not there, still gets a refusal rather than
        # a fabricated comparison.
        from tract.cli import _load_fold_results
        with pytest.raises(ValueError, match="No zero-shot baseline"):
            _load_fold_results(tmp_path, tmp_path / "corrected.json")
        with pytest.raises(ValueError, match="No zero-shot baseline"):
            _load_fold_results(tmp_path, tmp_path / "corrected.json",
                               tmp_path / "absent.json")
