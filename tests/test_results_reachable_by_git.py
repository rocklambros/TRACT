"""A run's output must be stageable, and .gitignore is where that silently broke.

`results/*` in .gitignore excluded results/phase1b, which is where the LOFO
orchestrator writes and where runpod_parallel's `collect` rsyncs a fleet's
output. Forty-five fold results were already tracked when the rule landed, so
`git status` stayed clean and nothing looked wrong, while every new result a
run produced was unstageable. A fleet could finish with its evidence on disk
and nothing to push.

The check is derived, not hand-written. It reads every output-directory
constant out of tract.config and asks git about a probe path under each, so a
directory added to config in future is covered the day it appears rather than
the day someone remembers to extend a list here. Directories that are excluded
on purpose are named in _DELIBERATELY_EXCLUDED with the reason, which makes
"ignored" a decision someone wrote down instead of a default nobody noticed.

Owner: TRACT
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Final

import pytest

from tract import config

REPO_ROOT: Final[Path] = config.PROJECT_ROOT

# Paths whose exclusion is a decision, with the reason it was made. An entry
# here is a claim that git should NOT see this path; the test below holds those
# to the same standard as the reachable ones, so a mistaken entry fails rather
# than quietly suppressing a real defect.
_DELIBERATELY_EXCLUDED: Final[dict[str, str]] = {
    "results/phase1c": "the directory defaults to excluded; calibration/ is carved back in",
    "results/phase1c/crosswalk.db": "a database, already covered by *.db",
    # deployment_model/ is not a run output at all. On a machine that has run
    # `tract assign`, every entry in it is a symlink into the HuggingFace cache
    # snapshot of the published rockCO78/tract-cre-assignment model. These
    # artifacts are distributed through HuggingFace and versioned by its
    # revision pin, so git is the wrong home for them.
    "results/phase1c/deployment_model": "published model weights, symlinked from the HF cache",
    "results/phase1c/deployment_model/deployment_artifacts.npz": (
        "packed deployment embeddings, shipped with the published model"
    ),
    "results/phase1c/deployment_model/calibration.json": (
        "the published model's calibration parameters (t_deploy, thresholds, "
        "conformal quantile); they describe one checkpoint and are meaningless "
        "without it. The ECE gate VERDICT is separate and is tracked, at "
        "results/phase1c/calibration/ece_gate.json"
    ),
    "results/phase1c/similarities": "per-fold .npz similarity arrays",
}

# Model artifacts that stay out of git wherever they land under results/phase1b.
# Every entry earns its place by killing a mutation the others survive.
#
# The two JSON probes exercise the checkpoint-*/ and model/ directory rules.
# Nothing else does: .safetensors and .pt never match the *.json allowlist in
# the first place, so a suite carrying only tensor probes stayed green with the
# checkpoint exclusion deleted.
#
# The last two sit directly in a fold directory rather than under checkpoint-*/
# or model/. No such file exists today, so widening the allowlist to every
# extension is unobservable without them. They keep the allowlist an allowlist
# for a layout nobody has written yet.
_MODEL_ARTIFACT_PROBES: Final[tuple[str, ...]] = (
    "checkpoint-999/adapter_config.json",
    "model/model/tokenizer.json",
    "checkpoint-999/adapter_model.safetensors",
    "checkpoint-999/optimizer.pt",
    "embeddings.npz",
    "weights.safetensors",
)


def _is_ignored(relative_path: str) -> bool:
    """True when git would refuse to stage *relative_path*.

    check-ignore exits 0 on a match and 1 on no match. Any other status is a
    real git failure and must not be read as "not ignored".
    """
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", relative_path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"git check-ignore failed on {relative_path!r} with "
            f"status {result.returncode}: {result.stderr.strip()}"
        )
    return result.returncode == 0


def _output_paths() -> dict[str, tuple[str, str]]:
    """Every results path tract.config declares, as (path, probe) pairs.

    Keyed by constant name so a failure names the constant an engineer has to
    go read, not just a path.

    A constant carrying a suffix names a single file, so the probe is that file
    itself. A constant with no suffix names a directory, and the probe is a
    plausible new file two levels beneath it, which is the shape a run actually
    writes: results/<phase>/<config>/<fold>/fold_result.json. Probing the
    directory itself would miss the case that broke here, where the directory
    is re-included but its contents are not.

    Deliberately does NOT call resolve() on the constant. The first version did,
    and it dropped PHASE1D_ARTIFACTS_PATH on the author's machine while keeping
    it in CI: results/phase1c/deployment_model/deployment_artifacts.npz is a
    symlink into the HuggingFace cache there, so resolve() followed it out of
    the repository and relative_to raised, which the loop swallowed as "not
    under results/". git does not follow the link when deciding what to ignore,
    so the declared path is the one that matters. A check that silently covers
    less on the machine where it is run is worse than no check.
    """
    results_root = REPO_ROOT / "results"
    found: dict[str, tuple[str, str]] = {}
    for name in dir(config):
        if not name.isupper():
            continue
        value = getattr(config, name)
        if not isinstance(value, Path):
            continue
        try:
            relative = value.relative_to(results_root)
        except ValueError:
            continue
        # The results root itself carries no output of its own.
        if str(relative) == ".":
            continue
        path = f"results/{relative}"
        probe = path if value.suffix else f"{path}/probe_run/probe_fold/fold_result.json"
        found[name] = (path, probe)
    return found


class TestEveryResultsDirectoryCanBeStaged:
    """A directory a run writes to must accept a new file, or say why not."""

    def test_config_declares_results_directories(self) -> None:
        """Guards the derivation itself.

        If tract.config stops exposing Path constants under results/, every
        other test in this file passes vacuously. This is the tripwire that
        turns that into a failure.
        """
        paths = _output_paths()
        assert paths, (
            "no results output paths found in tract.config; the "
            "derivation below would pass without checking anything"
        )
        assert "PHASE1B_RESULTS_DIR" in paths, (
            "PHASE1B_RESULTS_DIR is the path runpod_parallel collects into "
            f"and it must be covered; found {sorted(paths)}"
        )

    def test_no_results_constant_is_silently_dropped(self) -> None:
        """Counts by string prefix, which no symlink can bend.

        _output_paths uses relative_to, and its first version resolved the
        constant first, which silently dropped any path that happened to be a
        symlink out of the repository. This counts the same constants a second
        way, so a drop shows up as a mismatch rather than as coverage quietly
        shrinking on whichever machine has the symlink.
        """
        prefix = f"{REPO_ROOT / 'results'}/"
        expected = {
            name
            for name in dir(config)
            if name.isupper()
            and isinstance(getattr(config, name), Path)
            and str(getattr(config, name)).startswith(prefix)
        }
        assert expected == set(_output_paths()), (
            "constants under results/ counted by prefix do not match the ones "
            "_output_paths derived. Dropped: "
            f"{sorted(expected - set(_output_paths()))}. Extra: "
            f"{sorted(set(_output_paths()) - expected)}."
        )

    @pytest.mark.parametrize(
        ("constant", "path", "probe"),
        sorted((name, p, probe) for name, (p, probe) in _output_paths().items()),
    )
    def test_a_new_result_file_is_stageable(
        self, constant: str, path: str, probe: str
    ) -> None:
        """A fresh run's output must be visible to git, or excluded on purpose."""
        ignored = _is_ignored(probe)
        reason = _DELIBERATELY_EXCLUDED.get(path)

        if reason is not None:
            assert ignored, (
                f"{path} is listed in _DELIBERATELY_EXCLUDED ({reason}) "
                f"but git would stage {probe}. Either the .gitignore entry "
                f"went away or the exclusion is no longer wanted; remove the "
                f"_DELIBERATELY_EXCLUDED entry if the latter."
            )
            return

        assert not ignored, (
            f"{constant} points at {path}, but git would refuse to stage "
            f"{probe}. A run writing there finishes with its output on disk "
            f"and nothing to push, and `git status` stays clean the whole "
            f"time. Add a .gitignore negation for {path}, or record it "
            f"in _DELIBERATELY_EXCLUDED with the reason."
        )


class TestWeightsStayOut:
    """Re-including a results directory must not drag its tensors in."""

    @pytest.mark.parametrize("suffix", _MODEL_ARTIFACT_PROBES)
    def test_phase1b_model_artifacts_remain_excluded(self, suffix: str) -> None:
        """results/phase1b is 2.2 GB of weights against a few MB of evidence.

        The negation that makes fold results stageable is an allowlist for
        exactly this reason. The JSON probes matter most: adapter_config.json
        and tokenizer.json match the allowlist on their extension and have to
        be excluded by their directory instead. Tracking either without the
        tensors it describes reads like a checkpoint that is present when it
        is not.
        """
        probe = f"results/phase1b/probe_run/probe_fold/{suffix}"
        assert _is_ignored(probe), (
            f"git would stage {probe}. The results/phase1b negation has "
            f"widened past the evidence files it was scoped to and is now "
            f"admitting model artifacts."
        )
