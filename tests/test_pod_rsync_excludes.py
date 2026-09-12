"""What reaches a rented GPU host, asserted as a property rather than a list.

Two orchestrators shipped the working tree with two hand-copied exclude lists.
`runpod_retrain`'s own comment admitted the first divergence -- it omitted
`.env`, `*.db`, `data/raw` and `.claude`, so crosswalk.db went to every pod. It
also excluded only `results/phase0` and `results/phase1b`, letting the Tier-3
quarantined review export and the ceiling study's 400 LLM-written hub
descriptions through.

Phase 2C adds a sharper case: two volunteers' pseudonymised judgements and their
verbatim rationales. The merged corpus an arm trains on is shipped deliberately,
to a SECURE host, and its content is what the volunteers consented to publish.
The RAW per-annotator rounds and their sidecars are not read on a pod at all,
so they have no business leaving this machine.

These tests assert the PROPERTY (nothing matching these paths survives the
filter), not the literal patterns -- `.pod_state.json` and `.pod_state.json.*`
were both present and neither matched `.pod_state_0r.json`, which is a live
fleet roster of every pod's ip, port and id.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import pytest

from tract.config import POD_RSYNC_EXCLUDES


def _excluded(relative: str) -> bool:
    """Approximate rsync's matching: any path component may match a pattern."""
    parts = Path(relative).parts
    for pattern in POD_RSYNC_EXCLUDES:
        if "/" in pattern:
            if relative == pattern or relative.startswith(pattern + "/"):
                return True
            continue
        for i, part in enumerate(parts):
            if fnmatch.fnmatch(part, pattern):
                # A directory pattern excludes everything beneath it.
                if i < len(parts) - 1 or True:
                    return True
    return False


class TestVolunteerTextStaysHome:
    @pytest.mark.parametrize("path", [
        "data/training/bridge/vol-01.jsonl",
        "data/training/bridge/vol-01.reviewed.json",
        "data/training/bridge-r2/vol-02.jsonl",
        "data/training/bridge-r2/vol-02.reviewed.json",
    ])
    def test_raw_annotation_rounds_are_excluded(self, path: str) -> None:
        """Never read on a pod. Every record carries a pseudonym and free text."""
        assert _excluded(path), f"{path} would be rsynced to a rented host"

    @pytest.mark.parametrize("path", [
        "data/training/hub_links_bridge.r2.jsonl",
        "data/training/hub_links_bridge.placebo.jsonl",
    ])
    def test_the_merged_corpus_still_ships(self, path: str) -> None:
        """Excluding this would make every bridge arm train on nothing.

        A firewall that removes the treatment is not a safer experiment, it is
        a silent null.
        """
        assert not _excluded(path), (
            f"{path} is the corpus an arm trains on; excluding it makes the "
            "treatment arm identical to the comparator"
        )


class TestQuarantinedAndScratchArtifacts:
    @pytest.mark.parametrize("path", [
        "results/review/review_export.json",
        "results/ceiling_study/hub_reference.md",
        "results/ceiling_study/ceiling_answer_key.json",
        "results/phase1b/anything/fold_result.json",
    ])
    def test_results_are_excluded_wholesale(self, path: str) -> None:
        """Nothing on a pod reads results/, and two of these are quarantined.

        review_export.json is Tier 3 and may never be a gate denominator;
        hub_reference.md is 400 LLM-written hub descriptions that CLAUDE.md
        forbids sending anywhere.
        """
        assert _excluded(path)

    @pytest.mark.parametrize("path", [
        "claudedocs/curation-package.md",
        "claudedocs/jetson-runpod-start.md",
        "docs/session-prompts.md",
    ])
    def test_scratch_and_recruiting_material_is_excluded(
        self, path: str
    ) -> None:
        """claudedocs is gitignored scratch.

        It carries no identities today. That is accidental rather than
        enforced, and it is exactly where a "who is vol-02 again?" note gets
        written.
        """
        assert _excluded(path)


class TestThePodStateGlobIsAProperty:
    @pytest.mark.parametrize("path", [
        ".pod_state.json",
        ".pod_state.json.corrupt",
        ".pod_state.json.abc123.tmp",
        ".pod_state_0r.json",
        "scripts/phase1c/.pod_state_retrain.json",
    ])
    def test_every_pod_state_variant_is_excluded(self, path: str) -> None:
        """The two literal patterns matched the first three and not the last two.

        A live roster holds every pod's ip, port and id, world-readable. Owning
        one pod then disclosed the rest -- the exact harm the original exclude
        was written for, reintroduced by a filename that did not match it.
        """
        assert _excluded(path)


class TestTheListIsShared:
    def test_both_orchestrators_import_it(self) -> None:
        """Rather than each carrying a copy that drifts."""
        for module in (
            "scripts/phase1b/runpod_parallel.py",
            "scripts/phase1c/runpod_retrain.py",
        ):
            source = Path(module).read_text(encoding="utf-8")
            assert "POD_RSYNC_EXCLUDES" in source, (
                f"{module} does not use the shared exclude list, so it will "
                "diverge again"
            )

    def test_secrets_and_weights_are_still_covered(self) -> None:
        """Regression guard on what the original lists already got right."""
        for path in (".env", "data/raw/opencre.json", "models/x.safetensors",
                     "crosswalk.db", ".git/config"):
            assert _excluded(path), path
