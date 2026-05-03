"""Tests for nb_helpers module."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _patch_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch PROJECT_ROOT and all derived paths to use tmp_path."""
    import notebooks.nb_helpers as helpers

    monkeypatch.setattr(helpers, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(helpers, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(helpers, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(helpers, "PHASE0_DIR", tmp_path / "results" / "phase0")
    monkeypatch.setattr(helpers, "PHASE1B_DIR", tmp_path / "results" / "phase1b")
    monkeypatch.setattr(helpers, "PHASE1C_DIR", tmp_path / "results" / "phase1c")
    monkeypatch.setattr(helpers, "REVIEW_DIR", tmp_path / "results" / "review")
    monkeypatch.setattr(helpers, "BRIDGE_DIR", tmp_path / "results" / "bridge")
    monkeypatch.setattr(helpers, "DATASET_DIR", tmp_path / "build" / "dataset")

    phase0 = tmp_path / "results" / "phase0"
    phase1b = tmp_path / "results" / "phase1b"
    phase1c = tmp_path / "results" / "phase1c"
    review = tmp_path / "results" / "review"
    data = tmp_path / "data"
    dataset = tmp_path / "build" / "dataset"
    monkeypatch.setattr(helpers, "PREREQUISITE_PATHS", [
        phase0 / "exp1_embedding_baseline_bge.json",
        phase0 / "exp2_llm_probe.json",
        phase0 / "exp5_knn_baseline.json",
        phase0 / "exp6_fewshot_sonnet.json",
        phase0 / "exp3_hierarchy_paths_bge.json",
        phase0 / "exp4_hub_descriptions.json",
        phase1b / "zero_shot_firewalled_baseline" / "aggregate_metrics.json",
        phase1c / "calibration" / "t_deploy_result.json",
        phase1c / "calibration" / "ece_gate.json",
        phase1c / "calibration" / "ood.json",
        phase1c / "deployment_model" / "deployment_artifacts.npz",
        data / "processed" / "cre_hierarchy.json",
        review / "review_metrics.json",
        review / "review_export.json",
        dataset / "crosswalk_v1.0.jsonl",
        dataset / "framework_metadata.json",
    ])


class TestFigureCounter:
    def test_first_figure_in_section(self) -> None:
        from notebooks.nb_helpers import FigureCounter

        fc = FigureCounter()
        assert fc.next(1) == "Figure 1.1"

    def test_sequential_figures(self) -> None:
        from notebooks.nb_helpers import FigureCounter

        fc = FigureCounter()
        fc.next(3)
        assert fc.next(3) == "Figure 3.2"

    def test_multiple_sections(self) -> None:
        from notebooks.nb_helpers import FigureCounter

        fc = FigureCounter()
        fc.next(1)
        fc.next(2)
        fc.next(1)
        assert fc.next(2) == "Figure 2.2"

    def test_current_returns_last(self) -> None:
        from notebooks.nb_helpers import FigureCounter

        fc = FigureCounter()
        fc.next(5)
        fc.next(5)
        assert fc.current(5) == "Figure 5.2"

    def test_current_raises_if_no_figures(self) -> None:
        from notebooks.nb_helpers import FigureCounter

        fc = FigureCounter()
        with pytest.raises(ValueError, match="No figures yet"):
            fc.current(1)

    def test_reset(self) -> None:
        from notebooks.nb_helpers import FigureCounter

        fc = FigureCounter()
        fc.next(1)
        fc.next(1)
        fc.reset()
        assert fc.next(1) == "Figure 1.1"


class TestStyleAxes:
    def test_applies_title_and_labels(self) -> None:
        import matplotlib.pyplot as plt

        from notebooks.nb_helpers import style_axes

        fig, ax = plt.subplots()
        style_axes(ax, "Test Title", "X", "Y", "Figure 1.1")
        assert "Figure 1.1" in ax.get_title()
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        plt.close(fig)


class TestPalette:
    def test_okabe_ito_has_8_colors(self) -> None:
        from notebooks.nb_helpers import OKABE_ITO

        assert len(OKABE_ITO) == 8

    def test_all_hex(self) -> None:
        from notebooks.nb_helpers import OKABE_ITO

        for color in OKABE_ITO:
            assert color.startswith("#")
            assert len(color) == 7


class TestPaths:
    def test_project_root_is_directory(self, tmp_path: Path) -> None:
        from notebooks.nb_helpers import PROJECT_ROOT

        assert PROJECT_ROOT == tmp_path

    def test_results_dir(self, tmp_path: Path) -> None:
        from notebooks.nb_helpers import RESULTS_DIR

        assert RESULTS_DIR == tmp_path / "results"


class TestLoadPhase0Experiment:
    def test_loads_json(self, tmp_path: Path) -> None:
        from notebooks.nb_helpers import load_phase0_experiment

        p0 = tmp_path / "results" / "phase0"
        p0.mkdir(parents=True)
        (p0 / "exp1_test.json").write_text(json.dumps({"key": "val"}), encoding="utf-8")
        result = load_phase0_experiment("exp1_test")
        assert result == {"key": "val"}

    def test_raises_on_missing(self) -> None:
        from notebooks.nb_helpers import load_phase0_experiment

        with pytest.raises(FileNotFoundError):
            load_phase0_experiment("nonexistent")


class TestLoadFirewalledBaseline:
    def test_loads_aggregate(self, tmp_path: Path) -> None:
        from notebooks.nb_helpers import load_firewalled_baseline

        d = tmp_path / "results" / "phase1b" / "zero_shot_firewalled_baseline"
        d.mkdir(parents=True)
        (d / "aggregate_metrics.json").write_text(
            json.dumps({"aggregate_hit1": {"mean": 0.399}}), encoding="utf-8"
        )
        result = load_firewalled_baseline()
        assert result["aggregate_hit1"]["mean"] == 0.399


class TestLoadFoldMetrics:
    def test_loads_fold(self, tmp_path: Path) -> None:
        from notebooks.nb_helpers import load_fold_metrics

        d = tmp_path / "results" / "phase1b" / "run1" / "fold_A"
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps({"hit_at_1": 0.5}), encoding="utf-8")
        result = load_fold_metrics("run1", "fold_A")
        assert result["hit_at_1"] == 0.5


class TestLoadTrainingLogs:
    def test_loads_from_last_checkpoint(self, tmp_path: Path) -> None:
        from notebooks.nb_helpers import load_training_logs

        fold_dir = tmp_path / "results" / "phase1b" / "run1" / "fold_A"
        ckpt = fold_dir / "checkpoint-100"
        ckpt.mkdir(parents=True)
        (ckpt / "trainer_state.json").write_text(
            json.dumps({"log_history": [{"epoch": 1, "loss": 0.5}]}), encoding="utf-8"
        )
        result = load_training_logs("run1", "fold_A")
        assert result == [{"epoch": 1, "loss": 0.5}]

    def test_raises_on_no_checkpoints(self, tmp_path: Path) -> None:
        from notebooks.nb_helpers import load_training_logs

        fold_dir = tmp_path / "results" / "phase1b" / "run1" / "fold_B"
        fold_dir.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="No checkpoint"):
            load_training_logs("run1", "fold_B")


class TestLoadCalibrationData:
    def test_loads_all_three(self, tmp_path: Path) -> None:
        from notebooks.nb_helpers import load_calibration_data

        cal = tmp_path / "results" / "phase1c" / "calibration"
        cal.mkdir(parents=True)
        (cal / "t_deploy_result.json").write_text(json.dumps({"temperature": 0.074}), encoding="utf-8")
        (cal / "ece_gate.json").write_text(json.dumps({"ece": 0.079}), encoding="utf-8")
        (cal / "ood.json").write_text(json.dumps({"threshold": 0.568}), encoding="utf-8")
        result = load_calibration_data()
        assert result["temperature"]["temperature"] == 0.074
        assert result["ece"]["ece"] == 0.079
        assert result["ood"]["threshold"] == 0.568


class TestLoadDeploymentEmbeddings:
    def test_loads_arrays(self, tmp_path: Path) -> None:
        from notebooks.nb_helpers import load_deployment_embeddings

        d = tmp_path / "results" / "phase1c" / "deployment_model"
        d.mkdir(parents=True)
        np.savez(
            str(d / "deployment_artifacts.npz"),
            hub_embeddings=np.zeros((5, 10)),
            control_embeddings=np.ones((3, 10)),
            hub_ids=np.array(["h1", "h2", "h3", "h4", "h5"]),
            control_ids=np.array(["c1", "c2", "c3"]),
        )
        hubs, controls, hub_ids, ctrl_ids = load_deployment_embeddings()
        assert hubs.shape == (5, 10)
        assert controls.shape == (3, 10)
        assert len(hub_ids) == 5
        assert len(ctrl_ids) == 3


class TestLoadReviewMetrics:
    def test_loads(self, tmp_path: Path) -> None:
        from notebooks.nb_helpers import load_review_metrics

        d = tmp_path / "results" / "review"
        d.mkdir(parents=True)
        (d / "review_metrics.json").write_text(
            json.dumps({"overall": {"accepted": 680}}), encoding="utf-8"
        )
        result = load_review_metrics()
        assert result["overall"]["accepted"] == 680


class TestLoadCREHierarchy:
    def test_loads(self, tmp_path: Path) -> None:
        from notebooks.nb_helpers import load_cre_hierarchy

        d = tmp_path / "data" / "processed"
        d.mkdir(parents=True)
        (d / "cre_hierarchy.json").write_text(
            json.dumps({"hubs": {}, "roots": [3], "label_space": []}), encoding="utf-8"
        )
        result = load_cre_hierarchy()
        assert result["roots"] == [3]


class TestLoadCrosswalk:
    def test_loads_jsonl(self, tmp_path: Path) -> None:
        from notebooks.nb_helpers import load_crosswalk

        d = tmp_path / "build" / "dataset"
        d.mkdir(parents=True)
        lines = [json.dumps({"control_id": f"c{i}"}) for i in range(3)]
        (d / "crosswalk_v1.0.jsonl").write_text("\n".join(lines), encoding="utf-8")
        result = load_crosswalk()
        assert len(result) == 3
        assert result[0]["control_id"] == "c0"


class TestCheckPrerequisites:
    def test_all_missing_returns_all(self) -> None:
        from notebooks.nb_helpers import check_prerequisites

        missing = check_prerequisites()
        assert len(missing) > 0

    def test_all_present_returns_empty(self, tmp_path: Path) -> None:
        from notebooks.nb_helpers import PREREQUISITE_PATHS, check_prerequisites

        for path in PREREQUISITE_PATHS:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".npz":
                np.savez(str(path), dummy=np.array([0]))
            else:
                path.write_text("{}", encoding="utf-8")
        assert check_prerequisites() == []
