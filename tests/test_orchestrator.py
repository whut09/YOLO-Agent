"""Run orchestrator tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from yolo_agent.agents.orchestrator import LoopOrchestrator
from yolo_agent.cli import main
from yolo_agent.core.loop_state import LoopState


def _make_task(path: Path) -> Path:
    task_path = path / "task.yaml"
    task_path.write_text(
        yaml.safe_dump(
            {
                "task_type": "detect",
                "scene": "infrared_small_target",
                "class_names": ["target"],
                "primary_metric": {"name": "recall"},
                "secondary_metrics": [{"name": "map50_95"}, {"name": "latency_ms", "goal": "minimize"}],
                "max_latency_ms": 30,
                "max_model_size_mb": 20,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return task_path


def _make_dataset(root: Path) -> Path:
    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "img1.jpg").write_bytes(b"image-1")
    (image_dir / "img2.jpg").write_bytes(b"image-2")
    (label_dir / "img1.txt").write_text("0 0.5 0.5 0.03 0.03\n", encoding="utf-8")
    (label_dir / "img2.txt").write_text("", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                "path: .",
                "scene: infrared_small_target",
                "train: images/train",
                "names:",
                "  - target",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def _make_errors(path: Path) -> Path:
    errors_path = path / "errors.yaml"
    errors_path.write_text(
        yaml.safe_dump(
            {
                "errors": [
                    {"error_type": "small_object_miss", "count": 4, "severity": "high"},
                    {"error_type": "background_confusion", "count": 2, "severity": "medium"},
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return errors_path


def test_loop_orchestrator_blocks_when_detection_errors_are_missing(tmp_path: Path) -> None:
    """Auto loop should stop at diagnose_errors when required error evidence is absent."""
    task_path = _make_task(tmp_path)
    data_yaml = _make_dataset(tmp_path / "dataset")
    orchestrator = LoopOrchestrator.initialize(
        run_id="missing-errors",
        task_path=task_path,
        data_yaml=data_yaml,
        run_root=tmp_path / "runs",
    )

    results = orchestrator.run_until_blocked()

    assert [result.stage for result in results] == ["profile_data", "advise_labels", "diagnose_errors"]
    assert results[-1].status == "blocked"
    assert (orchestrator.context.artifact_path("dataset_report.json")).exists()
    state = LoopState.from_yaml(orchestrator.context.run_dir / "loop_state.yaml")
    assert state.stages["diagnose_errors"].status == "blocked"


def test_loop_orchestrator_runs_harness_until_metrics_import_block(tmp_path: Path) -> None:
    """With errors available, the loop should produce plans and stop before missing metrics."""
    task_path = _make_task(tmp_path)
    data_yaml = _make_dataset(tmp_path / "dataset")
    errors_path = _make_errors(tmp_path)
    orchestrator = LoopOrchestrator.initialize(
        run_id="loop-run",
        task_path=task_path,
        data_yaml=data_yaml,
        run_root=tmp_path / "runs",
        detection_errors_path=errors_path,
    )

    results = orchestrator.run_until_blocked()

    assert results[-1].stage == "import_metrics"
    assert results[-1].status == "blocked"
    assert (orchestrator.context.artifact_path("loop_diagnosis.json")).exists()
    assert (orchestrator.context.artifact_path("loop_plan.yaml")).exists()
    assert (orchestrator.context.artifact_path("policy_evaluation.yaml")).exists()
    assert (orchestrator.context.run_dir / "plan.yaml").exists()
    assert (orchestrator.context.run_dir / "ablation_plan.yaml").exists()
    assert (orchestrator.context.artifact_path("smoke_result.json")).exists()


def test_loop_cli_init_and_run_stage(tmp_path: Path) -> None:
    """Loop CLI should initialize a run and execute one state-machine stage."""
    task_path = _make_task(tmp_path)
    data_yaml = _make_dataset(tmp_path / "dataset")
    run_root = tmp_path / "runs"

    assert main(
        [
            "loop",
            "init",
            "--run-id",
            "cli-run",
            "--task",
            str(task_path),
            "--data",
            str(data_yaml),
            "--run-root",
            str(run_root),
        ]
    ) == 0
    assert main(["loop", "run-stage", "--run", str(run_root / "cli-run"), "--stage", "profile_data"]) == 0
    assert (run_root / "cli-run" / "artifacts" / "dataset_report.json").exists()
