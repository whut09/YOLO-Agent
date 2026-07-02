"""Run orchestrator tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from yolo_agent.agents.orchestrator import LoopOrchestrator
from yolo_agent.cli import main
from yolo_agent.core.event_log import EventLog
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


def _make_metrics(path: Path) -> Path:
    metrics_path = path / "metrics.csv"
    metrics_path.write_text(
        "metric,value\nmap50,0.6\nrecall,0.7\nlatency_ms,12\nmodel_size_mb,5\n",
        encoding="utf-8",
    )
    return metrics_path


def _make_node_metrics(path: Path) -> Path:
    metrics_path = path / "node_metrics.csv"
    metrics_path.write_text(
        "\n".join(
            [
                "candidate_id,node_id,dataset_version,split,metric_name,value,source",
                "baseline,node_baseline,dataset-v1,val,map50,0.6,benchmark",
                "baseline,node_baseline,dataset-v1,val,recall,0.7,benchmark",
                "baseline,node_baseline,dataset-v1,val,latency_ms,12,benchmark",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return metrics_path


def _make_reordered_loop_policy(path: Path) -> Path:
    policy_path = path / "loop_policy.yaml"
    data = yaml.safe_load(Path("configs/loop_policy.yaml").read_text(encoding="utf-8"))
    stages = data["stages"]
    profile_index = next(index for index, stage in enumerate(stages) if stage["id"] == "profile_data")
    advise_index = next(index for index, stage in enumerate(stages) if stage["id"] == "advise_labels")
    stages[profile_index], stages[advise_index] = stages[advise_index], stages[profile_index]
    policy_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return policy_path


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
    assert state.dataset_version == "unversioned"
    assert state.task_spec == task_path
    assert "profile_data" in state.completed
    assert "advise_labels" in state.completed
    assert "missing_detection_errors" in state.blocked
    assert "dataset_report" in state.artifacts
    events = EventLog(orchestrator.context.run_dir / "events.jsonl").read()
    assert [event.event_type for event in events if event.stage == "diagnose_errors"][-1] == "contract_blocked"
    assert events[-1].details["missing_required"] == ["detection_errors"]


def test_loop_orchestrator_uses_policy_stage_order(tmp_path: Path) -> None:
    """Auto loop should use stage order from loop_policy.yaml, not a code constant."""
    task_path = _make_task(tmp_path)
    data_yaml = _make_dataset(tmp_path / "dataset")
    policy_path = _make_reordered_loop_policy(tmp_path)
    orchestrator = LoopOrchestrator.initialize(
        run_id="policy-order",
        task_path=task_path,
        data_yaml=data_yaml,
        run_root=tmp_path / "runs",
        loop_policy_path=policy_path,
    )

    results = orchestrator.run_until_blocked()

    assert orchestrator.state.stage_order[:3] == ["init", "advise_labels", "profile_data"]
    assert [result.stage for result in results[:2]] == ["advise_labels", "profile_data"]
    assert results[-1].stage == "diagnose_errors"
    assert results[-1].status == "blocked"


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
    assert (orchestrator.context.artifact_path("evidence_status.json")).exists()
    events = EventLog(orchestrator.context.run_dir / "events.jsonl").read()
    assert any(event.event_type == "stage_completed" and event.stage == "smoke" for event in events)
    assert events[-1].event_type == "contract_blocked"
    assert events[-1].stage == "import_metrics"


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


def test_loop_cli_resume_retries_blocked_stage(tmp_path: Path) -> None:
    """Loop resume should reset the first blocked stage and continue when evidence appears."""
    task_path = _make_task(tmp_path)
    data_yaml = _make_dataset(tmp_path / "dataset")
    errors_path = _make_errors(tmp_path)
    run_root = tmp_path / "runs"
    orchestrator = LoopOrchestrator.initialize(
        run_id="resume-run",
        task_path=task_path,
        data_yaml=data_yaml,
        run_root=run_root,
    )
    results = orchestrator.run_until_blocked()
    assert results[-1].stage == "diagnose_errors"
    assert results[-1].status == "blocked"

    orchestrator.context.detection_errors_path = errors_path
    orchestrator.context.to_yaml()

    assert main(["loop", "--run", str(run_root / "resume-run"), "--resume"]) == 0

    state = LoopState.from_yaml(run_root / "resume-run" / "loop_state.yaml")
    assert state.stages["diagnose_errors"].status == "completed"
    assert state.stages["import_metrics"].status == "blocked"
    assert "missing_metrics" in state.blocked
    assert "loop_diagnosis" in state.artifacts


def test_loop_cli_workflow_commands_run_without_training(tmp_path: Path) -> None:
    """Dedicated loop CLI commands should drive the harness in explicit phases."""
    task_path = _make_task(tmp_path)
    data_yaml = _make_dataset(tmp_path / "dataset")
    errors_path = _make_errors(tmp_path)
    metrics_path = _make_metrics(tmp_path)
    run_root = tmp_path / "runs"
    run_dir = run_root / "phase-run"

    assert main(
        [
            "loop",
            "init",
            "--run-id",
            "phase-run",
            "--task",
            str(task_path),
            "--data",
            str(data_yaml),
            "--run-root",
            str(run_root),
        ]
    ) == 0
    assert main(["loop", "diagnose", "--run", str(run_dir), "--errors", str(errors_path)]) == 0
    assert main(["loop", "plan", "--run", str(run_dir)]) == 0
    assert main(["loop", "smoke", "--run", str(run_dir)]) == 0
    assert main(["loop", "ingest-metrics", "--run", str(run_dir), "--metrics", str(metrics_path)]) == 0
    assert main(["loop", "next", "--run", str(run_dir)]) == 0

    assert (run_dir / "artifacts" / "loop_diagnosis.json").exists()
    assert (run_dir / "artifacts" / "policy_evaluation.yaml").exists()
    assert (run_dir / "artifacts" / "smoke_result.json").exists()
    assert (run_dir / "artifacts" / "metrics_import.json").exists()
    assert (run_dir / "report.md").exists()
    state = LoopState.from_yaml(run_dir / "loop_state.yaml")
    assert state.stages["next_round"].status == "completed"


def test_loop_ingest_metrics_persists_candidate_records(tmp_path: Path) -> None:
    """Loop metrics ingest should persist candidate/node-level metric evidence."""
    task_path = _make_task(tmp_path)
    data_yaml = _make_dataset(tmp_path / "dataset")
    errors_path = _make_errors(tmp_path)
    metrics_path = _make_node_metrics(tmp_path)
    run_root = tmp_path / "runs"
    run_dir = run_root / "node-metrics-run"

    assert main(
        [
            "loop",
            "init",
            "--run-id",
            "node-metrics-run",
            "--task",
            str(task_path),
            "--data",
            str(data_yaml),
            "--run-root",
            str(run_root),
            "--errors",
            str(errors_path),
        ]
    ) == 0
    assert main(["loop", "diagnose", "--run", str(run_dir)]) == 0
    assert main(["loop", "plan", "--run", str(run_dir)]) == 0
    assert main(["loop", "smoke", "--run", str(run_dir)]) == 0
    assert main(["loop", "ingest-metrics", "--run", str(run_dir), "--metrics", str(metrics_path)]) == 0

    evidence = LoopOrchestrator.from_run_dir(run_dir).evidence_store.load_run("node-metrics-run")
    assert (run_dir / "metrics_by_node.jsonl").exists()
    assert {record.metric_name: record.value for record in evidence.metric_records} == {
        "map50": 0.6,
        "recall": 0.7,
        "latency_ms": 12,
    }
    assert evidence.metric_records[0].candidate_id == "baseline"
    assert evidence.metric_records[0].node_id == "node_baseline"


def test_loop_auto_can_initialize_from_task_and_data(tmp_path: Path) -> None:
    """loop auto should initialize a run when task/data are provided."""
    task_path = _make_task(tmp_path)
    data_yaml = _make_dataset(tmp_path / "dataset")
    run_root = tmp_path / "runs"

    assert main(
        [
            "loop",
            "auto",
            "--run-id",
            "auto-run",
            "--task",
            str(task_path),
            "--data",
            str(data_yaml),
            "--run-root",
            str(run_root),
        ]
    ) == 0

    run_dir = run_root / "auto-run"
    assert (run_dir / "run_context.yaml").exists()
    state = LoopState.from_yaml(run_dir / "loop_state.yaml")
    assert state.stages["diagnose_errors"].status == "blocked"
    assert "missing_detection_errors" in state.blocked
