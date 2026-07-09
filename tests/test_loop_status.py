"""Loop status panel tests."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

import yolo_agent.core.loop_status as loop_status_module
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.orchestrator import LoopOrchestrator
from yolo_agent.cli import main
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.execution_queue import ExecutionQueueStore
from yolo_agent.core.experiment_graph import ExperimentNode, ExperimentPlan
from yolo_agent.core.process_probe import ProcessProbeResult


def _make_task(path: Path) -> Path:
    task_path = path / "task.yaml"
    task_path.write_text(
        yaml.safe_dump(
            {
                "task_type": "detect",
                "scene": "generic",
                "class_names": ["object"],
                "primary_metric": {"name": "map50_95"},
                "secondary_metrics": [{"name": "latency_ms", "goal": "minimize"}],
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
    (image_dir / "img1.jpg").write_bytes(b"image")
    (label_dir / "img1.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "names:",
                "  - object",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_loop_status_shows_stage_queue_evidence_and_next_command(
    tmp_path: Path,
    capsys,  # type: ignore[no-untyped-def]
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    """The status command should print a compact progress panel for users."""
    monkeypatch.setattr(
        loop_status_module,
        "probe_command_process",
        lambda command: ProcessProbeResult(status="found", detail="pid=123 yolo.EXE", pid=123, name="yolo.EXE"),
    )
    task_path = _make_task(tmp_path)
    data_yaml = _make_dataset(tmp_path / "dataset")
    run_root = tmp_path / "runs"
    orchestrator = LoopOrchestrator.initialize(
        run_id="status-run",
        task_path=task_path,
        data_yaml=data_yaml,
        run_root=run_root,
    )
    candidate = CandidateConfig(
        candidate_id="baseline",
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
    )
    node = ExperimentNode(
        node_id="node_baseline",
        candidate_config=candidate,
        data_version="dataset-v1",
        command_spec=CommandSpec.ultralytics_train(
            model="yolo26n.pt",
            data=data_yaml,
            project=orchestrator.context.artifact_path("ultralytics"),
            name="node_baseline",
            epochs=10,
            imgsz=640,
            metadata={"training_budget_profile": "debug"},
        ),
    )
    ExperimentPlan(plan_id="status-plan", nodes=[node]).to_yaml(
        orchestrator.context.artifact_path("experiment_plan.yaml")
    )
    queue = orchestrator.enqueue()
    queue.items[0].mark_running()
    ExecutionQueueStore(orchestrator.context.run_dir).save(queue)
    orchestrator.evidence_store.log_candidate_metrics(
        run_id="status-run",
        candidate_id="baseline",
        node_id="node_baseline",
        metrics={"map50_95": 0.31, "latency_ms": 8.0},
        dataset_version="dataset-v1",
        source="test",
    )
    stdout_log = orchestrator.context.artifact_path("node_baseline_ultralytics_stdout.log")
    stdout_log.write_text(
        "\n".join(
            [
                "Epoch GPU_mem box_loss cls_loss Instances Size",
                "1/10 1.25G 0.10 0.20 12 640: 10%|#---------| 1/10 [00:01<00:09, 7.50it/s]",
                "2/10 1.30G 0.09 0.18 14 640: 20%|##--------| 2/10 [00:02<00:08, 8.25it/s]",
            ]
        ),
        encoding="utf-8",
    )
    runtime_jsonl = orchestrator.context.artifact_path("node_baseline_runtime_profile.jsonl")
    runtime_jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "record_type": "log_line",
                        "line": "2/10 1.30G 0.09 0.18 14 640: 20%|##--------| 2/10 [00:02<00:08, 8.25it/s]",
                        "metrics": {"runtime_stream_it_per_sec": 8.25},
                    }
                ),
                json.dumps(
                    {
                        "record_type": "gpu_sample",
                        "sample": {"gpu_util_percent": 72.0},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    assert main(["loop", "status", "--run", str(run_root / "status-run")]) == 0

    output = capsys.readouterr().out
    assert "YOLO Agent Status" in output
    assert "State:      debug training is running" in output
    assert "Progress:   training 2/10 (20%), epoch 2/10, GPU 72%, 8.25 it/s, ETA 00:08" in output
    assert "Trust:      none; debug only verifies the pipeline and is not effect evidence" in output
    assert "Active item" in output
    assert "Process:   found (pid=123 yolo.EXE)" in output
    assert "Recent training log" in output
    assert "Next:       wait for training to finish; evidence import runs after completion" in output
    assert "machine_status:" not in output
    assert "current_training_command=" not in output

    assert main(["loop", "status", "--run", str(run_root / "status-run"), "--verbose"]) == 0

    output = capsys.readouterr().out
    assert "YOLO Agent Status (verbose)" in output
    assert "Run:        status-run" in output
    assert "Loop" in output
    assert "Stage:     init (completed)" in output
    assert "Queue" in output
    assert "running=1" in output
    assert (
        "Heartbeat: node=node_baseline candidate=baseline progress=training:2/10(20%) "
        "epoch=2/10 it/s=8.25 gpu=72.0%"
    ) in output
    assert "Current item" in output
    assert "Status:    running" in output
    assert "Command:   yolo detect train" in output
    assert "1. Epoch GPU_mem box_loss cls_loss Instances Size" in output
    assert "3. 2/10 1.30G" in output
    assert "Metric records:    2" in output
    assert "Key metrics:       latency_ms=8.0 map50_95=0.31" in output
    assert "Next command: yolo-agent status --run" in output


def test_loop_status_cleans_ansi_and_wide_progress_glyphs(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """Status output should not leak ANSI escapes or mojibake-prone progress glyphs."""
    task_path = _make_task(tmp_path)
    data_yaml = _make_dataset(tmp_path / "dataset")
    orchestrator = LoopOrchestrator.initialize(
        run_id="ansi-run",
        task_path=task_path,
        data_yaml=data_yaml,
        run_root=tmp_path / "runs",
    )
    candidate = CandidateConfig(
        candidate_id="baseline",
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
    )
    node = ExperimentNode(
        node_id="node_ansi",
        candidate_config=candidate,
        data_version="dataset-v1",
        command_spec=CommandSpec.ultralytics_train(
            model="yolo26n.pt",
            data=data_yaml,
            project=orchestrator.context.artifact_path("ultralytics"),
            name="node_ansi",
            epochs=1,
            imgsz=640,
            metadata={"training_budget_profile": "debug"},
        ),
    )
    ExperimentPlan(plan_id="ansi-plan", nodes=[node]).to_yaml(
        orchestrator.context.artifact_path("experiment_plan.yaml")
    )
    queue = orchestrator.enqueue()
    queue.items[0].mark_running()
    ExecutionQueueStore(orchestrator.context.run_dir).save(queue)
    stdout_log = orchestrator.context.artifact_path("node_ansi_ultralytics_stdout.log")
    stdout_log.write_text(
        "\x1b[K\x1b[34m\x1b[1mtrain: \x1b[0mCaching images: 100% "
        "\u9239\u4f5d\u9232\u6523 1183/1183 17.0Kit/s\n",
        encoding="utf-8",
    )

    assert main(["loop", "status", "--run", str(orchestrator.context.run_dir)]) == 0

    output = capsys.readouterr().out
    assert "\x1b" not in output
    assert "\u9239" not in output
    assert "train: Caching images: 100%" in output


def test_loop_status_reports_stale_running_queue_when_process_missing(
    tmp_path: Path,
    capsys,  # type: ignore[no-untyped-def]
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    """A running queue item without a matching process should be called stale, not training."""
    monkeypatch.setattr(
        loop_status_module,
        "probe_command_process",
        lambda command: ProcessProbeResult(status="not_found", detail="no process matched marker"),
    )
    task_path = _make_task(tmp_path)
    data_yaml = _make_dataset(tmp_path / "dataset")
    orchestrator = LoopOrchestrator.initialize(
        run_id="stale-run",
        task_path=task_path,
        data_yaml=data_yaml,
        run_root=tmp_path / "runs",
    )
    node = ExperimentNode(
        node_id="node_stale",
        candidate_config=CandidateConfig(
            candidate_id="baseline",
            base_model="yolo26n.pt",
            scale="n",
            framework="ultralytics",
        ),
        data_version="dataset-v1",
        command_spec=CommandSpec.ultralytics_train(
            model="yolo26n.pt",
            data=data_yaml,
            project=orchestrator.context.artifact_path("ultralytics"),
            name="node_stale",
            metadata={"training_budget_profile": "debug"},
        ),
    )
    ExperimentPlan(plan_id="stale-plan", nodes=[node]).to_yaml(
        orchestrator.context.artifact_path("experiment_plan.yaml")
    )
    queue = orchestrator.enqueue()
    queue.items[0].mark_running()
    ExecutionQueueStore(orchestrator.context.run_dir).save(queue)

    assert main(["loop", "status", "--run", str(orchestrator.context.run_dir)]) == 0

    output = capsys.readouterr().out
    assert "State:      debug stale: no training process detected" in output
    assert "Progress:   no matching training process" in output
    assert "Process:   not found (no process matched marker)" in output
    assert "Next:       rerun the same optimize debug command; the stale queue will be requeued automatically" in output
