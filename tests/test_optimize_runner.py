"""One-command optimize runner tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from yolo_agent.agents.optimize_runner import OptimizeRunner
from yolo_agent.cli import COMMANDS, main
from yolo_agent.core.execution_queue import ExecutionQueue


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
                "  0: object",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_optimize_coco_prepares_debug_queue_without_execute(tmp_path: Path) -> None:
    """optimize coco should prepare a safe debug run without starting training by default."""
    data_yaml = _make_dataset(tmp_path / "dataset")

    result = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=tmp_path / "runs",
        profile="debug",
        execute=False,
    )

    assert result.ok is True
    assert result.executed is False
    assert result.profile == "debug"
    assert result.task_path.exists()
    assert result.experiment_plan_path.exists()
    assert result.queue_path.exists()
    assert result.report_path is not None and result.report_path.exists()
    assert result.training_loop is not None
    assert result.training_loop.stopped_reason == "next_round_blocked"
    assert result.queue_counts["completed"] == 1
    assert "Rerun with --execute" in result.next_action
    plan = yaml.safe_load(result.experiment_plan_path.read_text(encoding="utf-8-sig"))
    assert plan["metadata"]["preset"] is None
    queue = ExecutionQueue.from_yaml(result.queue_path)
    assert queue.items[0].status == "completed"
    assert queue.items[0].command.command_type == "train"
    assert queue.items[0].command.metadata["training_budget_profile"] == "debug"


def test_optimize_cli_runs_coco_dry_run(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """The top-level optimize CLI should be available and user-facing."""
    data_yaml = _make_dataset(tmp_path / "dataset")

    assert "optimize" in COMMANDS
    assert main(
        [
            "optimize",
            "coco",
            "--data",
            str(data_yaml),
            "--run-id",
            "cli-coco",
            "--run-root",
            str(tmp_path / "runs"),
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "run_dir=" in output
    assert "preset=coco_yolo26_auto" in output
    assert "profile=debug" in output
    assert "executed=False" in output
    assert "execution_queue=" in output
    assert (tmp_path / "runs" / "cli-coco" / "task.yaml").exists()
    task = yaml.safe_load((tmp_path / "runs" / "cli-coco" / "task.yaml").read_text(encoding="utf-8-sig"))
    assert task["primary_metric"]["name"] == "map50_95"
    plan = yaml.safe_load(
        (tmp_path / "runs" / "cli-coco" / "artifacts" / "experiment_plan.yaml").read_text(encoding="utf-8-sig")
    )
    assert plan["metadata"]["preset"] == "coco_yolo26_auto"
