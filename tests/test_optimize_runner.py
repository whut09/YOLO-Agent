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
    assert queue.metadata["queue_source_plan_hash"] == plan["metadata"]["plan_hash"]


def test_optimize_rebuilds_stale_queue_when_profile_changes(tmp_path: Path) -> None:
    """Changing profile for an existing run should not reuse the old completed queue."""
    data_yaml = _make_dataset(tmp_path / "dataset")
    runner = OptimizeRunner()

    debug = runner.run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=tmp_path / "runs",
        profile="debug",
        execute=False,
    )
    debug_queue = ExecutionQueue.from_yaml(debug.queue_path)
    debug_hash = str(debug_queue.metadata["queue_source_plan_hash"])
    assert debug_queue.items[0].status == "completed"
    assert debug_queue.items[0].command.metadata["training_budget_profile"] == "debug"

    pilot = runner.run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=tmp_path / "runs",
        profile="pilot",
        execute=False,
    )

    pilot_plan = yaml.safe_load(pilot.experiment_plan_path.read_text(encoding="utf-8-sig"))
    pilot_queue = ExecutionQueue.from_yaml(pilot.queue_path)
    assert pilot.profile == "pilot"
    assert pilot_queue.metadata["queue_source_plan_hash"] == pilot_plan["metadata"]["plan_hash"]
    assert pilot_queue.metadata["queue_source_plan_hash"] != debug_hash
    assert pilot_queue.items[0].command.metadata["training_budget_profile"] == "pilot"
    assert pilot_queue.items[0].command.metadata["training_budget_epochs"] == 10


def test_optimize_advance_reuses_existing_run_context(tmp_path: Path) -> None:
    """advance should move an existing run to a new profile without restating data/model."""
    data_yaml = _make_dataset(tmp_path / "dataset")
    runner = OptimizeRunner()
    debug = runner.run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=tmp_path / "runs",
        profile="debug",
        execute=False,
    )
    debug_queue = ExecutionQueue.from_yaml(debug.queue_path)

    pilot = runner.advance(
        run_dir=tmp_path / "runs" / "coco-yolo26n",
        to_profile="pilot",
        execute=False,
    )

    pilot_plan = yaml.safe_load(pilot.experiment_plan_path.read_text(encoding="utf-8-sig"))
    pilot_queue = ExecutionQueue.from_yaml(pilot.queue_path)
    assert pilot.profile == "pilot"
    assert pilot.executed is False
    assert pilot_plan["metadata"]["model"] == "yolo26n.pt"
    assert pilot_plan["metadata"]["data_yaml"] == data_yaml.as_posix()
    assert pilot_queue.metadata["queue_source_plan_hash"] != debug_queue.metadata["queue_source_plan_hash"]
    assert pilot_queue.items[0].command.metadata["training_budget_profile"] == "pilot"


def test_optimize_full_profile_execute_requires_confirmation(tmp_path: Path) -> None:
    """Full COCO profiles should not execute unless the user confirms the budget."""
    data_yaml = _make_dataset(tmp_path / "dataset")

    result = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=tmp_path / "runs",
        profile="baseline_full",
        execute=True,
    )

    assert result.ok is False
    assert result.executed is False
    assert any(check.name == "confirm_full_run" and not check.ok for check in result.preflight)
    assert "--confirm-full-run" in result.next_action
    assert not result.experiment_plan_path.exists()


def test_optimize_full_profile_dry_run_does_not_require_confirmation(tmp_path: Path) -> None:
    """Dry-run planning for full profiles should remain available without confirmation."""
    data_yaml = _make_dataset(tmp_path / "dataset")

    result = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=tmp_path / "runs",
        profile="baseline_full",
        execute=False,
    )

    assert result.ok is True
    assert result.executed is False
    assert not any(check.name == "confirm_full_run" and not check.ok for check in result.preflight)
    assert result.experiment_plan_path.exists()


def test_optimize_advance_cli_runs_existing_run(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """The optimize advance CLI should expose a short profile transition command."""
    data_yaml = _make_dataset(tmp_path / "dataset")
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
    capsys.readouterr()

    assert main(
        [
            "optimize",
            "advance",
            "--run",
            str(tmp_path / "runs" / "cli-coco"),
            "--to-profile",
            "pilot",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "profile=pilot" in output
    assert "executed=False" in output
    assert "execution_queue=" in output
    assert f"next: yolo-agent loop status --run {tmp_path / 'runs' / 'cli-coco'}" in output
    queue = ExecutionQueue.from_yaml(tmp_path / "runs" / "cli-coco" / "execution_queue.yaml")
    assert queue.items[0].command.metadata["training_budget_profile"] == "pilot"


def test_optimize_cli_blocks_full_execute_without_confirmation(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """The CLI should make full COCO execution an explicit opt-in."""
    data_yaml = _make_dataset(tmp_path / "dataset")

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
            "--profile",
            "baseline_full",
            "--execute",
        ]
    ) == 1

    output = capsys.readouterr().out
    assert "preflight.confirm_full_run=error" in output
    assert "--confirm-full-run" in output
    assert not (tmp_path / "runs" / "cli-coco" / "artifacts" / "experiment_plan.yaml").exists()


def test_optimize_advance_cli_blocks_full_execute_without_confirmation(
    tmp_path: Path, capsys
) -> None:  # type: ignore[no-untyped-def]
    """The advance shortcut should not bypass full-run confirmation."""
    data_yaml = _make_dataset(tmp_path / "dataset")
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
    capsys.readouterr()

    assert main(
        [
            "optimize",
            "advance",
            "--run",
            str(tmp_path / "runs" / "cli-coco"),
            "--to-profile",
            "candidate_full",
            "--execute",
        ]
    ) == 1

    output = capsys.readouterr().out
    assert "profile=candidate_full" in output
    assert "preflight.confirm_full_run=error" in output
    assert "--confirm-full-run" in output


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
    assert f"next: yolo-agent loop status --run {tmp_path / 'runs' / 'cli-coco'}" in output
    assert (tmp_path / "runs" / "cli-coco" / "task.yaml").exists()
    task = yaml.safe_load((tmp_path / "runs" / "cli-coco" / "task.yaml").read_text(encoding="utf-8-sig"))
    assert task["primary_metric"]["name"] == "map50_95"
    plan = yaml.safe_load(
        (tmp_path / "runs" / "cli-coco" / "artifacts" / "experiment_plan.yaml").read_text(encoding="utf-8-sig")
    )
    assert plan["metadata"]["preset"] == "coco_yolo26_auto"
