"""Fresh base run-id allocation tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import yolo_agent.cli as cli
from yolo_agent.core.run_allocation import allocate_base_run_id


def test_available_run_id_is_used_without_a_suffix(tmp_path: Path) -> None:
    allocation = allocate_base_run_id(tmp_path / "runs", "coco-yolo26n")

    assert allocation.allocated_run_id == "coco-yolo26n"
    assert allocation.sequence == 0
    assert allocation.reason == "requested_id_available"
    assert allocation.changed is False


def test_existing_run_uses_next_monotonic_sequence(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    (run_root / "coco-yolo26n").mkdir(parents=True)
    (run_root / "coco-yolo26n-2").mkdir()
    (run_root / "coco-yolo26n-r38").mkdir()

    allocation = allocate_base_run_id(run_root, "coco-yolo26n")

    assert allocation.allocated_run_id == "coco-yolo26n-3"
    assert allocation.sequence == 3
    assert allocation.reason == "existing_run_directory"
    assert allocation.changed is True


def test_active_child_round_reuses_existing_base_run(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    (run_root / "coco-yolo26n").mkdir(parents=True)
    child = run_root / "coco-yolo26n-r1"
    child.mkdir()
    (child / "execution_queue.yaml").write_text(
        yaml.safe_dump({"items": [{"status": "needs_resume"}]}),
        encoding="utf-8",
    )

    allocation = allocate_base_run_id(run_root, "coco-yolo26n")

    assert allocation.allocated_run_id == "coco-yolo26n"
    assert allocation.reason == "existing_run_has_active_work"


def test_waiting_asha_trial_reuses_existing_base_run_without_queue(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    base = run_root / "coco-yolo26n"
    (base / "artifacts").mkdir(parents=True)
    (base / "artifacts" / "asha_state.yaml").write_text(
        yaml.safe_dump(
            {
                "trials": [
                    {"trial_id": "trial-a", "status": "waiting", "pending_stage": "pilot_3"}
                ],
                "assignments": [],
            }
        ),
        encoding="utf-8",
    )

    allocation = allocate_base_run_id(run_root, "coco-yolo26n")

    assert allocation.allocated_run_id == "coco-yolo26n"
    assert allocation.reason == "existing_run_has_active_work"


def test_completed_asha_state_does_not_prevent_fresh_numbered_run(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    base = run_root / "coco-yolo26n"
    (base / "artifacts").mkdir(parents=True)
    (base / "artifacts" / "asha_state.yaml").write_text(
        yaml.safe_dump(
            {
                "trials": [
                    {"trial_id": "trial-a", "status": "rejected", "pending_stage": None}
                ],
                "assignments": [{"assignment_id": "a", "status": "completed"}],
            }
        ),
        encoding="utf-8",
    )

    allocation = allocate_base_run_id(run_root, "coco-yolo26n")

    assert allocation.allocated_run_id == "coco-yolo26n-1"


def test_explicit_existing_run_is_not_renumbered(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    (run_root / "coco-yolo26n").mkdir(parents=True)

    allocation = allocate_base_run_id(run_root, "coco-yolo26n", reuse_existing=True)

    assert allocation.allocated_run_id == "coco-yolo26n"
    assert allocation.reason == "explicit_existing_run"


def test_beginner_train_command_uses_allocated_run_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root = tmp_path / "runs"
    (run_root / "coco-yolo26n").mkdir(parents=True)
    captured: dict[str, object] = {}

    def fake_run_optimize_command(args: object) -> int:
        captured["args"] = args
        return 0

    monkeypatch.setattr(cli, "run_optimize_command", fake_run_optimize_command)
    args = SimpleNamespace(
        run_root=run_root,
        run_id="coco-yolo26n",
        profile=None,
        confirm_full_run=False,
        auto_rounds=0,
        kind="coco",
        dry_run=True,
    )

    assert cli.run_train_command(args) == 0

    resolved = captured["args"]
    assert getattr(resolved, "run_id") == "coco-yolo26n-1"
    assert getattr(resolved, "run_allocation").requested_run_id == "coco-yolo26n"


@pytest.mark.parametrize("run_id", ["", ".", "..", "nested/run", "nested\\run"])
def test_invalid_run_id_is_rejected(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        allocate_base_run_id(tmp_path / "runs", run_id)
