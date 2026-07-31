from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.core.run_initialization import (
    RunInitializationReport,
    RunInitializationTransaction,
    write_partial_run_migration_report,
)


def test_successful_initialization_writes_committed_status(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    with RunInitializationTransaction(run_root, "run-a") as transaction:
        (transaction.run_dir / "artifacts").mkdir(parents=True)
        (transaction.run_dir / "run_context.yaml").write_text(
            "run_id: run-a\n",
            encoding="utf-8",
        )
        status_path = transaction.commit()

    report = RunInitializationReport.from_yaml(status_path)
    assert report.status == "initialized"
    assert report.action == "continue"
    assert transaction.run_dir.is_dir()


def test_failed_initialization_is_archived_outside_active_namespace(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs"
    transaction = RunInitializationTransaction(run_root, "run-b")

    with pytest.raises(RuntimeError, match="broken initializer"):
        with transaction:
            transaction.run_dir.mkdir(parents=True)
            (transaction.run_dir / "task.yaml").write_text("partial: true\n")
            raise RuntimeError("broken initializer")

    assert not transaction.run_dir.exists()
    assert transaction.report_path is not None
    report = RunInitializationReport.from_yaml(transaction.report_path)
    assert report.status == "failed"
    assert report.action == "archived_failed_initialization"
    assert report.archived_run_dir is not None
    assert (report.archived_run_dir / "task.yaml").is_file()


def test_existing_partial_run_gets_migration_report_without_deletion(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs"
    partial = run_root / "run-c"
    partial.mkdir(parents=True)
    (partial / "task.yaml").write_text("partial: true\n", encoding="utf-8")

    path = write_partial_run_migration_report(run_root, "run-c", "run-c-1")

    assert path is not None
    report = RunInitializationReport.from_yaml(path)
    assert report.status == "partial_detected"
    assert report.allocated_run_id == "run-c-1"
    assert report.action == "preserve_and_allocate_new_run"
    assert (partial / "task.yaml").is_file()
