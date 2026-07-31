"""Transactional run initialization and partial-run migration artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from types import TracebackType
from typing import Literal

from pydantic import BaseModel, Field

from yolo_agent.core.yaml_io import YAMLModelMixin


class RunInitializationReport(BaseModel, YAMLModelMixin):
    """Auditable outcome for a completed, failed, or pre-existing partial run."""

    schema_version: str = "run_initialization_report.v1"
    run_id: str
    status: Literal["initialized", "failed", "partial_detected"]
    run_dir: Path
    action: Literal[
        "continue",
        "archived_failed_initialization",
        "preserve_and_allocate_new_run",
    ]
    reason: str
    allocated_run_id: str | None = None
    archived_run_dir: Path | None = None
    detected_files: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RunInitializationTransaction:
    """Remove failed initialization remnants from the active run namespace."""

    def __init__(self, run_root: Path | str, run_id: str) -> None:
        self.run_root = Path(run_root)
        self.run_id = run_id
        self.run_dir = self.run_root / run_id
        self.owned = not self.run_dir.exists()
        self.committed = False
        self.report_path: Path | None = None

    def __enter__(self) -> "RunInitializationTransaction":
        return self

    def commit(self) -> Path:
        if not self.run_dir.is_dir():
            raise RuntimeError(f"run initialization did not create {self.run_dir}")
        report = RunInitializationReport(
            run_id=self.run_id,
            status="initialized",
            run_dir=self.run_dir.resolve(),
            action="continue",
            reason="task_and_run_context_initialized",
            detected_files=_relative_files(self.run_dir),
        )
        self.report_path = report.to_yaml(
            self.run_dir / "artifacts" / "run_initialization_status.yaml",
            exclude_none=True,
            sort_keys=False,
        )
        self.committed = True
        return self.report_path

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, traceback
        if exc is None or self.committed or not self.owned or not self.run_dir.exists():
            return False
        archive = _available_failure_archive(self.run_root, self.run_id)
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.run_dir, archive)
        report = RunInitializationReport(
            run_id=self.run_id,
            status="failed",
            run_dir=self.run_dir.resolve(),
            archived_run_dir=archive.resolve(),
            action="archived_failed_initialization",
            reason=f"{type(exc).__name__}: {exc}",
            detected_files=_relative_files(archive),
        )
        report.to_yaml(
            archive / "initialization_failure_report.yaml",
            exclude_none=True,
            sort_keys=False,
        )
        self.report_path = report.to_yaml(
            self.run_root / "initialization_failures" / f"{archive.name}.yaml",
            exclude_none=True,
            sort_keys=False,
        )
        return False


def write_partial_run_migration_report(
    run_root: Path | str,
    requested_run_id: str,
    allocated_run_id: str,
) -> Path | None:
    """Report a directory that exists without a loadable run context."""
    run_dir = Path(run_root) / requested_run_id
    if not run_dir.is_dir() or (run_dir / "run_context.yaml").is_file():
        return None
    report = RunInitializationReport(
        run_id=requested_run_id,
        status="partial_detected",
        run_dir=run_dir.resolve(),
        allocated_run_id=allocated_run_id,
        action="preserve_and_allocate_new_run",
        reason="run_directory_exists_without_run_context",
        detected_files=_relative_files(run_dir),
    )
    return report.to_yaml(
        run_dir / "artifacts" / "run_initialization_migration.yaml",
        exclude_none=True,
        sort_keys=False,
    )


def _available_failure_archive(run_root: Path, run_id: str) -> Path:
    base = run_root / ".failed_initializations" / run_id
    candidate = base
    sequence = 1
    while candidate.exists():
        candidate = base.with_name(f"{run_id}-{sequence}")
        sequence += 1
    return candidate


def _relative_files(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    )


__all__ = [
    "RunInitializationReport",
    "RunInitializationTransaction",
    "write_partial_run_migration_report",
]
