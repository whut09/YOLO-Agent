"""CLI wrapper for resolving real frozen YOLO26 teacher assets."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.certification.distillation import (
    resolve_teacher_asset_report,
)
from yolo_agent.components.adapters.distillation.teacher_asset_resolver import (
    TeacherAssetReport,
)


def run_teacher_asset_resolution(
    *,
    model: str = "yolo26n.pt",
    data: Path | str,
    output_path: Path | str = Path("runs/paper-readiness/teacher_assets.yaml"),
    download_dir: Path | str | None = None,
    allow_download: bool = True,
) -> TeacherAssetReport:
    """Resolve and persist official teachers without starting training."""

    return resolve_teacher_asset_report(
        student_model=model,
        data=data,
        output_path=output_path,
        download_dir=download_dir,
        allow_download=allow_download,
    )


__all__ = ["run_teacher_asset_resolution"]
