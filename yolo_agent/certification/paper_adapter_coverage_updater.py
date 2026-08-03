"""Atomic machine-local coverage refresh after adapter certification."""

from __future__ import annotations

import os
from pathlib import Path

from yolo_agent.tools.paper_adapter_coverage import (
    LocalPaperAdapterCoverageReport,
    PaperCatalogAudit,
    build_local_report,
)


class PaperAdapterCoverageUpdater:
    def __init__(
        self,
        audit_path: Path | str = Path("configs/paper_catalog_audit.yaml"),
    ) -> None:
        self.audit_path = Path(audit_path)

    def refresh(
        self,
        *,
        registry_path: Path,
        output_path: Path,
    ) -> LocalPaperAdapterCoverageReport:
        report = build_local_report(
            PaperCatalogAudit.from_yaml(self.audit_path),
            registry_path=registry_path,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(
            f".{output_path.name}.{os.getpid()}.tmp"
        )
        report.to_yaml(temporary, exclude_none=True, sort_keys=False)
        temporary.replace(output_path)
        return report


__all__ = ["PaperAdapterCoverageUpdater"]
