"""Opt-in end-to-end acceptance for paper-driven automatic optimization."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Protocol

from yolo_agent.certification.paper_auto_optimization_research import (
    PaperAcceptanceResearchContext,
    PaperAcceptanceResearchPreparer,
)
from yolo_agent.certification.paper_auto_optimization_schemas import (
    PaperAutoOptimizationReport,
    PaperAutoOptimizationStage,
    PaperAutoOptimizationStageId,
    PaperAutoOptimizationStatus,
)
from yolo_agent.certification.runner import GpuAcceptanceBackend, UltralyticsGpuBackend


class PaperResearchPreparerProtocol(Protocol):
    def prepare(self, output_path: Path | str) -> PaperAcceptanceResearchContext: ...


class PaperAutoOptimizationAcceptanceSuite:
    """Certify one real paper recipe without granting full-run consent."""

    report_name = "paper_auto_optimization_report.yaml"

    def __init__(
        self,
        backend: GpuAcceptanceBackend | None = None,
        research_preparer: PaperResearchPreparerProtocol | None = None,
    ) -> None:
        self.backend = backend or UltralyticsGpuBackend()
        self.research_preparer = research_preparer

    def _resolve_preparer(
        self,
        *,
        research_root: Path | str,
        source: Path | str | None,
        maturity_registry: Path | str,
        source_commit: str | None,
    ) -> PaperResearchPreparerProtocol:
        if self.research_preparer is not None:
            return self.research_preparer
        return PaperAcceptanceResearchPreparer(
            research_root=research_root,
            source=source,
            maturity_registry=maturity_registry,
            source_commit=source_commit,
        )

    @classmethod
    def _write_report(
        cls,
        root: Path,
        report: PaperAutoOptimizationReport,
    ) -> PaperAutoOptimizationReport:
        path = root / cls.report_name
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        report.to_yaml(temporary, exclude_none=True, sort_keys=False)
        temporary.replace(path)
        return report


def _stage(
    stage_id: PaperAutoOptimizationStageId,
    *,
    status: PaperAutoOptimizationStatus = "passed",
    message: str = "",
    command: list[str] | None = None,
    artifacts: dict[str, str] | None = None,
    metrics: dict[str, Any] | None = None,
) -> PaperAutoOptimizationStage:
    now = datetime.now(timezone.utc)
    return PaperAutoOptimizationStage(
        stage_id=stage_id,
        status=status,
        message=message,
        command=command or [],
        artifacts=artifacts or {},
        metrics=metrics or {},
        started_at=now,
        completed_at=now,
    )


__all__ = [
    "PaperAutoOptimizationAcceptanceSuite",
    "PaperResearchPreparerProtocol",
]
