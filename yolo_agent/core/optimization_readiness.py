"""Preflight gate before automatic candidate exploration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from yolo_agent.certification.code_identity import certification_code_hash
from yolo_agent.certification.schemas import CertificationReport
from yolo_agent.core.yaml_io import YAMLModelMixin


ReadinessMode = Literal["dry_run", "certified_exploration", "blocked"]


class OptimizationReadinessResult(BaseModel, YAMLModelMixin):
    """Auditable decision about whether candidate exploration may start."""

    schema_version: str = "optimization_readiness.v1"
    ready: bool
    mode: ReadinessMode
    certification_report: Path | None = None
    certification_report_hash: str | None = None
    certified_code_hash: str | None = None
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    observed_capabilities: list[str] = Field(default_factory=list)


class OptimizationReadinessGate:
    """Block automatic search until a matching opt-in GPU certification exists."""

    required_capabilities = (
        "candidate_coco_error_facts",
        "error_delta_next_round",
        "asha_queue_control",
    )

    def __init__(self, *, default_report_name: str = "certification/mini-gpu/certification_report.yaml") -> None:
        self.default_report_name = default_report_name

    def evaluate(
        self,
        *,
        run_root: Path | str,
        execute: bool,
        require_certification: bool = True,
        report_path: Path | str | None = None,
    ) -> OptimizationReadinessResult:
        if not execute:
            return OptimizationReadinessResult(
                ready=True,
                mode="dry_run",
                warnings=["dry_run does not authorize candidate training"],
                required_capabilities=list(self.required_capabilities),
            )
        if not require_certification:
            return OptimizationReadinessResult(
                ready=True,
                mode="certified_exploration",
                warnings=["certification gate bypassed by an explicit test-only override"],
                required_capabilities=list(self.required_capabilities),
            )

        root = Path(run_root)
        path = Path(report_path) if report_path is not None else root / self.default_report_name
        blockers: list[str] = []
        observed: list[str] = []
        report: CertificationReport | None = None
        if not path.is_file():
            blockers.append("gpu_certification_report_missing")
        else:
            try:
                report = CertificationReport.load_verified(path)
            except (OSError, ValueError, TypeError) as exc:
                blockers.append(f"gpu_certification_report_invalid:{exc}")
        if report is not None:
            observed = sorted({claim.capability_id for claim in report.capability_claims})
            if report.status != "passed":
                blockers.append(f"gpu_certification_not_passed:{report.status}")
            if report.fixed_imgsz != 640:
                blockers.append("gpu_certification_imgsz_not_640")
            missing = sorted(set(self.required_capabilities) - set(observed))
            blockers.extend(f"gpu_certification_missing_capability:{item}" for item in missing)
            if not report.executed_recipe_id:
                blockers.append("gpu_certification_executed_recipe_missing")
            current_hash = certification_code_hash()
            if report.certified_code_hash != current_hash:
                blockers.append("gpu_certification_code_hash_mismatch")

        return OptimizationReadinessResult(
            ready=not blockers,
            mode="certified_exploration" if not blockers else "blocked",
            certification_report=path,
            certification_report_hash=report.report_hash if report is not None else None,
            certified_code_hash=report.certified_code_hash if report is not None else None,
            blockers=blockers,
            required_capabilities=list(self.required_capabilities),
            observed_capabilities=observed,
        )


__all__ = ["OptimizationReadinessGate", "OptimizationReadinessResult"]
