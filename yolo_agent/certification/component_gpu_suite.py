"""Priority runner for opt-in real GPU paper adapter certification."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.certification.component_gpu import GPU_CERTIFICATION_COMPONENTS
from yolo_agent.certification.component_runner import ComponentCertificationRunner
from yolo_agent.certification.component_schemas import ComponentCertificationReport
from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.components.distillation import DISTILLATION_COMPONENTS


class ComponentRunnerProtocol(Protocol):
    def run(self, **kwargs: object) -> ComponentCertificationReport: ...


class PaperComponentGPUResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    priority: int = Field(ge=1)
    component_id: str
    status: Literal["passed", "failed", "blocked", "not_run"]
    cpu_status: str | None = None
    gpu_status: str | None = None
    final_maturity: str | None = None
    cpu_report: Path | None = None
    gpu_report: Path | None = None
    reason: str | None = None


class PaperComponentGPUSuiteReport(BaseModel, YAMLModelMixin):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_component_gpu_suite.v1"
    status: Literal["passed", "failed", "blocked"]
    execute_real_gpu: bool
    model: str
    device: str
    registry_path: Path
    results: list[PaperComponentGPUResult] = Field(default_factory=list)
    stopped_at: str | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PaperComponentGPUSuiteRunner:
    """Certify high-value adapters sequentially under one explicit consent."""

    def __init__(self, runner: ComponentRunnerProtocol | None = None) -> None:
        self.runner = runner or ComponentCertificationRunner()

    def run(
        self,
        *,
        workdir: Path | str,
        registry_path: Path | str,
        model: str,
        device: str,
        teacher: str | None = None,
        ensemble_teacher: str | None = None,
        execute_real_gpu: bool = False,
    ) -> PaperComponentGPUSuiteReport:
        root = Path(workdir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        registry = Path(registry_path).resolve()
        if not execute_real_gpu:
            report = PaperComponentGPUSuiteReport(
                status="blocked",
                execute_real_gpu=False,
                model=model,
                device=device,
                registry_path=registry,
                results=[
                    PaperComponentGPUResult(
                        priority=index,
                        component_id=component_id,
                        status="not_run",
                        reason="gpu_execution_not_confirmed",
                    )
                    for index, component_id in enumerate(
                        GPU_CERTIFICATION_COMPONENTS,
                        start=1,
                    )
                ],
            )
            return self._write(root, report)

        results: list[PaperComponentGPUResult] = []
        stopped_at: str | None = None
        for index, component_id in enumerate(GPU_CERTIFICATION_COMPONENTS, start=1):
            if stopped_at is not None:
                results.append(
                    PaperComponentGPUResult(
                        priority=index,
                        component_id=component_id,
                        status="not_run",
                        reason=f"stopped_after:{stopped_at}",
                    )
                )
                continue
            component_root = root / component_id
            options = None
            if component_id in {
                "distillation.yolo26_teacher_student",
                *DISTILLATION_COMPONENTS,
            } and teacher:
                options = {"teacher": teacher}
                if component_id == "distillation.teacher_ensemble" and ensemble_teacher:
                    options["teachers"] = [ensemble_teacher]
            cpu = self.runner.run(
                component_id=component_id,
                mode="cpu",
                workdir=component_root,
                registry_path=registry,
                model=model,
                device="cpu",
                options=options,
                execute_gpu=False,
            )
            cpu_path = component_root / "component_certification.cpu.yaml"
            if cpu.status != "passed":
                stopped_at = component_id
                results.append(
                    PaperComponentGPUResult(
                        priority=index,
                        component_id=component_id,
                        status="failed",
                        cpu_status=cpu.status,
                        final_maturity=cpu.final_maturity,
                        cpu_report=cpu_path,
                        reason=cpu.errors[0] if cpu.errors else "cpu_smoke_failed",
                    )
                )
                continue
            gpu = self.runner.run(
                component_id=component_id,
                mode="gpu",
                workdir=component_root,
                registry_path=registry,
                model=model,
                device=device,
                options=options,
                execute_gpu=True,
            )
            gpu_path = component_root / "component_certification.gpu.yaml"
            passed = gpu.status == "passed" and gpu.final_maturity == "gpu_certified"
            if not passed:
                stopped_at = component_id
            results.append(
                PaperComponentGPUResult(
                    priority=index,
                    component_id=component_id,
                    status="passed" if passed else "failed",
                    cpu_status=cpu.status,
                    gpu_status=gpu.status,
                    final_maturity=gpu.final_maturity,
                    cpu_report=cpu_path,
                    gpu_report=gpu_path,
                    reason=(
                        None
                        if passed
                        else gpu.errors[0]
                        if gpu.errors
                        else "gpu_contract_failed"
                    ),
                )
            )
        report = PaperComponentGPUSuiteReport(
            status="passed" if stopped_at is None else "failed",
            execute_real_gpu=True,
            model=model,
            device=device,
            registry_path=registry,
            results=results,
            stopped_at=stopped_at,
        )
        return self._write(root, report)

    @staticmethod
    def _write(
        root: Path,
        report: PaperComponentGPUSuiteReport,
    ) -> PaperComponentGPUSuiteReport:
        path = root / "paper_component_gpu_suite.yaml"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        report.to_yaml(temporary, exclude_none=True, sort_keys=False)
        temporary.replace(path)
        return report


__all__ = [
    "PaperComponentGPUResult",
    "PaperComponentGPUSuiteReport",
    "PaperComponentGPUSuiteRunner",
]
