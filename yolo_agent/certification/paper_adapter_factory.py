"""Resumable factory for independent reusable adapter certification."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from yolo_agent.certification.component_runner import ComponentCertificationRunner
from yolo_agent.certification.component_schemas import ComponentCertificationReport
from yolo_agent.certification.paper_adapter_discovery import (
    ReusableAdapterDescriptor,
    ReusableAdapterDiscoveryResult,
    ReusablePaperAdapterDiscovery,
)
from yolo_agent.certification.paper_adapter_factory_schemas import (
    BatchCertificationMode,
    PaperAdapterCertificationReport,
    PaperAdapterCertificationResult,
)


class AdapterDiscoveryProtocol(Protocol):
    def discover(self) -> ReusableAdapterDiscoveryResult: ...


class AdapterCertificationRunnerProtocol(Protocol):
    def run(self, **kwargs: object) -> ComponentCertificationReport: ...


class PaperAdapterCertificationFactory:
    """Certify reusable adapters independently under one batch checkpoint."""

    report_name = "paper_adapter_certification.yaml"

    def __init__(
        self,
        *,
        discovery: AdapterDiscoveryProtocol | None = None,
        runner: AdapterCertificationRunnerProtocol | None = None,
    ) -> None:
        self.discovery = discovery or ReusablePaperAdapterDiscovery()
        self.runner = runner or ComponentCertificationRunner()

    def run(
        self,
        *,
        workdir: Path | str,
        registry_path: Path | str,
        mode: BatchCertificationMode = "cpu",
        model: str = "yolo26n.pt",
        data: str = "coco.yaml",
        device: str = "0",
        execute_real_gpu: bool = False,
        resume: bool = False,
        changed_only: bool = False,
        component_ids: list[str] | None = None,
        options_by_component: dict[str, dict[str, object]] | None = None,
    ) -> PaperAdapterCertificationReport:
        root = Path(workdir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        registry = Path(registry_path).resolve()
        discovered = self.discovery.discover()
        requested = set(component_ids or [])
        descriptors = [
            item
            for item in discovered.adapters
            if not requested or item.component_id in requested
        ]
        selected_ids = [item.component_id for item in descriptors]
        unknown = sorted(requested - set(selected_ids))
        discovery_errors = dict(discovered.errors)
        for component_id in unknown:
            discovery_errors.setdefault(
                component_id, "requested reusable adapter was not discovered"
            )

        report = PaperAdapterCertificationReport(
            status="blocked" if not descriptors else "passed",
            mode=mode,
            execute_real_gpu=execute_real_gpu,
            resume=resume,
            changed_only=changed_only,
            registry_path=registry,
            selected_component_ids=selected_ids,
            results=[],
            discovery_errors=discovery_errors,
        )
        self._write_report(root, report)
        options = options_by_component or {}
        results: list[PaperAdapterCertificationResult] = []
        for descriptor in descriptors:
            component_root = root / _component_directory(descriptor.component_id)
            component_root.mkdir(parents=True, exist_ok=True)
            result = self._certify_one(
                descriptor=descriptor,
                component_root=component_root,
                registry_path=registry,
                mode=mode,
                model=model,
                data=data,
                device=device,
                execute_real_gpu=execute_real_gpu,
                options=options.get(descriptor.component_id),
            )
            result.to_yaml(
                component_root / "batch_result.yaml",
                exclude_none=True,
                sort_keys=False,
            )
            results.append(result)
            report = report.model_copy(
                update={
                    "status": _batch_status(results),
                    "results": list(results),
                    "report_hash": "",
                }
            )
            report = PaperAdapterCertificationReport.model_validate(
                report.model_dump(mode="json", exclude={"report_hash"})
            )
            self._write_report(root, report)
        return report

    def _certify_one(
        self,
        *,
        descriptor: ReusableAdapterDescriptor,
        component_root: Path,
        registry_path: Path,
        mode: BatchCertificationMode,
        model: str,
        data: str,
        device: str,
        execute_real_gpu: bool,
        options: dict[str, object] | None,
    ) -> PaperAdapterCertificationResult:
        adapter = descriptor
        initial = adapter.contract.maturity
        if mode == "gpu" and not execute_real_gpu:
            return PaperAdapterCertificationResult(
                component_id=adapter.component_id,
                identity=adapter.identity,
                status="blocked",
                initial_maturity=initial,
                final_maturity=initial,
                selection_reason="gpu_execution_not_confirmed",
                errors=["gpu_execution_not_confirmed"],
            )
        try:
            cpu = self.runner.run(
                component_id=adapter.component_id,
                mode="cpu",
                workdir=component_root,
                registry_path=registry_path,
                model=model,
                data=data,
                device="cpu",
                protocol_hash=adapter.identity.protocol_hash,
                options=options,
                execute_gpu=False,
            )
            cpu_path = component_root / "component_certification.cpu.yaml"
            if cpu.status != "passed":
                return _result_from_report(
                    adapter=adapter,
                    report=cpu,
                    selection_reason="cpu_certification_failed",
                    cpu_report=cpu_path,
                )
            if mode == "cpu":
                return _result_from_report(
                    adapter=adapter,
                    report=cpu,
                    selection_reason="selected_all_reusable_adapters",
                    cpu_report=cpu_path,
                )
            gpu = self.runner.run(
                component_id=adapter.component_id,
                mode="gpu",
                workdir=component_root,
                registry_path=registry_path,
                model=model,
                data=data,
                device=device,
                protocol_hash=adapter.identity.protocol_hash,
                options=options,
                execute_gpu=True,
            )
            return _result_from_report(
                adapter=adapter,
                report=gpu,
                selection_reason="selected_explicit_gpu_batch",
                cpu_report=cpu_path,
                gpu_report=component_root / "component_certification.gpu.yaml",
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return PaperAdapterCertificationResult(
                component_id=adapter.component_id,
                identity=adapter.identity,
                status="failed",
                initial_maturity=initial,
                final_maturity=initial,
                selection_reason="certification_runner_exception",
                errors=[str(exc)],
            )

    @classmethod
    def _write_report(
        cls,
        root: Path,
        report: PaperAdapterCertificationReport,
    ) -> Path:
        path = root / cls.report_name
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        report.to_yaml(temporary, exclude_none=True, sort_keys=False)
        temporary.replace(path)
        return path


def _result_from_report(
    *,
    adapter: ReusableAdapterDescriptor,
    report: ComponentCertificationReport,
    selection_reason: str,
    cpu_report: Path | None = None,
    gpu_report: Path | None = None,
) -> PaperAdapterCertificationResult:
    return PaperAdapterCertificationResult(
        component_id=adapter.component_id,
        identity=adapter.identity,
        status=report.status,
        initial_maturity=report.initial_maturity,
        final_maturity=report.final_maturity,
        selection_reason=selection_reason,
        cpu_report=cpu_report,
        gpu_report=gpu_report,
        errors=list(report.errors),
    )


def _batch_status(
    results: list[PaperAdapterCertificationResult],
) -> str:
    failures = [item for item in results if item.status in {"failed", "blocked"}]
    successes = [
        item
        for item in results
        if item.status in {"passed", "skipped_resume", "skipped_unchanged"}
    ]
    if failures and successes:
        return "partial"
    if failures:
        return "failed"
    return "passed"


def _component_directory(component_id: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in component_id
    ).strip(".-") or "unknown-component"


__all__ = [
    "AdapterCertificationRunnerProtocol",
    "AdapterDiscoveryProtocol",
    "PaperAdapterCertificationFactory",
]
