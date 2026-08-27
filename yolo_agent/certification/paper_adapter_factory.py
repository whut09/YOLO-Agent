"""Resumable factory for independent reusable adapter certification."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

from yolo_agent.certification.component_runner import ComponentCertificationRunner
from yolo_agent.certification.component_schemas import ComponentCertificationReport
from yolo_agent.certification.distillation_paper_routes import (
    DistillationPaperRouteCertificationSummary,
    certify_all_paper_routes,
)
from yolo_agent.certification.matched_pilot_fixture import MatchedPilotFixtureBuilder
from yolo_agent.certification.paper_adapter_discovery import (
    ReusableAdapterDescriptor,
    ReusableAdapterDiscoveryResult,
    ReusablePaperAdapterDiscovery,
)
from yolo_agent.certification.paper_adapter_coverage_updater import (
    PaperAdapterCoverageUpdater,
)
from yolo_agent.certification.paper_adapter_factory_schemas import (
    AdapterCertificationIdentity,
    BatchCertificationMode,
    PaperAdapterCertificationReport,
    PaperAdapterCertificationResult,
)


class AdapterDiscoveryProtocol(Protocol):
    def discover(self) -> ReusableAdapterDiscoveryResult: ...


class AdapterCertificationRunnerProtocol(Protocol):
    def run(self, **kwargs: object) -> ComponentCertificationReport: ...


class MatchedPilotFixtureBuilderProtocol(Protocol):
    def build(self, **kwargs: object) -> object: ...


class AdapterCoverageUpdaterProtocol(Protocol):
    def refresh(self, **kwargs: object) -> object: ...


class PaperAdapterCertificationFactory:
    """Certify reusable adapters independently under one batch checkpoint."""

    report_name = "paper_adapter_certification.yaml"

    def __init__(
        self,
        *,
        discovery: AdapterDiscoveryProtocol | None = None,
        runner: AdapterCertificationRunnerProtocol | None = None,
        fixture_builder: MatchedPilotFixtureBuilderProtocol | None = None,
        coverage_updater: AdapterCoverageUpdaterProtocol | None = None,
    ) -> None:
        self.discovery = discovery or ReusablePaperAdapterDiscovery()
        self.runner = runner or ComponentCertificationRunner()
        self.fixture_builder = fixture_builder or MatchedPilotFixtureBuilder()
        self.coverage_updater = coverage_updater or PaperAdapterCoverageUpdater()

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
        previous = (
            self._load_previous(root) if resume or changed_only else None
        )
        previous_results = {
            item.component_id: item for item in previous.results
        } if previous is not None else {}
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
            status="blocked" if not descriptors else "partial",
            mode=mode,
            execute_real_gpu=execute_real_gpu,
            resume=resume,
            changed_only=changed_only,
            registry_path=registry,
            resumed_from_report_hash=(
                previous.report_hash if previous is not None else None
            ),
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
            prior = previous_results.get(descriptor.component_id)
            result = _selection_result(
                descriptor=descriptor,
                prior=prior,
                mode=mode,
                resume=resume,
                changed_only=changed_only,
            )
            if result is None:
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
                    selection_reason=_run_selection_reason(
                        prior, descriptor.identity, changed_only=changed_only
                    ),
                )
            result.to_yaml(
                component_root / "batch_result.yaml",
                exclude_none=True,
                sort_keys=False,
            )
            results.append(result)
            report = report.model_copy(
                update={
                    "status": _batch_status(
                        results, expected_count=len(descriptors)
                    ),
                    "results": list(results),
                    "report_hash": "",
                }
            )
            report = PaperAdapterCertificationReport.model_validate(
                report.model_dump(mode="json", exclude={"report_hash"})
            )
            self._write_report(root, report)
        coverage_path = root / "paper_adapter_coverage.yaml"
        coverage_error: str | None = None
        try:
            self.coverage_updater.refresh(
                registry_path=registry,
                output_path=coverage_path,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            coverage_error = str(exc)
        final_status = report.status
        relevant_discovery_errors = (
            discovery_errors
            if not requested
            else {
                component_id: message
                for component_id, message in discovery_errors.items()
                if component_id in requested
            }
        )
        if relevant_discovery_errors and final_status == "passed":
            final_status = "partial"
        if coverage_error and final_status == "passed":
            final_status = "partial"
        report = PaperAdapterCertificationReport.model_validate(
            report.model_dump(
                mode="json",
                exclude={"report_hash", "coverage_report_path", "coverage_error"},
            )
            | {
                "status": final_status,
                "coverage_report_path": (
                    str(coverage_path) if coverage_error is None else None
                ),
                "coverage_error": coverage_error,
            }
        )
        self._write_report(root, report)
        return report

    def certify_paper_routes(
        self,
        *,
        workdir: Path | str,
        workspace: Path | str,
        paper_ids: list[str] | None = None,
        teacher: str = "yolo26s.pt",
        student: str = "yolo26n.pt",
        expected_teacher_sha256: str | None = None,
        expected_student_sha256: str | None = None,
        dataset_manifest_hash: str | None = None,
        split: str = "train",
        imgsz: int = 640,
        matched_baseline: dict[str, Any] | None = None,
        asset_registry_path: Path | str | None = None,
    ) -> DistillationPaperRouteCertificationSummary:
        """Certify paper-specific distillation routes without GPU training."""
        root = Path(workdir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        return certify_all_paper_routes(
            output_path=root / "distillation_paper_route_certification.yaml",
            workspace=workspace,
            paper_ids=tuple(paper_ids) if paper_ids is not None else None,
            teacher=teacher,
            student=student,
            expected_teacher_sha256=expected_teacher_sha256,
            expected_student_sha256=expected_student_sha256,
            dataset_manifest_hash=dataset_manifest_hash,
            split=split,
            imgsz=imgsz,
            matched_baseline=matched_baseline,
            asset_registry_path=asset_registry_path,
        )

    @classmethod
    def _load_previous(
        cls, root: Path
    ) -> PaperAdapterCertificationReport | None:
        path = root / cls.report_name
        if not path.is_file():
            return None
        return PaperAdapterCertificationReport.from_yaml(path)

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
        selection_reason: str,
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
                    selection_reason=selection_reason,
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
            result = _result_from_report(
                adapter=adapter,
                report=gpu,
                selection_reason=selection_reason,
                cpu_report=cpu_path,
                gpu_report=component_root / "component_certification.gpu.yaml",
            )
            if gpu.status != "passed":
                return result
            fixture_path = component_root / "matched_pilot_fixture.yaml"
            try:
                self.fixture_builder.build(
                    report=gpu,
                    identity=adapter.identity,
                    model=model,
                    data=data,
                    output=fixture_path,
                )
            except (OSError, TypeError, ValueError) as exc:
                return _updated_result(
                    result,
                    status="failed",
                    selection_reason="matched_pilot_fixture_failed",
                    errors=[str(exc)],
                )
            return _updated_result(
                result, matched_pilot_fixture=fixture_path
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
    identity_errors = _report_identity_errors(adapter, report)
    status = "failed" if identity_errors else report.status
    return PaperAdapterCertificationResult(
        component_id=adapter.component_id,
        identity=adapter.identity,
        status=status,
        initial_maturity=report.initial_maturity,
        final_maturity=report.final_maturity,
        selection_reason=(
            "certification_identity_mismatch"
            if identity_errors
            else selection_reason
        ),
        cpu_report=cpu_report,
        gpu_report=gpu_report,
        errors=[*report.errors, *identity_errors],
    )


def _report_identity_errors(
    adapter: ReusableAdapterDescriptor,
    report: ComponentCertificationReport,
) -> list[str]:
    expected = adapter.identity
    checks = {
        "component_id": (report.component_id, expected.component_id),
        "adapter_hash": (report.adapter_hash, expected.adapter_hash),
        "code_commit": (report.code_commit, expected.code_commit),
        "ultralytics_version": (
            report.ultralytics_version,
            expected.ultralytics_version,
        ),
        "protocol_hash": (report.protocol_hash, expected.protocol_hash),
    }
    return [
        f"certification_identity_mismatch:{name}"
        for name, (actual, wanted) in checks.items()
        if actual != wanted
    ]


def _batch_status(
    results: list[PaperAdapterCertificationResult],
    *,
    expected_count: int,
) -> str:
    if len(results) < expected_count:
        return "partial"
    failures = [item for item in results if item.status in {"failed", "blocked"}]
    successes = [
        item
        for item in results
        if item.status in {"passed", "skipped_resume", "skipped_unchanged"}
    ]
    if failures and successes:
        return "partial"
    if failures:
        return (
            "blocked"
            if all(item.status == "blocked" for item in failures)
            else "failed"
        )
    return "passed"


def _selection_result(
    *,
    descriptor: ReusableAdapterDescriptor,
    prior: PaperAdapterCertificationResult | None,
    mode: BatchCertificationMode,
    resume: bool,
    changed_only: bool,
) -> PaperAdapterCertificationResult | None:
    if prior is None:
        return None
    changed = _identity_changes(prior.identity, descriptor.identity)
    if changed:
        return None
    if changed_only:
        return _updated_result(
            prior,
            status="skipped_unchanged",
            selection_reason=f"unchanged_identity;previous_status={prior.status}",
            errors=[],
        )
    if resume and _prior_stage_is_reusable(prior, mode):
        return _updated_result(
            prior,
            status="skipped_resume",
            selection_reason="matching_passed_report_reused",
            errors=[],
        )
    return None


def _prior_stage_is_reusable(
    prior: PaperAdapterCertificationResult,
    mode: BatchCertificationMode,
) -> bool:
    report_path = prior.gpu_report if mode == "gpu" else prior.cpu_report
    if report_path is None or not report_path.is_file():
        return False
    try:
        report = ComponentCertificationReport.from_yaml(report_path)
    except (OSError, ValueError):
        return False
    required_maturity = "gpu_certified" if mode == "gpu" else "smoke_passed"
    return bool(
        report.status == "passed"
        and report.component_id == prior.component_id
        and report.protocol_hash == prior.identity.protocol_hash
        and report.adapter_hash == prior.identity.adapter_hash
        and report.ultralytics_version == prior.identity.ultralytics_version
        and report.final_maturity == required_maturity
    )


def _identity_changes(
    previous: AdapterCertificationIdentity,
    current: AdapterCertificationIdentity,
) -> list[str]:
    return [
        name
        for name in (
            "adapter_hash",
            "ultralytics_version",
            "protocol_hash",
        )
        if getattr(previous, name) != getattr(current, name)
    ]


def _run_selection_reason(
    previous: PaperAdapterCertificationResult | None,
    current: AdapterCertificationIdentity,
    *,
    changed_only: bool,
) -> str:
    if previous is None:
        return (
            "new_adapter_selected_by_changed_only"
            if changed_only
            else "selected_all_reusable_adapters"
        )
    changes = _identity_changes(previous.identity, current)
    if changes:
        return "identity_changed:" + ",".join(changes)
    return "previous_result_not_reusable"


def _updated_result(
    result: PaperAdapterCertificationResult,
    **updates: object,
) -> PaperAdapterCertificationResult:
    payload = result.model_dump(mode="json", exclude={"result_hash"})
    payload.update(updates)
    return PaperAdapterCertificationResult.model_validate(payload)


def _component_directory(component_id: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in component_id
    ).strip(".-") or "unknown-component"


__all__ = [
    "AdapterCertificationRunnerProtocol",
    "AdapterCoverageUpdaterProtocol",
    "AdapterDiscoveryProtocol",
    "MatchedPilotFixtureBuilderProtocol",
    "PaperAdapterCertificationFactory",
]
