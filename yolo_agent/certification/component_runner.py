"""CPU and opt-in GPU certification for one runtime component adapter."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Protocol

from yolo_agent.certification.component_schemas import (
    ComponentCertificationMode,
    ComponentCertificationReport,
    ComponentCertificationStage,
    ComponentGPUCertificationEvidence,
    ComponentSmokeWorkerReport,
    ComponentSmokeWorkerRequest,
)
from yolo_agent.components.contracts import ComponentContract, load_contracts
from yolo_agent.components.adapters.assigners.yolo26_assignment import ASSIGNMENT_SPECS
from yolo_agent.components.distillation import DISTILLATION_COMPONENTS
from yolo_agent.components.maturity import (
    maturity_artifact,
    maturity_rank,
    record_maturity_artifact,
    transition_maturity,
)
from yolo_agent.components.maturity_registry import (
    ComponentMaturityRegistry,
    adapter_source_hash,
    current_code_commit,
    installed_ultralytics_version,
)
from yolo_agent.components.validation_bridge import ComponentValidationBridge
from yolo_agent.components.validation_schemas import ComponentValidationStageReport
from yolo_agent.resources import ResourcePaths


class ComponentSmokeBackend(Protocol):
    """Boundary for isolated smoke execution."""

    def run(
        self,
        request: ComponentSmokeWorkerRequest,
        *,
        workdir: Path,
    ) -> tuple[ComponentSmokeWorkerReport, Path]: ...


class SubprocessComponentSmokeBackend:
    """Run smoke checks in a separate Python process."""

    def __init__(self, *, timeout_seconds: int = 900) -> None:
        self.timeout_seconds = timeout_seconds

    def run(
        self,
        request: ComponentSmokeWorkerRequest,
        *,
        workdir: Path,
    ) -> tuple[ComponentSmokeWorkerReport, Path]:
        request_path = workdir / f"{request.mode}_worker_request.yaml"
        staging_path = workdir / f".{request.mode}_worker_report.yaml"
        log_path = workdir / f"{request.mode}_worker.log"
        request.to_yaml(request_path, exclude_none=True, sort_keys=False)
        command = [
            sys.executable,
            "-m",
            "yolo_agent.certification.component_worker",
            "--request",
            str(request_path),
            "--output",
            str(staging_path),
        ]
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout_seconds,
            check=False,
        )
        log_path.write_text(
            (completed.stdout or "") + (completed.stderr or ""),
            encoding="utf-8",
        )
        if not staging_path.is_file():
            raise RuntimeError(
                f"component smoke worker produced no report; exit_code={completed.returncode}"
            )
        report = ComponentSmokeWorkerReport.from_yaml(staging_path)
        immutable = _content_addressed_report(staging_path, request.mode)
        return report, immutable


class ComponentCertificationRunner:
    """Certify adapter runtime maturity without creating a training node."""

    def __init__(
        self,
        *,
        worker_backend: ComponentSmokeBackend | None = None,
        contract_paths: list[Path | str] | None = None,
    ) -> None:
        self.worker_backend = worker_backend or SubprocessComponentSmokeBackend()
        self.contract_paths = (
            [Path(item) for item in contract_paths]
            if contract_paths is not None
            else [
                ResourcePaths.COMPONENT_COMPATIBILITY,
                *sorted(ResourcePaths.COMPONENTS_DIR.rglob("*.yaml")),
            ]
        )

    def run(
        self,
        *,
        component_id: str,
        mode: ComponentCertificationMode,
        workdir: Path | str,
        registry_path: Path | str,
        model: str = "yolo26n.pt",
        data: str = "coco.yaml",
        device: str = "0",
        protocol_hash: str | None = None,
        options: dict[str, object] | None = None,
        execute_gpu: bool = False,
    ) -> ComponentCertificationReport:
        root = Path(workdir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        registry = ComponentMaturityRegistry(registry_path)
        source = self._find_source_contract(component_id)
        adapter_hash = adapter_source_hash(source)
        code_commit = current_code_commit()
        ultralytics_version = installed_ultralytics_version()
        resolved_protocol = protocol_hash or component_certification_protocol_hash(
            component_id=component_id,
            adapter_hash=adapter_hash,
            ultralytics_version=ultralytics_version,
        )
        contract = self._load_effective_contract(
            component_id,
            registry=registry,
            protocol_hash=resolved_protocol,
            ultralytics_version=ultralytics_version,
        )
        if mode == "gpu":
            return self._run_gpu(
                contract=contract,
                root=root,
                registry=registry,
                adapter_hash=adapter_hash,
                code_commit=code_commit,
                ultralytics_version=ultralytics_version,
                protocol_hash=resolved_protocol,
                model=model,
                device=device,
                options=dict(options or {}),
                execute_gpu=execute_gpu,
            )
        return self._run_cpu(
            contract=contract,
            root=root,
            registry=registry,
            adapter_hash=adapter_hash,
            code_commit=code_commit,
            ultralytics_version=ultralytics_version,
            protocol_hash=resolved_protocol,
            model=model,
            data=data,
            options=dict(options or {}),
        )

    def _run_cpu(
        self,
        *,
        contract: ComponentContract,
        root: Path,
        registry: ComponentMaturityRegistry,
        adapter_hash: str,
        code_commit: str,
        ultralytics_version: str,
        protocol_hash: str,
        model: str,
        data: str,
        options: dict[str, object],
    ) -> ComponentCertificationReport:
        initial_maturity = contract.maturity
        model, options = _cpu_fixture_inputs(
            contract=contract,
            root=root,
            model=model,
            data=data,
            options=options,
        )
        validation_root = root / "cpu_validation"
        validation = ComponentValidationBridge(
            maturity_registry=registry,
        ).validate(
            contract=contract,
            workspace=validation_root,
            protocol_hash=protocol_hash,
            base_command=[
                "yolo",
                "detect",
                "train",
                f"model={model}",
                f"data={data}",
                "imgsz=640",
            ],
            model_config={"model": model},
            training_config={"imgsz": 640},
            options=options,
            target_maturity="unit_tested",
        )
        stages = _validation_stages(validation)
        stages = _complete_reused_validation_stages(stages, validation.contract)
        if (
            validation.status != "completed"
            or maturity_rank(validation.contract.maturity) < maturity_rank("unit_tested")
            or validation.runtime_payload_path is None
        ):
            return self._write_report(
                root,
                ComponentCertificationReport(
                    component_id=contract.component_id,
                    mode="cpu",
                    status="blocked" if validation.status == "blocked" else "failed",
                    initial_maturity=initial_maturity,
                    final_maturity=validation.contract.maturity,
                    next_maturity=_next_maturity(validation.contract.maturity),
                    protocol_hash=protocol_hash,
                    adapter_hash=adapter_hash,
                    code_commit=code_commit,
                    ultralytics_version=ultralytics_version,
                    registry_path=registry.path,
                    workdir=root,
                    stages=stages,
                    missing_artifacts=_missing_cpu_artifacts(validation.contract),
                    generated_paths={
                        "validation": validation_root / "component_validation.yaml"
                    },
                    errors=list(validation.blocked_by),
                ),
            )
        request = ComponentSmokeWorkerRequest(
            contract=validation.contract,
            mode="cpu",
            protocol_hash=protocol_hash,
            runtime_payload_path=validation.runtime_payload_path,
            workspace=root / "cpu_smoke",
            device="cpu",
            options=options,
        )
        worker, worker_path = self.worker_backend.run(request, workdir=root)
        updated = _apply_worker_artifact(
            validation.contract,
            worker,
            worker_path,
            target="smoke_passed",
            protocol_hash=protocol_hash,
        )
        registry.record_contract(
            updated,
            adapter_hash=adapter_hash,
            code_commit=code_commit,
            ultralytics_version=ultralytics_version,
            protocol_hash=protocol_hash,
        )
        passed = worker.status == "passed" and worker.evidence_kind == "local"
        stages.append(
            ComponentCertificationStage(
                stage_id="isolated_smoke",
                status="passed" if passed else "failed",
                message=(
                    "isolated local CPU smoke passed"
                    if passed
                    else "; ".join(worker.errors) or "isolated CPU smoke failed"
                ),
                artifacts={
                    "worker_report": worker_path,
                    **_worker_generated_artifacts(worker),
                },
                checks=dict(worker.checks),
            )
        )
        return self._write_report(
            root,
            ComponentCertificationReport(
                component_id=contract.component_id,
                mode="cpu",
                status="passed" if passed else "failed",
                initial_maturity=initial_maturity,
                final_maturity=updated.maturity,
                next_maturity=_next_maturity(updated.maturity),
                protocol_hash=protocol_hash,
                adapter_hash=adapter_hash,
                code_commit=code_commit,
                ultralytics_version=ultralytics_version,
                registry_path=registry.path,
                workdir=root,
                stages=stages,
                missing_artifacts=[] if passed else ["smoke_passed"],
                generated_paths={
                    "validation": validation_root / "component_validation.yaml",
                    "runtime_payload": validation.runtime_payload_path,
                    "worker_report": worker_path,
                    **_worker_generated_artifacts(worker),
                },
                errors=[] if passed else list(worker.errors),
            ),
        )

    def _run_gpu(
        self,
        *,
        contract: ComponentContract,
        root: Path,
        registry: ComponentMaturityRegistry,
        adapter_hash: str,
        code_commit: str,
        ultralytics_version: str,
        protocol_hash: str,
        model: str,
        device: str,
        options: dict[str, object],
        execute_gpu: bool,
    ) -> ComponentCertificationReport:
        initial_maturity = contract.maturity
        if not execute_gpu:
            return self._blocked_gpu_report(
                contract=contract,
                root=root,
                registry=registry,
                adapter_hash=adapter_hash,
                code_commit=code_commit,
                ultralytics_version=ultralytics_version,
                protocol_hash=protocol_hash,
                reason="gpu_execution_not_confirmed",
                missing=[],
            )
        if not contract.can_execute:
            return self._blocked_gpu_report(
                contract=contract,
                root=root,
                registry=registry,
                adapter_hash=adapter_hash,
                code_commit=code_commit,
                ultralytics_version=ultralytics_version,
                protocol_hash=protocol_hash,
                reason="cpu_smoke_passed_required",
                missing=["smoke_passed"],
            )
        runtime_payload = _artifact_path(contract, "runtime_integrated")
        if runtime_payload is None:
            return self._blocked_gpu_report(
                contract=contract,
                root=root,
                registry=registry,
                adapter_hash=adapter_hash,
                code_commit=code_commit,
                ultralytics_version=ultralytics_version,
                protocol_hash=protocol_hash,
                reason="runtime_payload_artifact_missing",
                missing=["runtime_integrated"],
            )
        request = ComponentSmokeWorkerRequest(
            contract=contract,
            mode="gpu",
            protocol_hash=protocol_hash,
            runtime_payload_path=runtime_payload,
            workspace=root / "gpu_smoke",
            device=device,
            model=model,
            adapter_hash=adapter_hash,
            ultralytics_version=ultralytics_version,
            real_gpu_training=True,
            options=options,
        )
        worker, worker_path = self.worker_backend.run(request, workdir=root)
        passed = (
            worker.status == "passed"
            and worker.evidence_kind == "local"
            and worker.cuda_available
            and _verified_gpu_evidence(worker)
        )
        updated = _apply_worker_artifact(
            contract,
            worker,
            worker_path,
            target="gpu_certified",
            protocol_hash=protocol_hash,
        )
        registry.record_contract(
            updated,
            adapter_hash=adapter_hash,
            code_commit=code_commit,
            ultralytics_version=ultralytics_version,
            protocol_hash=protocol_hash,
        )
        stages = [
            ComponentCertificationStage(
                stage_id="cpu_smoke_precondition",
                status="passed",
                message="artifact-backed CPU smoke_passed is valid",
                artifacts={
                    "smoke_report": _artifact_path(contract, "smoke_passed")
                },
            ),
            ComponentCertificationStage(
                stage_id="isolated_gpu_smoke",
                status="passed" if passed else "failed",
                message=(
                    "isolated component GPU smoke passed"
                    if passed
                    else "; ".join(worker.errors) or "isolated GPU smoke failed"
                ),
                artifacts={
                    "worker_report": worker_path,
                    **_worker_generated_artifacts(worker),
                },
                checks=dict(worker.checks),
            ),
        ]
        return self._write_report(
            root,
            ComponentCertificationReport(
                component_id=contract.component_id,
                mode="gpu",
                status="passed" if passed else "failed",
                initial_maturity=initial_maturity,
                final_maturity=updated.maturity,
                next_maturity=_next_maturity(updated.maturity),
                protocol_hash=protocol_hash,
                adapter_hash=adapter_hash,
                code_commit=code_commit,
                ultralytics_version=ultralytics_version,
                registry_path=registry.path,
                workdir=root,
                stages=stages,
                missing_artifacts=[] if passed else ["gpu_certified"],
                generated_paths={
                    "runtime_payload": runtime_payload,
                    "worker_report": worker_path,
                    **_worker_generated_artifacts(worker),
                },
                errors=[] if passed else list(worker.errors),
            ),
        )

    def _blocked_gpu_report(
        self,
        *,
        contract: ComponentContract,
        root: Path,
        registry: ComponentMaturityRegistry,
        adapter_hash: str,
        code_commit: str,
        ultralytics_version: str,
        protocol_hash: str,
        reason: str,
        missing: list[str],
    ) -> ComponentCertificationReport:
        return self._write_report(
            root,
            ComponentCertificationReport(
                component_id=contract.component_id,
                mode="gpu",
                status="blocked",
                initial_maturity=contract.maturity,
                final_maturity=contract.maturity,
                next_maturity=_next_maturity(contract.maturity),
                protocol_hash=protocol_hash,
                adapter_hash=adapter_hash,
                code_commit=code_commit,
                ultralytics_version=ultralytics_version,
                registry_path=registry.path,
                workdir=root,
                stages=[
                    ComponentCertificationStage(
                        stage_id="cpu_smoke_precondition",
                        status="blocked",
                        message=reason,
                    )
                ],
                missing_artifacts=missing,
                errors=[reason],
            ),
        )

    def _find_source_contract(self, component_id: str) -> ComponentContract:
        found: ComponentContract | None = None
        for path in self.contract_paths:
            if not path.is_file():
                continue
            for contract in _load_contract_file(path):
                if contract.component_id == component_id:
                    found = contract
        if found is None:
            raise ValueError(f"component contract not found: {component_id}")
        return found

    def _load_effective_contract(
        self,
        component_id: str,
        *,
        registry: ComponentMaturityRegistry,
        protocol_hash: str,
        ultralytics_version: str,
    ) -> ComponentContract:
        found: ComponentContract | None = None
        for path in self.contract_paths:
            if not path.is_file():
                continue
            for contract in _load_contract_file(
                path,
                maturity_registry=registry,
                protocol_hash=protocol_hash,
                ultralytics_version=ultralytics_version,
            ):
                if contract.component_id == component_id:
                    found = contract
        if found is None:
            raise ValueError(f"component contract not found: {component_id}")
        return found

    @staticmethod
    def _write_report(
        root: Path,
        report: ComponentCertificationReport,
    ) -> ComponentCertificationReport:
        path = root / f"component_certification.{report.mode}.yaml"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        report.to_yaml(temporary, exclude_none=True, sort_keys=False)
        temporary.replace(path)
        return report


def component_certification_protocol_hash(
    *,
    component_id: str,
    adapter_hash: str,
    ultralytics_version: str,
) -> str:
    payload = {
        "schema_version": "component_certification_protocol.v2",
        "component_id": component_id,
        "adapter_hash": adapter_hash,
        "ultralytics_version": ultralytics_version,
        "imgsz": 640,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_contract_file(
    path: Path,
    *,
    maturity_registry: ComponentMaturityRegistry | None = None,
    protocol_hash: str | None = None,
    ultralytics_version: str | None = None,
) -> list[ComponentContract]:
    try:
        return load_contracts(
            path,
            maturity_registry=maturity_registry,
            protocol_hash=protocol_hash,
            ultralytics_version=ultralytics_version,
        )
    except (KeyError, TypeError, ValueError):
        # Legacy component-card YAML still lives beside typed runtime contracts.
        # Certification consumes only the typed contracts.
        return []


def _worker_generated_artifacts(
    report: ComponentSmokeWorkerReport,
) -> dict[str, Path]:
    generated: dict[str, Path] = {}
    for key, output_name in (
        ("cpu_golden_path_report", "cpu_golden_path"),
        ("gpu_evidence_path", "gpu_evidence"),
    ):
        value = report.checks.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            if path.is_file():
                generated[output_name] = path
    return generated


def _verified_gpu_evidence(worker: ComponentSmokeWorkerReport) -> bool:
    value = worker.checks.get("gpu_evidence_path")
    if not isinstance(value, str) or not value:
        return False
    try:
        evidence = ComponentGPUCertificationEvidence.from_yaml(value)
    except (OSError, ValueError):
        return False
    return bool(
        evidence.status == "passed"
        and evidence.component_id == worker.component_id
        and evidence.worker_protocol_hash == worker.protocol_hash
        and evidence.runtime_payload_hash == worker.payload_hash
    )


def _cpu_fixture_inputs(
    *,
    contract: ComponentContract,
    root: Path,
    model: str,
    data: str,
    options: dict[str, object],
) -> tuple[str, dict[str, object]]:
    if contract.component_id in ASSIGNMENT_SPECS:
        prepared = dict(options)
        prepared.setdefault("assignment.minimum_shadow_batches", 1)
        prepared.setdefault("assignment.maximum_conflict_rate", 1.0)
        prepared.setdefault("assignment.evidence_interval", 1)
        return model, prepared
    if contract.component_id not in {
        "distillation.yolo26_teacher_student",
        *DISTILLATION_COMPONENTS,
    }:
        return model, options
    prepared = dict(options)
    fixture_root = root / "cpu_fixture_inputs"
    fixture_root.mkdir(parents=True, exist_ok=True)
    teacher = Path(str(prepared.get("teacher", "yolo26s.pt")))
    student = Path(str(prepared.get("student", model)))
    if not teacher.is_file():
        teacher = fixture_root / "yolo26s.pt"
        teacher.write_bytes(b"yolo-agent-cpu-certification-teacher\n")
    if not student.is_file():
        student = fixture_root / "yolo26n.pt"
        student.write_bytes(b"yolo-agent-cpu-certification-student\n")
    ensemble_teachers: list[str] = []
    if contract.component_id == "distillation.teacher_ensemble":
        teacher_m = fixture_root / "yolo26m.pt"
        if not teacher_m.is_file():
            teacher_m.write_bytes(b"yolo-agent-cpu-certification-teacher-m\n")
        ensemble_teachers = [str(teacher_m.resolve())]
    prepared.update(
        {
            "teacher": str(teacher.resolve()),
            "student": str(student.resolve()),
            "teacher_data": data,
            "student_data": data,
            "imgsz": 640,
            "teachers": ensemble_teachers,
        }
    )
    return str(student.resolve()), prepared


def _validation_stages(validation: object) -> list[ComponentCertificationStage]:
    result = validation
    stages = [
        ComponentCertificationStage(
            stage_id="adapter_import",
            status="passed",
            message="component adapter imported",
        )
    ]
    stage_reports = getattr(result, "stage_reports", {})
    runtime_path = stage_reports.get("runtime_integrated")
    if runtime_path is not None and Path(runtime_path).is_file():
        runtime = ComponentValidationStageReport.from_yaml(runtime_path)
        runtime_passed = runtime.status == "passed"
        stages.extend(
            [
                ComponentCertificationStage(
                    stage_id="runtime_payload",
                    status="passed" if runtime_passed else "failed",
                    message="typed runtime payload validated",
                    artifacts=dict(runtime.artifacts),
                    checks=dict(runtime.checks),
                ),
                ComponentCertificationStage(
                    stage_id="hook_signature",
                    status=(
                        "passed"
                        if int(runtime.checks.get("runtime_hook_signatures_verified", 0)) > 0
                        else "failed"
                    ),
                    message="runtime plugin hook signatures validated",
                    artifacts={"runtime_report": Path(runtime_path)},
                    checks={
                        "verified_hook_count": int(
                            runtime.checks.get("runtime_hook_signatures_verified", 0)
                        )
                    },
                ),
            ]
        )
    unit_path = stage_reports.get("unit_tested")
    if unit_path is not None and Path(unit_path).is_file():
        unit = ComponentValidationStageReport.from_yaml(unit_path)
        stages.append(
            ComponentCertificationStage(
                stage_id="unit_tests",
                status="passed" if unit.status == "passed" else "failed",
                message="runtime contract unit tests completed",
                artifacts={"unit_report": Path(unit_path)},
                checks=dict(unit.checks),
            )
        )
    return stages


def _complete_reused_validation_stages(
    stages: list[ComponentCertificationStage],
    contract: ComponentContract,
) -> list[ComponentCertificationStage]:
    existing = {item.stage_id for item in stages}
    runtime_path = _artifact_path(contract, "runtime_integrated")
    if runtime_path is not None and "runtime_payload" not in existing:
        stages.extend(
            [
                ComponentCertificationStage(
                    stage_id="runtime_payload",
                    status="passed",
                    message="reused valid artifact-backed runtime payload",
                    artifacts={"runtime_payload": runtime_path},
                ),
                ComponentCertificationStage(
                    stage_id="hook_signature",
                    status="passed",
                    message="reused runtime-integrated hook signature evidence",
                    artifacts={"runtime_payload": runtime_path},
                ),
            ]
        )
    unit_path = _artifact_path(contract, "unit_tested")
    if unit_path is not None and "unit_tests" not in existing:
        stages.append(
            ComponentCertificationStage(
                stage_id="unit_tests",
                status="passed",
                message="reused valid artifact-backed unit evidence",
                artifacts={"unit_report": unit_path},
            )
        )
    return stages


def _apply_worker_artifact(
    contract: ComponentContract,
    worker: ComponentSmokeWorkerReport,
    worker_path: Path,
    *,
    target: str,
    protocol_hash: str,
) -> ComponentContract:
    passed = worker.status == "passed" and worker.evidence_kind == "local"
    if target == "gpu_certified":
        passed = passed and worker.cuda_available and _verified_gpu_evidence(worker)
    artifact = maturity_artifact(
        component_id=contract.component_id,
        target_maturity=target,
        artifact_path=worker_path,
        status="passed" if passed else "failed",
        producer="ComponentCertificationRunner",
        mock=worker.evidence_kind != "local",
        protocol_hash=protocol_hash,
        metadata={
            "mode": worker.mode,
            "process_id": worker.process_id,
            "payload_hash": worker.payload_hash,
        },
    )
    if any(
        item.target_maturity == artifact.target_maturity
        and item.artifact_sha256 == artifact.artifact_sha256
        for item in contract.maturity_artifacts
    ):
        return contract
    if passed and maturity_rank(target) == maturity_rank(contract.maturity) + 1:
        return transition_maturity(
            contract,
            target,
            reason=f"isolated {worker.mode} component certification passed",
            artifact=artifact,
        )
    return record_maturity_artifact(contract, artifact)


def _artifact_path(contract: ComponentContract, target: str) -> Path | None:
    artifact = next(
        (
            item
            for item in reversed(contract.maturity_artifacts)
            if item.target_maturity == target
            and item.status == "passed"
            and not item.mock
        ),
        None,
    )
    return artifact.artifact_path if artifact is not None else None


def _missing_cpu_artifacts(contract: ComponentContract) -> list[str]:
    return [
        target
        for target in ("runtime_integrated", "unit_tested", "smoke_passed")
        if _artifact_path(contract, target) is None
    ]


def _next_maturity(maturity: str) -> str | None:
    names = (
        "metadata_only",
        "recipe_idea_only",
        "adapter_implemented",
        "runtime_integrated",
        "unit_tested",
        "smoke_passed",
        "gpu_certified",
        "pilot_reproduced",
        "full_reproduced",
        "confirmed_multi_seed",
    )
    try:
        index = names.index(maturity)
    except ValueError:
        return None
    return names[index + 1] if index + 1 < len(names) else None


def _content_addressed_report(staging: Path, mode: str) -> Path:
    digest = hashlib.sha256(staging.read_bytes()).hexdigest()
    target = staging.parent / f"isolated_{mode}_smoke.{digest[:12]}.yaml"
    if target.is_file():
        staging.unlink()
    else:
        staging.replace(target)
    return target


__all__ = [
    "ComponentCertificationRunner",
    "ComponentSmokeBackend",
    "SubprocessComponentSmokeBackend",
    "component_certification_protocol_hash",
]
