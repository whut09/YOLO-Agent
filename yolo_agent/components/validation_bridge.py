"""Bootstrap component runtime maturity without creating a training node."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Literal

from yolo_agent.components.adapters import ComponentAdapterRegistry
from yolo_agent.components.adapters.base import AdapterContext, ComponentAdapter, PatchPreview
from yolo_agent.components.adapters.runtime import AdapterRuntimePayload
from yolo_agent.components.adapters.runtime import RUNTIME_PLUGIN_METHODS
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.maturity import (
    ComponentMaturityArtifact,
    MaturityName,
    maturity_artifact,
    maturity_rank,
    record_maturity_artifact,
    transition_maturity,
)
from yolo_agent.components.validation_schemas import (
    ComponentValidationResult,
    ComponentValidationStageReport,
    ValidationStage,
)


SmokeEvidence = Literal["mock", "local"]
_VALIDATION_STAGES: tuple[ValidationStage, ...] = (
    "runtime_integrated",
    "unit_tested",
    "smoke_passed",
)


class ComponentValidationBridge:
    """Generate artifact-backed runtime, unit, and smoke maturity evidence."""

    result_name = "component_validation.yaml"

    def __init__(self, *, adapter_registry: ComponentAdapterRegistry | None = None) -> None:
        self.adapter_registry = adapter_registry or ComponentAdapterRegistry()

    def validate(
        self,
        *,
        contract: ComponentContract,
        workspace: Path | str,
        protocol_hash: str,
        base_command: list[str],
        model_config: dict[str, Any] | None = None,
        training_config: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        target_maturity: ValidationStage = "smoke_passed",
        smoke_evidence: SmokeEvidence = "mock",
    ) -> ComponentValidationResult:
        """Validate one adapter and advance only adjacent artifact-backed states."""
        root = Path(workspace).resolve()
        root.mkdir(parents=True, exist_ok=True)
        initial_maturity = contract.maturity
        model_values = dict(model_config or {"model": "yolo26n.pt"})
        training_values = dict(training_config or {"imgsz": 640})
        option_values = dict(options or {})
        validation_key = _validation_key(
            contract=contract,
            protocol_hash=protocol_hash,
            base_command=base_command,
            model_config=model_values,
            training_config=training_values,
            options=option_values,
        )
        preflight_error = _preflight_error(
            contract,
            protocol_hash=protocol_hash,
            base_command=base_command,
            training_config=training_values,
            target_maturity=target_maturity,
        )
        if preflight_error:
            return self._blocked_result(
                root=root,
                contract=contract,
                initial_maturity=initial_maturity,
                target_maturity=target_maturity,
                protocol_hash=protocol_hash,
                validation_key=validation_key,
                reason=preflight_error,
            )

        try:
            adapter = self.adapter_registry.create_for_contract(contract)
        except (AttributeError, ImportError, KeyError, TypeError, ValueError) as exc:
            return self._failed_stage(
                root=root,
                contract=contract,
                initial_maturity=initial_maturity,
                target_maturity=target_maturity,
                protocol_hash=protocol_hash,
                validation_key=validation_key,
                stage="runtime_integrated",
                errors=[f"adapter_import_failed:{exc}"],
            )

        validation_key = _validation_key(
            contract=contract,
            protocol_hash=protocol_hash,
            base_command=base_command,
            model_config=model_values,
            training_config=training_values,
            options=option_values,
            adapter=adapter,
        )
        working, stage_reports = _recover_contract(
            root / self.result_name,
            contract=contract,
            validation_key=validation_key,
            protocol_hash=protocol_hash,
        )
        source_error = _source_artifact_error(working)
        if source_error:
            return self._blocked_result(
                root=root,
                contract=working,
                initial_maturity=initial_maturity,
                target_maturity=target_maturity,
                protocol_hash=protocol_hash,
                validation_key=validation_key,
                reason=source_error,
                stage_reports=stage_reports,
            )

        context = AdapterContext(
            contract=working,
            detector_family="yolo26",
            head="one_to_one",
            imgsz=640,
            workspace=root,
            options=option_values,
        )
        try:
            preview = adapter.prepare_patch(
                model_values,
                training_values,
                context,
                dry_run=True,
            )
        except (AttributeError, ImportError, KeyError, TypeError, ValueError) as exc:
            return self._failed_stage(
                root=root,
                contract=working,
                initial_maturity=initial_maturity,
                target_maturity=target_maturity,
                protocol_hash=protocol_hash,
                validation_key=validation_key,
                stage="runtime_integrated",
                errors=[f"adapter_patch_validation_failed:{exc}"],
                stage_reports=stage_reports,
            )
        preview_path = root / "patch_preview.yaml"
        preview.to_yaml(preview_path, exclude_none=True, sort_keys=False)

        runtime_payload_path = _passed_artifact_path(working, "runtime_integrated")
        if maturity_rank(working.maturity) < maturity_rank("runtime_integrated"):
            runtime = self._validate_runtime(
                root=root,
                contract=working,
                adapter=adapter,
                context=context,
                preview=preview,
                protocol_hash=protocol_hash,
                validation_key=validation_key,
                base_command=base_command,
            )
            if runtime[0] is None:
                return self._failed_stage(
                    root=root,
                    contract=working,
                    initial_maturity=initial_maturity,
                    target_maturity=target_maturity,
                    protocol_hash=protocol_hash,
                    validation_key=validation_key,
                    stage="runtime_integrated",
                    errors=runtime[2],
                    checks=runtime[3],
                    stage_reports=stage_reports,
                    patch_preview_path=preview_path,
                    existing_report_path=runtime[4],
                )
            working, runtime_payload_path, report_path = runtime[0], runtime[1], runtime[4]
            stage_reports["runtime_integrated"] = report_path
            self._persist_result(
                root=root,
                status="completed",
                contract=working,
                initial_maturity=initial_maturity,
                target_maturity=target_maturity,
                protocol_hash=protocol_hash,
                validation_key=validation_key,
                stage_reports=stage_reports,
                patch_preview_path=preview_path,
                runtime_payload_path=runtime_payload_path,
            )
        if target_maturity == "runtime_integrated":
            return self._completed_result(
                root=root,
                contract=working,
                initial_maturity=initial_maturity,
                target_maturity=target_maturity,
                protocol_hash=protocol_hash,
                validation_key=validation_key,
                stage_reports=stage_reports,
                patch_preview_path=preview_path,
                runtime_payload_path=runtime_payload_path,
            )

        if runtime_payload_path is None:
            return self._blocked_result(
                root=root,
                contract=working,
                initial_maturity=initial_maturity,
                target_maturity=target_maturity,
                protocol_hash=protocol_hash,
                validation_key=validation_key,
                reason="runtime_payload_artifact_missing",
                stage_reports=stage_reports,
                patch_preview_path=preview_path,
            )
        if maturity_rank(working.maturity) < maturity_rank("unit_tested"):
            unit = self._validate_unit(
                root=root,
                contract=working,
                adapter=adapter,
                context=context.model_copy(update={"contract": working}),
                preview=preview,
                runtime_payload_path=runtime_payload_path,
                model_config=model_values,
                training_config=training_values,
                protocol_hash=protocol_hash,
                validation_key=validation_key,
            )
            working, report_path = unit[0], unit[1]
            stage_reports["unit_tested"] = report_path
            if unit[2]:
                return self._failed_result(
                    root=root,
                    contract=working,
                    initial_maturity=initial_maturity,
                    target_maturity=target_maturity,
                    protocol_hash=protocol_hash,
                    validation_key=validation_key,
                    stage_reports=stage_reports,
                    errors=unit[2],
                    patch_preview_path=preview_path,
                    runtime_payload_path=runtime_payload_path,
                )
            self._persist_result(
                root=root,
                status="completed",
                contract=working,
                initial_maturity=initial_maturity,
                target_maturity=target_maturity,
                protocol_hash=protocol_hash,
                validation_key=validation_key,
                stage_reports=stage_reports,
                patch_preview_path=preview_path,
                runtime_payload_path=runtime_payload_path,
            )
        if target_maturity == "unit_tested":
            return self._completed_result(
                root=root,
                contract=working,
                initial_maturity=initial_maturity,
                target_maturity=target_maturity,
                protocol_hash=protocol_hash,
                validation_key=validation_key,
                stage_reports=stage_reports,
                patch_preview_path=preview_path,
                runtime_payload_path=runtime_payload_path,
            )

        if maturity_rank(working.maturity) < maturity_rank("smoke_passed"):
            smoke = self._validate_smoke(
                root=root,
                contract=working,
                adapter=adapter,
                context=context.model_copy(update={"contract": working}),
                protocol_hash=protocol_hash,
                validation_key=validation_key,
                smoke_evidence=smoke_evidence,
            )
            working, report_path, errors, retained_mock = smoke
            stage_reports["smoke_passed"] = report_path
            if errors:
                return self._failed_result(
                    root=root,
                    contract=working,
                    initial_maturity=initial_maturity,
                    target_maturity=target_maturity,
                    protocol_hash=protocol_hash,
                    validation_key=validation_key,
                    stage_reports=stage_reports,
                    errors=errors,
                    patch_preview_path=preview_path,
                    runtime_payload_path=runtime_payload_path,
                )
            if retained_mock:
                return self._blocked_result(
                    root=root,
                    contract=working,
                    initial_maturity=initial_maturity,
                    target_maturity=target_maturity,
                    protocol_hash=protocol_hash,
                    validation_key=validation_key,
                    reason="mock_smoke_evidence_cannot_promote",
                    stage_reports=stage_reports,
                    patch_preview_path=preview_path,
                    runtime_payload_path=runtime_payload_path,
                )
        return self._completed_result(
            root=root,
            contract=working,
            initial_maturity=initial_maturity,
            target_maturity=target_maturity,
            protocol_hash=protocol_hash,
            validation_key=validation_key,
            stage_reports=stage_reports,
            patch_preview_path=preview_path,
            runtime_payload_path=runtime_payload_path,
        )

    def _validate_runtime(
        self,
        *,
        root: Path,
        contract: ComponentContract,
        adapter: ComponentAdapter,
        context: AdapterContext,
        preview: PatchPreview,
        protocol_hash: str,
        validation_key: str,
        base_command: list[str],
    ) -> tuple[
        ComponentContract | None,
        Path | None,
        list[str],
        dict[str, bool | str | int | float],
        Path,
    ]:
        checks: dict[str, bool | str | int | float] = {
            "patch_preview_is_dry_run": preview.dry_run,
            "patch_preview_is_not_maturity_evidence": True,
        }
        errors: list[str] = []
        payload_path: Path | None = None
        try:
            payload = adapter.build_runtime_payload(
                context,
                protocol_hash=protocol_hash,
                base_command=base_command,
                generated_config={
                    "model_config": preview.patched_model_config,
                    "training_config": preview.patched_training_config,
                },
            )
            if payload is None:
                raise ValueError("adapter returned no runtime payload")
            payload.verify_imports()
            plugin_checks = _validate_runtime_plugins(payload)
            payload_path = payload.write(
                root / f"adapter_runtime_payload.{payload.payload_hash[:12]}.yaml"
            )
            restored = AdapterRuntimePayload.read(payload_path, verify_imports=True)
            checks.update(
                {
                    "payload_importable": True,
                    "payload_round_trip": restored.payload_hash == payload.payload_hash,
                    "runtime_hook_count": len(restored.plugin_references),
                    "protocol_bound": restored.protocol_hash == protocol_hash,
                    **plugin_checks,
                }
            )
            if not all(
                bool(checks[key])
                for key in ("payload_importable", "payload_round_trip", "protocol_bound")
            ):
                raise ValueError("runtime payload contract checks failed")
        except (AttributeError, ImportError, KeyError, OSError, TypeError, ValueError) as exc:
            errors.append(f"runtime_payload_validation_failed:{exc}")
        report = ComponentValidationStageReport(
            component_id=contract.component_id,
            stage="runtime_integrated",
            status="passed" if not errors else "failed",
            protocol_hash=protocol_hash,
            validation_key=validation_key,
            checks=checks,
            artifacts={
                **({"runtime_payload": payload_path} if payload_path is not None else {}),
                "patch_preview": root / "patch_preview.yaml",
            },
            errors=errors,
        )
        report_path = _write_stage_report(root, report)
        if errors or payload_path is None:
            return None, payload_path, errors, checks, report_path
        artifact = maturity_artifact(
            component_id=contract.component_id,
            target_maturity="runtime_integrated",
            artifact_path=payload_path,
            status="passed",
            producer="ComponentValidationBridge",
            protocol_hash=protocol_hash,
            metadata={"validation_key": validation_key, "stage_report": str(report_path)},
        )
        updated = transition_maturity(
            contract,
            "runtime_integrated",
            reason="runtime payload imports and round-trips",
            artifact=artifact,
        )
        return updated, payload_path, [], checks, report_path

    def _validate_unit(
        self,
        *,
        root: Path,
        contract: ComponentContract,
        adapter: ComponentAdapter,
        context: AdapterContext,
        preview: PatchPreview,
        runtime_payload_path: Path,
        model_config: dict[str, Any],
        training_config: dict[str, Any],
        protocol_hash: str,
        validation_key: str,
    ) -> tuple[ComponentContract, Path, list[str]]:
        errors: list[str] = []
        checks: dict[str, bool | str | int | float] = {}
        try:
            payload = AdapterRuntimePayload.read(runtime_payload_path, verify_imports=True)
            repeated = adapter.prepare_patch(
                model_config,
                training_config,
                context,
                dry_run=True,
            )
            checks = {
                "runtime_payload_hash_stable": payload.payload_hash
                == hashlib.sha256(
                    json.dumps(
                        payload.model_dump(mode="json", exclude_none=True),
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    ).encode("utf-8")
                ).hexdigest(),
                "patch_idempotency_key_stable": repeated.idempotency_key
                == preview.idempotency_key,
                "runtime_hooks_present": len(payload.plugin_references) > 0,
                "rollback_does_not_modify_global_source": not payload.rollback_plan.restores_global_source,
            }
            if not all(bool(value) for value in checks.values()):
                raise ValueError("one or more unit contract checks failed")
        except (AttributeError, ImportError, KeyError, OSError, TypeError, ValueError) as exc:
            errors.append(f"unit_contract_validation_failed:{exc}")
        report = ComponentValidationStageReport(
            component_id=contract.component_id,
            stage="unit_tested",
            status="passed" if not errors else "failed",
            protocol_hash=protocol_hash,
            validation_key=validation_key,
            checks=checks,
            artifacts={"runtime_payload": runtime_payload_path},
            errors=errors,
        )
        report_path = _write_stage_report(root, report)
        artifact = maturity_artifact(
            component_id=contract.component_id,
            target_maturity="unit_tested",
            artifact_path=report_path,
            status="passed" if not errors else "failed",
            producer="ComponentValidationBridge",
            protocol_hash=protocol_hash,
            metadata={"validation_key": validation_key},
        )
        updated = (
            transition_maturity(
                contract,
                "unit_tested",
                reason="runtime contract unit checks passed",
                artifact=artifact,
            )
            if not errors
            else _record_once(contract, artifact)
        )
        return updated, report_path, errors

    def _validate_smoke(
        self,
        *,
        root: Path,
        contract: ComponentContract,
        adapter: ComponentAdapter,
        context: AdapterContext,
        protocol_hash: str,
        validation_key: str,
        smoke_evidence: SmokeEvidence,
    ) -> tuple[ComponentContract, Path, list[str], bool]:
        errors: list[str] = []
        checks: dict[str, bool | str | int | float] = {}
        try:
            smoke = adapter.smoke_test(context)
            checks = dict(smoke.checks)
            checks["adapter_reported_passed"] = smoke.passed
            errors.extend(smoke.errors)
            if not smoke.passed and not errors:
                errors.append("adapter smoke test failed without details")
        except (AttributeError, ImportError, KeyError, OSError, TypeError, ValueError) as exc:
            errors.append(f"adapter_smoke_failed:{exc}")
        retained_mock = not errors and smoke_evidence == "mock"
        status = "failed" if errors else "retained_mock" if retained_mock else "passed"
        report = ComponentValidationStageReport(
            component_id=contract.component_id,
            stage="smoke_passed",
            status=status,
            protocol_hash=protocol_hash,
            validation_key=validation_key,
            mock=smoke_evidence == "mock",
            checks=checks,
            errors=errors,
            metadata={"smoke_evidence": smoke_evidence},
        )
        report_path = _write_stage_report(root, report)
        artifact = maturity_artifact(
            component_id=contract.component_id,
            target_maturity="smoke_passed",
            artifact_path=report_path,
            status="failed" if errors else "passed",
            producer="ComponentValidationBridge",
            mock=smoke_evidence == "mock",
            protocol_hash=protocol_hash,
            metadata={"validation_key": validation_key},
        )
        updated = (
            transition_maturity(
                contract,
                "smoke_passed",
                reason="local adapter smoke checks passed",
                artifact=artifact,
            )
            if not errors and not retained_mock
            else _record_once(contract, artifact)
        )
        return updated, report_path, errors, retained_mock

    def _failed_stage(
        self,
        *,
        root: Path,
        contract: ComponentContract,
        initial_maturity: MaturityName,
        target_maturity: ValidationStage,
        protocol_hash: str,
        validation_key: str,
        stage: ValidationStage,
        errors: list[str],
        checks: dict[str, bool | str | int | float] | None = None,
        stage_reports: dict[ValidationStage, Path] | None = None,
        patch_preview_path: Path | None = None,
        existing_report_path: Path | None = None,
    ) -> ComponentValidationResult:
        if existing_report_path is None:
            report = ComponentValidationStageReport(
                component_id=contract.component_id,
                stage=stage,
                status="failed",
                protocol_hash=protocol_hash,
                validation_key=validation_key,
                checks=checks or {},
                errors=errors,
            )
            report_path = _write_stage_report(root, report)
        else:
            report_path = existing_report_path
        artifact = maturity_artifact(
            component_id=contract.component_id,
            target_maturity=stage,
            artifact_path=report_path,
            status="failed",
            producer="ComponentValidationBridge",
            protocol_hash=protocol_hash or None,
            metadata={"validation_key": validation_key},
        )
        updated = _record_once(contract, artifact)
        reports = dict(stage_reports or {})
        reports[stage] = report_path
        return self._failed_result(
            root=root,
            contract=updated,
            initial_maturity=initial_maturity,
            target_maturity=target_maturity,
            protocol_hash=protocol_hash,
            validation_key=validation_key,
            stage_reports=reports,
            errors=errors,
            patch_preview_path=patch_preview_path,
        )

    def _completed_result(self, **kwargs: Any) -> ComponentValidationResult:
        return self._persist_result(status="completed", **kwargs)

    def _failed_result(self, *, errors: list[str], **kwargs: Any) -> ComponentValidationResult:
        return self._persist_result(status="failed", blocked_by=errors, **kwargs)

    def _blocked_result(self, *, reason: str, **kwargs: Any) -> ComponentValidationResult:
        return self._persist_result(status="blocked", blocked_by=[reason], **kwargs)

    def _persist_result(
        self,
        *,
        root: Path,
        status: Literal["completed", "failed", "blocked"],
        contract: ComponentContract,
        initial_maturity: MaturityName,
        target_maturity: ValidationStage,
        protocol_hash: str,
        validation_key: str,
        stage_reports: dict[ValidationStage, Path] | None = None,
        patch_preview_path: Path | None = None,
        runtime_payload_path: Path | None = None,
        blocked_by: list[str] | None = None,
    ) -> ComponentValidationResult:
        result = ComponentValidationResult(
            status=status,
            component_id=contract.component_id,
            initial_maturity=initial_maturity,
            final_maturity=contract.maturity,
            target_maturity=target_maturity,
            protocol_hash=protocol_hash,
            validation_key=validation_key,
            contract=contract,
            stage_reports=stage_reports or {},
            patch_preview_path=patch_preview_path,
            runtime_payload_path=runtime_payload_path,
            blocked_by=blocked_by or [],
        )
        result.to_yaml(root / self.result_name, exclude_none=True, sort_keys=False)
        return result


def _preflight_error(
    contract: ComponentContract,
    *,
    protocol_hash: str,
    base_command: list[str],
    training_config: dict[str, Any],
    target_maturity: ValidationStage,
) -> str | None:
    if target_maturity not in _VALIDATION_STAGES:
        return f"unsupported_validation_target:{target_maturity}"
    if maturity_rank(contract.maturity) < maturity_rank("adapter_implemented"):
        return f"component_maturity_below_adapter_implemented:{contract.maturity}"
    if not contract.implementation_path or not contract.adapter_class:
        return "adapter_implementation_identity_missing"
    if not protocol_hash.strip():
        return "validation_protocol_hash_missing"
    if not base_command:
        return "validation_base_command_missing"
    if training_config.get("imgsz", 640) != 640:
        return "fixed_imgsz_640_required"
    return None


def _validation_key(
    *,
    contract: ComponentContract,
    protocol_hash: str,
    base_command: list[str],
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    options: dict[str, Any],
    adapter: ComponentAdapter | None = None,
) -> str:
    payload = {
        "component_id": contract.component_id,
        "implementation_path": contract.implementation_path,
        "adapter_class": contract.adapter_class,
        "protocol_hash": protocol_hash,
        "base_command": base_command,
        "model_config": model_config,
        "training_config": training_config,
        "options": options,
        "adapter_version": getattr(adapter, "adapter_version", None),
        "source_commit": getattr(adapter, "source_commit", None),
        "adapter_source_sha256": _adapter_source_sha256(adapter),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _adapter_source_sha256(adapter: ComponentAdapter | None) -> str | None:
    if adapter is None:
        return None
    source = inspect.getsourcefile(type(adapter))
    if not source or not Path(source).is_file():
        return None
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


def _validate_runtime_plugins(
    payload: AdapterRuntimePayload,
) -> dict[str, bool | str | int | float]:
    """Instantiate declared plugins and verify their concrete runtime hooks."""
    loaded = 0
    hook_count = 0
    identities: list[str] = []
    for reference in payload.plugin_references:
        implementation = reference.resolve()
        instance = (
            implementation(**reference.options)
            if isinstance(implementation, type)
            else implementation
        )
        hooks = sorted(
            method
            for method in RUNTIME_PLUGIN_METHODS
            if callable(getattr(instance, method, None))
        )
        if not hooks:
            raise ValueError(f"runtime plugin has no callable hooks: {reference.reference}")
        loaded += 1
        hook_count += len(hooks)
        identities.append(f"{reference.reference}={','.join(hooks)}")
    return {
        "runtime_plugins_loaded": loaded,
        "runtime_plugin_hooks_verified": hook_count,
        "runtime_plugin_identities": ";".join(identities),
    }


def _write_stage_report(
    root: Path,
    report: ComponentValidationStageReport,
) -> Path:
    """Persist an immutable stage report addressed by its content hash."""
    temporary = root / f".{report.stage}_report.tmp.yaml"
    report.to_yaml(temporary, exclude_none=True, sort_keys=False)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    target = root / f"{report.stage}_report.{digest[:12]}.yaml"
    if target.is_file():
        temporary.unlink()
    else:
        temporary.replace(target)
    return target


def _recover_contract(
    result_path: Path,
    *,
    contract: ComponentContract,
    validation_key: str,
    protocol_hash: str,
) -> tuple[ComponentContract, dict[ValidationStage, Path]]:
    if not result_path.is_file():
        return contract, {}
    try:
        result = ComponentValidationResult.from_yaml(result_path)
        if (
            result.component_id != contract.component_id
            or result.validation_key != validation_key
            or result.protocol_hash != protocol_hash
            or maturity_rank(result.contract.maturity) < maturity_rank(contract.maturity)
        ):
            return contract, {}
        for artifact in result.contract.maturity_artifacts:
            artifact.verify()
    except (OSError, TypeError, ValueError):
        return contract, {}
    return result.contract, dict(result.stage_reports)


def _source_artifact_error(contract: ComponentContract) -> str | None:
    for stage in _VALIDATION_STAGES:
        if maturity_rank(contract.maturity) < maturity_rank(stage):
            break
        artifact = next(
            (
                item
                for item in contract.maturity_artifacts
                if item.target_maturity == stage and item.status == "passed" and not item.mock
            ),
            None,
        )
        if artifact is None:
            return f"source_maturity_artifact_missing:{stage}"
        try:
            artifact.verify()
        except ValueError as exc:
            return f"source_maturity_artifact_invalid:{stage}:{exc}"
    return None


def _passed_artifact_path(
    contract: ComponentContract,
    target: ValidationStage,
) -> Path | None:
    artifact = next(
        (
            item
            for item in reversed(contract.maturity_artifacts)
            if item.target_maturity == target and item.status == "passed" and not item.mock
        ),
        None,
    )
    return artifact.artifact_path if artifact is not None else None


def _record_once(
    contract: ComponentContract,
    artifact: ComponentMaturityArtifact,
) -> ComponentContract:
    if any(
        item.target_maturity == artifact.target_maturity
        and item.artifact_sha256 == artifact.artifact_sha256
        for item in contract.maturity_artifacts
    ):
        return contract
    return record_maturity_artifact(contract, artifact)


__all__ = ["ComponentValidationBridge", "SmokeEvidence"]
