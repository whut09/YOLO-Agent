"""Unified runtime-adapter contract built on the existing adapter SDK.

The project already has a mature ``ComponentAdapter`` and a typed
``AdapterRuntimePayload``.  This module adds the paper-facing vocabulary
around those objects instead of introducing a second execution framework.
Legacy adapters are exposed through :class:`RuntimeAdapterFacade`; new
capability-specific adapters can implement the small structural protocol
directly.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import textwrap
from collections.abc import Iterable, Mapping
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.components.adapters.base import (
    AdapterContext,
    AdapterValidationReport,
    ComponentAdapter,
    PatchPreview,
    RollbackPlan,
    SmokeTestResult,
)
from yolo_agent.components.adapters.runtime import (
    AdapterRuntimePayload,
)
from yolo_agent.components.adapters.validation import (
    validate_adapter_metadata,
    validate_rollback_plan,
    validate_runtime_plugin_hooks,
)
from yolo_agent.components.contracts import ComponentContract


RuntimeCapability = Literal[
    "data_pipeline",
    "annotation",
    "sampling",
    "augmentation",
    "graph_component",
    "loss",
    "assignment",
    "distillation",
    "training_strategy",
    "postprocess",
    "inference",
    "calibration",
]


class RuntimeAdapterMetadata(BaseModel):
    """Stable identity and capability declaration for one runtime adapter."""

    model_config = ConfigDict(extra="forbid")

    adapter_id: str
    adapter_version: str
    component_ids: list[str] = Field(min_length=1)
    supported_paper_ids: list[str] = Field(default_factory=list)
    supported_detector_families: list[str] = Field(min_length=1)
    supported_yolo_versions: list[str] = Field(min_length=1)
    insertion_point: str
    capabilities: list[RuntimeCapability] = Field(min_length=1)
    intentional_passthrough: bool = False

    @model_validator(mode="after")
    def validate_identity(self) -> "RuntimeAdapterMetadata":
        for name, values in (
            ("component_ids", self.component_ids),
            ("supported_paper_ids", self.supported_paper_ids),
            ("supported_detector_families", self.supported_detector_families),
            ("supported_yolo_versions", self.supported_yolo_versions),
            ("capabilities", self.capabilities),
        ):
            if any(not str(value).strip() for value in values):
                raise ValueError(f"runtime adapter {name} must not contain empty values")
            if len(values) != len(set(values)):
                raise ValueError(f"runtime adapter {name} must be unique")
        if not self.adapter_id.strip() or not self.adapter_version.strip():
            raise ValueError("runtime adapter identity requires id and version")
        if not self.insertion_point.strip():
            raise ValueError("runtime adapter insertion_point must not be empty")
        return self


@runtime_checkable
class RuntimeAdapterProtocol(Protocol):
    """Small common interface consumed by paper routing and validation."""

    @property
    def metadata(self) -> RuntimeAdapterMetadata: ...

    @property
    def adapter_id(self) -> str: ...

    @property
    def adapter_version(self) -> str: ...

    @property
    def component_ids(self) -> list[str]: ...

    @property
    def supported_paper_ids(self) -> list[str]: ...

    @property
    def supported_detector_families(self) -> list[str]: ...

    @property
    def supported_yolo_versions(self) -> list[str]: ...

    @property
    def insertion_point(self) -> str: ...

    def typed_runtime_payload(
        self,
        *,
        protocol_hash: str,
        base_command: list[str],
        generated_config: dict[str, Any],
    ) -> AdapterRuntimePayload: ...

    def validate_contract(self) -> AdapterValidationReport: ...

    def build(self) -> Any: ...

    def apply(
        self,
        model_config: dict[str, Any],
        training_config: dict[str, Any],
        *,
        dry_run: bool = True,
    ) -> PatchPreview: ...

    def rollback(self) -> RollbackPlan: ...

    def fingerprint(
        self,
        *,
        configuration: Any = None,
        payload: AdapterRuntimePayload | None = None,
    ) -> str: ...

    def validate_non_mock_smoke(self) -> SmokeTestResult: ...


# Capability protocols intentionally add no large shared base class.  They
# are typing markers for route-specific adapters while RuntimeAdapterProtocol
# remains the only common runtime surface.
class DataPipelineAdapter(RuntimeAdapterProtocol, Protocol):
    """Runtime adapter for dataset construction and pipeline changes."""


class AnnotationAdapter(RuntimeAdapterProtocol, Protocol):
    """Runtime adapter for annotation parsing or target construction."""


class SamplingAdapter(RuntimeAdapterProtocol, Protocol):
    """Runtime adapter for sampler changes."""


class AugmentationAdapter(RuntimeAdapterProtocol, Protocol):
    """Runtime adapter for train-time transforms."""


class GraphComponentAdapter(RuntimeAdapterProtocol, Protocol):
    """Runtime adapter for an isolated model-graph component."""


class LossAdapter(RuntimeAdapterProtocol, Protocol):
    """Runtime adapter for a loss injection."""


class AssignmentAdapter(RuntimeAdapterProtocol, Protocol):
    """Runtime adapter for assignment or matching changes."""


class DistillationAdapter(RuntimeAdapterProtocol, Protocol):
    """Runtime adapter for teacher/student training."""


class TrainingStrategyAdapter(RuntimeAdapterProtocol, Protocol):
    """Runtime adapter for a training strategy or trainer extension."""


class PostprocessAdapter(RuntimeAdapterProtocol, Protocol):
    """Runtime adapter for postprocess behavior."""


class InferenceAdapter(RuntimeAdapterProtocol, Protocol):
    """Runtime adapter for evaluation-only inference behavior."""


class CalibrationAdapter(RuntimeAdapterProtocol, Protocol):
    """Runtime adapter for confidence or score calibration."""


class RuntimeAdapterContractError(ValueError):
    """Raised when an adapter cannot satisfy the unified runtime contract."""


class RuntimeAdapterFacade:
    """Expose a legacy ``ComponentAdapter`` through the unified contract.

    The facade stores the existing invocation context.  ``apply`` returns the
    existing immutable ``PatchPreview`` rather than mutating a model, while
    ``typed_runtime_payload`` delegates to the already validated
    ``AdapterRuntimePayload`` builder.
    """

    def __init__(
        self,
        adapter: ComponentAdapter,
        contract: ComponentContract,
        context: AdapterContext,
        *,
        protocol_hash: str = "runtime-contract-unbound",
        base_command: Iterable[str] | None = None,
        generated_config: Mapping[str, Any] | None = None,
        supported_paper_ids: Iterable[str] | None = None,
        supported_yolo_versions: Iterable[str] | None = None,
        capabilities: Iterable[RuntimeCapability] | None = None,
    ) -> None:
        self.adapter = adapter
        self.contract = contract
        self.context = context
        self.protocol_hash = protocol_hash
        self.base_command = list(
            base_command
            or [
                "yolo",
                "detect",
                "train",
                "model=yolo26n.pt",
                "data=coco.yaml",
                "imgsz=640",
            ]
        )
        self.generated_config = dict(generated_config or {})
        self._metadata = _metadata_for(
            adapter,
            contract,
            supported_paper_ids=supported_paper_ids,
            supported_yolo_versions=supported_yolo_versions,
            capabilities=capabilities,
        )

    @property
    def metadata(self) -> RuntimeAdapterMetadata:
        return self._metadata

    @property
    def adapter_id(self) -> str:
        return self._metadata.adapter_id

    @property
    def adapter_version(self) -> str:
        return self._metadata.adapter_version

    @property
    def component_ids(self) -> list[str]:
        return list(self._metadata.component_ids)

    @property
    def supported_paper_ids(self) -> list[str]:
        return list(self._metadata.supported_paper_ids)

    @property
    def supported_detector_families(self) -> list[str]:
        return list(self._metadata.supported_detector_families)

    @property
    def supported_yolo_versions(self) -> list[str]:
        return list(self._metadata.supported_yolo_versions)

    @property
    def insertion_point(self) -> str:
        return self._metadata.insertion_point

    def typed_runtime_payload(
        self,
        *,
        protocol_hash: str | None = None,
        base_command: list[str] | None = None,
        generated_config: dict[str, Any] | None = None,
    ) -> AdapterRuntimePayload:
        """Return the existing typed payload, failing on legacy ``None``."""
        payload = self.adapter.build_runtime_payload(
            self.context,
            protocol_hash=protocol_hash or self.protocol_hash,
            base_command=list(base_command or self.base_command),
            generated_config=dict(
                self.generated_config if generated_config is None else generated_config
            ),
        )
        if not isinstance(payload, AdapterRuntimePayload):
            raise RuntimeAdapterContractError(
                f"adapter has no typed runtime payload: {type(self.adapter).__name__}"
            )
        return payload

    def validate_contract(
        self,
        *,
        payload: AdapterRuntimePayload | None = None,
    ) -> AdapterValidationReport:
        """Validate metadata, legacy SDK methods, payload, and hook binding.

        Callers that already built the final payload can pass it here so the
        bridge validates exactly what will execute instead of rebuilding a
        second payload from pre-patch configuration.
        """
        errors: list[str] = []
        warnings: list[str] = []
        checks: dict[str, bool | str | int | float] = {}
        try:
            validate_adapter_metadata(self.adapter)
            checks["legacy_adapter_metadata"] = True
        except (AttributeError, TypeError, ValueError) as exc:
            errors.append(f"adapter_metadata_invalid:{exc}")
            checks["legacy_adapter_metadata"] = False

        missing = [
            name
            for name in _LEGACY_ADAPTER_METHODS
            if not callable(getattr(self.adapter, name, None))
        ]
        if missing:
            errors.append("adapter_callable_missing:" + ",".join(missing))
        checks["adapter_callables_present"] = not missing

        if self.contract.component_id not in self.component_ids:
            errors.append(
                "component_identity_mismatch:"
                f"{self.contract.component_id} not in {self.component_ids}"
            )
        checks["component_identity_bound"] = self.contract.component_id in self.component_ids
        detector_family_supported = (
            self.context.detector_family in self.supported_detector_families
            or "generic" in self.supported_detector_families
        )
        if not detector_family_supported:
            errors.append(
                f"detector_family_unsupported:{self.context.detector_family}"
            )
        checks["detector_family_supported"] = detector_family_supported
        if self.context.yolo_version not in self.supported_yolo_versions:
            errors.append(
                f"yolo_version_unsupported:{self.context.yolo_version}"
            )
        checks["yolo_version_supported"] = (
            self.context.yolo_version in self.supported_yolo_versions
        )

        environment = _invoke_report(
            getattr(self.adapter, "validate_environment", None), self.context
        )
        compatibility = _invoke_report(
            getattr(self.adapter, "validate_compatibility", None), self.context
        )
        checks["environment_valid"] = environment.ok
        checks["compatibility_valid"] = compatibility.ok
        errors.extend(environment.errors)
        errors.extend(compatibility.errors)
        warnings.extend(environment.warnings)
        warnings.extend(compatibility.warnings)

        try:
            payload = payload or self.typed_runtime_payload()
            payload.verify_imports()
            checks["typed_payload"] = True
            checks.update(validate_runtime_plugin_hooks(payload))
            checks.update(
                _validate_hook_registration(
                    payload,
                    declared_hook=getattr(self.contract, "runtime_hook", None),
                )
            )
            checks["payload_identity"] = (
                self.contract.component_id in payload.component_ids
                and type(self.adapter).__name__ in payload.adapter_classes
                and payload.adapter_versions.get(self.contract.component_id)
                == self.adapter_version
            )
            if not checks["payload_identity"]:
                errors.append("runtime_payload_identity_mismatch")
            expected_changed_variable = self.contract.changed_variable
            checks["changed_variable_bound"] = (
                not expected_changed_variable
                or expected_changed_variable in payload.changed_variables
            )
            if not checks["changed_variable_bound"]:
                errors.append(
                    "runtime_payload_changed_variable_missing:"
                    f"{expected_changed_variable}"
                )
        except Exception as exc:
            errors.append(f"runtime_payload_invalid:{exc}")
            checks["typed_payload"] = False

        fake_errors = inspect_fake_implementation(
            self.adapter,
            contract=self.contract,
            built_module=None,
            payload=payload,
        )
        errors.extend(fake_errors)
        checks["fake_implementation_checks"] = not fake_errors
        return AdapterValidationReport(
            ok=not errors,
            errors=list(dict.fromkeys(str(error) for error in errors if str(error))),
            warnings=list(dict.fromkeys(str(warning) for warning in warnings if str(warning))),
            checks=checks,
        )

    def build(self) -> Any:
        """Build the existing local module and reject an undeclared Identity."""
        module = self.adapter.build_module(self.context)
        errors = inspect_fake_implementation(
            self.adapter,
            contract=self.contract,
            built_module=module,
        )
        if errors:
            raise RuntimeAdapterContractError("; ".join(errors))
        return module

    def apply(
        self,
        model_config: dict[str, Any],
        training_config: dict[str, Any],
        *,
        dry_run: bool = True,
    ) -> PatchPreview:
        """Apply the existing adapter patch API and retain its audit preview."""
        preview = self.adapter.prepare_patch(
            model_config,
            training_config,
            self.context,
            dry_run=dry_run,
        )
        if not isinstance(preview, PatchPreview):
            raise RuntimeAdapterContractError("adapter apply did not return PatchPreview")
        if not preview.operations and not self._metadata.intentional_passthrough:
            if self.contract.inference_only is not True:
                raise RuntimeAdapterContractError(
                    "adapter apply produced no declared operation without intentional_passthrough"
                )
        return preview

    def rollback(self) -> RollbackPlan:
        """Return and validate the existing workspace-scoped rollback plan."""
        plan = self.adapter.rollback_plan(self.context)
        if not isinstance(plan, RollbackPlan):
            raise RuntimeAdapterContractError("adapter rollback did not return RollbackPlan")
        validate_rollback_plan(plan, self.context.workspace)
        return plan

    def fingerprint(
        self,
        *,
        configuration: Any = None,
        payload: AdapterRuntimePayload | None = None,
    ) -> str:
        """Hash identity, contract, payload, and effective configuration."""
        content = {
            "metadata": self.metadata.model_dump(mode="json"),
            "contract_signature": self.contract.runtime_contract_signature,
            "source_commit": str(getattr(self.adapter, "source_commit", "")),
            "protocol_hash": self.protocol_hash,
            "context": self.context.model_dump(mode="json"),
            "base_command": self.base_command,
            "generated_config": self.generated_config,
            "payload_hash": payload.payload_hash if payload is not None else None,
            "configuration": configuration,
        }
        encoded = json.dumps(
            content,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def validate_non_mock_smoke(self) -> SmokeTestResult:
        """Run the existing smoke and refuse mock provenance as readiness."""
        try:
            smoke = self.adapter.smoke_test(self.context)
        except Exception as exc:
            return SmokeTestResult(
                passed=False,
                evidence_kind="local",
                errors=[f"adapter_smoke_failed:{exc}"],
            )
        if not smoke.passed:
            return smoke
        if smoke.evidence_kind == "mock":
            return smoke.model_copy(
                update={
                    "passed": False,
                    "errors":[*smoke.errors, "mock_smoke_provenance_rejected"],
                }
            )
        fake_errors = inspect_fake_implementation(
            self.adapter,
            contract=self.contract,
            built_module=None,
        )
        if fake_errors:
            return smoke.model_copy(update={"passed": False, "errors": fake_errors})
        return smoke


def adapt_component_adapter(
    adapter: ComponentAdapter,
    contract: ComponentContract,
    context: AdapterContext,
    **kwargs: Any,
) -> RuntimeAdapterFacade:
    """Create the compatibility facade used by current execution bridges."""
    return RuntimeAdapterFacade(adapter, contract, context, **kwargs)


def validate_runtime_adapter(
    adapter: ComponentAdapter,
    *,
    contract: ComponentContract,
    context: AdapterContext,
    protocol_hash: str = "runtime-contract-unbound",
    base_command: Iterable[str] | None = None,
    generated_config: Mapping[str, Any] | None = None,
    check_smoke: bool = False,
) -> AdapterValidationReport:
    """Validate a legacy adapter through the unified contract boundary."""
    facade = adapt_component_adapter(
        adapter,
        contract,
        context,
        protocol_hash=protocol_hash,
        base_command=base_command,
        generated_config=generated_config,
    )
    report = facade.validate_contract()
    if check_smoke:
        smoke = facade.validate_non_mock_smoke()
        checks = dict(report.checks)
        checks["non_mock_smoke"] = smoke.passed
        errors = [*report.errors, *smoke.errors]
        report = report.model_copy(
            update={
                "ok": report.ok and smoke.passed,
                "errors": list(dict.fromkeys(errors)),
                "checks": checks,
            }
        )
    return report


def assert_runtime_adapter(
    adapter: ComponentAdapter,
    *,
    contract: ComponentContract,
    context: AdapterContext,
    **kwargs: Any,
) -> RuntimeAdapterFacade:
    """Return a facade or raise a descriptive contract error."""
    facade = adapt_component_adapter(adapter, contract, context, **kwargs)
    report = facade.validate_contract()
    if not report.ok:
        raise RuntimeAdapterContractError("; ".join(report.errors))
    return facade


_LEGACY_ADAPTER_METHODS = (
    "validate_environment",
    "validate_compatibility",
    "patch_model_config",
    "patch_training_config",
    "build_module",
    "load_pretrained_weights",
    "smoke_test",
    "expected_artifacts",
    "rollback_plan",
    "build_runtime_payload",
)


def _metadata_for(
    adapter: Any,
    contract: ComponentContract,
    *,
    supported_paper_ids: Iterable[str] | None,
    supported_yolo_versions: Iterable[str] | None,
    capabilities: Iterable[RuntimeCapability] | None,
) -> RuntimeAdapterMetadata:
    component_ids = _string_list(
        getattr(adapter, "component_ids", None),
        fallback=[contract.component_id],
    )
    papers = _string_list(
        supported_paper_ids
        if supported_paper_ids is not None
        else getattr(adapter, "supported_paper_ids", None),
        fallback=contract.source_papers,
    )
    detector_families = _string_list(
        getattr(adapter, "supported_detector_families", None),
        fallback=contract.supported_detector_families or ["generic"],
    )
    versions = _string_list(
        supported_yolo_versions
        if supported_yolo_versions is not None
        else getattr(adapter, "supported_yolo_versions", None),
        fallback=getattr(contract, "supported_yolo_versions", None) or ["26"],
    )
    inferred_capabilities = _string_list(
        capabilities
        if capabilities is not None
        else getattr(adapter, "capabilities", None),
        fallback=[_capability_for_contract(contract)],
    )
    return RuntimeAdapterMetadata(
        adapter_id=str(getattr(adapter, "adapter_id", "") or contract.component_id),
        adapter_version=str(getattr(adapter, "adapter_version", "")),
        component_ids=component_ids,
        supported_paper_ids=papers,
        supported_detector_families=detector_families,
        supported_yolo_versions=versions,
        insertion_point=str(
            getattr(adapter, "insertion_point", "") or contract.insertion_point
        ),
        capabilities=inferred_capabilities,  # type: ignore[arg-type]
        intentional_passthrough=bool(
            getattr(adapter, "intentional_passthrough", False)
            or contract.inference_only is True
        ),
    )


def _string_list(value: Any, *, fallback: Iterable[str]) -> list[str]:
    if value is None:
        values = list(fallback)
    elif isinstance(value, str):
        values = [value]
    else:
        values = [str(item) for item in value]
    return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _capability_for_contract(contract: ComponentContract) -> RuntimeCapability:
    category = contract.category.lower()
    component = contract.component_id.lower()
    if component.startswith("inference.") or contract.inference_only is True:
        return "inference"
    if component.startswith("sampling."):
        return "sampling"
    if component.startswith("augmentation.") or category in {"augmentation", "transform"}:
        return "augmentation"
    if component.startswith("annotation."):
        return "annotation"
    if component.startswith("neck.") or component.startswith("attention.") or component.startswith("feature_pyramid.") or component.startswith("detection_head."):
        return "graph_component"
    if component.startswith("assigner."):
        return "assignment"
    if component.startswith("distillation."):
        return "distillation"
    if component.startswith("loss.calibration.") or category == "calibration":
        return "calibration"
    if component.startswith("loss.") or "loss" in category:
        return "loss"
    if "postprocess" in category or component.startswith("postprocess."):
        return "postprocess"
    if "strategy" in category:
        return "training_strategy"
    if "data" in category:
        return "data_pipeline"
    return "training_strategy"


def _invoke_report(method: Any, context: AdapterContext) -> AdapterValidationReport:
    if not callable(method):
        return AdapterValidationReport(ok=False, errors=["adapter_validation_callable_missing"])
    try:
        result = method(context)
    except Exception as exc:
        return AdapterValidationReport(ok=False, errors=[str(exc)])
    if not isinstance(result, AdapterValidationReport):
        return AdapterValidationReport(
            ok=False,
            errors=["adapter validation did not return AdapterValidationReport"],
        )
    return result


def _validate_hook_registration(
    payload: AdapterRuntimePayload,
    *,
    declared_hook: str | None,
) -> dict[str, bool | str | int | float]:
    references = payload.plugin_references
    if not references:
        raise RuntimeAdapterContractError("declared runtime hook has no plugin registration")
    registered = {
        hook
        for reference in references
        for hook in reference.required_hooks
    }
    if declared_hook and declared_hook not in registered:
        raise RuntimeAdapterContractError(
            f"declared runtime hook is not registered: {declared_hook}"
        )
    return {
        "declared_runtime_hook_registered": (
            declared_hook is None or declared_hook in registered
        ),
        "registered_runtime_hook_count": len(registered),
    }


def inspect_fake_implementation(
    adapter: Any,
    *,
    contract: ComponentContract | None = None,
    built_module: Any = None,
    payload: AdapterRuntimePayload | None = None,
) -> list[str]:
    """Return deterministic findings for common metadata-only fakes."""
    del payload
    intentional = bool(
        getattr(adapter, "intentional_passthrough", False)
        or contract is not None
        and contract.inference_only is True
    )
    findings: list[str] = []
    for name in ("build", "build_module"):
        if _source_returns_identity_only(getattr(adapter, name, None)) and not intentional:
            findings.append("fake_identity_only_implementation")
            break
    for name in ("apply",):
        if _source_returns_input(getattr(adapter, name, None)) and not intentional:
            findings.append("fake_passthrough_apply")
            break
    for name in ("compute_loss", "loss", "forward"):
        if _source_returns_constant(getattr(adapter, name, None)) and not intentional:
            findings.append("fake_constant_loss_or_forward")
            break
    if built_module is not None and _is_identity_module(built_module) and not intentional:
        findings.append("fake_identity_only_implementation")
    return list(dict.fromkeys(findings))


def _source_returns_identity_only(method: Any) -> bool:
    source = _method_source(method)
    if source is None:
        return False
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    returns = [node.value for node in ast.walk(tree) if isinstance(node, ast.Return)]
    if not returns:
        return False
    return all(
        isinstance(value, ast.Call)
        and _qualified_name(value.func).split(".")[-1].lower() == "identity"
        for value in returns
    )


def _source_returns_input(method: Any) -> bool:
    source = _method_source(method)
    if source is None:
        return False
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    function = next(
        (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if function is None:
        return False
    parameters = [
        argument.arg
        for argument in [*function.args.posonlyargs, *function.args.args]
        if argument.arg != "self"
    ]
    returns = [node.value for node in ast.walk(function) if isinstance(node, ast.Return)]
    return bool(parameters and returns) and all(
        isinstance(value, ast.Name) and value.id == parameters[0]
        for value in returns
    )


def _source_returns_constant(method: Any) -> bool:
    source = _method_source(method)
    if source is None:
        return False
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    returns = [node.value for node in ast.walk(tree) if isinstance(node, ast.Return)]
    if not returns:
        return False
    return all(
        isinstance(value, ast.Constant)
        or (
            isinstance(value, ast.Call)
            and _qualified_name(value.func).split(".")[-1].lower()
            in {"zeros", "ones", "full", "tensor"}
            and any(
                isinstance(argument, ast.Constant) and argument.value == 0
                for argument in value.args
            )
        )
        for value in returns
    )


def _method_source(method: Any) -> str | None:
    if not callable(method):
        return None
    try:
        return textwrap.dedent(inspect.getsource(method))
    except (OSError, TypeError):
        return None


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _is_identity_module(value: Any) -> bool:
    try:
        import torch.nn as nn
    except ImportError:
        return False
    if isinstance(value, nn.Identity):
        return True
    if isinstance(value, nn.Sequential):
        modules = list(value.children())
        return bool(modules) and all(isinstance(item, nn.Identity) for item in modules)
    return False


__all__ = [
    "AnnotationAdapter",
    "AssignmentAdapter",
    "AugmentationAdapter",
    "CalibrationAdapter",
    "DataPipelineAdapter",
    "DistillationAdapter",
    "GraphComponentAdapter",
    "InferenceAdapter",
    "LossAdapter",
    "PostprocessAdapter",
    "RuntimeAdapterContractError",
    "RuntimeAdapterFacade",
    "RuntimeAdapterMetadata",
    "RuntimeAdapterProtocol",
    "RuntimeCapability",
    "SamplingAdapter",
    "TrainingStrategyAdapter",
    "adapt_component_adapter",
    "assert_runtime_adapter",
    "inspect_fake_implementation",
    "validate_runtime_adapter",
]
