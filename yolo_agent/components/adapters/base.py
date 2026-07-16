"""Base interfaces and schemas for controlled component adapters."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.components.contracts import ComponentContract


AdapterStrategy = Literal[
    "custom_module",
    "custom_model_yaml",
    "callback",
    "trainer_subclass",
    "loss_injection",
    "assigner_injection",
    "inference_adapter",
]
PatchTarget = Literal["model_config", "training_config"]


class AdapterContext(BaseModel):
    """Immutable inputs supplied to an adapter invocation."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    contract: ComponentContract
    framework: str = "ultralytics"
    detector_family: str = "generic"
    head: str | None = None
    imgsz: int = Field(default=640, ge=1)
    workspace: Path = Path(".")
    environment: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)


class AdapterValidationReport(BaseModel):
    """Structured environment or compatibility validation result."""

    ok: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checks: dict[str, bool | str | int | float] = Field(default_factory=dict)


class PatchOperation(BaseModel):
    """One field-level configuration change produced by an adapter."""

    target: PatchTarget
    field: str
    before: Any = None
    after: Any = None


class ExpectedArtifact(BaseModel):
    """Artifact expected after applying and executing an adapter."""

    name: str
    relative_path: Path
    required: bool = True


class RollbackPlan(BaseModel):
    """How to discard an adapter patch without touching global packages."""

    reversible: bool = True
    actions: list[str] = Field(default_factory=list)
    files_to_remove: list[Path] = Field(default_factory=list)
    restores_global_source: bool = False


class SmokeTestResult(BaseModel):
    """Result of an adapter-level smoke check."""

    passed: bool
    checks: dict[str, bool | str] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class WeightLoadResult(BaseModel):
    """Result of optional component weight loading."""

    loaded: bool
    source: Path | None = None
    missing_keys: list[str] = Field(default_factory=list)
    unexpected_keys: list[str] = Field(default_factory=list)
    message: str = ""


class PatchPreview(BaseModel):
    """Auditable, non-destructive preview of an adapter patch."""

    component_id: str
    adapter_class: str
    adapter_version: str
    source_commit: str
    strategy: AdapterStrategy
    dry_run: bool = True
    operations: list[PatchOperation] = Field(default_factory=list)
    declared_modified_fields: list[str] = Field(default_factory=list)
    patched_model_config: dict[str, Any] = Field(default_factory=dict)
    patched_training_config: dict[str, Any] = Field(default_factory=dict)
    expected_artifacts: list[ExpectedArtifact] = Field(default_factory=list)
    rollback: RollbackPlan
    idempotency_key: str
    warnings: list[str] = Field(default_factory=list)


class ComponentAdapter(ABC):
    """Uniform SDK boundary for a locally implemented paper component."""

    adapter_version: str
    source_commit: str
    strategy: AdapterStrategy
    modified_model_fields: frozenset[str] = frozenset()
    modified_training_fields: frozenset[str] = frozenset()

    @abstractmethod
    def validate_environment(self, context: AdapterContext) -> AdapterValidationReport:
        """Check imports, files, and local runtime requirements."""

    @abstractmethod
    def validate_compatibility(self, context: AdapterContext) -> AdapterValidationReport:
        """Check the component contract against the target detector."""

    @abstractmethod
    def patch_model_config(
        self, config: dict[str, Any], context: AdapterContext, *, dry_run: bool = True
    ) -> dict[str, Any]:
        """Return a patched model configuration without mutating input."""

    @abstractmethod
    def patch_training_config(
        self, config: dict[str, Any], context: AdapterContext, *, dry_run: bool = True
    ) -> dict[str, Any]:
        """Return a patched training configuration without mutating input."""

    @abstractmethod
    def build_module(self, context: AdapterContext) -> Any:
        """Build a custom module or injection object in local code."""

    @abstractmethod
    def load_pretrained_weights(
        self, module: Any, weights: Path | str | None, context: AdapterContext
    ) -> WeightLoadResult:
        """Load optional component weights without modifying global packages."""

    @abstractmethod
    def smoke_test(self, context: AdapterContext) -> SmokeTestResult:
        """Run a cheap, local adapter smoke test."""

    @abstractmethod
    def expected_artifacts(self, context: AdapterContext) -> list[ExpectedArtifact]:
        """Declare artifacts the adapter execution must produce."""

    @abstractmethod
    def rollback_plan(self, context: AdapterContext) -> RollbackPlan:
        """Describe how generated local changes are discarded."""

    def prepare_patch(
        self,
        model_config: dict[str, Any],
        training_config: dict[str, Any],
        context: AdapterContext,
        *,
        dry_run: bool = True,
    ) -> PatchPreview:
        """Build, validate, and preview an idempotent configuration patch."""
        from yolo_agent.components.adapters.validation import (
            diff_config,
            validate_adapter_metadata,
            validate_declared_operations,
            validate_rollback_plan,
        )

        validate_adapter_metadata(self)
        environment = self.validate_environment(context)
        compatibility = self.validate_compatibility(context)
        if not environment.ok or not compatibility.ok:
            errors = [*environment.errors, *compatibility.errors]
            raise ValueError("Adapter validation failed: " + "; ".join(errors))

        original_model = _json_clone(model_config)
        original_training = _json_clone(training_config)
        patched_model = self.patch_model_config(_json_clone(model_config), context, dry_run=dry_run)
        patched_training = self.patch_training_config(_json_clone(training_config), context, dry_run=dry_run)
        operations = [
            *diff_config(original_model, patched_model, target="model_config"),
            *diff_config(original_training, patched_training, target="training_config"),
        ]
        validate_declared_operations(self, operations)
        rollback = self.rollback_plan(context)
        validate_rollback_plan(rollback, context.workspace)
        payload = {
            "component_id": context.contract.component_id,
            "adapter_class": type(self).__name__,
            "adapter_version": self.adapter_version,
            "source_commit": self.source_commit,
            "strategy": self.strategy,
        }
        fingerprint = {**payload, "operations": [item.model_dump(mode="json") for item in operations]}
        return PatchPreview(
            **payload,
            dry_run=dry_run,
            operations=operations,
            declared_modified_fields=sorted(
                {f"model_config.{item}" for item in self.modified_model_fields}
                | {f"training_config.{item}" for item in self.modified_training_fields}
            ),
            patched_model_config=patched_model,
            patched_training_config=patched_training,
            expected_artifacts=self.expected_artifacts(context),
            rollback=rollback,
            idempotency_key=hashlib.sha256(
                json.dumps(fingerprint, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest(),
            warnings=[*environment.warnings, *compatibility.warnings],
        )


def _json_clone(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, default=str))


__all__ = [
    "AdapterContext",
    "AdapterStrategy",
    "AdapterValidationReport",
    "ComponentAdapter",
    "ExpectedArtifact",
    "PatchOperation",
    "PatchPreview",
    "RollbackPlan",
    "SmokeTestResult",
    "WeightLoadResult",
]
