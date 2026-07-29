"""Serializable contracts for offline component runtime validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.maturity import MaturityName
from yolo_agent.core.yaml_io import YAMLModelMixin


ValidationStage = Literal["runtime_integrated", "unit_tested", "smoke_passed"]
ValidationStageStatus = Literal["passed", "failed", "retained_mock", "skipped"]
ComponentValidationStatus = Literal["completed", "failed", "blocked"]


class ComponentValidationStageReport(BaseModel, YAMLModelMixin):
    """One immutable validation-stage observation."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "component_validation_stage.v1"
    component_id: str
    stage: ValidationStage
    status: ValidationStageStatus
    protocol_hash: str
    validation_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    producer: str = "ComponentValidationBridge"
    mock: bool = False
    checks: dict[str, bool | str | int | float] = Field(default_factory=dict)
    artifacts: dict[str, Path] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ComponentValidationResult(BaseModel, YAMLModelMixin):
    """Recoverable result of validating one adapter without training."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "component_validation.v1"
    status: ComponentValidationStatus
    component_id: str
    initial_maturity: MaturityName
    final_maturity: MaturityName
    target_maturity: ValidationStage
    protocol_hash: str
    validation_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract: ComponentContract
    stage_reports: dict[ValidationStage, Path] = Field(default_factory=dict)
    patch_preview_path: Path | None = None
    runtime_payload_path: Path | None = None
    blocked_by: list[str] = Field(default_factory=list)


__all__ = [
    "ComponentValidationResult",
    "ComponentValidationStageReport",
    "ComponentValidationStatus",
    "ValidationStage",
    "ValidationStageStatus",
]
