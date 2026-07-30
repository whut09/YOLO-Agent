"""Machine-readable contracts for isolated component runtime certification."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.maturity import MaturityName
from yolo_agent.core.yaml_io import YAMLModelMixin


ComponentCertificationMode = Literal["cpu", "gpu"]
ComponentCertificationStatus = Literal["passed", "failed", "blocked"]
ComponentCertificationStageStatus = Literal["passed", "failed", "blocked", "skipped"]


class ComponentCertificationStage(BaseModel):
    """One auditable stage in a component certification run."""

    model_config = ConfigDict(extra="forbid")

    stage_id: str
    status: ComponentCertificationStageStatus
    message: str = ""
    artifacts: dict[str, Path] = Field(default_factory=dict)
    checks: dict[str, bool | str | int | float] = Field(default_factory=dict)


class ComponentSmokeWorkerReport(BaseModel, YAMLModelMixin):
    """Result emitted by the isolated CPU/GPU smoke subprocess."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "component_smoke_worker.v1"
    component_id: str
    mode: ComponentCertificationMode
    status: Literal["passed", "failed"]
    protocol_hash: str
    payload_hash: str
    evidence_kind: Literal["mock", "local"] = "mock"
    process_id: int = Field(ge=1)
    cuda_available: bool = False
    device: str = "cpu"
    checks: dict[str, bool | str | int | float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ComponentSmokeWorkerRequest(BaseModel, YAMLModelMixin):
    """Serializable input consumed by the isolated smoke subprocess."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "component_smoke_worker_request.v1"
    contract: ComponentContract
    mode: ComponentCertificationMode
    protocol_hash: str
    runtime_payload_path: Path
    workspace: Path
    device: str = "cpu"
    options: dict[str, Any] = Field(default_factory=dict)


class ComponentCertificationReport(BaseModel, YAMLModelMixin):
    """Terminal report for one component CPU or GPU certification."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "component_runtime_certification.v1"
    component_id: str
    mode: ComponentCertificationMode
    status: ComponentCertificationStatus
    initial_maturity: MaturityName
    final_maturity: MaturityName
    next_maturity: MaturityName | None = None
    protocol_hash: str
    adapter_hash: str | None = None
    code_commit: str | None = None
    ultralytics_version: str | None = None
    registry_path: Path
    workdir: Path
    stages: list[ComponentCertificationStage] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)
    generated_paths: dict[str, Path] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    report_hash: str = ""

    @model_validator(mode="after")
    def validate_terminal_report(self) -> "ComponentCertificationReport":
        if self.status == "passed":
            required = (
                {
                    "adapter_import",
                    "runtime_payload",
                    "hook_signature",
                    "unit_tests",
                    "isolated_smoke",
                }
                if self.mode == "cpu"
                else {"cpu_smoke_precondition", "isolated_gpu_smoke"}
            )
            passed = {
                item.stage_id for item in self.stages if item.status == "passed"
            }
            missing = sorted(required - passed)
            if missing:
                raise ValueError(
                    "passed component certification is missing stages: "
                    + ", ".join(missing)
                )
            if self.errors or self.missing_artifacts:
                raise ValueError("passed component certification cannot report blockers")
        expected = self.calculate_hash()
        if self.report_hash and self.report_hash != expected:
            raise ValueError("component certification report hash mismatch")
        self.report_hash = expected
        return self

    def calculate_hash(self) -> str:
        payload: dict[str, Any] = self.model_dump(
            mode="json",
            exclude={"report_hash", "generated_at"},
        )
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ComponentCertificationMode",
    "ComponentCertificationReport",
    "ComponentCertificationStage",
    "ComponentCertificationStatus",
    "ComponentSmokeWorkerReport",
    "ComponentSmokeWorkerRequest",
]
