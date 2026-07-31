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


class ComponentGPUProtocol(BaseModel):
    """Immutable identity for one real component GPU certification."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "component_gpu_protocol.v1"
    component_id: str
    adapter_hash: str
    runtime_payload_hash: str
    fixture_manifest_hash: str
    model_sha256: str
    ultralytics_version: str
    device: str
    imgsz: int = Field(default=640, frozen=True)
    batch: int = Field(default=2, ge=1)
    initial_epochs: int = Field(default=1, ge=1)
    resumed_epochs: int = Field(default=2, ge=2)
    amp: bool = True
    protocol_hash: str = ""

    @model_validator(mode="after")
    def bind_protocol_hash(self) -> "ComponentGPUProtocol":
        payload = self.model_dump(mode="json", exclude={"protocol_hash"})
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if self.protocol_hash and self.protocol_hash != expected:
            raise ValueError("component GPU protocol hash mismatch")
        self.protocol_hash = expected
        return self


class ComponentGPUResources(BaseModel):
    """Measured resources from a real CUDA training and resume run."""

    model_config = ConfigDict(extra="forbid")

    device: str
    gpu_name: str
    total_vram_mb: float = Field(ge=0.0)
    peak_vram_mb: float = Field(ge=0.0)
    train_duration_s: float = Field(ge=0.0)
    resume_duration_s: float = Field(ge=0.0)
    latency_ms: float = Field(ge=0.0)
    model_size_mb: float = Field(gt=0.0)


class ComponentGPUCertificationEvidence(BaseModel, YAMLModelMixin):
    """Artifact contract required before GPU maturity can advance."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "component_gpu_certification_evidence.v1"
    component_id: str
    status: Literal["passed", "failed"]
    worker_protocol_hash: str
    gpu_protocol: ComponentGPUProtocol
    runtime_payload_path: Path
    runtime_payload_hash: str
    train_command: list[str] = Field(default_factory=list)
    resume_command: list[str] = Field(default_factory=list)
    hook_call_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    checks: dict[str, bool | str | int | float] = Field(default_factory=dict)
    resources: ComponentGPUResources | None = None
    artifacts: dict[str, Path] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_passed_evidence(self) -> "ComponentGPUCertificationEvidence":
        if self.runtime_payload_hash != self.gpu_protocol.runtime_payload_hash:
            raise ValueError("GPU evidence runtime payload hash mismatch")
        if self.status == "passed":
            required = {
                "real_ultralytics_train",
                "required_hooks_observed",
                "backward_observed",
                "amp_enabled",
                "checkpoint_saved",
                "resume_completed",
                "resume_checkpoint_saved",
                "adapter_hash_matched",
                "fixture_manifest_matched",
                "adapter_artifacts_complete",
                "component_profile_verified",
                "stateful_resume_hook_observed",
            }
            failed = sorted(name for name in required if self.checks.get(name) is not True)
            if failed:
                raise ValueError(
                    "passed GPU evidence has failed checks: " + ", ".join(failed)
                )
            if self.resources is None or self.errors:
                raise ValueError("passed GPU evidence requires resources and no errors")
        return self


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
    model: str = "yolo26n.pt"
    adapter_hash: str | None = None
    ultralytics_version: str | None = None
    real_gpu_training: bool = False
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
    "ComponentGPUCertificationEvidence",
    "ComponentGPUProtocol",
    "ComponentGPUResources",
    "ComponentSmokeWorkerReport",
    "ComponentSmokeWorkerRequest",
]
