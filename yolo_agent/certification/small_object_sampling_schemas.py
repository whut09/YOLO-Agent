"""Contracts for the small-object sampling CPU golden path."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin


class SmallObjectSamplingCpuReport(BaseModel, YAMLModelMixin):
    """Artifact-backed result from the isolated sampling runtime fixture."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "small_object_sampling_cpu_golden_path.v1"
    component_id: Literal["sampling.small_object"] = "sampling.small_object"
    status: Literal["passed", "failed"]
    protocol_hash: str
    runtime_payload_hash: str
    sampler_manifest_path: Path | None = None
    runtime_evidence_path: Path | None = None
    sampler_state_path: Path | None = None
    checks: dict[str, bool | str | int | float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    report_hash: str = ""

    @model_validator(mode="after")
    def validate_report(self) -> "SmallObjectSamplingCpuReport":
        if self.status == "passed":
            required = {
                "train_dataloader_hook_called",
                "sampler_manifest_verified",
                "ddp_deterministic_sharding",
                "resume_state_restored",
                "validation_loader_unchanged",
            }
            missing = sorted(
                key for key in required if self.checks.get(key) is not True
            )
            if missing:
                raise ValueError(
                    "passed sampling CPU report is missing checks: "
                    + ", ".join(missing)
                )
            if self.errors:
                raise ValueError("passed sampling CPU report cannot contain errors")
            if not all(
                path is not None
                for path in (
                    self.sampler_manifest_path,
                    self.runtime_evidence_path,
                    self.sampler_state_path,
                )
            ):
                raise ValueError("passed sampling CPU report requires all runtime artifacts")
        expected = self.calculate_hash()
        if self.report_hash and self.report_hash != expected:
            raise ValueError("sampling CPU report hash mismatch")
        self.report_hash = expected
        return self

    def calculate_hash(self) -> str:
        payload: dict[str, Any] = self.model_dump(
            mode="json",
            exclude={"generated_at", "report_hash"},
        )
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()


__all__ = ["SmallObjectSamplingCpuReport"]
