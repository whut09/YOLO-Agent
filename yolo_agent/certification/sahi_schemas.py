"""Contracts for isolated SAHI inference certification."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from yolo_agent.components.adapters.inference.slicing import (
    SlicingInferenceMetrics,
    SlicingInferenceProtocol,
)


SahiCertificationStatus = Literal["passed", "failed", "skipped"]


class SahiDependencyStatus(BaseModel):
    available: bool
    version: str | None = None
    reason: str | None = None


class SahiCertificationReport(BaseModel):
    """Machine-readable proof that slicing stayed outside training attribution."""

    schema_version: str = "sahi_inference_certification.v1"
    status: SahiCertificationStatus
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model: str
    annotations: str
    protocol: SlicingInferenceProtocol
    protocol_hash: str
    dependency: SahiDependencyStatus
    standard_640_metrics: dict[str, float] = Field(default_factory=dict)
    sliced_inference_metrics: SlicingInferenceMetrics | None = None
    inference_policy_changed: bool = True
    training_attribution_allowed: bool = False
    checks: dict[str, bool] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    reason: str | None = None
    report_hash: str = ""

    @model_validator(mode="after")
    def _validate_isolation(self) -> "SahiCertificationReport":
        if any(name.startswith("sliced_") for name in self.standard_640_metrics):
            raise ValueError("standard_640_metrics cannot contain sliced metrics")
        if self.training_attribution_allowed:
            raise ValueError("SAHI certification cannot allow training attribution")
        expected_protocol = protocol_hash(self.protocol)
        if self.protocol_hash != expected_protocol:
            raise ValueError("SAHI protocol hash does not match protocol payload")
        if self.status == "passed":
            if self.sliced_inference_metrics is None:
                raise ValueError("passed SAHI certification requires sliced metrics")
            if not self.checks or not all(self.checks.values()):
                raise ValueError("passed SAHI certification requires all checks to pass")
        expected_report = self.calculate_hash()
        if self.report_hash and self.report_hash != expected_report:
            raise ValueError("SAHI certification report hash mismatch")
        self.report_hash = expected_report
        return self

    def calculate_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"report_hash"})
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                yaml.safe_dump(self.model_dump(mode="json"), file, sort_keys=False)
            os.replace(temporary_name, path)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        return path


def protocol_hash(protocol: SlicingInferenceProtocol) -> str:
    return hashlib.sha256(
        json.dumps(
            protocol.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "SahiCertificationReport",
    "SahiCertificationStatus",
    "SahiDependencyStatus",
    "protocol_hash",
]
