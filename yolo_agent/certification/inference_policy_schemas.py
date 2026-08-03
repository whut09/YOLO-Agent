"""Certification contracts for inference-only paper policies."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from yolo_agent.components.adapters.inference.policy import (
    InferencePolicyMetrics,
    InferencePolicyProtocol,
)


class InferencePolicyCertificationReport(BaseModel):
    schema_version: str = "inference_policy_certification.v1"
    status: Literal["passed", "failed", "skipped"]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model: str
    annotations: str
    protocol: InferencePolicyProtocol
    protocol_hash: str
    standard_640_metrics: dict[str, float] = Field(default_factory=dict)
    policy_metrics: InferencePolicyMetrics | None = None
    inference_policy_changed: Literal[True] = True
    training_attribution_allowed: Literal[False] = False
    checks: dict[str, bool] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    reason: str | None = None
    report_hash: str = ""

    @model_validator(mode="after")
    def _validate_report(self) -> "InferencePolicyCertificationReport":
        if self.protocol_hash != self.protocol.protocol_hash:
            raise ValueError("inference policy protocol hash mismatch")
        if any(_is_policy_metric(name) for name in self.standard_640_metrics):
            raise ValueError("standard_640_metrics cannot contain policy metrics")
        if self.status == "passed":
            if self.policy_metrics is None:
                raise ValueError("passed inference policy requires policy metrics")
            if not self.checks or not all(self.checks.values()):
                raise ValueError("passed inference policy requires all checks")
        expected = self.calculate_hash()
        if self.report_hash and self.report_hash != expected:
            raise ValueError("inference policy report hash mismatch")
        self.report_hash = expected
        return self

    def calculate_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"report_hash"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=path.name, suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                yaml.safe_dump(self.model_dump(mode="json"), file, sort_keys=False)
            os.replace(temporary_name, path)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        return path


def _is_policy_metric(name: Any) -> bool:
    text = str(name)
    return text == "inference_policy_changed" or text.startswith(
        ("sliced_", "tiled_multi_scale_", "tta_", "calibrated_", "class_threshold_", "merged_")
    )


__all__ = ["InferencePolicyCertificationReport"]
