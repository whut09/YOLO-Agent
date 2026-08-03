"""Contracts for resumable batch certification of reusable paper adapters."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.components.maturity import MaturityName
from yolo_agent.core.yaml_io import YAMLModelMixin


BatchCertificationMode = Literal["cpu", "gpu"]
BatchAdapterStatus = Literal[
    "passed",
    "failed",
    "blocked",
    "skipped_resume",
    "skipped_unchanged",
]
BatchCertificationStatus = Literal["passed", "partial", "failed", "blocked"]


class AdapterCertificationIdentity(BaseModel):
    """Environment identity that invalidates stale certification evidence."""

    model_config = ConfigDict(extra="forbid")

    component_id: str
    adapter_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    code_commit: str = Field(min_length=1)
    ultralytics_version: str = Field(min_length=1)
    protocol_hash: str = Field(min_length=1)

    @property
    def identity_hash(self) -> str:
        return _stable_hash(self.model_dump(mode="json"))


class PaperAdapterCertificationResult(BaseModel, YAMLModelMixin):
    """Independent terminal result for one discovered reusable adapter."""

    model_config = ConfigDict(extra="forbid")

    component_id: str
    identity: AdapterCertificationIdentity
    status: BatchAdapterStatus
    initial_maturity: MaturityName
    final_maturity: MaturityName
    selection_reason: str
    cpu_report: Path | None = None
    gpu_report: Path | None = None
    matched_pilot_fixture: Path | None = None
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _identity_matches_component(self) -> "PaperAdapterCertificationResult":
        if self.component_id != self.identity.component_id:
            raise ValueError("batch result identity must match component_id")
        if self.status == "passed" and self.errors:
            raise ValueError("passed batch adapter result cannot contain errors")
        return self


class PaperAdapterCertificationReport(BaseModel, YAMLModelMixin):
    """Resumable batch checkpoint and final certification report."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_adapter_certification_factory.v1"
    status: BatchCertificationStatus
    mode: BatchCertificationMode
    execute_real_gpu: bool = False
    resume: bool = False
    changed_only: bool = False
    registry_path: Path
    coverage_report_path: Path | None = None
    discovery_errors: dict[str, str] = Field(default_factory=dict)
    selected_component_ids: list[str] = Field(default_factory=list)
    results: list[PaperAdapterCertificationResult] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    report_hash: str = ""

    @model_validator(mode="after")
    def _bind_report_hash(self) -> "PaperAdapterCertificationReport":
        result_ids = [item.component_id for item in self.results]
        if len(result_ids) != len(set(result_ids)):
            raise ValueError("batch certification result components must be unique")
        if not set(result_ids).issubset(self.selected_component_ids):
            raise ValueError("batch results must be selected components")
        expected = self.calculate_hash()
        if self.report_hash and self.report_hash != expected:
            raise ValueError("batch certification report hash mismatch")
        self.report_hash = expected
        return self

    def calculate_hash(self) -> str:
        payload = self.model_dump(
            mode="json",
            exclude={"report_hash", "generated_at"},
        )
        return _stable_hash(payload)


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "AdapterCertificationIdentity",
    "BatchAdapterStatus",
    "BatchCertificationMode",
    "BatchCertificationStatus",
    "PaperAdapterCertificationReport",
    "PaperAdapterCertificationResult",
]
