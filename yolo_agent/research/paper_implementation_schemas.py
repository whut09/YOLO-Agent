"""Paper-level implementation readiness schemas.

Component maturity describes a reusable runtime primitive.  These models keep
the paper identity, configuration, and evidence boundary separate so a shared
adapter can never be mistaken for an implementation of every source paper.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin


PaperImplementationReadiness = Literal[
    "cataloged",
    "profiled",
    "spec_complete",
    "code_bound",
    "runtime_integrated",
    "unit_tested",
    "smoke_passed",
    "implementation_ready",
    "pilot_reproduced",
    "full_reproduced",
    "confirmed_multi_seed",
]

ImplementationEvidenceClass = Literal[
    "paper_specific",
    "generic_only",
    "metadata_only",
    "recipe_only",
    "mock_only",
    "no_op",
    "unknown",
]

SHA256_PATTERN = r"^[0-9a-f]{64}$"
_DISALLOWED_READY_CLASSES = {
    "generic_only",
    "metadata_only",
    "recipe_only",
    "mock_only",
    "no_op",
    "unknown",
}
_CRITICAL_BLOCKER_MARKERS = (
    "missing_",
    "missing_runtime",
    "missing_test",
    "unknown_mechanism",
    "mock_only",
    "no_op",
    "generic_only",
    "metadata_only",
    "recipe_only",
    "blocked_",
    "incompatible",
    "adapter_",
    "component_",
    "paper_specific_evidence",
)


class PaperImplementationSpec(BaseModel, YAMLModelMixin):
    """Independent implementation contract for one frozen paper."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_implementation_spec.v1"
    paper_id: str
    manifest_membership_hash: str = Field(pattern=SHA256_PATTERN)
    method_profile_id: str
    title: str
    year: int = Field(ge=1900, le=2100)
    implementation_domain: str
    mechanism_summary: str
    paper_specific_mechanisms: list[str] = Field(default_factory=list)
    shared_primitives: list[str] = Field(default_factory=list)
    component_ids: list[str] = Field(default_factory=list)
    adapter_ids: list[str] = Field(default_factory=list)
    runtime_insertion_points: list[str] = Field(default_factory=list)
    runtime_hooks: list[str] = Field(default_factory=list)
    architecture_changes: list[str] = Field(default_factory=list)
    data_changes: list[str] = Field(default_factory=list)
    annotation_changes: list[str] = Field(default_factory=list)
    sampling_changes: list[str] = Field(default_factory=list)
    augmentation_changes: list[str] = Field(default_factory=list)
    loss_changes: list[str] = Field(default_factory=list)
    assignment_changes: list[str] = Field(default_factory=list)
    training_changes: list[str] = Field(default_factory=list)
    postprocess_changes: list[str] = Field(default_factory=list)
    inference_changes: list[str] = Field(default_factory=list)
    paper_specific_config: dict[str, Any] = Field(default_factory=dict)
    paper_specific_hyperparameters: dict[str, Any] = Field(default_factory=dict)
    required_assets: list[str] = Field(default_factory=list)
    paper_evidence_refs: list[str] = Field(default_factory=list)
    official_code_refs: list[str] = Field(default_factory=list)
    source_license: str | None = None
    unit_test_refs: list[str] = Field(default_factory=list)
    smoke_test_refs: list[str] = Field(default_factory=list)
    compatibility_test_refs: list[str] = Field(default_factory=list)
    implementation_fingerprint: str = Field(pattern=SHA256_PATTERN)
    readiness: PaperImplementationReadiness = "cataloged"
    blockers: list[str] = Field(default_factory=list)
    implementation_evidence_class: ImplementationEvidenceClass = "unknown"
    runtime_implementation_verified: bool = False
    unit_tests_passed: bool = False
    non_mock_smoke_passed: bool = False
    compatibility_validation_passed: bool = False
    pilot_reproduced: bool = False
    full_reproduced: bool = False
    confirmed_multi_seed: bool = False
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_boundary(self) -> "PaperImplementationSpec":
        if not self.paper_id.strip():
            raise ValueError("paper implementation spec requires paper_id")
        if not self.method_profile_id.strip():
            raise ValueError("paper implementation spec requires method_profile_id")
        if not self.title.strip() or not self.implementation_domain.strip():
            raise ValueError("paper implementation spec requires paper metadata")
        if not self.mechanism_summary.strip():
            raise ValueError("paper implementation spec requires mechanism_summary")
        if self.readiness == "implementation_ready":
            self._validate_implementation_ready()
        if self.readiness == "pilot_reproduced" and not self.pilot_reproduced:
            raise ValueError("pilot_reproduced readiness requires pilot_reproduced=true")
        if self.readiness == "full_reproduced" and not self.full_reproduced:
            raise ValueError("full_reproduced readiness requires full_reproduced=true")
        if self.readiness == "confirmed_multi_seed" and not self.confirmed_multi_seed:
            raise ValueError(
                "confirmed_multi_seed readiness requires confirmed_multi_seed=true"
            )
        return self

    def _validate_implementation_ready(self) -> None:
        if self.implementation_evidence_class in _DISALLOWED_READY_CLASSES:
            raise ValueError(
                "implementation_ready requires paper-specific non-mock evidence"
            )
        required = {
            "paper_specific_mechanisms": self.paper_specific_mechanisms,
            "component_ids": self.component_ids,
            "adapter_ids": self.adapter_ids,
            "runtime_insertion_points": self.runtime_insertion_points,
            "runtime_hooks": self.runtime_hooks,
            "paper_specific_config": self.paper_specific_config,
            "implementation_fingerprint": self.implementation_fingerprint,
            "unit_test_refs": self.unit_test_refs,
            "smoke_test_refs": self.smoke_test_refs,
            "compatibility_test_refs": self.compatibility_test_refs,
        }
        missing = sorted(name for name, value in required.items() if not value)
        if missing:
            raise ValueError(
                "implementation_ready spec is incomplete: " + ", ".join(missing)
            )
        if not self.runtime_implementation_verified:
            raise ValueError("implementation_ready requires runtime implementation verification")
        if not self.unit_tests_passed:
            raise ValueError("implementation_ready requires passed unit-test evidence")
        if not self.non_mock_smoke_passed:
            raise ValueError("implementation_ready requires non-mock smoke evidence")
        if not self.compatibility_validation_passed:
            raise ValueError("implementation_ready requires compatibility validation")
        critical = [
            blocker
            for blocker in self.blockers
            if any(marker in blocker for marker in _CRITICAL_BLOCKER_MARKERS)
        ]
        if critical:
            raise ValueError(
                "implementation_ready cannot retain critical blockers: "
                + ", ".join(critical)
            )

    @property
    def is_implementation_ready(self) -> bool:
        """Return the strict paper-level authorization boundary."""
        return self.readiness == "implementation_ready"

    @property
    def is_not_audited(self) -> bool:
        """Whether the paper lacks a complete paper-specific audit."""
        return self.readiness in {"cataloged", "profiled", "spec_complete"}


class PaperImplementationRegistry(BaseModel, YAMLModelMixin):
    """Complete paper-level implementation report for the frozen campaign."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_implementation_registry.v1"
    manifest_path: str
    manifest_membership_hash: str = Field(pattern=SHA256_PATTERN)
    paper_count: int = Field(ge=0)
    source_inventory_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    source_method_coverage_hash: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
    )
    records: list[PaperImplementationSpec] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    registry_hash: str = ""

    @model_validator(mode="after")
    def validate_registry(self) -> "PaperImplementationRegistry":
        ids = [item.paper_id for item in self.records]
        if ids != sorted(set(ids)):
            raise ValueError("paper implementation records must be sorted and unique")
        if self.paper_count != len(ids):
            raise ValueError("paper_count must equal implementation record count")
        if any(
            item.manifest_membership_hash != self.manifest_membership_hash
            for item in self.records
        ):
            raise ValueError("all paper specs must use the registry membership hash")
        if self.registry_hash and self.registry_hash != self.calculate_hash():
            raise ValueError("paper implementation registry hash mismatch")
        return self

    @property
    def implementation_ready_count(self) -> int:
        return sum(item.is_implementation_ready for item in self.records)

    @property
    def blocked_count(self) -> int:
        return sum(bool(item.blockers) for item in self.records)

    @property
    def not_audited_count(self) -> int:
        return sum(item.is_not_audited for item in self.records)

    @property
    def readiness_counts(self) -> dict[str, int]:
        names = (
            "cataloged",
            "profiled",
            "spec_complete",
            "code_bound",
            "runtime_integrated",
            "unit_tested",
            "smoke_passed",
            "implementation_ready",
            "pilot_reproduced",
            "full_reproduced",
            "confirmed_multi_seed",
        )
        return {name: sum(item.readiness == name for item in self.records) for name in names}

    def calculate_hash(self) -> str:
        payload = self.model_dump(
            mode="json",
            exclude={"registry_hash", "generated_at"},
        )
        for record in payload["records"]:
            record.pop("generated_at", None)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def with_hash(self) -> "PaperImplementationRegistry":
        return self.model_copy(update={"registry_hash": self.calculate_hash()})


__all__ = [
    "ImplementationEvidenceClass",
    "PaperImplementationReadiness",
    "PaperImplementationRegistry",
    "PaperImplementationSpec",
]
