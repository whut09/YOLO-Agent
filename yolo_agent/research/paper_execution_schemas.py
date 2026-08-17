"""Schemas for the paper-level execution inventory.

Paper metadata and runtime authorization are deliberately kept separate.  An
inventory row describes what must happen to one compatible paper; it is not a
training result and it does not grant ASHA eligibility by itself.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin


PaperExecutionDisposition = Literal[
    "queued",
    "runtime_ready",
    "already_tested",
    "evidence_recovery",
    "implementation_request",
    "incompatible",
    "blocked_runtime",
    "deferred_budget",
]


class PaperExecutionSpec(BaseModel, YAMLModelMixin):
    """One auditable execution route for one compatible paper."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_execution_spec.v1"
    paper_id: str
    profile_id: str
    title: str
    source_locations: list[str] = Field(min_length=1)
    original_method_name: str = "unknown"
    original_method_family: str = "unknown"
    canonical_component_ids: list[str] = Field(default_factory=list)
    paper_specific_mechanism_ids: list[str] = Field(default_factory=list)
    generic_component_ids: list[str] = Field(default_factory=list)
    adaptation_mode: str = "component_adaptation"
    exact_reproduction_possible: bool = False
    required_dataset_protocol: dict[str, Any] = Field(default_factory=dict)
    required_checkpoints: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    recipe_ids: list[str] = Field(default_factory=list)
    execution_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    current_disposition: PaperExecutionDisposition = "implementation_request"
    disposition_reason: str
    reusable_adapter_ids: list[str] = Field(default_factory=list)
    runtime_ready_adapters: list[str] = Field(default_factory=list)
    matched_error_fact_ids: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @model_validator(mode="after")
    def validate_execution_boundary(self) -> "PaperExecutionSpec":
        if not self.paper_id.strip():
            raise ValueError("paper execution spec requires paper_id")
        if not self.profile_id.strip():
            raise ValueError("paper execution spec requires profile_id")
        if not self.title.strip():
            raise ValueError("paper execution spec requires title")
        if not self.disposition_reason.strip():
            raise ValueError("paper execution spec requires disposition_reason")
        canonical = set(self.canonical_component_ids)
        specific = set(self.paper_specific_mechanism_ids)
        generic = set(self.generic_component_ids)
        if not specific.issubset(canonical):
            raise ValueError(
                "paper-specific mechanisms must be canonical component IDs"
            )
        if not generic.issubset(canonical):
            raise ValueError("generic mechanisms must be canonical component IDs")
        if specific & generic:
            raise ValueError(
                "paper-specific and generic mechanisms must be disjoint"
            )
        if self.generic_component_ids and not self.paper_specific_mechanism_ids:
            if self.current_disposition not in {
                "implementation_request",
                "evidence_recovery",
            }:
                raise ValueError(
                    "generic mechanisms without paper-specific IDs cannot be executable"
                )
        if self.current_disposition == "runtime_ready":
            if not self.paper_specific_mechanism_ids:
                raise ValueError("runtime-ready paper requires a paper-specific mechanism")
            if self.generic_component_ids:
                raise ValueError("runtime-ready paper cannot retain unresolved generic mechanisms")
            if not self.runtime_ready_adapters:
                raise ValueError("runtime-ready paper requires runtime-ready adapters")
        return self


class PaperExecutionInventory(BaseModel, YAMLModelMixin):
    """Complete paper-level inventory for the YOLO26-compatible denominator."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_execution_inventory.v1"
    source_method_coverage_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_maturity_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    all_paper_count: int = Field(ge=0)
    compatible_paper_count: int = Field(ge=0)
    exact_reproduction_candidates: int = Field(ge=0)
    generic_mechanism_counts: dict[str, int] = Field(default_factory=dict)
    records: list[PaperExecutionSpec] = Field(default_factory=list)
    inventory_hash: str = ""
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @model_validator(mode="after")
    def validate_inventory(self) -> "PaperExecutionInventory":
        paper_ids = [item.paper_id for item in self.records]
        profile_ids = [item.profile_id for item in self.records]
        if len(paper_ids) != len(set(paper_ids)):
            duplicates = sorted(
                paper_id
                for paper_id in set(paper_ids)
                if paper_ids.count(paper_id) > 1
            )
            raise ValueError("paper execution inventory has duplicate paper IDs: " + ", ".join(duplicates))
        if len(profile_ids) != len(set(profile_ids)):
            raise ValueError("paper execution inventory has duplicate profile IDs")
        if paper_ids != sorted(paper_ids):
            raise ValueError("paper execution inventory records must be sorted by paper_id")
        if self.compatible_paper_count != len(self.records):
            raise ValueError(
                "compatible_paper_count must equal the number of inventory records"
            )
        if self.all_paper_count < self.compatible_paper_count:
            raise ValueError("all_paper_count cannot be smaller than compatible_paper_count")
        if self.exact_reproduction_candidates != sum(
            item.exact_reproduction_possible for item in self.records
        ):
            raise ValueError(
                "exact_reproduction_candidates does not match inventory records"
            )
        expected_generic: dict[str, int] = {}
        for record in self.records:
            for component_id in record.generic_component_ids:
                expected_generic[component_id] = expected_generic.get(component_id, 0) + 1
        if dict(sorted(self.generic_mechanism_counts.items())) != dict(sorted(expected_generic.items())):
            raise ValueError("generic_mechanism_counts does not match inventory records")
        if self.inventory_hash and self.inventory_hash != self.calculate_hash():
            raise ValueError("paper execution inventory hash mismatch")
        return self

    def calculate_hash(self) -> str:
        payload = self.model_dump(
            mode="json",
            exclude={"inventory_hash", "generated_at"},
        )
        for record in payload["records"]:
            record.pop("generated_at", None)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def with_hash(self) -> "PaperExecutionInventory":
        return self.model_copy(update={"inventory_hash": self.calculate_hash()})


__all__ = [
    "PaperExecutionDisposition",
    "PaperExecutionInventory",
    "PaperExecutionSpec",
]
