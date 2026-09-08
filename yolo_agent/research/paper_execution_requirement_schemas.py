"""Schemas for the per-paper execution requirements matrix.

This matrix is a planning artifact.  It describes what a paper would need to
run and never grants training or ASHA eligibility by itself.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.research.paper_execution_schemas import PaperExecutionDisposition


ExecutionRoute = Literal[
    "training",
    "inference",
    "blocked_runtime",
    "evidence_recovery",
    "implementation_request",
]


class AssetRequirementSource(BaseModel):
    """Why one declared asset belongs to a paper execution route."""

    model_config = ConfigDict(extra="forbid")

    source_mechanism_id: str
    source_reason: str

    @model_validator(mode="after")
    def validate_source(self) -> "AssetRequirementSource":
        if not self.source_mechanism_id.strip():
            raise ValueError("asset requirement source needs a mechanism ID")
        if not self.source_reason.strip():
            raise ValueError("asset requirement source needs a reason")
        return self


class PaperExecutionRequirement(BaseModel, YAMLModelMixin):
    """Requirements and authorization boundary for exactly one paper."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_execution_requirement.v1"
    paper_id: str
    paper_specific_mechanism: str
    paper_specific_mechanism_ids: list[str] = Field(default_factory=list)
    execution_route: ExecutionRoute
    required_adapter: str | None = None
    required_changed_variables: list[str] = Field(default_factory=list)
    required_runtime_payload: dict[str, Any] = Field(default_factory=dict)
    required_evidence: list[str] = Field(default_factory=list)
    required_dataset_protocol: dict[str, Any] = Field(default_factory=dict)
    required_teacher_assets: list[str] = Field(default_factory=list)
    required_domain_assets: list[str] = Field(default_factory=list)
    required_manifest_assets: list[str] = Field(default_factory=list)
    required_graph_assets: list[str] = Field(default_factory=list)
    required_control_assets: list[str] = Field(default_factory=list)
    asset_requirement_sources: dict[str, AssetRequirementSource] = Field(
        default_factory=dict
    )
    # These row-level fields make the primary provenance visible in compact
    # YAML/CLI views; asset_requirement_sources retains per-asset detail.
    source_mechanism_id: str = ""
    source_reason: str = ""
    compatible_with_yolo26: bool
    training_candidate_allowed: bool = False
    exact_blocker: str | None = None
    recovery_action: str
    recipe_ids: list[str] = Field(default_factory=list)
    current_disposition: PaperExecutionDisposition
    protocol_hash: str = ""
    execution_fingerprint: str

    @model_validator(mode="after")
    def validate_boundary(self) -> "PaperExecutionRequirement":
        if not self.paper_id.strip():
            raise ValueError("paper requirement requires paper_id")
        if not self.paper_specific_mechanism.strip():
            raise ValueError("paper requirement requires paper-specific mechanism")
        if self.paper_specific_mechanism in {
            "distillation.yolo26_teacher_student",
            "domain_adaptation.general",
            "quality_alignment.general",
        }:
            raise ValueError("generic mechanism cannot be a paper requirement")
        if self.paper_specific_mechanism_ids and self.paper_specific_mechanism not in self.paper_specific_mechanism_ids:
            raise ValueError("primary mechanism must be included in mechanism IDs")
        if not self.recovery_action.strip():
            raise ValueError("paper requirement requires a recovery action")
        if self.training_candidate_allowed:
            if self.execution_route != "training":
                raise ValueError("training candidate must use training route")
            if self.exact_blocker:
                raise ValueError("training candidate cannot retain an exact blocker")
            if not self.required_adapter:
                raise ValueError("training candidate requires an adapter")
        if self.execution_route == "inference" and self.training_candidate_allowed:
            raise ValueError("inference route cannot be a training candidate")
        if not self.compatible_with_yolo26 and self.training_candidate_allowed:
            raise ValueError("incompatible paper cannot be a training candidate")
        if not self.exact_blocker and not self.training_candidate_allowed:
            raise ValueError("non-training paper requires an exact blocker")
        declared_assets = {
            *self.required_teacher_assets,
            *self.required_domain_assets,
            *self.required_manifest_assets,
            *self.required_graph_assets,
            *self.required_control_assets,
        }
        unknown_sources = sorted(
            set(self.asset_requirement_sources).difference(declared_assets)
        )
        if unknown_sources:
            raise ValueError(
                "asset requirement sources refer to undeclared assets: "
                + ", ".join(unknown_sources)
            )
        if self.asset_requirement_sources and set(self.asset_requirement_sources) != declared_assets:
            missing_sources = sorted(
                declared_assets.difference(self.asset_requirement_sources)
            )
            raise ValueError(
                "declared asset requirements need sources: "
                + ", ".join(missing_sources)
            )
        if bool(self.source_mechanism_id) != bool(self.source_reason):
            raise ValueError(
                "source_mechanism_id and source_reason must be provided together"
            )
        return self


class PaperExecutionRequirementsMatrix(BaseModel, YAMLModelMixin):
    """Complete requirements matrix with an immutable inventory denominator."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_execution_requirements.v1"
    source_inventory_path: str
    source_inventory_hash: str
    compatible_paper_count: int = Field(ge=0)
    requirements: list[PaperExecutionRequirement] = Field(default_factory=list)
    generated_at: str

    @model_validator(mode="after")
    def validate_complete(self) -> "PaperExecutionRequirementsMatrix":
        ids = [item.paper_id for item in self.requirements]
        if len(ids) != len(set(ids)):
            raise ValueError("requirements matrix contains duplicate paper IDs")
        if ids != sorted(ids):
            raise ValueError("requirements matrix records must be sorted by paper_id")
        if self.compatible_paper_count != len(ids):
            raise ValueError("requirements matrix must contain every compatible paper")
        return self


__all__ = [
    "AssetRequirementSource",
    "ExecutionRoute",
    "PaperExecutionRequirement",
    "PaperExecutionRequirementsMatrix",
]
