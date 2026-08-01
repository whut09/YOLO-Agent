"""Schemas for auditable paper execution coverage denominators."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin


CompatibilityClass = Literal[
    "yolo26_runtime_ready",
    "yolo26_adapter_available",
    "yolo26_adapter_required",
    "yolo26_coupled_adaptation",
    "separate_detector_family",
    "incompatible",
    "insufficient_information",
]
AdaptationScope = Literal[
    "none",
    "single_component",
    "multiple_independent_components",
    "coupled_components",
    "whole_detector",
    "exact_reproduction",
]
CostLevel = Literal["low", "medium", "high", "unknown"]


class PaperImplementationCost(BaseModel):
    """Qualitative implementation cost derived from known runtime hooks."""

    model_config = ConfigDict(extra="forbid")

    level: CostLevel = "unknown"
    adapter_count: int = Field(default=0, ge=0)
    missing_adapter_count: int = Field(default=0, ge=0)
    hook_count: int = Field(default=0, ge=0)
    rationale: list[str] = Field(default_factory=list)


class PaperExpectedResourceCost(BaseModel):
    """Declared resource impact; unknown values stay unknown."""

    model_config = ConfigDict(extra="forbid")

    level: CostLevel = "unknown"
    latency: str = "unknown"
    model_size: str = "unknown"
    vram: str = "unknown"
    training_compute: str = "unknown"
    rationale: list[str] = Field(default_factory=list)


class PaperExecutableCoverageEntry(BaseModel):
    """One paper's complete implementation and execution audit."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    profile_id: str
    decision: str
    compatibility_class: CompatibilityClass
    adaptation_scope: AdaptationScope
    blocking_fields: list[str] = Field(default_factory=list)
    canonical_mechanisms: list[str] = Field(default_factory=list)
    reusable_adapter_candidates: list[str] = Field(default_factory=list)
    runtime_ready_adapters: list[str] = Field(default_factory=list)
    required_runtime_hooks: list[str] = Field(default_factory=list)
    implementation_cost: PaperImplementationCost = Field(
        default_factory=PaperImplementationCost
    )
    expected_resource_cost: PaperExpectedResourceCost = Field(
        default_factory=PaperExpectedResourceCost
    )
    exact_reproduction_possible: bool = False
    exclusion_reason: str | None = None
    source_locations: list[str] = Field(default_factory=list)


class PaperCoverageDenominator(BaseModel):
    """One named denominator with its auditable paper membership."""

    model_config = ConfigDict(extra="forbid")

    name: Literal[
        "all_papers",
        "yolo26_compatible_papers",
        "adaptable_component_papers",
        "exact_reproduction_candidates",
    ]
    definition: str
    paper_count: int = Field(default=0, ge=0)
    paper_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_membership(self) -> "PaperCoverageDenominator":
        if self.paper_ids != sorted(set(self.paper_ids)):
            raise ValueError(f"{self.name} paper_ids must be sorted and unique")
        if self.paper_count != len(self.paper_ids):
            raise ValueError(f"{self.name} paper_count does not match paper_ids")
        return self


class ExecutablePaperCoverageBaseline(BaseModel, YAMLModelMixin):
    """Four-denominator paper implementation coverage baseline."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "executable_paper_coverage.v1"
    source_method_coverage_hash: str
    source_maturity_hash: str | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    denominators: dict[str, PaperCoverageDenominator]
    compatibility_counts: dict[str, int] = Field(default_factory=dict)
    runtime_ready_paper_count: int = Field(default=0, ge=0)
    reusable_adapter_paper_count: int = Field(default=0, ge=0)
    entries: list[PaperExecutableCoverageEntry] = Field(default_factory=list)
    report_hash: str = ""

    @model_validator(mode="after")
    def validate_report(self) -> "ExecutablePaperCoverageBaseline":
        expected_names = {
            "all_papers",
            "yolo26_compatible_papers",
            "adaptable_component_papers",
            "exact_reproduction_candidates",
        }
        if set(self.denominators) != expected_names:
            raise ValueError("coverage baseline requires all four denominators")
        paper_ids = [item.paper_id for item in self.entries]
        if paper_ids != sorted(set(paper_ids)):
            raise ValueError("coverage entries must contain each paper exactly once")
        if self.denominators["all_papers"].paper_ids != paper_ids:
            raise ValueError("all_papers denominator must match coverage entries")
        expected = self.calculate_hash()
        if self.report_hash and self.report_hash != expected:
            raise ValueError("executable paper coverage report hash mismatch")
        self.report_hash = expected
        return self

    def calculate_hash(self) -> str:
        payload = self.model_dump(
            mode="json",
            exclude={"generated_at", "report_hash"},
        )
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


__all__ = [
    "AdaptationScope",
    "CompatibilityClass",
    "ExecutablePaperCoverageBaseline",
    "PaperCoverageDenominator",
    "PaperExecutableCoverageEntry",
    "PaperExpectedResourceCost",
    "PaperImplementationCost",
]
