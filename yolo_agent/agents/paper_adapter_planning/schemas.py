"""Typed contracts for diagnosis-driven paper adapter implementation planning."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.research.component_aliases import ImplementationStatus, YOLO26Compatibility


CostLevel = Literal["low", "medium", "high", "unknown"]
ImplementationTrack = Literal[
    "ready_to_materialize",
    "implementation_queue",
    "shadow_evaluation_queue",
    "incompatible",
    "separate_detector_family",
    "insufficient_information",
    "deferred",
]


class AdapterImplementationEstimate(BaseModel):
    """Human-reviewed implementation and deployment cost prior."""

    model_config = ConfigDict(extra="forbid")

    component_id: str
    implementation_cost: CostLevel = "unknown"
    expected_latency_cost: CostLevel = "unknown"
    expected_model_size_cost: CostLevel = "unknown"
    required_runtime_hook: str | None = None
    requires_shadow_evaluation: bool = False
    notes: list[str] = Field(default_factory=list)


class RuntimeHookAvailability(BaseModel):
    """Audited local runtime hook, never inferred from a component name."""

    model_config = ConfigDict(extra="forbid")

    hook_id: str
    available: bool
    verified: bool = False
    version: str = "unknown"
    evidence: str | None = None


class PaperAdapterImplementationRequest(BaseModel):
    """A bounded engineering request; it contains no generated implementation."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    component_id: str
    paper_ids: list[str]
    insertion_point: str
    required_runtime_hook: str | None = None
    reason: str
    acceptance_tests: list[str] = Field(default_factory=list)
    generated_code_allowed: bool = False


class PaperAdapterQueueItem(BaseModel):
    """One canonical component ranked into an implementation track."""

    model_config = ConfigDict(extra="forbid")

    component_id: str
    component_family: str
    mechanism_cluster_id: str | None = None
    adapter_family: str | None = None
    canonical_component_ids: list[str] = Field(default_factory=list)
    covered_paper_count: int = Field(default=0, ge=0)
    paper_ids: list[str]
    paper_year: int
    official_code_available: bool
    source_license: str
    yolo26_compatibility: YOLO26Compatibility
    implementation_status: ImplementationStatus
    insertion_point: str
    diagnosis_targets: list[str] = Field(default_factory=list)
    score: float = 0.0
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    fingerprint: str
    track: ImplementationTrack
    implementation_request: PaperAdapterImplementationRequest | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImplementationHistoryRecord(BaseModel):
    """Minimal history used for implementation cooldown and deduplication."""

    model_config = ConfigDict(extra="forbid")

    fingerprint: str
    component_family: str
    round_index: int = Field(ge=0)
    outcome: Literal["queued", "completed", "failed", "deferred"] = "queued"


class PaperAdapterImplementationPlan(BaseModel):
    """Auditable queues produced without generating or executing adapter code."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_adapter_implementation_plan.v1"
    current_round: int = Field(default=0, ge=0)
    ready_to_materialize: list[PaperAdapterQueueItem] = Field(default_factory=list)
    implementation_queue: list[PaperAdapterQueueItem] = Field(default_factory=list)
    shadow_evaluation_queue: list[PaperAdapterQueueItem] = Field(default_factory=list)
    incompatible: list[PaperAdapterQueueItem] = Field(default_factory=list)
    separate_detector_family: list[PaperAdapterQueueItem] = Field(default_factory=list)
    insufficient_information: list[PaperAdapterQueueItem] = Field(default_factory=list)
    deferred: list[PaperAdapterQueueItem] = Field(default_factory=list)
    auto_code_generation: bool = False
    summary: dict[str, int] = Field(default_factory=dict)
    plan_hash: str = ""


__all__ = [
    "AdapterImplementationEstimate",
    "CostLevel",
    "ImplementationHistoryRecord",
    "ImplementationTrack",
    "PaperAdapterImplementationPlan",
    "PaperAdapterImplementationRequest",
    "PaperAdapterQueueItem",
    "RuntimeHookAvailability",
]
