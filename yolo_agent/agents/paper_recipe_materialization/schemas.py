"""Typed outcomes for certified paper recipe materialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.components.compatibility import CompatibilityResult
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.recipes.paper_priors import RecipePrior
from yolo_agent.research.method_profiles import (
    PaperImplementationDecision,
    PaperMethodProfile,
)


GateAction = Literal[
    "evidence_recovery",
    "implementation_required",
    "registered_with_asha",
    "queue_assignment",
    "awaiting_cohort",
    "exhausted",
    "blocked",
]
CandidateGateAction = Literal[
    "implementation_request",
    "rejected",
    "registered_with_asha",
    "deferred",
]


class PaperRecipeCandidateInput(BaseModel):
    """Paper prior plus local execution inputs; still has no queue authority."""

    model_config = ConfigDict(extra="forbid")

    prior: RecipePrior
    method_profile: PaperMethodProfile
    implementation_decision: PaperImplementationDecision
    compatibility: CompatibilityResult
    source_node: ExperimentNode | None = None
    matched_control_node: ExperimentNode | None = None
    component_family: str
    bucket: Literal["exploration", "exploitation"] = "exploitation"


class PaperRecipeEvidenceRecovery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: Literal["evidence_recovery"] = "evidence_recovery"
    required_evidence: list[str]
    reason: str
    training_allowed: Literal[False] = False


class PaperRecipeImplementationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: Literal["implementation_request"] = "implementation_request"
    prior_id: str
    component_ids: list[str]
    required_adapters: list[str] = Field(default_factory=list)
    reason: str
    generated_code_allowed: Literal[False] = False


class MaterializedAdapterIdentity(BaseModel):
    """Runtime identity shown to users and persisted in the ledger."""

    model_config = ConfigDict(extra="forbid")

    component_ids: list[str]
    adapter_classes: dict[str, str]
    adapter_versions: dict[str, str]
    adapter_hashes: dict[str, str]
    component_maturity: dict[str, str]
    maturity_artifact_hashes: dict[str, list[str]]
    adapter_patch_hashes: dict[str, str]
    aggregate_patch_hash: str
    runtime_payload_hash: str
    runtime_payload_path: Path
    protocol_hash: str
    runtime_execution_ready: Literal[True] = True
    smoke_passed: Literal[True] = True

    @property
    def terminal_summary(self) -> str:
        adapters = ", ".join(
            f"{component}={self.adapter_classes[component]}@{self.adapter_versions[component]}"
            for component in self.component_ids
        )
        return (
            f"adapters={adapters}; hashes="
            f"{','.join(value[:12] for value in self.adapter_hashes.values())}; "
            f"patch={self.aggregate_patch_hash[:12]}; "
            f"runtime={self.runtime_payload_hash[:12]}"
        )


class PaperRecipeCandidateGateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prior_id: str
    action: CandidateGateAction
    candidate_id: str | None = None
    recipe_id: str | None = None
    reasons: list[str] = Field(default_factory=list)
    implementation_request: PaperRecipeImplementationRequest | None = None
    runtime_identity: MaterializedAdapterIdentity | None = None
    eligibility_token: str | None = None


class PaperRecipeMaterializationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_recipe_materialization_gate.v1"
    run_id: str
    action: GateAction
    candidates: list[PaperRecipeCandidateGateResult] = Field(default_factory=list)
    evidence_recovery: PaperRecipeEvidenceRecovery | None = None
    registration: dict[str, Any] = Field(default_factory=dict)
    round_execution_plan: dict[str, Any] | None = None
    execution_queue: dict[str, Any] | None = None
    asha_assignment_id: str | None = None
    scalar_hpo_enabled: Literal[False] = False
    stopped_reason: str | None = None
    terminal_lines: list[str] = Field(default_factory=list)
    ledger_path: Path | None = None


__all__ = [
    "CandidateGateAction",
    "GateAction",
    "MaterializedAdapterIdentity",
    "PaperRecipeCandidateInput",
    "PaperRecipeCandidateGateResult",
    "PaperRecipeEvidenceRecovery",
    "PaperRecipeImplementationRequest",
    "PaperRecipeMaterializationResult",
]
