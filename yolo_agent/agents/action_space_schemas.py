"""Unified detection optimization action space schemas.

The autonomous decision object of YOLO-Agent is the *whole* detection
system's controllable optimization action — not only paper recipes or model
components.  This module defines the frozen ``ActionFamily`` taxonomy and
the ``ActionSpec`` record that every candidate action (paper-derived or
local) must carry, plus the deterministic boundary types that keep LLM
output on the proposal side of the line.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin

ACTION_SPACE_SCHEMA_VERSION = "detection_action_space.v1"

# ---------------------------------------------------------------- taxonomy --

ActionFamily = Literal[
    # data side
    "annotation",
    "data_cleaning",
    "data_selection",
    "sampling",
    "augmentation",
    "preprocessing",
    # model capacity / input
    "model_scale",
    "input_resolution",
    # model graph
    "backbone",
    "neck",
    "feature_fusion",
    "head",
    # training objectives
    "assignment",
    "bbox_loss",
    "classification_loss",
    "auxiliary_loss",
    # teacher / distribution
    "distillation",
    "domain_adaptation",
    "semi_supervised",
    # training control
    "training_strategy",
    "optimizer",
    "regularization",
    # inference side
    "postprocess",
    "threshold",
    "inference",
    "calibration",
    # label acquisition
    "active_learning",
]

#: Families that never enter the training graph: they only change evaluation
#: or deployment behavior.  The boundary is enforced structurally (see
#: ``is_train_time_family``) instead of by prose.
INFERENCE_ONLY_FAMILIES: frozenset[str] = frozenset(
    {
        "postprocess",
        "threshold",
        "inference",
        "calibration",
    }
)

#: Families realized through the model-graph patch system.
MODEL_GRAPH_FAMILIES: frozenset[str] = frozenset(
    {
        "backbone",
        "neck",
        "feature_fusion",
        "head",
        "model_scale",
    }
)

#: Families that alter the data pipeline or its annotations.
DATA_FAMILIES: frozenset[str] = frozenset(
    {
        "annotation",
        "data_cleaning",
        "data_selection",
        "sampling",
        "augmentation",
        "preprocessing",
        "active_learning",
    }
)

TRAIN_GRAPH_FAMILIES: frozenset[str] = frozenset(set(ActionFamily.__args__) - INFERENCE_ONLY_FAMILIES)  # type: ignore[attr-defined]

#: Families whose mechanisms genuinely execute in either phase depending on
#: the concrete action (e.g. training imgsz vs inference imgsz).  Every other
#: train-graph family is strictly train/data-time.
DUAL_PHASE_FAMILIES: frozenset[str] = frozenset({"input_resolution"})


def is_train_time_family(family: str) -> bool:
    """Whether an action family participates in the training graph."""

    return family in TRAIN_GRAPH_FAMILIES


def is_inference_only_family(family: str) -> bool:
    """Whether an action family is inference/eval-side only."""

    return family in INFERENCE_ONLY_FAMILIES


RuntimePhase = Literal["data", "train", "inference", "eval"]

_COST = Literal["low", "medium", "high", "unknown"]


# --------------------------------------------------------------- action spec --


class ActionRollback(BaseModel):
    """How to undo one applied action."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal[
        "config_revert",
        "checkpoint_restore",
        "component_remove",
        "artifact_discard",
        "not_reversible",
    ]
    instructions: str
    values: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> "ActionRollback":
        if not self.instructions.strip():
            raise ValueError("rollback requires instructions")
        if self.strategy == "config_revert" and not self.values:
            raise ValueError("config_revert rollback requires values")
        return self


class ActionSpec(BaseModel, YAMLModelMixin):
    """One controllable optimization action over the detection system.

    Every action — paper-derived or local — carries the full record: which
    problem tags it targets, its paper/component lineage, its parameter
    surface, preconditions, cost profile, rollback, and the runtime phase it
    actually executes in.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = ACTION_SPACE_SCHEMA_VERSION
    action_id: str
    family: ActionFamily
    title: str
    description: str = ""

    problem_tags: list[str] = Field(default_factory=list)

    paper_ids: list[str] = Field(default_factory=list)
    component_ids: list[str] = Field(default_factory=list)
    paper_lineage: Literal["paper", "local", "shared_primitive_plus_paper"] = "local"

    parameters: dict[str, Any] = Field(default_factory=dict)
    search_space: dict[str, list[Any]] = Field(default_factory=dict)

    preconditions: list[str] = Field(default_factory=list)
    compatibility_constraints: list[str] = Field(default_factory=list)

    expected_metrics: list[str] = Field(min_length=1)
    possible_side_effects: list[str] = Field(default_factory=list)

    implementation_cost: _COST = "unknown"
    training_cost: _COST = "unknown"
    inference_cost: _COST = "unknown"

    rollback: ActionRollback
    required_evidence: list[str] = Field(min_length=1)

    runtime_phase: RuntimePhase

    @model_validator(mode="after")
    def validate_spec(self) -> "ActionSpec":
        if not self.action_id.strip():
            raise ValueError("action_id must not be empty")
        if self.paper_ids and self.paper_lineage == "local":
            raise ValueError(
                f"action {self.action_id} lists paper_ids but declares local lineage"
            )
        if self.paper_lineage == "paper" and not self.paper_ids:
            raise ValueError(
                f"action {self.action_id} declares paper lineage without paper_ids"
            )
        if self.family in INFERENCE_ONLY_FAMILIES and self.runtime_phase not in {
            "inference",
            "eval",
        }:
            raise ValueError(
                f"inference-only action {self.action_id} ({self.family}) must run in "
                f"inference/eval phase, got {self.runtime_phase}"
            )
        if (
            self.family not in INFERENCE_ONLY_FAMILIES
            and self.family not in DUAL_PHASE_FAMILIES
            and self.runtime_phase == "inference"
        ):
            raise ValueError(
                f"train-graph action {self.action_id} ({self.family}) cannot declare "
                "runtime_phase=inference; train-side actions run in data/train phase"
            )
        return self

    @property
    def train_time(self) -> bool:
        return is_train_time_family(self.family)

    @property
    def inference_only(self) -> bool:
        return is_inference_only_family(self.family)

    @property
    def is_paper_action(self) -> bool:
        return self.paper_lineage in {"paper", "shared_primitive_plus_paper"}


# ------------------------------------------------------- diagnosis chain ------


class RootCauseHypothesis(BaseModel, YAMLModelMixin):
    """One candidate root cause behind an observed detection error.

    A hypothesis is a *proposal*: it names the evidence that would confirm
    or refute it and the action families that could address it.  It never
    claims certainty and never authorizes execution by itself.
    """

    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    description: str
    problem_tags: list[str] = Field(default_factory=list)
    confirming_evidence: list[str] = Field(default_factory=list)
    refuting_evidence: list[str] = Field(default_factory=list)
    candidate_action_families: list[ActionFamily] = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    origin: Literal["deterministic_rule", "llm_proposal", "human"] = "deterministic_rule"

    @model_validator(mode="after")
    def validate_hypothesis(self) -> "RootCauseHypothesis":
        if not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id must not be empty")
        if not self.description.strip():
            raise ValueError("hypothesis requires a description")
        return self


class ActionFamilySelection(BaseModel, YAMLModelMixin):
    """Action families selected for one hypothesis, with the concrete specs."""

    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    families: list[ActionFamily] = Field(min_length=1)
    action_specs: list[ActionSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_selection(self) -> "ActionFamilySelection":
        unknown = sorted(set(self.families) - set(ActionFamily.__args__))  # type: ignore[attr-defined]
        if unknown:
            raise ValueError(f"unknown action families: {', '.join(unknown)}")
        for spec in self.action_specs:
            if spec.family not in self.families:
                raise ValueError(
                    f"action {spec.action_id} ({spec.family}) is not among the "
                    "selected families for hypothesis " + self.hypothesis_id
                )
        return self


class DiagnosisActionChain(BaseModel, YAMLModelMixin):
    """The full deterministic chain: ErrorFact → hypotheses → families → specs."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = ACTION_SPACE_SCHEMA_VERSION
    error_fact_refs: list[str] = Field(default_factory=list)
    problem_tags: list[str] = Field(default_factory=list)
    hypotheses: list[RootCauseHypothesis] = Field(min_length=1)
    selections: list[ActionFamilySelection] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_chain(self) -> "DiagnosisActionChain":
        hypothesis_ids = [item.hypothesis_id for item in self.hypotheses]
        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise ValueError("chain hypotheses must be unique")
        selection_ids = [item.hypothesis_id for item in self.selections]
        if len(selection_ids) != len(set(selection_ids)):
            raise ValueError("chain selections must reference unique hypotheses")
        for selection in self.selections:
            if selection.hypothesis_id not in hypothesis_ids:
                raise ValueError(
                    f"selection references unknown hypothesis {selection.hypothesis_id}"
                )
        return self


# --------------------------------------------------- deterministic boundary ---


#: What an LLM may do with this action space.  Anything outside this set is
#: structurally rejected by :class:`LLMActionBoundary`.
LLM_ALLOWED_OPERATIONS: frozenset[str] = frozenset(
    {
        "generate_hypothesis",
        "explain_evidence",
        "rank_candidates",
        "propose_action_spec",
    }
)

#: Operations that are *never* delegable to an LLM proposal.
LLM_FORBIDDEN_OPERATIONS: frozenset[str] = frozenset(
    {
        "mark_implementation_ready",
        "skip_compatibility",
        "skip_readiness",
        "request_gpu_allocation",
        "promote_candidate",
        "modify_frozen_membership",
    }
)


class LLMActionBoundary(BaseModel):
    """Structural boundary between LLM proposals and frozen guarantees."""

    model_config = ConfigDict(extra="forbid")

    allowed_operations: list[str] = Field(default_factory=list)
    rejected_operations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_boundary(self) -> "LLMActionBoundary":
        for operation in self.allowed_operations:
            if operation in LLM_FORBIDDEN_OPERATIONS:
                raise ValueError(
                    f"LLM boundary cannot allow forbidden operation: {operation}"
                )
        for operation in self.rejected_operations:
            if operation not in LLM_FORBIDDEN_OPERATIONS:
                raise ValueError(
                    f"non-forbidden operation recorded as rejected: {operation}"
                )
        return self

    def propose(self, operation: str) -> bool:
        """Whether an LLM proposal of this operation is structurally possible."""

        if operation in LLM_FORBIDDEN_OPERATIONS:
            return False
        return operation in LLM_ALLOWED_OPERATIONS


__all__ = [
    "ACTION_SPACE_SCHEMA_VERSION",
    "ActionFamily",
    "ActionFamilySelection",
    "ActionRollback",
    "ActionSpec",
    "DATA_FAMILIES",
    "DiagnosisActionChain",
    "DUAL_PHASE_FAMILIES",
    "INFERENCE_ONLY_FAMILIES",
    "LLM_ALLOWED_OPERATIONS",
    "LLM_FORBIDDEN_OPERATIONS",
    "LLMActionBoundary",
    "MODEL_GRAPH_FAMILIES",
    "RootCauseHypothesis",
    "RuntimePhase",
    "TRAIN_GRAPH_FAMILIES",
    "is_inference_only_family",
    "is_train_time_family",
]
