"""Semantic adaptation of training-control parameters to the runtime.

The frozen-campaign rule enforced here: a paper naming a training knob plus a
runtime parameter with a similar name is *not* an implementation.  Every
adaptation declares the paper's parameter semantics, the YOLO-Agent runtime
contract that actually realizes them, the Ultralytics/YOLO26 hook the contract
binds to, what is preserved, and what is approximated — and the record fails
validation if the runtime contract does not have a real binding point in this
repository's code.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin

AdaptationClass = Literal["faithful_adaptation", "exact_reproduction"]

#: Runtime hooks in this repository that training controls can bind to.
#: A paper adaptation may only reference hooks present in this mapping; a
#: claimed binding to any other hook fails validation (fail-closed against
#: name-similarity impostors).
RUNTIME_BINDING_POINTS: dict[str, str] = {
    "build_model": (
        "DomainAdaptationBranchPlugin.build_model — attach route modules "
        "before the optimizer is created"
    ),
    "compute_loss": (
        "DomainAdaptationBranchPlugin.compute_loss — YOLO26 (loss, "
        "loss_items) hook contract; schedule-controlled weight applied here"
    ),
    "on_train_batch_start": (
        "trainer callback — the schedule controller resolves the batch index "
        "and announces the effective auxiliary weight"
    ),
    "on_train_batch_end": (
        "trainer callback — records the weighted loss term as evidence"
    ),
    "train_overrides": (
        "UltralyticsTrainingConfig.train_overrides — explicit optimizer "
        "overrides only; optimizer=auto ignores lr/momentum knobs and the "
        "candidate builder rejects those combinations"
    ),
}


class TrainingParameterAdaptation(BaseModel, YAMLModelMixin):
    """One paper parameter's path from paper semantics to a runtime hook."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    parameter_name: str
    paper_semantics: str = Field(min_length=1)
    runtime_contract: str = Field(min_length=1)
    runtime_binding_point: str
    ultralytics_hook: str = Field(min_length=1)
    adaptation_class: AdaptationClass = "faithful_adaptation"
    preserved_information: list[str] = Field(min_length=1)
    approximated_information: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_adaptation(self) -> "TrainingParameterAdaptation":
        if not self.paper_id.strip() or not self.parameter_name.strip():
            raise ValueError("adaptation record requires paper and parameter identity")
        if self.runtime_binding_point not in RUNTIME_BINDING_POINTS:
            known = ", ".join(sorted(RUNTIME_BINDING_POINTS))
            raise ValueError(
                f"runtime binding point {self.runtime_binding_point!r} does not exist in "
                f"this repository; known bindings: {known}"
            )
        if self.adaptation_class == "exact_reproduction" and self.approximated_information:
            raise ValueError(
                "exact_reproduction cannot approximate information; mark it "
                "faithful_adaptation"
            )
        return self


class TrainingControlPaperRecord(BaseModel, YAMLModelMixin):
    """Per-paper training-control status that never collapses into primitives."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "training_control_paper_record.v1"
    paper_id: str
    title: str
    year: int = Field(ge=1900, le=2100)
    primary_domain: str
    in_scope: bool = Field(
        description=(
            "True only when frozen plan evidence declares a training-control "
            "insertion point (training_schedule) or an optimizer/regularization "
            "contract for this paper."
        )
    )
    scope_reason: str = Field(min_length=1)
    schedule_id: str | None = None
    adaptations: list[TrainingParameterAdaptation] = Field(default_factory=list)
    protocol_spec_ids: list[str] = Field(default_factory=list)
    behavior_passed: bool = False
    behavior_evidence: list[str] = Field(default_factory=list)
    remaining_dependencies: list[str] = Field(default_factory=list)
    status: Literal["ready", "blocked_missing_evidence", "out_of_scope"]
    blockers: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_record(self) -> "TrainingControlPaperRecord":
        if not self.paper_id.strip() or not self.scope_reason.strip():
            raise ValueError("training-control record requires identity and scope reason")
        if self.status == "ready" and self.blockers:
            raise ValueError("ready record must not carry blockers")
        if self.status == "blocked_missing_evidence" and not self.blockers:
            raise ValueError("blocked record must list at least one blocker")
        if self.status == "ready" and not self.in_scope:
            raise ValueError("out-of-scope paper cannot be training-control ready")
        if self.status == "ready" and not self.behavior_passed:
            raise ValueError("ready record requires real behavior evidence")
        if self.status == "ready" and not self.adaptations:
            raise ValueError("ready record requires at least one semantic adaptation")
        return self


__all__ = [
    "RUNTIME_BINDING_POINTS",
    "AdaptationClass",
    "TrainingControlPaperRecord",
    "TrainingParameterAdaptation",
]
