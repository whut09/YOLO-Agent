"""Matched paired-delta attribution for coupled recipe ablations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


CoupledArm = Literal["A", "B", "A+B"]
CoupledEffectKind = Literal["single", "combined_total", "interaction"]
ContributionConfidence = Literal["possible", "confirmed"]


class CoupledArmObservation(BaseModel):
    """One current-node paired delta against an exact matched control."""

    model_config = ConfigDict(extra="forbid")

    arm: CoupledArm
    node_id: str
    matched_control_node_id: str
    seed: int
    protocol_hash: str
    metric_deltas: dict[str, float] = Field(min_length=1)
    paired_result_verified: bool = False
    evidence_role: str = "current_observation"
    inheritance_depth: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _identity_complete(self) -> "CoupledArmObservation":
        if not self.node_id.strip() or not self.matched_control_node_id.strip():
            raise ValueError("coupled observation requires node and matched control IDs")
        if not self.protocol_hash.strip():
            raise ValueError("coupled observation requires protocol_hash")
        return self


class CoupledContributionEffect(BaseModel):
    effect_id: str
    effect_kind: CoupledEffectKind
    component_ids: list[str]
    metric_name: str
    seed_count: int = Field(ge=1)
    mean_delta: float
    confidence_interval_low: float | None = None
    confidence_interval_high: float | None = None
    confidence: ContributionConfidence
    direction: Literal["positive", "negative", "neutral", "uncertain"]
    reason: str


class CoupledContributionReport(BaseModel):
    schema_version: str = "coupled_contribution_report.v1"
    recipe_id: str
    component_a: str
    component_b: str
    effects: list[CoupledContributionEffect] = Field(default_factory=list)
    complete_seeds: list[int] = Field(default_factory=list)
    incomplete_seeds: dict[int, list[str]] = Field(default_factory=dict)
    rejected_observations: dict[str, str] = Field(default_factory=dict)


__all__ = [
    "ContributionConfidence",
    "CoupledArm",
    "CoupledArmObservation",
    "CoupledContributionEffect",
    "CoupledContributionReport",
    "CoupledEffectKind",
]
