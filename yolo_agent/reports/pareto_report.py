"""Strict local-evidence Pareto analysis for paper recipe outcomes."""
from __future__ import annotations
from pydantic import BaseModel, Field, model_validator

class PaperParetoCandidate(BaseModel):
    candidate_id: str
    recipe_id: str
    map50_95: float | None = None
    ap_small: float | None = None
    recall: float | None = None
    latency_ms: float | None = None
    model_size_mb: float | None = None
    verified_local: bool = False
    evidence_role: str = "current_observation"
    inheritance_depth: int = 0
    inference_policy_changed: bool = False
    slicing_metrics: dict[str, float] = Field(default_factory=dict)
    evidence_status: str = "possible"

    @model_validator(mode="after")
    def mark_slicing(self) -> "PaperParetoCandidate":
        if self.slicing_metrics:
            self.inference_policy_changed = True
        return self

class PaperParetoPoint(BaseModel):
    candidate_id: str
    recipe_id: str
    map50_95: float | None = None
    ap_small: float | None = None
    recall: float | None = None
    latency_ms: float | None = None
    model_size_mb: float | None = None
    evidence_status: str
    tradeoff: str

class PaperParetoReport(BaseModel):
    schema_version: str = "paper_pareto_report.v1"
    included: list[PaperParetoPoint] = Field(default_factory=list)
    dominated: list[str] = Field(default_factory=list)
    excluded: dict[str, str] = Field(default_factory=dict)

def build_paper_pareto_report(candidates: list[PaperParetoCandidate]) -> PaperParetoReport:
    eligible, excluded = [], {}
    for candidate in candidates:
        if not candidate.verified_local:
            excluded[candidate.candidate_id] = "local_verified_evidence_required"
        elif candidate.evidence_role != "current_observation" or candidate.inheritance_depth != 0:
            excluded[candidate.candidate_id] = "current_node_only_evidence_required"
        elif candidate.inference_policy_changed:
            excluded[candidate.candidate_id] = "inference_policy_changed_not_training_pareto"
        elif all(getattr(candidate, name) is None for name in ("map50_95", "ap_small", "recall", "latency_ms", "model_size_mb")):
            excluded[candidate.candidate_id] = "no_local_pareto_metrics"
        else:
            eligible.append(candidate)
    front, dominated = [], []
    for candidate in eligible:
        if any(_dominates(other, candidate) for other in eligible if other.candidate_id != candidate.candidate_id):
            dominated.append(candidate.candidate_id)
        else:
            front.append(candidate)
    return PaperParetoReport(included=[_point(item) for item in sorted(front, key=lambda x: (-(x.map50_95 or 0.0), x.latency_ms or float("inf")))], dominated=sorted(dominated), excluded=excluded)

def _dominates(left: PaperParetoCandidate, right: PaperParetoCandidate) -> bool:
    better = False
    for name, direction in (("map50_95", "max"), ("ap_small", "max"), ("recall", "max"), ("latency_ms", "min"), ("model_size_mb", "min")):
        a, b = getattr(left, name), getattr(right, name)
        if a is None or b is None:
            continue
        if (direction == "max" and a < b) or (direction == "min" and a > b):
            return False
        better = better or a != b
    return better

def _point(item: PaperParetoCandidate) -> PaperParetoPoint:
    return PaperParetoPoint(candidate_id=item.candidate_id, recipe_id=item.recipe_id, map50_95=item.map50_95, ap_small=item.ap_small, recall=item.recall, latency_ms=item.latency_ms, model_size_mb=item.model_size_mb, evidence_status=item.evidence_status, tradeoff=f"mAP50-95={item.map50_95}; AP_small={item.ap_small}; recall={item.recall}; latency_ms={item.latency_ms}; model_size_mb={item.model_size_mb}")

__all__ = ["PaperParetoCandidate", "PaperParetoPoint", "PaperParetoReport", "build_paper_pareto_report"]
