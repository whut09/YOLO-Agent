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


class SlicedPaperParetoPoint(BaseModel):
    candidate_id: str
    recipe_id: str
    sliced_map50_95: float | None = None
    sliced_ap_small: float | None = None
    sliced_latency_ms: float | None = None
    sliced_throughput: float | None = None
    inference_policy_changed: bool = True
    evidence_status: str
    tradeoff: str

class PaperParetoReport(BaseModel):
    schema_version: str = "paper_pareto_report.v1"
    included: list[PaperParetoPoint] = Field(default_factory=list)
    dominated: list[str] = Field(default_factory=list)
    inference_included: list[SlicedPaperParetoPoint] = Field(default_factory=list)
    inference_dominated: list[str] = Field(default_factory=list)
    excluded: dict[str, str] = Field(default_factory=dict)

def build_paper_pareto_report(candidates: list[PaperParetoCandidate]) -> PaperParetoReport:
    eligible, inference_eligible, excluded = [], [], {}
    for candidate in candidates:
        if not candidate.verified_local:
            excluded[candidate.candidate_id] = "local_verified_evidence_required"
        elif candidate.evidence_role != "current_observation" or candidate.inheritance_depth != 0:
            excluded[candidate.candidate_id] = "current_node_only_evidence_required"
        else:
            has_standard = not all(getattr(candidate, name) is None for name in ("map50_95", "ap_small", "recall", "latency_ms", "model_size_mb"))
            has_sliced = any(name.startswith("sliced_") for name in candidate.slicing_metrics)
            if has_standard:
                eligible.append(candidate)
            if has_sliced:
                inference_eligible.append(candidate)
            if not has_standard and not has_sliced:
                excluded[candidate.candidate_id] = "no_local_pareto_metrics"
    front, dominated = [], []
    for candidate in eligible:
        if any(_dominates(other, candidate) for other in eligible if other.candidate_id != candidate.candidate_id):
            dominated.append(candidate.candidate_id)
        else:
            front.append(candidate)
    inference_front, inference_dominated = [], []
    for candidate in inference_eligible:
        if any(_sliced_dominates(other, candidate) for other in inference_eligible if other.candidate_id != candidate.candidate_id):
            inference_dominated.append(candidate.candidate_id)
        else:
            inference_front.append(candidate)
    return PaperParetoReport(
        included=[_point(item) for item in sorted(front, key=lambda x: (-(x.map50_95 or 0.0), x.latency_ms or float("inf")))],
        dominated=sorted(dominated),
        inference_included=[_sliced_point(item) for item in sorted(inference_front, key=lambda x: (-x.slicing_metrics.get("sliced_map50_95", 0.0), x.slicing_metrics.get("sliced_latency_ms", float("inf"))))],
        inference_dominated=sorted(inference_dominated),
        excluded=excluded,
    )

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


def _sliced_dominates(left: PaperParetoCandidate, right: PaperParetoCandidate) -> bool:
    better = False
    for name, direction in (("sliced_map50_95", "max"), ("sliced_ap_small", "max"), ("sliced_latency_ms", "min"), ("sliced_throughput", "max")):
        a, b = left.slicing_metrics.get(name), right.slicing_metrics.get(name)
        if a is None or b is None:
            continue
        if (direction == "max" and a < b) or (direction == "min" and a > b):
            return False
        better = better or a != b
    return better


def _sliced_point(item: PaperParetoCandidate) -> SlicedPaperParetoPoint:
    metrics = item.slicing_metrics
    return SlicedPaperParetoPoint(
        candidate_id=item.candidate_id,
        recipe_id=item.recipe_id,
        sliced_map50_95=metrics.get("sliced_map50_95"),
        sliced_ap_small=metrics.get("sliced_ap_small"),
        sliced_latency_ms=metrics.get("sliced_latency_ms"),
        sliced_throughput=metrics.get("sliced_throughput"),
        evidence_status=item.evidence_status,
        tradeoff=(
            f"sliced_mAP50-95={metrics.get('sliced_map50_95')}; "
            f"sliced_AP_small={metrics.get('sliced_ap_small')}; "
            f"sliced_latency_ms={metrics.get('sliced_latency_ms')}; "
            f"sliced_throughput={metrics.get('sliced_throughput')}"
        ),
    )

__all__ = ["PaperParetoCandidate", "PaperParetoPoint", "PaperParetoReport", "SlicedPaperParetoPoint", "build_paper_pareto_report"]
