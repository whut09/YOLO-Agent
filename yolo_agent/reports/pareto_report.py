"""Strict local-evidence Pareto analysis for paper recipe outcomes."""
from __future__ import annotations
from pydantic import BaseModel, Field, model_validator


INFERENCE_PREFIXES = (
    "sliced_",
    "tiled_multi_scale_",
    "tta_",
    "calibrated_",
    "class_threshold_",
    "merged_",
)

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
    inference_metrics: dict[str, float] = Field(default_factory=dict)
    evidence_status: str = "possible"

    @model_validator(mode="after")
    def mark_slicing(self) -> "PaperParetoCandidate":
        if self.slicing_metrics or self.inference_metrics:
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
    metric_namespace: str = "sliced_inference"
    metrics: dict[str, float] = Field(default_factory=dict)
    inference_policy_changed: bool = True
    evidence_status: str
    tradeoff: str

class PaperParetoReport(BaseModel):
    schema_version: str = "paper_pareto_report.v1"
    included: list[PaperParetoPoint] = Field(default_factory=list)
    dominated: list[str] = Field(default_factory=list)
    inference_included: list[SlicedPaperParetoPoint] = Field(default_factory=list)
    inference_dominated: list[str] = Field(default_factory=list)
    inference_fronts: dict[str, list[SlicedPaperParetoPoint]] = Field(default_factory=dict)
    inference_dominated_by_namespace: dict[str, list[str]] = Field(default_factory=dict)
    excluded: dict[str, str] = Field(default_factory=dict)

def build_paper_pareto_report(candidates: list[PaperParetoCandidate]) -> PaperParetoReport:
    eligible, excluded = [], {}
    inference_by_namespace: dict[str, list[PaperParetoCandidate]] = {}
    for candidate in candidates:
        if not candidate.verified_local:
            excluded[candidate.candidate_id] = "local_verified_evidence_required"
        elif candidate.evidence_role != "current_observation" or candidate.inheritance_depth != 0:
            excluded[candidate.candidate_id] = "current_node_only_evidence_required"
        else:
            has_standard = not all(getattr(candidate, name) is None for name in ("map50_95", "ap_small", "recall", "latency_ms", "model_size_mb"))
            policy_metrics = _candidate_policy_metrics(candidate)
            namespaces = _policy_namespaces(policy_metrics)
            if has_standard:
                eligible.append(candidate)
            for namespace in namespaces:
                inference_by_namespace.setdefault(namespace, []).append(candidate)
            if not has_standard and not namespaces:
                excluded[candidate.candidate_id] = "no_local_pareto_metrics"
    front, dominated = [], []
    for candidate in eligible:
        if any(_dominates(other, candidate) for other in eligible if other.candidate_id != candidate.candidate_id):
            dominated.append(candidate.candidate_id)
        else:
            front.append(candidate)
    inference_fronts: dict[str, list[SlicedPaperParetoPoint]] = {}
    inference_dominated_by_namespace: dict[str, list[str]] = {}
    for namespace, namespace_candidates in inference_by_namespace.items():
        namespace_front: list[PaperParetoCandidate] = []
        namespace_dominated: list[str] = []
        for candidate in namespace_candidates:
            if any(
                _inference_dominates(other, candidate, namespace)
                for other in namespace_candidates
                if other.candidate_id != candidate.candidate_id
            ):
                namespace_dominated.append(candidate.candidate_id)
            else:
                namespace_front.append(candidate)
        inference_fronts[namespace] = [
            _inference_point(item, namespace)
            for item in sorted(
                namespace_front,
                key=lambda item: _inference_sort_key(item, namespace),
            )
        ]
        inference_dominated_by_namespace[namespace] = sorted(namespace_dominated)
    inference_included = [
        item for namespace in sorted(inference_fronts) for item in inference_fronts[namespace]
    ]
    inference_dominated = sorted(
        {item for values in inference_dominated_by_namespace.values() for item in values}
    )
    return PaperParetoReport(
        included=[_point(item) for item in sorted(front, key=lambda x: (-(x.map50_95 or 0.0), x.latency_ms or float("inf")))],
        dominated=sorted(dominated),
        inference_included=inference_included,
        inference_dominated=inference_dominated,
        inference_fronts=inference_fronts,
        inference_dominated_by_namespace=inference_dominated_by_namespace,
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


def _inference_dominates(
    left: PaperParetoCandidate, right: PaperParetoCandidate, namespace: str
) -> bool:
    prefix = namespace.removesuffix("_inference") + "_"
    left_metrics = _candidate_policy_metrics(left)
    right_metrics = _candidate_policy_metrics(right)
    better = False
    for suffix, direction in (("map50_95", "max"), ("ap_small", "max"), ("recall", "max"), ("latency_ms", "min"), ("throughput", "max"), ("peak_vram_mb", "min")):
        a, b = left_metrics.get(prefix + suffix), right_metrics.get(prefix + suffix)
        if a is None or b is None:
            continue
        if (direction == "max" and a < b) or (direction == "min" and a > b):
            return False
        better = better or a != b
    return better


def _inference_point(
    item: PaperParetoCandidate, namespace: str
) -> SlicedPaperParetoPoint:
    metrics = _candidate_policy_metrics(item)
    prefix = namespace.removesuffix("_inference") + "_"
    selected = {
        name: value for name, value in metrics.items() if name.startswith(prefix)
    }
    return SlicedPaperParetoPoint(
        candidate_id=item.candidate_id,
        recipe_id=item.recipe_id,
        sliced_map50_95=selected.get(prefix + "map50_95"),
        sliced_ap_small=selected.get(prefix + "ap_small"),
        sliced_latency_ms=selected.get(prefix + "latency_ms"),
        sliced_throughput=selected.get(prefix + "throughput"),
        metric_namespace=namespace,
        metrics=selected,
        evidence_status=item.evidence_status,
        tradeoff=(
            f"namespace={namespace}; mAP50-95={selected.get(prefix + 'map50_95')}; "
            f"AP_small={selected.get(prefix + 'ap_small')}; "
            f"latency_ms={selected.get(prefix + 'latency_ms')}; "
            f"throughput={selected.get(prefix + 'throughput')}"
        ),
    )


def _candidate_policy_metrics(item: PaperParetoCandidate) -> dict[str, float]:
    return {**item.slicing_metrics, **item.inference_metrics}


def _policy_namespaces(metrics: dict[str, float]) -> list[str]:
    return [
        prefix.removesuffix("_") + "_inference"
        for prefix in INFERENCE_PREFIXES
        if any(name.startswith(prefix) for name in metrics)
    ]


def _inference_sort_key(
    item: PaperParetoCandidate, namespace: str
) -> tuple[float, float]:
    prefix = namespace.removesuffix("_inference") + "_"
    metrics = _candidate_policy_metrics(item)
    return (
        -metrics.get(prefix + "map50_95", 0.0),
        metrics.get(prefix + "latency_ms", float("inf")),
    )

__all__ = ["PaperParetoCandidate", "PaperParetoPoint", "PaperParetoReport", "SlicedPaperParetoPoint", "build_paper_pareto_report"]
