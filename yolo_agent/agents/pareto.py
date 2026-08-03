"""Pareto-front selection for model trade-off analysis."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


MetricNamespace = Literal[
    "standard_640",
    "sliced_inference",
    "tiled_multi_scale_inference",
    "tta_inference",
    "calibrated_inference",
    "class_threshold_inference",
    "merged_inference",
]


POLICY_PREFIXES: dict[MetricNamespace, str] = {
    "sliced_inference": "sliced_",
    "tiled_multi_scale_inference": "tiled_multi_scale_",
    "tta_inference": "tta_",
    "calibrated_inference": "calibrated_",
    "class_threshold_inference": "class_threshold_",
    "merged_inference": "merged_",
}


class CandidateMetrics(BaseModel):
    """Metrics used for Pareto comparison."""

    candidate_id: str
    model: str
    accuracy: float | None = None
    latency: float | None = None
    model_size: float | None = None
    robustness: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    inference_policy_changed: bool = False
    metric_namespace: MetricNamespace = "standard_640"


class ParetoPoint(BaseModel):
    """One non-dominated candidate."""

    candidate_id: str
    model: str
    accuracy: float | None = None
    latency: float | None = None
    model_size: float | None = None
    robustness: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    inference_policy_changed: bool = False
    metric_namespace: MetricNamespace = "standard_640"
    tradeoff_summary: str = ""


class ParetoFront(BaseModel):
    """Pareto-front result."""

    points: list[ParetoPoint] = Field(default_factory=list)
    dominated: list[str] = Field(default_factory=list)


class PartitionedParetoFront(BaseModel):
    """Independent fronts for standard and changed inference protocols."""

    standard_640: ParetoFront = Field(default_factory=ParetoFront)
    sliced_inference: ParetoFront = Field(default_factory=ParetoFront)
    tiled_multi_scale_inference: ParetoFront = Field(default_factory=ParetoFront)
    tta_inference: ParetoFront = Field(default_factory=ParetoFront)
    calibrated_inference: ParetoFront = Field(default_factory=ParetoFront)
    class_threshold_inference: ParetoFront = Field(default_factory=ParetoFront)
    merged_inference: ParetoFront = Field(default_factory=ParetoFront)


class ParetoSelector:
    """Select non-dominated candidates across accuracy, latency, size, and robustness."""

    def select(self, candidates: list[CandidateMetrics]) -> ParetoFront:
        """Return non-dominated candidates."""
        points: list[ParetoPoint] = []
        dominated: list[str] = []
        for candidate in candidates:
            if any(_dominates(other, candidate) for other in candidates if other.candidate_id != candidate.candidate_id):
                dominated.append(candidate.candidate_id)
                continue
            points.append(
                ParetoPoint(
                    candidate_id=candidate.candidate_id,
                    model=candidate.model,
                    accuracy=candidate.accuracy,
                    latency=candidate.latency,
                    model_size=candidate.model_size,
                    robustness=candidate.robustness,
                    metrics=candidate.metrics,
                    inference_policy_changed=candidate.inference_policy_changed,
                    metric_namespace=candidate.metric_namespace,
                    tradeoff_summary=_tradeoff_summary(candidate),
                )
            )
        points.sort(key=lambda point: (-(point.accuracy or 0.0), point.latency if point.latency is not None else float("inf")))
        return ParetoFront(points=points, dominated=dominated)

    def select_partitioned(self, candidates: list[CandidateMetrics]) -> PartitionedParetoFront:
        """Never compare changed inference protocols with standard 640 or each other."""
        return PartitionedParetoFront.model_validate(
            {
                namespace: self.select(
                    [item for item in candidates if item.metric_namespace == namespace]
                )
                for namespace in ("standard_640", *POLICY_PREFIXES)
            }
        )


def candidate_metrics_from_row(row: dict[str, Any]) -> CandidateMetrics | None:
    """Build CandidateMetrics from a report row."""
    if not row.get("has_evidence"):
        return None
    metrics = row.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    namespaces = _policy_namespaces(metrics)
    inference_policy_changed = bool(
        row.get("inference_policy_changed")
        or metrics.get("inference_policy_changed")
        or namespaces
    )
    namespace: MetricNamespace = namespaces[0] if namespaces else "standard_640"
    prefix = POLICY_PREFIXES.get(namespace, "")
    accuracy = _first_number(metrics, f"{prefix}map50_95") if prefix else None
    latency = _first_number(metrics, f"{prefix}latency_ms") if prefix else None
    accuracy = accuracy if accuracy is not None else _first_number(metrics, "map", "mAP", "map50_95", "map50")
    latency = latency if latency is not None else _first_number(metrics, "latency", "latency_ms")
    model_size = _first_number(metrics, "model_size", "model_size_mb")
    robustness = _first_number(metrics, "robustness", "robustness_score")
    if accuracy is None and latency is None and model_size is None and robustness is None:
        return None
    components = row.get("components") or []
    component_text = " + ".join(str(component) for component in components)
    model = str(row.get("base_model") or row.get("id"))
    if component_text:
        model = f"{model} + {component_text}"
    return CandidateMetrics(
        candidate_id=str(row.get("id")),
        model=model,
        accuracy=accuracy,
        latency=latency,
        model_size=model_size,
        robustness=robustness,
        metrics=metrics,
        inference_policy_changed=inference_policy_changed,
        metric_namespace=namespace,
    )


def candidate_metric_variants_from_row(row: dict[str, Any]) -> list[CandidateMetrics]:
    """Build independent standard and inference-policy views from one evidence row."""
    if not row.get("has_evidence") or not isinstance(row.get("metrics"), dict):
        return []
    metrics: dict[str, Any] = row["metrics"]
    model = _model_label(row)
    candidate_id = str(row.get("id"))
    model_size = _first_number(metrics, "model_size", "model_size_mb")
    variants: list[CandidateMetrics] = []
    standard_accuracy = _first_number(metrics, "map", "mAP", "map50_95", "map50")
    standard_latency = _first_number(metrics, "latency", "latency_ms")
    robustness = _first_number(metrics, "robustness", "robustness_score")
    if any(value is not None for value in (standard_accuracy, standard_latency, model_size, robustness)):
        variants.append(
            CandidateMetrics(
                candidate_id=candidate_id,
                model=model,
                accuracy=standard_accuracy,
                latency=standard_latency,
                model_size=model_size,
                robustness=robustness,
                metrics={
                    key: value
                    for key, value in metrics.items()
                    if not any(str(key).startswith(prefix) for prefix in POLICY_PREFIXES.values())
                    and key != "inference_policy_changed"
                },
                metric_namespace="standard_640",
            )
        )
    for namespace in _policy_namespaces(metrics):
        prefix = POLICY_PREFIXES[namespace]
        policy_accuracy = _first_number(metrics, f"{prefix}map50_95")
        policy_latency = _first_number(metrics, f"{prefix}latency_ms")
        policy_robustness = _first_number(metrics, f"{prefix}robustness")
        if all(
            value is None
            for value in (policy_accuracy, policy_latency, policy_robustness)
        ):
            continue
        variants.append(
            CandidateMetrics(
                candidate_id=candidate_id,
                model=f"{model} + {namespace.replace('_', ' ')}",
                accuracy=policy_accuracy,
                latency=policy_latency,
                model_size=model_size,
                robustness=policy_robustness,
                metrics={
                    key: value
                    for key, value in metrics.items()
                    if str(key).startswith(prefix) or key == "inference_policy_changed"
                },
                inference_policy_changed=True,
                metric_namespace=namespace,
            )
        )
    return variants


def _model_label(row: dict[str, Any]) -> str:
    components = row.get("components") or []
    component_text = " + ".join(str(component) for component in components)
    model = str(row.get("base_model") or row.get("id"))
    return f"{model} + {component_text}" if component_text else model


def _policy_namespaces(metrics: dict[str, Any]) -> list[MetricNamespace]:
    return [
        namespace
        for namespace, prefix in POLICY_PREFIXES.items()
        if any(str(name).startswith(prefix) for name in metrics)
    ]


def _dominates(left: CandidateMetrics, right: CandidateMetrics) -> bool:
    comparable = False
    strictly_better = False
    for metric, direction in {
        "accuracy": "max",
        "robustness": "max",
        "latency": "min",
        "model_size": "min",
    }.items():
        left_value = getattr(left, metric)
        right_value = getattr(right, metric)
        if left_value is None or right_value is None:
            continue
        comparable = True
        if direction == "max":
            if left_value < right_value:
                return False
            strictly_better = strictly_better or left_value > right_value
        else:
            if left_value > right_value:
                return False
            strictly_better = strictly_better or left_value < right_value
    return comparable and strictly_better


def _tradeoff_summary(candidate: CandidateMetrics) -> str:
    parts = []
    if candidate.accuracy is not None:
        parts.append(f"accuracy={candidate.accuracy}")
    if candidate.latency is not None:
        parts.append(f"latency={candidate.latency}")
    if candidate.model_size is not None:
        parts.append(f"model_size={candidate.model_size}")
    if candidate.robustness is not None:
        parts.append(f"robustness={candidate.robustness}")
    return ", ".join(parts) if parts else "metrics unavailable"


def _first_number(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None
