"""Pareto-front selection for model trade-off analysis."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


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
    tradeoff_summary: str = ""


class ParetoFront(BaseModel):
    """Pareto-front result."""

    points: list[ParetoPoint] = Field(default_factory=list)
    dominated: list[str] = Field(default_factory=list)


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
                    tradeoff_summary=_tradeoff_summary(candidate),
                )
            )
        points.sort(key=lambda point: (-(point.accuracy or 0.0), point.latency if point.latency is not None else float("inf")))
        return ParetoFront(points=points, dominated=dominated)


def candidate_metrics_from_row(row: dict[str, Any]) -> CandidateMetrics | None:
    """Build CandidateMetrics from a report row."""
    if not row.get("has_evidence"):
        return None
    metrics = row.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    inference_policy_changed = bool(
        row.get("inference_policy_changed")
        or metrics.get("inference_policy_changed")
        or any(str(name).startswith("sliced_") for name in metrics)
    )
    accuracy = _first_number(metrics, "sliced_map50_95") if inference_policy_changed else None
    latency = _first_number(metrics, "sliced_latency_ms") if inference_policy_changed else None
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
    )


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
