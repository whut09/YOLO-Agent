"""Audit runtime-hook readiness without inferring it from component IDs."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, Field

from yolo_agent.agents.paper_adapter_planning.schemas import (
    AdapterImplementationEstimate,
    RuntimeHookAvailability,
)


class RuntimeHookAssessment(BaseModel):
    required_hook: str | None = None
    available: bool
    verified: bool
    score: float = 0.0
    reasons: list[str] = Field(default_factory=list)


def assess_runtime_hook(
    estimate: AdapterImplementationEstimate,
    hooks: Iterable[RuntimeHookAvailability],
    *,
    verified_weight: float,
) -> RuntimeHookAssessment:
    required = estimate.required_runtime_hook
    if not required:
        return RuntimeHookAssessment(
            available=True,
            verified=True,
            reasons=["no_additional_runtime_hook_required"],
        )
    by_id = {item.hook_id: item for item in hooks}
    hook = by_id.get(required)
    if hook is None or not hook.available:
        return RuntimeHookAssessment(
            required_hook=required,
            available=False,
            verified=False,
            score=-verified_weight,
            reasons=[f"runtime_hook_unavailable:{required}"],
        )
    if not hook.verified:
        return RuntimeHookAssessment(
            required_hook=required,
            available=True,
            verified=False,
            score=verified_weight * 0.25,
            reasons=[f"runtime_hook_available_not_verified:{required}"],
        )
    return RuntimeHookAssessment(
        required_hook=required,
        available=True,
        verified=True,
        score=verified_weight,
        reasons=[f"runtime_hook_verified:{required}:{hook.version}"],
    )


__all__ = ["RuntimeHookAssessment", "assess_runtime_hook"]
