"""Map structured detection errors to next-action policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


ErrorCategory = Literal[
    "false_negative",
    "false_positive",
    "localization_error",
    "classification_error",
]

DetectionErrorType = Literal[
    "small_object_miss",
    "occlusion_miss",
    "low_contrast_miss",
    "out_of_distribution_miss",
    "background_confusion",
    "hard_negative",
    "label_noise_induced",
    "loose_box",
    "shifted_box",
    "partial_object_box",
    "class_confusion",
    "long_tail_bias",
]


class DetectionErrorObservation(BaseModel):
    """Observed detection error signal from eval analysis."""

    error_type: DetectionErrorType
    count: int = Field(default=1, ge=0)
    severity: Literal["low", "medium", "high"] = "medium"
    notes: list[str] = Field(default_factory=list)


class ActionPolicy(BaseModel):
    """A recommended action for an observed error type."""

    id: str
    description: str
    target_components: list[str] = Field(default_factory=list)
    target_variables: dict[str, Any] = Field(default_factory=dict)
    expected_effect: str = ""
    risks: list[str] = Field(default_factory=list)


class ErrorPolicy(BaseModel):
    """Policy bundle for one error type."""

    error_category: ErrorCategory
    actions: list[ActionPolicy] = Field(default_factory=list)


class ActionRecommendation(BaseModel):
    """Prioritized action recommendation produced by the mapper."""

    error_type: DetectionErrorType
    error_category: ErrorCategory
    action: ActionPolicy
    priority: float
    rationale: str


class ErrorActionPlan(BaseModel):
    """A diagnosis-to-action plan for observed detection errors."""

    observations: list[DetectionErrorObservation]
    recommendations: list[ActionRecommendation]
    unresolved_errors: list[DetectionErrorType] = Field(default_factory=list)


class ErrorActionMapper:
    """Map detection error observations to configured action policies."""

    def __init__(self, policies: dict[DetectionErrorType, ErrorPolicy]) -> None:
        self.policies = policies

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "ErrorActionMapper":
        """Load action policies from YAML."""
        policy_path = Path(path) if path is not None else default_error_policy_path()
        with policy_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Error action policy YAML must contain a mapping: {policy_path}")
        raw_policies = data.get("policies", {})
        if not isinstance(raw_policies, dict):
            raise ValueError("Error action policy YAML requires a 'policies' mapping.")
        policies = {
            error_type: ErrorPolicy.model_validate(policy)
            for error_type, policy in raw_policies.items()
        }
        return cls(policies)  # type: ignore[arg-type]

    def map_errors(self, observations: list[DetectionErrorObservation]) -> ErrorActionPlan:
        """Map observed errors to prioritized actions."""
        recommendations: list[ActionRecommendation] = []
        unresolved: list[DetectionErrorType] = []
        for observation in observations:
            policy = self.policies.get(observation.error_type)
            if policy is None:
                unresolved.append(observation.error_type)
                continue
            for action in policy.actions:
                recommendations.append(
                    ActionRecommendation(
                        error_type=observation.error_type,
                        error_category=policy.error_category,
                        action=action,
                        priority=_priority(observation),
                        rationale=_rationale(observation, action),
                    )
                )

        recommendations.sort(key=lambda item: item.priority, reverse=True)
        return ErrorActionPlan(
            observations=observations,
            recommendations=recommendations,
            unresolved_errors=unresolved,
        )


def default_error_policy_path() -> Path:
    """Return bundled error-action policy config."""
    return Path(__file__).resolve().parents[2] / "configs" / "error_action_policies.yaml"


def _priority(observation: DetectionErrorObservation) -> float:
    severity_weight = {"low": 1.0, "medium": 2.0, "high": 3.0}[observation.severity]
    return severity_weight * max(observation.count, 1)


def _rationale(observation: DetectionErrorObservation, action: ActionPolicy) -> str:
    return (
        f"{observation.error_type} observed {observation.count} times "
        f"with {observation.severity} severity; action={action.id}."
    )

