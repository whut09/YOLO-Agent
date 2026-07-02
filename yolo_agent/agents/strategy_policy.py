"""Policy proposal and evaluator boundary.

LLMs or humans may propose policies, but they do not directly select final
experiments. The evaluator validates constraints and produces accepted candidates.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.components.compatibility import BaseModelSpec, CompatibilityChecker, RiskLevel
from yolo_agent.components.registry import ComponentRegistry
from yolo_agent.core.task_spec import TaskSpec


PolicySource = Literal["llm", "human", "rule_engine"]


class PolicyConstraint(BaseModel):
    """Constraint attached to a candidate policy proposal."""

    name: str
    value: Any
    hard: bool = True


class CandidatePolicy(BaseModel):
    """A proposal, not a decision."""

    policy_id: str
    source: PolicySource = "rule_engine"
    base_model: str
    scale: str
    framework: str
    components: list[str] = Field(default_factory=list)
    train_overrides: dict[str, Any] = Field(default_factory=dict)
    constraints: list[PolicyConstraint] = Field(default_factory=list)
    expected_effect: list[str] = Field(default_factory=list)
    risk: RiskLevel = "medium"
    rationale: str = ""


class PolicyEvaluation(BaseModel):
    """Evaluator output for one policy."""

    policy_id: str
    accepted: bool
    score: float = 0.0
    candidate_config: CandidateConfig | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    expected_effect: list[str] = Field(default_factory=list)
    risk: RiskLevel = "medium"


class PolicyEvaluationReport(BaseModel):
    """Batch policy evaluation report."""

    evaluations: list[PolicyEvaluation]

    @property
    def accepted_candidates(self) -> list[CandidateConfig]:
        """Return only evaluator-approved candidates."""
        return [
            evaluation.candidate_config
            for evaluation in self.evaluations
            if evaluation.accepted and evaluation.candidate_config is not None
        ]


class PolicyEvaluator:
    """Validate and score candidate policies."""

    def __init__(self, registry: ComponentRegistry, checker: CompatibilityChecker | None = None) -> None:
        self.registry = registry
        self.checker = checker or CompatibilityChecker()

    def evaluate(self, policies: list[CandidatePolicy], task_spec: TaskSpec) -> PolicyEvaluationReport:
        """Evaluate policy proposals against task constraints."""
        return PolicyEvaluationReport(
            evaluations=[self.evaluate_one(policy, task_spec) for policy in policies]
        )

    def evaluate_one(self, policy: CandidatePolicy, task_spec: TaskSpec) -> PolicyEvaluation:
        """Evaluate one policy proposal."""
        errors: list[str] = []
        warnings: list[str] = []
        components = []
        for component_id in policy.components:
            component = next((card for card in self.registry.cards if card.id == component_id), None)
            if component is None:
                errors.append(f"Unknown component: {component_id}")
            else:
                components.append(component)

        hard_constraint_errors = _check_policy_constraints(policy, task_spec)
        errors.extend(hard_constraint_errors)

        base_model = BaseModelSpec(
            name=policy.base_model,
            framework=policy.framework,
            model_family=_model_family(policy.base_model),
            export_format=str(_constraint_value(policy, "export_format", "none")),
            estimated_latency_ms=_optional_float(_constraint_value(policy, "estimated_latency_ms")),
            estimated_model_size_mb=_optional_float(_constraint_value(policy, "estimated_model_size_mb")),
        )
        compatibility = self.checker.check(task_spec, base_model, components)
        errors.extend(compatibility.errors)
        warnings.extend(compatibility.warnings)

        accepted = not errors
        candidate = None
        if accepted:
            candidate = CandidateConfig(
                candidate_id=policy.policy_id,
                base_model=policy.base_model,
                scale=policy.scale,
                framework=policy.framework,
                components=policy.components,
                train_overrides=policy.train_overrides,
                expected_effect=policy.expected_effect,
                risk=compatibility.estimated_risk,
            )

        return PolicyEvaluation(
            policy_id=policy.policy_id,
            accepted=accepted,
            score=_score_policy(policy, compatibility.estimated_risk, accepted),
            candidate_config=candidate,
            errors=errors,
            warnings=warnings,
            expected_effect=policy.expected_effect,
            risk=compatibility.estimated_risk if accepted else policy.risk,
        )


def _check_policy_constraints(policy: CandidatePolicy, task_spec: TaskSpec) -> list[str]:
    errors: list[str] = []
    for constraint in policy.constraints:
        if constraint.name == "max_latency_ms" and task_spec.max_latency_ms is not None:
            if float(constraint.value) > task_spec.max_latency_ms and constraint.hard:
                errors.append(
                    f"Policy latency {constraint.value} exceeds task max_latency_ms={task_spec.max_latency_ms}."
                )
        if constraint.name == "max_model_size_mb" and task_spec.max_model_size_mb is not None:
            if float(constraint.value) > task_spec.max_model_size_mb and constraint.hard:
                errors.append(
                    f"Policy model size {constraint.value} exceeds task max_model_size_mb={task_spec.max_model_size_mb}."
                )
    return errors


def _score_policy(policy: CandidatePolicy, risk: RiskLevel, accepted: bool) -> float:
    if not accepted:
        return 0.0
    risk_penalty = {"low": 0.0, "medium": 0.2, "high": 0.5}[risk]
    effect_bonus = min(0.3, len(policy.expected_effect) * 0.1)
    return max(0.0, 1.0 + effect_bonus - risk_penalty)


def _constraint_value(policy: CandidatePolicy, name: str, default: Any = None) -> Any:
    for constraint in policy.constraints:
        if constraint.name == name:
            return constraint.value
    return default


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _model_family(base_model: str) -> str:
    lowered = base_model.lower()
    for family in ("yolov11", "yolov10", "yolov9", "yolov8", "yolov7", "yolov6", "yolov5"):
        if family.replace("v", "") in lowered or family in lowered:
            return family
    if "yolo11" in lowered:
        return "yolov11"
    return "generic"

