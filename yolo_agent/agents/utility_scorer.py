"""Utility scoring for policy proposals.

The scorer makes proposal ordering explicit:

utility = expected_gain * confidence * target_error_relevance
          - training_cost - latency_risk - model_size_risk
          - implementation_risk - evidence_gap_penalty
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal

import yaml
from pydantic import BaseModel, Field

from yolo_agent.adapters.ultralytics.training import UltralyticsTrainingConfig
from yolo_agent.agents.strategy_policy import CandidatePolicy, PolicyConstraint
from yolo_agent.components.compatibility import RiskLevel
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.core.optimization_objective import OptimizationObjective
from yolo_agent.core.policy_memory import (
    ActionFingerprint,
    PilotToFullPrediction,
    PolicyMemoryRecord,
    PolicyMemoryStore,
    pilot_to_full_prediction,
)
from yolo_agent.core.task_spec import TaskSpec
from yolo_agent.resources import ResourcePaths


UtilityDecision = Literal["run_now", "defer", "needs_evidence", "reject"]


class UtilityCost(BaseModel):
    """Estimated proposal cost terms."""

    gpu_hours: float = 0.0
    training_cost: float = 0.0
    latency_risk: float = 0.0
    model_size_risk: float = 0.0
    implementation_risk: float = 0.0
    evidence_gap_penalty: float = 0.0


class UtilityScore(BaseModel):
    """Transparent utility score for one proposal."""

    expected_gain: dict[str, float] = Field(default_factory=dict)
    aggregate_expected_gain: float = 0.0
    confidence: float = 0.0
    target_error_relevance: float = 0.0
    cost: UtilityCost = Field(default_factory=UtilityCost)
    utility: float = 0.0
    decision: UtilityDecision = "defer"
    objective_hash: str | None = None
    objective_progress: float | None = None
    pilot_to_full_posterior: PilotToFullPrediction | None = None
    reasons: list[str] = Field(default_factory=list)


class UtilityPolicy(BaseModel):
    """Configurable weights for utility scoring."""

    metric_weights: dict[str, float] = Field(default_factory=dict)
    source_confidence: dict[str, float] = Field(
        default_factory=lambda: {"rule_engine": 0.55, "human": 0.5, "llm": 0.35}
    )
    risk_penalties: dict[RiskLevel, float] = Field(
        default_factory=lambda: {"low": 0.05, "medium": 0.25, "high": 0.6}
    )
    fallback_gain_per_priority_hint: float = 0.1
    target_bound_bonus: float = 0.2
    single_variable_bonus: float = 0.1
    evidence_gap_penalty_per_item: float = 0.15
    evidence_required_penalty_per_item: float = 0.03
    training_cost_per_gpu_hour: float = 0.02
    action_domain_cost_multiplier: dict[str, float] = Field(
        default_factory=lambda: {
            "model": 1.0,
            "training": 0.7,
            "augmentation": 0.6,
            "data": 0.35,
            "label": 0.25,
            "postprocess": 0.2,
            "evidence": 0.05,
        }
    )
    action_domain_confidence_adjustment: dict[str, float] = Field(
        default_factory=lambda: {
            "model": 0.0,
            "training": 0.0,
            "augmentation": 0.03,
            "data": 0.08,
            "label": 0.05,
            "postprocess": 0.04,
            "evidence": 0.12,
        }
    )
    latency_risk_weight: float = 0.5
    model_size_risk_weight: float = 0.4
    default_target_error_relevance: float = 0.35
    target_error_bound_relevance: float = 0.8
    target_action_match_relevance: float = 1.0
    run_now_threshold: float = 0.15
    defer_threshold: float = 0.0

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "UtilityPolicy":
        """Load utility policy from YAML."""
        policy_path = Path(path) if path is not None else ResourcePaths.UTILITY_POLICY
        with policy_path.open("r", encoding="utf-8-sig") as file:
            raw = yaml.safe_load(file) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Utility policy YAML must contain a mapping: {policy_path}")
        return cls.model_validate(raw)


class UtilityScorer:
    """Score policy proposals using expected value, confidence, cost, and risk."""

    def __init__(self, policy: UtilityPolicy | None = None) -> None:
        self.policy = policy or UtilityPolicy.from_yaml()

    def score(
        self,
        proposal: CandidatePolicy,
        task_spec: TaskSpec,
        changed_variables: dict[str, Any],
        missing_evidence: list[str] | None = None,
        error_facts: list[ErrorFact] | None = None,
        training_config: UltralyticsTrainingConfig | None = None,
        optimization_objective: OptimizationObjective | None = None,
        policy_memory: PolicyMemoryStore | Iterable[PolicyMemoryRecord] | None = None,
        action_fingerprint: ActionFingerprint | None = None,
        observed_pilot_delta: float | None = None,
    ) -> UtilityScore:
        """Return explicit utility decomposition for one proposal."""
        missing = list(missing_evidence or [])
        posterior = _full_gain_posterior(
            policy_memory,
            action_fingerprint,
            proposal=proposal,
            objective=optimization_objective,
            observed_pilot_delta=observed_pilot_delta,
        )
        memory_requested = policy_memory is not None and action_fingerprint is not None
        expected_gain = _expected_gain(
            proposal,
            self.policy,
            posterior=posterior,
            memory_requested=memory_requested,
            objective=optimization_objective,
        )
        aggregate_gain = _aggregate_gain(expected_gain, self.policy)
        objective_progress = _objective_progress(expected_gain, optimization_objective)
        if objective_progress is not None:
            aggregate_gain += objective_progress
        confidence = _confidence(
            proposal,
            changed_variables,
            self.policy,
            posterior=posterior,
            memory_requested=memory_requested,
        )
        relevance = _target_error_relevance(proposal, error_facts, self.policy)
        cost = _cost(
            proposal,
            task_spec,
            missing,
            training_config,
            self.policy,
            optimization_objective,
            posterior=posterior,
        )
        utility = round(
            aggregate_gain * confidence * relevance
            - cost.training_cost
            - cost.latency_risk
            - cost.model_size_risk
            - cost.implementation_risk
            - cost.evidence_gap_penalty,
            6,
        )
        decision = _decision(utility, missing, self.policy)
        objective_blockers = _objective_blockers(proposal, cost, optimization_objective)
        if memory_requested and not expected_gain and not missing and not objective_blockers:
            decision = "defer"
        if objective_blockers:
            decision = "reject"
        return UtilityScore(
            expected_gain=expected_gain,
            aggregate_expected_gain=aggregate_gain,
            confidence=confidence,
            target_error_relevance=relevance,
            cost=cost,
            utility=utility,
            decision=decision,
            objective_hash=optimization_objective.objective_hash if optimization_objective is not None else None,
            objective_progress=objective_progress,
            pilot_to_full_posterior=posterior,
            reasons=_reasons(
                expected_gain,
                confidence,
                relevance,
                cost,
                missing,
                decision,
                optimization_objective,
                objective_blockers,
            ),
        )


def _full_gain_posterior(
    memory: PolicyMemoryStore | Iterable[PolicyMemoryRecord] | None,
    fingerprint: ActionFingerprint | None,
    *,
    proposal: CandidatePolicy,
    objective: OptimizationObjective | None,
    observed_pilot_delta: float | None,
) -> PilotToFullPrediction | None:
    if memory is None or fingerprint is None:
        return None
    records = memory.read() if isinstance(memory, PolicyMemoryStore) else list(memory)
    metric_name = objective.primary_metric if objective is not None else _proposal_metric(proposal)
    return pilot_to_full_prediction(
        records,
        fingerprint,
        target=None,
        metric_name=metric_name,
        observed_pilot_delta=observed_pilot_delta,
    )


def _primary_target(proposal: CandidatePolicy) -> str | None:
    if not proposal.target_error_facts:
        return None
    fact = proposal.target_error_facts[0]
    if not isinstance(fact, dict):
        return str(fact)
    fact_type = str(fact.get("fact_type") or "error_fact")
    subject = str(fact.get("class_name") or fact.get("class_pair") or fact.get("area") or fact.get("subject") or "unknown")
    metric = str(fact.get("metric_name") or "count")
    return f"{fact_type}:{subject}:{metric}"


def _proposal_metric(proposal: CandidatePolicy) -> str | None:
    raw = proposal.expected_improvement if isinstance(proposal.expected_improvement, dict) else {}
    metric = raw.get("metric_name")
    if isinstance(metric, str) and metric:
        return metric
    expected = raw.get("expected_gain")
    if isinstance(expected, dict) and len(expected) == 1:
        return str(next(iter(expected)))
    if proposal.target_error_facts and isinstance(proposal.target_error_facts[0], dict):
        value = proposal.target_error_facts[0].get("metric_name")
        return str(value) if value else None
    return None


def _expected_gain(
    proposal: CandidatePolicy,
    policy: UtilityPolicy,
    *,
    posterior: PilotToFullPrediction | None = None,
    memory_requested: bool = False,
    objective: OptimizationObjective | None = None,
) -> dict[str, float]:
    if posterior is not None and posterior.expected_full_gain is not None:
        metric = posterior.metric_name or (objective.primary_metric if objective is not None else "expected_full_gain")
        return {metric: posterior.expected_full_gain}
    if memory_requested:
        return {}
    raw = proposal.expected_improvement
    gains: dict[str, float] = {}
    expected_gain = raw.get("expected_gain") if isinstance(raw, dict) else None
    if isinstance(expected_gain, dict):
        gains.update(_numeric_mapping(expected_gain))
    gains.update(
        _numeric_mapping(
            {
                key: value
                for key, value in raw.items()
                if key
                not in {
                    "expected_gain",
                    "metric_name",
                    "direction",
                    "target",
                    "confidence",
                    "minimum_expected_delta",
                    "summary",
                }
            }
            if isinstance(raw, dict)
            else {}
        )
    )
    if not gains:
        gains["proposal_prior"] = round(proposal.priority_hint * policy.fallback_gain_per_priority_hint, 6)
    return gains


def _numeric_mapping(values: dict[str, Any]) -> dict[str, float]:
    gains: dict[str, float] = {}
    for key, value in values.items():
        numeric = _float_or_none(value)
        if numeric is not None:
            gains[str(key)] = numeric
    return gains


def _aggregate_gain(gains: dict[str, float], policy: UtilityPolicy) -> float:
    total = 0.0
    for metric, value in gains.items():
        total += value * policy.metric_weights.get(metric, 1.0)
    return round(total, 6)


def _confidence(
    proposal: CandidatePolicy,
    changed_variables: dict[str, Any],
    policy: UtilityPolicy,
    *,
    posterior: PilotToFullPrediction | None = None,
    memory_requested: bool = False,
) -> float:
    if memory_requested:
        if posterior is None or posterior.expected_full_gain is None:
            if posterior is None or posterior.pilot_observation_count == 0:
                return 0.1
            if posterior.pilot_mean_effect_delta is not None and posterior.pilot_mean_effect_delta < 0:
                return 0.05
            return 0.08
        bounded = posterior.confidence
        if proposal.target_error_facts:
            bounded += min(policy.target_bound_bonus, 0.1)
        if len(changed_variables) == 1:
            bounded += min(policy.single_variable_bonus, 0.05)
        return round(max(0.0, min(0.8, bounded)), 6)
    raw = proposal.expected_improvement.get("confidence") if isinstance(proposal.expected_improvement, dict) else None
    explicit = _float_or_none(raw)
    confidence = explicit if explicit is not None else policy.source_confidence.get(proposal.source, 0.4)
    if proposal.target_error_facts:
        confidence += policy.target_bound_bonus
    if len(changed_variables) == 1:
        confidence += policy.single_variable_bonus
    confidence += policy.action_domain_confidence_adjustment.get(proposal.action_domain, 0.0)
    return round(max(0.0, min(1.0, confidence)), 6)


def _target_error_relevance(
    proposal: CandidatePolicy,
    error_facts: list[ErrorFact] | None,
    policy: UtilityPolicy,
) -> float:
    if proposal.action_domain == "evidence":
        return policy.target_action_match_relevance
    if not proposal.target_error_facts:
        return policy.default_target_error_relevance
    target_actions = _target_actions(proposal)
    if target_actions and error_facts:
        for fact in error_facts:
            if set(target_actions) & set(fact.action_candidates):
                return policy.target_action_match_relevance
    return policy.target_error_bound_relevance


def _cost(
    proposal: CandidatePolicy,
    task_spec: TaskSpec,
    missing_evidence: list[str],
    training_config: UltralyticsTrainingConfig | None,
    policy: UtilityPolicy,
    objective: OptimizationObjective | None,
    *,
    posterior: PilotToFullPrediction | None = None,
) -> UtilityCost:
    gpu_hours = _constraint_float(proposal.constraints, "estimated_gpu_hours")
    if gpu_hours is None:
        gpu_hours = _profile_gpu_hours(training_config)
    gpu_hours *= policy.action_domain_cost_multiplier.get(proposal.action_domain, 1.0)
    estimated_latency = _constraint_float(proposal.constraints, "estimated_latency_ms")
    estimated_size = _constraint_float(proposal.constraints, "estimated_model_size_mb")
    latency_risk = _budget_risk(
        value=estimated_latency,
        budget=task_spec.max_latency_ms,
        weight=policy.latency_risk_weight,
    )
    size_risk = _budget_risk(
        value=estimated_size,
        budget=task_spec.max_model_size_mb,
        weight=policy.model_size_risk_weight,
    )
    if objective is not None:
        latency_risk += _regression_risk(
            _constraint_float(proposal.constraints, "estimated_latency_regression"),
            objective.max_latency_regression,
        )
        size_risk += _regression_risk(
            _constraint_float(proposal.constraints, "estimated_model_size_regression"),
            objective.max_model_size_regression,
        )
        if posterior is not None:
            latency_risk += _regression_risk(
                _percent_to_ratio(posterior.expected_latency_delta_pct),
                objective.max_latency_regression,
            )
            size_risk += _regression_risk(
                _percent_to_ratio(posterior.expected_model_size_delta_pct),
                objective.max_model_size_regression,
            )
    evidence_penalty = (
        len(missing_evidence) * policy.evidence_gap_penalty_per_item
        + len(proposal.evidence_required) * policy.evidence_required_penalty_per_item
    )
    implementation_risk = policy.risk_penalties.get(proposal.risk, 0.25)
    if proposal.action_domain == "evidence":
        implementation_risk = min(implementation_risk, 0.02)
    return UtilityCost(
        gpu_hours=round(gpu_hours, 6),
        training_cost=round(gpu_hours * policy.training_cost_per_gpu_hour, 6),
        latency_risk=latency_risk,
        model_size_risk=size_risk,
        implementation_risk=implementation_risk,
        evidence_gap_penalty=round(evidence_penalty, 6),
    )


def _budget_risk(value: float | None, budget: float | None, weight: float) -> float:
    if value is None or budget is None or budget <= 0:
        return 0.0
    ratio = value / budget
    if ratio <= 0.8:
        return 0.0
    return round((ratio - 0.8) * weight, 6)


def _profile_gpu_hours(training_config: UltralyticsTrainingConfig | None) -> float:
    if training_config is None:
        return 0.1
    if training_config.budget_profile is None:
        return max(0.1, float(training_config.epochs) / 10.0)
    profile = training_config.selected_budget_profile()
    return max(0.1, float(profile.epochs) * len(profile.seeds) * float(profile.fraction))


def _decision(utility: float, missing_evidence: list[str], policy: UtilityPolicy) -> UtilityDecision:
    if missing_evidence:
        return "needs_evidence"
    if utility >= policy.run_now_threshold:
        return "run_now"
    if utility >= policy.defer_threshold:
        return "defer"
    return "reject"


def _reasons(
    expected_gain: dict[str, float],
    confidence: float,
    relevance: float,
    cost: UtilityCost,
    missing_evidence: list[str],
    decision: UtilityDecision,
    objective: OptimizationObjective | None,
    objective_blockers: list[str],
) -> list[str]:
    reasons = [
        f"expected_gain={expected_gain}",
        f"confidence={confidence}",
        f"target_error_relevance={relevance}",
        f"cost={cost.model_dump(mode='json')}",
        f"utility_decision={decision}",
    ]
    if missing_evidence:
        reasons.append(f"missing_evidence={missing_evidence}")
    if objective is not None:
        reasons.append(
            f"objective={objective.objective_hash}:{objective.primary_metric}:delta={objective.required_delta()}"
        )
    if objective_blockers:
        reasons.append(f"objective_blockers={objective_blockers}")
    return reasons


def _objective_progress(
    expected_gain: dict[str, float],
    objective: OptimizationObjective | None,
) -> float | None:
    if objective is None:
        return None
    expected = expected_gain.get(objective.primary_metric)
    required = objective.required_delta()
    if expected is None or required is None or required <= 0:
        return 0.0
    normalized = expected / 100.0 if abs(expected) >= 0.1 else expected
    return round(normalized / required, 6)


def _regression_risk(value: float | None, allowed: float) -> float:
    if value is None or value <= allowed:
        return 0.0
    return round((value - allowed) * 5.0, 6)


def _percent_to_ratio(value: float | None) -> float | None:
    return None if value is None else value / 100.0


def _objective_blockers(
    proposal: CandidatePolicy,
    cost: UtilityCost,
    objective: OptimizationObjective | None,
) -> list[str]:
    if objective is None:
        return []
    blockers: list[str] = []
    latency_ratio = _constraint_float(proposal.constraints, "estimated_latency_regression")
    size_ratio = _constraint_float(proposal.constraints, "estimated_model_size_regression")
    if latency_ratio is not None and latency_ratio > objective.max_latency_regression:
        blockers.append("estimated_latency_regression_exceeds_objective")
    if size_ratio is not None and size_ratio > objective.max_model_size_regression:
        blockers.append("estimated_model_size_regression_exceeds_objective")
    if cost.gpu_hours > objective.max_gpu_hours:
        blockers.append("estimated_gpu_hours_exceed_objective_budget")
    return blockers


def _target_actions(proposal: CandidatePolicy) -> list[str]:
    actions: list[str] = []
    if proposal.action_id:
        actions.append(proposal.action_id)
    for key in ("target_actions", "target_error_actions", "action_candidates"):
        value = proposal.train_overrides.get(key)
        if isinstance(value, list):
            actions.extend(str(item) for item in value)
        if isinstance(value, str) and value.strip():
            actions.extend(part.strip() for part in value.split(",") if part.strip())
    for component in proposal.components:
        actions.extend(_component_target_actions(component))
    return list(dict.fromkeys(actions))


def _component_target_actions(component_id: str) -> list[str]:
    mapping: dict[str, list[str]] = {
        "loss.bbox.nwd": ["small_object_recipe", "bbox_loss_recipe"],
        "loss.bbox.wiou": ["bbox_loss_recipe", "label_box_audit"],
        "loss.bbox.mpdiou": ["bbox_loss_recipe", "assigner_recipe"],
        "assigner.stal": ["assigner_recipe", "increase_recall_recipe"],
        "head.p2_small_object": ["small_object_recipe"],
    }
    return mapping.get(component_id, [])


def _constraint_float(constraints: list[PolicyConstraint], name: str) -> float | None:
    for constraint in constraints:
        if constraint.name == name:
            return _float_or_none(constraint.value)
    return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
