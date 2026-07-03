"""Loop-level policy proposal evaluator.

This module keeps the boundary explicit:

PolicyProposal -> LoopPolicyEvaluation -> CandidateConfig -> ExperimentNode

LLMs, humans, and rule engines may create proposals, but only this evaluator can
turn accepted proposals into candidate configs and reproducible experiment nodes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.strategy_policy import CandidatePolicy, PolicyConstraint, PolicyEvaluator
from yolo_agent.components.compatibility import RiskLevel
from yolo_agent.components.registry import ComponentRegistry
from yolo_agent.core.evidence_contract import EvidenceGateResult
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.task_spec import TaskSpec


PolicyProposal = CandidatePolicy
LoopPolicyDecision = Literal["accepted", "rejected", "needs_evidence", "split_required", "deferred", "needs_approval"]
LatencyBudgetPolicy = Literal["strict", "warn", "manual_confirm"]
BudgetBucket = Literal["exploration", "exploitation"]


class BudgetPolicy(BaseModel):
    """Round-level experiment budget policy."""

    max_candidates_per_round: int = Field(default=6, ge=1)
    max_high_risk_candidates: int = Field(default=1, ge=0)
    latency_budget_policy: LatencyBudgetPolicy = "manual_confirm"
    latency_warning_ratio: float = Field(default=0.8, ge=0.0)
    exploration_ratio: float = Field(default=0.3, ge=0.0, le=1.0)
    exploitation_ratio: float = Field(default=0.7, ge=0.0, le=1.0)


class BudgetAllocationSummary(BaseModel):
    """Summary of proposal allocation decisions for one round."""

    max_candidates_per_round: int
    max_high_risk_candidates: int
    selected: list[str] = Field(default_factory=list)
    deferred: list[str] = Field(default_factory=list)
    needs_approval: list[str] = Field(default_factory=list)
    exploration_selected: int = 0
    exploitation_selected: int = 0


class LoopPolicyEvaluation(BaseModel):
    """Loop-level evaluation for one policy proposal."""

    policy_id: str
    decision: LoopPolicyDecision
    priority: float = 0.0
    candidate_config: CandidateConfig | None = None
    experiment_node: ExperimentNode | None = None
    split_proposals: list[PolicyProposal] = Field(default_factory=list)
    blocked_by_deployment: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    changed_variables: dict[str, Any] = Field(default_factory=dict)
    budget_bucket: BudgetBucket | None = None
    budget_reason: str = ""
    requires_human_confirmation: bool = False
    rationale: str = ""


class LoopPolicyEvaluationReport(BaseModel):
    """Batch loop-policy evaluation report."""

    evaluations: list[LoopPolicyEvaluation]
    budget_policy: BudgetPolicy = Field(default_factory=BudgetPolicy)
    budget_allocation: BudgetAllocationSummary | None = None

    @property
    def accepted_candidates(self) -> list[CandidateConfig]:
        """Return accepted candidate configs."""
        return [
            evaluation.candidate_config
            for evaluation in self.evaluations
            if evaluation.decision == "accepted" and evaluation.candidate_config is not None
        ]

    @property
    def experiment_nodes(self) -> list[ExperimentNode]:
        """Return planned experiment nodes for accepted candidates."""
        return [
            evaluation.experiment_node
            for evaluation in self.evaluations
            if evaluation.decision == "accepted" and evaluation.experiment_node is not None
        ]


class BudgetAllocator:
    """Allocate eligible proposals into this-round, deferred, or manual-confirm buckets."""

    def __init__(self, policy: BudgetPolicy | None = None) -> None:
        self.policy = policy or BudgetPolicy()

    def allocate(
        self,
        evaluations: list[LoopPolicyEvaluation],
        proposals_by_id: dict[str, PolicyProposal],
        task_spec: TaskSpec,
    ) -> tuple[list[LoopPolicyEvaluation], BudgetAllocationSummary]:
        """Apply round budget constraints to accepted evaluations."""
        selected: list[str] = []
        deferred: list[str] = []
        needs_approval: list[str] = []
        high_risk_selected = 0
        exploration_selected = 0
        exploitation_selected = 0
        bucket_limits = _bucket_limits(evaluations, proposals_by_id, self.policy)
        allocated: list[LoopPolicyEvaluation] = []

        for evaluation in evaluations:
            if evaluation.decision != "accepted":
                allocated.append(evaluation)
                continue
            proposal = proposals_by_id[evaluation.policy_id]
            bucket = _budget_bucket(proposal)
            evaluation = evaluation.model_copy(update={"budget_bucket": bucket})
            approval_reason = _manual_confirmation_reason(proposal, evaluation, task_spec, self.policy, high_risk_selected)
            if approval_reason:
                needs_approval.append(evaluation.policy_id)
                allocated.append(
                    evaluation.model_copy(
                        update={
                            "decision": "needs_approval",
                            "requires_human_confirmation": True,
                            "budget_reason": approval_reason,
                            "warnings": [*evaluation.warnings, approval_reason],
                        }
                    )
                )
                continue
            if len(selected) >= self.policy.max_candidates_per_round:
                deferred.append(evaluation.policy_id)
                allocated.append(
                    evaluation.model_copy(
                        update={
                            "decision": "deferred",
                            "budget_reason": "Round candidate budget exhausted.",
                        }
                    )
                )
                continue
            if bucket == "exploration" and exploration_selected >= bucket_limits["exploration"]:
                deferred.append(evaluation.policy_id)
                allocated.append(
                    evaluation.model_copy(
                        update={
                            "decision": "deferred",
                            "budget_reason": "Exploration budget exhausted for this round.",
                        }
                    )
                )
                continue
            if bucket == "exploitation" and exploitation_selected >= bucket_limits["exploitation"]:
                deferred.append(evaluation.policy_id)
                allocated.append(
                    evaluation.model_copy(
                        update={
                            "decision": "deferred",
                            "budget_reason": "Exploitation budget exhausted for this round.",
                        }
                    )
                )
                continue

            selected.append(evaluation.policy_id)
            if _effective_risk(proposal, evaluation) == "high":
                high_risk_selected += 1
            if bucket == "exploration":
                exploration_selected += 1
            else:
                exploitation_selected += 1
            allocated.append(
                evaluation.model_copy(
                    update={"budget_reason": "Selected within round budget."}
                )
            )

        return allocated, BudgetAllocationSummary(
            max_candidates_per_round=self.policy.max_candidates_per_round,
            max_high_risk_candidates=self.policy.max_high_risk_candidates,
            selected=selected,
            deferred=deferred,
            needs_approval=needs_approval,
            exploration_selected=exploration_selected,
            exploitation_selected=exploitation_selected,
        )


class LoopPolicyEvaluator:
    """Evaluate proposals for ordering, evidence, constraints, and ablation hygiene."""

    def __init__(
        self,
        registry: ComponentRegistry,
        base_evaluator: PolicyEvaluator | None = None,
        budget_policy: BudgetPolicy | None = None,
    ) -> None:
        self.registry = registry
        self.base_evaluator = base_evaluator or PolicyEvaluator(registry)
        self.budget_policy = budget_policy or BudgetPolicy()
        self.budget_allocator = BudgetAllocator(self.budget_policy)

    def evaluate(
        self,
        proposals: list[PolicyProposal],
        task_spec: TaskSpec,
        evidence_gate: EvidenceGateResult | None = None,
        data_version: str = "unversioned",
        seed: int = 42,
        plan_path: Path | str | None = None,
        data_path: Path | str | None = None,
    ) -> LoopPolicyEvaluationReport:
        """Evaluate proposals and return ordered loop decisions."""
        evaluations = [
            self.evaluate_one(
                proposal,
                task_spec,
                evidence_gate,
                data_version,
                seed,
                plan_path=plan_path,
                data_path=data_path,
            )
            for proposal in proposals
        ]
        evaluations.sort(key=lambda evaluation: evaluation.priority, reverse=True)
        allocated, allocation = self.budget_allocator.allocate(
            evaluations,
            {proposal.policy_id: proposal for proposal in proposals},
            task_spec,
        )
        return LoopPolicyEvaluationReport(
            evaluations=allocated,
            budget_policy=self.budget_policy,
            budget_allocation=allocation,
        )

    def evaluate_one(
        self,
        proposal: PolicyProposal,
        task_spec: TaskSpec,
        evidence_gate: EvidenceGateResult | None = None,
        data_version: str = "unversioned",
        seed: int = 42,
        plan_path: Path | str | None = None,
        data_path: Path | str | None = None,
    ) -> LoopPolicyEvaluation:
        """Evaluate one policy proposal."""
        changed_variables = infer_changed_variables(proposal)
        priority = _priority(proposal, changed_variables)
        split_proposals = split_policy_proposal(proposal, changed_variables)
        if len(changed_variables) > 1:
            return LoopPolicyEvaluation(
                policy_id=proposal.policy_id,
                decision="split_required",
                priority=priority,
                split_proposals=split_proposals,
                changed_variables=changed_variables,
                warnings=["Policy changes multiple primary variables and must be split before ablation."],
                rationale=proposal.rationale,
            )

        deployment_errors = _deployment_errors(proposal, task_spec)
        if deployment_errors:
            return LoopPolicyEvaluation(
                policy_id=proposal.policy_id,
                decision="rejected",
                priority=priority,
                blocked_by_deployment=deployment_errors,
                errors=deployment_errors,
                changed_variables=changed_variables,
                rationale=proposal.rationale,
            )

        missing_evidence = _missing_evidence(proposal, evidence_gate)
        if missing_evidence:
            return LoopPolicyEvaluation(
                policy_id=proposal.policy_id,
                decision="needs_evidence",
                priority=priority,
                missing_evidence=missing_evidence,
                changed_variables=changed_variables,
                warnings=[f"Missing required evidence: {', '.join(missing_evidence)}"],
                rationale=proposal.rationale,
            )

        base = self.base_evaluator.evaluate_one(proposal, task_spec)
        if not base.accepted or base.candidate_config is None:
            return LoopPolicyEvaluation(
                policy_id=proposal.policy_id,
                decision="rejected",
                priority=priority,
                errors=base.errors,
                warnings=base.warnings,
                changed_variables=changed_variables,
                rationale=proposal.rationale,
            )

        experiment_node = ExperimentNode(
            node_id=f"node_{base.candidate_config.candidate_id}",
            candidate_config=base.candidate_config,
            data_version=data_version,
            seed=seed,
            command=_command_for_candidate(base.candidate_config, plan_path=plan_path, data_path=data_path),
            status="planned",
            changed_variables=changed_variables,
        )
        return LoopPolicyEvaluation(
            policy_id=proposal.policy_id,
            decision="accepted",
            priority=priority + base.score,
            candidate_config=base.candidate_config,
            experiment_node=experiment_node,
            warnings=base.warnings,
            changed_variables=changed_variables,
            rationale=proposal.rationale,
        )


def infer_changed_variables(proposal: PolicyProposal) -> dict[str, Any]:
    """Infer primary ablation variables changed by a policy proposal."""
    changed: dict[str, Any] = {}
    component_groups = {
        "bbox_loss": "loss.bbox.",
        "head_component": "head.",
        "assigner": "assigner.",
        "neck_component": "neck.",
        "augmentation_policy": "augmentation.",
    }
    for variable, prefix in component_groups.items():
        values = [component for component in proposal.components if component.startswith(prefix)]
        if values:
            changed[variable] = values

    if "imgsz" in proposal.train_overrides:
        changed["imgsz"] = proposal.train_overrides["imgsz"]
    if "augmentation_policy" in proposal.train_overrides:
        changed["augmentation_policy"] = proposal.train_overrides["augmentation_policy"]
    if "postprocess" in proposal.train_overrides:
        changed["postprocess"] = proposal.train_overrides["postprocess"]
    if proposal.scale not in {"", "baseline"} and proposal.scale != "n":
        changed["model_scale"] = proposal.scale
    return changed


def split_policy_proposal(
    proposal: PolicyProposal,
    changed_variables: dict[str, Any] | None = None,
) -> list[PolicyProposal]:
    """Split a multi-variable proposal into single-variable proposals."""
    changes = changed_variables or infer_changed_variables(proposal)
    if len(changes) <= 1:
        return []
    split: list[PolicyProposal] = []
    for variable, value in changes.items():
        split.append(
            PolicyProposal(
                policy_id=f"{proposal.policy_id}_{variable}",
                source=proposal.source,
                base_model=proposal.base_model,
                scale=proposal.scale if variable == "model_scale" else "n",
                framework=proposal.framework,
                components=_components_for_variable(proposal.components, variable),
                train_overrides=_train_overrides_for_variable(proposal.train_overrides, variable),
                constraints=proposal.constraints,
                evidence_required=proposal.evidence_required,
                priority_hint=proposal.priority_hint,
                expected_effect=proposal.expected_effect,
                risk=proposal.risk,
                rationale=f"Split from {proposal.policy_id}; single variable={variable}; value={value}.",
            )
        )
    return split


def _components_for_variable(components: list[str], variable: str) -> list[str]:
    prefixes = {
        "bbox_loss": ("loss.bbox.",),
        "head_component": ("head.",),
        "assigner": ("assigner.",),
        "neck_component": ("neck.",),
        "augmentation_policy": ("augmentation.",),
    }.get(variable, ())
    return [component for component in components if component.startswith(prefixes)]


def _train_overrides_for_variable(train_overrides: dict[str, Any], variable: str) -> dict[str, Any]:
    keys = {
        "imgsz": ["imgsz"],
        "augmentation_policy": ["augmentation_policy"],
        "postprocess": ["postprocess"],
    }.get(variable, [])
    return {key: train_overrides[key] for key in keys if key in train_overrides}


def _deployment_errors(proposal: PolicyProposal, task_spec: TaskSpec) -> list[str]:
    errors: list[str] = []
    for constraint in proposal.constraints:
        if constraint.name == "estimated_latency_ms":
            max_latency = task_spec.max_latency_ms or _constraint_value(proposal.constraints, "max_latency_ms")
            if max_latency is not None and float(constraint.value) > float(max_latency):
                errors.append(f"estimated_latency_ms={constraint.value} exceeds max_latency_ms={max_latency}.")
        if constraint.name == "estimated_model_size_mb":
            max_size = task_spec.max_model_size_mb or _constraint_value(proposal.constraints, "max_model_size_mb")
            if max_size is not None and float(constraint.value) > float(max_size):
                errors.append(f"estimated_model_size_mb={constraint.value} exceeds max_model_size_mb={max_size}.")
        if constraint.name in {"max_latency_ms", "max_model_size_mb"} and constraint.hard:
            task_value = getattr(task_spec, constraint.name, None)
            if task_value is not None and float(constraint.value) > float(task_value):
                errors.append(f"{constraint.name}={constraint.value} exceeds task {constraint.name}={task_value}.")
    return errors


def _missing_evidence(proposal: PolicyProposal, gate: EvidenceGateResult | None) -> list[str]:
    if not proposal.evidence_required:
        return []
    if gate is None:
        return proposal.evidence_required
    missing = set(gate.missing_required)
    return [requirement for requirement in proposal.evidence_required if requirement in missing]


def _priority(proposal: PolicyProposal, changed_variables: dict[str, Any]) -> float:
    source_bonus = {"rule_engine": 0.4, "human": 0.3, "llm": 0.1}[proposal.source]
    risk_penalty = {"low": 0.0, "medium": 0.2, "high": 0.5}[proposal.risk]
    single_variable_bonus = 0.3 if len(changed_variables) == 1 else 0.0
    evidence_penalty = min(0.4, len(proposal.evidence_required) * 0.05)
    return max(0.0, proposal.priority_hint + source_bonus + single_variable_bonus - risk_penalty - evidence_penalty)


def _budget_bucket(proposal: PolicyProposal) -> BudgetBucket:
    if proposal.source == "llm" or proposal.risk == "high" or proposal.priority_hint < 1.0:
        return "exploration"
    return "exploitation"


def _bucket_limits(
    evaluations: list[LoopPolicyEvaluation],
    proposals_by_id: dict[str, PolicyProposal],
    budget: BudgetPolicy,
) -> dict[BudgetBucket, int]:
    eligible = [
        _budget_bucket(proposals_by_id[evaluation.policy_id])
        for evaluation in evaluations
        if evaluation.decision == "accepted"
    ]
    counts = {
        "exploration": eligible.count("exploration"),
        "exploitation": eligible.count("exploitation"),
    }
    total_ratio = budget.exploration_ratio + budget.exploitation_ratio
    exploration_share = budget.exploration_ratio / total_ratio if total_ratio > 0 else 0.0
    exploration_limit = min(
        counts["exploration"],
        round(budget.max_candidates_per_round * exploration_share),
    )
    exploitation_limit = min(
        counts["exploitation"],
        budget.max_candidates_per_round - exploration_limit,
    )
    unused = budget.max_candidates_per_round - exploration_limit - exploitation_limit
    if unused > 0:
        extra_exploration = min(unused, counts["exploration"] - exploration_limit)
        exploration_limit += extra_exploration
        unused -= extra_exploration
    if unused > 0:
        exploitation_limit += min(unused, counts["exploitation"] - exploitation_limit)
    return {"exploration": exploration_limit, "exploitation": exploitation_limit}


def _manual_confirmation_reason(
    proposal: PolicyProposal,
    evaluation: LoopPolicyEvaluation,
    task_spec: TaskSpec,
    budget: BudgetPolicy,
    high_risk_selected: int,
) -> str:
    candidate_risk = _effective_risk(proposal, evaluation)
    if candidate_risk == "high" and high_risk_selected >= budget.max_high_risk_candidates:
        return "High-risk candidate budget exhausted; human confirmation required."
    estimated_latency = _constraint_value(proposal.constraints, "estimated_latency_ms")
    if estimated_latency is None or task_spec.max_latency_ms is None:
        return ""
    latency = float(estimated_latency)
    max_latency = float(task_spec.max_latency_ms)
    if latency > max_latency and budget.latency_budget_policy == "manual_confirm":
        return f"estimated_latency_ms={latency} exceeds max_latency_ms={max_latency}; human confirmation required."
    if latency >= max_latency * budget.latency_warning_ratio and budget.latency_budget_policy == "manual_confirm":
        return f"estimated_latency_ms={latency} is near max_latency_ms={max_latency}; human confirmation required."
    return ""


def _effective_risk(proposal: PolicyProposal, evaluation: LoopPolicyEvaluation) -> RiskLevel:
    candidate_risk: RiskLevel = evaluation.candidate_config.risk if evaluation.candidate_config is not None else proposal.risk
    order = {"low": 0, "medium": 1, "high": 2}
    return proposal.risk if order[proposal.risk] >= order[candidate_risk] else candidate_risk


def _constraint_value(constraints: list[PolicyConstraint], name: str) -> Any:
    for constraint in constraints:
        if constraint.name == name:
            return constraint.value
    return None


def _command_for_candidate(
    candidate: CandidateConfig,
    plan_path: Path | str | None = None,
    data_path: Path | str | None = None,
) -> str:
    plan_arg = _cli_path_arg(plan_path or Path("runs") / "plan.yaml")
    data_arg = _cli_path_arg(data_path or Path("data.yaml"))
    run_id = _cli_arg(f"smoke_{candidate.candidate_id}")
    return f"yolo-agent smoke --plan {plan_arg} --data {data_arg} --run-id {run_id}"


def _cli_path_arg(path: Path | str) -> str:
    return _cli_arg(Path(path).as_posix())


def _cli_arg(value: object) -> str:
    text = str(value)
    if any(character.isspace() for character in text):
        return '"' + text.replace('"', '\\"') + '"'
    return text
