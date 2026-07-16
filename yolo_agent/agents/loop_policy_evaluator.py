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

from yolo_agent.adapters.ultralytics.baseline_acceptance import BaselineAcceptanceResult
from yolo_agent.adapters.ultralytics.candidate_promotion import CandidatePromotionResult
from yolo_agent.adapters.ultralytics.training import UltralyticsTrainingConfig, command_from_training_config
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.strategy_policy import CandidatePolicy, PolicyConstraint, PolicyEvaluator
from yolo_agent.agents.utility_scorer import UtilityScore, UtilityScorer
from yolo_agent.components.compatibility import RiskLevel
from yolo_agent.components.registry import ComponentRegistry
from yolo_agent.components.yolo26_compatibility import YOLO26CompatibilityChecker
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.evidence_contract import EvidenceGateResult
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.core.optimization_objective import OptimizationObjective
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.policy_variables import PolicyVariableClassification, classify_policy_variables
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
    utility_score: UtilityScore | None = None
    candidate_config: CandidateConfig | None = None
    experiment_node: ExperimentNode | None = None
    split_proposals: list[PolicyProposal] = Field(default_factory=list)
    blocked_by_deployment: list[str] = Field(default_factory=list)
    evidence_required: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    fixed_variables: dict[str, Any] = Field(default_factory=dict)
    effective_overrides: dict[str, Any] = Field(default_factory=dict)
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
        fixed_imgsz: int | None = None,
        utility_scorer: UtilityScorer | None = None,
        optimization_objective: OptimizationObjective | None = None,
    ) -> None:
        self.registry = registry
        self.base_evaluator = base_evaluator or PolicyEvaluator(registry)
        self.budget_policy = budget_policy or BudgetPolicy()
        self.budget_allocator = BudgetAllocator(self.budget_policy)
        self.fixed_imgsz = fixed_imgsz
        self.utility_scorer = utility_scorer or UtilityScorer()
        self.optimization_objective = optimization_objective

    def evaluate(
        self,
        proposals: list[PolicyProposal],
        task_spec: TaskSpec,
        evidence_gate: EvidenceGateResult | None = None,
        data_version: str = "unversioned",
        seed: int = 42,
        plan_path: Path | str | None = None,
        data_path: Path | str | None = None,
        run_id: str | None = None,
        training_config: UltralyticsTrainingConfig | None = None,
        baseline_acceptance: BaselineAcceptanceResult | None = None,
        candidate_promotions: dict[str, CandidatePromotionResult] | None = None,
        error_facts: list[ErrorFact] | None = None,
        proposal_mode: str | None = None,
        allowed_training_profiles: list[str] | None = None,
        required_proposal_bindings: list[str] | None = None,
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
                run_id=run_id,
                training_config=training_config,
                baseline_acceptance=baseline_acceptance,
                candidate_promotions=candidate_promotions,
                error_facts=error_facts,
                proposal_mode=proposal_mode,
                allowed_training_profiles=allowed_training_profiles,
                required_proposal_bindings=required_proposal_bindings,
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
        run_id: str | None = None,
        training_config: UltralyticsTrainingConfig | None = None,
        baseline_acceptance: BaselineAcceptanceResult | None = None,
        candidate_promotions: dict[str, CandidatePromotionResult] | None = None,
        error_facts: list[ErrorFact] | None = None,
        proposal_mode: str | None = None,
        allowed_training_profiles: list[str] | None = None,
        required_proposal_bindings: list[str] | None = None,
    ) -> LoopPolicyEvaluation:
        """Evaluate one policy proposal."""
        effective_fixed_imgsz = (
            self.fixed_imgsz
            if self.fixed_imgsz is not None
            else self.optimization_objective.fixed_imgsz
            if self.optimization_objective is not None
            else training_config.imgsz
            if training_config is not None
            else None
        )
        variable_classification = classify_loop_policy_variables(
            proposal,
            fixed_imgsz=effective_fixed_imgsz,
        )
        changed_variables = variable_classification.changed_variables
        missing_evidence = _missing_evidence(proposal, evidence_gate)
        utility_score = self.utility_scorer.score(
            proposal=proposal,
            task_spec=task_spec,
            changed_variables=changed_variables,
            missing_evidence=missing_evidence,
            error_facts=error_facts,
            training_config=training_config,
            optimization_objective=self.optimization_objective,
        )
        priority = utility_score.utility
        split_proposals = split_policy_proposal(proposal, changed_variables)

        proposal_contract_errors = _proposal_contract_errors(
            proposal,
            training_config,
            proposal_mode=proposal_mode,
            allowed_training_profiles=allowed_training_profiles,
            required_proposal_bindings=required_proposal_bindings,
        )
        training_profile_errors = _training_profile_errors(training_config)
        if proposal_contract_errors or training_profile_errors:
            return LoopPolicyEvaluation(
                policy_id=proposal.policy_id,
                decision="rejected",
                priority=priority,
                utility_score=utility_score,
                evidence_required=list(proposal.evidence_required),
                errors=[*proposal_contract_errors, *training_profile_errors],
                fixed_variables=variable_classification.fixed_variables,
                effective_overrides=variable_classification.effective_overrides,
                changed_variables=changed_variables,
                rationale=proposal.rationale,
            )

        imgsz_errors = _imgsz_guard_errors(proposal, fixed_imgsz=effective_fixed_imgsz)
        if imgsz_errors:
            return LoopPolicyEvaluation(
                policy_id=proposal.policy_id,
                decision="rejected",
                priority=priority,
                utility_score=utility_score,
                evidence_required=list(proposal.evidence_required),
                errors=imgsz_errors,
                fixed_variables=variable_classification.fixed_variables,
                effective_overrides=variable_classification.effective_overrides,
                changed_variables=changed_variables,
                rationale=proposal.rationale,
            )

        if "yolo26" in proposal.base_model.lower():
            proposal_components = [
                card for component_id in proposal.components
                if (card := next((item for item in self.registry.cards if item.id == component_id), None)) is not None
            ]
            yolo26 = YOLO26CompatibilityChecker().check(
                components=proposal_components,
                train_overrides=proposal.train_overrides,
                changed_variables=changed_variables,
                single_variable=bool(_constraint_value(proposal.constraints, "single_variable")),
                export_format=str(_constraint_value(proposal.constraints, "export_format") or "none"),
                execution_requested=proposal.execution_action == "run_training",
            )
            if yolo26.incompatible:
                return LoopPolicyEvaluation(
                    policy_id=proposal.policy_id,
                    decision="rejected",
                    priority=priority,
                    utility_score=utility_score,
                    evidence_required=list(proposal.evidence_required),
                    errors=yolo26.blocked_by,
                    warnings=yolo26.warnings,
                    fixed_variables=variable_classification.fixed_variables,
                    effective_overrides=variable_classification.effective_overrides,
                    changed_variables=changed_variables,
                    rationale=proposal.rationale,
                )

        if len(changed_variables) > 1:
            return LoopPolicyEvaluation(
                policy_id=proposal.policy_id,
                decision="split_required",
                priority=priority,
                utility_score=utility_score,
                split_proposals=split_proposals,
                evidence_required=list(proposal.evidence_required),
                fixed_variables=variable_classification.fixed_variables,
                effective_overrides=variable_classification.effective_overrides,
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
                utility_score=utility_score,
                blocked_by_deployment=deployment_errors,
                evidence_required=list(proposal.evidence_required),
                errors=deployment_errors,
                fixed_variables=variable_classification.fixed_variables,
                effective_overrides=variable_classification.effective_overrides,
                changed_variables=changed_variables,
                rationale=proposal.rationale,
            )

        if missing_evidence:
            return LoopPolicyEvaluation(
                policy_id=proposal.policy_id,
                decision="needs_evidence",
                priority=priority,
                utility_score=utility_score,
                missing_evidence=missing_evidence,
                evidence_required=list(proposal.evidence_required),
                fixed_variables=variable_classification.fixed_variables,
                effective_overrides=variable_classification.effective_overrides,
                changed_variables=changed_variables,
                warnings=[f"Missing required evidence: {', '.join(missing_evidence)}"],
                rationale=proposal.rationale,
            )

        baseline_blockers = _candidate_full_baseline_blockers(training_config, baseline_acceptance)
        promotion_blockers = _candidate_full_promotion_blockers(
            proposal,
            training_config,
            candidate_promotions,
        )
        error_delta_blockers = _candidate_full_error_delta_blockers(
            proposal,
            training_config,
            error_facts,
        )
        if baseline_blockers or promotion_blockers or error_delta_blockers:
            missing = []
            if baseline_blockers:
                missing.append("baseline_trusted")
            if promotion_blockers:
                missing.append("candidate_full_allowed")
            if error_delta_blockers:
                missing.append("error_facts")
            evidence_required = list(dict.fromkeys([*proposal.evidence_required, *missing]))
            return LoopPolicyEvaluation(
                policy_id=proposal.policy_id,
                decision="needs_evidence",
                priority=priority,
                utility_score=utility_score,
                missing_evidence=missing,
                evidence_required=evidence_required,
                fixed_variables=variable_classification.fixed_variables,
                effective_overrides=variable_classification.effective_overrides,
                changed_variables=changed_variables,
                warnings=[
                    *(
                        ["candidate_full is blocked until COCO baseline acceptance passes."]
                        if baseline_blockers
                        else []
                    ),
                    *baseline_blockers,
                    *(
                        ["candidate_full is blocked until candidate pilot promotion passes."]
                        if promotion_blockers
                        else []
                    ),
                    *promotion_blockers,
                    *(
                        ["candidate_full is blocked until targeted COCO error facts exist."]
                        if error_delta_blockers
                        else []
                    ),
                    *error_delta_blockers,
                ],
                rationale=proposal.rationale,
            )

        enforce_utility_gate = proposal_mode == "pilot_only"
        if enforce_utility_gate and utility_score.decision == "reject":
            return LoopPolicyEvaluation(
                policy_id=proposal.policy_id,
                decision="rejected",
                priority=priority,
                utility_score=utility_score,
                evidence_required=list(proposal.evidence_required),
                errors=["utility_score rejected this proposal; do not enqueue for training."],
                fixed_variables=variable_classification.fixed_variables,
                effective_overrides=variable_classification.effective_overrides,
                changed_variables=changed_variables,
                rationale=proposal.rationale,
            )

        if enforce_utility_gate and utility_score.decision == "defer":
            return LoopPolicyEvaluation(
                policy_id=proposal.policy_id,
                decision="deferred",
                priority=priority,
                utility_score=utility_score,
                evidence_required=list(proposal.evidence_required),
                warnings=["utility_score deferred this proposal; collect stronger evidence or try higher-utility actions first."],
                fixed_variables=variable_classification.fixed_variables,
                effective_overrides=variable_classification.effective_overrides,
                changed_variables=changed_variables,
                rationale=proposal.rationale,
            )

        base = self.base_evaluator.evaluate_one(proposal, task_spec)
        if not base.accepted or base.candidate_config is None:
            return LoopPolicyEvaluation(
                policy_id=proposal.policy_id,
                decision="rejected",
                priority=priority,
                utility_score=utility_score,
                errors=base.errors,
                evidence_required=list(proposal.evidence_required),
                warnings=base.warnings,
                fixed_variables=variable_classification.fixed_variables,
                effective_overrides=variable_classification.effective_overrides,
                changed_variables=changed_variables,
                rationale=proposal.rationale,
            )

        experiment_node = ExperimentNode(
            node_id=f"node_{base.candidate_config.candidate_id}",
            candidate_config=base.candidate_config,
            data_version=data_version,
            seed=seed,
            status="planned",
            fixed_variables=variable_classification.fixed_variables,
            effective_overrides=variable_classification.effective_overrides,
            changed_variables=changed_variables,
        )
        command_spec = _command_for_candidate(
            experiment_node,
            plan_path=plan_path,
            data_path=data_path,
            run_id=run_id,
            training_config=training_config,
        )
        if self.optimization_objective is not None:
            command_spec = command_spec.model_copy(
                update={
                    "metadata": {
                        **command_spec.metadata,
                        "optimization_objective_hash": self.optimization_objective.objective_hash,
                        "baseline_protocol_hash": self.optimization_objective.baseline_protocol_hash,
                        "optimization_primary_metric": self.optimization_objective.primary_metric,
                        "optimization_target_delta": self.optimization_objective.required_delta(),
                    }
                }
            )
        experiment_node.command = command_spec.display()
        experiment_node.command_spec = command_spec
        return LoopPolicyEvaluation(
            policy_id=proposal.policy_id,
            decision="accepted",
            priority=priority,
            utility_score=utility_score,
            candidate_config=base.candidate_config,
            experiment_node=experiment_node,
            evidence_required=list(proposal.evidence_required),
            warnings=base.warnings,
            fixed_variables=variable_classification.fixed_variables,
            effective_overrides=variable_classification.effective_overrides,
            changed_variables=changed_variables,
            rationale=proposal.rationale,
        )


def classify_loop_policy_variables(
    proposal: PolicyProposal,
    *,
    fixed_imgsz: int | None = None,
) -> PolicyVariableClassification:
    """Classify loop variables against the effective baseline protocol."""
    declared_fixed = dict(proposal.fixed_variables)
    constraint_imgsz = _constraint_value(proposal.constraints, "fixed_imgsz")
    if constraint_imgsz is not None:
        declared_fixed.setdefault("imgsz", constraint_imgsz)
    return classify_policy_variables(
        components=proposal.components,
        train_overrides=proposal.train_overrides,
        action_domain=proposal.action_domain,
        action_id=proposal.action_id,
        scale=proposal.scale,
        declared_fixed_variables=declared_fixed,
        baseline_protocol={"imgsz": fixed_imgsz},
    )


def infer_changed_variables(
    proposal: PolicyProposal,
    *,
    fixed_imgsz: int | None = None,
) -> dict[str, Any]:
    """Infer true ablation changes while excluding fixed protocol values."""
    return classify_loop_policy_variables(proposal, fixed_imgsz=fixed_imgsz).changed_variables


def split_policy_proposal(
    proposal: PolicyProposal,
    changed_variables: dict[str, Any] | None = None,
) -> list[PolicyProposal]:
    """Split a multi-variable proposal into single-variable proposals."""
    changes = changed_variables or infer_changed_variables(proposal)
    if len(changes) <= 1:
        return []
    split: list[PolicyProposal] = []
    fixed_override_keys = set(proposal.fixed_variables)
    if _constraint_value(proposal.constraints, "fixed_imgsz") is not None:
        fixed_override_keys.add("imgsz")
    for variable, value in changes.items():
        fixed_overrides = {
            key: proposal.train_overrides[key]
            for key in fixed_override_keys
            if key in proposal.train_overrides
        }
        split.append(
            PolicyProposal(
                policy_id=f"{proposal.policy_id}_{variable}",
                source=proposal.source,
                action_domain=proposal.action_domain,
                action_id=proposal.action_id,
                execution_action=proposal.execution_action,
                base_model=proposal.base_model,
                scale=proposal.scale if variable == "model_scale" else "n",
                framework=proposal.framework,
                components=_components_for_variable(proposal.components, variable),
                train_overrides={
                    **fixed_overrides,
                    **_train_overrides_for_variable(proposal.train_overrides, variable),
                },
                fixed_variables=proposal.fixed_variables,
                constraints=proposal.constraints,
                evidence_required=proposal.evidence_required,
                target_error_facts=proposal.target_error_facts,
                expected_improvement=proposal.expected_improvement,
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
        "data_action": ["data_action", "sampling_target", "sampling_parameters"],
        "label_action": ["label_action"],
        "training_action": ["training_action", "focal_loss_gamma"],
        "postprocess_action": ["postprocess_action", "inference_tiling"],
        "augmentation_action": ["augmentation_action", "mosaic"],
        "evidence_action": ["evidence_action", "missing_evidence"],
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


def _imgsz_guard_errors(proposal: PolicyProposal, fixed_imgsz: int | None) -> list[str]:
    """Return hard guard errors for unfair input-size increases."""
    if fixed_imgsz is None or "imgsz" not in proposal.train_overrides:
        return []
    value = proposal.train_overrides["imgsz"]
    try:
        requested = int(value)
    except (TypeError, ValueError):
        return [
            "imgsz changes are blocked for COCO/YOLO26 baseline comparability unless they are an explicit numeric value "
            f"<= fixed_imgsz={fixed_imgsz}."
        ]
    if requested > fixed_imgsz:
        return [
            f"imgsz increase is blocked for COCO/YOLO26 baseline comparability: requested imgsz={requested} "
            f"> fixed_imgsz={fixed_imgsz}."
        ]
    return []


def _missing_evidence(proposal: PolicyProposal, gate: EvidenceGateResult | None) -> list[str]:
    if not proposal.evidence_required:
        return []
    if gate is None:
        return proposal.evidence_required
    missing = set(gate.missing_required)
    return [requirement for requirement in proposal.evidence_required if requirement in missing]


def _proposal_contract_errors(
    proposal: PolicyProposal,
    training_config: UltralyticsTrainingConfig | None,
    proposal_mode: str | None,
    allowed_training_profiles: list[str] | None,
    required_proposal_bindings: list[str] | None,
) -> list[str]:
    """Return proposal-mode contract violations before candidate creation."""
    if proposal.action_domain == "evidence":
        return []
    if proposal_mode != "pilot_only":
        return []
    errors: list[str] = []
    profile = training_config.budget_profile if training_config is not None else None
    allowed = set(allowed_training_profiles or ["debug", "pilot"])
    if profile == "candidate_full":
        errors.append("candidate_full_blocked_by_pilot_only_proposal_mode")
    elif profile is not None and allowed and profile not in allowed:
        errors.append(f"training_profile_not_allowed_in_pilot_only_mode:{profile}")
    required = set(required_proposal_bindings or ["target_error_facts", "expected_improvement"])
    if "target_error_facts" in required and not proposal.target_error_facts:
        errors.append("missing_target_error_facts_binding")
    if "expected_improvement" in required and not proposal.expected_improvement:
        errors.append("missing_expected_improvement")
    return errors


def _training_profile_errors(training_config: UltralyticsTrainingConfig | None) -> list[str]:
    """Return hard training-budget protocol violations."""
    if training_config is None or training_config.budget_profile != "candidate_full":
        return []
    profile = training_config.selected_budget_profile()
    seed_count = len(set(profile.seeds))
    errors: list[str] = []
    if seed_count < 3:
        errors.append(f"candidate_full_requires_3_seeds:{seed_count}/3")
    if not profile.confirms_contribution:
        errors.append("candidate_full_must_confirm_contribution")
    return errors


def _candidate_full_baseline_blockers(
    training_config: UltralyticsTrainingConfig | None,
    baseline_acceptance: BaselineAcceptanceResult | None,
) -> list[str]:
    """Return blockers that prevent full candidate runs before trusted baseline evidence."""
    if training_config is None or training_config.budget_profile != "candidate_full":
        return []
    if baseline_acceptance is None:
        return ["baseline_acceptance_not_evaluated"]
    if baseline_acceptance.baseline_trusted:
        return []
    return baseline_acceptance.baseline_rejection_reason or ["baseline_trusted_false"]


def _candidate_full_promotion_blockers(
    proposal: PolicyProposal,
    training_config: UltralyticsTrainingConfig | None,
    candidate_promotions: dict[str, CandidatePromotionResult] | None,
) -> list[str]:
    """Return blockers that prevent full candidate promotion after pilot."""
    if training_config is None or training_config.budget_profile != "candidate_full":
        return []
    if not training_config.selected_budget_profile().requires_pilot_pass:
        return []
    if candidate_promotions is None:
        return ["candidate_promotion_not_evaluated"]
    promotion = candidate_promotions.get(proposal.policy_id)
    if promotion is None:
        return ["missing_candidate_promotion_decision"]
    if promotion.candidate_full_allowed:
        return []
    return promotion.candidate_promotion_rejection_reason or ["candidate_full_allowed_false"]


def _candidate_full_error_delta_blockers(
    proposal: PolicyProposal,
    training_config: UltralyticsTrainingConfig | None,
    error_facts: list[ErrorFact] | None,
) -> list[str]:
    """Return blockers that prevent full candidates without targeted error facts."""
    if training_config is None or training_config.budget_profile != "candidate_full":
        return []
    if not proposal.target_error_facts:
        return ["missing_target_error_facts_binding"]
    if not proposal.expected_improvement:
        return ["missing_expected_improvement"]
    if error_facts is None:
        return ["error_delta_gate_not_evaluated"]
    if not error_facts:
        return ["missing_error_facts"]
    target_actions = _target_actions(proposal)
    if not target_actions:
        return ["missing_target_error_actions"]
    actionable = [
        fact
        for fact in error_facts
        if fact.severity in {"high", "medium"}
        and bool(set(fact.action_candidates) & set(target_actions))
    ]
    if actionable:
        return []
    return [f"missing_target_error_facts:{','.join(target_actions)}"]


def _target_actions(proposal: PolicyProposal) -> list[str]:
    """Return action tags this proposal claims to address."""
    for key in ("target_actions", "target_error_actions", "action_candidates"):
        value = proposal.train_overrides.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str) and value.strip():
            return [part.strip() for part in value.split(",") if part.strip()]
    actions: list[str] = []
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
    node: ExperimentNode,
    plan_path: Path | str | None = None,
    data_path: Path | str | None = None,
    run_id: str | None = None,
    training_config: UltralyticsTrainingConfig | None = None,
) -> CommandSpec:
    candidate = node.candidate_config
    if candidate.action_domain == "evidence":
        return _command_for_evidence_action(
            node=node,
            plan_path=plan_path,
            data_path=data_path,
            run_id=run_id,
        )
    if training_config is not None:
        return command_from_training_config(
            node=node,
            config=training_config,
            run_id=run_id or "loop",
            data_path=data_path,
        )
    return CommandSpec.smoke(
        plan_path=plan_path or Path("runs") / "plan.yaml",
        data_path=data_path or Path("data.yaml"),
        run_id=f"smoke_{candidate.candidate_id}",
        expected_artifacts={
            "smoke_result": Path("artifacts") / "smoke_result.json",
            "generated_models": Path("artifacts") / "generated_models",
        },
        metadata={
            "run_id": run_id or f"smoke_{candidate.candidate_id}",
            "node_id": node.node_id,
            "candidate_id": candidate.candidate_id,
            "dataset_version": node.data_version,
            "seed": node.seed,
            "framework": candidate.framework,
        },
    )


def _command_for_evidence_action(
    node: ExperimentNode,
    plan_path: Path | str | None = None,
    data_path: Path | str | None = None,
    run_id: str | None = None,
) -> CommandSpec:
    """Build a typed non-training command for evidence acquisition actions."""
    candidate = node.candidate_config
    action = candidate.execution_action
    run_dir = _run_dir_from_plan(plan_path, run_id)
    data = Path(data_path or "data.yaml").as_posix()
    metadata = {
        "run_id": run_id or "loop",
        "node_id": node.node_id,
        "candidate_id": candidate.candidate_id,
        "dataset_version": node.data_version,
        "seed": node.seed,
        "action_domain": candidate.action_domain,
        "action_id": candidate.action_id or "",
        "execution_action": action,
    }
    if action == "profile_data":
        out = run_dir / "artifacts" / "dataset_report"
        argv = ["yolo-agent", "profile-data", "--data", data, "--out", out.as_posix()]
        return CommandSpec(
            command_type="profile_data",
            argv=argv,
            expected_artifacts={
                "dataset_report": out.with_suffix(".json"),
                "dataset_report_md": out.with_suffix(".md"),
            },
            expected_metrics=["dataset_health_score"],
            metadata=metadata,
        )
    if action == "advise_labels":
        out = run_dir / "artifacts" / "annotation_advice"
        argv = ["yolo-agent", "advise-labels", "--data", data, "--out", out.as_posix()]
        return CommandSpec(
            command_type="advise_labels",
            argv=argv,
            expected_artifacts={
                "label_quality_report": out.with_suffix(".json"),
                "annotation_advice_md": out.with_suffix(".md"),
            },
            expected_metrics=["label_quality_score"],
            metadata=metadata,
        )
    if action == "mine_errors":
        out = run_dir / "artifacts" / "coco_error_report"
        argv = [
            "yolo-agent",
            "mine-coco-errors",
            "--gt",
            "<instances_val2017.json>",
            "--predictions",
            "<predictions.json>",
            "--out",
            out.as_posix(),
        ]
        return CommandSpec(
            command_type="mine_errors",
            argv=argv,
            expected_artifacts={
                "coco_error_report": out.with_suffix(".json"),
                "coco_error_report_md": out.with_suffix(".md"),
                "error_observations": out.with_name(out.name + "_errors.yaml"),
            },
            expected_metrics=["false_positive_count", "false_negative_count", "localization_error_rate"],
            metadata=metadata,
        )
    if action == "benchmark_latency":
        argv = ["yolo-agent", "benchmark", "--run", run_dir.as_posix()]
        return CommandSpec(
            command_type="benchmark",
            argv=argv,
            expected_metrics=["latency_ms", "fps", "model_size_mb"],
            metadata=metadata,
        )
    argv = [
        "yolo-agent",
        "loop",
        "ingest-metrics",
        "--run",
        run_dir.as_posix(),
        "--metrics",
        "<metrics.yaml>",
    ]
    return CommandSpec(
        command_type="import_metrics",
        argv=argv,
        expected_metrics=[
            "map50_95",
            "map50",
            "precision",
            "recall",
            "ap_small",
            "per_class_ap/*",
            "per_class_ar/*",
        ],
        metadata=metadata,
    )


def _run_dir_from_plan(plan_path: Path | str | None, run_id: str | None) -> Path:
    if plan_path is not None:
        path = Path(plan_path)
        if path.name in {"plan.yaml", "experiment_plan.yaml"}:
            return path.parent
    return Path("runs") / (run_id or "loop")
