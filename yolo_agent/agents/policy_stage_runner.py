"""Policy planning and evaluation loop stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from yolo_agent.adapters.ultralytics.baseline_acceptance import BaselineAcceptanceGate
from yolo_agent.adapters.ultralytics.candidate_promotion import CandidatePromotionGate, CandidatePromotionResult
from yolo_agent.adapters.ultralytics.training import TrainingBudgetProfileName, UltralyticsTrainingConfig
from yolo_agent.agents.error_driven_loop import ErrorDrivenLoopReport
from yolo_agent.agents.budget_optimizer import BudgetOptimizer
from yolo_agent.agents.loop_evidence import LoopEvidence
from yolo_agent.agents.loop_io import read_json, read_yaml, write_yaml
from yolo_agent.agents.loop_policy_evaluator import (
    BudgetPolicy,
    LoopPolicyEvaluation,
    LoopPolicyEvaluationReport,
    LoopPolicyEvaluator,
)
from yolo_agent.agents.loop_types import StageResult
from yolo_agent.agents.strategy_policy import CandidatePolicy
from yolo_agent.agents.successive_halving import SuccessiveHalvingPlanner
from yolo_agent.components.registry import ComponentRegistry
from yolo_agent.core.decision_ledger import (
    DecisionLedger,
    DecisionLedgerRecord,
    DecisionReplaySnapshot,
    build_replay_snapshot,
)
from yolo_agent.core.error_facts import ErrorFactStore
from yolo_agent.core.experiment_graph import ExperimentPlan
from yolo_agent.core.loop_state import LoopStage
from yolo_agent.core.run_context import RunContext
from yolo_agent.core.stage_contract import LoopStageContracts
from yolo_agent.core.task_spec import TaskSpec


POLICY_VERSION = "LoopPolicyEvaluator@1.0"


class PolicyStageRunner:
    """Run loop-policy proposal and evaluation stages."""

    def __init__(
        self,
        context: RunContext,
        policy: LoopStageContracts,
        evidence: LoopEvidence,
    ) -> None:
        self.context = context
        self.policy = policy
        self.evidence = evidence

    def generate_loop_plan(self) -> StageResult:
        """Convert diagnosis into loop policy proposals."""
        diagnosis_path = self.context.artifact_path("loop_diagnosis.json")
        if not diagnosis_path.is_file():
            return _blocked("generate_loop_plan", "Missing loop_diagnosis; run diagnose_errors first.")
        report = ErrorDrivenLoopReport.model_validate(read_json(diagnosis_path))
        path = self.context.artifact_path("loop_plan.yaml")
        candidate_policies, contract_guardrails = _apply_inherited_pilot_contract(
            self.context,
            list(report.next_round.candidate_policies),
        )
        data = {
            "candidate_policies": [policy.model_dump(mode="json") for policy in candidate_policies],
            "changed_variables": report.next_round.changed_variables,
            "evidence_required": report.next_round.evidence_required,
            "guardrails": list(dict.fromkeys([*report.next_round.guardrails, *contract_guardrails])),
        }
        write_yaml(path, data)
        return StageResult(
            stage="generate_loop_plan",
            status="completed",
            message=f"Generated {len(candidate_policies)} policy proposals.",
            artifacts={"loop_plan": path},
        )

    def evaluate_policies(self) -> StageResult:
        """Evaluate loop policy proposals and persist experiment graph artifacts."""
        loop_plan_path = self.context.artifact_path("loop_plan.yaml")
        if not loop_plan_path.is_file():
            return _blocked("evaluate_policies", "Missing loop_plan; run generate_loop_plan first.")
        if not self.context.component_path.exists():
            return _blocked("evaluate_policies", f"Missing component registry: {self.context.component_path}")
        raw_plan = read_yaml(loop_plan_path)
        policies = [CandidatePolicy.model_validate(item) for item in raw_plan.get("candidate_policies", [])]
        registry = ComponentRegistry.from_path(self.context.component_path)
        task_spec = TaskSpec.from_yaml(self.context.task_path)
        evidence_gate = self.evidence.current_gate()
        training_config = _training_config_from_context(self.context)
        baseline_acceptance = None
        error_facts = ErrorFactStore(self.context.run_root).read(self.context.run_id)
        if training_config is not None and training_config.budget_profile == "candidate_full":
            expected_sha = self.context.metadata.get("coco_manifest_sha256")
            baseline_acceptance = BaselineAcceptanceGate(training_config.baseline_acceptance).check(
                self.evidence.evidence_store.load_run(self.context.run_id),
                expected_dataset_manifest_sha256=str(expected_sha) if isinstance(expected_sha, str) else None,
                actual_dataset_manifest_sha256=self.context.dataset_manifest_sha256,
            )
            BaselineAcceptanceGate(training_config.baseline_acceptance).persist_decision(
                self.evidence.evidence_store,
                self.context.run_id,
                baseline_acceptance,
                dataset_version=self.context.dataset_version,
            )
        candidate_promotions = _candidate_promotions_for_policies(
            context=self.context,
            evidence=self.evidence,
            policies=policies,
            training_config=training_config,
        )
        evaluation = LoopPolicyEvaluator(
            registry,
            budget_policy=BudgetPolicy.model_validate(self.policy.policy_budget),
            fixed_imgsz=training_config.imgsz if training_config is not None else None,
        ).evaluate(
            proposals=policies,
            task_spec=task_spec,
            evidence_gate=evidence_gate,
            data_version=self.context.dataset_version,
            seed=self.context.seed,
            plan_path=self.context.run_dir / "plan.yaml",
            data_path=self.context.data_yaml,
            run_id=self.context.run_id,
            training_config=training_config,
            baseline_acceptance=baseline_acceptance,
            candidate_promotions=candidate_promotions,
            error_facts=error_facts,
            proposal_mode=_proposal_mode(self.context),
            allowed_training_profiles=_context_list(self.context.metadata.get("inherited_proposal_budget_profiles_allowed", [])),
            required_proposal_bindings=_context_list(self.context.metadata.get("inherited_proposal_required_bindings", [])),
        )
        budget_optimization = BudgetOptimizer().optimize(evaluation.evaluations)
        halving_plan = SuccessiveHalvingPlanner().plan(budget_optimization.selected_arms)
        budget_optimization_path = self.context.artifact_path("budget_optimization.yaml")
        write_yaml(
            budget_optimization_path,
            {
                "budget_optimizer": budget_optimization.model_dump(mode="json"),
                "successive_halving": halving_plan.model_dump(mode="json"),
            },
        )
        path = self.context.artifact_path("policy_evaluation.yaml")
        write_yaml(path, evaluation.model_dump(mode="json"))
        ledger_path = self.context.artifact_path("decision_ledger.jsonl")
        write_decision_ledger(
            path=ledger_path,
            run_id=self.context.run_id,
            proposals=policies,
            evaluation=evaluation,
            replay_snapshot=build_replay_snapshot(
                task_spec_path=self.context.task_path,
                component_registry_path=self.context.component_path,
                loop_plan_path=loop_plan_path,
                evidence_gate=evidence_gate,
                policy_version=POLICY_VERSION,
            ),
        )
        experiment_plan_path = self.context.artifact_path("experiment_plan.yaml")
        ExperimentPlan(
            plan_id=f"{self.context.run_id}_loop_policy_plan",
            nodes=evaluation.experiment_nodes,
            metadata={
                "source": "LoopPolicyEvaluator",
                "split_required": [
                    item.policy_id for item in evaluation.evaluations if item.decision == "split_required"
                ],
                "needs_evidence": [
                    item.policy_id for item in evaluation.evaluations if item.decision == "needs_evidence"
                ],
                "deferred": [
                    item.policy_id for item in evaluation.evaluations if item.decision == "deferred"
                ],
                "needs_approval": [
                    item.policy_id for item in evaluation.evaluations if item.decision == "needs_approval"
                ],
                "budget_allocation": (
                    evaluation.budget_allocation.model_dump(mode="json")
                    if evaluation.budget_allocation is not None
                    else {}
                ),
                "baseline_acceptance": (
                    baseline_acceptance.model_dump(mode="json")
                    if baseline_acceptance is not None
                    else {}
                ),
                "candidate_promotion": {
                    policy_id: result.model_dump(mode="json")
                    for policy_id, result in (candidate_promotions or {}).items()
                },
                "budget_optimizer": budget_optimization.model_dump(mode="json"),
                "successive_halving": halving_plan.model_dump(mode="json"),
            },
        ).to_yaml(experiment_plan_path)
        return StageResult(
            stage="evaluate_policies",
            status="completed",
            message=f"Accepted {len(evaluation.accepted_candidates)}/{len(evaluation.evaluations)} policies.",
            artifacts={
                "policy_evaluation": path,
                "experiment_plan": experiment_plan_path,
                "decision_ledger": ledger_path,
                "budget_optimization": budget_optimization_path,
                **({"baseline_acceptance": self.context.artifact_path("baseline_acceptance.json")} if baseline_acceptance is not None else {}),
                **({"candidate_promotion": self.context.artifact_path("candidate_promotion.json")} if candidate_promotions else {}),
            },
        )


def write_decision_ledger(
    path: Path,
    run_id: str,
    proposals: list[CandidatePolicy],
    evaluation: LoopPolicyEvaluationReport,
    replay_snapshot: DecisionReplaySnapshot | None = None,
) -> Path:
    """Write proposal evaluation decisions as an audit ledger."""
    proposals_by_id = {proposal.policy_id: proposal for proposal in proposals}
    records = [
        decision_record(run_id, proposals_by_id.get(item.policy_id), item, replay_snapshot)
        for item in evaluation.evaluations
    ]
    return DecisionLedger(path).write(records)


def decision_record(
    run_id: str,
    proposal: CandidatePolicy | None,
    evaluation: LoopPolicyEvaluation,
    replay_snapshot: DecisionReplaySnapshot | None = None,
) -> DecisionLedgerRecord:
    """Build one decision ledger record."""
    candidate = evaluation.candidate_config
    node = evaluation.experiment_node
    proposal_data = proposal.model_dump(mode="json") if proposal is not None else {"policy_id": evaluation.policy_id}
    deployment_constraints = [
        constraint.model_dump(mode="json")
        for constraint in (proposal.constraints if proposal is not None else [])
    ]
    return DecisionLedgerRecord(
        run_id=run_id,
        policy_id=evaluation.policy_id,
        proposal=proposal_data,
        decision=evaluation.decision,
        priority=evaluation.priority,
        blocked_by=blocked_by_decision(evaluation),
        missing_evidence=list(evaluation.missing_evidence),
        deployment_constraints=deployment_constraints,
        compatibility_warnings=list(evaluation.warnings),
        errors=list(evaluation.errors),
        budget_bucket=evaluation.budget_bucket,
        budget_reason=evaluation.budget_reason,
        requires_human_confirmation=evaluation.requires_human_confirmation,
        created_candidate_id=candidate.candidate_id if candidate is not None else None,
        created_node_id=node.node_id if node is not None else None,
        candidate_config=candidate.model_dump(mode="json") if candidate is not None else None,
        experiment_node=node.model_dump(mode="json") if node is not None else None,
        rationale=evaluation.rationale,
        task_spec_sha256=replay_snapshot.task_spec_sha256 if replay_snapshot is not None else None,
        component_registry_sha256=replay_snapshot.component_registry_sha256 if replay_snapshot is not None else None,
        loop_plan_sha256=replay_snapshot.loop_plan_sha256 if replay_snapshot is not None else None,
        evidence_gate_sha256=replay_snapshot.evidence_gate_sha256 if replay_snapshot is not None else None,
        policy_version=replay_snapshot.policy_version if replay_snapshot is not None else POLICY_VERSION,
        replay_snapshot=replay_snapshot,
    )


def blocked_by_decision(evaluation: LoopPolicyEvaluation) -> list[str]:
    """Summarize blocking causes for a policy evaluation."""
    blocked_by: list[str] = []
    blocked_by.extend(str(item) for item in evaluation.blocked_by_deployment)
    blocked_by.extend(str(item) for item in evaluation.missing_evidence)
    blocked_by.extend(str(item) for item in evaluation.errors)
    if evaluation.decision == "split_required":
        blocked_by.append("multi_variable_policy")
    if evaluation.decision == "deferred":
        blocked_by.append(evaluation.budget_reason or "budget_deferred")
    if evaluation.decision == "needs_approval":
        blocked_by.append(evaluation.budget_reason or "human_confirmation_required")
    return list(dict.fromkeys(blocked_by))


def _blocked(stage: LoopStage, message: str) -> StageResult:
    return StageResult(stage=stage, status="blocked", message=message)


def _training_config_from_context(context: RunContext) -> UltralyticsTrainingConfig | None:
    """Load optional Ultralytics training config for executable experiment nodes."""
    raw_path = context.metadata.get("training_config_path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_file():
        return None
    return UltralyticsTrainingConfig.from_yaml(path, budget_profile=_training_profile_from_context(context))


def _training_profile_from_context(context: RunContext) -> TrainingBudgetProfileName | None:
    """Return a validated training profile from run metadata."""
    value = context.metadata.get("training_profile")
    if value in {"debug", "pilot", "baseline_full", "baseline_confirm", "candidate_full"}:
        return value  # type: ignore[return-value]
    return None


def _candidate_promotions_for_policies(
    context: RunContext,
    evidence: LoopEvidence,
    policies: list[CandidatePolicy],
    training_config: UltralyticsTrainingConfig | None,
) -> dict[str, CandidatePromotionResult] | None:
    """Evaluate candidate pilot promotion decisions when planning full candidates."""
    if training_config is None or training_config.budget_profile != "candidate_full":
        return None
    if not training_config.selected_budget_profile().requires_pilot_pass:
        return None
    run_evidence = evidence.evidence_store.load_run(context.run_id)
    error_facts = ErrorFactStore(context.run_root).read(context.run_id)
    gate = CandidatePromotionGate(training_config.candidate_promotion)
    results = [
        gate.check(
            run_evidence,
            error_facts,
            candidate_id=policy.policy_id,
            target_actions=_target_actions(policy),
            target_error_facts=policy.target_error_facts,
        )
        for policy in policies
    ]
    gate.persist_decisions(
        evidence.evidence_store,
        context.run_id,
        results,
        dataset_version=context.dataset_version,
    )
    return {result.candidate_id: result for result in results}


def _target_actions(policy: CandidatePolicy) -> list[str]:
    """Return explicit target actions from policy metadata when provided."""
    actions: list[str] = []
    if policy.action_id:
        actions.append(policy.action_id)
    for key in ("target_actions", "target_error_actions", "action_candidates"):
        value = policy.train_overrides.get(key)
        if isinstance(value, list):
            actions.extend(str(item) for item in value)
        if isinstance(value, str) and value.strip():
            actions.extend(part.strip() for part in value.split(",") if part.strip())
    for component in policy.components:
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


def _apply_inherited_pilot_contract(
    context: RunContext,
    policies: list[CandidatePolicy],
) -> tuple[list[CandidatePolicy], list[str]]:
    """Bind inherited error-delta focus to next-round pilot proposals."""
    if _proposal_mode(context) == "blocked":
        evidence_policies = [policy for policy in policies if policy.action_domain == "evidence"]
        return evidence_policies, [
            "proposal_generation_blocked_until_error_facts_exist",
            "no_candidate_full_without_error_facts",
            "evidence_actions_allowed_while_training_proposals_blocked",
        ]
    if _proposal_mode(context) != "pilot_only":
        return policies, []
    focus_items = _context_mapping_list(context.metadata.get("inherited_current_round_focus", []))
    allowed_actions = set(_context_list(context.metadata.get("inherited_current_round_error_actions", [])))
    if not focus_items or not allowed_actions:
        return [], ["pilot_only_requires_target_error_facts"]

    bound: list[CandidatePolicy] = []
    for policy in policies:
        if policy.action_domain == "evidence":
            expected_improvement = _expected_improvement_from_targets(focus_items, set(_target_actions(policy)))
            expected_improvement["summary"] = f"Collect evidence before training action: {policy.action_id}."
            train_overrides = dict(policy.train_overrides)
            train_overrides["target_actions"] = sorted(allowed_actions)
            bound.append(
                policy.model_copy(
                    update={
                        "train_overrides": train_overrides,
                        "target_error_facts": focus_items,
                        "expected_improvement": expected_improvement,
                    }
                )
            )
            continue
        actions = set(_target_actions(policy)) & allowed_actions
        if not actions:
            continue
        targets = _target_facts_for_actions(focus_items, actions)
        if not targets:
            continue
        expected_improvement = _expected_improvement_from_targets(targets, actions)
        train_overrides = dict(policy.train_overrides)
        train_overrides["target_actions"] = sorted(actions)
        bound.append(
            policy.model_copy(
                update={
                    "train_overrides": train_overrides,
                    "target_error_facts": targets,
                    "expected_improvement": expected_improvement,
                    "expected_effect": list(
                        dict.fromkeys(
                            [
                                *policy.expected_effect,
                                str(expected_improvement["summary"]),
                            ]
                        )
                    ),
                }
            )
        )
    return bound, [
        "pilot_only_proposals",
        "candidate_full_blocked_until_pilot_promotion",
        "target_error_facts_required",
        "expected_improvement_required",
    ]


def _target_facts_for_actions(
    focus_items: list[dict[str, Any]],
    actions: set[str],
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for item in focus_items:
        raw_actions = item.get("action_candidates", [])
        item_actions = {str(action) for action in raw_actions} if isinstance(raw_actions, list) else set()
        if not item_actions.intersection(actions):
            continue
        target = {
            "fact_type": item.get("fact_type"),
            "subject": item.get("subject"),
            "class_name": item.get("class_name"),
            "class_pair": item.get("class_pair"),
            "area": item.get("area"),
            "metric_name": item.get("metric_name"),
            "current_value": item.get("current_value", item.get("value")),
            "current_severity": item.get("current_severity", item.get("severity")),
            "trend": item.get("trend", "current"),
            "action_candidates": sorted(item_actions),
            "node_id": item.get("node_id"),
            "candidate_id": item.get("candidate_id"),
        }
        targets.append({key: value for key, value in target.items() if value is not None})
    return targets


def _expected_improvement_from_targets(
    targets: list[dict[str, Any]],
    actions: set[str],
) -> dict[str, Any]:
    primary = targets[0]
    metric_name = str(primary.get("metric_name") or primary.get("fact_type") or "target_error")
    direction = "decrease" if primary.get("fact_type") in {
        "false_negative_heavy_class",
        "localization_heavy_class",
        "class_confusion_pair",
        "background_false_positive_class",
    } else "increase"
    subject = primary.get("class_name") or primary.get("class_pair") or primary.get("area") or primary.get("subject")
    return {
        "metric_name": metric_name,
        "direction": direction,
        "target": subject,
        "actions": sorted(actions),
        "minimum_expected_delta": "pilot_positive_delta",
        "summary": f"Pilot should {direction} {metric_name} for {subject}.",
    }


def _proposal_mode(context: RunContext) -> str | None:
    value = context.metadata.get("inherited_proposal_mode") or context.metadata.get("proposal_mode")
    return str(value) if value else None


def _context_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _context_mapping_list(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []
