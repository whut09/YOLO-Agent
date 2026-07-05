"""Policy planning and evaluation loop stages."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.adapters.ultralytics.baseline_acceptance import BaselineAcceptanceGate
from yolo_agent.adapters.ultralytics.training import TrainingBudgetProfileName, UltralyticsTrainingConfig
from yolo_agent.agents.error_driven_loop import ErrorDrivenLoopReport
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
from yolo_agent.components.registry import ComponentRegistry
from yolo_agent.core.decision_ledger import (
    DecisionLedger,
    DecisionLedgerRecord,
    DecisionReplaySnapshot,
    build_replay_snapshot,
)
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
        data = {
            "candidate_policies": [policy.model_dump(mode="json") for policy in report.next_round.candidate_policies],
            "changed_variables": report.next_round.changed_variables,
            "evidence_required": report.next_round.evidence_required,
            "guardrails": report.next_round.guardrails,
        }
        write_yaml(path, data)
        return StageResult(
            stage="generate_loop_plan",
            status="completed",
            message=f"Generated {len(report.next_round.candidate_policies)} policy proposals.",
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
                **({"baseline_acceptance": self.context.artifact_path("baseline_acceptance.json")} if baseline_acceptance is not None else {}),
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
