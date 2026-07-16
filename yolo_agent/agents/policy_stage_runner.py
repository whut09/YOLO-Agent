"""Policy planning and evaluation loop stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from yolo_agent.adapters.ultralytics.baseline_acceptance import BaselineAcceptanceGate
from yolo_agent.adapters.ultralytics.candidate_promotion import CandidatePromotionGate, CandidatePromotionResult
from yolo_agent.adapters.ultralytics.training import (
    TrainingBudgetProfileName,
    UltralyticsTrainingConfig,
    command_from_training_config,
)
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.error_driven_loop import ErrorDrivenLoopReport
from yolo_agent.agents.decision_bundle import DecisionContext, LLMDecisionBundle
from yolo_agent.agents.budget_optimizer import BudgetOptimizer
from yolo_agent.agents.llm_decision_advisor import LLMDecisionAdvisor, LLMDecisionAdvisorResult
from yolo_agent.agents.llm_proposal_critic import LLMProposalQualityReport, LLMProposalCritic
from yolo_agent.agents.loop_evidence import LoopEvidence
from yolo_agent.agents.loop_io import read_json, read_yaml, write_yaml
from yolo_agent.agents.policy_memory_context import build_policy_memory_context
from yolo_agent.agents.loop_policy_evaluator import (
    BudgetPolicy,
    LoopPolicyEvaluation,
    LoopPolicyEvaluationReport,
    LoopPolicyEvaluator,
)
from yolo_agent.agents.loop_types import StageResult
from yolo_agent.agents.strategy_policy import CandidatePolicy
from yolo_agent.agents.training_recipe_planner import TrainingRecipePlanner
from yolo_agent.components.registry import ComponentRegistry
from yolo_agent.core.decision_ledger import (
    DecisionLedger,
    DecisionLedgerRecord,
    DecisionReplaySnapshot,
    build_replay_snapshot,
)
from yolo_agent.core.error_facts import ErrorFactStore
from yolo_agent.core.experiment_graph import Evidence, ExperimentNode
from yolo_agent.core.loop_state import LoopStage
from yolo_agent.core.optimization_objective import load_optimization_objective
from yolo_agent.core.policy_memory import PolicyMemoryStore
from yolo_agent.core.run_context import RunContext
from yolo_agent.core.round_execution_plan import build_round_execution_plan
from yolo_agent.core.stage_contract import LoopStageContracts
from yolo_agent.core.task_spec import TaskSpec


POLICY_VERSION = "LoopPolicyEvaluator@1.0"


def _baseline_control_node(
    context: RunContext,
    training_config: UltralyticsTrainingConfig | None,
) -> ExperimentNode | None:
    """Build the native matched control that accompanies every pilot fidelity."""
    if training_config is None:
        return None
    plan_path = context.run_dir / "plan.yaml"
    raw = read_yaml(plan_path) if plan_path.is_file() else {}
    candidates = raw.get("candidates", []) if isinstance(raw, dict) else []
    baseline_raw = next(
        (
            item
            for item in candidates
            if isinstance(item, dict) and "baseline" in str(item.get("candidate_id") or "").lower()
        ),
        None,
    )
    candidate = (
        CandidateConfig.model_validate(baseline_raw)
        if baseline_raw is not None
        else CandidateConfig(
            candidate_id="matched_baseline_control",
            base_model=training_config.model,
            scale="n",
            framework="ultralytics",
            expected_effect=["Matched native control for paired candidate deltas."],
        )
    )
    candidate = candidate.model_copy(
        update={
            "candidate_id": "matched_baseline_control",
            "base_model": training_config.model,
            "components": [],
            "train_overrides": {},
            "action_id": None,
        }
    )
    node = ExperimentNode(
        node_id=f"node_{candidate.candidate_id}_matched_control",
        candidate_config=candidate,
        data_version=context.dataset_version,
        seed=context.seed,
        fixed_variables={"imgsz": 640},
        changed_variables={},
    )
    command = command_from_training_config(node, training_config, context.run_id, context.data_yaml)
    command = command.model_copy(
        update={
            "metadata": {
                **command.metadata,
                "dataset_manifest_sha256": context.dataset_manifest_sha256 or "",
                "baseline_protocol_hash": context.metadata.get("baseline_protocol_hash") or "",
                "optimization_objective_hash": context.metadata.get("optimization_objective_hash") or "",
                "matched_baseline_control": True,
            }
        }
    )
    return node.model_copy(update={"command_spec": command, "command": command.display()})


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
        task_spec = TaskSpec.from_yaml(self.context.task_path)
        run_evidence = self.evidence.evidence_store.load_run(self.context.run_id)
        diagnostic_missing = _missing_diagnostic_evidence(
            evidence=run_evidence,
            run_artifacts={},
            requested_evidence=[
                *report.next_round.evidence_required,
                *_context_list(self.context.metadata.get("inherited_evidence_required", [])),
                *_context_list(self.context.metadata.get("inherited_missing_evidence", [])),
            ],
        )
        training_config = _training_config_from_context(self.context)
        policy_memory_context = build_policy_memory_context(
            PolicyMemoryStore(self.context.run_root),
            dataset_version=self.context.dataset_version,
            dataset_signature=self.context.dataset_manifest_sha256,
            scenario=task_spec.scene,
            model_family=_policy_memory_model_family(
                training_config.model if training_config is not None else self.context.metadata.get("training_model")
            ),
            target_metrics=_target_metrics_for_memory(report),
            target_actions=_target_actions_for_memory(report),
        )
        focus_items = _context_mapping_list(self.context.metadata.get("inherited_current_round_focus", []))
        allowed_actions = set(_context_list(self.context.metadata.get("inherited_current_round_error_actions", [])))
        tried_actions = set(_context_list(self.context.metadata.get("inherited_tried_action_ids", [])))
        recipe_plan = TrainingRecipePlanner().plan(
            context=self.context,
            evidence=run_evidence,
            focus_items=focus_items,
            allowed_actions=allowed_actions,
            tried_actions=tried_actions,
        ) if _proposal_mode(self.context) == "pilot_only" else None
        recipe_context = recipe_plan.model_dump(mode="json") if recipe_plan is not None else {}
        rule_policies = list(report.next_round.candidate_policies)
        recipe_policies = recipe_plan.policies if recipe_plan is not None else []
        paper_plan_path = self.context.artifact_path("paper_recipe_plan.yaml")
        paper_plan = read_yaml(paper_plan_path) if paper_plan_path.is_file() else {}
        paper_recipe_policies = _paper_recipe_policies(paper_plan_path)
        fallback_policies = _merge_policy_proposals(
            [*rule_policies, *recipe_policies, *paper_recipe_policies]
        )
        objective = load_optimization_objective(self.context.metadata.get("optimization_objective_path"))
        paper_inputs = paper_plan.get("decision_context_inputs", {}) if isinstance(paper_plan, dict) else {}
        decision_context = DecisionContext(
            run_id=self.context.run_id,
            research_snapshot_hash=self.context.metadata.get("research_snapshot_hash"),
            research_snapshot_path=self.context.metadata.get("research_snapshot_path"),
            research_snapshot_verified=bool(self.context.metadata.get("research_snapshot_verified", False)),
            baseline_evidence=[
                item.model_dump(mode="json")
                for item in run_evidence.metric_records
                if item.evidence_role == "baseline_reference"
            ][-100:],
            current_evidence=[
                item.model_dump(mode="json")
                for item in run_evidence.metric_records
                if item.evidence_role == "current_observation"
                and item.origin_run_id in {None, self.context.run_id}
            ][-100:],
            error_delta=(
                self.context.metadata.get("inherited_error_fact_delta", {})
                if isinstance(self.context.metadata.get("inherited_error_fact_delta", {}), dict)
                else {}
            ),
            diagnosis=report.model_dump(mode="json"),
            paper_candidates=(
                paper_inputs.get("paper_candidates", [])
                if isinstance(paper_inputs, dict) and isinstance(paper_inputs.get("paper_candidates", []), list)
                else []
            ),
            deterministic_recipe_candidates=[policy.model_dump(mode="json") for policy in fallback_policies],
            executable_adapters=(
                paper_inputs.get("executable_adapters", [])
                if isinstance(paper_inputs, dict) and isinstance(paper_inputs.get("executable_adapters", []), list)
                else []
            ),
            component_maturity=(
                paper_inputs.get("component_maturity", {})
                if isinstance(paper_inputs, dict) and isinstance(paper_inputs.get("component_maturity", {}), dict)
                else {}
            ),
            compatibility=(
                paper_inputs.get("compatibility", {})
                if isinstance(paper_inputs, dict) and isinstance(paper_inputs.get("compatibility", {}), dict)
                else {}
            ),
            policy_memory=policy_memory_context,
            tried_actions=sorted(tried_actions),
            rejected_actions=_context_list(self.context.metadata.get("inherited_rejected_action_ids", [])),
            objective=objective.model_dump(mode="json") if objective is not None else {},
            budget={
                "proposal_mode": _proposal_mode(self.context),
                "training_profile": training_config.budget_profile if training_config is not None else None,
                "policy_budget": self.policy.policy_budget,
            },
            guardrails=list(dict.fromkeys([
                *report.next_round.guardrails,
                *_context_list(self.context.metadata.get("inherited_guardrails", [])),
            ])),
            missing_evidence=diagnostic_missing,
            fallback_policies=fallback_policies,
        )
        llm_result = LLMDecisionAdvisor().propose(
            task_spec=task_spec,
            diagnosis_report=report,
            inherited_context={
                "run_id": self.context.run_id,
                "dataset_version": self.context.dataset_version,
                "proposal_mode": _proposal_mode(self.context),
                "missing_diagnostic_evidence": diagnostic_missing,
                "llm_evidence_only_mode": bool(diagnostic_missing),
                "policy_memory_context": policy_memory_context,
                "training_recipe_plan": recipe_context,
                "decision_context": decision_context.model_dump(mode="json"),
                "decision_context_hash": decision_context.context_hash,
                "inherited_current_round_focus": self.context.metadata.get("inherited_current_round_focus", []),
                "inherited_current_round_error_actions": self.context.metadata.get("inherited_current_round_error_actions", []),
                "inherited_guardrails": self.context.metadata.get("inherited_guardrails", []),
            },
        )
        llm_quality = LLMProposalCritic().critique(
            llm_result.proposals,
            fixed_imgsz=training_config.imgsz if training_config is not None else None,
            missing_diagnostic_evidence=diagnostic_missing,
            require_target_error_facts=True,
            require_expected_improvement=True,
        )
        llm_path = self.context.artifact_path("llm_decision.yaml")
        write_yaml(llm_path, llm_result.model_dump(mode="json"))
        llm_quality_path = self.context.artifact_path("llm_proposal_quality.yaml")
        write_yaml(llm_quality_path, llm_quality.model_dump(mode="json"))
        ledger_path = self.context.artifact_path("decision_ledger.jsonl")
        append_llm_decision_record(ledger_path, self.context.run_id, llm_result)
        append_llm_proposal_quality_record(ledger_path, self.context.run_id, llm_quality)
        accepted_llm_policies = [
            policy for policy in llm_result.proposals if policy.policy_id in llm_quality.accepted_policy_ids
        ]
        decision_mode = "llm" if llm_result.status == "used" else "deterministic_fallback"
        source_policies = (
            _merge_policy_proposals(accepted_llm_policies)
            if decision_mode == "llm"
            else fallback_policies
        )
        candidate_policies, contract_guardrails = _apply_inherited_pilot_contract(
            self.context,
            source_policies,
        )
        candidate_policies = _drop_satisfied_evidence_policies(
            candidate_policies,
            run_evidence,
        )
        candidate_policies = _normalize_policies_for_training_context(
            self.context,
            candidate_policies,
            training_config,
        )
        decision_bundle = LLMDecisionBundle(
            run_id=self.context.run_id,
            context=decision_context,
            llm_status=llm_result.status,
            provider=llm_result.provider,
            model=llm_result.model,
            prompt_sha256=llm_result.prompt_sha256,
            doctor_report_draft=(
                llm_result.doctor_report_draft.model_dump(mode="json")
                if llm_result.doctor_report_draft is not None
                else None
            ),
            proposed_policies=list(llm_result.proposals),
            evidence_requests=[item.model_dump(mode="json") for item in llm_result.evidence_requests],
            rejected_actions=[item.model_dump(mode="json") for item in llm_result.rejected_actions],
            critic_result=llm_quality.model_dump(mode="json"),
            critic_accepted_policy_ids=list(llm_quality.accepted_policy_ids),
            selected_for_evaluation_policy_ids=[item.policy_id for item in candidate_policies],
            decision_mode=decision_mode,
            warnings=list(llm_result.warnings),
        )
        decision_bundle_path = self.context.artifact_path("llm_decision_bundle.yaml")
        decision_bundle.to_yaml(decision_bundle_path)
        append_unified_decision_bundle_record(ledger_path, decision_bundle)
        data = {
            "candidate_policies": [policy.model_dump(mode="json") for policy in candidate_policies],
            "paper_recipe_policy_ids": [policy.policy_id for policy in paper_recipe_policies],
            "decision_bundle_hash": decision_bundle.decision_hash,
            "decision_context_hash": decision_context.context_hash,
            "research_snapshot_hash": decision_context.research_snapshot_hash,
            "research_snapshot_verified": decision_context.research_snapshot_verified,
            "decision_mode": decision_mode,
            "changed_variables": report.next_round.changed_variables,
            "evidence_required": report.next_round.evidence_required,
            "guardrails": list(dict.fromkeys([*report.next_round.guardrails, *contract_guardrails])),
            "llm_decision": {
                "status": llm_result.status,
                "provider": llm_result.provider,
                "model": llm_result.model,
                "model_alias": llm_result.model_alias,
                "proposal_count": len(llm_result.proposals),
                "warnings": llm_result.warnings,
            },
            "llm_proposal_quality": llm_quality.model_dump(mode="json"),
            "llm_evidence_first_gate": {
                "missing_diagnostic_evidence": diagnostic_missing,
                "evidence_only_mode": bool(diagnostic_missing),
            },
            "policy_memory_context": policy_memory_context,
            "training_recipe_plan": recipe_context,
            "proposal_sources": {
                "llm": len(accepted_llm_policies) if decision_mode == "llm" else 0,
                "llm_rejected_by_critic": llm_quality.rejected,
                "rule_engine": len(rule_policies) if decision_mode == "deterministic_fallback" else 0,
                "training_recipes": len(recipe_policies) if decision_mode == "deterministic_fallback" else 0,
                "paper_recipes": len(paper_recipe_policies) if decision_mode == "deterministic_fallback" else 0,
                "after_contract": len(candidate_policies),
            },
        }
        write_yaml(path, data)
        return StageResult(
            stage="generate_loop_plan",
            status="completed",
            message=f"Generated {len(candidate_policies)} policy proposals; llm_status={llm_result.status}.",
            artifacts={
                "loop_plan": path,
                "llm_decision": llm_path,
                "llm_decision_bundle": decision_bundle_path,
                "llm_proposal_quality": llm_quality_path,
                "decision_ledger": ledger_path,
            },
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
        objective = load_optimization_objective(self.context.metadata.get("optimization_objective_path"))
        evaluation = LoopPolicyEvaluator(
            registry,
            budget_policy=BudgetPolicy.model_validate(self.policy.policy_budget),
            fixed_imgsz=training_config.imgsz if training_config is not None else None,
            optimization_objective=objective,
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
        for item in evaluation.evaluations:
            node = item.experiment_node
            if node is None or node.command_spec is None:
                continue
            node.command_spec = node.command_spec.model_copy(
                update={
                    "metadata": {
                        **node.command_spec.metadata,
                        "dataset_manifest_sha256": self.context.dataset_manifest_sha256 or "",
                        "seed": node.seed,
                    }
                }
            )
            node.command = node.command_spec.display()
        budget_optimization = BudgetOptimizer().optimize(evaluation.evaluations)
        selected_node_ids = {arm.node_id for arm in budget_optimization.selected_arms}
        selected_nodes = [node for node in evaluation.experiment_nodes if node.node_id in selected_node_ids]
        ranks = {
            selection.arm.candidate_id: int(selection.rank or index)
            for index, selection in enumerate(budget_optimization.selected, start=1)
        }
        evaluation_hash = hashlib.sha256(
            json.dumps(evaluation.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        round_plan = build_round_execution_plan(
            run_id=self.context.run_id,
            nodes=selected_nodes,
            ranks=ranks,
            objective_hash=objective.objective_hash if objective is not None else None,
            decision_context_hash=str(raw_plan.get("decision_context_hash") or "") or None,
            source_decision_bundle_hash=str(raw_plan.get("decision_bundle_hash") or "") or None,
            source_policy_evaluation_hash=evaluation_hash,
            primary_metric=objective.primary_metric if objective is not None else "map50_95",
            baseline_control_node=_baseline_control_node(self.context, training_config),
        )
        round_plan.selected_recipes = [
            {"policy_id": item.policy_id, "decision": item.decision}
            for item in evaluation.evaluations
            if item.decision == "accepted"
        ]
        round_plan.critic_results = [
            {
                "policy_id": item.policy_id,
                "decision": item.decision,
                "errors": item.errors,
                "warnings": item.warnings,
            }
            for item in evaluation.evaluations
        ]
        round_plan_path = self.context.artifact_path("round_execution_plan.yaml")
        round_plan.to_yaml(round_plan_path)
        decision_bundle_path = self.context.artifact_path("llm_decision_bundle.yaml")
        if decision_bundle_path.is_file():
            decision_bundle = LLMDecisionBundle.from_yaml(decision_bundle_path)
            decision_bundle.deterministic_outcome = {
                "policy_evaluations": [
                    {
                        "policy_id": item.policy_id,
                        "decision": item.decision,
                        "candidate_id": (
                            item.candidate_config.candidate_id
                            if item.candidate_config is not None
                            else None
                        ),
                        "node_id": item.experiment_node.node_id if item.experiment_node is not None else None,
                    }
                    for item in evaluation.evaluations
                ],
                "budget_selected_candidate_ids": [arm.candidate_id for arm in budget_optimization.selected_arms],
                "round_execution_plan_hash": round_plan.plan_hash(),
                "execution_node_ids": [node.node_id for node in round_plan.execution_nodes],
            }
            decision_bundle.to_yaml(decision_bundle_path)
        budget_optimization_path = self.context.artifact_path("budget_optimization.yaml")
        write_yaml(
            budget_optimization_path,
            {
                **round_plan.budget_projection(),
                "budget_optimizer": budget_optimization.model_dump(mode="json"),
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
        round_plan.experiment_projection().to_yaml(experiment_plan_path)
        return StageResult(
            stage="evaluate_policies",
            status="completed",
            message=f"Accepted {len(evaluation.accepted_candidates)}/{len(evaluation.evaluations)} policies.",
            artifacts={
                "policy_evaluation": path,
                "experiment_plan": experiment_plan_path,
                "round_execution_plan": round_plan_path,
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
    ledger = DecisionLedger(path)
    preserved = [
        record for record in ledger.read() if record.decision_type != "policy_evaluation"
    ]
    records = [
        decision_record(run_id, proposals_by_id.get(item.policy_id), item, replay_snapshot)
        for item in evaluation.evaluations
    ]
    return ledger.write([*preserved, *records])


def append_llm_decision_record(
    path: Path,
    run_id: str,
    llm_result: LLMDecisionAdvisorResult,
) -> DecisionLedgerRecord:
    """Append the LLM proposal-generation step to the decision ledger."""
    proposal_bundle = (
        llm_result.proposal_bundle.model_dump(mode="json")
        if llm_result.proposal_bundle is not None
        else None
    )
    evidence_request_ids = [request.evidence_id for request in llm_result.evidence_requests]
    record = DecisionLedgerRecord(
        run_id=run_id,
        policy_id="llm_decision",
        decision_type="llm_proposal_generation",
        proposal={
            "policy_id": "llm_decision",
            "status": llm_result.status,
            "proposal_bundle": proposal_bundle,
            "candidate_policies": [policy.model_dump(mode="json") for policy in llm_result.proposals],
            "doctor_report_draft": (
                llm_result.doctor_report_draft.model_dump(mode="json")
                if llm_result.doctor_report_draft is not None
                else None
            ),
            "evidence_requests": [request.model_dump(mode="json") for request in llm_result.evidence_requests],
            "rejected_actions": [action.model_dump(mode="json") for action in llm_result.rejected_actions],
            "warnings": list(llm_result.warnings),
        },
        decision=llm_result.status,
        prompt_sha256=llm_result.prompt_sha256,
        input_summary=llm_result.input_summary,
        model_metadata={
            "provider": llm_result.provider,
            "model": llm_result.model,
            "model_alias": llm_result.model_alias,
            "temperature": llm_result.temperature,
            "max_output_tokens": llm_result.max_output_tokens,
        },
        missing_evidence=evidence_request_ids,
        blocked_by=list(llm_result.warnings),
        compatibility_warnings=list(llm_result.warnings),
        created_candidate_id=None,
        created_node_id=None,
        rationale="LLM generated proposal input for guarded policy evaluation.",
        policy_version="LLMDecisionAdvisor@1.0",
    )
    return DecisionLedger(path).append(record)


def append_llm_proposal_quality_record(
    path: Path,
    run_id: str,
    quality: LLMProposalQualityReport,
) -> DecisionLedgerRecord:
    """Append deterministic LLM proposal critique to the decision ledger."""
    record = DecisionLedgerRecord(
        run_id=run_id,
        policy_id="llm_proposal_quality",
        decision_type="llm_proposal_critic",
        proposal={"policy_id": "llm_proposal_quality", **quality.model_dump(mode="json")},
        decision="accepted" if quality.rejected == 0 else "rejected_some",
        blocked_by=list(quality.rejection_reasons),
        errors=list(quality.rejection_reasons),
        rationale="Deterministic critic filtered LLM proposals before policy evaluation.",
        policy_version="LLMProposalCritic@1.0",
    )
    return DecisionLedger(path).append(record)


def append_unified_decision_bundle_record(
    path: Path,
    bundle: LLMDecisionBundle,
) -> DecisionLedgerRecord:
    """Append the canonical one-LLM round decision boundary to the ledger."""
    record = DecisionLedgerRecord(
        run_id=bundle.run_id,
        policy_id="unified_llm_decision_bundle",
        decision_type="unified_llm_decision_bundle",
        proposal={
            "policy_id": "unified_llm_decision_bundle",
            "context_hash": bundle.context.context_hash,
            "decision_hash": bundle.decision_hash,
            "decision_mode": bundle.decision_mode,
            "proposed_policy_ids": [item.policy_id for item in bundle.proposed_policies],
            "critic_accepted_policy_ids": list(bundle.critic_accepted_policy_ids),
            "selected_for_evaluation_policy_ids": list(bundle.selected_for_evaluation_policy_ids),
        },
        decision=bundle.decision_mode,
        prompt_sha256=bundle.prompt_sha256,
        input_summary={
            "decision_context_hash": bundle.context.context_hash,
            "missing_evidence": list(bundle.context.missing_evidence),
            "paper_candidate_count": len(bundle.context.paper_candidates),
            "fallback_policy_count": len(bundle.context.fallback_policies),
        },
        model_metadata={
            "provider": bundle.provider,
            "model": bundle.model,
            "llm_status": bundle.llm_status,
        },
        missing_evidence=list(bundle.context.missing_evidence),
        blocked_by=list(bundle.warnings),
        rationale=(
            "One doctor-style LLM decision supplied proposals; deterministic critics, utility, "
            "budget, and ablation gates retain execution authority."
        ),
        policy_version="LLMDecisionBundle@1.0",
    )
    return DecisionLedger(path).append(record)


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


CRITICAL_DIAGNOSTIC_EVIDENCE_ALIASES: dict[str, set[str]] = {
    "ap_small": {"ap_small", "AP_small", "mAP_small", "map_small", "mAP_small"},
    "per_class_ap": {"per_class_ap", "per_class_ap/*", "per_class_ap_by_class", "class_ap"},
    "per_class_ar": {"per_class_ar", "per_class_ar/*", "per_class_ar_by_class", "class_ar"},
    "confusion_matrix": {"confusion_matrix", "confusion_matrix.json", "confusion_matrix.png"},
}


def _missing_diagnostic_evidence(
    *,
    evidence: Evidence,
    run_artifacts: dict[str, Path],
    requested_evidence: list[str],
) -> list[str]:
    """Return missing critical diagnostic evidence requested by the current loop."""
    requested = {str(item) for item in requested_evidence if str(item).strip()}
    missing: list[str] = []
    for canonical, aliases in CRITICAL_DIAGNOSTIC_EVIDENCE_ALIASES.items():
        if not requested.intersection(aliases | {canonical}):
            continue
        if not _has_diagnostic_evidence(canonical, evidence, run_artifacts):
            missing.append(canonical)
    return missing


def _has_diagnostic_evidence(
    canonical: str,
    evidence: Evidence,
    run_artifacts: dict[str, Path],
) -> bool:
    if canonical == "ap_small":
        return _has_metric(evidence, {"ap_small", "AP_small", "mAP_small", "map_small"})
    if canonical == "per_class_ap":
        return _has_metric_group(evidence, "per_class_ap/")
    if canonical == "per_class_ar":
        return _has_metric_group(evidence, "per_class_ar/")
    if canonical == "confusion_matrix":
        return _has_artifact(evidence, run_artifacts, {"confusion_matrix", "confusion_matrix.json", "confusion_matrix.png"})
    return False


def _has_metric(evidence: Evidence, names: set[str]) -> bool:
    if any(evidence.metrics.get(name) is not None for name in names):
        return True
    return any(record.verified and record.value is not None and record.metric_name in names for record in evidence.metric_records)


def _has_metric_group(evidence: Evidence, prefix: str) -> bool:
    if any(value is not None and name.startswith(prefix) for name, value in evidence.metrics.items()):
        return True
    return any(
        record.verified and record.value is not None and record.metric_name.startswith(prefix)
        for record in evidence.metric_records
    )


def _has_artifact(evidence: Evidence, run_artifacts: dict[str, Path], names: set[str]) -> bool:
    for name in names:
        path = run_artifacts.get(name)
        if path is not None and path.is_file():
            return True
    return any(
        entry.verify() and (entry.name in names or entry.path.name in names)
        for entry in evidence.artifact_manifest
    )


def _target_metrics_for_memory(report: ErrorDrivenLoopReport) -> list[str]:
    metrics: list[str] = []
    metrics.extend(report.next_round.evidence_required)
    for policy in report.next_round.candidate_policies:
        value = policy.expected_improvement.get("metric_name") if isinstance(policy.expected_improvement, dict) else None
        if value:
            metrics.append(str(value))
        for fact in policy.target_error_facts:
            metric_name = fact.get("metric_name") if isinstance(fact, dict) else None
            if metric_name:
                metrics.append(str(metric_name))
    for diagnostic in report.diagnostics:
        metrics.extend(diagnostic.expected_metrics)
    return list(dict.fromkeys(metrics))


def _target_actions_for_memory(report: ErrorDrivenLoopReport) -> list[str]:
    actions: list[str] = []
    for policy in report.next_round.candidate_policies:
        if policy.action_id:
            actions.append(policy.action_id)
        actions.extend(_target_actions(policy))
        actions.extend(policy.components)
    for values in report.next_round.changed_variables.values():
        actions.extend(str(item) for item in values)
    for diagnostic in report.diagnostics:
        actions.extend(diagnostic.next_actions)
    return list(dict.fromkeys(action for action in actions if action))


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
    config = UltralyticsTrainingConfig.from_yaml(path, budget_profile=_training_profile_from_context(context))
    model = context.metadata.get("training_model")
    if isinstance(model, str) and model.strip():
        return config.model_copy(update={"model": model.strip()})
    return config


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
    objective = load_optimization_objective(context.metadata.get("optimization_objective_path"))
    gate = CandidatePromotionGate(training_config.candidate_promotion, optimization_objective=objective)
    results = [
        gate.check(
            run_evidence,
            error_facts,
            candidate_id=policy.policy_id,
            target_actions=_target_actions(policy),
            target_error_facts=policy.target_error_facts,
            dataset_manifest_sha256=context.dataset_manifest_sha256,
            seed=context.seed,
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


def _merge_policy_proposals(policies: list[CandidatePolicy]) -> list[CandidatePolicy]:
    """Merge LLM and rule proposals while preserving source priority and IDs."""
    merged: list[CandidatePolicy] = []
    seen: set[str] = set()
    for policy in policies:
        policy_id = policy.policy_id
        if policy_id in seen:
            suffix = 2
            while f"{policy_id}_{suffix}" in seen:
                suffix += 1
            policy = policy.model_copy(update={"policy_id": f"{policy_id}_{suffix}"})
        seen.add(policy.policy_id)
        merged.append(policy)
    return merged


def _normalize_policies_for_training_context(
    context: RunContext,
    policies: list[CandidatePolicy],
    training_config: UltralyticsTrainingConfig | None,
) -> list[CandidatePolicy]:
    """Use the run's actual training model instead of generic rule-engine defaults."""
    model = context.metadata.get("training_model")
    if not isinstance(model, str) or not model.strip():
        model = training_config.model if training_config is not None else ""
    model = str(model).strip()
    if not model:
        return policies
    scale = _scale_from_model(model)
    normalized: list[CandidatePolicy] = []
    for policy in policies:
        if policy.base_model.lower() in {"yolo11n", "yolo11s", "yolo11n.pt", "yolo11s.pt"}:
            normalized.append(policy.model_copy(update={"base_model": model, "scale": scale or policy.scale}))
        else:
            normalized.append(policy)
    return normalized


def _scale_from_model(model: str) -> str:
    stem = Path(model).stem.lower()
    for scale in ("n", "s", "m", "l", "x"):
        if stem.endswith(scale):
            return scale
    return ""


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
    tried_actions = set(_context_list(context.metadata.get("inherited_tried_action_ids", [])))
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
        if policy.action_id in tried_actions and len(allowed_actions - tried_actions) > 0:
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
                    "risk": "low" if policy.action_domain in {"augmentation", "training"} else policy.risk,
                    "priority_hint": max(policy.priority_hint, 3.2)
                    if policy.action_domain in {"augmentation", "training"}
                    else policy.priority_hint,
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
        item_actions = _expanded_target_actions(item_actions)
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


def _expanded_target_actions(actions: set[str]) -> set[str]:
    expansions = {
        "hard_negative_mining": {"reduce_mosaic_strength"},
        "background_only_sampling": {"reduce_mosaic_strength"},
        "precision_threshold_tuning": {"reduce_mosaic_strength"},
        "bbox_loss_recipe": {"increase_box_loss_gain", "reduce_cls_loss_gain"},
        "assigner_recipe": {"increase_box_loss_gain"},
        "increase_recall_recipe": {"reduce_cls_loss_gain", "light_copy_paste", "light_mixup"},
        "class_balanced_sampling": {"light_copy_paste", "light_mixup"},
    }
    expanded = set(actions)
    for action in list(actions):
        expanded.update(expansions.get(action, set()))
    return expanded


def _synthetic_executable_pilot_policies(
    context: RunContext,
    *,
    focus_items: list[dict[str, Any]],
    allowed_actions: set[str],
    tried_actions: set[str],
    existing_policy_ids: set[str],
) -> list[CandidatePolicy]:
    """Materialize safe Ultralytics-native pilot actions from diagnosis actions."""
    model = str(context.metadata.get("training_model") or "yolo26n.pt")
    scale = _scale_from_model(model)
    recipe_families = [
        {
            "domain": "training",
            "unlocks": {"bbox_loss_recipe", "assigner_recipe", "increase_box_loss_gain"},
            "effect": "Increase box loss weight to target localization-heavy classes.",
            "metric": "localization_heavy_class",
            "priority": 3.2,
            "variants": [
                ("increase_box_loss_gain", {"training_action": "increase_box_loss_gain", "box": 9.0}),
                ("tune_box_loss_gain_8_25", {"training_action": "tune_box_loss_gain_8_25", "box": 8.25}),
                ("reduce_box_loss_gain_6_5", {"training_action": "reduce_box_loss_gain_6_5", "box": 6.5}),
            ],
        },
        {
            "domain": "training",
            "unlocks": {"increase_recall_recipe", "bbox_loss_recipe", "reduce_cls_loss_gain"},
            "effect": "Reduce classification loss weight to test recall/localization tradeoff.",
            "metric": "false_negative_heavy_class",
            "priority": 3.0,
            "variants": [
                ("reduce_cls_loss_gain", {"training_action": "reduce_cls_loss_gain", "cls": 0.35}),
                ("tune_cls_loss_gain_0_425", {"training_action": "tune_cls_loss_gain_0_425", "cls": 0.425}),
                ("increase_cls_loss_gain_0_65", {"training_action": "increase_cls_loss_gain_0_65", "cls": 0.65}),
            ],
        },
        {
            "domain": "augmentation",
            "unlocks": {"increase_recall_recipe", "class_balanced_sampling", "light_mixup"},
            "effect": "Use light mixup to test recall robustness without changing image size.",
            "metric": "false_negative_heavy_class",
            "priority": 2.4,
            "variants": [
                ("light_mixup", {"augmentation_action": "light_mixup", "mixup": 0.05}),
                ("mixup_0_1", {"augmentation_action": "mixup_0_1", "mixup": 0.1}),
                ("mixup_0_2", {"augmentation_action": "mixup_0_2", "mixup": 0.2}),
            ],
        },
        {
            "domain": "augmentation",
            "unlocks": {"hard_negative_mining", "background_only_sampling", "precision_threshold_tuning", "close_mosaic_early"},
            "effect": "Close mosaic earlier to reduce background/context artifacts.",
            "metric": "background_false_positive_class",
            "priority": 2.6,
            "variants": [
                ("close_mosaic_early", {"augmentation_action": "close_mosaic_early", "close_mosaic": 5}),
                ("close_mosaic_3", {"augmentation_action": "close_mosaic_3", "close_mosaic": 3}),
                ("mosaic_0_5", {"augmentation_action": "mosaic_0_5", "mosaic": 0.5}),
            ],
        },
    ]
    policies: list[CandidatePolicy] = []
    for recipe in recipe_families:
        unlocks = set(recipe["unlocks"])
        if not unlocks.intersection(allowed_actions):
            continue
        variants = list(recipe["variants"])
        next_variant = next(
            ((str(action), dict(overrides)) for action, overrides in variants if str(action) not in tried_actions),
            None,
        )
        if next_variant is None:
            continue
        action, variant_overrides = next_variant
        targets = _target_facts_for_actions(focus_items, unlocks)
        if not targets:
            continue
        policy_id = f"next_{recipe['domain']}_{action}"
        if policy_id in existing_policy_ids:
            continue
        expected = _expected_improvement_from_targets(targets, {action})
        expected["metric_name"] = recipe["metric"]
        expected["minimum_expected_delta"] = 0.002
        expected["expected_gain"] = {
            str(recipe["metric"]): max(float(recipe["priority"]), 3.2) * 0.1
        }
        expected["summary"] = str(recipe["effect"])
        train_overrides = variant_overrides
        train_overrides["target_actions"] = [action]
        policies.append(
            CandidatePolicy(
                policy_id=policy_id,
                source="rule_engine",
                action_domain=recipe["domain"],  # type: ignore[arg-type]
                action_id=action,
                execution_action="run_training",
                base_model=model,
                scale=scale or "n",
                framework="ultralytics",
                components=[],
                train_overrides=train_overrides,
                target_error_facts=targets,
                expected_improvement=expected,
                priority_hint=float(recipe["priority"]),
                expected_effect=[str(recipe["effect"])],
                risk="low",
                rationale=f"Ultralytics-native single-variable pilot for {action}.",
            )
        )
    return policies


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


def _policy_memory_model_family(value: Any) -> str | None:
    model = str(value or "").lower()
    if not model:
        return None
    for family in ("yolo26", "yolo11", "yolov10", "yolov9", "yolov8"):
        if family in model:
            return family
    return model.rsplit("/", 1)[-1].split(".", 1)[0]


def _paper_recipe_policies(path: Path) -> list[CandidatePolicy]:
    """Load only critic-approved pilot policies produced by Paper Intelligence."""
    if not path.is_file():
        return []
    raw = read_yaml(path)
    policies = raw.get("executable_pilot_policies", [])
    if not isinstance(policies, list):
        return []
    selected: list[CandidatePolicy] = []
    for item in policies:
        if not isinstance(item, dict):
            continue
        try:
            policy = CandidatePolicy.model_validate(item)
        except ValueError:
            continue
        if policy.execution_action != "run_training":
            continue
        if policy.train_overrides.get("imgsz", 640) != 640:
            continue
        selected.append(policy)
    return selected


def _drop_satisfied_evidence_policies(
    policies: list[CandidatePolicy],
    evidence: Evidence,
) -> list[CandidatePolicy]:
    """Remove evidence actions whose requested verified metrics already exist."""
    verified_names = {
        record.metric_name
        for record in evidence.metric_records
        if record.verified and record.value is not None
    }
    verified_names.update(
        name for name, value in evidence.metrics.items() if value is not None
    )
    selected: list[CandidatePolicy] = []
    for policy in policies:
        if policy.action_domain != "evidence":
            selected.append(policy)
            continue
        missing = policy.train_overrides.get("missing_evidence", [])
        requested = {str(item) for item in missing} if isinstance(missing, list) else set()
        if requested and requested.issubset(verified_names):
            continue
        selected.append(policy)
    return selected
