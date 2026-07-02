"""Concrete stage implementations for the loop harness."""

from __future__ import annotations

import shutil
from pathlib import Path

from yolo_agent.agents.ablation_planner import AblationPlanner
from yolo_agent.agents.annotation_advisor import advise_annotations
from yolo_agent.agents.candidate_generator import CandidateConfig, CandidatePlan
from yolo_agent.agents.error_driven_loop import ErrorDrivenLoopEngine, ErrorDrivenLoopReport
from yolo_agent.agents.error_to_action import DetectionErrorObservation
from yolo_agent.agents.loop_evidence import LoopEvidence
from yolo_agent.agents.loop_io import read_json, read_yaml, write_json, write_yaml
from yolo_agent.agents.loop_policy_evaluator import (
    BudgetPolicy,
    LoopPolicyEvaluation,
    LoopPolicyEvaluationReport,
    LoopPolicyEvaluator,
)
from yolo_agent.agents.loop_types import StageResult
from yolo_agent.agents.run_initializer import attach_dataset_manifest_to_context
from yolo_agent.agents.strategy_policy import CandidatePolicy
from yolo_agent.components.registry import ComponentRegistry
from yolo_agent.core.decision_ledger import DecisionLedger, DecisionLedgerRecord
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.executor import BenchmarkImporter
from yolo_agent.core.experiment_graph import ExperimentPlan
from yolo_agent.core.loop_state import LoopStage
from yolo_agent.core.run_context import RunContext
from yolo_agent.core.schemas import DeploymentConstraints
from yolo_agent.core.stage_contract import LoopStageContracts
from yolo_agent.core.task_spec import TaskSpec
from yolo_agent.reports.experiment_report import generate_experiment_report
from yolo_agent.tools.dataset_stats import DatasetReport, profile_dataset
from yolo_agent.tools.smoke_runner import SmokeRunner, log_smoke_guard_evidence


class StageRunner:
    """Run concrete loop stages without owning the state machine."""

    def __init__(
        self,
        context: RunContext,
        policy: LoopStageContracts,
        evidence_store: EvidenceStore,
        evidence: LoopEvidence,
    ) -> None:
        self.context = context
        self.policy = policy
        self.evidence_store = evidence_store
        self.evidence = evidence

    def run(self, stage: LoopStage) -> StageResult:
        """Run one concrete stage."""
        dispatch = {
            "init": self._stage_init,
            "profile_data": self._stage_profile_data,
            "advise_labels": self._stage_advise_labels,
            "diagnose_errors": self._stage_diagnose_errors,
            "generate_loop_plan": self._stage_generate_loop_plan,
            "evaluate_policies": self._stage_evaluate_policies,
            "generate_candidates": self._stage_generate_candidates,
            "ablate": self._stage_ablate,
            "smoke": self._stage_smoke,
            "import_metrics": self._stage_import_metrics,
            "report": self._stage_report,
            "next_round": self._stage_next_round,
        }
        return dispatch[stage]()

    def _stage_init(self) -> StageResult:
        self.context.ensure_dirs()
        dataset_manifest_path = attach_dataset_manifest_to_context(self.context)
        context_path = self.context.to_yaml()
        self.context.to_json()
        return StageResult(
            stage="init",
            status="completed",
            message="Run context initialized.",
            artifacts={"run_context": context_path, "dataset_manifest": dataset_manifest_path},
        )

    def _stage_profile_data(self) -> StageResult:
        if not self.context.data_yaml.is_file():
            return self._blocked("profile_data", f"Missing data_yaml: {self.context.data_yaml}")
        report = profile_dataset(self.context.data_yaml, self.context.artifact_path("dataset_report"))
        return StageResult(
            stage="profile_data",
            status="completed",
            message=f"Profiled images={report.image_count} labels={report.label_count}.",
            artifacts={
                "dataset_report": self.context.artifact_path("dataset_report.json"),
                "dataset_report_md": self.context.artifact_path("dataset_report.md"),
            },
        )

    def _stage_advise_labels(self) -> StageResult:
        if not self.context.data_yaml.is_file():
            return self._blocked("advise_labels", f"Missing data_yaml: {self.context.data_yaml}")
        report = advise_annotations(
            data_yaml=self.context.data_yaml,
            out_prefix=self.context.artifact_path("annotation_advice"),
            predictions_path=self.context.predictions_path,
        )
        return StageResult(
            stage="advise_labels",
            status="completed",
            message=f"Found label_issues={len(report.label_quality.issues)}.",
            artifacts={
                "annotation_advice": self.context.artifact_path("annotation_advice.json"),
                "annotation_advice_md": self.context.artifact_path("annotation_advice.md"),
            },
        )

    def _stage_diagnose_errors(self) -> StageResult:
        errors_path = self.context.detection_errors_path
        if errors_path is None or not errors_path.is_file():
            return self._blocked("diagnose_errors", "Missing detection_errors_path; cannot diagnose model errors.")
        dataset_report_path = self.context.artifact_path("dataset_report.json")
        if not dataset_report_path.is_file():
            return self._blocked("diagnose_errors", "Missing dataset_report; run profile_data first.")

        task_spec = TaskSpec.from_yaml(self.context.task_path)
        dataset_report = DatasetReport.model_validate(read_json(dataset_report_path))
        observations = _read_detection_errors(errors_path)
        deployment = DeploymentConstraints(
            target="unknown",
            max_latency_ms=task_spec.max_latency_ms,
            max_model_size_mb=task_spec.max_model_size_mb,
            preferred_export="none",
        )
        report = ErrorDrivenLoopEngine().run(task_spec, dataset_report, observations, deployment)
        path = self.context.artifact_path("loop_diagnosis.json")
        write_json(path, report.model_dump(mode="json"))
        return StageResult(
            stage="diagnose_errors",
            status="completed",
            message=f"Created loop diagnosis with {len(report.diagnostics)} diagnostics.",
            artifacts={"loop_diagnosis": path},
        )

    def _stage_generate_loop_plan(self) -> StageResult:
        diagnosis_path = self.context.artifact_path("loop_diagnosis.json")
        if not diagnosis_path.is_file():
            return self._blocked("generate_loop_plan", "Missing loop_diagnosis; run diagnose_errors first.")
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

    def _stage_evaluate_policies(self) -> StageResult:
        loop_plan_path = self.context.artifact_path("loop_plan.yaml")
        if not loop_plan_path.is_file():
            return self._blocked("evaluate_policies", "Missing loop_plan; run generate_loop_plan first.")
        if not self.context.component_path.exists():
            return self._blocked("evaluate_policies", f"Missing component registry: {self.context.component_path}")
        raw_plan = read_yaml(loop_plan_path)
        policies = [CandidatePolicy.model_validate(item) for item in raw_plan.get("candidate_policies", [])]
        registry = ComponentRegistry.from_path(self.context.component_path)
        task_spec = TaskSpec.from_yaml(self.context.task_path)
        evaluation = LoopPolicyEvaluator(
            registry,
            budget_policy=BudgetPolicy.model_validate(self.policy.policy_budget),
        ).evaluate(
            proposals=policies,
            task_spec=task_spec,
            evidence_gate=self.evidence.current_gate(),
            data_version=self.context.dataset_version,
            seed=self.context.seed,
        )
        path = self.context.artifact_path("policy_evaluation.yaml")
        write_yaml(path, evaluation.model_dump(mode="json"))
        ledger_path = self.context.artifact_path("decision_ledger.jsonl")
        write_decision_ledger(
            path=ledger_path,
            run_id=self.context.run_id,
            proposals=policies,
            evaluation=evaluation,
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
            },
        )

    def _stage_generate_candidates(self) -> StageResult:
        evaluation_path = self.context.artifact_path("policy_evaluation.yaml")
        if not evaluation_path.is_file():
            return self._blocked("generate_candidates", "Missing policy_evaluation; run evaluate_policies first.")
        task_spec = TaskSpec.from_yaml(self.context.task_path)
        evaluation = LoopPolicyEvaluationReport.model_validate(read_yaml(evaluation_path))
        candidates = [baseline_candidate(), *evaluation.accepted_candidates]
        plan = CandidatePlan(task_scene=task_spec.scene, candidates=dedupe_candidates(candidates))
        plan_path = self.context.run_dir / "plan.yaml"
        plan.to_yaml(plan_path)
        shutil.copy2(plan_path, self.context.artifact_path("candidate_plan.yaml"))
        return StageResult(
            stage="generate_candidates",
            status="completed",
            message=f"Generated {len(plan.candidates)} candidates.",
            artifacts={"candidate_plan": plan_path},
        )

    def _stage_ablate(self) -> StageResult:
        plan_path = self.context.run_dir / "plan.yaml"
        if not plan_path.is_file():
            return self._blocked("ablate", "Missing candidate plan; run generate_candidates first.")
        candidate_plan = CandidatePlan.from_yaml(plan_path)
        ablation_plan = AblationPlanner().plan(candidate_plan.candidates)
        path = self.context.run_dir / "ablation_plan.yaml"
        ablation_plan.to_yaml(path)
        shutil.copy2(path, self.context.artifact_path("ablation_plan.yaml"))
        return StageResult(
            stage="ablate",
            status="completed",
            message=f"Created {len(ablation_plan.nodes)} ablation nodes.",
            artifacts={"ablation_plan": path},
        )

    def _stage_smoke(self) -> StageResult:
        plan_path = self.context.run_dir / "plan.yaml"
        if not plan_path.is_file():
            return self._blocked("smoke", "Missing candidate plan; run generate_candidates first.")
        result = SmokeRunner(EvidenceStore(self.context.run_dir / "smoke_evidence")).run(
            plan_path=plan_path,
            data_path=self.context.data_yaml,
            run_id="smoke",
            generated_dir=self.context.artifact_path("generated_models"),
        )
        path = self.context.artifact_path("smoke_result.json")
        write_json(path, result.model_dump(mode="json"))
        log_smoke_guard_evidence(
            evidence_store=self.evidence_store,
            run_id=self.context.run_id,
            result=result,
            dataset_version=self.context.dataset_version,
            source_artifact=path,
        )
        return StageResult(
            stage="smoke",
            status="failed" if result.status == "failed" else "completed",
            message=f"Smoke status={result.status}.",
            artifacts={"smoke_result": path},
        )

    def _stage_import_metrics(self) -> StageResult:
        metrics_path = self.context.metrics_input_path
        if metrics_path is None or not metrics_path.is_file():
            gate_path = self.evidence.write_status()
            return StageResult(
                stage="import_metrics",
                status="blocked",
                message="Missing metrics_input_path; import external benchmark metrics later.",
                artifacts={"evidence_status": gate_path},
            )
        import_result = BenchmarkImporter(self.evidence_store).import_metrics(
            run_id=self.context.run_id,
            metrics_path=metrics_path,
            dataset_version=self.context.dataset_version,
            source="loop_ingest_metrics",
        )
        output_path = self.context.artifact_path("metrics_import.json")
        write_json(output_path, import_result.run_metrics)
        gate_path = self.evidence.write_status()
        return StageResult(
            stage="import_metrics",
            status="completed",
            message=(
                f"Imported {len(import_result.run_metrics)} run metrics "
                f"and {len(import_result.metric_records)} node metrics."
            ),
            artifacts={"metrics": output_path, "evidence_status": gate_path},
        )

    def _stage_report(self) -> StageResult:
        gate_path = self.evidence.write_status()
        output_path = self.context.run_dir / "report.md"
        generate_experiment_report(self.context.run_dir, output_path)
        return StageResult(
            stage="report",
            status="completed",
            message=f"Wrote {output_path}.",
            artifacts={"report": output_path, "evidence_status": gate_path},
        )

    def _stage_next_round(self) -> StageResult:
        loop_plan_path = self.context.artifact_path("loop_plan.yaml")
        if not loop_plan_path.is_file():
            return self._blocked("next_round", "Missing loop_plan; run generate_loop_plan first.")
        raw_plan = read_yaml(loop_plan_path)
        gate_path = self.evidence.write_status()
        output_path = self.context.artifact_path("next_round.yaml")
        write_yaml(output_path, self.evidence.next_round_payload(raw_plan))
        return StageResult(
            stage="next_round",
            status="completed",
            message="Next-round checklist generated.",
            artifacts={"next_round": output_path, "evidence_status": gate_path},
        )

    def _blocked(self, stage: LoopStage, message: str) -> StageResult:
        return StageResult(stage=stage, status="blocked", message=message)


def read_detection_errors(path: Path) -> list[DetectionErrorObservation]:
    """Load detection error observations from JSON or YAML."""
    raw = read_yaml(path) if path.suffix.lower() in {".yaml", ".yml"} else read_json(path)
    items = raw.get("errors", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError("Detection errors must be a list or an 'errors' list.")
    return [DetectionErrorObservation.model_validate(item) for item in items]


def _read_detection_errors(path: Path) -> list[DetectionErrorObservation]:
    return read_detection_errors(path)


def baseline_candidate() -> CandidateConfig:
    """Return the default baseline candidate."""
    return CandidateConfig(
        candidate_id="yolo11n_baseline_n",
        base_model="yolo11n",
        scale="n",
        framework="ultralytics",
        expected_effect=["Baseline reference experiment."],
        risk="low",
    )


def dedupe_candidates(candidates: list[CandidateConfig]) -> list[CandidateConfig]:
    """Keep candidates unique by candidate_id while preserving order."""
    seen: set[str] = set()
    deduped: list[CandidateConfig] = []
    for candidate in candidates:
        if candidate.candidate_id in seen:
            continue
        seen.add(candidate.candidate_id)
        deduped.append(candidate)
    return deduped


def write_decision_ledger(
    path: Path,
    run_id: str,
    proposals: list[CandidatePolicy],
    evaluation: LoopPolicyEvaluationReport,
) -> Path:
    """Write proposal evaluation decisions as an audit ledger."""
    proposals_by_id = {proposal.policy_id: proposal for proposal in proposals}
    records = [
        decision_record(run_id, proposals_by_id.get(item.policy_id), item)
        for item in evaluation.evaluations
    ]
    return DecisionLedger(path).write(records)


def decision_record(
    run_id: str,
    proposal: CandidatePolicy | None,
    evaluation: LoopPolicyEvaluation,
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
