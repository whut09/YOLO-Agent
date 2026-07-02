"""Run orchestrator for the YOLO Agent loop harness."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from yolo_agent.agents.ablation_planner import AblationPlanner
from yolo_agent.agents.annotation_advisor import advise_annotations
from yolo_agent.agents.candidate_generator import CandidateConfig, CandidatePlan
from yolo_agent.agents.error_driven_loop import ErrorDrivenLoopEngine, ErrorDrivenLoopReport
from yolo_agent.agents.error_to_action import DetectionErrorObservation
from yolo_agent.agents.loop_policy_evaluator import LoopPolicyEvaluation, LoopPolicyEvaluationReport, LoopPolicyEvaluator
from yolo_agent.agents.strategy_policy import CandidatePolicy
from yolo_agent.components.registry import ComponentRegistry
from yolo_agent.core.artifact_manifest import sha256_file
from yolo_agent.core.dataset_versioning import DatasetVersionStore
from yolo_agent.core.decision_ledger import DecisionLedger, DecisionLedgerRecord
from yolo_agent.core.evidence_contract import EvidenceGate, default_loop_evidence_requirements
from yolo_agent.core.event_log import EventLog, EventType
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.executor import BenchmarkImporter
from yolo_agent.core.experiment_graph import ExperimentPlan
from yolo_agent.core.loop_state import LoopStage, LoopState, StageStatus
from yolo_agent.core.run_context import RunContext
from yolo_agent.core.run_lineage import RunLineageStore, build_lineage_record
from yolo_agent.core.schemas import DeploymentConstraints
from yolo_agent.core.stage_contract import LoopStageContracts, StageContractCheck
from yolo_agent.core.task_spec import TaskSpec
from yolo_agent.reports.experiment_report import generate_experiment_report
from yolo_agent.tools.dataset_stats import DatasetProfiler, DatasetReport, profile_dataset
from yolo_agent.tools.smoke_runner import SmokeRunner


class StageResult(BaseModel):
    """Result of one orchestrated stage."""

    stage: LoopStage
    status: StageStatus
    message: str = ""
    artifacts: dict[str, Path] = Field(default_factory=dict)


class LoopOrchestrator:
    """State-machine orchestrator for the full optimization harness loop."""

    def __init__(self, context: RunContext, state: LoopState | None = None) -> None:
        self.context = context
        self.context.ensure_dirs()
        self.policy = LoopStageContracts.from_yaml(context.loop_policy_path)
        self.state = state or self._load_or_create_state()
        self.evidence_store = EvidenceStore(context.run_root)
        self.event_log = EventLog(context.run_dir / "events.jsonl")
        self.lineage_store = RunLineageStore(context.run_root)

    @classmethod
    def initialize(
        cls,
        run_id: str,
        task_path: Path | str,
        data_yaml: Path | str,
        run_root: Path | str = "runs",
        component_path: Path | str = "configs/components",
        search_space_path: Path | str = "configs/search_space.yaml",
        loop_policy_path: Path | str = "configs/loop_policy.yaml",
        predictions_path: Path | str | None = None,
        detection_errors_path: Path | str | None = None,
        metrics_input_path: Path | str | None = None,
        dataset_version: str = "unversioned",
        seed: int = 42,
    ) -> "LoopOrchestrator":
        """Create a run context, initial state, and evidence config."""
        context = RunContext(
            run_id=run_id,
            run_root=Path(run_root),
            task_path=Path(task_path),
            data_yaml=Path(data_yaml),
            component_path=Path(component_path),
            search_space_path=Path(search_space_path),
            loop_policy_path=Path(loop_policy_path),
            predictions_path=Path(predictions_path) if predictions_path is not None else None,
            detection_errors_path=Path(detection_errors_path) if detection_errors_path is not None else None,
            metrics_input_path=Path(metrics_input_path) if metrics_input_path is not None else None,
            dataset_version=dataset_version,
            seed=seed,
        )
        context.ensure_dirs()
        dataset_manifest_path = _attach_dataset_manifest_to_context(context)
        context.to_yaml()
        context.to_json()
        policy = LoopStageContracts.from_yaml(loop_policy_path)
        state = LoopState.create(
            run_id,
            policy.stage_order,
            dataset_version=dataset_version,
            task_spec=Path(task_path),
        )
        state.mark(
            "init",
            "completed",
            "Run context initialized.",
            {
                "run_context": context.run_dir / "run_context.yaml",
                "dataset_manifest": dataset_manifest_path,
            },
        )
        state.to_yaml(context.run_dir / "loop_state.yaml")
        orchestrator = cls(context, state)
        orchestrator.evidence_store.log_config(run_id, {"run_context": context.model_dump(mode="json")})
        orchestrator._record_artifacts(
            "init",
            {
                "run_context": context.run_dir / "run_context.yaml",
                "loop_state": context.run_dir / "loop_state.yaml",
                "dataset_manifest": dataset_manifest_path,
            },
        )
        orchestrator.event_log.append(
            run_id=run_id,
            event_type="run_initialized",
            stage="init",
            status="completed",
            message="Run context initialized.",
            artifacts={
                "run_context": context.run_dir / "run_context.yaml",
                "dataset_manifest": dataset_manifest_path,
            },
        )
        orchestrator._record_lineage()
        return orchestrator

    @classmethod
    def from_run_dir(cls, run_dir: Path | str) -> "LoopOrchestrator":
        """Load an orchestrator from an existing run directory."""
        context = RunContext.from_run_dir(run_dir)
        state_path = context.run_dir / "loop_state.yaml"
        policy = LoopStageContracts.from_yaml(context.loop_policy_path)
        state = (
            LoopState.from_yaml(state_path)
            if state_path.exists()
            else LoopState.create(
                context.run_id,
                policy.stage_order,
                dataset_version=context.dataset_version,
                task_spec=context.task_path,
            )
        )
        return cls(context, state)

    def run_until_blocked(self) -> list[StageResult]:
        """Run pending stages until completion, block, or failure."""
        results: list[StageResult] = []
        while (stage := self.state.next_pending()) is not None:
            result = self.run_stage(stage)
            results.append(result)
            if result.status in {"blocked", "failed"}:
                break
        return results

    def resume(self) -> list[StageResult]:
        """Resume a blocked loop by retrying the first blocked stage."""
        blocked_stage = self.state.first_blocked()
        self.event_log.append(
            run_id=self.context.run_id,
            event_type="resume_requested",
            stage=blocked_stage,
            message="Resume requested.",
        )
        self.state.reset_for_resume()
        self._save_state()
        return self.run_until_blocked()

    def diagnose(self, errors_path: Path | str | None = None) -> list[StageResult]:
        """Run data profiling, label advice, and error diagnosis."""
        if errors_path is not None:
            self.context.detection_errors_path = Path(errors_path)
            self.context.to_yaml()
        return self.run_stages(["profile_data", "advise_labels", "diagnose_errors"])

    def plan_loop(self) -> list[StageResult]:
        """Generate loop plan, evaluate policies, candidates, and ablations."""
        return self.run_stages(["generate_loop_plan", "evaluate_policies", "generate_candidates", "ablate"])

    def smoke(self) -> StageResult:
        """Run smoke stage."""
        return self.run_stage("smoke")

    def ingest_metrics(self, metrics_path: Path | str) -> StageResult:
        """Set metrics input and run import_metrics."""
        self.context.metrics_input_path = Path(metrics_path)
        self.context.to_yaml()
        return self.run_stage("import_metrics")

    def next_round(self) -> list[StageResult]:
        """Generate report and next-round checklist."""
        return self.run_stages(["report", "next_round"])

    def fork_next(self, new_run_id: str) -> "LoopOrchestrator":
        """Materialize the next round into a fresh run that inherits loop context."""
        if not new_run_id or any(separator in new_run_id for separator in ("/", "\\")):
            raise ValueError("new_run_id must be a non-empty single path segment.")
        next_round_path = self.context.artifact_path("next_round.yaml")
        if not next_round_path.is_file():
            raise FileNotFoundError(f"Missing next_round.yaml: {next_round_path}")
        next_round = _read_yaml(next_round_path)
        new_run_dir = self.context.run_root / new_run_id
        if new_run_dir.exists():
            raise FileExistsError(f"Run already exists: {new_run_dir}")
        missing_evidence = _missing_evidence_from_status(self.context.artifact_path("evidence_status.json"))
        context = RunContext(
            run_id=new_run_id,
            run_root=self.context.run_root,
            task_path=self.context.task_path,
            data_yaml=self.context.data_yaml,
            component_path=self.context.component_path,
            search_space_path=self.context.search_space_path,
            loop_policy_path=self.context.loop_policy_path,
            predictions_path=self.context.predictions_path,
            detection_errors_path=self.context.detection_errors_path,
            metrics_input_path=None,
            dataset_version=self.context.dataset_version,
            dataset_root=self.context.dataset_root,
            dataset_version_store_path=self.context.dataset_version_store_path,
            seed=self.context.seed,
            metadata={
                "parent_run_id": self.context.run_id,
                "parent_run_dir": self.context.run_dir.as_posix(),
                "parent_next_round_path": next_round_path.as_posix(),
                "inherited_next_round_path": (self.context.run_root / new_run_id / "artifacts" / "parent_next_round.yaml").as_posix(),
                "inherited_evidence_required": list(next_round.get("evidence_required", [])),
                "inherited_missing_evidence": missing_evidence,
                "inherited_changed_variables": next_round.get("changed_variables", {}),
                "inherited_guardrails": list(next_round.get("guardrails", [])),
            },
        )
        context.ensure_dirs()
        dataset_manifest_path = _inherit_dataset_manifest_to_context(self.context, context)
        inherited_next_round_path = context.artifact_path("parent_next_round.yaml")
        shutil.copy2(next_round_path, inherited_next_round_path)
        fork_context_path = context.artifact_path("fork_context.yaml")
        _write_yaml(
            fork_context_path,
            {
                "parent_run_id": self.context.run_id,
                "parent_run_dir": self.context.run_dir.as_posix(),
                "parent_next_round_path": next_round_path.as_posix(),
                "inherited_next_round_path": inherited_next_round_path.as_posix(),
                "inherited_evidence_required": list(next_round.get("evidence_required", [])),
                "inherited_missing_evidence": missing_evidence,
                "inherited_changed_variables": next_round.get("changed_variables", {}),
                "inherited_guardrails": list(next_round.get("guardrails", [])),
                "dataset_version": context.dataset_version,
                "dataset_manifest_sha256": context.dataset_manifest_sha256,
            },
        )
        context.to_yaml()
        context.to_json()
        state = LoopState.create(
            new_run_id,
            self.policy.stage_order,
            dataset_version=context.dataset_version,
            task_spec=context.task_path,
        )
        state.mark(
            "init",
            "completed",
            "Forked from parent run.",
            {
                "run_context": context.run_dir / "run_context.yaml",
                "dataset_manifest": dataset_manifest_path,
                "fork_context": fork_context_path,
                "parent_next_round": inherited_next_round_path,
            },
        )
        state.to_yaml(context.run_dir / "loop_state.yaml")
        orchestrator = LoopOrchestrator(context, state)
        orchestrator.evidence_store.log_config(
            new_run_id,
            {
                "run_context": context.model_dump(mode="json"),
                "forked_from": self.context.run_id,
                "parent_next_round": next_round,
            },
        )
        orchestrator._record_artifacts(
            "init",
            {
                "run_context": context.run_dir / "run_context.yaml",
                "loop_state": context.run_dir / "loop_state.yaml",
                "dataset_manifest": dataset_manifest_path,
                "fork_context": fork_context_path,
                "parent_next_round": inherited_next_round_path,
            },
        )
        orchestrator.event_log.append(
            run_id=new_run_id,
            event_type="run_initialized",
            stage="init",
            status="completed",
            message=f"Forked from parent run {self.context.run_id}.",
            artifacts={
                "run_context": context.run_dir / "run_context.yaml",
                "dataset_manifest": dataset_manifest_path,
                "fork_context": fork_context_path,
                "parent_next_round": inherited_next_round_path,
            },
            details={
                "parent_run_id": self.context.run_id,
                "inherited_evidence_required": list(next_round.get("evidence_required", [])),
                "inherited_missing_evidence": missing_evidence,
            },
        )
        orchestrator._record_lineage(
            parent_run_id=self.context.run_id,
            inherited_missing_evidence=missing_evidence,
            current_missing_evidence=missing_evidence,
            metadata={"parent_next_round_path": inherited_next_round_path.as_posix()},
        )
        return orchestrator

    def run_stages(self, stages: list[LoopStage]) -> list[StageResult]:
        """Run selected stages in order until one blocks or fails."""
        results: list[StageResult] = []
        for stage in stages:
            result = self.run_stage(stage)
            results.append(result)
            if result.status in {"blocked", "failed"}:
                break
        return results

    def run_stage(self, stage: LoopStage) -> StageResult:
        """Run one loop stage and persist state."""
        contract_result = self._check_stage_contract(stage)
        if not contract_result.ok:
            artifacts = self._blocked_contract_artifacts(stage)
            result = StageResult(
                stage=stage,
                status="blocked",
                message="Missing required stage inputs: " + ", ".join(contract_result.missing_required),
                artifacts=artifacts,
            )
            self.event_log.append(
                run_id=self.context.run_id,
                event_type="contract_blocked",
                stage=stage,
                status="blocked",
                message=result.message,
                details={"missing_required": contract_result.missing_required},
                artifacts=artifacts,
            )
            self._record_artifacts(result.stage, result.artifacts)
            self.state.mark(result.stage, result.status, result.message, result.artifacts)
            self._save_state()
            return result
        self.state.mark(stage, "running", f"Running {stage}.")
        self._save_state()
        self.event_log.append(
            run_id=self.context.run_id,
            event_type="stage_started",
            stage=stage,
            status="running",
            message=f"Running {stage}.",
            details={"contract_warnings": contract_result.warnings},
        )
        try:
            result = self._run_stage(stage)
        except Exception as exc:  # pragma: no cover - defensive state guard
            result = StageResult(stage=stage, status="failed", message=str(exc))
        self._record_artifacts(result.stage, result.artifacts)
        self.state.mark(result.stage, result.status, result.message, result.artifacts)
        self._save_state()
        self.event_log.append(
            run_id=self.context.run_id,
            event_type=_event_type_for_status(result.status),
            stage=result.stage,
            status=result.status,
            message=result.message,
            artifacts=result.artifacts,
            details={"provides": self._stage_provides(result.stage)},
        )
        return result

    def _record_artifacts(self, stage: LoopStage, artifacts: dict[str, Path]) -> None:
        """Record artifact manifest entries for stage outputs."""
        for name, path in artifacts.items():
            artifact_path = Path(path)
            if not artifact_path.exists():
                continue
            self.evidence_store.log_artifact_manifest(
                run_id=self.context.run_id,
                name=name,
                artifact_path=artifact_path,
                producer_stage=stage,
            )

    def _check_stage_contract(self, stage: LoopStage) -> StageContractCheck:
        """Validate configured requirements for a stage."""
        return self.policy.get(stage).check(self._available_contract_items())

    def _stage_provides(self, stage: LoopStage) -> list[str]:
        """Return configured outputs for event logging."""
        return self.policy.get(stage).provides

    def _blocked_contract_artifacts(self, stage: LoopStage) -> dict[str, Path]:
        """Write any artifacts a blocked contract should still produce."""
        contract = self.policy.get(stage)
        artifacts: dict[str, Path] = {}
        if "evidence_status" in contract.producer_artifacts:
            artifacts["evidence_status"] = self._write_evidence_status()
        return artifacts

    def _available_contract_items(self) -> set[str]:
        """Return currently available contract inputs for this run."""
        available: set[str] = set()
        if self.context.task_path.is_file():
            available.add("task_spec")
        if self.context.data_yaml.is_file():
            available.add("data_yaml")
        if self.context.component_path.exists():
            available.add("component_registry")
        if self.context.detection_errors_path is not None and self.context.detection_errors_path.is_file():
            available.add("detection_errors")
        if self.context.metrics_input_path is not None and self.context.metrics_input_path.is_file():
            available.add("metrics_input")
        if (self.context.run_dir / "run_context.yaml").is_file():
            available.add("run_context")
        if (self.context.run_dir / "loop_state.yaml").is_file():
            available.add("loop_state")
        if self.context.dataset_manifest_path is not None and self.context.dataset_manifest_path.is_file():
            available.add("dataset_manifest")
        if (self.context.run_dir / "metrics.json").is_file():
            available.add("metrics")
        if (self.context.run_dir / "report.md").is_file():
            available.add("report")

        available.update(self.state.artifacts.keys())
        available.update(_existing_artifact_items(self.context))
        return available

    def _run_stage(self, stage: LoopStage) -> StageResult:
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
        dataset_manifest_path = _attach_dataset_manifest_to_context(self.context)
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
        dataset_report = DatasetReport.model_validate(_read_json(dataset_report_path))
        observations = _read_detection_errors(errors_path)
        deployment = DeploymentConstraints(
            target="unknown",
            max_latency_ms=task_spec.max_latency_ms,
            max_model_size_mb=task_spec.max_model_size_mb,
            preferred_export="none",
        )
        report = ErrorDrivenLoopEngine().run(task_spec, dataset_report, observations, deployment)
        path = self.context.artifact_path("loop_diagnosis.json")
        _write_json(path, report.model_dump(mode="json"))
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
        report = ErrorDrivenLoopReport.model_validate(_read_json(diagnosis_path))
        path = self.context.artifact_path("loop_plan.yaml")
        data = {
            "candidate_policies": [policy.model_dump(mode="json") for policy in report.next_round.candidate_policies],
            "changed_variables": report.next_round.changed_variables,
            "evidence_required": report.next_round.evidence_required,
            "guardrails": report.next_round.guardrails,
        }
        _write_yaml(path, data)
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
        raw_plan = _read_yaml(loop_plan_path)
        policies = [CandidatePolicy.model_validate(item) for item in raw_plan.get("candidate_policies", [])]
        registry = ComponentRegistry.from_path(self.context.component_path)
        task_spec = TaskSpec.from_yaml(self.context.task_path)
        evaluation = LoopPolicyEvaluator(registry).evaluate(
            proposals=policies,
            task_spec=task_spec,
            evidence_gate=self._current_evidence_gate(),
            data_version=self.context.dataset_version,
            seed=self.context.seed,
        )
        path = self.context.artifact_path("policy_evaluation.yaml")
        _write_yaml(path, evaluation.model_dump(mode="json"))
        ledger_path = self.context.artifact_path("decision_ledger.jsonl")
        _write_decision_ledger(
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
        evaluation = LoopPolicyEvaluationReport.model_validate(_read_yaml(evaluation_path))
        candidates = [_baseline_candidate(), *evaluation.accepted_candidates]
        plan = CandidatePlan(task_scene=task_spec.scene, candidates=_dedupe_candidates(candidates))
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
        _write_json(path, result.model_dump(mode="json"))
        return StageResult(
            stage="smoke",
            status="failed" if result.status == "failed" else "completed",
            message=f"Smoke status={result.status}.",
            artifacts={"smoke_result": path},
        )

    def _stage_import_metrics(self) -> StageResult:
        metrics_path = self.context.metrics_input_path
        if metrics_path is None or not metrics_path.is_file():
            gate_path = self._write_evidence_status()
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
        _write_json(output_path, import_result.run_metrics)
        gate_path = self._write_evidence_status()
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
        gate_path = self._write_evidence_status()
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
        raw_plan = _read_yaml(loop_plan_path)
        gate_path = self._write_evidence_status()
        output_path = self.context.artifact_path("next_round.yaml")
        _write_yaml(
            output_path,
            {
                "changed_variables": raw_plan.get("changed_variables", {}),
                "evidence_required": raw_plan.get("evidence_required", []),
                "guardrails": raw_plan.get("guardrails", []),
                "status": "ready_for_evidence_collection",
            },
        )
        return StageResult(
            stage="next_round",
            status="completed",
            message="Next-round checklist generated.",
            artifacts={"next_round": output_path, "evidence_status": gate_path},
        )

    def _blocked(self, stage: LoopStage, message: str) -> StageResult:
        return StageResult(stage=stage, status="blocked", message=message)

    def _load_or_create_state(self) -> LoopState:
        state_path = self.context.run_dir / "loop_state.yaml"
        if state_path.exists():
            return LoopState.from_yaml(state_path)
        return LoopState.create(
            self.context.run_id,
            self.policy.stage_order,
            dataset_version=self.context.dataset_version,
            task_spec=self.context.task_path,
        )

    def _save_state(self) -> None:
        self.state.to_yaml(self.context.run_dir / "loop_state.yaml")

    def _write_evidence_status(self) -> Path:
        gate = self._current_evidence_gate()
        path = self.context.artifact_path("evidence_status.json")
        _write_json(path, gate.model_dump(mode="json"))
        self._record_lineage(
            current_missing_evidence=gate.missing_required,
            trusted=gate.trusted,
        )
        return path

    def _current_evidence_gate(self):
        evidence = self.evidence_store.load_run(self.context.run_id)
        extra = _loop_plan_evidence_required(self.context.artifact_path("loop_plan.yaml"))
        return EvidenceGate(default_loop_evidence_requirements(extra)).evaluate(
            evidence=evidence,
            artifacts=self.state.artifacts,
        )

    def _record_lineage(
        self,
        parent_run_id: str | None = None,
        inherited_missing_evidence: list[str] | None = None,
        current_missing_evidence: list[str] | None = None,
        trusted: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a lineage snapshot for the current run."""
        context_parent = self.context.metadata.get("parent_run_id")
        parent = parent_run_id or (str(context_parent) if context_parent is not None else None)
        inherited = inherited_missing_evidence
        if inherited is None:
            raw_inherited = self.context.metadata.get("inherited_missing_evidence", [])
            inherited = [str(item) for item in raw_inherited] if isinstance(raw_inherited, list) else []
        current = current_missing_evidence
        if current is None:
            current = _missing_evidence_from_status(self.context.artifact_path("evidence_status.json"))
        evidence = self.evidence_store.load_run(self.context.run_id)
        merged_metadata = dict(self.context.metadata)
        if metadata:
            merged_metadata.update(metadata)
        self.lineage_store.append(
            build_lineage_record(
                run_id=self.context.run_id,
                run_dir=self.context.run_dir,
                parent_run_id=parent,
                dataset_version=self.context.dataset_version,
                dataset_manifest_sha256=self.context.dataset_manifest_sha256,
                inherited_missing_evidence=inherited,
                current_missing_evidence=current,
                trusted=bool(trusted) if trusted is not None else _trusted_from_status(self.context.artifact_path("evidence_status.json")),
                metrics=evidence.metrics,
                metadata=merged_metadata,
            )
        )


def _read_detection_errors(path: Path) -> list[DetectionErrorObservation]:
    raw = _read_yaml(path) if path.suffix.lower() in {".yaml", ".yml"} else _read_json(path)
    items = raw.get("errors", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError("Detection errors must be a list or an 'errors' list.")
    return [DetectionErrorObservation.model_validate(item) for item in items]


def _baseline_candidate() -> CandidateConfig:
    return CandidateConfig(
        candidate_id="yolo11n_baseline_n",
        base_model="yolo11n",
        scale="n",
        framework="ultralytics",
        expected_effect=["Baseline reference experiment."],
        risk="low",
    )


def _dedupe_candidates(candidates: list[CandidateConfig]) -> list[CandidateConfig]:
    seen: set[str] = set()
    deduped: list[CandidateConfig] = []
    for candidate in candidates:
        if candidate.candidate_id in seen:
            continue
        seen.add(candidate.candidate_id)
        deduped.append(candidate)
    return deduped


def _write_decision_ledger(
    path: Path,
    run_id: str,
    proposals: list[CandidatePolicy],
    evaluation: LoopPolicyEvaluationReport,
) -> Path:
    """Write proposal evaluation decisions as an audit ledger."""
    proposals_by_id = {proposal.policy_id: proposal for proposal in proposals}
    records = [
        _decision_record(run_id, proposals_by_id.get(item.policy_id), item)
        for item in evaluation.evaluations
    ]
    return DecisionLedger(path).write(records)


def _decision_record(
    run_id: str,
    proposal: CandidatePolicy | None,
    evaluation: LoopPolicyEvaluation,
) -> DecisionLedgerRecord:
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
        blocked_by=_blocked_by_decision(evaluation),
        missing_evidence=list(evaluation.missing_evidence),
        deployment_constraints=deployment_constraints,
        compatibility_warnings=list(evaluation.warnings),
        errors=list(evaluation.errors),
        created_candidate_id=candidate.candidate_id if candidate is not None else None,
        created_node_id=node.node_id if node is not None else None,
        candidate_config=candidate.model_dump(mode="json") if candidate is not None else None,
        experiment_node=node.model_dump(mode="json") if node is not None else None,
        rationale=evaluation.rationale,
    )


def _blocked_by_decision(evaluation: LoopPolicyEvaluation) -> list[str]:
    blocked_by: list[str] = []
    blocked_by.extend(str(item) for item in evaluation.blocked_by_deployment)
    blocked_by.extend(str(item) for item in evaluation.missing_evidence)
    blocked_by.extend(str(item) for item in evaluation.errors)
    if evaluation.decision == "split_required":
        blocked_by.append("multi_variable_policy")
    return list(dict.fromkeys(blocked_by))


def _loop_plan_evidence_required(path: Path) -> list[str]:
    if not path.is_file():
        return []
    raw = _read_yaml(path)
    values = raw.get("evidence_required", [])
    return [str(value) for value in values] if isinstance(values, list) else []


def _missing_evidence_from_status(path: Path) -> list[str]:
    """Return missing evidence names from a persisted evidence gate result."""
    if not path.is_file():
        return []
    raw = _read_json(path)
    values = raw.get("missing_required", []) if isinstance(raw, dict) else []
    return [str(value) for value in values] if isinstance(values, list) else []


def _trusted_from_status(path: Path) -> bool:
    """Return trusted flag from a persisted evidence gate result."""
    if not path.is_file():
        return False
    raw = _read_json(path)
    return bool(raw.get("trusted")) if isinstance(raw, dict) else False


def _attach_dataset_manifest_to_context(context: RunContext) -> Path:
    """Create a dataset manifest for the run and attach its hash to context."""
    dataset_root = _resolve_yolo_dataset_root(context.data_yaml)
    store_path = context.run_dir / "dataset_versions"
    DatasetVersionStore(store_path).create_version(
        dataset_root=dataset_root,
        version=context.dataset_version,
        notes=[
            f"run_id={context.run_id}",
            f"data_yaml={context.data_yaml.as_posix()}",
            "created_by=LoopOrchestrator.initialize",
        ],
        copy_data=False,
    )
    manifest_path = store_path / context.dataset_version / "manifest.json"
    context.dataset_root = dataset_root
    context.dataset_version_store_path = store_path
    context.dataset_manifest_path = manifest_path
    context.dataset_manifest_sha256 = sha256_file(manifest_path)
    return manifest_path


def _inherit_dataset_manifest_to_context(parent: RunContext, child: RunContext) -> Path:
    """Copy the parent's dataset manifest into the child run when available."""
    parent_manifest_path = parent.dataset_manifest_path
    if parent_manifest_path is None or not parent_manifest_path.is_file():
        return _attach_dataset_manifest_to_context(child)
    store_path = child.run_dir / "dataset_versions"
    manifest_path = store_path / child.dataset_version / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(parent_manifest_path, manifest_path)
    child.dataset_root = parent.dataset_root or _resolve_yolo_dataset_root(child.data_yaml)
    child.dataset_version_store_path = store_path
    child.dataset_manifest_path = manifest_path
    child.dataset_manifest_sha256 = sha256_file(manifest_path)
    return manifest_path


def _resolve_yolo_dataset_root(data_yaml: Path) -> Path:
    """Resolve the dataset root represented by a YOLO data.yaml."""
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data_yaml does not exist: {data_yaml}")
    raw = _read_yaml(data_yaml)
    configured_path = raw.get("path")
    if configured_path is None:
        return data_yaml.parent
    dataset_root = Path(str(configured_path))
    if not dataset_root.is_absolute():
        dataset_root = data_yaml.parent / dataset_root
    return dataset_root


def _existing_artifact_items(context: RunContext) -> set[str]:
    """Return contract item names inferred from existing artifact files."""
    path_candidates = {
        "dataset_report": [context.artifact_path("dataset_report.json")],
        "label_quality_report": [context.artifact_path("annotation_advice.json")],
        "annotation_advice": [context.artifact_path("annotation_advice.json")],
        "loop_diagnosis": [context.artifact_path("loop_diagnosis.json")],
        "loop_plan": [context.artifact_path("loop_plan.yaml")],
        "policy_evaluation": [context.artifact_path("policy_evaluation.yaml")],
        "decision_ledger": [context.artifact_path("decision_ledger.jsonl")],
        "experiment_plan": [context.artifact_path("experiment_plan.yaml")],
        "candidate_plan": [context.run_dir / "plan.yaml", context.artifact_path("candidate_plan.yaml")],
        "ablation_plan": [context.run_dir / "ablation_plan.yaml", context.artifact_path("ablation_plan.yaml")],
        "smoke_result": [context.artifact_path("smoke_result.json")],
        "evidence_status": [context.artifact_path("evidence_status.json")],
        "artifact_manifest": [context.artifact_path("artifact_manifest.jsonl")],
        "dataset_manifest": [
            context.dataset_manifest_path,
            context.run_dir / "dataset_versions" / context.dataset_version / "manifest.json",
        ],
        "next_round": [context.artifact_path("next_round.yaml")],
    }
    return {
        name
        for name, paths in path_candidates.items()
        if any(path is not None and path.is_file() for path in paths)
    }


def _event_type_for_status(status: StageStatus) -> EventType:
    if status == "completed":
        return "stage_completed"
    if status == "blocked":
        return "stage_blocked"
    if status == "failed":
        return "stage_failed"
    if status == "skipped":
        return "stage_skipped"
    return "stage_completed"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return data


def _write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)
