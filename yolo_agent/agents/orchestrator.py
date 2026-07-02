"""Run orchestrator for the YOLO Agent loop harness."""

from __future__ import annotations

import csv
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
from yolo_agent.agents.loop_policy_evaluator import LoopPolicyEvaluationReport, LoopPolicyEvaluator
from yolo_agent.agents.strategy_policy import CandidatePolicy
from yolo_agent.components.registry import ComponentRegistry
from yolo_agent.core.evidence_contract import EvidenceGate, default_loop_evidence_requirements
from yolo_agent.core.event_log import EventLog, EventType
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.experiment_graph import ExperimentPlan, MetricEvidence, MetricValue
from yolo_agent.core.loop_state import LoopStage, LoopState, StageStatus
from yolo_agent.core.run_context import RunContext
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
        context.to_yaml()
        context.to_json()
        policy = LoopStageContracts.from_yaml(loop_policy_path)
        state = LoopState.create(
            run_id,
            policy.stage_order,
            dataset_version=dataset_version,
            task_spec=Path(task_path),
        )
        state.mark("init", "completed", "Run context initialized.", {"run_context": context.run_dir / "run_context.yaml"})
        state.to_yaml(context.run_dir / "loop_state.yaml")
        orchestrator = cls(context, state)
        orchestrator.evidence_store.log_config(run_id, {"run_context": context.model_dump(mode="json")})
        orchestrator.event_log.append(
            run_id=run_id,
            event_type="run_initialized",
            stage="init",
            status="completed",
            message="Run context initialized.",
            artifacts={"run_context": context.run_dir / "run_context.yaml"},
        )
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
        context_path = self.context.to_yaml()
        self.context.to_json()
        return StageResult(
            stage="init",
            status="completed",
            message="Run context initialized.",
            artifacts={"run_context": context_path},
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
            artifacts={"policy_evaluation": path, "experiment_plan": experiment_plan_path},
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
        metrics = _read_metric_mapping(metrics_path)
        metric_records = _read_metric_records(metrics_path, self.context)
        output_path = self.context.artifact_path("metrics_import.json")
        _write_json(output_path, metrics)
        self.evidence_store.log_metrics(self.context.run_id, metrics)
        if metric_records:
            self.evidence_store.log_metric_records(self.context.run_id, metric_records)
        gate_path = self._write_evidence_status()
        return StageResult(
            stage="import_metrics",
            status="completed",
            message=f"Imported {len(metrics)} run metrics and {len(metric_records)} node metrics.",
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
        return path

    def _current_evidence_gate(self):
        evidence = self.evidence_store.load_run(self.context.run_id)
        extra = _loop_plan_evidence_required(self.context.artifact_path("loop_plan.yaml"))
        return EvidenceGate(default_loop_evidence_requirements(extra)).evaluate(
            evidence=evidence,
            artifacts=self.state.artifacts,
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


def _read_metric_mapping(path: Path) -> dict[str, MetricValue]:
    if path.suffix.lower() == ".csv":
        return _read_csv_metrics(path)
    data = _read_yaml(path) if path.suffix.lower() in {".yaml", ".yml"} else _read_json(path)
    if isinstance(data.get("metrics"), list):
        return _metric_records_to_mapping(_metric_records_from_items(data["metrics"], None))
    if isinstance(data.get("metric_records"), list):
        return _metric_records_to_mapping(_metric_records_from_items(data["metric_records"], None))
    if not isinstance(data, dict):
        raise ValueError("Metrics input must contain a mapping.")
    return {
        str(key): value
        for key, value in data.items()
        if isinstance(value, (float, int, str, bool)) or value is None
    }


def _read_csv_metrics(path: Path) -> dict[str, MetricValue]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return {}
    if {"metric_name", "value"}.issubset(rows[0]):
        return {
            str(row["metric_name"]): _coerce_metric_value(row["value"])
            for row in rows
            if row.get("metric_name")
        }
    if {"metric", "value"}.issubset(rows[0]):
        return {str(row["metric"]): _coerce_metric_value(row["value"]) for row in rows if row.get("metric")}
    return {key: _coerce_metric_value(value) for key, value in rows[0].items() if key}


def _read_metric_records(path: Path, context: RunContext) -> list[MetricEvidence]:
    """Read candidate/node metric records from CSV, YAML, or JSON input."""
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        return _metric_records_from_items(rows, context)

    data = _read_yaml(path) if path.suffix.lower() in {".yaml", ".yml"} else _read_json(path)
    if isinstance(data, dict):
        for key in ("metric_records", "metrics"):
            values = data.get(key)
            if isinstance(values, list):
                return _metric_records_from_items(values, context)
    if isinstance(data, list):
        return _metric_records_from_items(data, context)
    return []


def _metric_records_from_items(items: list[Any], context: RunContext | None) -> list[MetricEvidence]:
    records: list[MetricEvidence] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        metric_name = item.get("metric_name", item.get("metric"))
        if metric_name is None:
            continue
        candidate_id = item.get("candidate_id")
        node_id = item.get("node_id")
        if candidate_id is None and node_id is None:
            continue
        records.append(
            MetricEvidence(
                candidate_id=str(candidate_id or node_id),
                node_id=str(node_id or candidate_id),
                dataset_version=str(item.get("dataset_version") or (context.dataset_version if context else "unversioned")),
                split=str(item.get("split", "val")),
                metric_name=str(metric_name),
                value=_coerce_metric_value(item.get("value")),
                source=str(item.get("source", "import")),
            )
        )
    return records


def _metric_records_to_mapping(records: list[MetricEvidence]) -> dict[str, MetricValue]:
    metrics: dict[str, MetricValue] = {}
    for record in records:
        metrics.setdefault(record.metric_name, record.value)
    return metrics


def _coerce_metric_value(value: object) -> MetricValue:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() else number


def _loop_plan_evidence_required(path: Path) -> list[str]:
    if not path.is_file():
        return []
    raw = _read_yaml(path)
    values = raw.get("evidence_required", [])
    return [str(value) for value in values] if isinstance(values, list) else []


def _existing_artifact_items(context: RunContext) -> set[str]:
    """Return contract item names inferred from existing artifact files."""
    path_candidates = {
        "dataset_report": [context.artifact_path("dataset_report.json")],
        "label_quality_report": [context.artifact_path("annotation_advice.json")],
        "annotation_advice": [context.artifact_path("annotation_advice.json")],
        "loop_diagnosis": [context.artifact_path("loop_diagnosis.json")],
        "loop_plan": [context.artifact_path("loop_plan.yaml")],
        "policy_evaluation": [context.artifact_path("policy_evaluation.yaml")],
        "experiment_plan": [context.artifact_path("experiment_plan.yaml")],
        "candidate_plan": [context.run_dir / "plan.yaml", context.artifact_path("candidate_plan.yaml")],
        "ablation_plan": [context.run_dir / "ablation_plan.yaml", context.artifact_path("ablation_plan.yaml")],
        "smoke_result": [context.artifact_path("smoke_result.json")],
        "evidence_status": [context.artifact_path("evidence_status.json")],
        "next_round": [context.artifact_path("next_round.yaml")],
    }
    return {
        name
        for name, paths in path_candidates.items()
        if any(path.is_file() for path in paths)
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
