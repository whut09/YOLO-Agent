"""Automatic pilot optimization loop driver.

The driver connects completed pilot evidence to the guarded loop machinery:

PilotResult -> LLMAnalysis -> PolicyProposal -> GuardedCandidate -> PilotRun -> DeltaAnalysis

It intentionally does not pretend that every metadata proposal can be trained.
Each accepted candidate is classified before queue materialization so the loop
only executes candidates backed by real adapter support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_serializer

from yolo_agent.adapters.ultralytics.training import TrainingBudgetProfileName, UltralyticsTrainingConfig
from yolo_agent.agents.error_driven_loop import ErrorDrivenLoopEngine
from yolo_agent.agents.error_to_action import DetectionErrorObservation, DetectionErrorType
from yolo_agent.agents.loop_io import read_json, read_yaml, write_json, write_yaml
from yolo_agent.agents.loop_policy_evaluator import LoopPolicyEvaluationReport
from yolo_agent.agents.orchestrator import LoopOrchestrator, TrainingLoopResult
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.error_facts import ErrorFact, ErrorFactStore
from yolo_agent.core.experiment_graph import Evidence, ExperimentNode, ExperimentPlan
from yolo_agent.core.task_spec import TaskSpec
from yolo_agent.tools.dataset_stats import DatasetReport


CandidateExecutionClass = Literal["executable", "recommendation_only", "adapter_required"]


ADAPTER_REQUIRED_COMPONENT_PREFIXES = (
    "loss.bbox.",
    "head.",
    "neck.",
    "assigner.",
    "backbone.",
    "backbone_block.",
)

SAFE_ULTRALYTICS_OVERRIDE_KEYS = {
    "optimizer",
    "patience",
    "amp",
    "workers",
    "device",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "warmup_momentum",
    "warmup_bias_lr",
    "box",
    "cls",
    "dfl",
    "mosaic",
    "mixup",
    "copy_paste",
    "close_mosaic",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "erasing",
    "crop_fraction",
    "target_actions",
}

NON_TRAINING_DOMAINS = {"data", "label", "postprocess", "evidence"}


class CandidateExecutionAssessment(BaseModel):
    """Whether a guarded candidate can be executed by the current harness."""

    policy_id: str
    candidate_id: str | None = None
    node_id: str | None = None
    execution_class: CandidateExecutionClass
    command_type: str | None = None
    action_domain: str = ""
    action_id: str | None = None
    reasons: list[str] = Field(default_factory=list)
    required_adapters: list[str] = Field(default_factory=list)
    command: str = ""


class AutoRoundResult(BaseModel):
    """One automatic optimization round."""

    round_index: int
    run_id: str
    run_dir: Path
    parent_run_id: str
    status: Literal["completed", "blocked", "failed", "skipped"] = "completed"
    stop_reason: str = ""
    llm_decision_path: Path | None = None
    doctor_report_path: Path | None = None
    policy_evaluation_path: Path | None = None
    auto_round_summary_path: Path
    next_round_path: Path | None = None
    training_loop: TrainingLoopResult | None = None
    candidate_assessments: list[CandidateExecutionAssessment] = Field(default_factory=list)

    @field_serializer(
        "run_dir",
        "llm_decision_path",
        "doctor_report_path",
        "policy_evaluation_path",
        "auto_round_summary_path",
        "next_round_path",
    )
    def serialize_path(self, value: Path | None) -> str | None:
        """Serialize paths portably."""
        return value.as_posix() if value is not None else None

    @property
    def executable_count(self) -> int:
        """Return how many accepted candidates can truly execute."""
        return sum(1 for item in self.candidate_assessments if item.execution_class == "executable")


class AutoOptimizationResult(BaseModel):
    """Summary for an automatic pilot optimization loop."""

    base_run_id: str
    base_run_dir: Path
    requested_rounds: int
    executed: bool
    profile: TrainingBudgetProfileName = "pilot"
    rounds: list[AutoRoundResult] = Field(default_factory=list)
    stopped_reason: str = ""
    summary_path: Path
    full_candidate_recommendations_path: Path

    @field_serializer("base_run_dir", "summary_path", "full_candidate_recommendations_path")
    def serialize_path(self, value: Path) -> str:
        """Serialize paths portably."""
        return value.as_posix()


class AutoOptimizationLoopDriver:
    """Drive bounded automatic pilot rounds from error facts and guarded policy evaluation."""

    def run(
        self,
        base_run_dir: Path | str,
        auto_rounds: int,
        *,
        execute: bool,
        executor: str,
        max_steps: int = 8,
        auto_import: bool = True,
        profile: TrainingBudgetProfileName = "pilot",
    ) -> AutoOptimizationResult:
        """Run up to ``auto_rounds`` child pilot rounds."""
        base_orchestrator = LoopOrchestrator.from_run_dir(base_run_dir)
        base_context = base_orchestrator.context
        summary_path = base_context.artifact_path("auto_optimization_summary.md")
        recommendations_path = base_context.artifact_path("full_candidate_recommendations.yaml")
        result = AutoOptimizationResult(
            base_run_id=base_context.run_id,
            base_run_dir=base_context.run_dir,
            requested_rounds=auto_rounds,
            executed=execute,
            profile=profile,
            summary_path=summary_path,
            full_candidate_recommendations_path=recommendations_path,
        )
        if auto_rounds <= 0:
            result.stopped_reason = "auto_rounds_zero"
            _write_final_outputs(result)
            return result

        parent = base_orchestrator
        for round_index in range(1, auto_rounds + 1):
            parent_next = _ensure_next_round(parent)
            parent_facts = ErrorFactStore(parent.context.run_root).read(parent.context.run_id)
            if not parent_facts:
                round_result = _empty_round(
                    round_index=round_index,
                    parent=parent,
                    status="blocked",
                    stop_reason="missing_error_facts",
                )
                result.rounds.append(round_result)
                result.stopped_reason = "missing_error_facts"
                break

            child_run_id = f"{base_context.run_id}-r{round_index}"
            child = _fork_or_load_child(parent, child_run_id)
            _prepare_child_training_context(child, parent, profile)
            _inherit_parent_metric_evidence(child, parent)
            round_result = self._run_one_round(
                round_index=round_index,
                parent=parent,
                child=child,
                parent_facts=parent_facts,
                parent_next_round=parent_next,
                execute=execute,
                executor=executor,
                max_steps=max_steps,
                auto_import=auto_import,
                profile=profile,
            )
            result.rounds.append(round_result)
            if round_result.status != "completed" or round_result.stop_reason in {
                "no_guarded_candidates",
                "no_executable_candidates",
                "queue_blocked",
                "training_failed",
            }:
                result.stopped_reason = round_result.stop_reason or round_result.status
                break
            parent = child
        if not result.stopped_reason:
            result.stopped_reason = "requested_rounds_completed"
        _write_final_outputs(result)
        return result

    def _run_one_round(
        self,
        *,
        round_index: int,
        parent: LoopOrchestrator,
        child: LoopOrchestrator,
        parent_facts: list[ErrorFact],
        parent_next_round: dict[str, Any],
        execute: bool,
        executor: str,
        max_steps: int,
        auto_import: bool,
        profile: TrainingBudgetProfileName,
    ) -> AutoRoundResult:
        """Run one child round through diagnosis, policy evaluation, and pilot execution."""
        status: Literal["completed", "blocked", "failed", "skipped"] = "completed"
        stop_reason = ""
        training_loop: TrainingLoopResult | None = None

        diagnosis_path = _ensure_loop_diagnosis_from_error_facts(child, parent_facts, parent_next_round)
        for stage in ["generate_loop_plan", "evaluate_policies", "generate_candidates", "ablate"]:
            stage_result = child.run_stage(stage)  # type: ignore[arg-type]
            if stage_result.status in {"blocked", "failed"}:
                status = stage_result.status
                stop_reason = f"{stage}_{stage_result.status}"
                break

        assessments = _assess_policy_evaluation(child.context.artifact_path("policy_evaluation.yaml"))
        if status == "completed":
            if not assessments:
                status = "blocked"
                stop_reason = "no_guarded_candidates"
            else:
                executable_nodes = _executable_nodes(child.context.artifact_path("experiment_plan.yaml"), assessments)
                if not executable_nodes:
                    status = "blocked"
                    stop_reason = "no_executable_candidates"
                else:
                    _write_filtered_experiment_plan(child, executable_nodes, assessments)
                    if execute:
                        training_loop = child.run_training_loop(
                            profile=profile,
                            executor=executor,
                            max_steps=max_steps,
                            auto_import=auto_import,
                        )
                        if not training_loop.completed:
                            status = "blocked"
                            stop_reason = (
                                "training_failed"
                                if training_loop.queue_counts.get("failed", 0)
                                else "queue_blocked"
                            )
                    else:
                        training_loop = child.run_training_loop(
                            profile=profile,
                            executor="dry-run",
                            max_steps=max_steps,
                            auto_import=auto_import,
                        )
                    child.next_round()

        next_round_path = child.context.artifact_path("next_round.yaml")
        round_result = AutoRoundResult(
            round_index=round_index,
            run_id=child.context.run_id,
            run_dir=child.context.run_dir,
            parent_run_id=parent.context.run_id,
            status=status,
            stop_reason=stop_reason or "round_completed",
            llm_decision_path=_existing_or_none(child.context.artifact_path("llm_decision.yaml")),
            doctor_report_path=diagnosis_path,
            policy_evaluation_path=_existing_or_none(child.context.artifact_path("policy_evaluation.yaml")),
            auto_round_summary_path=child.context.artifact_path("auto_round_summary.yaml"),
            next_round_path=_existing_or_none(next_round_path),
            training_loop=training_loop,
            candidate_assessments=assessments,
        )
        write_yaml(round_result.auto_round_summary_path, round_result.model_dump(mode="json"))
        child.evidence_store.log_artifact_manifest(
            run_id=child.context.run_id,
            name="auto_round_summary",
            artifact_path=round_result.auto_round_summary_path,
            producer_stage="auto_optimization_loop",
        )
        return round_result


def assess_candidate_execution(report: LoopPolicyEvaluationReport) -> list[CandidateExecutionAssessment]:
    """Classify guarded policy evaluations by real execution support."""
    assessments: list[CandidateExecutionAssessment] = []
    for evaluation in report.evaluations:
        if evaluation.decision != "accepted" or evaluation.candidate_config is None:
            continue
        candidate = evaluation.candidate_config
        node = evaluation.experiment_node
        command = node.command_spec if node is not None else None
        execution_class: CandidateExecutionClass = "executable"
        reasons: list[str] = []
        required_adapters: list[str] = []

        if command is None:
            execution_class = "recommendation_only"
            reasons.append("accepted policy has no command_spec")
        elif command.command_type != "train":
            execution_class = "recommendation_only"
            reasons.append(f"command_type={command.command_type} is not a pilot training command")

        if candidate.action_domain in NON_TRAINING_DOMAINS:
            execution_class = "recommendation_only"
            reasons.append(f"action_domain={candidate.action_domain} is advisory or evidence-first")

        component_adapters = _required_component_adapters(candidate.components)
        if component_adapters:
            execution_class = "adapter_required"
            required_adapters.extend(component_adapters)
            reasons.append("candidate uses metadata-only model components")

        unsupported_overrides = (
            _unsupported_train_overrides(candidate.train_overrides)
            if command is not None
            and command.command_type == "train"
            and candidate.action_domain not in NON_TRAINING_DOMAINS
            else []
        )
        if unsupported_overrides:
            execution_class = "adapter_required"
            required_adapters.extend(f"ultralytics_override:{key}" for key in unsupported_overrides)
            reasons.append("candidate train_overrides are not mapped to safe Ultralytics CLI options")

        if execution_class == "executable" and not reasons:
            reasons.append("train command uses only currently supported Ultralytics CLI options")

        assessments.append(
            CandidateExecutionAssessment(
                policy_id=evaluation.policy_id,
                candidate_id=candidate.candidate_id,
                node_id=node.node_id if node is not None else None,
                execution_class=execution_class,
                command_type=command.command_type if command is not None else None,
                action_domain=candidate.action_domain,
                action_id=candidate.action_id,
                reasons=list(dict.fromkeys(reasons)),
                required_adapters=list(dict.fromkeys(required_adapters)),
                command=command.display() if command is not None else "",
            )
        )
    return assessments


def _ensure_next_round(orchestrator: LoopOrchestrator) -> dict[str, Any]:
    path = orchestrator.context.artifact_path("next_round.yaml")
    if not path.is_file():
        orchestrator.next_round()
    return read_yaml(path) if path.is_file() else {}


def _fork_or_load_child(parent: LoopOrchestrator, child_run_id: str) -> LoopOrchestrator:
    child_dir = parent.context.run_root / child_run_id
    if child_dir.exists():
        return LoopOrchestrator.from_run_dir(child_dir)
    return parent.fork_next(child_run_id)


def _prepare_child_training_context(
    child: LoopOrchestrator,
    parent: LoopOrchestrator,
    profile: TrainingBudgetProfileName,
) -> None:
    parent_meta = parent.context.metadata
    child.context.metadata["training_profile"] = profile
    for key in ("training_config_path", "training_model"):
        if key in parent_meta and key not in child.context.metadata:
            child.context.metadata[key] = parent_meta[key]
    child.context.metadata["auto_optimization_round"] = child.context.metadata.get("auto_optimization_round", "")
    child.context.to_yaml()
    child.context.to_json()


def _ensure_loop_diagnosis_from_error_facts(
    child: LoopOrchestrator,
    parent_facts: list[ErrorFact],
    parent_next_round: dict[str, Any],
) -> Path:
    diagnosis_path = child.context.artifact_path("loop_diagnosis.json")
    dataset_report_path = child.context.artifact_path("dataset_report.json")
    if not dataset_report_path.is_file():
        child.run_stage("profile_data")
    dataset_report = DatasetReport.model_validate(read_json(dataset_report_path))
    task_spec = TaskSpec.from_yaml(child.context.task_path)
    observations = _observations_from_error_facts(parent_facts, parent_next_round)
    training_config = _training_config_from_context(child)
    report = ErrorDrivenLoopEngine().run(
        task_spec=task_spec,
        dataset_report=dataset_report,
        detection_errors=observations,
        evidence_status=_evidence_status_from_parent(child, parent_facts),
        fixed_imgsz=training_config.imgsz if training_config is not None else None,
    )
    write_json(diagnosis_path, report.model_dump(mode="json"))
    child.artifacts.record("diagnose_errors", {"loop_diagnosis": diagnosis_path})
    child.state.mark(
        "diagnose_errors",
        "completed",
        f"Created loop diagnosis from {len(parent_facts)} parent error facts.",
        {"loop_diagnosis": diagnosis_path},
    )
    child.state.to_yaml(child.context.run_dir / "loop_state.yaml")
    child.event_log.append(
        run_id=child.context.run_id,
        event_type="stage_completed",
        stage="diagnose_errors",
        status="completed",
        message="Created loop diagnosis from parent error facts for auto optimization.",
        artifacts={"loop_diagnosis": diagnosis_path},
        details={
            "parent_error_fact_count": len(parent_facts),
            "observation_count": len(observations),
        },
    )
    return diagnosis_path


def _inherit_parent_metric_evidence(child: LoopOrchestrator, parent: LoopOrchestrator) -> None:
    """Copy parent metric records into the child as inherited context evidence."""
    parent_evidence = parent.evidence_store.load_run(parent.context.run_id)
    if parent_evidence.metrics:
        child.evidence_store.log_metrics(child.context.run_id, parent_evidence.metrics)
    if parent_evidence.metric_records:
        child.evidence_store.log_metric_records(
            child.context.run_id,
            [
                record.model_copy(
                    update={
                        "source": f"inherited:{parent.context.run_id}:{record.source}",
                        "validator": record.validator or "inherited_parent_evidence",
                    }
                )
                for record in parent_evidence.metric_records
            ],
        )
    child.event_log.append(
        run_id=child.context.run_id,
        event_type="stage_completed",
        stage="init",
        status="completed",
        message="Inherited parent metric evidence for auto optimization planning.",
        details={
            "parent_run_id": parent.context.run_id,
            "run_metric_count": len(parent_evidence.metrics),
            "metric_record_count": len(parent_evidence.metric_records),
        },
    )


def _evidence_status_from_parent(child: LoopOrchestrator, parent_facts: list[ErrorFact]) -> dict[str, str]:
    """Build evidence status for the diagnosis engine from inherited parent evidence."""
    evidence = child.evidence_store.load_run(child.context.run_id)
    present = _present_metric_names(evidence)
    status: dict[str, str] = {name: "present" for name in present}
    if child.context.artifact_path("dataset_report.json").is_file():
        status["dataset_report"] = "present"
    if child.context.artifact_path("annotation_advice.json").is_file():
        status["label_quality_report"] = "present"
    if parent_facts:
        status.update(
            {
                "error_facts": "present",
                "localization_error_rate": "present",
                "false_negative_count": "present",
                "false_positive_count": "present",
                "class_confusion_pairs": "present",
                "confusion_matrix": "present",
            }
        )
    for fact in parent_facts:
        if fact.metric_name:
            status[str(fact.metric_name)] = "present"
        if fact.fact_type == "area_metric" and fact.area:
            status[f"ap_{fact.area}"] = "present"
        if fact.metric_name == "per_class_ap":
            status["per_class_ap"] = "present"
        if fact.metric_name == "per_class_ar":
            status["per_class_ar"] = "present"
    return status


def _present_metric_names(evidence: Evidence) -> set[str]:
    names = {name for name, value in evidence.metrics.items() if value is not None}
    names.update(record.metric_name for record in evidence.metric_records if record.value is not None and record.verified)
    return names


def _observations_from_error_facts(
    facts: list[ErrorFact],
    parent_next_round: dict[str, Any],
) -> list[DetectionErrorObservation]:
    focus = parent_next_round.get("current_round_focus", [])
    focus_keys = {
        _focus_key(item)
        for item in focus
        if isinstance(item, dict)
    }
    selected = [
        fact
        for fact in sorted(facts, key=_fact_rank)
        if fact.severity in {"high", "medium"}
        and (not focus_keys or _fact_key(fact) in focus_keys)
    ]
    if not selected:
        selected = [fact for fact in sorted(facts, key=_fact_rank) if fact.severity in {"high", "medium"}]
    observations: list[DetectionErrorObservation] = []
    for fact in selected[:8]:
        observations.append(
            DetectionErrorObservation(
                error_type=_error_type_for_fact(fact),
                count=max(int(fact.count or 1), 1),
                severity=fact.severity,
                notes=[
                    f"fact_type={fact.fact_type}",
                    f"subject={fact.subject}",
                    f"metric={fact.metric_name or 'unknown'}",
                    f"actions={','.join(fact.action_candidates)}",
                ],
            )
        )
    return observations or [
        DetectionErrorObservation(
            error_type="out_of_distribution_miss",
            count=1,
            severity="medium",
            notes=["fallback observation because no medium/high error facts were selected"],
        )
    ]


def _error_type_for_fact(fact: ErrorFact) -> DetectionErrorType:
    if fact.fact_type in {"area_metric", "subset_performance"} and fact.area == "small":
        return "small_object_miss"
    if fact.fact_type == "false_negative_heavy_class":
        return "out_of_distribution_miss"
    if fact.fact_type == "localization_heavy_class":
        return "loose_box"
    if fact.fact_type == "background_false_positive_class":
        return "background_confusion"
    if fact.fact_type == "class_confusion_pair":
        return "class_confusion"
    if fact.fact_type in {"class_low_ap", "per_class_metric"}:
        if fact.metric_name == "per_class_ar":
            return "long_tail_bias"
        return "out_of_distribution_miss"
    return "out_of_distribution_miss"


def _fact_rank(fact: ErrorFact) -> tuple[int, int, float]:
    severity = {"high": 0, "medium": 1, "low": 2}[fact.severity]
    rank = fact.rank if fact.rank is not None else 999
    value = float(fact.value) if isinstance(fact.value, (int, float)) and not isinstance(fact.value, bool) else 999.0
    return (severity, rank, value)


def _focus_key(item: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(item.get("fact_type", "")),
        str(item.get("subject", "")),
        str(item.get("class_name", "")),
        str(item.get("area", "")),
        str(item.get("metric_name", "")),
    )


def _fact_key(fact: ErrorFact) -> tuple[str, str, str, str, str]:
    return (
        fact.fact_type,
        fact.subject,
        fact.class_name or "",
        fact.area or "",
        fact.metric_name or "",
    )


def _training_config_from_context(child: LoopOrchestrator) -> UltralyticsTrainingConfig | None:
    raw_path = child.context.metadata.get("training_config_path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_file():
        return None
    profile = child.context.metadata.get("training_profile")
    return UltralyticsTrainingConfig.from_yaml(
        path,
        budget_profile=profile if profile in {"debug", "pilot", "baseline_full", "baseline_confirm", "candidate_full"} else None,
    )


def _assess_policy_evaluation(path: Path) -> list[CandidateExecutionAssessment]:
    if not path.is_file():
        return []
    report = LoopPolicyEvaluationReport.model_validate(read_yaml(path))
    return assess_candidate_execution(report)


def _required_component_adapters(components: list[str]) -> list[str]:
    adapters: list[str] = []
    for component in components:
        if component.startswith(ADAPTER_REQUIRED_COMPONENT_PREFIXES):
            adapters.append(f"component_adapter:{component}")
    return adapters


def _unsupported_train_overrides(overrides: dict[str, Any]) -> list[str]:
    unsupported = []
    for key in overrides:
        if key in SAFE_ULTRALYTICS_OVERRIDE_KEYS:
            continue
        if key == "imgsz":
            continue
        unsupported.append(str(key))
    return unsupported


def _executable_nodes(path: Path, assessments: list[CandidateExecutionAssessment]) -> list[ExperimentNode]:
    if not path.is_file():
        return []
    executable_node_ids = {
        item.node_id
        for item in assessments
        if item.execution_class == "executable" and item.node_id is not None
    }
    plan = ExperimentPlan.from_yaml(path)
    return [node for node in plan.nodes if node.node_id in executable_node_ids]


def _write_filtered_experiment_plan(
    child: LoopOrchestrator,
    executable_nodes: list[ExperimentNode],
    assessments: list[CandidateExecutionAssessment],
) -> Path:
    source_path = child.context.artifact_path("experiment_plan.yaml")
    original = ExperimentPlan.from_yaml(source_path)
    filtered = ExperimentPlan(
        plan_id=f"{child.context.run_id}_auto_executable_pilot_plan",
        nodes=executable_nodes,
        metadata={
            **original.metadata,
            "source": "AutoOptimizationLoopDriver",
            "original_plan_id": original.plan_id,
            "original_node_count": len(original.nodes),
            "executable_node_count": len(executable_nodes),
            "candidate_execution_assessments": [item.model_dump(mode="json") for item in assessments],
        },
    )
    filtered.metadata["plan_hash"] = filtered.plan_hash()
    filtered.to_yaml(source_path)
    child.evidence_store.log_artifact_manifest(
        run_id=child.context.run_id,
        name="experiment_plan",
        artifact_path=source_path,
        producer_stage="auto_optimization_loop",
    )
    return source_path


def _empty_round(
    *,
    round_index: int,
    parent: LoopOrchestrator,
    status: Literal["completed", "blocked", "failed", "skipped"],
    stop_reason: str,
) -> AutoRoundResult:
    path = parent.context.artifact_path(f"auto_round_{round_index}_blocked.yaml")
    result = AutoRoundResult(
        round_index=round_index,
        run_id=parent.context.run_id,
        run_dir=parent.context.run_dir,
        parent_run_id=parent.context.run_id,
        status=status,
        stop_reason=stop_reason,
        auto_round_summary_path=path,
    )
    write_yaml(path, result.model_dump(mode="json"))
    return result


def _existing_or_none(path: Path) -> Path | None:
    return path if path.exists() else None


def _write_final_outputs(result: AutoOptimizationResult) -> None:
    result.summary_path.parent.mkdir(parents=True, exist_ok=True)
    recommendations = _full_candidate_recommendations(result)
    write_yaml(result.full_candidate_recommendations_path, recommendations)
    result.summary_path.write_text(_summary_markdown(result, recommendations), encoding="utf-8")


def _full_candidate_recommendations(result: AutoOptimizationResult) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for round_result in result.rounds:
        for assessment in round_result.candidate_assessments:
            if assessment.execution_class != "executable":
                continue
            items.append(
                {
                    "source_run_id": round_result.run_id,
                    "candidate_id": assessment.candidate_id,
                    "node_id": assessment.node_id,
                    "promotion_status": "pilot_only_evidence_required",
                    "next_profile": "candidate_full",
                    "requires": [
                        "candidate_promotion_gate_passed",
                        "baseline_trusted",
                        "3_seed_confirmation",
                        "explicit --confirm-full-run",
                    ],
                    "command_hint": (
                        f"yolo-agent optimize advance --run {result.base_run_dir} "
                        "--to-profile candidate_full --execute --confirm-full-run"
                    ),
                }
            )
    return {
        "schema_version": "full_candidate_recommendations.v1",
        "base_run_id": result.base_run_id,
        "stopped_reason": result.stopped_reason,
        "full_run_started": False,
        "recommendations": items,
        "adapter_required": [
            {
                "round": round_result.round_index,
                "run_id": round_result.run_id,
                **assessment.model_dump(mode="json"),
            }
            for round_result in result.rounds
            for assessment in round_result.candidate_assessments
            if assessment.execution_class == "adapter_required"
        ],
        "recommendation_only": [
            {
                "round": round_result.round_index,
                "run_id": round_result.run_id,
                **assessment.model_dump(mode="json"),
            }
            for round_result in result.rounds
            for assessment in round_result.candidate_assessments
            if assessment.execution_class == "recommendation_only"
        ],
    }


def _summary_markdown(result: AutoOptimizationResult, recommendations: dict[str, Any]) -> str:
    lines = [
        "# Auto Optimization Summary",
        "",
        f"- Base run: `{result.base_run_id}`",
        f"- Requested rounds: {result.requested_rounds}",
        f"- Executed training: {result.executed}",
        f"- Stop reason: `{result.stopped_reason}`",
        f"- Full run started: false",
        "",
        "## Rounds",
        "",
    ]
    if not result.rounds:
        lines.append("- No automatic rounds ran.")
    for round_result in result.rounds:
        counts = {
            "executable": round_result.executable_count,
            "adapter_required": sum(1 for item in round_result.candidate_assessments if item.execution_class == "adapter_required"),
            "recommendation_only": sum(1 for item in round_result.candidate_assessments if item.execution_class == "recommendation_only"),
        }
        lines.append(
            f"- Round {round_result.round_index}: `{round_result.run_id}` "
            f"status={round_result.status} stop={round_result.stop_reason} "
            f"executable={counts['executable']} adapter_required={counts['adapter_required']} "
            f"recommendation_only={counts['recommendation_only']}"
        )
    lines.extend(["", "## Full Candidate Recommendations", ""])
    recs = recommendations.get("recommendations", [])
    if not recs:
        lines.append("- No full candidates are recommended yet. Pilot evidence or adapters are still missing.")
    else:
        for item in recs:
            lines.append(
                f"- `{item.get('candidate_id')}` from `{item.get('source_run_id')}`: "
                "requires candidate promotion, trusted baseline, 3 seeds, and explicit full-run confirmation."
            )
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "The auto loop only runs pilot-safe executable candidates. Metadata-only components, label/data work, "
            "post-processing policies, and unsupported Ultralytics overrides are recorded as recommendations or "
            "adapter-required items instead of being fake-trained.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "AutoOptimizationLoopDriver",
    "AutoOptimizationResult",
    "AutoRoundResult",
    "CandidateExecutionAssessment",
    "assess_candidate_execution",
]
