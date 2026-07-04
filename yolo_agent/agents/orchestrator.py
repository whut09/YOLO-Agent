"""Run orchestrator for the YOLO Agent loop harness."""

from __future__ import annotations

import time
from pathlib import Path

from yolo_agent.agents.active_learning import ActiveLearningPlan, LabelingTarget, MiningConfig
from yolo_agent.agents.loop_artifacts import LoopArtifacts
from yolo_agent.agents.loop_evidence import LoopEvidence
from yolo_agent.agents.loop_io import read_json, read_yaml, write_json
from yolo_agent.agents.loop_types import StageResult
from yolo_agent.agents.loop_policy_evaluator import LoopPolicyEvaluationReport
from yolo_agent.agents.next_round_forker import NextRoundForker
from yolo_agent.agents.run_initializer import RunInitializer
from yolo_agent.agents.stage_runner import StageRunner
from yolo_agent.core.evidence_contract import EvidenceContract
from yolo_agent.core.event_log import EventLog, EventType
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.execution_queue import ExecutionQueue, ExecutionQueueStore, QueueStatus
from yolo_agent.resources import ResourcePaths
from yolo_agent.core.executor import (
    BenchmarkImporter,
    DryRunExecutor,
    ExperimentExecutor,
    ShellExecutor,
    UltralyticsExecutor,
    UltralyticsTrainExecutor,
)
from yolo_agent.core.experiment_graph import ExperimentPlan
from yolo_agent.core.loop_state import LoopStage, LoopState, StageStatus
from yolo_agent.core.run_context import RunContext
from yolo_agent.core.run_lineage import RunLineageStore
from yolo_agent.core.stage_contract import LoopStageContracts, RetryPolicy


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
        self.evidence = LoopEvidence(self.context, self.state, self.evidence_store, self.lineage_store)
        self.artifacts = LoopArtifacts(
            context=self.context,
            state=self.state,
            policy=self.policy,
            evidence_store=self.evidence_store,
            evidence_status_writer=self.evidence.write_status,
        )
        self.stage_runner = StageRunner(
            context=self.context,
            policy=self.policy,
            evidence_store=self.evidence_store,
            evidence=self.evidence,
        )

    @classmethod
    def initialize(
        cls,
        run_id: str,
        task_path: Path | str,
        data_yaml: Path | str,
        run_root: Path | str = "runs",
        component_path: Path | str = ResourcePaths.COMPONENTS_DIR,
        search_space_path: Path | str = ResourcePaths.SEARCH_SPACE,
        loop_policy_path: Path | str = ResourcePaths.LOOP_POLICY,
        predictions_path: Path | str | None = None,
        detection_errors_path: Path | str | None = None,
        metrics_input_path: Path | str | None = None,
        training_config_path: Path | str | None = None,
        dataset_version: str = "unversioned",
        seed: int = 42,
    ) -> "LoopOrchestrator":
        """Create a run context, initial state, and evidence config."""
        initialization = RunInitializer().initialize(
            run_id=run_id,
            task_path=task_path,
            data_yaml=data_yaml,
            run_root=run_root,
            component_path=component_path,
            search_space_path=search_space_path,
            loop_policy_path=loop_policy_path,
            predictions_path=predictions_path,
            detection_errors_path=detection_errors_path,
            metrics_input_path=metrics_input_path,
            training_config_path=training_config_path,
            dataset_version=dataset_version,
            seed=seed,
        )
        context = initialization.context
        orchestrator = cls(context, initialization.state)
        orchestrator.evidence_store.log_config(run_id, {"run_context": context.model_dump(mode="json")})
        orchestrator.artifacts.record(
            "init",
            {
                "run_context": context.run_dir / "run_context.yaml",
                "loop_state": context.run_dir / "loop_state.yaml",
                "dataset_manifest": initialization.dataset_manifest_path,
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
                "dataset_manifest": initialization.dataset_manifest_path,
            },
        )
        orchestrator.evidence.record_lineage()
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

    def mine(
        self,
        predictions_path: Path | str,
        labeling_target: LabelingTarget = "generic",
        mining_config: MiningConfig | None = None,
    ) -> ActiveLearningPlan:
        """Run active-learning stages and return the mined plan."""
        self.context.predictions_path = Path(predictions_path)
        self.context.metadata["labeling_target"] = labeling_target
        self.context.to_yaml()
        self.context.to_json()
        if mining_config is not None:
            self.stage_runner.active_learning.mining_config = mining_config
        results = self.run_stages(["mine_samples", "label_handoff", "dataset_promote"])
        if results and results[-1].status in {"blocked", "failed"}:
            raise RuntimeError(results[-1].message)
        plan_path = self.context.artifact_path("active_learning_plan.json")
        return ActiveLearningPlan.model_validate(read_json(plan_path))

    def next_round(self) -> list[StageResult]:
        """Generate report and next-round checklist."""
        return self.run_stages(["report", "next_round"])

    def promote_dataset(self, reviewed_labels_path: Path | str | None = None) -> StageResult:
        """Evaluate dataset promotion with optional reviewed labels."""
        if reviewed_labels_path is not None:
            self.context.reviewed_labels_path = Path(reviewed_labels_path)
            self.context.to_yaml()
            self.context.to_json()
        return self.run_stage("dataset_promote")

    def fork_next(self, new_run_id: str) -> "LoopOrchestrator":
        """Materialize the next round into a fresh run that inherits loop context."""
        return NextRoundForker(self.context, self.policy).fork(new_run_id, LoopOrchestrator)

    def enqueue(self) -> ExecutionQueue:
        """Materialize the experiment plan into a persistent execution queue."""
        experiment_plan_path = self.context.artifact_path("experiment_plan.yaml")
        if not experiment_plan_path.is_file():
            raise FileNotFoundError(f"Missing experiment_plan.yaml: {experiment_plan_path}")
        plan = ExperimentPlan.from_yaml(experiment_plan_path)
        requires_evidence_by_node = self._queue_evidence_requirements(plan)
        queue = ExecutionQueueStore(self.context.run_dir).enqueue_from_plan(
            self.context.run_id,
            plan,
            requires_evidence_by_node=requires_evidence_by_node,
        )
        queue_path = self.context.run_dir / "execution_queue.yaml"
        self.evidence_store.log_artifact_manifest(
            run_id=self.context.run_id,
            name="execution_queue",
            artifact_path=queue_path,
            producer_stage="enqueue",
        )
        self.event_log.append(
            run_id=self.context.run_id,
            event_type="queue_enqueued",
            status="completed",
            message=f"Enqueued {len(queue.items)} experiment nodes.",
            artifacts={"execution_queue": queue_path},
            details={
                "counts": queue.counts(),
                "requires_evidence_by_node": requires_evidence_by_node,
            },
        )
        return queue

    def refresh_queue(self) -> ExecutionQueue:
        """Refresh needs_evidence queue items against current run evidence."""
        store = ExecutionQueueStore(self.context.run_dir)
        queue = store.load()
        missing_by_node = self._queue_missing_evidence(queue)
        queue, summary = store.refresh_needs_evidence(missing_by_node)
        queue_path = self.context.run_dir / "execution_queue.yaml"
        self.evidence_store.log_artifact_manifest(
            run_id=self.context.run_id,
            name="execution_queue",
            artifact_path=queue_path,
            producer_stage="queue_refresh",
        )
        self.event_log.append(
            run_id=self.context.run_id,
            event_type="queue_refreshed",
            status="completed",
            message=(
                f"Refreshed execution queue; unblocked={summary['unblocked']} "
                f"still_blocked={summary['still_blocked']}."
            ),
            artifacts={"execution_queue": queue_path},
            details={
                "counts": queue.counts(),
                "summary": summary,
                "missing_by_node": missing_by_node,
            },
        )
        return queue

    def _queue_evidence_requirements(self, plan: ExperimentPlan) -> dict[str, list[str]]:
        """Return missing evidence requirements for each planned node."""
        requirements = self._policy_evidence_requirements_by_node()
        evidence = self.evidence_store.load_run(self.context.run_id)
        by_node: dict[str, list[str]] = {}
        for node in plan.nodes:
            names = requirements.get(node.node_id, [])
            if not names:
                continue
            gate = EvidenceContract.from_names(names).evaluate(
                evidence=evidence,
                artifacts=self.state.artifacts,
            )
            if gate.missing_required:
                by_node[node.node_id] = gate.missing_required
        return by_node

    def _policy_evidence_requirements_by_node(self) -> dict[str, list[str]]:
        """Load accepted policy evidence requirements keyed by experiment node."""
        evaluation_path = self.context.artifact_path("policy_evaluation.yaml")
        if not evaluation_path.is_file():
            return {}
        report = LoopPolicyEvaluationReport.model_validate(read_yaml(evaluation_path))
        by_node: dict[str, list[str]] = {}
        for evaluation in report.evaluations:
            node = evaluation.experiment_node
            if node is None or not evaluation.evidence_required:
                continue
            by_node[node.node_id] = list(dict.fromkeys(evaluation.evidence_required))
        return by_node

    def _queue_missing_evidence(self, queue: ExecutionQueue) -> dict[str, list[str]]:
        """Return currently missing evidence keyed by needs_evidence queue node."""
        evidence = self.evidence_store.load_run(self.context.run_id)
        missing_by_node: dict[str, list[str]] = {}
        for item in queue.items:
            if item.status != "needs_evidence":
                continue
            gate = EvidenceContract.from_names(item.requires_evidence).evaluate(
                evidence=evidence,
                artifacts=self.state.artifacts,
            )
            if gate.missing_required:
                missing_by_node[item.node_id] = gate.missing_required
        return missing_by_node

    def execute_queue(self, executor_name: str = "dry-run") -> ExecutionQueue:
        """Execute queued items with an explicit executor."""
        executor = _executor_for_name(executor_name, self)
        store = ExecutionQueueStore(self.context.run_dir)
        queue = store.load()
        results_dir = self.context.artifact_path("execution_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        for item in list(queue.items):
            if item.status != "queued":
                continue
            item.mark_running()
            queue = store.update_item(item)
            self.event_log.append(
                run_id=self.context.run_id,
                event_type="queue_item_started",
                status="running",
                message=f"Executing queue item {item.queue_id}.",
                details={
                    "executor": executor_name,
                    "queue_id": item.queue_id,
                    "node_id": item.node_id,
                    "candidate_id": item.candidate_id,
                },
            )
            result = executor.execute(item.experiment_node, self.context.run_id, item.command)
            result_path = results_dir / f"{item.node_id}.json"
            write_json(result_path, result.model_dump(mode="json"))
            item.mark_result(result, result_path)
            queue = store.update_item(item)
            self.evidence_store.log_artifact_manifest(
                run_id=self.context.run_id,
                name=f"execution_result_{item.node_id}",
                artifact_path=result_path,
                producer_stage="execute_queue",
            )
            self.evidence_store.log_candidate_metrics(
                run_id=self.context.run_id,
                candidate_id=item.candidate_id,
                node_id=item.node_id,
                metrics={
                    "execution_duration_seconds": result.duration_seconds,
                    "execution_return_code": result.return_code,
                },
                dataset_version=item.experiment_node.data_version,
                source=f"executor:{executor_name}",
            )
            self.event_log.append(
                run_id=self.context.run_id,
                event_type=_queue_event_type(item.status),
                status=_stage_status_from_queue_status(item.status),
                message=item.message,
                artifacts={"execution_result": result_path},
                details={
                    "executor": executor_name,
                    "queue_id": item.queue_id,
                    "node_id": item.node_id,
                    "candidate_id": item.candidate_id,
                    "execution_status": result.status,
                },
            )
        self.evidence_store.log_artifact_manifest(
            run_id=self.context.run_id,
            name="execution_queue",
            artifact_path=self.context.run_dir / "execution_queue.yaml",
            producer_stage="execute_queue",
        )
        self.evidence_store.log_artifact_manifest(
            run_id=self.context.run_id,
            name="execution_results",
            artifact_path=results_dir,
            producer_stage="execute_queue",
        )
        return queue

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
        contract_result = self.artifacts.check_stage_contract(stage)
        if not contract_result.ok:
            artifacts = self.artifacts.blocked_contract_artifacts(stage)
            missing_items = [
                *contract_result.missing_required,
                *contract_result.missing_evidence,
                *contract_result.invalid_artifacts,
            ]
            result = StageResult(
                stage=stage,
                status="blocked",
                message="Missing or invalid required stage inputs: " + ", ".join(missing_items),
                artifacts=artifacts,
            )
            self.event_log.append(
                run_id=self.context.run_id,
                event_type="contract_blocked",
                stage=stage,
                status="blocked",
                message=result.message,
                details={
                    "missing_required": contract_result.missing_required,
                    "missing_evidence": contract_result.missing_evidence,
                    "invalid_artifacts": contract_result.invalid_artifacts,
                    "evidence_gate": (
                        contract_result.evidence_gate.model_dump(mode="json")
                        if contract_result.evidence_gate is not None
                        else None
                    ),
                },
                artifacts=artifacts,
            )
            self.artifacts.record(result.stage, result.artifacts)
            self.state.mark(result.stage, result.status, result.message, result.artifacts)
            self._save_state()
            return result
        retry_policy = self.policy.get(stage).retry_policy
        result = self._run_stage_with_retry(stage, retry_policy, contract_result.warnings)
        self.artifacts.record(result.stage, result.artifacts)
        self.state.mark(result.stage, result.status, result.message, result.artifacts)
        self._save_state()
        self.event_log.append(
            run_id=self.context.run_id,
            event_type=_event_type_for_status(result.status),
            stage=result.stage,
            status=result.status,
            message=result.message,
            artifacts=result.artifacts,
            details={"provides": self.artifacts.stage_provides(result.stage)},
        )
        return result

    def _run_stage_with_retry(
        self,
        stage: LoopStage,
        retry_policy: RetryPolicy,
        contract_warnings: list[str],
    ) -> StageResult:
        """Run one stage, retrying failed attempts according to stage policy."""
        max_attempts = retry_policy.max_attempts
        result = StageResult(stage=stage, status="failed", message="Stage did not run.")
        for attempt in range(1, max_attempts + 1):
            self.state.mark(stage, "running", f"Running {stage} (attempt {attempt}/{max_attempts}).")
            self._save_state()
            self.event_log.append(
                run_id=self.context.run_id,
                event_type="stage_started",
                stage=stage,
                status="running",
                message=f"Running {stage} (attempt {attempt}/{max_attempts}).",
                details={
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "retry_backoff": retry_policy.backoff,
                    "contract_warnings": contract_warnings,
                },
            )
            try:
                result = self.stage_runner.run(stage)
            except Exception as exc:  # pragma: no cover - defensive state guard
                result = StageResult(stage=stage, status="failed", message=str(exc))
            if result.status != "failed" or attempt >= max_attempts:
                return result
            delay_seconds = _retry_delay_seconds(retry_policy, attempt)
            self.event_log.append(
                run_id=self.context.run_id,
                event_type="stage_failed",
                stage=stage,
                status="failed",
                message=f"Attempt {attempt}/{max_attempts} failed; retrying {stage}.",
                details={
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "retry_backoff": retry_policy.backoff,
                    "backoff_seconds": delay_seconds,
                    "failure_message": result.message,
                },
                artifacts=result.artifacts,
            )
            if delay_seconds > 0:
                time.sleep(delay_seconds)
        return result

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


def _retry_delay_seconds(policy: RetryPolicy, failed_attempt: int) -> float:
    """Return a bounded retry delay for a failed stage attempt."""
    if policy.backoff == "none":
        return 0.0
    if policy.backoff == "linear":
        return min(2.0, 0.1 * failed_attempt)
    if policy.backoff == "exponential":
        return min(5.0, 0.1 * (2 ** max(0, failed_attempt - 1)))
    return 0.0


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


def _executor_for_name(name: str, orchestrator: LoopOrchestrator | None = None) -> ExperimentExecutor:
    """Return an executor by explicit CLI name."""
    if name == "dry-run":
        return DryRunExecutor()
    if name == "ultralytics-train":
        if orchestrator is None:
            return UltralyticsTrainExecutor()
        training_config = _training_config_from_context(orchestrator.context)
        return UltralyticsTrainExecutor(
            evidence_store=orchestrator.evidence_store,
            training_config=training_config,
            data_path=orchestrator.context.data_yaml,
        )
    if name == "ultralytics":
        return UltralyticsExecutor()
    if name == "shell":
        return ShellExecutor()
    raise ValueError(f"Unknown executor: {name}")


def _training_config_from_context(context: RunContext) -> object | None:
    from yolo_agent.adapters.ultralytics.training import UltralyticsTrainingConfig

    raw_path = context.metadata.get("training_config_path")
    if isinstance(raw_path, str) and raw_path:
        path = Path(raw_path)
        if path.is_file():
            return UltralyticsTrainingConfig.from_yaml(path)
    model = str(context.metadata.get("training_model", "yolo26s.pt"))
    return UltralyticsTrainingConfig(model=model, data=context.data_yaml)


def _queue_event_type(status: QueueStatus) -> EventType:
    if status == "completed":
        return "queue_item_completed"
    if status == "failed":
        return "queue_item_failed"
    if status in {"skipped", "needs_evidence"}:
        return "queue_item_skipped"
    return "queue_item_failed"


def _stage_status_from_queue_status(status: QueueStatus) -> StageStatus:
    if status == "completed":
        return "completed"
    if status == "failed":
        return "failed"
    if status in {"skipped", "needs_evidence"}:
        return "skipped"
    if status == "running":
        return "running"
    return "completed"


