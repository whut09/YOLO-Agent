"""Run orchestrator for the YOLO Agent loop harness."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.agents.loop_artifacts import LoopArtifacts
from yolo_agent.agents.loop_evidence import LoopEvidence
from yolo_agent.agents.loop_io import write_json
from yolo_agent.agents.loop_types import StageResult
from yolo_agent.agents.next_round_forker import NextRoundForker
from yolo_agent.agents.run_initializer import RunInitializer
from yolo_agent.agents.stage_runner import StageRunner
from yolo_agent.core.event_log import EventLog, EventType
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.execution_queue import ExecutionQueue, ExecutionQueueStore, QueueStatus
from yolo_agent.core.executor import (
    BenchmarkImporter,
    DryRunExecutor,
    ExperimentExecutor,
    ShellExecutor,
    UltralyticsExecutor,
)
from yolo_agent.core.experiment_graph import ExperimentPlan
from yolo_agent.core.loop_state import LoopStage, LoopState, StageStatus
from yolo_agent.core.run_context import RunContext
from yolo_agent.core.run_lineage import RunLineageStore
from yolo_agent.core.stage_contract import LoopStageContracts


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

    def next_round(self) -> list[StageResult]:
        """Generate report and next-round checklist."""
        return self.run_stages(["report", "next_round"])

    def fork_next(self, new_run_id: str) -> "LoopOrchestrator":
        """Materialize the next round into a fresh run that inherits loop context."""
        return NextRoundForker(self.context, self.policy).fork(new_run_id, LoopOrchestrator)

    def enqueue(self) -> ExecutionQueue:
        """Materialize the experiment plan into a persistent execution queue."""
        experiment_plan_path = self.context.artifact_path("experiment_plan.yaml")
        if not experiment_plan_path.is_file():
            raise FileNotFoundError(f"Missing experiment_plan.yaml: {experiment_plan_path}")
        plan = ExperimentPlan.from_yaml(experiment_plan_path)
        queue = ExecutionQueueStore(self.context.run_dir).enqueue_from_plan(self.context.run_id, plan)
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
            details={"counts": queue.counts()},
        )
        return queue

    def execute_queue(self, executor_name: str = "dry-run") -> ExecutionQueue:
        """Execute queued items with an explicit executor."""
        executor = _executor_for_name(executor_name)
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
            missing_items = [*contract_result.missing_required, *contract_result.invalid_artifacts]
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
                    "invalid_artifacts": contract_result.invalid_artifacts,
                },
                artifacts=artifacts,
            )
            self.artifacts.record(result.stage, result.artifacts)
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
            result = self.stage_runner.run(stage)
        except Exception as exc:  # pragma: no cover - defensive state guard
            result = StageResult(stage=stage, status="failed", message=str(exc))
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


def _executor_for_name(name: str) -> ExperimentExecutor:
    """Return an executor by explicit CLI name."""
    if name == "dry-run":
        return DryRunExecutor()
    if name == "ultralytics":
        return UltralyticsExecutor()
    if name == "shell":
        return ShellExecutor()
    raise ValueError(f"Unknown executor: {name}")


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


