"""One-command optimization entrypoints for common YOLO workflows."""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from yolo_agent.adapters.ultralytics.training import (
    TrainingBudgetProfileName,
    UltralyticsTrainingConfig,
    command_from_training_config,
)
from yolo_agent.adapters.ultralytics.baseline_acceptance import BaselineAcceptanceGate
from yolo_agent.agents.auto_optimization_loop import AutoOptimizationLoopDriver, AutoOptimizationResult
from yolo_agent.agents.asha_scheduler import ASHAScheduler, ASHAStudy
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.orchestrator import LoopOrchestrator, TrainingLoopResult
from yolo_agent.core.execution_queue import ExecutionQueue
from yolo_agent.core.experiment_graph import ExperimentNode, ExperimentPlan
from yolo_agent.core.optimization_budget import AutoOptimizationBudget
from yolo_agent.core.optimization_objective import (
    build_baseline_protocol_hash,
    evaluate_optimization_objective,
    parse_optimization_goal,
)
from yolo_agent.core.full_run_consent import (
    FullRunConsentDecision,
    FullRunConsentDriver,
    FullRunStageStatus,
)
from yolo_agent.core.process_probe import probe_command_process
from yolo_agent.core.run_migration import assess_run_protocol, write_migration_report
from yolo_agent.core.run_protocol import RunProtocolVersion, build_run_protocol_version
from yolo_agent.core.task_spec import MetricPriority, ScenarioHint, TaskSpec
from yolo_agent.research.snapshot import bind_research_snapshot
from yolo_agent.resources import ResourcePaths


OptimizeKind = Literal["coco", "custom"]


FULL_RUN_CONFIRMATION_PROFILES = {"baseline_full", "baseline_confirm", "candidate_full"}


COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


class PreflightCheck(BaseModel):
    """One optimize preflight check result."""

    name: str
    ok: bool
    level: Literal["info", "warning", "error"] = "info"
    message: str = ""


class OptimizeResult(BaseModel):
    """Summary returned by a one-command optimize run."""

    kind: OptimizeKind
    run_id: str
    run_dir: Path
    model: str | None = None
    data_yaml: Path | None = None
    profile: str
    executor: str
    executed: bool
    preflight: list[PreflightCheck] = Field(default_factory=list)
    task_path: Path
    experiment_plan_path: Path
    queue_path: Path
    report_path: Path | None = None
    queue_counts: dict[str, int] = Field(default_factory=dict)
    training_loop: TrainingLoopResult | None = None
    auto_optimization: AutoOptimizationResult | None = None
    optimization_budget: AutoOptimizationBudget | None = None
    profile_history: list[str] = Field(default_factory=list)
    next_action: str = ""
    migration_report_path: Path | None = None
    migration_suggested_run_id: str | None = None
    full_run_consent: FullRunConsentDecision | None = None
    full_run_status: FullRunStageStatus | None = None

    @property
    def ok(self) -> bool:
        """Return whether no hard preflight error occurred."""
        return not any(check.level == "error" and not check.ok for check in self.preflight)


class OptimizeRunner:
    """Run a friendly one-command baseline optimization entrypoint."""

    def advance(
        self,
        run_dir: Path | str,
        to_profile: TrainingBudgetProfileName,
        execute: bool = False,
        confirm_full_run: bool = False,
        auto_advance: bool = True,
        max_steps: int = 8,
        auto_import: bool = True,
        auto_rounds: int = 0,
    ) -> OptimizeResult:
        """Advance an existing optimize run to a new budget profile."""
        orchestrator = LoopOrchestrator.from_run_dir(run_dir)
        context = orchestrator.context
        previous = _previous_optimize_metadata(context.artifact_path("experiment_plan.yaml"))
        kind = _coerce_kind(previous.get("kind"))
        model = str(previous.get("model") or _previous_model(context.artifact_path("experiment_plan.yaml")))
        if not model:
            model = "yolo26n.pt" if kind == "coco" else "yolo11n.pt"
        training_config_path = Path(
            str(
                previous.get("training_config_path")
                or context.metadata.get("training_config_path")
                or ResourcePaths.YOLO26_COCO_GOAL
            )
        )
        preset = previous.get("preset")
        return self.run(
            kind=kind,
            model=model,
            data_yaml=context.data_yaml,
            run_id=context.run_id,
            run_root=context.run_root,
            goal=str(previous.get("goal") or "+2map"),
            profile=to_profile,
            execute=execute,
            confirm_full_run=confirm_full_run,
            auto_advance=auto_advance,
            training_config_path=training_config_path,
            dataset_manifest_mode=str(context.metadata.get("dataset_manifest_mode", "metadata")),
            component_path=context.component_path,
            search_space_path=context.search_space_path,
            loop_policy_path=context.loop_policy_path,
            preset_name=str(preset) if preset is not None else None,
            max_steps=max_steps,
            auto_import=auto_import,
            auto_rounds=auto_rounds,
        )

    def run(
        self,
        kind: OptimizeKind,
        model: str,
        data_yaml: Path | str,
        run_id: str,
        run_root: Path | str = "runs",
        goal: str = "+2map",
        profile: TrainingBudgetProfileName = "debug",
        execute: bool = False,
        training_config_path: Path | str = ResourcePaths.YOLO26_COCO_GOAL,
        dataset_manifest_mode: str = "metadata",
        component_path: Path | str = ResourcePaths.COMPONENTS_DIR,
        search_space_path: Path | str = ResourcePaths.SEARCH_SPACE,
        loop_policy_path: Path | str = ResourcePaths.LOOP_POLICY,
        preset_name: str | None = None,
        max_steps: int = 8,
        auto_import: bool = True,
        confirm_full_run: bool = False,
        auto_advance: bool = True,
        auto_rounds: int = 0,
    ) -> OptimizeResult:
        """Initialize, queue, and optionally execute a baseline optimization run."""
        data_path = Path(data_yaml)
        run_root_path = Path(run_root)
        run_dir = run_root_path / run_id
        preflight = optimize_preflight(kind, data_path, execute=execute)
        confirm_check = _confirm_full_run_check(
            profile,
            execute=execute,
            confirmed=confirm_full_run or (run_dir / "artifacts" / "full_run_consent.yaml").is_file(),
        )
        if confirm_check is not None:
            preflight.append(confirm_check)
        task_path = run_dir / "task.yaml"
        plan_path = run_dir / "artifacts" / "experiment_plan.yaml"
        queue_path = run_dir / "execution_queue.yaml"
        hard_error = any(check.level == "error" and not check.ok for check in preflight)

        if hard_error:
            return OptimizeResult(
                kind=kind,
                run_id=run_id,
                run_dir=run_dir,
                model=model,
                data_yaml=data_path,
                profile=profile,
                executor="ultralytics-train" if execute else "dry-run",
                executed=False,
                preflight=preflight,
                task_path=task_path,
                experiment_plan_path=plan_path,
                queue_path=queue_path,
                next_action=_preflight_next_action(preflight),
            )

        if (run_dir / "run_context.yaml").is_file():
            orchestrator = LoopOrchestrator.from_run_dir(run_dir)
            assessment = assess_run_protocol(orchestrator.context, orchestrator.evidence_store)
            if assessment.legacy_run:
                migration = write_migration_report(orchestrator.context, assessment)
                preflight.append(
                    PreflightCheck(
                        name="legacy_run",
                        ok=False,
                        level="error",
                        message=(
                            f"Run uses a legacy protocol ({', '.join(migration.reasons)}). "
                            f"Start a new isolated run-id: {migration.suggested_run_id}."
                        ),
                    )
                )
                return OptimizeResult(
                    kind=kind,
                    run_id=run_id,
                    run_dir=run_dir,
                    model=model,
                    data_yaml=data_path,
                    profile=profile,
                    executor="ultralytics-train" if execute else "dry-run",
                    executed=False,
                    preflight=preflight,
                    task_path=task_path,
                    experiment_plan_path=plan_path,
                    queue_path=queue_path,
                    migration_report_path=orchestrator.context.artifact_path("run_migration_report.yaml"),
                    migration_suggested_run_id=migration.suggested_run_id,
                    next_action="legacy_run_requires_isolated_run_id",
                )
        else:
            task_path.parent.mkdir(parents=True, exist_ok=True)
            _task_spec_for(kind, data_path, goal).to_yaml(task_path)
            orchestrator = LoopOrchestrator.initialize(
                run_id=run_id,
                task_path=task_path,
                data_yaml=data_path,
                run_root=run_root_path,
                component_path=component_path,
                search_space_path=search_space_path,
                loop_policy_path=loop_policy_path,
                training_config_path=training_config_path,
                training_profile=profile,
                dataset_version="coco2017" if kind == "coco" else "dataset-v1",
                dataset_manifest_mode=dataset_manifest_mode,
            )

        running_result = _existing_running_queue_result(
            kind=kind,
            run_id=run_id,
            run_dir=orchestrator.context.run_dir,
            requested_profile=profile,
            executor="ultralytics-train" if execute else "dry-run",
            preflight=preflight,
            task_path=task_path,
            plan_path=plan_path,
            queue_path=queue_path,
            execute=execute,
        )
        if running_result is not None:
            return running_result

        training_config = UltralyticsTrainingConfig.from_yaml(training_config_path, budget_profile=profile)
        nodes = _baseline_nodes(kind, model, profile, orchestrator.context.dataset_version)
        node = nodes[0]
        protocol_hash = build_baseline_protocol_hash(
            model=model,
            data_yaml=data_path,
            training_config=training_config,
            dataset_version=orchestrator.context.dataset_version,
            dataset_manifest_sha256=orchestrator.context.dataset_manifest_sha256,
        )
        objective = parse_optimization_goal(
            goal,
            baseline_run_id=run_id,
            baseline_candidate_id=_baseline_node(
                kind,
                model,
                "baseline_full",
                orchestrator.context.dataset_version,
                seed=1,
            ).candidate_config.candidate_id,
            baseline_protocol_hash=protocol_hash,
            defaults=_objective_defaults(training_config_path),
        )
        current_task = TaskSpec.from_yaml(orchestrator.context.task_path)
        if current_task.primary_metric.name != objective.primary_metric:
            current_task.model_copy(
                update={
                    "primary_metric": MetricPriority(
                        name=objective.primary_metric,
                        weight=current_task.primary_metric.weight,
                        goal="maximize",
                    )
                }
            ).to_yaml(orchestrator.context.task_path)
        objective_path = orchestrator.context.artifact_path("optimization_objective.yaml")
        objective.to_yaml(objective_path, exclude_none=True, sort_keys=False)
        consent_driver = FullRunConsentDriver(orchestrator.context.run_dir)
        if execute and confirm_full_run:
            consent_driver.grant(
                run_id=run_id,
                objective=objective,
                dataset_manifest_sha256=orchestrator.context.dataset_manifest_sha256,
            )
        objective_status = evaluate_optimization_objective(
            objective,
            run_root=orchestrator.context.run_root,
            base_run_id=run_id,
        )
        consent_decision = consent_driver.validate(
            run_id=run_id,
            objective=objective,
            dataset_manifest_sha256=orchestrator.context.dataset_manifest_sha256,
            objective_status=objective_status,
        )
        if execute and profile in FULL_RUN_CONFIRMATION_PROFILES and not consent_decision.allowed:
            preflight.append(
                PreflightCheck(
                    name="full_run_consent",
                    ok=False,
                    level="error",
                    message=consent_decision.reason,
                )
            )
            return OptimizeResult(
                kind=kind,
                run_id=run_id,
                run_dir=orchestrator.context.run_dir,
                model=model,
                data_yaml=data_path,
                profile=profile,
                executor="ultralytics-train",
                executed=False,
                preflight=preflight,
                task_path=task_path,
                experiment_plan_path=plan_path,
                queue_path=queue_path,
                next_action=consent_decision.reason,
                full_run_consent=consent_decision,
                full_run_status=_full_run_status(profile, None, consent_decision, "stopped", consent_decision.reason),
            )
        orchestrator.context.metadata.update(
            {
                "optimization_objective_path": objective_path.resolve().as_posix(),
                "optimization_objective_hash": objective.objective_hash,
                "baseline_protocol_hash": objective.baseline_protocol_hash,
            }
        )
        research_binding = bind_research_snapshot(
            run_root_path.parent / "research",
            expected_hash=orchestrator.context.metadata.get("research_snapshot_hash"),
            snapshot_path=orchestrator.context.metadata.get("research_snapshot_path"),
        )
        orchestrator.context.metadata.update(research_binding.model_dump(mode="json"))
        orchestrator.context.to_yaml()
        orchestrator.evidence_store.log_artifact_manifest(
            run_id=run_id,
            name="optimization_objective",
            artifact_path=objective_path,
            producer_stage="optimize_init",
        )
        run_protocol = build_run_protocol_version(
            model=model, context=orchestrator.context, training_config=training_config, profile=profile, seed=node.seed
        )
        _persist_run_protocol(orchestrator, run_protocol)
        for current_node in nodes:
            node_protocol = build_run_protocol_version(
                model=model,
                context=orchestrator.context,
                training_config=training_config,
                profile=profile,
                seed=current_node.seed,
            )
            command = command_from_training_config(
                current_node,
                training_config.model_copy(update={"model": model}),
                run_id=run_id,
                data_path=data_path,
            )
            command = command.model_copy(
                update={
                    "metadata": {
                        **command.metadata,
                        "optimization_objective_hash": objective.objective_hash,
                        "baseline_protocol_hash": objective.baseline_protocol_hash,
                        "optimization_primary_metric": objective.primary_metric,
                        "optimization_target_delta": objective.required_delta(),
                        "run_protocol_hash": node_protocol.protocol_hash,
                        "subset_manifest_sha256": node_protocol.subset_manifest_sha256,
                        "batch_policy_hash": node_protocol.batch_policy_hash,
                        "eval_protocol_hash": node_protocol.eval_protocol_hash,
                        "ultralytics_version": node_protocol.ultralytics_version,
                        "code_version": node_protocol.code_version,
                        "full_run_seed": current_node.seed,
                    }
                }
            )
            current_node.command = command.display()
            current_node.command_spec = command
        plan = ExperimentPlan(
            plan_id=f"{run_id}_optimize_{kind}_{profile}",
            nodes=nodes,
            metadata={
                "source": "OptimizeRunner",
                "kind": kind,
                "goal": goal,
                "optimization_objective_path": objective_path.as_posix(),
                "optimization_objective_hash": objective.objective_hash,
                "baseline_protocol_hash": objective.baseline_protocol_hash,
                "model": model,
                "data_yaml": data_path.as_posix(),
                "profile": profile,
                "training_config_path": Path(training_config_path).as_posix(),
                "execute": execute,
                "confirm_full_run": confirm_full_run,
                "preset": preset_name,
                "run_protocol_hash": run_protocol.protocol_hash,
            },
            run_protocol_hash=run_protocol.protocol_hash,
        )
        plan.metadata["plan_hash"] = plan.plan_hash()
        plan.to_yaml(plan_path)
        orchestrator.evidence_store.log_artifact_manifest(
            run_id=run_id,
            name="experiment_plan",
            artifact_path=plan_path,
            producer_stage="optimize",
        )
        training_loop = orchestrator.run_training_loop(
            profile=profile,
            executor="ultralytics-train" if execute else "dry-run",
            max_steps=max_steps,
            auto_import=auto_import,
        )
        report_path = orchestrator.context.run_dir / "report.md"
        next_action = _next_action(profile, execute, training_loop.queue_counts, orchestrator.context.run_dir)
        result = OptimizeResult(
            kind=kind,
            run_id=run_id,
            run_dir=orchestrator.context.run_dir,
            model=model,
            data_yaml=data_path,
            profile=profile,
            executor="ultralytics-train" if execute else "dry-run",
            executed=execute,
            preflight=preflight,
            task_path=task_path,
            experiment_plan_path=plan_path,
            queue_path=queue_path,
            report_path=report_path,
            queue_counts=training_loop.queue_counts,
            training_loop=training_loop,
            profile_history=[profile],
            next_action=next_action,
            full_run_consent=consent_decision if profile in FULL_RUN_CONFIRMATION_PROFILES else None,
            full_run_status=(
                _full_run_status(profile, nodes[-1].seed, consent_decision, "completed", "continue")
                if profile in FULL_RUN_CONFIRMATION_PROFILES
                else None
            ),
        )
        baseline_trusted_for_full = False
        if execute and profile == "baseline_confirm" and training_loop.completed:
            strict_acceptance_config = training_config.baseline_acceptance.model_copy(
                update={"require_artifact_provenance": True}
            )
            acceptance = BaselineAcceptanceGate(strict_acceptance_config).check(
                orchestrator.evidence_store.load_run(run_id),
                expected_dataset_manifest_sha256=orchestrator.context.dataset_manifest_sha256,
                actual_dataset_manifest_sha256=orchestrator.context.dataset_manifest_sha256,
            )
            BaselineAcceptanceGate(strict_acceptance_config).persist_decision(
                orchestrator.evidence_store,
                run_id,
                acceptance,
                dataset_version=orchestrator.context.dataset_version,
            )
            if not acceptance.baseline_trusted:
                result.next_action = "baseline_acceptance_failed:" + ",".join(acceptance.baseline_rejection_reason)
                result.full_run_status = _full_run_status(
                    "baseline_confirm",
                    nodes[-1].seed,
                    consent_decision,
                    "stopped",
                    "baseline_not_trusted",
                )
            else:
                baseline_trusted_for_full = True
                result.full_run_status = _full_run_status(
                    "baseline_acceptance",
                    None,
                    consent_decision,
                    "completed",
                    "continue",
                )
        next_profile = _next_auto_profile(profile, confirm_full_run=confirm_full_run)
        if _should_auto_advance(result, execute=execute, auto_advance=auto_advance) and next_profile is not None:
            advanced = self.run(
                kind=kind,
                model=model,
                data_yaml=data_path,
                run_id=run_id,
                run_root=run_root_path,
                goal=goal,
                profile=next_profile,
                execute=execute,
                training_config_path=training_config_path,
                dataset_manifest_mode=dataset_manifest_mode,
                component_path=component_path,
                search_space_path=search_space_path,
                loop_policy_path=loop_policy_path,
                preset_name=preset_name,
                max_steps=max_steps,
                auto_import=auto_import,
                confirm_full_run=confirm_full_run,
                auto_advance=auto_advance,
                auto_rounds=auto_rounds,
            )
            advanced.profile_history = [*result.profile_history, *advanced.profile_history]
            return advanced
        if baseline_trusted_for_full and _has_pending_full_candidate(orchestrator.context.run_dir):
            auto = AutoOptimizationLoopDriver().run(
                base_run_dir=orchestrator.context.run_dir,
                auto_rounds=objective.confirmation_seeds,
                execute=True,
                executor="ultralytics-train",
                max_steps=max_steps,
                auto_import=auto_import,
                profile="candidate_full",
                confirm_full_run=True,
            )
            result.auto_optimization = auto
            result.next_action = _auto_optimization_next_action(auto, result.next_action)
            result.full_run_status = _full_run_status(
                "candidate_full",
                None,
                consent_decision,
                "completed" if auto.rounds else "stopped",
                auto.stopped_reason,
            )
            return result
        if baseline_trusted_for_full:
            result.next_action = "no_candidate_full_pending"
            result.full_run_status = _full_run_status(
                "baseline_acceptance",
                None,
                consent_decision,
                "completed",
                "no_candidate_full_pending",
            )
        bounded_auto_rounds = _bounded_auto_rounds(
            run_root=run_root_path,
            run_id=run_id,
            requested_rounds=auto_rounds,
            safety_limit=objective.max_auto_rounds_safety,
        )
        if _should_run_auto_optimization(result, execute=execute, auto_rounds=bounded_auto_rounds):
            auto = AutoOptimizationLoopDriver().run(
                base_run_dir=orchestrator.context.run_dir,
                auto_rounds=bounded_auto_rounds,
                execute=execute,
                executor="ultralytics-train" if execute else "dry-run",
                max_steps=max_steps,
                auto_import=auto_import,
                profile="pilot",
                confirm_full_run=bool(consent_decision.allowed),
            )
            result.auto_optimization = auto
            result.next_action = _auto_optimization_next_action(auto, result.next_action)
        elif auto_rounds > 0 and bounded_auto_rounds == 0:
            result.next_action = "automatic optimization stopped: internal round safety cap reached"
        return result


def _persist_run_protocol(orchestrator: LoopOrchestrator, protocol: RunProtocolVersion) -> Path:
    """Persist the current run protocol and initialize protocol-bound ASHA state."""
    path = orchestrator.context.artifact_path("run_protocol.yaml")
    protocol.to_yaml(path)
    orchestrator.context.run_protocol_path = path
    orchestrator.context.run_protocol_hash = protocol.protocol_hash
    orchestrator.context.legacy_run = False
    orchestrator.context.metadata.update(
        {
            "run_protocol_schema_version": protocol.schema_version,
            "run_protocol_hash": protocol.protocol_hash,
            "post_eval_protocol_hash": protocol.eval_protocol_hash,
            "subset_manifest_sha256": protocol.subset_manifest_sha256,
            "batch_policy_hash": protocol.batch_policy_hash,
            "ultralytics_version": protocol.ultralytics_version,
            "code_version": protocol.code_version,
        }
    )
    orchestrator.context.to_yaml()
    orchestrator.context.to_json()
    asha_path = orchestrator.context.artifact_path("asha_state.yaml")
    if not asha_path.is_file():
        scheduler = ASHAScheduler.create(orchestrator.context.run_id)
        scheduler.study.run_protocol_hash = protocol.protocol_hash
        scheduler.study.to_yaml(asha_path)
    orchestrator.context.metadata["asha_state_path"] = asha_path.as_posix()
    orchestrator.context.to_yaml()
    orchestrator.context.to_json()
    orchestrator.evidence_store.log_artifact_manifest(
        run_id=orchestrator.context.run_id,
        name="run_protocol",
        artifact_path=path,
        producer_stage="optimize_init",
    )
    orchestrator.evidence_store.log_artifact_manifest(
        run_id=orchestrator.context.run_id,
        name="asha_state",
        artifact_path=asha_path,
        producer_stage="optimize_init",
    )
    return path


def optimize_preflight(kind: OptimizeKind, data_yaml: Path, execute: bool = False) -> list[PreflightCheck]:
    """Run best-effort checks before one-command optimization."""
    checks: list[PreflightCheck] = []
    checks.append(
        PreflightCheck(
            name="python",
            ok=sys.version_info >= (3, 10),
            level="error" if sys.version_info < (3, 10) else "info",
            message=f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )
    )
    checks.append(
        PreflightCheck(
            name="data_yaml",
            ok=data_yaml.is_file(),
            level="error",
            message=f"data yaml {'found' if data_yaml.is_file() else 'missing'}: {data_yaml}",
        )
    )
    if data_yaml.is_file():
        dataset_root = _dataset_root(data_yaml)
        checks.append(
            PreflightCheck(
                name="dataset_root",
                ok=dataset_root.exists(),
                level="error" if not dataset_root.exists() else "info",
                message=str(dataset_root),
            )
        )
        if kind == "coco":
            checks.extend(_coco_path_checks(dataset_root))
        usage = shutil.disk_usage(dataset_root if dataset_root.exists() else data_yaml.parent)
        free_gb = usage.free / (1024**3)
        checks.append(
            PreflightCheck(
                name="disk_free",
                ok=free_gb >= 10,
                level="warning" if free_gb < 10 else "info",
                message=f"{free_gb:.1f} GB free",
            )
        )
    ultralytics_available = importlib.util.find_spec("ultralytics") is not None
    checks.append(
        PreflightCheck(
            name="ultralytics",
            ok=ultralytics_available,
            level="error" if execute and not ultralytics_available else ("info" if ultralytics_available else "warning"),
            message="installed" if ultralytics_available else "not installed; dry-run is still available",
        )
    )
    gpu_ok, gpu_message = _gpu_status()
    checks.append(
        PreflightCheck(
            name="gpu",
            ok=gpu_ok,
            level="error" if execute and not gpu_ok else ("info" if gpu_ok else "warning"),
            message=gpu_message,
        )
    )
    return checks


def _confirm_full_run_check(profile: str, execute: bool, confirmed: bool) -> PreflightCheck | None:
    if not execute or profile not in FULL_RUN_CONFIRMATION_PROFILES:
        return None
    return PreflightCheck(
        name="confirm_full_run",
        ok=confirmed,
        level="error" if not confirmed else "info",
        message=(
            f"profile {profile} is a full COCO training profile; rerun with --confirm-full-run "
            "to acknowledge the 100-epoch budget."
        )
        if not confirmed
        else f"profile {profile} full-run confirmation accepted",
    )


def _preflight_next_action(preflight: list[PreflightCheck]) -> str:
    if any(check.name == "confirm_full_run" and not check.ok for check in preflight):
        return "Fix preflight: add --confirm-full-run to execute this full COCO profile, or use debug/pilot first."
    return "Fix preflight errors and rerun the same optimize command."


def _should_auto_advance(result: OptimizeResult, execute: bool, auto_advance: bool) -> bool:
    if not execute or not auto_advance or not result.ok or result.training_loop is None:
        return False
    if not result.training_loop.completed:
        return False
    if result.queue_counts.get("failed", 0):
        return False
    blocked_statuses = ("running", "paused", "blocked_by_resource", "needs_resume", "needs_evidence")
    return not any(result.queue_counts.get(status, 0) for status in blocked_statuses)


def _should_run_auto_optimization(result: OptimizeResult, execute: bool, auto_rounds: int) -> bool:
    """Return whether a completed pilot should enter automatic candidate rounds."""
    if auto_rounds <= 0 or not execute or not result.ok or result.training_loop is None:
        return False
    if result.profile != "pilot":
        return False
    if not result.training_loop.completed:
        return False
    blocked_statuses = ("running", "paused", "blocked_by_resource", "needs_resume", "needs_evidence", "failed")
    return not any(result.queue_counts.get(status, 0) for status in blocked_statuses)


def _bounded_auto_rounds(
    *,
    run_root: Path,
    run_id: str,
    requested_rounds: int,
    safety_limit: int,
) -> int:
    """Keep the absolute child round index within the final safety limit."""
    latest_round = 0
    if run_root.is_dir():
        pattern = re.compile(re.escape(run_id) + r"-r(?P<round>\d+)$")
        for path in run_root.iterdir():
            if not path.is_dir():
                continue
            match = pattern.fullmatch(path.name)
            if match:
                latest_round = max(latest_round, int(match.group("round")))
    remaining = max(0, safety_limit - latest_round)
    return min(max(0, requested_rounds), remaining)


def _auto_optimization_next_action(auto: AutoOptimizationResult, fallback: str) -> str:
    """Return a concise next action after automatic optimization rounds."""
    if auto.stopped_reason == "objective_confirmed":
        return "Optimization objective is confirmed; review the full candidate recommendation and Pareto guards."
    if auto.stopped_reason == "target_reached_pending_full_confirmation":
        return (
            "Pilot reached the objective; the selected candidate now needs full-budget "
            "and multi-seed confirmation."
        )
    if auto.stopped_reason == "target_reached_pending_guard_evidence":
        return (
            "Accuracy target was reached, but latency/model-size evidence is incomplete "
            "or outside the objective guards."
        )
    if auto.stopped_reason in {
        "gpu_budget_exhausted",
        "max_pilot_rounds_reached",
        "no_improvement_patience_reached",
        "no_improvement_patience",
        "family_exhaustion",
    }:
        return (
            f"Automatic search stopped at its objective boundary: {auto.stopped_reason}. "
            f"Review {auto.summary_path}."
        )
    if auto.stopped_reason == "missing_error_facts":
        return (
            "Auto loop stopped before training a new candidate because COCO error facts are missing. "
            "Generate/import COCO error facts, then rerun yolo-agent train for the same run."
        )
    if auto.stopped_reason == "no_executable_candidates":
        return (
            "Auto loop produced guarded recommendations, but no currently executable candidate. "
            f"Review {auto.full_candidate_recommendations_path}."
        )
    if auto.rounds:
        return f"Auto loop stopped: {auto.stopped_reason}. Review {auto.summary_path}."
    return fallback


def _next_auto_profile(profile: str, confirm_full_run: bool) -> TrainingBudgetProfileName | None:
    if profile == "debug":
        return "pilot"
    if profile == "pilot" and confirm_full_run:
        return "baseline_full"
    if profile == "baseline_full" and confirm_full_run:
        return "baseline_confirm"
    return None


def _previous_optimize_metadata(plan_path: Path) -> dict[str, object]:
    if not plan_path.is_file():
        return {}
    plan = ExperimentPlan.from_yaml(plan_path)
    return dict(plan.metadata)


def _previous_model(plan_path: Path) -> str:
    if not plan_path.is_file():
        return ""
    plan = ExperimentPlan.from_yaml(plan_path)
    if not plan.nodes:
        return ""
    return plan.nodes[0].candidate_config.base_model


def _coerce_kind(value: object) -> OptimizeKind:
    return "custom" if value == "custom" else "coco"


def _task_spec_for(kind: OptimizeKind, data_yaml: Path, goal: str) -> TaskSpec:
    names = _class_names(data_yaml)
    if kind == "coco" and not names:
        names = COCO_NAMES
    if not names:
        names = ["object"]
    objective = parse_optimization_goal(
        goal,
        baseline_run_id="pending",
        baseline_candidate_id="pending",
        baseline_protocol_hash="pending",
    )
    return TaskSpec(
        task_type="detect",
        scene="generic",
        class_names=names,
        primary_metric=MetricPriority(name=objective.primary_metric),
        secondary_metrics=[
            MetricPriority(name="latency_ms", goal="minimize"),
            MetricPriority(name="model_size_mb", goal="minimize"),
        ],
        scenario_hint=ScenarioHint(
            name=f"{kind}_optimize",
            description=f"One-command optimize run targeting {goal}.",
            suggested_model_size="auto",
            notes=["Generated by yolo-agent train/optimize."],
        ),
    )


def _objective_defaults(training_config_path: Path | str) -> dict[str, object]:
    """Load objective guardrails from the training config goal section."""
    path = Path(training_config_path)
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    goal = raw.get("goal", {}) if isinstance(raw, dict) else {}
    if not isinstance(goal, dict):
        return {}
    mapping = {
        "fixed_imgsz": goal.get("fixed_imgsz", 640),
        "max_latency_regression": goal.get("max_latency_regression", 0.05),
        "max_model_size_regression": goal.get("max_model_size_regression", 0.10),
        "confirmation_seeds": goal.get("minimum_seeds", 3),
        "confidence_level": goal.get("confidence_level", 0.95),
        "max_gpu_hours": goal.get("max_gpu_hours", 24.0),
        "max_pilot_rounds": goal.get("max_pilot_rounds", 12),
        "no_improvement_patience": goal.get("no_improvement_patience", 4),
        "max_concurrent_pilots": goal.get("max_concurrent_pilots", 1),
        "max_auto_rounds_safety": goal.get("max_auto_rounds_safety", 60),
        "full_requires_confirmation": goal.get("full_requires_confirmation", True),
    }
    return {key: value for key, value in mapping.items() if value is not None}


def _baseline_node(
    kind: OptimizeKind,
    model: str,
    profile: str,
    dataset_version: str,
    seed: int = 1,
) -> ExperimentNode:
    stem = Path(model).stem.replace(".", "_").replace("-", "_")
    candidate_id = f"{stem}_{kind}_{profile}" + (f"_seed_{seed}" if seed != 1 else "")
    candidate = CandidateConfig(
        candidate_id=candidate_id,
        base_model=model,
        scale=_scale_from_model(model),
        framework="ultralytics",
        expected_effect=[f"{profile} baseline sanity run for {model}."],
        risk="low",
    )
    return ExperimentNode(
        node_id=f"node_{candidate_id}",
        candidate_config=candidate,
        data_version=dataset_version,
        seed=seed,
        changed_variables={},
    )


def _baseline_nodes(
    kind: OptimizeKind,
    model: str,
    profile: str,
    dataset_version: str,
) -> list[ExperimentNode]:
    seeds = [2, 3] if profile == "baseline_confirm" else [1]
    return [_baseline_node(kind, model, profile, dataset_version, seed=seed) for seed in seeds]


def _has_pending_full_candidate(run_dir: Path) -> bool:
    path = run_dir / "artifacts" / "asha_state.yaml"
    if not path.is_file():
        return False
    study = ASHAStudy.from_yaml(path)
    return any(trial.status == "full_pending_confirmation" for trial in study.trials)


def _full_run_status(
    profile: str,
    seed: int | None,
    decision: FullRunConsentDecision,
    progress: str,
    stop_reason: str,
) -> FullRunStageStatus:
    stage = (
        profile
        if profile in {"baseline_full", "baseline_confirm", "baseline_acceptance", "candidate_full"}
        else "stopped"
    )
    return FullRunStageStatus(
        stage=stage,
        seed=seed,
        progress=progress,
        gpu_hours_used=decision.gpu_hours_used,
        gpu_hours_authorized=(
            decision.consent.authorized_max_gpu_hours if decision.consent is not None else 0.0
        ),
        stop_reason=stop_reason,
    )


def _next_action(profile: str, execute: bool, counts: dict[str, int], run_dir: Path) -> str:
    if not execute:
        return f"Dry-run completed. Rerun with --execute to start the {profile} training command."
    if counts.get("running", 0):
        return f"Training is already running. Watch it with yolo-agent status --run {run_dir}."
    if counts.get("completed", 0):
        if profile == "debug":
            return "Debug execution completed. Auto-advance will continue to pilot when enabled."
        if profile == "pilot":
            return (
                "Pilot execution completed. Use pilot evidence for error diagnosis and pilot-only candidate proposals "
                "before any full COCO confirmation."
            )
        if profile == "baseline_full":
            return "Full baseline completed. Run baseline_confirm with --confirm-full-run for 3-seed confirmation."
        return "Execution completed. Inspect report.md and evidence_status.json."
    if counts.get("blocked_by_resource", 0) or counts.get("paused", 0):
        issue = _queue_blocked_issue(run_dir)
        if issue:
            return issue
        return "Execution was resource-blocked. Free GPU resources, then rerun the same yolo-agent train command."
    if counts.get("needs_resume", 0):
        return (
            "Execution is blocked because the queue expects a resume checkpoint. "
            "For debug runs, rerun the same train command to recover a stale queue and start fresh."
        )
    if counts.get("needs_evidence", 0):
        return "Execution is waiting for required evidence. Rerun yolo-agent train after evidence is available."
    if counts.get("queued", 0):
        return "Execution is queued and ready. Rerun yolo-agent train to start it."
    if counts.get("failed", 0):
        return "Execution failed. Inspect events.jsonl and artifacts/execution_results."
    if counts.get("skipped", 0):
        issue = _queue_skipped_issue(run_dir)
        if issue:
            return issue
        return "Execution was skipped by a guard. Inspect execution_queue.yaml."
    return "No queued item ran. Inspect execution_queue.yaml."


def _queue_blocked_issue(run_dir: Path) -> str:
    """Return a specific next action for the first resource-blocked queue item."""
    queue_path = run_dir / "execution_queue.yaml"
    if not queue_path.is_file():
        return ""
    try:
        queue = ExecutionQueue.from_yaml(queue_path)
    except Exception:
        return ""
    for item in queue.items:
        if item.status not in {"blocked_by_resource", "paused"}:
            continue
        blockers = set(item.resource_blockers)
        if "missing_batch_tuning_result" in blockers:
            profile = item.command.metadata.get("training_budget_profile") or item.command.metadata.get("profile") or "pilot"
            return (
                f"{profile} is waiting for batch tuning. Rerun the same train command; "
                "UltralyticsTrainExecutor will run BatchTuner first, then start training."
            )
        if blockers:
            return f"Execution is blocked by: {', '.join(sorted(blockers))}. Resolve it, then rerun yolo-agent train."
        if item.message:
            return item.message
    return ""


def _queue_skipped_issue(run_dir: Path) -> str:
    """Return a specific next action for skipped queue items."""
    queue_path = run_dir / "execution_queue.yaml"
    if not queue_path.is_file():
        return ""
    try:
        queue = ExecutionQueue.from_yaml(queue_path)
    except Exception:
        return ""
    for item in queue.items:
        if item.status != "skipped":
            continue
        profile = item.command.metadata.get("training_budget_profile") or item.command.metadata.get("profile") or "debug"
        if "Fast Baseline Gate blocked" in (item.message or ""):
            return (
                f"{profile} was skipped by Fast Baseline Gate. Rerun the same train command; "
                "the queue will be rebuilt and prior sanity evidence will be reused."
            )
        if item.message:
            return item.message
    return ""


def _existing_running_queue_result(
    *,
    kind: OptimizeKind,
    run_id: str,
    run_dir: Path,
    requested_profile: TrainingBudgetProfileName,
    executor: str,
    preflight: list[PreflightCheck],
    task_path: Path,
    plan_path: Path,
    queue_path: Path,
    execute: bool,
) -> OptimizeResult | None:
    """Return a non-mutating result when an existing run already has active training."""
    if not queue_path.is_file():
        return None
    try:
        queue = ExecutionQueue.from_yaml(queue_path)
    except Exception:
        return None
    counts = {key: int(value) for key, value in queue.counts().items()}
    active_statuses = ("running", "paused", "blocked_by_resource", "needs_resume", "needs_evidence")
    if not any(counts.get(status, 0) > 0 for status in active_statuses):
        return None
    running_profile = _running_queue_profile(queue) or requested_profile
    if _running_queue_is_stale(queue):
        if running_profile == requested_profile:
            return None
        return OptimizeResult(
            kind=kind,
            run_id=run_id,
            run_dir=run_dir,
            profile=running_profile,
            executor=executor,
            executed=execute,
            preflight=preflight,
            task_path=task_path,
            experiment_plan_path=plan_path,
            queue_path=queue_path,
            report_path=run_dir / "report.md",
            queue_counts=counts,
            training_loop=TrainingLoopResult(
                run_id=run_id,
                profile=running_profile,
                executor=executor,
                auto_import=True,
                max_steps=0,
                steps=[],
                queue_counts=counts,
                stopped_reason="queue_stale",
                completed=False,
            ),
            profile_history=[running_profile],
            next_action=(
                f"Stale {running_profile} queue detected. Rerun yolo-agent train for the same run "
                "to recover it before advancing."
            ),
        )
    if counts.get("running", 0) <= 0:
        if running_profile == requested_profile and _queue_has_only_batch_tuning_blocker(queue):
            return None
        next_action = _queue_blocked_issue(run_dir) or f"Rerun yolo-agent train after resolving queue blockers for {run_dir}."
        return OptimizeResult(
            kind=kind,
            run_id=run_id,
            run_dir=run_dir,
            profile=running_profile,
            executor=executor,
            executed=execute,
            preflight=preflight,
            task_path=task_path,
            experiment_plan_path=plan_path,
            queue_path=queue_path,
            report_path=run_dir / "report.md",
            queue_counts=counts,
            training_loop=TrainingLoopResult(
                run_id=run_id,
                profile=running_profile,
                executor=executor,
                auto_import=True,
                max_steps=0,
                steps=[],
                queue_counts=counts,
                stopped_reason="queue_blocked",
                completed=False,
            ),
            profile_history=[running_profile],
            next_action=next_action,
        )
    training_loop = TrainingLoopResult(
        run_id=run_id,
        profile=running_profile,
        executor=executor,
        auto_import=True,
        max_steps=0,
        steps=[],
        queue_counts=counts,
        stopped_reason="queue_running",
        completed=False,
    )
    return OptimizeResult(
        kind=kind,
        run_id=run_id,
        run_dir=run_dir,
        profile=running_profile,
        executor=executor,
        executed=execute,
        preflight=preflight,
        task_path=task_path,
        experiment_plan_path=plan_path,
        queue_path=queue_path,
        report_path=run_dir / "report.md",
        queue_counts=counts,
        training_loop=training_loop,
        profile_history=[running_profile],
        next_action=f"Training is already running. Watch it with yolo-agent status --run {run_dir}.",
    )


def _running_queue_profile(queue: ExecutionQueue) -> TrainingBudgetProfileName | None:
    """Return the training profile for the active queue item, if known."""
    for item in queue.items:
        if item.status not in {"running", "paused", "blocked_by_resource", "needs_resume", "needs_evidence"}:
            continue
        raw = item.command.metadata.get("training_budget_profile") or item.command.metadata.get("profile")
        if raw in {"debug", "pilot", "baseline_full", "baseline_confirm", "candidate_full"}:
            return raw  # type: ignore[return-value]
    return None


def _queue_has_only_batch_tuning_blocker(queue: ExecutionQueue) -> bool:
    """Return whether blocked items can be recovered by the Ultralytics executor."""
    blocked_items = [
        item
        for item in queue.items
        if item.status in {"blocked_by_resource", "paused"}
    ]
    if not blocked_items:
        return False
    return all(set(item.resource_blockers) == {"missing_batch_tuning_result"} for item in blocked_items)


def _running_queue_is_stale(queue: ExecutionQueue) -> bool:
    """Return whether every running train item has no matching local process."""
    running_train_items = [
        item for item in queue.items if item.status == "running" and item.command.command_type == "train"
    ]
    if not running_train_items:
        return False
    return all(probe_command_process(item.command).status == "not_found" for item in running_train_items)


def _class_names(data_yaml: Path) -> list[str]:
    if not data_yaml.is_file():
        return []
    raw = _read_yaml(data_yaml)
    names = raw.get("names")
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        return [str(value) for _, value in sorted(names.items(), key=lambda item: int(item[0]))]
    return []


def _dataset_root(data_yaml: Path) -> Path:
    raw = _read_yaml(data_yaml) if data_yaml.is_file() else {}
    configured = raw.get("path")
    if configured is None:
        return data_yaml.parent
    root = Path(str(configured))
    return root if root.is_absolute() else data_yaml.parent / root


def _coco_path_checks(dataset_root: Path) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    for relative in [
        Path("images") / "train2017",
        Path("images") / "val2017",
        Path("annotations") / "instances_val2017.json",
    ]:
        path = dataset_root / relative
        checks.append(
            PreflightCheck(
                name=f"coco_{relative.as_posix()}",
                ok=path.exists(),
                level="warning",
                message=f"{'found' if path.exists() else 'missing'}: {path}",
            )
        )
    return checks


def _gpu_status() -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return False, "nvidia-smi unavailable"
    if completed.returncode != 0:
        return False, "nvidia-smi returned an error"
    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return bool(names), ", ".join(names) if names else "no visible GPU"


def _scale_from_model(model: str) -> str:
    stem = Path(model).stem.lower()
    for scale in ("n", "s", "m", "l", "x"):
        if stem.endswith(scale):
            return scale
    return "n"


def _read_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig") as file:
        data = yaml.safe_load(file) or {}
    return data if isinstance(data, dict) else {}
