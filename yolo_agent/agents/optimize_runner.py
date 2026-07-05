"""One-command optimization entrypoints for common YOLO workflows."""

from __future__ import annotations

import importlib.util
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
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.orchestrator import LoopOrchestrator, TrainingLoopResult
from yolo_agent.core.experiment_graph import ExperimentNode, ExperimentPlan
from yolo_agent.core.task_spec import MetricPriority, ScenarioHint, TaskSpec
from yolo_agent.resources import ResourcePaths


OptimizeKind = Literal["coco", "custom"]


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
    next_action: str = ""

    @property
    def ok(self) -> bool:
        """Return whether no hard preflight error occurred."""
        return not any(check.level == "error" and not check.ok for check in self.preflight)


class OptimizeRunner:
    """Run a friendly one-command baseline optimization entrypoint."""

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
    ) -> OptimizeResult:
        """Initialize, queue, and optionally execute a baseline optimization run."""
        data_path = Path(data_yaml)
        run_root_path = Path(run_root)
        run_dir = run_root_path / run_id
        preflight = optimize_preflight(kind, data_path, execute=execute)
        task_path = run_dir / "task.yaml"
        plan_path = run_dir / "artifacts" / "experiment_plan.yaml"
        queue_path = run_dir / "execution_queue.yaml"
        hard_error = any(check.level == "error" and not check.ok for check in preflight)

        if hard_error:
            return OptimizeResult(
                kind=kind,
                run_id=run_id,
                run_dir=run_dir,
                profile=profile,
                executor="ultralytics-train" if execute else "dry-run",
                executed=False,
                preflight=preflight,
                task_path=task_path,
                experiment_plan_path=plan_path,
                queue_path=queue_path,
                next_action="Fix preflight errors and rerun the same optimize command.",
            )

        if (run_dir / "run_context.yaml").is_file():
            orchestrator = LoopOrchestrator.from_run_dir(run_dir)
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

        node = _baseline_node(kind, model, profile, orchestrator.context.dataset_version)
        training_config = UltralyticsTrainingConfig.from_yaml(training_config_path, budget_profile=profile)
        command = command_from_training_config(
            node,
            training_config.model_copy(update={"model": model}),
            run_id=run_id,
            data_path=data_path,
        )
        node.command = command.display()
        node.command_spec = command
        plan = ExperimentPlan(
            plan_id=f"{run_id}_optimize_{kind}_{profile}",
            nodes=[node],
            metadata={
                "source": "OptimizeRunner",
                "kind": kind,
                "goal": goal,
                "model": model,
                "data_yaml": data_path.as_posix(),
                "profile": profile,
                "training_config_path": Path(training_config_path).as_posix(),
                "execute": execute,
                "preset": preset_name,
            },
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
        next_action = _next_action(profile, execute, training_loop.queue_counts)
        return OptimizeResult(
            kind=kind,
            run_id=run_id,
            run_dir=orchestrator.context.run_dir,
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
            next_action=next_action,
        )


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


def _task_spec_for(kind: OptimizeKind, data_yaml: Path, goal: str) -> TaskSpec:
    names = _class_names(data_yaml)
    if kind == "coco" and not names:
        names = COCO_NAMES
    if not names:
        names = ["object"]
    return TaskSpec(
        task_type="detect",
        scene="generic",
        class_names=names,
        primary_metric=MetricPriority(name="map50_95"),
        secondary_metrics=[
            MetricPriority(name="latency_ms", goal="minimize"),
            MetricPriority(name="model_size_mb", goal="minimize"),
        ],
        scenario_hint=ScenarioHint(
            name=f"{kind}_optimize",
            description=f"One-command optimize run targeting {goal}.",
            suggested_model_size="auto",
            notes=["Generated by yolo-agent optimize."],
        ),
    )


def _baseline_node(
    kind: OptimizeKind,
    model: str,
    profile: str,
    dataset_version: str,
) -> ExperimentNode:
    stem = Path(model).stem.replace(".", "_").replace("-", "_")
    candidate_id = f"{stem}_{kind}_{profile}"
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
        seed=1,
        changed_variables={},
    )


def _next_action(profile: str, execute: bool, counts: dict[str, int]) -> str:
    if not execute:
        return f"Dry-run completed. Rerun with --execute to start the {profile} training command."
    if counts.get("completed", 0):
        if profile == "debug":
            return "Debug execution completed. Inspect report.md, then rerun optimize with --profile pilot."
        if profile == "pilot":
            return "Pilot execution completed. Import COCO error facts, then generate candidate proposals."
        return "Execution completed. Inspect report.md and evidence_status.json."
    if counts.get("blocked_by_resource", 0) or counts.get("paused", 0):
        return "Execution was resource-blocked. Free GPU resources, then rerun yolo-agent loop queue-refresh and loop execute."
    if counts.get("failed", 0):
        return "Execution failed. Inspect events.jsonl and artifacts/execution_results."
    return "No queued item ran. Inspect execution_queue.yaml."


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
