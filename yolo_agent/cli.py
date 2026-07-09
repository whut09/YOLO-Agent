"""Command line interface for yolo-agent."""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Callable, TypeVar, cast

from yolo_agent.agents.ablation_planner import create_ablation_plan
from yolo_agent.agents.annotation_advisor import advise_annotations
from yolo_agent.agents.candidate_generator import generate_plan
from yolo_agent.adapters.ultralytics.training import TrainingBudgetProfileName
from yolo_agent.adapters.ultralytics.training import UltralyticsRunImporter
from yolo_agent.adapters.ultralytics.training import parse_ultralytics_run
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.orchestrator import LoopOrchestrator
from yolo_agent.agents.auto_optimization_loop import AutoOptimizationLoopDriver, AutoOptimizationResult
from yolo_agent.agents.optimize_runner import OptimizeKind, OptimizeResult, OptimizeRunner
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.evidence_index import EvidenceIndex
from yolo_agent.core.execution_queue import ExecutionQueue, ExecutionQueueStore
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.event_log import EventLog
from yolo_agent.core.loop_status import load_loop_status, render_loop_status
from yolo_agent.core.loop_state import LoopStage
from yolo_agent.core.llm_config import LLMDecisionConfig, load_llm_decision_config
from yolo_agent.core.process_probe import terminate_command_process, terminate_run_processes
from yolo_agent.core.runbook_preset import load_runbook_preset
from yolo_agent.core.run_lineage import RunLineageStore
from yolo_agent.core.schemas import AgentConfig
from yolo_agent.core.task_spec import TaskSpec
from yolo_agent.resources import ResourcePaths
from yolo_agent.reports.cross_run_report import generate_cross_run_comparison_report
from yolo_agent.reports.experiment_report import generate_experiment_report
from yolo_agent.tools.coco_error_mining import mine_coco_errors, write_coco_error_report
from yolo_agent.tools.coco_error_importer import import_coco_eval_metrics
from yolo_agent.tools.dataset_stats import profile_dataset
from yolo_agent.tools.doctor import DatasetKind, DoctorReport, run_doctor
from yolo_agent.tools.setup_wizard import run_setup_wizard, setup_result_to_text
from yolo_agent.tools.smoke_runner import SmokeRunner


T = TypeVar("T")


CLI_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CLI_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


COMMANDS: tuple[str, ...] = (
    "init",
    "profile-data",
    "advise-labels",
    "plan",
    "check",
    "smoke",
    "search",
    "ablate",
    "ablate-plan",
    "benchmark",
    "report",
    "loop",
    "optimize",
    "doctor",
    "setup",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="yolo-agent",
        description="Componentized YOLO optimization harness.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="yolo-agent 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser(
        "init",
        help="Initialize a task.yaml from a scenario template.",
    )
    init_parser.add_argument(
        "--scenario",
        choices=available_scenarios(),
        help="Scenario template to use when generating task.yaml.",
    )
    init_parser.add_argument(
        "--output",
        type=Path,
        default=Path("task.yaml"),
        help="Output path for the generated task spec.",
    )
    init_parser.set_defaults(handler=run_init_command)

    plan_parser = subparsers.add_parser(
        "plan",
        help="Generate compatible candidate experiment configurations.",
    )
    plan_parser.add_argument(
        "--task",
        type=Path,
        required=True,
        help="Path to task.yaml.",
    )
    plan_parser.add_argument(
        "--components",
        type=Path,
        required=True,
        help="Path to component card YAML file or directory.",
    )
    plan_parser.add_argument(
        "--search-space",
        type=Path,
        default=ResourcePaths.SEARCH_SPACE,
        help="Path to search-space YAML.",
    )
    plan_parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs") / "plan.yaml",
        help="Output path for the generated plan.",
    )
    plan_parser.set_defaults(handler=run_plan_command)

    smoke_parser = subparsers.add_parser(
        "smoke",
        help="Run pre-training smoke checks for a candidate plan.",
    )
    smoke_parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Path to runs/plan.yaml.",
    )
    smoke_parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to dataset data.yaml.",
    )
    smoke_parser.add_argument(
        "--base-template",
        type=Path,
        default=ResourcePaths.ULTRALYTICS_BASE_TEMPLATE,
        help="Base Ultralytics model YAML template.",
    )
    smoke_parser.add_argument(
        "--run-id",
        default="smoke",
        help="EvidenceStore run id.",
    )
    smoke_parser.add_argument(
        "--try-forward",
        action="store_true",
        help="When ultralytics is installed, try model.info() for generated YAMLs.",
    )
    smoke_parser.set_defaults(handler=run_smoke_command)

    profile_parser = subparsers.add_parser(
        "profile-data",
        help="Profile a YOLO data.yaml and write dataset reports.",
    )
    profile_parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to YOLO data.yaml.",
    )
    profile_parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs") / "dataset_report",
        help="Output prefix for JSON and Markdown reports.",
    )
    profile_parser.set_defaults(handler=run_profile_data_command)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check training environment, CUDA, data paths, LLM config, and run directory writability.",
    )
    doctor_parser.add_argument("--data", type=Path, help="Path to YOLO data.yaml.")
    doctor_parser.add_argument("--model", default="yolo26n.pt", help="YOLO model checkpoint/name.")
    doctor_parser.add_argument("--run-root", type=Path, default=Path("runs"), help="Run root to test for writability.")
    doctor_parser.add_argument(
        "--kind",
        choices=["coco", "custom"],
        default="coco",
        help="Dataset convention to validate; coco checks train/val/test2017 and annotations.",
    )
    doctor_parser.add_argument("--min-disk-gb", type=float, default=10.0, help="Minimum free disk space required.")
    doctor_parser.add_argument("--min-vram-gb", type=float, default=4.0, help="Minimum free GPU VRAM required.")
    doctor_parser.add_argument("--imgsz", type=int, default=640, help="Image size used for conservative batch estimation.")
    doctor_parser.add_argument(
        "--batch-candidates",
        default="32,48,64,96",
        help="Comma-separated batch candidates for the preflight estimate.",
    )
    doctor_parser.add_argument(
        "--llm",
        action="store_true",
        help="Also check the local decision-analysis LLM config and API-key fallback behavior.",
    )
    doctor_parser.set_defaults(handler=run_doctor_command)

    setup_parser = subparsers.add_parser(
        "setup",
        help="Run a first-use setup wizard for common workflows.",
    )
    setup_parser.set_defaults(handler=run_scaffold_command)
    setup_subparsers = setup_parser.add_subparsers(dest="setup_command")
    setup_coco = setup_subparsers.add_parser(
        "coco",
        help="Prepare local config, LLM config, run id, and COCO path report.",
    )
    setup_coco.add_argument("--data", type=Path, required=True, help="Path to COCO data.yaml.")
    setup_coco.add_argument("--model", default="yolo26n.pt", help="YOLO model checkpoint/name.")
    setup_coco.add_argument("--run-id", help="Run id under --run-root. Defaults to coco-{model stem}.")
    setup_coco.add_argument("--run-root", type=Path, default=Path("runs"), help="Run root directory.")
    setup_coco.add_argument("--env-file", type=Path, default=Path(".env.local"), help="Local env file to create.")
    setup_coco.add_argument(
        "--llm-config",
        type=Path,
        default=ResourcePaths.LLM_DECISION_LOCAL,
        help="Ignored local LLM config path to create.",
    )
    setup_coco.add_argument("--report", type=Path, help="Setup report path. Defaults to runs/{run_id}/setup_report.yaml.")
    setup_coco.add_argument("--overwrite", action="store_true", help="Overwrite existing local setup files.")
    setup_coco.set_defaults(handler=run_setup_coco_command)

    advise_parser = subparsers.add_parser(
        "advise-labels",
        help="Analyze YOLO labels and optional predictions for annotation advice.",
    )
    advise_parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to YOLO data.yaml.",
    )
    advise_parser.add_argument(
        "--predictions",
        type=Path,
        help="Optional prediction YAML/JSON with normalized boxes.",
    )
    advise_parser.add_argument(
        "--rules",
        type=Path,
        help="Optional annotation rules YAML.",
    )
    advise_parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs") / "annotation_advice",
        help="Output prefix for JSON and Markdown reports.",
    )
    advise_parser.set_defaults(handler=run_advise_labels_command)

    coco_errors_parser = subparsers.add_parser(
        "mine-coco-errors",
        help="Mine COCO validation errors from GT annotations and prediction JSON.",
    )
    coco_errors_parser.add_argument("--gt", type=Path, required=True, help="COCO instances JSON.")
    coco_errors_parser.add_argument("--predictions", type=Path, required=True, help="COCO detection prediction JSON.")
    coco_errors_parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs") / "coco_error_report",
        help="Output prefix for JSON, Markdown, and errors YAML.",
    )
    coco_errors_parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for TP matching.")
    coco_errors_parser.add_argument("--score", type=float, default=0.001, help="Minimum prediction confidence.")
    coco_errors_parser.set_defaults(handler=run_mine_coco_errors_command)

    ablate_plan_parser = subparsers.add_parser(
        "ablate-plan",
        help="Create a single-variable ablation plan from candidate plan YAML.",
    )
    ablate_plan_parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Path to runs/plan.yaml.",
    )
    ablate_plan_parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs") / "ablation_plan.yaml",
        help="Output path for the ablation plan.",
    )
    ablate_plan_parser.set_defaults(handler=run_ablate_plan_command)

    report_parser = subparsers.add_parser(
        "report",
        help="Generate a Markdown experiment report from a run directory.",
    )
    report_parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Path to runs/{run_id}.",
    )
    report_parser.add_argument(
        "--out",
        type=Path,
        default=Path("report.md"),
        help="Output Markdown path.",
    )
    report_parser.set_defaults(handler=run_report_command)

    loop_parser = subparsers.add_parser(
        "loop",
        help="Run the state-machine optimization loop harness.",
    )
    loop_parser.add_argument("--run", type=Path, help="Path to runs/{run_id}.")
    loop_parser.add_argument("--resume", action="store_true", help="Resume from the first blocked loop stage.")
    loop_parser.set_defaults(handler=run_loop_command)
    loop_subparsers = loop_parser.add_subparsers(dest="loop_command")

    loop_init = loop_subparsers.add_parser(
        "init",
        help="Initialize a loop run context and state.",
    )
    loop_init.add_argument("--run-id", required=True, help="Run id under runs/.")
    loop_init.add_argument("--task", type=Path, required=True, help="Path to task.yaml.")
    loop_init.add_argument("--data", type=Path, required=True, help="Path to YOLO data.yaml.")
    loop_init.add_argument("--run-root", type=Path, default=Path("runs"), help="Run root directory.")
    loop_init.add_argument("--components", type=Path, default=ResourcePaths.COMPONENTS_DIR, help="Component registry path.")
    loop_init.add_argument("--search-space", type=Path, default=ResourcePaths.SEARCH_SPACE, help="Search-space YAML path.")
    loop_init.add_argument("--loop-policy", type=Path, default=ResourcePaths.LOOP_POLICY, help="Loop policy YAML path.")
    loop_init.add_argument("--predictions", type=Path, help="Optional prediction YAML/JSON for label advice.")
    loop_init.add_argument("--errors", type=Path, help="Optional detection error YAML/JSON.")
    loop_init.add_argument("--metrics", type=Path, help="Optional metrics YAML/JSON to import.")
    loop_init.add_argument("--training-config", type=Path, help="Optional Ultralytics training config YAML.")
    loop_init.add_argument(
        "--training-profile",
        choices=["debug", "pilot", "baseline_full", "baseline_confirm", "candidate_full"],
        help="Optional TrainingBudgetProfile to apply to the training config.",
    )
    loop_init.add_argument("--dataset-version", default="unversioned", help="Dataset version label.")
    loop_init.add_argument(
        "--dataset-manifest-mode",
        choices=["sha256", "metadata"],
        default="sha256",
        help="Dataset manifest fingerprint mode. Use metadata for fast large-dataset loop setup.",
    )
    loop_init.set_defaults(handler=run_loop_init_command)

    loop_run_stage = loop_subparsers.add_parser(
        "run-stage",
        help="Run one loop stage from an existing run directory.",
    )
    loop_run_stage.add_argument("--run", type=Path, required=True, help="Path to runs/{run_id}.")
    loop_run_stage.add_argument("--stage", required=True, help="Stage to run; valid stages come from the run loop policy.")
    loop_run_stage.set_defaults(handler=run_loop_stage_command)

    loop_diagnose = loop_subparsers.add_parser(
        "diagnose",
        help="Run profile-data, label advice, and error diagnosis for a loop run.",
    )
    loop_diagnose.add_argument("--run", type=Path, required=True, help="Path to runs/{run_id}.")
    loop_diagnose.add_argument("--errors", type=Path, help="Detection error YAML/JSON.")
    loop_diagnose.set_defaults(handler=run_loop_diagnose_command)

    loop_plan = loop_subparsers.add_parser(
        "plan",
        help="Generate loop plan, evaluate policies, candidates, and ablations.",
    )
    loop_plan.add_argument("--run", type=Path, required=True, help="Path to runs/{run_id}.")
    loop_plan.set_defaults(handler=run_loop_plan_command)

    loop_enqueue = loop_subparsers.add_parser(
        "enqueue",
        help="Materialize experiment_plan.yaml into execution_queue.yaml.",
    )
    loop_enqueue.add_argument("--run", type=Path, required=True, help="Path to runs/{run_id}.")
    loop_enqueue.set_defaults(handler=run_loop_enqueue_command)

    loop_queue_refresh = loop_subparsers.add_parser(
        "queue-refresh",
        help="Refresh needs_evidence queue items against current run evidence.",
    )
    loop_queue_refresh.add_argument("--run", type=Path, required=True, help="Path to runs/{run_id}.")
    loop_queue_refresh.set_defaults(handler=run_loop_queue_refresh_command)

    loop_status = loop_subparsers.add_parser(
        "status",
        help="Show a user-facing progress panel for a loop run.",
    )
    loop_status.add_argument("--run", type=Path, required=True, help="Path to runs/{run_id}.")
    loop_status.add_argument("--verbose", action="store_true", help="Show machine-readable status details.")
    loop_status.set_defaults(handler=run_loop_status_command)

    loop_stop = loop_subparsers.add_parser(
        "stop",
        help="Stop local optimize/train processes for a run and mark running queue items interrupted.",
    )
    loop_stop.add_argument("--run", type=Path, required=True, help="Path to runs/{run_id}.")
    loop_stop.set_defaults(handler=run_loop_stop_command)

    loop_execute = loop_subparsers.add_parser(
        "execute",
        help="Execute queued experiment nodes with an explicit executor.",
    )
    loop_execute.add_argument("--run", type=Path, required=True, help="Path to runs/{run_id}.")
    loop_execute.add_argument(
        "--executor",
        choices=["dry-run", "shell", "ultralytics", "ultralytics-train"],
        default="dry-run",
        help="Executor to use. dry-run is the default and does not start training.",
    )
    loop_execute.set_defaults(handler=run_loop_execute_command)

    loop_train = loop_subparsers.add_parser(
        "train",
        help="Run the automatic training-loop driver for an existing run.",
    )
    loop_train.add_argument("--run", type=Path, required=True, help="Path to runs/{run_id}.")
    loop_train.add_argument(
        "--profile",
        choices=["debug", "pilot", "baseline_full", "baseline_confirm", "candidate_full"],
        default="debug",
        help="TrainingBudgetProfile to apply to the run context.",
    )
    loop_train.add_argument(
        "--executor",
        choices=["dry-run", "shell", "ultralytics", "ultralytics-train"],
        default="dry-run",
        help="Executor to use. dry-run does not start training.",
    )
    loop_train.add_argument("--max-steps", type=int, default=8, help="Maximum automatic driver steps to run.")
    loop_train.add_argument("--no-auto-import", action="store_true", help="Disable metrics auto-import attempts.")
    loop_train.set_defaults(handler=run_loop_train_command)

    loop_smoke = loop_subparsers.add_parser(
        "smoke",
        help="Run loop smoke guard.",
    )
    loop_smoke.add_argument("--run", type=Path, required=True, help="Path to runs/{run_id}.")
    loop_smoke.set_defaults(handler=run_loop_smoke_command)

    loop_ingest = loop_subparsers.add_parser(
        "ingest-metrics",
        help="Import external benchmark metrics from YAML, JSON, or CSV.",
    )
    loop_ingest.add_argument("--run", type=Path, required=True, help="Path to runs/{run_id}.")
    loop_ingest.add_argument("--metrics", type=Path, required=True, help="Metrics YAML/JSON/CSV.")
    loop_ingest.set_defaults(handler=run_loop_ingest_metrics_command)

    loop_import_ultralytics = loop_subparsers.add_parser(
        "import-ultralytics",
        help="Import an Ultralytics run directory into node-level evidence.",
    )
    loop_import_ultralytics.add_argument("--run", type=Path, required=True, help="Path to runs/{run_id}.")
    loop_import_ultralytics.add_argument("--ultralytics-run", type=Path, required=True, help="Ultralytics run directory.")
    loop_import_ultralytics.add_argument("--candidate-id", required=True, help="Candidate id for imported evidence.")
    loop_import_ultralytics.add_argument("--node-id", required=True, help="Experiment node id for imported evidence.")
    loop_import_ultralytics.add_argument("--base-model", default="yolo26n.pt", help="Base model used by the run.")
    loop_import_ultralytics.add_argument("--scale", default="n", help="Model scale label.")
    loop_import_ultralytics.add_argument("--seed", type=int, default=1, help="Experiment seed.")
    loop_import_ultralytics.add_argument("--dataset-version", help="Override dataset version.")
    loop_import_ultralytics.add_argument("--log", type=Path, help="Optional Ultralytics stdout/stderr log to profile.")
    loop_import_ultralytics.set_defaults(handler=run_loop_import_ultralytics_command)

    loop_import_coco_eval = loop_subparsers.add_parser(
        "import-coco-eval",
        help="Import official COCO eval metrics into node-level evidence.",
    )
    loop_import_coco_eval.add_argument("--run", type=Path, required=True, help="Path to runs/{run_id}.")
    loop_import_coco_eval.add_argument("--eval", type=Path, required=True, help="COCO eval JSON or text output.")
    loop_import_coco_eval.add_argument("--candidate-id", required=True, help="Candidate id for imported evidence.")
    loop_import_coco_eval.add_argument("--node-id", required=True, help="Experiment node id for imported evidence.")
    loop_import_coco_eval.add_argument("--dataset-version", help="Override dataset version.")
    loop_import_coco_eval.add_argument("--split", default="val2017", help="Dataset split label.")
    loop_import_coco_eval.set_defaults(handler=run_loop_import_coco_eval_command)

    loop_mine = loop_subparsers.add_parser(
        "mine",
        help="Mine unlabeled predictions into an active-learning labeling manifest.",
    )
    loop_mine.add_argument("--run", type=Path, required=True, help="Path to runs/{run_id}.")
    loop_mine.add_argument("--predictions", type=Path, required=True, help="Unlabeled prediction JSON.")
    loop_mine.add_argument(
        "--target",
        choices=["generic", "cvat", "label_studio"],
        default="generic",
        help="Labeling handoff target.",
    )
    loop_mine.set_defaults(handler=run_loop_mine_command)

    loop_dataset_promote = loop_subparsers.add_parser(
        "dataset-promote",
        help="Evaluate dataset promotion after reviewed labels are available.",
    )
    loop_dataset_promote.add_argument("--run", type=Path, required=True, help="Path to runs/{run_id}.")
    loop_dataset_promote.add_argument("--reviewed-labels", type=Path, help="Reviewed labels YAML/JSON.")
    loop_dataset_promote.set_defaults(handler=run_loop_dataset_promote_command)

    loop_next = loop_subparsers.add_parser(
        "next",
        help="Generate report and next-round checklist.",
    )
    loop_next.add_argument("--run", type=Path, required=True, help="Path to runs/{run_id}.")
    loop_next.set_defaults(handler=run_loop_next_command)

    loop_fork_next = loop_subparsers.add_parser(
        "fork-next",
        help="Materialize next_round.yaml into a fresh child loop run.",
    )
    loop_fork_next.add_argument("--run", type=Path, required=True, help="Path to parent runs/{run_id}.")
    loop_fork_next.add_argument("--new-run-id", required=True, help="Child run id under the same run root.")
    loop_fork_next.set_defaults(handler=run_loop_fork_next_command)

    loop_lineage = loop_subparsers.add_parser(
        "lineage",
        help="Query cross-run lineage graph.",
    )
    loop_lineage.add_argument("--run-root", type=Path, default=Path("runs"), help="Run root containing lineage.jsonl.")
    loop_lineage.add_argument("--run", help="Optional run id to inspect.")
    loop_lineage.add_argument("--best", action="store_true", help="Show the current best trusted run.")
    loop_lineage.set_defaults(handler=run_loop_lineage_command)

    loop_compare = loop_subparsers.add_parser(
        "compare",
        help="Generate a cross-run comparison report.",
    )
    loop_compare.add_argument("--runs", type=Path, nargs="+", required=True, help="Run directories to compare.")
    loop_compare.add_argument("--out", type=Path, default=Path("comparison.md"), help="Output Markdown path.")
    loop_compare.set_defaults(handler=run_loop_compare_command)

    loop_auto = loop_subparsers.add_parser(
        "auto",
        help="Initialize or run pending loop stages until blocked, failed, or complete.",
    )
    loop_auto.add_argument("--run", type=Path, help="Path to runs/{run_id}.")
    loop_auto.add_argument("--run-id", default="auto", help="Run id when initializing.")
    loop_auto.add_argument("--task", type=Path, help="Path to task.yaml when initializing.")
    loop_auto.add_argument("--data", type=Path, help="Path to YOLO data.yaml when initializing.")
    loop_auto.add_argument("--run-root", type=Path, default=Path("runs"), help="Run root directory.")
    loop_auto.add_argument("--components", type=Path, default=ResourcePaths.COMPONENTS_DIR, help="Component registry path.")
    loop_auto.add_argument("--search-space", type=Path, default=ResourcePaths.SEARCH_SPACE, help="Search-space YAML path.")
    loop_auto.add_argument("--loop-policy", type=Path, default=ResourcePaths.LOOP_POLICY, help="Loop policy YAML path.")
    loop_auto.add_argument("--predictions", type=Path, help="Optional prediction YAML/JSON for label advice.")
    loop_auto.add_argument("--errors", type=Path, help="Optional detection error YAML/JSON.")
    loop_auto.add_argument("--metrics", type=Path, help="Optional metrics YAML/JSON/CSV.")
    loop_auto.add_argument("--training-config", type=Path, help="Optional Ultralytics training config YAML.")
    loop_auto.add_argument(
        "--training-profile",
        choices=["debug", "pilot", "baseline_full", "baseline_confirm", "candidate_full"],
        help="Optional TrainingBudgetProfile to apply to the training config.",
    )
    loop_auto.add_argument("--dataset-version", default="unversioned", help="Dataset version label.")
    loop_auto.add_argument(
        "--dataset-manifest-mode",
        choices=["sha256", "metadata"],
        default="sha256",
        help="Dataset manifest fingerprint mode. Use metadata for fast large-dataset loop setup.",
    )
    loop_auto.set_defaults(handler=run_loop_auto_command)

    optimize_parser = subparsers.add_parser(
        "optimize",
        help="One-command optimization runbooks for common workflows.",
    )
    optimize_subparsers = optimize_parser.add_subparsers(dest="optimize_command")
    optimize_advance = optimize_subparsers.add_parser(
        "advance",
        help="Advance an existing optimize run to the next budget profile.",
    )
    optimize_advance.add_argument("--run", type=Path, required=True, help="Path to runs/{run_id}.")
    optimize_advance.add_argument(
        "--to-profile",
        choices=["debug", "pilot", "baseline_full", "baseline_confirm", "candidate_full"],
        required=True,
        help="TrainingBudgetProfile to materialize for the existing run.",
    )
    optimize_advance.add_argument(
        "--execute",
        action="store_true",
        help="Actually run ultralytics-train. Without this flag, only prepare the run and queue.",
    )
    optimize_advance.add_argument(
        "--confirm-full-run",
        action="store_true",
        help="Required with --execute for baseline_full, baseline_confirm, or candidate_full profiles.",
    )
    optimize_advance.add_argument(
        "--no-auto-advance",
        action="store_true",
        help="Disable bounded profile auto-advance after a successful profile completes.",
    )
    optimize_advance.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Maximum automatic driver steps to run.",
    )
    optimize_advance.add_argument(
        "--no-auto-import",
        action="store_true",
        help="Disable automatic metrics import when metrics_input_path is configured.",
    )
    optimize_advance.set_defaults(handler=run_optimize_advance_command)

    optimize_auto_loop = optimize_subparsers.add_parser(
        "auto-loop",
        help="Continue pilot-only auto optimization from an existing run without rerunning the baseline.",
    )
    optimize_auto_loop.add_argument("--run", type=Path, required=True, help="Path to runs/{run_id}.")
    optimize_auto_loop.add_argument(
        "--auto-rounds",
        type=int,
        default=1,
        help="Number of child pilot-only optimization rounds to run.",
    )
    optimize_auto_loop.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute currently supported pilot training candidates. Without this flag, dry-run only.",
    )
    optimize_auto_loop.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Maximum automatic driver steps per child round.",
    )
    optimize_auto_loop.add_argument(
        "--no-auto-import",
        action="store_true",
        help="Disable automatic metrics import when metrics_input_path is configured.",
    )
    optimize_auto_loop.set_defaults(handler=run_optimize_auto_loop_command)

    for kind, default_run_id in [
        ("coco", "coco-yolo26n"),
        ("custom", "custom-yolo26n"),
    ]:
        optimize_kind = optimize_subparsers.add_parser(
            kind,
            help=f"Start a one-command {kind} optimization run.",
        )
        optimize_kind.add_argument(
            "--preset",
            type=Path,
            default=ResourcePaths.COCO_YOLO26_AUTO_PRESET,
            help="Runbook preset YAML. Defaults to presets/coco_yolo26_auto.yaml.",
        )
        optimize_kind.add_argument("--model", help="YOLO model checkpoint/name. Defaults to the preset model.")
        optimize_kind.add_argument("--data", type=Path, required=True, help="YOLO data.yaml.")
        optimize_kind.add_argument("--goal", help="Human-readable optimization goal. Defaults to the preset goal.")
        optimize_kind.add_argument("--run-id", default=default_run_id, help="Run id under --run-root.")
        optimize_kind.add_argument("--run-root", type=Path, default=Path("runs"), help="Run root directory.")
        optimize_kind.add_argument(
            "--profile",
            choices=["debug", "pilot", "baseline_full", "baseline_confirm", "candidate_full"],
            help="TrainingBudgetProfile; defaults to the preset default profile.",
        )
        optimize_kind.add_argument(
            "--training-config",
            type=Path,
            help="Override the preset Ultralytics training config YAML.",
        )
        optimize_kind.add_argument("--components", type=Path, help="Override the preset component registry path.")
        optimize_kind.add_argument("--search-space", type=Path, help="Override the preset search-space YAML.")
        optimize_kind.add_argument("--loop-policy", type=Path, help="Override the preset loop policy YAML.")
        optimize_kind.add_argument(
            "--dataset-manifest-mode",
            choices=["sha256", "metadata"],
            help="Override the preset dataset manifest mode.",
        )
        optimize_kind.add_argument(
            "--execute",
            action="store_true",
            help="Actually run ultralytics-train. Without this flag, only prepare the run and queue.",
        )
        optimize_kind.add_argument(
            "--confirm-full-run",
            action="store_true",
            help="Required with --execute for baseline_full, baseline_confirm, or candidate_full profiles.",
        )
        optimize_kind.add_argument(
            "--no-auto-advance",
            action="store_true",
            help="Disable bounded profile auto-advance after a successful profile completes.",
        )
        optimize_kind.add_argument(
            "--auto-rounds",
            type=int,
            default=0,
            help=(
                "After a successful pilot, automatically fork and run this many pilot-only optimization rounds. "
                "Full COCO training is never started by this flag."
            ),
        )
        optimize_kind.add_argument(
            "--max-steps",
            type=int,
            default=8,
            help="Maximum automatic driver steps to run.",
        )
        optimize_kind.add_argument(
            "--no-auto-import",
            action="store_true",
            help="Disable automatic metrics import when metrics_input_path is configured.",
        )
        optimize_kind.set_defaults(handler=run_optimize_command, optimize_kind=kind)

    for command in COMMANDS:
        if command in {
            "init",
            "plan",
            "smoke",
            "profile-data",
            "advise-labels",
            "mine-coco-errors",
            "ablate-plan",
            "report",
            "loop",
            "optimize",
            "doctor",
            "setup",
        }:
            continue
        command_parser = subparsers.add_parser(
            command,
            help=f"Run the {command} workflow scaffold.",
        )
        command_parser.set_defaults(handler=run_scaffold_command)

    return parser


def scenarios_dir() -> Path:
    """Return the bundled scenario template directory."""
    return ResourcePaths.SCENARIOS_DIR


def available_scenarios() -> list[str]:
    """List available scenario template names."""
    directory = scenarios_dir()
    if not directory.exists():
        return []
    return sorted(path.stem for path in directory.glob("*.yaml"))


def run_init_command(args: argparse.Namespace) -> int:
    """Generate task.yaml from a validated scenario template."""
    if args.scenario is None:
        print("yolo-agent init: scaffold ready")
        print("available_scenarios=" + ", ".join(available_scenarios()))
        return 0

    scenario_path = scenarios_dir() / f"{args.scenario}.yaml"
    task_spec = TaskSpec.from_yaml(scenario_path)
    task_spec.to_yaml(args.output)
    print(f"created {args.output} from scenario={args.scenario}")
    return 0


def run_plan_command(args: argparse.Namespace) -> int:
    """Generate candidate plan YAML."""
    plan = generate_plan(
        task_path=args.task,
        component_path=args.components,
        search_space_path=args.search_space,
        out_path=args.out,
    )
    print(f"created {args.out} with {len(plan.candidates)} candidates")
    if plan.skipped:
        print(f"skipped={len(plan.skipped)}")
    return 0


def run_smoke_command(args: argparse.Namespace) -> int:
    """Run smoke checks for a generated plan."""
    result = SmokeRunner().run(
        plan_path=args.plan,
        data_path=args.data,
        run_id=args.run_id,
        base_template=args.base_template,
        try_forward=args.try_forward,
    )
    print(f"smoke status={result.status}")
    print(f"candidates={len(result.candidates)}")
    if result.warnings:
        print(f"warnings={len(result.warnings)}")
    if result.errors:
        print(f"errors={len(result.errors)}")
    return 1 if result.status == "failed" else 0


def run_profile_data_command(args: argparse.Namespace) -> int:
    """Profile a YOLO dataset."""
    report = profile_dataset(args.data, args.out)
    json_path = args.out.with_suffix(".json") if args.out.suffix else Path(f"{args.out}.json")
    markdown_path = args.out.with_suffix(".md") if args.out.suffix else Path(f"{args.out}.md")
    print(f"profiled images={report.image_count} labels={report.label_count}")
    print(f"dataset_health={report.dataset_health.score}/100")
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    return 0


def run_doctor_command(args: argparse.Namespace) -> int:
    """Run environment doctor checks."""
    if args.data is None:
        if args.llm:
            _print_llm_doctor_report()
            return 0
        print("doctor error=missing_data")
        print("  fix: pass --data data.yaml, or use --llm for an LLM-only check.")
        return 2

    report = run_doctor(
        data_yaml=args.data,
        model=args.model,
        run_root=args.run_root,
        kind=cast("DatasetKind", args.kind),
        min_disk_gb=args.min_disk_gb,
        min_vram_gb=args.min_vram_gb,
        imgsz=args.imgsz,
        candidate_batches=_parse_batch_candidates(args.batch_candidates),
    )
    _print_doctor_report(report)
    if args.llm:
        _print_llm_doctor_report()
    return 0 if report.ok else 1


def run_setup_coco_command(args: argparse.Namespace) -> int:
    """Run the COCO onboarding setup wizard."""
    result = run_setup_wizard(
        kind="coco",
        data_yaml=args.data,
        model=args.model,
        run_id=args.run_id,
        run_root=args.run_root,
        env_file=args.env_file,
        llm_config_path=args.llm_config,
        setup_report_path=args.report,
        overwrite=args.overwrite,
    )
    print(setup_result_to_text(result))
    return 0 if result.ok else 1


def _print_doctor_report(report: DoctorReport) -> None:
    print(f"doctor status={'ok' if report.ok else 'failed'} errors={report.error_count} warnings={report.warning_count}")
    print(f"data={report.data_yaml}")
    print(f"model={report.model}")
    print(f"run_root={report.run_root}")
    if report.batch_estimate is not None:
        estimate = report.batch_estimate
        selected = estimate.selected_batch if estimate.selected_batch is not None else "unknown"
        candidates = ",".join(str(value) for value in estimate.candidate_batches)
        print(
            "batch_estimate="
            f"{selected} candidates={candidates} imgsz={estimate.imgsz} "
            f"free_vram_gb={_format_optional_float(estimate.free_vram_gb)} "
            f"confidence={estimate.confidence}"
        )
        if estimate.limiting_reason:
            print(f"batch_reason={estimate.limiting_reason}")
        print(f"batch_note={estimate.note}")
    for check in report.checks:
        status = "ok" if check.ok else check.level
        print(f"{check.name}: {status} - {check.message}")
        if not check.ok and check.fix:
            print(f"  fix: {check.fix}")


def _parse_batch_candidates(value: str) -> list[int]:
    """Parse a comma-separated batch candidate list."""
    candidates: list[int] = []
    for item in value.split(","):
        text = item.strip()
        if not text:
            continue
        candidates.append(int(text))
    return candidates


def _format_optional_float(value: float | None) -> str:
    """Format optional floats for compact CLI output."""
    return "unknown" if value is None else f"{value:.1f}"


def _print_llm_doctor_report() -> None:
    """Print a non-failing LLM readiness summary for beginner setup."""
    try:
        config = load_llm_decision_config()
    except (OSError, ValueError) as exc:
        print(f"llm status=failed - {exc}")
        print("llm fallback=rule_engine")
        return

    status = _llm_doctor_status(config)
    print(f"llm status={status}")
    print(f"llm enabled={str(config.enabled).lower()}")
    print(f"llm use_by_default={str(config.use_by_default).lower()}")
    print(f"llm provider={config.provider}")
    print(f"llm model={config.model}")
    if config.model_alias:
        print(f"llm model_alias={config.model_alias}")
    print(f"llm api_key_source={config.api_key_source()}")
    print(f"llm base_url_source={config.base_url_source()}")
    print(f"llm decision_role={config.decision_role}")
    print(f"llm executable_decisions_allowed={str(config.executable_decisions_allowed).lower()}")
    if status in {"disabled", "redacted", "missing_key", "failed"}:
        print("llm fallback=rule_engine")
    if status == "missing_key":
        print(f"  fix: set {config.api_key_env}, for example: $env:{config.api_key_env}=\"...\"")
    elif status == "redacted":
        print("  fix: copy configs/llm_decision.example.yaml to configs/local/llm_decision.local.yaml and fill local values.")


def _llm_doctor_status(config: LLMDecisionConfig) -> str:
    """Return the user-facing LLM readiness status."""
    if not config.enabled or not config.use_by_default:
        return "disabled"
    if config.provider == "XX" or config.model == "XX" or (config.api_key_env == "XX" and not config.api_key):
        return "redacted"
    if config.require_api_key and not config.resolved_api_key():
        return "missing_key"
    return "ready"


def run_advise_labels_command(args: argparse.Namespace) -> int:
    """Analyze labels and write annotation advice reports."""
    report = advise_annotations(args.data, args.out, args.predictions, args.rules)
    json_path = args.out.with_suffix(".json") if args.out.suffix else Path(f"{args.out}.json")
    markdown_path = args.out.with_suffix(".md") if args.out.suffix else Path(f"{args.out}.md")
    print(f"label_issues={len(report.label_quality.issues)}")
    print(f"samples_for_review={len(report.samples_for_review)}")
    print(f"boxes_to_redraw={len(report.boxes_to_redraw)}")
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    return 0


def run_mine_coco_errors_command(args: argparse.Namespace) -> int:
    """Mine COCO error facts from predictions."""
    report = mine_coco_errors(
        gt_json=args.gt,
        predictions_json=args.predictions,
        iou_threshold=args.iou,
        score_threshold=args.score,
    )
    json_path, markdown_path, errors_path = write_coco_error_report(report, args.out)
    print(f"classes={len(report.class_summaries)}")
    print(f"observations={len(report.observations)}")
    print(f"small_recall={report.area_recall.get('small', 0.0):.6f}")
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    print(f"wrote {errors_path}")
    return 0


def run_ablate_plan_command(args: argparse.Namespace) -> int:
    """Create a single-variable ablation plan."""
    plan = create_ablation_plan(args.plan, args.out)
    print(f"created {args.out} with {len(plan.nodes)} ablations")
    if plan.invalid_candidates:
        print(f"invalid={len(plan.invalid_candidates)}")
    return 0


def run_report_command(args: argparse.Namespace) -> int:
    """Generate a Markdown experiment report."""
    generate_experiment_report(args.run, args.out)
    print(f"wrote {args.out}")
    return 0


def run_loop_init_command(args: argparse.Namespace) -> int:
    """Initialize a loop run."""
    orchestrator = LoopOrchestrator.initialize(
        run_id=args.run_id,
        task_path=args.task,
        data_yaml=args.data,
        run_root=args.run_root,
        component_path=args.components,
        search_space_path=args.search_space,
        loop_policy_path=args.loop_policy,
        predictions_path=args.predictions,
        detection_errors_path=args.errors,
        metrics_input_path=args.metrics,
        training_config_path=args.training_config,
        training_profile=cast("TrainingBudgetProfileName | None", args.training_profile),
        dataset_version=args.dataset_version,
        dataset_manifest_mode=args.dataset_manifest_mode,
    )
    print(f"created {orchestrator.context.run_dir}")
    print(f"state={orchestrator.context.run_dir / 'loop_state.yaml'}")
    return 0


def run_loop_command(args: argparse.Namespace) -> int:
    """Run top-level loop actions such as resume."""
    if args.run is None:
        print("yolo-agent loop: provide --run with --resume, or use a loop subcommand.")
        return 0
    orchestrator = LoopOrchestrator.from_run_dir(args.run)
    results = orchestrator.resume() if args.resume else orchestrator.run_until_blocked()
    for result in results:
        print(f"{result.stage} status={result.status}")
        if result.message:
            print(result.message)
    if results and results[-1].status == "failed":
        return 1
    return 0


def run_loop_stage_command(args: argparse.Namespace) -> int:
    """Run one loop stage."""
    orchestrator = LoopOrchestrator.from_run_dir(args.run)
    if args.stage not in orchestrator.policy.stage_order:
        print(f"Unknown stage for this loop policy: {args.stage}")
        print("valid_stages=" + ", ".join(orchestrator.policy.stage_order))
        return 1
    result = orchestrator.run_stage(cast(LoopStage, args.stage))
    print(f"{result.stage} status={result.status}")
    if result.message:
        print(result.message)
    return 1 if result.status == "failed" else 0


def run_loop_diagnose_command(args: argparse.Namespace) -> int:
    """Run loop diagnosis stages."""
    return _print_loop_results(LoopOrchestrator.from_run_dir(args.run).diagnose(args.errors))


def run_loop_plan_command(args: argparse.Namespace) -> int:
    """Run loop planning stages."""
    return _print_loop_results(LoopOrchestrator.from_run_dir(args.run).plan_loop())


def run_loop_enqueue_command(args: argparse.Namespace) -> int:
    """Materialize an execution queue."""
    queue = LoopOrchestrator.from_run_dir(args.run).enqueue()
    print(f"execution_queue={args.run / 'execution_queue.yaml'}")
    print(_format_queue_counts(queue.counts()))
    return 0


def run_loop_queue_refresh_command(args: argparse.Namespace) -> int:
    """Refresh needs_evidence queue items."""
    queue = LoopOrchestrator.from_run_dir(args.run).refresh_queue()
    print(f"execution_queue={args.run / 'execution_queue.yaml'}")
    print(_format_queue_counts(queue.counts()))
    return 0


def run_loop_status_command(args: argparse.Namespace) -> int:
    """Print a user-facing loop progress panel."""
    print(render_loop_status(load_loop_status(args.run), verbose=args.verbose))
    return 0


def run_loop_stop_command(args: argparse.Namespace) -> int:
    """Stop local run processes and mark running queue items interrupted."""
    run_dir = args.run
    run_id = run_dir.name
    terminations = terminate_run_processes(run_id)
    stopped = sum(1 for result in terminations if result.terminated)
    marked = 0
    queue_path = run_dir / "execution_queue.yaml"
    if queue_path.is_file():
        store = ExecutionQueueStore(run_dir)
        queue = store.load()
        for item in queue.items:
            if item.status != "running":
                continue
            command_termination = terminate_command_process(item.command)
            if command_termination.terminated:
                stopped += 1
            item.mark_interrupted("Stopped by yolo-agent loop stop.")
            queue = store.update_item(item)
            marked += 1
            EventLog(run_dir / "events.jsonl").append(
                run_id=queue.run_id,
                event_type="queue_item_failed",
                status="blocked",
                message=item.message,
                details={
                    "queue_id": item.queue_id,
                    "node_id": item.node_id,
                    "candidate_id": item.candidate_id,
                    "stopped_by_user": True,
                    "termination": command_termination.model_dump(mode="json"),
                },
            )
    print(f"stop run={run_dir}")
    print(f"stopped_processes={stopped}")
    print(f"marked_running_items={marked}")
    for result in terminations:
        state = "stopped" if result.terminated else "not_stopped"
        print(f"{state} pid={result.pid} name={result.name} detail={_clean_cli_line(result.detail, limit=160)}")
    print(f"next: yolo-agent loop queue-refresh --run {run_dir}")
    print(f"next: yolo-agent loop status --run {run_dir}")
    return 0 if stopped or marked else 1


def run_loop_execute_command(args: argparse.Namespace) -> int:
    """Execute queued nodes with an explicit executor."""
    queue = LoopOrchestrator.from_run_dir(args.run).execute_queue(args.executor)
    print(f"executor={args.executor}")
    print(_format_queue_counts(queue.counts()))
    counts = queue.counts()
    return 1 if counts["failed"] else 0


def run_loop_train_command(args: argparse.Namespace) -> int:
    """Run the automatic training-loop driver."""
    result = LoopOrchestrator.from_run_dir(args.run).run_training_loop(
        profile=cast("TrainingBudgetProfileName", args.profile),
        executor=args.executor,
        max_steps=args.max_steps,
        auto_import=not args.no_auto_import,
    )
    print(f"profile={result.profile}")
    print(f"executor={result.executor}")
    print(f"driver_steps={len(result.steps)}")
    print(f"driver_stopped={result.stopped_reason}")
    print(_format_queue_counts(result.queue_counts))
    for step in result.steps:
        print(f"{step.action} status={step.status}")
        if step.message:
            print(step.message)
    return 1 if any(step.status == "failed" for step in result.steps) else 0


def run_loop_smoke_command(args: argparse.Namespace) -> int:
    """Run loop smoke stage."""
    return _print_loop_results([LoopOrchestrator.from_run_dir(args.run).smoke()])


def run_loop_ingest_metrics_command(args: argparse.Namespace) -> int:
    """Import loop metrics."""
    return _print_loop_results([LoopOrchestrator.from_run_dir(args.run).ingest_metrics(args.metrics)])


def run_loop_import_ultralytics_command(args: argparse.Namespace) -> int:
    """Import Ultralytics run evidence into the loop EvidenceStore."""
    context = LoopOrchestrator.from_run_dir(args.run).context
    dataset_version = args.dataset_version or context.dataset_version
    node = ExperimentNode(
        node_id=args.node_id,
        candidate_config=CandidateConfig(
            candidate_id=args.candidate_id,
            base_model=args.base_model,
            scale=args.scale,
            framework="ultralytics",
        ),
        data_version=dataset_version,
        seed=args.seed,
    )
    store = EvidenceStore(context.run_root)
    metrics = UltralyticsRunImporter(store).import_run(
        context.run_id,
        node,
        args.ultralytics_run,
        log_path=args.log,
        data_path=context.data_yaml,
    )
    store.log_metrics(context.run_id, metrics)
    print(f"imported_metrics={len(metrics)}")
    print(f"candidate_id={args.candidate_id}")
    print(f"node_id={args.node_id}")
    print(f"metrics_by_node={context.run_dir / 'metrics_by_node.jsonl'}")
    return 0


def run_loop_import_coco_eval_command(args: argparse.Namespace) -> int:
    """Import official COCO eval metrics into node-level evidence."""
    context = LoopOrchestrator.from_run_dir(args.run).context
    result = import_coco_eval_metrics(
        eval_path=args.eval,
        evidence_store=EvidenceStore(context.run_root),
        run_id=context.run_id,
        candidate_id=args.candidate_id,
        node_id=args.node_id,
        dataset_version=args.dataset_version or context.dataset_version,
        split=args.split,
    )
    print(f"imported_metrics={len(result.metrics)}")
    print(f"candidate_id={args.candidate_id}")
    print(f"node_id={args.node_id}")
    print(f"metrics_by_node={result.metrics_by_node_path}")
    if result.error_facts_path is not None:
        print(f"error_facts={result.error_facts_path}")
        print(f"error_fact_count={result.error_fact_count}")
    return 0


def run_loop_mine_command(args: argparse.Namespace) -> int:
    """Mine unlabeled predictions for active learning."""
    orchestrator = LoopOrchestrator.from_run_dir(args.run)
    plan = orchestrator.mine(args.predictions, labeling_target=args.target)
    manifest_path = orchestrator.context.artifact_path("labeling_manifest.json")
    plan_path = orchestrator.context.artifact_path("active_learning_plan.json")
    print(f"labeling_manifest={manifest_path}")
    print(f"active_learning_plan={plan_path}")
    print(f"mined_samples={len(plan.mined_samples)}")
    print(f"next_dataset_version={plan.next_dataset_version}")
    return 0


def run_loop_dataset_promote_command(args: argparse.Namespace) -> int:
    """Evaluate active-learning dataset promotion."""
    result = LoopOrchestrator.from_run_dir(args.run).promote_dataset(args.reviewed_labels)
    print(f"{result.stage} status={result.status}")
    if result.message:
        print(result.message)
    return 1 if result.status == "failed" else 0


def run_loop_next_command(args: argparse.Namespace) -> int:
    """Run loop report and next-round stages."""
    return _print_loop_results(LoopOrchestrator.from_run_dir(args.run).next_round())


def run_loop_fork_next_command(args: argparse.Namespace) -> int:
    """Fork an existing run's next-round checklist into a child run."""
    orchestrator = LoopOrchestrator.from_run_dir(args.run).fork_next(args.new_run_id)
    missing = orchestrator.context.metadata.get("inherited_missing_evidence", [])
    print(f"created {orchestrator.context.run_dir}")
    print(f"parent_run_id={orchestrator.context.metadata.get('parent_run_id')}")
    print(f"inherited_missing_evidence={len(missing) if isinstance(missing, list) else 0}")
    return 0


def run_loop_lineage_command(args: argparse.Namespace) -> int:
    """Query the run lineage graph."""
    graph = RunLineageStore(args.run_root).graph()
    if args.best:
        best = graph.best_trusted_run()
        if best is None:
            print("best_trusted_run=none")
            return 0
        print(f"best_trusted_run={best.run_id}")
        print(f"candidate={best.best_candidate_id or 'unknown'}")
        print(f"node={best.best_node_id or 'unknown'}")
        print(f"metric={best.best_metric_name or 'unknown'}")
        print(f"value={best.best_metric_value if best.best_metric_value is not None else 'unknown'}")
        return 0
    if args.run:
        record = graph.records.get(args.run)
        if record is None:
            print(f"run_not_found={args.run}")
            return 1
        delta = graph.evidence_delta(args.run)
        print(f"run_id={record.run_id}")
        print(f"parent_run_id={record.parent_run_id or 'none'}")
        print(f"children={','.join(graph.children_of(args.run)) or 'none'}")
        print(f"dataset_manifest_sha256={record.dataset_manifest_sha256 or 'unknown'}")
        print(f"trusted={record.trusted}")
        print(f"inherited_missing={','.join(delta['inherited_missing']) or 'none'}")
        print(f"current_missing={','.join(delta['current_missing']) or 'none'}")
        print(f"resolved={','.join(delta['resolved']) or 'none'}")
        return 0
    for record in graph.records.values():
        print(
            f"{record.run_id} parent={record.parent_run_id or 'none'} "
            f"trusted={record.trusted} sha={record.dataset_manifest_sha256 or 'unknown'}"
        )
    return 0


def run_loop_compare_command(args: argparse.Namespace) -> int:
    """Generate a cross-run comparison report."""
    if len(args.runs) < 2:
        print("yolo-agent loop compare: provide at least two run directories.")
        return 1
    generate_cross_run_comparison_report(args.runs, args.out)
    print(f"wrote {args.out}")
    return 0


def run_loop_auto_command(args: argparse.Namespace) -> int:
    """Run pending stages until blocked or complete."""
    if args.run is not None:
        orchestrator = LoopOrchestrator.from_run_dir(args.run)
    else:
        if args.task is None or args.data is None:
            print("yolo-agent loop auto: provide --run, or provide --task and --data to initialize.")
            return 1
        orchestrator = LoopOrchestrator.initialize(
            run_id=args.run_id,
            task_path=args.task,
            data_yaml=args.data,
            run_root=args.run_root,
            component_path=args.components,
            search_space_path=args.search_space,
            loop_policy_path=args.loop_policy,
            predictions_path=args.predictions,
            detection_errors_path=args.errors,
            metrics_input_path=args.metrics,
            training_config_path=args.training_config,
            training_profile=cast("TrainingBudgetProfileName | None", args.training_profile),
            dataset_version=args.dataset_version,
            dataset_manifest_mode=args.dataset_manifest_mode,
        )
        print(f"created {orchestrator.context.run_dir}")
    return _print_loop_results(orchestrator.run_until_blocked())


def run_optimize_command(args: argparse.Namespace) -> int:
    """Run a one-command optimization runbook."""
    try:
        preset = load_runbook_preset(args.preset)
        profile = cast("TrainingBudgetProfileName", args.profile or preset.default_profile)
        preset.require_profile(profile)
    except (OSError, ValueError) as exc:
        print(f"preset error: {exc}")
        return 1
    model = args.model or preset.default_model
    goal = args.goal or preset.default_goal
    training_config = args.training_config or preset.training_config
    component_path = args.components or preset.components
    search_space_path = args.search_space or preset.search_space
    loop_policy_path = args.loop_policy or preset.loop_policy
    dataset_manifest_mode = args.dataset_manifest_mode or preset.dataset_manifest_mode
    run_dir = args.run_root / args.run_id
    print("Starting YOLO Agent optimize", flush=True)
    print(f"Run: {args.run_id}  Profile: {profile}  Mode: {'execute' if args.execute else 'dry-run'}", flush=True)
    print(f"Data: {args.data}", flush=True)
    if args.execute:
        print("progress: real execution requested; watching run events. Use Ctrl+C to stop the CLI.", flush=True)
    result = _run_with_event_progress(
        run_dir,
        lambda: OptimizeRunner().run(
            kind=cast("OptimizeKind", args.optimize_kind),
            model=model,
            data_yaml=args.data,
            run_id=args.run_id,
            run_root=args.run_root,
            goal=goal,
            profile=profile,
            execute=args.execute,
            confirm_full_run=args.confirm_full_run,
            auto_advance=not args.no_auto_advance,
            auto_rounds=args.auto_rounds,
            training_config_path=training_config,
            dataset_manifest_mode=dataset_manifest_mode,
            component_path=component_path,
            search_space_path=search_space_path,
            loop_policy_path=loop_policy_path,
            preset_name=preset.name,
            max_steps=args.max_steps,
            auto_import=not args.no_auto_import,
        ),
        enabled=args.execute,
    )
    _print_optimize_summary(result, preset_name=preset.name)
    if not result.ok:
        return 1
    return 0


def run_optimize_advance_command(args: argparse.Namespace) -> int:
    """Advance an existing one-command optimization run."""
    print("Starting YOLO Agent optimize advance", flush=True)
    print(f"Run dir: {args.run}  Profile: {args.to_profile}  Mode: {'execute' if args.execute else 'dry-run'}", flush=True)
    if args.execute:
        print("progress: real execution requested; watching run events. Use Ctrl+C to stop the CLI.", flush=True)
    result = _run_with_event_progress(
        args.run,
        lambda: OptimizeRunner().advance(
            run_dir=args.run,
            to_profile=cast("TrainingBudgetProfileName", args.to_profile),
            execute=args.execute,
            confirm_full_run=args.confirm_full_run,
            auto_advance=not args.no_auto_advance,
            max_steps=args.max_steps,
            auto_import=not args.no_auto_import,
        ),
        enabled=args.execute,
    )
    _print_optimize_summary(result, preset_name=None)
    if not result.ok:
        return 1
    return 0


def run_optimize_auto_loop_command(args: argparse.Namespace) -> int:
    """Continue auto optimization from an existing pilot run."""
    print("Starting YOLO Agent optimize auto-loop", flush=True)
    print(
        f"Run dir: {args.run}  Auto rounds: {args.auto_rounds}  Mode: {'execute' if args.execute else 'dry-run'}",
        flush=True,
    )
    if args.execute:
        print("progress: auto-loop may fork child runs; use loop status on the latest child run for live training details.", flush=True)
    try:
        llm_config = load_llm_decision_config()
    except Exception:
        llm_config = None
    if llm_config is not None and llm_config.can_generate_proposals:
        print(
            "progress: generating diagnosis and guarded proposals; "
            f"LLM analysis may wait up to {llm_config.timeout_seconds}s before rule fallback.",
            flush=True,
        )
    else:
        print("progress: generating diagnosis and guarded proposals with rule fallback.", flush=True)
    result = _run_with_event_progress(
        args.run,
        lambda: AutoOptimizationLoopDriver().run(
            base_run_dir=args.run,
            auto_rounds=args.auto_rounds,
            execute=args.execute,
            executor="ultralytics-train" if args.execute else "dry-run",
            max_steps=args.max_steps,
            auto_import=not args.no_auto_import,
            profile="pilot",
        ),
        enabled=False,
    )
    _print_auto_optimization_summary(result)
    return 0


def _print_auto_optimization_summary(result: AutoOptimizationResult) -> None:
    """Print a readable panel for an existing-run auto loop."""
    latest = result.rounds[-1] if result.rounds else None
    print("")
    print("YOLO Agent Auto Loop")
    print("--------------------")
    print(f"Base run: {result.base_run_id}")
    print(f"Mode:     {'execute' if result.executed else 'dry-run'}")
    print(f"Rounds:   {len(result.rounds)}/{result.requested_rounds}")
    print(f"Stop:     {result.stopped_reason}")
    if latest is None:
        print("State:    no round was created")
    elif latest.status == "completed" and latest.executable_count:
        print(f"State:    ready/running pilot candidates in child run {latest.run_id}")
    elif latest.stop_reason == "no_executable_candidates":
        print("State:    guarded stop; no trainable candidate is supported by current adapters")
    elif latest.status in {"blocked", "failed"}:
        print(f"State:    {latest.status}; inspect child run {latest.run_id}")
    else:
        print(f"State:    {latest.status}")
    for round_result in result.rounds:
        print(
            f"  - r{round_result.round_index}: {round_result.run_id} "
            f"status={round_result.status} stop={round_result.stop_reason} "
            f"executable={round_result.executable_count}"
        )
        runnable = [
            item for item in round_result.candidate_assessments
            if item.execution_class == "executable"
        ]
        blocked = [
            item for item in round_result.candidate_assessments
            if item.execution_class != "executable"
        ]
        if runnable:
            print("    runnable:")
            for item in runnable[:3]:
                print(f"      - {item.policy_id}: {item.action_id or item.action_domain}")
                if item.command:
                    print(f"        command: {item.command}")
        if blocked:
            print("    not run:")
            for item in blocked[:4]:
                reason = "; ".join(item.reasons[:2]) if item.reasons else item.execution_class
                print(f"      - {item.policy_id}: {item.execution_class} ({reason})")
    print(f"Summary:  {result.summary_path}")
    print(f"Full candidates: {result.full_candidate_recommendations_path}")
    if latest is not None and latest.executable_count:
        if result.executed:
            print(f"Next:     yolo-agent loop status --run {latest.run_dir}")
        else:
            print(
                "Next:     rerun with --execute to launch the runnable pilot, or inspect the summary first."
            )
    elif latest is not None and latest.stop_reason == "no_executable_candidates":
        print("Next:     implement the required adapters or collect the listed evidence; no training was launched.")
    else:
        print("Next:     Review summary; full COCO still requires --confirm-full-run.")


def _print_optimize_summary(result: OptimizeResult, preset_name: str | None) -> None:
    """Print a readable final panel for one-command optimize runs."""
    queue_issue = _optimize_queue_issue(result)
    evidence_summary = _optimize_evidence_summary(result)
    print("")
    print("YOLO Agent Optimize")
    print("-------------------")
    if preset_name:
        print(f"Preset:   {preset_name}")
    print(f"Run:      {result.run_id}")
    print(f"Run dir:  {result.run_dir}")
    print(f"Profile:  {result.profile}")
    print(f"Mode:     {'execute' if result.executed else 'dry-run'}")
    print(f"State:    {_optimize_state(result)}")
    print(f"Training: {_optimize_training_state(result)}")
    print(f"Queue:    {_format_active_queue_counts(result.queue_counts)}")
    reason = _optimize_reason(result)
    if reason:
        print(f"Reason:   {reason}")
    if queue_issue["blocked_by"]:
        print(f"Blocked:  {queue_issue['blocked_by']}")
    if queue_issue["why"]:
        print(f"Why:      {queue_issue['why']}")
    if result.profile_history:
        print(f"Profiles: {', '.join(result.profile_history)}")
    if evidence_summary:
        print("Result:")
        for line in evidence_summary:
            print(f"  {line}")
    if result.auto_optimization is not None:
        auto = result.auto_optimization
        print("Auto loop:")
        print(f"  rounds={len(auto.rounds)}/{auto.requested_rounds} stop={auto.stopped_reason}")
        if auto.rounds:
            latest = auto.rounds[-1]
            print(
                "  latest="
                f"{latest.run_id} status={latest.status} "
                f"executable={latest.executable_count}"
            )
        print(f"  summary={auto.summary_path}")
        print(f"  full_candidates={auto.full_candidate_recommendations_path}")
    if result.ok:
        print(f"Plan:     {result.experiment_plan_path}")
        print(f"Queue:    {result.queue_path}")
        if result.report_path is not None:
            print(f"Report:   {result.report_path}")
    else:
        print("Preflight errors:")
        for check in result.preflight:
            if check.ok:
                continue
            print(f"  - {check.name}: {check.level} - {check.message}")
    warnings = [check for check in result.preflight if check.level == "warning" and not check.ok]
    if result.ok and warnings:
        print("Warnings:")
        for check in warnings:
            print(f"  - {check.name}: {check.message}")
    next_action = queue_issue["next"] or result.next_action
    print(f"Next:     {next_action}")
    if result.ok:
        print(f"Status:   yolo-agent loop status --run {result.run_dir}")


def _optimize_evidence_summary(result: OptimizeResult) -> list[str]:
    """Return a short evidence-backed training result summary."""
    if not result.ok:
        return []
    item = _optimize_completed_queue_item(result)
    node_id = item.node_id if item is not None else None
    candidate_id = item.candidate_id if item is not None else None
    index = _load_evidence_index(result)
    metrics = _selected_metric_mapping(index, node_id=node_id, candidate_id=candidate_id)
    run_dir = _discover_ultralytics_results_dir(result, node_id=node_id)
    if run_dir is not None:
        metrics = {**parse_ultralytics_run(run_dir), **metrics}
    batch_result = _load_batch_tuning_result(result, node_id=node_id)
    lines: list[str] = []
    completed = result.queue_counts.get("completed", 0)
    if completed:
        lines.append(f"completed profile={result.profile}; training is not running now")
    if node_id:
        lines.append(f"candidate={candidate_id or 'unknown'} node={node_id}")
    score_parts = _format_metric_parts(
        metrics,
        [
            ("map50_95", "mAP50-95"),
            ("map50", "mAP50"),
            ("precision", "precision"),
            ("recall", "recall"),
        ],
    )
    if score_parts:
        lines.append("metrics " + " ".join(score_parts))
    if "model_size_mb" in metrics:
        lines.append(f"model_size={_format_metric_value(metrics['model_size_mb'])} MB")
    runtime_parts = _format_metric_parts(
        metrics,
        [
            ("execution_duration_seconds", "duration_s"),
            ("runtime_avg_it_per_sec", "avg_it/s"),
            ("runtime_avg_gpu_util_percent", "avg_gpu%"),
            ("runtime_max_gpu_memory_used_mb", "max_vram_mb"),
        ],
    )
    if runtime_parts:
        lines.append("runtime " + " ".join(runtime_parts))
    if batch_result:
        selected = batch_result.get("selected_batch")
        reason = str(batch_result.get("reason") or "").strip()
        if selected:
            text = f"batch={selected}"
            if reason:
                text += f" ({reason})"
            lines.append(text)
    gate_metric = _metric_value(index, "fast_baseline_pilot_passed", node_id=node_id, candidate_id=candidate_id)
    if result.profile == "pilot" and gate_metric is True:
        lines.append("conclusion=pilot passed; execution strategy is viable")
        lines.extend(_pilot_screening_advice(metrics))
    elif result.profile == "debug" and result.queue_counts.get("completed", 0):
        lines.append("conclusion=debug sanity passed; continue to pilot")
    if result.profile in {"debug", "pilot"} and result.queue_counts.get("completed", 0):
        lines.append("trust=not a final COCO claim; +2 mAP still needs full baseline, error facts, candidates, and seeds")
    return lines


def _pilot_screening_advice(metrics: dict[str, object]) -> list[str]:
    """Return early, non-final strategy guidance from pilot metrics."""
    advice: list[str] = []
    precision = _float_metric(metrics.get("precision"))
    recall = _float_metric(metrics.get("recall"))
    map50 = _float_metric(metrics.get("map50"))
    map50_95 = _float_metric(metrics.get("map50_95"))
    if precision is not None and recall is not None:
        if recall + 0.08 < precision:
            advice.append(
                "pilot_signal=recall lags precision; prioritize false-negative mining, small-object/long-tail sampling, and threshold analysis"
            )
        elif precision + 0.08 < recall:
            advice.append(
                "pilot_signal=precision lags recall; prioritize background false-positive mining and hard negatives"
            )
    if map50 is not None and map50_95 is not None and map50 - map50_95 >= 0.12:
        advice.append(
            "pilot_signal=large mAP50-to-mAP50-95 gap; prioritize localization error facts before changing model components"
        )
    if not advice:
        advice.append("pilot_signal=metrics are usable for screening; mine error facts before proposing full-budget candidates")
    advice.append("next_screening=generate COCO error facts and pilot-only proposals; reserve full COCO for selected candidates")
    return advice


def _float_metric(value: object) -> float | None:
    """Coerce a metric-like object to float when possible."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def _load_evidence_index(result: OptimizeResult) -> EvidenceIndex:
    """Load candidate metric evidence for an optimize run."""
    try:
        evidence = EvidenceStore(result.run_dir.parent).load_run(result.run_dir.name)
    except Exception:
        return EvidenceIndex([])
    return EvidenceIndex(evidence.metric_records)


def _selected_metric_mapping(
    index: EvidenceIndex,
    *,
    node_id: str | None,
    candidate_id: str | None,
) -> dict[str, object]:
    """Select one trusted value per useful optimize metric."""
    metric_names = [
        "map50_95",
        "map50",
        "precision",
        "recall",
        "model_size_mb",
        "execution_duration_seconds",
        "runtime_avg_it_per_sec",
        "runtime_avg_gpu_util_percent",
        "runtime_max_gpu_memory_used_mb",
    ]
    metrics: dict[str, object] = {}
    for metric_name in metric_names:
        value = _metric_value(index, metric_name, node_id=node_id, candidate_id=candidate_id)
        if value is not None:
            metrics[metric_name] = value
    return metrics


def _metric_value(
    index: EvidenceIndex,
    metric_name: str,
    *,
    node_id: str | None,
    candidate_id: str | None,
) -> object:
    """Return a trusted metric value, scoped when possible."""
    filters = {"metric_name": metric_name, "verified": True}
    if node_id:
        filters["node_id"] = node_id
    if candidate_id:
        filters["candidate_id"] = candidate_id
    record = index.select_one(**filters)
    if record is None and (node_id or candidate_id):
        record = index.select_one(metric_name=metric_name, verified=True)
    return record.value if record is not None else None


def _format_metric_parts(metrics: dict[str, object], names: list[tuple[str, str]]) -> list[str]:
    """Format selected metric values for the optimize summary."""
    parts: list[str] = []
    for key, label in names:
        if key in metrics:
            parts.append(f"{label}={_format_metric_value(metrics[key])}")
    return parts


def _format_metric_value(value: object) -> str:
    """Format compact metric values without hiding precision."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.5g}"
    return str(value)


def _optimize_completed_queue_item(result: OptimizeResult):
    """Return the completed queue item that best matches the current profile."""
    if not result.queue_path.is_file():
        return None
    try:
        queue = ExecutionQueue.from_yaml(result.queue_path)
    except Exception:
        return None
    completed = [item for item in queue.items if item.status == "completed"]
    if not completed:
        return None
    for item in reversed(completed):
        if str(item.command.metadata.get("training_budget_profile", "")) == result.profile:
            return item
    return completed[-1]


def _discover_ultralytics_results_dir(result: OptimizeResult, *, node_id: str | None) -> Path | None:
    """Find the actual Ultralytics results directory for the completed node."""
    item = _optimize_completed_queue_item(result)
    expected = item.command.expected_artifacts.get("results_csv") if item is not None else None
    if expected is not None:
        expected_path = Path(expected)
        if expected_path.is_file():
            return expected_path.parent
    if node_id is None:
        return None
    pattern = f"{result.run_id}_{node_id}"
    candidates = sorted(
        (
            path
            for path in Path("runs").rglob("results.csv")
            if pattern in path.parent.name
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0].parent if candidates else None


def _load_batch_tuning_result(result: OptimizeResult, *, node_id: str | None) -> dict[str, object]:
    """Load a BatchTuner summary artifact for the completed node."""
    if node_id is None:
        return {}
    path = result.run_dir / "artifacts" / f"{node_id}_batch_tuning_result.json"
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _optimize_state(result: OptimizeResult) -> str:
    """Return a short user-facing optimize state."""
    if not result.ok:
        return "preflight failed"
    counts = result.queue_counts
    if counts.get("running", 0):
        return "running"
    if counts.get("failed", 0):
        return "failed"
    if counts.get("needs_resume", 0):
        return "blocked: needs resume checkpoint"
    if counts.get("blocked_by_resource", 0):
        return "blocked: resource limits"
    if counts.get("paused", 0):
        return "paused"
    if counts.get("needs_evidence", 0):
        return "blocked: waiting for evidence"
    if counts.get("queued", 0):
        return "queued"
    if counts.get("completed", 0):
        return "completed"
    if result.training_loop is not None and result.training_loop.stopped_reason:
        return result.training_loop.stopped_reason
    return "ready"


def _optimize_training_state(result: OptimizeResult) -> str:
    """Return whether a training process should be active after optimize."""
    counts = result.queue_counts
    if not result.executed:
        return "no; dry-run only"
    if counts.get("running", 0):
        return "yes; training process is expected to be running"
    if counts.get("queued", 0):
        return "no; command is queued"
    if any(counts.get(status, 0) for status in ("needs_resume", "blocked_by_resource", "paused", "needs_evidence")):
        return "no; blocked before training"
    if counts.get("completed", 0):
        return "no; this profile finished"
    if counts.get("failed", 0):
        return "no; execution failed"
    return "no active training detected"


def _optimize_reason(result: OptimizeResult) -> str:
    """Return the clearest stop reason for optimize output."""
    failed_checks = [check for check in result.preflight if check.level == "error" and not check.ok]
    if failed_checks:
        return "; ".join(f"{check.name}: {check.message}" for check in failed_checks)
    if result.training_loop is not None and result.training_loop.stopped_reason:
        return result.training_loop.stopped_reason
    return ""


def _optimize_queue_issue(result: OptimizeResult) -> dict[str, str]:
    """Return a beginner-readable queue blocker explanation."""
    empty = {"blocked_by": "", "why": "", "next": ""}
    if not result.queue_path.is_file():
        return empty
    try:
        queue = ExecutionQueue.from_yaml(result.queue_path)
    except Exception:
        return empty
    for item in queue.items:
        if item.status not in {"blocked_by_resource", "paused", "needs_resume", "needs_evidence", "failed", "skipped"}:
            continue
        blockers = list(item.resource_blockers)
        blocked_by = ", ".join(blockers) if blockers else item.status
        if "missing_batch_tuning_result" in blockers:
            profile = item.command.metadata.get("training_budget_profile") or item.command.metadata.get("profile") or "pilot"
            model = item.experiment_node.candidate_config.base_model
            kind = "coco" if result.kind == "coco" else "custom"
            data = _command_arg_value(item.command.argv, "data") or str(result.task_path.parent / "data.yaml")
            return {
                "blocked_by": blocked_by,
                "why": (
                    f"{profile} uses batch=auto and needs a BatchTuner-selected batch before training. "
                    "The Ultralytics executor will generate this evidence automatically before the run."
                ),
                "next": (
                    f"yolo-agent optimize {kind} --model {model} --data {data} "
                    f"--run-id {result.run_id} --profile {profile} --execute"
                ),
            }
        if blockers:
            return {
                "blocked_by": blocked_by,
                "why": item.message or "The execution queue is blocked by a guard.",
                "next": "Resolve the blocker, then rerun the same optimize command.",
            }
        if item.status == "skipped" and "Fast Baseline Gate blocked" in (item.message or ""):
            profile = item.command.metadata.get("training_budget_profile") or item.command.metadata.get("profile") or result.profile
            model = item.experiment_node.candidate_config.base_model
            data = _command_arg_value(item.command.argv, "data") or str(result.task_path.parent / "data.yaml")
            return {
                "blocked_by": "fast_baseline_gate",
                "why": (
                    "Fast Baseline Gate did not recognize the previous debug sanity evidence. "
                    "The gate now reuses prior baseline sanity evidence across debug/pilot/full profiles."
                ),
                "next": (
                    f"yolo-agent optimize {result.kind} --model {model} --data {data} "
                    f"--run-id {result.run_id} --profile {profile} --execute"
                ),
            }
        return {
            "blocked_by": item.status,
            "why": item.message,
            "next": result.next_action,
        }
    return empty


def _command_arg_value(argv: Sequence[str], name: str) -> str | None:
    """Return a value from argv entries like name=value."""
    prefix = f"{name}="
    for arg in argv:
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
    return None


def _format_active_queue_counts(counts: dict[str, int]) -> str:
    """Print only queue statuses that matter to a user."""
    active = {name: value for name, value in sorted(counts.items()) if value}
    if not active:
        return "none"
    return " ".join(f"{name}={value}" for name, value in active.items())


def _print_optimize_next(result: object) -> None:
    """Print machine-readable and copy-paste next steps for optimize commands."""
    next_action = str(getattr(result, "next_action", ""))
    run_dir = getattr(result, "run_dir", None)
    print(f"next_action={next_action}")
    if run_dir is not None and not next_action.lower().startswith("fix preflight"):
        print(f"next: yolo-agent loop status --run {run_dir}")
    elif next_action:
        print(f"next: {next_action}")


def _run_with_event_progress(run_dir: Path, action: Callable[[], T], *, enabled: bool) -> T:
    """Run an action while tailing the run event log for user-visible progress."""
    if not enabled:
        return action()
    _print_existing_queue_hint(run_dir)
    stop_event = threading.Event()
    watcher = threading.Thread(
        target=_watch_event_log,
        args=(run_dir / "events.jsonl", stop_event),
        daemon=True,
    )
    watcher.start()
    try:
        return action()
    except KeyboardInterrupt:
        stop_event.set()
        _handle_user_interrupt(run_dir)
        raise
    finally:
        stop_event.set()
        watcher.join(timeout=1.0)
        _print_recent_queue_hint(run_dir)


def _handle_user_interrupt(run_dir: Path) -> None:
    """Stop known run processes and make the recovery path visible after Ctrl+C."""
    print("\ninterrupt: Ctrl+C received; stopping known training process for this run...", flush=True)
    queue_path = run_dir / "execution_queue.yaml"
    if not queue_path.is_file():
        print(f"interrupt: no execution queue found at {queue_path}", flush=True)
        return
    try:
        store = ExecutionQueueStore(run_dir)
        queue = store.load()
    except Exception as exc:
        print(f"interrupt: could not read execution queue: {exc}", flush=True)
        return
    stopped = 0
    updated = False
    for item in queue.items:
        if item.status != "running":
            continue
        termination = terminate_command_process(item.command)
        if termination.terminated:
            stopped += 1
        item.mark_interrupted(
            "Execution interrupted by user. Rerun queue-refresh before continuing."
        )
        queue = store.update_item(item)
        updated = True
        EventLog(run_dir / "events.jsonl").append(
            run_id=queue.run_id,
            event_type="queue_item_failed",
            status="blocked",
            message=item.message,
            details={
                "queue_id": item.queue_id,
                "node_id": item.node_id,
                "candidate_id": item.candidate_id,
                "interrupted_by_user": True,
                "termination": termination.model_dump(mode="json"),
            },
        )
    if updated:
        print(f"interrupt: marked running queue item as needs_resume; stopped_processes={stopped}", flush=True)
    else:
        print("interrupt: no running queue item was recorded.", flush=True)
    print(f"next: yolo-agent loop status --run {run_dir}", flush=True)
    print(f"next: yolo-agent loop queue-refresh --run {run_dir}", flush=True)


def _watch_event_log(path: Path, stop_event: threading.Event) -> None:
    """Tail events.jsonl and render concise progress lines."""
    offset = path.stat().st_size if path.is_file() else 0
    last_activity = time.monotonic()
    while not stop_event.is_set():
        if path.is_file():
            try:
                size = path.stat().st_size
                if size < offset:
                    offset = 0
                if size > offset:
                    with path.open("r", encoding="utf-8-sig") as file:
                        file.seek(offset)
                        for line in file:
                            _print_event_progress(line)
                            if _is_terminal_optimizer_event(line):
                                stop_event.set()
                                return
                        offset = file.tell()
                    last_activity = time.monotonic()
            except OSError:
                pass
        if time.monotonic() - last_activity > 15:
            _print_live_status_progress(path.parent)
            last_activity = time.monotonic()
        stop_event.wait(1.0)


def _print_event_progress(line: str) -> None:
    """Print one event log line as a user-facing progress message."""
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return
    event_type = str(event.get("event_type") or "event")
    status = str(event.get("status") or "unknown")
    details = event.get("details")
    if not isinstance(details, dict):
        details = {}
    stage = event.get("stage") or details.get("node_id") or details.get("queue_id") or "-"
    message = str(event.get("message") or "")
    if event_type not in {
        "run_initialized",
        "executor_started",
        "executor_log",
        "executor_metric",
        "stage_started",
        "stage_completed",
        "stage_failed",
        "stage_blocked",
        "queue_enqueued",
        "queue_refreshed",
        "queue_item_started",
        "queue_item_completed",
        "queue_item_failed",
        "queue_item_resource_blocked",
        "queue_item_skipped",
        "executor_completed",
        "executor_failed",
        "executor_timeout",
    }:
        return
    if event_type == "executor_log":
        clean = _clean_cli_line(message, limit=160)
        if clean:
            prefix = "preflight" if clean.lower().startswith("batch tuning") else "training"
            print(f"{prefix}: {clean}", flush=True)
        return
    if event_type == "executor_metric":
        return
    print(f"progress: {event_type} stage={stage} status={status} - {_clean_cli_line(message, limit=140)}", flush=True)


def _is_terminal_optimizer_event(line: str) -> bool:
    """Return whether the optimize progress watcher can stop tailing events."""
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return False
    event_type = str(event.get("event_type") or "")
    if event_type not in {"executor_completed", "executor_failed", "executor_timeout"}:
        return False
    message = str(event.get("message") or "")
    return "Training loop driver stopped" in message


def _print_live_status_progress(run_dir: Path) -> None:
    """Print a concise live status snapshot while optimize is waiting."""
    try:
        status = load_loop_status(run_dir)
    except Exception as exc:  # pragma: no cover - defensive UX guard
        print(f"progress: still running; status unavailable: {exc}", flush=True)
        return
    heartbeat = status.training_heartbeat
    if heartbeat is None:
        print("progress: still running; waiting for training heartbeat", flush=True)
        return
    parts: list[str] = []
    is_batch_tuning = "batch_tuning=b" in heartbeat.process_detail
    if is_batch_tuning:
        batch = heartbeat.process_detail.split("batch_tuning=", 1)[1].split()[0]
        parts.append(f"batch tuning {batch} (not formal training yet)")
    if heartbeat.phase and heartbeat.progress_current is not None and heartbeat.progress_total is not None:
        progress = f"{heartbeat.phase} {heartbeat.progress_current}/{heartbeat.progress_total}"
        if heartbeat.progress_percent is not None:
            progress += f" ({heartbeat.progress_percent:g}%)"
        parts.append(progress)
    if heartbeat.epoch is not None and heartbeat.total_epochs is not None:
        parts.append(f"epoch {heartbeat.epoch}/{heartbeat.total_epochs}")
    if heartbeat.gpu_util_percent is not None:
        parts.append(f"GPU {heartbeat.gpu_util_percent:g}%")
    if heartbeat.it_per_sec is not None:
        parts.append(f"{heartbeat.it_per_sec:g} it/s")
    if heartbeat.eta:
        parts.append(f"ETA {heartbeat.eta}")
    if not parts and heartbeat.recent_log_lines:
        parts.append(_clean_cli_line(heartbeat.recent_log_lines[-1], limit=140))
    if parts:
        prefix = "preflight" if is_batch_tuning else "training"
        print(f"{prefix}: {', '.join(parts)}", flush=True)
    else:
        print("progress: running; waiting for Ultralytics output or batch-tuning result", flush=True)


def _clean_cli_line(text: str, limit: int = 140) -> str:
    """Return a terminal-safe single line for progress output."""
    cleaned = CLI_ANSI_ESCAPE_RE.sub("", text)
    cleaned = CLI_CONTROL_CHARS_RE.sub("", cleaned)
    cleaned = cleaned.replace("\r", " ").replace("\n", " ")
    cleaned = "".join(char if 32 <= ord(char) <= 126 else " " for char in cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


def _print_existing_queue_hint(run_dir: Path) -> None:
    """Print current queue state before a long optimize action starts."""
    queue_path = run_dir / "execution_queue.yaml"
    if not queue_path.is_file():
        return
    try:
        queue = ExecutionQueue.from_yaml(queue_path)
    except Exception as exc:  # pragma: no cover - defensive UX guard
        print(f"progress: existing execution queue could not be read: {exc}", flush=True)
        return
    active = {name: int(value) for name, value in queue.counts().items() if value}
    if not active:
        return
    print(f"progress: existing queue state {active}", flush=True)
    for item in queue.items:
        if item.status in {"running", "queued", "paused", "blocked_by_resource", "needs_resume", "needs_evidence"}:
            profile = item.command.metadata.get("training_budget_profile") or item.command.metadata.get("profile") or ""
            print(
                f"progress: queue item {item.node_id} status={item.status} profile={profile}",
                flush=True,
            )
            break


def _print_recent_queue_hint(run_dir: Path) -> None:
    """Print a small hint if the queue still contains running items after the command returns."""
    queue_path = run_dir / "execution_queue.yaml"
    try:
        queue = ExecutionQueue.from_yaml(queue_path)
    except Exception:
        return
    counts = queue.counts()
    if counts.get("running", 0) or counts.get("queued", 0):
        print(f"progress: queue still has pending work; inspect with yolo-agent loop status --run {run_dir}", flush=True)


def _print_loop_results(results: list[object]) -> int:
    for result in results:
        stage = getattr(result, "stage", "unknown")
        status = getattr(result, "status", "unknown")
        message = getattr(result, "message", "")
        print(f"{stage} status={status}")
        if message:
            print(message)
    if results and getattr(results[-1], "status", None) == "failed":
        return 1
    return 0


def _format_queue_counts(counts: dict[str, int]) -> str:
    return " ".join(f"{name}={counts.get(name, 0)}" for name in sorted(counts))


def run_scaffold_command(args: argparse.Namespace) -> int:
    """Run a placeholder command while the harness is being built."""
    config = AgentConfig()
    print(f"yolo-agent {args.command}: scaffold ready")
    print(f"experiment_root={config.experiment_root}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the yolo-agent CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    try:
        return int(handler(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user.", flush=True)
        return 130


if __name__ == "__main__":  # pragma: no cover - exercised by Python's module runner
    raise SystemExit(main())
