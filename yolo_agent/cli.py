"""Command line interface for yolo-agent."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from yolo_agent.agents.ablation_planner import create_ablation_plan
from yolo_agent.agents.annotation_advisor import advise_annotations
from yolo_agent.agents.candidate_generator import generate_plan
from yolo_agent.adapters.ultralytics.training import TrainingBudgetProfileName
from yolo_agent.adapters.ultralytics.training import UltralyticsRunImporter
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.orchestrator import LoopOrchestrator
from yolo_agent.agents.optimize_runner import OptimizeKind, OptimizeRunner
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.loop_status import load_loop_status, render_loop_status
from yolo_agent.core.loop_state import LoopStage
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
from yolo_agent.tools.smoke_runner import SmokeRunner


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
        help="Check training environment, CUDA, data paths, and run directory writability.",
    )
    doctor_parser.add_argument("--data", type=Path, required=True, help="Path to YOLO data.yaml.")
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
    doctor_parser.set_defaults(handler=run_doctor_command)

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
    loop_status.set_defaults(handler=run_loop_status_command)

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
    report = run_doctor(
        data_yaml=args.data,
        model=args.model,
        run_root=args.run_root,
        kind=cast("DatasetKind", args.kind),
        min_disk_gb=args.min_disk_gb,
        min_vram_gb=args.min_vram_gb,
    )
    _print_doctor_report(report)
    return 0 if report.ok else 1


def _print_doctor_report(report: DoctorReport) -> None:
    print(f"doctor status={'ok' if report.ok else 'failed'} errors={report.error_count} warnings={report.warning_count}")
    print(f"data={report.data_yaml}")
    print(f"model={report.model}")
    print(f"run_root={report.run_root}")
    for check in report.checks:
        status = "ok" if check.ok else check.level
        print(f"{check.name}: {status} - {check.message}")
        if not check.ok and check.fix:
            print(f"  fix: {check.fix}")


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
    print(render_loop_status(load_loop_status(args.run)))
    return 0


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
    result = OptimizeRunner().run(
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
        training_config_path=training_config,
        dataset_manifest_mode=dataset_manifest_mode,
        component_path=component_path,
        search_space_path=search_space_path,
        loop_policy_path=loop_policy_path,
        preset_name=preset.name,
        max_steps=args.max_steps,
        auto_import=not args.no_auto_import,
    )
    print(f"preset={preset.name}")
    print(f"run_dir={result.run_dir}")
    print(f"profile={result.profile}")
    print(f"executor={result.executor}")
    print(f"executed={result.executed}")
    if result.profile_history:
        print(f"profile_history={','.join(result.profile_history)}")
    if result.training_loop is not None:
        print(f"driver_steps={len(result.training_loop.steps)}")
        print(f"driver_stopped={result.training_loop.stopped_reason}")
    for check in result.preflight:
        status = "ok" if check.ok else check.level
        print(f"preflight.{check.name}={status} {check.message}")
    if not result.ok:
        _print_optimize_next(result)
        return 1
    print(f"task={result.task_path}")
    print(f"experiment_plan={result.experiment_plan_path}")
    print(f"execution_queue={result.queue_path}")
    print(_format_queue_counts(result.queue_counts))
    if result.report_path is not None:
        print(f"report={result.report_path}")
    _print_optimize_next(result)
    return 0


def run_optimize_advance_command(args: argparse.Namespace) -> int:
    """Advance an existing one-command optimization run."""
    result = OptimizeRunner().advance(
        run_dir=args.run,
        to_profile=cast("TrainingBudgetProfileName", args.to_profile),
        execute=args.execute,
        confirm_full_run=args.confirm_full_run,
        auto_advance=not args.no_auto_advance,
        max_steps=args.max_steps,
        auto_import=not args.no_auto_import,
    )
    print(f"run_dir={result.run_dir}")
    print(f"profile={result.profile}")
    print(f"executor={result.executor}")
    print(f"executed={result.executed}")
    if result.profile_history:
        print(f"profile_history={','.join(result.profile_history)}")
    if result.training_loop is not None:
        print(f"driver_steps={len(result.training_loop.steps)}")
        print(f"driver_stopped={result.training_loop.stopped_reason}")
    for check in result.preflight:
        status = "ok" if check.ok else check.level
        print(f"preflight.{check.name}={status} {check.message}")
    if not result.ok:
        _print_optimize_next(result)
        return 1
    print(f"experiment_plan={result.experiment_plan_path}")
    print(f"execution_queue={result.queue_path}")
    print(_format_queue_counts(result.queue_counts))
    if result.report_path is not None:
        print(f"report={result.report_path}")
    _print_optimize_next(result)
    return 0


def _print_optimize_next(result: object) -> None:
    """Print machine-readable and copy-paste next steps for optimize commands."""
    next_action = str(getattr(result, "next_action", ""))
    run_dir = getattr(result, "run_dir", None)
    print(f"next_action={next_action}")
    if run_dir is not None and not next_action.lower().startswith("fix preflight"):
        print(f"next: yolo-agent loop status --run {run_dir}")
    elif next_action:
        print(f"next: {next_action}")


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
    return int(handler(args))
