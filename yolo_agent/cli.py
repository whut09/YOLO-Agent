"""Command line interface for yolo-agent."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import re
import threading
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

from yolo_agent.agents.ablation_planner import create_ablation_plan
from yolo_agent.agents.annotation_advisor import advise_annotations
from yolo_agent.agents.candidate_generator import generate_plan
from yolo_agent.adapters.ultralytics.training import TrainingBudgetProfileName
from yolo_agent.adapters.ultralytics.training import UltralyticsRunImporter
from yolo_agent.adapters.ultralytics.training import parse_ultralytics_run
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.loop_io import read_yaml
from yolo_agent.agents.orchestrator import LoopOrchestrator
from yolo_agent.agents.auto_optimization_loop import AutoOptimizationLoopDriver, AutoOptimizationResult
from yolo_agent.agents.llm_decision_advisor import openai_responses_transport
from yolo_agent.agents.optimize_runner import OptimizeKind, OptimizeResult, OptimizeRunner
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.evidence_index import EvidenceIndex
from yolo_agent.core.execution_failure import ExecutionFailure, classify_execution_failure
from yolo_agent.core.execution_queue import ExecutionQueue, ExecutionQueueStore
from yolo_agent.core.experiment_graph import ExperimentNode, ExperimentPlan
from yolo_agent.core.event_log import EventLog
from yolo_agent.core.loop_status import load_loop_status, render_loop_status
from yolo_agent.core.loop_state import LoopStage
from yolo_agent.core.llm_config import LLMDecisionConfig, load_llm_decision_config
from yolo_agent.core.optimization_budget import AutoOptimizationBudget
from yolo_agent.core.optimization_objective import (
    OPTIMIZATION_TARGET_METRICS,
    OptimizationGoalError,
    resolve_optimization_objective,
)
from yolo_agent.core.process_probe import terminate_command_process, terminate_run_processes
from yolo_agent.core.run_allocation import RunAllocation, allocate_base_run_id
from yolo_agent.core.run_initialization import write_partial_run_migration_report
from yolo_agent.core.run_migration import assess_run_protocol, write_migration_report
from yolo_agent.core.runbook_preset import load_runbook_preset
from yolo_agent.core.run_lineage import RunLineageStore
from yolo_agent.core.run_context import RunContext
from yolo_agent.core.schemas import AgentConfig
from yolo_agent.core.task_spec import MetricName, TaskSpec
from yolo_agent.certification.runner import RealGpuAcceptanceSuite
from yolo_agent.certification.component_runner import ComponentCertificationRunner
from yolo_agent.certification.component_gpu_suite import PaperComponentGPUSuiteRunner
from yolo_agent.certification.component_schemas import ComponentCertificationReport
from yolo_agent.certification.paper_adapter_factory import (
    PaperAdapterCertificationFactory,
)
from yolo_agent.certification.paper_auto_optimization import (
    PaperAutoOptimizationAcceptanceSuite,
)
from yolo_agent.certification.paper_auto_optimization_terminal import (
    render_paper_auto_optimization_report,
)
from yolo_agent.certification.sahi_runner import SahiInferenceCertificationRunner
from yolo_agent.certification.inference_policy_runner import (
    InferencePolicyCertificationRunner,
)
from yolo_agent.components.adapters.inference.policy import InferencePolicyConfig
from yolo_agent.components.adapters.inference.slicing import SlicingInferenceConfig
from yolo_agent.components.distillation import DISTILLATION_COMPONENTS
from yolo_agent.resources import ResourcePaths
from yolo_agent.research.paper_registry import PaperRegistry
from yolo_agent.research.awesome_snapshot_builder import AwesomeSnapshotBuilder
from yolo_agent.research.snapshot import preflight_research_snapshot
from yolo_agent.research.paper_scout import PaperScout, PaperScoutConfig
from yolo_agent.research.production_pipeline import ResearchProductionPipeline
from yolo_agent.research.llm_paper_analyzer import LLMPaperAnalyzer
from yolo_agent.reports.cross_run_report import generate_cross_run_comparison_report
from yolo_agent.reports.experiment_report import generate_experiment_report
from yolo_agent.tools.coco_error_mining import mine_coco_errors, write_coco_error_report
from yolo_agent.tools.coco_error_importer import import_coco_eval_metrics
from yolo_agent.tools.dataset_stats import profile_dataset
from yolo_agent.tools.executable_paper_coverage import (
    build_executable_coverage_baseline,
)
from yolo_agent.research.executable_coverage_report import (
    write_executable_coverage_artifacts,
)
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

USER_COMMANDS: tuple[str, ...] = (
    "setup",
    "train",
    "status",
    "stop",
)

_HIDDEN_HELP = argparse.SUPPRESS

AUTO_PAPER_RUNTIME_COMPONENTS: tuple[str, ...] = (
    "sampling.small_object",
    "head.p2_small_object",
    "loss.quality.correlation",
    "loss.calibration.bpc",
    "loss.quality.pseudo_iou",
    "distillation.yolo26_teacher_student",
    "assigner.task_aligned",
    "assigner.optimal_transport",
    "assigner.dynamic_smooth_label",
    "neck.multi_scale_fusion",
    "neck.gold_gather_distribute",
    "neck.rtmdet_large_kernel",
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

    subparsers = parser.add_subparsers(dest="command", metavar="{" + ",".join(USER_COMMANDS) + "}")

    advanced_parser = subparsers.add_parser(
        "advanced",
        help=_HIDDEN_HELP,
        description=(
            "Advanced compatibility namespace for doctor, loop, optimize, evidence, "
            "queue, reporting, and reproduction commands."
        ),
    )
    advanced_parser.add_argument("advanced_args", nargs=argparse.REMAINDER)
    advanced_parser.set_defaults(handler=run_advanced_command)

    research_parser = subparsers.add_parser(
        "research",
        help=_HIDDEN_HELP,
    )
    research_subparsers = research_parser.add_subparsers(dest="research_command")
    research_list = research_subparsers.add_parser("list", help="List local research papers.")
    _add_research_filter_arguments(research_list)
    research_list.set_defaults(handler=run_research_list_command)
    research_show = research_subparsers.add_parser("show", help="Show one local research paper.")
    research_show.add_argument("--paper-id", required=True)
    research_show.add_argument("--root", type=Path, default=Path("research"))
    research_show.set_defaults(handler=run_research_show_command)
    research_search = research_subparsers.add_parser("search", help="Search local research papers.")
    research_search.add_argument("--component")
    research_search.add_argument("--component-category")
    research_search.add_argument("--year-from", type=int)
    research_search.add_argument("--year-to", type=int)
    research_search.add_argument("--task-family")
    research_search.add_argument("--detector-family")
    research_search.add_argument("--dataset")
    research_search.add_argument("--metric")
    research_search.add_argument("--framework")
    research_search.add_argument("--official-code", type=_parse_optional_bool)
    research_search.add_argument("--license", dest="license_name")
    research_search.add_argument("--evidence-level")
    research_search.add_argument("--applicability")
    research_search.add_argument("--root", type=Path, default=Path("research"))
    research_search.set_defaults(handler=run_research_search_command)
    research_sync = research_subparsers.add_parser("sync", help="Incrementally sync paper metadata.")
    research_sync.add_argument("--root", type=Path, default=Path("research"))
    research_sync.add_argument("--config", type=Path, default=ResourcePaths.PAPER_SOURCES)
    research_sync.add_argument("--year-from", type=int)
    research_sync.add_argument("--since", type=_parse_iso_datetime)
    research_sync.add_argument("--dry-run", action="store_true")
    research_sync.set_defaults(handler=run_research_sync_command)
    research_import_awesome = research_subparsers.add_parser(
        "import-awesome",
        help="Import a local Awesome-object-detection catalog without network access.",
    )
    research_import_awesome.add_argument("--source", type=Path, required=True)
    research_import_awesome.add_argument("--root", type=Path, default=Path("research"))
    research_import_awesome.add_argument("--config", type=Path, default=ResourcePaths.RESEARCH_SOURCES)
    research_import_awesome.add_argument("--source-commit")
    research_import_awesome.add_argument("--dry-run", action="store_true")
    research_import_awesome.set_defaults(handler=run_research_import_awesome_command)
    research_build = research_subparsers.add_parser(
        "build-snapshot",
        help="Build a frozen offline Paper Intelligence snapshot.",
    )
    research_build.add_argument("--root", type=Path, default=Path("research"))
    research_build.add_argument("--source", choices=["awesome_object_detection"])
    research_build.add_argument(
        "--maturity-registry",
        type=Path,
        default=Path("runs/component_maturity_registry.yaml"),
        help="Machine-local maturity overlays to freeze into the snapshot.",
    )
    research_build.add_argument(
        "--cached-code-root",
        type=Path,
        help=(
            "Optional local cache containing owner/project README and config "
            "metadata; never fetched during snapshot construction."
        ),
    )
    research_build.add_argument("--sync", action="store_true", help="Sync metadata before the offline build.")
    research_build.add_argument("--config", type=Path, default=ResourcePaths.PAPER_SOURCES)
    research_build.add_argument("--year-from", type=int)
    research_build.add_argument("--since", type=_parse_iso_datetime)
    research_build.add_argument("--force", action="store_true", help="Re-run component extraction for all papers.")
    research_build.add_argument(
        "--extract-components",
        action="store_true",
        help="Use the configured LLM during snapshot production; training itself remains offline.",
    )
    research_build.set_defaults(handler=run_research_build_snapshot_command)
    research_coverage = research_subparsers.add_parser(
        "coverage-baseline",
        help="Audit executable paper coverage from a frozen snapshot.",
    )
    research_coverage.add_argument(
        "--root", type=Path, default=Path("research")
    )
    research_coverage.add_argument("--snapshot", type=Path)
    research_coverage.add_argument(
        "--output",
        type=Path,
        default=Path("runs/coverage_baseline.yaml"),
    )
    research_coverage.add_argument("--markdown", type=Path)
    research_coverage.set_defaults(
        handler=run_research_coverage_baseline_command
    )

    init_parser = subparsers.add_parser(
        "init",
        help=_HIDDEN_HELP,
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
        help=_HIDDEN_HELP,
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
        help=_HIDDEN_HELP,
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
        help=_HIDDEN_HELP,
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
        help=_HIDDEN_HELP,
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
    for setup_kind in ("coco", "custom"):
        setup_kind_parser = setup_subparsers.add_parser(
            setup_kind,
            help=f"Prepare local config, LLM config, run id, and {setup_kind} dataset report.",
        )
        setup_kind_parser.add_argument("--data", type=Path, required=True, help="Path to YOLO data.yaml.")
        setup_kind_parser.add_argument("--model", default="yolo26n.pt", help="YOLO model checkpoint/name.")
        setup_kind_parser.add_argument("--run-id", help=f"Run id under --run-root. Defaults to {setup_kind}-{{model stem}}.")
        setup_kind_parser.add_argument("--run-root", type=Path, default=Path("runs"), help="Run root directory.")
        setup_kind_parser.add_argument("--env-file", type=Path, default=Path(".env.local"), help="Local env file to create.")
        setup_kind_parser.add_argument(
            "--llm-config",
            type=Path,
            default=ResourcePaths.LLM_DECISION_LOCAL,
            help="Ignored local LLM config path to create.",
        )
        setup_kind_parser.add_argument(
            "--report",
            type=Path,
            help="Setup report path. Defaults to runs/{run_id}/setup_report.yaml.",
        )
        setup_kind_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing local setup files.")
        setup_kind_parser.set_defaults(handler=run_setup_command, setup_kind=setup_kind)

    train_parser = subparsers.add_parser(
        "train",
        help="One-command automatic YOLO training and pilot optimization.",
    )
    train_parser.add_argument("--model", default="yolo26n.pt", help="YOLO model checkpoint/name.")
    train_parser.add_argument("--data", type=Path, required=True, help="YOLO data.yaml.")
    train_parser.add_argument(
        "--run-id",
        default="coco-yolo26n",
        help="Base run id under --run-root; stale or completed collisions receive an incremented suffix.",
    )
    train_parser.add_argument("--run-root", type=Path, default=Path("runs"), help="Run root directory.")
    train_parser.add_argument(
        "--profile",
        choices=["debug", "pilot", "baseline_full", "baseline_confirm", "candidate_full"],
        help="Training profile. Existing runs infer this automatically; new runs start with debug.",
    )
    train_parser.add_argument(
        "--kind",
        choices=["coco", "custom"],
        default="coco",
        help="Dataset workflow preset.",
    )
    train_parser.add_argument(
        "--goal",
        help="Structured objective such as +2map or +2%%map; defaults to +2map.",
    )
    train_parser.add_argument(
        "--target-metric",
        choices=OPTIMIZATION_TARGET_METRICS,
        help="Explicit primary metric; requires --target-delta and cannot be combined with --goal.",
    )
    train_parser.add_argument(
        "--target-delta",
        type=float,
        help="Normalized absolute gain for --target-metric; use 0.02 for two AP points.",
    )
    train_parser.add_argument(
        "--goal-description",
        help="Natural-language intent stored for diagnosis; it does not replace the executable objective.",
    )
    train_parser.add_argument(
        "--auto-rounds",
        type=int,
        default=None,
        help=(
            "Advanced override for the internal round safety cap. By default budget=auto uses GPU-hour, "
            "pilot-count, and no-improvement limits. Use 0 to stop after pilot."
        ),
    )
    train_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare and validate the run without starting training.",
    )
    train_parser.add_argument(
        "--confirm-full-run",
        action="store_true",
        help="Required for full COCO profiles; prevents accidental long training.",
    )
    train_parser.add_argument(
        "--no-auto-advance",
        action="store_true",
        help="Stop after the requested profile instead of advancing debug to pilot.",
    )
    train_parser.add_argument("--max-steps", type=int, default=8, help="Maximum automatic driver steps.")
    train_parser.add_argument("--no-auto-import", action="store_true", help="Disable automatic evidence import attempts.")
    train_parser.set_defaults(handler=run_train_command)

    status_parser = subparsers.add_parser(
        "status",
        help="Show the current training state, progress, blockers, and next action.",
    )
    status_parser.add_argument("--run", type=Path, required=True, help="Path to runs/{run_id}.")
    status_parser.add_argument("--verbose", action="store_true", help="Show machine-readable details.")
    status_parser.set_defaults(handler=run_loop_status_command)

    stop_parser = subparsers.add_parser(
        "stop",
        help="Stop local training processes for a run.",
    )
    stop_parser.add_argument("--run", type=Path, required=True, help="Path to runs/{run_id}.")
    stop_parser.set_defaults(handler=run_stop_command)

    advise_parser = subparsers.add_parser(
        "advise-labels",
        help=_HIDDEN_HELP,
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
        help=_HIDDEN_HELP,
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
        help=_HIDDEN_HELP,
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
        help=_HIDDEN_HELP,
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
        help=_HIDDEN_HELP,
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
        help=_HIDDEN_HELP,
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
    optimize_auto_loop.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip decision-analysis LLM calls and use deterministic rule proposals for this run.",
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
        optimize_kind.add_argument(
            "--goal",
            help="Structured objective such as +2map; defaults to the preset goal.",
        )
        optimize_kind.add_argument(
            "--target-metric",
            choices=OPTIMIZATION_TARGET_METRICS,
            help="Explicit primary metric; requires --target-delta.",
        )
        optimize_kind.add_argument(
            "--target-delta",
            type=float,
            help="Normalized absolute gain; use 0.02 for two AP points.",
        )
        optimize_kind.add_argument(
            "--goal-description",
            help="Natural-language intent stored separately from the executable objective.",
        )
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
            help=_HIDDEN_HELP,
        )
        command_parser.set_defaults(handler=run_scaffold_command)

    visible_actions = [
        action for action in subparsers._choices_actions if action.help != _HIDDEN_HELP  # type: ignore[attr-defined]
    ]
    order = {name: index for index, name in enumerate(USER_COMMANDS)}
    subparsers._choices_actions = sorted(  # type: ignore[attr-defined]
        visible_actions,
        key=lambda action: order.get(action.dest, len(order)),
    )
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


def run_setup_command(args: argparse.Namespace) -> int:
    """Run the COCO or custom-dataset onboarding setup wizard."""
    result = run_setup_wizard(
        kind=cast("DatasetKind", args.setup_kind),
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
    return _stop_run(args.run, internal=True)


def run_stop_command(args: argparse.Namespace) -> int:
    """Stop local run processes through the beginner-facing command."""
    return _stop_run(args.run, internal=False)


def _stop_run(run_dir: Path, *, internal: bool) -> int:
    """Stop run processes and print the right next command for the caller."""
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
    print(f"next: {_train_command_for_run_dir(run_dir)}")
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


def run_train_command(args: argparse.Namespace) -> int:
    """Run the beginner-facing one-command training workflow."""
    args.allocate_fresh_run = True
    args.run_allocation = None
    budget = AutoOptimizationBudget.from_training_config(
        ResourcePaths.YOLO26_COCO_GOAL,
        explicit_rounds=args.auto_rounds,
    )
    args.optimization_budget = budget
    args.auto_rounds = budget.effective_round_limit
    args.optimize_kind = args.kind
    args.preset = ResourcePaths.COCO_YOLO26_AUTO_PRESET
    args.training_config = None
    args.components = None
    args.search_space = None
    args.loop_policy = None
    args.dataset_manifest_mode = None
    args.execute = not args.dry_run
    args.display_command = "train"
    return run_optimize_command(args)


def _print_auto_budget_startup(budget: AutoOptimizationBudget) -> None:
    """Show bounded cost expectations without promising a fixed round count."""
    if budget.mode == "auto":
        print("Budget: auto; stops when the first cost, evidence, or patience limit is reached", flush=True)
        print(
            f"Expected: {budget.expected_pilot_range} pilot experiments; "
            f"stop after {budget.no_improvement_patience} consecutive no-improvement pilots",
            flush=True,
        )
        print(
            f"Limits: <= {budget.max_gpu_hours:g} GPU hours; "
            f"concurrency={budget.max_concurrent_pilots}; "
            f"internal safety cap={budget.max_rounds_safety} state-machine rounds",
            flush=True,
        )
    else:
        print(
            f"Budget: fixed-round override={budget.explicit_rounds}; objective cost guards still apply",
            flush=True,
        )
    if budget.full_requires_confirmation:
        print("Full: excluded from the automatic budget unless --confirm-full-run is explicit", flush=True)


def _resolve_train_profile(args: argparse.Namespace) -> TrainingBudgetProfileName | None:
    """Infer the active profile for the one-command train entrypoint."""
    if args.profile:
        return cast("TrainingBudgetProfileName", args.profile)
    run_dir = args.run_root / args.run_id
    if run_dir.is_dir():
        try:
            context = RunContext.from_run_dir(run_dir)
        except (OSError, ValueError):
            return None
        profile = str(context.metadata.get("training_profile", "")).strip()
        if profile == "pilot" and _fast_baseline_debug_recovery_required(run_dir):
            return "debug"
        if profile in {"debug", "pilot", "baseline_full", "baseline_confirm", "candidate_full"}:
            return cast("TrainingBudgetProfileName", profile)
    return None


def _fast_baseline_debug_recovery_required(run_dir: Path) -> bool:
    """Return whether a fresh pilot was skipped because debug sanity is absent."""
    queue_path = run_dir / "execution_queue.yaml"
    if not queue_path.is_file():
        return False
    try:
        queue = ExecutionQueue.from_yaml(queue_path)
    except (OSError, TypeError, ValueError):
        return False
    return any(
        item.status == "skipped"
        and "Fast Baseline Gate blocked" in (item.message or "")
        for item in queue.items
    )


def _auto_migrate_legacy_train_run(
    args: argparse.Namespace,
    allocation: RunAllocation,
) -> RunAllocation:
    """Move an executable beginner train request off a stale protocol run."""
    requested_run_id = allocation.requested_run_id
    legacy_dir = args.run_root / requested_run_id
    context_path = legacy_dir / "run_context.yaml"
    if not context_path.is_file():
        return allocation
    try:
        context = RunContext.from_run_dir(legacy_dir)
        assessment = assess_run_protocol(context, EvidenceStore(args.run_root))
    except (OSError, TypeError, ValueError):
        return allocation
    if not assessment.legacy_run:
        return allocation
    migration = write_migration_report(context, assessment)
    migrated_run_id = migration.suggested_run_id
    if not migrated_run_id:
        return allocation
    return RunAllocation(
        requested_run_id=requested_run_id,
        allocated_run_id=migrated_run_id,
        sequence=0,
        reason="legacy_run_migration",
        partial_run_migration_report=(legacy_dir / "artifacts" / "run_migration_report.yaml")
        .resolve()
        .as_posix(),
    )


def run_optimize_command(args: argparse.Namespace) -> int:
    """Run a one-command optimization runbook."""
    try:
        preset = load_runbook_preset(args.preset)
    except (OSError, ValueError) as exc:
        print(f"preset error: {exc}")
        return 1
    model = args.model or preset.default_model
    training_config = args.training_config or preset.training_config
    component_path = args.components or preset.components
    search_space_path = args.search_space or preset.search_space
    loop_policy_path = args.loop_policy or preset.loop_policy
    dataset_manifest_mode = args.dataset_manifest_mode or preset.dataset_manifest_mode
    target_metric = getattr(args, "target_metric", None)
    target_delta = getattr(args, "target_delta", None)
    goal_description = getattr(args, "goal_description", None)
    explicit_target = target_metric is not None or target_delta is not None
    goal = args.goal if explicit_target else (args.goal or preset.default_goal)
    try:
        resolve_optimization_objective(
            goal_expression=goal,
            target_metric=cast("MetricName | None", target_metric),
            target_delta=target_delta,
            goal_description=goal_description,
            baseline_run_id="pending",
            baseline_candidate_id="pending",
            baseline_protocol_hash="pending",
        )
    except (OSError, OptimizationGoalError, ValueError) as exc:
        _print_objective_input_error(args, model=model, error=exc)
        return 2
    research_binding = None
    if getattr(args, "display_command", "optimize") == "train" and args.execute:
        research_root = args.run_root.parent / "research"
        snapshot_preflight = preflight_research_snapshot(research_root)
        snapshot_path = (
            getattr(snapshot_preflight.binding, "research_snapshot_path", None)
            if snapshot_preflight.binding is not None
            else None
        )
        missing_runtime_components = (
            _research_snapshot_missing_automatic_components(
                Path(snapshot_path)
            )
            if snapshot_preflight.ok and snapshot_path is not None
            else []
        )
        refresh_snapshot = False
        if missing_runtime_components:
            print(
                "progress: preparing implemented paper methods for this machine "
                f"({len(missing_runtime_components)} adapters; CPU only).",
                flush=True,
            )
            try:
                certification = PaperAdapterCertificationFactory().run(
                    workdir=args.run_root / "certification" / "auto-paper-adapters",
                    registry_path=args.run_root / "component_maturity_registry.yaml",
                    mode="cpu",
                    model=model,
                    data=str(args.data),
                    resume=True,
                    component_ids=missing_runtime_components,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                print("Paper method preparation failed before run initialization.")
                print(f"  - {exc}")
                return 2
            if certification.status != "passed":
                print("Paper method preparation failed before run initialization.")
                for item in certification.results:
                    if item.status != "passed" and not item.status.startswith("skipped_"):
                        reason = item.errors[0] if item.errors else item.status
                        print(f"  - {item.component_id}: {reason}")
                for component_id, error in certification.discovery_errors.items():
                    print(f"  - {component_id}: {error}")
                return 2
            refresh_snapshot = True
        if (
            snapshot_preflight.ok
            and snapshot_path is not None
            and _research_snapshot_needs_recipe_refresh(Path(snapshot_path))
        ):
            refresh_snapshot = True
        if refresh_snapshot:
            print(
                "progress: updating the local paper recipe snapshot before training.",
                flush=True,
            )
            refreshed = ResearchProductionPipeline(
                research_root,
                maturity_registry=args.run_root / "component_maturity_registry.yaml",
            ).run(include_local_implementations=True)
            if refreshed.status != "completed":
                print("Paper recipe snapshot refresh failed before run initialization.")
                for error in refreshed.errors:
                    print(f"  - {error}")
                return 2
            snapshot_preflight = preflight_research_snapshot(research_root)
        if not snapshot_preflight.ok:
            _print_research_snapshot_preflight_error(
                snapshot_preflight.status,
                snapshot_preflight.reasons,
                research_root=research_root,
                maturity_registry=args.run_root / "component_maturity_registry.yaml",
            )
            return 2
        research_binding = snapshot_preflight.binding
    if getattr(args, "allocate_fresh_run", False):
        explicit_profile = args.profile
        inherited_profile = explicit_profile or _resolve_train_profile(args)
        try:
            allocation = allocate_base_run_id(
                args.run_root,
                args.run_id,
                reuse_existing=bool(args.profile or args.confirm_full_run),
            )
        except ValueError as exc:
            print(f"run-id error: {exc}")
            return 2
        migration_report = write_partial_run_migration_report(
            args.run_root,
            allocation.requested_run_id,
            allocation.allocated_run_id,
        )
        if migration_report is not None:
            allocation = replace(
                allocation,
                partial_run_migration_report=migration_report.resolve().as_posix(),
            )
        args.run_allocation = allocation
        args.run_id = allocation.allocated_run_id
        args.profile = inherited_profile if not allocation.changed else explicit_profile
        if args.execute and getattr(args, "display_command", "optimize") == "train":
            allocation = _auto_migrate_legacy_train_run(args, allocation)
            args.run_allocation = allocation
            args.run_id = allocation.allocated_run_id
            if allocation.reason == "legacy_run_migration":
                args.profile = explicit_profile
    profile = cast("TrainingBudgetProfileName", args.profile or preset.default_profile)
    try:
        preset.require_profile(profile)
    except ValueError as exc:
        print(f"preset error: {exc}")
        return 1
    run_dir = args.run_root / args.run_id
    display_command = getattr(args, "display_command", "optimize")
    optimization_budget = getattr(args, "optimization_budget", None)
    run_allocation = getattr(args, "run_allocation", None)
    print(f"Starting YOLO Agent {display_command}", flush=True)
    if isinstance(run_allocation, RunAllocation) and run_allocation.changed:
        print(f"Requested run: {run_allocation.requested_run_id}", flush=True)
        allocation_detail = (
            "isolated current protocol"
            if run_allocation.reason == "legacy_run_migration"
            else f"sequence {run_allocation.sequence}"
        )
        print(
            f"Allocated run: {run_allocation.allocated_run_id} ({allocation_detail})",
            flush=True,
        )
    if (
        isinstance(run_allocation, RunAllocation)
        and run_allocation.partial_run_migration_report
    ):
        message = (
            "Migration: isolated legacy run; report="
            if run_allocation.reason == "legacy_run_migration"
            else "Migration: preserved incomplete requested run; report="
        )
        print(message + run_allocation.partial_run_migration_report, flush=True)
    print(f"Run: {args.run_id}  Profile: {profile}  Mode: {'execute' if args.execute else 'dry-run'}", flush=True)
    print(f"Data: {args.data}", flush=True)
    if isinstance(optimization_budget, AutoOptimizationBudget):
        _print_auto_budget_startup(optimization_budget)
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
            target_metric=cast("MetricName | None", target_metric),
            target_delta=target_delta,
            goal_description=goal_description,
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
            run_allocation=run_allocation if isinstance(run_allocation, RunAllocation) else None,
            research_binding=research_binding,
        ),
        enabled=args.execute,
        include_child_runs=args.auto_rounds > 0,
    )
    result.optimization_budget = optimization_budget
    _print_optimize_summary(result, preset_name=preset.name)
    if not result.ok:
        return 1
    return 0


def _print_objective_input_error(
    args: argparse.Namespace,
    *,
    model: str,
    error: Exception,
) -> None:
    """Render an actionable objective correction without exposing a traceback."""
    print(f"objective error: {error}")
    raw_goal = str(getattr(args, "goal", "") or "").strip()
    description = str(getattr(args, "goal_description", "") or raw_goal).strip()
    metric = "ap_small" if "ap_small" in description.lower() else "map50_95"
    command = [
        "yolo-agent",
        "train",
        "--model",
        model,
        "--data",
        str(args.data),
        "--run-id",
        str(args.run_id),
        "--target-metric",
        metric,
        "--target-delta",
        "0.02",
    ]
    if description:
        command.extend(["--goal-description", description])
    print("Next: " + " ".join(_powershell_argument(item) for item in command))


def _research_snapshot_missing_automatic_components(snapshot_dir: Path) -> list[str]:
    """Return implemented priority adapters absent from the frozen runtime contract."""
    contracts_path = snapshot_dir / "component_contracts.yaml"
    if not contracts_path.is_file():
        return []
    try:
        payload = read_yaml(contracts_path)
    except (OSError, TypeError, ValueError):
        return []
    components = payload.get("components", []) if isinstance(payload, dict) else []
    if not isinstance(components, dict):
        return []
    executable_maturities = {
        "smoke_passed",
        "gpu_certified",
        "pilot_reproduced",
        "full_reproduced",
        "confirmed_multi_seed",
    }
    missing: list[str] = []
    for component_id in AUTO_PAPER_RUNTIME_COMPONENTS:
        item = components.get(component_id)
        if not isinstance(item, dict):
            continue
        maturity = str(item.get("maturity", "metadata_only"))
        if maturity not in executable_maturities:
            missing.append(component_id)
    return missing


def _research_snapshot_needs_recipe_refresh(snapshot_dir: Path) -> bool:
    """Detect the pre-contract scale bindings produced by older local snapshots."""
    recipes_path = snapshot_dir / "recipes.yaml"
    if not recipes_path.is_file():
        return False
    try:
        payload = read_yaml(recipes_path)
    except (OSError, TypeError, ValueError):
        return False
    recipes = payload.get("recipes", []) if isinstance(payload, dict) else []
    if not isinstance(recipes, list):
        return False
    scale_types = {
        "scale_variation",
        "small_object_false_negative",
        "cross_scale_information_decay",
    }
    for recipe in recipes:
        if not isinstance(recipe, dict):
            continue
        recipe_id = str(recipe.get("recipe_id") or "")
        if not recipe_id.startswith("paper.neck."):
            continue
        targets = recipe.get("target_error_facts", [])
        if not isinstance(targets, list):
            continue
        target_types = {
            str(target.get("fact_type"))
            for target in targets
            if isinstance(target, dict)
        }
        has_small_area_contract = any(
            isinstance(target, dict)
            and target.get("fact_type") == "area_metric"
            and target.get("area") == "small"
            and target.get("metric_name") == "ap_small"
            for target in targets
        )
        if scale_types.intersection(target_types) and not has_small_area_contract:
            return True
    return False


def _print_research_snapshot_preflight_error(
    status: str,
    reasons: list[str],
    *,
    research_root: Path,
    maturity_registry: Path,
) -> None:
    """Render a deterministic snapshot recovery command without a traceback."""
    print("research snapshot preflight failed")
    print(f"Status: {status}")
    print(f"Reason: {', '.join(reasons) or 'snapshot_not_current'}")
    command = [
        "yolo-agent",
        "research",
        "build-snapshot",
        "--root",
        str(research_root),
        "--source",
        "awesome_object_detection",
        "--maturity-registry",
        str(maturity_registry),
    ]
    print("Next: " + " ".join(_powershell_argument(item) for item in command))


def _powershell_argument(value: str) -> str:
    if value and not any(
        character.isspace() or character in "'\"`$"
        for character in value
    ):
        return value
    return "'" + value.replace("'", "''") + "'"


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
    previous_disable_local_llm = os.environ.get("YOLO_AGENT_DISABLE_LOCAL_LLM")
    if args.no_llm:
        os.environ["YOLO_AGENT_DISABLE_LOCAL_LLM"] = "1"
        llm_config = None
        print("progress: LLM disabled for this run; using deterministic rule proposals.", flush=True)
    else:
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
    try:
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
            enabled=args.execute,
            include_child_runs=True,
        )
    finally:
        if args.no_llm:
            if previous_disable_local_llm is None:
                os.environ.pop("YOLO_AGENT_DISABLE_LOCAL_LLM", None)
            else:
                os.environ["YOLO_AGENT_DISABLE_LOCAL_LLM"] = previous_disable_local_llm
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
    elif not result.executed and latest.status == "completed" and latest.executable_count:
        print(f"State:    dry-run planned pilot candidates in child run {latest.run_id}; no training was launched")
    elif latest.status == "completed" and latest.executable_count:
        print(f"State:    ready/running pilot candidates in child run {latest.run_id}")
    elif latest.stop_reason == "no_executable_candidates":
        print("State:    guarded stop; no trainable candidate is supported by current adapters")
    elif latest.stop_reason == "method_candidates_exhausted":
        print("State:    method search exhausted; scalar HPO stayed disabled")
    elif latest.stop_reason == "paper_adapter_implementation_required":
        print("State:    paper methods found; runtime adapter implementation is required")
    elif latest.stop_reason == "no_certified_paper_components":
        print("State:    no certified paper component can enter automatic training")
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
        for line in _auto_round_paper_lines(round_result):
            print(f"    {line}")
    print(f"Summary:  {result.summary_path}")
    print(f"Full candidates: {result.full_candidate_recommendations_path}")
    if latest is not None and latest.executable_count:
        if result.executed:
            print("Next:     system will automatically continue while training work remains")
        else:
            print(f"Next:     {_train_command_for_run_dir(result.base_run_dir)}")
    else:
        print(f"Next:     {_train_command_for_run_dir(result.base_run_dir)}")


def _print_optimize_summary(result: OptimizeResult, preset_name: str | None) -> None:
    """Print a readable final panel for one-command optimize runs."""
    if result.full_run_status is not None:
        status = result.full_run_status
        print("")
        print("YOLO Agent Full Run")
        print("-------------------")
        print(f"Stage:    {status.stage}")
        print(f"Seed:     {status.seed or '-'} / {status.seed_total}")
        print(f"Progress: {status.progress}")
        print(f"Cost:     {status.gpu_hours_used:.2f}/{status.gpu_hours_authorized:.2f} GPU hours")
        print(f"Stop:     {status.stop_reason}")
        return
    queue_issue = _optimize_queue_issue(result)
    evidence_summary = _optimize_evidence_summary(result)
    latest_auto = result.auto_optimization.rounds[-1] if result.auto_optimization and result.auto_optimization.rounds else None
    print("")
    print("YOLO Agent Optimize")
    print("-------------------")
    if preset_name:
        print(f"Preset:   {preset_name}")
    print(f"Run:      {result.run_id}")
    print(f"Run dir:  {result.run_dir}")
    print(f"Profile:  {result.profile}")
    print(f"Mode:     {_optimize_mode_label(result)}")
    if latest_auto is None:
        print(f"State:    {_optimize_state(result)}")
        print(f"Training: {_optimize_training_state(result)}")
        print(f"Queue:    {_format_active_queue_counts(result.queue_counts)}")
    else:
        print(f"State:    {_auto_round_state_label(latest_auto)}")
        print(f"Training: {_auto_round_training_state(latest_auto)}")
        auto_counts = latest_auto.training_loop.queue_counts if latest_auto.training_loop is not None else {}
        round_issue = _auto_round_execution_issue(latest_auto)
        print(
            f"Queue:    {_auto_round_queue_label(round_issue)}"
            if round_issue is not None
            else f"Queue:    {_format_active_queue_counts(auto_counts)}"
        )
    reason = _optimize_reason(result)
    if reason:
        print(f"Reason:   {reason}")
    user_summary = _optimize_user_summary_lines(result, evidence_summary)
    if user_summary:
        print("Outcome:")
        for line in user_summary:
            print(f"  {line}")
    if latest_auto is not None and _auto_round_execution_issue(latest_auto) is not None:
        issue = _auto_round_execution_issue(latest_auto)
        next_command = (
            _train_command_after_adapter_fix(result)
            if issue is not None
            and getattr(issue.get("failure"), "kind", None) == "adapter_runtime_failed"
            else _train_command_for_optimize_result(result)
        )
        print(f"Next:     {next_command}")
        print(f"Status:   yolo-agent status --run {latest_auto.run_dir}")
        return
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
        if auto.certification_attempted:
            print("Safety check:")
            if auto.certification_status == "passed":
                print("  automatic GPU certification=PASSED; candidate optimization was authorized")
            else:
                print("  automatic GPU certification=FAILED; candidate optimization was not started")
            if auto.certification_report_path is not None:
                print(f"  report={auto.certification_report_path}")
            if auto.certification_failure:
                print(f"  reason={auto.certification_failure}")
        budget = result.optimization_budget
        if budget is not None and budget.mode == "auto":
            print("Auto budget:")
            print(f"  mode=auto stop={_auto_stop_display(auto)}")
            objective_status = auto.objective_status
            completed_pilots = (
                objective_status.completed_pilot_rounds
                if objective_status is not None
                else len(auto.rounds)
            )
            print(
                f"  pilots={completed_pilots}/{budget.max_pilots} "
                f"no_improvement_patience={budget.no_improvement_patience}"
            )
            if objective_status is not None:
                gpu_used = max(0.0, budget.max_gpu_hours - objective_status.gpu_budget_remaining)
                print(f"  gpu_hours={gpu_used:.3f}/{budget.max_gpu_hours:g}")
            if auto.rounds:
                print(
                    f"  safety_round={auto.rounds[-1].round_index}/{budget.max_rounds_safety} "
                    "(internal guard, not a promised experiment count)"
                )
        else:
            print("Auto loop:")
            print(f"  rounds={len(auto.rounds)}/{auto.requested_rounds} stop={auto.stopped_reason}")
        if auto.rounds:
            latest = auto.rounds[-1]
            print(
                "  latest="
                f"{latest.run_id} status={latest.status} "
                f"executable={latest.executable_count}"
            )
            print(f"  outcome={_auto_round_outcome(latest)}")
            paper_lines = _auto_round_paper_lines(latest)
            if paper_lines:
                print("Paper components:")
                for line in paper_lines:
                    print(f"  {line}")
            comparison_lines = _auto_round_comparison_lines(latest)
            if comparison_lines:
                print("Paired comparison:")
                for line in comparison_lines:
                    print(f"  {line}")
        print(f"  summary={auto.summary_path}")
        print(f"  full_candidates={auto.full_candidate_recommendations_path}")
        if auto.asha_state_path is not None:
            print(f"  asha_state={auto.asha_state_path}")
        decision_lines = _auto_optimization_decision_lines(auto)
        if decision_lines:
            print("Decision:")
            for line in decision_lines:
                print(f"  {line}")
    if result.ok:
        print(f"Plan:     {result.experiment_plan_path}")
        print(f"Queue:    {result.queue_path}")
        if result.report_path is not None:
            print(f"Report:   {result.report_path}")
    else:
        if result.migration_report_path is not None:
            print(f"Migration: {result.migration_report_path}")
        if result.migration_suggested_run_id:
            print(f"New run:   {result.migration_suggested_run_id}")
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
    if not result.ok and result.next_action:
        next_action = result.next_action
    elif (
        result.auto_optimization is not None
        and result.auto_optimization.stopped_reason == "optimization_readiness_blocked"
    ):
        next_action = _train_command_for_optimize_result(result)
    elif (
        result.auto_optimization is not None
        and result.auto_optimization.stopped_reason == "method_candidates_exhausted"
    ):
        next_action = "do not rerun this run-id; no untried executable candidates remain"
    elif _result_has_running_work(result, latest_auto):
        next_action = "system will automatically continue after current training and validation"
    else:
        next_action = _train_command_for_optimize_result(result)
    print(f"Next:     {next_action}")
    if result.ok:
        status_dir = latest_auto.run_dir if latest_auto is not None else result.run_dir
        print(f"Status:   yolo-agent status --run {status_dir}")


def _auto_round_training_state(round_result: object) -> str:
    issue = _auto_round_execution_issue(round_result)
    if issue is not None:
        if getattr(issue.get("failure"), "kind", None) == "adapter_runtime_failed":
            return f"stopped; {issue['failed_role']} failed in an adapter hook"
        completed_role = str(issue.get("completed_role") or "one paired run")
        failed_role = str(issue.get("failed_role") or "the other paired run")
        return f"stopped; {completed_role} completed, {failed_role} needs automatic retry"
    training_loop = getattr(round_result, "training_loop", None)
    stop_reason = str(getattr(round_result, "stop_reason", ""))
    if stop_reason == "asha_evidence_incomplete":
        return "finished; checkpoints saved, missing COCO evaluation will retry automatically"
    if training_loop is None:
        if stop_reason == "no_guarded_candidates":
            return "no; baseline completed but no candidate pilot was scheduled"
        if stop_reason == "no_executable_candidates":
            return "no; candidates were considered but none passed executable gates"
        if stop_reason == "method_candidates_exhausted":
            return "no; method candidates are exhausted and scalar HPO is disabled"
        if stop_reason == "paper_adapter_implementation_required":
            return "no; relevant paper candidates require executable runtime adapters"
        if stop_reason == "no_certified_paper_components":
            return "no; no artifact-backed paper component passed all training gates"
        if stop_reason == "no_new_asha_trials":
            return "no; executable candidates were found but ASHA registered no pilot trials"
        return "no; round stopped before executable training"
    counts = getattr(training_loop, "queue_counts", {})
    if int(counts.get("running", 0)) > 0:
        return "yes; latest auto-round candidate is training"
    if int(counts.get("completed", 0)) > 0:
        return "no; latest auto-round pilot completed"
    return "no; no executable candidate ran"


def _auto_round_state_label(round_result: object) -> str:
    """Render a concise state that distinguishes planning stops from training results."""
    if _auto_round_execution_issue(round_result) is not None:
        issue = _auto_round_execution_issue(round_result)
        if issue is not None and getattr(issue.get("failure"), "kind", None) == "adapter_runtime_failed":
            return "BLOCKED - paper adapter failed during training"
        return "BLOCKED - paired comparison incomplete"
    stop_reason = str(getattr(round_result, "stop_reason", ""))
    round_index = getattr(round_result, "round_index", "?")
    if stop_reason == "no_guarded_candidates":
        return f"auto round {round_index} blocked during candidate planning"
    if stop_reason == "no_executable_candidates":
        return f"auto round {round_index} blocked by executable-component gates"
    if stop_reason == "method_candidates_exhausted":
        return f"auto round {round_index} stopped after method candidates were exhausted"
    if stop_reason == "paper_adapter_implementation_required":
        return f"auto round {round_index} stopped for paper adapter implementation"
    if stop_reason == "no_certified_paper_components":
        return f"auto round {round_index} stopped with no certified paper components"
    if stop_reason == "no_new_asha_trials":
        return f"auto round {round_index} blocked before ASHA registration"
    if stop_reason == "asha_evidence_incomplete":
        return "BLOCKED - candidate COCO evaluation incomplete"
    return f"auto round {round_index} {getattr(round_result, 'status', 'unknown')}"


def _auto_round_outcome(round_result: object) -> str:
    """Explain what an auto-round terminal state means without implying a model result."""
    issue = _auto_round_execution_issue(round_result)
    if issue is not None:
        if getattr(issue.get("failure"), "kind", None) == "adapter_runtime_failed":
            return "candidate_training=failed; adapter crashed; no paired optimization decision"
        return (
            f"comparison=not_available; {issue['failed_role']} failed with "
            f"{issue['failure'].kind}; no optimization decision was made"
        )
    stop_reason = str(getattr(round_result, "stop_reason", ""))
    if stop_reason == "no_guarded_candidates":
        return "candidate_training=not_started; zero proposals reached deterministic evaluation"
    if stop_reason == "no_executable_candidates":
        return "candidate_training=not_started; candidate proposals failed execution gates"
    if stop_reason == "method_candidates_exhausted":
        return "candidate_training=not_started; method recipes are exhausted and scalar HPO is disabled"
    if stop_reason == "paper_adapter_implementation_required":
        return "candidate_training=not_started; relevant paper recipes need runtime-integrated adapters"
    if stop_reason == "no_certified_paper_components":
        return "candidate_training=not_started; no paper component has a valid maturity and method-profile binding"
    if stop_reason == "no_new_asha_trials":
        count = int(getattr(round_result, "executable_count", 0))
        return (
            f"candidate_training=not_started; candidates_planned={count}; "
            "ASHA_trials_registered=0"
        )
    if stop_reason == "missing_error_facts":
        return "candidate_training=not_started; required COCO error facts are missing"
    if stop_reason == "asha_evidence_incomplete":
        return "training=completed; comparison=no decision; missing COCO evidence will be retried"
    return "candidate_training=completed" if getattr(round_result, "training_loop", None) is not None else "candidate_training=not_started"


def _auto_round_comparison_lines(round_result: object) -> list[str]:
    """Render candidate/control values and paired deltas from the latest ASHA artifact."""
    run_dir = getattr(round_result, "run_dir", None)
    if run_dir is None:
        return []
    paths = sorted(Path(run_dir).glob("artifacts/*_paired_experiment_result.json"))
    if not paths:
        issue = _auto_round_execution_issue(round_result)
        if issue is not None:
            metric = issue.get("completed_map50_95")
            completed = str(issue.get("completed_candidate_id") or issue["completed_role"])
            metric_text = f" mAP50-95={float(metric):.6f}" if isinstance(metric, (int, float)) else ""
            return [
                f"completed={completed}{metric_text}",
                f"missing={issue['failed_role']}; {issue['failure'].summary}",
                "paired_delta=unavailable; improvement or regression cannot be determined",
                f"automatic_retry={_settings_text(issue['failure'].recovery_overrides)}",
            ]
        return ["status=unavailable; paired experiment result artifact was not written"]
    payload = _read_json_mapping(paths[0])
    if payload is None:
        return ["status=unavailable; paired experiment result artifact could not be read"]
    candidate_id = str(payload.get("candidate_id") or "unknown")
    baseline_id = str(payload.get("baseline_candidate_id") or "unknown")
    lines = [
        f"candidate={candidate_id} baseline={baseline_id} protocol={payload.get('protocol_match_status', 'unknown')}"
    ]
    deltas = payload.get("metric_deltas")
    if not isinstance(deltas, dict):
        return [*lines, "status=incomplete; metric deltas were not recorded"]
    labels = {
        "map50_95": "mAP50-95",
        "latency_ms": "latency_ms",
        "model_size_mb": "model_size_mb",
    }
    for metric_name in ("map50_95", "latency_ms", "model_size_mb"):
        metric = deltas.get(metric_name)
        if not isinstance(metric, dict):
            lines.append(f"{labels[metric_name]}=missing")
            continue
        baseline = _float_metric(metric.get("baseline_value"))
        candidate = _float_metric(metric.get("candidate_value"))
        delta = _float_metric(metric.get("paired_delta"))
        if baseline is None or candidate is None or delta is None:
            lines.append(f"{labels[metric_name]}=incomplete")
            continue
        direction = _comparison_direction(metric_name, delta)
        lines.append(
            f"{labels[metric_name]} candidate={candidate:.6f} baseline={baseline:.6f} "
            f"paired_delta={delta:+.6f} ({direction})"
        )
    blockers = payload.get("blockers")
    if payload.get("verified") is True:
        lines.append("status=verified; ASHA may evaluate promotion")
    else:
        lines.append("status=incomplete; promotion blocked")
        if _primary_pair_missing(payload):
            provisional = _provisional_training_map_values(Path(run_dir), payload)
            candidate_value = provisional.get("candidate")
            baseline_value = provisional.get("baseline")
            if candidate_value is not None:
                lines.append(f"candidate_training_mAP50-95={candidate_value:.6f} (provisional)")
            if baseline_value is not None:
                lines.append(f"baseline_training_mAP50-95={baseline_value:.6f} (provisional)")
            lines.append("blocker=candidate/control fixed COCO val2017 metrics are not both available")
        elif isinstance(blockers, list) and blockers:
            lines.append(f"blockers={'; '.join(str(item) for item in blockers[:3])}")
    primary = deltas.get("map50_95")
    primary_delta = _float_metric(primary.get("paired_delta")) if isinstance(primary, dict) else None
    if primary_delta is not None and primary_delta <= 0:
        lines.append("conclusion=accuracy regressed or did not improve; reject candidate despite resource changes")
    elif primary_delta is not None:
        lines.append("conclusion=accuracy improved; promotion still requires diagnosis and guard gates")
    return lines


def _read_json_mapping(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _primary_pair_missing(payload: dict[str, Any]) -> bool:
    deltas = payload.get("metric_deltas")
    if not isinstance(deltas, dict):
        return True
    primary = deltas.get("map50_95")
    return not isinstance(primary, dict) or any(
        _float_metric(primary.get(key)) is None
        for key in ("baseline_value", "candidate_value", "paired_delta")
    )


def _provisional_training_map_values(
    run_dir: Path,
    paired_payload: dict[str, Any],
) -> dict[str, float]:
    candidate_id = str(paired_payload.get("candidate_id") or "")
    values: dict[str, float] = {}
    for path in (run_dir / "artifacts" / "execution_results").glob("*.json"):
        result = _read_json_mapping(path)
        if result is None:
            continue
        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            continue
        value = _float_metric(metrics.get("map50_95"))
        if value is None:
            continue
        result_candidate = str(result.get("candidate_id") or "")
        if result_candidate == candidate_id:
            values["candidate"] = value
        elif result_candidate == "matched_baseline_control":
            values["baseline"] = value
    return values


def _auto_round_paper_lines(round_result: object) -> list[str]:
    """Render frozen paper provenance and effective runtime eligibility."""
    path = getattr(round_result, "paper_recipe_plan_path", None)
    if path is None or not Path(path).is_file():
        return []
    try:
        raw = read_yaml(Path(path))
    except (OSError, TypeError, ValueError):
        return []
    rows = raw.get("paper_component_decisions", [])
    executable = raw.get("executable_pilot_policies", [])
    if not isinstance(rows, list):
        rows = []
    if not isinstance(executable, list):
        executable = []
    lines: list[str] = []
    selected = [item for item in executable if isinstance(item, dict)]
    if selected:
        paper_selected = [item for item in selected if _policy_paper_ids(item)]
        lines.append(
            f"candidate recipes selected={len(selected)} "
            f"(paper={len(paper_selected)}, local={len(selected) - len(paper_selected)})"
        )
        for item in selected[:3]:
            expected = item.get("expected_improvement")
            expected = expected if isinstance(expected, dict) else {}
            paper_ids = _policy_paper_ids(item)
            component_ids = item.get("components") or expected.get("component_ids") or []
            component_text = ",".join(str(value) for value in component_ids) if isinstance(component_ids, list) else str(component_ids)
            source = f"paper:{','.join(paper_ids)}" if paper_ids else "local evidence"
            lines.append(
                f"recipe={item.get('action_id') or item.get('candidate_id') or 'unknown'} "
                f"source={source} component={component_text or 'unknown'}"
            )
    else:
        lines.append(
            "paper recipes selected=0; current cohort uses local evidence-bound method recipes"
        )
    eligible = [
        row for row in rows
        if isinstance(row, dict) and row.get("eligible") is True
    ]
    if eligible:
        component_ids = ",".join(
            str(row.get("component_id") or "unknown") for row in eligible[:3]
        )
        lines.append(f"certified paper components available={len(eligible)}: {component_ids}")
    stale = 0
    rejected = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        reasons = row.get("rejection_reasons") or []
        if row.get("eligible") is not True:
            rejected += 1
        if any("frozen_adapter_hash_mismatch" in str(reason) for reason in reasons):
            stale += 1
    if rejected or stale:
        lines.append(f"paper component summary: eligible={len(eligible)} rejected={rejected} stale={stale}")
    if not selected and eligible:
        critic_reports = raw.get("recipe_critic_reports", [])
        blocked_recipes: list[str] = []
        if isinstance(critic_reports, list):
            for report in critic_reports:
                if not isinstance(report, dict):
                    continue
                findings = report.get("findings", [])
                if not isinstance(findings, list) or not any(
                    isinstance(finding, dict)
                    and finding.get("code") == "missing_bound_error_facts"
                    for finding in findings
                ):
                    continue
                recipe_id = str(report.get("recipe_id") or "unknown")
                if recipe_id.startswith("paper."):
                    blocked_recipes.append(recipe_id)
        if blocked_recipes:
            names = ",".join(blocked_recipes[:3])
            lines.append(
                "paper blocker=certified components did not enter training because "
                f"{names} lacked a matching local error-fact binding"
            )
    if str(getattr(round_result, "stop_reason", "")) == "no_certified_paper_components":
        lines.append("Scalar HPO: disabled")
    return lines


def _policy_paper_ids(policy: dict[str, object]) -> list[str]:
    """Read paper provenance from the policy identity without inventing it."""
    expected = policy.get("expected_improvement")
    if not isinstance(expected, dict):
        return []
    values = expected.get("paper_ids")
    if isinstance(values, list):
        return [str(value) for value in values if str(value)]
    if isinstance(values, str) and values:
        return [values]
    return []


def _comparison_direction(metric_name: str, delta: float) -> str:
    """Describe whether a paired delta helps its metric objective."""
    if metric_name in {"latency_ms", "model_size_mb"}:
        return "improved" if delta < 0 else "regressed" if delta > 0 else "unchanged"
    return "improved" if delta > 0 else "regressed" if delta < 0 else "unchanged"


def _auto_optimization_decision_lines(auto: AutoOptimizationResult) -> list[str]:
    """Return user-facing full-run and next-round decisions from auto-loop outputs."""
    if auto.stopped_reason == "optimization_readiness_blocked":
        next_step = (
            "fix the automatic certification issue and rerun the same train command"
            if auto.certification_attempted
            else "rerun the same train command; certification is handled automatically"
        )
        return [
            "candidate_training=not_started",
            "measured_improvement=none; no candidate was trained or compared",
            f"blocked_by={_readiness_blocker_summary(auto)}",
            f"next={next_step}",
        ]
    if auto.stopped_reason == "no_guarded_candidates":
        return [
            "candidate_training=not_started",
            "why=baseline pilot completed, but candidate planning selected zero proposals; this is not a negative optimization result",
            "next=repair or rerun candidate planning before spending additional GPU budget",
        ]
    if auto.stopped_reason == "no_executable_candidates":
        return [
            "candidate_training=not_started",
            "why=candidate proposals exist, but no candidate passed component maturity, compatibility, and budget gates",
            "next=use an executable adapter or collect the evidence required by the blocked candidates",
        ]
    if auto.stopped_reason == "method_candidates_exhausted":
        return [
            *_exhausted_search_result_lines(auto),
            "why=all eligible method variants were tested or rejected; optimizer/lr/weight-decay fallback is disabled",
            "next=do not rerun this search; no untried executable candidates remain",
        ]
    if auto.stopped_reason == "paper_adapter_implementation_required":
        return [
            "candidate_training=not_started",
            "why=relevant paper recipes exist, but no runtime-integrated smoke-passed adapter can execute them",
            "next=complete the highest-priority adapter implementation and GPU smoke certification",
        ]
    if auto.stopped_reason == "no_certified_paper_components":
        return [
            "candidate_training=not_started",
            "why=no paper component has both a trainable MethodProfile route and valid artifact-backed maturity",
            "next=certify a diagnosis-relevant component; scalar HPO remains disabled",
        ]
    if auto.stopped_reason == "no_new_asha_trials":
        planned = auto.rounds[-1].executable_count if auto.rounds else 0
        return [
            "candidate_training=not_started",
            f"candidates_planned={planned}; candidates_trained=0",
            "measured_improvement=none; no candidate result exists",
            "why=executable candidates were found but no ASHA pilot trial was registered",
            "next=rerun the same train command after updating YOLO Agent",
        ]
    if auto.stopped_reason == "missing_error_facts":
        return [
            "candidate_training=not_started",
            "why=current COCO error facts are missing or incomplete",
            "next=complete COCO post-evaluation evidence recovery before proposing another pilot",
        ]
    try:
        recommendations = read_yaml(auto.full_candidate_recommendations_path)
    except (OSError, ValueError):
        recommendations = {}
    full_candidates = recommendations.get("recommendations", [])
    adapter_required = recommendations.get("adapter_required", [])
    recommendation_only = recommendations.get("recommendation_only", [])
    full_ready = [
        item for item in full_candidates
        if isinstance(item, dict) and str(item.get("promotion_status", "")).lower() in {"full_ready", "promoted"}
    ]
    lines: list[str] = []
    if full_ready:
        names = ", ".join(_unique_candidate_names(full_ready, key="candidate_id", limit=3))
        lines.append(f"full_candidate=ready; consider full training for {names}")
        lines.append("why=pilot evidence passed promotion gate; full still requires 3 seeds and --confirm-full-run")
    elif full_candidates:
        names = ", ".join(_unique_candidate_names(full_candidates, key="candidate_id", limit=3))
        lines.append("full_candidate=not_ready; do not start full COCO for current candidates")
        if names:
            lines.append(f"blocked_candidates={names}")
        lines.append("why=candidate promotion or trusted full-baseline evidence is still missing")
        lines.append("next=continue pilot-only rounds or collect the missing evidence before full training")
    elif adapter_required:
        actions = ", ".join(
            str(item.get("action_id", item.get("policy_id", "unknown")))
            for item in adapter_required[:3]
            if isinstance(item, dict)
        )
        lines.append("full_candidate=none")
        lines.append(f"why=best-looking actions need adapters before they can be honestly trained: {actions}")
        lines.append("next=implement adapters or continue with executable pilot-safe policies")
    else:
        lines.append("full_candidate=none")
        lines.append("why=no candidate passed the guarded auto-loop as full-ready")
        lines.append("next=continue pilot-only rounds; do not spend full COCO budget yet")
    if recommendation_only:
        evidence_actions = ", ".join(
            _unique_candidate_names(recommendation_only, key="action_id", fallback_key="policy_id", limit=3)
        )
        if evidence_actions:
            lines.append(f"evidence_first={evidence_actions}")
    return lines


def _unique_candidate_names(
    items: object,
    *,
    key: str,
    limit: int,
    fallback_key: str | None = None,
) -> list[str]:
    """Return unique display names from recommendation mappings."""
    if not isinstance(items, list):
        return []
    names: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        raw = item.get(key)
        if raw is None and fallback_key is not None:
            raw = item.get(fallback_key)
        name = str(raw or "unknown")
        if name in names:
            continue
        names.append(name)
        if len(names) >= limit:
            break
    return names


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
    if _latest_resource_failure(result) is not None:
        return "RECOVERABLE RESOURCE FAILURE"
    if _optimize_queue_issue(result)["blocked_by"] == "fast_baseline_gate":
        return "BLOCKED before training: debug sanity is missing"
    if (
        result.auto_optimization is not None
        and result.auto_optimization.stopped_reason == "optimization_readiness_blocked"
    ):
        return "blocked: baseline finished, candidate optimization did not start"
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
    if not result.ok:
        return "no; preflight failed before execution"
    if _latest_resource_failure(result) is not None:
        return "stopped; pilot did not complete"
    if _optimize_queue_issue(result)["blocked_by"] == "fast_baseline_gate":
        return "no; pilot did not start and no candidate metrics were produced"
    if (
        result.auto_optimization is not None
        and result.auto_optimization.stopped_reason == "optimization_readiness_blocked"
    ):
        return "no; baseline pilot finished, but no optimization candidate was trained"
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


def _optimize_mode_label(result: OptimizeResult) -> str:
    """Distinguish an unstarted execute request from an intentional dry-run."""
    if not result.ok and result.executor != "dry-run":
        return "execute requested; not started"
    return "execute" if result.executed else "dry-run"


def _optimize_reason(result: OptimizeResult) -> str:
    """Return the clearest stop reason for optimize output."""
    failed_checks = [check for check in result.preflight if check.level == "error" and not check.ok]
    if failed_checks:
        return "; ".join(f"{check.name}: {check.message}" for check in failed_checks)
    resource_failure = _latest_resource_failure(result)
    if resource_failure is not None:
        return resource_failure.summary
    if result.auto_optimization is not None and result.auto_optimization.rounds:
        issue = _auto_round_execution_issue(result.auto_optimization.rounds[-1])
        if issue is not None:
            if issue["failure"].kind == "adapter_runtime_failed":
                return f"{issue['failed_role']} failed inside a paper adapter hook; no comparison was made"
            return (
                f"{issue['failed_role']} ran out of "
                f"{'GPU memory' if issue['failure'].kind == 'gpu_memory_exhausted' else 'system memory'}; "
                "the matched comparison did not finish"
            )
    if _optimize_queue_issue(result)["blocked_by"] == "fast_baseline_gate":
        return "pilot requires a completed debug sanity run"
    if result.auto_optimization is not None:
        auto_reason = result.auto_optimization.stopped_reason
        if auto_reason == "optimization_readiness_blocked":
            return _readiness_blocker_summary(result.auto_optimization)
        if auto_reason == "no_new_asha_trials":
            return "executable candidates were found but no ASHA pilot trial was registered"
        if auto_reason == "method_candidates_exhausted":
            return "search finished without a candidate that reached the requested improvement"
        if auto_reason == "asha_evidence_incomplete":
            return "training finished, but matched COCO evaluation is incomplete; no optimization decision yet"
    if result.training_loop is not None and result.training_loop.stopped_reason:
        return result.training_loop.stopped_reason
    return ""


def _optimize_user_summary_lines(
    result: OptimizeResult,
    evidence_summary: list[str],
) -> list[str]:
    """Put the user decision before protocol and artifact details."""
    queue_issue = _optimize_queue_issue(result)
    resource_failure = _latest_resource_failure(result)
    if resource_failure is not None:
        if resource_failure.waiting_for_external_gpu:
            return _gpu_conflict_summary_lines(resource_failure)
        failed = _settings_text(resource_failure.failed_settings)
        lines = [
            "RESOURCE FAILURE - this was not a model-quality result.",
            f"Problem: {resource_failure.root_cause}",
            f"Failed setting: {failed or 'not recorded'}.",
            "Model result: none; the pilot stopped before a valid mAP comparison was produced.",
        ]
        if resource_failure.recoverable:
            lines.extend(
                [
                    f"Automatic recovery: {_failure_recovery_text(resource_failure)}.",
                    "Action: rerun the same train command; recovery is automatic and bounded.",
                ]
            )
        else:
            lines.append(
                "Action: automatic retries are exhausted; free system RAM, then start a new isolated run-id."
            )
        return lines
    if result.auto_optimization is not None and result.auto_optimization.rounds:
        issue = _auto_round_execution_issue(result.auto_optimization.rounds[-1])
        if issue is not None:
            failure = issue["failure"]
            if isinstance(failure, ExecutionFailure) and failure.waiting_for_external_gpu:
                return [
                    "NO DECISION YET - the candidate and matched baseline were not both completed.",
                    *_gpu_conflict_summary_lines(failure, heading=False),
                    "Comparison: unavailable; this output does not prove improvement or regression.",
                ]
            if isinstance(failure, ExecutionFailure) and failure.kind == "adapter_runtime_failed":
                metric = issue.get("completed_map50_95")
                metric_text = (
                    f"mAP50-95={float(metric):.6f}"
                    if isinstance(metric, (int, float))
                    else "metrics saved"
                )
                return [
                    "CANDIDATE FAILED - the baseline is valid, but the paper adapter crashed.",
                    f"Completed: {issue['completed_role']} ({metric_text}).",
                    f"Failed: {issue['failed_role']} - {failure.root_cause}",
                    "mAP improvement: not measured; this candidate has no valid comparison.",
                    "Action: fix the adapter before it can re-enter training.",
                    "Next run: start a new run-id after the code update; do not retry this failed candidate.",
                ]
            metric = issue.get("completed_map50_95")
            metric_text = (
                f"mAP50-95={float(metric):.6f}"
                if isinstance(metric, (int, float))
                else "metrics saved"
            )
            return [
                "NO DECISION YET - the candidate and matched baseline were not both completed.",
                f"Completed: {issue['completed_role']} ({metric_text}).",
                f"Failed: {issue['failed_role']} - {issue['failure'].summary}",
                "Comparison: unavailable; this output does not prove improvement or regression.",
                f"Recovery: rerun the same command; {_failure_recovery_text(issue['failure'])}.",
            ]
    if queue_issue["blocked_by"] == "fast_baseline_gate":
        return [
            "BLOCKED - training did not start.",
            "Problem: this fresh run started at pilot, but the required debug sanity run is missing.",
            "Measured result: none; no baseline or candidate mAP was produced in this attempt.",
            "Action: rerun the same train command; YOLO Agent will recover with debug, then continue to pilot automatically.",
        ]
    auto = result.auto_optimization
    if auto is None:
        return []
    if auto.stopped_reason == "method_candidates_exhausted":
        return [
            "SEARCH FINISHED - no improving candidate was found.",
            *_exhausted_search_result_lines(auto),
            "Result: the requested target was not reached.",
            "Action: do not rerun this run-id; all executable method variants were already tested.",
        ]
    if auto.stopped_reason == "no_new_asha_trials":
        latest = auto.rounds[-1] if auto.rounds else None
        planned = latest.executable_count if latest is not None else 0
        return [
            "BLOCKED - candidates were planned, but candidate training did not start.",
            f"Candidates planned: {planned}.",
            "Candidates trained: 0.",
            "mAP improvement: not measured; there is no candidate result to compare.",
            "Action: rerun the same train command after updating YOLO Agent.",
        ]
    if auto.stopped_reason == "asha_evidence_incomplete" and auto.rounds:
        latest = auto.rounds[-1]
        paths = sorted(Path(latest.run_dir).glob("artifacts/*_paired_experiment_result.json"))
        payload = _read_json_mapping(paths[0]) if paths else None
        provisional = (
            _provisional_training_map_values(Path(latest.run_dir), payload)
            if payload is not None
            else {}
        )
        lines = [
            "NO DECISION - training finished, but the matched COCO evaluation is incomplete.",
        ]
        candidate_value = provisional.get("candidate")
        baseline_value = provisional.get("baseline")
        if candidate_value is not None:
            lines.append(
                f"Candidate training mAP50-95={candidate_value:.6f} (provisional, not the paired COCO result)."
            )
        if baseline_value is not None:
            lines.append(
                f"Baseline training mAP50-95={baseline_value:.6f} (provisional, not the paired COCO result)."
            )
        lines.extend(
            [
                "Problem: candidate and baseline do not yet have matching COCO val2017 metrics and AP_small/FN facts.",
                "Action: rerun the same command; saved checkpoints are reused and only missing evaluation is retried.",
            ]
        )
        return lines
    if auto.stopped_reason != "optimization_readiness_blocked":
        return []
    metrics = next(
        (line.removeprefix("metrics ") for line in evidence_summary if line.startswith("metrics ")),
        "pilot metrics unavailable",
    )
    budget = result.optimization_budget
    max_pilots = budget.max_pilots if budget is not None else 0
    report = getattr(auto.readiness, "certification_report", None) if auto.readiness else None
    lines = [
        "BLOCKED - baseline pilot completed successfully; automatic optimization did not start.",
        f"Baseline pilot: {metrics}",
        f"Optimization candidates: 0 trained (0/{max_pilots or '?'} pilot budget used).",
        "mAP improvement: not measured; there is no candidate result to compare.",
        f"Blocker: {_readiness_blocker_summary(auto)}",
    ]
    if report is not None:
        lines.append(f"Certification report: {report}")
    if auto.certification_attempted:
        lines.append("Action: fix the certification issue above, then rerun this same train command.")
    else:
        lines.append("Action: rerun this same train command; the safety check will run automatically.")
    return lines


def _readiness_blocker_summary(auto: AutoOptimizationResult) -> str:
    """Translate readiness contract IDs and validation errors into user language."""
    blockers = list(auto.readiness.blockers) if auto.readiness is not None else []
    if not blockers:
        return "automatic optimization readiness requirements were not satisfied"
    blocker = str(blockers[0])
    if blocker.startswith("gpu_certification_report_invalid:"):
        if "report hash does not match" in blocker:
            return "GPU certification report is invalid or stale (report hash mismatch)"
        return "GPU certification report is invalid or unreadable"
    if blocker == "gpu_certification_report_missing":
        return "GPU certification report is missing"
    if blocker.startswith("gpu_certification_not_passed:"):
        return "GPU certification has not passed"
    if blocker == "gpu_certification_code_hash_mismatch":
        return "GPU certification is stale because the code changed"
    if blocker.startswith("gpu_certification_missing_capability:"):
        return "GPU certification is missing a required automatic-optimization capability"
    if blocker.startswith("gpu_certification_auto_run_failed:"):
        return "automatic GPU certification failed before it could authorize candidate optimization"
    return blocker.split(":", 1)[0].replace("_", " ")


def _auto_stop_display(auto: AutoOptimizationResult) -> str:
    """Render common automatic-loop stops without exposing internal status IDs first."""
    if auto.stopped_reason == "optimization_readiness_blocked":
        return f"blocked - {_readiness_blocker_summary(auto)}"
    if auto.stopped_reason == "no_new_asha_trials":
        return "blocked - candidates planned but ASHA registered no pilot trials"
    if auto.stopped_reason == "method_candidates_exhausted":
        return "finished - no improving candidate found"
    if auto.stopped_reason == "asha_evidence_incomplete":
        return "blocked - matched COCO evaluation incomplete; training will not be repeated"
    return auto.stopped_reason


def _exhausted_search_result_lines(auto: AutoOptimizationResult) -> list[str]:
    """Summarize a completed ASHA search without implying that the final round trained."""
    objective = auto.objective_status
    lines: list[str] = []
    if objective is not None:
        if objective.baseline_value is not None:
            lines.append(f"baseline_mAP50-95={objective.baseline_value:.6f}")
        if objective.observed_delta is not None:
            lines.append(f"best_objective_delta={objective.observed_delta:+.6f}")
        if objective.required_delta is not None:
            lines.append(f"required_delta={objective.required_delta:+.6f}")

    study = _read_asha_summary(auto.asha_state_path)
    trials = study.get("trials") if isinstance(study, dict) else None
    if not isinstance(trials, list):
        return lines
    observed_trials = [
        trial for trial in trials
        if isinstance(trial, dict) and isinstance(trial.get("observations"), list) and trial["observations"]
    ]
    observations = [
        observation
        for trial in observed_trials
        for observation in trial.get("observations", [])
        if isinstance(observation, dict)
    ]
    lines.append(f"candidates_tested={len(observed_trials)}; training_observations={len(observations)}")

    pilot_3 = _best_asha_observation(observed_trials, "pilot_3")
    if pilot_3 is not None:
        candidate_id, observation, trial = pilot_3
        delta = _float_metric(observation.get("paired_delta"))
        if delta is not None:
            lines.append(f"best_pilot_3={_short_candidate_id(candidate_id)} paired_delta={delta:+.6f}")
        pilot_10 = next(
            (
                item for item in trial.get("observations", [])
                if isinstance(item, dict) and item.get("stage_id") == "pilot_10"
            ),
            None,
        )
        if isinstance(pilot_10, dict):
            pilot_10_delta = _float_metric(pilot_10.get("paired_delta"))
            if pilot_10_delta is not None:
                verdict = "rejected" if pilot_10_delta <= 0 else "improved"
                lines.append(f"pilot_10={_short_candidate_id(candidate_id)} paired_delta={pilot_10_delta:+.6f} ({verdict})")
    return lines


def _read_asha_summary(path: Path | None) -> dict[str, object]:
    if path is None or not Path(path).is_file():
        return {}
    try:
        payload = read_yaml(Path(path))
    except (OSError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _best_asha_observation(
    trials: list[dict[str, object]],
    stage_id: str,
) -> tuple[str, dict[str, object], dict[str, object]] | None:
    matches: list[tuple[float, str, dict[str, object], dict[str, object]]] = []
    for trial in trials:
        candidate_id = str(trial.get("candidate_id") or "unknown")
        observations = trial.get("observations")
        if not isinstance(observations, list):
            continue
        for observation in observations:
            if not isinstance(observation, dict) or observation.get("stage_id") != stage_id:
                continue
            delta = _float_metric(observation.get("paired_delta"))
            if delta is not None:
                matches.append((delta, candidate_id, observation, trial))
    if not matches:
        return None
    _, candidate_id, observation, trial = max(matches, key=lambda item: item[0])
    return candidate_id, observation, trial


def _short_candidate_id(candidate_id: str) -> str:
    for prefix in ("next_augmentation_", "next_training_", "next_sampling_", "next_model_"):
        if candidate_id.startswith(prefix):
            return candidate_id.removeprefix(prefix)
    return candidate_id


def _gpu_certification_command(result: OptimizeResult) -> str:
    """Return the concrete command that repairs a blocked optimization readiness gate."""
    auto = result.auto_optimization
    report = (
        getattr(auto.readiness, "certification_report", None)
        if auto is not None and auto.readiness is not None
        else None
    )
    workdir = Path(report).parent if report is not None else result.run_dir.parent / "certification" / "mini-gpu"
    parts = [
        "yolo-agent",
        "advanced",
        "certify-gpu",
        "--workdir",
        str(workdir),
        "--model",
        result.model or "yolo26n.pt",
        "--device",
        "0",
        "--recipe",
        "reduce_mosaic",
        "--execute-real-gpu",
    ]
    return " ".join(_powershell_argument(item) for item in parts)


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
        if item.status == "failed" and item.last_result is not None:
            failure = item.last_result.failure or classify_execution_failure(
                stdout=item.last_result.stdout,
                stderr=item.last_result.stderr,
                command=item.last_result.command,
            )
            if failure is not None:
                return {
                    "blocked_by": failure.kind,
                    "why": failure.root_cause,
                    "next": (
                        "Rerun the same train command; YOLO Agent will apply bounded recovery automatically."
                        if failure.recoverable
                        else "Free system RAM and start a new isolated run-id."
                    ),
                }
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
                "next": _canonical_train_command(
                    kind=kind,
                    model=model,
                    data=data,
                    run_id=result.run_id,
                    run_root=result.run_dir.parent,
                    profile=str(profile),
                ),
            }
        if blockers:
            return {
                "blocked_by": blocked_by,
                "why": item.message or "The execution queue is blocked by a guard.",
                "next": "Resolve the blocker, then rerun the same train command.",
            }
        if item.status == "skipped" and "Fast Baseline Gate blocked" in (item.message or ""):
            profile = item.command.metadata.get("training_budget_profile") or item.command.metadata.get("profile") or result.profile
            model = item.experiment_node.candidate_config.base_model
            data = _command_arg_value(item.command.argv, "data") or str(result.task_path.parent / "data.yaml")
            return {
                "blocked_by": "fast_baseline_gate",
                "why": (
                    "This run entered pilot without a completed debug sanity run. "
                    "No training or accuracy comparison was performed."
                ),
                "next": _canonical_train_command(
                    kind=result.kind,
                    model=model,
                    data=data,
                    run_id=result.run_id,
                    run_root=result.run_dir.parent,
                    profile=str(profile),
                ),
            }
        return {
            "blocked_by": item.status,
            "why": item.message,
            "next": result.next_action,
        }
    return empty


def _latest_resource_failure(result: OptimizeResult) -> ExecutionFailure | None:
    """Load the latest classified queue failure, including legacy generic results."""
    if not result.queue_path.is_file():
        return None
    try:
        queue = ExecutionQueue.from_yaml(result.queue_path)
    except Exception:
        return None
    failed_items = [item for item in queue.items if item.status == "failed" and item.last_result is not None]
    for item in reversed(failed_items):
        execution = item.last_result
        if execution is None:
            continue
        failure = execution.failure or classify_execution_failure(
            stdout=execution.stdout,
            stderr=execution.stderr,
            command=execution.command,
        )
        if failure is not None:
            return failure
    return None


def _settings_text(settings: dict[str, str | int | float | bool | None]) -> str:
    return " ".join(f"{key}={value}" for key, value in settings.items() if value is not None)


def _failure_recovery_text(failure: ExecutionFailure) -> str:
    batch = failure.recovery_overrides.get("batch")
    if failure.recovery_strategy in {
        "retry_same_batch_on_clean_gpu",
        "retry_same_batch_after_external_gpu_cleared",
    }:
        return f"retry the original batch={batch} after confirming the GPU is free"
    settings = _settings_text(failure.recovery_overrides)
    return f"the next attempt will use {settings}" if settings else "wait for the GPU conflict to clear"


def _gpu_conflict_summary_lines(
    failure: ExecutionFailure,
    *,
    heading: bool = True,
) -> list[str]:
    snapshot = failure.gpu_snapshot
    lines = ["GPU BUSY - training is paused; this is not a model failure."] if heading else []
    if snapshot is not None and snapshot.used_memory_mb is not None:
        total = snapshot.total_memory_mb
        usage = f"{snapshot.used_memory_mb} MB" if total is None else f"{snapshot.used_memory_mb}/{total} MB"
        lines.append(f"GPU memory in use: {usage} before YOLO training.")
        for process in snapshot.external_processes[:3]:
            detail = process.command_line or process.process_name
            lines.append(f"Blocking process: PID {process.pid} - {detail}.")
    batch = failure.failed_settings.get("batch")
    lines.extend(
        [
            f"Batch: preserved at {batch}; it was not reduced.",
            "Action: stop the listed GPU workload, then rerun the same train command.",
        ]
    )
    return lines


def _auto_round_execution_issue(round_result: object) -> dict[str, object] | None:
    """Return a user-facing paired-run resource blocker from the child queue."""
    run_dir = getattr(round_result, "run_dir", None)
    if run_dir is None:
        return None
    queue_path = Path(run_dir) / "execution_queue.yaml"
    if not queue_path.is_file():
        return None
    try:
        queue = ExecutionQueue.from_yaml(queue_path)
    except Exception:
        return None
    completed = next((item for item in queue.items if item.status == "completed"), None)
    failed = next(
        (
            item
            for item in queue.items
            if item.status in {"failed", "needs_resume", "queued"}
            and item.last_result is not None
            and item.last_result.status == "failed"
        ),
        None,
    )
    if completed is None or failed is None or failed.last_result is None:
        return None
    failure = classify_execution_failure(
        stdout=failed.last_result.stdout,
        stderr=failed.last_result.stderr,
        command=failed.last_result.command,
    ) or failed.last_result.failure
    if failure is None:
        return None
    completed_is_baseline = _queue_item_is_matched_baseline(completed)
    failed_is_baseline = _queue_item_is_matched_baseline(failed)
    if completed_is_baseline == failed_is_baseline:
        return None
    completed_role = "matched baseline" if completed_is_baseline else "candidate"
    failed_role = "matched baseline" if failed_is_baseline else "candidate"
    completed_metrics = completed.last_result.metrics if completed.last_result is not None else {}
    return {
        "completed_role": completed_role,
        "failed_role": failed_role,
        "completed_candidate_id": completed.candidate_id,
        "completed_map50_95": completed_metrics.get("map50_95"),
        "failure": failure,
    }


def _queue_item_is_matched_baseline(item: object) -> bool:
    candidate_id = str(getattr(item, "candidate_id", ""))
    command = getattr(item, "command", None)
    metadata = getattr(command, "metadata", {}) if command is not None else {}
    return candidate_id == "matched_baseline_control" or metadata.get("matched_baseline_control") is True


def _auto_round_queue_label(issue: dict[str, object] | None) -> str:
    if issue is None:
        return "none"
    failure = issue.get("failure")
    if isinstance(failure, ExecutionFailure) and failure.kind == "adapter_runtime_failed":
        return f"{issue['completed_role']}=done; {issue['failed_role']}=failed in adapter"
    if isinstance(failure, ExecutionFailure) and failure.waiting_for_external_gpu:
        return f"{issue['completed_role']}=done; {issue['failed_role']}=waiting for GPU (batch preserved)"
    return f"{issue['completed_role']}=done; {issue['failed_role']}=automatic retry required"


def _canonical_train_command(
    *,
    kind: str,
    model: str,
    data: str,
    run_id: str,
    run_root: Path,
    profile: str,
) -> str:
    """Return the shortest safe train command for user-facing output."""
    parts = ["yolo-agent", "train"]
    if kind != "coco":
        parts.extend(["--kind", kind])
    parts.extend(["--model", model, "--data", data, "--run-id", run_id])
    if run_root != Path("runs"):
        parts.extend(["--run-root", str(run_root)])
    if profile in {"baseline_full", "baseline_confirm", "candidate_full"}:
        parts.extend(["--profile", profile, "--confirm-full-run"])
    return " ".join(parts)


def _train_command_for_run_dir(run_dir: Path | str) -> str:
    """Return the canonical beginner command, collapsing child runs to their base run."""
    resolved = Path(run_dir)
    child_match = re.fullmatch(r"(?P<base>.+)-r\d+", resolved.name)
    if child_match:
        base_dir = resolved.parent / child_match.group("base")
        if (base_dir / "run_context.yaml").is_file():
            resolved = base_dir
    try:
        context = RunContext.from_run_dir(resolved)
    except (OSError, ValueError):
        return f"yolo-agent train --run-id {resolved.name} --data <data.yaml>"

    model = "yolo26n.pt"
    profile = str(context.metadata.get("training_profile") or "pilot")
    plan_path = context.artifact_path("experiment_plan.yaml")
    if plan_path.is_file():
        try:
            plan = ExperimentPlan.from_yaml(plan_path)
        except (OSError, ValueError):
            plan = None
        if plan is not None and plan.nodes:
            node = plan.nodes[0]
            model = node.candidate_config.base_model
            if node.command_spec is not None:
                profile = str(
                    node.command_spec.metadata.get("training_budget_profile")
                    or node.command_spec.metadata.get("profile")
                    or profile
                )
    kind = "coco" if context.dataset_version.startswith("coco") or "coco" in context.run_id.lower() else "custom"
    return _canonical_train_command(
        kind=kind,
        model=model,
        data=str(context.data_yaml),
        run_id=context.run_id,
        run_root=context.run_dir.parent,
        profile=profile,
    )


def _train_command_for_optimize_result(result: OptimizeResult) -> str:
    """Return a train-only Next command even when preflight stopped before run initialization."""
    if result.migration_suggested_run_id:
        return _canonical_train_command(
            kind=result.kind,
            model=result.model or "yolo26n.pt",
            data=str(result.data_yaml or Path("<data.yaml>")),
            run_id=result.migration_suggested_run_id,
            run_root=result.run_dir.parent,
            profile=result.profile,
        )
    if (result.run_dir / "run_context.yaml").is_file():
        return _train_command_for_run_dir(result.run_dir)
    return _canonical_train_command(
        kind=result.kind,
        model=result.model or "yolo26n.pt",
        data=str(result.data_yaml or Path("<data.yaml>")),
        run_id=result.run_id,
        run_root=result.run_dir.parent,
        profile=result.profile,
    )


def _train_command_after_adapter_fix(result: OptimizeResult) -> str:
    """Start clean ASHA state after a candidate failed because its adapter crashed."""
    match = re.fullmatch(r"(?P<base>.+)-v(?P<version>\d+)", result.run_id)
    base = match.group("base") if match else result.run_id
    version = int(match.group("version")) + 1 if match else 2
    while (result.run_dir.parent / f"{base}-v{version}").exists():
        version += 1
    command = _canonical_train_command(
        kind=result.kind,
        model=result.model or "yolo26n.pt",
        data=str(result.data_yaml or Path("<data.yaml>")),
        run_id=f"{base}-v{version}",
        run_root=result.run_dir.parent,
        profile=result.profile,
    )
    objective_path = result.run_dir / "artifacts" / "optimization_objective.yaml"
    if objective_path.is_file():
        try:
            objective = read_yaml(objective_path)
        except (OSError, TypeError, ValueError):
            objective = None
        if isinstance(objective, dict) and objective.get("goal_expression"):
            command += f" --goal {_powershell_argument(str(objective['goal_expression']))}"
    return command


def _result_has_running_work(result: OptimizeResult, latest_auto: object | None) -> bool:
    """Return whether Next should describe automatic continuation instead of a command."""
    if int(result.queue_counts.get("running", 0)) > 0:
        return True
    training_loop = getattr(latest_auto, "training_loop", None)
    counts = getattr(training_loop, "queue_counts", {}) if training_loop is not None else {}
    return int(counts.get("running", 0)) > 0


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
    run_dir = getattr(result, "run_dir", None)
    next_action = _train_command_for_run_dir(run_dir) if run_dir is not None else str(getattr(result, "next_action", ""))
    print(f"next_action={next_action}")
    if next_action:
        print(f"next: {next_action}")


def _run_with_event_progress(
    run_dir: Path,
    action: Callable[[], T],
    *,
    enabled: bool,
    include_child_runs: bool = False,
) -> T:
    """Run an action while tailing the run event log for user-visible progress."""
    if not enabled:
        return action()
    _print_existing_queue_hint(run_dir)
    stop_event = threading.Event()
    watcher_target = _watch_run_tree_events if include_child_runs else _watch_event_log
    initial_tree_paths = set(_run_tree_event_paths(run_dir)) if include_child_runs else set()
    watcher_args: tuple[object, ...] = (
        (run_dir, stop_event, initial_tree_paths)
        if include_child_runs
        else (run_dir / "events.jsonl", stop_event)
    )
    watcher = threading.Thread(
        target=watcher_target,
        args=watcher_args,
        daemon=False,
    )
    watcher.start()
    try:
        return action()
    except KeyboardInterrupt:
        stop_event.set()
        _handle_user_interrupt(_latest_run_tree_dir(run_dir) if include_child_runs else run_dir)
        raise
    finally:
        stop_event.set()
        watcher.join(timeout=5.0)
        _print_recent_queue_hint(_latest_run_tree_dir(run_dir) if include_child_runs else run_dir)


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
    print(f"next: {_train_command_for_run_dir(run_dir)}", flush=True)


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
                            if stop_event.is_set():
                                break
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


def _watch_run_tree_events(
    base_run_dir: Path,
    stop_event: threading.Event,
    initial_paths: set[Path] | None = None,
) -> None:
    """Tail the base run and any forked child run event logs."""
    offsets: dict[Path, int] = {}
    initial_paths = initial_paths if initial_paths is not None else set(_run_tree_event_paths(base_run_dir))
    last_activity = time.monotonic()
    last_status_dir: Path | None = None
    while True:
        event_paths = _run_tree_event_paths(base_run_dir)
        for path in event_paths:
            if path not in offsets:
                is_initial_path = path in initial_paths
                offsets[path] = path.stat().st_size if is_initial_path and path.is_file() else 0
                run_dir = path.parent
                if run_dir != base_run_dir and not is_initial_path:
                    print(f"progress: following child run {run_dir}", flush=True)
                if is_initial_path:
                    continue
            if not path.is_file():
                continue
            try:
                size = path.stat().st_size
                if size < offsets[path]:
                    offsets[path] = 0
                if size > offsets[path]:
                    with path.open("r", encoding="utf-8-sig") as file:
                        file.seek(offsets[path])
                        for line in file:
                            _print_event_progress(line)
                        offsets[path] = file.tell()
                    last_activity = time.monotonic()
            except OSError:
                pass
        if time.monotonic() - last_activity > 15:
            status_dir = _latest_run_tree_dir(base_run_dir)
            if status_dir != last_status_dir:
                print(f"progress: active run {status_dir}", flush=True)
                last_status_dir = status_dir
            _print_live_status_progress(status_dir)
            last_activity = time.monotonic()
        if stop_event.is_set():
            break
        stop_event.wait(1.0)


def _run_tree_event_paths(base_run_dir: Path) -> list[Path]:
    """Return event logs for a base run and forked auto-loop child runs."""
    root = base_run_dir.parent
    prefix = f"{base_run_dir.name}-r"
    paths = [base_run_dir / "events.jsonl"]
    if root.is_dir():
        children = sorted(
            [path for path in root.iterdir() if path.is_dir() and path.name.startswith(prefix)],
            key=lambda path: path.stat().st_mtime,
        )
        paths.extend(child / "events.jsonl" for child in children)
    return [path for path in paths if path.exists()]


def _latest_run_tree_dir(base_run_dir: Path) -> Path:
    """Return the newest child run directory, or the base run if no child exists."""
    root = base_run_dir.parent
    prefix = f"{base_run_dir.name}-r"
    if not root.is_dir():
        return base_run_dir
    children = [path for path in root.iterdir() if path.is_dir() and path.name.startswith(prefix)]
    if not children:
        return base_run_dir
    return max(children, key=lambda path: path.stat().st_mtime)


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
        "stage_progress",
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
        "auto_round_started",
        "auto_round_decision",
        "auto_round_completed",
        "auto_round_blocked",
        "gpu_certification_started",
        "gpu_certification_completed",
        "gpu_certification_failed",
    }:
        return
    if event_type.startswith("auto_round_"):
        _print_auto_round_progress(event_type, status=status, message=message, details=details)
        return
    if event_type == "executor_log":
        clean = _clean_cli_line(message, limit=160)
        prefix = _executor_log_prefix(clean)
        if prefix:
            print(f"{prefix}: {clean}", flush=True)
        return
    if event_type == "executor_metric":
        return
    print(f"progress: {event_type} stage={stage} status={status} - {_clean_cli_line(message, limit=140)}", flush=True)


def _print_auto_round_progress(event_type: str, *, status: str, message: str, details: dict[str, object]) -> None:
    """Render auto-loop progress with round and strategy context."""
    round_index = details.get("round_index", "?")
    total_rounds = details.get("total_rounds", "?")
    prefix = f"auto[{round_index}/{total_rounds}]"
    if event_type == "auto_round_decision":
        strategy = str(details.get("strategy") or details.get("policy_id") or "unknown")
        diagnosis = str(details.get("diagnosis") or "")
        recipe = str(details.get("recipe") or strategy)
        changed = str(details.get("changed_variable") or strategy)
        execution_class = str(details.get("execution_class") or status)
        remaining = details.get("remaining_candidates")
        reasons = details.get("reasons")
        reason_text = ""
        if isinstance(reasons, list) and reasons:
            reason_text = f" reason={_clean_cli_line(str(reasons[0]), limit=80)}"
        context = f" diagnosis={_clean_cli_line(diagnosis, limit=48)}" if diagnosis else ""
        remaining_text = f" remaining={remaining}" if remaining is not None else ""
        print(
            f"progress: {prefix} strategy={strategy} class={execution_class}"
            f"{context} recipe={recipe} changed={changed}{remaining_text}{reason_text}",
            flush=True,
        )
        return
    clean = _clean_cli_line(message, limit=140)
    print(f"progress: {prefix} {event_type.replace('auto_round_', '')} status={status} - {clean}", flush=True)


def _executor_log_prefix(clean: str) -> str:
    """Return a user-facing prefix for important executor logs, or empty to hide noise."""
    if not clean:
        return ""
    lowered = clean.lower()
    if lowered.startswith("batch tuning") or "batch tuning cache hit" in lowered:
        return "preflight"
    if "traceback" in lowered or lowered.startswith(("error", "runtimeerror", "exception")):
        return "error"
    if "results saved to" in lowered or "training complete" in lowered or "ultralytics training completed" in lowered:
        return "training"
    if _is_ultralytics_noise_line(clean):
        return ""
    if _is_progress_log_line(clean):
        return "training"
    return ""


def _is_progress_log_line(clean: str) -> bool:
    """Return whether a line is a concise train/val progress line worth printing."""
    lowered = clean.lower()
    if "it/s" not in lowered and "eta" not in lowered:
        return False
    if "%" not in clean and not re.search(r"\b\d+/\d+\b", clean):
        return False
    if re.search(r"^\d+/\d+\b", clean):
        return True
    return any(token in lowered for token in ("epoch", "train", "val", "valid", "box", "map", "gpu", "eta"))


def _is_ultralytics_noise_line(clean: str) -> bool:
    """Suppress verbose Ultralytics boilerplate that is still saved in log files."""
    lowered = clean.lower()
    noise_prefixes = (
        "ultralytics ",
        "engine\\trainer:",
        "engine/trainer:",
        "from n params module arguments",
        "transferred ",
        "freezing layer",
        "optimizer:",
        "albumentations:",
        "image sizes ",
        "using ",
        "logging results",
        "starting training",
        "learn more at ",
    )
    if lowered.startswith(noise_prefixes):
        return True
    if re.match(r"^\d+\s+[-\[]", clean):
        return True
    if re.match(r"^[A-Za-z][A-Za-z0-9_ /-]{1,32}\s+\d+\s+\d+\s+0?\.\d+", clean):
        return True
    if "ultralytics.nn.modules" in lowered or "torch.nn.modules" in lowered:
        return True
    if "class images instances" in lowered:
        return True
    return False


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
    if not (run_dir / "run_context.yaml").is_file():
        print("progress: initializing run context", flush=True)
        return
    try:
        status = load_loop_status(run_dir)
    except Exception as exc:  # pragma: no cover - defensive UX guard
        print(f"progress: still running; status unavailable: {exc}", flush=True)
        return
    heartbeat = status.training_heartbeat
    if heartbeat is None:
        if status.current_stage_status == "running":
            if status.current_stage == "profile_data" and _print_profile_data_progress(run_dir):
                return
            print(f"progress: stage {status.current_stage} is still running", flush=True)
        else:
            print("progress: still running; waiting for execution heartbeat", flush=True)
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


def _print_profile_data_progress(run_dir: Path, *, stale_after_seconds: float = 60.0) -> bool:
    """Render the profiler heartbeat and identify abandoned profiling stages."""
    path = run_dir / "artifacts" / "dataset_profile_progress.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        phase = str(payload["phase"])
        status = str(payload.get("status") or "running")
        current = int(payload.get("current") or 0)
        total_value = payload.get("total")
        total = None if total_value is None else int(total_value)
        percent_value = payload.get("percent")
        percent = None if percent_value is None else float(percent_value)
        pid = int(payload["pid"])
        updated_at = datetime.fromisoformat(str(payload["updated_at"]).replace("Z", "+00:00"))
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return False

    age_seconds = max(0.0, (datetime.now(timezone.utc) - updated_at).total_seconds())
    if status == "running" and age_seconds > stale_after_seconds and not _profile_process_is_alive(pid):
        print(
            f"progress: profile_data stale: profiler process {pid} is not running "
            f"(last heartbeat {age_seconds:.0f}s ago)",
            flush=True,
        )
        return True

    progress = f"{current}/{total}" if total is not None else str(current)
    if percent is not None:
        progress += f" ({percent:g}%)"
    prefix = "progress: profile_data"
    if status == "failed":
        print(f"{prefix} failed during {phase}: {progress}", flush=True)
    elif status == "completed":
        print(f"{prefix} completed {phase}: {progress}", flush=True)
    else:
        print(f"{prefix} {phase}: {progress}", flush=True)
    return True


def _profile_process_is_alive(pid: int) -> bool:
    """Check process liveness without sending Windows control signals."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))) and exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


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
        print(f"progress: queue still has pending work; inspect with yolo-agent status --run {run_dir}", flush=True)


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


def _parse_optional_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _parse_iso_datetime(value: str) -> object:
    from datetime import datetime, timezone

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected ISO date or datetime") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _add_research_filter_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=Path("research"))
    parser.add_argument("--year-from", type=int)
    parser.add_argument("--year-to", type=int)
    parser.add_argument("--task-family")
    parser.add_argument("--detector-family")
    parser.add_argument("--component")
    parser.add_argument("--component-category")
    parser.add_argument("--dataset")
    parser.add_argument("--metric")
    parser.add_argument("--framework")
    parser.add_argument("--official-code", type=_parse_optional_bool)
    parser.add_argument("--license", dest="license_name")
    parser.add_argument("--evidence-level")
    parser.add_argument("--applicability")


def _research_filters(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: value
        for key, value in {
            "year_from": args.year_from,
            "year_to": args.year_to,
            "task_family": args.task_family,
            "detector_family": args.detector_family,
            "component": args.component,
            "component_category": args.component_category,
            "dataset": args.dataset,
            "metric": args.metric,
            "framework": args.framework,
            "official_code": args.official_code,
            "license": args.license_name,
            "evidence_level": args.evidence_level,
            "applicability": args.applicability,
        }.items()
        if value is not None
    }


def run_research_list_command(args: argparse.Namespace) -> int:
    """List local paper records without accessing the network."""
    papers = PaperRegistry(args.root).list(**_research_filters(args))
    for paper in papers:
        code = "code" if paper.official_code_url else "no-code"
        print(f"{paper.paper_id}	{paper.year}	{code}	{paper.title}")
    print(f"papers={len(papers)}")
    return 0


def run_research_show_command(args: argparse.Namespace) -> int:
    """Show one local paper as JSON."""
    paper = PaperRegistry(args.root).get(args.paper_id)
    if paper is None:
        print(f"paper_not_found={args.paper_id}")
        return 1
    print(json.dumps(paper.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run_research_search_command(args: argparse.Namespace) -> int:
    """Search local paper records using structured filters."""
    papers = PaperRegistry(args.root).list(**_research_filters(args))
    for paper in papers:
        print(f"{paper.paper_id}	{paper.year}	{paper.title}")
    print(f"papers={len(papers)}")
    return 0


def run_research_sync_command(args: argparse.Namespace) -> int:
    """Incrementally collect metadata without affecting the training harness."""
    config = PaperScoutConfig.from_yaml(args.config)
    result = PaperScout(PaperRegistry(args.root), config=config).sync(
        since=args.since,
        year_from=args.year_from,
        dry_run=args.dry_run,
    )
    mode = "dry-run" if result.dry_run else "write"
    print(f"paper_sync mode={mode} sources={result.sources_attempted} queries={result.queries_attempted}")
    print(
        f"pages={result.pages_fetched} seen={result.records_seen} "
        f"normalized={result.records_normalized} writes={result.registry_writes}"
    )
    for error in result.errors:
        print(f"error: {error}")
    return 1 if result.errors else 0


def run_research_import_awesome_command(args: argparse.Namespace) -> int:
    """Import a local Awesome catalog; this command never accesses the network."""
    try:
        result = AwesomeSnapshotBuilder(args.root, config_path=args.config).import_catalog(
            args.source,
            dry_run=args.dry_run,
            source_commit=args.source_commit,
        )
    except Exception as exc:
        print(f"error: {exc}")
        return 1
    mode = "dry-run" if result.dry_run else "write"
    print("Awesome Catalog Import")
    print("----------------------")
    print(f"Mode:       {mode}")
    print(f"Records:    {result.catalog_record_count}")
    print(f"Would add:  {result.would_import_count}")
    print(f"Imported:   {result.imported_count}")
    print(f"Skipped:    {result.skipped_count}")
    print(f"Conflicts:  {result.conflict_count}")
    print(f"Catalog:    {result.catalog_hash}")
    print(f"Commit:     {result.source_commit}")
    if not result.dry_run:
        print(f"Next:       yolo-agent research build-snapshot --root {args.root} --source awesome_object_detection")
    return 0


def run_research_build_snapshot_command(args: argparse.Namespace) -> int:
    """Produce a replayable research snapshot before training starts."""
    if args.source:
        analyzer = (
            LLMPaperAnalyzer(
                transport=openai_responses_transport,
                ledger_path=args.root / "production" / "research_decision_ledger.jsonl",
            )
            if args.extract_components
            else None
        )
        result = AwesomeSnapshotBuilder(
            args.root,
            config_path=ResourcePaths.RESEARCH_SOURCES,
            analyzer=analyzer,
            maturity_registry=args.maturity_registry,
            cached_code_root=args.cached_code_root,
        ).build(source_name=args.source, force=args.force)
        print("Research Snapshot")
        print("-----------------")
        print(f"Status:     {result.status}")
        if result.import_result:
            print(f"Records:    {result.import_result.catalog_record_count}")
            print(f"Catalog:    {result.import_result.catalog_hash}")
            print(f"Commit:     {result.import_result.source_commit}")
        print(f"Paper AI:   {result.paper_intelligence}")
        if result.unavailable_reason:
            print(f"Reason:     {result.unavailable_reason}")
        if result.snapshot_hash:
            print(f"Snapshot:   {result.snapshot_hash}")
            print(f"Path:       {result.snapshot_path}")
            print(f"Maturity:   {args.maturity_registry}")
        for error in result.errors:
            print(f"Error:      {error}")
        return 0 if result.status == "completed" else 1
    scout = None
    if args.sync:
        scout = PaperScout(
            PaperRegistry(args.root),
            config=PaperScoutConfig.from_yaml(args.config),
        )
    analyzer = (
        LLMPaperAnalyzer(
            transport=openai_responses_transport,
            ledger_path=args.root / "production" / "research_decision_ledger.jsonl",
        )
        if args.extract_components
        else None
    )
    result = ResearchProductionPipeline(
        args.root,
        analyzer=analyzer,
        maturity_registry=args.maturity_registry,
        cached_code_root=args.cached_code_root,
    ).run(
        sync=args.sync,
        scout=scout,
        since=args.since,
        year_from=args.year_from,
        force=args.force,
        include_local_implementations=True,
    )
    print("Research Snapshot")
    print("-----------------")
    print(f"Status:     {result.status}")
    print(f"Papers:     {result.paper_count}")
    print(f"Components: {result.component_count}")
    print(f"Recipes:    {result.recipe_count}")
    print(f"Paper AI:   {result.paper_intelligence}")
    if result.unavailable_reason:
        print(f"Reason:     {result.unavailable_reason}")
    print(f"Maturity:   {result.maturity_summary.model_dump_json()}")
    if result.snapshot_hash:
        print(f"Snapshot:   {result.snapshot_hash}")
        print(f"Path:       {result.snapshot_path}")
        print(f"Registry:   {args.maturity_registry}")
    for error in result.errors:
        print(f"Error:      {error}")
    return 0 if result.status == "completed" else 1


def run_research_coverage_baseline_command(args: argparse.Namespace) -> int:
    """Render four explicit paper coverage denominators from frozen evidence."""
    try:
        report = build_executable_coverage_baseline(
            snapshot=args.snapshot,
            research_root=args.root,
        )
        markdown = args.markdown or args.output.with_suffix(".md")
        write_executable_coverage_artifacts(
            report,
            yaml_path=args.output,
            markdown_path=markdown,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    print("Executable Paper Coverage")
    print("-------------------------")
    for name, denominator in report.denominators.items():
        print(f"{name}: {denominator.paper_count}")
    print(f"Reusable:  {report.reusable_adapter_paper_count}")
    print(f"Runtime:   {report.runtime_ready_paper_count}")
    print(f"YAML:      {args.output}")
    print(f"Markdown:  {markdown}")
    return 0


def run_scaffold_command(args: argparse.Namespace) -> int:
    """Run a placeholder command while the harness is being built."""
    config = AgentConfig()
    print(f"yolo-agent {args.command}: scaffold ready")
    print(f"experiment_root={config.experiment_root}")
    return 0


def run_advanced_command(args: argparse.Namespace) -> int:
    """Dispatch advanced commands through their existing compatibility parsers."""
    advanced_args = list(args.advanced_args)
    if not advanced_args:
        print("yolo-agent advanced: choose doctor, loop, optimize, or another internal command")
        print("Research commands remain under yolo-agent research.")
        return 0
    if advanced_args[0] == "certify-paper-auto":
        parser = argparse.ArgumentParser(
            prog="yolo-agent advanced certify-paper-auto"
        )
        parser.add_argument(
            "--workdir",
            type=Path,
            default=Path("runs/certification/paper-auto"),
        )
        parser.add_argument("--research-root", type=Path, default=Path("research"))
        parser.add_argument(
            "--source",
            type=Path,
            help=(
                "Local Awesome-object-detection checkout or exported papers.json. "
                "When omitted, the existing offline source manifest is rebuilt."
            ),
        )
        parser.add_argument("--source-commit")
        parser.add_argument(
            "--registry",
            type=Path,
            default=Path("runs/component_maturity_registry.yaml"),
        )
        parser.add_argument("--policy-root", type=Path, default=Path("runs"))
        parser.add_argument("--model", default="yolo26n.pt")
        parser.add_argument("--device", default="0")
        parser.add_argument("--execute-real-gpu", action="store_true")
        certify_args = parser.parse_args(advanced_args[1:])
        report = PaperAutoOptimizationAcceptanceSuite().run(
            workdir=certify_args.workdir,
            research_root=certify_args.research_root,
            source=certify_args.source,
            source_commit=certify_args.source_commit,
            maturity_registry=certify_args.registry,
            policy_memory_root=certify_args.policy_root,
            model=certify_args.model,
            device=certify_args.device,
            execute_real_gpu=certify_args.execute_real_gpu,
        )
        for line in render_paper_auto_optimization_report(report):
            print(line)
        print(
            "Report:    "
            f"{certify_args.workdir / PaperAutoOptimizationAcceptanceSuite.report_name}"
        )
        return 0 if report.status in {"passed", "skipped"} else 1
    if advanced_args[0] == "certify-gpu":
        parser = argparse.ArgumentParser(prog="yolo-agent advanced certify-gpu")
        parser.add_argument("--workdir", type=Path, default=Path("runs/certification/mini-gpu"))
        parser.add_argument("--model", default="yolo26n.pt")
        parser.add_argument("--device", default="0")
        parser.add_argument("--recipe", default="reduce_mosaic")
        parser.add_argument("--execute-real-gpu", action="store_true")
        certify_args = parser.parse_args(advanced_args[1:])
        report = RealGpuAcceptanceSuite().run(
            workdir=certify_args.workdir,
            model=certify_args.model,
            device=certify_args.device,
            execute_real_gpu=certify_args.execute_real_gpu,
            recipe_id=certify_args.recipe,
        )
        print("YOLO Agent GPU Certification")
        print("----------------------------")
        print(f"Status:   {report.status}")
        component_stage = next(
            (
                stage
                for stage in report.stages
                if stage.stage_id == "component_runtime_certification"
            ),
            None,
        )
        if component_stage is not None:
            print(f"Component: {component_stage.status}")
        runtime_stage = next(
            (stage for stage in report.stages if stage.stage_id == "runtime_adapter"),
            None,
        )
        if runtime_stage is not None:
            print(
                "Runtime:  "
                f"hook_called={runtime_stage.metrics.get('train_dataloader_hook_called')} "
                f"manifest_matched={runtime_stage.metrics.get('manifest_payload_matched')}"
            )
        if report.objective is not None:
            print(
                "Objective: "
                f"{report.objective.primary_metric} "
                f"delta={report.objective.observed_delta} "
                f"passed={report.objective.passed}"
            )
            if report.objective.target_error_fact_deltas:
                error_deltas = ", ".join(
                    f"{name}={value:+.6f}"
                    for name, value in sorted(
                        report.objective.target_error_fact_deltas.items()
                    )
                )
                print(f"Error:    {error_deltas}")
        if report.asha_survivor:
            print(f"ASHA:     survivor={report.asha_survivor}")
        if report.failures:
            print(f"Reason:   {report.failures[0]}")
        print(f"Report:   {certify_args.workdir / 'certification_report.yaml'}")
        return 0 if report.status in {"passed", "skipped"} else 1
    if advanced_args[0] == "certify-paper-adapters":
        parser = argparse.ArgumentParser(
            prog="yolo-agent advanced certify-paper-adapters"
        )
        parser.add_argument(
            "--workdir",
            type=Path,
            default=Path("runs/certification/paper-adapters"),
        )
        parser.add_argument(
            "--registry",
            type=Path,
            default=Path("runs/component_maturity_registry.yaml"),
        )
        mode = parser.add_mutually_exclusive_group()
        mode.add_argument(
            "--cpu", action="store_true", help="Run offline CPU certification (default)."
        )
        mode.add_argument(
            "--gpu", action="store_true", help="Prepare an opt-in real GPU batch."
        )
        selection = parser.add_mutually_exclusive_group()
        selection.add_argument(
            "--resume", action="store_true", help="Resume from verified per-adapter reports."
        )
        selection.add_argument(
            "--changed-only",
            action="store_true",
            help="Certify only new or changed runtime identities.",
        )
        parser.add_argument(
            "--component",
            action="append",
            dest="components",
            help="Limit the batch to one component; repeat to select multiple.",
        )
        parser.add_argument("--model", default="yolo26n.pt")
        parser.add_argument("--data", default="coco.yaml")
        parser.add_argument("--device", default="0")
        parser.add_argument("--teacher")
        parser.add_argument("--ensemble-teacher")
        parser.add_argument(
            "--execute-real-gpu",
            action="store_true",
            help="Explicitly allow real CUDA execution; valid only with --gpu.",
        )
        certify_args = parser.parse_args(advanced_args[1:])
        certification_mode = "gpu" if certify_args.gpu else "cpu"
        if certify_args.execute_real_gpu and certification_mode != "gpu":
            parser.error("--execute-real-gpu requires --gpu")
        options_by_component = _batch_certification_options(
            teacher=certify_args.teacher,
            ensemble_teacher=certify_args.ensemble_teacher,
        )
        try:
            report = PaperAdapterCertificationFactory().run(
                workdir=certify_args.workdir,
                registry_path=certify_args.registry,
                mode=certification_mode,
                model=certify_args.model,
                data=certify_args.data,
                device=certify_args.device,
                execute_real_gpu=certify_args.execute_real_gpu,
                resume=certify_args.resume,
                changed_only=certify_args.changed_only,
                component_ids=certify_args.components,
                options_by_component=options_by_component,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print("YOLO Agent Paper Adapter Certification")
            print("--------------------------------------")
            print("Status:    failed")
            print(f"Reason:    {exc}")
            print(f"Workdir:   {certify_args.workdir}")
            return 1
        print("YOLO Agent Paper Adapter Certification")
        print("--------------------------------------")
        print(f"Status:    {report.status}")
        print(f"Mode:      {report.mode}")
        print(f"Selected:  {len(report.selected_component_ids)}")
        for result in report.results:
            print(
                f"{result.component_id}: {result.status} "
                f"maturity={result.initial_maturity}->{result.final_maturity} "
                f"identity={result.identity.identity_hash[:12]} "
                f"reason={result.selection_reason}"
            )
            if result.cpu_report:
                print(f"  cpu_report={result.cpu_report}")
            if result.gpu_report:
                print(f"  gpu_report={result.gpu_report}")
            if result.matched_pilot_fixture:
                print(f"  matched_fixture={result.matched_pilot_fixture}")
            if result.errors:
                print(f"  error={result.errors[0]}")
        if report.discovery_errors:
            print(f"Discovery: {len(report.discovery_errors)} error(s)")
        if report.coverage_report_path:
            print(f"Coverage:  {report.coverage_report_path}")
        if report.coverage_error:
            print(f"Coverage:  failed: {report.coverage_error}")
        print("Ceiling:   gpu_certified; matched fixture is not pilot evidence")
        print(
            "Report:    "
            f"{certify_args.workdir / 'paper_adapter_certification.yaml'}"
        )
        return 0 if report.status == "passed" else 1
    if advanced_args[0] == "certify-paper-components":
        parser = argparse.ArgumentParser(
            prog="yolo-agent advanced certify-paper-components"
        )
        parser.add_argument(
            "--workdir",
            type=Path,
            default=Path("runs/certification/paper-components"),
        )
        parser.add_argument(
            "--registry",
            type=Path,
            default=Path("runs/component_maturity_registry.yaml"),
        )
        parser.add_argument("--model", default="yolo26n.pt")
        parser.add_argument("--teacher")
        parser.add_argument(
            "--ensemble-teacher",
            help="Second local teacher required by teacher-ensemble distillation.",
        )
        parser.add_argument("--device", default="0")
        parser.add_argument("--execute-real-gpu", action="store_true")
        certify_args = parser.parse_args(advanced_args[1:])
        report = PaperComponentGPUSuiteRunner().run(
            workdir=certify_args.workdir,
            registry_path=certify_args.registry,
            model=certify_args.model,
            teacher=certify_args.teacher,
            ensemble_teacher=certify_args.ensemble_teacher,
            device=certify_args.device,
            execute_real_gpu=certify_args.execute_real_gpu,
        )
        print("YOLO Agent Paper Component GPU Suite")
        print("------------------------------------")
        print(f"Status:    {report.status}")
        print(f"Model:     {report.model}")
        print(f"Device:    {report.device}")
        for result in report.results:
            line = (
                f"{result.priority}. {result.component_id}: {result.status}"
                f" maturity={result.final_maturity or '-'}"
            )
            if result.reason:
                line += f" reason={result.reason}"
            print(line)
        print("Ceiling:   gpu_certified; pilot_reproduced is not granted")
        print(f"Report:    {certify_args.workdir / 'paper_component_gpu_suite.yaml'}")
        return 0 if report.status == "passed" else 1
    if advanced_args[0] == "certify-component":
        parser = argparse.ArgumentParser(
            prog="yolo-agent advanced certify-component"
        )
        parser.add_argument("--component", required=True)
        mode = parser.add_mutually_exclusive_group(required=True)
        mode.add_argument("--cpu", action="store_true")
        mode.add_argument("--gpu", action="store_true")
        parser.add_argument("--workdir", type=Path)
        parser.add_argument(
            "--registry",
            type=Path,
            default=Path("runs/component_maturity_registry.yaml"),
        )
        parser.add_argument("--model", default="yolo26n.pt")
        parser.add_argument(
            "--teacher",
            help="Local yolo26s.pt or yolo26m.pt for distillation certification.",
        )
        parser.add_argument("--data", default="coco.yaml")
        parser.add_argument("--device", default="0")
        parser.add_argument("--protocol-hash")
        certify_args = parser.parse_args(advanced_args[1:])
        certification_mode = "gpu" if certify_args.gpu else "cpu"
        workdir = certify_args.workdir or (
            Path("runs/certification/components")
            / _component_certification_directory(certify_args.component)
        )
        try:
            report = ComponentCertificationRunner().run(
                component_id=certify_args.component,
                mode=certification_mode,
                workdir=workdir,
                registry_path=certify_args.registry,
                model=certify_args.model,
                data=certify_args.data,
                device=certify_args.device,
                protocol_hash=certify_args.protocol_hash,
                options=(
                    {"teacher": certify_args.teacher}
                    if certify_args.teacher
                    else None
                ),
                execute_gpu=certify_args.gpu,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print("YOLO Agent Component Certification")
            print("----------------------------------")
            print(f"Component: {certify_args.component}")
            print(f"Mode:      {certification_mode}")
            print("Status:    failed")
            print(f"Reason:    {exc}")
            print(f"Workdir:   {workdir}")
            return 1
        _print_component_certification_report(report)
        return 0 if report.status == "passed" else 1
    if advanced_args[0] == "certify-inference-policy":
        parser = argparse.ArgumentParser(
            prog="yolo-agent advanced certify-inference-policy"
        )
        parser.add_argument(
            "--workdir",
            type=Path,
            default=Path("runs/certification/inference-policy"),
        )
        parser.add_argument("--model", required=True)
        parser.add_argument("--images", type=Path, required=True)
        parser.add_argument("--annotations", type=Path, required=True)
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("--standard-metrics", type=Path)
        parser.add_argument("--execute", action="store_true")
        certify_args = parser.parse_args(advanced_args[1:])
        try:
            raw_config = read_yaml(certify_args.config)
            if isinstance(raw_config.get("inference_policy"), dict):
                raw_config = raw_config["inference_policy"]
            config = InferencePolicyConfig.model_validate(raw_config)
            report = InferencePolicyCertificationRunner().run(
                workdir=certify_args.workdir,
                model=certify_args.model,
                images=certify_args.images,
                annotations=certify_args.annotations,
                config=config,
                standard_metrics=certify_args.standard_metrics,
                execute=certify_args.execute,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print("YOLO Agent Inference Policy Certification")
            print("-----------------------------------------")
            print("Status:    failed")
            print("Training:  unchanged; attribution disabled")
            print(f"Reason:    {exc}")
            print(f"Report:    {certify_args.workdir / 'inference_policy_certification_report.yaml'}")
            return 1
        print("YOLO Agent Inference Policy Certification")
        print("-----------------------------------------")
        print(f"Status:    {report.status}")
        print(f"Policy:    {report.protocol.config.policy_id} ({report.protocol.config.kind})")
        print(f"Namespace: {report.protocol.metric_namespace}")
        print("Training:  unchanged; attribution disabled")
        print(f"Protocol:  {report.protocol_hash}")
        if report.standard_640_metrics:
            standard = " ".join(
                f"{name}={value:.6f}"
                for name, value in sorted(report.standard_640_metrics.items())
            )
            print(f"Standard:  {standard}")
        if report.policy_metrics is not None:
            metrics = report.policy_metrics
            print(
                "Policy AP: "
                f"mAP50-95={metrics.map50_95} AP_small={metrics.ap_small} "
                f"recall={metrics.recall}"
            )
            print(
                "Runtime:   "
                f"latency_ms={metrics.resources.latency_ms:.6f} "
                f"throughput={metrics.resources.throughput:.6f} "
                f"peak_vram_mb={metrics.resources.peak_vram_mb:.3f}"
            )
            print(f"Merge:     {report.protocol.config.merge_policy}")
        if report.reason:
            print(f"Reason:    {report.reason}")
        print(
            f"Report:    {certify_args.workdir / 'inference_policy_certification_report.yaml'}"
        )
        return 0 if report.status in {"passed", "skipped"} else 1
    if advanced_args[0] == "certify-sahi":
        parser = argparse.ArgumentParser(prog="yolo-agent advanced certify-sahi")
        parser.add_argument("--workdir", type=Path, default=Path("runs/certification/sahi"))
        parser.add_argument("--model", required=True)
        parser.add_argument("--images", type=Path, required=True)
        parser.add_argument("--annotations", type=Path, required=True)
        parser.add_argument("--device", default="cpu")
        parser.add_argument("--slice-height", type=int, default=640)
        parser.add_argument("--slice-width", type=int, default=640)
        parser.add_argument("--overlap-height", type=float, default=0.2)
        parser.add_argument("--overlap-width", type=float, default=0.2)
        parser.add_argument("--merge-policy", choices=["none", "nms", "nmm"], default="none")
        parser.add_argument("--confidence-threshold", type=float, default=0.001)
        parser.add_argument("--standard-metrics", type=Path)
        parser.add_argument("--execute", action="store_true")
        certify_args = parser.parse_args(advanced_args[1:])
        report = SahiInferenceCertificationRunner().run(
            workdir=certify_args.workdir,
            model=certify_args.model,
            images=certify_args.images,
            annotations=certify_args.annotations,
            config=SlicingInferenceConfig(
                device=certify_args.device,
                slice_height=certify_args.slice_height,
                slice_width=certify_args.slice_width,
                overlap_height_ratio=certify_args.overlap_height,
                overlap_width_ratio=certify_args.overlap_width,
                merge_policy=certify_args.merge_policy,
                confidence_threshold=certify_args.confidence_threshold,
            ),
            standard_metrics=certify_args.standard_metrics,
            execute=certify_args.execute,
        )
        print("YOLO Agent SAHI Inference Certification")
        print("---------------------------------------")
        print(f"Status:    {report.status}")
        print("Training:  unchanged; attribution disabled")
        print(f"Protocol:  {report.protocol_hash}")
        if report.sliced_inference_metrics is not None:
            metrics = report.sliced_inference_metrics
            print(f"Sliced AP: mAP50-95={metrics.sliced_map50_95} AP_small={metrics.sliced_ap_small}")
            print(f"Runtime:   latency_ms={metrics.sliced_latency_ms} throughput={metrics.sliced_throughput}")
        if report.reason:
            print(f"Reason:    {report.reason}")
        print(f"Report:    {certify_args.workdir / 'sahi_certification_report.yaml'}")
        return 0 if report.status in {"passed", "skipped"} else 1
    if advanced_args[0] in {*USER_COMMANDS, "advanced", "research"}:
        print(f"yolo-agent advanced: {advanced_args[0]} is not an advanced command")
        return 2
    return main(advanced_args)


def _component_certification_directory(component_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", component_id).strip(".-")
    return normalized or "unknown-component"


def _batch_certification_options(
    *, teacher: str | None, ensemble_teacher: str | None
) -> dict[str, dict[str, object]]:
    if not teacher:
        return {}
    component_ids = {
        "distillation.yolo26_teacher_student",
        *DISTILLATION_COMPONENTS,
    }
    output = {component_id: {"teacher": teacher} for component_id in component_ids}
    if ensemble_teacher:
        output.setdefault("distillation.teacher_ensemble", {})["teachers"] = [
            ensemble_teacher
        ]
    return output


def _print_component_certification_report(
    report: ComponentCertificationReport,
) -> None:
    print("YOLO Agent Component Certification")
    print("----------------------------------")
    print(f"Component: {report.component_id}")
    print(f"Mode:      {report.mode}")
    print(f"Status:    {report.status}")
    print(f"Maturity:  {report.initial_maturity} -> {report.final_maturity}")
    if report.mode == "gpu":
        print("Ceiling:   gpu_certified (pilot reproduction is a separate suite)")
        gpu_stage = next(
            (item for item in report.stages if item.stage_id == "isolated_gpu_smoke"),
            None,
        )
        if gpu_stage is not None:
            checks = gpu_stage.checks
            if checks.get("gpu_name"):
                print(f"GPU:       {checks['gpu_name']}")
            if checks.get("peak_vram_mb") is not None:
                print(f"VRAM:      peak={float(checks['peak_vram_mb']):.1f} MB")
            if checks.get("latency_ms") is not None:
                print(f"Latency:   {float(checks['latency_ms']):.3f} ms")
            if checks.get("model_size_mb") is not None:
                print(f"Size:      {float(checks['model_size_mb']):.3f} MB")
    if report.missing_artifacts:
        print(f"Missing:   {', '.join(report.missing_artifacts)}")
    else:
        print("Missing:   none")
    if report.generated_paths:
        print("Generated:")
        for name, path in sorted(report.generated_paths.items()):
            print(f"  {name}={path}")
    report_path = report.workdir / f"component_certification.{report.mode}.yaml"
    print(f"  report={report_path}")
    if report.next_maturity:
        print(f"Next:      {report.next_maturity}")
    else:
        print("Next:      maturity sequence complete")
    if report.errors:
        print(f"Reason:    {report.errors[0]}")
    print(f"Registry:  {report.registry_path}")


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
