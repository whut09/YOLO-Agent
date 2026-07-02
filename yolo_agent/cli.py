"""Command line interface for yolo-agent."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from yolo_agent.agents.ablation_planner import create_ablation_plan
from yolo_agent.agents.annotation_advisor import advise_annotations
from yolo_agent.agents.candidate_generator import default_search_space_path, generate_plan
from yolo_agent.agents.orchestrator import LoopOrchestrator
from yolo_agent.core.loop_state import LoopStage
from yolo_agent.core.schemas import AgentConfig
from yolo_agent.core.task_spec import TaskSpec
from yolo_agent.reports.experiment_report import generate_experiment_report
from yolo_agent.tools.dataset_stats import profile_dataset
from yolo_agent.tools.smoke_runner import SmokeRunner, default_ultralytics_template_path


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
        default=default_search_space_path(),
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
        default=default_ultralytics_template_path(),
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
    loop_init.add_argument("--components", type=Path, default=Path("configs/components"), help="Component registry path.")
    loop_init.add_argument("--search-space", type=Path, default=default_search_space_path(), help="Search-space YAML path.")
    loop_init.add_argument("--loop-policy", type=Path, default=Path("configs/loop_policy.yaml"), help="Loop policy YAML path.")
    loop_init.add_argument("--predictions", type=Path, help="Optional prediction YAML/JSON for label advice.")
    loop_init.add_argument("--errors", type=Path, help="Optional detection error YAML/JSON.")
    loop_init.add_argument("--metrics", type=Path, help="Optional metrics YAML/JSON to import.")
    loop_init.add_argument("--dataset-version", default="unversioned", help="Dataset version label.")
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

    loop_next = loop_subparsers.add_parser(
        "next",
        help="Generate report and next-round checklist.",
    )
    loop_next.add_argument("--run", type=Path, required=True, help="Path to runs/{run_id}.")
    loop_next.set_defaults(handler=run_loop_next_command)

    loop_auto = loop_subparsers.add_parser(
        "auto",
        help="Initialize or run pending loop stages until blocked, failed, or complete.",
    )
    loop_auto.add_argument("--run", type=Path, help="Path to runs/{run_id}.")
    loop_auto.add_argument("--run-id", default="auto", help="Run id when initializing.")
    loop_auto.add_argument("--task", type=Path, help="Path to task.yaml when initializing.")
    loop_auto.add_argument("--data", type=Path, help="Path to YOLO data.yaml when initializing.")
    loop_auto.add_argument("--run-root", type=Path, default=Path("runs"), help="Run root directory.")
    loop_auto.add_argument("--components", type=Path, default=Path("configs/components"), help="Component registry path.")
    loop_auto.add_argument("--search-space", type=Path, default=default_search_space_path(), help="Search-space YAML path.")
    loop_auto.add_argument("--loop-policy", type=Path, default=Path("configs/loop_policy.yaml"), help="Loop policy YAML path.")
    loop_auto.add_argument("--predictions", type=Path, help="Optional prediction YAML/JSON for label advice.")
    loop_auto.add_argument("--errors", type=Path, help="Optional detection error YAML/JSON.")
    loop_auto.add_argument("--metrics", type=Path, help="Optional metrics YAML/JSON/CSV.")
    loop_auto.add_argument("--dataset-version", default="unversioned", help="Dataset version label.")
    loop_auto.set_defaults(handler=run_loop_auto_command)

    for command in COMMANDS:
        if command in {"init", "plan", "smoke", "profile-data", "advise-labels", "ablate-plan", "report", "loop"}:
            continue
        command_parser = subparsers.add_parser(
            command,
            help=f"Run the {command} workflow scaffold.",
        )
        command_parser.set_defaults(handler=run_scaffold_command)

    return parser


def scenarios_dir() -> Path:
    """Return the bundled scenario template directory."""
    return Path(__file__).resolve().parents[1] / "configs" / "scenarios"


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
        dataset_version=args.dataset_version,
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


def run_loop_smoke_command(args: argparse.Namespace) -> int:
    """Run loop smoke stage."""
    return _print_loop_results([LoopOrchestrator.from_run_dir(args.run).smoke()])


def run_loop_ingest_metrics_command(args: argparse.Namespace) -> int:
    """Import loop metrics."""
    return _print_loop_results([LoopOrchestrator.from_run_dir(args.run).ingest_metrics(args.metrics)])


def run_loop_next_command(args: argparse.Namespace) -> int:
    """Run loop report and next-round stages."""
    return _print_loop_results(LoopOrchestrator.from_run_dir(args.run).next_round())


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
            dataset_version=args.dataset_version,
        )
        print(f"created {orchestrator.context.run_dir}")
    return _print_loop_results(orchestrator.run_until_blocked())


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
