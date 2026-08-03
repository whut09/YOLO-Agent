"""CLI smoke tests for the yolo-agent scaffold."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from yolo_agent.cli import COMMANDS, USER_COMMANDS, build_parser, main


def test_cli_import() -> None:
    """The CLI module should import and expose scaffold commands."""
    assert "init" in COMMANDS
    assert "report" in COMMANDS
    assert "optimize" in COMMANDS
    assert "doctor" in COMMANDS
    assert USER_COMMANDS == ("setup", "train", "status", "stop")


def test_cli_help_runs(capsys) -> None:  # type: ignore[no-untyped-def]
    """Running without a command should print help and succeed."""
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "Componentized YOLO optimization harness" in output
    assert "{setup,train,status,stop}" in output
    assert "doctor" not in output
    assert "loop" not in output
    assert "optimize" not in output


def test_scaffold_commands_run(capsys) -> None:  # type: ignore[no-untyped-def]
    """Every declared command should execute the current scaffold."""
    for command in COMMANDS:
        if command in {
            "plan",
            "smoke",
            "profile-data",
            "advise-labels",
            "ablate-plan",
            "report",
            "loop",
            "optimize",
            "doctor",
        }:
            continue
        assert main([command]) == 0
        output = capsys.readouterr().out
        assert f"yolo-agent {command}: scaffold ready" in output


def test_train_command_runs_dry_run(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    """The beginner-facing train command should prepare a run without exposing optimize."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    data_yaml = dataset / "coco.yaml"
    data_yaml.write_text(
        "path: .\ntrain: images/train2017\nval: images/val2017\nnames: {0: person}\n",
        encoding="utf-8",
    )

    assert main(
        [
            "train",
            "--data",
            str(data_yaml),
            "--run-id",
            "cli-train",
            "--run-root",
            str(tmp_path / "runs"),
            "--dry-run",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "Starting YOLO Agent train" in output
    assert "Run: cli-train  Profile: debug  Mode: dry-run" in output
    assert "Budget: auto; stops when the first cost, evidence, or patience limit is reached" in output
    assert "Expected: 1-12 pilot experiments" in output
    assert "Limits: <= 24 GPU hours; concurrency=1" in output
    assert "Full: excluded from the automatic budget unless --confirm-full-run is explicit" in output
    assert f"Status:   yolo-agent status --run {tmp_path / 'runs' / 'cli-train'}" in output


def test_train_defaults_to_bounded_auto_optimization() -> None:
    """One-command train should select auto budget instead of promising a fixed round count."""
    args = build_parser().parse_args(["train", "--data", "data.yaml"])
    assert args.auto_rounds is None
    assert args.profile is None
    assert args.goal is None
    assert args.target_metric is None
    assert args.target_delta is None
    assert args.goal_description is None

    explicit = build_parser().parse_args([
        "train",
        "--data",
        "data.yaml",
        "--target-metric",
        "ap_small",
        "--target-delta",
        "0.02",
        "--goal-description",
        "Reduce false negatives",
    ])
    assert explicit.target_metric == "ap_small"
    assert explicit.target_delta == 0.02
    assert explicit.goal_description == "Reduce false negatives"


def test_train_help_renders_percent_goal_example(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["train", "--help"])

    assert exc_info.value.code == 0
    assert "+2%map" in capsys.readouterr().out


def test_build_snapshot_defaults_to_machine_maturity_registry() -> None:
    args = build_parser().parse_args(["research", "build-snapshot"])

    assert args.maturity_registry == Path("runs/component_maturity_registry.yaml")
    assert args.cached_code_root is None


def test_research_coverage_baseline_cli_defaults_to_frozen_snapshot() -> None:
    args = build_parser().parse_args(["research", "coverage-baseline"])

    assert args.root == Path("research")
    assert args.snapshot is None
    assert args.output == Path("runs/coverage_baseline.yaml")
    assert args.markdown is None


def test_real_train_requires_current_snapshot_before_run_allocation(
    tmp_path: Path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    data_yaml = tmp_path / "coco.yaml"
    data_yaml.write_text("names: {0: person}\n", encoding="utf-8")
    run_root = tmp_path / "runs"

    code = main([
        "train",
        "--data",
        str(data_yaml),
        "--run-root",
        str(run_root),
        "--run-id",
        "snapshot-preflight",
    ])

    output = capsys.readouterr().out
    assert code == 2
    assert "research snapshot preflight failed" in output
    assert "Status: missing" in output
    assert (
        "yolo-agent research build-snapshot "
        f"--root {tmp_path / 'research'} --source awesome_object_detection "
        f"--maturity-registry {run_root / 'component_maturity_registry.yaml'}"
    ) in output
    assert "Traceback" not in output
    assert not run_root.exists()


def test_train_rejects_natural_language_goal_without_traceback_or_run_dir(
    tmp_path: Path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    data_yaml = tmp_path / "coco.yaml"
    data_yaml.write_text("names: {0: person}\n", encoding="utf-8")
    run_root = tmp_path / "runs"

    code = main([
        "train",
        "--data",
        str(data_yaml),
        "--run-root",
        str(run_root),
        "--run-id",
        "natural-goal",
        "--goal",
        "Improve AP_small and reduce false negatives",
    ])

    output = capsys.readouterr().out
    assert code == 2
    assert "objective error:" in output
    assert "--target-metric ap_small --target-delta 0.02" in output
    assert "--goal-description 'Improve AP_small and reduce false negatives'" in output
    assert "Traceback" not in output
    assert not run_root.exists()


def test_train_accepts_explicit_ap_small_target_and_description(
    tmp_path: Path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    data_yaml = dataset / "coco.yaml"
    data_yaml.write_text(
        "path: .\ntrain: images/train2017\nval: images/val2017\nnames: {0: person}\n",
        encoding="utf-8",
    )
    run_root = tmp_path / "runs"

    code = main([
        "train",
        "--data",
        str(data_yaml),
        "--run-root",
        str(run_root),
        "--run-id",
        "explicit-goal",
        "--target-metric",
        "ap_small",
        "--target-delta",
        "0.02",
        "--goal-description",
        "Reduce small-object false negatives",
        "--dry-run",
    ])

    capsys.readouterr()
    assert code == 0
    objective = yaml.safe_load(
        (run_root / "explicit-goal" / "artifacts" / "optimization_objective.yaml").read_text(
            encoding="utf-8-sig"
        )
    )
    assert objective["primary_metric"] == "ap_small"
    assert objective["target_absolute_delta"] == 0.02
    assert objective["goal_description"] == "Reduce small-object false negatives"


def test_train_allocates_incremented_run_only_after_valid_objective(
    tmp_path: Path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    data_yaml = dataset / "coco.yaml"
    data_yaml.write_text(
        "path: .\ntrain: images/train2017\nval: images/val2017\nnames: {0: person}\n",
        encoding="utf-8",
    )
    run_root = tmp_path / "runs"
    (run_root / "numbered").mkdir(parents=True)

    code = main([
        "train",
        "--data",
        str(data_yaml),
        "--run-root",
        str(run_root),
        "--run-id",
        "numbered",
        "--goal",
        "+2map",
        "--dry-run",
    ])

    output = capsys.readouterr().out
    assert code == 0
    assert "Allocated run: numbered-1" in output
    assert "Migration: preserved incomplete requested run" in output
    migration = run_root / "numbered" / "artifacts" / "run_initialization_migration.yaml"
    assert migration.is_file()
    assert yaml.safe_load(migration.read_text(encoding="utf-8-sig"))["allocated_run_id"] == (
        "numbered-1"
    )
    assert (run_root / "numbered-1" / "run_context.yaml").is_file()


def test_advanced_namespace_dispatches_hidden_compatibility_commands(capsys) -> None:  # type: ignore[no-untyped-def]
    """Advanced commands should remain available without appearing in beginner help."""
    assert main(["advanced"]) == 0
    output = capsys.readouterr().out
    assert "choose doctor, loop, optimize" in output
    args = build_parser().parse_args(["advanced", "doctor", "--data", "data.yaml"])
    assert args.advanced_args == ["doctor", "--data", "data.yaml"]


def test_advanced_gpu_certification_is_safe_by_default(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    workdir = tmp_path / "certification"
    assert main(["advanced", "certify-gpu", "--workdir", str(workdir)]) == 0
    output = capsys.readouterr().out
    assert "Status:   skipped" in output
    assert "real_gpu_execution_not_confirmed" in output
    assert (workdir / "certification_report.yaml").is_file()


def test_advanced_paper_auto_certification_is_safe_by_default(
    tmp_path: Path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    workdir = tmp_path / "paper-auto"

    result = main(
        ["advanced", "certify-paper-auto", "--workdir", str(workdir)]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "YOLO Agent Paper Auto-Optimization Acceptance" in output
    assert "Status:    skipped" in output
    assert "Component: sampling.small_object" in output
    assert "Scalar HPO: disabled" in output
    assert (workdir / "paper_auto_optimization_report.yaml").is_file()


def test_sampling_gpu_certification_prints_golden_path_outcome(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    report = SimpleNamespace(
        status="passed",
        stages=[
            SimpleNamespace(
                stage_id="component_runtime_certification",
                status="passed",
                metrics={},
            ),
            SimpleNamespace(
                stage_id="runtime_adapter",
                status="passed",
                metrics={
                    "train_dataloader_hook_called": True,
                    "manifest_payload_matched": True,
                },
            ),
        ],
        objective=SimpleNamespace(
            primary_metric="ap_small",
            observed_delta=0.012,
            passed=True,
            target_error_fact_deltas={"false_negative/object": 2.0},
        ),
        asha_survivor="small_object_sampling",
        failures=[],
    )

    class FakeSuite:
        def run(self, **kwargs):  # type: ignore[no-untyped-def]
            return report

    monkeypatch.setattr("yolo_agent.cli.RealGpuAcceptanceSuite", FakeSuite)

    result = main(
        [
            "advanced",
            "certify-gpu",
            "--workdir",
            str(tmp_path),
            "--recipe",
            "small_object_sampling",
            "--execute-real-gpu",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "Component: passed" in output
    assert "Runtime:  hook_called=True manifest_matched=True" in output
    assert "Objective: ap_small delta=0.012 passed=True" in output
    assert "Error:    false_negative/object=+2.000000" in output
    assert "ASHA:     survivor=small_object_sampling" in output


def test_advanced_sahi_certification_is_safe_by_default(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    workdir = tmp_path / "sahi-certification"
    images = tmp_path / "images"
    annotations = tmp_path / "annotations.json"
    assert main([
        "advanced", "certify-sahi",
        "--workdir", str(workdir),
        "--model", "yolo26n.pt",
        "--images", str(images),
        "--annotations", str(annotations),
    ]) == 0
    output = capsys.readouterr().out
    assert "Status:    skipped" in output
    assert "Training:  unchanged; attribution disabled" in output
    assert (workdir / "sahi_certification_report.yaml").is_file()


def test_advanced_inference_policy_is_safe_and_explicit(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    workdir = tmp_path / "inference-policy"
    config = tmp_path / "policy.yaml"
    config.write_text(
        "policy_id: tta-fixture\n"
        "kind: test_time_augmentation\n"
        "scales: [1.0, 1.2]\n",
        encoding="utf-8",
    )

    result = main(
        [
            "advanced",
            "certify-inference-policy",
            "--workdir",
            str(workdir),
            "--model",
            "yolo26n.pt",
            "--images",
            str(tmp_path / "images"),
            "--annotations",
            str(tmp_path / "annotations.json"),
            "--config",
            str(config),
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "Status:    skipped" in output
    assert "Policy:    tta-fixture (test_time_augmentation)" in output
    assert "Namespace: tta_inference" in output
    assert "Training:  unchanged; attribution disabled" in output
    assert (workdir / "inference_policy_certification_report.yaml").is_file()


def test_setup_supports_coco_and_custom_without_new_top_level_commands() -> None:
    parser = build_parser()
    coco = parser.parse_args(["setup", "coco", "--data", "coco.yaml"])
    custom = parser.parse_args(["setup", "custom", "--data", "custom.yaml"])

    assert coco.setup_kind == "coco"
    assert custom.setup_kind == "custom"
    assert USER_COMMANDS == ("setup", "train", "status", "stop")
