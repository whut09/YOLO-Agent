"""Fast tests for the prepared paper-cohort execution authority."""

from __future__ import annotations

import pytest

import yolo_agent.cli as cli
from yolo_agent.cli import build_parser
from yolo_agent.research.paper_training_plan_schemas import (
    PaperTrainingPlan,
    PaperTrainingPlanRecord,
)


def _record() -> PaperTrainingPlanRecord:
    return PaperTrainingPlanRecord(
        paper_ids=["arxiv:paper-001"],
        profile_ids=["profile-001"],
        mechanism_ids=["loss.quality.correlation"],
        recipe_id="quality-correlation",
        recipe_version="v1",
        execution_fingerprint="a" * 64,
        candidate_node_id="node-candidate",
        baseline_node_id="node-baseline",
        asha_trial_id="trial-001",
        protocol_hash="b" * 64,
        dataset_manifest_hash="c" * 64,
        runtime_payload_path="/tmp/payload.yaml",
        runtime_payload_hash="d" * 64,
        candidate_command=["candidate"],
        baseline_command=["baseline"],
    )


def test_paper_training_plan_requires_one_control_and_trial_per_fingerprint() -> None:
    plan = PaperTrainingPlan(
        run_id="paper-cohort",
        run_dir="/tmp/paper-cohort",
        inventory_path="/tmp/inventory.yaml",
        requirements_path="/tmp/requirements.yaml",
        assets_path="/tmp/assets.yaml",
        readiness_path="/tmp/readiness.yaml",
        asha_path="/tmp/asha.yaml",
        round_plan_path="/tmp/round.yaml",
        queue_path="/tmp/queue.yaml",
        run_protocol_hash="e" * 64,
        total_papers=1,
        trainable_fingerprints=1,
        matched_controls_planned=1,
        asha_trials_registered=1,
        records=[_record()],
        training_allowed=True,
    ).with_hash()

    assert plan.plan_hash == plan.calculate_hash()
    assert plan.training_allowed is True


def test_paper_training_plan_cannot_be_marked_as_real_execution() -> None:
    with pytest.raises(ValueError, match="dry-run artifact"):
        PaperTrainingPlan(
            run_id="paper-cohort",
            run_dir="/tmp/paper-cohort",
            inventory_path="/tmp/inventory.yaml",
            requirements_path="/tmp/requirements.yaml",
            assets_path="/tmp/assets.yaml",
            readiness_path="/tmp/readiness.yaml",
            asha_path="/tmp/asha.yaml",
            round_plan_path="/tmp/round.yaml",
            queue_path="/tmp/queue.yaml",
            run_protocol_hash="e" * 64,
            total_papers=1,
            trainable_fingerprints=1,
            matched_controls_planned=1,
            asha_trials_registered=1,
            records=[_record()],
            dry_run_only=False,
            training_allowed=True,
        )


def test_paper_training_plan_cli_is_registered() -> None:
    args = build_parser().parse_args(
        ["research", "paper-training-plan", "--run-id", "paper-cohort"]
    )

    assert args.run_id == "paper-cohort"
    assert args.handler.__name__ == "run_research_paper_training_plan_command"


def test_train_reuses_prepared_paper_cohort(monkeypatch: pytest.MonkeyPatch) -> None:
    args = build_parser().parse_args(
        [
            "train",
            "--model",
            "yolo26n.pt",
            "--data",
            "coco.yaml",
            "--run-id",
            "paper-cohort",
        ]
    )
    captured: list[object] = []

    monkeypatch.setattr(cli, "_paper_training_cohort_marked", lambda *_: True)
    monkeypatch.setattr(
        cli,
        "run_optimize_command",
        lambda value: captured.append(value) or 0,
    )

    assert cli.run_train_command(args) == 0
    assert captured == [args]
    assert args.profile == "pilot"
    assert args.no_auto_advance is True
    assert args.run_id == "paper-cohort"
