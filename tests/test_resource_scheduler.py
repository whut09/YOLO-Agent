"""GPU-aware resource scheduler tests."""

from __future__ import annotations

from datetime import datetime

from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.core.command_spec import CommandSpec, ResourceRequirements
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.execution_queue import ExecutionQueue
from yolo_agent.core.experiment_graph import ExperimentNode, ExperimentPlan
from yolo_agent.core.resource_scheduler import (
    GPUResource,
    ResourceScheduler,
    ResourceSchedulerConfig,
    ResourceSnapshot,
)


def test_scheduler_blocks_when_gpu_is_unavailable() -> None:
    """GPU-required commands should not run when no GPU is visible."""
    command = _train_command()

    decision = ResourceScheduler(snapshot=ResourceSnapshot(gpus=[])).evaluate(command)

    assert decision.status == "blocked_by_resource"
    assert decision.reasons == ["gpu_unavailable"]


def test_scheduler_selects_idle_gpu_with_sufficient_vram() -> None:
    """An idle GPU with enough free VRAM should make the item runnable."""
    command = _train_command(min_free_vram_mb=8000)
    snapshot = ResourceSnapshot(
        gpus=[
            GPUResource(gpu_id=0, util_percent=90, memory_used_mb=1000, memory_total_mb=24000),
            GPUResource(gpu_id=1, util_percent=5, memory_used_mb=2000, memory_total_mb=24000),
        ]
    )

    decision = ResourceScheduler(snapshot=snapshot).evaluate(command)

    assert decision.status == "runnable"
    assert decision.selected_gpu_id == 1


def test_scheduler_blocks_when_vram_is_insufficient() -> None:
    """GPU selection should respect minimum free VRAM."""
    command = _train_command(min_free_vram_mb=16000)
    snapshot = ResourceSnapshot(
        gpus=[GPUResource(gpu_id=0, util_percent=5, memory_used_mb=12000, memory_total_mb=24000)]
    )

    decision = ResourceScheduler(snapshot=snapshot).evaluate(command)

    assert decision.status == "blocked_by_resource"
    assert decision.reasons == ["insufficient_vram:0:12000<16000"]


def test_scheduler_requires_batch_tuning_result_when_requested(tmp_path: Path) -> None:
    """Batch auto full runs should wait for batch tuner evidence."""
    command = _train_command(requires_batch_tuning=True)
    snapshot = ResourceSnapshot(
        gpus=[GPUResource(gpu_id=0, util_percent=5, memory_used_mb=1000, memory_total_mb=24000)]
    )
    scheduler = ResourceScheduler(snapshot=snapshot)
    store = EvidenceStore(tmp_path / "runs")
    store.create_run("exp")

    missing = scheduler.evaluate(command, evidence=store.load_run("exp"))

    assert missing.status == "blocked_by_resource"
    assert missing.reasons == ["missing_batch_tuning_result"]

    store.log_candidate_metrics(
        "exp",
        "candidate",
        "node_candidate",
        {"batch_tuning_selected_batch": 48},
        split="runtime",
        source="batch_tuner",
    )
    ready = scheduler.evaluate(command, evidence=store.load_run("exp"))

    assert ready.status == "runnable"


def test_scheduler_accepts_candidate_level_batch_tuning_from_prior_node(tmp_path: Path) -> None:
    """A full node may reuse prior pilot/debug batch tuning for the same candidate."""
    command = _train_command(requires_batch_tuning=True)
    snapshot = ResourceSnapshot(
        gpus=[GPUResource(gpu_id=0, util_percent=5, memory_used_mb=1000, memory_total_mb=24000)]
    )
    store = EvidenceStore(tmp_path / "runs")
    store.log_candidate_metrics(
        "exp",
        "candidate",
        "node_candidate_pilot",
        {"batch_tuning_selected_batch": 48},
        split="runtime",
        source="batch_tuner",
    )

    decision = ResourceScheduler(snapshot=snapshot).evaluate(command, evidence=store.load_run("exp"))

    assert decision.status == "runnable"


def test_scheduler_pauses_high_risk_and_full_run_outside_budget_window() -> None:
    """High-risk and full-budget policies should pause before resource execution."""
    command = _train_command(
        requirements=ResourceRequirements(
            requires_gpu=True,
            high_risk=True,
            full_run=True,
            allowed_start_hours=[23],
        )
    )

    decision = ResourceScheduler(
        snapshot=ResourceSnapshot(gpus=[GPUResource(gpu_id=0, util_percent=0, memory_used_mb=0, memory_total_mb=24000)]),
        now=datetime(2026, 7, 5, 14, 0, 0),
    ).evaluate(command)

    assert decision.status == "paused"
    assert decision.reasons == ["high_risk_candidate_deferred", "outside_full_run_budget_window:14"]


def test_scheduler_marks_retried_training_without_checkpoint_as_needs_resume() -> None:
    """A retried train command should require a resume checkpoint when resume is expected."""
    command = _train_command()

    decision = ResourceScheduler().evaluate(command, attempts=1)

    assert decision.status == "needs_resume"
    assert decision.reasons == ["missing_resume_checkpoint_after_attempt"]


def test_execution_queue_refreshes_resource_blocked_items() -> None:
    """Resource-blocked queue items should become queued when the scheduler later allows them."""
    node = _node()
    queue = ExecutionQueue.from_experiment_plan("run-1", ExperimentPlan(plan_id="plan", nodes=[node]))
    blocked = ResourceScheduler(snapshot=ResourceSnapshot(gpus=[])).evaluate(queue.items[0].command)
    queue.items[0].mark_resource_decision(blocked)

    ready = ResourceScheduler(
        snapshot=ResourceSnapshot(gpus=[GPUResource(gpu_id=0, util_percent=0, memory_used_mb=0, memory_total_mb=24000)])
    ).evaluate(queue.items[0].command)
    summary = queue.refresh_resources({queue.items[0].queue_id: ready})

    assert summary == {"refreshed": 1, "unblocked": 1, "still_blocked": 0}
    assert queue.items[0].status == "queued"


def _train_command(
    min_free_vram_mb: int | None = None,
    requires_batch_tuning: bool = False,
    requirements: ResourceRequirements | None = None,
) -> CommandSpec:
    return CommandSpec(
        command_type="train",
        command="yolo",
        argv=["yolo", "detect", "train"],
        resource_requirements=requirements or ResourceRequirements(
            requires_gpu=True,
            min_free_vram_mb=min_free_vram_mb,
            requires_batch_tuning=requires_batch_tuning,
        ),
        metadata={"candidate_id": "candidate", "node_id": "node_candidate"},
    )


def _node() -> ExperimentNode:
    command = _train_command()
    return ExperimentNode(
        node_id="node_candidate",
        candidate_config=CandidateConfig(
            candidate_id="candidate",
            base_model="yolo26n.pt",
            scale="n",
            framework="ultralytics",
        ),
        data_version="coco2017",
        command=command.display(),
        command_spec=command,
    )
