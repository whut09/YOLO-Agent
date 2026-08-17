"""Execution infrastructure failure classification and recovery tests."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.orchestrator import (
    LoopOrchestrator,
    _recover_failed_resource_item,
    _resource_recovery_message,
)
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.execution_failure import (
    apply_cached_resource_policy,
    apply_execution_recovery,
    classify_execution_failure,
    resolve_external_gpu_wait,
    save_successful_resource_policy,
)
from yolo_agent.core.execution_queue import ExecutionQueueItem
from yolo_agent.core.executor import ExecutionResult
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.experiment_graph import ExperimentPlan
from yolo_agent.core.gpu_runtime import GPURuntimeSnapshot, GPUProcessInfo


HOST_OOM_OUTPUT = """
SystemError: Caught SystemError in DataLoader worker process 0.
Original numpy._core._exceptions._ArrayMemoryError: Unable to allocate 1.17 MiB
for an array with shape (640, 640, 3) and data type uint8
SystemError: <built-in function warpAffine> returned a result with an exception set
"""

GPU_OOM_OUTPUT = """
SystemError: Caught SystemError in DataLoader worker process 0.
Unable to allocate an intermediate buffer.
torch.AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocation` in the CUDA runtime documentation.
"""

ADAPTER_RECTANGULAR_VALIDATION_OUTPUT = """
Traceback (most recent call last):
  File "ultralytics\\engine\\trainer.py", line 810, in validate
    metrics = self.validator(self)
ValueError: feature 0 has stride (14, 8); expected 8
"""


def _command(*, batch: int = 48, workers: int = 8) -> CommandSpec:
    return CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data="coco.yaml",
        project="runs/ultralytics",
        name="pilot",
        batch=batch,
        workers=workers,
        overrides={"cache": "disk"},
    )


def _node(command: CommandSpec) -> ExperimentNode:
    return ExperimentNode(
        node_id="node_pilot",
        candidate_config=CandidateConfig(
            candidate_id="pilot",
            base_model="yolo26n.pt",
            scale="n",
            framework="ultralytics",
        ),
        data_version="coco2017",
        command=command.display(),
        command_spec=command,
    )


def test_classifies_host_memory_failure_from_stdout() -> None:
    failure = classify_execution_failure(stdout=HOST_OOM_OUTPUT, stderr="", command=_command())

    assert failure is not None
    assert failure.kind == "host_memory_exhausted"
    assert failure.recoverable is True
    assert failure.failed_settings == {"batch": 48, "workers": 8, "cache": "disk"}
    assert failure.recovery_overrides == {"workers": 2}


def test_second_host_memory_failure_reduces_workers_and_batch() -> None:
    first = classify_execution_failure(stdout=HOST_OOM_OUTPUT, stderr="", command=_command())
    assert first is not None
    first_retry = apply_execution_recovery(_command(), first)

    second = classify_execution_failure(stdout=HOST_OOM_OUTPUT, stderr="", command=first_retry)

    assert second is not None
    assert second.recovery_attempt == 1
    assert second.recovery_overrides == {"workers": 0, "batch": 24}
    second_retry = apply_execution_recovery(first_retry, second)
    assert "workers=0" in second_retry.argv
    assert "batch=24" in second_retry.argv


def test_cuda_oom_takes_priority_and_retries_known_batch_once() -> None:
    failure = classify_execution_failure(stdout=GPU_OOM_OUTPUT, stderr="", command=_command())

    assert failure is not None
    assert failure.kind == "gpu_memory_exhausted"
    assert failure.recovery_overrides == {"batch": 48}
    assert failure.recovery_strategy == "retry_same_batch_on_clean_gpu"
    retry = apply_execution_recovery(_command(), failure)
    assert "batch=48" in retry.argv
    assert "workers=8" in retry.argv
    assert retry.metadata["gpu_clean_retry_attempted"] is True


def test_classifies_adapter_traceback_from_runtime_entrypoint() -> None:
    command = _command().model_copy(
        update={
            "metadata": {
                **_command().metadata,
                "adapter_runtime_entrypoint": "yolo_agent.adapters.ultralytics.runtime_entrypoint",
            }
        }
    )

    failure = classify_execution_failure(
        stdout=ADAPTER_RECTANGULAR_VALIDATION_OUTPUT,
        stderr="",
        command=command,
    )

    assert failure is not None
    assert failure.kind == "adapter_runtime_failed"
    assert "validation" in failure.summary
    assert "feature 0 has stride (14, 8)" in failure.root_cause
    assert failure.recoverable is False


def test_classifies_concise_adapter_entrypoint_failure_without_traceback() -> None:
    command = _command().model_copy(
        update={
            "metadata": {
                **_command().metadata,
                "adapter_runtime_entrypoint": (
                    "yolo_agent.adapters.ultralytics.runtime_entrypoint"
                ),
            }
        }
    )

    failure = classify_execution_failure(
        stdout="",
        stderr=(
            "adapter_runtime_failed: PluginExecutionError: "
            "plugin hook failed: quality:compute_loss: invalid tensor"
        ),
        command=command,
    )

    assert failure is not None
    assert failure.kind == "adapter_runtime_failed"
    assert "adapter_runtime_failed" in failure.evidence_patterns
    assert failure.recoverable is False


def test_repeated_cuda_oom_on_clean_gpu_reduces_batch() -> None:
    first = classify_execution_failure(stdout=GPU_OOM_OUTPUT, stderr="", command=_command())
    assert first is not None
    same_batch_retry = apply_execution_recovery(_command(), first)

    second = classify_execution_failure(
        stdout=GPU_OOM_OUTPUT,
        stderr="",
        command=same_batch_retry,
    )

    assert second is not None
    assert second.recovery_overrides == {"batch": 24}
    assert second.recovery_strategy == "reduce_batch_after_clean_gpu_oom"


def test_matched_candidate_clean_gpu_oom_isolated_without_batch_retry() -> None:
    command = _command().model_copy(
        update={
            "metadata": {
                **_command().metadata,
                "matched_pilot_required": True,
                "matched_baseline_control": False,
            }
        }
    )

    failure = classify_execution_failure(
        stdout=GPU_OOM_OUTPUT,
        stderr="",
        command=command,
        gpu_snapshot=GPURuntimeSnapshot(
            used_memory_mb=1200,
            total_memory_mb=24564,
        ),
    )

    assert failure is not None
    assert failure.kind == "gpu_memory_exhausted"
    assert failure.recoverable is False
    assert failure.recovery_overrides == {}
    assert failure.recovery_strategy == "fail_candidate_after_clean_gpu_oom"


def test_matched_candidate_external_gpu_oom_remains_resumable() -> None:
    command = _command().model_copy(
        update={
            "metadata": {
                **_command().metadata,
                "matched_pilot_required": True,
                "matched_baseline_control": False,
            }
        }
    )
    busy = GPURuntimeSnapshot(
        used_memory_mb=9726,
        total_memory_mb=24564,
        processes=[GPUProcessInfo(pid=32452, process_name="python.exe")],
    )

    failure = classify_execution_failure(
        stdout=GPU_OOM_OUTPUT,
        stderr="",
        command=command,
        gpu_snapshot=busy,
    )

    assert failure is not None
    assert failure.recoverable is True
    assert failure.waiting_for_external_gpu is True
    assert failure.recovery_strategy == "wait_for_external_gpu_then_retry_same_batch"


def test_external_gpu_pressure_preserves_batch_until_process_clears() -> None:
    busy = GPURuntimeSnapshot(
        used_memory_mb=8325,
        total_memory_mb=24564,
        processes=[
            GPUProcessInfo(
                pid=15584,
                process_name="python.exe",
                command_line="python indextts-worker.py",
            )
        ],
    )
    failure = classify_execution_failure(
        stdout=GPU_OOM_OUTPUT,
        stderr="",
        command=_command(),
        gpu_snapshot=busy,
    )

    assert failure is not None
    assert failure.waiting_for_external_gpu is True
    assert failure.recovery_overrides == {}
    assert "PID 15584" in failure.root_cause
    assert resolve_external_gpu_wait(failure, busy) is None

    cleared = GPURuntimeSnapshot(used_memory_mb=1595, total_memory_mb=24564)
    resolved = resolve_external_gpu_wait(failure, cleared)
    assert resolved is not None
    assert resolved.recovery_overrides == {"batch": 48}
    assert resolved.waiting_for_external_gpu is False


def test_failed_queue_item_is_requeued_without_model_evidence() -> None:
    command = _command()
    item = ExecutionQueueItem.from_node("run-1", _node(command))
    item.mark_running()
    item.mark_result(
        ExecutionResult(
            run_id="run-1",
            node_id="node_pilot",
            candidate_id="pilot",
            status="failed",
            command=command,
            stdout=HOST_OOM_OUTPUT,
            message="Ultralytics training failed.",
        )
    )

    failure = _recover_failed_resource_item(item)

    assert failure is not None
    assert item.status == "queued"
    assert item.attempts == 1
    assert "workers=2" in item.command.argv
    assert item.command.metadata["resource_recovery_excluded_from_model_evidence"] is True


def test_external_gpu_queue_waits_then_retries_original_batch(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import yolo_agent.agents.orchestrator as orchestrator_mod

    command = _command()
    busy = GPURuntimeSnapshot(
        used_memory_mb=8325,
        total_memory_mb=24564,
        processes=[GPUProcessInfo(pid=15584, process_name="python.exe")],
    )
    failure = classify_execution_failure(
        stdout=GPU_OOM_OUTPUT,
        stderr="",
        command=command,
        gpu_snapshot=busy,
    )
    assert failure is not None
    item = ExecutionQueueItem.from_node("run-1", _node(command))
    item.mark_running()
    item.mark_result(
        ExecutionResult(
            run_id="run-1",
            node_id="node_pilot",
            candidate_id="pilot",
            status="failed",
            command=command,
            failure=failure,
        )
    )
    monkeypatch.setattr(orchestrator_mod, "inspect_gpu_runtime", lambda command: busy)
    monkeypatch.setattr(orchestrator_mod, "terminate_stale_run_processes", lambda snapshot: [])

    assert _recover_failed_resource_item(item) is None
    assert item.status == "needs_resume"
    assert item.resource_blockers == ["external_gpu_process"]

    cleared = GPURuntimeSnapshot(used_memory_mb=1595, total_memory_mb=24564)
    monkeypatch.setattr(orchestrator_mod, "inspect_gpu_runtime", lambda command: cleared)
    recovered = _recover_failed_resource_item(item)

    assert recovered is not None
    assert recovered.recovery_strategy == "retry_same_batch_after_external_gpu_cleared"
    assert item.status == "queued"
    assert "batch=48" in item.command.argv


def test_external_gpu_wait_is_not_rewritten_as_missing_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    import yolo_agent.agents.orchestrator as orchestrator_mod

    command = _command()
    busy = GPURuntimeSnapshot(
        used_memory_mb=8325,
        total_memory_mb=24564,
        processes=[GPUProcessInfo(pid=15584, process_name="python.exe")],
    )
    failure = classify_execution_failure(
        stdout=GPU_OOM_OUTPUT,
        stderr="",
        command=command,
        gpu_snapshot=busy,
    )
    assert failure is not None
    item = ExecutionQueueItem.from_node("run-1", _node(command))
    item.mark_running()
    item.mark_result(
        ExecutionResult(
            run_id="run-1",
            node_id=item.node_id,
            candidate_id=item.candidate_id,
            status="failed",
            command=command,
            failure=failure,
        )
    )
    context = orchestrator_mod.RunContext(
        run_id="run-1",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "data.yaml",
    )
    context.ensure_dirs()
    orchestrator = object.__new__(LoopOrchestrator)
    orchestrator.context = context
    orchestrator.evidence_store = orchestrator_mod.EvidenceStore(context.run_root)
    queue = orchestrator_mod.ExecutionQueue(run_id="run-1", items=[item])

    decisions = orchestrator._queue_resource_decisions(queue)

    assert decisions == {}
    assert item.status == "needs_resume"
    assert item.resource_blockers == ["external_gpu_process"]


def test_stale_plan_recovers_cleared_external_gpu_before_blocking(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    import yolo_agent.agents.orchestrator as orchestrator_mod

    command = _command()
    busy = GPURuntimeSnapshot(
        used_memory_mb=10153,
        total_memory_mb=24564,
        processes=[GPUProcessInfo(pid=50600, process_name="python.exe")],
    )
    failure = classify_execution_failure(
        stdout=GPU_OOM_OUTPUT,
        stderr="",
        command=command,
        gpu_snapshot=busy,
    )
    assert failure is not None
    item = ExecutionQueueItem.from_node("run-1", _node(command))
    item.mark_running()
    item.mark_result(
        ExecutionResult(
            run_id="run-1",
            node_id=item.node_id,
            candidate_id=item.candidate_id,
            status="failed",
            command=command,
            failure=failure,
        )
    )
    context = orchestrator_mod.RunContext(
        run_id="run-1",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "data.yaml",
    )
    context.ensure_dirs()
    plan = ExperimentPlan(plan_id="new-plan", nodes=[_node(command)])
    plan.to_yaml(context.artifact_path("experiment_plan.yaml"))
    queue = orchestrator_mod.ExecutionQueue(
        run_id="run-1",
        items=[item],
        metadata={"queue_source_plan_hash": "old-plan-hash"},
    )
    queue.to_yaml(context.run_dir / "execution_queue.yaml")
    orchestrator = object.__new__(LoopOrchestrator)
    orchestrator.context = context
    orchestrator.evidence_store = orchestrator_mod.EvidenceStore(context.run_root)
    orchestrator.event_log = orchestrator_mod.EventLog(context.run_root)
    cleared = GPURuntimeSnapshot(used_memory_mb=879, total_memory_mb=24564)
    monkeypatch.setattr(orchestrator_mod, "_stage_status", lambda *_args: "completed")
    monkeypatch.setattr(orchestrator_mod, "inspect_gpu_runtime", lambda command: cleared)
    monkeypatch.setattr(orchestrator_mod, "terminate_stale_run_processes", lambda snapshot: [])

    step = orchestrator._next_training_loop_step("debug", "ultralytics-train", True)
    recovered_queue = orchestrator_mod.ExecutionQueue.from_yaml(
        context.run_dir / "execution_queue.yaml"
    )

    assert step.action == "queue_resource_recovery"
    assert step.status == "completed"
    assert recovered_queue.items[0].status == "queued"
    assert "batch=48" in recovered_queue.items[0].command.argv


def test_resource_recovery_message_names_external_gpu_retry() -> None:
    message = _resource_recovery_message(
        [{"failure": {"kind": "gpu_memory_exhausted"}}]
    )

    assert "original batch" in message
    assert "comparison decision" in message


def test_resource_recovery_message_keeps_host_memory_wording() -> None:
    message = _resource_recovery_message(
        [{"failure": {"kind": "host_memory_exhausted"}}]
    )

    assert "host-memory DataLoader" in message


def test_needs_resume_resource_failure_is_requeued_from_original_result() -> None:
    command = _command()
    item = ExecutionQueueItem.from_node("run-1", _node(command))
    item.mark_running()
    item.mark_result(
        ExecutionResult(
            run_id="run-1",
            node_id="node_pilot",
            candidate_id="pilot",
            status="failed",
            command=command,
            stdout=GPU_OOM_OUTPUT,
            message="Ultralytics training failed.",
        )
    )
    item.status = "needs_resume"
    item.resource_blockers = ["missing_resume_checkpoint_after_attempt"]

    failure = _recover_failed_resource_item(item)

    assert failure is not None and failure.kind == "gpu_memory_exhausted"
    assert item.status == "queued"
    assert "batch=48" in item.command.argv


def test_gpu_retry_preserves_prior_host_memory_worker_recovery() -> None:
    original = _command(batch=48, workers=8)
    host_failure = classify_execution_failure(
        stdout=HOST_OOM_OUTPUT,
        stderr="",
        command=original,
    )
    assert host_failure is not None
    host_recovered = apply_execution_recovery(original, host_failure)
    item = ExecutionQueueItem.from_node("run-1", _node(host_recovered))
    item.mark_running()
    item.mark_result(
        ExecutionResult(
            run_id="run-1",
            node_id="node_pilot",
            candidate_id="pilot",
            status="failed",
            command=original,
            stdout=GPU_OOM_OUTPUT,
        )
    )
    item.status = "needs_resume"

    failure = _recover_failed_resource_item(item)

    assert failure is not None
    assert item.status == "queued"
    assert "batch=48" in item.command.argv
    assert "workers=2" in item.command.argv


def test_successful_recovery_caps_future_machine_commands(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    cache_path = tmp_path / "resource-policy.json"
    monkeypatch.setenv("YOLO_AGENT_RESOURCE_POLICY_CACHE", str(cache_path))
    failure = classify_execution_failure(stdout=HOST_OOM_OUTPUT, stderr="", command=_command())
    assert failure is not None
    recovered = apply_execution_recovery(_command(), failure)

    assert save_successful_resource_policy(recovered) == cache_path

    future, applied = apply_cached_resource_policy(_command(batch=64, workers=8))
    assert applied is True
    assert "workers=2" in future.argv
    assert "batch=48" in future.argv
    assert future.metadata["host_memory_policy_applied"] is True


def test_unrelated_training_failure_is_not_retried() -> None:
    failure = classify_execution_failure(
        stdout="RuntimeError: invalid model graph",
        stderr="",
        command=_command(),
    )

    assert failure is None
