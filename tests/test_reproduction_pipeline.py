from pathlib import Path

import pytest

from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.execution_queue import ExecutionQueueStore
from yolo_agent.research.reproduction_pipeline import ReproductionPipeline, ReproductionTransitionError


def _pipeline(tmp_path: Path) -> ReproductionPipeline:
    return ReproductionPipeline(tmp_path / "run", "paper:demo", "sampling.demo")


def test_state_contracts_and_resume(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    state = pipeline.initialize(paper_claims=[{"delta": "+1 AP_small"}], evidence={"paper_record": True})
    assert state.status == "registered"
    pipeline.transition("license_checked", evidence={"source_verified": True, "license_checked": True})
    pipeline.transition("adapter_required")
    pipeline.transition("adapter_implemented", evidence={"adapter_implemented": True})
    resumed = ReproductionPipeline(tmp_path / "run", "paper:demo", "sampling.demo").load()
    assert resumed.status == "adapter_implemented"
    assert resumed.paper_claims[0]["delta"] == "+1 AP_small"


def test_smoke_requires_all_local_checks_and_preserves_failure(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline.initialize(evidence={"paper_record": True})
    pipeline.transition("license_checked", evidence={"source_verified": True, "license_checked": True})
    pipeline.transition("adapter_required")
    pipeline.transition("adapter_implemented", evidence={"adapter_implemented": True})
    pipeline.transition("unit_tested", evidence={"unit_tested": True})
    with pytest.raises(ReproductionTransitionError, match="missing required evidence"):
        pipeline.transition("smoke_passed", evidence={"construction_test": True})
    pipeline.fail("tensor shape mismatch")
    assert pipeline.load().last_error == "tensor shape mismatch"
    assert (tmp_path / "run" / "artifacts" / "reproduction_state.yaml").exists()


def test_pilot_then_explicit_full_gate(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline.initialize(evidence={"paper_record": True})
    pipeline.transition("license_checked", evidence={"source_verified": True, "license_checked": True})
    pipeline.transition("adapter_implemented", evidence={"adapter_implemented": True})
    pipeline.transition("unit_tested", evidence={"unit_tested": True})
    pipeline.transition("smoke_passed", evidence={"construction_test": True, "tensor_shape": True, "backward": True, "amp": True, "cpu_smoke": True})
    pipeline.transition("debug_passed", evidence={"debug_evidence": True})
    pipeline.transition("pilot_running", evidence={"pilot_queued": True})
    pipeline.transition("pilot_reproduced", evidence={"pilot_evidence": True}, local_delta={"map50_95": 0.01})
    with pytest.raises(ReproductionTransitionError, match="confirm_full"):
        pipeline.transition("full_reproduced", evidence={"full_evidence": True})
    pipeline.transition("full_pending_confirmation", evidence={"full_confirmation_requested": True})
    state = pipeline.transition("full_reproduced", evidence={"full_evidence": True}, confirm_full=True)
    assert state.status == "full_reproduced"
    assert state.local_delta["map50_95"] == 0.01


def test_enqueue_uses_existing_execution_queue_and_blocks_full(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline.initialize(evidence={"paper_record": True})
    store = ExecutionQueueStore(tmp_path / "run")
    command = CommandSpec(command_type="custom", argv=["echo", "debug"])
    with pytest.raises(ReproductionTransitionError, match="debug_passed"):
        pipeline.enqueue("pilot", run_id="run", queue_store=store, command=command)
    pipeline.transition("debug_passed", evidence={"debug_evidence": True, "smoke_passed": True})
    pipeline.enqueue("pilot", run_id="run", queue_store=store, command=command)
    assert store.load().counts()["queued"] == 1
    with pytest.raises(ReproductionTransitionError, match="pilot_reproduced"):
        pipeline.enqueue("full", run_id="run", queue_store=store, command=command, confirm_full=True)


def test_reconcile_queue_requires_imported_pilot_evidence(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline.initialize(evidence={"paper_record": True})
    pipeline.transition("debug_passed", evidence={"debug_evidence": True, "smoke_passed": True})
    store = ExecutionQueueStore(tmp_path / "run")
    queue = pipeline.enqueue("pilot", run_id="run", queue_store=store, command=CommandSpec(argv=["echo", "pilot"]))
    item = queue.items[0]
    item.status = "completed"
    store.update_item(item)
    with pytest.raises(ReproductionTransitionError, match="pilot_evidence"):
        pipeline.reconcile_queue(store.load())
    state = pipeline.reconcile_queue(store.load(), evidence={"pilot_evidence": True}, local_delta={"ap_small": 0.02})
    assert state.status == "pilot_reproduced"
    assert state.local_delta["ap_small"] == 0.02
