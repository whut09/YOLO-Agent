from pathlib import Path

import pytest

from yolo_agent.certification.assignment_pilot_state import (
    AssignmentPilotState,
    AssignmentPilotStateLedger,
    assignment_state_path,
)


def _state(component: str = "assigner.task_aligned") -> AssignmentPilotState:
    return AssignmentPilotState(
        run_id="run-1",
        trial_id=f"run-1:{component}",
        candidate_id=f"candidate-{component}",
        canonical_component_id=component,
        shadow_recipe_id=f"shadow-{component}",
        protocol_hash="protocol-1",
        matched_control_node_id="control",
        matched_control_protocol_hash="protocol-1",
    )


def test_assignment_state_machine_is_ordered_and_idempotent() -> None:
    item = _state()
    for state in (
        "shadow_evidence_complete",
        "active_candidate_eligible",
        "active_pilot",
        "promoted",
    ):
        item.transition(state)  # type: ignore[arg-type]
    item.transition("promoted")
    assert item.state == "promoted"
    with pytest.raises(ValueError, match="cannot move backward"):
        item.transition("active_pilot")


def test_rejected_is_terminal_and_reason_is_persisted(tmp_path: Path) -> None:
    ledger = AssignmentPilotStateLedger(run_id="run-1")
    ledger.upsert(_state("assigner.optimal_transport"))
    ledger.transition(
        "run-1:assigner.optimal_transport",
        "rejected",
        disposition="blocked_runtime",
        reason_codes=["native_loss_equivalence_failed"],
    )
    path = assignment_state_path(tmp_path / "run-1")
    ledger.save(path)
    loaded = AssignmentPilotStateLedger.load_or_create(path, run_id="run-1")
    record = loaded.record("run-1:assigner.optimal_transport")
    assert record is not None
    assert record.state == "rejected"
    assert record.disposition == "blocked_runtime"
    assert record.reason_codes == ["native_loss_equivalence_failed"]


def test_task_aligned_and_ota_have_independent_records() -> None:
    ledger = AssignmentPilotStateLedger(run_id="run-1")
    ledger.upsert(_state())
    ledger.upsert(_state("assigner.optimal_transport"))
    assert [item.canonical_component_id for item in ledger.records] == [
        "assigner.task_aligned",
        "assigner.optimal_transport",
    ]
