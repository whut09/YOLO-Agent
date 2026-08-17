from __future__ import annotations

import json
from pathlib import Path

from yolo_agent.agents.paper_recipe_materialization_gate import (
    PaperRecipeMaterializationGate,
)
from yolo_agent.agents.paper_proposal_ledger import PaperCandidateCoverage
from yolo_agent.components.compatibility import CompatibilityResult
from yolo_agent.certification.component_queue_gate import (
    ComponentQueueCertificationResult,
)
from tests.paper_materialization_fixtures import (
    adapter_registry,
    candidate_input,
    gate_kwargs,
    node,
)


def test_certified_recipe_enters_asha_plan_with_runtime_identity(tmp_path: Path) -> None:
    run_dir = tmp_path / "paper-run"
    gate = PaperRecipeMaterializationGate(
        run_dir,
        base_run_id="paper-run",
        adapter_registry=adapter_registry(),
    )

    result = gate.materialize(**gate_kwargs(tmp_path, candidates=[candidate_input()]))

    assert result.action == "queue_assignment"
    assert result.scalar_hpo_enabled is False
    assert result.asha_assignment_id
    assert result.round_execution_plan is not None
    assert result.execution_queue is not None
    assert result.execution_queue["metadata"]["source_authority"] == "RoundExecutionPlan"
    assert result.execution_queue["metadata"]["scheduler_mode"] == "external_asha"
    candidate_commands = [
        item["command"]
        for item in result.execution_queue["items"]
        if not item["command"]["metadata"].get("matched_baseline_control")
    ]
    assert len(candidate_commands) == 1
    assert "--payload" in candidate_commands[0]["argv"]
    assert "imgsz=640" in candidate_commands[0]["argv"]
    assert any(line.startswith("Adapter: dummy.component=DummyAdapter@") for line in result.terminal_lines)
    assert "Paper: paper-dummy" in result.terminal_lines
    assert "Component: dummy.component" in result.terminal_lines
    assert any(line.startswith("Adapter hash: dummy.component=") for line in result.terminal_lines)
    assert "Maturity: dummy.component=smoke_passed" in result.terminal_lines
    assert any(line.startswith("Adapter patch: ") for line in result.terminal_lines)
    assert any(line.startswith("Runtime payload: ") for line in result.terminal_lines)
    assert "Budget authority: ASHA" in result.terminal_lines
    assert any(line.startswith("Planning priority: score=") for line in result.terminal_lines)
    assert result.candidates[0].planning_priority is not None
    assert result.candidates[0].planning_priority.covered_paper_count == 1

    ledger = [
        json.loads(line)
        for line in (run_dir / "artifacts" / "decision_ledger.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    registration = next(
        item for item in ledger if item["decision_type"] == "paper_candidate_registration"
    )
    assert registration["proposal"]["adapter_ids"] == ["dummy.component"]
    assert registration["proposal"]["adapter_patch_hash"]
    assert registration["proposal"]["adapter_runtime_payload_hash"]
    assert registration["proposal"]["planning_priority"]["covered_paper_count"] == 1

    coverage = PaperCandidateCoverage.from_yaml(
        run_dir / "artifacts" / "paper_candidate_coverage.yaml"
    )
    record = coverage.records[0]
    assert record.paper_ids == ["paper-dummy"]
    assert record.method_profile_ids == ["profile-paper-dummy"]
    assert [event.source_stage for event in record.stage_history] == [
        "materialization_input",
        "materialization",
        "asha_registration",
        "round_execution_plan",
    ]


def test_matched_control_is_required_before_asha_registration(tmp_path: Path) -> None:
    item = candidate_input()
    item.matched_control_node = None
    gate = PaperRecipeMaterializationGate(
        tmp_path / "paper-run",
        base_run_id="paper-run",
        adapter_registry=adapter_registry(),
    )

    result = gate.materialize(**gate_kwargs(tmp_path, candidates=[item]))

    assert result.action == "exhausted"
    assert "matched_control_missing" in result.candidates[0].reasons
    assert result.execution_queue is None
    assert gate.orchestrator.scheduler.study.trials == []


def test_component_certification_blocker_prevents_asha_registration(
    tmp_path: Path,
) -> None:
    gate = PaperRecipeMaterializationGate(
        tmp_path / "paper-run",
        base_run_id="paper-run",
        adapter_registry=adapter_registry(),
    )
    gate.component_certification_gate.evaluate = lambda **_: (  # type: ignore[method-assign]
        ComponentQueueCertificationResult(
            allowed=False,
            component_ids=["sampling.small_object"],
            blockers=["sampling_end_to_end_certification_report_missing"],
        )
    )

    result = gate.materialize(
        **gate_kwargs(tmp_path, candidates=[candidate_input()])
    )

    assert result.action == "exhausted"
    assert result.candidates[0].reasons == [
        "sampling_end_to_end_certification_report_missing"
    ]
    assert gate.orchestrator.scheduler.study.trials == []


def test_protocol_mismatched_control_is_rejected(tmp_path: Path) -> None:
    item = candidate_input()
    control = node("old-control", control=True)
    metadata = dict(control.command_spec.metadata)
    metadata["protocol_hash"] = "old-protocol"
    metadata["run_protocol_hash"] = "old-protocol"
    control.command_spec = control.command_spec.model_copy(update={"metadata": metadata})
    item.matched_control_node = control
    gate = PaperRecipeMaterializationGate(
        tmp_path / "paper-run",
        base_run_id="paper-run",
        adapter_registry=adapter_registry(),
    )

    result = gate.materialize(**gate_kwargs(tmp_path, candidates=[item]))

    assert "matched_control_protocol_mismatch" in result.candidates[0].reasons
    assert gate.orchestrator.scheduler.study.trials == []


def test_method_profile_mismatch_cannot_enter_asha(tmp_path: Path) -> None:
    item = candidate_input()
    item.method_profile = item.method_profile.model_copy(
        update={"paper_id": "different-paper"}
    )
    gate = PaperRecipeMaterializationGate(
        tmp_path / "paper-run",
        base_run_id="paper-run",
        adapter_registry=adapter_registry(),
    )

    result = gate.materialize(
        **gate_kwargs(tmp_path, candidates=[item])
    )

    assert result.action == "exhausted"
    assert result.candidates[0].reasons == [
        "paper_method_profile_paper_mismatch",
        "paper_method_profile_decision_mismatch",
    ]
    assert result.stopped_reason == "no_certified_paper_components"
    output = "\n".join(result.terminal_lines)
    assert "paper_id=paper-dummy" in output
    assert "component_id=dummy.component" in output
    assert "reason=paper_method_profile_paper_mismatch" in output
    assert "Scalar HPO: disabled" in output
    assert gate.orchestrator.scheduler.study.trials == []


def test_compatibility_failure_cannot_be_overridden(tmp_path: Path) -> None:
    item = candidate_input()
    item.compatibility = CompatibilityResult(
        ok=False,
        errors=["component violates YOLO26 native head semantics"],
        estimated_risk="high",
    )
    gate = PaperRecipeMaterializationGate(
        tmp_path / "paper-run",
        base_run_id="paper-run",
        adapter_registry=adapter_registry(),
    )

    result = gate.materialize(**gate_kwargs(tmp_path, candidates=[item]))

    reasons = ";".join(result.candidates[0].reasons)
    assert "compatibility_error:component violates YOLO26 native head semantics" in reasons
    assert "compatibility_failed" in reasons
    assert gate.orchestrator.scheduler.study.trials == []


def test_non_640_source_node_never_reaches_asha(tmp_path: Path) -> None:
    item = candidate_input()
    item.source_node = node("wrong-imgsz", imgsz=672)
    gate = PaperRecipeMaterializationGate(
        tmp_path / "paper-run",
        base_run_id="paper-run",
        adapter_registry=adapter_registry(),
    )

    result = gate.materialize(**gate_kwargs(tmp_path, candidates=[item]))

    assert "fixed_imgsz_must_equal_640" in result.candidates[0].reasons
    assert gate.orchestrator.scheduler.study.trials == []
