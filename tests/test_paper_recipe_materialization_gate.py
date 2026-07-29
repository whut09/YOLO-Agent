from __future__ import annotations

import json
from pathlib import Path

from yolo_agent.agents.paper_recipe_materialization_gate import (
    PaperRecipeMaterializationGate,
)
from tests.paper_materialization_fixtures import (
    adapter_registry,
    candidate_input,
    gate_kwargs,
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
    assert any(line.startswith("Adapter patch: ") for line in result.terminal_lines)
    assert any(line.startswith("Runtime payload: ") for line in result.terminal_lines)
    assert "Budget authority: ASHA" in result.terminal_lines

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
