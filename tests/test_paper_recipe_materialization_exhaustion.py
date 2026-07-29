from __future__ import annotations

import json
from pathlib import Path

from yolo_agent.agents.paper_recipe_materialization_gate import (
    PaperRecipeMaterializationGate,
)
from tests.paper_materialization_fixtures import adapter_registry, gate_kwargs


def test_empty_paper_recipe_space_stops_without_scalar_hpo(tmp_path: Path) -> None:
    run_dir = tmp_path / "paper-run"
    gate = PaperRecipeMaterializationGate(
        run_dir,
        base_run_id="paper-run",
        adapter_registry=adapter_registry(),
    )

    result = gate.materialize(**gate_kwargs(tmp_path, candidates=[]))

    assert result.action == "exhausted"
    assert result.stopped_reason == "paper_component_recipes_exhausted"
    assert result.scalar_hpo_enabled is False
    assert result.execution_queue is None
    assert result.round_execution_plan is None
    assert not (run_dir / "execution_queue.yaml").exists()
    assert not (run_dir / "artifacts" / "asha_state.yaml").exists()
    output = "\n".join(result.terminal_lines)
    assert "scalar HPO is disabled" in output
    assert "no ASHA assignment created" in output

    ledger = [
        json.loads(line)
        for line in (run_dir / "artifacts" / "decision_ledger.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert ledger[-1]["decision"] == "exhausted"
    assert ledger[-1]["proposal"] == {
        "queue_authority": "ASHA/RoundExecutionPlan",
        "scalar_hpo_enabled": False,
    }
