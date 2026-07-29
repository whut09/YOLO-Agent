from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.agents.paper_recipe_materialization_gate import (
    PaperRecipeMaterializationGate,
)
from yolo_agent.components.adapters import ComponentAdapterRegistry
from tests.paper_materialization_fixtures import candidate_input, contract, gate_kwargs


@pytest.mark.parametrize(
    ("component_contract", "expected_reason"),
    [
        (
            contract(
                maturity="metadata_only",
                implementation_path=None,
                adapter_class=None,
            ),
            "metadata_only_components_cannot_materialize_recipe",
        ),
        (
            contract(
                maturity="smoke_passed",
                implementation_path="missing.paper_adapter",
                adapter_class="MissingAdapter",
            ),
            "runtime_adapter_missing:dummy.component",
        ),
    ],
)
def test_missing_runtime_adapter_only_emits_implementation_request(
    tmp_path: Path,
    component_contract,
    expected_reason: str,
) -> None:
    run_dir = tmp_path / "paper-run"
    gate = PaperRecipeMaterializationGate(
        run_dir,
        base_run_id="paper-run",
        adapter_registry=ComponentAdapterRegistry(),
    )
    values = gate_kwargs(tmp_path, candidates=[candidate_input()])
    values["component_contracts"] = {"dummy.component": component_contract}

    result = gate.materialize(**values)

    assert result.action == "implementation_required"
    assert result.scalar_hpo_enabled is False
    assert result.execution_queue is None
    assert result.round_execution_plan is None
    assert not (run_dir / "execution_queue.yaml").exists()
    request = result.candidates[0].implementation_request
    assert request is not None
    assert request.generated_code_allowed is False
    assert expected_reason in ";".join(result.candidates[0].reasons)
    assert "scalar HPO is disabled" in "\n".join(result.terminal_lines)
