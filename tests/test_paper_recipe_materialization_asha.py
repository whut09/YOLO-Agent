from __future__ import annotations

from pathlib import Path

from yolo_agent.agents.paper_recipe_materialization_gate import (
    PaperRecipeMaterializationGate,
)
from tests.paper_materialization_fixtures import (
    adapter_registry,
    candidate_input,
    gate_kwargs,
)


def test_materialization_registers_cohort_but_asha_assigns_one_pilot(
    tmp_path: Path,
) -> None:
    candidates = []
    for suffix in ("a", "b", "c"):
        item = candidate_input(
            prior_id=f"paper-prior-{suffix}",
            candidate_id=f"paper-candidate-{suffix}",
        )
        config = item.source_node.candidate_config.model_copy(
            update={"action_id": f"paper-action-{suffix}"}
        )
        item.source_node = item.source_node.model_copy(update={"candidate_config": config})
        item.component_family = f"sampling-{suffix}"
        candidates.append(item)
    gate = PaperRecipeMaterializationGate(
        tmp_path / "paper-run",
        base_run_id="paper-run",
        adapter_registry=adapter_registry(),
    )

    result = gate.materialize(**gate_kwargs(tmp_path, candidates=candidates))

    assert result.action == "queue_assignment"
    assert sorted(result.registration["registered"]) == [
        "paper-candidate-a",
        "paper-candidate-b",
        "paper-candidate-c",
    ]
    assert len(gate.orchestrator.scheduler.study.trials) == 3
    assert result.round_execution_plan["scheduler_mode"] == "external_asha"
    assert result.round_execution_plan["assignments"][0]["stage_id"] == "pilot_3"
    assert result.execution_queue["metadata"]["source_authority"] == "RoundExecutionPlan"
    train_items = [
        item
        for item in result.execution_queue["items"]
        if item["command"]["command_type"] == "train"
    ]
    assert len(train_items) == 2
    candidate = next(
        item
        for item in train_items
        if not item["command"]["metadata"].get("matched_baseline_control")
    )
    control = next(
        item
        for item in train_items
        if item["command"]["metadata"].get("matched_baseline_control")
    )
    assert "--payload" in candidate["command"]["argv"]
    assert "--payload" not in control["command"]["argv"]
    assert candidate["command"]["metadata"]["adapter_runtime_payload_hash"]
    assert result.candidates[0].runtime_identity is not None
    assert all(item.prior.model_dump().get("execution_queue") is None for item in candidates)
