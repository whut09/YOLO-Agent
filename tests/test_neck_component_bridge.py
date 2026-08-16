"""Execution bridge tests for guarded model-graph neck recipes."""

from __future__ import annotations

from pathlib import Path

from tests.neck_fixtures import neck_contracts, neck_node, neck_recipes
from tests.maturity_helpers import with_smoke_artifact
from yolo_agent.components.adapters.runtime import AdapterRuntimePayload
from yolo_agent.components.execution_bridge import ComponentExecutionBridge


def test_each_neck_recipe_builds_one_model_graph_runtime_payload(tmp_path: Path) -> None:
    contracts = {
        component_id: with_smoke_artifact(contract)
        for component_id, contract in neck_contracts().items()
    }
    for recipe in neck_recipes():
        workspace = tmp_path / recipe.recipe_id
        result = ComponentExecutionBridge().prepare(
            recipe=recipe,
            node=neck_node(recipe, tmp_path),
            contracts=contracts,
            training_config=dict(recipe.train_overrides),
            workspace=workspace,
            protocol_hash="neck-protocol",
        )

        assert result.status == "executable", result.blocked_by
        assert result.runtime_payload_path is not None
        assert set(result.changed_variables) == {"model_config.neck_plugin"}
        payload = AdapterRuntimePayload.read(result.runtime_payload_path)
        assert len(payload.model_graph_plugin) == 1
        assert not payload.loss_plugin
        assert not payload.assigner_plugin
        assert not payload.dataloader_plugin
        assert result.node.command_spec is not None
        assert result.node.command_spec.metadata["adapter_guard_metrics"] == (
            "latency_ms,peak_vram_mb,model_size_mb"
        )
        assert result.node.command_spec.metadata["adapter_peak_vram_source"] == (
            "runtime_measurement_after_training"
        )


def test_missing_deformable_dependency_becomes_implementation_request(
    tmp_path: Path,
) -> None:
    recipe = neck_recipes()[1].model_copy(
        update={
            "train_overrides": {
                **neck_recipes()[1].train_overrides,
                "deformable_module": "missing_yolo_agent_deformable_operator",
            }
        }
    )
    result = ComponentExecutionBridge().prepare(
        recipe=recipe,
        node=neck_node(recipe, tmp_path),
        contracts={
            component_id: with_smoke_artifact(contract)
            for component_id, contract in neck_contracts().items()
        },
        training_config=dict(recipe.train_overrides),
        workspace=tmp_path / "missing-deformable",
        protocol_hash="neck-protocol",
    )

    assert result.status == "adapter_required"
    assert any(item.startswith("adapter_implementation_request:") for item in result.blocked_by)
