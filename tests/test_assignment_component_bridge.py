"""Assignment adapter to execution bridge tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from tests.assignment_fixtures import assignment_node, assignment_recipes
from tests.maturity_helpers import with_smoke_artifact
from yolo_agent.components.contracts import load_contracts
from yolo_agent.components.execution_bridge import ComponentExecutionBridge


def test_assignment_bridge_materializes_only_declared_shadow_variable(
    tmp_path: Path,
) -> None:
    contracts = {
        item.component_id: with_smoke_artifact(item)
        for item in load_contracts("configs/components/assigner/yolo26_assignment.yaml")
    }
    for recipe in assignment_recipes():
        result = ComponentExecutionBridge().prepare(
            recipe=recipe,
            node=assignment_node(recipe, tmp_path),
            contracts=contracts,
            training_config=dict(recipe.train_overrides),
            workspace=tmp_path / recipe.recipe_id,
            protocol_hash="protocol-1",
        )
        assert result.status == "executable", result.blocked_by
        assert set(result.changed_variables) == {
            f"training_config.{recipe.primary_changed_variable}"
        }
        assert result.runtime_payload_path is not None
        payload = yaml.safe_load(
            result.runtime_payload_path.read_text(encoding="utf-8")
        )
        assert len(payload["assigner_plugin"]) == 1
        assert payload["assigner_plugin"][0]["options"]["mode"] == "shadow"
        assert payload["loss_plugin"] == []
        assert payload["model_graph_plugin"] == []
        assert result.node.command_spec is not None
        assert result.node.command_spec.metadata["assignment_execution_mode"] == "shadow"
        assert result.node.command_spec.metadata["evidence_only"] is True
        assert result.node.command_spec.metadata["optimization_metric_eligible"] is False
