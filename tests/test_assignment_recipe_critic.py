"""Structural assignment recipe critic tests."""

from __future__ import annotations

from yolo_agent.agents.recipe_critic import RecipeCritic
from yolo_agent.components.contracts import ComponentContract, load_contracts
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.recipes.schemas import AtomicRecipe


def test_assigner_plus_head_or_loss_requires_coupled_recipe() -> None:
    assigner = load_contracts("configs/components/assigner/yolo26_assignment.yaml")[0]
    loss = ComponentContract(
        component_id="loss.test",
        display_name="Test Loss",
        category="classification_loss",
        implementation_path="tests.fixture",
        adapter_class="FixtureAdapter",
        maturity="smoke_passed",
    )
    recipe = AtomicRecipe(
        recipe_id="invalid_atomic_assignment_and_loss",
        version="v1",
        target_error_facts=[{"fact_type": "localization_heavy_class"}],
        target_metrics=["latency", "model_size"],
        component_ids=[assigner.component_id, loss.component_id],
        train_overrides={"imgsz": 640},
        fixed_variables={"imgsz": 640},
        primary_changed_variable="assigner",
        stop_conditions=["latency_guard", "model_size_guard"],
        maturity="smoke_passed",
    )
    fact = ErrorFact(
        run_id="run-1",
        candidate_id="candidate-1",
        node_id="node-1",
        dataset_version="coco",
        fact_type="localization_heavy_class",
        subject="all",
        severity="high",
    )

    report = RecipeCritic().critique(
        recipe,
        error_facts=[fact],
        component_contracts=[assigner, loss],
        compatibility={assigner.component_id: True, loss.component_id: True},
    )

    assert "assigner_head_loss_requires_coupled_recipe" in report.blocked_by
