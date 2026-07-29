"""Shared offline fixtures for guarded neck plugin tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.components.adapters.base import AdapterContext
from yolo_agent.components.contracts import ComponentContract, load_contracts
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.recipes.schemas import RecipeSpec, recipe_from_mapping


def neck_contracts() -> dict[str, ComponentContract]:
    return {
        item.component_id: item
        for item in load_contracts("configs/components/neck/yolo26_multi_scale.yaml")
    }


def neck_recipes() -> list[RecipeSpec]:
    raw = yaml.safe_load(
        Path("configs/recipes/yolo26_multi_scale_necks.yaml").read_text(encoding="utf-8")
    )
    return [recipe_from_mapping(item) for item in raw["recipes"]]


def neck_context(
    contract: ComponentContract,
    workspace: Path,
    options: dict[str, object] | None = None,
) -> AdapterContext:
    return AdapterContext(
        contract=contract,
        detector_family="yolo26",
        head="one_to_one",
        imgsz=640,
        workspace=workspace,
        options=options or {"imgsz": 640, "audit_imgsz": 64, "latency_iterations": 1},
    )


def neck_node(recipe: RecipeSpec, workspace: Path) -> ExperimentNode:
    candidate = CandidateConfig(
        candidate_id=recipe.recipe_id,
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
        components=list(recipe.component_ids),
        train_overrides=dict(recipe.train_overrides),
        action_id=recipe.recipe_id,
    )
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data=workspace / "coco.yaml",
        project=workspace / "runs",
        name=recipe.recipe_id,
        epochs=3,
        imgsz=640,
    )
    return ExperimentNode(
        node_id=f"node_{recipe.recipe_id}",
        candidate_config=candidate,
        data_version="coco2017",
        seed=1,
        command=command.display(),
        command_spec=command,
    )


__all__ = ["neck_context", "neck_contracts", "neck_node", "neck_recipes"]
