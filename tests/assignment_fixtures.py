"""Shared offline fixtures for guarded YOLO26 assignment tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.components.adapters.assigners.yolo26_assignment import (
    ASSIGNMENT_SPECS,
    AssignmentPaperPrior,
    YOLO26AssignmentRuntimePlugin,
)
from yolo_agent.components.assignment import AssignerInputs
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.recipes.schemas import AtomicRecipe, recipe_from_mapping


def assignment_inputs(*, anchor_representation: str = "point") -> AssignerInputs:
    points = torch.stack(
        [torch.linspace(4.0, 156.0, 20), torch.linspace(4.0, 156.0, 20)],
        dim=-1,
    )
    boxes = torch.stack(
        [points[:, 0] - 10, points[:, 1] - 10, points[:, 0] + 10, points[:, 1] + 10],
        dim=-1,
    ).unsqueeze(0)
    scores = torch.full((1, 20, 2), 0.1)
    scores[0, :10, 0] = torch.linspace(0.95, 0.55, 10)
    return AssignerInputs(
        predicted_scores=scores,
        predicted_boxes_xyxy=boxes,
        anchor_points_xy=points,
        stride_per_anchor=torch.full((20, 1), 8.0),
        gt_labels=torch.tensor([[[0.0]]]),
        gt_boxes_xyxy=torch.tensor([[[0.0, 0.0, 96.0, 96.0]]]),
        gt_mask=torch.tensor([[[True]]]),
        num_classes=2,
        path="one_to_many",
        anchor_representation=anchor_representation,  # type: ignore[arg-type]
    )


def native_model_and_criterion() -> tuple[torch.nn.Module, object]:
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel("yolo26n.yaml", ch=3, nc=3, verbose=False)
    model.args = get_cfg(overrides={"imgsz": 640})
    model.train()
    return model, model.init_criterion()


def detection_batch(image: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "img": image,
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
    }


def runtime_context(tmp_path: Path, method: str) -> SimpleNamespace:
    spec = next(item for item in ASSIGNMENT_SPECS.values() if item.method == method)
    return SimpleNamespace(
        payload_path=tmp_path / "adapter_runtime_payload.yaml",
        payload=SimpleNamespace(
            protocol_hash=f"protocol-{spec.method}",
            payload_hash=f"payload-{spec.method}-shadow",
            changed_variables={spec.changed_variable: "shadow"},
        ),
    )


def runtime_options(
    method: str,
    *,
    mode: str,
    minimum_shadow_batches: int,
    shadow_evidence_path: str | None = None,
    shadow_payload_hash: str | None = None,
) -> dict[str, object]:
    spec = next(item for item in ASSIGNMENT_SPECS.values() if item.method == method)
    return {
        "component_id": spec.component_id,
        "method": spec.method,
        "changed_variable": spec.changed_variable,
        "assignment_path": "one_to_many",
        "mode": mode,
        "imgsz": 640,
        "minimum_shadow_batches": minimum_shadow_batches,
        "maximum_conflict_rate": 1.0,
        "evidence_interval": 1,
        "shadow_evidence_path": shadow_evidence_path,
        "shadow_payload_hash": shadow_payload_hash,
        "paper_prior": AssignmentPaperPrior(
            paper_id=spec.paper_id,
            adaptation=spec.adaptation,
        ).model_dump(mode="json"),
    }


def run_one_shadow_batch(directory: Path, method: str) -> None:
    model, criterion = native_model_and_criterion()
    context = runtime_context(directory, method)
    plugin = YOLO26AssignmentRuntimePlugin(
        **runtime_options(method, mode="shadow", minimum_shadow_batches=1)
    )
    plugin.build_criterion(
        context=context,
        trainer=SimpleNamespace(),
        model=model,
        criterion=criterion,
    )
    image = torch.rand(1, 3, 64, 64)
    batch = detection_batch(image)
    predictions = model(image)
    output = criterion(predictions, batch)
    plugin.compute_loss(
        context=context,
        trainer=SimpleNamespace(),
        model=model,
        criterion=criterion,
        predictions=predictions,
        batch=batch,
        loss_output=output,
    )


def assignment_recipes() -> list[AtomicRecipe]:
    raw = yaml.safe_load(
        Path("configs/recipes/yolo26_assignment_shadow.yaml").read_text(
            encoding="utf-8"
        )
    )
    return [recipe_from_mapping(item) for item in raw["recipes"]]  # type: ignore[return-value]


def assignment_node(recipe: AtomicRecipe, tmp_path: Path) -> ExperimentNode:
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data=tmp_path / "coco.yaml",
        project=tmp_path / "runs",
        name=recipe.recipe_id,
        epochs=1,
        imgsz=640,
        overrides={
            recipe.primary_changed_variable: recipe.train_overrides[
                recipe.primary_changed_variable
            ]
        },
    )
    candidate = CandidateConfig(
        candidate_id=recipe.recipe_id,
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
        components=list(recipe.component_ids),
        train_overrides=dict(recipe.train_overrides),
    )
    return ExperimentNode(
        node_id=f"node_{recipe.recipe_id}",
        candidate_config=candidate,
        data_version="coco2017",
        seed=1,
        command=command.display(),
        command_spec=command,
    )
