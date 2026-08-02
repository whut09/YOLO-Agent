"""Quality-alignment auxiliary loss API and YOLO26 runtime tests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.components.adapters.base import AdapterContext
from yolo_agent.components.adapters.losses.quality_alignment import (
    LOSS_SPECS,
    AuxiliaryPaperPrior,
    QualityAlignmentAuxiliaryLossAdapter,
    QualityAlignmentRuntimePlugin,
)
from yolo_agent.components.auxiliary_losses import (
    AuxiliaryLossInputs,
    AuxiliaryLossPlugin,
    build_auxiliary_loss,
)
from yolo_agent.components.contracts import load_contracts
from yolo_agent.components.maturity import ComponentMaturityArtifact
from yolo_agent.components.execution_bridge import ComponentExecutionBridge
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.recipes.schemas import AtomicRecipe, recipe_from_mapping


def _synthetic_inputs() -> AuxiliaryLossInputs:
    return AuxiliaryLossInputs(
        class_logits=torch.tensor(
            [[[2.0, -1.0], [-0.5, 1.5], [1.0, 0.0]]], requires_grad=True
        ),
        predicted_boxes_xyxy=torch.tensor(
            [[[1.0, 1.0, 5.0, 5.0], [5.0, 5.0, 9.0, 9.0], [0.0] * 4]],
            requires_grad=True,
        ),
        target_boxes_xyxy=torch.tensor(
            [[[1.0, 1.0, 5.0, 5.0], [4.0, 4.0, 9.0, 9.0], [0.0] * 4]]
        ),
        target_classes=torch.tensor([[0, 1, 0]]),
        foreground_mask=torch.tensor([[True, True, False]]),
        anchor_points_xy=torch.tensor([[3.0, 3.0], [7.0, 7.0], [1.0, 1.0]]),
    )


@pytest.mark.parametrize("loss_name", ["correlation", "bpc_calibration", "pseudo_iou"])
def test_auxiliary_loss_api_shape_backward_and_amp(loss_name: str) -> None:
    inputs = _synthetic_inputs()
    plugin = build_auxiliary_loss(loss_name)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = plugin.compute(inputs)
    output.loss.backward()

    assert isinstance(plugin, AuxiliaryLossPlugin)
    assert output.loss.ndim == 0 and torch.isfinite(output.loss)
    assert inputs.class_logits.grad is not None
    assert inputs.predicted_boxes_xyxy.grad is None


@pytest.mark.parametrize(
    "loss_name",
    [
        "iou_aware_classification",
        "localization_aware_classification",
        "boundary_aware",
        "uncertainty_weighted_regression",
        "hard_negative_classification",
        "class_balanced_focal",
    ],
)
def test_extended_auxiliary_loss_family_shape_backward_and_amp(
    loss_name: str,
) -> None:
    inputs = _synthetic_inputs()
    plugin = build_auxiliary_loss(loss_name)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = plugin.compute(inputs)
    output.loss.backward()

    assert isinstance(plugin, AuxiliaryLossPlugin)
    assert output.loss.ndim == 0 and torch.isfinite(output.loss)
    assert inputs.class_logits.grad is not None or inputs.predicted_boxes_xyxy.grad is not None


def test_regression_auxiliary_terms_preserve_gradient_to_decoded_boxes() -> None:
    for loss_name in ("boundary_aware", "uncertainty_weighted_regression"):
        inputs = _synthetic_inputs()
        output = build_auxiliary_loss(loss_name).compute(inputs)
        output.loss.backward()
        assert inputs.predicted_boxes_xyxy.grad is not None


@pytest.mark.parametrize("loss_name", ["correlation", "bpc_calibration", "pseudo_iou"])
def test_zero_weight_runtime_is_native_loss_equivalent(
    loss_name: str, tmp_path: Path
) -> None:
    plugin = QualityAlignmentRuntimePlugin(**_runtime_options(loss_name, weight=0.0))
    trainer = SimpleNamespace(loss_names=("box_loss", "cls_loss", "dfl_loss"))
    criterion = _mock_criterion()
    context = _runtime_context(tmp_path)
    native_loss = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    native_items = torch.tensor([0.1, 0.2, 0.3])

    output_loss, output_items = plugin.compute_loss(
        context=context,
        trainer=trainer,
        model=object(),
        criterion=criterion,
        predictions={},
        batch={},
        loss_output=(native_loss, native_items),
    )

    assert torch.equal(output_loss, native_loss)
    assert torch.equal(output_items[:3], native_items)
    assert output_items[3].item() == 0.0
    assert trainer.loss_names[-1] == f"aux_{loss_name}_loss"


def test_native_yolo26_runtime_logs_loss_and_checkpoint_metadata(tmp_path: Path) -> None:
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel("yolo26n.yaml", ch=3, nc=3, verbose=False)
    model.args = get_cfg(overrides={"imgsz": 640})
    model.train()
    criterion = model.init_criterion()
    native_bbox_loss = criterion.one2many.bbox_loss
    context = _runtime_context(tmp_path)
    image = torch.rand(1, 3, 64, 64)
    batch = {
        "img": image,
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
    }

    for loss_name in ("correlation", "bpc_calibration", "pseudo_iou"):
        model.zero_grad(set_to_none=True)
        trainer = SimpleNamespace(loss_names=("box_loss", "cls_loss", "dfl_loss"))
        plugin = QualityAlignmentRuntimePlugin(**_runtime_options(loss_name, weight=0.1))
        assert plugin.build_model(context=context, trainer=trainer, model=model) is model
        assert trainer.loss_names[-1] == f"aux_{loss_name}_loss"
        trainer.loss_names = ("box_loss", "cls_loss", "dfl_loss")
        validator = object()
        assert plugin.build_validator(
            context=context,
            trainer=trainer,
            validator=validator,
        ) is validator
        assert trainer.loss_names[-1] == f"aux_{loss_name}_loss"
        assert plugin.build_criterion(
            context=context,
            trainer=trainer,
            model=model,
            criterion=criterion,
        ) is criterion
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            predictions = model(image)
            native = criterion(predictions, batch)
            loss, loss_items = plugin.compute_loss(
                context=context,
                trainer=trainer,
                model=model,
                criterion=criterion,
                predictions=predictions,
                batch=batch,
                loss_output=native,
            )
        loss.sum().backward()
        assert loss_items.shape == (4,)
        assert trainer.loss_names[-1] == f"aux_{loss_name}_loss"
        assert loss_name in trainer.auxiliary_loss_terms
        assert criterion.one2many.bbox_loss is native_bbox_loss
        assert criterion.one2many.use_dfl is False
        evidence_path = tmp_path / f"auxiliary_loss_{loss_name}_evidence.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        assert evidence["replaces_bbox_regression"] is False
        assert evidence["replaces_assigner"] is False
        assert evidence["changes_inference_graph"] is False
        assert evidence["paper_prior"]["evidence_level"] == "paper_prior"
        assert evidence["paper_prior"]["reported_delta"] == {}

        checkpoint = tmp_path / f"{loss_name}.pt"
        checkpoint.write_bytes(b"checkpoint")
        plugin.on_checkpoint_save(
            context=context,
            trainer=trainer,
            checkpoints={"last": checkpoint},
        )
        metadata_path = checkpoint.with_suffix(
            checkpoint.suffix + f".auxiliary_loss.{loss_name}.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["batch_log_name"] == f"aux_{loss_name}_loss"
        assert metadata["checkpoint_sha256"]


def test_contracts_and_recipes_are_atomic_and_paper_prior_only() -> None:
    contracts = load_contracts("configs/components/loss/quality_alignment.yaml")
    recipes = _recipes()

    assert len(contracts) == len(recipes) == 3
    assert all(item.maturity == "adapter_implemented" and not item.can_execute for item in contracts)
    assert all(isinstance(item, AtomicRecipe) and not item.is_executable for item in recipes)
    assert {item.primary_changed_variable for item in recipes} == {
        "loss.correlation.weight",
        "loss.bpc_calibration.weight",
        "loss.pseudo_iou.weight",
    }
    for recipe in recipes:
        assert recipe.train_overrides["imgsz"] == 640
        assert recipe.train_overrides[recipe.primary_changed_variable] >= 0.0
        assert all(item["evidence_level"] == "paper_prior" for item in recipe.evidence_prior)
        assert all(item.get("local_evidence") is False for item in recipe.evidence_prior)


def test_bpc_runtime_payload_preserves_certification_threshold(tmp_path: Path) -> None:
    contract = next(
        item
        for item in load_contracts("configs/components/loss/quality_alignment.yaml")
        if item.component_id == "loss.calibration.bpc"
    )
    payload = QualityAlignmentAuxiliaryLossAdapter().build_runtime_payload(
        AdapterContext(
            contract=contract,
            workspace=tmp_path,
            options={"confidence_threshold": 0.0},
        ),
        protocol_hash="protocol",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={},
    )

    assert payload.loss_plugin[0].options["confidence_threshold"] == 0.0
    assert set(payload.changed_variables) == {"loss.bpc_calibration.weight"}


def test_component_bridge_materializes_exact_weight_variable(tmp_path: Path) -> None:
    artifact_path = Path(__file__)
    contracts = {}
    for item in load_contracts("configs/components/loss/quality_alignment.yaml"):
        artifact = ComponentMaturityArtifact(
            component_id=item.component_id,
            target_maturity="smoke_passed",
            artifact_type="smoke_report",
            artifact_path=artifact_path,
            artifact_sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
            status="passed",
            producer="pytest_fixture",
        )
        contracts[item.component_id] = item.model_copy(
            update={"maturity": "smoke_passed", "maturity_artifacts": [artifact]}
        )
    for recipe in _recipes():
        recipe = recipe.model_copy(update={"maturity": "smoke_passed"})
        command = CommandSpec.ultralytics_train(
            model="yolo26n.pt",
            data=tmp_path / "coco.yaml",
            project=tmp_path / "runs",
            name=recipe.recipe_id,
            epochs=3,
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
        node = ExperimentNode(
            node_id=f"node_{recipe.recipe_id}",
            candidate_config=candidate,
            data_version="coco2017",
            seed=1,
            command=command.display(),
            command_spec=command,
        )

        result = ComponentExecutionBridge().prepare(
            recipe=recipe,
            node=node,
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("YOLO_AGENT_RUN_GPU_TESTS") != "1",
    reason="set YOLO_AGENT_RUN_GPU_TESTS=1 for optional GPU adapter smoke",
)
@pytest.mark.parametrize(
    "component_id",
    sorted(LOSS_SPECS),
)
def test_optional_quality_loss_gpu_smoke(
    component_id: str,
    tmp_path: Path,
) -> None:
    contract = next(
        item
        for item in load_contracts("configs/components/loss/quality_alignment.yaml")
        if item.component_id == component_id
    )
    context = AdapterContext(
        contract=contract,
        detector_family="yolo26",
        head="one_to_one",
        imgsz=640,
        workspace=tmp_path,
    )

    result = QualityAlignmentAuxiliaryLossAdapter().gpu_smoke_test(context)

    assert result.passed, result.errors
    assert result.checks["zero_weight_native_equivalent"] is True


def _runtime_options(loss_name: str, *, weight: float) -> dict[str, object]:
    spec = next(item for item in LOSS_SPECS.values() if item.loss_name == loss_name)
    return {
        "loss_name": loss_name,
        "component_id": spec.component_id,
        "changed_variable": spec.changed_variable,
        "weight": weight,
        "imgsz": 640,
        "paper_prior": AuxiliaryPaperPrior(
            paper_id=spec.paper_id,
            adaptation=spec.adaptation,
        ).model_dump(mode="json"),
    }


def _runtime_context(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        payload_path=tmp_path / "adapter_runtime_payload.yaml",
        payload=SimpleNamespace(protocol_hash="protocol-1"),
    )


def _mock_criterion() -> SimpleNamespace:
    native = SimpleNamespace(
        assigner=object(),
        bbox_loss=object(),
        bbox_decode=lambda *args: None,
        preprocess=lambda *args, **kwargs: None,
        stride=torch.tensor([8.0, 16.0, 32.0]),
        device=torch.device("cpu"),
        use_dfl=False,
    )
    return SimpleNamespace(one2many=native)


def _recipes() -> list[AtomicRecipe]:
    raw = yaml.safe_load(
        Path("configs/recipes/yolo26_quality_alignment.yaml").read_text(encoding="utf-8")
    )
    return [recipe_from_mapping(item) for item in raw["recipes"]]  # type: ignore[return-value]
