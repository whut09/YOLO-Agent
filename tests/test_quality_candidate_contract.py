from __future__ import annotations

import json

from yolo_agent.agents.candidate_generator import (
    CandidateConfig,
    CandidateEvaluationContract,
)
from yolo_agent.agents.paper_recipe_planner import _evaluation_contract
from yolo_agent.recipes.registry import RecipeRegistry
from yolo_agent.agents.paper_recipe_materialization.runtime_identity import (
    validate_certified_runtime_node,
)
from yolo_agent.components.adapters.runtime import AdapterRuntimePayload
from yolo_agent.components.adapters.base import RollbackPlan
from yolo_agent.components.adapters.runtime import RuntimePluginReference
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.experiment_graph import ExperimentNode


QUALITY_RECIPE_IDS = {
    "yolo26_correlation_auxiliary_loss",
    "yolo26_pseudo_iou_quality_auxiliary_loss",
}


def test_quality_recipes_preserve_primary_localization_and_resource_guards() -> None:
    registry = RecipeRegistry.from_paths(
        ["configs/recipes/yolo26_quality_alignment.yaml"], strict=False
    )
    recipes = [recipe for recipe in registry.list() if recipe.recipe_id in QUALITY_RECIPE_IDS]

    assert {recipe.recipe_id for recipe in recipes} == QUALITY_RECIPE_IDS
    for recipe in recipes:
        contract = _evaluation_contract(recipe)
        assert contract.primary_metric == "map50_95"
        assert "map50_95" in contract.evaluation_metrics
        assert any(
            metric in contract.evaluation_metrics
            for metric in {"ap75", "confidence_iou_correlation"}
        )
        assert contract.latency_metric == "latency_ms"
        assert contract.model_size_metric == "model_size_mb"
        assert "latency_guard" in " ".join(contract.promotion_requirements)
        assert "model_size_guard" in " ".join(contract.promotion_requirements)


def test_candidate_evaluation_contract_is_backward_compatible_and_deduplicated() -> None:
    legacy = CandidateConfig(
        candidate_id="legacy",
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
    )
    assert legacy.evaluation_contract.evaluation_metrics == [
        "map50_95",
        "latency_ms",
        "model_size_mb",
    ]

    contract = CandidateEvaluationContract(
        primary_metric="map50_95",
        evaluation_metrics=["map50_95", "ap75", "ap75"],
    )
    assert contract.evaluation_metrics == [
        "map50_95",
        "ap75",
        "latency_ms",
        "model_size_mb",
    ]


def _quality_node(tmp_path, *, imgsz: int = 640, changed: bool = True) -> ExperimentNode:
    variable = {"loss.correlation.weight": 0.2}
    payload = AdapterRuntimePayload(
        component_ids=["loss.quality.correlation"],
        adapter_classes=["QualityAlignmentAuxiliaryLossAdapter"],
        adapter_versions={"loss.quality.correlation": "v1"},
        source_commits={"loss.quality.correlation": "test"},
        loss_plugin=[
            RuntimePluginReference(
                reference="yolo_agent.components.adapters.losses.quality_alignment:QualityAlignmentRuntimePlugin",
                options={"loss_name": "correlation", "weight": 0.2},
                required_hooks=["compute_loss"],
            )
        ],
        changed_variables=variable,
        rollback_plan=RollbackPlan(actions=["remove quality adapter"]),
        protocol_hash="quality-protocol",
        base_command=["yolo", "detect", "train", f"imgsz={imgsz}"],
    )
    payload_path = payload.write(tmp_path / "payload.yaml")
    if not changed:
        raw = payload.model_dump(mode="json")
        raw["changed_variables"] = {}
        import yaml

        payload_path.write_text(
            yaml.safe_dump(raw, sort_keys=False),
            encoding="utf-8",
        )
    command = CommandSpec(
        command_type="train",
        command="python",
        argv=["python", "-m", "runtime", "--payload", str(payload_path), "--", "yolo", "detect", "train", f"imgsz={imgsz}"],
        args=[],
        metadata={
            "component_ids": "loss.quality.correlation",
            "adapter_patch_hash": "patch",
            "adapter_hashes": json.dumps({"loss.quality.correlation": "hash"}),
            "component_maturity": json.dumps({"loss.quality.correlation": "smoke_passed"}),
            "maturity_artifact_hashes": json.dumps({"loss.quality.correlation": ["artifact"]}),
            "adapter_runtime_payload_hash": payload.payload_hash,
            "adapter_runtime_payload_path": str(payload_path),
            "adapter_runtime_protocol_hash": payload.protocol_hash,
            "adapter_runtime_entrypoint": "runtime",
        },
    )
    return ExperimentNode(
        node_id="quality-node",
        candidate_config=CandidateConfig(
            candidate_id="quality",
            base_model="yolo26n.pt",
            scale="n",
            framework="ultralytics",
            components=["loss.quality.correlation"],
        ),
        data_version="fixture",
        command_spec=command,
    )


def test_quality_runtime_gate_rejects_missing_changed_variable(tmp_path) -> None:
    errors = validate_certified_runtime_node(_quality_node(tmp_path, changed=False))
    assert "adapter_changed_variable_missing" in errors


def test_quality_runtime_gate_rejects_non_640_imgsz(tmp_path) -> None:
    errors = validate_certified_runtime_node(_quality_node(tmp_path, imgsz=1280))
    assert "fixed_imgsz_must_equal_640" in errors
