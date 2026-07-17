"""Executable contract tests for the first high-value YOLO26 recipes."""

from __future__ import annotations

from pathlib import Path

import yaml

from yolo_agent.agents.auto_optimization_loop import _register_guarded_pilot_trials
from yolo_agent.agents.recipe_ablation_planner import RecipeAblationPlanner
from yolo_agent.agents.asha_scheduler import ASHAScheduler
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.components.adapters.base import AdapterContext
from yolo_agent.components.adapters.distillation.yolo26_distillation import YOLO26DistillationAdapter
from yolo_agent.components.adapters.head.p2_head import P2HeadAdapter
from yolo_agent.components.adapters.sampling.small_object_sampling import SmallObjectSamplingAdapter
from yolo_agent.components.contracts import ComponentContract, load_contracts
from yolo_agent.components.execution_bridge import ComponentExecutionBridge
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.recipes.schemas import CoupledRecipe, RecipeSpec, recipe_from_mapping


def _recipe_records(path: str) -> list[RecipeSpec]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    values = raw.get("recipes", [raw])
    return [recipe_from_mapping(item) for item in values]


def _node(recipe: RecipeSpec, tmp_path: Path) -> ExperimentNode:
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
        model="yolo26n.pt", data=tmp_path / "coco.yaml", project=tmp_path / "runs",
        name=recipe.recipe_id, epochs=3, imgsz=640,
    )
    return ExperimentNode(
        node_id=f"node_{recipe.recipe_id}", candidate_config=candidate,
        data_version="coco2017", seed=1, command=command.display(), command_spec=command,
    )


def _context(contract: ComponentContract, tmp_path: Path, options: dict) -> AdapterContext:
    return AdapterContext(
        contract=contract, detector_family="yolo26", head="one_to_one",
        imgsz=640, workspace=tmp_path, options=options,
    )


def test_atomic_recipes_are_smoke_passed_fixed_640_and_guarded() -> None:
    recipes = [
        *_recipe_records("configs/recipes/yolo26_small_object.yaml")[:2],
        *_recipe_records("configs/recipes/yolo26n_distillation.yaml"),
    ]
    assert {item.recipe_id for item in recipes} == {
        "yolo26_small_object_p2", "yolo26_small_object_sampling", "yolo26n_distillation"
    }
    for recipe in recipes:
        assert recipe.maturity == "smoke_passed" and recipe.is_executable
        assert recipe.train_overrides["imgsz"] == 640
        assert recipe.fixed_variables["imgsz"] == 640
        assert any("latency" in item for item in [*recipe.stop_conditions, *recipe.promotion_requirements])
        assert any("model_size" in item for item in [*recipe.stop_conditions, *recipe.promotion_requirements])
        assert "matched_pilot" in recipe.promotion_requirements


def test_all_three_adapters_pass_shape_backward_amp_smoke(tmp_path: Path) -> None:
    p2 = load_contracts("configs/components/head/yolo26_p2_small_object.yaml")[0]
    sampler = load_contracts("configs/components/sampling/small_object_sampling.yaml")[0]
    distill = load_contracts("configs/components/distillation/yolo26_teacher_student.yaml")[0]
    results = [
        P2HeadAdapter().smoke_test(_context(p2, tmp_path, {"imgsz": 640})),
        SmallObjectSamplingAdapter().smoke_test(_context(sampler, tmp_path, {"imgsz": 640})),
        YOLO26DistillationAdapter().smoke_test(_context(distill, tmp_path, {
            "teacher": "yolo26s.pt", "student": "yolo26n.pt",
            "teacher_data": "coco.yaml", "student_data": "coco.yaml", "imgsz": 640,
        })),
    ]
    for result in results:
        assert result.passed, result.errors
        assert result.checks["backward"] is True
        assert result.checks["amp"] is True
        assert "shape" in result.checks


def test_component_bridge_prepares_each_atomic_recipe(tmp_path: Path) -> None:
    contracts = {
        item.component_id: item
        for path in [
            "configs/components/head/yolo26_p2_small_object.yaml",
            "configs/components/sampling/small_object_sampling.yaml",
            "configs/components/distillation/yolo26_teacher_student.yaml",
        ]
        for item in load_contracts(path)
    }
    recipes = [
        *_recipe_records("configs/recipes/yolo26_small_object.yaml")[:2],
        *_recipe_records("configs/recipes/yolo26n_distillation.yaml"),
    ]
    for recipe in recipes:
        result = ComponentExecutionBridge().prepare(
            recipe=recipe, node=_node(recipe, tmp_path), contracts=contracts,
            workspace=tmp_path / recipe.recipe_id,
        )
        assert result.status == "executable", result.blocked_by
        assert result.node.command_spec is not None
        assert result.node.command_spec.metadata["matched_pilot_required"] is True
        assert result.node.command_spec.metadata["adapter_guard_metrics"] == "latency_ms,model_size_mb"


def test_p2_sampler_is_only_coupled_and_declares_baseline_a_b_a_plus_b() -> None:
    recipe = _recipe_records("configs/recipes/yolo26_small_object.yaml")[2]
    assert isinstance(recipe, CoupledRecipe)
    assert recipe.component_ids == ["head.p2_small_object", "sampling.small_object"]
    assert [item["name"] for item in recipe.internal_ablation_plan] == [
        "baseline", "p2_only", "sampling_only", "p2_plus_sampling"
    ]
    baseline = CandidateConfig(
        candidate_id="baseline", base_model="yolo26n.pt", scale="n", framework="ultralytics"
    )
    plan = RecipeAblationPlanner().plan(recipe, baseline, max_nodes=4)
    assert [item.role for item in plan.nodes] == ["baseline", "single", "single", "full"]
    assert [item.component_ids for item in plan.nodes] == [
        [], ["head.p2_small_object"], ["sampling.small_object"],
        ["head.p2_small_object", "sampling.small_object"],
    ]


def test_component_recipe_requires_matched_control_before_asha_registration(tmp_path: Path) -> None:
    recipe = _recipe_records("configs/recipes/yolo26_small_object.yaml")[1]
    node = _node(recipe, tmp_path)
    node.command_spec = node.command_spec.model_copy(
        update={"metadata": {"matched_pilot_required": True}}
    )
    scheduler = ASHAScheduler.create("base")

    class Context:
        def artifact_path(self, name: str) -> Path:
            return tmp_path / name

    class Child:
        context = Context()

    from yolo_agent.core.round_execution_plan import build_asha_assignment_plan
    plan = build_asha_assignment_plan(
        run_id="child", source_node=node, stage_id="pilot_3",
        epochs=3, fraction=0.1, seed=1,
    )
    plan.to_yaml(tmp_path / "round_execution_plan.yaml")
    assert _register_guarded_pilot_trials(scheduler, Child(), plan.execution_nodes) == 0  # type: ignore[arg-type]
