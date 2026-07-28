from pathlib import Path

import yaml
import pytest

from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.recipe_ablation_planner import AblationObservation, RecipeAblationPlanner
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.experiment_graph import ExperimentNode, MetricEvidence
from yolo_agent.recipes.schemas import CoupledRecipe, recipe_from_mapping


def _baseline() -> CandidateConfig:
    return CandidateConfig(candidate_id="baseline", base_model="yolo26n.pt", scale="n", framework="ultralytics", train_overrides={"imgsz": 640})


def _recipe(components) -> CoupledRecipe:
    variables = [f"component_{index}" for index in range(len(components))]
    return CoupledRecipe.model_validate({
        "recipe_id": "coupled_small", "version": "v1", "primary_changed_variable": variables[0],
        "component_ids": components, "target_error_facts": [{"fact_type": "area_metric", "area": "small"}],
        "target_metrics": ["ap_small", "latency_ms", "model_size_mb"],
        "fixed_variables": {"imgsz": 640}, "train_overrides": {"imgsz": 640},
        "coupled_variables": variables, "coupling_reason": "Components jointly implement the paper recipe.",
        "coupling_source_papers": ["paper:x"],
        "internal_ablation_plan": [{"name": "single_and_full"}],
        "stop_conditions": ["pilot_no_gain", "latency_regressed", "model_size_regressed"],
        "promotion_requirements": ["latency_guard", "model_size_guard"],
    })


def _small_object_recipe() -> CoupledRecipe:
    raw = yaml.safe_load(Path("configs/recipes/yolo26_small_object.yaml").read_text(encoding="utf-8"))
    recipe = recipe_from_mapping(raw["recipes"][2])
    assert isinstance(recipe, CoupledRecipe)
    return recipe


def _experiment(candidate: CandidateConfig, changed_variables: dict) -> ExperimentNode:
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data="coco.yaml",
        project="runs/ablation",
        name=candidate.candidate_id,
        epochs=3,
        imgsz=640,
    )
    return ExperimentNode(
        node_id=f"node_{candidate.candidate_id}",
        candidate_config=candidate,
        data_version="coco2017",
        seed=1,
        command=command.display(),
        command_spec=command,
        changed_variables=changed_variables,
    )


def test_two_component_recipe_generates_baseline_singles_and_full() -> None:
    plan = RecipeAblationPlanner().plan(_recipe(["component.a", "component.b"]), _baseline(), max_nodes=4)
    assert [node.role for node in plan.nodes] == ["baseline", "single", "single", "full"]
    assert [node.component_ids for node in plan.nodes] == [[], ["component.a"], ["component.b"], ["component.a", "component.b"]]
    assert len(plan.single_variable_plan.nodes) == 2
    assert plan.successive_halving is not None


def test_three_component_recipe_supports_full_matrix() -> None:
    plan = RecipeAblationPlanner().plan(_recipe(["component.a", "component.b", "component.c"]), _baseline(), max_nodes=8)
    assert len(plan.nodes) == 8
    assert sum(node.role == "pair" for node in plan.nodes) == 3
    assert plan.omitted_combinations == []
    assert plan.nodes[-1].component_ids == ["component.a", "component.b", "component.c"]


def test_budget_prunes_pairs_but_keeps_singles_and_full_recipe() -> None:
    plan = RecipeAblationPlanner().plan(_recipe(["component.a", "component.b", "component.c"]), _baseline(), max_nodes=6)
    assert len(plan.nodes) == 6
    assert sum(node.role == "single" for node in plan.nodes) == 3
    assert sum(node.role == "pair" for node in plan.nodes) == 1
    assert plan.nodes[-1].role == "full"
    assert len(plan.omitted_combinations) == 2
    assert plan.budget_report is not None and plan.budget_report.selected_count == 1


def test_budget_cannot_remove_mandatory_atomic_and_full_nodes() -> None:
    try:
        RecipeAblationPlanner().plan(_recipe(["component.a", "component.b", "component.c"]), _baseline(), max_nodes=4)
    except ValueError as exc:
        assert "at least 5 nodes" in str(exc)
    else:
        raise AssertionError("insufficient ablation budget should fail")


def test_contribution_requires_repeated_seeds_for_confirmation() -> None:
    planner = RecipeAblationPlanner()
    plan = planner.plan(_recipe(["component.a", "component.b"]), _baseline(), max_nodes=4)
    node = next(item for item in plan.nodes if item.role == "single")
    possible = planner.assess_contributions(plan, [AblationObservation(node_id=node.node_id, seed=1, deltas={"ap_small": 0.01})])
    assert possible[0].confidence == "possible"
    confirmed = planner.assess_contributions(plan, [
        AblationObservation(node_id=node.node_id, seed=1, deltas={"ap_small": 0.01}),
        AblationObservation(node_id=node.node_id, seed=2, deltas={"ap_small": 0.011}),
        AblationObservation(node_id=node.node_id, seed=3, deltas={"ap_small": 0.012}),
    ])
    assert confirmed[0].confidence == "confirmed"
    assert confirmed[0].seed_count == 3
    assert confirmed[0].mean_deltas["ap_small"] == pytest.approx(0.011)
    assert confirmed[0].confidence_interval_low > 0


def test_repeated_seeds_with_conflicting_direction_remain_possible() -> None:
    planner = RecipeAblationPlanner()
    plan = planner.plan(_recipe(["component.a", "component.b"]), _baseline(), max_nodes=4)
    node = next(item for item in plan.nodes if item.role == "single")
    result = planner.assess_contributions(plan, [
        AblationObservation(node_id=node.node_id, seed=1, deltas={"ap_small": 0.01}),
        AblationObservation(node_id=node.node_id, seed=2, deltas={"ap_small": -0.02}),
        AblationObservation(node_id=node.node_id, seed=3, deltas={"ap_small": 0.03}),
    ])
    assert result[0].confidence == "possible"
    assert result[0].reason == "paired_seed_confidence_interval_not_strictly_positive"


def test_small_object_coupled_recipe_materializes_guarded_paired_round() -> None:
    recipe = _small_object_recipe()
    baseline = _baseline()
    planner = RecipeAblationPlanner()
    ablation = planner.plan(recipe, baseline, max_nodes=4)
    prepared = {
        item.candidate_config.candidate_id: _experiment(
            item.candidate_config,
            item.changed_variables,
        )
        for item in ablation.nodes
        if item.role != "baseline"
    }
    round_plan = planner.materialize_round_execution_plan(
        run_id="small-object-round",
        recipe=recipe,
        ablation_plan=ablation,
        baseline_control_node=_experiment(baseline, {}),
        prepared_nodes=prepared,
    )

    assert [item.role for item in round_plan.ablation_nodes] == [
        "baseline",
        "single",
        "single",
        "full",
    ]
    assert [item.component_ids for item in round_plan.ablation_nodes] == [
        [],
        ["sampling.small_object"],
        ["head.p2_small_object"],
        ["sampling.small_object", "head.p2_small_object"],
    ]
    assert round_plan.require_complete_post_eval is True
    assert len(round_plan.execution_nodes) == 4
    control = next(item for item in round_plan.assignments if item.role == "baseline_control")
    candidates = [item for item in round_plan.assignments if item.role == "candidate"]
    assert all(item.matched_control_execution_node_id == control.execution_node_id for item in candidates)
    assert all(
        round_plan.evidence_requirements[item.execution_node_id][-1] == "verified_paired_delta"
        for item in round_plan.assignments
    )
    sampling = next(
        item
        for item in round_plan.execution_nodes
        if item.candidate_config.components == ["sampling.small_object"]
    )
    p2 = next(
        item
        for item in round_plan.execution_nodes
        if item.candidate_config.components == ["head.p2_small_object"]
    )
    combined = next(
        item
        for item in round_plan.execution_nodes
        if len(item.candidate_config.components) == 2
    )
    assert sampling.command_spec.metadata["attribution_excluded_metrics"] == (
        "latency_ms,throughput,model_size_mb"
    )
    assert p2.command_spec.metadata["guard_metrics"] == "latency_ms,model_size_mb"
    assert len(combined.changed_variables) == 2


def test_coupled_round_refuses_promotion_until_every_node_finishes_post_eval() -> None:
    recipe = _small_object_recipe()
    baseline = _baseline()
    planner = RecipeAblationPlanner()
    ablation = planner.plan(recipe, baseline, max_nodes=4)
    prepared = {
        item.candidate_config.candidate_id: _experiment(item.candidate_config, item.changed_variables)
        for item in ablation.nodes
        if item.role != "baseline"
    }
    round_plan = planner.materialize_round_execution_plan(
        run_id="small-object-round",
        recipe=recipe,
        ablation_plan=ablation,
        baseline_control_node=_experiment(baseline, {}),
        prepared_nodes=prepared,
    )

    assert round_plan.reconcile([]) is False
    assert "complete COCO post-eval required" in round_plan.blocked_reason
    completions = [
        MetricEvidence(
            run_id="small-object-round",
            origin_run_id="small-object-round",
            candidate_id=assignment.candidate_id,
            node_id=assignment.execution_node_id,
            metric_name="coco_post_eval_complete",
            value=True,
            verified=True,
            evidence_role=(
                "baseline_reference"
                if assignment.role == "baseline_control"
                else "current_observation"
            ),
        )
        for assignment in round_plan.assignments
    ]
    assert round_plan.reconcile(completions) is False
    assert "needs matched baseline ap_small" in round_plan.blocked_reason
