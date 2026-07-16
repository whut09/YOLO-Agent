from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.recipe_ablation_planner import AblationObservation, RecipeAblationPlanner
from yolo_agent.recipes.schemas import CoupledRecipe


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
        AblationObservation(node_id=node.node_id, seed=2, deltas={"ap_small": 0.02}),
        AblationObservation(node_id=node.node_id, seed=3, deltas={"ap_small": 0.03}),
    ])
    assert confirmed[0].confidence == "confirmed"
    assert confirmed[0].seed_count == 3
    assert confirmed[0].mean_deltas["ap_small"] == 0.02


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
    assert result[0].reason == "repeated_seeds_but_inconsistent_direction"
