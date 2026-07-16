from pathlib import Path

from yolo_agent.agents.paper_recipe_planner import PaperRecipePlanner
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.registry import ComponentRegistry
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.core.experiment_graph import MetricEvidence
from yolo_agent.core.policy_memory import PolicyMemoryRecord
from yolo_agent.core.schemas import DeploymentConstraints
from yolo_agent.recipes.registry import RecipeRegistry
from yolo_agent.recipes.schemas import AtomicRecipe
from yolo_agent.research.paper_registry import PaperRegistry
from yolo_agent.research.schemas import PaperComponentClaim, PaperRecord


def _fact() -> ErrorFact:
    return ErrorFact(run_id="r", candidate_id="base", node_id="n", fact_type="area_metric", subject="small", area="small", metric_name="ap_small", value=0.2, severity="high", action_candidates=["small_object_recipe"])


def _recipe(recipe_id="small_object_recipe", **updates):
    data = {"recipe_id": recipe_id, "version": "v1.0.0", "primary_changed_variable": "data_sampling", "fixed_variables": {"imgsz": 640}, "train_overrides": {"imgsz": 640}, "target_error_facts": [{"area": "small"}], "target_metrics": ["ap_small"], "expected_effects": {"ap_small": 0.1}, "implementation_risk": "low"}
    data.update(updates)
    return AtomicRecipe.model_validate(data)


def _paper() -> PaperRecord:
    return PaperRecord(paper_id="p1", title="Small object sampling", year=2024, component_ids=["sampling.small_object"], claimed_effects=[PaperComponentClaim(component_id="sampling.small_object", component_category="sampling", claimed_effect="improves small AP", evidence_level="paper_claim", target_metrics=["ap_small"], target_error_types=["area_metric"])])


def _planner(tmp_path: Path, recipes: list[AtomicRecipe], contracts: list[ComponentContract] = ()):
    papers = PaperRegistry(tmp_path / "research")
    papers.add(_paper())
    return PaperRecipePlanner(), papers, ComponentRegistry(list(contracts)), RecipeRegistry(recipes, contracts)


def test_no_error_facts_only_requests_evidence(tmp_path: Path) -> None:
    planner, papers, components, recipes = _planner(tmp_path, [_recipe()])
    plan = planner.plan(error_facts=[], dataset_report=None, node_metrics=[], policy_memory=[], paper_registry=papers, component_registry=components, recipe_registry=recipes)
    assert not plan.selected_recipes and not plan.rejected_recipes
    assert "mine_coco_error_facts" in plan.evidence_actions


def test_metadata_recipe_becomes_implementation_proposal(tmp_path: Path) -> None:
    contract = ComponentContract(component_id="head.paper", display_name="Paper", category="head")
    recipe = _recipe(component_ids=[contract.component_id], primary_changed_variable="head")
    planner, papers, components, recipes = _planner(tmp_path, [recipe], [contract])
    plan = planner.plan(error_facts=[_fact()], dataset_report=None, node_metrics=[], policy_memory=[], paper_registry=papers, component_registry=components, recipe_registry=recipes)
    assert plan.deferred_recipes[0].decision == "implementation_proposal"


def test_executable_recipe_is_selected_as_pilot_not_full(tmp_path: Path) -> None:
    contract = ComponentContract(component_id="sampling.small_object", display_name="Sampling", category="sampling", implementation_path="x", adapter_class="A", maturity="smoke_passed")
    recipe = _recipe(component_ids=[contract.component_id], primary_changed_variable="sampling")
    planner, papers, components, recipes = _planner(tmp_path, [recipe], [contract])
    metric = MetricEvidence(candidate_id="base", node_id="n", metric_name="map50_95", value=0.3, verified=True)
    plan = planner.plan(error_facts=[_fact()], dataset_report=None, node_metrics=[metric], policy_memory=[], paper_registry=papers, component_registry=components, recipe_registry=recipes, training_budget={"profile": "candidate_full"})
    assert plan.selected_recipes[0].recipe_id == recipe.recipe_id and plan.training_profile == "pilot" and plan.fixed_imgsz == 640


def test_single_failed_pilot_does_not_permanently_reject_family(tmp_path: Path) -> None:
    recipe = _recipe()
    planner, papers, components, recipes = _planner(tmp_path, [recipe])
    memory = [PolicyMemoryRecord(run_id="r", action="small_object_recipe", target="ap_small", effect_delta=-0.1, confidence="high")]
    plan = planner.plan(error_facts=[_fact()], dataset_report=None, node_metrics=[], policy_memory=memory, paper_registry=papers, component_registry=components, recipe_registry=recipes)
    assert not plan.rejected_recipes
    assert plan.selected_recipes or plan.deferred_recipes
    considered = [*plan.selected_recipes, *plan.deferred_recipes][0]
    no_prior = planner.plan(
        error_facts=[_fact()],
        dataset_report=None,
        node_metrics=[],
        policy_memory=[],
        paper_registry=papers,
        component_registry=components,
        recipe_registry=recipes,
    )
    no_prior_considered = [*no_prior.selected_recipes, *no_prior.deferred_recipes][0]
    assert considered.confidence < no_prior_considered.confidence


def test_repeated_stable_negative_memory_rejects_recipe(tmp_path: Path) -> None:
    recipe = _recipe()
    planner, papers, components, recipes = _planner(tmp_path, [recipe])
    memory = [
        PolicyMemoryRecord(
            run_id=f"r{index}",
            action="small_object_recipe",
            target="ap_small",
            metric_name="ap_small",
            effect_delta=value,
        )
        for index, value in enumerate([-0.10, -0.09, -0.11], start=1)
    ]
    plan = planner.plan(
        error_facts=[_fact()],
        dataset_report=None,
        node_metrics=[],
        policy_memory=memory,
        paper_registry=papers,
        component_registry=components,
        recipe_registry=recipes,
    )
    assert plan.rejected_recipes[0].decision == "rejected"
    assert "stable_historical_no_gain" in plan.rejected_recipes[0].reasons[0]


def test_incompatible_recipe_is_rejected(tmp_path: Path) -> None:
    contract = ComponentContract(component_id="head.nms", display_name="NMS", category="nms", implementation_path="x", adapter_class="A", maturity="smoke_passed")
    recipe = _recipe(component_ids=[contract.component_id], primary_changed_variable="head", train_overrides={"imgsz": 640, "postprocess": "soft_nms"})
    planner, papers, components, recipes = _planner(tmp_path, [recipe], [contract])
    plan = planner.plan(error_facts=[_fact()], dataset_report=None, node_metrics=[], policy_memory=[], paper_registry=papers, component_registry=components, recipe_registry=recipes)
    assert plan.rejected_recipes[0].decision == "rejected"


def test_deployment_constraints_are_passed_to_planner(tmp_path: Path) -> None:
    planner, papers, components, recipes = _planner(tmp_path, [_recipe()])
    plan = planner.plan(error_facts=[_fact()], dataset_report=None, node_metrics=[], policy_memory=[], paper_registry=papers, component_registry=components, recipe_registry=recipes, deployment=DeploymentConstraints(max_latency_ms=10))
    assert plan.fixed_imgsz == 640
