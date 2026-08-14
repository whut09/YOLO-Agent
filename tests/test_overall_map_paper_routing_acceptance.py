from __future__ import annotations

from pathlib import Path

from yolo_agent.agents.paper_recipe_planner import PaperRecipePlanner
from yolo_agent.components.contracts import load_contracts
from yolo_agent.components.registry import ComponentRegistry
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.core.experiment_graph import MetricEvidence
from yolo_agent.core.optimization_objective import OptimizationObjective
from yolo_agent.recipes.registry import RecipeRegistry
from yolo_agent.research.paper_registry import PaperRegistry
from yolo_agent.resources import ResourcePaths
from tests.maturity_helpers import with_smoke_artifact


TARGET_COMPONENTS = {
    "loss.hard_negative_classification",
    "sampling.hard_negative_replay",
    "loss.quality.correlation",
    "loss.quality.pseudo_iou",
    "assigner.task_aligned",
    "assigner.optimal_transport",
    "distillation.yolo26_teacher_student",
    "neck.rtmdet_large_kernel",
}


def test_overall_map_diagnosis_routes_all_runtime_ready_paper_methods(
    tmp_path: Path,
) -> None:
    contracts = {}
    for path in sorted(ResourcePaths.COMPONENTS_DIR.rglob("*.yaml")):
        try:
            loaded = load_contracts(path)
        except (KeyError, TypeError, ValueError):
            continue
        for contract in loaded:
            contracts[contract.component_id] = (
                with_smoke_artifact(contract)
                if contract.component_id in TARGET_COMPONENTS
                else contract
            )
    registry = RecipeRegistry.from_paths(
        [ResourcePaths.RECIPE_BUNDLES, *sorted(ResourcePaths.RECIPES_DIR.glob("*.yaml"))],
        component_contracts=contracts.values(),
        strict=False,
    )
    facts = [
        _fact("background_false_positive_class", "person"),
        _fact("localization_heavy_class", "overall"),
        _fact("class_confusion_pair", "person:bicycle"),
        _fact("class_low_ap", "long_tail_classes", severity="high"),
    ]
    metric = MetricEvidence(
        candidate_id="baseline",
        node_id="node-baseline",
        metric_name="map50_95",
        value=0.394,
        verified=True,
    )

    plan = PaperRecipePlanner().plan(
        error_facts=facts,
        dataset_report=None,
        node_metrics=[metric],
        policy_memory=[],
        paper_registry=PaperRegistry(tmp_path / "research"),
        component_registry=ComponentRegistry(contracts.values()),
        recipe_registry=registry,
        training_budget={
            "profile": "pilot",
            "fidelity": "pilot_3",
            "imgsz": 640,
            "dataset_signature": "coco2017",
            "protocol_hash": "protocol-640",
        },
        optimization_objective=OptimizationObjective(
            goal_description="Improve overall mAP",
            primary_metric="map50_95",
            baseline_run_id="improve-map-11",
            baseline_candidate_id="baseline",
            baseline_protocol_hash="protocol-640",
        ),
    )

    inventory_components = {
        component_id
        for planned in plan.candidate_inventory
        for recipe in [registry.get(planned.recipe_id, planned.version)]
        if recipe is not None
        for component_id in recipe.component_ids
    }
    assert TARGET_COMPONENTS.issubset(inventory_components)

    decisions = {
        planned.recipe_id: planned
        for planned in plan.candidate_inventory
        if (recipe := registry.get(planned.recipe_id, planned.version)) is not None
        and set(recipe.component_ids).issubset(TARGET_COMPONENTS)
    }
    assert decisions
    assert all(
        planned.decision != "implementation_proposal"
        for planned in decisions.values()
    )


def _fact(
    fact_type: str,
    subject: str,
    *,
    severity: str = "high",
) -> ErrorFact:
    return ErrorFact(
        run_id="improve-map-11",
        candidate_id="baseline",
        node_id="node-baseline",
        fact_type=fact_type,
        subject=subject,
        severity=severity,
    )
