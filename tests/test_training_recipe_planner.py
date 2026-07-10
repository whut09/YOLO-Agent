"""Evidence-driven training recipe planner tests."""

from pathlib import Path

from yolo_agent.agents.training_recipe_planner import (
    TrainingRecipe,
    TrainingRecipeCatalog,
    TrainingRecipePlanner,
    TrainingRecipeVariant,
)
from yolo_agent.core.experiment_graph import Evidence, MetricEvidence
from yolo_agent.core.run_context import RunContext


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        run_id="recipe-test",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "data.yaml",
        metadata={"training_model": "yolo26n.pt"},
    )


def _catalog() -> TrainingRecipeCatalog:
    return TrainingRecipeCatalog(
        max_recipes_per_round=1,
        recipes=[
            TrainingRecipe(
                family="optimizer",
                action_domain="training",
                trigger_actions=["increase_recall_recipe"],
                target_fact_types=["false_negative_heavy_class"],
                effect="Test optimizer choice.",
                stop_after_non_positive=2,
                variants=[
                    TrainingRecipeVariant(action_id="optimizer_adamw", overrides={"optimizer": "AdamW"}),
                    TrainingRecipeVariant(action_id="optimizer_sgd", overrides={"optimizer": "SGD"}),
                ],
            )
        ],
    )


def _focus() -> list[dict[str, object]]:
    return [{
        "fact_type": "false_negative_heavy_class",
        "class_name": "person",
        "action_candidates": ["increase_recall_recipe"],
    }]


def test_planner_selects_next_untried_single_variable_recipe(tmp_path: Path) -> None:
    plan = TrainingRecipePlanner(_catalog()).plan(
        context=_context(tmp_path),
        evidence=Evidence(run_id="recipe-test"),
        focus_items=_focus(),
        allowed_actions={"increase_recall_recipe"},
        tried_actions={"optimizer_adamw"},
    )

    assert [policy.action_id for policy in plan.policies] == ["optimizer_sgd"]
    policy = plan.policies[0]
    assert policy.train_overrides["optimizer"] == "SGD"
    assert "increase_recall_recipe" in policy.train_overrides["target_actions"]
    assert "imgsz" not in policy.train_overrides


def test_planner_rejects_family_after_two_non_positive_pilots(tmp_path: Path) -> None:
    evidence = Evidence(
        run_id="recipe-test",
        metric_records=[
            MetricEvidence(candidate_id="yolo26n_coco_pilot", node_id="baseline", metric_name="map50_95", value=0.40),
            MetricEvidence(candidate_id="next_optimizer_adamw", node_id="adamw", metric_name="map50_95", value=0.399),
            MetricEvidence(candidate_id="next_optimizer_sgd", node_id="sgd", metric_name="map50_95", value=0.40),
        ],
    )
    plan = TrainingRecipePlanner(_catalog()).plan(
        context=_context(tmp_path),
        evidence=evidence,
        focus_items=_focus(),
        allowed_actions={"increase_recall_recipe"},
        tried_actions={"optimizer_adamw", "optimizer_sgd"},
    )

    assert plan.policies == []
    assert plan.family_decisions[0].decision == "rejected_by_evidence"
