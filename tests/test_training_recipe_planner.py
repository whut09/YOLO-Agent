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
        enable_scalar_hpo=True,
        max_scalar_hpo_per_round=1,
        max_scalar_hpo_per_run=2,
        recipes=[
            TrainingRecipe(
                family="optimizer",
                search_tier="scalar_hpo",
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


def test_default_catalog_prioritizes_methods_and_excludes_scalar_hpo_from_fn_cohort(tmp_path: Path) -> None:
    plan = TrainingRecipePlanner().plan(
        context=_context(tmp_path),
        evidence=Evidence(run_id="recipe-test"),
        focus_items=_focus(),
        allowed_actions={"increase_recall_recipe", "class_balanced_sampling", "light_mixup"},
        tried_actions=set(),
    )

    assert [policy.action_id for policy in plan.policies] == [
        "scale_aug_0_3",
        "copy_paste_0_1",
        "mixup_0_05",
    ]
    assert not any(
        key in policy.train_overrides
        for policy in plan.policies
        for key in ("optimizer", "lr0", "weight_decay")
    )


def test_default_catalog_stops_instead_of_falling_back_to_scalar_hpo(tmp_path: Path) -> None:
    plan = TrainingRecipePlanner().plan(
        context=_context(tmp_path),
        evidence=Evidence(run_id="recipe-test"),
        focus_items=_focus(),
        allowed_actions={"increase_recall_recipe", "class_balanced_sampling", "light_mixup"},
        tried_actions={
            "scale_aug_0_3",
            "scale_aug_0_7",
            "copy_paste_0_1",
            "copy_paste_0_2",
            "mixup_0_05",
            "mixup_0_1",
        },
    )

    assert plan.policies == []
    scalar_decisions = [
        item for item in plan.family_decisions
        if item.family in {"optimizer", "learning_rate", "weight_decay"}
    ]
    assert scalar_decisions
    assert all("disabled" in item.reason for item in scalar_decisions)
