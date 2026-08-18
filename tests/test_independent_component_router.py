"""Independent YOLO26 paper-component routing tests. No GPU training."""

from __future__ import annotations

from pathlib import Path

import yaml

from yolo_agent.components.independent_component_router import (
    GRAPH_IDENTITIES,
    INDEPENDENT_COMPONENT_IDS,
    QUALITY_PAIR,
    IndependentComponentRouter,
    default_independent_component_router,
)
from yolo_agent.recipes.paper_recipe_guards import inference_train_reasons
from yolo_agent.recipes.schemas import AtomicRecipe


def test_router_covers_all_twelve_identities() -> None:
    coverage = default_independent_component_router().coverage(
        has_shadow_evidence=True,
        has_payload=True,
        has_changed_variable=True,
        has_evidence=True,
        has_adapter_hash=True,
    )
    assert coverage.components_total == 12
    assert {item.component_id for item in coverage.routes} == set(INDEPENDENT_COMPONENT_IDS)
    assert coverage.swallowed_identities == []


def test_sahi_is_inference_only_not_training() -> None:
    route = IndependentComponentRouter().route("inference.sahi_slicing")
    assert route.inference_only is True
    assert route.queue_track == "inference"
    assert route.asha_eligible is False
    assert "inference_only_not_training_candidate" in route.reason_codes
    recipe = AtomicRecipe.model_validate(
        {
            "recipe_id": "sahi_slicing_inference",
            "version": "v1.0.0",
            "component_ids": ["inference.sahi_slicing"],
            "inference_actions": ["sahi_slicing"],
            "primary_changed_variable": "inference_policy",
        }
    )
    assert inference_train_reasons(recipe) == ["inference_only_not_training_candidate"]


def test_task_aligned_head_is_not_swallowed_by_assigner() -> None:
    router = IndependentComponentRouter()
    head = router.route("detection_head.task_aligned")
    assigner = router.route("assigner.task_aligned", has_shadow_evidence=True)
    assert head.graph_identity == "detection_head.task_aligned"
    assert assigner.graph_identity == "assigner.task_aligned"
    assert head.graph_identity != assigner.graph_identity
    assert head.disposition == "implementation_request"


def test_neck_and_pyramid_graph_identities_stay_distinct() -> None:
    router = IndependentComponentRouter()
    identities = {
        router.route("neck.gold_gather_distribute").graph_identity,
        router.route("neck.multi_scale_fusion").graph_identity,
        router.route("feature_pyramid.multi_scale").graph_identity,
    }
    assert identities == {
        "neck.gold_gather_distribute",
        "neck.multi_scale_fusion",
        "feature_pyramid.multi_scale",
    }
    pyramid = router.route("feature_pyramid.multi_scale")
    assert pyramid.disposition == "implementation_request"


def test_quality_losses_remain_independent_candidates() -> None:
    router = IndependentComponentRouter()
    correlation = router.route("loss.quality.correlation")
    pseudo = router.route("loss.quality.pseudo_iou")
    assert set(QUALITY_PAIR) == {correlation.component_id, pseudo.component_id}
    assert correlation.recipe_id != pseudo.recipe_id
    assert correlation.changed_variable != pseudo.changed_variable
    assert correlation.graph_identity != pseudo.graph_identity
    assert correlation.asha_eligible is True
    assert pseudo.asha_eligible is True


def test_assignment_requires_shadow_before_active_queue() -> None:
    router = IndependentComponentRouter()
    blocked = router.route("assigner.optimal_transport", has_shadow_evidence=False)
    ready = router.route("assigner.optimal_transport", has_shadow_evidence=True)
    assert blocked.disposition == "evidence_recovery"
    assert "assignment_shadow_evidence_required" in blocked.reason_codes
    assert blocked.asha_eligible is False
    assert ready.asha_eligible is True
    assert ready.requires_shadow_evidence is True


def test_missing_payload_changed_variable_or_evidence_cannot_queue() -> None:
    router = IndependentComponentRouter()
    for kwargs, code in (
        ({"has_payload": False}, "runtime_payload_missing"),
        ({"has_changed_variable": False}, "changed_variable_missing"),
        ({"has_evidence": False}, "evidence_artifact_missing"),
        ({"has_adapter_hash": False}, "adapter_hash_missing"),
    ):
        route = router.route("loss.calibration.bpc", **kwargs)
        assert route.asha_eligible is False
        assert route.queue_track == "blocked"
        assert code in route.reason_codes


def test_catalog_graph_identities_are_stable() -> None:
    assert GRAPH_IDENTITIES["detection_head.task_aligned"] != GRAPH_IDENTITIES["assigner.task_aligned"]
    fixture = Path("tests/fixtures/independent_component_routes.yaml")
    payload = yaml.safe_load(fixture.read_text(encoding="utf-8"))
    assert payload["components_total"] == 12
