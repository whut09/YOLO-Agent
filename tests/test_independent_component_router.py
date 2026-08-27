"""Independent YOLO26 paper-component routing tests. No GPU training."""

from __future__ import annotations

from pathlib import Path

import yaml

from yolo_agent.components.independent_component_router import (
    COMPONENT_CATALOG,
    GRAPH_IDENTITIES,
    INDEPENDENT_COMPONENT_IDS,
    QUALITY_PAIR,
    IndependentComponentRouter,
    default_independent_component_router,
)
from yolo_agent.recipes.paper_recipe_guards import inference_train_reasons
from yolo_agent.recipes.registry import RecipeRegistry
from yolo_agent.recipes.schemas import AtomicRecipe
from yolo_agent.research.component_aliases import ComponentAliasResolver


def test_router_covers_all_thirteen_identities() -> None:
    coverage = default_independent_component_router().coverage(
        has_shadow_evidence=True,
        has_payload=True,
        has_changed_variable=True,
        has_evidence=True,
        has_adapter_hash=True,
        paired_baseline=True,
        contract_can_execute=True,
    )
    assert coverage.components_total == 13
    assert {item.component_id for item in coverage.routes} == set(INDEPENDENT_COMPONENT_IDS)
    assert coverage.swallowed_identities == []


def test_sahi_is_inference_only_not_training() -> None:
    route = IndependentComponentRouter().route("inference.sahi_slicing", paired_baseline=True)
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
    assert head.disposition == "evidence_recovery"


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
    assert pyramid.disposition == "evidence_recovery"


def test_rtmdet_neck_keeps_its_own_graph_identity_and_queue() -> None:
    router = IndependentComponentRouter()
    rtmdet = router.route(
        "neck.rtmdet_large_kernel",
        has_payload=True,
        has_changed_variable=True,
        has_evidence=True,
        has_adapter_hash=True,
        paired_baseline=True,
        contract_can_execute=True,
    )
    assert rtmdet.asha_eligible is True
    assert rtmdet.queue_track == "training"
    assert rtmdet.disposition == "queued"
    assert rtmdet.recipe_id == "yolo26_rtmdet_large_kernel_neck"
    assert rtmdet.adapter_class == "RTMDetLargeKernelNeckAdapter"
    assert rtmdet.changed_variable == "model.neck_plugin"
    assert rtmdet.runtime_hook == "build_model"
    assert rtmdet.evidence_artifact == "neck_rtmdet_large_kernel_manifest.json"
    identities = {
        router.route("neck.gold_gather_distribute").graph_identity,
        router.route("neck.multi_scale_fusion").graph_identity,
        router.route("attention.spatial").graph_identity,
        router.route("feature_pyramid.multi_scale").graph_identity,
        rtmdet.graph_identity,
    }
    assert identities == {
        "neck.gold_gather_distribute",
        "neck.multi_scale_fusion",
        "neck.rtmdet_large_kernel",
        "attention.spatial",
        "feature_pyramid.multi_scale",
    }
    blocked = router.route("neck.rtmdet_large_kernel")
    assert blocked.asha_eligible is False
    assert blocked.queue_track == "blocked"
    assert blocked.disposition == "evidence_recovery"
    assert "runtime_payload_missing" in blocked.reason_codes


def test_quality_losses_remain_independent_candidates() -> None:
    router = IndependentComponentRouter()
    common = {
        "has_payload": True,
        "has_changed_variable": True,
        "has_evidence": True,
        "has_adapter_hash": True,
        "paired_baseline": True,
        "contract_can_execute": True,
    }
    correlation = router.route("loss.quality.correlation", **common)
    pseudo = router.route("loss.quality.pseudo_iou", **common)
    assert set(QUALITY_PAIR) == {correlation.component_id, pseudo.component_id}
    assert correlation.recipe_id != pseudo.recipe_id
    assert correlation.changed_variable != pseudo.changed_variable
    assert correlation.graph_identity != pseudo.graph_identity
    assert correlation.asha_eligible is True
    assert pseudo.asha_eligible is True


def test_assignment_requires_shadow_before_active_queue() -> None:
    router = IndependentComponentRouter()
    blocked = router.route("assigner.optimal_transport", has_shadow_evidence=False)
    ready = router.route(
        "assigner.optimal_transport",
        has_shadow_evidence=True,
        has_payload=True,
        has_changed_variable=True,
        has_evidence=True,
        has_adapter_hash=True,
        paired_baseline=True,
        contract_can_execute=True,
    )
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
    assert payload["components_total"] == 13


def test_cpu_audit_covers_all_components_and_fails_closed() -> None:
    coverage = IndependentComponentRouter().audit_coverage()
    assert coverage.components_total == 13
    assert {route.component_id for route in coverage.routes} == set(INDEPENDENT_COMPONENT_IDS)
    assert all(route.asha_eligible is False for route in coverage.routes)
    sahi = next(route for route in coverage.routes if route.component_id == "inference.sahi_slicing")
    assert sahi.queue_track == "inference"
    assert sahi.asha_eligible is False
    head = next(route for route in coverage.routes if route.component_id == "detection_head.task_aligned")
    pyramid = next(route for route in coverage.routes if route.component_id == "feature_pyramid.multi_scale")
    assert head.disposition == "evidence_recovery"
    assert pyramid.disposition == "evidence_recovery"
    for component_id in {
        "assigner.optimal_transport",
        "assigner.task_aligned",
        "assigner.dynamic_smooth_label",
    }:
        route = next(item for item in coverage.routes if item.component_id == component_id)
        assert "contract_execution_gate_not_satisfied" in route.reason_codes


def test_rtmdet_cpu_audit_emits_hashes_and_keeps_maturity_gate() -> None:
    coverage = IndependentComponentRouter().audit_coverage()
    rtmdet = next(
        route for route in coverage.routes if route.component_id == "neck.rtmdet_large_kernel"
    )
    assert rtmdet.adapter_source_sha256 and len(rtmdet.adapter_source_sha256) == 64
    assert rtmdet.runtime_payload_hash and len(rtmdet.runtime_payload_hash) == 64
    assert not any(code.startswith("runtime_probe_failed") for code in rtmdet.reason_codes)
    assert "runtime_payload_missing" not in rtmdet.reason_codes
    assert "changed_variable_missing" not in rtmdet.reason_codes
    assert "contract_execution_gate_not_satisfied" in rtmdet.reason_codes
    assert rtmdet.asha_eligible is False


def test_rtmdet_audit_fails_closed_when_adapter_probe_breaks(monkeypatch) -> None:
    import yolo_agent.components.independent_component_router as router_module

    class BrokenResolver:
        def __init__(self, real) -> None:
            self.contracts = dict(real.contracts)
            broken = self.contracts["neck.rtmdet_large_kernel"].model_copy(
                update={"implementation_path": "yolo_agent.components.adapters.neck.missing_rtmdet"}
            )
            self.contracts["neck.rtmdet_large_kernel"] = broken

        @classmethod
        def from_yaml(cls) -> "BrokenResolver":
            return cls(ComponentAliasResolver.from_yaml())

    monkeypatch.setattr(router_module, "ComponentAliasResolver", BrokenResolver)
    route = IndependentComponentRouter().audit("neck.rtmdet_large_kernel")
    assert route.asha_eligible is False
    assert route.queue_track == "blocked"
    assert route.disposition == "implementation_request"
    assert any(code.startswith("runtime_probe_failed") for code in route.reason_codes)


def test_contract_gate_is_required_even_when_payload_evidence_is_present() -> None:
    route = IndependentComponentRouter().route(
        "loss.quality.correlation",
        has_payload=True,
        has_changed_variable=True,
        has_evidence=True,
        has_adapter_hash=True,
        paired_baseline=True,
        contract_can_execute=False,
    )
    assert route.asha_eligible is False
    assert route.disposition == "evidence_recovery"
    assert "contract_execution_gate_not_satisfied" in route.reason_codes


def test_recipe_binding_keeps_independent_identities() -> None:
    router = IndependentComponentRouter()
    assert router.recipe_binding_reasons(
        "yolo26_correlation_auxiliary_loss",
        ["loss.quality.correlation"],
    ) == []
    assert router.recipe_binding_reasons(
        "yolo26_pseudo_iou_quality_auxiliary_loss",
        ["loss.quality.correlation"],
    )
    assert router.recipe_binding_reasons(
        "yolo26_tood_tal_assignment_shadow",
        ["detection_head.task_aligned"],
    )


def test_all_independent_components_have_registered_recipe_entries() -> None:
    registry = RecipeRegistry.from_paths(sorted(Path("configs/recipes").glob("*.yaml")))
    recipe_ids = {recipe.recipe_id for recipe in registry.list()}
    for component_id in INDEPENDENT_COMPONENT_IDS:
        expected = str(COMPONENT_CATALOG[component_id]["recipe_id"])
        assert expected in recipe_ids, (component_id, expected)


def test_routes_expose_all_runtime_boundary_fields() -> None:
    coverage = IndependentComponentRouter().coverage(
        has_payload=True,
        has_changed_variable=True,
        has_evidence=True,
        has_adapter_hash=True,
        paired_baseline=True,
        contract_can_execute=True,
        has_shadow_evidence=True,
    )
    for route in coverage.routes:
        assert route.implementation_path
        assert route.adapter_class
        assert route.changed_variable
        assert route.runtime_hook
        assert route.runtime_payload_field
        assert route.evidence_artifact
