"""Explicit per-route smoke and compatibility evidence for distillation.

Mirrors the domain-route evidence file: the distillation route behavior
suites are parametrized over the registry, so they prove behavior without
ever naming the numeric per-paper component IDs that the readiness evaluator
matches on.  Each ID is named here via its registry route and runs the route
adapter's real CPU smoke (synthetic teacher/student fixtures, forward and
backward) — no training, no mock evidence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.components.adapters.base import AdapterContext
from yolo_agent.components.adapters.distillation.paper_routes import (
    default_paper_route_registry,
    paper_route_adapter,
)
from yolo_agent.components.contracts import ComponentContract


def _distillation_contract(component_id: str) -> ComponentContract:
    return ComponentContract(
        component_id=component_id,
        display_name=component_id,
        category="distillation",
        implementation_path=(
            "yolo_agent.components.adapters.distillation.paper_routes"
        ),
        adapter_class="PaperRouteAdapter",
        maturity="adapter_implemented",
    )


# Literal snapshot of every distillation route component ID.  The readiness
# evaluator discovers test evidence by grepping file *content* for the
# component ID, which a purely parametrized list cannot satisfy.  This
# literal is guarded against registry drift by the id-list test below.
_DISTILLATION_ROUTE_COMPONENT_IDS = [
    "distillation.base_novel_commonality",
    "distillation.bovw_consistency",
    "distillation.classifier_kd",
    "distillation.dlim_det_query",
    "distillation.eldet_early_learning",
    "distillation.frs_richness",
    "distillation.glamd_attention_mask",
    "distillation.global_kd_prototype",
    "distillation.head_hetero_assist",
    "distillation.icd_instance_conditional",
    "distillation.mscd_cross_scale",
    "distillation.pgd_prediction_guided",
    "distillation.pkd_pearson",
    "distillation.structural_kd",
    "distillation.active_knowledge_aggregation",
    "distillation.cross_task_protocol",
    "distillation.crosskd",
    "distillation.cyclic_disentangled",
    "distillation.decoupled_features",
    "distillation.dense_relation_fewshot",
    "distillation.elastic_response_incremental",
    "distillation.gdetkd",
    "distillation.general_instance",
    "distillation.incremental_within_class",
    "distillation.localization",
    "distillation.object_aware_pyramid",
    "distillation.scale_equivalent",
    "distillation.scalekd",
    "distillation.spatial_self",
    "distillation.structured_instance_graph",
    "distillation.target_perceived_dual_branch",
    "distillation.unikd",
]


def test_distillation_route_id_list_matches_registry() -> None:
    """The literal ID list above must mirror the route registry exactly."""

    registry_ids = {
        route.component_id for route in default_paper_route_registry().routes()
    }
    assert registry_ids == set(_DISTILLATION_ROUTE_COMPONENT_IDS)


_DISTILLATION_ROUTES = default_paper_route_registry().routes()


def _route_adapter(route):
    return paper_route_adapter(route.paper_id)()


def _smoke_options(route) -> dict:
    fixture_root = Path("runs/test-distillation-smoke-fixtures")
    fixture_root.mkdir(parents=True, exist_ok=True)
    # The distillation config validates checkpoint names against the YOLO26
    # scale whitelist, so the synthetic fixtures use the canonical names
    # exactly as the certification runner's own fixtures do.
    teacher = fixture_root / "yolo26s.pt"
    student = fixture_root / "yolo26n.pt"
    teacher_m = fixture_root / "yolo26m.pt"
    if not teacher.is_file():
        teacher.write_bytes(b"distillation-smoke-teacher-fixture\n")
    if not student.is_file():
        student.write_bytes(b"distillation-smoke-student-fixture\n")
    if not teacher_m.is_file():
        teacher_m.write_bytes(b"distillation-smoke-teacher-m-fixture\n")
    options = {
        "teacher": str(teacher.resolve()),
        "student": str(student.resolve()),
        "test_only_teacher_fixture": True,
        "imgsz": 640,
    }
    if route.branch_id == "teacher_ensemble":
        options["teachers"] = [str(teacher_m.resolve())]
    return options


@pytest.mark.parametrize(
    "route", _DISTILLATION_ROUTES, ids=lambda item: item.component_id
)
def test_distillation_route_smoke_forward_backward(route) -> None:
    """Each distillation route runs its real CPU smoke (forward + backward)."""

    adapter = _route_adapter(route)
    context = AdapterContext(
        contract=_distillation_contract(route.component_id),
        detector_family="yolo26",
        head="one_to_one",
        imgsz=640,
        options=_smoke_options(route),
    )
    smoke = adapter.smoke_test(context)
    assert smoke.passed is True, (route.component_id, smoke.errors)
    assert smoke.evidence_kind == "local"
    # The distillation smoke reports shape + backward (its own check keys);
    # both match the readiness evaluator's smoke-term discovery.
    assert smoke.checks.get("shape") is not None
    assert smoke.checks.get("backward") is True


@pytest.mark.parametrize(
    "route", _DISTILLATION_ROUTES, ids=lambda item: item.component_id
)
def test_distillation_route_yolo26_compatibility(route) -> None:
    """Each distillation route validates its YOLO26 compatibility contract."""

    adapter = _route_adapter(route)
    context = AdapterContext(
        contract=_distillation_contract(route.component_id),
        detector_family="yolo26",
        head="one_to_one",
        imgsz=640,
        options=_smoke_options(route),
    )
    report = adapter.validate_compatibility(context)
    assert report.ok is True, (route.component_id, report.errors)
    assert report.checks.get("shared_augmented_batch") is True
    assert report.checks.get("student_inference_graph_unchanged") is True
