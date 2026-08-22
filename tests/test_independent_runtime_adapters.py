"""CPU-only regression tests for the twelve independent YOLO26 adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.components.adapters import AdapterContext
from yolo_agent.components.adapters.audit_contract import (
    validate_audited_runtime_payload,
)
from yolo_agent.components.adapters.heads.task_aligned import TaskAlignedHeadAdapter
from yolo_agent.components.adapters.neck.feature_pyramid_adapter import (
    FeaturePyramidMultiScaleAdapter,
)
from yolo_agent.components.independent_component_router import (
    INDEPENDENT_COMPONENT_IDS,
    IndependentComponentRouter,
)
from yolo_agent.research.component_aliases import ComponentAliasResolver


def test_cpu_audit_finds_real_payloads_for_all_independent_components(
    tmp_path: Path,
) -> None:
    coverage = IndependentComponentRouter().audit_coverage(workspace=tmp_path)

    assert {item.component_id for item in coverage.routes} == set(INDEPENDENT_COMPONENT_IDS)
    for route in coverage.routes:
        assert route.adapter_source_sha256 and len(route.adapter_source_sha256) == 64
        assert route.runtime_payload_hash and len(route.runtime_payload_hash) == 64
        assert not any(code.startswith("runtime_probe_failed") for code in route.reason_codes)
        assert "runtime_payload_missing" not in route.reason_codes
        assert "changed_variable_missing" not in route.reason_codes
    assert next(
        item for item in coverage.routes if item.component_id == "inference.sahi_slicing"
    ).asha_eligible is False


def test_contracts_declare_independent_runtime_boundaries() -> None:
    resolver = ComponentAliasResolver.from_yaml()
    for component_id in INDEPENDENT_COMPONENT_IDS:
        contract = resolver.contracts[component_id]
        assert contract.implementation_path
        assert contract.adapter_class
        assert contract.changed_variable
        assert contract.paper_specific_mechanism_ids == [component_id]
        assert contract.runtime_payload_schema
        assert contract.evidence_protocol
        assert contract.fixed_imgsz_compatible is True


def test_task_aligned_head_payload_is_independent_from_assignment() -> None:
    contract = ComponentAliasResolver.from_yaml().contracts["detection_head.task_aligned"]
    context = AdapterContext(
        contract=contract,
        detector_family="yolo26",
        head="one_to_one",
        imgsz=640,
        workspace=Path("."),
    )
    payload = TaskAlignedHeadAdapter().build_runtime_payload(
        context,
        protocol_hash="head-cpu-protocol",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={"imgsz": 640},
    )
    checks = validate_audited_runtime_payload(payload, "detection_head.task_aligned")
    assert checks["audited_required_hook"] == "build_model"
    assert payload.model_graph_plugin[0].reference.endswith(
        "heads.task_aligned:TaskAlignedHeadRuntimePlugin"
    )


def test_feature_pyramid_payload_keeps_its_own_changed_variable() -> None:
    contract = ComponentAliasResolver.from_yaml().contracts["feature_pyramid.multi_scale"]
    context = AdapterContext(
        contract=contract,
        detector_family="yolo26",
        head="one_to_one",
        imgsz=640,
        workspace=Path("."),
    )
    payload = FeaturePyramidMultiScaleAdapter().build_runtime_payload(
        context,
        protocol_hash="pyramid-cpu-protocol",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={"imgsz": 640},
    )
    checks = validate_audited_runtime_payload(payload, "feature_pyramid.multi_scale")
    assert checks["audited_changed_variable"] == "model.feature_pyramid"
    assert payload.changed_variables.keys() == {"model.feature_pyramid"}


@pytest.mark.parametrize("component_id", [
    "detection_head.task_aligned",
    "feature_pyramid.multi_scale",
])
def test_new_graph_adapters_reject_non_640_protocol(component_id: str) -> None:
    contract = ComponentAliasResolver.from_yaml().contracts[component_id]
    adapter = (
        TaskAlignedHeadAdapter()
        if component_id == "detection_head.task_aligned"
        else FeaturePyramidMultiScaleAdapter()
    )
    context = AdapterContext(
        contract=contract,
        detector_family="yolo26",
        head="one_to_one",
        imgsz=608,
    )
    report = adapter.validate_compatibility(context)
    assert report.ok is False
    assert any("640" in error for error in report.errors)


def test_missing_payload_or_evidence_cannot_become_asha_candidate() -> None:
    router = IndependentComponentRouter()
    for component_id in (
        "detection_head.task_aligned",
        "feature_pyramid.multi_scale",
    ):
        route = router.route(
            component_id,
            has_payload=False,
            has_changed_variable=False,
            has_evidence=False,
            has_adapter_hash=True,
            paired_baseline=True,
            contract_can_execute=True,
        )
        assert route.asha_eligible is False
        assert route.queue_track == "blocked"
        assert "runtime_payload_missing" in route.reason_codes
        assert "evidence_artifact_missing" in route.reason_codes


def test_assignment_shadow_and_inference_boundaries_remain_explicit() -> None:
    router = IndependentComponentRouter()
    blocked = router.route(
        "assigner.task_aligned",
        has_payload=True,
        has_changed_variable=True,
        has_evidence=True,
        has_adapter_hash=True,
        paired_baseline=True,
        contract_can_execute=True,
    )
    assert blocked.asha_eligible is False
    assert "assignment_shadow_evidence_required" in blocked.reason_codes
    sahi = router.route(
        "inference.sahi_slicing",
        has_payload=True,
        has_changed_variable=True,
        has_evidence=True,
        has_adapter_hash=True,
        paired_baseline=True,
        contract_can_execute=True,
    )
    assert sahi.queue_track == "inference"
    assert sahi.asha_eligible is False
