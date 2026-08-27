"""Fail-closed CPU certification tests for the thirteen independent routes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from yolo_agent.certification.independent_component_routes import (
    IndependentComponentRouteReport,
    REQUIRED_ROUTE_CHECKS,
    certify_independent_component_route,
    certify_independent_component_routes,
    verify_independent_route_report_hash,
)
from yolo_agent.components.independent_component_router import (
    INDEPENDENT_COMPONENT_IDS,
)
from yolo_agent.research.component_aliases import ComponentAliasResolver


def _patch_contract(monkeypatch, component_id: str, **updates: object) -> None:
    import yolo_agent.certification.independent_component_routes as cert_module

    class PatchedResolver:
        @classmethod
        def from_yaml(cls) -> ComponentAliasResolver:
            resolver = ComponentAliasResolver.from_yaml()
            contract = resolver.contracts[component_id]
            resolver.contracts = dict(resolver.contracts)
            resolver.contracts[component_id] = contract.model_copy(update=updates)
            return resolver

    monkeypatch.setattr(cert_module, "ComponentAliasResolver", PatchedResolver)


@pytest.fixture(scope="module")
def certified_reports() -> list:
    return certify_independent_component_routes(workspace=None)


def _certified(report) -> None:
    assert report.disposition == "certified_route"
    assert report.runtime_ready is False
    assert not any(code.startswith("probe_failed") for code in report.reason_codes)


def test_certify_all_thirteen_routes_pass_required_field_checks(
    certified_reports: list,
) -> None:
    assert len(certified_reports) == 13
    assert {item.component_id for item in certified_reports} == set(
        INDEPENDENT_COMPONENT_IDS
    )
    for report in certified_reports:
        _certified(report)
        assert set(report.checks) == set(REQUIRED_ROUTE_CHECKS)
        assert all(report.checks.values()), report.component_id
        assert report.implementation_path
        assert report.adapter_class
        assert report.changed_variable
        assert report.runtime_hook
        assert report.runtime_payload_field
        assert report.evidence_artifact
        assert report.recipe_id
        assert report.graph_identity
        assert report.fixed_imgsz == 640
        assert report.paired_baseline_required is True
        assert verify_independent_route_report_hash(report)


def test_certify_binds_real_adapter_source_hashes(certified_reports: list) -> None:
    for report in certified_reports:
        assert report.adapter_source_sha256 and len(report.adapter_source_sha256) == 64
        assert report.protocol_hash and len(report.protocol_hash) == 64
        source = Path(
            report.implementation_path.replace(".", "/") + ".py"
        ).resolve()
        expected = hashlib.sha256(source.read_bytes()).hexdigest()
        assert report.adapter_source_sha256 == expected


def test_certify_marks_inference_only_and_shadow_routes(certified_reports: list) -> None:
    sahi = next(
        item for item in certified_reports if item.component_id == "inference.sahi_slicing"
    )
    assert sahi.inference_only is True
    _certified(sahi)
    shadow_components = {
        item.component_id
        for item in certified_reports
        if item.requires_shadow_evidence
    }
    assert shadow_components == {
        "assigner.optimal_transport",
        "assigner.task_aligned",
        "assigner.dynamic_smooth_label",
    }


def test_certify_all_independent_routes_end_to_end(tmp_path: Path) -> None:
    from yolo_agent.certification.independent_component_routes import (
        certify_all_independent_component_routes,
    )
    from yolo_agent.components.independent_component_router import (
        COMPONENT_CATALOG,
        IndependentComponentRouter,
    )

    output = tmp_path / "independent_component_route_certification.yaml"
    summary = certify_all_independent_component_routes(
        output_path=output, workspace=None
    )
    assert summary.components_total == 13
    assert summary.certified_routes == 13
    assert summary.blocked_missing_field == 0
    assert summary.probe_failed == 0
    assert summary.silent_drops == []
    assert summary.inference_only_components == ["inference.sahi_slicing"]
    assert summary.shadow_evidence_components == [
        "assigner.optimal_transport",
        "assigner.task_aligned",
        "assigner.dynamic_smooth_label",
    ]
    assert all(item.runtime_ready is False for item in summary.reports)
    audit_ids = {
        item.component_id
        for item in IndependentComponentRouter().audit_coverage().routes
    }
    assert {item.component_id for item in summary.reports} == audit_ids
    assert {item.component_id for item in summary.reports} == set(
        INDEPENDENT_COMPONENT_IDS
    )
    by_id = {item.component_id: item for item in summary.reports}
    for component_id, catalog in COMPONENT_CATALOG.items():
        assert by_id[component_id].recipe_id == catalog["recipe_id"]
        assert by_id[component_id].graph_identity == catalog["graph_identity"]
    reloaded = type(summary).from_yaml(output)
    assert reloaded.summary_hash == summary.summary_hash
    assert reloaded.components_total == 13


def test_certify_rejects_runtime_ready_reports() -> None:
    report = certify_independent_component_route(
        "loss.calibration.bpc", workspace=None
    )
    dumped = report.model_dump(mode="python")
    dumped["runtime_ready"] = True
    with pytest.raises(ValueError, match="runtime ready"):
        IndependentComponentRouteReport.model_validate(dumped)


def test_certify_dfl_dependent_contract_blocks(monkeypatch) -> None:
    _patch_contract(
        monkeypatch,
        "neck.rtmdet_large_kernel",
        tensor_input_contract={
            "compatibility_constraints": {"requires_dfl": True}
        },
    )
    report = certify_independent_component_route(
        "neck.rtmdet_large_kernel", workspace=None
    )
    assert report.disposition == "blocked_missing_field"
    assert "missing_field:native_dfl_free_regression" in report.reason_codes
    assert report.runtime_ready is False


def test_certify_missing_fixed_imgsz_compatibility_blocks(monkeypatch) -> None:
    _patch_contract(
        monkeypatch, "neck.rtmdet_large_kernel", fixed_imgsz_compatible=False
    )
    report = certify_independent_component_route(
        "neck.rtmdet_large_kernel", workspace=None
    )
    assert report.disposition == "blocked_missing_field"
    assert "missing_field:fixed_imgsz_640" in report.reason_codes


def test_certify_changed_variable_contract_mismatch_blocks(monkeypatch) -> None:
    _patch_contract(
        monkeypatch,
        "neck.rtmdet_large_kernel",
        changed_variable="model.wrong_neck",
    )
    report = certify_independent_component_route(
        "neck.rtmdet_large_kernel", workspace=None
    )
    assert report.disposition == "blocked_missing_field"
    assert "changed_variable_contract_mismatch" in report.reason_codes


def test_certify_missing_evidence_protocol_blocks(monkeypatch) -> None:
    _patch_contract(
        monkeypatch, "loss.calibration.bpc", evidence_protocol=[]
    )
    report = certify_independent_component_route(
        "loss.calibration.bpc", workspace=None
    )
    assert report.disposition == "blocked_missing_field"
    assert "missing_field:evidence_artifact" in report.reason_codes


def test_certify_without_matched_baseline_blocks() -> None:
    report = certify_independent_component_route(
        "loss.calibration.bpc", workspace=None, matched_baseline=False
    )
    assert report.disposition == "blocked_missing_field"
    assert "missing_field:matched_baseline" in report.reason_codes


def test_certify_non_640_imgsz_blocks() -> None:
    report = certify_independent_component_route(
        "neck.rtmdet_large_kernel", workspace=None, imgsz=1280
    )
    assert report.disposition == "blocked_missing_field"
    assert "missing_field:fixed_imgsz_640" in report.reason_codes
    assert report.runtime_ready is False


def test_certify_broken_adapter_module_is_probe_failed(monkeypatch) -> None:
    _patch_contract(
        monkeypatch,
        "neck.rtmdet_large_kernel",
        implementation_path="yolo_agent.components.adapters.neck.nonexistent_rtmdet",
    )
    report = certify_independent_component_route(
        "neck.rtmdet_large_kernel", workspace=None
    )
    assert report.disposition == "probe_failed"
    assert "probe_failed:ModuleNotFoundError" in report.reason_codes
    assert report.runtime_ready is False


def test_factory_certify_independent_routes_persists_summary(tmp_path: Path) -> None:
    from yolo_agent.certification.paper_adapter_factory import (
        PaperAdapterCertificationFactory,
    )

    summary = PaperAdapterCertificationFactory().certify_independent_component_routes(
        workdir=tmp_path
    )
    assert summary.components_total == 13
    assert summary.certified_routes == 13
    assert summary.blocked_missing_field == 0
    assert summary.probe_failed == 0
    assert summary.inference_only_components == ["inference.sahi_slicing"]
    assert summary.silent_drops == []
    assert all(item.runtime_ready is False for item in summary.reports)
    output = tmp_path / "independent_component_route_certification.yaml"
    assert output.is_file()
    reloaded = type(summary).from_yaml(output)
    assert reloaded.summary_hash == summary.summary_hash


def test_route_report_hash_is_tamper_evident(certified_reports: list) -> None:
    report = certified_reports[0]
    dumped = report.model_dump(mode="python")
    dumped["disposition"] = "blocked_missing_field"
    with pytest.raises(ValueError, match="report hash mismatch"):
        IndependentComponentRouteReport.model_validate(dumped)
    dumped = report.model_dump(mode="python")
    dumped["checks"]["fixed_imgsz_640"] = False
    with pytest.raises(ValueError, match="report hash mismatch"):
        IndependentComponentRouteReport.model_validate(dumped)


def test_summary_hash_is_tamper_evident(tmp_path: Path) -> None:
    from yolo_agent.certification.paper_adapter_factory import (
        PaperAdapterCertificationFactory,
    )

    summary = PaperAdapterCertificationFactory().certify_independent_component_routes(
        workdir=tmp_path
    )
    dumped = summary.model_dump(mode="python")
    dumped["inference_only_components"] = []
    with pytest.raises(ValueError, match="summary hash mismatch"):
        type(summary).model_validate(dumped)
    dumped = summary.model_dump(mode="python")
    dumped["reports"] = dumped["reports"][:-1]
    with pytest.raises(ValueError, match="exactly one report"):
        type(summary).model_validate(dumped)


def test_factory_independent_summary_rejects_runtime_ready(tmp_path: Path) -> None:
    from yolo_agent.certification.paper_adapter_factory import (
        PaperAdapterCertificationFactory,
    )

    summary = PaperAdapterCertificationFactory().certify_independent_component_routes(
        workdir=tmp_path
    )
    dumped = summary.model_dump(mode="python")
    dumped["reports"][0]["runtime_ready"] = True
    with pytest.raises(ValueError, match="runtime ready"):
        type(summary).model_validate(dumped)


def _digest(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


def _shadow_recipe(recipe_id: str = "yolo26_tood_tal_assignment_shadow"):
    import yaml

    from yolo_agent.recipes.schemas import AtomicRecipe

    data = yaml.safe_load(
        Path("configs/recipes/yolo26_assignment_shadow.yaml").read_text(
            encoding="utf-8-sig"
        )
    )
    return AtomicRecipe.model_validate(
        next(item for item in data["recipes"] if item["recipe_id"] == recipe_id)
    )


def test_assignment_active_requires_shadow_evidence_before_active() -> None:
    from yolo_agent.certification.assignment_pilot_gate import (
        AssignmentActivePilotMaterializer,
    )

    recipe = _shadow_recipe()
    materializer = AssignmentActivePilotMaterializer()
    missing = materializer.materialize(
        shadow_recipe=recipe,
        shadow_evidence_path=Path("does_not_exist.json"),
        candidate_protocol_hash="abc123",
        control_protocol_hash="abc123",
        matched_control_available=True,
    )
    assert missing.allowed is False
    assert missing.execution_class == "blocked"
    assert "shadow_evidence_missing" in missing.blocked_by
    assert missing.active_recipe is None
    no_control = materializer.materialize(
        shadow_recipe=recipe,
        shadow_evidence_path=Path("does_not_exist.json"),
        candidate_protocol_hash="abc123",
        control_protocol_hash="abc123",
        matched_control_available=False,
    )
    assert "matched_control_missing" in no_control.blocked_by
    mismatch = materializer.materialize(
        shadow_recipe=recipe,
        shadow_evidence_path=Path("does_not_exist.json"),
        candidate_protocol_hash="abc123",
        control_protocol_hash="different",
        matched_control_available=True,
    )
    assert "matched_control_protocol_mismatch" in mismatch.blocked_by


def test_assignment_router_requires_shadow_evidence_for_active_queue() -> None:
    from yolo_agent.components.independent_component_router import (
        ASSIGNMENT_SHADOW_COMPONENTS,
        IndependentComponentRouter,
    )

    router = IndependentComponentRouter()
    full = {
        "has_payload": True,
        "has_changed_variable": True,
        "has_evidence": True,
        "has_adapter_hash": True,
        "paired_baseline": True,
        "contract_can_execute": True,
    }
    assert len(ASSIGNMENT_SHADOW_COMPONENTS) == 3
    for component_id in ASSIGNMENT_SHADOW_COMPONENTS:
        blocked = router.route(component_id, has_shadow_evidence=False, **full)
        assert blocked.asha_eligible is False
        assert "assignment_shadow_evidence_required" in blocked.reason_codes
        ready = router.route(component_id, has_shadow_evidence=True, **full)
        assert ready.asha_eligible is True
        assert ready.requires_shadow_evidence is True
        assert ready.queue_track == "training"


def test_assignment_pilot_state_rejects_backward_and_terminal_conflicts() -> None:
    from yolo_agent.certification.assignment_pilot_state import (
        AssignmentPilotState,
    )

    state = AssignmentPilotState(
        run_id="run",
        trial_id="trial",
        candidate_id="candidate",
        canonical_component_id="assigner.task_aligned",
        shadow_recipe_id="yolo26_tood_tal_assignment_shadow",
        protocol_hash="protocol-1",
    )
    assert state.state == "shadow_planned"
    state.transition("shadow_evidence_complete")
    state.transition("active_candidate_eligible")
    with pytest.raises(ValueError, match="cannot move backward"):
        state.transition("shadow_planned")
    state.transition("active_pilot")
    state.transition("promoted")
    with pytest.raises(ValueError, match="terminal state conflict"):
        state.transition("rejected")


def _minimal_independent_inventory(component_id: str, recipe_id: str):
    from yolo_agent.research.paper_execution_schemas import (
        PaperExecutionInventory,
        PaperExecutionSpec,
    )

    spec = PaperExecutionSpec(
        paper_id="arxiv:2212.07784",
        profile_id=f"profile-{component_id}",
        title="Independent Component Paper",
        source_locations=["paper.md"],
        canonical_component_ids=[component_id],
        paper_specific_mechanism_ids=[component_id],
        generic_component_ids=[],
        required_checkpoints=[],
        required_evidence=[],
        recipe_ids=[recipe_id],
        execution_fingerprint=_digest("execution-fingerprint"),
        disposition_reason="test",
    )
    return PaperExecutionInventory(
        source_method_coverage_hash=_digest("coverage"),
        source_maturity_hash=_digest("maturity"),
        all_paper_count=1,
        compatible_paper_count=1,
        exact_reproduction_candidates=0,
        generic_mechanism_counts={},
        records=[spec],
    )


def test_graph_neck_identities_stay_distinct_in_certification(
    certified_reports: list,
) -> None:
    by_id = {item.component_id: item for item in certified_reports}
    graph_ids = {
        "neck.gold_gather_distribute",
        "neck.multi_scale_fusion",
        "neck.rtmdet_large_kernel",
        "attention.spatial",
        "feature_pyramid.multi_scale",
    }
    reports = [by_id[component_id] for component_id in graph_ids]
    for report in reports:
        _certified(report)
        assert report.graph_identity == report.component_id
    assert {item.graph_identity for item in reports} == graph_ids
    assert len({item.protocol_hash for item in reports}) == len(reports)
    necks = {
        "neck.gold_gather_distribute",
        "neck.multi_scale_fusion",
        "neck.rtmdet_large_kernel",
    }
    neck_variable = by_id["neck.gold_gather_distribute"].changed_variable
    assert all(
        by_id[component_id].changed_variable == neck_variable for component_id in necks
    )
    assert (
        by_id["feature_pyramid.multi_scale"].changed_variable != neck_variable
    )


def test_quality_pair_stays_independent_end_to_end(certified_reports: list) -> None:
    from yolo_agent.components.independent_component_router import (
        IndependentComponentRouter,
    )

    by_id = {item.component_id: item for item in certified_reports}
    correlation = by_id["loss.quality.correlation"]
    pseudo_iou = by_id["loss.quality.pseudo_iou"]
    _certified(correlation)
    _certified(pseudo_iou)
    assert correlation.recipe_id != pseudo_iou.recipe_id
    assert correlation.changed_variable != pseudo_iou.changed_variable
    assert correlation.evidence_artifact != pseudo_iou.evidence_artifact
    assert correlation.graph_identity != pseudo_iou.graph_identity
    assert correlation.protocol_hash != pseudo_iou.protocol_hash
    for report in (correlation, pseudo_iou):
        assert report.cpu_smoke_checks["native_bbox_regression_preserved"] is True
        assert report.cpu_smoke_checks["native_assigner_preserved"] is True
    router = IndependentComponentRouter()
    common = {
        "has_payload": True,
        "has_changed_variable": True,
        "has_evidence": True,
        "has_adapter_hash": True,
        "paired_baseline": True,
        "contract_can_execute": True,
    }
    routed = {
        "loss.quality.correlation": router.route("loss.quality.correlation", **common),
        "loss.quality.pseudo_iou": router.route("loss.quality.pseudo_iou", **common),
    }
    assert all(item.asha_eligible is True for item in routed.values())
    assert routed["loss.quality.correlation"].recipe_id != routed[
        "loss.quality.pseudo_iou"
    ].recipe_id


def test_inference_only_route_model_cannot_enter_training_asha() -> None:
    from yolo_agent.components.independent_component_router import (
        IndependentComponentRoute,
    )

    base = {
        "component_id": "inference.sahi_slicing",
        "recipe_id": "sahi_slicing_inference",
        "implementation_path": "yolo_agent.components.adapters.inference.slicing",
        "adapter_class": "SlicingInferenceAdapter",
        "changed_variable": "inference.slicing_policy",
        "runtime_hook": "prepare_command",
        "runtime_payload_field": "inference_plugin",
        "evidence_artifact": "slicing_inference_protocol.json",
        "graph_identity": "inference.sahi_slicing",
        "inference_only": True,
        "disposition": "queued",
    }
    with pytest.raises(ValueError, match="cannot be training candidates"):
        IndependentComponentRoute(**base, queue_track="training", asha_eligible=False)
    with pytest.raises(ValueError, match="cannot enter training ASHA"):
        IndependentComponentRoute(
            **base, queue_track="inference", asha_eligible=True
        )


def test_inference_only_route_stays_off_training_in_router_and_audit() -> None:
    from yolo_agent.components.independent_component_router import (
        IndependentComponentRouter,
    )

    router = IndependentComponentRouter()
    route = router.route(
        "inference.sahi_slicing",
        has_payload=True,
        has_changed_variable=True,
        has_evidence=True,
        has_adapter_hash=True,
        paired_baseline=True,
        contract_can_execute=True,
    )
    assert route.inference_only is True
    assert route.queue_track == "inference"
    assert route.asha_eligible is False
    assert "inference_only_not_training_candidate" in route.reason_codes
    audit = next(
        item
        for item in router.audit_coverage().routes
        if item.component_id == "inference.sahi_slicing"
    )
    assert audit.queue_track == "inference"
    assert audit.asha_eligible is False


def test_inference_only_certification_never_reports_training_readiness(
    certified_reports: list,
) -> None:
    sahi = next(
        item
        for item in certified_reports
        if item.component_id == "inference.sahi_slicing"
    )
    _certified(sahi)
    assert sahi.inference_only is True
    assert sahi.runtime_ready is False


def test_readiness_validation_passes_for_all_independent_routes() -> None:
    from yolo_agent.research.paper_execution_requirements import (
        validate_independent_standard_routes,
    )

    assert validate_independent_standard_routes() == []


def test_readiness_rows_carry_all_independent_route_fields(tmp_path: Path) -> None:
    from yolo_agent.research.paper_execution_requirements import (
        PaperExecutionRequirementsBuilder,
    )

    row = PaperExecutionRequirementsBuilder().build(
        _minimal_independent_inventory(
            "neck.rtmdet_large_kernel", "yolo26_rtmdet_large_kernel_neck"
        ),
        source_inventory_path=tmp_path / "inventory.yaml",
    ).requirements[0]
    assert row.required_adapter == "neck.rtmdet_large_kernel"
    assert row.required_changed_variables
    assert row.required_runtime_payload
    assert row.recipe_ids == ["yolo26_rtmdet_large_kernel_neck"]
    assert row.required_graph_assets == [
        "yolo26_one_to_one_head",
        "native_dfl_free_regression",
        "imgsz_640",
    ]
    assert len(row.protocol_hash) == 64


def test_task_aligned_head_keeps_independent_identity_in_certification(
    certified_reports: list,
) -> None:
    by_id = {item.component_id: item for item in certified_reports}
    head = by_id["detection_head.task_aligned"]
    assigner = by_id["assigner.task_aligned"]
    _certified(head)
    _certified(assigner)
    assert head.graph_identity == "detection_head.task_aligned"
    assert assigner.graph_identity == "assigner.task_aligned"
    assert head.graph_identity != assigner.graph_identity
    assert head.recipe_id != assigner.recipe_id
    assert head.changed_variable != assigner.changed_variable
    assert head.implementation_path != assigner.implementation_path
    assert head.adapter_class != assigner.adapter_class
    assert head.evidence_artifact != assigner.evidence_artifact
    assert head.protocol_hash != assigner.protocol_hash
    assert head.cpu_smoke_checks["native_one_to_one"] is True
    assert head.cpu_smoke_checks["native_dfl_free"] is True
    assert assigner.cpu_smoke_checks["shadow_mode"] is True
    assert "shadow_mode" not in head.cpu_smoke_checks


def test_readiness_build_rejects_broken_independent_binding(
    monkeypatch, tmp_path: Path
) -> None:
    import yolo_agent.research.paper_execution_requirements as requirements_module
    from yolo_agent.research.paper_execution_requirements import (
        PaperExecutionRequirementsBuilder,
    )

    broken = dict(requirements_module._STANDARD_ROUTES)
    broken_rtmdet = dict(broken["neck.rtmdet_large_kernel"])
    broken_rtmdet.pop("payload")
    broken["neck.rtmdet_large_kernel"] = broken_rtmdet
    monkeypatch.setattr(requirements_module, "_STANDARD_ROUTES", broken)
    with pytest.raises(
        ValueError, match="independent_route_payload_missing:neck.rtmdet_large_kernel"
    ):
        PaperExecutionRequirementsBuilder().build(
            _minimal_independent_inventory(
                "neck.rtmdet_large_kernel", "yolo26_rtmdet_large_kernel_neck"
            ),
            source_inventory_path=tmp_path / "inventory.yaml",
        )


def test_certify_cpu_smoke_checks_cover_contract_shape_forward_backward(
    certified_reports: list,
) -> None:
    by_id = {item.component_id: item for item in certified_reports}
    for report in certified_reports:
        assert report.cpu_smoke_checks, report.component_id
    graph_components = {
        "neck.gold_gather_distribute",
        "neck.multi_scale_fusion",
        "neck.rtmdet_large_kernel",
        "attention.spatial",
        "feature_pyramid.multi_scale",
    }
    for component_id in graph_components:
        checks = by_id[component_id].cpu_smoke_checks
        assert checks["forward"] is True
        assert checks["backward"] is True
        assert checks["shape_contract"] is True
    head_checks = by_id["detection_head.task_aligned"].cpu_smoke_checks
    assert head_checks["forward"] is True
    assert head_checks["backward"] is True
    assert head_checks["native_one_to_one"] is True
    assert head_checks["native_dfl_free"] is True
    for component_id in (
        "loss.quality.correlation",
        "loss.quality.pseudo_iou",
        "loss.calibration.bpc",
    ):
        checks = by_id[component_id].cpu_smoke_checks
        assert checks["backward"] is True
        assert checks["native_bbox_regression_preserved"] is True
        assert checks["native_assigner_preserved"] is True
    for component_id in (
        "assigner.optimal_transport",
        "assigner.task_aligned",
        "assigner.dynamic_smooth_label",
    ):
        checks = by_id[component_id].cpu_smoke_checks
        assert checks["native_loss_unchanged"] is True
        assert checks["shadow_mode"] is True
        assert checks["assignment_path"] == "one_to_many"
    sahi_checks = by_id["inference.sahi_slicing"].cpu_smoke_checks
    assert sahi_checks["standard_metrics_preserved"] is True
    assert sahi_checks["extra_nms"] is False
