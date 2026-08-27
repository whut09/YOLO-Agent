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
