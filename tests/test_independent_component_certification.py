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
