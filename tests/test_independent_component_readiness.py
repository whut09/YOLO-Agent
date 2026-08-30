"""CPU-only tests for the independent component readiness boundary."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from yolo_agent.certification import independent_component_readiness as readiness
from yolo_agent.certification.independent_component_routes import (
    IndependentComponentRouteReport,
)
from yolo_agent.components.independent_component_router import INDEPENDENT_COMPONENT_IDS


def _fake_route(component_id: str, *, imgsz: int = 640, complete: bool = True) -> IndependentComponentRouteReport:
    catalog = readiness.COMPONENT_CATALOG[component_id]
    checks = {
        name: complete
        for name in (
            "contract_present",
            "recipe_id",
            "implementation_path",
            "adapter_class",
            "changed_variable",
            "runtime_hook",
            "runtime_payload_field",
            "graph_identity",
            "payload_schema",
            "evidence_artifact",
            "adapter_hash",
            "protocol_hash",
            "fixed_imgsz_640",
            "yolo26_one_to_one_head",
            "native_dfl_free_regression",
            "matched_baseline",
            "cpu_smoke",
        )
    }
    if imgsz != 640:
        checks["fixed_imgsz_640"] = False
    return IndependentComponentRouteReport(
        component_id=component_id,
        recipe_id=str(catalog["recipe_id"]),
        graph_identity=str(catalog["graph_identity"]),
        implementation_path=str(catalog["implementation_path"]),
        adapter_class=str(catalog["adapter_class"]),
        changed_variable=str(catalog["changed_variable"]),
        runtime_hook=str(catalog["runtime_hook"]),
        runtime_payload_field=str(catalog["runtime_payload_field"]),
        evidence_artifact=str(catalog["evidence_artifact"]),
        inference_only=component_id == "inference.sahi_slicing",
        requires_shadow_evidence=component_id in readiness.ASSIGNMENT_SHADOW_COMPONENTS,
        paired_baseline_required=True,
        fixed_imgsz=imgsz,
        checks=checks,
        cpu_smoke_checks={"shape": True, "forward": True, "backward": True},
        adapter_source_sha256="b" * 64,
        protocol_hash="a" * 64,
        disposition="certified_route" if complete else "blocked_missing_field",
        reason_codes=[] if complete else ["missing_field:payload_schema"],
    )


def _patch_cert(monkeypatch, *, complete: bool = True) -> None:
    monkeypatch.setattr(
        readiness,
        "certify_independent_component_route",
        lambda component_id, **kwargs: _fake_route(
            component_id,
            imgsz=int(kwargs.get("imgsz", 640)),
            complete=complete,
        ),
    )


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _valid_artifacts(tmp_path: Path, component_id: str) -> tuple[Path, Path, Path | None]:
    protocol = "a" * 64
    evidence = _write(
        tmp_path / "evidence.yaml",
        {"component_id": component_id, "protocol_hash": protocol, "status": "passed"},
    )
    baseline = _write(
        tmp_path / "baseline.yaml",
        {"protocol_hash": protocol, "imgsz": 640, "split": "train", "status": "verified"},
    )
    shadow = None
    if component_id in readiness.ASSIGNMENT_SHADOW_COMPONENTS:
        shadow = _write(
            tmp_path / "shadow.yaml",
            {
                "component_id": component_id,
                "protocol_hash": protocol,
                "status": "shadow_evidence_complete",
                "minimum_batches": 2,
                "positive_assignment_valid": True,
                "native_loss_equivalence": True,
            },
        )
    return evidence, baseline, shadow


def test_missing_real_artifacts_never_become_runtime_ready(monkeypatch, tmp_path: Path) -> None:
    _patch_cert(monkeypatch)
    result = readiness.assess_independent_component_readiness(
        "loss.quality.correlation",
        workspace=tmp_path,
    )
    assert result.runtime_ready is False
    assert result.asha_eligible is False
    assert result.disposition == "evidence_recovery"
    assert "evidence_artifact_missing" in result.reason_codes
    assert "matched_baseline_artifact_missing" in result.reason_codes


def test_complete_quality_route_is_runtime_ready_only_with_real_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_cert(monkeypatch)
    evidence, baseline, _ = _valid_artifacts(tmp_path, "loss.quality.correlation")
    result = readiness.assess_independent_component_readiness(
        "loss.quality.correlation",
        evidence_artifact=evidence,
        matched_baseline_artifact=baseline,
        workspace=tmp_path,
    )
    assert result.runtime_ready is True
    assert result.asha_eligible is True
    assert result.disposition == "runtime_ready"
    assert result.reason_codes == []


def test_assignment_requires_passed_shadow_evidence(monkeypatch, tmp_path: Path) -> None:
    _patch_cert(monkeypatch)
    evidence, baseline, _ = _valid_artifacts(tmp_path, "assigner.task_aligned")
    result = readiness.assess_independent_component_readiness(
        "assigner.task_aligned",
        evidence_artifact=evidence,
        matched_baseline_artifact=baseline,
        workspace=tmp_path,
    )
    assert result.runtime_ready is False
    assert result.asha_eligible is False
    assert "assignment_shadow_evidence_missing" in result.reason_codes


def test_assignment_shadow_and_matched_protocol_enable_active_route(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_cert(monkeypatch)
    evidence, baseline, shadow = _valid_artifacts(tmp_path, "assigner.task_aligned")
    result = readiness.assess_independent_component_readiness(
        "assigner.task_aligned",
        evidence_artifact=evidence,
        matched_baseline_artifact=baseline,
        shadow_evidence_artifact=shadow,
        workspace=tmp_path,
    )
    assert result.runtime_ready is True
    assert result.asha_eligible is True


def test_baseline_protocol_mismatch_blocks_route(monkeypatch, tmp_path: Path) -> None:
    _patch_cert(monkeypatch)
    evidence, baseline, _ = _valid_artifacts(tmp_path, "loss.quality.pseudo_iou")
    _write(baseline, {"protocol_hash": "c" * 64, "imgsz": 640, "split": "train", "status": "verified"})
    result = readiness.assess_independent_component_readiness(
        "loss.quality.pseudo_iou",
        evidence_artifact=evidence,
        matched_baseline_artifact=baseline,
        workspace=tmp_path,
    )
    assert result.runtime_ready is False
    assert result.asha_eligible is False
    assert "matched_baseline_protocol_mismatch" in result.reason_codes


def test_non_640_route_is_blocked(monkeypatch, tmp_path: Path) -> None:
    _patch_cert(monkeypatch)
    result = readiness.assess_independent_component_readiness(
        "neck.rtmdet_large_kernel",
        workspace=tmp_path,
        imgsz=608,
    )
    assert result.runtime_ready is False
    assert result.asha_eligible is False
    assert "fixed_imgsz_640_required" in result.reason_codes


@pytest.mark.parametrize(
    "missing_field",
    ["recipe_id", "runtime_payload_field", "graph_identity", "evidence_artifact"],
)
def test_missing_route_identity_field_cannot_be_certified(
    monkeypatch, missing_field: str
) -> None:
    import yolo_agent.certification.independent_component_routes as routes

    original = routes.COMPONENT_CATALOG["neck.rtmdet_large_kernel"]
    routes.COMPONENT_CATALOG["neck.rtmdet_large_kernel"] = {
        **original,
        missing_field: "",
    }
    try:
        report = routes.certify_independent_component_route(
            "neck.rtmdet_large_kernel"
        )
    finally:
        routes.COMPONENT_CATALOG["neck.rtmdet_large_kernel"] = original
    assert report.runtime_ready is False
    assert report.disposition == "blocked_missing_field"
    assert f"missing_field:{missing_field}" in report.reason_codes


def test_inference_route_never_becomes_training_asha(monkeypatch, tmp_path: Path) -> None:
    _patch_cert(monkeypatch)
    evidence, baseline, _ = _valid_artifacts(tmp_path, "inference.sahi_slicing")
    result = readiness.assess_independent_component_readiness(
        "inference.sahi_slicing",
        evidence_artifact=evidence,
        matched_baseline_artifact=baseline,
        workspace=tmp_path,
    )
    assert result.runtime_ready is True
    assert result.training_candidate_allowed is False
    assert result.asha_eligible is False


def test_summary_covers_all_independent_routes_without_silent_drop(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_cert(monkeypatch)
    summary = readiness.assess_independent_component_routes(workspace=tmp_path)
    assert summary.components_total == len(INDEPENDENT_COMPONENT_IDS)
    assert {item.component_id for item in summary.reports} == set(INDEPENDENT_COMPONENT_IDS)
    assert summary.silent_drops == []
    assert summary.asha_eligible_count == 0
    assert summary.blocked_count == 12
