"""Inference-side audit tests: scope, probes, and the deployment boundary.

No test here executes a model or starts training.  Probes run the real
inference-adapter code on synthetic detections.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pydantic
import pytest

from yolo_agent.research.paper_inference_side import (
    PaperInferenceSideAuditBuilder,
    _behavior_probes,
)
from yolo_agent.research.paper_inference_side_schemas import (
    InferenceDeploymentBoundary,
    PaperInferenceSideRecord,
)


def test_audit_scope_comes_from_frozen_evidence() -> None:
    audit = PaperInferenceSideAuditBuilder().build()
    assert audit.paper_count == 83
    # No frozen paper carries a postprocess/inference/calibration domain;
    # the audit records that honestly instead of inflating the count.
    assert audit.inference_side_paper_count == 0
    assert audit.summary["out_of_scope"] == 83
    assert audit.audit_hash

    by_id = {record.paper_id: record for record in audit.records}
    bpc = by_id["arxiv:2303.14404"]
    assert bpc.status == "out_of_scope"
    assert "train-time loss" in bpc.scope_reason
    assert "loss.calibration.bpc" in bpc.scope_reason


def test_calibration_paper_is_not_rebranded_as_inference_policy() -> None:
    audit = PaperInferenceSideAuditBuilder().build()
    bpc = next(r for r in audit.records if r.paper_id == "arxiv:2303.14404")
    # The train-time calibration loss keeps its loss-side identity; no
    # inference-side mechanism_id or policy is invented for it.
    assert bpc.mechanism_id == ""
    assert bpc.policy_kind is None
    assert bpc.deployment_boundary is None


def test_behavior_probes_cover_all_required_mechanisms() -> None:
    probes = _behavior_probes()
    assert probes.passed, probes.errors
    checks = probes.checks
    # NMS: overlap / non-overlap / class-aware edges.
    assert checks["nms_overlap_suppressed"]
    assert checks["nms_keeps_higher_score"]
    assert checks["class_aware_nms_keeps_cross_class"]
    # Soft-NMS family: score decay on overlapping boxes.
    assert checks["wbf_fuses_overlaps"]
    assert checks["wbf_cluster_max_score_preserved"]
    # WBF: coordinate fusion toward the weighted centroid.
    assert checks["wbf_fuses_coordinates"]
    # Thresholds: per-class difference is real.
    assert checks["per_class_threshold_difference"]
    # Calibration: score distribution changes, ordering survives.
    assert checks["temperature_changes_distribution"]
    assert checks["calibration_monotone"]
    assert checks["calibration_in_range"]
    # Tiles: tile -> global coordinate mapping is exact.
    assert checks["tile_global_mapping"]
    # TTA: transform -> inverse transform is the identity.
    assert checks["tta_inverse_transform"]
    assert len(probes.observed_changes) >= 6


def test_deployment_boundary_is_literal_typed() -> None:
    boundary = InferenceDeploymentBoundary(
        policy_kind="confidence_calibration",
        metric_namespace="calibrated_inference",
        adapter_path="yolo_agent.components.adapters.inference.policy",
        latency_risk="low",
        export_compatibility_risk="post-export; graphs untouched",
        nms_free_compatibility="native_one_to_one",
        one_to_one_guard_required=False,
    )
    assert boundary.train_time is False
    assert boundary.inference_time is True
    # A training recipe can never serialize train_time=true: the Literal
    # type itself rejects it.
    with pytest.raises(pydantic.ValidationError):
        InferenceDeploymentBoundary.model_validate(
            {**boundary.model_dump(mode="json"), "train_time": True}
        )
    with pytest.raises(pydantic.ValidationError):
        InferenceDeploymentBoundary.model_validate(
            {**boundary.model_dump(mode="json"), "inference_time": False}
        )


def test_cross_view_merge_requires_one_to_one_guard() -> None:
    with pytest.raises(pydantic.ValidationError, match="allow_cross_view_merge"):
        InferenceDeploymentBoundary(
            policy_kind="merge_policy",
            metric_namespace="merged_inference",
            adapter_path="yolo_agent.components.adapters.inference.policy",
            latency_risk="medium",
            export_compatibility_risk="post-export",
            nms_free_compatibility="requires_allow_cross_view_merge",
            one_to_one_guard_required=False,
        )


def test_standard_640_namespace_must_be_preserved() -> None:
    with pytest.raises(pydantic.ValidationError, match="640"):
        InferenceDeploymentBoundary(
            policy_kind="test_time_augmentation",
            metric_namespace="tta_inference",
            adapter_path="yolo_agent.components.adapters.inference.policy",
            latency_risk="medium",
            export_compatibility_risk="post-export",
            nms_free_compatibility="native_one_to_one",
            one_to_one_guard_required=False,
            standard_640_namespace_preserved=False,
        )


def test_training_graph_never_imports_inference_adapters() -> None:
    """The deployment boundary in the import graph, not just prose."""

    training_graph_packages = (
        "yolo_agent/components/adapters/distillation",
        "yolo_agent/components/adapters/domain_adaptation",
        "yolo_agent/components/adapters/losses",
        "yolo_agent/components/adapters/assigners",
        "yolo_agent/components/adapters/neck",
        "yolo_agent/components/adapters/heads",
        "yolo_agent/components/adapters/data_pipeline",
        "yolo_agent/components/training_control.py",
        "yolo_agent/components/teacher_ema.py",
        "yolo_agent/components/pseudo_label_filter.py",
        "yolo_agent/components/distillation",
    )
    inference_root = "yolo_agent.components.adapters.inference"
    violations: list[str] = []
    for package in training_graph_packages:
        package_path = Path(package)
        files = (
            [package_path]
            if package_path.is_file()
            else sorted(package_path.rglob("*.py"))
        )
        for path in files:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    if name.startswith(inference_root):
                        violations.append(f"{path} -> {name}")
    assert not violations, f"training graph imports inference adapters: {violations}"


def test_record_validation_keeps_ready_strict() -> None:
    boundary = InferenceDeploymentBoundary(
        policy_kind="class_aware_thresholding",
        metric_namespace="class_threshold_inference",
        adapter_path="yolo_agent.components.adapters.inference.policy",
        latency_risk="low",
        export_compatibility_risk="post-export",
        nms_free_compatibility="native_one_to_one",
        one_to_one_guard_required=False,
    )
    probes = _behavior_probes()
    base = dict(
        paper_id="arxiv:9999.99999",
        title="Hypothetical inference paper",
        year=2025,
        primary_domain="postprocess",
        in_scope=True,
        scope_reason="frozen plan declares an inference_policy insertion point",
        mechanism_id="inference.hypothetical",
        policy_kind="class_aware_thresholding",
        parameter_semantics="per-class thresholds from validation PR curves",
        evidence_refs=["frozen plan"],
        deployment_boundary=boundary,
        behavior=probes,
    )
    record = PaperInferenceSideRecord.model_validate({**base, "status": "ready"})
    assert record.status == "ready"
    # Ready requires real behavior evidence.
    with pytest.raises(pydantic.ValidationError, match="behavior"):
        PaperInferenceSideRecord.model_validate(
            {
                **base,
                "status": "ready",
                "behavior": probes.model_copy(
                    update={"passed": False, "errors": ["synthetic failure"]}
                ),
            }
        )
    # Ready requires per-paper parameter semantics.
    with pytest.raises(pydantic.ValidationError, match="semantics"):
        PaperInferenceSideRecord.model_validate(
            {**base, "status": "ready", "parameter_semantics": ""}
        )
