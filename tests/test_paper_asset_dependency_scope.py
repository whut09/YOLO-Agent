"""Regression tests for exact paper asset dependency propagation."""

from __future__ import annotations

from types import SimpleNamespace

from yolo_agent.certification.paper_readiness import _manifest_result
from yolo_agent.core.paper_training_readiness import _required_asset_blocker
from yolo_agent.research.paper_execution_requirement_schemas import (
    PaperExecutionRequirement,
)
from yolo_agent.research.paper_training_cohort import _asset_blocker
from yolo_agent.research.paper_asset_dependencies import (
    asset_scope_violations,
    is_inference_only,
    requires_domain_assets,
    requires_graph_config,
    requires_hard_negative_replay,
    requires_teacher_checkpoint,
)


def test_protocol_manifests_do_not_imply_hard_negative_replay() -> None:
    assert not requires_hard_negative_replay(
        ["distillation.feature", "teacher_student_dataset_manifest"]
    )
    assert not requires_hard_negative_replay(
        ["domain_adaptation.feature_alignment", "target_domain_manifest"]
    )
    assert requires_hard_negative_replay(["sampling.hard_negative_replay"])


def test_teacher_and_domain_assets_require_their_own_mechanism() -> None:
    assert requires_teacher_checkpoint(["distillation.feature"])
    assert requires_teacher_checkpoint(["cross_domain_teacher"])
    assert not requires_teacher_checkpoint(["domain_adaptation.feature_alignment"])
    assert requires_domain_assets(["domain_adaptation.feature_alignment"])
    assert requires_domain_assets(["feature_alignment"])
    assert not requires_domain_assets(["distillation.feature"])


def test_scope_invariant_reports_cross_family_contamination() -> None:
    assert asset_scope_violations(
        ["distillation.feature"],
        manifest_assets=["hard_negative_manifest"],
    ) == [
        "hard_negative_manifest_without_sampling.hard_negative_replay"
    ]
    assert asset_scope_violations(
        ["domain_adaptation.feature_alignment"],
        teacher_assets=["frozen_teacher_checkpoint"],
    ) == ["teacher_assets_without_teacher_mechanism"]
    assert asset_scope_violations(
        ["assigner.task_aligned"],
        graph_assets=["graph_identity"],
    ) == ["graph_assets_without_graph_mechanism"]


def test_assignment_is_not_a_graph_asset_dependency() -> None:
    assert not requires_graph_config(["assigner.task_aligned"])
    assert requires_graph_config(["neck.rtmdet_large_kernel"])
    assert requires_graph_config(["attention.spatial"])
    assert is_inference_only(["inference.sahi_slicing"])


def test_distillation_dataset_manifest_does_not_trigger_replay_gates() -> None:
    requirement = PaperExecutionRequirement(
        paper_id="fixture:distillation",
        paper_specific_mechanism="distillation.feature",
        paper_specific_mechanism_ids=["distillation.feature"],
        execution_route="training",
        required_adapter="distillation.feature",
        required_changed_variables=["loss.distillation.feature.weight"],
        required_runtime_payload={"loss_mode": "feature"},
        required_manifest_assets=["teacher_student_dataset_manifest"],
        compatible_with_yolo26=True,
        training_candidate_allowed=True,
        recovery_action="none",
        current_disposition="runtime_ready",
        protocol_hash="a" * 64,
        execution_fingerprint="b" * 64,
    )
    item = SimpleNamespace(
        canonical_component_ids=["distillation.yolo26_teacher_student"],
        paper_specific_mechanism_ids=["feature_distillation"],
    )
    asset = SimpleNamespace(
        teacher_checkpoint="C:/teacher.pt",
        teacher_sha256="c" * 64,
        source_dataset_manifest="C:/source.yaml",
        target_dataset_manifest=None,
        hard_negative_manifest=None,
    )
    assert _required_asset_blocker(item=item, requirement=requirement, asset=asset) is None
    assert _asset_blocker(item, requirement, asset) is None

    record = SimpleNamespace(
        canonical_component_ids=["distillation.yolo26_teacher_student"],
        paper_specific_mechanism_ids=["feature_distillation"],
        required_evidence=[],
    )
    result = _manifest_result(
        record,
        requirement=requirement,
        asset_record=None,
        strict_assets=False,
    )
    assert result.passed is True
    assert result.status == "not_applicable"
