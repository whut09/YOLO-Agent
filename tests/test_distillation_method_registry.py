"""Distillation method registry and 32-paper coverage. No GPU training."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from yolo_agent.components.adapters.distillation.method_registry import (
    BRANCH_TO_MECHANISM,
    CERTIFIED_DISTILLATION_PAPERS,
    DistillationMethodRegistry,
    DistillationTeacherMissingError,
)
from yolo_agent.components.distillation.mechanism_losses import (
    DistillationInputs,
    build_distillation_mechanism_loss,
)


def test_registry_has_eleven_independent_branches() -> None:
    registry = DistillationMethodRegistry()
    branches = registry.branches()
    assert len(branches) == 11
    assert {item.branch_id for item in branches} == set(BRANCH_TO_MECHANISM)
    fingerprints = {item.execution_fingerprint for item in branches}
    assert len(fingerprints) == 11
    variables = {tuple(sorted(item.changed_variables)) for item in branches}
    assert len(variables) == 11
    modes = {item.loss_mode for item in branches}
    assert len(modes) == 11
    for branch in branches:
        assert branch.export_teacher is False
        assert branch.measure_student_only is True
        assert branch.allow_dfl is False
        assert branch.allow_head_replacement is False
        assert branch.requires_teacher_checkpoint is True
        assert "teacher_checkpoint" in branch.evidence_schema
        assert branch.teacher_protocol["export_forbidden"] is True
        assert branch.student_protocol["export_and_measure"] == "student_only"


def test_missing_teacher_is_not_silent() -> None:
    branch = DistillationMethodRegistry().get("logits_distillation")
    with pytest.raises(DistillationTeacherMissingError, match="silent skip is forbidden"):
        branch.require_teacher_checkpoint(None)


def test_certified_distillation_papers_have_branch_or_blocker() -> None:
    coverage = DistillationMethodRegistry().coverage()
    assert coverage.papers_total == 32
    assert len(CERTIFIED_DISTILLATION_PAPERS) == 32
    assert coverage.silent_drops == []
    assert coverage.assigned + coverage.implementation_request == 32
    assert {item.paper_id for item in coverage.assignments} == set(CERTIFIED_DISTILLATION_PAPERS)
    for item in coverage.assignments:
        if item.disposition == "assigned":
            assert item.branch_id in BRANCH_TO_MECHANISM
            assert item.execution_fingerprint
        else:
            assert item.disposition == "implementation_request"
            assert item.reason_codes


@pytest.mark.parametrize("branch_id,mechanism", list(BRANCH_TO_MECHANISM.items()))
def test_each_branch_has_cpu_shape_backward_and_zero_weight(
    branch_id: str,
    mechanism: str,
) -> None:
    student_logits = torch.randn(2, 4, 7, requires_grad=True)
    teacher_logits = torch.randn(2, 4, 7)
    student_features = [torch.randn(2, 5, 8, 8, requires_grad=True)]
    teacher_features = [torch.randn(2, 5, 8, 8)]
    if mechanism == "teacher_ensemble":
        teacher_logits = [torch.randn(2, 4, 7), torch.randn(2, 4, 7)]
    inputs = DistillationInputs(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        student_features=student_features,
        teacher_features=teacher_features,
        student_boxes=torch.randn(2, 4, 7, requires_grad=True),
        teacher_boxes=torch.randn(2, 4, 7),
    )
    options = {"class_dim": 1} if mechanism in {
        "logits",
        "quality_aware",
        "teacher_ensemble",
        "source_free_teacher",
        "cross_domain_teacher",
    } else {}
    loss_fn = build_distillation_mechanism_loss(mechanism, **options)
    output = loss_fn.compute(inputs)
    assert output.loss.ndim == 0
    output.loss.backward()
    has_grad = (
        student_logits.grad is not None
        or any(item.grad is not None for item in student_features)
        or inputs.student_boxes.grad is not None
    )
    assert has_grad
    zero = build_distillation_mechanism_loss(mechanism, **options)
    student_logits.grad = None
    for item in student_features:
        item.grad = None
    scaled = zero.compute(inputs).loss * 0.0
    if scaled.requires_grad:
        scaled.backward()
    assert student_logits.grad is None or float(student_logits.grad.abs().sum()) == 0.0
    assert DistillationMethodRegistry().get(branch_id).loss_mode == branch_id


def test_coverage_fixture_matches_live_registry() -> None:
    payload = yaml.safe_load(
        Path("tests/fixtures/distillation_paper_coverage.yaml").read_text(encoding="utf-8")
    )
    live = DistillationMethodRegistry().coverage()
    assert payload["papers_total"] == 32
    assert payload["assigned"] == live.assigned
    assert payload["implementation_request"] == live.implementation_request
    assert {item["paper_id"] for item in payload["assignments"]} == {
        item.paper_id for item in live.assignments
    }
