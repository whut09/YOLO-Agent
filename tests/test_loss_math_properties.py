"""Paper-specific mathematical property tests for in-scope losses.

These tests go beyond the shared finiteness matrix: each in-scope loss must
behave the way its formula provenance says it behaves, including the
zero-positive guard, the direction of change when geometry or confidence
changes, and the refusal to become a CIoU weight alias.
"""

from __future__ import annotations

import pytest
import torch

from yolo_agent.components.auxiliary_losses import (
    AuxiliaryLossInputs,
    BPCCalibrationAuxiliaryLoss,
    CorrelationAuxiliaryLoss,
    MutualSupervisionAuxiliaryLoss,
    PseudoIoUQualityAuxiliaryLoss,
    build_auxiliary_loss,
)
from yolo_agent.loss_math.formula_provenance import (
    LOSS_FORMULA_PROVENANCE,
    validate_no_loss_alias,
)
from yolo_agent.loss_math.loss_math_matrix import evaluate_loss_math_matrix

IN_SCOPE_LOSSES = (
    "loss.quality.pseudo_iou",
    "loss.2109_05986",
    "loss.quality.correlation",
    "loss.calibration.bpc",
)


def _inputs(
    predicted: list[list[float]],
    target: list[list[float]],
    *,
    classes: list[int] | None = None,
    logits: torch.Tensor | None = None,
    count: int | None = None,
) -> AuxiliaryLossInputs:
    rows = count if count is not None else max(len(predicted), len(target), 1)
    predicted_tensor = torch.tensor(predicted or [[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    if len(predicted_tensor) < rows:
        predicted_tensor = torch.cat(
            [predicted_tensor, predicted_tensor.new_zeros(rows - len(predicted_tensor), 4)]
        )
    predicted_tensor = predicted_tensor.unsqueeze(0)
    target_tensor = torch.tensor(target or [[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    if len(target_tensor) < rows:
        target_tensor = torch.cat(
            [target_tensor, target_tensor.new_zeros(rows - len(target_tensor), 4)]
        )
    target_tensor = target_tensor.unsqueeze(0)
    if logits is None:
        logits = torch.full((1, rows, 3), 1.2)
    mask = torch.zeros((1, rows), dtype=torch.bool)
    target_classes = torch.full((1, rows), -1.0)
    for index, class_id in enumerate(classes or []):
        mask[0, index] = True
        target_classes[0, index] = float(class_id)
    anchor = torch.full((rows, 2), 0.45)
    for index, _ in enumerate(classes or []):
        anchor[index] = 0.45
    return AuxiliaryLossInputs(
        class_logits=logits,
        predicted_boxes_xyxy=predicted_tensor,
        target_boxes_xyxy=target_tensor,
        target_classes=target_classes,
        foreground_mask=mask,
        anchor_points_xy=anchor,
    )


@pytest.mark.parametrize("loss_name", IN_SCOPE_LOSSES)
def test_in_scope_losses_pass_full_math_matrix(loss_name: str) -> None:
    results = evaluate_loss_math_matrix(
        LOSS_FORMULA_PROVENANCE[loss_name].implementation_name,
        build_loss=build_auxiliary_loss,
    )

    failures = [(item.case, item.errors) for item in results if not item.passed]
    assert failures == []
    assert {item.case for item in results} == {
        "identical_boxes",
        "non_overlapping_boxes",
        "tiny_boxes",
        "degenerate_boxes",
        "extreme_aspect_ratio",
        "zero_positives",
        "mixed_batch",
    }


@pytest.mark.parametrize("loss_name", ("pseudo_iou", "mutual_supervision", "correlation"))
def test_guarded_losses_return_exact_zero_without_positives(loss_name: str) -> None:
    plugin = build_auxiliary_loss(loss_name)
    inputs = _inputs(
        [[0.4, 0.4, 0.6, 0.6]], [[0.4, 0.4, 0.6, 0.6]], classes=[], count=2
    )

    output = plugin.compute(inputs)

    assert float(output.loss.detach()) == 0.0


def test_mutual_supervision_pulls_confidence_toward_matched_iou() -> None:
    plugin = MutualSupervisionAuxiliaryLoss()
    target = [[0.4, 0.4, 0.6, 0.6]]
    matched_logits = torch.full((1, 1, 3), 3.0)
    matched_logits[0, 0, 1] = 4.0
    bad_logits = torch.full((1, 1, 3), -3.0)

    matched_loss = float(
        plugin.compute(_inputs([[0.4, 0.4, 0.6, 0.6]], target, classes=[1], logits=matched_logits)).loss
    )
    bad_loss = float(
        plugin.compute(_inputs([[0.4, 0.4, 0.6, 0.6]], target, classes=[1], logits=bad_logits)).loss
    )

    # Perfect box with high true-class confidence: the confidence->IoU term
    # is near zero.  Same box with low confidence must cost strictly more.
    assert bad_loss > matched_loss
    assert matched_loss >= 0.0


def test_mutual_supervision_localization_branch_penalizes_worse_boxes() -> None:
    plugin = MutualSupervisionAuxiliaryLoss()
    target = [[0.4, 0.4, 0.6, 0.6]]
    logits = torch.full((1, 1, 3), 3.0)
    logits[0, 0, 1] = 4.0

    good = float(
        plugin.compute(_inputs([[0.41, 0.41, 0.61, 0.61]], target, classes=[1], logits=logits)).loss
    )
    bad = float(
        plugin.compute(_inputs([[0.0, 0.0, 0.2, 0.2]], target, classes=[1], logits=logits)).loss
    )

    assert bad > good


def test_pseudo_iou_target_follows_anchor_point_distance() -> None:
    plugin = PseudoIoUQualityAuxiliaryLoss()
    target = [[0.4, 0.4, 0.6, 0.6]]
    near = torch.full((1, 1, 3), 3.0)
    near[0, 0, 1] = 4.0
    far = near.clone()

    near_inputs = _inputs([[0.4, 0.4, 0.6, 0.6]], target, classes=[1], logits=near)
    near_inputs = AuxiliaryLossInputs(
        class_logits=near_inputs.class_logits,
        predicted_boxes_xyxy=near_inputs.predicted_boxes_xyxy,
        target_boxes_xyxy=near_inputs.target_boxes_xyxy,
        target_classes=near_inputs.target_classes,
        foreground_mask=near_inputs.foreground_mask,
        anchor_points_xy=torch.tensor([[0.45, 0.45]]),
    )
    far_inputs = _inputs([[0.4, 0.4, 0.6, 0.6]], target, classes=[1], logits=far)
    far_inputs = AuxiliaryLossInputs(
        class_logits=far_inputs.class_logits,
        predicted_boxes_xyxy=far_inputs.predicted_boxes_xyxy,
        target_boxes_xyxy=far_inputs.target_boxes_xyxy,
        target_classes=far_inputs.target_classes,
        foreground_mask=far_inputs.foreground_mask,
        anchor_points_xy=torch.tensor([[0.9, 0.9]]),
    )

    near_loss = float(plugin.compute(near_inputs).loss)
    far_loss = float(plugin.compute(far_inputs).loss)

    # Anchor farther from the GT center lowers the pseudo-IoU target, so the
    # same high-confidence logit is penalized more.
    assert far_loss > near_loss


def test_pseudo_iou_regression_gradient_stays_bounded_for_tiny_boxes() -> None:
    plugin = PseudoIoUQualityAuxiliaryLoss()
    logits = torch.full((1, 1, 3), 2.0)
    logits = logits + torch.tensor([[[0.0, 0.0, 1.0]]])
    logits.requires_grad_(True)
    predicted = torch.tensor([[[0.5, 0.5, 0.5 + 1e-3, 0.5 + 1e-3]]], requires_grad=True)
    target = torch.tensor([[[0.5, 0.5, 0.5 + 1e-3, 0.5 + 1e-3]]])
    inputs = AuxiliaryLossInputs(
        class_logits=logits,
        predicted_boxes_xyxy=predicted,
        target_boxes_xyxy=target,
        target_classes=torch.tensor([[2.0]]),
        foreground_mask=torch.tensor([[True]]),
        anchor_points_xy=torch.tensor([[0.5, 0.5]]),
    )

    output = plugin.compute(inputs)
    output.loss.backward()

    assert torch.isfinite(output.loss).all()
    # The pseudo-IoU target is anchor/GT geometry and detached by design, so the
    # only differentiable path is the classification logits.
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().max() < 1e3
    assert predicted.grad is None  # prediction-independent quality target


def test_pseudo_iou_target_is_prediction_independent() -> None:
    """Paper property: pseudo-IoU uses anchor/GT overlap, not predictions."""
    plugin = PseudoIoUQualityAuxiliaryLoss()
    logits = torch.full((1, 1, 3), 2.0)
    target = torch.tensor([[[0.4, 0.4, 0.6, 0.6]]])
    anchor = torch.tensor([[0.5, 0.5]])
    common = dict(
        class_logits=logits,
        target_boxes_xyxy=target,
        target_classes=torch.tensor([[2.0]]),
        foreground_mask=torch.tensor([[True]]),
        anchor_points_xy=anchor,
    )
    close = AuxiliaryLossInputs(
        predicted_boxes_xyxy=torch.tensor([[[0.41, 0.41, 0.59, 0.59]]]), **common
    )
    far = AuxiliaryLossInputs(
        predicted_boxes_xyxy=torch.tensor([[[0.0, 0.0, 0.1, 0.1]]]), **common
    )
    assert float(plugin.compute(close).loss) == pytest.approx(
        float(plugin.compute(far).loss), abs=1e-6
    )


def test_correlation_increases_as_confidence_and_iou_disagree() -> None:
    plugin = CorrelationAuxiliaryLoss()
    target = [[0.4, 0.4, 0.6, 0.6], [0.4, 0.4, 0.6, 0.6]]
    predicted = [[0.4, 0.4, 0.6, 0.6], [0.0, 0.0, 0.2, 0.2]]
    aligned = torch.tensor(
        [[[3.0, 4.0, 0.0], [3.0, 3.4, 0.0]]]
    )
    misaligned = torch.tensor(
        [[[3.0, 3.4, 0.0], [3.0, 4.0, 0.0]]]
    )

    aligned_loss = float(
        plugin.compute(_inputs(predicted, target, classes=[1, 1], logits=aligned)).loss
    )
    misaligned_loss = float(
        plugin.compute(_inputs(predicted, target, classes=[1, 1], logits=misaligned)).loss
    )

    # When high confidence aligns with high IoU, concordance is higher and
    # the loss is lower than the swapped (anti-correlated) configuration.
    assert misaligned_loss > aligned_loss


def test_correlation_requires_two_positives_and_is_autograd_safe() -> None:
    plugin = CorrelationAuxiliaryLoss()
    logits = torch.full((1, 2, 3), 1.0)
    inputs = _inputs(
        [[0.4, 0.4, 0.6, 0.6], [0.41, 0.41, 0.61, 0.61]],
        [[0.4, 0.4, 0.6, 0.6], [0.4, 0.4, 0.6, 0.6]],
        classes=[1, 1],
        logits=logits,
    )
    logits = logits.clone().requires_grad_(True)
    inputs = AuxiliaryLossInputs(
        class_logits=logits,
        predicted_boxes_xyxy=inputs.predicted_boxes_xyxy,
        target_boxes_xyxy=inputs.target_boxes_xyxy,
        target_classes=inputs.target_classes,
        foreground_mask=inputs.foreground_mask,
        anchor_points_xy=inputs.anchor_points_xy,
    )

    output = plugin.compute(inputs)
    output.loss.backward()

    assert output.metrics["positive_count"] == 2.0
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_bpc_confident_but_inaccurate_candidates_raise_the_loss() -> None:
    plugin = BPCCalibrationAuxiliaryLoss()
    target = [[0.4, 0.4, 0.6, 0.6]]
    accurate_confident = torch.full((1, 1, 3), 4.0)
    accurate_confident[0, 0, 1] = 5.0
    inaccurate_confident = torch.full((1, 1, 3), 4.0)
    inaccurate_confident[0, 0, 1] = 5.0

    good = float(
        plugin.compute(
            _inputs([[0.4, 0.4, 0.6, 0.6]], target, classes=[1], logits=accurate_confident)
        ).loss
    )
    bad = float(
        plugin.compute(
            _inputs([[0.0, 0.0, 0.2, 0.2]], target, classes=[1], logits=inaccurate_confident)
        ).loss
    )

    assert bad > good


def test_bpc_quadrant_counts_are_reported_and_bounded() -> None:
    plugin = BPCCalibrationAuxiliaryLoss()
    logits = torch.tensor([[[4.0, 5.0, 0.0], [4.0, 0.0, 5.0]]])
    inputs = _inputs(
        [[0.4, 0.4, 0.6, 0.6], [0.0, 0.0, 0.2, 0.2]],
        [[0.4, 0.4, 0.6, 0.6], [0.4, 0.4, 0.6, 0.6]],
        classes=[1, 1],
        logits=logits,
    )

    output = plugin.compute(inputs)

    assert output.metrics["accurate_confident"] == 1.0
    assert output.metrics["inaccurate_confident"] == 1.0
    assert 0.0 <= float(output.loss.detach()) <= float("inf")


def test_loss_provenance_registry_covers_every_in_scope_mechanism() -> None:
    for mechanism_id in IN_SCOPE_LOSSES:
        provenance = LOSS_FORMULA_PROVENANCE[mechanism_id]
        assert provenance.formula.strip()
        assert provenance.source_anchor.strip()
        assert provenance.parameter_semantics or provenance.formula_family in {
            "pseudo_iou",
            "mutual_supervision",
            "concordance_correlation",
        }
        assert provenance.shared_primitives, "IoU geometry may be shared, but must be declared"


def test_wiou_mpdiou_nwd_are_not_registered_as_paper_losses() -> None:
    # These mechanism families have no in-scope paper in the frozen 83.  They
    # must not appear in the provenance registry as paper-specific losses.
    assert "loss.bbox.wiou" not in LOSS_FORMULA_PROVENANCE
    assert "loss.bbox.mpdiou" not in LOSS_FORMULA_PROVENANCE
    assert "loss.bbox.nwd" not in LOSS_FORMULA_PROVENANCE


def test_alias_guard_rejects_weight_only_implementations() -> None:
    class WeightOnlyCIoUAlias:
        DISTINGUISHING_TERMS: tuple[str, ...] = ()

    with pytest.raises(ValueError, match="refusing to treat it as a new loss"):
        validate_no_loss_alias("loss.quality.correlation", WeightOnlyCIoUAlias())

    class RealCorrelation:
        DISTINGUISHING_TERMS = ("batch_moment_alignment",)

    validate_no_loss_alias("loss.quality.correlation", RealCorrelation())


def test_alias_guard_requires_registered_mechanism() -> None:
    with pytest.raises(KeyError):
        validate_no_loss_alias("loss.bbox.unknown_mechanism", object())
