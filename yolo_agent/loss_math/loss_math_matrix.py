"""Reusable degenerate-input math matrix for paper-specific losses.

Every in-scope loss must produce a finite output and a finite autograd
gradient on each constructed case: identical boxes, non-overlapping boxes,
tiny boxes, degenerate (zero-area) boxes, extreme aspect ratios, batches
without positives, and mixed batches.  The matrix is data, so a new paper
loss inherits the whole suite by calling :func:`evaluate_loss_math_matrix`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from yolo_agent.components.auxiliary_losses import AuxiliaryLossInputs


@dataclass(frozen=True)
class LossMathCase:
    """One named input regime for the math matrix.

    The matrix asserts finiteness, autograd, and gradient health only.
    Mechanism-specific expectations (for example "no positives means zero
    loss") are paper property tests because they differ by design: BPC
    deliberately penalizes confident background while the guarded quality
    losses return exactly zero.
    """

    name: str
    inputs: AuxiliaryLossInputs
    expects_positive_loss: bool = False


def _make(
    boxes: list[list[float]],
    *,
    classes: list[int] | None = None,
    anchors: list[list[float]] | None = None,
    logits_value: float = 1.2,
) -> tuple[Any, ...]:
    """Build a one-image batch with one candidate per given box layout.

    ``boxes`` holds predicted boxes; targets repeat the first box so IoU
    quality is well defined.  ``classes`` selects the target class per
    candidate row; an empty ``classes`` means no positives at all.
    """

    count = max(len(boxes), 1)
    predicted = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    predicted = torch.cat(
        [predicted, predicted.new_zeros(count - len(boxes), 4)], dim=0
    ).unsqueeze(0)
    target_boxes = torch.tensor(boxes[:1] or [[0.4, 0.4, 0.5, 0.5]], dtype=torch.float32)
    target_boxes = target_boxes.reshape(-1, 4).expand(count, 4).unsqueeze(0).contiguous()
    logits = torch.full((1, count, 3), logits_value, dtype=torch.float32)
    if classes:
        for index, class_id in enumerate(classes):
            logits[0, index, :] = logits_value
            logits[0, index, class_id] = logits_value + 1.5
    mask = torch.zeros((1, count), dtype=torch.bool)
    if classes:
        mask[0, : len(classes)] = True
    anchor = torch.tensor(anchors or [[0.45, 0.45]] * count, dtype=torch.float32)
    target_classes = torch.full((1, count), -1.0, dtype=torch.float32)
    for index, class_id in enumerate(classes or []):
        target_classes[0, index] = float(class_id)
    return logits, predicted, target_boxes, target_classes, mask, anchor


def identical_boxes_case() -> LossMathCase:
    logits, predicted, target, classes, mask, anchor = _make(
        [[0.3, 0.3, 0.5, 0.5]], classes=[1], anchors=[[0.4, 0.4]]
    )
    return LossMathCase(
        name="identical_boxes",
        inputs=AuxiliaryLossInputs(
            class_logits=logits,
            predicted_boxes_xyxy=predicted,
            target_boxes_xyxy=target,
            target_classes=classes,
            foreground_mask=mask,
            anchor_points_xy=anchor,
        ),
    )


def non_overlapping_boxes_case() -> LossMathCase:
    logits, predicted, target, classes, mask, anchor = _make(
        [[0.0, 0.0, 0.2, 0.2]], classes=[1], anchors=[[0.1, 0.1]]
    )
    target[0, 0] = torch.tensor([0.7, 0.7, 0.95, 0.95])
    return LossMathCase(
        name="non_overlapping_boxes",
        inputs=AuxiliaryLossInputs(
            class_logits=logits,
            predicted_boxes_xyxy=predicted,
            target_boxes_xyxy=target,
            target_classes=classes,
            foreground_mask=mask,
            anchor_points_xy=anchor,
        ),
    )


def tiny_boxes_case() -> LossMathCase:
    logits, predicted, target, classes, mask, anchor = _make(
        [[0.5, 0.5, 0.5 + 2e-4, 0.5 + 2e-4]], classes=[2], anchors=[[0.5, 0.5]]
    )
    return LossMathCase(
        name="tiny_boxes",
        inputs=AuxiliaryLossInputs(
            class_logits=logits,
            predicted_boxes_xyxy=predicted,
            target_boxes_xyxy=target,
            target_classes=classes,
            foreground_mask=mask,
            anchor_points_xy=anchor,
        ),
    )


def degenerate_boxes_case() -> LossMathCase:
    """A zero-width target: IoU is zero but every tensor stays finite."""

    logits, predicted, target, classes, mask, anchor = _make(
        [[0.3, 0.3, 0.6, 0.6]], classes=[0], anchors=[[0.45, 0.45]]
    )
    target[0, 0] = torch.tensor([0.4, 0.4, 0.4, 0.7])
    return LossMathCase(
        name="degenerate_boxes",
        inputs=AuxiliaryLossInputs(
            class_logits=logits,
            predicted_boxes_xyxy=predicted,
            target_boxes_xyxy=target,
            target_classes=classes,
            foreground_mask=mask,
            anchor_points_xy=anchor,
        ),
    )


def extreme_aspect_ratio_case() -> LossMathCase:
    logits, predicted, target, classes, mask, anchor = _make(
        [[0.0, 0.45, 1.0, 0.55]], classes=[1], anchors=[[0.5, 0.5]]
    )
    return LossMathCase(
        name="extreme_aspect_ratio",
        inputs=AuxiliaryLossInputs(
            class_logits=logits,
            predicted_boxes_xyxy=predicted,
            target_boxes_xyxy=target,
            target_classes=classes,
            foreground_mask=mask,
            anchor_points_xy=anchor,
        ),
    )


def zero_positives_case() -> LossMathCase:
    logits, predicted, target, classes, mask, anchor = _make(
        [[0.3, 0.3, 0.5, 0.5]], classes=[], anchors=[[0.4, 0.4]]
    )
    return LossMathCase(
        name="zero_positives",
        inputs=AuxiliaryLossInputs(
            class_logits=logits,
            predicted_boxes_xyxy=predicted,
            target_boxes_xyxy=target,
            target_classes=classes,
            foreground_mask=mask,
            anchor_points_xy=anchor,
        ),
    )


def mixed_batch_case() -> LossMathCase:
    """Genuinely mixed: a good match, a far-off box, a wrong class, a tiny box."""

    logits, predicted, target, classes, mask, anchor = _make(
        [
            [0.30, 0.30, 0.52, 0.52],
            [0.40, 0.40, 0.60, 0.60],
            [0.49, 0.49, 0.5005, 0.5005],
            [0.0, 0.0, 0.05, 0.5],
        ],
        classes=[1, 2, 0, 1],
        anchors=[[0.41, 0.41], [0.5, 0.5], [0.495, 0.495], [0.02, 0.25]],
    )
    target[0, 1] = torch.tensor([0.88, 0.88, 0.98, 0.98])
    target[0, 2] = torch.tensor([0.49, 0.49, 0.5005, 0.5005])
    target[0, 3] = torch.tensor([0.0, 0.0, 0.05, 0.5])
    logits[0, 3, :] = 1.2
    logits[0, 3, 2] = 2.7
    logits[0, 3, 1] = 0.1
    return LossMathCase(
        name="mixed_batch",
        inputs=AuxiliaryLossInputs(
            class_logits=logits,
            predicted_boxes_xyxy=predicted,
            target_boxes_xyxy=target,
            target_classes=classes,
            foreground_mask=mask,
            anchor_points_xy=anchor,
        ),
        expects_positive_loss=True,
    )


def _all_cases() -> tuple[LossMathCase, ...]:
    return (
        identical_boxes_case(),
        non_overlapping_boxes_case(),
        tiny_boxes_case(),
        degenerate_boxes_case(),
        extreme_aspect_ratio_case(),
        zero_positives_case(),
        mixed_batch_case(),
    )


LOSS_MATH_CASES: tuple[str, ...] = tuple(item.name for item in _all_cases())


@dataclass(frozen=True)
class LossMathCaseResult:
    case: str
    finite_output: bool
    finite_gradient: bool
    backward_ran: bool
    loss_value: float
    positive_count: float
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors


def evaluate_loss_math_matrix(
    loss_name: str,
    *,
    build_loss: Callable[[str], Any],
) -> list[LossMathCaseResult]:
    """Run every math case against one auxiliary loss plugin.

    The callable receives ``loss_name`` and must return an
    :class:`AuxiliaryLossPlugin`.  Gradients are taken with respect to the
    candidate class logits and predicted boxes, mirroring runtime autograd.
    """

    plugin = build_loss(loss_name)
    results: list[LossMathCaseResult] = []
    for case in _all_cases():
        errors: list[str] = []
        logits, predicted, target, classes, mask, anchor = (
            case.inputs.class_logits,
            case.inputs.predicted_boxes_xyxy,
            case.inputs.target_boxes_xyxy,
            case.inputs.target_classes,
            case.inputs.foreground_mask,
            case.inputs.anchor_points_xy,
        )
        logits = logits.detach().clone().requires_grad_(True)
        predicted = predicted.detach().clone().requires_grad_(True)
        inputs = AuxiliaryLossInputs(
            class_logits=logits,
            predicted_boxes_xyxy=predicted,
            target_boxes_xyxy=target,
            target_classes=classes,
            foreground_mask=mask,
            anchor_points_xy=anchor,
        )
        backward_ran = False
        try:
            output = plugin.compute(inputs)
            loss = output.loss
            finite_output = bool(torch.isfinite(loss.detach()).all())
            if finite_output:
                loss.backward()
                backward_ran = True
            else:
                errors.append("non_finite_output")
            if logits.grad is None and predicted.grad is None:
                errors.append("missing_autograd_gradient")
                finite_gradient = False
            else:
                finite_gradient = True
                for gradient in (logits.grad, predicted.grad):
                    if gradient is not None and not bool(torch.isfinite(gradient).all()):
                        finite_gradient = False
                        errors.append("non_finite_gradient")
        except (AssertionError, RuntimeError, TypeError, ValueError) as exc:
            finite_output = False
            finite_gradient = False
            errors.append(f"{type(exc).__name__}:{exc}")
            loss = torch.tensor(float("nan"))
            loss_value = float("nan")
            results.append(
                LossMathCaseResult(
                    case=case.name,
                    finite_output=False,
                    finite_gradient=False,
                    backward_ran=False,
                    loss_value=float("nan"),
                    positive_count=float("nan"),
                    errors=errors,
                )
            )
            continue
        loss_value = float(loss.detach().reshape(-1)[0])
        if case.expects_positive_loss and not loss_value > 0.0:
            errors.append("expected_positive_loss")
        results.append(
            LossMathCaseResult(
                case=case.name,
                finite_output=finite_output,
                finite_gradient=finite_gradient,
                backward_ran=backward_ran,
                loss_value=loss_value,
                positive_count=0.0,
                errors=errors,
            )
        )
    return results


__all__ = [
    "LOSS_MATH_CASES",
    "LossMathCase",
    "LossMathCaseResult",
    "evaluate_loss_math_matrix",
]
