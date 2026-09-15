"""Pure-torch task-aligned assignment reference profiles.

This module exists so paper-assignment difference tests can compare a paper
assigner against a task-aligned baseline without importing the training
runtime.  It is a *reference implementation*: it is not registered as a
runtime component.  Two profiles share one metric core:

- the YOLOv8-family baseline profile (alpha=0.5, beta=6.0, topk=10, raw-IoU
  quality targets), used as the comparison baseline;
- the PP-YOLOE paper profile (alpha=1.0, beta=6.0, topk=13,
  alignment-normalized soft targets), which is the faithful adaptation of
  arxiv:2203.16250 Sec. 3.2 recorded in the provenance registry.

The profiles differ exactly in the hyperparameters and target construction
the paper specifies, so a disagreement test proves the paper configuration
changes the assignment rather than aliasing the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from yolo_agent.components.assignment import AssignerInputs, _targets_from_matches


@dataclass(frozen=True)
class TALReferenceParams:
    """Task-alignment profile parameters shared by both reference profiles."""

    topk: int = 10
    alpha: float = 0.5
    beta: float = 6.0
    normalize_targets: bool = False


def ppyoloe_tal_profile() -> TALReferenceParams:
    """PP-YOLOE paper profile from arxiv:2203.16250 Sec. 3.2."""

    return TALReferenceParams(
        topk=13,
        alpha=1.0,
        beta=6.0,
        normalize_targets=True,
    )


def yolo_baseline_profile() -> TALReferenceParams:
    """YOLOv8-family baseline profile used as the comparison baseline."""

    return TALReferenceParams()


def _pairwise_iou(gt_boxes: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0) * (
        gt_boxes[:, 3] - gt_boxes[:, 1]
    ).clamp(min=0)
    pred_area = (predicted[..., 2] - predicted[..., 0]).clamp(min=0) * (
        predicted[..., 3] - predicted[..., 1]
    ).clamp(min=0)
    lt = torch.maximum(gt_boxes[:, None, :2], predicted[None, :, :2])
    rb = torch.minimum(gt_boxes[:, None, 2:], predicted[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = gt_area[:, None] + pred_area[None, :] - inter
    return inter / union.clamp(min=1e-7)


def _points_inside_boxes(
    points: torch.Tensor, boxes: torch.Tensor
) -> torch.Tensor:
    lt = points[None, :, :] - boxes[:, None, :2]
    rb = boxes[:, None, 2:] - points[None, :, :]
    return torch.cat([lt, rb], dim=-1).amin(dim=-1) > 0


def tal_reference_assignment(
    inputs: AssignerInputs,
    params: TALReferenceParams | None = None,
) -> "object":
    """Compute the task-aligned reference assignment for one batch.

    Positive selection uses ``s^alpha * u^beta`` over points inside each GT
    with top-k selection.  The baseline profile writes raw IoU quality into
    the target scores; the PP-YOLOE profile writes the alignment value
    normalized per GT by its maximum (the paper's soft classification
    target).  The returned object is the same ``AssignerOutput`` shape the
    paper plugins produce, so :func:`compare_assignments` works directly.
    """

    active = params or TALReferenceParams()
    batch, anchors, _ = inputs.predicted_scores.shape
    matched = torch.zeros((batch, anchors), dtype=torch.long, device=inputs.predicted_scores.device)
    foreground = torch.zeros((batch, anchors), dtype=torch.bool, device=inputs.predicted_scores.device)
    quality = torch.zeros((batch, anchors), dtype=inputs.predicted_scores.dtype, device=inputs.predicted_scores.device)
    for batch_index in range(batch):
        valid = inputs.gt_mask[batch_index].reshape(-1).bool()
        boxes = inputs.gt_boxes_xyxy[batch_index, valid]
        labels = inputs.gt_labels[batch_index, valid].reshape(-1).long()
        if boxes.numel() == 0:
            continue
        pair_iou = _pairwise_iou(boxes, inputs.predicted_boxes_xyxy[batch_index])
        class_probability = inputs.predicted_scores[batch_index, :, labels].transpose(0, 1)
        alignment = pair_iou.clamp(min=0) ** active.beta * class_probability.clamp(min=0) ** active.alpha
        inside = _points_inside_boxes(inputs.anchor_points_xy, boxes).to(alignment.dtype)
        candidate = alignment * inside
        top_count = min(active.topk, anchors)
        selected = torch.zeros_like(candidate, dtype=torch.bool)
        per_gt_max = candidate.amax(dim=-1, keepdim=True).clamp(min=1e-12)
        for gt_index in range(candidate.shape[0]):
            indices = candidate[gt_index].topk(top_count).indices
            selected[gt_index, indices] = True
        masked = candidate.masked_fill(~selected, 0.0)
        best_score, best_gt = masked.max(dim=0)
        overlap = pair_iou[best_gt, torch.arange(anchors, device=best_gt.device)]
        positive = (best_score > 0) & (overlap > 0)
        matched[batch_index] = best_gt
        foreground[batch_index] = positive
        if active.normalize_targets:
            # PP-YOLOE soft target: normalized alignment of the selected GT row.
            normalized = candidate / per_gt_max
            quality[batch_index, positive] = normalized[
                best_gt[positive], torch.arange(anchors, device=best_gt.device)[positive]
            ].clamp(0, 1)
        else:
            quality[batch_index, positive] = overlap[positive].clamp(0, 1)
    return _targets_from_matches(inputs, matched, foreground, quality)


__all__ = [
    "TALReferenceParams",
    "ppyoloe_tal_profile",
    "tal_reference_assignment",
    "yolo_baseline_profile",
]
