"""Framework-neutral auxiliary loss plugins for detection quality alignment."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AuxiliaryLossInputs:
    """Matched native-detector tensors consumed by auxiliary losses."""

    class_logits: Any
    predicted_boxes_xyxy: Any
    target_boxes_xyxy: Any
    target_classes: Any
    foreground_mask: Any
    anchor_points_xy: Any


@dataclass(frozen=True)
class AuxiliaryLossOutput:
    """One scalar auxiliary term plus detached diagnostics."""

    loss: Any
    metrics: dict[str, float] = field(default_factory=dict)


class AuxiliaryLossPlugin(ABC):
    """Stable API for additive, training-only detector losses."""

    loss_name: str

    @abstractmethod
    def compute(self, inputs: AuxiliaryLossInputs) -> AuxiliaryLossOutput:
        """Return an unweighted scalar without replacing native detector loss."""


class CorrelationAuxiliaryLoss(AuxiliaryLossPlugin):
    """Align true-class confidence with detached localization quality."""

    loss_name = "correlation"

    def __init__(self, *, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon

    def compute(self, inputs: AuxiliaryLossInputs) -> AuxiliaryLossOutput:
        mask = inputs.foreground_mask.bool()
        if int(mask.sum()) < 2:
            return AuxiliaryLossOutput(loss=inputs.class_logits.sum() * 0.0)
        true_logits = _true_class_logits(inputs)[mask]
        scores = true_logits.sigmoid().float()
        quality = _matched_iou(inputs).detach()[mask].float()
        score_delta = scores - scores.mean()
        quality_delta = quality - quality.mean()
        covariance = (score_delta * quality_delta).mean()
        denominator = (
            score_delta.square().mean()
            + quality_delta.square().mean()
            + (scores.mean() - quality.mean()).square()
            + self.epsilon
        )
        concordance = (2.0 * covariance / denominator).clamp(min=-1.0, max=1.0)
        loss = 1.0 - concordance
        return AuxiliaryLossOutput(
            loss=loss,
            metrics={
                "concordance": float(concordance.detach().cpu()),
                "positive_count": float(mask.sum().detach().cpu()),
            },
        )


class BPCCalibrationAuxiliaryLoss(AuxiliaryLossPlugin):
    """BPC-style differentiable precision-confidence quadrant objective."""

    loss_name = "bpc_calibration"

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        max_candidates_per_image: int = 300,
        epsilon: float = 1e-6,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_candidates_per_image = max_candidates_per_image
        self.epsilon = epsilon

    def compute(self, inputs: AuxiliaryLossInputs) -> AuxiliaryLossOutput:
        import torch

        probabilities = inputs.class_logits.sigmoid()
        confidence, predicted_class = probabilities.max(dim=-1)
        target_class = inputs.target_classes.long()
        iou = _matched_iou(inputs).detach()
        accurate = (
            inputs.foreground_mask.bool()
            & predicted_class.detach().eq(target_class)
            & iou.ge(self.iou_threshold)
        )
        selected = _top_candidate_mask(
            confidence.detach(), self.max_candidates_per_image
        )
        confidence = confidence[selected].float()
        accurate = accurate[selected]
        if confidence.numel() == 0:
            return AuxiliaryLossOutput(loss=inputs.class_logits.sum() * 0.0)
        confident = confidence.detach().ge(self.confidence_threshold)
        accurate_confident = accurate & confident
        accurate_not_confident = accurate & ~confident
        inaccurate_confident = ~accurate & confident
        inaccurate_not_confident = ~accurate & ~confident
        high_proxy = torch.tanh(confidence)
        low_proxy = torch.tanh(1.0 - confidence)
        good = (
            high_proxy[accurate_confident].sum()
            + low_proxy[inaccurate_not_confident].sum()
        )
        bad = (
            low_proxy[accurate_not_confident].sum()
            + high_proxy[inaccurate_confident].sum()
        )
        loss = torch.log1p(bad / (good + self.epsilon))
        return AuxiliaryLossOutput(
            loss=loss,
            metrics={
                "accurate_confident": float(accurate_confident.sum().detach().cpu()),
                "accurate_not_confident": float(
                    accurate_not_confident.sum().detach().cpu()
                ),
                "inaccurate_confident": float(
                    inaccurate_confident.sum().detach().cpu()
                ),
                "inaccurate_not_confident": float(
                    inaccurate_not_confident.sum().detach().cpu()
                ),
            },
        )


class PseudoIoUQualityAuxiliaryLoss(AuxiliaryLossPlugin):
    """Use anchor/GT pseudo-IoU as a classification quality target."""

    loss_name = "pseudo_iou"

    def compute(self, inputs: AuxiliaryLossInputs) -> AuxiliaryLossOutput:
        import torch

        mask = inputs.foreground_mask.bool()
        if not bool(mask.any()):
            return AuxiliaryLossOutput(loss=inputs.class_logits.sum() * 0.0)
        logits = _true_class_logits(inputs)[mask].float()
        quality = _pseudo_iou(inputs).detach()[mask].float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, quality)
        return AuxiliaryLossOutput(
            loss=loss,
            metrics={
                "mean_pseudo_iou": float(quality.mean().cpu()),
                "positive_count": float(mask.sum().detach().cpu()),
            },
        )


def build_auxiliary_loss(name: str, **options: Any) -> AuxiliaryLossPlugin:
    """Build an explicitly supported auxiliary loss by stable name."""
    implementations: dict[str, type[AuxiliaryLossPlugin]] = {
        CorrelationAuxiliaryLoss.loss_name: CorrelationAuxiliaryLoss,
        BPCCalibrationAuxiliaryLoss.loss_name: BPCCalibrationAuxiliaryLoss,
        PseudoIoUQualityAuxiliaryLoss.loss_name: PseudoIoUQualityAuxiliaryLoss,
    }
    try:
        implementation = implementations[name]
    except KeyError as exc:
        raise KeyError(f"unknown auxiliary loss: {name}") from exc
    return implementation(**options)


def _true_class_logits(inputs: AuxiliaryLossInputs) -> Any:
    classes = inputs.target_classes.long().clamp(min=0)
    return inputs.class_logits.gather(-1, classes.unsqueeze(-1)).squeeze(-1)


def _matched_iou(inputs: AuxiliaryLossInputs) -> Any:
    return _elementwise_iou(
        inputs.predicted_boxes_xyxy.detach(), inputs.target_boxes_xyxy.detach()
    )


def _pseudo_iou(inputs: AuxiliaryLossInputs) -> Any:
    import torch

    target = inputs.target_boxes_xyxy.detach()
    points = inputs.anchor_points_xy.detach()
    if points.ndim == 2:
        points = points.unsqueeze(0).expand(target.shape[0], -1, -1)
    width = (target[..., 2] - target[..., 0]).clamp(min=0.0)
    height = (target[..., 3] - target[..., 1]).clamp(min=0.0)
    half_size = 0.5 * torch.stack((width, height), dim=-1)
    pseudo = torch.cat((points - half_size, points + half_size), dim=-1)
    return _elementwise_iou(pseudo, target)


def _elementwise_iou(first: Any, second: Any, epsilon: float = 1e-7) -> Any:
    import torch

    intersection_min = torch.maximum(first[..., :2], second[..., :2])
    intersection_max = torch.minimum(first[..., 2:], second[..., 2:])
    intersection_wh = (intersection_max - intersection_min).clamp(min=0.0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    first_wh = (first[..., 2:] - first[..., :2]).clamp(min=0.0)
    second_wh = (second[..., 2:] - second[..., :2]).clamp(min=0.0)
    union = (
        first_wh[..., 0] * first_wh[..., 1]
        + second_wh[..., 0] * second_wh[..., 1]
        - intersection
    )
    return intersection / (union + epsilon)


def _top_candidate_mask(confidence: Any, limit: int) -> Any:
    import torch

    count = min(limit, confidence.shape[1])
    indices = confidence.topk(count, dim=1).indices
    mask = torch.zeros_like(confidence, dtype=torch.bool)
    return mask.scatter(1, indices, True)


__all__ = [
    "AuxiliaryLossInputs",
    "AuxiliaryLossOutput",
    "AuxiliaryLossPlugin",
    "BPCCalibrationAuxiliaryLoss",
    "CorrelationAuxiliaryLoss",
    "PseudoIoUQualityAuxiliaryLoss",
    "build_auxiliary_loss",
]
