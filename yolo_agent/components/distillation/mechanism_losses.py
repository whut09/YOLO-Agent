"""Framework-neutral losses for independent distillation mechanisms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from yolo_agent.components.distillation.losses import channel_agnostic_feature_loss
from yolo_agent.components.distillation.mechanisms import DistillationMechanism


@dataclass(frozen=True)
class DistillationInputs:
    student_logits: Any
    teacher_logits: Any
    student_features: Any | None = None
    teacher_features: Any | None = None
    student_boxes: Any | None = None
    teacher_boxes: Any | None = None


@dataclass(frozen=True)
class DistillationLossOutput:
    loss: Any
    metrics: dict[str, float] = field(default_factory=dict)


class DistillationMechanismLoss(ABC):
    mechanism: DistillationMechanism

    @abstractmethod
    def compute(self, inputs: DistillationInputs) -> DistillationLossOutput:
        """Return one unweighted training-only distillation scalar."""


class LogitsDistillationLoss(DistillationMechanismLoss):
    mechanism = "logits"

    def __init__(self, *, temperature: float = 2.0, class_dim: int = -1) -> None:
        if temperature <= 0.0:
            raise ValueError("distillation temperature must be positive")
        self.temperature = temperature
        self.class_dim = class_dim

    def compute(self, inputs: DistillationInputs) -> DistillationLossOutput:
        import torch

        student, teacher = _same_shape(inputs.student_logits, inputs.teacher_logits)
        temperature = self.temperature
        loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(
                student.float() / temperature,
                dim=self.class_dim,
            ),
            torch.nn.functional.softmax(
                teacher.detach().float() / temperature,
                dim=self.class_dim,
            ),
            reduction="batchmean",
        ) * (temperature**2)
        return DistillationLossOutput(
            loss=loss,
            metrics={"temperature": temperature},
        )


class LocalizationDistillationLoss(DistillationMechanismLoss):
    mechanism = "localization"

    def compute(self, inputs: DistillationInputs) -> DistillationLossOutput:
        import torch

        if inputs.student_boxes is None or inputs.teacher_boxes is None:
            raise ValueError("localization distillation requires student and teacher boxes")
        student, teacher = _same_shape(inputs.student_boxes, inputs.teacher_boxes)
        loss = torch.nn.functional.smooth_l1_loss(
            student.float(),
            teacher.detach().float(),
        )
        return DistillationLossOutput(
            loss=loss,
            metrics={"box_element_count": float(student.numel())},
        )


class FeatureDistillationLoss(DistillationMechanismLoss):
    mechanism = "feature"

    def compute(self, inputs: DistillationInputs) -> DistillationLossOutput:
        if inputs.student_features is None or inputs.teacher_features is None:
            raise ValueError("feature distillation requires student and teacher features")
        loss = channel_agnostic_feature_loss(
            inputs.student_features,
            inputs.teacher_features,
        )
        return DistillationLossOutput(
            loss=loss,
            metrics={"feature_level_count": float(len(_feature_levels(inputs.student_features)))},
        )


class RelationDistillationLoss(DistillationMechanismLoss):
    mechanism = "relation"

    def __init__(self, *, max_spatial_tokens: int = 256) -> None:
        if max_spatial_tokens < 4:
            raise ValueError("relation distillation requires at least four spatial tokens")
        self.max_spatial_tokens = max_spatial_tokens

    def compute(self, inputs: DistillationInputs) -> DistillationLossOutput:
        import torch

        if inputs.student_features is None or inputs.teacher_features is None:
            raise ValueError("relation distillation requires student and teacher features")
        pairs = _feature_pairs(inputs.student_features, inputs.teacher_features)
        losses = []
        observed_tokens = 0
        for student, teacher in pairs:
            student = _bound_spatial_tokens(student.float(), self.max_spatial_tokens)
            teacher = _bound_spatial_tokens(
                teacher.detach().float(), self.max_spatial_tokens
            )
            if student.shape[2:] != teacher.shape[2:]:
                teacher = torch.nn.functional.interpolate(
                    teacher,
                    size=student.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            student_tokens = torch.nn.functional.normalize(
                student.flatten(2), dim=1
            ).transpose(1, 2)
            teacher_tokens = torch.nn.functional.normalize(
                teacher.flatten(2), dim=1
            ).transpose(1, 2)
            observed_tokens = max(observed_tokens, student_tokens.shape[1])
            student_relation = torch.bmm(student_tokens, student_tokens.transpose(1, 2))
            teacher_relation = torch.bmm(teacher_tokens, teacher_tokens.transpose(1, 2))
            losses.append(
                torch.nn.functional.mse_loss(student_relation, teacher_relation)
            )
        return DistillationLossOutput(
            loss=torch.stack(losses).mean(),
            metrics={
                "feature_level_count": float(len(pairs)),
                "max_relation_tokens": float(observed_tokens),
            },
        )


class AttentionDistillationLoss(DistillationMechanismLoss):
    mechanism = "attention"

    def __init__(self, *, channel_bins: int = 32) -> None:
        if channel_bins < 1:
            raise ValueError("attention distillation channel_bins must be positive")
        self.channel_bins = channel_bins

    def compute(self, inputs: DistillationInputs) -> DistillationLossOutput:
        import torch

        if inputs.student_features is None or inputs.teacher_features is None:
            raise ValueError("attention distillation requires student and teacher features")
        pairs = _feature_pairs(inputs.student_features, inputs.teacher_features)
        channel_losses = []
        spatial_losses = []
        for student, teacher in pairs:
            student = student.float()
            teacher = teacher.detach().float()
            student_channel = torch.nn.functional.adaptive_avg_pool1d(
                student.abs().mean(dim=(-2, -1)).unsqueeze(1),
                self.channel_bins,
            ).squeeze(1)
            teacher_channel = torch.nn.functional.adaptive_avg_pool1d(
                teacher.abs().mean(dim=(-2, -1)).unsqueeze(1),
                self.channel_bins,
            ).squeeze(1)
            student_channel = torch.nn.functional.normalize(student_channel, dim=1)
            teacher_channel = torch.nn.functional.normalize(teacher_channel, dim=1)
            channel_losses.append(
                torch.nn.functional.mse_loss(student_channel, teacher_channel)
            )
            student_spatial = student.abs().mean(dim=1, keepdim=True)
            teacher_spatial = teacher.abs().mean(dim=1, keepdim=True)
            if student_spatial.shape[2:] != teacher_spatial.shape[2:]:
                teacher_spatial = torch.nn.functional.interpolate(
                    teacher_spatial,
                    size=student_spatial.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            student_spatial = torch.nn.functional.normalize(
                student_spatial.flatten(1), dim=1
            )
            teacher_spatial = torch.nn.functional.normalize(
                teacher_spatial.flatten(1), dim=1
            )
            spatial_losses.append(
                torch.nn.functional.mse_loss(student_spatial, teacher_spatial)
            )
        channel_loss = torch.stack(channel_losses).mean()
        spatial_loss = torch.stack(spatial_losses).mean()
        return DistillationLossOutput(
            loss=0.5 * (channel_loss + spatial_loss),
            metrics={
                "channel_attention_loss": float(channel_loss.detach().cpu()),
                "spatial_attention_loss": float(spatial_loss.detach().cpu()),
                "feature_level_count": float(len(pairs)),
            },
        )


class MaskedFeatureDistillationLoss(DistillationMechanismLoss):
    mechanism = "masked_feature"

    def __init__(self, *, mask_ratio: float = 0.5) -> None:
        if not 0.0 < mask_ratio <= 1.0:
            raise ValueError("masked feature ratio must be in (0, 1]")
        self.mask_ratio = mask_ratio

    def compute(self, inputs: DistillationInputs) -> DistillationLossOutput:
        import torch

        if inputs.student_features is None or inputs.teacher_features is None:
            raise ValueError("masked feature distillation requires feature tensors")
        pairs = _feature_pairs(inputs.student_features, inputs.teacher_features)
        losses = []
        selected = 0
        available = 0
        for student, teacher in pairs:
            student_energy = student.float().square().mean(dim=1)
            teacher_energy = teacher.detach().float().square().mean(dim=1)
            if student_energy.shape[1:] != teacher_energy.shape[1:]:
                teacher_energy = torch.nn.functional.interpolate(
                    teacher_energy.unsqueeze(1),
                    size=student_energy.shape[1:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            student_flat = torch.nn.functional.normalize(
                student_energy.flatten(1), dim=1
            )
            teacher_flat = torch.nn.functional.normalize(
                teacher_energy.flatten(1), dim=1
            )
            count = max(1, int(student_flat.shape[1] * self.mask_ratio))
            indices = teacher_flat.topk(count, dim=1).indices
            mask = torch.zeros_like(teacher_flat, dtype=torch.bool).scatter(
                1, indices, True
            )
            losses.append(
                torch.nn.functional.mse_loss(student_flat[mask], teacher_flat[mask])
            )
            selected += int(mask.sum().item())
            available += mask.numel()
        return DistillationLossOutput(
            loss=torch.stack(losses).mean(),
            metrics={
                "masked_position_count": float(selected),
                "masked_position_fraction": float(selected / max(available, 1)),
                "feature_level_count": float(len(pairs)),
            },
        )


def build_distillation_mechanism_loss(
    mechanism: DistillationMechanism,
    **options: Any,
) -> DistillationMechanismLoss:
    implementations: dict[str, type[DistillationMechanismLoss]] = {
        AttentionDistillationLoss.mechanism: AttentionDistillationLoss,
        FeatureDistillationLoss.mechanism: FeatureDistillationLoss,
        LogitsDistillationLoss.mechanism: LogitsDistillationLoss,
        LocalizationDistillationLoss.mechanism: LocalizationDistillationLoss,
        MaskedFeatureDistillationLoss.mechanism: MaskedFeatureDistillationLoss,
        RelationDistillationLoss.mechanism: RelationDistillationLoss,
    }
    try:
        implementation = implementations[mechanism]
    except KeyError as exc:
        raise KeyError(f"distillation mechanism loss is not implemented: {mechanism}") from exc
    return implementation(**options)


def _same_shape(student: Any, teacher: Any) -> tuple[Any, Any]:
    if not hasattr(student, "shape") or not hasattr(teacher, "shape"):
        raise TypeError("distillation tensors must expose shape")
    if tuple(student.shape) != tuple(teacher.shape):
        raise ValueError(
            "distillation tensors require identical shapes: "
            f"{tuple(student.shape)} vs {tuple(teacher.shape)}"
        )
    return student, teacher


def _feature_levels(features: Any) -> list[Any]:
    if isinstance(features, (list, tuple)):
        return list(features)
    return [features]


def _feature_pairs(student_features: Any, teacher_features: Any) -> list[tuple[Any, Any]]:
    students = _feature_levels(student_features)
    teachers = _feature_levels(teacher_features)
    if not students or len(students) != len(teachers):
        raise ValueError("student and teacher feature levels must match")
    pairs = list(zip(students, teachers, strict=True))
    if any(
        student.ndim != 4
        or teacher.ndim != 4
        or student.shape[0] != teacher.shape[0]
        for student, teacher in pairs
    ):
        raise ValueError("distillation feature levels require matching 4D batches")
    return pairs


def _bound_spatial_tokens(feature: Any, limit: int) -> Any:
    import math
    import torch

    tokens = feature.shape[-2] * feature.shape[-1]
    if tokens <= limit:
        return feature
    side = max(2, int(math.sqrt(limit)))
    return torch.nn.functional.adaptive_avg_pool2d(feature, (side, side))


__all__ = [
    "AttentionDistillationLoss",
    "DistillationInputs",
    "DistillationLossOutput",
    "DistillationMechanismLoss",
    "FeatureDistillationLoss",
    "LocalizationDistillationLoss",
    "LogitsDistillationLoss",
    "MaskedFeatureDistillationLoss",
    "RelationDistillationLoss",
    "build_distillation_mechanism_loss",
]
