"""Typed, point-based assignment plugins for guarded YOLO26 experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field


AssignmentPath = Literal["one_to_many", "one_to_one"]
AnchorRepresentation = Literal["point", "anchor_box"]


@dataclass(frozen=True)
class AssignerInputs:
    """Native YOLO26 assignment tensors in input-image coordinates."""

    predicted_scores: Any
    predicted_boxes_xyxy: Any
    anchor_points_xy: Any
    stride_per_anchor: Any
    gt_labels: Any
    gt_boxes_xyxy: Any
    gt_mask: Any
    num_classes: int
    path: AssignmentPath
    anchor_representation: AnchorRepresentation = "point"

    def validate(self) -> None:
        """Fail closed before a paper method sees native training tensors."""
        import torch

        if self.anchor_representation != "point":
            raise ValueError("YOLO26 assignment plugins accept point anchors only")
        if self.path not in {"one_to_many", "one_to_one"}:
            raise ValueError(f"unsupported YOLO26 assignment path: {self.path}")
        tensors = {
            "predicted_scores": self.predicted_scores,
            "predicted_boxes_xyxy": self.predicted_boxes_xyxy,
            "anchor_points_xy": self.anchor_points_xy,
            "stride_per_anchor": self.stride_per_anchor,
            "gt_labels": self.gt_labels,
            "gt_boxes_xyxy": self.gt_boxes_xyxy,
            "gt_mask": self.gt_mask,
        }
        if any(not torch.is_tensor(value) for value in tensors.values()):
            raise TypeError("assignment inputs must be torch tensors")
        batch, anchors, classes = self.predicted_scores.shape
        if classes != self.num_classes or self.num_classes < 1:
            raise ValueError("predicted score classes do not match num_classes")
        if self.predicted_boxes_xyxy.shape != (batch, anchors, 4):
            raise ValueError("predicted boxes must have shape [batch, anchors, 4]")
        if self.anchor_points_xy.shape != (anchors, 2):
            raise ValueError("anchor points must have shape [anchors, 2]")
        if self.stride_per_anchor.shape not in {(anchors,), (anchors, 1)}:
            raise ValueError("stride_per_anchor must have one value per point")
        if self.gt_labels.shape[:2] != self.gt_boxes_xyxy.shape[:2]:
            raise ValueError("GT label and box dimensions do not match")
        if self.gt_boxes_xyxy.shape[0] != batch or self.gt_boxes_xyxy.shape[-1] != 4:
            raise ValueError("GT boxes must have shape [batch, max_gt, 4]")
        if self.gt_mask.shape[:2] != self.gt_boxes_xyxy.shape[:2]:
            raise ValueError("GT mask dimensions do not match")
        if not all(torch.isfinite(value).all() for value in tensors.values() if value.is_floating_point()):
            raise ValueError("assignment inputs contain non-finite values")


@dataclass(frozen=True)
class AssignerOutput:
    """Five tensors consumed by the native YOLO26 detection loss."""

    target_labels: Any
    target_boxes_xyxy: Any
    target_scores: Any
    foreground_mask: Any
    target_gt_indices: Any

    def validate(self, inputs: AssignerInputs) -> "AssignerOutput":
        """Validate exact native output shapes and value bounds."""
        import torch

        batch, anchors, classes = inputs.predicted_scores.shape
        expected = {
            "target_labels": (batch, anchors),
            "target_boxes_xyxy": (batch, anchors, 4),
            "target_scores": (batch, anchors, classes),
            "foreground_mask": (batch, anchors),
            "target_gt_indices": (batch, anchors),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if not torch.is_tensor(value) or tuple(value.shape) != shape:
                raise ValueError(f"assignment output {name} must have shape {shape}")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"assignment output {name} contains non-finite values")
        if not bool((self.target_scores >= 0).all() and (self.target_scores <= 1).all()):
            raise ValueError("assignment target scores must remain in [0, 1]")
        return self

    def native_tuple(self) -> tuple[Any, Any, Any, Any, Any]:
        """Return the exact tuple expected by Ultralytics detection loss."""
        return (
            self.target_labels,
            self.target_boxes_xyxy,
            self.target_scores,
            self.foreground_mask,
            self.target_gt_indices,
        )


class AssignmentComparison(BaseModel):
    """One batch of baseline-versus-candidate assignment evidence."""

    total_candidates: int = Field(ge=0)
    baseline_positive_count: int = Field(ge=0)
    candidate_positive_count: int = Field(ge=0)
    baseline_positive_ratio: float = Field(ge=0.0, le=1.0)
    candidate_positive_ratio: float = Field(ge=0.0, le=1.0)
    foreground_disagreement_count: int = Field(ge=0)
    gt_conflict_count: int = Field(ge=0)
    gt_conflict_rate: float = Field(ge=0.0, le=1.0)
    conflict_count: int = Field(ge=0)
    conflict_rate: float = Field(ge=0.0, le=1.0)
    matching_stability: float = Field(ge=0.0, le=1.0)


class YOLO26AssignerPlugin(ABC):
    """Stable API for one independently implemented assignment method."""

    plugin_id: ClassVar[str]
    plugin_version: ClassVar[str]
    paper_id: ClassVar[str | None] = None
    exact_paper_reproduction: ClassVar[bool] = False
    supported_paths: ClassVar[frozenset[AssignmentPath]] = frozenset({"one_to_many"})
    anchor_representation: ClassVar[AnchorRepresentation] = "point"
    replaces_head: ClassVar[bool] = False
    replaces_loss: ClassVar[bool] = False
    changes_inference_path: ClassVar[bool] = False

    def run(self, inputs: AssignerInputs) -> AssignerOutput:
        """Validate the safety contract around an implementation call."""
        inputs.validate()
        if inputs.path not in self.supported_paths:
            raise ValueError(f"{self.plugin_id} does not support {inputs.path}")
        if self.anchor_representation != "point":
            raise ValueError(f"anchor-based plugin is forbidden for YOLO26: {self.plugin_id}")
        return self.assign(inputs).validate(inputs)

    @abstractmethod
    def assign(self, inputs: AssignerInputs) -> AssignerOutput:
        """Compute an assignment without changing head or regression semantics."""


class NativeYOLO26AssignerPlugin(YOLO26AssignerPlugin):
    """Explicit baseline plugin around the installed native assigner."""

    plugin_id = "yolo26.native_task_aligned"
    plugin_version = "ultralytics-runtime"
    supported_paths = frozenset({"one_to_many", "one_to_one"})

    def __init__(self, native_assigner: Any) -> None:
        if not callable(native_assigner):
            raise TypeError("native YOLO26 assigner must be callable")
        self.native_assigner = native_assigner

    def assign(self, inputs: AssignerInputs) -> AssignerOutput:
        output = self.native_assigner(
            inputs.predicted_scores,
            inputs.predicted_boxes_xyxy,
            inputs.anchor_points_xy,
            inputs.gt_labels,
            inputs.gt_boxes_xyxy,
            inputs.gt_mask,
        )
        return _output_from_native(output)


class TOODTaskAlignedAssignerPlugin(YOLO26AssignerPlugin):
    """TOOD TAL assignment-only profile using paper task alignment parameters."""

    plugin_id = "tood.task_aligned_learning"
    plugin_version = "tood_tal_shadow.v1"
    paper_id = "arxiv:2108.07755"

    def __init__(self, *, topk: int = 13, alpha: float = 1.0, beta: float = 6.0) -> None:
        if topk < 1 or alpha <= 0 or beta <= 0:
            raise ValueError("TOOD TAL parameters must be positive")
        self.topk = topk
        self.alpha = alpha
        self.beta = beta

    def assign(self, inputs: AssignerInputs) -> AssignerOutput:
        from ultralytics.utils.tal import TaskAlignedAssigner

        if inputs.predicted_scores.shape[1] < self.topk:
            raise ValueError("TOOD TAL requires at least topk candidate points")
        strides = sorted({float(value) for value in inputs.stride_per_anchor.reshape(-1).tolist()})
        implementation = TaskAlignedAssigner(
            topk=self.topk,
            num_classes=inputs.num_classes,
            alpha=self.alpha,
            beta=self.beta,
            stride=strides,
        )
        return _output_from_native(
            implementation(
                inputs.predicted_scores,
                inputs.predicted_boxes_xyxy,
                inputs.anchor_points_xy,
                inputs.gt_labels,
                inputs.gt_boxes_xyxy,
                inputs.gt_mask,
            )
        )


class OTAAssignerPlugin(YOLO26AssignerPlugin):
    """Entropic optimal-transport assignment with dynamic positive supply."""

    plugin_id = "ota.optimal_transport"
    plugin_version = "ota_sinkhorn_shadow.v1"
    paper_id = "arxiv:2103.14259"

    def __init__(
        self,
        *,
        regression_weight: float = 1.5,
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_iterations: int = 40,
        dynamic_topk: int = 20,
        center_penalty: float = 20.0,
    ) -> None:
        if min(regression_weight, sinkhorn_epsilon, center_penalty) <= 0:
            raise ValueError("OTA cost parameters must be positive")
        if sinkhorn_iterations < 1 or dynamic_topk < 1:
            raise ValueError("OTA iteration and top-k parameters must be positive")
        self.regression_weight = regression_weight
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        self.dynamic_topk = dynamic_topk
        self.center_penalty = center_penalty

    def assign(self, inputs: AssignerInputs) -> AssignerOutput:
        import torch

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
            classification_cost = -torch.log(class_probability.clamp(min=1e-7))
            inside = _points_inside_boxes(inputs.anchor_points_xy, boxes)
            foreground_cost = (
                classification_cost
                + self.regression_weight * (1.0 - pair_iou)
                + (~inside).to(classification_cost.dtype) * self.center_penalty
            )
            background_cost = -torch.log(
                (1.0 - inputs.predicted_scores[batch_index].amax(dim=-1)).clamp(min=1e-7)
            ).unsqueeze(0)
            supply = _dynamic_positive_supply(pair_iou, self.dynamic_topk, anchors)
            background_supply = anchors - int(supply.sum().item())
            supply = torch.cat(
                [supply.to(foreground_cost.dtype), foreground_cost.new_tensor([background_supply])]
            )
            transport = _sinkhorn_transport(
                torch.cat([foreground_cost, background_cost], dim=0),
                supply,
                epsilon=self.sinkhorn_epsilon,
                iterations=self.sinkhorn_iterations,
            )
            selected = torch.zeros_like(foreground_cost, dtype=torch.bool)
            for gt_index, count in enumerate(supply[:-1].long().tolist()):
                if count > 0:
                    indices = transport[gt_index].topk(min(count, anchors)).indices
                    selected[gt_index, indices] = True
            selected_cost = foreground_cost.masked_fill(~selected, float("inf"))
            best_cost, supplier = selected_cost.min(dim=0)
            active = torch.isfinite(best_cost)
            matched[batch_index] = supplier
            foreground[batch_index] = active
            anchor_indices = torch.arange(anchors, device=supplier.device)
            quality[batch_index, active] = pair_iou[
                supplier[active], anchor_indices[active]
            ]
        return _targets_from_matches(inputs, matched, foreground, quality)


class DSLAAssignerPlugin(YOLO26AssignerPlugin):
    """Dynamic smooth labels from interval, core-zone, and online-IoU quality."""

    plugin_id = "dsla.dynamic_smooth_label"
    plugin_version = "dsla_shadow.v1"
    paper_id = "arxiv:2208.00817"

    def __init__(self, *, interval_relaxation: float = 0.2) -> None:
        if not 0.0 <= interval_relaxation < 1.0:
            raise ValueError("DSLA interval relaxation must be in [0, 1)")
        self.interval_relaxation = interval_relaxation

    def assign(self, inputs: AssignerInputs) -> AssignerOutput:
        import torch

        batch, anchors, _ = inputs.predicted_scores.shape
        matched = torch.zeros((batch, anchors), dtype=torch.long, device=inputs.predicted_scores.device)
        foreground = torch.zeros((batch, anchors), dtype=torch.bool, device=inputs.predicted_scores.device)
        quality = torch.zeros((batch, anchors), dtype=inputs.predicted_scores.dtype, device=inputs.predicted_scores.device)
        strides = inputs.stride_per_anchor.reshape(-1).to(inputs.predicted_scores.dtype)
        for batch_index in range(batch):
            valid = inputs.gt_mask[batch_index].reshape(-1).bool()
            boxes = inputs.gt_boxes_xyxy[batch_index, valid]
            if boxes.numel() == 0:
                continue
            distances = _point_box_distances(inputs.anchor_points_xy, boxes)
            inside = distances.amin(dim=-1) > 0
            scale_score = _dsla_interval_score(
                distances.amax(dim=-1),
                strides,
                relaxation=self.interval_relaxation,
            )
            centerness = _dsla_centerness(distances, strides)
            online_iou = _pairwise_iou(boxes, inputs.predicted_boxes_xyxy[batch_index])
            smooth_quality = inside.to(online_iou.dtype) * scale_score * centerness * online_iou
            best_quality, best_gt = smooth_quality.max(dim=0)
            active = best_quality > 0
            matched[batch_index] = best_gt
            foreground[batch_index] = active
            quality[batch_index] = torch.where(active, best_quality, best_quality.new_zeros(()))
        return _targets_from_matches(inputs, matched, foreground, quality)


def build_yolo26_assigner_plugin(method: str, **options: Any) -> YOLO26AssignerPlugin:
    """Construct only explicit, independently implemented shadow methods."""
    implementations: dict[str, type[YOLO26AssignerPlugin]] = {
        "tood_tal": TOODTaskAlignedAssignerPlugin,
        "ota": OTAAssignerPlugin,
        "dsla": DSLAAssignerPlugin,
    }
    try:
        implementation = implementations[method]
    except KeyError as exc:
        raise KeyError(f"unknown YOLO26 assignment method: {method}") from exc
    return implementation(**options)


def compare_assignments(
    baseline: AssignerOutput,
    candidate: AssignerOutput,
) -> AssignmentComparison:
    """Measure foreground and GT identity disagreement for one batch."""
    base_fg = baseline.foreground_mask.bool()
    candidate_fg = candidate.foreground_mask.bool()
    total = int(base_fg.numel())
    foreground_disagreement = base_fg ^ candidate_fg
    gt_conflict = base_fg & candidate_fg & (
        baseline.target_gt_indices.long() != candidate.target_gt_indices.long()
    )
    conflict = foreground_disagreement | gt_conflict
    baseline_count = int(base_fg.sum().item())
    candidate_count = int(candidate_fg.sum().item())
    return AssignmentComparison(
        total_candidates=total,
        baseline_positive_count=baseline_count,
        candidate_positive_count=candidate_count,
        baseline_positive_ratio=baseline_count / max(total, 1),
        candidate_positive_ratio=candidate_count / max(total, 1),
        foreground_disagreement_count=int(foreground_disagreement.sum().item()),
        gt_conflict_count=int(gt_conflict.sum().item()),
        gt_conflict_rate=float(gt_conflict.sum().item()) / max(total, 1),
        conflict_count=int(conflict.sum().item()),
        conflict_rate=float(conflict.sum().item()) / max(total, 1),
        matching_stability=1.0 - float(conflict.sum().item()) / max(total, 1),
    )


def _output_from_native(output: Any) -> AssignerOutput:
    if not isinstance(output, tuple) or len(output) != 5:
        raise TypeError("native YOLO26 assigner must return five tensors")
    return AssignerOutput(
        target_labels=output[0],
        target_boxes_xyxy=output[1],
        target_scores=output[2],
        foreground_mask=output[3].bool(),
        target_gt_indices=output[4].long(),
    )


def _targets_from_matches(
    inputs: AssignerInputs,
    matched: Any,
    foreground: Any,
    quality: Any,
) -> AssignerOutput:
    import torch

    batch, anchors = matched.shape
    labels = torch.full(
        (batch, anchors),
        inputs.num_classes,
        dtype=torch.long,
        device=matched.device,
    )
    boxes = inputs.predicted_boxes_xyxy.new_zeros((batch, anchors, 4))
    scores = inputs.predicted_scores.new_zeros((batch, anchors, inputs.num_classes))
    for batch_index in range(batch):
        valid_gt = inputs.gt_mask[batch_index].reshape(-1).bool()
        gt_boxes = inputs.gt_boxes_xyxy[batch_index, valid_gt]
        gt_labels = inputs.gt_labels[batch_index, valid_gt].reshape(-1).long()
        active = foreground[batch_index]
        if not bool(active.any()) or gt_boxes.numel() == 0:
            continue
        indices = matched[batch_index, active].clamp(min=0, max=gt_boxes.shape[0] - 1)
        active_labels = gt_labels[indices]
        labels[batch_index, active] = active_labels
        boxes[batch_index, active] = gt_boxes[indices]
        anchor_indices = torch.where(active)[0]
        scores[batch_index, anchor_indices, active_labels] = quality[batch_index, active].clamp(0, 1)
    return AssignerOutput(labels, boxes, scores, foreground.bool(), matched.long())


def _pairwise_iou(gt_boxes: Any, predicted_boxes: Any, epsilon: float = 1e-7) -> Any:
    import torch

    left_top = torch.maximum(gt_boxes[:, None, :2], predicted_boxes[None, :, :2])
    right_bottom = torch.minimum(gt_boxes[:, None, 2:], predicted_boxes[None, :, 2:])
    intersection_wh = (right_bottom - left_top).clamp(min=0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    gt_wh = (gt_boxes[:, 2:] - gt_boxes[:, :2]).clamp(min=0)
    predicted_wh = (predicted_boxes[:, 2:] - predicted_boxes[:, :2]).clamp(min=0)
    union = (
        gt_wh[:, None, 0] * gt_wh[:, None, 1]
        + predicted_wh[None, :, 0] * predicted_wh[None, :, 1]
        - intersection
    )
    return intersection / (union + epsilon)


def _points_inside_boxes(points: Any, boxes: Any) -> Any:
    return _point_box_distances(points, boxes).amin(dim=-1) > 0


def _point_box_distances(points: Any, boxes: Any) -> Any:
    import torch

    left_top = points[None, :, :] - boxes[:, None, :2]
    right_bottom = boxes[:, None, 2:] - points[None, :, :]
    return torch.cat((left_top, right_bottom), dim=-1)


def _dynamic_positive_supply(pair_iou: Any, topk: int, anchors: int) -> Any:
    import torch

    count = min(topk, anchors)
    supply = pair_iou.topk(count, dim=-1).values.sum(dim=-1).floor().clamp(min=1).long()
    capacity = max(anchors - 1, 1)
    while int(supply.sum().item()) > capacity:
        reducible = torch.where(supply > 1)[0]
        if reducible.numel() == 0:
            supply[-1] = max(0, capacity - int(supply[:-1].sum().item()))
            break
        index = reducible[supply[reducible].argmax()]
        supply[index] -= 1
    return supply


def _sinkhorn_transport(cost: Any, supply: Any, *, epsilon: float, iterations: int) -> Any:
    import torch

    demand = torch.ones(cost.shape[1], dtype=cost.dtype, device=cost.device)
    kernel = torch.exp(-cost / epsilon).clamp(min=torch.finfo(cost.dtype).tiny)
    left = torch.ones_like(supply)
    right = torch.ones_like(demand)
    for _ in range(iterations):
        left = supply / (kernel @ right).clamp(min=1e-12)
        right = demand / (kernel.transpose(0, 1) @ left).clamp(min=1e-12)
    return left[:, None] * kernel * right[None, :]


def _dsla_interval_score(max_distance: Any, strides: Any, *, relaxation: float) -> Any:
    import torch

    unique_strides = sorted({float(value) for value in strides.tolist()})
    score = torch.zeros_like(max_distance)
    for level, stride in enumerate(unique_strides):
        level_mask = torch.isclose(strides, strides.new_tensor(stride))
        lower = 0.0 if level == 0 else unique_strides[level - 1] * 8.0
        upper = float("inf") if level == len(unique_strides) - 1 else stride * 8.0
        values = max_distance[:, level_mask]
        current = torch.ones_like(values)
        if lower > 0:
            relaxed_lower = lower * (1.0 - relaxation)
            current *= ((values - relaxed_lower) / max(lower - relaxed_lower, 1e-6)).clamp(0, 1)
        if upper != float("inf"):
            relaxed_upper = upper * (1.0 + relaxation)
            current *= ((relaxed_upper - values) / max(relaxed_upper - upper, 1e-6)).clamp(0, 1)
        score[:, level_mask] = current
    return score


def _dsla_centerness(distances: Any, strides: Any) -> Any:
    import torch

    left, top, right, bottom = distances.unbind(dim=-1)
    horizontal = torch.minimum(left, right) / torch.maximum(left, right).clamp(min=1e-7)
    vertical = torch.minimum(top, bottom) / torch.maximum(top, bottom).clamp(min=1e-7)
    centerness = (horizontal.clamp(min=0) * vertical.clamp(min=0)).sqrt()
    core = (
        (torch.abs(left - right) <= strides[None, :])
        & (torch.abs(top - bottom) <= strides[None, :])
    )
    return torch.where(core, torch.ones_like(centerness), centerness)


__all__ = [
    "AnchorRepresentation",
    "AssignerInputs",
    "AssignerOutput",
    "AssignmentComparison",
    "AssignmentPath",
    "DSLAAssignerPlugin",
    "NativeYOLO26AssignerPlugin",
    "OTAAssignerPlugin",
    "TOODTaskAlignedAssignerPlugin",
    "YOLO26AssignerPlugin",
    "build_yolo26_assigner_plugin",
    "compare_assignments",
]
