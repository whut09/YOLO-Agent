"""Loss adapter abstractions for future Ultralytics trainer integration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel


UNVERIFIED_LOSS_MESSAGE = "This loss requires a verified runtime adapter before training."


class UnavailableLossError(RuntimeError):
    """Raised when a metadata-only loss is requested for execution."""


class LossAvailability(BaseModel):
    """Explicit executable status for one legacy bbox loss name."""

    name: str
    executable: bool
    implementation_status: Literal["runtime_integrated", "adapter_required"]
    reason: str = ""


class BBoxLossAdapter(ABC):
    """Base interface for bbox loss plugins."""

    name: str

    def supports_head(self, head_type: str) -> bool:
        """Return whether this loss can be used with a head type."""
        return head_type in {"detect", "anchor_free", "anchor_based", "generic"}

    def build(self, config: dict[str, Any] | None = None) -> "BBoxLossAdapter":
        """Build or configure the loss adapter."""
        self.config = config or {}
        return self

    @abstractmethod
    def compute(self, pred_boxes: Any, target_boxes: Any) -> Any:
        """Compute the loss for predicted and target boxes."""


class CIoULossAdapter(BBoxLossAdapter):
    """Simplified CIoU loss for tests and adapter wiring."""

    name = "ciou"

    def compute(self, pred_boxes: Any, target_boxes: Any) -> Any:
        """Compute a simplified CIoU loss using torch tensors in xyxy format."""
        torch = _require_torch()
        pred = pred_boxes.float()
        target = target_boxes.float()
        eps = 1e-7

        pred_x1, pred_y1, pred_x2, pred_y2 = pred.unbind(dim=-1)
        target_x1, target_y1, target_x2, target_y2 = target.unbind(dim=-1)

        inter_x1 = torch.maximum(pred_x1, target_x1)
        inter_y1 = torch.maximum(pred_y1, target_y1)
        inter_x2 = torch.minimum(pred_x2, target_x2)
        inter_y2 = torch.minimum(pred_y2, target_y2)
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        intersection = inter_w * inter_h

        pred_w = (pred_x2 - pred_x1).clamp(min=eps)
        pred_h = (pred_y2 - pred_y1).clamp(min=eps)
        target_w = (target_x2 - target_x1).clamp(min=eps)
        target_h = (target_y2 - target_y1).clamp(min=eps)
        union = pred_w * pred_h + target_w * target_h - intersection + eps
        iou = intersection / union

        pred_cx = (pred_x1 + pred_x2) / 2
        pred_cy = (pred_y1 + pred_y2) / 2
        target_cx = (target_x1 + target_x2) / 2
        target_cy = (target_y1 + target_y2) / 2
        center_distance = (pred_cx - target_cx).pow(2) + (pred_cy - target_cy).pow(2)

        enclosing_x1 = torch.minimum(pred_x1, target_x1)
        enclosing_y1 = torch.minimum(pred_y1, target_y1)
        enclosing_x2 = torch.maximum(pred_x2, target_x2)
        enclosing_y2 = torch.maximum(pred_y2, target_y2)
        enclosing_diagonal = (
            (enclosing_x2 - enclosing_x1).pow(2)
            + (enclosing_y2 - enclosing_y1).pow(2)
            + eps
        )

        v = (4 / torch.pi**2) * (
            torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h)
        ).pow(2)
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)
        ciou = iou - center_distance / enclosing_diagonal - alpha * v
        return (1 - ciou).mean()


class LossRegistry:
    """Registry for bbox loss adapters."""

    def __init__(self) -> None:
        self._adapters: dict[str, type[BBoxLossAdapter]] = {}
        self._unavailable: dict[str, LossAvailability] = {}

    def register(self, adapter_cls: type[BBoxLossAdapter]) -> None:
        """Register a loss adapter class by its ``name``."""
        if not adapter_cls.name:
            raise ValueError("Loss adapter must define a non-empty name.")
        self._adapters[adapter_cls.name] = adapter_cls

    def register_unavailable(self, name: str, reason: str) -> None:
        """Record a known paper loss without exposing an executable adapter."""
        self._unavailable[name] = LossAvailability(
            name=name,
            executable=False,
            implementation_status="adapter_required",
            reason=reason,
        )

    def get(self, name: str) -> BBoxLossAdapter:
        """Instantiate a registered loss adapter."""
        unavailable = self._unavailable.get(name)
        if unavailable is not None:
            raise UnavailableLossError(f"{name}: {unavailable.reason}")
        try:
            adapter_cls = self._adapters[name]
        except KeyError as exc:
            raise KeyError(f"Unknown bbox loss adapter: {name}") from exc
        return adapter_cls()

    def names(self) -> list[str]:
        """Return registered loss names."""
        return sorted(self._adapters)

    def availability(self) -> list[LossAvailability]:
        """Return executable and unavailable names without conflating their status."""
        available = [
            LossAvailability(
                name=name,
                executable=True,
                implementation_status="runtime_integrated",
            )
            for name in self._adapters
        ]
        return sorted([*available, *self._unavailable.values()], key=lambda item: item.name)


def default_loss_registry() -> LossRegistry:
    """Build the executable bbox loss registry."""
    registry = LossRegistry()
    registry.register(CIoULossAdapter)
    for name in ("wiou", "mpdiou", "nwd"):
        registry.register_unavailable(name, UNVERIFIED_LOSS_MESSAGE)
    return registry


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("CIoULossAdapter requires torch for tensor computation.") from exc
    return torch
