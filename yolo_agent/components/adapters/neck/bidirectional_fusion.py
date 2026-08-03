"""Explicit bidirectional feature-fusion identity for the reusable neck core."""

from __future__ import annotations

from yolo_agent.components.adapters.neck.multi_scale_fusion import (
    MultiScaleFusionNeck,
)


class BidirectionalFeatureFusionNeck(MultiScaleFusionNeck):
    """Top-down and bottom-up fusion with an independent runtime identity."""

    plugin_id = "neck.bidirectional_feature_fusion"
    plugin_version = "bidirectional_feature_fusion.v1"
    paper_ids: tuple[str, ...] = ()
    exact_paper_reproduction = False

    def __init__(self, channels: list[int], *, fusion_channels: int = 64) -> None:
        try:
            super().__init__(channels, fusion_channels=fusion_channels)
        except TypeError as exc:  # pragma: no cover - torch-free installation
            raise ImportError(
                "BidirectionalFeatureFusionNeck requires torch"
            ) from exc


__all__ = ["BidirectionalFeatureFusionNeck"]
