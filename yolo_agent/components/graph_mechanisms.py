"""Stable identities for reusable YOLO26 graph mechanisms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


GraphMechanismKind = Literal[
    "multi_scale_fusion",
    "weighted_feature_pyramid",
    "bidirectional_feature_fusion",
    "gold_gather_distribute",
    "rtmdet_large_kernel",
    "lightweight_neck",
    "reparameterized_convolution",
    "channel_attention",
    "spatial_attention",
    "deformable_feature_aggregation",
    "feature_pyramid_multi_scale",
]


@dataclass(frozen=True)
class GraphMechanismSpec:
    kind: GraphMechanismKind
    component_id: str
    display_name: str
    insertion_point: Literal["before_detect"] = "before_detect"
    input_strides: tuple[int, ...] = (8, 16, 32)
    output_strides: tuple[int, ...] = (8, 16, 32)
    channels_unchanged: bool = True
    requires_deformable_operator: bool = False


GRAPH_MECHANISMS = {
    item.kind: item
    for item in (
        GraphMechanismSpec(
            kind="multi_scale_fusion",
            component_id="neck.multi_scale_fusion",
            display_name="Legacy generic multi-scale fusion",
        ),
        GraphMechanismSpec(
            kind="weighted_feature_pyramid",
            component_id="neck.weighted_feature_pyramid",
            display_name="Weighted feature pyramid",
        ),
        GraphMechanismSpec(
            kind="bidirectional_feature_fusion",
            component_id="neck.bidirectional_feature_fusion",
            display_name="Bidirectional feature fusion",
        ),
        GraphMechanismSpec(
            kind="gold_gather_distribute",
            component_id="neck.gold_gather_distribute",
            display_name="Gather-distribute fusion",
        ),
        GraphMechanismSpec(
            kind="rtmdet_large_kernel",
            component_id="neck.rtmdet_large_kernel",
            display_name="Large-kernel neck block",
        ),
        GraphMechanismSpec(
            kind="lightweight_neck",
            component_id="neck.lightweight",
            display_name="Lightweight depthwise neck",
        ),
        GraphMechanismSpec(
            kind="reparameterized_convolution",
            component_id="block.reparameterized_convolution",
            display_name="Re-parameterized convolution block",
        ),
        GraphMechanismSpec(
            kind="channel_attention",
            component_id="attention.channel",
            display_name="Channel attention block",
        ),
        GraphMechanismSpec(
            kind="spatial_attention",
            component_id="attention.spatial",
            display_name="Spatial attention block",
        ),
        GraphMechanismSpec(
            kind="deformable_feature_aggregation",
            component_id="neck.deformable_feature_aggregation",
            display_name="Deformable feature aggregation",
            requires_deformable_operator=True,
        ),
        GraphMechanismSpec(
            kind="feature_pyramid_multi_scale",
            component_id="feature_pyramid.multi_scale",
            display_name="Independent multi-scale feature pyramid",
        ),
    )
}

GRAPH_COMPONENTS = {item.component_id: item for item in GRAPH_MECHANISMS.values()}

P2_GRAPH_COMPONENT_ID = "head.p2_small_object"
P2_GRAPH_STRIDES = (4, 8, 16, 32)


__all__ = [
    "GRAPH_COMPONENTS",
    "GRAPH_MECHANISMS",
    "GraphMechanismKind",
    "GraphMechanismSpec",
    "P2_GRAPH_COMPONENT_ID",
    "P2_GRAPH_STRIDES",
]
