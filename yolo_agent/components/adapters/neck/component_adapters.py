"""Independent adapter identities for reusable pre-Detect graph components."""

from yolo_agent.components.adapters.neck.runtime import GuardedYOLO26NeckAdapter


class WeightedFeaturePyramidAdapter(GuardedYOLO26NeckAdapter):
    component_id = "neck.weighted_feature_pyramid"
    neck_kind = "weighted_feature_pyramid"
    adapter_version = "weighted_feature_pyramid_adapter.v1"


class BidirectionalFeatureFusionAdapter(GuardedYOLO26NeckAdapter):
    component_id = "neck.bidirectional_feature_fusion"
    neck_kind = "bidirectional_feature_fusion"
    adapter_version = "bidirectional_feature_fusion_adapter.v1"


class LightweightNeckAdapter(GuardedYOLO26NeckAdapter):
    component_id = "neck.lightweight"
    neck_kind = "lightweight_neck"
    adapter_version = "lightweight_neck_adapter.v1"


class ReparameterizedConvolutionAdapter(GuardedYOLO26NeckAdapter):
    component_id = "block.reparameterized_convolution"
    neck_kind = "reparameterized_convolution"
    adapter_version = "reparameterized_convolution_adapter.v1"


class ChannelAttentionAdapter(GuardedYOLO26NeckAdapter):
    component_id = "attention.channel"
    neck_kind = "channel_attention"
    adapter_version = "channel_attention_adapter.v1"


class SpatialAttentionAdapter(GuardedYOLO26NeckAdapter):
    component_id = "attention.spatial"
    neck_kind = "spatial_attention"
    adapter_version = "spatial_attention_adapter.v1"


class DeformableFeatureAggregationAdapter(GuardedYOLO26NeckAdapter):
    component_id = "neck.deformable_feature_aggregation"
    neck_kind = "deformable_feature_aggregation"
    adapter_version = "deformable_feature_aggregation_adapter.v1"
    default_options = {
        "deformable_module": "torchvision.ops",
        "deformable_operator": "DeformConv2d",
    }


__all__ = [
    "BidirectionalFeatureFusionAdapter",
    "ChannelAttentionAdapter",
    "DeformableFeatureAggregationAdapter",
    "LightweightNeckAdapter",
    "ReparameterizedConvolutionAdapter",
    "SpatialAttentionAdapter",
    "WeightedFeaturePyramidAdapter",
]
