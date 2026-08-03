from yolo_agent.components.adapters.neck.component_adapters import (
    BidirectionalFeatureFusionAdapter,
    ChannelAttentionAdapter,
    DeformableFeatureAggregationAdapter,
    LightweightNeckAdapter,
    ReparameterizedConvolutionAdapter,
    SpatialAttentionAdapter,
    WeightedFeaturePyramidAdapter,
)


ADAPTERS = [
    WeightedFeaturePyramidAdapter,
    BidirectionalFeatureFusionAdapter,
    LightweightNeckAdapter,
    ReparameterizedConvolutionAdapter,
    ChannelAttentionAdapter,
    SpatialAttentionAdapter,
    DeformableFeatureAggregationAdapter,
]


def test_graph_component_adapters_have_independent_identities() -> None:
    instances = [adapter_type() for adapter_type in ADAPTERS]

    assert len({item.component_id for item in instances}) == len(instances)
    assert len({item.neck_kind for item in instances}) == len(instances)
    assert len({item.adapter_version for item in instances}) == len(instances)


def test_deformable_adapter_declares_real_operator_dependency() -> None:
    adapter = DeformableFeatureAggregationAdapter()

    assert adapter.default_options == {
        "deformable_module": "torchvision.ops",
        "deformable_operator": "DeformConv2d",
    }
