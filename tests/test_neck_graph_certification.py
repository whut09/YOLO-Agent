from __future__ import annotations

from pathlib import Path

import pytest

from tests.neck_fixtures import neck_context, neck_contracts
from yolo_agent.certification.neck_graph import run_neck_graph_cpu_fixture
from yolo_agent.components.adapters.neck import (
    BidirectionalFeatureFusionAdapter,
    ChannelAttentionAdapter,
    DeformableFeatureAggregationAdapter,
    GoldGatherDistributeAdapter,
    LightweightNeckAdapter,
    MultiScaleFusionAdapter,
    ReparameterizedConvolutionAdapter,
    RTMDetLargeKernelNeckAdapter,
    SpatialAttentionAdapter,
    WeightedFeaturePyramidAdapter,
)


CASES = [
    ("neck.multi_scale_fusion", MultiScaleFusionAdapter),
    ("neck.gold_gather_distribute", GoldGatherDistributeAdapter),
    ("neck.rtmdet_large_kernel", RTMDetLargeKernelNeckAdapter),
    ("neck.weighted_feature_pyramid", WeightedFeaturePyramidAdapter),
    ("neck.bidirectional_feature_fusion", BidirectionalFeatureFusionAdapter),
    ("neck.lightweight", LightweightNeckAdapter),
    ("block.reparameterized_convolution", ReparameterizedConvolutionAdapter),
    ("attention.channel", ChannelAttentionAdapter),
    ("attention.spatial", SpatialAttentionAdapter),
    ("neck.deformable_feature_aggregation", DeformableFeatureAggregationAdapter),
]


@pytest.mark.parametrize(("component_id", "adapter_type"), CASES)
def test_neck_graph_cpu_fixture_certifies_runtime_component(
    component_id: str,
    adapter_type: type,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / component_id
    context = neck_context(
        neck_contracts()[component_id],
        workspace,
        {
            "imgsz": 640,
            "audit_imgsz": 64,
            "latency_warmup": 0,
            "latency_iterations": 1,
            "context_channels": 16,
            "resource_limits": {
                "max_latency_regression": 100.0,
                "max_vram_regression": 10.0,
                "max_parameter_regression": 10.0,
                "max_model_size_regression": 10.0,
            },
        },
    )
    payload = adapter_type().build_runtime_payload(
        context,
        protocol_hash=f"{component_id}-protocol",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={},
    )
    payload_path = payload.write(workspace / "adapter_runtime_payload.yaml")

    report = run_neck_graph_cpu_fixture(
        runtime_payload_path=payload_path,
        workspace=workspace / "golden",
    )

    assert report.status == "passed", report.errors
    assert report.component_id == component_id
    assert report.checks["real_forward"] is True
    assert report.checks["native_loss_preserved"] is True
    assert report.checks["backward"] is True
    assert report.checks["amp"] is True
    assert report.checks["partial_checkpoint_audit"] is True
    assert report.checks["export"] is True
    assert report.checks["resource_guard"] is True
    assert report.checks["matched_control_required"] is True
    assert report.checks["mechanism_bound"] is True
    assert report.checks["deformable_operator_verified"] is True
    assert report.checks["training_deploy_equivalence"] is True
