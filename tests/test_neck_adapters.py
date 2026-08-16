"""Adapter SDK tests for the three guarded YOLO26 neck components."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.neck_fixtures import neck_context, neck_contracts
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
from yolo_agent.components.adapters.neck.runtime import YOLO26NeckRuntimePlugin


ADAPTERS = {
    "neck.multi_scale_fusion": MultiScaleFusionAdapter,
    "neck.gold_gather_distribute": GoldGatherDistributeAdapter,
    "neck.rtmdet_large_kernel": RTMDetLargeKernelNeckAdapter,
    "neck.weighted_feature_pyramid": WeightedFeaturePyramidAdapter,
    "neck.bidirectional_feature_fusion": BidirectionalFeatureFusionAdapter,
    "neck.lightweight": LightweightNeckAdapter,
    "block.reparameterized_convolution": ReparameterizedConvolutionAdapter,
    "attention.channel": ChannelAttentionAdapter,
    "attention.spatial": SpatialAttentionAdapter,
    "neck.deformable_feature_aggregation": DeformableFeatureAggregationAdapter,
}


@pytest.mark.parametrize(("component_id", "adapter_type"), ADAPTERS.items())
def test_neck_adapters_patch_one_graph_variable_and_pass_smoke(
    component_id: str,
    adapter_type,
    tmp_path: Path,
) -> None:
    contract = neck_contracts()[component_id]
    adapter = adapter_type()
    context = neck_context(contract, tmp_path)
    preview = adapter.prepare_patch({}, {}, context)
    smoke = adapter.smoke_test(context)

    assert [item.field for item in preview.operations] == ["neck_plugin"]
    assert smoke.passed, smoke.errors
    assert smoke.checks["shape"] is True
    assert smoke.checks["backward"] is True
    assert smoke.checks["amp"] is True
    assert smoke.checks["export"] is True
    payload = adapter.build_runtime_payload(
        context,
        protocol_hash="neck-protocol",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={},
    )
    assert len(payload.model_graph_plugin) == 1
    assert not payload.loss_plugin and not payload.assigner_plugin
    assert payload.supports_amp and payload.supports_ddp
    assert payload.supports_resume is True
    payload.verify_imports()


def test_neck_adapter_rejects_changed_imgsz(tmp_path: Path) -> None:
    contract = neck_contracts()["neck.multi_scale_fusion"]
    context = neck_context(contract, tmp_path).model_copy(update={"imgsz": 1280})

    with pytest.raises(ValueError, match="fixed imgsz=640"):
        MultiScaleFusionAdapter().prepare_patch({}, {}, context)


def test_rtmdet_payload_binds_graph_identity_and_shape_contract(tmp_path: Path) -> None:
    component_id = "neck.rtmdet_large_kernel"
    adapter = RTMDetLargeKernelNeckAdapter()
    payload = adapter.build_runtime_payload(
        neck_context(neck_contracts()[component_id], tmp_path),
        protocol_hash="rtmdet-neck-protocol",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={},
    )
    options = payload.model_graph_plugin[0].options
    identity = options["graph_identity"]

    assert identity["component_id"] == component_id
    assert identity["neck_kind"] == "rtmdet_large_kernel"
    assert identity["target_node"] == "terminal_native_detect"
    assert identity["imgsz"] == 640
    assert identity["input_shape_contract"] == identity["output_shape_contract"]
    assert identity["input_shape_contract"]["strides"] == [8, 16, 32]
    assert identity["input_shape_contract"]["channels_source"] == (
        "runtime_detect_channels"
    )
    assert identity["preserves_one_to_one_head"] is True
    assert identity["preserves_native_dfl_free_regression"] is True
    assert identity["adapter_version"] == adapter.adapter_version
    assert identity["adapter_hash"] == options["adapter_hash"]
    assert len(options["graph_identity_hash"]) == 64
    assert payload.rollback_plan.actions


def test_rtmdet_runtime_rejects_tampered_graph_identity(tmp_path: Path) -> None:
    component_id = "neck.rtmdet_large_kernel"
    payload = RTMDetLargeKernelNeckAdapter().build_runtime_payload(
        neck_context(neck_contracts()[component_id], tmp_path),
        protocol_hash="rtmdet-neck-protocol",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={},
    )
    options = {
        **payload.model_graph_plugin[0].options,
        "graph_identity_hash": "0" * 64,
    }

    with pytest.raises(ValueError, match="graph identity hash mismatch"):
        YOLO26NeckRuntimePlugin(**options)


def test_neck_adapter_returns_deformable_implementation_request(tmp_path: Path) -> None:
    contract = neck_contracts()["neck.deformable_feature_aggregation"]
    context = neck_context(
        contract,
        tmp_path,
        {
            "imgsz": 640,
            "deformable_module": "missing_yolo_agent_deformable_operator",
        },
    )
    adapter = DeformableFeatureAggregationAdapter()

    report = adapter.validate_environment(context)
    request = adapter.implementation_request(context)

    assert report.ok is False
    assert report.checks["execution_class"] == "implementation_request"
    assert request is not None
    assert request.component_id == "neck.deformable_feature_aggregation"
