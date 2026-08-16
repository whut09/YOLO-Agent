"""Real installed-Ultralytics runtime tests without dataset or GPU training."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402

from tests.neck_fixtures import neck_context, neck_contracts  # noqa: E402
from yolo_agent.adapters.ultralytics.plugin_bridge import (  # noqa: E402
    PluginExecutionError,
    UltralyticsTrainerPluginBridge,
)
from yolo_agent.components.adapters.neck import (  # noqa: E402
    BidirectionalFeatureFusionAdapter,
    ChannelAttentionAdapter,
    DeformableFeatureAggregationAdapter,
    DetectWithFeaturePyramidNeck,
    GoldGatherDistributeAdapter,
    LightweightNeckAdapter,
    MultiScaleFusionAdapter,
    ReparameterizedConvolutionAdapter,
    RTMDetLargeKernelNeckAdapter,
    SpatialAttentionAdapter,
    WeightedFeaturePyramidAdapter,
    YOLO26NeckManifest,
)


RUNTIME_CASES = [
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


@pytest.mark.parametrize(("component_id", "adapter_type"), RUNTIME_CASES)
def test_neck_runtime_preserves_native_head_loss_and_writes_audit(
    component_id: str,
    adapter_type,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / component_id.replace(".", "_")
    workspace.mkdir()
    model = DetectionModel("yolo26n.yaml", nc=3, verbose=False)
    model.args = get_cfg(overrides={"imgsz": 640})
    checkpoint = workspace / "native.pt"
    torch.save(model.state_dict(), checkpoint)
    model.pt_path = checkpoint
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
    adapter = adapter_type()
    payload = adapter.build_runtime_payload(
        context,
        protocol_hash="neck-protocol",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={},
    )
    assert payload.supports_resume is True
    bridge = UltralyticsTrainerPluginBridge(payload.write(workspace / "runtime.yaml"))
    trainer = SimpleNamespace(args=get_cfg(overrides={"imgsz": 640}))
    transformed = bridge.invoke_transform("build_model", model, trainer=trainer)

    assert transformed is model
    assert isinstance(model.model[-1], DetectWithFeaturePyramidNeck)
    assert model.model[-1].stride.tolist() == [8.0, 16.0, 32.0]
    assert model.model[-1].reg_max == 1
    assert type(model.model[-1].dfl).__name__ == "Identity"
    assert model.end2end is True

    image = torch.rand(1, 3, 64, 64)
    model.train()
    predictions = model(image)
    assert set(predictions) == {"one2many", "one2one"}
    batch = {
        "img": image,
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
    }
    loss, _ = model.loss(batch)
    loss.sum().backward()
    assert any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if ".neck." in name
    )

    model.eval()
    with torch.no_grad():
        model(torch.rand(1, 3, 64, 96))

    manifest_path = workspace / f"{component_id.replace('.', '_')}_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["input_strides"] == [8, 16, 32]
    assert manifest["output_strides"] == [8, 16, 32]
    assert manifest["input_channels"] == [64, 128, 256]
    assert manifest["output_channels"] == manifest["input_channels"]
    assert manifest["mechanism"] == adapter.neck_kind
    assert len(manifest["configuration_hash"]) == 64
    assert manifest["dependency_available"] is True
    assert manifest["checkpoint"]["matched_keys"]
    assert manifest["checkpoint"]["newly_initialized_keys"]
    assert manifest["checkpoint"]["checkpoint_sha256"]
    assert manifest["resources"]["passed"] is True
    assert set(manifest["runtime_metrics"]) == {
        "latency_ms",
        "peak_vram_mb",
        "model_size_mb",
        "peak_vram_source",
    }
    assert manifest["runtime_metrics"]["peak_vram_source"] == (
        "cpu_preflight_estimate"
    )
    legacy_manifest = dict(manifest)
    legacy_manifest.pop("runtime_metrics")
    assert YOLO26NeckManifest.model_validate(legacy_manifest).runtime_metrics is None
    assert manifest["export_dry_run"] is True
    assert manifest["external_nms_added"] is False
    if component_id == "neck.deformable_feature_aggregation":
        assert manifest["operator_module"] == "torchvision.ops"
        assert manifest["operator_class"] == "DeformConv2d"
        assert manifest["operator_call_count"] > 0
        assert model.model[-1].neck.operator_calls > 0
    else:
        assert manifest["operator_module"] is None
        assert manifest["operator_class"] is None


def test_neck_runtime_enforces_resource_guard_before_training(tmp_path: Path) -> None:
    component_id = "neck.rtmdet_large_kernel"
    model = DetectionModel("yolo26n.yaml", nc=3, verbose=False)
    model.args = get_cfg(overrides={"imgsz": 640})
    context = neck_context(
        neck_contracts()[component_id],
        tmp_path,
        {
            "imgsz": 640,
            "audit_imgsz": 64,
            "latency_warmup": 0,
            "latency_iterations": 1,
            "resource_limits": {
                "max_latency_regression": 100.0,
                "max_vram_regression": 0.0,
                "max_parameter_regression": 0.0,
                "max_model_size_regression": 0.0,
            },
        },
    )
    adapter = RTMDetLargeKernelNeckAdapter()
    payload = adapter.build_runtime_payload(
        context,
        protocol_hash="neck-protocol",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={},
    )
    bridge = UltralyticsTrainerPluginBridge(payload.write(tmp_path / "runtime.yaml"))

    with pytest.raises(RuntimeError, match="resource guards failed"):
        bridge.invoke_transform(
            "build_model",
            model,
            trainer=SimpleNamespace(args=get_cfg(overrides={"imgsz": 640})),
        )
    manifest = json.loads(
        (tmp_path / "neck_rtmdet_large_kernel_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["resources"]["passed"] is False
    assert manifest["resources"]["checks"]["vram"] is False


def test_rtmdet_runtime_refuses_duplicate_detect_wrapper(tmp_path: Path) -> None:
    component_id = "neck.rtmdet_large_kernel"
    model = DetectionModel("yolo26n.yaml", nc=3, verbose=False)
    model.args = get_cfg(overrides={"imgsz": 640})
    context = neck_context(
        neck_contracts()[component_id],
        tmp_path,
        {
            "imgsz": 640,
            "audit_imgsz": 64,
            "latency_warmup": 0,
            "latency_iterations": 1,
            "resource_limits": {
                "max_latency_regression": 100.0,
                "max_vram_regression": 10.0,
                "max_parameter_regression": 10.0,
                "max_model_size_regression": 10.0,
            },
        },
    )
    payload = RTMDetLargeKernelNeckAdapter().build_runtime_payload(
        context,
        protocol_hash="neck-protocol",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={},
    )
    bridge = UltralyticsTrainerPluginBridge(payload.write(tmp_path / "runtime.yaml"))
    trainer = SimpleNamespace(args=get_cfg(overrides={"imgsz": 640}))
    bridge.invoke_transform("build_model", model, trainer=trainer)

    with pytest.raises(
        PluginExecutionError,
        match="refuses to wrap an existing neck plugin",
    ):
        bridge.invoke_transform("build_model", model, trainer=trainer)


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("YOLO_AGENT_RUN_GPU_TESTS") != "1",
    reason="set YOLO_AGENT_RUN_GPU_TESTS=1 for optional neck GPU smoke",
)
@pytest.mark.parametrize(("component_id", "adapter_type"), RUNTIME_CASES)
def test_optional_neck_graph_gpu_smoke(
    component_id: str,
    adapter_type,
    tmp_path: Path,
) -> None:
    context = neck_context(
        neck_contracts()[component_id],
        tmp_path / component_id.replace(".", "_"),
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
    context.workspace.mkdir(parents=True, exist_ok=True)

    result = adapter_type().gpu_smoke_test(context)

    assert result.passed, result.errors
    assert result.checks["actual_graph"] is True
    assert result.checks["native_loss_preserved"] is True
    assert result.checks["backward"] is True
    assert result.checks["amp"] is True
    assert result.checks["partial_checkpoint_audit"] is True
    assert result.checks["export"] is True
    assert result.checks["resource_guard"] is True
