import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402

from yolo_agent.adapters.ultralytics.plugin_bridge import (  # noqa: E402
    UltralyticsTrainerPluginBridge,
)
from yolo_agent.components.adapters.base import AdapterContext  # noqa: E402
from yolo_agent.components.adapters.head.p2_head import (  # noqa: E402
    P2Head,
    P2HeadAdapter,
    P2HeadConfig,
)
from yolo_agent.components.contracts import ComponentContract  # noqa: E402


def _context(tmp_path: Path, **options):
    return AdapterContext(contract=ComponentContract(
        component_id="head.p2_small_object", display_name="P2", category="detection_head",
        implementation_path="local", adapter_class="P2HeadAdapter", changes_model_graph=True,
        fixed_imgsz_compatible=True,
    ), detector_family="yolo26", head="one_to_one", workspace=tmp_path, options=options)


@pytest.fixture(scope="module")
def runtime_graph(tmp_path_factory: pytest.TempPathFactory):
    workspace = tmp_path_factory.mktemp("p2-runtime")
    source = DetectionModel("yolo26n.yaml", nc=80, verbose=False)
    source.args = get_cfg(overrides={"imgsz": 640})
    checkpoint = workspace / "native_yolo26n.pt"
    torch.save(source.state_dict(), checkpoint)
    source.pt_path = checkpoint
    adapter = P2HeadAdapter()
    context = _context(workspace, audit_imgsz=64, latency_iterations=1)
    payload = adapter.build_runtime_payload(
        context,
        protocol_hash="p2-protocol",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={},
    )
    payload_path = payload.write(workspace / "runtime.yaml")
    bridge = UltralyticsTrainerPluginBridge(payload_path)
    trainer = SimpleNamespace(args=get_cfg(overrides={"imgsz": 640}))
    model = bridge.invoke_transform("build_model", source, trainer=trainer)
    manifest = json.loads((workspace / "p2_head_manifest.json").read_text(encoding="utf-8"))
    return workspace, source, model, manifest, bridge


def test_p2_head_shape_and_backward() -> None:
    config = P2HeadConfig(in_channels=[16, 32, 64, 128], p2_channels=8)
    module = P2Head(config.in_channels, config)
    features = [torch.randn(2, c, 40 // (2 ** i), 40 // (2 ** i), requires_grad=True) for i, c in enumerate(config.in_channels)]
    output = module(features)
    assert output.shape == (2, 8, 40, 40)
    output.mean().backward()
    assert features[0].grad is not None


def test_p2_checks_real_feature_strides() -> None:
    features = [torch.zeros(1, c, 160 // (2 ** i), 160 // (2 ** i)) for i, c in enumerate([16, 32, 64, 128])]
    assert P2Head.validate_feature_strides(features, 640) == {"p2": 4, "p3": 8, "p4": 16, "p5": 32}
    features[1] = torch.zeros(1, 32, 70, 70)
    with pytest.raises(ValueError, match="expected 8"):
        P2Head.validate_feature_strides(features, 640)


def test_p2_adapter_patch_smoke_and_checkpoint_policy(tmp_path: Path) -> None:
    adapter = P2HeadAdapter()
    context = _context(tmp_path, in_channels=[16, 32, 64, 128], p2_channels=8)
    preview = adapter.prepare_patch({}, {}, context)
    assert preview.operations[0].field == "p2_head"
    assert adapter.smoke_test(context).passed
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"checkpoint")
    assert "partial" in adapter.load_pretrained_weights({}, checkpoint, context).message.lower()


def test_p2_rejects_changed_imgsz(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fixed imgsz=640"):
        P2HeadAdapter().prepare_patch({}, {}, _context(tmp_path).model_copy(update={"imgsz": 1280}))


def test_p2_runtime_builds_loadable_native_four_scale_graph(runtime_graph) -> None:
    workspace, _, model, manifest, bridge = runtime_graph
    generated = workspace / "generated_yolo26_p2.yaml"
    reloaded = DetectionModel(str(generated), nc=80, verbose=False)

    assert model.stride.tolist() == [4.0, 8.0, 16.0, 32.0]
    assert reloaded.stride.tolist() == [4.0, 8.0, 16.0, 32.0]
    assert model.model[-1].f == [19, 22, 25, 28]
    assert model.end2end is True
    assert model.model[-1].reg_max == 1
    assert type(model.model[-1].dfl).__name__ == "Identity"
    assert manifest["actual_tensor_strides"] == [4, 8, 16, 32]
    assert manifest["graph_integrated"] is True
    assert manifest["detection_head_integrated"] is True
    assert manifest["native_loss_integrated"] is True
    assert manifest["checkpoint_integrated"] is True
    assert manifest["runtime_payload_hash"] == bridge.payload.payload_hash
    assert manifest["detect_input_count"] == 4
    assert manifest["external_nms_added"] is False
    assert bridge.context.evidence.hook_call_counts[
        "yolo_agent.components.adapters.head.p2_head:P2HeadRuntimePlugin"
    ]["build_model"] == 1


def test_p2_runtime_uses_native_loss_backward_amp_and_export(runtime_graph) -> None:
    _, _, model, _, _ = runtime_graph
    model.args = get_cfg(overrides={"imgsz": 640})
    image = torch.rand(1, 3, 64, 64)
    model.train()
    predictions = model(image)
    assert set(predictions) == {"one2many", "one2one"}
    assert len(predictions["one2many"]["feats"]) == 4
    assert len(predictions["one2one"]["feats"]) == 4
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
        if name.startswith("model.19.")
    )
    model.zero_grad(set_to_none=True)
    model.criterion = None
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        amp_loss, _ = model.loss(batch)
    amp_loss.sum().backward()
    model.eval()
    detect = model.model[-1]
    detect.export = True
    detect.format = "torchscript"
    with torch.no_grad():
        exported = model(image)
    detect.export = False
    assert isinstance(exported, torch.Tensor)
    assert exported.shape[-1] == 6


def test_p2_runtime_records_partial_checkpoint_and_resource_risks(runtime_graph) -> None:
    _, source, _, manifest, _ = runtime_graph
    checkpoint = manifest["checkpoint"]

    assert checkpoint["checkpoint_path"] == Path(source.pt_path).resolve().as_posix()
    assert len(checkpoint["checkpoint_sha256"]) == 64
    assert checkpoint["matched_keys"]
    assert checkpoint["missing_keys"]
    assert checkpoint["unexpected_keys"]
    assert checkpoint["newly_initialized_keys"] == checkpoint["missing_keys"]
    assert 0.0 < checkpoint["matched_parameter_fraction"] < 1.0
    assert not any(
        target.startswith("model.19.") and source_key.startswith("model.19.")
        for target, source_key in checkpoint["key_mapping"].items()
    )
    assert any(
        target.startswith("model.25.") and source_key.startswith("model.19.")
        for target, source_key in checkpoint["key_mapping"].items()
    )
    assert len(manifest["generated_yaml_sha256"]) == 64
    assert manifest["parameter_delta"] > 0
    assert manifest["model_size_delta_mb"] > 0
    assert isinstance(manifest["latency_delta_ms"], float)
    assert manifest["latency_risk"] in {"increase_guarded", "measured_no_increase"}
    assert manifest["resources"]["passed"] is True
    assert all(manifest["resources"]["checks"].values())


def test_p2_runtime_fails_closed_when_resource_guard_is_exceeded(
    tmp_path: Path,
) -> None:
    source = DetectionModel("yolo26n.yaml", nc=3, verbose=False)
    source.args = get_cfg(overrides={"imgsz": 640})
    context = _context(
        tmp_path,
        audit_imgsz=64,
        latency_warmup=0,
        latency_iterations=1,
        resource_limits={
            "max_latency_regression": 100.0,
            "max_vram_regression": 0.0,
            "max_parameter_regression": 0.0,
            "max_model_size_regression": 0.0,
        },
    )
    payload = P2HeadAdapter().build_runtime_payload(
        context,
        protocol_hash="p2-guard-protocol",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={},
    )
    bridge = UltralyticsTrainerPluginBridge(
        payload.write(tmp_path / "runtime.yaml")
    )

    with pytest.raises(RuntimeError, match="P2 model graph resource guards failed"):
        bridge.invoke_transform(
            "build_model",
            source,
            trainer=SimpleNamespace(args=get_cfg(overrides={"imgsz": 640})),
        )

    manifest = json.loads(
        (tmp_path / "p2_head_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["resources"]["passed"] is False
    assert manifest["resources"]["checks"]["parameters"] is False


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("YOLO_AGENT_RUN_GPU_TESTS") != "1",
    reason="set YOLO_AGENT_RUN_GPU_TESTS=1 for optional P2 GPU smoke",
)
def test_optional_p2_graph_gpu_smoke(tmp_path: Path) -> None:
    result = P2HeadAdapter().gpu_smoke_test(
        _context(
            tmp_path,
            num_classes=3,
            audit_imgsz=64,
            latency_warmup=0,
            latency_iterations=1,
        )
    )

    assert result.passed, result.errors
    assert result.checks["actual_p2_graph"] is True
    assert result.checks["native_loss_preserved"] is True
    assert result.checks["backward"] is True
    assert result.checks["amp"] is True
