from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from yolo_agent.components.adapters.base import AdapterContext
from yolo_agent.components.adapters.head.p2_head import P2Head, P2HeadAdapter, P2HeadConfig
from yolo_agent.components.contracts import ComponentContract


def _context(tmp_path: Path, **options):
    return AdapterContext(contract=ComponentContract(
        component_id="head.p2_small_object", display_name="P2", category="detection_head",
        implementation_path="local", adapter_class="P2HeadAdapter", changes_model_graph=True,
        fixed_imgsz_compatible=True,
    ), detector_family="yolo26", head="one_to_one", workspace=tmp_path, options=options)


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
