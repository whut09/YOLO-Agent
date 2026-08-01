"""Offline and CPU tests for the Ultralytics trainer plugin bridge."""

from __future__ import annotations

import copy
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytest.importorskip("ultralytics")

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

from yolo_agent.adapters.ultralytics.plugin_bridge import (
    PluginDetectionTrainer,
    PluginCriterionWrapper,
    PluginExecutionError,
    UltralyticsTrainerPluginBridge,
)
from yolo_agent.adapters.ultralytics.plugin_context import (
    PluginRuntimeEvidence,
    audit_installed_ultralytics,
)
from yolo_agent.adapters.ultralytics.runtime_entrypoint import (
    parse_ultralytics_train_command,
    run_ultralytics_training,
)
from yolo_agent.components.adapters import (
    AdapterRuntimePayload,
    RollbackPlan,
    RuntimePluginReference,
)
from yolo_agent.components.adapters.validation import validate_runtime_plugin_hooks


PLUGIN_REFERENCE = "yolo_agent.components.adapters.dummy:DummyRuntimePlugin"


def test_checkpoint_deepcopy_strips_training_only_bridge_state() -> None:
    native = SimpleNamespace(name="native-criterion")
    bridge = SimpleNamespace(lock=threading.Lock())
    wrapper = PluginCriterionWrapper(native, bridge, object(), object())

    restored = copy.deepcopy(wrapper)

    assert isinstance(restored, SimpleNamespace)
    assert restored.name == "native-criterion"


def _payload(tmp_path: Path, *, options: dict[str, Any] | None = None) -> AdapterRuntimePayload:
    return AdapterRuntimePayload(
        component_ids=["dummy.component"],
        adapter_classes=["DummyAdapter"],
        adapter_versions={"dummy.component": "dummy.v1"},
        source_commits={"dummy.component": "local-test"},
        trainer_plugin=[
            RuntimePluginReference(
                reference=PLUGIN_REFERENCE,
                options=options or {},
                required_hooks=["build_model"],
            )
        ],
        generated_config={"training_config": {"imgsz": 640, "amp": True}},
        changed_variables={"training.adapter_marker": "active"},
        rollback_plan=RollbackPlan(actions=["discard test runtime"]),
        protocol_hash="protocol-1",
        base_command=[
            "yolo",
            "detect",
            "train",
            "model=yolo26n.pt",
            "data=coco.yaml",
            "imgsz=640",
            "epochs=3",
        ],
        supports_amp=True,
        supports_ddp=True,
        supports_resume=True,
    )


def _bridge(tmp_path: Path, *, options: dict[str, Any] | None = None) -> UltralyticsTrainerPluginBridge:
    path = _payload(tmp_path, options=options).write(tmp_path / "adapter_runtime_payload.yaml")
    return UltralyticsTrainerPluginBridge(path)


def test_installed_ultralytics_signatures_are_audited() -> None:
    audit = audit_installed_ultralytics()

    assert audit.compatible, audit.blocked_by
    assert audit.version.startswith("8.4.")
    assert audit.method_parameters["DetectionTrainer.get_model"] == [
        "self",
        "cfg",
        "weights",
        "verbose",
    ]
    assert len(audit.signature_hash) == 64


def test_unreviewed_ultralytics_version_fails_closed() -> None:
    audit = audit_installed_ultralytics(
        version="9.0.0",
        trainer_class=DetectionTrainer,
        model_class=DetectionModel,
    )

    assert not audit.compatible
    assert audit.blocked_by[0] == "unsupported_ultralytics_version:9.0.0"


def test_train_command_parser_preserves_original_overrides() -> None:
    task, arguments = parse_ultralytics_train_command(
        [
            "yolo",
            "detect",
            "train",
            "model=yolo26n.pt",
            "data=E:/dataset/coco.yaml",
            "imgsz=640",
            "epochs=10",
            "amp=True",
            "resume=last.pt",
            "device=0,1",
        ]
    )

    assert task == "detect"
    assert arguments == {
        "model": "yolo26n.pt",
        "data": "E:/dataset/coco.yaml",
        "imgsz": 640,
        "epochs": 10,
        "amp": True,
        "resume": "last.pt",
        "device": "0,1",
    }


def test_bridge_rejects_non_fixed_imgsz_and_multi_scale(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)

    with pytest.raises(PluginExecutionError, match="fixed imgsz=640"):
        bridge.validate_training_args({"imgsz": 672})
    with pytest.raises(PluginExecutionError, match="multi_scale"):
        bridge.validate_training_args({"imgsz": 640, "multi_scale": 0.5})


@pytest.mark.parametrize(
    ("capability", "arguments", "message"),
    [
        ("supports_amp", {"imgsz": 640, "amp": True}, "does not support AMP"),
        (
            "supports_resume",
            {"imgsz": 640, "resume": "last.pt"},
            "does not support checkpoint resume",
        ),
        ("supports_ddp", {"imgsz": 640, "device": "0,1"}, "does not support DDP"),
    ],
)
def test_declared_runtime_capabilities_are_enforced(
    tmp_path: Path,
    capability: str,
    arguments: dict[str, Any],
    message: str,
) -> None:
    payload = _payload(tmp_path).model_copy(update={capability: False})
    path = payload.write(tmp_path / f"{capability}.yaml")
    bridge = UltralyticsTrainerPluginBridge(path)

    with pytest.raises(PluginExecutionError, match=message):
        bridge.validate_training_args(arguments)


def test_plugin_loading_failure_does_not_fall_back(tmp_path: Path) -> None:
    payload = _payload(tmp_path, options={"unknown_constructor_option": True})
    path = payload.write(tmp_path / "adapter_runtime_payload.yaml")

    with pytest.raises(PluginExecutionError, match="failed to load"):
        UltralyticsTrainerPluginBridge(path)
    evidence = PluginRuntimeEvidence.model_validate_json(
        (tmp_path / "plugin_runtime_evidence.json").read_text(encoding="utf-8")
    )
    assert evidence.failures and "failed to load" in evidence.failures[0]


def test_runtime_evidence_merges_stale_bridge_state(tmp_path: Path) -> None:
    payload_path = _payload(tmp_path).write(tmp_path / "adapter_runtime_payload.yaml")
    first = UltralyticsTrainerPluginBridge(payload_path)
    second = UltralyticsTrainerPluginBridge(payload_path)
    model = SimpleNamespace()
    first.invoke_transform("build_model", model, trainer=SimpleNamespace())

    second.context.record_failure("runtime_entrypoint", "train", "synthetic failure")

    evidence = PluginRuntimeEvidence.model_validate_json(
        second.context.evidence_path.read_text(encoding="utf-8")
    )
    assert evidence.hook_call_counts[PLUGIN_REFERENCE]["build_model"] == 1
    assert evidence.failures == ["runtime_entrypoint:train:synthetic failure"]


def test_required_runtime_hooks_must_be_observed(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)

    with pytest.raises(PluginExecutionError, match="required runtime hooks were not called"):
        bridge.verify_required_hooks()

    bridge.invoke_transform("build_model", SimpleNamespace(), trainer=SimpleNamespace())
    bridge.verify_required_hooks()


def test_mock_trainer_invokes_all_hooks_and_records_identity(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path, options={"loss_scale": 2.0})
    trainer = SimpleNamespace()

    class TinyCriterion:
        def __call__(self, predictions: torch.Tensor, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            loss = (predictions - batch["target"]).square().mean()
            return loss, loss.detach().reshape(1)

    class TinyModel(torch.nn.Module):
        def init_criterion(self) -> TinyCriterion:
            return TinyCriterion()

    model = TinyModel()
    model = bridge.install_model_hooks(model, trainer=trainer)
    dataset = bridge.invoke_transform("build_train_dataset", [1, 2], trainer=trainer)
    dataloader = bridge.invoke_transform("build_train_dataloader", iter(dataset), trainer=trainer)
    validator = bridge.invoke_transform("build_validator", object(), trainer=trainer)
    bridge.invoke_event("on_train_batch_start", trainer=trainer, batch={"img": "batch"})
    bridge.invoke_event("on_train_batch_end", trainer=trainer, batch={}, loss=None)
    bridge.invoke_event("on_checkpoint_save", trainer=trainer, checkpoints={})
    bridge.invoke_event("on_checkpoint_load", trainer=trainer, checkpoint={"epoch": 1})

    assert model.yolo_agent_dummy_runtime is True
    assert dataset == [1, 2]
    assert list(dataloader) == [1, 2]
    assert validator is not None
    criterion = model.init_criterion()
    predictions = torch.randn(2, 4, requires_grad=True)
    batch = {"target": torch.zeros(2, 4)}
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss, loss_items = criterion(predictions, batch)
    expected = (predictions - batch["target"]).square().mean() * 2.0
    assert loss.shape == torch.Size([])
    assert loss_items.shape == torch.Size([1])
    assert torch.allclose(loss, expected)
    loss.backward()
    assert predictions.grad is not None and predictions.grad.shape == predictions.shape

    evidence = PluginRuntimeEvidence.model_validate_json(
        bridge.context.evidence_path.read_text(encoding="utf-8")
    )
    descriptor = evidence.plugins[0]
    assert evidence.component_ids == ["dummy.component"]
    assert evidence.changed_variables == {"training.adapter_marker": "active"}
    assert descriptor.class_name == "DummyRuntimePlugin"
    assert descriptor.version == "dummy_runtime.v1"
    assert len(descriptor.source_hash) == 64
    counts = evidence.hook_call_counts[PLUGIN_REFERENCE]
    assert counts == {
        "build_criterion": 1,
        "build_model": 1,
        "build_train_dataloader": 1,
        "build_train_dataset": 1,
        "build_validator": 1,
        "compute_loss": 1,
        "on_checkpoint_load": 1,
        "on_checkpoint_save": 1,
        "on_train_batch_end": 1,
        "on_train_batch_start": 1,
    }


def test_runtime_entrypoint_uses_plugin_trainer_and_preserves_kwargs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload(tmp_path)
    path = payload.write(tmp_path / "adapter_runtime_payload.yaml")
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        UltralyticsTrainerPluginBridge,
        "verify_required_hooks",
        lambda self: captured.update(required_hooks_verified=True),
    )

    class FakeYOLO:
        def __init__(self, model: str, task: str, verbose: bool = False) -> None:
            captured["model"] = model
            captured["task"] = task
            captured["verbose"] = verbose

        def train(self, *, trainer: type[Any], **kwargs: Any) -> None:
            captured["trainer"] = trainer
            captured["kwargs"] = kwargs

    monkeypatch.setattr("ultralytics.YOLO", FakeYOLO)

    result = run_ultralytics_training(
        path,
        [
            "yolo",
            "detect",
            "train",
            "model=yolo26n.pt",
            "data=coco.yaml",
            "imgsz=640",
            "epochs=3",
            "amp=True",
            "resume=last.pt",
        ],
    )

    assert result == 0
    assert captured["model"] == "yolo26n.pt"
    assert captured["task"] == "detect"
    assert captured["trainer"] is PluginDetectionTrainer
    assert captured["required_hooks_verified"] is True
    assert captured["kwargs"] == {
        "data": "coco.yaml",
        "imgsz": 640,
        "epochs": 3,
        "amp": True,
        "resume": "last.pt",
    }


def test_plugin_detection_trainer_dispatches_real_lifecycle_methods(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _bridge(tmp_path)

    class TinyCriterion:
        def __call__(self, predictions: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
            return predictions.square().mean()

    class TinyModel(torch.nn.Module):
        def init_criterion(self) -> TinyCriterion:
            return TinyCriterion()

    monkeypatch.setattr(DetectionTrainer, "get_model", lambda *_args, **_kwargs: TinyModel())
    monkeypatch.setattr(DetectionTrainer, "build_dataset", lambda *_args, **_kwargs: ["dataset"])
    monkeypatch.setattr(DetectionTrainer, "get_dataloader", lambda *_args, **_kwargs: ["loader"])
    monkeypatch.setattr(DetectionTrainer, "get_validator", lambda *_args, **_kwargs: "validator")
    monkeypatch.setattr(DetectionTrainer, "preprocess_batch", lambda _self, batch: batch)
    monkeypatch.setattr(DetectionTrainer, "save_model", lambda _self: True)
    monkeypatch.setattr(DetectionTrainer, "resume_training", lambda _self, _ckpt: None)

    trainer = object.__new__(PluginDetectionTrainer)
    trainer.plugin_bridge = bridge
    trainer._plugin_batch = None
    trainer.last = tmp_path / "last.pt"
    trainer.best = tmp_path / "best.pt"
    trainer.resume = True
    trainer.loss = torch.tensor(1.0)
    trainer.loss_items = torch.tensor([1.0])

    model = trainer.get_model()
    assert model.yolo_agent_dummy_runtime is True
    assert trainer.build_dataset("images", mode="train") == ["dataset"]
    assert trainer.get_dataloader("images", mode="train") == ["loader"]
    assert trainer.get_validator() == "validator"
    batch = {"img": torch.ones(1, 3, 8, 8)}
    assert trainer.preprocess_batch(batch) is batch
    trainer._run_plugin_batch_end(trainer)
    assert trainer.save_model() is True
    trainer.resume_training({"epoch": 0})

    evidence = PluginRuntimeEvidence.model_validate_json(
        bridge.context.evidence_path.read_text(encoding="utf-8")
    )
    counts = evidence.hook_call_counts[PLUGIN_REFERENCE]
    for hook in (
        "build_model",
        "build_train_dataset",
        "build_train_dataloader",
        "build_validator",
        "on_train_batch_start",
        "on_train_batch_end",
        "on_checkpoint_save",
        "on_checkpoint_load",
    ):
        assert counts[hook] == 1


def test_checkpoint_serialization_hooks_wrap_native_save_even_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class RecordingBridge:
        def invoke_event(self, hook: str, **_kwargs: Any) -> None:
            calls.append(hook)

    def fail_save(_trainer: Any) -> None:
        calls.append("native_save")
        raise RuntimeError("checkpoint failed")

    monkeypatch.setattr(DetectionTrainer, "save_model", fail_save)
    trainer = object.__new__(PluginDetectionTrainer)
    trainer.plugin_bridge = RecordingBridge()

    with pytest.raises(RuntimeError, match="checkpoint failed"):
        trainer.save_model()

    assert calls == [
        "on_model_serialize_start",
        "native_save",
        "on_model_serialize_end",
    ]


def test_runtime_validation_accepts_model_serialization_hooks(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload.trainer_plugin[0].required_hooks = [
        "on_model_serialize_start",
        "on_model_serialize_end",
    ]
    payload.verify_imports()

    report = validate_runtime_plugin_hooks(payload)

    assert report["runtime_plugin_hooks_verified"] >= 2
    assert "on_model_serialize_end" in str(report["runtime_plugin_identities"])


def test_dataloader_plugin_boundary_is_train_only(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    trainer = object.__new__(PluginDetectionTrainer)
    trainer.plugin_bridge = bridge
    train_loader = ["train"]
    validation_loader = ["validation"]

    transformed = trainer.apply_dataloader_plugins(
        train_loader,
        dataset_path="images/train",
        batch_size=2,
        rank=-1,
        mode="train",
    )
    unchanged = trainer.apply_dataloader_plugins(
        validation_loader,
        dataset_path="images/val",
        batch_size=2,
        rank=-1,
        mode="val",
    )

    assert transformed == train_loader
    assert unchanged is validation_loader
    evidence = PluginRuntimeEvidence.model_validate_json(
        bridge.context.evidence_path.read_text(encoding="utf-8")
    )
    assert evidence.hook_call_counts[PLUGIN_REFERENCE]["build_train_dataloader"] == 1
