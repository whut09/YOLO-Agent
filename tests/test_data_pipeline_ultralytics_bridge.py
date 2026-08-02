from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from tests.test_data_pipeline_dataset import TransformDataset
from tests.test_data_pipeline_runtime import TinyDataset
from yolo_agent.adapters.ultralytics.plugin_bridge import (
    PluginDetectionTrainer,
    UltralyticsTrainerPluginBridge,
)
from yolo_agent.adapters.ultralytics.plugin_context import PluginRuntimeEvidence
from yolo_agent.components.adapters.base import AdapterContext
from yolo_agent.components.adapters.data_pipeline import (
    ClassBalancedSamplingAdapter,
    ScaleAwareCropAdapter,
)
from yolo_agent.components.contracts import ComponentContract


def _context(adapter, tmp_path: Path, options: dict) -> AdapterContext:  # type: ignore[no-untyped-def]
    return AdapterContext(
        contract=ComponentContract(
            component_id=adapter.component_id,
            display_name=adapter.mechanism_id,
            category="sampling",
            implementation_path=(
                "yolo_agent.components.adapters.data_pipeline.adapters"
            ),
            adapter_class=type(adapter).__name__,
            maturity="adapter_implemented",
        ),
        detector_family="yolo26",
        workspace=tmp_path,
        options=options,
    )


def _bridge(adapter, tmp_path: Path, options: dict) -> UltralyticsTrainerPluginBridge:  # type: ignore[no-untyped-def]
    context = _context(adapter, tmp_path, options)
    payload = adapter.build_runtime_payload(
        context,
        protocol_hash="protocol-v1",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={},
    )
    path = payload.write(tmp_path / "adapter_runtime_payload.yaml")
    return UltralyticsTrainerPluginBridge(path)


def test_sampling_payload_changes_real_train_loader_and_records_hook(
    tmp_path: Path,
) -> None:
    bridge = _bridge(ClassBalancedSamplingAdapter(), tmp_path, {"imgsz": 640})
    native = DataLoader(TinyDataset(), batch_size=2, num_workers=0)
    transformed = bridge.invoke_transform(
        "build_train_dataloader",
        native,
        trainer=SimpleNamespace(),
        dataset_path="train",
        batch_size=2,
        rank=-1,
    )

    assert transformed is not native
    assert transformed.sampler.mechanism_id == "class_balanced_sampling"
    bridge.verify_required_hooks()
    evidence = PluginRuntimeEvidence.model_validate_json(
        bridge.context.evidence_path.read_text()
    )
    reference = bridge.payload.dataloader_plugin[0].reference
    assert evidence.changed_variables == {
        "data.class_balanced_sampling": bridge.payload.changed_variables[
            "data.class_balanced_sampling"
        ]
    }
    assert evidence.hook_call_counts[reference]["build_train_dataloader"] == 1


def test_transform_payload_changes_train_dataset_and_keeps_val_loader(
    tmp_path: Path,
) -> None:
    bridge = _bridge(
        ScaleAwareCropAdapter(),
        tmp_path,
        {"imgsz": 640, "crop_scale": 0.75},
    )
    transformed = bridge.invoke_transform(
        "build_train_dataset",
        TransformDataset(),
        trainer=SimpleNamespace(),
        image_path="train",
        batch_size=2,
    )
    trainer = object.__new__(PluginDetectionTrainer)
    trainer.plugin_bridge = bridge
    validation_loader = DataLoader(TinyDataset(), batch_size=2, num_workers=0)
    unchanged = trainer.apply_dataloader_plugins(
        validation_loader,
        dataset_path="val",
        batch_size=2,
        rank=-1,
        mode="val",
    )

    output = transformed[1]
    native = TransformDataset()[1]
    assert transformed.transform_count == 1
    assert output["bboxes"].shape != native["bboxes"].shape or not torch.equal(
        output["bboxes"], native["bboxes"]
    )
    assert unchanged is validation_loader
    bridge.verify_required_hooks()
