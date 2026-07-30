from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from ultralytics.data.build import InfiniteDataLoader

from yolo_agent.adapters.ultralytics.plugin_bridge import PluginDetectionTrainer
from yolo_agent.components.adapters.base import AdapterContext
from yolo_agent.components.adapters.sampling.small_object_sampling import (
    DeterministicDistributedWeightedSampler,
    SmallObjectSample,
    SmallObjectSampler,
    SmallObjectSamplingAdapter,
    SmallObjectSamplingConfig,
    SmallObjectSamplingManifest,
    SmallObjectSamplingRuntimePlugin,
)
from yolo_agent.components.contracts import ComponentContract


class TinyYOLODataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        self.im_files = ["small-rare.jpg", "large-common.jpg", "small-fn.jpg"]
        self.labels = [
            {
                "im_file": self.im_files[0],
                "normalized": True,
                "bbox_format": "xywh",
                "bboxes": torch.tensor([[0.5, 0.5, 0.05, 0.05]]),
                "cls": torch.tensor([[2.0]]),
            },
            {
                "im_file": self.im_files[1],
                "normalized": True,
                "bbox_format": "xywh",
                "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
                "cls": torch.tensor([[1.0]]),
            },
            {
                "im_file": self.im_files[2],
                "normalized": True,
                "bbox_format": "xywh",
                "bboxes": torch.tensor([[0.5, 0.5, 0.08, 0.08]]),
                "cls": torch.tensor([[3.0]]),
            },
        ]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"img": torch.full((3, 16, 16), float(index)), "index": torch.tensor(index)}


def _context(tmp_path: Path) -> AdapterContext:
    return AdapterContext(
        contract=ComponentContract(
            component_id="sampling.small_object",
            display_name="Sampler",
            category="sampling",
            implementation_path=(
                "yolo_agent.components.adapters.sampling.small_object_sampling"
            ),
            adapter_class="SmallObjectSamplingAdapter",
            fixed_imgsz_compatible=True,
            maturity="smoke_passed",
        ),
        workspace=tmp_path,
        options={"imgsz": 640, "fn_heavy_class_ids": [3], "seed": 17},
    )


def _loader() -> DataLoader[dict[str, torch.Tensor]]:
    return DataLoader(TinyYOLODataset(), batch_size=2, shuffle=False, num_workers=0)


def _plugin_context(tmp_path: Path) -> SimpleNamespace:
    payload_path = tmp_path / "adapter_runtime_payload.yaml"
    payload_path.write_text("payload", encoding="utf-8")
    return SimpleNamespace(
        payload_path=payload_path,
        payload=SimpleNamespace(
            protocol_hash="protocol-1",
            payload_hash="payload-hash-1",
        ),
    )


def test_sampler_boosts_small_rare_and_fn_heavy_images_with_bounded_weights() -> None:
    config = SmallObjectSamplingConfig(
        fn_heavy_class_ids=[3],
        max_weight=2.5,
        max_oversampling_ratio=2.0,
    )
    values, manifest = SmallObjectSampler(config).weights(
        [
            SmallObjectSample(
                image_path="small-rare.jpg", normalized_areas=[0.005], class_ids=[2]
            ),
            SmallObjectSample(
                image_path="large-common.jpg",
                normalized_areas=[0.2, 0.3, 0.4],
                class_ids=[1, 1, 1],
            ),
            SmallObjectSample(
                image_path="small-fn.jpg", normalized_areas=[0.006], class_ids=[3]
            ),
        ],
        dataset_manifest="dataset-v1",
    )

    assert values[0] > values[1]
    assert values[2] > values[1]
    assert max(values) / min(values) <= 2.0
    assert manifest.dataset_manifest == "dataset-v1"
    assert manifest.class_counts == {"1": 3, "2": 1, "3": 1}
    assert manifest.small_image_count == 2
    assert manifest.clipping_statistics["clipped_count"] >= 1
    assert len(manifest.adapter_hash) == 64


def test_validation_samples_are_not_resampled() -> None:
    values, manifest = SmallObjectSampler().weights(
        [
            SmallObjectSample(
                image_path="train.jpg",
                split="train",
                normalized_areas=[0.01],
                class_ids=[1],
            ),
            SmallObjectSample(
                image_path="val.jpg",
                split="val",
                normalized_areas=[0.001],
                class_ids=[1],
            ),
        ]
    )

    assert len(values) == 1
    assert manifest.val_unchanged
    assert "val.jpg" not in manifest.weights


def test_runtime_plugin_rebuilds_train_loader_and_writes_complete_manifest(
    tmp_path: Path,
) -> None:
    plugin = SmallObjectSamplingRuntimePlugin(
        imgsz=640,
        seed=17,
        fn_heavy_class_ids=[3],
        dataset_manifest="tiny-coco-v1",
    )
    trainer = SimpleNamespace()
    context = _plugin_context(tmp_path)
    rebuilt = plugin.build_train_dataloader(
        context=context,
        trainer=trainer,
        dataloader=_loader(),
        dataset_path="train/images",
        batch_size=2,
        rank=-1,
    )

    assert rebuilt is not None
    assert isinstance(rebuilt.sampler, DeterministicDistributedWeightedSampler)
    batch = next(iter(rebuilt))
    assert batch["img"].shape == torch.Size([2, 3, 16, 16])
    assert trainer.small_object_sampler is rebuilt.sampler
    manifest = SmallObjectSamplingManifest.model_validate_json(
        (tmp_path / "sampler_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.dataset_manifest == "tiny-coco-v1"
    assert manifest.split == "train"
    assert manifest.seed == 17
    assert manifest.sample_count == 3
    assert len(manifest.raw_weights) == len(manifest.final_weights) == 3
    assert manifest.area_thresholds == {"small": 0.01}
    assert manifest.protocol_hash == "protocol-1"
    assert manifest.runtime_payload_hash == context.payload.payload_hash
    assert manifest.plugin_version == "small_object_sampling_runtime.v1"


def test_runtime_plugin_rebuilds_ultralytics_infinite_loader_on_cpu(tmp_path: Path) -> None:
    original = InfiniteDataLoader(
        dataset=TinyYOLODataset(),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    plugin = SmallObjectSamplingRuntimePlugin(imgsz=640)
    trainer = SimpleNamespace(args=SimpleNamespace(seed=31))

    rebuilt = plugin.build_train_dataloader(
        context=_plugin_context(tmp_path),
        trainer=trainer,
        dataloader=original,
        dataset_path="train/images",
        batch_size=2,
        rank=-1,
    )

    assert isinstance(rebuilt, InfiniteDataLoader)
    assert rebuilt.sampler.seed == 31
    assert next(iter(rebuilt))["img"].shape == torch.Size([2, 3, 16, 16])
    manifest = SmallObjectSamplingManifest.model_validate_json(
        (tmp_path / "sampler_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.seed == 31


def test_distributed_sampler_is_deterministic_and_shards_global_positions() -> None:
    kwargs = {
        "weights": [1.0, 2.0, 3.0, 4.0],
        "sample_count": 7,
        "seed": 23,
        "world_size": 2,
        "dataset_manifest": "dataset-v1",
        "adapter_hash": "adapter-v1",
    }
    rank0 = DeterministicDistributedWeightedSampler(rank=0, **kwargs)
    rank1 = DeterministicDistributedWeightedSampler(rank=1, **kwargs)
    global_stream = rank0.global_indices()

    assert list(rank0) == global_stream[0::2]
    assert list(rank1) == global_stream[1::2]
    assert len(rank0) == len(rank1) == 4
    assert rank0.global_indices() == rank1.global_indices()
    rank0.set_epoch(2)
    rank1.set_epoch(2)
    assert rank0.global_indices() == rank1.global_indices()
    assert rank0.global_indices() != global_stream


def test_only_primary_rank_writes_sampler_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _plugin_context(tmp_path)
    monkeypatch.setenv("WORLD_SIZE", "2")
    primary = SmallObjectSamplingRuntimePlugin(imgsz=640, seed=5)
    primary.build_train_dataloader(
        context=context,
        trainer=SimpleNamespace(),
        dataloader=_loader(),
        dataset_path="train/images",
        batch_size=2,
        rank=0,
    )
    manifest_path = tmp_path / "sampler_manifest.json"
    original = manifest_path.read_text(encoding="utf-8")

    secondary = SmallObjectSamplingRuntimePlugin(imgsz=640, seed=99)
    secondary.build_train_dataloader(
        context=context,
        trainer=SimpleNamespace(),
        dataloader=_loader(),
        dataset_path="train/images",
        batch_size=2,
        rank=1,
    )

    assert manifest_path.read_text(encoding="utf-8") == original


def test_sampler_checkpoint_state_round_trip_and_mismatch_rejection(tmp_path: Path) -> None:
    context = _plugin_context(tmp_path)
    checkpoint = tmp_path / "last.pt"
    plugin = SmallObjectSamplingRuntimePlugin(imgsz=640, seed=11)
    trainer = SimpleNamespace(epoch=4, args=SimpleNamespace(resume=False))
    plugin.build_train_dataloader(
        context=context,
        trainer=trainer,
        dataloader=_loader(),
        dataset_path="train/images",
        batch_size=2,
        rank=-1,
    )
    plugin.on_checkpoint_save(
        context=context,
        trainer=trainer,
        checkpoints={"last": checkpoint},
    )

    resumed = SmallObjectSamplingRuntimePlugin(imgsz=640, seed=11)
    resume_trainer = SimpleNamespace(args=SimpleNamespace(resume=str(checkpoint)))
    resumed.build_train_dataloader(
        context=context,
        trainer=resume_trainer,
        dataloader=_loader(),
        dataset_path="train/images",
        batch_size=2,
        rank=-1,
    )
    resumed.on_checkpoint_load(
        context=context,
        trainer=resume_trainer,
        checkpoint={},
    )
    assert resumed.sampler is not None and resumed.sampler.epoch == 4

    incompatible = SmallObjectSamplingRuntimePlugin(imgsz=640, seed=12)
    incompatible.build_train_dataloader(
        context=context,
        trainer=resume_trainer,
        dataloader=_loader(),
        dataset_path="train/images",
        batch_size=2,
        rank=-1,
    )
    with pytest.raises(ValueError, match="resume mismatch for seed"):
        incompatible.on_checkpoint_load(
            context=context,
            trainer=resume_trainer,
            checkpoint=json.loads(
                (tmp_path / "small_object_sampler_state.rank0.json").read_text(
                    encoding="utf-8"
                )
            ),
        )


def test_plugin_detection_trainer_leaves_val_loader_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = object()
    calls: list[str] = []
    trainer = PluginDetectionTrainer.__new__(PluginDetectionTrainer)
    trainer.plugin_bridge = SimpleNamespace(
        invoke_transform=lambda hook, value, **kwargs: calls.append(hook) or value
    )
    monkeypatch.setattr(
        "ultralytics.models.yolo.detect.train.DetectionTrainer.get_dataloader",
        lambda self, dataset_path, batch_size=16, rank=0, mode="train": original,
    )

    result = trainer.get_dataloader("val/images", mode="val")

    assert result is original
    assert calls == []


def test_sampler_adapter_has_real_runtime_payload_and_single_changed_variable(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    adapter = SmallObjectSamplingAdapter()
    preview = adapter.prepare_patch({}, {}, context)
    payload = adapter.build_runtime_payload(
        context,
        protocol_hash="protocol-1",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={"training_config": preview.patched_training_config},
    )

    assert [item.field for item in preview.operations] == ["data.sampling_policy"]
    assert preview.declared_modified_fields == ["training_config.data.sampling_policy"]
    assert len(payload.dataloader_plugin) == 1
    assert payload.supports_amp and payload.supports_ddp and payload.supports_resume
    payload.verify_imports()
    assert adapter.smoke_test(context).passed
