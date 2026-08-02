from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.test_data_pipeline_dataset import TransformDataset
from yolo_agent.components.adapters.data_pipeline import (
    DataPipelineManifest,
    DataPipelinePlugin,
)


def _context(tmp_path: Path) -> SimpleNamespace:
    payload = tmp_path / "payload.yaml"
    payload.write_text("payload", encoding="utf-8")
    return SimpleNamespace(
        payload_path=payload,
        payload=SimpleNamespace(protocol_hash="protocol", payload_hash="payload"),
    )


def _plugin(mechanism: str, **options: object) -> DataPipelinePlugin:
    return DataPipelinePlugin(
        mechanism_id=mechanism,
        component_id=f"augmentation.{mechanism}",
        adapter_family=f"data.augmentation.{mechanism}",
        changed_variable=f"data.{mechanism}",
        **options,
    )


def test_plugin_wraps_only_train_dataset_and_emits_manifest(tmp_path: Path) -> None:
    plugin = _plugin("scale_aware_crop", crop_scale=0.75)
    wrapped = plugin.build_train_dataset(
        context=_context(tmp_path),
        trainer=SimpleNamespace(),
        dataset=TransformDataset(),
        image_path="train",
        batch_size=2,
    )

    assert wrapped[0]["img"].shape == torch.Size([3, 16, 16])
    manifest = DataPipelineManifest.model_validate_json(
        (tmp_path / "scale_aware_crop_manifest.json").read_text()
    )
    assert manifest.identity.changed_variable == "data.scale_aware_crop"
    assert manifest.val_unchanged and manifest.test_unchanged


def test_plugin_resume_restores_schedule_epoch(tmp_path: Path) -> None:
    context = _context(tmp_path)
    plugin = _plugin(
        "multi_image_sampling_schedule",
        multi_image_count=2,
        active_epoch_start=2,
    )
    trainer = SimpleNamespace(epoch=3, args=SimpleNamespace(resume=False))
    plugin.build_train_dataset(
        context=context,
        trainer=trainer,
        dataset=TransformDataset(),
        image_path="train",
        batch_size=2,
    )
    plugin.on_checkpoint_save(context=context, trainer=trainer, checkpoints={})

    resumed = _plugin(
        "multi_image_sampling_schedule",
        multi_image_count=2,
        active_epoch_start=2,
    )
    resumed.build_train_dataset(
        context=context,
        trainer=SimpleNamespace(),
        dataset=TransformDataset(),
        image_path="train",
        batch_size=2,
    )
    resumed.on_checkpoint_load(
        context=context,
        trainer=SimpleNamespace(args=SimpleNamespace(resume=False)),
        checkpoint={},
    )

    assert resumed.dataset is not None and resumed.dataset.epoch == 3


def test_zero_probability_returns_native_dataset_object(tmp_path: Path) -> None:
    native = TransformDataset()
    plugin = _plugin("scale_aware_crop", probability=0)

    result = plugin.build_train_dataset(
        context=_context(tmp_path),
        trainer=SimpleNamespace(),
        dataset=native,
        image_path="train",
        batch_size=2,
    )

    assert result is native


def test_only_primary_rank_writes_transform_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    plugin = _plugin("object_centric_crop", crop_scale=0.75)

    plugin.build_train_dataset(
        context=context,
        trainer=SimpleNamespace(),
        dataset=TransformDataset(),
        image_path="train",
        batch_size=2,
    )

    assert not (tmp_path / "object_centric_crop_manifest.json").exists()


def test_transform_resume_state_is_rank_scoped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    plugin = _plugin("object_centric_crop", crop_scale=0.75)
    trainer = SimpleNamespace(epoch=2)
    plugin.build_train_dataset(
        context=context,
        trainer=trainer,
        dataset=TransformDataset(),
        image_path="train",
        batch_size=2,
    )
    plugin.on_checkpoint_save(context=context, trainer=trainer, checkpoints={})

    assert (tmp_path / "object_centric_crop_dataset_state.rank1.json").is_file()
