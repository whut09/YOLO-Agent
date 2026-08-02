from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from torch.utils.data import DataLoader

from tests.test_data_pipeline_runtime import TinyDataset
from yolo_agent.components.adapters.data_pipeline import (
    DataPipelineManifest,
    SamplingPlugin,
)


def _context(tmp_path: Path) -> SimpleNamespace:
    path = tmp_path / "payload.yaml"
    path.write_text("payload", encoding="utf-8")
    return SimpleNamespace(
        payload_path=path,
        payload=SimpleNamespace(protocol_hash="protocol", payload_hash="payload"),
    )


def _plugin(mechanism: str, **options: object) -> SamplingPlugin:
    return SamplingPlugin(
        mechanism_id=mechanism,
        component_id=f"sampling.{mechanism}",
        adapter_family=f"data.sampling.{mechanism}",
        changed_variable=f"data.{mechanism}",
        **options,
    )


def test_plugin_rebuilds_train_loader_and_writes_identity_manifest(
    tmp_path: Path,
) -> None:
    plugin = _plugin("class_balanced_sampling", seed=7)
    loader = DataLoader(TinyDataset(), batch_size=2, num_workers=0)
    trainer = SimpleNamespace(epoch=0)

    rebuilt = plugin.build_train_dataloader(
        context=_context(tmp_path),
        trainer=trainer,
        dataloader=loader,
        dataset_path="train",
        batch_size=2,
        rank=-1,
    )

    assert rebuilt.dataset is loader.dataset
    manifest = DataPipelineManifest.model_validate_json(
        (tmp_path / "class_balanced_sampling_manifest.json").read_text()
    )
    assert manifest.identity.changed_variable == "data.class_balanced_sampling"
    assert manifest.exact_reproduction is False
    assert manifest.manifest_hash


def test_plugin_resume_restores_epoch_and_rejects_other_mechanism(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    loader = DataLoader(TinyDataset(), batch_size=2, num_workers=0)
    plugin = _plugin("repeat_factor_sampling", seed=11, repeat_threshold=0.8)
    trainer = SimpleNamespace(epoch=4, args=SimpleNamespace(resume=False))
    plugin.build_train_dataloader(
        context=context,
        trainer=trainer,
        dataloader=loader,
        dataset_path="train",
        batch_size=2,
        rank=-1,
    )
    plugin.on_checkpoint_save(context=context, trainer=trainer, checkpoints={})

    resumed = _plugin("repeat_factor_sampling", seed=11, repeat_threshold=0.8)
    resumed.build_train_dataloader(
        context=context,
        trainer=SimpleNamespace(),
        dataloader=loader,
        dataset_path="train",
        batch_size=2,
        rank=-1,
    )
    resumed.on_checkpoint_load(
        context=context,
        trainer=SimpleNamespace(args=SimpleNamespace(resume=False)),
        checkpoint={},
    )
    assert resumed.sampler is not None and resumed.sampler.epoch == 4

    state_path = tmp_path / "repeat_factor_sampling_state.rank0.json"
    state = json.loads(state_path.read_text())
    state["mechanism_id"] = "class_balanced_sampling"
    with pytest.raises(ValueError, match="mechanism_id"):
        resumed.on_checkpoint_load(
            context=context,
            trainer=SimpleNamespace(args=SimpleNamespace(resume=False)),
            checkpoint={"repeat_factor_sampling_sampler_state": state},
        )


def test_evidence_dependent_plugins_fail_closed(tmp_path: Path) -> None:
    dataset = TinyDataset()
    dataset.labels = [
        {key: value for key, value in label.items() if key not in {
            "is_hard_negative",
            "false_negative_score",
        }}
        for label in dataset.labels
    ]
    loader = DataLoader(dataset, batch_size=2, num_workers=0)
    kwargs = dict(
        context=_context(tmp_path),
        trainer=SimpleNamespace(),
        dataloader=loader,
        dataset_path="train",
        batch_size=2,
        rank=-1,
    )
    with pytest.raises(ValueError, match="hard-negative evidence"):
        _plugin("hard_negative_replay").build_train_dataloader(**kwargs)
    with pytest.raises(ValueError, match="class IDs and FN scores"):
        _plugin("false_negative_class_boost").build_train_dataloader(**kwargs)


def test_zero_strength_returns_native_loader_object(tmp_path: Path) -> None:
    native = DataLoader(TinyDataset(), batch_size=2, num_workers=0)
    plugin = _plugin("class_balanced_sampling", strength=0)

    result = plugin.build_train_dataloader(
        context=_context(tmp_path),
        trainer=SimpleNamespace(),
        dataloader=native,
        dataset_path="train",
        batch_size=2,
        rank=-1,
    )

    assert result is native
    manifest = DataPipelineManifest.model_validate_json(
        (tmp_path / "class_balanced_sampling_manifest.json").read_text()
    )
    assert manifest.final_exposure == [1.0, 1.0]
