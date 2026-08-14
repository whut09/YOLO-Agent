from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.components.adapters.base import AdapterContext
from yolo_agent.components.adapters.data_pipeline import (
    ClassBalancedSamplingAdapter,
    FalseNegativeClassBoostAdapter,
    HardNegativeReplayAdapter,
    MultiImageSamplingScheduleAdapter,
    ObjectCentricCropAdapter,
    RareClassCopyPasteAdapter,
    RepeatFactorSamplingAdapter,
    ScaleAwareCropAdapter,
    SmallObjectWeightedSamplingAdapter,
)
from yolo_agent.components.adapters.data_pipeline.hard_negative import (
    HardNegativeManifest,
    HardNegativeRecord,
)
from yolo_agent.components.contracts import ComponentContract


def _context(adapter, tmp_path: Path) -> AdapterContext:  # type: ignore[no-untyped-def]
    return AdapterContext(
        contract=ComponentContract(
            component_id=adapter.component_id,
            display_name=adapter.mechanism_id,
            category="sampling",
            implementation_path=(
                "yolo_agent.components.adapters.data_pipeline.adapters"
            ),
            adapter_class=type(adapter).__name__,
            fixed_imgsz_compatible=True,
            maturity="adapter_implemented",
        ),
        detector_family="yolo26",
        workspace=tmp_path,
        options={"imgsz": 640},
    )


@pytest.mark.parametrize(
    "adapter",
    [SmallObjectWeightedSamplingAdapter(), ClassBalancedSamplingAdapter()],
)
def test_sampling_adapter_payload_has_one_exact_runtime_identity(
    adapter,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    context = _context(adapter, tmp_path)
    preview = adapter.prepare_patch({}, {}, context)
    payload = adapter.build_runtime_payload(
        context,
        protocol_hash="protocol",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={"training_config": preview.patched_training_config},
    )

    assert [item.field for item in preview.operations] == [adapter.changed_variable]
    assert set(payload.changed_variables) == {adapter.changed_variable}
    assert payload.component_ids == [adapter.component_id]
    assert payload.dataloader_plugin[0].options["mechanism_id"] == adapter.mechanism_id
    assert payload.dataloader_plugin[0].required_hooks == ["build_train_dataloader"]
    assert adapter.smoke_test(context).passed
    payload.verify_imports()


@pytest.mark.parametrize(
    ("adapter", "options"),
    [
        (RepeatFactorSamplingAdapter(), {"repeat_threshold": 0.8}),
        (HardNegativeReplayAdapter(), {}),
        (FalseNegativeClassBoostAdapter(), {"target_class_ids": [2]}),
    ],
)
def test_evidence_sampling_adapters_keep_distinct_payloads(
    adapter,
    options: dict[str, object],
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    context = _context(adapter, tmp_path).model_copy(update={"options": options})
    if isinstance(adapter, HardNegativeReplayAdapter):
        manifest = HardNegativeManifest.from_records(
            dataset_manifest_hash="train-dataset",
            source_run_id="baseline",
            baseline_protocol_hash="protocol",
            records=[HardNegativeRecord(
                image_id="1",
                sample_index=0,
                predicted_class=1,
                score=0.9,
                bbox=[0.0, 0.0, 1.0, 1.0],
                error_type="background_false_positive",
            )],
        )
        manifest_path = tmp_path / "hard_negative_manifest.json"
        manifest.write(manifest_path)
        context = context.model_copy(update={"options": {
            "manifest_path": manifest_path,
            "manifest_hash": manifest.manifest_hash,
            "dataset_manifest_hash": "train-dataset",
            "baseline_protocol_hash": "protocol",
            "evidence_id": manifest.evidence_id,
        }})
    payload = adapter.build_runtime_payload(
        context,
        protocol_hash="protocol",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={},
    )

    assert set(payload.changed_variables) == {f"data.{adapter.mechanism_id}"}
    assert payload.component_ids == [adapter.component_id]
    assert payload.dataloader_plugin[0].options["mechanism_id"] == (
        adapter.mechanism_id
    )
    assert adapter.smoke_test(context).passed


@pytest.mark.parametrize(
    ("adapter", "options"),
    [
        (RareClassCopyPasteAdapter(), {"rare_class_ids": [3]}),
        (ScaleAwareCropAdapter(), {"crop_scale": 0.75}),
        (ObjectCentricCropAdapter(), {"crop_scale": 0.75}),
        (
            MultiImageSamplingScheduleAdapter(),
            {"multi_image_count": 2, "active_epoch_start": 0},
        ),
    ],
)
def test_transform_adapters_use_train_dataset_hook(
    adapter,
    options: dict[str, object],
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    context = _context(adapter, tmp_path).model_copy(update={"options": options})
    payload = adapter.build_runtime_payload(
        context,
        protocol_hash="protocol",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={},
    )

    assert set(payload.changed_variables) == {adapter.changed_variable}
    assert payload.dataloader_plugin[0].required_hooks == ["build_train_dataset"]
    assert payload.dataloader_plugin[0].options["mechanism_id"] == (
        adapter.mechanism_id
    )
    assert adapter.smoke_test(context).passed
    payload.verify_imports()


@pytest.mark.parametrize(
    "adapter",
    [
        SmallObjectWeightedSamplingAdapter(),
        ClassBalancedSamplingAdapter(),
        RepeatFactorSamplingAdapter(),
        HardNegativeReplayAdapter(),
        FalseNegativeClassBoostAdapter(),
        RareClassCopyPasteAdapter(),
        ScaleAwareCropAdapter(),
        ObjectCentricCropAdapter(),
        MultiImageSamplingScheduleAdapter(),
    ],
)
def test_all_data_adapters_emit_auditable_train_only_payload(
    adapter,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    context = _context(adapter, tmp_path).model_copy(
        update={"options": {"imgsz": 640}}
    )
    generated_config: dict[str, object] = {}
    if isinstance(adapter, HardNegativeReplayAdapter):
        manifest = HardNegativeManifest.from_records(
            dataset_manifest_hash="train-dataset",
            source_run_id="baseline",
            baseline_protocol_hash="protocol-v1",
            records=[HardNegativeRecord(
                image_id="1",
                sample_index=0,
                predicted_class=1,
                score=0.9,
                bbox=[0.0, 0.0, 1.0, 1.0],
                error_type="background_false_positive",
            )],
        )
        path = tmp_path / "hard_negative_manifest.json"
        manifest.write(path)
        context = context.model_copy(update={"options": {
            "imgsz": 640,
            "manifest_path": path,
            "manifest_hash": manifest.manifest_hash,
            "dataset_manifest_hash": "train-dataset",
            "baseline_protocol_hash": "protocol-v1",
            "evidence_id": manifest.evidence_id,
        }})
        generated_config = {"training_config": {}}
    payload = adapter.build_runtime_payload(
        context,
        protocol_hash="protocol-v1",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config=generated_config,
    )
    plugin = payload.dataloader_plugin[0]

    assert payload.protocol_hash == "protocol-v1"
    assert len(payload.payload_hash) == 64
    assert payload.component_ids == [adapter.component_id]
    assert set(payload.changed_variables) == {f"data.{adapter.mechanism_id}"}
    assert plugin.options["changed_variable"] == f"data.{adapter.mechanism_id}"
    assert "paper_method_profiles" not in plugin.options
    assert payload.supports_ddp and payload.supports_resume
    assert payload.base_command[-1] == "imgsz=640"
    assert payload.expected_artifacts[0].relative_path.name == (
        f"{adapter.mechanism_id}_manifest.json"
    )
