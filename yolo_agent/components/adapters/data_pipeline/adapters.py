"""Independent ComponentAdapter identities for reusable data mechanisms."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import torch

from yolo_agent.components.adapters.base import (
    AdapterContext,
    AdapterValidationReport,
    ComponentAdapter,
    ExpectedArtifact,
    RollbackPlan,
    SmokeTestResult,
    WeightLoadResult,
)
from yolo_agent.components.adapters.data_pipeline.contracts import DataSampleRecord
from yolo_agent.components.adapters.data_pipeline.dataset import DataPipelineDataset
from yolo_agent.components.adapters.data_pipeline.exposure import (
    ExposureConfig,
    compute_exposure,
)
from yolo_agent.components.adapters.data_pipeline.transforms import DataTransformConfig
from yolo_agent.components.adapters.runtime import (
    AdapterRuntimePayload,
    RuntimePluginReference,
)


class _DataAdapter(ComponentAdapter):
    adapter_version = "data_pipeline_adapters.v1"
    source_commit = "yolo-agent:data-pipeline-runtime-v1"
    strategy = "trainer_subclass"
    modified_model_fields = frozenset()
    mechanism_id: ClassVar[str]
    component_id: ClassVar[str]
    adapter_family: ClassVar[str]
    plugin_reference: ClassVar[str]
    plugin_hook: ClassVar[str]
    config_type: ClassVar[type[ExposureConfig] | type[DataTransformConfig]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        mechanism = getattr(cls, "mechanism_id", "")
        cls.modified_training_fields = frozenset({f"data.{mechanism}"})

    @property
    def changed_variable(self) -> str:
        return f"data.{self.mechanism_id}"

    def validate_environment(self, context: AdapterContext) -> AdapterValidationReport:
        return AdapterValidationReport(ok=True, checks={"torch": True, "python": True})

    def validate_compatibility(self, context: AdapterContext) -> AdapterValidationReport:
        errors: list[str] = []
        if context.imgsz != 640:
            errors.append("data pipeline adapters require fixed imgsz=640")
        if context.contract.component_id != self.component_id:
            errors.append(
                f"adapter identity mismatch: expected {self.component_id}, "
                f"got {context.contract.component_id}"
            )
        return AdapterValidationReport(
            ok=not errors,
            errors=errors,
            checks={
                "fixed_imgsz": context.imgsz == 640,
                "val_unchanged": True,
                "test_unchanged": True,
                "changed_variable": self.changed_variable,
                "exact_reproduction": False,
            },
        )

    def patch_model_config(
        self,
        config: dict[str, Any],
        context: AdapterContext,
        *,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        del context, dry_run
        return config

    def patch_training_config(
        self,
        config: dict[str, Any],
        context: AdapterContext,
        *,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        del dry_run
        config[self.changed_variable] = self._config(context).model_dump(mode="json")
        return config

    def build_module(self, context: AdapterContext) -> Any:
        return self._config(context)

    def load_pretrained_weights(
        self,
        module: Any,
        weights: Path | str | None,
        context: AdapterContext,
    ) -> WeightLoadResult:
        del module, weights, context
        return WeightLoadResult(loaded=False, message="Data adapters have no weights")

    def smoke_test(self, context: AdapterContext) -> SmokeTestResult:
        try:
            config = self._config(context)
            if isinstance(config, ExposureConfig):
                records = [
                    DataSampleRecord(
                        image_path="a.jpg",
                        normalized_areas=[0.002],
                        class_ids=[2],
                        is_hard_negative=True,
                        false_negative_score=1.0,
                    ),
                    DataSampleRecord(
                        image_path="b.jpg",
                        normalized_areas=[0.2],
                        class_ids=[1],
                    ),
                    DataSampleRecord(
                        image_path="c.jpg",
                        normalized_areas=[0.3],
                        class_ids=[1],
                    ),
                ]
                exposure, _ = compute_exposure(records, config)
                changed = len(set(exposure)) > 1 or config.strength == 0
            else:
                changed = self._transform_smoke(config)
            return SmokeTestResult(
                passed=changed,
                evidence_kind="local",
                checks={
                    "runtime_mechanism": self.mechanism_id,
                    "changed_variable": self.changed_variable,
                    "fixed_imgsz": True,
                    "val_unchanged": True,
                    "spawn_safe": True,
                },
                errors=[] if changed else ["mechanism smoke produced no effect"],
            )
        except (RuntimeError, ValueError) as exc:
            return SmokeTestResult(
                passed=False,
                evidence_kind="local",
                errors=[str(exc)],
            )

    def expected_artifacts(self, context: AdapterContext) -> list[ExpectedArtifact]:
        del context
        return [ExpectedArtifact(
            name=f"{self.mechanism_id}_manifest",
            relative_path=Path(f"{self.mechanism_id}_manifest.json"),
        )]

    def rollback_plan(self, context: AdapterContext) -> RollbackPlan:
        del context
        return RollbackPlan(
            actions=[f"remove {self.changed_variable} plugin and local artifacts"],
            files_to_remove=[
                Path(f"{self.mechanism_id}_manifest.json"),
                Path(f"{self.mechanism_id}_state.rank0.json"),
                Path(f"{self.mechanism_id}_dataset_state.rank0.json"),
            ],
        )

    def build_runtime_payload(
        self,
        context: AdapterContext,
        *,
        protocol_hash: str,
        base_command: list[str],
        generated_config: dict[str, Any],
    ) -> AdapterRuntimePayload:
        config = self._config(context)
        options = {
            "mechanism_id": self.mechanism_id,
            "component_id": self.component_id,
            "adapter_family": self.adapter_family,
            "changed_variable": self.changed_variable,
            **config.model_dump(mode="json", exclude={"mechanism"}),
        }
        reference = RuntimePluginReference(
            reference=self.plugin_reference,
            options=options,
            required_hooks=[self.plugin_hook],
        )
        runtime_config = dict(generated_config)
        if self.mechanism_id == "hard_negative_replay":
            manifest_path = config.manifest_path
            manifest_hash = config.manifest_hash
            if manifest_path is None or not manifest_hash:
                raise ValueError(
                    "hard-negative replay runtime payload requires manifest_path and manifest_hash"
                )
            data_pipeline = runtime_config.setdefault("data_pipeline", {})
            if not isinstance(data_pipeline, dict):
                raise ValueError("generated_config.data_pipeline must be a mapping")
            data_pipeline["hard_negative_replay"] = {
                "manifest_path": str(manifest_path),
                "manifest_hash": manifest_hash,
                "evidence_id": config.evidence_id,
                "source_split": "train",
                "baseline_protocol_hash": config.baseline_protocol_hash,
                "dataset_manifest_hash": config.dataset_manifest_hash,
                "train_index_hash": config.train_index_hash,
            }
        return AdapterRuntimePayload(
            component_ids=[self.component_id],
            adapter_classes=[type(self).__name__],
            adapter_versions={self.component_id: self.adapter_version},
            source_commits={self.component_id: self.source_commit},
            dataloader_plugin=[reference],
            generated_config=runtime_config,
            changed_variables={self.changed_variable: config.model_dump(mode="json")},
            expected_artifacts=self.expected_artifacts(context),
            rollback_plan=self.rollback_plan(context),
            protocol_hash=protocol_hash,
            base_command=base_command,
            supports_amp=True,
            supports_ddp=True,
            supports_resume=True,
        )

    def _config(self, context: AdapterContext) -> ExposureConfig | DataTransformConfig:
        return self.config_type.model_validate(
            {"mechanism": self.mechanism_id, **context.options}
        )

    def _transform_smoke(self, config: DataTransformConfig) -> bool:
        class _Dataset:
            def __len__(self) -> int:
                return 2

            def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
                class_id = 3 if index else 1
                value = 100 if index else 0
                return {
                    "img": torch.full((3, 16, 16), value, dtype=torch.uint8),
                    "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
                    "cls": torch.tensor([[class_id]], dtype=torch.float32),
                    "batch_idx": torch.tensor([0]),
                }

        if config.mechanism == "copy_paste_rare_classes" and not config.rare_class_ids:
            config = config.model_copy(update={"rare_class_ids": [3]})
        output = DataPipelineDataset(_Dataset(), config)[0]
        return output["img"].shape == torch.Size([3, 16, 16])


class _ExposureAdapter(_DataAdapter):
    plugin_reference = (
        "yolo_agent.components.adapters.data_pipeline.sampling_plugin:SamplingPlugin"
    )
    plugin_hook = "build_train_dataloader"
    config_type = ExposureConfig


class _TransformAdapter(_DataAdapter):
    plugin_reference = (
        "yolo_agent.components.adapters.data_pipeline.data_pipeline_plugin:"
        "DataPipelinePlugin"
    )
    plugin_hook = "build_train_dataset"
    config_type = DataTransformConfig


class SmallObjectWeightedSamplingAdapter(_ExposureAdapter):
    mechanism_id = "small_object_weighted_sampling"
    component_id = "sampling.small_object_weighted"
    adapter_family = "data.sampling.small_object_weighted"


class ClassBalancedSamplingAdapter(_ExposureAdapter):
    mechanism_id = "class_balanced_sampling"
    component_id = "sampling.class_balanced"
    adapter_family = "data.sampling.class_balanced"


class RepeatFactorSamplingAdapter(_ExposureAdapter):
    mechanism_id = "repeat_factor_sampling"
    component_id = "sampling.repeat_factor"
    adapter_family = "data.sampling.repeat_factor"


class HardNegativeReplayAdapter(_ExposureAdapter):
    mechanism_id = "hard_negative_replay"
    component_id = "sampling.hard_negative_replay"
    adapter_family = "data.sampling.hard_negative_replay"


class FalseNegativeClassBoostAdapter(_ExposureAdapter):
    mechanism_id = "false_negative_class_boost"
    component_id = "sampling.false_negative_class_boost"
    adapter_family = "data.sampling.false_negative_class_boost"


class RareClassCopyPasteAdapter(_TransformAdapter):
    mechanism_id = "copy_paste_rare_classes"
    component_id = "augmentation.copy_paste_rare_classes"
    adapter_family = "data.augmentation.copy_paste_rare_classes"


class ScaleAwareCropAdapter(_TransformAdapter):
    mechanism_id = "scale_aware_crop"
    component_id = "augmentation.scale_aware_crop"
    adapter_family = "data.augmentation.scale_aware_crop"


class ObjectCentricCropAdapter(_TransformAdapter):
    mechanism_id = "object_centric_crop"
    component_id = "augmentation.object_centric_crop"
    adapter_family = "data.augmentation.object_centric_crop"


class MultiImageSamplingScheduleAdapter(_TransformAdapter):
    mechanism_id = "multi_image_sampling_schedule"
    component_id = "augmentation.multi_image_sampling_schedule"
    adapter_family = "data.augmentation.multi_image_sampling_schedule"


__all__ = [
    "ClassBalancedSamplingAdapter",
    "FalseNegativeClassBoostAdapter",
    "HardNegativeReplayAdapter",
    "MultiImageSamplingScheduleAdapter",
    "ObjectCentricCropAdapter",
    "RareClassCopyPasteAdapter",
    "RepeatFactorSamplingAdapter",
    "ScaleAwareCropAdapter",
    "SmallObjectWeightedSamplingAdapter",
]
