"""Runtime hooks for annotation, preprocessing, and active-learning routes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, model_validator

from yolo_agent.agents.active_learning import ActiveLearningMiner, MiningConfig
from yolo_agent.components.adapters.data_pipeline.annotation import (
    AnnotationFilterConfig,
    AnnotationFilterDataset,
)
from yolo_agent.components.adapters.data_pipeline.contracts import (
    DataPipelineIdentity,
    DataPipelineManifest,
)
from yolo_agent.components.adapters.data_pipeline.preprocessing import (
    PreprocessingConfig,
    PreprocessingDataset,
)


class ActiveLearningConfig(MiningConfig):
    """Active-learning route configuration with the campaign image-size guard."""

    model_config = ConfigDict(extra="forbid")

    imgsz: int = 640

    @model_validator(mode="after")
    def validate_imgsz(self) -> "ActiveLearningConfig":
        if self.imgsz != 640:
            raise ValueError("active-learning adapters require fixed imgsz=640")
        return self


class AnnotationFilterPlugin:
    """Attach an annotation filter to the train dataset construction hook."""

    plugin_version = "annotation_filter_plugin.v1"

    def __init__(
        self,
        *,
        mechanism_id: str,
        component_id: str,
        adapter_family: str,
        changed_variable: str,
        **options: Any,
    ) -> None:
        del adapter_family, changed_variable
        self.mechanism_id = mechanism_id
        self.component_id = component_id
        self.config = AnnotationFilterConfig.model_validate(options)

    def build_train_dataset(
        self,
        *,
        context: Any,
        trainer: Any,
        dataset: Any,
        image_path: str,
        batch_size: int | None,
    ) -> Any:
        del image_path, batch_size
        wrapped = AnnotationFilterDataset(dataset, self.config)
        setattr(trainer, f"{self.mechanism_id}_dataset", wrapped)
        _write_manifest(
            context=context,
            mechanism_id=self.mechanism_id,
            component_id=self.component_id,
            adapter_family="data.annotation.quality_filter",
            plugin_version=self.plugin_version,
            config=self.config.model_dump(mode="json"),
            dataset=dataset,
        )
        return wrapped


class PreprocessingPlugin:
    """Attach deterministic image preprocessing to train dataset construction."""

    plugin_version = "preprocessing_plugin.v1"

    def __init__(
        self,
        *,
        mechanism_id: str,
        component_id: str,
        adapter_family: str,
        changed_variable: str,
        **options: Any,
    ) -> None:
        del adapter_family, changed_variable
        self.mechanism_id = mechanism_id
        self.component_id = component_id
        self.config = PreprocessingConfig.model_validate(options)

    def build_train_dataset(
        self,
        *,
        context: Any,
        trainer: Any,
        dataset: Any,
        image_path: str,
        batch_size: int | None,
    ) -> Any:
        del image_path, batch_size
        wrapped = PreprocessingDataset(dataset, self.config)
        setattr(trainer, f"{self.mechanism_id}_dataset", wrapped)
        _write_manifest(
            context=context,
            mechanism_id=self.mechanism_id,
            component_id=self.component_id,
            adapter_family="data.preprocessing.normalization",
            plugin_version=self.plugin_version,
            config=self.config.model_dump(mode="json"),
            dataset=dataset,
        )
        return wrapped


class ActiveLearningPlugin:
    """Register the existing active-learning miner at validator construction."""

    plugin_version = "active_learning_plugin.v1"

    def __init__(
        self,
        *,
        mechanism_id: str,
        component_id: str,
        adapter_family: str,
        changed_variable: str,
        **options: Any,
    ) -> None:
        del adapter_family, changed_variable
        self.mechanism_id = mechanism_id
        self.component_id = component_id
        self.config = ActiveLearningConfig.model_validate(options)
        self.miner = ActiveLearningMiner(
            MiningConfig.model_validate(self.config.model_dump(exclude={"imgsz"}))
        )

    def build_validator(
        self,
        *,
        context: Any,
        trainer: Any,
        validator: Any,
    ) -> Any:
        _write_manifest(
            context=context,
            mechanism_id=self.mechanism_id,
            component_id=self.component_id,
            adapter_family="data.active_learning.acquisition",
            plugin_version=self.plugin_version,
            config=self.config.model_dump(mode="json"),
            dataset=None,
        )
        setattr(validator, f"{self.mechanism_id}_miner", self.miner)
        setattr(trainer, f"{self.mechanism_id}_miner", self.miner)
        return validator

    def mine(self, predictions: list[Any], *, dataset_version: str) -> Any:
        """Expose the acquisition operation used by the active-learning stage."""

        return self.miner.mine(predictions, dataset_version=dataset_version)


def _write_manifest(
    *,
    context: Any,
    mechanism_id: str,
    component_id: str,
    adapter_family: str,
    plugin_version: str,
    config: dict[str, Any],
    dataset: Any,
) -> Path:
    """Persist a runtime binding artifact without claiming training evidence."""

    payload = {
        "plugin_version": plugin_version,
        "mechanism_id": mechanism_id,
        "component_id": component_id,
        "config": config,
    }
    adapter_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    image_paths = [str(value) for value in getattr(dataset, "im_files", [])] if dataset is not None else []
    dataset_manifest = str(
        getattr(dataset, "manifest_hash", None)
        or getattr(dataset, "dataset_manifest", None)
        or adapter_hash
    ) if dataset is not None else "unbound"
    identity = DataPipelineIdentity(
        mechanism_id=mechanism_id,
        component_id=component_id,
        adapter_family=adapter_family,
        mechanism_kind="schedule" if mechanism_id == "active_sample_selection" else "transform",
        changed_variable=f"data.{mechanism_id}",
    )
    manifest = DataPipelineManifest(
        identity=identity,
        dataset_manifest=dataset_manifest,
        protocol_hash=context.payload.protocol_hash,
        runtime_payload_hash=context.payload.payload_hash,
        adapter_hash=adapter_hash,
        plugin_version=plugin_version,
        image_paths=image_paths,
        transform_parameters=config,
        sample_count=len(dataset) if dataset is not None else 0,
    )
    output = Path(context.payload_path).parent / f"{mechanism_id}_manifest.json"
    return manifest.write(output)


__all__ = [
    "ActiveLearningConfig",
    "ActiveLearningPlugin",
    "AnnotationFilterPlugin",
    "PreprocessingPlugin",
]
