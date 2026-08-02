"""Ultralytics train-dataset plugin for one explicit transform mechanism."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from yolo_agent.components.adapters.data_pipeline.contracts import (
    DataPipelineIdentity,
    DataPipelineManifest,
)
from yolo_agent.components.adapters.data_pipeline.dataset import DataPipelineDataset
from yolo_agent.components.adapters.data_pipeline.runtime import read_json, write_json_atomic
from yolo_agent.components.adapters.data_pipeline.transforms import DataTransformConfig


class DataPipelinePlugin:
    """Reusable dataset hook with mechanism-specific runtime and artifact identity."""

    plugin_version = "data_pipeline_plugin.v1"

    def __init__(
        self,
        *,
        mechanism_id: str,
        component_id: str,
        adapter_family: str,
        changed_variable: str,
        **options: Any,
    ) -> None:
        self.identity = DataPipelineIdentity(
            mechanism_id=mechanism_id,
            component_id=component_id,
            adapter_family=adapter_family,
            mechanism_kind=(
                "schedule"
                if mechanism_id == "multi_image_sampling_schedule"
                else "transform"
            ),
            changed_variable=changed_variable,
        )
        self.config = DataTransformConfig.model_validate(
            {"mechanism": mechanism_id, **options}
        )
        self.dataset: DataPipelineDataset | None = None

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
        wrapped = DataPipelineDataset(dataset, self.config)
        self.dataset = wrapped
        setattr(trainer, f"{self.identity.mechanism_id}_dataset", wrapped)
        manifest = DataPipelineManifest(
            identity=self.identity,
            dataset_manifest=str(
                getattr(dataset, "manifest_hash", None)
                or getattr(dataset, "dataset_manifest", None)
                or self._dataset_hash(dataset)
            ),
            protocol_hash=context.payload.protocol_hash,
            runtime_payload_hash=context.payload.payload_hash,
            adapter_hash=self._adapter_hash(),
            plugin_version=self.plugin_version,
            seed=self.config.seed,
            image_paths=[str(value) for value in getattr(dataset, "im_files", [])],
            transform_parameters=self.config.model_dump(mode="json"),
            sample_count=len(dataset),
        ).with_hash()
        manifest.write(self._manifest_path(context.payload_path.parent))
        if self.config.probability == 0:
            return dataset
        return wrapped

    def on_train_batch_start(
        self,
        *,
        context: Any,
        trainer: Any,
        batch: Any,
    ) -> None:
        del context, batch
        if self.dataset is not None:
            self.dataset.set_epoch(int(getattr(trainer, "epoch", self.dataset.epoch)))

    def on_checkpoint_save(
        self,
        *,
        context: Any,
        trainer: Any,
        checkpoints: dict[str, Any],
    ) -> None:
        if self.dataset is None:
            return
        self.dataset.set_epoch(int(getattr(trainer, "epoch", self.dataset.epoch)))
        state = self.dataset.state_dict()
        write_json_atomic(self._state_path(context.payload_path.parent), state)
        for checkpoint in checkpoints.values():
            if checkpoint:
                write_json_atomic(self._checkpoint_path(Path(checkpoint)), state)

    def on_checkpoint_load(
        self,
        *,
        context: Any,
        trainer: Any,
        checkpoint: Any,
    ) -> None:
        if self.dataset is None:
            raise ValueError("data pipeline dataset was not constructed before resume")
        key = f"{self.identity.mechanism_id}_dataset_state"
        state = checkpoint.get(key) if isinstance(checkpoint, dict) else None
        if not isinstance(state, dict):
            resume = getattr(getattr(trainer, "args", None), "resume", None)
            paths: list[Path] = []
            if isinstance(resume, (str, Path)) and str(resume).lower() not in {
                "true",
                "false",
            }:
                paths.append(self._checkpoint_path(Path(resume)))
            paths.append(self._state_path(context.payload_path.parent))
            state = next((read_json(path) for path in paths if path.is_file()), None)
        if not isinstance(state, dict):
            raise ValueError(f"{self.identity.mechanism_id} resume state is missing")
        self.dataset.load_state_dict(state)

    def _adapter_hash(self) -> str:
        payload = {
            "plugin_version": self.plugin_version,
            "identity": self.identity.model_dump(mode="json"),
            "config": self.config.model_dump(mode="json"),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _dataset_hash(self, dataset: Any) -> str:
        payload = {
            "length": len(dataset),
            "images": [str(value) for value in getattr(dataset, "im_files", [])],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _manifest_path(self, root: Path) -> Path:
        return root / f"{self.identity.mechanism_id}_manifest.json"

    def _state_path(self, root: Path) -> Path:
        return root / f"{self.identity.mechanism_id}_dataset_state.json"

    def _checkpoint_path(self, checkpoint: Path) -> Path:
        return checkpoint.with_name(
            f"{checkpoint.name}.{self.identity.mechanism_id}.dataset.json"
        )


__all__ = ["DataPipelinePlugin"]
