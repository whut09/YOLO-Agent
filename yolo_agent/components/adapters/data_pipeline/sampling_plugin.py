"""Ultralytics train-dataloader plugin for one explicit exposure mechanism."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from yolo_agent.components.adapters.data_pipeline.contracts import (
    DataPipelineIdentity,
    DataPipelineManifest,
)
from yolo_agent.components.adapters.data_pipeline.exposure import (
    ExposureConfig,
    compute_exposure_details,
)
from yolo_agent.components.adapters.data_pipeline.runtime import (
    dataset_manifest_hash,
    read_json,
    rebuild_dataloader,
    records_from_yolo_dataset,
    world_size,
    write_json_atomic,
)
from yolo_agent.components.adapters.data_pipeline.sampling import (
    DistributedExposureSampler,
)


class SamplingPlugin:
    """Reusable implementation whose runtime identity remains mechanism-specific."""

    plugin_version = "data_sampling_plugin.v1"

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
                "replay" if mechanism_id == "hard_negative_replay" else "weighted_sampler"
            ),
            changed_variable=changed_variable,
        )
        self.config = ExposureConfig.model_validate(
            {"mechanism": mechanism_id, **options}
        )
        self.sampler: DistributedExposureSampler | None = None

    def build_train_dataloader(
        self,
        *,
        context: Any,
        trainer: Any,
        dataloader: Any,
        dataset_path: str,
        batch_size: int,
        rank: int,
    ) -> Any:
        del dataset_path, batch_size
        records = records_from_yolo_dataset(dataloader.dataset)
        if self.identity.mechanism_id == "hard_negative_replay" and not any(
            item.is_hard_negative for item in records
        ):
            raise ValueError("hard-negative replay requires local hard-negative evidence")
        if self.identity.mechanism_id == "false_negative_class_boost" and not (
            self.config.target_class_ids
            and any(item.false_negative_score > 0 for item in records)
        ):
            raise ValueError("false-negative class boost requires class IDs and FN scores")
        raw_exposure, exposure, clipping = compute_exposure_details(
            records, self.config
        )
        resolved_rank = rank if rank >= 0 else 0
        resolved_world_size = world_size(rank)
        manifest_id = dataset_manifest_hash(dataloader.dataset, records)
        adapter_hash = self._adapter_hash()
        sampler = DistributedExposureSampler(
            exposure,
            sample_count=self.config.sample_count or len(records),
            seed=self.config.seed,
            rank=resolved_rank,
            world_size=resolved_world_size,
            dataset_manifest=manifest_id,
            adapter_hash=adapter_hash,
            mechanism_id=self.identity.mechanism_id,
        )
        self.sampler = sampler
        setattr(trainer, f"{self.identity.mechanism_id}_sampler", sampler)
        manifest = DataPipelineManifest(
            identity=self.identity,
            dataset_manifest=manifest_id,
            protocol_hash=context.payload.protocol_hash,
            runtime_payload_hash=context.payload.payload_hash,
            adapter_hash=adapter_hash,
            plugin_version=self.plugin_version,
            seed=self.config.seed,
            rank=resolved_rank,
            world_size=resolved_world_size,
            image_paths=[item.image_path for item in records],
            class_counts=_class_counts(records),
            raw_exposure=raw_exposure,
            final_exposure=exposure,
            clipping_statistics=clipping,
            sample_count=self.config.sample_count or len(records),
        ).with_hash()
        if resolved_rank == 0:
            manifest.write(self._manifest_path(context.payload_path.parent))
        if self.config.strength == 0:
            return dataloader
        return rebuild_dataloader(dataloader, sampler)

    def on_checkpoint_save(
        self,
        *,
        context: Any,
        trainer: Any,
        checkpoints: dict[str, Any],
    ) -> None:
        if self.sampler is None:
            return
        self.sampler.set_epoch(int(getattr(trainer, "epoch", self.sampler.epoch)))
        state = self.sampler.state_dict()
        write_json_atomic(self._state_path(context.payload_path.parent), state)
        if self.sampler.rank == 0:
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
        if self.sampler is None:
            raise ValueError("data sampler was not constructed before resume")
        key = f"{self.identity.mechanism_id}_sampler_state"
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
        self.sampler.load_state_dict(state)

    def _adapter_hash(self) -> str:
        payload = {
            "plugin_version": self.plugin_version,
            "identity": self.identity.model_dump(mode="json"),
            "config": self.config.model_dump(mode="json"),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _manifest_path(self, root: Path) -> Path:
        return root / f"{self.identity.mechanism_id}_manifest.json"

    def _state_path(self, root: Path) -> Path:
        rank = self.sampler.rank if self.sampler is not None else 0
        return root / f"{self.identity.mechanism_id}_state.rank{rank}.json"

    def _checkpoint_path(self, checkpoint: Path) -> Path:
        return checkpoint.with_name(
            f"{checkpoint.name}.{self.identity.mechanism_id}.json"
        )


def _class_counts(records: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for class_id in record.class_ids:
            key = str(class_id)
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


__all__ = ["SamplingPlugin"]
