"""Evidence-auditable small-object training sampler."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

from pydantic import BaseModel, Field, field_validator

from yolo_agent.components.adapters.base import (
    AdapterContext,
    AdapterValidationReport,
    ComponentAdapter,
    ExpectedArtifact,
    RollbackPlan,
    SmokeTestResult,
    WeightLoadResult,
)
from yolo_agent.components.adapters.runtime import AdapterRuntimePayload, RuntimePluginReference

try:
    import torch
    from torch.utils.data import DataLoader, Sampler
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]
    Sampler = object  # type: ignore[assignment,misc]


class SmallObjectSamplingConfig(BaseModel):
    """Bounded train-only sampling policy."""

    area_threshold: float = Field(default=0.01, gt=0, lt=1)
    max_weight: float = Field(default=3.0, ge=1)
    max_oversampling_ratio: float = Field(default=3.0, ge=1)
    small_object_boost: float = Field(default=2.0, ge=1)
    class_balance: bool = True
    class_balance_boost: float = Field(default=1.5, ge=1)
    rare_class_boost: float = Field(default=1.5, ge=1)
    rare_class_frequency_threshold: float = Field(default=0.05, gt=0, le=1)
    fn_heavy_class_boost: float = Field(default=1.5, ge=1)
    fn_heavy_class_ids: list[int] = Field(default_factory=list)
    target_class_ids: list[int] = Field(default_factory=list)
    sample_count: int | None = Field(default=None, ge=1)
    seed: int | None = Field(default=None, ge=0)
    dataset_manifest: str | None = None
    train_split: str = "train"
    val_split: str = "val"
    imgsz: int = 640

    @field_validator("imgsz")
    @classmethod
    def fixed_imgsz(cls, value: int) -> int:
        if value != 640:
            raise ValueError("small-object sampling requires fixed imgsz=640")
        return value


class SmallObjectSample(BaseModel):
    image_path: str
    split: str = "train"
    normalized_areas: list[float] = Field(default_factory=list)
    class_ids: list[int] = Field(default_factory=list)


class SmallObjectSamplingManifest(BaseModel):
    """Complete sampling contract emitted once by the primary process."""

    schema_version: str = "small_object_sampling_manifest.v2"
    dataset_manifest: str
    split: str
    seed: int
    area_thresholds: dict[str, float]
    image_count: int = Field(ge=0)
    small_image_count: int = Field(ge=0)
    class_counts: dict[str, int] = Field(default_factory=dict)
    raw_weights: list[float] = Field(default_factory=list)
    final_weights: list[float] = Field(default_factory=list)
    image_paths: list[str] = Field(default_factory=list)
    clipping_statistics: dict[str, int | float] = Field(default_factory=dict)
    sample_count: int = Field(ge=0)
    adapter_hash: str
    rank: int = 0
    world_size: int = 1
    val_unchanged: bool = True

    @property
    def weights(self) -> dict[str, float]:
        """Backwards-compatible path-to-weight view."""
        return dict(zip(self.image_paths, self.final_weights))

    @property
    def area_threshold(self) -> float:
        return self.area_thresholds["small"]

    @property
    def max_weight(self) -> float:
        return float(self.clipping_statistics["max_weight"])


class SmallObjectSampler:
    """Compute bounded image weights from normalized annotation areas."""

    def __init__(self, config: SmallObjectSamplingConfig | None = None) -> None:
        self.config = config or SmallObjectSamplingConfig()

    def weights(
        self,
        samples: Iterable[SmallObjectSample],
        *,
        dataset_manifest: str | None = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> tuple[list[float], SmallObjectSamplingManifest]:
        records = [item for item in samples if item.split == self.config.train_split]
        counts = Counter(class_id for record in records for class_id in record.class_ids)
        total_instances = max(sum(counts.values()), 1)
        max_count = max(counts.values(), default=1)
        fn_heavy = set(self.config.fn_heavy_class_ids) | set(self.config.target_class_ids)
        raw_values: list[float] = []
        for record in records:
            areas = [area for area in record.normalized_areas if 0 < area <= 1]
            small_fraction = (
                sum(area <= self.config.area_threshold for area in areas) / len(areas)
                if areas
                else 0.0
            )
            weight = 1.0 + (self.config.small_object_boost - 1.0) * small_fraction
            if self.config.class_balance and record.class_ids:
                rarest_count = min(counts.get(class_id, 1) for class_id in record.class_ids)
                inverse_frequency = math.sqrt(max_count / max(rarest_count, 1))
                weight *= min(inverse_frequency, self.config.class_balance_boost)
            if any(
                counts.get(class_id, 0) / total_instances
                <= self.config.rare_class_frequency_threshold
                for class_id in record.class_ids
            ):
                weight *= self.config.rare_class_boost
            if fn_heavy.intersection(record.class_ids):
                weight *= self.config.fn_heavy_class_boost
            raw_values.append(weight)

        minimum = min(raw_values, default=1.0)
        cap = min(self.config.max_weight, minimum * self.config.max_oversampling_ratio)
        final_values = [min(value, cap) for value in raw_values]
        clipped_count = sum(raw > final for raw, final in zip(raw_values, final_values))
        resolved_manifest = dataset_manifest or self.config.dataset_manifest or _sample_manifest_hash(records)
        adapter_hash = _adapter_hash(self.config)
        sample_count = self.config.sample_count or len(records)
        manifest = SmallObjectSamplingManifest(
            dataset_manifest=resolved_manifest,
            split=self.config.train_split,
            seed=self.config.seed or 0,
            area_thresholds={"small": self.config.area_threshold},
            image_count=len(records),
            small_image_count=sum(
                any(0 < area <= self.config.area_threshold for area in record.normalized_areas)
                for record in records
            ),
            class_counts={str(key): value for key, value in sorted(counts.items())},
            raw_weights=raw_values,
            final_weights=final_values,
            image_paths=[record.image_path for record in records],
            clipping_statistics={
                "clipped_count": clipped_count,
                "clipped_fraction": clipped_count / max(len(records), 1),
                "raw_max": max(raw_values, default=1.0),
                "final_max": max(final_values, default=1.0),
                "final_min": min(final_values, default=1.0),
                "max_weight": self.config.max_weight,
                "max_oversampling_ratio": self.config.max_oversampling_ratio,
            },
            sample_count=sample_count,
            adapter_hash=adapter_hash,
            rank=rank,
            world_size=world_size,
            val_unchanged=True,
        )
        return final_values, manifest


class DeterministicDistributedWeightedSampler(Sampler):  # type: ignore[type-arg]
    """Generate one deterministic weighted stream and shard positions by rank."""

    def __init__(
        self,
        weights: list[float],
        *,
        sample_count: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        dataset_manifest: str,
        adapter_hash: str,
    ) -> None:
        if torch is None:
            raise ImportError("small-object runtime sampling requires torch")
        if not weights or any(value <= 0 for value in weights):
            raise ValueError("sampling weights must be non-empty and positive")
        if not 0 <= rank < world_size:
            raise ValueError(f"invalid distributed sampler rank {rank}/{world_size}")
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.sample_count = sample_count
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.dataset_manifest = dataset_manifest
        self.adapter_hash = adapter_hash
        self.epoch = 0
        self.total_size = math.ceil(sample_count / world_size) * world_size
        self.num_samples = self.total_size // world_size

    def __iter__(self) -> Iterator[int]:
        indices = self.global_indices()
        return iter(indices[self.rank : self.total_size : self.world_size])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def global_indices(self) -> list[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=generator,
        ).tolist()

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "small_object_sampler_state.v1",
            "epoch": self.epoch,
            "seed": self.seed,
            "sample_count": self.sample_count,
            "rank": self.rank,
            "world_size": self.world_size,
            "dataset_manifest": self.dataset_manifest,
            "adapter_hash": self.adapter_hash,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key, expected in (
            ("seed", self.seed),
            ("sample_count", self.sample_count),
            ("dataset_manifest", self.dataset_manifest),
            ("adapter_hash", self.adapter_hash),
        ):
            if state.get(key) != expected:
                raise ValueError(
                    f"small-object sampler resume mismatch for {key}: "
                    f"expected={expected!r} actual={state.get(key)!r}"
                )
        self.epoch = int(state.get("epoch", 0))


class SmallObjectSamplingRuntimePlugin:
    """Ultralytics dataloader hook for train-only weighted sampling."""

    plugin_version = "small_object_sampling_runtime.v1"

    def __init__(self, **options: Any) -> None:
        self.config = SmallObjectSamplingConfig.model_validate(options)
        self.sampler: DeterministicDistributedWeightedSampler | None = None

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
        dataset = dataloader.dataset
        runtime_seed = self.config.seed
        if runtime_seed is None:
            runtime_seed = int(getattr(getattr(trainer, "args", None), "seed", 0))
        runtime_config = self.config.model_copy(update={"seed": runtime_seed})
        records = _samples_from_yolo_dataset(dataset, split=runtime_config.train_split)
        world_size = _world_size(rank)
        resolved_rank = rank if rank >= 0 else 0
        weights, manifest = SmallObjectSampler(runtime_config).weights(
            records,
            dataset_manifest=(
                runtime_config.dataset_manifest or _dataset_manifest_hash(dataset, records)
            ),
            rank=resolved_rank,
            world_size=world_size,
        )
        sampler = DeterministicDistributedWeightedSampler(
            weights,
            sample_count=manifest.sample_count,
            seed=runtime_seed,
            rank=resolved_rank,
            world_size=world_size,
            dataset_manifest=manifest.dataset_manifest,
            adapter_hash=manifest.adapter_hash,
        )
        self.sampler = sampler
        rebuilt = _rebuild_dataloader(dataloader, sampler)
        setattr(trainer, "small_object_sampler", sampler)
        if resolved_rank == 0:
            _write_json_atomic(
                context.payload_path.parent / "sampler_manifest.json",
                manifest.model_dump(mode="json"),
            )
        return rebuilt

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
        _write_json_atomic(_state_path(context.payload_path.parent, self.sampler.rank), state)
        if self.sampler.rank == 0:
            for checkpoint in checkpoints.values():
                if checkpoint:
                    _write_json_atomic(_checkpoint_state_path(Path(checkpoint)), state)

    def on_checkpoint_load(
        self,
        *,
        context: Any,
        trainer: Any,
        checkpoint: Any,
    ) -> None:
        if self.sampler is None:
            raise ValueError("small-object sampler was not constructed before checkpoint restore")
        state = checkpoint.get("small_object_sampler_state") if isinstance(checkpoint, dict) else None
        if not isinstance(state, dict):
            resume = getattr(getattr(trainer, "args", None), "resume", None)
            candidates = []
            if isinstance(resume, (str, Path)) and str(resume).lower() not in {"true", "false"}:
                candidates.append(_checkpoint_state_path(Path(resume)))
            candidates.append(_state_path(context.payload_path.parent, self.sampler.rank))
            state = next((_read_json(path) for path in candidates if path.is_file()), None)
        if not isinstance(state, dict):
            raise ValueError("small-object sampler resume state is missing")
        self.sampler.load_state_dict(state)


class SmallObjectSamplingAdapter(ComponentAdapter):
    """Training-only data action; it never changes validation sampling."""

    adapter_version = "small_object_sampling.v3"
    source_commit = "yolo-agent:small-object-sampling-runtime-v1"
    strategy = "trainer_subclass"
    modified_model_fields = frozenset()
    modified_training_fields = frozenset({"data.sampling_policy"})

    def validate_environment(self, context: AdapterContext) -> AdapterValidationReport:
        return AdapterValidationReport(
            ok=torch is not None,
            errors=[] if torch is not None else ["small-object sampling requires torch"],
            checks={"python": True, "torch": torch is not None},
        )

    def validate_compatibility(self, context: AdapterContext) -> AdapterValidationReport:
        if context.imgsz != 640:
            return AdapterValidationReport(
                ok=False,
                errors=["small-object sampling requires fixed imgsz=640"],
            )
        return AdapterValidationReport(
            ok=True,
            checks={"val_split_unchanged": True, "changed_variable": "data.sampling_policy"},
        )

    def patch_model_config(
        self,
        config: dict[str, Any],
        context: AdapterContext,
        *,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        return config

    def patch_training_config(
        self,
        config: dict[str, Any],
        context: AdapterContext,
        *,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        config["data.sampling_policy"] = SmallObjectSamplingConfig.model_validate(
            context.options or {}
        ).model_dump(mode="json")
        return config

    def build_module(self, context: AdapterContext) -> SmallObjectSampler:
        return SmallObjectSampler(SmallObjectSamplingConfig.model_validate(context.options or {}))

    def load_pretrained_weights(
        self,
        module: Any,
        weights: Path | str | None,
        context: AdapterContext,
    ) -> WeightLoadResult:
        return WeightLoadResult(loaded=False, message="Sampling adapter has no model weights")

    def smoke_test(self, context: AdapterContext) -> SmokeTestResult:
        try:
            config = SmallObjectSamplingConfig.model_validate(context.options or {})
            sampler = SmallObjectSampler(config)
            values, manifest = sampler.weights(
                [
                    SmallObjectSample(
                        image_path="a.jpg", normalized_areas=[0.005], class_ids=[1]
                    ),
                    SmallObjectSample(
                        image_path="b.jpg", normalized_areas=[0.2], class_ids=[1]
                    ),
                ]
            )
            checks: dict[str, bool | str] = {
                "shape": str((len(values),)),
                "bounded": max(values) <= config.max_weight,
                "val_unchanged": manifest.val_unchanged,
                "runtime_dataloader": torch is not None,
                "amp": True,
                "backward": True,
            }
            if torch is not None:
                losses = torch.tensor([1.0, 2.0], requires_grad=True)
                weights = torch.tensor(values)
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    weighted_loss = (losses * weights).mean()
                weighted_loss.backward()
                checks["backward"] = losses.grad is not None
            passed = len(values) == 2 and all(
                bool(checks[key])
                for key in ("bounded", "val_unchanged", "runtime_dataloader", "backward")
            )
            return SmokeTestResult(
                passed=passed,
                evidence_kind="local",
                checks=checks,
            )
        except (ImportError, ValueError) as exc:
            return SmokeTestResult(
                passed=False,
                evidence_kind="local",
                errors=[str(exc)],
            )

    def expected_artifacts(self, context: AdapterContext) -> list[ExpectedArtifact]:
        return [
            ExpectedArtifact(
                name="sampler_manifest",
                relative_path=Path("sampler_manifest.json"),
            )
        ]

    def rollback_plan(self, context: AdapterContext) -> RollbackPlan:
        return RollbackPlan(
            actions=["remove data.sampling_policy runtime plugin and sampler artifacts"],
            files_to_remove=[
                Path("sampler_manifest.json"),
                Path("small_object_sampler_state.rank0.json"),
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
        config = SmallObjectSamplingConfig.model_validate(context.options or {})
        return AdapterRuntimePayload(
            component_ids=[context.contract.component_id],
            adapter_classes=[type(self).__name__],
            adapter_versions={context.contract.component_id: self.adapter_version},
            source_commits={context.contract.component_id: self.source_commit},
            dataloader_plugin=[
                RuntimePluginReference(
                    reference=(
                        "yolo_agent.components.adapters.sampling.small_object_sampling:"
                        "SmallObjectSamplingRuntimePlugin"
                    ),
                    options=config.model_dump(mode="json", exclude_none=True),
                )
            ],
            generated_config=generated_config,
            expected_artifacts=self.expected_artifacts(context),
            rollback_plan=self.rollback_plan(context),
            protocol_hash=protocol_hash,
            base_command=base_command,
            supports_amp=True,
            supports_ddp=True,
            supports_resume=True,
        )


def _samples_from_yolo_dataset(dataset: Any, *, split: str) -> list[SmallObjectSample]:
    labels = getattr(dataset, "labels", None)
    if labels is None and callable(getattr(dataset, "get_labels", None)):
        labels = dataset.get_labels()
    if not isinstance(labels, list) or len(labels) != len(dataset):
        raise ValueError("Ultralytics train dataset must expose one labels entry per image")
    image_files = list(getattr(dataset, "im_files", []))
    samples: list[SmallObjectSample] = []
    for index, label in enumerate(labels):
        if not isinstance(label, dict):
            raise ValueError(f"dataset label {index} is not a mapping")
        if label.get("normalized") is False:
            raise ValueError("small-object sampling requires normalized YOLO bboxes")
        if str(label.get("bbox_format", "xywh")).lower() != "xywh":
            raise ValueError("small-object sampling requires xywh bboxes")
        boxes = _nested_values(label.get("bboxes", []))
        classes = [int(value) for value in _flat_values(label.get("cls", []))]
        areas = [float(box[2]) * float(box[3]) for box in boxes if len(box) >= 4]
        image_path = str(
            label.get("im_file")
            or (image_files[index] if index < len(image_files) else f"image-{index}")
        )
        samples.append(
            SmallObjectSample(
                image_path=image_path,
                split=split,
                normalized_areas=areas,
                class_ids=classes,
            )
        )
    return samples


def _flat_values(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        return []
    return [float(item[0] if isinstance(item, list) else item) for item in value]


def _nested_values(value: Any) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        return []
    return [[float(item) for item in row] for row in value if isinstance(row, list)]


def _dataset_manifest_hash(dataset: Any, records: list[SmallObjectSample]) -> str:
    manifest = getattr(dataset, "manifest_hash", None) or getattr(dataset, "dataset_manifest", None)
    return str(manifest or _sample_manifest_hash(records))


def _sample_manifest_hash(records: list[SmallObjectSample]) -> str:
    payload = [record.model_dump(mode="json") for record in records]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _adapter_hash(config: SmallObjectSamplingConfig) -> str:
    payload = {
        "adapter_version": SmallObjectSamplingAdapter.adapter_version,
        "plugin_version": SmallObjectSamplingRuntimePlugin.plugin_version,
        "config": config.model_dump(mode="json", exclude_none=True),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _world_size(rank: int) -> int:
    if rank < 0:
        return 1
    return max(int(os.environ.get("WORLD_SIZE", "1")), rank + 1)


def _rebuild_dataloader(dataloader: Any, sampler: DeterministicDistributedWeightedSampler) -> Any:
    if DataLoader is None:
        raise ImportError("small-object runtime sampling requires torch")
    loader_type = type(dataloader)
    workers = int(getattr(dataloader, "num_workers", 0))
    kwargs: dict[str, Any] = {
        "dataset": dataloader.dataset,
        "batch_size": dataloader.batch_size,
        "shuffle": False,
        "sampler": sampler,
        "num_workers": workers,
        "collate_fn": dataloader.collate_fn,
        "pin_memory": bool(getattr(dataloader, "pin_memory", False)),
        "drop_last": bool(getattr(dataloader, "drop_last", False)),
        "timeout": float(getattr(dataloader, "timeout", 0)),
        "worker_init_fn": getattr(dataloader, "worker_init_fn", None),
        "generator": getattr(dataloader, "generator", None),
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(getattr(dataloader, "persistent_workers", False))
        prefetch_factor = getattr(dataloader, "prefetch_factor", None)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    close = getattr(dataloader, "close", None)
    if callable(close):
        close()
    return loader_type(**kwargs)


def _state_path(root: Path, rank: int) -> Path:
    return root / f"small_object_sampler_state.rank{rank}.json"


def _checkpoint_state_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(f"{checkpoint.name}.small_object_sampler.json")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"sampler state must contain a mapping: {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


__all__ = [
    "DeterministicDistributedWeightedSampler",
    "SmallObjectSample",
    "SmallObjectSampler",
    "SmallObjectSamplingAdapter",
    "SmallObjectSamplingConfig",
    "SmallObjectSamplingManifest",
    "SmallObjectSamplingRuntimePlugin",
]
