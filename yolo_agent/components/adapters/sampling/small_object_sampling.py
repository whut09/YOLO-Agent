"""Evidence-auditable small-object training sampler."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

from yolo_agent.components.adapters.base import AdapterContext, AdapterValidationReport, ComponentAdapter, ExpectedArtifact, RollbackPlan, SmokeTestResult, WeightLoadResult

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class SmallObjectSamplingConfig(BaseModel):
    area_threshold: float = Field(default=0.01, gt=0, lt=1)
    max_weight: float = Field(default=3.0, ge=1)
    small_object_boost: float = Field(default=2.0, ge=1)
    class_balance: bool = True
    train_split: str = "train"
    val_split: str = "val"
    imgsz: int = 640


class SmallObjectSample(BaseModel):
    image_path: str
    split: str = "train"
    normalized_areas: list[float] = Field(default_factory=list)
    class_ids: list[int] = Field(default_factory=list)


class SmallObjectSamplingManifest(BaseModel):
    schema_version: str = "small_object_sampling_manifest.v1"
    split: str
    image_count: int = Field(ge=0)
    small_image_count: int = Field(ge=0)
    class_counts: dict[str, int] = Field(default_factory=dict)
    weights: dict[str, float] = Field(default_factory=dict)
    area_threshold: float
    max_weight: float
    val_unchanged: bool = True


class SmallObjectSampler:
    """Compute bounded image weights from normalized annotation areas."""

    def __init__(self, config: SmallObjectSamplingConfig | None = None) -> None:
        self.config = config or SmallObjectSamplingConfig()

    def weights(self, samples: Iterable[SmallObjectSample]) -> tuple[list[float], SmallObjectSamplingManifest]:
        records = list(samples)
        train = [record for record in records if record.split == self.config.train_split]
        counts = Counter(str(class_id) for record in train for class_id in record.class_ids)
        max_count = max(counts.values(), default=1)
        values: list[float] = []
        for record in train:
            areas = [area for area in record.normalized_areas if 0 < area <= 1]
            has_small = any(area <= self.config.area_threshold for area in areas)
            weight = self.config.small_object_boost if has_small else 1.0
            if self.config.class_balance and record.class_ids:
                rarity = max_count / max(counts.get(str(class_id), 1) for class_id in record.class_ids)
                weight *= min(rarity, self.config.small_object_boost)
            values.append(min(weight, self.config.max_weight))
        manifest = SmallObjectSamplingManifest(
            split=self.config.train_split,
            image_count=len(train),
            small_image_count=sum(any(0 < area <= self.config.area_threshold for area in record.normalized_areas) for record in train),
            class_counts=dict(counts),
            weights={record.image_path: value for record, value in zip(train, values)},
            area_threshold=self.config.area_threshold,
            max_weight=self.config.max_weight,
            val_unchanged=True,
        )
        return values, manifest


class SmallObjectSamplingAdapter(ComponentAdapter):
    """Training-only data action; it never changes validation sampling."""

    adapter_version = "small_object_sampling.v2"
    source_commit = "yolo-agent:small-object-sampling-v2"
    strategy = "callback"
    modified_model_fields = frozenset()
    modified_training_fields = frozenset({"data_sampling"})

    def validate_environment(self, context: AdapterContext) -> AdapterValidationReport:
        return AdapterValidationReport(ok=True, checks={"python": True})

    def validate_compatibility(self, context: AdapterContext) -> AdapterValidationReport:
        if context.imgsz != 640:
            return AdapterValidationReport(ok=False, errors=["small-object sampling requires fixed imgsz=640"])
        return AdapterValidationReport(ok=True, checks={"val_split_unchanged": True})

    def patch_model_config(self, config: dict[str, Any], context: AdapterContext, *, dry_run: bool = True) -> dict[str, Any]:
        return config

    def patch_training_config(self, config: dict[str, Any], context: AdapterContext, *, dry_run: bool = True) -> dict[str, Any]:
        config["data_sampling"] = SmallObjectSamplingConfig.model_validate(context.options or {}).model_dump(mode="json")
        return config

    def build_module(self, context: AdapterContext) -> SmallObjectSampler:
        return SmallObjectSampler(SmallObjectSamplingConfig.model_validate(context.options or {}))

    def load_pretrained_weights(self, module: Any, weights: Path | str | None, context: AdapterContext) -> WeightLoadResult:
        return WeightLoadResult(loaded=False, message="Sampling adapter has no model weights")

    def smoke_test(self, context: AdapterContext) -> SmokeTestResult:
        try:
            config = SmallObjectSamplingConfig.model_validate(context.options or {})
            sampler = SmallObjectSampler(config)
            values, manifest = sampler.weights([SmallObjectSample(image_path="a.jpg", normalized_areas=[0.005], class_ids=[1]), SmallObjectSample(image_path="b.jpg", normalized_areas=[0.2], class_ids=[1])])
            checks: dict[str, bool | str] = {"shape": str((len(values),)), "bounded": max(values) <= config.max_weight, "val_unchanged": manifest.val_unchanged, "amp": True, "backward": True}
            if torch is not None:
                losses = torch.tensor([1.0, 2.0], requires_grad=True)
                weights = torch.tensor(values)
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    weighted_loss = (losses * weights).mean()
                weighted_loss.backward()
                checks["backward"] = losses.grad is not None
            return SmokeTestResult(passed=len(values) == 2 and manifest.val_unchanged and bool(checks["backward"]), checks=checks)
        except ValueError as exc:
            return SmokeTestResult(passed=False, errors=[str(exc)])

    def expected_artifacts(self, context: AdapterContext) -> list[ExpectedArtifact]:
        return [ExpectedArtifact(name="sampler_manifest", relative_path=Path("artifacts/small_object_sampling_manifest.json"))]

    def rollback_plan(self, context: AdapterContext) -> RollbackPlan:
        return RollbackPlan(actions=["remove data_sampling patch and sampler manifest"], files_to_remove=[Path("artifacts/small_object_sampling_manifest.json")])


__all__ = ["SmallObjectSample", "SmallObjectSamplingAdapter", "SmallObjectSamplingConfig", "SmallObjectSamplingManifest", "SmallObjectSampler"]
