"""CPU golden-path validation for the small-object sampling runtime."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

from yolo_agent.adapters.ultralytics.plugin_bridge import (
    PluginDetectionTrainer,
    UltralyticsTrainerPluginBridge,
)
from yolo_agent.adapters.ultralytics.plugin_context import (
    PluginRuntimeEvidence,
    runtime_evidence_path,
)
from yolo_agent.certification.small_object_sampling_schemas import (
    SmallObjectSamplingCpuReport,
)
from yolo_agent.components.adapters.runtime import AdapterRuntimePayload
from yolo_agent.components.adapters.sampling.small_object_sampling import (
    DeterministicDistributedWeightedSampler,
    SmallObjectSamplingManifest,
)


def run_small_object_sampling_cpu_fixture(
    *,
    runtime_payload_path: Path | str,
    workspace: Path | str,
) -> SmallObjectSamplingCpuReport:
    """Exercise the actual Ultralytics train dataloader hook without training."""
    root = Path(workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload_path = Path(runtime_payload_path).resolve()
    payload = AdapterRuntimePayload.read(payload_path, verify_imports=True)
    report_path = root / "small_object_sampling_cpu_golden_path.yaml"
    manifest_path = payload_path.parent / "sampler_manifest.json"
    evidence_path = runtime_evidence_path(payload_path)
    state_path = payload_path.parent / "small_object_sampler_state.rank1.json"
    checks: dict[str, bool | str | int | float] = {}
    errors: list[str] = []
    try:
        if payload.component_ids != ["sampling.small_object"]:
            raise ValueError("sampling CPU fixture requires sampling.small_object payload")
        bridge = UltralyticsTrainerPluginBridge(payload_path)
        trainer = object.__new__(PluginDetectionTrainer)
        trainer.plugin_bridge = bridge
        trainer.args = SimpleNamespace(seed=17, resume=False)
        trainer.epoch = 4

        with _temporary_environment(WORLD_SIZE="2"):
            rank0_loader = trainer.apply_dataloader_plugins(
                _fixture_loader(),
                dataset_path="fixture/images/train",
                batch_size=2,
                rank=0,
                mode="train",
            )
            rank0_sampler = _sampling_sampler(rank0_loader)
            manifest_after_rank0 = manifest_path.read_bytes()
            rank1_loader = trainer.apply_dataloader_plugins(
                _fixture_loader(),
                dataset_path="fixture/images/train",
                batch_size=2,
                rank=1,
                mode="train",
            )
            rank1_sampler = _sampling_sampler(rank1_loader)

        manifest = SmallObjectSamplingManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8-sig")
        )
        checks["sampler_manifest_verified"] = bool(
            manifest.protocol_hash == payload.protocol_hash
            and manifest.runtime_payload_hash == payload.payload_hash
            and manifest.dataset_manifest == "cpu-fixture-manifest-v1"
            and manifest.split == "train"
            and manifest.val_unchanged
            and len(manifest.raw_weights) == len(manifest.final_weights) == 4
        )
        checks["ddp_deterministic_sharding"] = _ddp_shards_match(
            rank0_sampler,
            rank1_sampler,
        ) and manifest_path.read_bytes() == manifest_after_rank0

        validation_loader = _fixture_loader()
        before_val = _train_hook_calls(evidence_path)
        unchanged = trainer.apply_dataloader_plugins(
            validation_loader,
            dataset_path="fixture/images/val",
            batch_size=2,
            rank=0,
            mode="val",
        )
        after_val = _train_hook_calls(evidence_path)
        checks["validation_loader_unchanged"] = bool(
            unchanged is validation_loader and before_val == after_val
        )

        bridge.invoke_event(
            "on_checkpoint_save",
            trainer=trainer,
            checkpoints={},
        )
        state = _read_mapping(state_path)
        resumed_bridge = UltralyticsTrainerPluginBridge(payload_path)
        resumed_trainer = object.__new__(PluginDetectionTrainer)
        resumed_trainer.plugin_bridge = resumed_bridge
        resumed_trainer.args = SimpleNamespace(seed=17, resume=False)
        with _temporary_environment(WORLD_SIZE="2"):
            resumed_trainer.apply_dataloader_plugins(
                _fixture_loader(),
                dataset_path="fixture/images/train",
                batch_size=2,
                rank=1,
                mode="train",
            )
        resumed_bridge.invoke_event(
            "on_checkpoint_load",
            trainer=resumed_trainer,
            checkpoint={"small_object_sampler_state": state},
        )
        bridge.context.persist()
        resumed_bridge.context.persist()
        checks["resume_state_restored"] = bool(
            resumed_trainer.small_object_sampler.epoch == trainer.epoch
        )

        evidence = PluginRuntimeEvidence.model_validate_json(
            evidence_path.read_text(encoding="utf-8-sig")
        )
        hook_calls = sum(
            hooks.get("build_train_dataloader", 0)
            for hooks in evidence.hook_call_counts.values()
        )
        checks["train_dataloader_hook_calls"] = hook_calls
        # Two calls prove both simulated DDP ranks traversed the train-only hook;
        # resume validation is checked separately below.
        checks["train_dataloader_hook_called"] = hook_calls >= 2
        checks["plugin_failures_empty"] = not evidence.failures
        failed_checks = sorted(
            key
            for key, value in checks.items()
            if key != "train_dataloader_hook_calls" and value is not True
        )
        if failed_checks:
            errors.append("failed sampling CPU checks: " + ", ".join(failed_checks))
    except (AttributeError, ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        errors.append(str(exc))

    report = SmallObjectSamplingCpuReport(
        status="failed" if errors else "passed",
        protocol_hash=payload.protocol_hash,
        runtime_payload_hash=payload.payload_hash,
        sampler_manifest_path=manifest_path if manifest_path.is_file() else None,
        runtime_evidence_path=evidence_path if evidence_path.is_file() else None,
        sampler_state_path=state_path if state_path.is_file() else None,
        checks=checks,
        errors=errors,
    )
    report.to_yaml(report_path, exclude_none=True, sort_keys=False)
    return report


def _fixture_loader():  # type: ignore[no-untyped-def]
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:  # pragma: no cover - certification dependency
        raise ImportError("sampling CPU fixture requires torch") from exc

    class FixtureDataset(Dataset):  # type: ignore[type-arg]
        manifest_hash = "cpu-fixture-manifest-v1"
        im_files = [f"fixture-{index}.jpg" for index in range(4)]
        labels = [
            {
                "im_file": im_files[0],
                "normalized": True,
                "bbox_format": "xywh",
                "bboxes": [[0.5, 0.5, 0.04, 0.04]],
                "cls": [[0]],
            },
            {
                "im_file": im_files[1],
                "normalized": True,
                "bbox_format": "xywh",
                "bboxes": [[0.5, 0.5, 0.4, 0.4]],
                "cls": [[1]],
            },
            {
                "im_file": im_files[2],
                "normalized": True,
                "bbox_format": "xywh",
                "bboxes": [[0.5, 0.5, 0.06, 0.06]],
                "cls": [[2]],
            },
            {
                "im_file": im_files[3],
                "normalized": True,
                "bbox_format": "xywh",
                "bboxes": [[0.5, 0.5, 0.3, 0.3]],
                "cls": [[1]],
            },
        ]

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> dict[str, object]:
            return {
                "img": torch.full((3, 16, 16), float(index)),
                "index": index,
            }

    return DataLoader(FixtureDataset(), batch_size=2, shuffle=False, num_workers=0)


def _sampling_sampler(loader: object) -> DeterministicDistributedWeightedSampler:
    sampler = getattr(loader, "sampler", None)
    if not isinstance(sampler, DeterministicDistributedWeightedSampler):
        raise TypeError("train dataloader hook did not install the sampling policy")
    return sampler


def _ddp_shards_match(
    rank0: DeterministicDistributedWeightedSampler,
    rank1: DeterministicDistributedWeightedSampler,
) -> bool:
    global_stream = rank0.global_indices()
    return bool(
        rank0.world_size == rank1.world_size == 2
        and rank0.global_indices() == rank1.global_indices()
        and list(rank0) == global_stream[0::2]
        and list(rank1) == global_stream[1::2]
    )


def _train_hook_calls(path: Path) -> int:
    evidence = PluginRuntimeEvidence.model_validate_json(
        path.read_text(encoding="utf-8-sig")
    )
    return sum(
        hooks.get("build_train_dataloader", 0)
        for hooks in evidence.hook_call_counts.values()
    )


def _read_mapping(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"sampling state must be a mapping: {path}")
    return value


@contextmanager
def _temporary_environment(**updates: str) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


__all__ = ["run_small_object_sampling_cpu_fixture"]
