"""Build a persistent registry of real assets for every paper identity."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from yolo_agent.research.paper_execution_inventory import (
    PaperExecutionInventory,
    PaperExecutionSpec,
)
from yolo_agent.research.paper_execution_requirement_schemas import (
    PaperExecutionRequirement,
    PaperExecutionRequirementsMatrix,
)
from yolo_agent.research.paper_asset_schemas import (
    PaperAssetRecord,
    PaperAssetRegistry,
)


_ASSET_FIELDS = (
    "source_dataset_manifest",
    "target_dataset_manifest",
    "teacher_checkpoint",
    "hard_negative_manifest",
    "graph_config",
    "matched_baseline_artifact",
)
_DOMAIN_MARKERS = {
    "domain_adaptation.general",
    "adversarial_alignment",
    "feature_alignment",
    "pseudo_label_adaptation",
    "domain_distillation",
    "source_free_adaptation",
    "cross_domain_teacher",
    "contrastive_domain_alignment",
    "active_domain_adaptation",
}
_DISTILLATION_MARKERS = {
    "distillation.yolo26_teacher_student",
    "logits_distillation",
    "feature_distillation",
    "relation_distillation",
    "localization_distillation",
    "attention_distillation",
    "masked_feature_distillation",
    "quality_aware_distillation",
    "teacher_ensemble",
}


class PaperAssetRegistryBuilder:
    """Validate actual files without manufacturing missing research assets."""

    def build(
        self,
        inventory: PaperExecutionInventory,
        requirements: PaperExecutionRequirementsMatrix,
        *,
        source_inventory_path: Path | str,
        source_requirements_path: Path | str,
        dataset_manifest: Path | str | None = None,
        assets_by_paper: Mapping[str, Mapping[str, Path | str | None]] | None = None,
    ) -> PaperAssetRegistry:
        if inventory.inventory_hash != requirements.source_inventory_hash:
            raise ValueError("requirements were generated from a different inventory")
        if requirements.compatible_paper_count != inventory.compatible_paper_count:
            raise ValueError("inventory and requirements paper counts differ")
        requirement_by_id = {item.paper_id: item for item in requirements.requirements}
        if set(requirement_by_id) != {item.paper_id for item in inventory.records}:
            raise ValueError("requirements do not cover every inventory paper")
        overrides = assets_by_paper or {}
        records = [
            self._build_record(
                paper,
                requirement_by_id[paper.paper_id],
                default_dataset_manifest=dataset_manifest,
                overrides=overrides.get(paper.paper_id, {}),
            )
            for paper in inventory.records
        ]
        registry = PaperAssetRegistry(
            source_inventory_path=str(Path(source_inventory_path).resolve()),
            source_inventory_hash=inventory.inventory_hash,
            source_requirements_path=str(Path(source_requirements_path).resolve()),
            source_requirements_hash=_file_sha256(Path(source_requirements_path)),
            compatible_paper_count=inventory.compatible_paper_count,
            records=records,
        )
        return registry.with_hash()

    def _build_record(
        self,
        paper: PaperExecutionSpec,
        requirement: PaperExecutionRequirement,
        *,
        default_dataset_manifest: Path | str | None,
        overrides: Mapping[str, Path | str | None],
    ) -> PaperAssetRecord:
        mechanism = requirement.paper_specific_mechanism
        markers = set(paper.canonical_component_ids) | set(
            paper.paper_specific_mechanism_ids
        ) | set(requirement.paper_specific_mechanism_ids)
        is_domain = bool(markers & _DOMAIN_MARKERS) or bool(
            requirement.required_domain_assets
        )
        is_distillation = bool(markers & _DISTILLATION_MARKERS) or bool(
            requirement.required_teacher_assets
        )
        is_hard_negative = "hard_negative" in mechanism or any(
            "hard_negative" in item for item in markers
        )
        is_graph = bool(requirement.required_graph_assets) or any(
            mechanism.startswith(prefix)
            for prefix in ("neck.", "feature_pyramid.", "attention.", "detection_head.")
        )

        paths: dict[str, str | None] = {}
        for field_name in _ASSET_FIELDS:
            supplied = overrides.get(field_name)
            if supplied is None and field_name == "source_dataset_manifest":
                supplied = None if is_domain else default_dataset_manifest
            paths[field_name] = self._existing_absolute_path(supplied)

        blockers = self._requirement_blockers(paper, requirement)
        if requirement.execution_route == "training" and not is_domain:
            if paths["source_dataset_manifest"] is None:
                blockers.append("source_dataset_manifest_missing")
        if is_domain:
            if paths["source_dataset_manifest"] is None:
                blockers.append("domain_source_dataset_manifest_missing")
            if paths["target_dataset_manifest"] is None:
                blockers.append("domain_target_dataset_manifest_missing")
            if (
                paths["source_dataset_manifest"]
                and paths["target_dataset_manifest"]
                and Path(paths["source_dataset_manifest"]).resolve()
                == Path(paths["target_dataset_manifest"]).resolve()
            ):
                blockers.append("domain_source_target_manifest_identical")
            if default_dataset_manifest is not None and not overrides.get(
                "target_dataset_manifest"
            ):
                blockers.append("domain_target_manifest_not_provided")
        if is_distillation and paths["teacher_checkpoint"] is None:
            blockers.append("teacher_checkpoint_missing")
        if is_hard_negative:
            manifest = paths["hard_negative_manifest"]
            if manifest is None:
                blockers.append("train_side_hard_negative_manifest_missing")
            else:
                blockers.extend(self._validate_replay_manifest(Path(manifest)))
        if is_graph and paths["graph_config"] is None:
            blockers.append("graph_config_missing")
        if requirement.execution_route != "inference" and paths["matched_baseline_artifact"] is None:
            blockers.append("matched_baseline_artifact_missing")
        if paths["matched_baseline_artifact"] is not None:
            blockers.extend(
                self._validate_matched_baseline(
                    Path(paths["matched_baseline_artifact"]),
                    requirement,
                    paths["source_dataset_manifest"],
                )
            )

        hashes = {
            field_name: _file_sha256(Path(path))
            for field_name, path in paths.items()
            if path is not None
        }
        teacher_sha = hashes.get("teacher_checkpoint")
        available = not blockers
        aggregate = _aggregate_hash(hashes) if hashes else None
        return PaperAssetRecord(
            paper_id=paper.paper_id,
            mechanism_id=mechanism,
            source_dataset_manifest=paths["source_dataset_manifest"],
            target_dataset_manifest=paths["target_dataset_manifest"],
            teacher_checkpoint=paths["teacher_checkpoint"],
            teacher_sha256=teacher_sha,
            hard_negative_manifest=paths["hard_negative_manifest"],
            graph_config=paths["graph_config"],
            matched_baseline_artifact=paths["matched_baseline_artifact"],
            asset_sha256=aggregate,
            protocol_hash=requirement.protocol_hash,
            availability="available" if available else "unavailable",
            exact_blocker=";".join(dict.fromkeys(blockers)),
            recovery_action=self._recovery_action(blockers),
            current_disposition=paper.current_disposition,
            asset_hashes=hashes,
            validated_assets=sorted(hashes),
        )

    @staticmethod
    def _existing_absolute_path(value: Path | str | None) -> str | None:
        if value is None:
            return None
        path = Path(value).expanduser().resolve(strict=False)
        return str(path) if path.is_file() else None

    @staticmethod
    def _requirement_blockers(
        paper: PaperExecutionSpec,
        requirement: PaperExecutionRequirement,
    ) -> list[str]:
        blockers: list[str] = []
        if requirement.exact_blocker:
            blockers.extend(
                item for item in requirement.exact_blocker.split(";") if item
            )
        if not requirement.required_adapter and requirement.execution_route != "inference":
            blockers.append("required_adapter_missing")
        if not paper.paper_specific_mechanism_ids:
            blockers.append("paper_specific_mechanism_missing")
        if requirement.execution_route == "inference":
            blockers.append("inference_only_not_training_asset")
        return blockers

    @staticmethod
    def _validate_replay_manifest(path: Path) -> list[str]:
        try:
            payload = _load_mapping(path)
        except (OSError, ValueError) as exc:
            return [f"hard_negative_manifest_invalid:{exc}"]
        split = payload.get("source_split") or payload.get("split")
        if split != "train":
            return ["hard_negative_manifest_not_train_split"]
        entries = payload.get("records", payload.get("samples", []))
        if isinstance(entries, list) and any(
            isinstance(item, dict) and item.get("split") not in {None, "train"}
            for item in entries
        ):
            return ["hard_negative_manifest_contains_non_train_sample"]
        return []

    @staticmethod
    def _validate_matched_baseline(
        path: Path,
        requirement: PaperExecutionRequirement,
        source_manifest: str | None,
    ) -> list[str]:
        try:
            payload = _load_mapping(path)
        except (OSError, ValueError) as exc:
            return [f"matched_baseline_artifact_invalid:{exc}"]
        blockers: list[str] = []
        if payload.get("protocol_hash") != requirement.protocol_hash:
            blockers.append("matched_baseline_protocol_mismatch")
        if payload.get("imgsz") not in {None, 640}:
            blockers.append("matched_baseline_imgsz_mismatch")
        if source_manifest and payload.get("dataset_manifest_hash"):
            if payload["dataset_manifest_hash"] != _file_sha256(Path(source_manifest)):
                blockers.append("matched_baseline_dataset_manifest_mismatch")
        return blockers

    @staticmethod
    def _recovery_action(blockers: list[str]) -> str:
        if not blockers:
            return "none; assets are file-backed and protocol-validated"
        actions: list[str] = []
        if any("domain" in blocker for blocker in blockers):
            actions.append("provide distinct real source and target manifests with an explicit domain pair")
        if any("teacher" in blocker for blocker in blockers):
            actions.append("provide a frozen teacher checkpoint and verify its SHA-256")
        if any("hard_negative" in blocker for blocker in blockers):
            actions.append("generate a train-split hard-negative manifest without validation leakage")
        if any("baseline" in blocker for blocker in blockers):
            actions.append("produce a matched baseline artifact with the same protocol, split, and imgsz")
        if any("graph" in blocker for blocker in blockers):
            actions.append("provide the real graph configuration used by the adapter")
        if any("mechanism" in blocker or "adapter" in blocker for blocker in blockers):
            actions.append("implement and bind the paper-specific adapter and recipe")
        return "; ".join(dict.fromkeys(actions)) or "resolve the recorded asset blocker"


def build_paper_asset_registry(
    inventory_path: Path | str = Path("runs/coverage-audit/paper_execution_inventory.yaml"),
    requirements_path: Path | str = Path("runs/coverage-audit/paper_execution_requirements.yaml"),
    output_path: Path | str = Path("runs/paper-readiness/paper_asset_registry.yaml"),
    *,
    dataset_manifest: Path | str | None = None,
    assets_by_paper: Mapping[str, Mapping[str, Path | str | None]] | None = None,
) -> PaperAssetRegistry:
    inventory_file = Path(inventory_path).resolve()
    requirements_file = Path(requirements_path).resolve()
    inventory = PaperExecutionInventory.from_yaml(inventory_file)
    requirements = PaperExecutionRequirementsMatrix.from_yaml(requirements_file)
    registry = PaperAssetRegistryBuilder().build(
        inventory,
        requirements,
        source_inventory_path=inventory_file,
        source_requirements_path=requirements_file,
        dataset_manifest=dataset_manifest,
        assets_by_paper=assets_by_paper,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(registry.model_dump(mode="json", exclude_none=True), sort_keys=False),
        encoding="utf-8",
    )
    return registry


def _load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"asset artifact must contain a mapping: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate_hash(asset_hashes: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(asset_hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "PaperAssetRegistryBuilder",
    "build_paper_asset_registry",
]
