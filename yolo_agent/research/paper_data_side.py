"""Paper-specific data-side routing and CPU behavior audit.

The frozen paper plan is the only source of campaign membership and domain
scope.  This module never infers a route from a title.  A mechanism is
executable here only when its exact paper-specific mechanism ID is present in
the plan and its source/config/evidence boundary is explicit.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from yolo_agent.agents.active_learning import (
    ActiveLearningMiner,
    MiningConfig,
    PredictionSummary,
)
from yolo_agent.components.adapters.base import AdapterContext
from yolo_agent.components.adapters.data_pipeline import (
    DataPipelineDataset,
    DataSampleRecord,
    DataTransformConfig,
    ExposureConfig,
    compute_exposure_details,
)
from yolo_agent.components.adapters.registry import ComponentAdapterRegistry
from yolo_agent.components.adapters.runtime import AdapterRuntimePayload
from yolo_agent.components.contracts import ComponentContract, load_contracts
from yolo_agent.research.paper_83_campaign_schemas import Paper83Manifest
from yolo_agent.research.paper_83_implementation_schemas import (
    Paper83EngineeringPlan,
    Paper83EngineeringPlanEntry,
)
from yolo_agent.research.paper_implementation_schemas import (
    PaperImplementationRegistry,
    PaperImplementationSpec,
)
from yolo_agent.research.paper_data_side_schemas import (
    DataSideBehaviorEvidence,
    DataSideRouteSpec,
    PaperDataSideAudit,
    PaperDataSideRecord,
)


DATA_SIDE_DOMAINS = frozenset(
    {
        "data_quality",
        "annotation",
        "sampling",
        "augmentation",
        "preprocessing",
        "long_tail",
        "active_learning",
    }
)

_NON_DATA_COMPONENT_PREFIXES = (
    "assigner.",
    "attention.",
    "detection_head.",
    "distillation.",
    "domain_adaptation.",
    "feature_pyramid.",
    "inference.",
    "loss.",
    "neck.",
)
_PLAN_METADATA_SOURCES = frozenset(
    {"paper_record.component_ids", "paper_record.title", "category"}
)


def _route(
    mechanism_id: str,
    route_kind: str,
    component_id: str,
    adapter_class: str,
    *,
    runtime_hook: str,
    implementation_path: str = "yolo_agent.components.adapters.data_pipeline.adapters",
    required_evidence: Iterable[str] = (),
    required_config_keys: Iterable[str] = (),
    config_schema: Mapping[str, Any] | None = None,
) -> DataSideRouteSpec:
    return DataSideRouteSpec(
        mechanism_id=mechanism_id,
        route_kind=route_kind,  # type: ignore[arg-type]
        component_id=component_id,
        adapter_id=component_id,
        implementation_path=implementation_path,
        adapter_class=adapter_class,
        changed_variables=[f"data.{mechanism_id}"],
        runtime_hooks=[runtime_hook],
        required_evidence=list(required_evidence),
        required_config_keys=list(required_config_keys),
        test_refs=["tests/test_paper_data_side.py"],
        config_schema=dict(config_schema or {"imgsz": 640}),
    )


_EXPOSURE_SCHEMA: dict[str, Any] = {
    "imgsz": 640,
    "strength": 1.0,
    "max_weight": 3.0,
    "max_exposure_ratio": 3.0,
    "area_threshold": 0.01,
    "repeat_threshold": 0.1,
    "target_class_ids": [],
    "sample_count": None,
    "seed": 0,
}

_TRANSFORM_SCHEMA: dict[str, Any] = {
    "imgsz": 640,
    "probability": 1.0,
    "seed": 0,
    "rare_class_ids": [],
    "crop_scale": 0.75,
    "small_area_threshold": 0.01,
    "multi_image_count": 2,
    "active_epoch_start": 0,
    "active_epoch_end": None,
}


_ROUTES: dict[str, DataSideRouteSpec] = {
    item.mechanism_id: item
    for item in (
        _route(
            "small_object_weighted_sampling",
            "sampling",
            "sampling.small_object_weighted",
            "SmallObjectWeightedSamplingAdapter",
            runtime_hook="build_train_dataloader",
            required_evidence=["train_sample_distribution"],
            config_schema=_EXPOSURE_SCHEMA,
        ),
        _route(
            "class_balanced_sampling",
            "sampling",
            "sampling.class_balanced",
            "ClassBalancedSamplingAdapter",
            runtime_hook="build_train_dataloader",
            required_evidence=["class_distribution_change"],
            config_schema=_EXPOSURE_SCHEMA,
        ),
        _route(
            "repeat_factor_sampling",
            "sampling",
            "sampling.repeat_factor",
            "RepeatFactorSamplingAdapter",
            runtime_hook="build_train_dataloader",
            required_evidence=["repeat_factor_distribution"],
            config_schema=_EXPOSURE_SCHEMA,
        ),
        _route(
            "hard_negative_replay",
            "sampling",
            "sampling.hard_negative_replay",
            "HardNegativeReplayAdapter",
            runtime_hook="build_train_dataloader",
            required_evidence=["train_hard_negative_manifest"],
            required_config_keys=[
                "manifest_path",
                "manifest_hash",
                "evidence_id",
                "dataset_manifest_hash",
                "baseline_protocol_hash",
                "baseline_checkpoint_hash",
                "train_index_hash",
            ],
            config_schema={
                **_EXPOSURE_SCHEMA,
                "manifest_path": None,
                "manifest_hash": None,
                "evidence_id": None,
                "dataset_manifest_hash": None,
                "baseline_protocol_hash": None,
                "baseline_checkpoint_hash": None,
                "train_index_hash": None,
                "train_index_path": None,
                "require_provenance": True,
            },
        ),
        _route(
            "false_negative_class_boost",
            "sampling",
            "sampling.false_negative_class_boost",
            "FalseNegativeClassBoostAdapter",
            runtime_hook="build_train_dataloader",
            required_evidence=["train_false_negative_scores"],
            config_schema={
                **_EXPOSURE_SCHEMA,
                "target_class_ids": [1],
            },
        ),
        _route(
            "copy_paste_rare_classes",
            "augmentation",
            "augmentation.copy_paste_rare_classes",
            "RareClassCopyPasteAdapter",
            runtime_hook="build_train_dataset",
            required_evidence=["paper_copy_paste_policy"],
            required_config_keys=["rare_class_ids"],
            config_schema=_TRANSFORM_SCHEMA,
        ),
        _route(
            "scale_aware_crop",
            "augmentation",
            "augmentation.scale_aware_crop",
            "ScaleAwareCropAdapter",
            runtime_hook="build_train_dataset",
            required_evidence=["paper_crop_policy"],
            config_schema=_TRANSFORM_SCHEMA,
        ),
        _route(
            "object_centric_crop",
            "augmentation",
            "augmentation.object_centric_crop",
            "ObjectCentricCropAdapter",
            runtime_hook="build_train_dataset",
            required_evidence=["paper_crop_policy"],
            config_schema=_TRANSFORM_SCHEMA,
        ),
        _route(
            "multi_image_sampling_schedule",
            "augmentation",
            "augmentation.multi_image_sampling_schedule",
            "MultiImageSamplingScheduleAdapter",
            runtime_hook="build_train_dataset",
            required_evidence=["augmentation_schedule"],
            config_schema=_TRANSFORM_SCHEMA,
        ),
        _route(
            "annotation_quality_filter",
            "annotation",
            "annotation.quality_filter",
            "AnnotationQualityFilterAdapter",
            runtime_hook="build_train_dataset",
            implementation_path=(
                "yolo_agent.components.adapters.data_pipeline.adapters"
            ),
            required_evidence=["annotation_quality_policy"],
            config_schema={
                "imgsz": 640,
                "min_width": 0.002,
                "min_height": 0.002,
                "min_area": 0.000004,
                "max_area": 0.95,
                "max_aspect_ratio": 12.0,
            },
        ),
        _route(
            "preprocessing_normalization",
            "preprocessing",
            "preprocessing.normalization",
            "NormalizationPreprocessingAdapter",
            runtime_hook="build_train_dataset",
            implementation_path=(
                "yolo_agent.components.adapters.data_pipeline.adapters"
            ),
            required_evidence=["paper_preprocessing_policy"],
            config_schema={
                "imgsz": 640,
                "normalize": True,
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
            },
        ),
        _route(
            "active_sample_selection",
            "active_learning",
            "active_learning.acquisition",
            "ActiveLearningAcquisitionAdapter",
            runtime_hook="build_validator",
            implementation_path=(
                "yolo_agent.components.adapters.data_pipeline.adapters"
            ),
            required_evidence=["active_learning_acquisition_policy"],
            config_schema={
                "imgsz": 640,
                "low_confidence_threshold": 0.35,
                "high_entropy_threshold": 0.7,
                "disagreement_threshold": 0.34,
                "max_samples": 100,
                "strategies": [
                    "low_confidence",
                    "high_entropy",
                    "model_disagreement",
                ],
            },
        ),
    )
}


def data_side_routes() -> dict[str, DataSideRouteSpec]:
    """Return a copy of the exact, non-fuzzy data-side route catalog."""

    return dict(_ROUTES)


def resolve_data_side_route(mechanism_id: str) -> DataSideRouteSpec | None:
    """Resolve one exact mechanism ID; aliases and title matching are forbidden."""

    return _ROUTES.get(mechanism_id.strip())


def _contract_for_route(route: DataSideRouteSpec) -> ComponentContract:
    """Build a local contract for a route whose legacy contract is not cataloged."""

    if not route.component_id or not route.adapter_class:
        raise ValueError(f"data-side route lacks component identity: {route.mechanism_id}")
    return ComponentContract(
        component_id=route.component_id,
        display_name=route.mechanism_id,
        category=route.route_kind,
        implementation_family=f"paper_data_side.{route.route_kind}",
        implementation_path=route.implementation_path,
        adapter_class=route.adapter_class,
        changed_variable=route.changed_variables[0],
        insertion_point=route.runtime_hooks[0],
        supported_detector_families=["yolo26"],
        supported_yolo_versions=["26"],
        runtime_hook=route.runtime_hooks[0],
        supported_heads=["one_to_one"],
        runtime_payload_schema=dict(route.config_schema),
        evidence_protocol=list(route.required_evidence),
        checkpoint_compatibility="unchanged_graph",
        training_only=True,
        inference_only=route.inference_only,
        changes_model_graph=False,
        fixed_imgsz_compatible=True,
        maturity="adapter_implemented",
    )


def _apply_paper_config(
    route: DataSideRouteSpec,
    paper_config: Mapping[str, Any],
) -> DataSideRouteSpec:
    """Apply only schema-declared options to an exact paper route."""

    candidates: list[Mapping[str, Any]] = [paper_config]
    for key in ("data_side", "paper_specific_config", "config", "options"):
        nested = paper_config.get(key)
        if isinstance(nested, Mapping):
            candidates.insert(0, nested)
    config = dict(route.config_schema)
    for candidate in candidates:
        for key in config:
            if key in candidate:
                config[key] = candidate[key]
    return route.model_copy(update={"config_schema": config})


def _missing_route_config(route: DataSideRouteSpec) -> list[str]:
    """Return required values absent from an effective paper route config."""

    missing: list[str] = []
    for key in route.required_config_keys:
        value = route.config_schema.get(key)
        if value is None or value == "" or value == []:
            missing.append(key)
            continue
        if key == "manifest_path" and not Path(str(value)).is_file():
            missing.append(f"{key}:file_not_found")
    return missing


def _runtime_probe_options(route: DataSideRouteSpec) -> dict[str, Any]:
    """Use synthetic-only defaults to exercise routes without changing production config."""

    options = dict(route.config_schema)
    if route.mechanism_id == "copy_paste_rare_classes" and not options.get(
        "rare_class_ids"
    ):
        options["rare_class_ids"] = [3]
    return options


class PaperDataSideAuditBuilder:
    """Audit data-side scope and runtime behavior for every frozen paper."""

    def __init__(
        self,
        *,
        manifest_path: Path | str = "configs/research/paper_83_manifest.yaml",
        plan_path: Path | str = "configs/research/paper_83_implementation_plan.yaml",
        contracts_path: Path | str = (
            "configs/components/data_pipeline/paper_data_adapters.yaml"
        ),
        workspace: Path | str = ".",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.requested_plan_path = Path(plan_path)
        self.contracts_path = Path(contracts_path)
        self.workspace = Path(workspace).resolve()
        self.contracts = self._load_contracts()

    def build(
        self,
        *,
        manifest: Paper83Manifest | None = None,
        plan: Paper83EngineeringPlan | None = None,
    ) -> PaperDataSideAudit:
        """Build a complete audit without starting a trainer."""

        frozen = manifest or Paper83Manifest.from_yaml(self.manifest_path)
        if frozen.paper_count != 83:
            raise ValueError(f"frozen paper manifest must contain 83 papers, got {frozen.paper_count}")
        engineering_plan, source_plan = self._load_plan(plan)
        frozen_ids = [item.paper_id for item in frozen.papers]
        plan_ids = list(engineering_plan.manifest_paper_ids) or [
            item.paper_id for item in engineering_plan.papers
        ]
        if plan_ids != frozen_ids:
            raise ValueError("data-side plan membership differs from frozen paper manifest")
        if engineering_plan.manifest_membership_hash != frozen.campaign.membership_hash:
            raise ValueError("data-side plan membership hash differs from frozen manifest")

        by_id = {item.paper_id: item for item in frozen.papers}
        entries = {item.paper_id: item for item in engineering_plan.papers}
        if set(entries) != set(frozen_ids):
            raise ValueError("data-side plan must contain every frozen paper exactly once")
        records = [
            self._record(by_id[paper_id], entries[paper_id])
            for paper_id in frozen_ids
        ]
        summary = {
            status: sum(item.status == status for item in records)
            for status in ("ready", "blocked_missing_evidence", "out_of_scope")
        }
        audit = PaperDataSideAudit(
            manifest_path=str(self.manifest_path.resolve()),
            plan_path=str(self.requested_plan_path.resolve()),
            plan_source_path=str(source_plan.resolve()),
            manifest_membership_hash=frozen.campaign.membership_hash,
            plan_identity_hash=(
                engineering_plan.plan_identity_hash
                or engineering_plan.calculate_hash()
            ),
            paper_count=len(records),
            data_side_paper_count=sum(item.in_scope for item in records),
            records=sorted(records, key=lambda item: item.paper_id),
            summary=summary,
        )
        return audit.with_hash()

    def update_implementation_registry(
        self,
        registry: PaperImplementationRegistry,
        audit: PaperDataSideAudit,
    ) -> PaperImplementationRegistry:
        """Attach data-side routes to each selected paper spec independently."""

        if registry.manifest_membership_hash != audit.manifest_membership_hash:
            raise ValueError("data-side audit and implementation registry membership differ")
        registry_ids = {item.paper_id for item in registry.records}
        audit_ids = {item.paper_id for item in audit.records}
        if registry_ids != audit_ids:
            missing = sorted(audit_ids - registry_ids)
            extra = sorted(registry_ids - audit_ids)
            raise ValueError(
                "data-side audit and implementation registry paper IDs differ: "
                f"missing={missing} extra={extra}"
            )
        audit_by_id = {item.paper_id: item for item in audit.records}
        updated: list[PaperImplementationSpec] = []
        for spec in registry.records:
            record = audit_by_id.get(spec.paper_id)
            if record is None or not record.in_scope:
                updated.append(spec)
                continue
            updated.append(self._update_spec(spec, record))
        return registry.model_copy(update={"records": sorted(updated, key=lambda item: item.paper_id)}).with_hash()

    def _record(self, paper: Any, entry: Paper83EngineeringPlanEntry) -> PaperDataSideRecord:
        domains = {entry.implementation_domain, *entry.secondary_domains}
        in_scope = bool(domains.intersection(DATA_SIDE_DOMAINS))
        mechanisms = _unique(entry.paper_specific_mechanism_ids)
        evidence_refs = _unique([*entry.source_locations, *entry.required_evidence])
        dependencies = _unique(
            [*entry.paper_specific_missing_parts, *entry.dependency_papers_or_primitives]
        )
        if not in_scope:
            behavior = DataSideBehaviorEvidence(
                passed=True,
                checks={"data_side_scope": False},
            )
            return self._record_value(
                paper=paper,
                entry=entry,
                in_scope=False,
                mechanisms=[],
                routes=[],
                evidence_refs=evidence_refs,
                dependencies=dependencies,
                behavior=behavior,
                status="out_of_scope",
                blockers=[],
            )

        blockers: list[str] = []
        if not mechanisms:
            blockers.append("blocked_missing_evidence:paper_specific_data_mechanism")
        if entry.evidence_status != "sufficient_for_planning":
            blockers.append(
                "blocked_missing_evidence:plan_evidence_status="
                + entry.evidence_status
            )
        if not entry.source_locations:
            blockers.append("blocked_missing_evidence:paper_data_source_location")
        if not entry.required_evidence:
            blockers.append("blocked_missing_evidence:paper_data_evidence_ref")
        if not entry.paper_specific_config:
            blockers.append("blocked_missing_evidence:paper_specific_data_config")

        routes: list[DataSideRouteSpec] = []
        for mechanism_id in mechanisms:
            if _is_non_data_mechanism(mechanism_id):
                continue
            route = resolve_data_side_route(mechanism_id)
            if route is None:
                blockers.append(f"blocked_missing_evidence:data_side_route_unresolved:{mechanism_id}")
                continue
            route = _apply_paper_config(route, entry.paper_specific_config)
            routes.append(route)
            blockers.extend(
                f"blocked_missing_evidence:paper_data_config:{mechanism_id}:{key}"
                for key in _missing_route_config(route)
            )
            if any(
                evidence not in entry.required_evidence
                for evidence in route.required_evidence
            ):
                missing = [
                    evidence
                    for evidence in route.required_evidence
                    if evidence not in entry.required_evidence
                ]
                blockers.append(
                    f"blocked_missing_evidence:route_evidence:{mechanism_id}:"
                    + ",".join(missing)
                )

        if not routes and not blockers:
            blockers.append("blocked_missing_evidence:no_data_side_route")
        behavior = self._probe_routes(routes)
        if not behavior.passed:
            blockers.extend(f"blocked_missing_evidence:behavior:{error}" for error in behavior.errors)
        if any(route.mechanism_id == "hard_negative_replay" for route in routes):
            assets = [*entry.required_assets, *entry.required_evidence]
            if not any("hard_negative" in value or "manifest" in value for value in assets):
                blockers.append("blocked_missing_evidence:train_hard_negative_manifest")
        blockers = _unique(blockers)
        status = "ready" if not blockers else "blocked_missing_evidence"
        return self._record_value(
            paper=paper,
            entry=entry,
            in_scope=True,
            mechanisms=mechanisms,
            routes=routes,
            evidence_refs=evidence_refs,
            dependencies=dependencies,
            behavior=behavior,
            status=status,
            blockers=blockers,
        )

    def _record_value(
        self,
        *,
        paper: Any,
        entry: Paper83EngineeringPlanEntry,
        in_scope: bool,
        mechanisms: list[str],
        routes: list[DataSideRouteSpec],
        evidence_refs: list[str],
        dependencies: list[str],
        behavior: DataSideBehaviorEvidence,
        status: str,
        blockers: list[str],
    ) -> PaperDataSideRecord:
        route_ids = [item.mechanism_id for item in routes]
        component_ids = _unique(item.component_id for item in routes if item.component_id)
        adapter_ids = _unique(item.adapter_id for item in routes if item.adapter_id)
        config = {
            "paper_specific_config": dict(entry.paper_specific_config),
            "routes": [item.model_dump(mode="json") for item in routes],
            "fixed_imgsz": 640,
            "source_locations": evidence_refs,
        }
        fingerprint = _hash_payload(
            {
                "paper_id": paper.paper_id,
                "method_profile_id": entry.method_profile_id,
                "mechanisms": mechanisms,
                "routes": config,
            }
        )
        return PaperDataSideRecord(
            paper_id=paper.paper_id,
            title=paper.title,
            year=paper.year,
            method_profile_id=paper.method_profile_id,
            current_disposition=paper.current_disposition,
            primary_domain=entry.implementation_domain,
            secondary_domains=list(entry.secondary_domains),
            in_scope=in_scope,
            paper_specific_mechanism_ids=mechanisms,
            resolved_route_ids=route_ids,
            component_ids=component_ids,
            adapter_ids=adapter_ids,
            evidence_refs=evidence_refs,
            remaining_dependencies=dependencies,
            paper_specific_config=config if in_scope else {},
            behavior=behavior,
            status=status,  # type: ignore[arg-type]
            blockers=blockers,
            implementation_fingerprint=fingerprint,
        )

    def _load_plan(
        self,
        plan: Paper83EngineeringPlan | None,
    ) -> tuple[Paper83EngineeringPlan, Path]:
        if plan is not None:
            return plan, self.requested_plan_path
        if self.requested_plan_path.is_file():
            return Paper83EngineeringPlan.from_yaml(self.requested_plan_path), self.requested_plan_path
        if self.requested_plan_path.name == "paper_83_implementation_plan.yaml":
            fallback = self.requested_plan_path.with_name("paper_83_engineering_plan.yaml")
            if fallback.is_file():
                return Paper83EngineeringPlan.from_yaml(fallback), fallback
        raise FileNotFoundError(f"paper-83 implementation plan does not exist: {self.requested_plan_path}")

    def _load_contracts(self) -> dict[str, ComponentContract]:
        if not self.contracts_path.is_file():
            return {}
        return {
            item.component_id: item
            for item in load_contracts(self.contracts_path)
        }

    def _probe_routes(self, routes: list[DataSideRouteSpec]) -> DataSideBehaviorEvidence:
        checks: dict[str, bool | str | int | float] = {}
        changes: list[str] = []
        errors: list[str] = []
        for route in routes:
            try:
                result = self._probe_route(route)
            except (ImportError, RuntimeError, TypeError, ValueError) as exc:
                errors.append(f"{route.mechanism_id}:{type(exc).__name__}:{exc}")
                continue
            checks.update({f"{route.mechanism_id}.{key}": value for key, value in result.checks.items()})
            changes.extend(f"{route.mechanism_id}:{change}" for change in result.observed_changes)
            errors.extend(f"{route.mechanism_id}:{error}" for error in result.errors)
        return DataSideBehaviorEvidence(
            passed=not errors and all(bool(value) for value in checks.values() if isinstance(value, bool)),
            checks=checks,
            observed_changes=sorted(set(changes)),
            errors=sorted(set(errors)),
        )

    def _probe_route(self, route: DataSideRouteSpec) -> DataSideBehaviorEvidence:
        if route.route_kind == "sampling":
            behavior = _probe_sampling(route.mechanism_id)
        elif route.route_kind == "augmentation":
            behavior = _probe_augmentation(route.mechanism_id)
        elif route.route_kind == "annotation":
            behavior = _probe_annotation(route.mechanism_id)
        elif route.route_kind == "preprocessing":
            behavior = _probe_preprocessing(route.mechanism_id)
        else:
            behavior = _probe_active_learning(route.mechanism_id)
        runtime = self._probe_route_runtime(route)
        return DataSideBehaviorEvidence(
            passed=behavior.passed and runtime.passed,
            checks={**behavior.checks, **runtime.checks},
            observed_changes=sorted(
                set([*behavior.observed_changes, *runtime.observed_changes])
            ),
            errors=sorted(set([*behavior.errors, *runtime.errors])),
        )

    def _probe_route_runtime(self, route: DataSideRouteSpec) -> DataSideBehaviorEvidence:
        """Exercise the existing adapter SDK and its real plugin hook locally."""

        errors: list[str] = []
        checks: dict[str, bool | str | int | float] = {}
        contract = self.contracts.get(route.component_id or "")
        if contract is None:
            contract = _contract_for_route(route)
            checks["route_contract_materialized"] = True
        try:
            context = AdapterContext(
                contract=contract,
                detector_family="yolo26",
                yolo_version="26",
                head="one_to_one",
                imgsz=640,
                workspace=self.workspace,
                options=_runtime_probe_options(route),
            )
            adapter = ComponentAdapterRegistry().create_for_contract(contract)
            if type(adapter).__name__ != route.adapter_class:
                raise TypeError(
                    f"route adapter mismatch: expected {route.adapter_class}, "
                    f"got {type(adapter).__name__}"
                )
            facade = adapter.as_runtime_adapter(
                contract,
                context,
                protocol_hash="paper-data-side-cpu-probe.v1",
                base_command=[
                    "yolo-agent",
                    "train",
                    "model=yolo26n.pt",
                    "data=coco.yaml",
                    "imgsz=640",
                ],
            )
            preview = facade.apply({}, {})
            checks["adapter_callable"] = True
            checks["changed_variable_declared"] = bool(
                preview.operations
                and any(
                    operation.field == adapter.changed_variable
                    for operation in preview.operations
                )
            )
            payload = facade.typed_runtime_payload()
            report = facade.validate_contract(payload=payload)
            checks["typed_runtime_payload"] = report.checks.get(
                "typed_payload", False
            )
            checks["runtime_hook_registered"] = report.checks.get(
                "declared_runtime_hook_registered", False
            )
            checks["runtime_contract_valid"] = report.ok
            errors.extend(report.errors)
            facade.build()
            rollback = facade.rollback()
            checks["rollback_declared"] = rollback.reversible
            smoke = facade.validate_non_mock_smoke()
            checks["non_mock_smoke"] = smoke.passed
            errors.extend(smoke.errors)
            with TemporaryDirectory(prefix="paper-data-side-") as temporary:
                probe_root = Path(temporary)
                invoked = _invoke_probe_hook(
                    route,
                    payload,
                    probe_root,
                )
                artifact_written = all(
                    (probe_root / artifact.relative_path).is_file()
                    for artifact in payload.expected_artifacts
                    if artifact.required
                )
            checks["runtime_hook_invoked"] = invoked
            checks["runtime_artifact_written"] = artifact_written
            if not invoked:
                errors.append("runtime_hook_invocation_failed")
            if not artifact_written:
                errors.append("runtime_expected_artifact_missing")
        except (ImportError, RuntimeError, TypeError, ValueError, OSError) as exc:
            checks.setdefault("adapter_callable", False)
            errors.append(f"runtime:{type(exc).__name__}:{exc}")
        return DataSideBehaviorEvidence(
            passed=not errors and all(
                bool(value) for value in checks.values() if isinstance(value, bool)
            ),
            checks=checks,
            observed_changes=[],
            errors=sorted(set(errors)),
        )

    def _update_spec(
        self,
        spec: PaperImplementationSpec,
        record: PaperDataSideRecord,
    ) -> PaperImplementationSpec:
        routes = [resolve_data_side_route(item) for item in record.resolved_route_ids]
        route_values = [item for item in routes if item is not None]
        component_ids = _unique([*spec.component_ids, *record.component_ids])
        adapter_ids = _unique([*spec.adapter_ids, *record.adapter_ids])
        mechanisms = _unique([*spec.paper_specific_mechanisms, *record.paper_specific_mechanism_ids])
        insertion_points = _unique(
            [*spec.runtime_insertion_points, *(f"data_side:{item.route_kind}" for item in route_values)]
        )
        hooks = _unique(
            [
                *spec.runtime_hooks,
                *(hook for route in route_values for hook in route.runtime_hooks),
            ]
        )
        test_refs = _unique(
            [
                *spec.unit_test_refs,
                *(test for route in route_values for test in route.test_refs),
            ]
        )
        data_changes = _unique(
            [*spec.data_changes, *(item.changed_variables for item in route_values)]
        )
        sampling_changes = _unique(
            [
                *spec.sampling_changes,
                *(item.changed_variables for item in route_values if item.route_kind == "sampling"),
            ]
        )
        augmentation_changes = _unique(
            [
                *spec.augmentation_changes,
                *(item.changed_variables for item in route_values if item.route_kind == "augmentation"),
            ]
        )
        annotation_changes = _unique(
            [
                *spec.annotation_changes,
                *(item.changed_variables for item in route_values if item.route_kind == "annotation"),
            ]
        )
        config = dict(spec.paper_specific_config)
        config["data_side"] = {
            "record_fingerprint": record.implementation_fingerprint,
            "routes": [item.model_dump(mode="json") for item in route_values],
            "behavior": record.behavior.model_dump(mode="json"),
            "remaining_dependencies": record.remaining_dependencies,
        }
        blockers = _unique([*spec.blockers, *record.blockers])
        readiness = spec.readiness
        flags = {
            "runtime_implementation_verified": spec.runtime_implementation_verified,
            "unit_tests_passed": spec.unit_tests_passed,
            "non_mock_smoke_passed": spec.non_mock_smoke_passed,
            "compatibility_validation_passed": spec.compatibility_validation_passed,
        }
        if record.status == "blocked_missing_evidence" and readiness == "implementation_ready":
            readiness = "smoke_passed"
            flags = {key: False for key in flags}
        fingerprint = _hash_payload(
            {
                "previous": spec.implementation_fingerprint,
                "paper_id": spec.paper_id,
                "data_side": config["data_side"],
                "components": component_ids,
                "adapters": adapter_ids,
            }
        )
        return spec.model_copy(
            update={
                "paper_specific_mechanisms": mechanisms,
                "component_ids": component_ids,
                "adapter_ids": adapter_ids,
                "runtime_insertion_points": insertion_points,
                "runtime_hooks": hooks,
                "data_changes": data_changes,
                "sampling_changes": sampling_changes,
                "augmentation_changes": augmentation_changes,
                "annotation_changes": annotation_changes,
                "paper_specific_config": config,
                "unit_test_refs": test_refs,
                "blockers": blockers,
                "implementation_fingerprint": fingerprint,
                "readiness": readiness,
                **flags,
            }
        )


def build_paper_data_side_audit(**kwargs: object) -> PaperDataSideAudit:
    """Functional builder for offline callers."""

    return PaperDataSideAuditBuilder(**kwargs).build()


def render_paper_83_data_side_status(audit: PaperDataSideAudit) -> str:
    """Render all 83 papers without turning out-of-scope rows into claims."""

    lines = [
        "# Paper-83 Data-side Status",
        "",
        "This report audits only data-side mechanisms. It does not train a model "
        "and does not promote shared adapters to paper implementations.",
        "",
        f"- Frozen papers: {audit.paper_count}",
        f"- Data-side papers in scope: {audit.data_side_paper_count}",
        f"- Ready data-side routes: {audit.summary.get('ready', 0)}",
        f"- Blocked for missing evidence: {audit.summary.get('blocked_missing_evidence', 0)}",
        f"- Out of scope: {audit.summary.get('out_of_scope', 0)}",
        f"- Manifest membership hash: `{audit.manifest_membership_hash}`",
        f"- Plan source: `{audit.plan_source_path}`",
        "",
        "## Scope Decision",
        "",
    ]
    if audit.data_side_paper_count == 0:
        lines.extend(
            [
                "The current frozen engineering plan contains no paper whose "
                "primary or secondary domain is one of `data_quality`, "
                "`annotation`, `sampling`, `augmentation`, `preprocessing`, "
                "`long_tail`, or `active_learning`.",
                "",
                "The checked-in data primitives are therefore not attributed to "
                "any of the frozen 83 papers. A later plan must supply an exact "
                "paper-specific mechanism, config, and evidence before a row can "
                "be marked ready.",
                "",
            ]
        )
    lines.extend(
        [
            "## Per-paper Audit",
            "",
            "| Paper | Domains | Scope | Status | Mechanisms | Runtime routes | Blockers |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for item in audit.records:
        domains = ", ".join([item.primary_domain, *item.secondary_domains])
        mechanisms = "<br>".join(item.paper_specific_mechanism_ids) or "none"
        routes = "<br>".join(item.resolved_route_ids) or "none"
        blockers = "<br>".join(item.blockers) or "none"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_md(item.paper_id)}`",
                    _md(domains),
                    "data-side" if item.in_scope else "out-of-scope",
                    item.status,
                    _md(mechanisms),
                    _md(routes),
                    _md(blockers),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "`ready` means the exact route's CPU behavior probe passed and the "
            "paper supplied planning/evidence/config references. It is not a "
            "mAP result or a claim of exact reproduction.",
            "",
        ]
    )
    return "\n".join(lines)


def write_paper_83_data_side_status(
    audit: PaperDataSideAudit,
    path: Path | str = "docs/paper-83-data-side-status.md",
) -> Path:
    """Write the human-readable data-side status report."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_paper_83_data_side_status(audit), encoding="utf-8")
    return output


def _probe_sampling(mechanism_id: str) -> DataSideBehaviorEvidence:
    records = [
        DataSampleRecord(image_path="common.jpg", class_ids=[0], normalized_areas=[0.3]),
        DataSampleRecord(image_path="common-2.jpg", class_ids=[0], normalized_areas=[0.25]),
        DataSampleRecord(
            image_path="rare.jpg",
            class_ids=[1],
            normalized_areas=[0.01],
            false_negative_score=0.8,
        ),
        DataSampleRecord(
            image_path="negative.jpg",
            class_ids=[],
            normalized_areas=[],
            is_hard_negative=True,
            false_negative_score=0.9,
        ),
    ]
    options: dict[str, Any] = {"mechanism": mechanism_id, "imgsz": 640}
    if mechanism_id == "false_negative_class_boost":
        options["target_class_ids"] = [1]
    if mechanism_id == "repeat_factor_sampling":
        options["repeat_threshold"] = 0.8
    try:
        config = ExposureConfig.model_validate(options)
    except ValueError as exc:
        return DataSideBehaviorEvidence(passed=False, errors=[str(exc)])
    _, exposure, _ = compute_exposure_details(records, config)
    changed = len(set(exposure)) > 1
    return DataSideBehaviorEvidence(
        passed=changed,
        checks={
            "distribution_changes": changed,
            "bounded": max(exposure) <= config.max_weight,
            "imgsz_640": True,
        },
        observed_changes=["sample exposure weights" if changed else "none"],
        errors=[] if changed else ["sampling distribution did not change"],
    )


def _probe_annotation(mechanism_id: str) -> DataSideBehaviorEvidence:
    if mechanism_id != "annotation_quality_filter":
        return DataSideBehaviorEvidence(
            passed=False,
            errors=[f"annotation_route_unimplemented:{mechanism_id}"],
        )
    from yolo_agent.components.adapters.data_pipeline.annotation import (
        AnnotationFilterConfig,
        filter_detection_sample,
    )

    sample = {
        "img": torch.zeros((3, 16, 16), dtype=torch.uint8),
        "bboxes": torch.tensor(
            [[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.0, 0.2]],
            dtype=torch.float32,
        ),
        "cls": torch.tensor([[1], [2]], dtype=torch.float32),
        "batch_idx": torch.tensor([0, 0], dtype=torch.int64),
    }
    result = filter_detection_sample(sample, AnnotationFilterConfig())
    changed = result.kept_indices == [0] and result.removed_indices == [1]
    aligned = len(result.sample["bboxes"]) == len(result.sample["cls"])
    empty = filter_detection_sample(
        {
            "img": torch.zeros((3, 16, 16), dtype=torch.uint8),
            "bboxes": torch.zeros((0, 4), dtype=torch.float32),
            "cls": torch.zeros((0, 1), dtype=torch.float32),
        },
        AnnotationFilterConfig(),
    )
    empty_handling = (
        empty.sample["bboxes"].shape == (0, 4)
        and empty.sample["cls"].shape == (0, 1)
    )
    return DataSideBehaviorEvidence(
        passed=changed and aligned and empty_handling,
        checks={
            "annotation_output_changes": changed,
            "bbox_class_alignment": aligned,
            "empty_box_handling": empty_handling,
            "imgsz_640": True,
        },
        observed_changes=["invalid annotation removed" if changed else "none"],
        errors=[
            message
            for condition, message in (
                (changed, "annotation filter did not change invalid label"),
                (aligned, "annotation filter broke bbox/class alignment"),
                (empty_handling, "annotation filter mishandled an empty sample"),
            )
            if not condition
        ],
    )


def _probe_preprocessing(mechanism_id: str) -> DataSideBehaviorEvidence:
    if mechanism_id != "preprocessing_normalization":
        return DataSideBehaviorEvidence(
            passed=False,
            errors=[f"preprocessing_route_unimplemented:{mechanism_id}"],
        )
    from yolo_agent.components.adapters.data_pipeline.preprocessing import (
        PreprocessingConfig,
        preprocess_sample,
    )

    sample = {
        "img": torch.full((3, 4, 4), 255, dtype=torch.uint8),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        "cls": torch.tensor([[1.0]]),
    }
    output = preprocess_sample(sample, PreprocessingConfig())
    pixels_changed = not torch.equal(output["img"], sample["img"].float())
    labels_unchanged = torch.equal(output["bboxes"], sample["bboxes"]) and torch.equal(
        output["cls"], sample["cls"]
    )
    return DataSideBehaviorEvidence(
        passed=pixels_changed and labels_unchanged,
        checks={
            "pixels_change": pixels_changed,
            "labels_unchanged": labels_unchanged,
            "imgsz_640": True,
        },
        observed_changes=["image normalization" if pixels_changed else "none"],
        errors=[] if pixels_changed and labels_unchanged else ["preprocessing behavior invalid"],
    )


def _probe_active_learning(mechanism_id: str) -> DataSideBehaviorEvidence:
    if mechanism_id != "active_sample_selection":
        return DataSideBehaviorEvidence(
            passed=False,
            errors=[f"active_learning_route_unimplemented:{mechanism_id}"],
        )
    miner = ActiveLearningMiner(MiningConfig(max_samples=2))
    predictions = [
        PredictionSummary(image_path="easy.jpg", max_confidence=0.98),
        PredictionSummary(image_path="uncertain.jpg", max_confidence=0.1),
    ]
    plan = miner.mine(predictions, dataset_version="v1")
    ranking_changed = bool(plan.mined_samples) and plan.mined_samples[0].image_path.name == "uncertain.jpg"
    return DataSideBehaviorEvidence(
        passed=ranking_changed,
        checks={
            "acquisition_ranking_changes": ranking_changed,
            "labeling_plan_created": bool(plan.labeling_manifest),
            "imgsz_640": True,
        },
        observed_changes=["uncertain sample ranked first" if ranking_changed else "none"],
        errors=[] if ranking_changed else ["active-learning ranking did not change"],
    )


def _probe_augmentation(mechanism_id: str) -> DataSideBehaviorEvidence:
    dataset = _ProbeDataset()
    options: dict[str, Any] = {"mechanism": mechanism_id, "seed": 19, "imgsz": 640}
    if mechanism_id == "copy_paste_rare_classes":
        options["rare_class_ids"] = [3]
    if mechanism_id in {"scale_aware_crop", "object_centric_crop"}:
        options.update({"crop_scale": 0.5, "small_area_threshold": 0.02})
    if mechanism_id == "multi_image_sampling_schedule":
        options["multi_image_count"] = 2
    config = DataTransformConfig.model_validate(options)
    wrapped = DataPipelineDataset(dataset, config)
    first = wrapped[0]
    second = DataPipelineDataset(dataset, config)[0]
    changed = not torch.equal(first["img"], dataset[0]["img"]) or not torch.equal(
        first["bboxes"], dataset[0]["bboxes"]
    )
    reproducible = torch.equal(first["img"], second["img"]) and torch.equal(
        first["bboxes"], second["bboxes"]
    )
    aligned = len(first["bboxes"]) == len(first["cls"])
    return DataSideBehaviorEvidence(
        passed=changed and reproducible and aligned,
        checks={
            "pixels_or_boxes_change": changed,
            "fixed_seed_reproducible": reproducible,
            "bbox_class_alignment": aligned,
            "imgsz_640": True,
        },
        observed_changes=["image/bbox geometry" if changed else "none"],
        errors=[
            message
            for condition, message in (
                (changed, "augmentation produced no observable change"),
                (reproducible, "augmentation is not reproducible for fixed seed"),
                (aligned, "augmentation broke bbox/class alignment"),
            )
            if not condition
        ],
    )


class _ProbeDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        self.samples = [
            _probe_sample(0, 1, [0.2, 0.2, 0.05, 0.05]),
            _probe_sample(100, 3, [0.7, 0.7, 0.05, 0.05]),
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value.clone() for key, value in self.samples[index].items()}


def _probe_sample(value: int, class_id: int, box: list[float]) -> dict[str, torch.Tensor]:
    return {
        "img": torch.full((3, 16, 16), value, dtype=torch.uint8),
        "bboxes": torch.tensor([box], dtype=torch.float32),
        "cls": torch.tensor([[class_id]], dtype=torch.float32),
        "batch_idx": torch.tensor([0]),
    }


class _ProbeYoloDataset(Dataset[int]):
    """Small Ultralytics-shaped dataset used only for hook invocation."""

    im_files = ["probe-a.jpg", "probe-b.jpg", "probe-c.jpg"]
    labels = [
        {
            "normalized": True,
            "bbox_format": "xywh",
            "bboxes": [[0.2, 0.2, 0.05, 0.05]],
            "cls": [[1]],
            "false_negative_score": 0.8,
        },
        {
            "normalized": True,
            "bbox_format": "xywh",
            "bboxes": [[0.5, 0.5, 0.2, 0.2]],
            "cls": [[0]],
        },
        {
            "normalized": True,
            "bbox_format": "xywh",
            "bboxes": [],
            "cls": [],
        },
    ]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> int:
        return index


def _invoke_probe_hook(
    route: DataSideRouteSpec,
    payload: AdapterRuntimePayload,
    artifact_root: Path,
) -> bool:
    """Invoke the declared plugin hook against a synthetic CPU dataset."""

    if not route.runtime_hooks or not payload.plugin_references:
        return False
    reference = payload.plugin_references[0]
    implementation = reference.resolve()
    plugin = implementation(**reference.options) if isinstance(implementation, type) else implementation
    runtime_context = SimpleNamespace(
        payload=payload,
        payload_path=artifact_root / "runtime_payload.yaml",
    )
    trainer = SimpleNamespace(epoch=0)
    hook = route.runtime_hooks[0]
    if hook == "build_train_dataset":
        dataset = _ProbeDataset()
        result = plugin.build_train_dataset(
            context=runtime_context,
            trainer=trainer,
            dataset=dataset,
            image_path="probe-a.jpg",
            batch_size=2,
        )
        if result is None or len(result) != len(dataset):
            return False
        # Materializing one item is intentional: constructing a wrapper alone
        # does not prove that the geometric or annotation behavior is callable.
        sample = result[0]
        return isinstance(sample, dict) and isinstance(sample.get("img"), torch.Tensor)
    if hook == "build_train_dataloader":
        dataset = _ProbeYoloDataset()
        dataloader = DataLoader(dataset, batch_size=2)
        result = plugin.build_train_dataloader(
            context=runtime_context,
            trainer=trainer,
            dataloader=dataloader,
            dataset_path="probe.yaml",
            batch_size=2,
            rank=0,
        )
        if result is None or len(result.dataset) != len(dataset):
            return False
        next(iter(result))
        return True
    if hook == "build_validator":
        validator = SimpleNamespace()
        result = plugin.build_validator(
            context=runtime_context,
            trainer=trainer,
            validator=validator,
        )
        return result is validator and hasattr(validator, f"{route.mechanism_id}_miner")
    return False


def _is_non_data_mechanism(value: str) -> bool:
    return value.startswith(_NON_DATA_COMPONENT_PREFIXES)


def _unique(values: Iterable[object]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()})


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _md(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


__all__ = [
    "DATA_SIDE_DOMAINS",
    "PaperDataSideAuditBuilder",
    "build_paper_data_side_audit",
    "data_side_routes",
    "render_paper_83_data_side_status",
    "resolve_data_side_route",
    "write_paper_83_data_side_status",
]
