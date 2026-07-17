"""Bridge component contracts and adapters into executable experiment nodes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from yolo_agent.components.adapters import (
    AdapterContext,
    ComponentAdapterRegistry,
    PatchPreview,
    RollbackPlan,
    SmokeTestResult,
)
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.yolo26_compatibility import (
    YOLO26CompatibilityChecker,
    YOLO26CompatibilityResult,
)
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.recipes.schemas import RecipeSpec
from yolo_agent.core.yaml_io import YAMLModelMixin


BridgeStatus = Literal["executable", "adapter_required", "blocked"]


class AdapterExecutionRecord(BaseModel):
    """Auditable adapter preparation result for one component."""

    component_id: str
    adapter_class: str
    adapter_version: str
    source_commit: str
    patch_hash: str
    changed_variables: dict[str, Any] = Field(default_factory=dict)
    rollback_plan: RollbackPlan
    smoke_test: SmokeTestResult
    patch_preview_path: Path


class ComponentExecutionResult(BaseModel, YAMLModelMixin):
    """Result of converting one component recipe into an executable node."""

    schema_version: str = "component_execution_bridge.v1"
    status: BridgeStatus
    node: ExperimentNode
    recipe_id: str
    component_ids: list[str] = Field(default_factory=list)
    compatibility: YOLO26CompatibilityResult | None = None
    adapters: list[AdapterExecutionRecord] = Field(default_factory=list)
    aggregate_patch_hash: str | None = None
    changed_variables: dict[str, Any] = Field(default_factory=dict)
    blocked_by: list[str] = Field(default_factory=list)
    evidence_path: Path | None = None


class ComponentExecutionBridge:
    """Apply executable adapters and bind their provenance to a training node."""

    def __init__(
        self,
        *,
        adapter_registry: ComponentAdapterRegistry | None = None,
        compatibility_checker: YOLO26CompatibilityChecker | None = None,
    ) -> None:
        self.adapter_registry = adapter_registry or ComponentAdapterRegistry()
        self.compatibility_checker = compatibility_checker or YOLO26CompatibilityChecker()

    def prepare(
        self,
        *,
        recipe: RecipeSpec,
        node: ExperimentNode,
        contracts: dict[str, ComponentContract],
        model_config: dict[str, Any] | None = None,
        training_config: dict[str, Any] | None = None,
        workspace: Path | str,
        evidence_store: EvidenceStore | None = None,
        run_id: str | None = None,
        protocol_hash: str | None = None,
        dry_run: bool = True,
    ) -> ComponentExecutionResult:
        """Validate, preview, smoke-test, and attach adapters to ``node``."""
        workdir = Path(workspace)
        workdir.mkdir(parents=True, exist_ok=True)
        selected: list[ComponentContract] = []
        blocked: list[str] = []
        for component_id in recipe.component_ids:
            contract = contracts.get(component_id)
            if contract is None:
                blocked.append(f"missing_component_contract:{component_id}")
                continue
            if not contract.can_execute:
                blocked.append(f"component_maturity_below_smoke_passed:{component_id}:{contract.maturity}")
                continue
            selected.append(contract)
        if blocked:
            return ComponentExecutionResult(
                status="adapter_required",
                node=node,
                recipe_id=recipe.recipe_id,
                component_ids=list(recipe.component_ids),
                blocked_by=blocked,
            )

        compatibility = self.compatibility_checker.check(
            components=selected,
            train_overrides={
                key: value
                for key, value in recipe.train_overrides.items()
                if key in {"imgsz", "amp", "ddp", "head_mode", "allow_imgsz_increase"}
            },
            changed_variables=None,
            single_variable=len(recipe.component_ids) <= 1 and not recipe.coupled_variables,
            checkpoint=node.candidate_config.base_model,
            amp=_optional_bool(recipe.train_overrides.get("amp")),
            execution_requested=True,
        )
        if not compatibility.compatible:
            return ComponentExecutionResult(
                status="blocked",
                node=node,
                recipe_id=recipe.recipe_id,
                component_ids=list(recipe.component_ids),
                compatibility=compatibility,
                blocked_by=list(compatibility.blocked_by),
            )

        current_model = dict(model_config or {"model": node.candidate_config.base_model})
        current_training = dict(training_config or recipe.train_overrides)
        records: list[AdapterExecutionRecord] = []
        changed: dict[str, Any] = {}
        for contract in selected:
            try:
                contract.assert_executable(detector_family="yolo26", imgsz=640)
                adapter = self.adapter_registry.create_for_contract(contract)
                context = AdapterContext(
                    contract=contract,
                    detector_family="yolo26",
                    head="one_to_one",
                    imgsz=640,
                    workspace=workdir,
                    options=dict(recipe.train_overrides),
                )
                preview = adapter.prepare_patch(
                    current_model,
                    current_training,
                    context,
                    dry_run=dry_run,
                )
                smoke = adapter.smoke_test(context)
                if not smoke.passed:
                    blocked.extend(f"adapter_smoke_failed:{contract.component_id}:{error}" for error in smoke.errors or ["unknown"])
                    continue
                preview_path = _write_patch_preview(workdir, node.node_id, preview)
                operation_values = {
                    f"{item.target}.{item.field}": item.after
                    for item in preview.operations
                }
                changed.update(operation_values)
                records.append(
                    AdapterExecutionRecord(
                        component_id=contract.component_id,
                        adapter_class=preview.adapter_class,
                        adapter_version=preview.adapter_version,
                        source_commit=preview.source_commit,
                        patch_hash=preview.idempotency_key,
                        changed_variables=operation_values,
                        rollback_plan=preview.rollback,
                        smoke_test=smoke,
                        patch_preview_path=preview_path,
                    )
                )
                current_model = preview.patched_model_config
                current_training = preview.patched_training_config
            except (ImportError, KeyError, TypeError, ValueError) as exc:
                blocked.append(f"adapter_prepare_failed:{contract.component_id}:{exc}")

        if blocked or len(records) != len(selected):
            return ComponentExecutionResult(
                status="blocked",
                node=node,
                recipe_id=recipe.recipe_id,
                component_ids=list(recipe.component_ids),
                compatibility=compatibility,
                adapters=records,
                changed_variables=changed,
                blocked_by=blocked or ["adapter_preparation_incomplete"],
            )
        if node.command_spec is None or node.command_spec.command_type != "train":
            return ComponentExecutionResult(
                status="blocked",
                node=node,
                recipe_id=recipe.recipe_id,
                component_ids=list(recipe.component_ids),
                compatibility=compatibility,
                adapters=records,
                changed_variables=changed,
                blocked_by=["training_command_spec_missing"],
            )

        aggregate_hash = _aggregate_patch_hash(records)
        evidence_path = workdir / "component_execution.yaml"
        metadata = {
            **node.command_spec.metadata,
            "component_bridge_schema": "component_execution_bridge.v1",
            "component_recipe_id": recipe.recipe_id,
            "component_ids": ",".join(recipe.component_ids),
            "adapter_versions": json.dumps({item.component_id: item.adapter_version for item in records}, sort_keys=True),
            "adapter_source_commits": json.dumps({item.component_id: item.source_commit for item in records}, sort_keys=True),
            "adapter_patch_hash": aggregate_hash,
            "adapter_changed_variables": json.dumps(changed, sort_keys=True, default=str),
            "adapter_rollback_plan": json.dumps(
                {item.component_id: item.rollback_plan.model_dump(mode="json") for item in records},
                sort_keys=True,
            ),
            "adapter_evidence_path": evidence_path.as_posix(),
            "adapter_guard_metrics": "latency_ms,model_size_mb",
            "matched_pilot_required": True,
        }
        expected_artifacts = dict(node.command_spec.expected_artifacts)
        expected_artifacts["component_execution"] = evidence_path
        node.command_spec = node.command_spec.model_copy(
            update={"metadata": metadata, "expected_artifacts": expected_artifacts}
        )
        node.command = node.command_spec.display()
        node.changed_variables = {**node.changed_variables, **changed}
        node.effective_overrides = {**node.effective_overrides, **current_training}
        result = ComponentExecutionResult(
            status="executable",
            node=node,
            recipe_id=recipe.recipe_id,
            component_ids=list(recipe.component_ids),
            compatibility=compatibility,
            adapters=records,
            aggregate_patch_hash=aggregate_hash,
            changed_variables=changed,
            evidence_path=evidence_path,
        )
        result.to_yaml(evidence_path, exclude_none=True, sort_keys=False)
        if evidence_store is not None and run_id is not None:
            evidence_store.log_artifact_manifest(
                run_id=run_id,
                name=f"{node.node_id}_component_execution",
                artifact_path=evidence_path,
                producer_stage="component_execution_bridge",
                candidate_id=node.candidate_config.candidate_id,
                node_id=node.node_id,
                protocol_hash=protocol_hash,
            )
            evidence_store.log_candidate_metrics(
                run_id=run_id,
                candidate_id=node.candidate_config.candidate_id,
                node_id=node.node_id,
                metrics={
                    "adapter_smoke_passed": True,
                    "adapter_patch_hash": aggregate_hash,
                    "adapter_versions": metadata["adapter_versions"],
                    "adapter_changed_variables": metadata["adapter_changed_variables"],
                },
                dataset_version=node.data_version,
                split="runtime",
                source="component_execution_bridge",
                verified=True,
                validator="component_execution_bridge",
                source_artifact=evidence_path,
                seed=node.seed,
                protocol_hash=protocol_hash,
            )
        return result


def _write_patch_preview(workspace: Path, node_id: str, preview: PatchPreview) -> Path:
    path = workspace / f"{node_id}_{preview.component_id.replace('.', '_')}_patch.yaml"
    preview.to_yaml(path, exclude_none=True, sort_keys=False)
    return path


def _aggregate_patch_hash(records: list[AdapterExecutionRecord]) -> str:
    payload = [(item.component_id, item.patch_hash) for item in records]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


__all__ = ["AdapterExecutionRecord", "ComponentExecutionBridge", "ComponentExecutionResult"]
