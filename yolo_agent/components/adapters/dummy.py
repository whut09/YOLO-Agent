"""Deterministic adapter used by SDK tests and examples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from yolo_agent.components.adapters.base import (
    AdapterContext,
    AdapterValidationReport,
    ComponentAdapter,
    ExpectedArtifact,
    RollbackPlan,
    SmokeTestResult,
    WeightLoadResult,
)


class DummyAdapter(ComponentAdapter):
    """A local-only adapter that adds an explicit marker to training config."""

    adapter_version = "dummy.v1"
    source_commit = "local-test"
    strategy = "callback"
    modified_model_fields = frozenset()
    modified_training_fields = frozenset({"adapter_marker"})

    def validate_environment(self, context: AdapterContext) -> AdapterValidationReport:
        return AdapterValidationReport(ok=True, checks={"workspace": str(context.workspace)})

    def validate_compatibility(self, context: AdapterContext) -> AdapterValidationReport:
        if context.imgsz != 640:
            return AdapterValidationReport(ok=False, errors=["DummyAdapter requires imgsz=640"])
        return AdapterValidationReport(ok=True, checks={"fixed_imgsz": True})

    def patch_model_config(self, config: dict[str, Any], context: AdapterContext, *, dry_run: bool = True) -> dict[str, Any]:
        return config

    def patch_training_config(self, config: dict[str, Any], context: AdapterContext, *, dry_run: bool = True) -> dict[str, Any]:
        config["adapter_marker"] = context.contract.component_id
        return config

    def build_module(self, context: AdapterContext) -> dict[str, str]:
        return {"component_id": context.contract.component_id}

    def load_pretrained_weights(self, module: Any, weights: Path | str | None, context: AdapterContext) -> WeightLoadResult:
        return WeightLoadResult(loaded=False, source=Path(weights) if weights else None, message="No dummy weights required")

    def smoke_test(self, context: AdapterContext) -> SmokeTestResult:
        return SmokeTestResult(passed=True, checks={"local_only": True})

    def expected_artifacts(self, context: AdapterContext) -> list[ExpectedArtifact]:
        return [ExpectedArtifact(name="adapter_patch", relative_path=Path("adapter_patch.yaml"))]

    def rollback_plan(self, context: AdapterContext) -> RollbackPlan:
        return RollbackPlan(actions=["discard generated adapter patch"], files_to_remove=[Path("adapter_patch.yaml")])
