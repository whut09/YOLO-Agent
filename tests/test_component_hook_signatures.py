from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.components.adapters import (
    AdapterRuntimePayload,
    RollbackPlan,
    RuntimePluginReference,
)
from yolo_agent.components.adapters.validation import validate_runtime_plugin_hooks


class ValidHookPlugin:
    plugin_version = "valid.v1"

    def build_model(self, *, context: object, trainer: object, model: object) -> object:
        return model


class MissingModelArgumentPlugin:
    plugin_version = "invalid.v1"

    def build_model(self, *, context: object, trainer: object) -> object:
        return trainer


def _payload(reference: str) -> AdapterRuntimePayload:
    return AdapterRuntimePayload(
        component_ids=["test.component"],
        adapter_classes=["TestAdapter"],
        adapter_versions={"test.component": "v1"},
        source_commits={"test.component": "commit"},
        trainer_plugin=[
            RuntimePluginReference(reference=reference, required_hooks=["build_model"])
        ],
        changed_variables={"training.test_component": True},
        rollback_plan=RollbackPlan(actions=["discard"]),
        protocol_hash="protocol-1",
        base_command=["yolo", "detect", "train", "imgsz=640"],
    )


def test_runtime_hook_signature_contract_accepts_complete_hook() -> None:
    result = validate_runtime_plugin_hooks(
        _payload(f"{__name__}:ValidHookPlugin")
    )

    assert result["runtime_plugins_loaded"] == 1
    assert result["runtime_hook_signatures_verified"] == 1


def test_runtime_hook_signature_contract_rejects_missing_transform_argument() -> None:
    with pytest.raises(ValueError, match="signature mismatch.*missing=model"):
        validate_runtime_plugin_hooks(
            _payload(f"{__name__}:MissingModelArgumentPlugin")
        )


def test_validation_bridge_runtime_report_records_signature_check(
    tmp_path: Path,
) -> None:
    payload = _payload(f"{__name__}:ValidHookPlugin")
    path = payload.write(tmp_path / "payload.yaml")

    restored = AdapterRuntimePayload.read(path)
    checks = validate_runtime_plugin_hooks(restored)
    assert checks["runtime_hook_signatures_verified"] == 1
