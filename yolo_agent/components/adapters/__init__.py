"""Component Adapter SDK for controlled detector extensions."""

from yolo_agent.components.adapters.base import (
    AdapterContext,
    AdapterValidationReport,
    ComponentAdapter,
    ExpectedArtifact,
    PatchOperation,
    PatchPreview,
    RollbackPlan,
    SmokeTestResult,
    WeightLoadResult,
)
from yolo_agent.components.adapters.dummy import DummyAdapter
from yolo_agent.components.adapters.registry import ComponentAdapterRegistry

__all__ = [
    "AdapterContext",
    "AdapterValidationReport",
    "ComponentAdapter",
    "ComponentAdapterRegistry",
    "DummyAdapter",
    "ExpectedArtifact",
    "PatchOperation",
    "PatchPreview",
    "RollbackPlan",
    "SmokeTestResult",
    "WeightLoadResult",
]
