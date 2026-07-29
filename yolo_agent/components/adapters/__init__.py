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
from yolo_agent.components.adapters.runtime import AdapterRuntimePayload, RuntimePluginReference
from yolo_agent.components.adapters.neck import (
    GoldGatherDistributeAdapter,
    MultiScaleFusionAdapter,
    RTMDetLargeKernelNeckAdapter,
)

__all__ = [
    "AdapterContext",
    "AdapterValidationReport",
    "AdapterRuntimePayload",
    "ComponentAdapter",
    "ComponentAdapterRegistry",
    "DummyAdapter",
    "ExpectedArtifact",
    "PatchOperation",
    "PatchPreview",
    "RollbackPlan",
    "RuntimePluginReference",
    "GoldGatherDistributeAdapter",
    "MultiScaleFusionAdapter",
    "RTMDetLargeKernelNeckAdapter",
    "SmokeTestResult",
    "WeightLoadResult",
]
