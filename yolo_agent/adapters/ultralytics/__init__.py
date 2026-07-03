"""Ultralytics adapter scaffold."""

from yolo_agent.adapters.ultralytics.adapter import UltralyticsAdapter
from yolo_agent.adapters.ultralytics.yaml_generator import (
    UltralyticsYamlGenerator,
    YamlGenerationResult,
    generate_ultralytics_yaml,
)
from yolo_agent.adapters.ultralytics.loss_adapter import (
    BBoxLossAdapter,
    CIoULossAdapter,
    LossRegistry,
    MPDIoULossAdapter,
    NWDLossAdapter,
    WIoULossAdapter,
    default_loss_registry,
)

__all__ = [
    "BBoxLossAdapter",
    "CIoULossAdapter",
    "LossRegistry",
    "MPDIoULossAdapter",
    "NWDLossAdapter",
    "UltralyticsAdapter",
    "UltralyticsYamlGenerator",
    "WIoULossAdapter",
    "YamlGenerationResult",
    "default_loss_registry",
    "generate_ultralytics_yaml",
]
