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
from yolo_agent.adapters.ultralytics.training import (
    TrainingBudgetProfile,
    TrainingBudgetProfileName,
    UltralyticsRunImporter,
    UltralyticsTrainingConfig,
    Yolo26CocoGoal,
    command_from_training_config,
    default_training_budget_profiles,
    parse_results_csv,
    parse_ultralytics_run,
)

__all__ = [
    "BBoxLossAdapter",
    "CIoULossAdapter",
    "LossRegistry",
    "MPDIoULossAdapter",
    "NWDLossAdapter",
    "TrainingBudgetProfile",
    "TrainingBudgetProfileName",
    "UltralyticsAdapter",
    "UltralyticsRunImporter",
    "UltralyticsTrainingConfig",
    "UltralyticsYamlGenerator",
    "WIoULossAdapter",
    "YamlGenerationResult",
    "Yolo26CocoGoal",
    "command_from_training_config",
    "default_loss_registry",
    "default_training_budget_profiles",
    "generate_ultralytics_yaml",
    "parse_results_csv",
    "parse_ultralytics_run",
]
