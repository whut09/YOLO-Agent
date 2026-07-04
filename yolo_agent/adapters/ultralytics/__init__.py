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
from yolo_agent.adapters.ultralytics.runtime_profiler import (
    RuntimeProfile,
    RuntimeProfiler,
    RuntimeSampler,
    RuntimeSample,
    sample_nvidia_smi,
    write_runtime_profile,
)
from yolo_agent.adapters.ultralytics.batch_tuner import (
    BatchTuner,
    BatchTuningConfig,
    BatchTuningResult,
    BatchTrialResult,
    apply_selected_batch,
    build_batch_trial_command,
    should_tune_batch,
)

__all__ = [
    "BatchTuner",
    "BatchTuningConfig",
    "BatchTuningResult",
    "BatchTrialResult",
    "BBoxLossAdapter",
    "CIoULossAdapter",
    "LossRegistry",
    "MPDIoULossAdapter",
    "NWDLossAdapter",
    "TrainingBudgetProfile",
    "TrainingBudgetProfileName",
    "RuntimeProfile",
    "RuntimeProfiler",
    "RuntimeSampler",
    "RuntimeSample",
    "UltralyticsAdapter",
    "UltralyticsRunImporter",
    "UltralyticsTrainingConfig",
    "UltralyticsYamlGenerator",
    "WIoULossAdapter",
    "YamlGenerationResult",
    "Yolo26CocoGoal",
    "apply_selected_batch",
    "build_batch_trial_command",
    "command_from_training_config",
    "default_loss_registry",
    "default_training_budget_profiles",
    "generate_ultralytics_yaml",
    "parse_results_csv",
    "parse_ultralytics_run",
    "sample_nvidia_smi",
    "should_tune_batch",
    "write_runtime_profile",
]
