"""Centralized resource paths for the YOLO Agent harness."""

from __future__ import annotations

from pathlib import Path


class ResourcePaths:
    """Shared filesystem paths for bundled configs and templates."""

    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
    CONFIG_DIR: Path = PROJECT_ROOT / "configs"

    COMPONENTS_DIR: Path = CONFIG_DIR / "components"
    DATASETS_DIR: Path = CONFIG_DIR / "datasets"
    TRAINING_DIR: Path = CONFIG_DIR / "training"
    COCO_DATASET: Path = DATASETS_DIR / "coco.yaml"
    YOLO26_COCO_GOAL: Path = TRAINING_DIR / "yolo26_coco_goal.yaml"
    SEARCH_SPACE: Path = CONFIG_DIR / "search_space.yaml"
    LOOP_POLICY: Path = CONFIG_DIR / "loop_policy.yaml"
    ANNOTATION_RULES: Path = CONFIG_DIR / "annotation_rules.yaml"
    AUGMENTATION_POLICIES: Path = CONFIG_DIR / "augmentation_policies.yaml"
    COMPATIBILITY_RULES: Path = CONFIG_DIR / "compatibility_rules.yaml"
    ERROR_ACTION_POLICIES: Path = CONFIG_DIR / "error_action_policies.yaml"
    OPTIMIZATION_RECIPES: Path = CONFIG_DIR / "optimization_recipes.yaml"
    POSTPROCESS_STRATEGIES: Path = CONFIG_DIR / "postprocess_strategies.yaml"
    TRAINING_FAILURE_MODES: Path = CONFIG_DIR / "training_failure_modes.yaml"
    TEMPLATES_DIR: Path = CONFIG_DIR / "templates"
    ULTRALYTICS_BASE_TEMPLATE: Path = TEMPLATES_DIR / "ultralytics_base.yaml"
    SCENARIOS_DIR: Path = CONFIG_DIR / "scenarios"


__all__ = [
    "ResourcePaths",
]
