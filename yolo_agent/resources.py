"""Centralized resource paths for the YOLO Agent harness."""

from __future__ import annotations

from pathlib import Path


class ResourcePaths:
    """Shared filesystem paths for bundled configs and templates."""

    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
    CONFIG_DIR: Path = PROJECT_ROOT / "configs"
    PRESETS_DIR: Path = PROJECT_ROOT / "presets"

    COMPONENTS_DIR: Path = CONFIG_DIR / "components"
    COMPONENT_TAXONOMY: Path = CONFIG_DIR / "component_taxonomy.yaml"
    COMPONENT_COMPATIBILITY: Path = CONFIG_DIR / "component_compatibility.yaml"
    YOLO26_COMPATIBILITY: Path = CONFIG_DIR / "yolo26_compatibility.yaml"
    RECIPE_BUNDLES: Path = CONFIG_DIR / "recipe_bundles.yaml"
    PAPER_SOURCES: Path = CONFIG_DIR / "paper_sources.yaml"
    RESEARCH_PRIORITY: Path = CONFIG_DIR / "research_priority.yaml"
    REPRODUCTION_POLICY: Path = CONFIG_DIR / "reproduction_policy.yaml"
    DATASETS_DIR: Path = CONFIG_DIR / "datasets"
    TRAINING_DIR: Path = CONFIG_DIR / "training"
    YOLO26_NATIVE_RECIPE: Path = TRAINING_DIR / "yolo26_native_recipe.yaml"
    COCO_DATASET: Path = DATASETS_DIR / "coco.yaml"
    YOLO26_COCO_GOAL: Path = TRAINING_DIR / "yolo26_coco_goal.yaml"
    SEARCH_SPACE: Path = CONFIG_DIR / "search_space.yaml"
    LOOP_POLICY: Path = CONFIG_DIR / "loop_policy.yaml"
    ANNOTATION_RULES: Path = CONFIG_DIR / "annotation_rules.yaml"
    AUGMENTATION_POLICIES: Path = CONFIG_DIR / "augmentation_policies.yaml"
    COMPATIBILITY_RULES: Path = CONFIG_DIR / "compatibility_rules.yaml"
    ERROR_ACTION_POLICIES: Path = CONFIG_DIR / "error_action_policies.yaml"
    DIAGNOSIS_GRAPH: Path = CONFIG_DIR / "diagnosis_graph.yaml"
    UTILITY_POLICY: Path = CONFIG_DIR / "utility_policy.yaml"
    LLM_DECISION_EXAMPLE: Path = CONFIG_DIR / "llm_decision.example.yaml"
    LLM_DECISION_LOCAL: Path = CONFIG_DIR / "local" / "llm_decision.local.yaml"
    OPTIMIZATION_RECIPES: Path = CONFIG_DIR / "optimization_recipes.yaml"
    POSTPROCESS_STRATEGIES: Path = CONFIG_DIR / "postprocess_strategies.yaml"
    TRAINING_FAILURE_MODES: Path = CONFIG_DIR / "training_failure_modes.yaml"
    TRAINING_RECIPES: Path = CONFIG_DIR / "training_recipes.yaml"
    TEMPLATES_DIR: Path = CONFIG_DIR / "templates"
    ULTRALYTICS_BASE_TEMPLATE: Path = TEMPLATES_DIR / "ultralytics_base.yaml"
    SCENARIOS_DIR: Path = CONFIG_DIR / "scenarios"
    COCO_YOLO26_AUTO_PRESET: Path = PRESETS_DIR / "coco_yolo26_auto.yaml"


__all__ = [
    "ResourcePaths",
]
