"""Ultralytics adapter scaffold."""

from yolo_agent.adapters.ultralytics.yaml_generator import (
    UltralyticsYamlGenerator,
    YamlGenerationResult,
    generate_ultralytics_yaml,
)

__all__ = ["UltralyticsYamlGenerator", "YamlGenerationResult", "generate_ultralytics_yaml"]
