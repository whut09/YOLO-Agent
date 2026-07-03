"""Experimental adapter boundary for Ultralytics YOLO integration.

@experimental
This module provides unverified interfaces for model YAML generation, loss
adapter lookup, training command planning, and forward-checking. Real training
execution should be validated through the SmokeRunner forward checks before
use in production.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any

from yolo_agent.adapters.ultralytics.loss_adapter import default_loss_registry
from yolo_agent.adapters.ultralytics.yaml_generator import YamlGenerationResult, generate_ultralytics_yaml


class UltralyticsAdapter:
    """Experimental adapter boundary for Ultralytics YOLO workflows."""

    name: str = "ultralytics"

    def __init__(self) -> None:
        self._loss_registry = default_loss_registry()

    def is_available(self) -> bool:
        """Return whether the external integration is available."""
        return _import_ultralytics() is not None

    def generate_model_yaml(
        self,
        candidate: Any,
        base_template: Path | str | None = None,
        output_dir: Path | str | None = None,
        nc: int | None = None,
        scales: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> YamlGenerationResult:
        """Generate a model YAML draft for a candidate.

        @experimental
        """
        if base_template is None:
            from yolo_agent.tools.smoke_runner import default_ultralytics_template_path
            base_template = default_ultralytics_template_path()
        if output_dir is None:
            output_dir = "generated_models"
        return generate_ultralytics_yaml(
            candidate=candidate,
            base_template=base_template,
            output_dir=output_dir,
            nc=nc,
            scales=scales,
            dry_run=dry_run,
        )

    def available_losses(self) -> list[str]:
        """Return registered loss adapter names."""
        return self._loss_registry.names()

    def get_loss(self, name: str = "ciou") -> Any:
        """Instantiate a registered loss adapter by name."""
        return self._loss_registry.get(name)

    def build_train_command(
        self,
        node: Any,
        command: str | None = None,
        model_yaml_path: Path | str | None = None,
    ) -> str:
        """Build a ``yolo train`` subprocess command.

        @experimental
        Real training command assembly requires verified CLI argument mapping.
        """
        if model_yaml_path is None:
            model_yaml_path = _default_model_yaml_path(node)
        if command is not None:
            return command
        return f"yolo train model={Path(model_yaml_path).as_posix()}"

    def smoke_check(self, model_yaml_path: Path | str) -> bool:
        """Try importing ultralytics and loading the generated model YAML.

        @experimental
        """
        ultralytics = _import_ultralytics()
        if ultralytics is None:
            return False
        try:
            yolo_cls = getattr(ultralytics, "YOLO")
            model = yolo_cls(str(Path(model_yaml_path)))
            if hasattr(model, "info"):
                model.info()
            return True
        except Exception:
            return False


def _import_ultralytics() -> Any | None:
    try:
        return importlib.import_module("ultralytics")
    except ImportError:
        return None


def _default_model_yaml_path(node: Any) -> Path:
    candidates_dir = Path("generated_models")
    candidate_id = getattr(getattr(node, "candidate_config", None), "candidate_id", "candidate")
    return candidates_dir / f"{candidate_id}.yaml"
