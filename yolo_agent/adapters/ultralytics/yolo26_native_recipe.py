"""Canonical native YOLO26 recipe loader and audit facade."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from yolo_agent.adapters.ultralytics.yolo26_native_audit import YOLO26NativeAuditor, YOLO26NativeRecipeAudit
from yolo_agent.resources import ResourcePaths


class YOLO26NativeRecipe(BaseModel):
    """Project-side recipe contract; values are audited, not assumed native."""

    recipe_id: str = "yolo26_native"
    version: str = "v1"
    model_family: str = "yolo26"
    model: str = "yolo26n.pt"
    imgsz: int = 640
    fixed_variables: dict[str, object] = Field(default_factory=lambda: {"imgsz": 640, "end2end": True, "reg_max": 1})
    training: dict[str, object] = Field(default_factory=dict)
    audit_notes: list[str] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "YOLO26NativeRecipe":
        source = Path(path) if path is not None else ResourcePaths.YOLO26_NATIVE_RECIPE
        with source.open("r", encoding="utf-8-sig") as file:
            raw = yaml.safe_load(file) or {}
        return cls.model_validate(raw.get("recipe", raw))

    def audit(self, *, config_path: Path | str, model_path: Path | str | None = None) -> YOLO26NativeRecipeAudit:
        return YOLO26NativeAuditor().audit(config_path, model_path=model_path)


__all__ = ["YOLO26NativeRecipe"]
