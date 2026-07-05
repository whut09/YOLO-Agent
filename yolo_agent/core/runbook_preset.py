"""Runbook preset schema and resolution helpers."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_serializer, model_validator

from yolo_agent.adapters.ultralytics.training import TrainingBudgetProfileName
from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.resources import ResourcePaths


class RunbookPreset(BaseModel, YAMLModelMixin):
    """User-facing preset that hides harness wiring paths behind a named runbook."""

    name: str
    description: str = ""
    kind: str = "coco"
    default_model: str = "yolo26n.pt"
    default_goal: str = "+2map"
    default_profile: TrainingBudgetProfileName = "debug"
    allowed_profiles: list[TrainingBudgetProfileName] = Field(
        default_factory=lambda: ["debug", "pilot", "baseline_full", "baseline_confirm", "candidate_full"]
    )
    training_config: Path = ResourcePaths.YOLO26_COCO_GOAL
    loop_policy: Path = ResourcePaths.LOOP_POLICY
    components: Path = ResourcePaths.COMPONENTS_DIR
    search_space: Path = ResourcePaths.SEARCH_SPACE
    dataset_manifest_mode: str = "metadata"
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_default_profile(self) -> "RunbookPreset":
        """Ensure the default profile is selectable."""
        if self.default_profile not in self.allowed_profiles:
            raise ValueError("default_profile must be included in allowed_profiles")
        return self

    @field_serializer("training_config", "loop_policy", "components", "search_space")
    def serialize_path(self, value: Path) -> str:
        """Serialize preset paths as portable strings."""
        return value.as_posix()

    @classmethod
    def from_yaml(cls, path: Path | str) -> "RunbookPreset":
        """Load a preset and resolve relative paths against the project root."""
        preset_path = Path(path)
        preset = super().from_yaml(preset_path)
        return preset.resolve_paths(ResourcePaths.PROJECT_ROOT)

    def resolve_paths(self, root: Path | str = ResourcePaths.PROJECT_ROOT) -> "RunbookPreset":
        """Return a copy with relative paths resolved against a root directory."""
        root_path = Path(root)
        return self.model_copy(
            update={
                "training_config": _resolve_path(root_path, self.training_config),
                "loop_policy": _resolve_path(root_path, self.loop_policy),
                "components": _resolve_path(root_path, self.components),
                "search_space": _resolve_path(root_path, self.search_space),
            }
        )

    def require_profile(self, profile: TrainingBudgetProfileName) -> None:
        """Raise if the profile is not allowed by this preset."""
        if profile not in self.allowed_profiles:
            allowed = ", ".join(self.allowed_profiles)
            raise ValueError(f"profile {profile!r} is not allowed by preset {self.name!r}; allowed: {allowed}")


def load_runbook_preset(path: Path | str | None = None) -> RunbookPreset:
    """Load a runbook preset, defaulting to the bundled COCO YOLO26 auto preset."""
    return RunbookPreset.from_yaml(path or ResourcePaths.COCO_YOLO26_AUTO_PRESET)


def _resolve_path(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value
