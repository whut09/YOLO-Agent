"""Run context for orchestrated YOLO Agent loops."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_serializer

from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.resources import ResourcePaths


class RunContext(BaseModel, YAMLModelMixin):
    """Stable paths and identifiers for one harness run."""

    run_id: str
    run_root: Path = Path("runs")
    task_path: Path
    data_yaml: Path
    component_path: Path = ResourcePaths.COMPONENTS_DIR
    search_space_path: Path = ResourcePaths.SEARCH_SPACE
    loop_policy_path: Path = ResourcePaths.LOOP_POLICY
    predictions_path: Path | None = None
    detection_errors_path: Path | None = None
    metrics_input_path: Path | None = None
    dataset_version: str = "unversioned"
    dataset_root: Path | None = None
    dataset_version_store_path: Path | None = None
    dataset_manifest_path: Path | None = None
    dataset_manifest_sha256: str | None = None
    seed: int = 42
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        """Return run directory."""
        return self.run_root / self.run_id

    @property
    def artifacts_dir(self) -> Path:
        """Return artifact directory."""
        return self.run_dir / "artifacts"

    @field_serializer(
        "run_root",
        "task_path",
        "data_yaml",
        "component_path",
        "search_space_path",
        "loop_policy_path",
        "predictions_path",
        "detection_errors_path",
        "metrics_input_path",
        "dataset_root",
        "dataset_version_store_path",
        "dataset_manifest_path",
    )
    def serialize_path(self, value: Path | None) -> str | None:
        """Serialize paths portably."""
        return value.as_posix() if value is not None else None

    def ensure_dirs(self) -> None:
        """Create run directories."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def artifact_path(self, name: str) -> Path:
        """Return a path under artifacts."""
        return self.artifacts_dir / name

    def to_yaml(self, path: Path | str | None = None) -> Path:
        """Write context YAML."""
        target = self.run_dir / "run_context.yaml" if path is None else Path(path)
        return super().to_yaml(target)

    def to_json(self, path: Path | str | None = None) -> Path:
        """Write context JSON."""
        target = self.run_dir / "run_context.json" if path is None else Path(path)
        return super().to_json(target)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "RunContext":
        """Load run context YAML."""
        loaded = super().from_yaml(path)
        # Keep RunContext-specific validation if needed later.
        return loaded

    @classmethod
    def from_run_dir(cls, run_dir: Path | str) -> "RunContext":
        """Load run context from a run directory."""
        return cls.from_yaml(Path(run_dir) / "run_context.yaml")
