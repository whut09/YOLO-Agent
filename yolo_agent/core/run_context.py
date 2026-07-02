"""Run context for orchestrated YOLO Agent loops."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_serializer


class RunContext(BaseModel):
    """Stable paths and identifiers for one harness run."""

    run_id: str
    run_root: Path = Path("runs")
    task_path: Path
    data_yaml: Path
    component_path: Path = Path("configs/components")
    search_space_path: Path = Path("configs/search_space.yaml")
    loop_policy_path: Path = Path("configs/loop_policy.yaml")
    predictions_path: Path | None = None
    detection_errors_path: Path | None = None
    metrics_input_path: Path | None = None
    dataset_version: str = "unversioned"
    dataset_root: Path | None = None
    dataset_version_store_path: Path | None = None
    dataset_manifest_path: Path | None = None
    dataset_manifest_sha256: str | None = None
    seed: int = 42
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

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
        output_path = Path(path) if path is not None else self.run_dir / "run_context.yaml"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(self.model_dump(mode="json"), file, sort_keys=False)
        return output_path

    def to_json(self, path: Path | str | None = None) -> Path:
        """Write context JSON."""
        output_path = Path(path) if path is not None else self.run_dir / "run_context.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def from_yaml(cls, path: Path | str) -> "RunContext":
        """Load run context YAML."""
        input_path = Path(path)
        with input_path.open("r", encoding="utf-8-sig") as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Run context YAML must contain a mapping: {input_path}")
        return cls.model_validate(data)

    @classmethod
    def from_run_dir(cls, run_dir: Path | str) -> "RunContext":
        """Load run context from a run directory."""
        return cls.from_yaml(Path(run_dir) / "run_context.yaml")
