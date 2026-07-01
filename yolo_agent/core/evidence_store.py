"""Local filesystem evidence store for experiment runs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from yolo_agent.core.experiment_graph import Evidence


class EvidenceStore:
    """Persist run configuration, metrics, and artifact paths locally."""

    def __init__(self, root: Path | str = "runs") -> None:
        self.root = Path(root)

    def create_run(self, run_id: str) -> Path:
        """Create the local run directory structure."""
        run_dir = self._run_dir(run_id)
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        return run_dir

    def log_config(self, run_id: str, config: dict[str, Any]) -> Path:
        """Write run configuration to config.yaml."""
        run_dir = self.create_run(run_id)
        config_path = run_dir / "config.yaml"
        with config_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, sort_keys=False)
        return config_path

    def log_metrics(self, run_id: str, metrics: dict[str, float | int | str | bool | None]) -> Path:
        """Write run metrics to metrics.json."""
        run_dir = self.create_run(run_id)
        metrics_path = run_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2, sort_keys=True)
        return metrics_path

    def log_artifact(self, run_id: str, artifact_path: Path | str, name: str | None = None) -> Path:
        """Copy an artifact into runs/{run_id}/artifacts/."""
        run_dir = self.create_run(run_id)
        source = Path(artifact_path)
        if not source.is_file():
            raise FileNotFoundError(f"Artifact does not exist or is not a file: {source}")
        destination = run_dir / "artifacts" / (name or source.name)
        shutil.copy2(source, destination)
        return destination

    def load_run(self, run_id: str) -> Evidence:
        """Load config, metrics, and artifact paths for a run."""
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run does not exist: {run_id}")

        config_path = run_dir / "config.yaml"
        metrics_path = run_dir / "metrics.json"
        artifacts_dir = run_dir / "artifacts"
        config = _read_yaml_mapping(config_path) if config_path.exists() else {}
        metrics = _read_json_mapping(metrics_path) if metrics_path.exists() else {}
        artifacts = {
            path.name: path
            for path in sorted(artifacts_dir.iterdir())
            if path.is_file()
        } if artifacts_dir.exists() else {}

        return Evidence(
            run_id=run_id,
            config_path=config_path if config_path.exists() else None,
            metrics_path=metrics_path if metrics_path.exists() else None,
            artifacts_dir=artifacts_dir if artifacts_dir.exists() else None,
            config=config,
            metrics=metrics,
            artifacts=artifacts,
        )

    def _run_dir(self, run_id: str) -> Path:
        if not run_id or any(separator in run_id for separator in ("/", "\\")):
            raise ValueError("run_id must be a non-empty single path segment.")
        return self.root / run_id


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def _read_json_mapping(path: Path) -> dict[str, float | int | str | bool | None]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain a mapping: {path}")
    return data

