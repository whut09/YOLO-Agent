"""Content-addressed frozen research snapshots for replayable decisions."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from yolo_agent.core.artifact_manifest import sha256_file
from yolo_agent.core.yaml_io import YAMLModelMixin


class ResearchSnapshotArtifact(BaseModel):
    """One immutable file included in a research snapshot."""

    name: str
    path: str
    sha256: str
    size_bytes: int = Field(ge=0)


class ResearchSnapshot(BaseModel, YAMLModelMixin):
    """Frozen Paper Intelligence inputs used by every optimization round."""

    schema_version: str = "research_snapshot.v1"
    snapshot_hash: str
    papers_version: str
    component_registry_version: str
    recipe_registry_version: str
    classifications_version: str
    extractions_version: str
    compatibility_version: str
    reproduction_queue_version: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    paper_count: int = Field(default=0, ge=0)
    component_count: int = Field(default=0, ge=0)
    recipe_count: int = Field(default=0, ge=0)
    artifacts: dict[str, ResearchSnapshotArtifact] = Field(default_factory=dict)
    frozen: bool = True

    @model_validator(mode="after")
    def validate_snapshot_hash(self) -> "ResearchSnapshot":
        expected = research_snapshot_hash(self.version_payload())
        if self.snapshot_hash != expected:
            raise ValueError(f"research snapshot hash mismatch: expected {expected}, got {self.snapshot_hash}")
        return self

    def version_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "papers_version": self.papers_version,
            "component_registry_version": self.component_registry_version,
            "recipe_registry_version": self.recipe_registry_version,
            "classifications_version": self.classifications_version,
            "extractions_version": self.extractions_version,
            "compatibility_version": self.compatibility_version,
            "reproduction_queue_version": self.reproduction_queue_version,
            "paper_count": self.paper_count,
            "component_count": self.component_count,
            "recipe_count": self.recipe_count,
        }

    def verify(self, snapshot_dir: Path | str) -> list[str]:
        """Return integrity failures without mutating the frozen snapshot."""
        root = Path(snapshot_dir)
        failures: list[str] = []
        for name, artifact in self.artifacts.items():
            path = root / artifact.path
            if not path.is_file():
                failures.append(f"missing_snapshot_artifact:{name}:{path}")
                continue
            if path.stat().st_size != artifact.size_bytes:
                failures.append(f"snapshot_size_mismatch:{name}")
            actual = sha256_file(path)
            if actual != artifact.sha256:
                failures.append(f"snapshot_sha256_mismatch:{name}:{actual}")
        return failures

    @classmethod
    def from_snapshot_dir(cls, snapshot_dir: Path | str) -> "ResearchSnapshot":
        return cls.from_yaml(Path(snapshot_dir) / "snapshot.yaml")


def freeze_research_snapshot(
    research_root: Path | str,
    artifacts: dict[str, Path | str],
    *,
    paper_count: int,
    component_count: int,
    recipe_count: int,
) -> tuple[ResearchSnapshot, Path]:
    """Copy production artifacts into an immutable content-addressed directory."""
    root = Path(research_root)
    materialized = {name: Path(path) for name, path in artifacts.items()}
    missing = [f"{name}:{path}" for name, path in materialized.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing research snapshot inputs: " + ", ".join(missing))
    versions = {name: sha256_file(path) for name, path in materialized.items()}
    payload = {
        "schema_version": "research_snapshot.v1",
        "papers_version": versions["papers"],
        "component_registry_version": versions["component_contracts"],
        "recipe_registry_version": versions["recipes"],
        "classifications_version": versions["classifications"],
        "extractions_version": versions["component_extractions"],
        "compatibility_version": versions["compatibility_reviews"],
        "reproduction_queue_version": versions["reproduction_queue"],
        "paper_count": paper_count,
        "component_count": component_count,
        "recipe_count": recipe_count,
    }
    snapshot_hash = research_snapshot_hash(payload)
    snapshot_dir = root / "snapshots" / snapshot_hash
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    frozen_artifacts: dict[str, ResearchSnapshotArtifact] = {}
    for name, source in sorted(materialized.items()):
        suffix = "".join(source.suffixes) or ".dat"
        target_name = f"{name}{suffix}"
        target = snapshot_dir / target_name
        if not target.is_file() or sha256_file(target) != versions[name]:
            _atomic_copy(source, target)
        frozen_artifacts[name] = ResearchSnapshotArtifact(
            name=name,
            path=target_name,
            sha256=versions[name],
            size_bytes=target.stat().st_size,
        )
    snapshot = ResearchSnapshot(
        snapshot_hash=snapshot_hash,
        papers_version=versions["papers"],
        component_registry_version=versions["component_contracts"],
        recipe_registry_version=versions["recipes"],
        classifications_version=versions["classifications"],
        extractions_version=versions["component_extractions"],
        compatibility_version=versions["compatibility_reviews"],
        reproduction_queue_version=versions["reproduction_queue"],
        paper_count=paper_count,
        component_count=component_count,
        recipe_count=recipe_count,
        artifacts=frozen_artifacts,
    )
    snapshot.to_yaml(snapshot_dir / "snapshot.yaml", exclude_none=True, sort_keys=False)
    _atomic_write_yaml(
        root / "latest_snapshot.yaml",
        {
            "schema_version": "research_snapshot_pointer.v1",
            "snapshot_hash": snapshot.snapshot_hash,
            "snapshot_path": snapshot_dir.resolve().as_posix(),
            "generated_at": snapshot.generated_at.isoformat(),
        },
    )
    return snapshot, snapshot_dir


def load_research_snapshot(
    research_root: Path | str,
    snapshot: Path | str | None = None,
) -> tuple[ResearchSnapshot, Path] | None:
    """Load an explicit snapshot or the latest frozen snapshot pointer."""
    root = Path(research_root)
    if snapshot is not None:
        path = Path(snapshot)
        snapshot_dir = path if path.is_dir() else path.parent
    else:
        pointer = root / "latest_snapshot.yaml"
        if not pointer.is_file():
            return None
        raw = yaml.safe_load(pointer.read_text(encoding="utf-8-sig")) or {}
        path = Path(str(raw.get("snapshot_path") or ""))
        snapshot_dir = path if path.is_absolute() else root / path
    manifest = ResearchSnapshot.from_snapshot_dir(snapshot_dir)
    failures = manifest.verify(snapshot_dir)
    if failures:
        raise ValueError("invalid frozen research snapshot: " + "; ".join(failures))
    return manifest, snapshot_dir


def research_snapshot_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


__all__ = [
    "ResearchSnapshot",
    "ResearchSnapshotArtifact",
    "freeze_research_snapshot",
    "load_research_snapshot",
    "research_snapshot_hash",
]
