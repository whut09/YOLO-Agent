"""Content-addressed frozen research snapshots for replayable decisions."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

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


class ResearchMaturitySummary(BaseModel):
    """Audited component maturity counts frozen with the research inputs."""

    metadata_only: int = Field(default=0, ge=0)
    adapter_implemented: int = Field(default=0, ge=0)
    smoke_passed: int = Field(default=0, ge=0)
    pilot_reproduced: int = Field(default=0, ge=0)


class ResearchSnapshot(BaseModel, YAMLModelMixin):
    """Frozen Paper Intelligence inputs used by every optimization round."""

    schema_version: str = "research_snapshot.v3"
    snapshot_hash: str
    paper_intelligence: Literal["available", "unavailable"] = "available"
    unavailable_reason: str | None = None
    papers_version: str
    component_registry_version: str
    recipe_registry_version: str
    source_repository: str | None = None
    source_commit: str | None = None
    source_catalog_hash: str | None = None
    importer_version: str | None = None
    classifications_version: str
    extractions_version: str
    compatibility_version: str
    reproduction_queue_version: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    paper_count: int = Field(default=0, ge=0)
    component_count: int = Field(default=0, ge=0)
    recipe_count: int = Field(default=0, ge=0)
    maturity_summary: ResearchMaturitySummary = Field(default_factory=ResearchMaturitySummary)
    artifacts: dict[str, ResearchSnapshotArtifact] = Field(default_factory=dict)
    frozen: bool = True

    @model_validator(mode="after")
    def validate_snapshot_hash(self) -> "ResearchSnapshot":
        expected = research_snapshot_hash(self.version_payload())
        if self.snapshot_hash != expected:
            raise ValueError(f"research snapshot hash mismatch: expected {expected}, got {self.snapshot_hash}")
        return self

    def version_payload(self) -> dict[str, Any]:
        payload = {
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
        if self.schema_version != "research_snapshot.v1":
            payload.update({
                "paper_intelligence": self.paper_intelligence,
                "unavailable_reason": self.unavailable_reason,
                "maturity_summary": self.maturity_summary.model_dump(mode="json"),
            })
        if self.schema_version == "research_snapshot.v3":
            payload.update({
                "source_repository": self.source_repository,
                "source_commit": self.source_commit,
                "source_catalog_hash": self.source_catalog_hash,
                "importer_version": self.importer_version,
            })
        return payload

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


class ResearchRuntimeBinding(BaseModel):
    """The immutable research context attached to one training run/round."""

    research_snapshot_hash: str
    research_snapshot_path: str | None = None
    research_snapshot_verified: bool = False
    paper_intelligence: Literal["available", "unavailable"] = "unavailable"
    unavailable_reason: str | None = None
    maturity_summary: ResearchMaturitySummary = Field(default_factory=ResearchMaturitySummary)
    research_network_allowed: bool = False


UNAVAILABLE_RESEARCH_SNAPSHOT_HASH = hashlib.sha256(
    json.dumps(
        {"paper_intelligence": "unavailable", "schema_version": "research_runtime_unavailable.v1"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def bind_research_snapshot(
    research_root: Path | str,
    *,
    expected_hash: str | None = None,
    snapshot_path: Path | str | None = None,
) -> ResearchRuntimeBinding:
    """Resolve one frozen snapshot; never fall back to the live registry."""
    resolved = load_research_snapshot(research_root, snapshot_path)
    if resolved is None:
        if expected_hash not in {None, "", "none", UNAVAILABLE_RESEARCH_SNAPSHOT_HASH}:
            raise ValueError(f"bound research snapshot is unavailable: {expected_hash}")
        return ResearchRuntimeBinding(
            research_snapshot_hash=UNAVAILABLE_RESEARCH_SNAPSHOT_HASH,
            unavailable_reason="snapshot_missing",
        )
    snapshot, directory = resolved
    if expected_hash not in {None, "", "none", snapshot.snapshot_hash}:
        raise ValueError(f"bound research snapshot changed: expected {expected_hash}, got {snapshot.snapshot_hash}")
    return ResearchRuntimeBinding(
        research_snapshot_hash=snapshot.snapshot_hash,
        research_snapshot_path=directory.resolve().as_posix(),
        research_snapshot_verified=True,
        paper_intelligence=snapshot.paper_intelligence,
        unavailable_reason=snapshot.unavailable_reason,
        maturity_summary=snapshot.maturity_summary,
        research_network_allowed=False,
    )


def freeze_research_snapshot(
    research_root: Path | str,
    artifacts: dict[str, Path | str],
    *,
    paper_count: int,
    component_count: int,
    recipe_count: int,
    papers_version: str | None = None,
    maturity_summary: ResearchMaturitySummary | dict[str, int] | None = None,
    source_repository: str | None = None,
    source_commit: str | None = None,
    source_catalog_hash: str | None = None,
    importer_version: str | None = None,
    unavailable_reason_override: str | None = None,
) -> tuple[ResearchSnapshot, Path]:
    """Copy production artifacts into an immutable content-addressed directory."""
    root = Path(research_root)
    materialized = {name: Path(path) for name, path in artifacts.items()}
    missing = [f"{name}:{path}" for name, path in materialized.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing research snapshot inputs: " + ", ".join(missing))
    versions = {name: sha256_file(path) for name, path in materialized.items()}
    maturity = ResearchMaturitySummary.model_validate(maturity_summary or {})
    paper_intelligence = "unavailable" if unavailable_reason_override else "available" if paper_count > 0 else "unavailable"
    unavailable_reason = unavailable_reason_override or (None if paper_count > 0 else "empty_registry")
    semantic_papers_version = papers_version or versions["papers"]
    payload = {
        "schema_version": "research_snapshot.v3",
        "paper_intelligence": paper_intelligence,
        "unavailable_reason": unavailable_reason,
        "papers_version": semantic_papers_version,
        "component_registry_version": versions["component_contracts"],
        "recipe_registry_version": versions["recipes"],
        "source_repository": source_repository,
        "source_commit": source_commit,
        "source_catalog_hash": source_catalog_hash,
        "importer_version": importer_version,
        "classifications_version": versions["classifications"],
        "extractions_version": versions["component_extractions"],
        "compatibility_version": versions["compatibility_reviews"],
        "reproduction_queue_version": versions["reproduction_queue"],
        "paper_count": paper_count,
        "component_count": component_count,
        "recipe_count": recipe_count,
        "maturity_summary": maturity.model_dump(mode="json"),
    }
    snapshot_hash = research_snapshot_hash(payload)
    snapshot_dir = root / "snapshots" / snapshot_hash
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest = snapshot_dir / "snapshot.yaml"
    if existing_manifest.is_file():
        existing = ResearchSnapshot.from_snapshot_dir(snapshot_dir)
        if not existing.verify(snapshot_dir):
            _write_latest_pointer(root, existing, snapshot_dir)
            return existing, snapshot_dir
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
        paper_intelligence=paper_intelligence,
        unavailable_reason=unavailable_reason,
        papers_version=semantic_papers_version,
        component_registry_version=versions["component_contracts"],
        recipe_registry_version=versions["recipes"],
        source_repository=source_repository,
        source_commit=source_commit,
        source_catalog_hash=source_catalog_hash,
        importer_version=importer_version,
        classifications_version=versions["classifications"],
        extractions_version=versions["component_extractions"],
        compatibility_version=versions["compatibility_reviews"],
        reproduction_queue_version=versions["reproduction_queue"],
        paper_count=paper_count,
        component_count=component_count,
        recipe_count=recipe_count,
        maturity_summary=maturity,
        artifacts=frozen_artifacts,
    )
    snapshot.to_yaml(snapshot_dir / "snapshot.yaml", exclude_none=True, sort_keys=False)
    _write_latest_pointer(root, snapshot, snapshot_dir)
    return snapshot, snapshot_dir


def _write_latest_pointer(root: Path, snapshot: ResearchSnapshot, snapshot_dir: Path) -> None:
    _atomic_write_yaml(
        root / "latest_snapshot.yaml",
        {
            "schema_version": "research_snapshot_pointer.v1",
            "snapshot_hash": snapshot.snapshot_hash,
            "snapshot_path": snapshot_dir.resolve().as_posix(),
            "generated_at": snapshot.generated_at.isoformat(),
        },
    )


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
    "ResearchRuntimeBinding",
    "ResearchMaturitySummary",
    "ResearchSnapshot",
    "ResearchSnapshotArtifact",
    "freeze_research_snapshot",
    "load_research_snapshot",
    "bind_research_snapshot",
    "UNAVAILABLE_RESEARCH_SNAPSHOT_HASH",
    "research_snapshot_hash",
]
