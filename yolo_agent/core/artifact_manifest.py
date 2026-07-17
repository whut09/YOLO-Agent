"""Artifact manifest records with content hashes."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_serializer


ARTIFACT_MANIFEST_SCHEMA_VERSION = "1.0"
ArtifactType = Literal["file", "directory"]


class ArtifactManifestEntry(BaseModel):
    """One artifact manifest entry."""

    name: str
    type: ArtifactType
    path: Path
    sha256: str
    producer_stage: str
    run_id: str | None = None
    candidate_id: str | None = None
    node_id: str | None = None
    protocol_hash: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = ARTIFACT_MANIFEST_SCHEMA_VERSION

    @field_serializer("path")
    def serialize_path(self, value: Path) -> str:
        """Serialize paths portably."""
        return value.as_posix()

    @classmethod
    def from_path(
        cls,
        name: str,
        path: Path | str,
        producer_stage: str,
        *,
        run_id: str | None = None,
        candidate_id: str | None = None,
        node_id: str | None = None,
        protocol_hash: str | None = None,
    ) -> "ArtifactManifestEntry":
        """Create a manifest entry from a file or directory path."""
        artifact_path = Path(path)
        if artifact_path.is_file():
            artifact_type: ArtifactType = "file"
            digest = sha256_file(artifact_path)
        elif artifact_path.is_dir():
            artifact_type = "directory"
            digest = sha256_directory(artifact_path)
        else:
            raise FileNotFoundError(f"Artifact does not exist: {artifact_path}")
        return cls(
            name=name,
            type=artifact_type,
            path=artifact_path,
            sha256=digest,
            producer_stage=producer_stage,
            run_id=run_id,
            candidate_id=candidate_id,
            node_id=node_id,
            protocol_hash=protocol_hash,
        )

    def verify(self) -> bool:
        """Return whether the artifact still matches the recorded hash."""
        if self.type == "file":
            return self.path.is_file() and sha256_file(self.path) == self.sha256
        if self.type == "directory":
            return self.path.is_dir() and sha256_directory(self.path) == self.sha256
        return False


class ArtifactManifest:
    """Append-only JSONL artifact manifest."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def append(self, entry: ArtifactManifestEntry) -> ArtifactManifestEntry:
        """Append one manifest entry."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry.model_dump(mode="json"), sort_keys=True) + "\n")
        return entry

    def append_many(self, entries: list[ArtifactManifestEntry]) -> Path:
        """Append many manifest entries."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            for entry in entries:
                file.write(json.dumps(entry.model_dump(mode="json"), sort_keys=True) + "\n")
        return self.path

    def read(self) -> list[ArtifactManifestEntry]:
        """Read all manifest entries."""
        if not self.path.exists():
            return []
        entries: list[ArtifactManifestEntry] = []
        with self.path.open("r", encoding="utf-8-sig") as file:
            for line in file:
                text = line.strip()
                if text:
                    entries.append(ArtifactManifestEntry.model_validate(json.loads(text)))
        return entries


def sha256_file(path: Path | str) -> str:
    """Return SHA-256 for one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path | str) -> str:
    """Return a stable SHA-256 for a directory tree."""
    root = Path(path)
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = item.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()
