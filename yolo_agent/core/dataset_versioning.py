"""Lightweight dataset versioning with manifests and diffs."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


class DatasetFileRecord(BaseModel):
    """One file tracked in a dataset version."""

    path: str
    sha256: str
    size_bytes: int


class DatasetVersionManifest(BaseModel):
    """Dataset version manifest."""

    version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_root: Path
    files: list[DatasetFileRecord] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def to_json(self, path: Path | str) -> None:
        """Write manifest JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: Path | str) -> "DatasetVersionManifest":
        """Load manifest JSON."""
        with Path(path).open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError(f"Dataset manifest must contain a mapping: {path}")
        return cls.model_validate(data)


class DatasetDiff(BaseModel):
    """File-level diff between dataset versions."""

    from_version: str
    to_version: str
    added: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    modified: list[str] = Field(default_factory=list)

    def to_json(self, path: Path | str) -> None:
        """Write diff JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )


class DatasetVersionStore:
    """Local filesystem dataset version store."""

    def __init__(self, root: Path | str = "dataset_versions") -> None:
        self.root = Path(root)

    def create_version(
        self,
        dataset_root: Path | str,
        version: str,
        notes: list[str] | None = None,
        copy_data: bool = False,
    ) -> DatasetVersionManifest:
        """Create a dataset version manifest, optionally copying data."""
        source_root = Path(dataset_root)
        if not source_root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {source_root}")
        version_dir = self._version_dir(version)
        version_dir.mkdir(parents=True, exist_ok=True)

        manifest = DatasetVersionManifest(
            version=version,
            source_root=source_root,
            files=_scan_files(source_root),
            notes=notes or [],
        )
        manifest.to_json(version_dir / "manifest.json")
        if copy_data:
            data_dir = version_dir / "data"
            if data_dir.exists():
                shutil.rmtree(data_dir)
            shutil.copytree(source_root, data_dir)
        return manifest

    def load_version(self, version: str) -> DatasetVersionManifest:
        """Load a dataset version manifest."""
        return DatasetVersionManifest.from_json(self._version_dir(version) / "manifest.json")

    def diff_versions(self, from_version: str, to_version: str) -> DatasetDiff:
        """Diff two dataset versions and write diff JSON under the target version."""
        before = self.load_version(from_version)
        after = self.load_version(to_version)
        diff = diff_manifests(before, after)
        diff.to_json(self._version_dir(to_version) / f"diff_from_{from_version}.json")
        return diff

    def _version_dir(self, version: str) -> Path:
        if not version or any(separator in version for separator in ("/", "\\")):
            raise ValueError("version must be a non-empty single path segment.")
        return self.root / version


def diff_manifests(
    before: DatasetVersionManifest,
    after: DatasetVersionManifest,
) -> DatasetDiff:
    """Compute file-level added/removed/modified diff."""
    before_by_path = {record.path: record for record in before.files}
    after_by_path = {record.path: record for record in after.files}
    before_paths = set(before_by_path)
    after_paths = set(after_by_path)
    common = before_paths & after_paths
    return DatasetDiff(
        from_version=before.version,
        to_version=after.version,
        added=sorted(after_paths - before_paths),
        removed=sorted(before_paths - after_paths),
        modified=sorted(
            path
            for path in common
            if before_by_path[path].sha256 != after_by_path[path].sha256
            or before_by_path[path].size_bytes != after_by_path[path].size_bytes
        ),
    )


def _scan_files(root: Path) -> list[DatasetFileRecord]:
    records: list[DatasetFileRecord] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        records.append(
            DatasetFileRecord(
                path=relative,
                sha256=_sha256(path),
                size_bytes=path.stat().st_size,
            )
        )
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

