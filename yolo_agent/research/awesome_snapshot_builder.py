"""Offline Awesome catalog import and frozen ResearchSnapshot production."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from yolo_agent.research.awesome_catalog_importer import (
    IMPORTER_VERSION,
    AwesomeCatalogImporter,
    PaperImportResult,
)
from yolo_agent.research.paper_registry import PaperRegistry
from yolo_agent.research.production_pipeline import ResearchProductionPipeline, ResearchProductionResult
from yolo_agent.research.provenance import assert_research_production_allowed
from yolo_agent.resources import ResourcePaths


SOURCE_NAME = "awesome_object_detection"


class ResearchSourceEntry(BaseModel):
    kind: Literal["awesome_catalog"]
    importer_version: str = IMPORTER_VERSION
    manifest_path: Path
    network_allowed_during_training: bool = False


class ResearchSourcesConfig(BaseModel):
    schema_version: str = "research_sources.v1"
    sources: dict[str, ResearchSourceEntry] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path | str = ResourcePaths.RESEARCH_SOURCES) -> "ResearchSourcesConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig")) or {}
        return cls.model_validate(payload)


class AwesomeSourceManifest(BaseModel):
    schema_version: str = "awesome_source_manifest.v1"
    source_name: str = SOURCE_NAME
    source_repository: str
    source_commit: str
    source_catalog_hash: str
    catalog_path: str
    importer_version: str
    catalog_record_count: int = Field(default=0, ge=0)
    imported_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AwesomeSnapshotBuildResult(BaseModel):
    status: Literal["completed", "dry_run", "failed"]
    source_name: str = SOURCE_NAME
    import_result: PaperImportResult | None = None
    production_result: ResearchProductionResult | None = None
    source_manifest_path: str | None = None
    snapshot_hash: str | None = None
    snapshot_path: str | None = None
    paper_intelligence: str = "unavailable"
    unavailable_reason: str | None = None
    errors: list[str] = Field(default_factory=list)


class AwesomeSnapshotBuilder:
    """Run Awesome catalog import followed by the existing offline production pipeline."""

    def __init__(
        self,
        research_root: Path | str = "research",
        *,
        config_path: Path | str = ResourcePaths.RESEARCH_SOURCES,
        analyzer: Any | None = None,
    ) -> None:
        self.root = Path(research_root)
        self.config_path = Path(config_path)
        self.config = ResearchSourcesConfig.from_yaml(self.config_path)
        self.analyzer = analyzer

    def import_catalog(
        self,
        source: Path | str,
        *,
        dry_run: bool = False,
        source_commit: str | None = None,
    ) -> PaperImportResult:
        assert_research_production_allowed()
        entry = self._source_entry(SOURCE_NAME)
        registry = PaperRegistry(self.root)
        result = AwesomeCatalogImporter(registry, importer_version=entry.importer_version).import_source(
            source,
            dry_run=dry_run,
            source_commit=source_commit,
        )
        if not dry_run:
            manifest = AwesomeSourceManifest(
                source_repository=result.source_repository,
                source_commit=result.source_commit,
                source_catalog_hash=result.catalog_hash,
                catalog_path=result.source_path,
                importer_version=entry.importer_version,
                catalog_record_count=result.catalog_record_count,
            )
            _atomic_write_yaml(self._manifest_path(entry), manifest.model_dump(mode="json"))
        return result

    def build(
        self,
        *,
        source_name: str = SOURCE_NAME,
        source: Path | str | None = None,
        source_commit: str | None = None,
        force: bool = False,
    ) -> AwesomeSnapshotBuildResult:
        assert_research_production_allowed()
        entry = self._source_entry(source_name)
        manifest_path = self._manifest_path(entry)
        try:
            existing_manifest = self._load_manifest(manifest_path) if source is None else None
            source_path = Path(source) if source is not None else Path(existing_manifest.catalog_path)
            effective_commit = source_commit or (existing_manifest.source_commit if existing_manifest else None)
            imported = self.import_catalog(source_path, source_commit=effective_commit)
            manifest = self._load_manifest(manifest_path)
            production = ResearchProductionPipeline(self.root, analyzer=self.analyzer).run(
                force=force,
                snapshot_source={
                    "source_repository": manifest.source_repository,
                    "source_commit": manifest.source_commit,
                    "source_catalog_hash": manifest.source_catalog_hash,
                    "importer_version": manifest.importer_version,
                },
                unavailable_reason_override="empty_catalog" if imported.catalog_record_count == 0 else None,
            )
            status: Literal["completed", "dry_run", "failed"] = (
                "completed" if production.status == "completed" else "failed"
            )
            return AwesomeSnapshotBuildResult(
                status=status,
                import_result=imported,
                production_result=production,
                source_manifest_path=manifest_path.resolve().as_posix(),
                snapshot_hash=production.snapshot_hash,
                snapshot_path=production.snapshot_path,
                paper_intelligence=production.paper_intelligence,
                unavailable_reason=production.unavailable_reason,
                errors=list(production.errors),
            )
        except Exception as exc:
            return AwesomeSnapshotBuildResult(status="failed", errors=[str(exc)])

    def _source_entry(self, source_name: str) -> ResearchSourceEntry:
        entry = self.config.sources.get(source_name)
        if entry is None:
            raise ValueError(f"unknown research source: {source_name}")
        if entry.network_allowed_during_training:
            raise ValueError(f"research source permits training-time networking: {source_name}")
        return entry

    def _manifest_path(self, entry: ResearchSourceEntry) -> Path:
        return self.root / entry.manifest_path

    @staticmethod
    def _load_manifest(path: Path) -> AwesomeSourceManifest:
        if not path.is_file():
            raise FileNotFoundError(
                f"Awesome source manifest is missing: {path}. Run yolo-agent research import-awesome first."
            )
        payload = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
        return AwesomeSourceManifest.model_validate(payload)


def _atomic_write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            yaml.safe_dump(payload, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


__all__ = [
    "AwesomeSnapshotBuildResult",
    "AwesomeSnapshotBuilder",
    "AwesomeSourceManifest",
    "ResearchSourceEntry",
    "ResearchSourcesConfig",
]
