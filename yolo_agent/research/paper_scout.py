"""Incremental metadata synchronization for the local paper registry."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.research.paper_registry import PaperRegistry
from yolo_agent.research.provenance import assert_research_production_allowed
from yolo_agent.research.paper_sources import (
    ArxivSourceAdapter,
    CachedHttpClient,
    CrossrefSourceAdapter,
    PaperSourceAdapter,
    SemanticScholarSourceAdapter,
)
from yolo_agent.resources import ResourcePaths


DEFAULT_TOPICS = [
    "object detection",
    "real-time object detection",
    "small object detection",
    "object detection assignment matching",
    "bounding box regression",
    "object detection knowledge distillation",
    "object detection augmentation",
    "object detection domain adaptation",
    "open-vocabulary detection",
    "object detection inference acceleration",
]


class PaperSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    optional: bool = False
    page_size: int = Field(default=100, ge=1, le=1000)
    rate_limit_seconds: float = Field(default=0.0, ge=0.0)


class PaperScoutConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "paper_sources.v1"
    year_from: int = Field(default=2020, ge=1900, le=2100)
    max_pages_per_query: int = Field(default=25, ge=1, le=1000)
    retries: int = Field(default=2, ge=0, le=5)
    retry_backoff_seconds: float = Field(default=1.0, ge=0.0)
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    topics: list[str] = Field(default_factory=lambda: list(DEFAULT_TOPICS))
    sources: dict[str, PaperSourceConfig] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "PaperScoutConfig":
        config_path = Path(path) if path is not None else ResourcePaths.PAPER_SOURCES
        data = yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
        return cls.model_validate(data)


class SourceCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    query: str
    cursor: str | None = None
    last_sync: datetime | None = None
    completed: bool = False


class PaperScoutState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "paper_scout_state.v1"
    last_sync: datetime | None = None
    checkpoints: dict[str, SourceCheckpoint] = Field(default_factory=dict)


class PaperScoutResult(BaseModel):
    sources_attempted: int = 0
    queries_attempted: int = 0
    pages_fetched: int = 0
    records_seen: int = 0
    records_normalized: int = 0
    registry_writes: int = 0
    dry_run: bool = False
    errors: list[str] = Field(default_factory=list)


class PaperScout:
    """Synchronize configured metadata sources into PaperRegistry."""

    def __init__(
        self,
        registry: PaperRegistry,
        *,
        config: PaperScoutConfig | None = None,
        adapters: dict[str, PaperSourceAdapter] | None = None,
        client: CachedHttpClient | None = None,
    ) -> None:
        self.registry = registry
        self.config = config or PaperScoutConfig.from_yaml()
        self.state_path = registry.root / "paper_scout_state.json"
        self.cache_dir = registry.root / "cache" / "paper_sources"
        self.client = client or CachedHttpClient(
            self.cache_dir,
            retries=self.config.retries,
            backoff_seconds=self.config.retry_backoff_seconds,
            timeout_seconds=self.config.timeout_seconds,
        )
        self.adapters = adapters or self._default_adapters()

    def sync(
        self,
        *,
        since: datetime | None = None,
        year_from: int | None = None,
        dry_run: bool = False,
    ) -> PaperScoutResult:
        assert_research_production_allowed()
        state = self.load_state()
        result = PaperScoutResult(dry_run=dry_run)
        minimum_year = year_from or self.config.year_from
        effective_since = since or state.last_sync
        for source_name, source_config in self.config.sources.items():
            if not source_config.enabled:
                continue
            adapter = self.adapters.get(source_name)
            if adapter is None:
                if not source_config.optional:
                    result.errors.append(f"source_not_available:{source_name}")
                continue
            result.sources_attempted += 1
            for query in self.config.topics:
                result.queries_attempted += 1
                key = f"{source_name}:{query}"
                checkpoint = state.checkpoints.get(key) or SourceCheckpoint(source=source_name, query=query)
                cursor = None if checkpoint.completed else checkpoint.cursor
                for _ in range(self.config.max_pages_per_query):
                    try:
                        url = adapter.search(query, since=effective_since, year_from=minimum_year, cursor=cursor)
                        page = adapter.fetch(url, self.client)
                        adapter.rate_limit()
                        result.pages_fetched += 1
                        result.records_seen += len(page.records)
                        normalized = []
                        for record in page.records:
                            paper = adapter.normalize(record)
                            if paper.year < minimum_year:
                                continue
                            normalized.append(paper)
                        result.records_normalized += len(normalized)
                        if not dry_run:
                            self.registry.upsert_many(normalized)
                            result.registry_writes += len(normalized)
                        cursor = adapter.checkpoint(page)
                        checkpoint = checkpoint.model_copy(update={
                            "cursor": cursor,
                            "last_sync": datetime.now(timezone.utc),
                            "completed": page.done,
                        })
                        state.checkpoints[key] = checkpoint
                        if not dry_run:
                            self.save_state(state)
                        if page.done or cursor is None:
                            break
                    except Exception as exc:
                        result.errors.append(f"{source_name}:{query}:{exc}")
                        break
        if not dry_run and not result.errors:
            state.last_sync = datetime.now(timezone.utc)
            self.save_state(state)
        return result

    def load_state(self) -> PaperScoutState:
        if not self.state_path.is_file():
            return PaperScoutState()
        try:
            return PaperScoutState.model_validate_json(self.state_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            raise ValueError(f"Invalid paper scout checkpoint: {self.state_path}: {exc}") from exc

    def save_state(self, state: PaperScoutState) -> Path:
        _atomic_write_json(self.state_path, state.model_dump(mode="json"))
        return self.state_path

    def _default_adapters(self) -> dict[str, PaperSourceAdapter]:
        result: dict[str, PaperSourceAdapter] = {}
        classes: dict[str, type[PaperSourceAdapter]] = {
            "arxiv": ArxivSourceAdapter,
            "crossref": CrossrefSourceAdapter,
            "semantic_scholar": SemanticScholarSourceAdapter,
        }
        for name, source_config in self.config.sources.items():
            adapter_class = classes.get(name)
            if adapter_class is not None:
                result[name] = adapter_class(
                    page_size=source_config.page_size,
                    rate_limit_seconds=source_config.rate_limit_seconds,
                )
        return result


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


__all__ = [
    "DEFAULT_TOPICS",
    "PaperScout",
    "PaperScoutConfig",
    "PaperScoutResult",
    "PaperScoutState",
    "PaperSourceConfig",
    "SourceCheckpoint",
]
