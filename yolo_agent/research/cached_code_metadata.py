"""Read explicitly cached paper-code README and config metadata offline."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.research.code_metadata import parse_official_code_metadata
from yolo_agent.research.paper_method_evidence import MethodEvidenceSource
from yolo_agent.research.schemas import PaperRecord


_MAX_FILE_BYTES = 1_000_000
_CONFIG_SUFFIXES = {".yaml", ".yml", ".json", ".toml"}


class CachedCodeText(BaseModel):
    """One bounded local README or config text available to extraction."""

    model_config = ConfigDict(extra="forbid")

    source: MethodEvidenceSource
    source_location: str
    text: str
    content_hash: str


class CachedCodeMetadataBundle(BaseModel):
    """Offline cached-code input for one paper."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    repository_slug: str | None = None
    records: list[CachedCodeText] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def extractor_sources(self) -> list[tuple[MethodEvidenceSource, str, str]]:
        return [
            (record.source, record.source_location, record.text)
            for record in self.records
        ]


class CachedCodeMetadataLoader:
    """Load only allowlisted text files beneath a caller-provided cache root."""

    def __init__(self, cache_root: str | Path) -> None:
        self.cache_root = Path(cache_root).resolve()

    def load(self, paper: PaperRecord) -> CachedCodeMetadataBundle:
        code = parse_official_code_metadata(paper)
        if not code.owner or not code.project:
            return CachedCodeMetadataBundle(
                paper_id=paper.paper_id,
                repository_slug=code.repository_slug,
            )
        repository_root = (self.cache_root / code.owner / code.project).resolve()
        if not repository_root.is_relative_to(self.cache_root):
            return CachedCodeMetadataBundle(
                paper_id=paper.paper_id,
                repository_slug=code.repository_slug,
                warnings=["cached_repository_path_outside_root"],
            )
        if not repository_root.is_dir():
            return CachedCodeMetadataBundle(
                paper_id=paper.paper_id,
                repository_slug=code.repository_slug,
                warnings=["cached_repository_not_found"],
            )
        records: list[CachedCodeText] = []
        warnings: list[str] = []
        for path in sorted(repository_root.rglob("*")):
            if not path.is_file() or not _allowlisted(path):
                continue
            try:
                size = path.stat().st_size
                if size > _MAX_FILE_BYTES:
                    warnings.append(f"cached_file_too_large:{path.relative_to(repository_root)}")
                    continue
                raw = path.read_bytes()
                text = raw.decode("utf-8", errors="replace")
            except OSError:
                warnings.append(f"cached_file_unreadable:{path.relative_to(repository_root)}")
                continue
            relative = path.relative_to(repository_root).as_posix()
            source: MethodEvidenceSource = (
                "cached_readme" if path.name.lower().startswith("readme")
                else "cached_config"
            )
            records.append(CachedCodeText(
                source=source,
                source_location=f"cached_code:{code.repository_slug}/{relative}",
                text=text,
                content_hash=hashlib.sha256(raw).hexdigest(),
            ))
        return CachedCodeMetadataBundle(
            paper_id=paper.paper_id,
            repository_slug=code.repository_slug,
            records=records,
            warnings=sorted(warnings),
        )


def _allowlisted(path: Path) -> bool:
    return (
        path.name.lower().startswith("readme") and path.suffix.lower() in {".md", ".txt"}
    ) or path.suffix.lower() in _CONFIG_SUFFIXES


__all__ = [
    "CachedCodeMetadataBundle",
    "CachedCodeMetadataLoader",
    "CachedCodeText",
]
