"""Offline normalization of paper-provided official code metadata."""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.research.schemas import PaperRecord


CodeProvider = Literal["github", "gitlab", "gitee", "other", "unknown"]


class OfficialCodeMetadata(BaseModel):
    """Metadata explicitly available in a PaperRecord, without network lookup."""

    model_config = ConfigDict(extra="forbid")

    available: bool = False
    url: str | None = None
    host: str | None = None
    provider: CodeProvider = "unknown"
    repository_slug: str | None = None
    owner: str | None = None
    project: str | None = None
    license: str = "unknown"
    framework: str = "unknown"
    source_locations: list[str] = Field(default_factory=list)


def parse_official_code_metadata(paper: PaperRecord) -> OfficialCodeMetadata:
    """Parse only fields already present in the frozen paper record."""
    raw_url = (paper.official_code_url or "").strip()
    if not raw_url:
        return OfficialCodeMetadata(
            license=paper.code_license or "unknown",
            framework=paper.framework or "unknown",
        )
    parsed = urlparse(raw_url)
    host = (parsed.hostname or "").lower()
    provider: CodeProvider = "other"
    if host in {"github.com", "www.github.com"}:
        provider = "github"
    elif host in {"gitlab.com", "www.gitlab.com"}:
        provider = "gitlab"
    elif host in {"gitee.com", "www.gitee.com"}:
        provider = "gitee"
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    owner = parts[0] if len(parts) >= 2 else None
    project = parts[1].removesuffix(".git") if len(parts) >= 2 else None
    slug = f"{owner}/{project}" if owner and project else None
    locations = ["paper_record.official_code_url"]
    if paper.code_license:
        locations.append("paper_record.code_license")
    if paper.framework:
        locations.append("paper_record.framework")
    return OfficialCodeMetadata(
        available=True,
        url=raw_url,
        host=host or None,
        provider=provider,
        repository_slug=slug,
        owner=owner,
        project=project,
        license=paper.code_license or "unknown",
        framework=paper.framework or "unknown",
        source_locations=locations,
    )


__all__ = ["CodeProvider", "OfficialCodeMetadata", "parse_official_code_metadata"]
