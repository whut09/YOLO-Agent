"""Paper metadata source adapters and cached HTTP transport."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.research.schemas import PaperRecord


HttpTransport = Callable[[str, dict[str, str], float], bytes]


class SourcePage(BaseModel):
    """One normalized pagination response from a paper source."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    records: list[dict[str, Any]] = Field(default_factory=list)
    next_cursor: str | None = None
    done: bool = True


class CachedHttpClient:
    """Small GET-only HTTP client with disk cache and bounded retries."""

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        retries: int = 2,
        backoff_seconds: float = 1.0,
        timeout_seconds: float = 30.0,
        transport: HttpTransport | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.retries = max(0, retries)
        self.backoff_seconds = max(0.0, backoff_seconds)
        self.timeout_seconds = timeout_seconds
        self.transport = transport or _urllib_transport

    def get(self, url: str, headers: dict[str, str] | None = None) -> bytes:
        request_headers = headers or {}
        key = hashlib.sha256(
            json.dumps({"url": url, "headers": request_headers}, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / f"{key}.bin"
        if cache_path.is_file():
            return cache_path.read_bytes()
        errors: list[str] = []
        for attempt in range(self.retries + 1):
            try:
                payload = self.transport(url, request_headers, self.timeout_seconds)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(".tmp")
                temporary.write_bytes(payload)
                temporary.replace(cache_path)
                return payload
            except (OSError, TimeoutError, urllib.error.URLError) as exc:
                errors.append(f"attempt_{attempt + 1}:{exc}")
                if attempt >= self.retries:
                    break
                time.sleep(self.backoff_seconds * (attempt + 1))
        raise RuntimeError(f"HTTP request failed after {self.retries + 1} attempts: {' | '.join(errors)}")


class PaperSourceAdapter(ABC):
    """Abstract incremental metadata source."""

    source_name: str

    def __init__(self, *, page_size: int = 100, rate_limit_seconds: float = 0.0) -> None:
        self.page_size = page_size
        self.rate_limit_seconds = max(0.0, rate_limit_seconds)

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        since: datetime | None,
        year_from: int,
        cursor: str | None,
    ) -> str:
        """Build the URL for one incremental page."""

    @abstractmethod
    def fetch(self, url: str, client: CachedHttpClient) -> SourcePage:
        """Fetch and parse one page."""

    @abstractmethod
    def normalize(self, record: dict[str, Any]) -> PaperRecord:
        """Normalize a raw source record into PaperRecord."""

    def checkpoint(self, page: SourcePage) -> str | None:
        return page.next_cursor

    def rate_limit(self) -> None:
        if self.rate_limit_seconds:
            time.sleep(self.rate_limit_seconds)


class ArxivSourceAdapter(PaperSourceAdapter):
    source_name = "arxiv"
    endpoint = "https://export.arxiv.org/api/query"

    def search(self, query: str, *, since: datetime | None, year_from: int, cursor: str | None) -> str:
        start = int(cursor or 0)
        start_date = max(f"{year_from}01010000", since.strftime("%Y%m%d%H%M") if since else "")
        terms = f'all:"{query}" AND submittedDate:[{start_date} TO 299912312359]'
        params = {
            "search_query": terms,
            "start": str(start),
            "max_results": str(self.page_size),
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        }
        return f"{self.endpoint}?{urllib.parse.urlencode(params)}"

    def fetch(self, url: str, client: CachedHttpClient) -> SourcePage:
        root = ET.fromstring(client.get(url))
        namespace = {"atom": "http://www.w3.org/2005/Atom", "opensearch": "http://a9.com/-/spec/opensearch/1.1/"}
        records: list[dict[str, Any]] = []
        for entry in root.findall("atom:entry", namespace):
            records.append({
                "id": _xml_text(entry, "atom:id", namespace),
                "title": _xml_text(entry, "atom:title", namespace),
                "summary": _xml_text(entry, "atom:summary", namespace),
                "published": _xml_text(entry, "atom:published", namespace),
                "updated": _xml_text(entry, "atom:updated", namespace),
                "authors": [_xml_text(author, "atom:name", namespace) for author in entry.findall("atom:author", namespace)],
                "categories": [category.attrib.get("term", "") for category in entry.findall("atom:category", namespace)],
            })
        start = int(_xml_text(root, "opensearch:startIndex", namespace) or 0)
        total = int(_xml_text(root, "opensearch:totalResults", namespace) or len(records))
        next_start = start + len(records)
        return SourcePage(records=records, next_cursor=str(next_start) if next_start < total and records else None, done=next_start >= total or not records)

    def normalize(self, record: dict[str, Any]) -> PaperRecord:
        paper_url = str(record["id"])
        published = _parse_datetime(record.get("published"))
        return PaperRecord(
            paper_id=paper_url.removeprefix("https://arxiv.org/abs/"),
            title=_clean_text(record.get("title")),
            abstract=_clean_text(record.get("summary")),
            year=published.year,
            published_at=published,
            updated_at=_parse_datetime(record.get("updated")),
            authors=[str(item) for item in record.get("authors", []) if item],
            task_families=["object_detection"],
            source_url=paper_url,
            paper_url=paper_url,
            source="arxiv",
            ingestion_version="arxiv.v1",
        )


class CrossrefSourceAdapter(PaperSourceAdapter):
    source_name = "crossref"
    endpoint = "https://api.crossref.org/works"

    def search(self, query: str, *, since: datetime | None, year_from: int, cursor: str | None) -> str:
        params = {
            "query": query,
            "rows": str(self.page_size),
            "cursor": cursor or "*",
            "filter": f"from-pub-date:{year_from}-01-01",
            "select": "DOI,title,abstract,published,author,URL,link,license,subject",
        }
        if since is not None:
            params["filter"] += f",from-index-date:{since.date().isoformat()}"
        return f"{self.endpoint}?{urllib.parse.urlencode(params)}"

    def fetch(self, url: str, client: CachedHttpClient) -> SourcePage:
        data = json.loads(client.get(url, {"User-Agent": "YOLO-Agent/0.1"}).decode("utf-8"))
        message = data.get("message", {})
        records = message.get("items", []) if isinstance(message, dict) else []
        cursor = message.get("next-cursor") if isinstance(message, dict) else None
        return SourcePage(records=records if isinstance(records, list) else [], next_cursor=str(cursor) if cursor and records else None, done=not records)

    def normalize(self, record: dict[str, Any]) -> PaperRecord:
        published = _crossref_date(record.get("published"))
        titles = record.get("title", [])
        authors = record.get("author", [])
        licenses = record.get("license", [])
        links = record.get("link", [])
        code_url = record.get("official_code_url") or _first_url(links, content_type="application/vnd.github+json")
        return PaperRecord(
            paper_id=f"doi:{record['DOI']}",
            doi=str(record["DOI"]),
            title=_clean_text(titles[0] if isinstance(titles, list) and titles else "Untitled"),
            abstract=_clean_text(record.get("abstract")),
            year=published.year,
            published_at=published,
            authors=[_crossref_author(item) for item in authors if isinstance(item, dict)],
            task_families=["object_detection"],
            source_url=str(record.get("URL") or "") or None,
            paper_url=_first_url(links),
            official_code_url=str(code_url) if code_url else None,
            code_license=_first_license(licenses),
            source="crossref",
            ingestion_version="crossref.v1",
        )


class SemanticScholarSourceAdapter(PaperSourceAdapter):
    source_name = "semantic_scholar"
    endpoint = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

    def search(self, query: str, *, since: datetime | None, year_from: int, cursor: str | None) -> str:
        params = {
            "query": query,
            "year": f"{year_from}-",
            "token": cursor or "",
            "fields": "paperId,title,abstract,year,authors,url,externalIds,openAccessPdf,publicationDate",
        }
        return f"{self.endpoint}?{urllib.parse.urlencode(params)}"

    def fetch(self, url: str, client: CachedHttpClient) -> SourcePage:
        data = json.loads(client.get(url).decode("utf-8"))
        records = data.get("data", []) if isinstance(data, dict) else []
        token = data.get("token") if isinstance(data, dict) else None
        return SourcePage(records=records if isinstance(records, list) else [], next_cursor=str(token) if token else None, done=not token)

    def normalize(self, record: dict[str, Any]) -> PaperRecord:
        external = record.get("externalIds") or {}
        doi = external.get("DOI") if isinstance(external, dict) else None
        arxiv = external.get("ArXiv") if isinstance(external, dict) else None
        published = _parse_datetime(record.get("publicationDate")) if record.get("publicationDate") else datetime(int(record["year"]), 1, 1, tzinfo=timezone.utc)
        authors = record.get("authors", [])
        return PaperRecord(
            paper_id=f"arxiv:{arxiv}" if arxiv else f"s2:{record['paperId']}",
            doi=str(doi) if doi else None,
            title=_clean_text(record.get("title")),
            abstract=_clean_text(record.get("abstract")),
            year=int(record.get("year") or published.year),
            published_at=published,
            authors=[str(item.get("name")) for item in authors if isinstance(item, dict) and item.get("name")],
            task_families=["object_detection"],
            source_url=str(record.get("url") or "") or None,
            paper_url=str((record.get("openAccessPdf") or {}).get("url") or "") or None,
            official_code_url=str(record.get("official_code_url") or "") or None,
            source="semantic_scholar",
            ingestion_version="semantic_scholar.v1",
        )


def _urllib_transport(url: str, headers: dict[str, str], timeout: float) -> bytes:
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _xml_text(element: ET.Element, path: str, namespace: dict[str, str]) -> str:
    child = element.find(path, namespace)
    return child.text.strip() if child is not None and child.text else ""


def _parse_datetime(value: Any) -> datetime:
    text = str(value or "").replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _crossref_date(value: Any) -> datetime:
    parts = ((value or {}).get("date-parts") or [[1970, 1, 1]])[0]
    padded = [*parts, 1, 1][:3]
    return datetime(int(padded[0]), int(padded[1]), int(padded[2]), tzinfo=timezone.utc)


def _crossref_author(value: dict[str, Any]) -> str:
    return " ".join(str(value.get(key, "")).strip() for key in ("given", "family")).strip()


def _first_url(values: Any, content_type: str | None = None) -> str | None:
    if not isinstance(values, list):
        return None
    for value in values:
        if not isinstance(value, dict) or not value.get("URL"):
            continue
        if content_type is None or value.get("content-type") == content_type:
            return str(value["URL"])
    return None


def _first_license(values: Any) -> str | None:
    if not isinstance(values, list) or not values or not isinstance(values[0], dict):
        return None
    return str(values[0].get("URL") or "") or None


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("<jats:p>", "").replace("</jats:p>", "").split())


__all__ = [
    "ArxivSourceAdapter",
    "CachedHttpClient",
    "CrossrefSourceAdapter",
    "PaperSourceAdapter",
    "SemanticScholarSourceAdapter",
    "SourcePage",
]
