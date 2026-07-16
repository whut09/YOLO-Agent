"""Local JSONL registry for paper metadata and component claims."""

from __future__ import annotations

import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Iterable

from yolo_agent.research.paper_index import PaperIndex
from yolo_agent.research.schemas import PaperRecord


_ARXIV_VERSION_RE = re.compile(r"(arxiv[:/ ]*)?([0-9]{4}\\.[0-9]{4,5}|[a-z-]+/[0-9]{7})(?:v[0-9]+)$", re.IGNORECASE)
_SPACE_RE = re.compile(r"\\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9]+")


class PaperRegistry:
    """Durable local registry with atomic JSONL replacement."""

    def __init__(self, root: Path | str = "research") -> None:
        self.root = Path(root)
        self.papers_path = self.root / "papers.jsonl"
        self.paper_components_path = self.root / "paper_components.jsonl"
        self.paper_index_path = self.root / "paper_index.json"
        self._papers: dict[str, PaperRecord] = {}
        self.index = PaperIndex()
        self.load()

    def load(self) -> list[PaperRecord]:
        """Load papers and rebuild the query index."""
        papers = [_paper_from_line(line, self.papers_path) for line in _read_jsonl(self.papers_path)]
        self._papers = {paper.paper_id: paper for paper in deduplicate_papers(papers)}
        self.index.replace(self._papers.values())
        return self.list()

    def add(self, paper: PaperRecord) -> PaperRecord:
        """Add a new paper, rejecting an existing identity."""
        self.load()
        existing = self._find_identity(paper)
        if existing is not None:
            raise ValueError(f"Paper already exists: {existing.paper_id}")
        self._papers[paper.paper_id] = paper
        self._persist()
        return paper

    def update(self, paper: PaperRecord) -> PaperRecord:
        """Replace an existing paper record by identity."""
        self.load()
        existing = self._find_identity(paper)
        if existing is None:
            raise KeyError(f"Paper not found: {paper.paper_id}")
        if existing.paper_id != paper.paper_id:
            self._papers.pop(existing.paper_id, None)
        self._papers[paper.paper_id] = paper
        self._persist()
        return paper

    def upsert(self, paper: PaperRecord) -> PaperRecord:
        """Insert or replace a paper using paper id, DOI, arXiv id, or title."""
        self.load()
        existing = self._find_identity(paper)
        if existing is not None and existing.paper_id != paper.paper_id:
            self._papers.pop(existing.paper_id, None)
        self._papers[paper.paper_id] = paper
        self._persist()
        return paper

    def upsert_many(self, papers: Iterable[PaperRecord]) -> list[PaperRecord]:
        """Atomically upsert a complete source page."""
        self.load()
        incoming = list(papers)
        updated = dict(self._papers)
        for paper in incoming:
            existing = _find_identity_in(updated.values(), paper)
            if existing is not None and existing.paper_id != paper.paper_id:
                updated.pop(existing.paper_id, None)
            updated[paper.paper_id] = paper
        previous = self._papers
        self._papers = updated
        try:
            self._persist()
        except Exception:
            self._papers = previous
            raise
        return incoming

    def get(self, paper_id: str) -> PaperRecord | None:
        self.load()
        return self._papers.get(paper_id)

    def list(self, **filters: object) -> list[PaperRecord]:
        """List papers using the same filters as PaperIndex."""
        self.load_index_only()
        return self.index.list(**filters)  # type: ignore[arg-type]

    def remove(self, paper_id: str) -> PaperRecord:
        """Remove a paper and its claims."""
        self.load()
        paper = self._papers.pop(paper_id, None)
        if paper is None:
            raise KeyError(f"Paper not found: {paper_id}")
        self._persist()
        return paper

    def deduplicate(self) -> list[PaperRecord]:
        """Collapse duplicate records using paper id, DOI, arXiv id, and title."""
        self.load()
        deduplicated = deduplicate_papers(self._papers.values())
        self._papers = {paper.paper_id: paper for paper in deduplicated}
        self._persist()
        return self.list()

    def rebuild_index(self) -> Path:
        """Rebuild and persist the JSON query index."""
        self.load_index_only()
        payload = {
            "schema_version": "research_index.v1",
            "paper_count": len(self._papers),
            "paper_ids": sorted(self._papers),
            "years": sorted({paper.year for paper in self._papers.values()}),
        }
        _atomic_write_text(self.paper_index_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return self.paper_index_path

    def load_index_only(self) -> None:
        papers = [_paper_from_line(line, self.papers_path) for line in _read_jsonl(self.papers_path)]
        self._papers = {paper.paper_id: paper for paper in deduplicate_papers(papers)}
        self.index.replace(self._papers.values())

    def _find_identity(self, candidate: PaperRecord) -> PaperRecord | None:
        return _find_identity_in(self._papers.values(), candidate)

    def _persist(self) -> None:
        records = sorted(self._papers.values(), key=lambda paper: (paper.year, paper.paper_id))
        paper_lines = "".join(json.dumps(paper.model_dump(mode="json"), sort_keys=True) + "\n" for paper in records)
        component_lines = "".join(
            json.dumps({"paper_id": paper.paper_id, "claim": claim.model_dump(mode="json")}, sort_keys=True) + "\n"
            for paper in records
            for claim in paper.claimed_effects
        )
        _atomic_write_text(self.papers_path, paper_lines)
        _atomic_write_text(self.paper_components_path, component_lines)
        self.index.replace(records)
        payload = {
            "schema_version": "research_index.v1",
            "paper_count": len(records),
            "paper_ids": sorted(self._papers),
            "years": sorted({paper.year for paper in records}),
        }
        _atomic_write_text(self.paper_index_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def deduplicate_papers(papers: Iterable[PaperRecord]) -> list[PaperRecord]:
    """Deduplicate papers while preserving the richest/latest record."""
    by_identity: dict[str, PaperRecord] = {}
    for paper in papers:
        key = _identity_key(paper)
        current = by_identity.get(key)
        if current is None or _paper_preference(paper) >= _paper_preference(current):
            by_identity[key] = paper
    return sorted(by_identity.values(), key=lambda paper: (paper.year, paper.paper_id))


def _find_identity_in(papers: Iterable[PaperRecord], candidate: PaperRecord) -> PaperRecord | None:
    for paper in papers:
        if paper.paper_id == candidate.paper_id:
            return paper
        if _normalize_doi(paper.doi) and _normalize_doi(paper.doi) == _normalize_doi(candidate.doi):
            return paper
        if _normalize_arxiv(paper.paper_id) and _normalize_arxiv(paper.paper_id) == _normalize_arxiv(candidate.paper_id):
            return paper
        if _normalize_title(paper.title) == _normalize_title(candidate.title):
            return paper
    return None


def _identity_key(paper: PaperRecord) -> str:
    arxiv = _normalize_arxiv(paper.paper_id)
    if arxiv:
        return f"arxiv:{arxiv}"
    doi = _normalize_doi(paper.doi)
    if doi:
        return f"doi:{doi}"
    title = _normalize_title(paper.title)
    return f"title:{title}" if title else f"id:{paper.paper_id.casefold()}"


def _paper_preference(paper: PaperRecord) -> tuple[int, int, int, int]:
    return (
        int(bool(paper.official_code_url)),
        max((_evidence_rank(benchmark.evidence_level) for benchmark in paper.benchmarks), default=0),
        len(paper.claimed_effects),
        paper.year,
    )


def _evidence_rank(level: str) -> int:
    levels = {
        "paper_claim": 0,
        "official_code_available": 1,
        "externally_reproduced": 2,
        "locally_smoke_tested": 3,
        "locally_pilot_reproduced": 4,
        "locally_full_reproduced": 5,
        "confirmed_multi_seed": 6,
    }
    return levels.get(level, 0)


def _normalize_doi(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().lower().removeprefix("https://doi.org/").removeprefix("doi:")


def _normalize_arxiv(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "")
    normalized = normalized.removeprefix("https://arxiv.org/abs/")
    normalized = normalized.removeprefix("arxiv:")
    if not re.fullmatch(r"(?:[0-9]{4}\.[0-9]{4,5}|[a-z-]+/[0-9]{7})(?:v[0-9]+)?", normalized):
        return ""
    return re.sub(r"v[0-9]+$", "", normalized)


def _normalize_title(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    return _SPACE_RE.sub(" ", _PUNCT_RE.sub(" ", value)).strip()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
        records.append(value)
    return records


def _paper_from_line(line: dict[str, object], path: Path) -> PaperRecord:
    try:
        return PaperRecord.model_validate(line)
    except Exception as exc:
        raise ValueError(f"Invalid paper record in {path}: {exc}") from exc


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


__all__ = ["PaperRegistry", "deduplicate_papers"]
