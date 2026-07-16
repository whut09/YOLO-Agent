"""Offline Paper Scout tests with mocked HTTP transport."""

import json
from pathlib import Path

from yolo_agent.research.paper_registry import PaperRegistry
from yolo_agent.research.paper_scout import PaperScout, PaperScoutConfig, PaperSourceConfig
from yolo_agent.research.paper_sources import ArxivSourceAdapter, CachedHttpClient


ARXIV_XML = b'''<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>1</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <entry>
    <id>https://arxiv.org/abs/2401.12345v1</id>
    <title>  A Detection Method  </title>
    <summary>A metadata-only paper.</summary>
    <published>2024-01-02T00:00:00Z</published>
    <updated>2024-01-03T00:00:00Z</updated>
    <author><name>Author One</name></author>
    <category term="cs.CV" />
  </entry>
</feed>'''


def _config() -> PaperScoutConfig:
    return PaperScoutConfig(
        year_from=2020,
        max_pages_per_query=3,
        retries=1,
        topics=["object detection"],
        sources={"arxiv": PaperSourceConfig(page_size=10, rate_limit_seconds=0)},
    )


def test_arxiv_adapter_builds_incremental_query() -> None:
    adapter = ArxivSourceAdapter(page_size=10)
    url = adapter.search("object detection", since=None, year_from=2020, cursor="10")
    assert "start=10" in url
    assert "max_results=10" in url
    assert "202001010000" in url


def test_scout_sync_normalizes_and_writes_checkpoint(tmp_path: Path) -> None:
    calls: list[str] = []

    def transport(url: str, headers: dict[str, str], timeout: float) -> bytes:
        calls.append(url)
        return ARXIV_XML

    client = CachedHttpClient(tmp_path / "cache", transport=transport)
    registry = PaperRegistry(tmp_path / "research")
    scout = PaperScout(registry, config=_config(), client=client)
    result = scout.sync()

    assert result.errors == []
    assert result.pages_fetched == 1
    assert result.registry_writes == 1
    assert registry.list()[0].paper_id == "2401.12345v1"
    assert registry.list()[0].title == "A Detection Method"
    state = json.loads((tmp_path / "research" / "paper_scout_state.json").read_text(encoding="utf-8"))
    assert state["last_sync"] is not None
    assert state["checkpoints"]["arxiv:object detection"]["completed"] is True
    assert len(calls) == 1


def test_scout_reuses_http_cache(tmp_path: Path) -> None:
    calls = 0

    def transport(url: str, headers: dict[str, str], timeout: float) -> bytes:
        nonlocal calls
        calls += 1
        return b"cached"

    client = CachedHttpClient(tmp_path / "cache", transport=transport)
    assert client.get("https://example.invalid/paper") == b"cached"
    assert client.get("https://example.invalid/paper") == b"cached"
    assert calls == 1


def test_scout_dry_run_does_not_write_registry_or_checkpoint(tmp_path: Path) -> None:
    client = CachedHttpClient(tmp_path / "cache", transport=lambda url, headers, timeout: ARXIV_XML)
    root = tmp_path / "research"
    result = PaperScout(PaperRegistry(root), config=_config(), client=client).sync(dry_run=True)

    assert result.dry_run is True
    assert result.records_normalized == 1
    assert not (root / "papers.jsonl").exists()
    assert not (root / "paper_scout_state.json").exists()


def test_scout_http_failure_preserves_existing_state_and_registry(tmp_path: Path) -> None:
    root = tmp_path / "research"
    registry = PaperRegistry(root)
    existing = ArxivSourceAdapter().normalize({
        "id": "https://arxiv.org/abs/2301.00001v1",
        "title": "Existing Paper",
        "summary": "Existing",
        "published": "2023-01-01T00:00:00Z",
        "updated": "2023-01-02T00:00:00Z",
        "authors": [],
    })
    registry.add(existing)
    old_state = {"schema_version": "paper_scout_state.v1", "last_sync": None, "checkpoints": {}}
    (root / "paper_scout_state.json").write_text(json.dumps(old_state), encoding="utf-8")

    def fail(url: str, headers: dict[str, str], timeout: float) -> bytes:
        raise OSError("offline")

    config = _config().model_copy(update={"retries": 0})
    result = PaperScout(registry, config=config, client=CachedHttpClient(tmp_path / "cache", transport=fail)).sync()

    assert result.errors
    assert len(registry.list()) == 1
    assert json.loads((root / "paper_scout_state.json").read_text(encoding="utf-8")) == old_state


def test_scout_resumes_from_cursor(tmp_path: Path) -> None:
    urls: list[str] = []
    page_one = ARXIV_XML.replace(b"<opensearch:totalResults>1</opensearch:totalResults>", b"<opensearch:totalResults>2</opensearch:totalResults>")
    page_two = page_one.replace(b"<opensearch:startIndex>0</opensearch:startIndex>", b"<opensearch:startIndex>1</opensearch:startIndex>").replace(b"2401.12345", b"2401.12346").replace(b"A Detection Method", b"A Second Detection Method")

    def transport(url: str, headers: dict[str, str], timeout: float) -> bytes:
        urls.append(url)
        return page_one if "start=0" in url else page_two

    client = CachedHttpClient(tmp_path / "cache", transport=transport)
    config = _config().model_copy(update={"max_pages_per_query": 1})
    scout = PaperScout(PaperRegistry(tmp_path / "research"), config=config, client=client)
    first = scout.sync()
    assert first.errors == []
    assert first.pages_fetched == 1
    assert "start=0" in urls[0]
    second = PaperScout(PaperRegistry(tmp_path / "research"), config=config, client=client).sync()
    assert second.errors == []
    assert any("start=1" in url for url in urls)
    assert len(PaperRegistry(tmp_path / "research").list()) == 2
