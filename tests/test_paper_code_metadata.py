from __future__ import annotations

from yolo_agent.research.code_metadata import parse_official_code_metadata
from yolo_agent.research.schemas import PaperRecord


def test_parses_explicit_github_metadata_without_network() -> None:
    paper = PaperRecord(
        paper_id="paper-1",
        title="Paper",
        year=2025,
        official_code_url="https://github.com/owner/project.git",
        code_license="Apache-2.0",
        framework="pytorch",
    )

    metadata = parse_official_code_metadata(paper)

    assert metadata.available is True
    assert metadata.provider == "github"
    assert metadata.repository_slug == "owner/project"
    assert metadata.license == "Apache-2.0"
    assert metadata.framework == "pytorch"


def test_missing_code_metadata_stays_unknown() -> None:
    paper = PaperRecord(paper_id="paper-2", title="Paper", year=2025)

    metadata = parse_official_code_metadata(paper)

    assert metadata.available is False
    assert metadata.repository_slug is None
    assert metadata.license == "unknown"
