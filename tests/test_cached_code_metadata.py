from __future__ import annotations

from yolo_agent.research.cached_code_metadata import CachedCodeMetadataLoader
from yolo_agent.research.component_aliases import ComponentAliasResolver
from yolo_agent.research.paper_method_evidence_extractor import (
    PaperMethodEvidenceExtractor,
)
from yolo_agent.research.schemas import PaperRecord


def _paper() -> PaperRecord:
    return PaperRecord(
        paper_id="cached-paper",
        title="Detector",
        year=2025,
        official_code_url="https://github.com/owner/project",
    )


def test_loads_only_cached_readme_and_config_text(tmp_path) -> None:
    repository = tmp_path / "owner" / "project"
    repository.mkdir(parents=True)
    (repository / "README.md").write_text(
        "Uses small object sampling in the train dataloader with image weights.",
        encoding="utf-8",
    )
    (repository / "sampler.yaml").write_text(
        "sampling_policy: small_object_sampling",
        encoding="utf-8",
    )
    (repository / "weights.pt").write_bytes(b"not-readable-metadata")

    bundle = CachedCodeMetadataLoader(tmp_path).load(_paper())

    assert [item.source for item in bundle.records] == [
        "cached_readme",
        "cached_config",
    ]
    assert all(item.content_hash for item in bundle.records)
    assert all(item.source_location.startswith("cached_code:owner/project/") for item in bundle.records)


def test_cached_metadata_can_authorize_explicit_method_boundary(tmp_path) -> None:
    repository = tmp_path / "owner" / "project"
    repository.mkdir(parents=True)
    (repository / "README.md").write_text(
        "Small object sampling modifies image weights in the train dataloader.",
        encoding="utf-8",
    )
    sources = CachedCodeMetadataLoader(tmp_path).load(_paper()).extractor_sources()

    profile = PaperMethodEvidenceExtractor(
        ComponentAliasResolver.from_yaml()
    ).extract(_paper(), cached_metadata=sources)

    assert profile.authorizes_method_profile is True
    assert profile.canonical_mechanisms == ["sampling.small_object"]
    assert {item.source for item in profile.observations} == {"cached_readme"}


def test_missing_cache_is_non_blocking_and_does_not_access_network(tmp_path, monkeypatch) -> None:
    def fail_network(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr("urllib.request.urlopen", fail_network)

    bundle = CachedCodeMetadataLoader(tmp_path).load(_paper())

    assert bundle.records == []
    assert bundle.warnings == ["cached_repository_not_found"]
