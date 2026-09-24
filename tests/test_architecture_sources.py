"""The architecture source snapshot must match the tracked key packages."""

from __future__ import annotations

from tests.docs_inventory import REPO_ROOT
from yolo_agent.tools import architecture_sources

EXPECTED_AREAS = {
    "cli",
    "task",
    "evidence",
    "action_space",
    "runtime",
    "experiment",
    "research",
    "release",
}


def test_architecture_sources_are_fresh() -> None:
    assert architecture_sources.check(REPO_ROOT) == []


def test_architecture_spec_covers_every_area() -> None:
    spec = architecture_sources.load_spec(REPO_ROOT)
    assert set(spec.get("areas") or {}) == EXPECTED_AREAS
    for area, config in (spec.get("areas") or {}).items():
        assert config.get("paths"), f"area '{area}' tracks no paths"
    snapshot = spec.get("snapshot") or {}
    assert set(snapshot) == EXPECTED_AREAS
    for area, digest in snapshot.items():
        assert isinstance(digest, str) and len(digest) == 64, (
            f"area '{area}' snapshot is not a sha256 hex digest"
        )
