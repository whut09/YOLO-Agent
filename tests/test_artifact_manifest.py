"""Artifact manifest tests."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.core.artifact_manifest import ArtifactManifest, ArtifactManifestEntry, sha256_directory, sha256_file


def test_artifact_manifest_records_file_hash(tmp_path: Path) -> None:
    """Manifest entries should record and verify file hashes."""
    artifact = tmp_path / "report.json"
    artifact.write_text('{"ok": true}\n', encoding="utf-8")
    entry = ArtifactManifestEntry.from_path("report", artifact, "profile_data")

    assert entry.type == "file"
    assert entry.sha256 == sha256_file(artifact)
    assert entry.producer_stage == "profile_data"
    assert entry.schema_version == "1.0"
    assert entry.verify() is True

    artifact.write_text('{"ok": false}\n', encoding="utf-8")
    assert entry.verify() is False


def test_artifact_manifest_records_directory_hash(tmp_path: Path) -> None:
    """Directory hashes should be stable and change when contents change."""
    directory = tmp_path / "generated_models"
    directory.mkdir()
    (directory / "a.yaml").write_text("nc: 1\n", encoding="utf-8")
    entry = ArtifactManifestEntry.from_path("generated_models", directory, "smoke")
    original_hash = entry.sha256

    assert entry.type == "directory"
    assert original_hash == sha256_directory(directory)
    assert entry.verify() is True

    (directory / "b.yaml").write_text("nc: 2\n", encoding="utf-8")
    assert entry.verify() is False


def test_artifact_manifest_writes_and_reads_jsonl(tmp_path: Path) -> None:
    """ArtifactManifest should persist JSONL entries."""
    artifact = tmp_path / "model.yaml"
    artifact.write_text("nc: 1\n", encoding="utf-8")
    manifest = ArtifactManifest(tmp_path / "artifact_manifest.jsonl")
    entry = ArtifactManifestEntry.from_path("model", artifact, "smoke")

    manifest.append(entry)
    records = manifest.read()

    assert len(records) == 1
    assert records[0].name == "model"
    assert records[0].path == artifact
    assert records[0].verify() is True
