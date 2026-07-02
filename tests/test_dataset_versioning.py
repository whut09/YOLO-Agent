"""Dataset versioning tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.core.dataset_versioning import DatasetVersionStore, diff_manifests


def _write_dataset(root: Path) -> None:
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    (root / "images" / "train" / "a.jpg").write_bytes(b"image-a")
    (root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")


def test_dataset_version_store_creates_manifest_and_diff(tmp_path: Path) -> None:
    """Version store should track file additions, removals, and modifications."""
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)
    store = DatasetVersionStore(tmp_path / "versions")

    v1 = store.create_version(dataset_root, "dataset_v1", notes=["initial labels"])
    (dataset_root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    (dataset_root / "images" / "train" / "b.jpg").write_bytes(b"image-b")
    (dataset_root / "labels" / "train" / "b.txt").write_text("0 0.4 0.4 0.1 0.1\n", encoding="utf-8")
    (dataset_root / "images" / "train" / "a.jpg").unlink()
    v2 = store.create_version(dataset_root, "dataset_v2", notes=["relabel and add b"])

    diff = store.diff_versions("dataset_v1", "dataset_v2")

    assert v1.version == "dataset_v1"
    assert v2.version == "dataset_v2"
    assert "images/train/b.jpg" in diff.added
    assert "labels/train/b.txt" in diff.added
    assert "images/train/a.jpg" in diff.removed
    assert "labels/train/a.txt" in diff.modified
    assert (tmp_path / "versions" / "dataset_v2" / "diff_from_dataset_v1.json").exists()


def test_diff_manifests_without_store_roundtrip(tmp_path: Path) -> None:
    """Manifest diff should also work as a pure function."""
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)
    store = DatasetVersionStore(tmp_path / "versions")
    before = store.create_version(dataset_root, "v1")
    (dataset_root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    after = store.create_version(dataset_root, "v2")

    diff = diff_manifests(before, after)

    assert diff.modified == ["labels/train/a.txt"]


def test_copy_data_and_reject_nested_version(tmp_path: Path) -> None:
    """copy_data should snapshot files and version names should stay safe."""
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)
    store = DatasetVersionStore(tmp_path / "versions")

    store.create_version(dataset_root, "v1", copy_data=True)

    assert (tmp_path / "versions" / "v1" / "data" / "images" / "train" / "a.jpg").exists()
    with pytest.raises(ValueError):
        store.create_version(dataset_root, "../bad")

