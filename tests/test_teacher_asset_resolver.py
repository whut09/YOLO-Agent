"""CPU-only tests for real YOLO26 teacher asset resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from yolo_agent.certification.distillation import teacher_runtime_binding
from yolo_agent.components.adapters.distillation.teacher_asset_resolver import (
    TeacherAssetResolver,
    verify_teacher_asset,
)


def _write_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "coco.yaml"
    path.write_text("path: /data/coco\ntrain: train2017\nval: val2017\n", encoding="utf-8")
    return path


def _write_checkpoint(path: Path) -> Path:
    torch.save(
        {
            "architecture": "yolo26s",
            "split": "train",
            "imgsz": 640,
            "state_dict": {},
        },
        path,
    )
    return path


def test_local_teacher_is_protocol_bound_and_hash_verified(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    teacher_dir = tmp_path / "teachers"
    teacher_dir.mkdir()
    teacher = _write_checkpoint(teacher_dir / "yolo26s.pt")

    report = TeacherAssetResolver().resolve(
        dataset=dataset,
        download_dir=teacher_dir,
        preferred_teachers=("yolo26s.pt",),
        allow_download=False,
        output_path=tmp_path / "teacher_assets.yaml",
    )

    record = report.preferred_record
    assert record is not None
    assert record.availability == "available"
    assert record.checkpoint_path == str(teacher.resolve())
    assert record.sha256
    assert record.architecture == "yolo26s"
    assert record.dataset == str(dataset.resolve())
    assert record.dataset_manifest_hash
    assert record.split == "train"
    assert record.imgsz == 640
    assert record.frozen is True
    assert record.metadata_verified is True
    assert record.checkpoint_loadable is True
    assert verify_teacher_asset(
        record,
        dataset_manifest_hash=report.dataset_manifest_hash,
    ) == []

    loaded = type(report).from_yaml(tmp_path / "teacher_assets.yaml")
    assert loaded.report_hash == report.report_hash


def test_test_only_teacher_never_becomes_available(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    teacher_dir = tmp_path / "teachers"
    teacher_dir.mkdir()
    _write_checkpoint(teacher_dir / "yolo26s.pt")

    report = TeacherAssetResolver().resolve(
        dataset=dataset,
        download_dir=teacher_dir,
        preferred_teachers=("yolo26s.pt",),
        allow_download=False,
        test_only=True,
    )

    record = report.records[0]
    assert report.preferred_teacher is None
    assert record.availability == "unavailable"
    assert record.test_only is True
    assert "test_only_teacher_not_production_asset" in record.exact_blocker


def test_missing_teacher_is_a_precise_recoverable_blocker(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    report = TeacherAssetResolver().resolve(
        dataset=dataset,
        download_dir=tmp_path / "empty-teachers",
        preferred_teachers=("yolo26s.pt",),
        allow_download=False,
    )

    record = report.records[0]
    assert record.availability == "unavailable"
    assert record.exact_blocker == "official_teacher_download_disabled"
    assert report.preferred_teacher is None


def test_file_drift_is_detected_after_resolution(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    teacher_dir = tmp_path / "teachers"
    teacher_dir.mkdir()
    teacher = _write_checkpoint(teacher_dir / "yolo26s.pt")
    report = TeacherAssetResolver().resolve(
        dataset=dataset,
        download_dir=teacher_dir,
        preferred_teachers=("yolo26s.pt",),
        allow_download=False,
    )

    teacher.write_bytes(b"changed after resolution")
    record = report.records[0]
    assert verify_teacher_asset(
        record,
        dataset_manifest_hash=report.dataset_manifest_hash,
    ) == ["teacher_checkpoint_sha256_mismatch"]


def test_training_binding_exports_only_the_student_measurement_policy(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path)
    teacher_dir = tmp_path / "teachers"
    teacher_dir.mkdir()
    _write_checkpoint(teacher_dir / "yolo26s.pt")
    report = TeacherAssetResolver().resolve(
        dataset=dataset,
        download_dir=teacher_dir,
        preferred_teachers=("yolo26s.pt",),
        allow_download=False,
    )

    binding = teacher_runtime_binding(
        report.preferred_record,
        dataset_manifest_hash=report.dataset_manifest_hash,
    )
    assert binding["teacher_frozen"] is True
    assert binding["teacher_exported"] is False
    assert binding["measure_student_only"] is True
    assert binding["student_architecture"] == "yolo26n"
    assert binding["teacher_checkpoint_sha256"] == report.preferred_record.sha256


def test_official_loader_is_used_without_a_hardcoded_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _write_dataset(tmp_path)
    calls: list[tuple[Path, str, str]] = []

    def fake_download(file: str | Path, *, repo: str, release: str) -> str:
        destination = Path(file)
        calls.append((destination, repo, release))
        _write_checkpoint(destination)
        return str(destination)

    monkeypatch.setattr(
        "ultralytics.utils.downloads.attempt_download_asset",
        fake_download,
    )
    report = TeacherAssetResolver().resolve(
        dataset=dataset,
        download_dir=tmp_path / "official-cache",
        preferred_teachers=("yolo26s.pt",),
        allow_download=True,
    )

    assert calls
    assert calls[0][0].name == "yolo26s.pt"
    assert calls[0][1] == "ultralytics/assets"
    assert calls[0][2] == "latest"
    assert report.preferred_teacher == "yolo26s.pt"
    assert report.records[0].source_kind == "ultralytics_official"


def test_cli_resolve_teachers_is_asset_only(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    from yolo_agent.cli import main

    dataset = _write_dataset(tmp_path)
    teacher_dir = tmp_path / "teachers"
    teacher_dir.mkdir()
    _write_checkpoint(teacher_dir / "yolo26s.pt")
    output = tmp_path / "teacher_assets.yaml"

    monkeypatch.chdir(tmp_path)
    assert main(
        [
            "research",
            "resolve-teachers",
            "--model",
            "yolo26n.pt",
            "--data",
            str(dataset),
            "--download-dir",
            str(teacher_dir),
            "--no-download",
            "--output",
            str(output),
        ]
    ) == 0
    captured = capsys.readouterr().out
    assert "Training: not started (asset resolution only)" in captured
    assert "yolo26s.pt\tavailable" in captured
    assert output.is_file()


def test_student_and_dataset_contracts_are_checked(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    with pytest.raises(ValueError, match="fixed student yolo26n"):
        TeacherAssetResolver().resolve(student_model="yolo26s.pt", dataset=dataset)
    with pytest.raises(FileNotFoundError, match="dataset manifest"):
        TeacherAssetResolver().resolve(
            student_model="yolo26n.pt",
            dataset=tmp_path / "missing.yaml",
        )
