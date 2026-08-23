from __future__ import annotations

import hashlib
import json
from pathlib import Path

from yolo_agent.components.adapters.distillation.teacher_evidence import (
    resolve_student_checkpoint,
    resolve_teacher_checkpoint,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata(path: Path, *, architecture: str = "yolo26s") -> None:
    path.with_suffix(path.suffix + ".metadata.json").write_text(
        json.dumps(
            {
                "architecture": architecture,
                "dataset_hash": "d" * 64,
                "split": "train",
                "imgsz": 640,
            }
        ),
        encoding="utf-8",
    )


def test_teacher_resolution_records_identity_and_protocol(tmp_path: Path) -> None:
    teacher = tmp_path / "yolo26s.pt"
    teacher.write_bytes(b"teacher")
    _metadata(teacher)

    result = resolve_teacher_checkpoint(
        teacher,
        workspace=tmp_path,
        expected_sha256=_sha(teacher),
        expected_dataset_hash="d" * 64,
    )

    assert result.ready
    assert result.identity.architecture == "yolo26s"
    assert result.identity.metadata_verified
    assert result.identity.split == "train"
    assert result.identity.imgsz == 640


def test_teacher_hash_mismatch_is_evidence_recovery(tmp_path: Path) -> None:
    teacher = tmp_path / "yolo26s.pt"
    teacher.write_bytes(b"teacher")
    _metadata(teacher)

    result = resolve_teacher_checkpoint(
        teacher,
        workspace=tmp_path,
        expected_sha256="0" * 64,
        expected_dataset_hash="d" * 64,
    )

    assert result.disposition == "evidence_recovery"
    assert "teacher_checkpoint_sha256_mismatch" in result.reason_codes
    assert "replace" in result.recovery_action


def test_teacher_split_mismatch_is_not_ready(tmp_path: Path) -> None:
    teacher = tmp_path / "yolo26s.pt"
    teacher.write_bytes(b"teacher")
    _metadata(teacher)
    metadata = teacher.with_suffix(teacher.suffix + ".metadata.json")
    metadata.write_text(
        json.dumps(
            {
                "architecture": "yolo26s",
                "dataset_hash": "d" * 64,
                "split": "val",
                "imgsz": 640,
            }
        ),
        encoding="utf-8",
    )

    result = resolve_teacher_checkpoint(
        teacher,
        workspace=tmp_path,
        expected_dataset_hash="d" * 64,
        expected_split="train",
    )

    assert result.disposition == "evidence_recovery"
    assert "teacher_split_mismatch" in result.reason_codes


def test_missing_teacher_is_recoverable_and_not_silently_ready(tmp_path: Path) -> None:
    result = resolve_teacher_checkpoint(
        tmp_path / "yolo26s.pt",
        workspace=tmp_path,
        expected_dataset_hash="d" * 64,
    )

    assert result.disposition == "evidence_recovery"
    assert result.identity.sha256 is None
    assert any(item.startswith("teacher_checkpoint_missing:") for item in result.reason_codes)


def test_student_identity_rejects_non_yolo26n(tmp_path: Path) -> None:
    student = tmp_path / "yolo26s.pt"
    student.write_bytes(b"student")
    _metadata(student, architecture="yolo26s")

    result = resolve_student_checkpoint(student, workspace=tmp_path)

    assert result.disposition == "evidence_recovery"
    assert "student_architecture_not_yolo26n" in result.reason_codes
