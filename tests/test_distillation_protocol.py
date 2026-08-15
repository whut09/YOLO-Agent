from pathlib import Path

import pytest

from yolo_agent.components.adapters.distillation.protocol import (
    dataset_identity_hash,
    resolve_local_checkpoint,
    sha256_file,
    validate_checkpoint_name,
)


def test_teacher_resolution_is_local_and_deterministic(tmp_path: Path) -> None:
    teacher = tmp_path / "yolo26s.pt"
    teacher.write_bytes(b"teacher")
    assert resolve_local_checkpoint("yolo26s.pt", workspace=tmp_path) == teacher.resolve()
    assert sha256_file(teacher) == sha256_file(teacher)
    assert len(dataset_identity_hash(tmp_path / "missing.yaml")) == 64


def test_missing_teacher_does_not_trigger_download(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="automatic download is disabled"):
        resolve_local_checkpoint("yolo26s.pt", workspace=tmp_path)


def test_checkpoint_role_names_are_explicit() -> None:
    validate_checkpoint_name("yolo26s.pt", student=False)
    validate_checkpoint_name("yolo26n.pt", student=True)
    with pytest.raises(ValueError, match="teacher checkpoint"):
        validate_checkpoint_name("yolo26n.pt", student=False)
    with pytest.raises(ValueError, match="student checkpoint"):
        validate_checkpoint_name("yolo26s.pt", student=True)
