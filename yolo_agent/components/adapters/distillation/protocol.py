"""Deterministic teacher/student checkpoint and dataset protocol helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


TEACHER_NAMES = {"yolo26s.pt", "yolo26m.pt"}
STUDENT_NAME = "yolo26n.pt"


def sha256_file(path: Path | str) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_local_checkpoint(
    value: Path | str,
    *,
    workspace: Path | str | None = None,
    required: bool = True,
) -> Path:
    """Resolve a local checkpoint without downloading or guessing its architecture."""
    requested = Path(value).expanduser()
    candidates = [requested]
    if not requested.is_absolute():
        roots = [Path.cwd()]
        if workspace is not None:
            roots.insert(0, Path(workspace).expanduser())
        roots.extend(
            [
                Path(__file__).resolve().parents[4],
                Path(__file__).resolve().parents[4] / "weights",
            ]
        )
        candidates.extend(root / requested for root in roots)
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    if required:
        raise FileNotFoundError(
            "teacher/student checkpoint is not a local file and automatic download "
            f"is disabled: {value}"
        )
    return requested


def dataset_identity_hash(value: Path | str) -> str:
    """Hash a dataset manifest when available, otherwise its explicit resource identity."""
    path = Path(value).expanduser()
    if path.is_file():
        return sha256_file(path)
    identity = str(path.resolve())
    return hashlib.sha256(f"dataset-resource:{identity}".encode("utf-8")).hexdigest()


def validate_checkpoint_name(path: Path | str, *, student: bool) -> None:
    name = Path(path).name
    allowed = {STUDENT_NAME} if student else TEACHER_NAMES
    if name not in allowed:
        role = "student" if student else "teacher"
        raise ValueError(f"{role} checkpoint must be one of {sorted(allowed)}: {name}")


__all__ = [
    "STUDENT_NAME",
    "TEACHER_NAMES",
    "dataset_identity_hash",
    "resolve_local_checkpoint",
    "sha256_file",
    "validate_checkpoint_name",
]
