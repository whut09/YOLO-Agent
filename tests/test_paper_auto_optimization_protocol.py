"""Matched protocol tests for paper auto-optimization acceptance."""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.certification.paper_auto_optimization_protocol import (
    compare_paper_protocols,
    hash_files,
)
from yolo_agent.certification.paper_auto_optimization_schemas import (
    PaperProtocolIdentity,
)


def _identity(**updates: object) -> PaperProtocolIdentity:
    values: dict[str, object] = {
        "dataset_manifest_hash": "dataset",
        "subset_manifest_hash": "subset",
        "seed": 1,
        "epochs": 3,
        "batch_policy_hash": "batch",
        "ultralytics_version": "8.4.0",
        "eval_protocol_hash": "eval",
        "objective_hash": "objective",
        "protocol_hash": "protocol",
    }
    values.update(updates)
    return PaperProtocolIdentity.model_validate(values)


def test_identical_candidate_control_protocol_is_matched() -> None:
    result = compare_paper_protocols(_identity(), _identity())

    assert result.matched is True
    assert result.protocol_hash == "protocol"
    assert result.mismatched_fields == {}


def test_each_fairness_field_mismatch_is_reported() -> None:
    fields = {
        "dataset_manifest_hash": "other-dataset",
        "subset_manifest_hash": "other-subset",
        "seed": 2,
        "epochs": 10,
        "batch_policy_hash": "other-batch",
        "ultralytics_version": "different",
        "eval_protocol_hash": "other-eval",
        "imgsz": 1280,
        "objective_hash": "other-objective",
        "protocol_hash": "other-protocol",
    }
    for field, value in fields.items():
        if field == "imgsz":
            continue
        result = compare_paper_protocols(_identity(), _identity(**{field: value}))
        assert result.matched is False
        assert field in result.mismatched_fields


def test_missing_protocol_identity_fails_closed() -> None:
    result = compare_paper_protocols(_identity(), None)

    assert result.matched is False
    assert result.mismatched_fields["protocol_identity"] == ("present", "missing")


def test_protocol_identity_rejects_non_640_imgsz() -> None:
    with pytest.raises(ValueError):
        _identity(imgsz=1280)


def test_dataset_protocol_hash_ignores_runtime_cache_files(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "sample.png").write_bytes(b"image")
    before = hash_files(tmp_path)

    (tmp_path / "labels.cache").write_bytes(b"runtime cache")
    (tmp_path / "images" / "worker.tmp").write_bytes(b"temporary")

    assert hash_files(tmp_path) == before
