"""Model-graph paper protocol tests. No GPU training."""

from __future__ import annotations

import pytest

from yolo_agent.research.paper_evidence_requirements import evidence_artifacts_for_family
from yolo_agent.research.paper_protocol_contract import (
    PaperProtocolContext,
    PaperProtocolContract,
    evaluate_paper_protocol,
)


def _graph_contract(**overrides: object) -> PaperProtocolContract:
    payload = {
        "paper_id": "test:graph",
        "dataset_role": "coco_standard",
        "source_split": "coco_train",
        "target_split": "coco_val",
        "annotation_requirement": "fully_labeled",
        "train_val_test_protocol": "coco_official",
        "imgsz": 640,
        "model_family": "yolo26",
        "head_constraint": "yolo26_one_to_one",
        "teacher_requirement": "none",
        "graph_change": "required",
        "loss_change": "none",
        "inference_change": "none",
        "required_metrics": ["map50_95"],
        "paired_baseline_requirement": True,
        "required_evidence_artifacts": evidence_artifacts_for_family("model_graph"),
        "protocol_family": "model_graph",
        "graph_identity": "neck.rtmdet_large_kernel",
        "yolo26_one_to_one_head": True,
        "native_dfl_free_regression": True,
        "mechanism_ids": ["neck.rtmdet_large_kernel"],
    }
    payload.update(overrides)
    return PaperProtocolContract.model_validate(payload)


def test_model_graph_accepts_declared_yolo26_identity() -> None:
    evaluation = evaluate_paper_protocol(
        _graph_contract(graph_identity="neck.rtmdet_large_kernel"),
        PaperProtocolContext(),
    )
    assert evaluation.ok is True


def test_model_graph_requires_graph_identity() -> None:
    with pytest.raises(ValueError, match="graph_identity"):
        _graph_contract(graph_identity=None)


def test_model_graph_rejects_non_yolo26_constraints() -> None:
    evaluation = evaluate_paper_protocol(
        _graph_contract(
            paper_id="test:graph-incompatible",
            yolo26_one_to_one_head=False,
            native_dfl_free_regression=False,
            model_family="separate_detector_family",
            head_constraint="incompatible",
        )
    )
    assert evaluation.disposition == "incompatible"
    assert "yolo26_one_to_one_head_unsatisfied" in evaluation.reason_codes
    assert "native_dfl_free_regression_unsatisfied" in evaluation.reason_codes
    assert "model_graph_requires_yolo26_family" in evaluation.reason_codes
