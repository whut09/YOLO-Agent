"""Inference-only paper protocol tests. No GPU training."""

from __future__ import annotations

from yolo_agent.research.paper_protocol_catalog import inference_only_protocol
from yolo_agent.research.paper_protocol_contract import (
    PaperProtocolContext,
    evaluate_paper_protocol,
)


def test_inference_only_is_not_a_training_candidate() -> None:
    contract = inference_only_protocol()
    evaluation = evaluate_paper_protocol(contract, PaperProtocolContext(asha_track="training"))
    assert evaluation.execution_class == "inference_candidate"
    assert evaluation.allows_asha_registration is False
    assert evaluation.allows_materialization is False
    assert "inference_only_not_training_candidate" in evaluation.reason_codes
    assert "inference_only_excluded_from_training_asha" in evaluation.reason_codes
