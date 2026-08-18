"""Distillation paper protocol tests. No GPU training."""

from __future__ import annotations

from yolo_agent.research.paper_protocol_catalog import build_paper_protocol_contract
from yolo_agent.research.paper_protocol_contract import (
    PaperProtocolContext,
    evaluate_paper_protocol,
)


def test_distillation_requires_bound_teacher() -> None:
    contract = build_paper_protocol_contract(
        "cvf:cvpr2021:Dai_General_Instance_Distillation_for_Object_Detection"
    )
    missing = evaluate_paper_protocol(contract, PaperProtocolContext())
    assert missing.disposition == "evidence_recovery"
    assert "teacher_checkpoint_missing" in missing.reason_codes
    assert "teacher_checkpoint_sha256_missing" in missing.reason_codes

    ready = evaluate_paper_protocol(
        contract,
        PaperProtocolContext(
            teacher_checkpoint_exists=True,
            teacher_sha256="a" * 64,
            teacher_dataset_manifest="dataset-v1",
            student_dataset_manifest="dataset-v1",
            teacher_split="train",
            student_split="train",
        ),
    )
    assert ready.ok is True
    assert ready.allows_asha_registration is True


def test_distillation_rejects_split_mismatch_and_teacher_eval() -> None:
    contract = build_paper_protocol_contract(
        "cvf:cvpr2024:Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection"
    )
    evaluation = evaluate_paper_protocol(
        contract,
        PaperProtocolContext(
            teacher_checkpoint_exists=True,
            teacher_sha256="b" * 64,
            teacher_dataset_manifest="dataset-v1",
            student_dataset_manifest="dataset-v2",
            teacher_split="train",
            student_split="val",
            evaluate_teacher=True,
        ),
    )
    assert evaluation.disposition == "incompatible"
    assert "teacher_student_dataset_manifest_mismatch" in evaluation.reason_codes
    assert "teacher_student_split_mismatch" in evaluation.reason_codes
    assert "teacher_must_not_be_evaluated" in evaluation.reason_codes
