from __future__ import annotations

import json
from pathlib import Path

import yaml

from yolo_agent.agents.paper_adapter_implementation_planner import (
    PaperAdapterImplementationPlan,
    PaperAdapterQueueItem,
    PaperAdapterPlanningPolicy,
    record_implementation_plan,
    write_implementation_plan,
)
from yolo_agent.agents.paper_adapter_planning.fingerprints import (
    component_family,
    implementation_fingerprint,
)
from yolo_agent.core.decision_ledger import DecisionLedger


def test_implementation_fingerprint_identifies_work_not_paper_record() -> None:
    first = implementation_fingerprint(
        component_id="sampling.small_object",
        insertion_point="train_dataloader_sampler",
        required_runtime_hook="train_dataloader_sampler",
    )
    second = implementation_fingerprint(
        component_id="sampling.small-object",
        insertion_point="train-dataloader-sampler",
        required_runtime_hook="train_dataloader_sampler",
    )

    assert first == second
    assert component_family("head.p2_small_object", "detection_head") == "detection_head"
    assert component_family("inference.sahi_slicing", "slicing") == "inference_policy"


def test_scoring_policy_loads_from_reviewed_yaml() -> None:
    policy = PaperAdapterPlanningPolicy.from_yaml(
        "configs/paper_adapter_implementation.yaml"
    )

    assert policy.ap_small_priority_weights["sampling.small_object"] == 25.0
    assert policy.local_negative_max_penalty < -policy.freshness_max_weight
    assert policy.family_cooldown_rounds == 2


def test_plan_artifact_writes_yaml_and_json_atomically(tmp_path: Path) -> None:
    plan = PaperAdapterImplementationPlan(
        current_round=3,
        summary={"ready_to_materialize": 0},
        plan_hash="stable-hash",
    )

    yaml_path = write_implementation_plan(plan, tmp_path / "implementation_plan.yaml")
    json_path = write_implementation_plan(plan, tmp_path / "implementation_plan.json")

    assert yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["auto_code_generation"] is False
    assert json.loads(json_path.read_text(encoding="utf-8"))["plan_hash"] == "stable-hash"


def test_plan_summary_is_written_to_decision_ledger(tmp_path: Path) -> None:
    ledger = DecisionLedger(tmp_path / "decision_ledger.jsonl")
    plan = PaperAdapterImplementationPlan(
        current_round=4,
        summary={"implementation_queue": 0},
        plan_hash="plan-hash",
    )

    record = record_implementation_plan(ledger, run_id="run", plan=plan)

    assert record.decision == "no_actionable_implementation"
    assert record.proposal["plan_hash"] == "plan-hash"
    assert record.input_summary["auto_code_generation"] is False
    assert ledger.read()[0].decision_type == "paper_adapter_implementation_queue"


def test_queue_item_can_describe_reusable_mechanism_adapter_family() -> None:
    item = PaperAdapterQueueItem(
        component_id="loss.quality.correlation",
        component_family="loss",
        mechanism_cluster_id="quality_alignment",
        adapter_family="loss.quality_alignment",
        canonical_component_ids=[
            "loss.quality.correlation",
            "loss.quality.pseudo_iou",
        ],
        covered_paper_count=12,
        paper_ids=["paper-a", "paper-b"],
        paper_year=2025,
        official_code_available=True,
        source_license="Apache-2.0",
        yolo26_compatibility="compatible",
        implementation_status="adapter_required",
        insertion_point="trainer_loss",
        fingerprint="fingerprint",
        track="implementation_queue",
    )

    assert item.adapter_family == "loss.quality_alignment"
    assert item.covered_paper_count == 12
