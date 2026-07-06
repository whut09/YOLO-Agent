"""Diagnosis graph tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from yolo_agent.agents.diagnosis_graph import DiagnosisGraphReport
from yolo_agent.agents.doctor_report import build_doctor_decision_report, merge_evidence_grounded_doctor_report
from yolo_agent.agents.diagnosis_graph import DiagnosisGraph, diagnosis_graph_from_error_facts
from yolo_agent.core.error_facts import ErrorFact, ErrorFactStore
from yolo_agent.core.experiment_graph import Evidence, MetricEvidence
from yolo_agent.agents.orchestrator import LoopOrchestrator


def _make_task(path: Path) -> Path:
    task_path = path / "task.yaml"
    task_path.write_text(
        yaml.safe_dump(
            {
                "task_type": "detect",
                "scene": "generic",
                "class_names": ["object"],
                "primary_metric": {"name": "map50_95"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return task_path


def test_diagnosis_graph_maps_small_object_fact_to_causal_hypotheses() -> None:
    """Small-object AP facts should produce causes, evidence needs, and actions."""
    fact = ErrorFact(
        run_id="exp001",
        candidate_id="baseline",
        node_id="node_baseline",
        fact_type="area_metric",
        subject="small",
        area="small",
        metric_name="ap_small",
        value=0.19,
        severity="high",
        action_candidates=["small_object_recipe"],
    )

    report = diagnosis_graph_from_error_facts([fact])

    assert report.findings
    finding = report.findings[0]
    assert finding.diagnosis_id == "small_object_ap_low"
    assert {cause.cause_id for cause in finding.possible_causes} >= {
        "object_too_small_for_stride",
        "insufficient_positive_assignment",
        "slicing_or_inference_resolution_missing",
    }
    assert "bbox_area_histogram" in finding.evidence_needed
    assert "small_object_oversampling" in finding.actions
    assert "small_object_recipe" in finding.actions
    assert report.action_candidates[0] in finding.actions


def test_diagnosis_graph_is_configurable(tmp_path: Path) -> None:
    """Custom diagnosis YAML should be loadable without code changes."""
    graph_path = tmp_path / "diagnosis_graph.yaml"
    graph_path.write_text(
        yaml.safe_dump(
            {
                "rules": [
                    {
                        "id": "custom_background_fp",
                        "symptom": "Custom background false positives.",
                        "match": {"fact_types": ["background_false_positive_class"]},
                        "possible_causes": [
                            {
                                "cause_id": "custom_hard_negative_gap",
                                "description": "Missing custom hard negatives.",
                                "evidence_needed": ["custom_fp_gallery"],
                                "actions": ["custom_hard_negative_mining"],
                            }
                        ],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    fact = ErrorFact(
        run_id="exp001",
        candidate_id="baseline",
        node_id="node_baseline",
        fact_type="background_false_positive_class",
        subject="person",
        class_name="person",
        count=12,
        severity="medium",
    )

    report = DiagnosisGraph.from_yaml(graph_path).diagnose([fact])

    assert report.findings[0].diagnosis_id == "custom_background_fp"
    assert report.evidence_needed == ["custom_fp_gallery"]
    assert report.action_candidates == ["custom_hard_negative_mining"]


def test_next_round_payload_includes_diagnosis_graph(tmp_path: Path) -> None:
    """Next-round evidence should expose causal diagnoses derived from error facts."""
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        yaml.safe_dump(
            {
                "task_type": "detect",
                "scene": "generic",
                "class_names": ["object"],
                "primary_metric": {"name": "map50_95"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    dataset_root = tmp_path / "dataset"
    (dataset_root / "images" / "train").mkdir(parents=True)
    (dataset_root / "labels" / "train").mkdir(parents=True)
    (dataset_root / "images" / "train" / "img1.jpg").write_bytes(b"image")
    (dataset_root / "labels" / "train" / "img1.txt").write_text("", encoding="utf-8")
    data_yaml = dataset_root / "data.yaml"
    data_yaml.write_text("path: .\ntrain: images/train\nnames:\n  0: object\n", encoding="utf-8")
    run_root = tmp_path / "runs"
    orchestrator = LoopOrchestrator.initialize("exp001", task_path, data_yaml, run_root=run_root)
    ErrorFactStore(run_root).append(
        "exp001",
        [
            ErrorFact(
                run_id="exp001",
                candidate_id="baseline",
                node_id="node_baseline",
                fact_type="area_metric",
                subject="small",
                area="small",
                metric_name="ap_small",
                value=0.2,
                severity="high",
                action_candidates=["small_object_recipe"],
            )
        ],
    )

    payload = orchestrator.evidence.next_round_payload({})

    assert payload["diagnosis_graph"]["findings"][0]["diagnosis_id"] == "small_object_ap_low"
    assert "bbox_area_histogram" in payload["diagnosis_graph_evidence_needed"]
    assert "small_object_oversampling" in payload["diagnosis_graph_action_candidates"]
    assert payload["doctor_report"]["primary_problem"] == "AP_small low"
    assert any("AP_small=0.2" in item for item in payload["doctor_report"]["evidence"])
    assert "small_object_oversampling" in {
        item["action"] for item in payload["doctor_report"]["selected_actions"]
    }
    assert payload["doctor_report"]["expected_improvement"]["AP_small"] == (
        "increase; pilot_positive_delta required"
    )
    assert "Pilot does not improve the bound target error facts." in payload["doctor_report"]["stop_condition"]


def test_doctor_report_rejects_imgsz_increase_guardrail() -> None:
    """Doctor report should explain why blocked actions were not selected."""
    report = build_doctor_decision_report(
        diagnosis_graph=DiagnosisGraphReport(),
        current_round_focus=[
            {
                "diagnosis_kind": "small_object_ap",
                "fact_type": "area_metric",
                "subject": "small",
                "area": "small",
                "metric_name": "ap_small",
                "value": 0.21,
                "severity": "high",
                "action_candidates": ["small_object_oversampling"],
                "reason": "Small-object AP is an unresolved baseline weakness.",
            }
        ],
        current_round_error_actions=["small_object_oversampling"],
        error_delta_policy={
            "proposal_mode": "pilot_only",
            "status": "ready_for_baseline_error_pilot_proposals",
            "full_candidate_proposal_allowed": False,
            "proposal_budget_profiles_blocked": ["candidate_full"],
            "guardrails": [],
        },
        error_delta={"parent_fact_count": 0},
        raw_plan={
            "guardrails": [
                "blocked_imgsz_increase: requested imgsz=960 exceeds fixed baseline imgsz=640; keep input size fixed."
            ]
        },
        current_missing_evidence=[],
        newly_available_evidence=[],
    )

    assert report.primary_problem == "AP_small low"
    assert any(item.action == "increase_imgsz" for item in report.rejected_actions)
    assert report.selected_actions[0].action == "small_object_oversampling"
    assert report.why


def test_doctor_report_merges_only_evidence_grounded_llm_draft() -> None:
    """LLM doctor drafts may supplement explanations only when backed by evidence."""
    fact = ErrorFact(
        run_id="exp001",
        candidate_id="baseline",
        node_id="node_baseline",
        fact_type="area_metric",
        subject="small",
        area="small",
        metric_name="ap_small",
        value=0.2,
        severity="high",
        action_candidates=["small_object_oversampling"],
    )
    rule_report = build_doctor_decision_report(
        diagnosis_graph=diagnosis_graph_from_error_facts([fact]),
        current_round_focus=[
            {
                "diagnosis_kind": "small_object_ap",
                "fact_type": "area_metric",
                "subject": "small",
                "area": "small",
                "metric_name": "ap_small",
                "value": 0.2,
                "severity": "high",
                "action_candidates": ["small_object_oversampling"],
            }
        ],
        current_round_error_actions=["small_object_oversampling"],
        error_delta_policy={"proposal_mode": "pilot_only", "status": "ready"},
        error_delta={"parent_fact_count": 0},
        raw_plan={},
        current_missing_evidence=[],
        newly_available_evidence=[],
    )
    evidence = Evidence(
        run_id="exp001",
        metrics={"latency_ms": 12.0},
        metric_records=[
            MetricEvidence(
                candidate_id="baseline",
                node_id="node_baseline",
                metric_name="ap_small",
                value=0.2,
                validator="unit-test",
            )
        ],
    )
    llm_draft = {
        "primary_problem": "Small-object weakness",
        "evidence": [
            "AP_small=0.2 for small objects",
            "mAP50-95=0.99 proves the model is excellent",
        ],
        "why": [
            "AP_small=0.2 supports a small-object pilot.",
            "A private benchmark proves a large gain.",
        ],
        "selected_actions": ["small_object_oversampling", "increase_imgsz"],
        "expected_improvement": {"AP_small": "+0.5 to +1.5", "private_score": "+9"},
    }

    merged = merge_evidence_grounded_doctor_report(
        rule_report=rule_report,
        llm_draft=llm_draft,
        evidence=evidence,
        error_facts=[fact],
    )

    assert "AP_small=0.2 for small objects" in merged.evidence
    assert "mAP50-95=0.99 proves the model is excellent" not in merged.evidence
    assert "small_object_oversampling" in {item.action for item in merged.selected_actions}
    assert "increase_imgsz" not in {item.action for item in merged.selected_actions}
    assert merged.expected_improvement["AP_small"] == "increase; pilot_positive_delta required"
    assert merged.llm_merge["accepted_evidence"] == ["AP_small=0.2 for small objects"]
    assert merged.llm_merge["rejected_evidence"] == ["mAP50-95=0.99 proves the model is excellent"]


def test_next_round_uses_grounded_llm_doctor_draft(tmp_path: Path) -> None:
    """next_round should merge grounded LLM doctor drafts into the rule report."""
    task_path = _make_task(tmp_path)
    dataset_root = tmp_path / "dataset"
    (dataset_root / "images" / "train").mkdir(parents=True)
    data_yaml = dataset_root / "data.yaml"
    data_yaml.write_text("path: .\ntrain: images/train\nnames:\n  0: object\n", encoding="utf-8")
    run_root = tmp_path / "runs"
    orchestrator = LoopOrchestrator.initialize("exp-llm-doctor", task_path, data_yaml, run_root=run_root)
    ErrorFactStore(run_root).append(
        "exp-llm-doctor",
        [
            ErrorFact(
                run_id="exp-llm-doctor",
                candidate_id="baseline",
                node_id="node_baseline",
                fact_type="area_metric",
                subject="small",
                area="small",
                metric_name="ap_small",
                value=0.2,
                severity="high",
                action_candidates=["small_object_recipe"],
            )
        ],
    )
    orchestrator.context.artifact_path("llm_decision.yaml").write_text(
        yaml.safe_dump(
            {
                "doctor_report_draft": {
                    "evidence": [
                        "AP_small=0.2 for small objects",
                        "unverified private score=0.99",
                    ],
                    "why": ["AP_small=0.2 makes small-object actions the right first pilot."],
                }
            }
        ),
        encoding="utf-8",
    )

    payload = orchestrator.evidence.next_round_payload({})

    doctor_report = payload["doctor_report"]
    assert "AP_small=0.2 for small objects" in doctor_report["evidence"]
    assert "unverified private score=0.99" not in doctor_report["evidence"]
    assert doctor_report["llm_merge"]["used"] is True
    assert doctor_report["llm_merge"]["rejected_evidence"] == ["unverified private score=0.99"]
