"""Diagnosis graph tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from yolo_agent.agents.diagnosis_graph import DiagnosisGraph, diagnosis_graph_from_error_facts
from yolo_agent.core.error_facts import ErrorFact, ErrorFactStore
from yolo_agent.agents.orchestrator import LoopOrchestrator


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

