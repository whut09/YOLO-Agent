"""COCO eval importer tests."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from yolo_agent.cli import main
from yolo_agent.agents.orchestrator import LoopOrchestrator
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.error_facts import ErrorFact, ErrorFactStore
from yolo_agent.tools.coco_error_importer import import_coco_eval_metrics, parse_coco_eval_metrics


def test_parse_coco_eval_stats_and_per_class_ap(tmp_path: Path) -> None:
    """Importer should parse COCO stats array and per-class AP facts."""
    eval_path = tmp_path / "coco_eval.json"
    eval_path.write_text(
        json.dumps(
            {
                "stats": [0.4, 0.61, 0.43, 0.21, 0.45, 0.58, 0.3, 0.5, 0.6, 0.25, 0.52, 0.66],
                "per_class_ap": {
                    "person": 0.52,
                    "traffic light": 0.31,
                },
                "per_class_ar": {
                    "traffic light": 0.22,
                },
            }
        ),
        encoding="utf-8",
    )

    metrics = parse_coco_eval_metrics(eval_path)

    assert metrics["coco_ap50_95"] == 0.4
    assert metrics["ap_small"] == 0.21
    assert metrics["ap_medium"] == 0.45
    assert metrics["ap_large"] == 0.58
    assert metrics["per_class_ap/person"] == 0.52
    assert metrics["per_class_ap/traffic light"] == 0.31
    assert metrics["per_class_ar/traffic light"] == 0.22


def test_import_coco_eval_writes_node_level_evidence(tmp_path: Path) -> None:
    """COCO eval import should force candidate/node-level evidence."""
    eval_path = tmp_path / "coco_eval.json"
    eval_path.write_text(
        json.dumps({"AP": 0.37, "AP_small": 0.2, "AP_medium": 0.4, "AP_large": 0.5}),
        encoding="utf-8",
    )
    store = EvidenceStore(tmp_path / "runs")

    result = import_coco_eval_metrics(
        eval_path=eval_path,
        evidence_store=store,
        run_id="exp001",
        candidate_id="yolo26n_seed1",
        node_id="node_yolo26n_seed1",
        dataset_version="coco2017",
    )
    evidence = store.load_run("exp001")

    assert result.metrics["ap_small"] == 0.2
    assert result.metrics["map50_95"] == 0.37
    assert {record.metric_name for record in evidence.metric_records} >= {"map50_95", "ap_small", "ap_medium", "ap_large"}
    assert all(record.candidate_id == "yolo26n_seed1" for record in evidence.metric_records)
    assert all(record.node_id == "node_yolo26n_seed1" for record in evidence.metric_records)
    assert all(record.validator == "coco_error_importer" for record in evidence.metric_records)


def test_import_coco_eval_writes_queryable_error_facts(tmp_path: Path) -> None:
    """COCO eval import should write facts that can drive next-round policies."""
    eval_path = tmp_path / "coco_eval.json"
    eval_path.write_text(
        json.dumps(
            {
                "stats": [0.4, 0.61, 0.43, 0.21, 0.45, 0.58, 0.3, 0.5, 0.6, 0.25, 0.52, 0.66],
                "per_class": [
                    {"class": "bottle", "ap": 0.18, "ar": 0.25},
                    {"class": "person", "ap": 0.52, "ar": 0.7},
                ],
                "false_negative_top_classes": [
                    {"category_id": 44, "name": "bottle", "false_negative": 80, "recall": 0.25}
                ],
                "localization_error_top_classes": [
                    {"category_id": 44, "name": "bottle", "localization_error": 12}
                ],
                "background_false_positive_top_classes": [
                    {"category_id": 1, "name": "person", "background_false_positive": 11}
                ],
                "class_confusion_pairs": {"cup->bottle": 9},
                "area_recall": {"small": 0.24, "medium": 0.55, "large": 0.72},
            }
        ),
        encoding="utf-8",
    )
    store = EvidenceStore(tmp_path / "runs")

    result = import_coco_eval_metrics(
        eval_path=eval_path,
        evidence_store=store,
        run_id="exp001",
        candidate_id="node_yolo26s_baseline_candidate",
        node_id="node_yolo26s_baseline",
    )
    facts = ErrorFactStore(tmp_path / "runs").index("exp001")

    assert result.error_facts_path == tmp_path / "runs" / "exp001" / "error_facts_by_node.jsonl"
    assert result.error_fact_count > 0
    assert facts.query(node_id="node_yolo26s_baseline", fact_type="class_low_ap", class_name="bottle")
    assert facts.query(fact_type="false_negative_heavy_class", class_name="bottle")[0].severity == "high"
    assert facts.query(fact_type="localization_heavy_class", class_name="bottle")[0].action_candidates == [
        "bbox_loss_recipe",
        "assigner_recipe",
        "label_box_audit",
    ]
    assert facts.query(fact_type="class_confusion_pair", subject="cup->bottle")
    assert facts.query(fact_type="background_false_positive_class", class_name="person")
    assert "small_object_recipe" in facts.action_candidates(node_id="node_yolo26s_baseline")


def test_loop_import_coco_eval_cli(tmp_path: Path) -> None:
    """loop import-coco-eval should attach official COCO metrics to a loop run."""
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
    eval_path = tmp_path / "coco_eval.json"
    eval_path.write_text(json.dumps({"stats": [0.5, 0.7, 0.55, 0.25, 0.44, 0.6]}), encoding="utf-8")

    assert main(
        [
            "loop",
            "init",
            "--run-id",
            "coco-eval-run",
            "--task",
            str(task_path),
            "--data",
            str(data_yaml),
            "--run-root",
            str(run_root),
            "--dataset-version",
            "coco2017",
        ]
    ) == 0
    assert main(
        [
            "loop",
            "import-coco-eval",
            "--run",
            str(run_root / "coco-eval-run"),
            "--eval",
            str(eval_path),
            "--candidate-id",
            "yolo26s_coco_baseline",
            "--node-id",
            "node_yolo26s_coco_baseline",
        ]
    ) == 0

    evidence = EvidenceStore(run_root).load_run("coco-eval-run")
    facts = ErrorFactStore(run_root).read("coco-eval-run")
    next_round_payload = LoopOrchestrator.from_run_dir(run_root / "coco-eval-run").evidence.next_round_payload({})

    assert any(record.metric_name == "ap_small" for record in evidence.metric_records)
    assert any(record.metric_name == "coco_ap50_95" for record in evidence.metric_records)
    assert any(fact.fact_type == "area_metric" for fact in facts)
    assert "small_object_recipe" in next_round_payload["error_fact_action_candidates"]
    assert next_round_payload["diagnosis_graph"]["findings"][0]["diagnosis_id"] == "small_object_ap_low"
    assert "bbox_area_histogram" in next_round_payload["diagnosis_graph_evidence_needed"]
    assert next_round_payload["proposal_mode"] == "pilot_only"
    assert next_round_payload["full_candidate_proposal_allowed"] is False
    assert next_round_payload["proposal_budget_profiles_allowed"] == ["debug", "pilot"]
    assert next_round_payload["proposal_budget_profiles_blocked"] == ["candidate_full"]
    assert next_round_payload["proposal_required_bindings"] == ["target_error_facts", "expected_improvement"]
    assert next_round_payload["current_round_focus"][0]["diagnosis_kind"] == "small_object_ap"
    assert "small_object_recipe" in next_round_payload["current_round_error_actions"]


def test_next_round_blocks_candidate_proposals_without_error_facts(tmp_path: Path) -> None:
    """Next-round planning must not emit full candidate readiness without COCO error facts."""
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
    orchestrator = LoopOrchestrator.initialize("no-error-facts", task_path, data_yaml, run_root=run_root)

    payload = orchestrator.evidence.next_round_payload({})

    assert payload["proposal_mode"] == "blocked"
    assert payload["status"] == "blocked_missing_error_facts"
    assert payload["pilot_candidate_proposal_allowed"] is False
    assert payload["full_candidate_proposal_allowed"] is False
    assert payload["proposal_budget_profiles_allowed"] == []
    assert payload["proposal_budget_profiles_blocked"] == ["candidate_full"]
    assert "missing_error_facts" in payload["error_delta_proposal_policy"]["rejection_reasons"]


def test_next_round_compares_parent_and_current_error_fact_delta(tmp_path: Path) -> None:
    """Next-round planning should focus actions on unresolved/regressed error facts."""
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        yaml.safe_dump(
            {
                "task_type": "detect",
                "scene": "generic",
                "class_names": ["bottle", "person"],
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
    data_yaml.write_text("path: .\ntrain: images/train\nnames:\n  0: bottle\n  1: person\n", encoding="utf-8")
    run_root = tmp_path / "runs"
    parent = LoopOrchestrator.initialize("parent-run", task_path, data_yaml, run_root=run_root)
    child = LoopOrchestrator.initialize("child-run", task_path, data_yaml, run_root=run_root)
    child.context.metadata["parent_run_id"] = "parent-run"
    child.context.to_yaml()
    store = ErrorFactStore(run_root)
    matched = {
        "dataset_manifest_sha256": "dataset-sha",
        "subset_manifest_sha256": "subset-sha",
        "seed": 1,
        "epochs": 10,
        "fidelity": "pilot_10",
        "batch_policy_hash": "batch-policy",
        "ultralytics_version": "9.0.0",
        "imgsz": 640,
        "eval_protocol_hash": "eval-protocol",
    }
    store.append(
        "parent-run",
        [
            ErrorFact(
                run_id="parent-run",
                candidate_id="baseline",
                node_id="node_baseline",
                fact_type="area_metric",
                subject="small",
                area="small",
                metric_name="ap_small",
                value=0.2,
                severity="high",
                action_candidates=["small_object_recipe"],
                evidence_role="baseline_reference",
                **matched,
            ),
            ErrorFact(
                run_id="parent-run",
                candidate_id="baseline",
                node_id="node_baseline",
                fact_type="localization_heavy_class",
                subject="bottle",
                class_name="bottle",
                count=10,
                severity="medium",
                action_candidates=["bbox_loss_recipe"],
                evidence_role="baseline_reference",
                **matched,
            ),
            ErrorFact(
                run_id="parent-run",
                candidate_id="baseline",
                node_id="node_baseline",
                fact_type="background_false_positive_class",
                subject="person",
                class_name="person",
                count=20,
                severity="medium",
                action_candidates=["hard_negative_mining"],
                evidence_role="baseline_reference",
                **matched,
            ),
        ],
    )
    store.append(
        "child-run",
        [
            ErrorFact(
                run_id="child-run",
                candidate_id="candidate",
                node_id="node_candidate",
                fact_type="area_metric",
                subject="small",
                area="small",
                metric_name="ap_small",
                value=0.33,
                severity="medium",
                action_candidates=["small_object_recipe"],
                **matched,
            ),
            ErrorFact(
                run_id="child-run",
                candidate_id="candidate",
                node_id="node_candidate",
                fact_type="localization_heavy_class",
                subject="bottle",
                class_name="bottle",
                count=14,
                severity="high",
                action_candidates=["bbox_loss_recipe"],
                **matched,
            ),
            ErrorFact(
                run_id="child-run",
                candidate_id="candidate",
                node_id="node_candidate",
                fact_type="background_false_positive_class",
                subject="person",
                class_name="person",
                count=20,
                severity="medium",
                action_candidates=["hard_negative_mining"],
                **matched,
            ),
        ],
    )

    payload = child.evidence.next_round_payload({})
    delta = payload["error_fact_delta"]

    assert any(item["subject"] == "small" and item["trend"] == "improved" for item in delta["improved_errors"])
    assert any(item["subject"] == "bottle" and item["trend"] == "regressed" for item in delta["regressed_errors"])
    assert any(item["subject"] == "person" and item["trend"] == "unchanged" for item in delta["unchanged_errors"])
    assert "small_object_recipe" in delta["effective_action_candidates"]
    assert payload["error_delta_proposal_policy"]["focus_source"] == "parent_current_error_delta"
    assert payload["proposal_mode"] == "pilot_only"
    assert payload["proposal_budget_profiles_allowed"] == ["debug", "pilot"]
    assert "small_object_recipe" not in payload["current_round_error_actions"]
    assert "bbox_loss_recipe" in payload["next_error_actions"]
    assert "hard_negative_mining" in payload["next_error_actions"]
    assert "bbox_loss_recipe" in payload["current_round_error_actions"]
    assert "hard_negative_mining" in payload["current_round_error_actions"]
