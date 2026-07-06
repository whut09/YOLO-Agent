"""Error-driven closed-loop tests."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.agents.error_driven_loop import ErrorDrivenLoopEngine
from yolo_agent.agents.error_to_action import DetectionErrorObservation
from yolo_agent.core.schemas import DeploymentConstraints
from yolo_agent.core.task_spec import MetricPriority, TaskSpec
from yolo_agent.tools.dataset_stats import DatasetHealth, DatasetReport


def _task() -> TaskSpec:
    return TaskSpec(
        task_type="detect",
        scene="infrared_small_target",
        class_names=["target"],
        primary_metric=MetricPriority(name="recall"),
        secondary_metrics=[MetricPriority(name="map50_95"), MetricPriority(name="latency_ms", goal="minimize")],
        max_latency_ms=20,
        max_model_size_mb=20,
    )


def _dataset_report() -> DatasetReport:
    return DatasetReport(
        data_yaml=Path("data.yaml"),
        dataset_root=Path("."),
        scene="infrared_small_target",
        image_count=100,
        label_count=50,
        class_distribution={"target": 50},
        object_size_ratio={"small": 0.8, "medium": 0.2, "large": 0.0},
        empty_label_images=0,
        dataset_health=DatasetHealth(
            score=55,
            problems=["severe_small_object_bias", "missing_hard_backgrounds", "annotation_noise"],
        ),
        potential_issues=["1 images are missing label files."],
    )


def test_error_driven_loop_answers_diagnosis_and_next_round_questions() -> None:
    """Closed loop should compose diagnosis, recipes, policies, and evidence requirements."""
    report = ErrorDrivenLoopEngine().run(
        task_spec=_task(),
        dataset_report=_dataset_report(),
        detection_errors=[
            DetectionErrorObservation(error_type="small_object_miss", count=6, severity="high"),
            DetectionErrorObservation(error_type="background_confusion", count=2, severity="medium"),
        ],
        deployment=DeploymentConstraints(target="edge", max_latency_ms=20, preferred_export="onnx"),
    )

    categories = {diagnosis.category for diagnosis in report.diagnostics}
    assert {"data", "annotation", "model_capacity", "loss_assigner_head", "postprocess", "deployment"} <= categories
    assert "small_object_localization" in {item.recipe_id for item in report.optimization_recipes.recommendations}
    assert "small_object_oversampling" in {action.action_type for action in report.sampling_policy.actions}
    assert "sahi_slicing" in report.postprocess_policy.ids
    assert "switch_to_nwd_loss" in {item.action.id for item in report.action_policy.recommendations}

    changed = report.next_round.changed_variables
    assert "bbox_loss" in changed
    assert "head_component" in changed
    assert "assigner" in changed
    assert "postprocess" in changed
    assert "data_action" in changed
    assert "label_action" in changed
    assert "augmentation_policy" in changed
    assert "training_action" in changed
    assert "record_dataset_version" in report.next_round.guardrails
    assert "latency_ms" in report.next_round.evidence_required
    assert report.evidence_status["latency_ms"] == "missing"


def test_error_driven_loop_candidate_policies_are_single_variable() -> None:
    """Next-round candidate policies should keep variables separated for ablation."""
    report = ErrorDrivenLoopEngine().run(
        task_spec=_task(),
        dataset_report=_dataset_report(),
        detection_errors=[DetectionErrorObservation(error_type="small_object_miss", count=3, severity="high")],
    )

    policies = {policy.policy_id: policy for policy in report.next_round.candidate_policies}
    assert "next_bbox_loss_loss_bbox_nwd" in policies
    assert policies["next_bbox_loss_loss_bbox_nwd"].components == ["loss.bbox.nwd"]
    assert policies["next_head_component_head_p2_small_object"].components == ["head.p2_small_object"]
    assert policies["next_imgsz_960"].train_overrides == {"imgsz": 960}
    assert policies["next_postprocess_policy"].components == []
    assert "postprocess" in policies["next_postprocess_policy"].train_overrides


def test_error_driven_loop_competes_data_label_model_postprocess_actions() -> None:
    """Background false positives should create non-model actions as first-class policies."""
    report = ErrorDrivenLoopEngine().run(
        task_spec=_task(),
        dataset_report=_dataset_report(),
        detection_errors=[DetectionErrorObservation(error_type="background_confusion", count=8, severity="high")],
    )

    policies = {policy.policy_id: policy for policy in report.next_round.candidate_policies}
    domains = {policy.action_domain for policy in policies.values()}

    assert {"data", "augmentation", "postprocess", "label", "training"} <= domains
    assert policies["next_data_hard_negative_sampling"].action_domain == "data"
    assert policies["next_label_check_missing_labels"].action_domain == "label"
    assert policies["next_augmentation_reduce_mosaic_strength"].action_domain == "augmentation"
    assert policies["next_training_increase_focal_loss_gamma"].action_domain == "training"
    assert "hard_negative_sampling" in report.next_round.changed_variables["data_action"]
    assert "check_missing_labels" in report.next_round.changed_variables["label_action"]


def test_error_driven_loop_blocks_higher_imgsz_when_fixed_baseline_is_set() -> None:
    """Fixed-baseline COCO/YOLO26 loops should not propose larger input size."""
    report = ErrorDrivenLoopEngine().run(
        task_spec=_task(),
        dataset_report=_dataset_report(),
        detection_errors=[DetectionErrorObservation(error_type="small_object_miss", count=3, severity="high")],
        fixed_imgsz=640,
    )

    policies = {policy.policy_id: policy for policy in report.next_round.candidate_policies}
    assert "next_imgsz_960" not in policies
    assert any("blocked_imgsz_increase" in guardrail for guardrail in report.next_round.guardrails)
    assert "do_not_increase_imgsz_for_baseline_comparison" in report.next_round.guardrails
