"""Auto optimization loop driver tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from yolo_agent.agents.auto_optimization_loop import AutoOptimizationLoopDriver, assess_candidate_execution
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.loop_policy_evaluator import LoopPolicyEvaluation, LoopPolicyEvaluationReport
from yolo_agent.agents.optimize_runner import OptimizeRunner
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.error_facts import ErrorFact, ErrorFactStore
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.task_spec import MetricPriority, TaskSpec


def _make_dataset(root: Path) -> Path:
    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "img1.jpg").write_bytes(b"image")
    (label_dir / "img1.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "names:",
                "  0: object",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_assess_candidate_execution_splits_real_and_metadata_only_candidates(tmp_path: Path) -> None:
    """The auto loop must not fake-train metadata-only component proposals."""
    executable_candidate = CandidateConfig(
        candidate_id="safe_optimizer",
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
        action_domain="training",
        action_id="optimizer",
        train_overrides={"optimizer": "AdamW"},
    )
    executable_node = ExperimentNode(
        node_id="node_safe_optimizer",
        candidate_config=executable_candidate,
        data_version="coco2017",
    )
    executable_node.command_spec = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data=tmp_path / "data.yaml",
        project=tmp_path / "ultralytics",
        name="safe",
    )

    adapter_candidate = CandidateConfig(
        candidate_id="nwd_loss",
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
        components=["loss.bbox.nwd"],
    )
    adapter_node = ExperimentNode(
        node_id="node_nwd_loss",
        candidate_config=adapter_candidate,
        data_version="coco2017",
    )
    adapter_node.command_spec = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data=tmp_path / "data.yaml",
        project=tmp_path / "ultralytics",
        name="nwd",
    )

    advisory_candidate = CandidateConfig(
        candidate_id="postprocess",
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
        action_domain="postprocess",
        action_id="soft_nms",
        train_overrides={"postprocess": ["soft_nms"]},
    )
    advisory_node = ExperimentNode(
        node_id="node_postprocess",
        candidate_config=advisory_candidate,
        data_version="coco2017",
    )
    advisory_node.command_spec = CommandSpec.smoke(
        plan_path=tmp_path / "plan.yaml",
        data_path=tmp_path / "data.yaml",
        run_id="smoke",
    )

    report = LoopPolicyEvaluationReport(
        evaluations=[
            LoopPolicyEvaluation(
                policy_id="safe_optimizer",
                decision="accepted",
                candidate_config=executable_candidate,
                experiment_node=executable_node,
            ),
            LoopPolicyEvaluation(
                policy_id="nwd_loss",
                decision="accepted",
                candidate_config=adapter_candidate,
                experiment_node=adapter_node,
            ),
            LoopPolicyEvaluation(
                policy_id="postprocess",
                decision="accepted",
                candidate_config=advisory_candidate,
                experiment_node=advisory_node,
            ),
        ]
    )

    by_policy = {item.policy_id: item for item in assess_candidate_execution(report)}

    assert by_policy["safe_optimizer"].execution_class == "executable"
    assert by_policy["nwd_loss"].execution_class == "adapter_required"
    assert "component_adapter:loss.bbox.nwd" in by_policy["nwd_loss"].required_adapters
    assert by_policy["postprocess"].execution_class == "recommendation_only"


def test_auto_optimization_driver_stops_without_fake_executable_candidates(tmp_path: Path) -> None:
    """A dry auto round should produce artifacts and stop if all candidates need adapters."""
    data_yaml = _make_dataset(tmp_path / "dataset")
    run_root = tmp_path / "runs"
    task_path = run_root / "coco-yolo26n" / "task.yaml"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    TaskSpec(
        task_type="detect",
        scene="generic",
        class_names=["object"],
        primary_metric=MetricPriority(name="map50_95"),
    ).to_yaml(task_path)

    base = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="coco-yolo26n",
        run_root=run_root,
        profile="pilot",
        execute=False,
    )
    ErrorFactStore(run_root).append(
        base.run_id,
        [
            ErrorFact(
                run_id=base.run_id,
                candidate_id="yolo26n_coco_pilot",
                node_id="node_yolo26n_coco_pilot",
                dataset_version="coco2017",
                fact_type="area_metric",
                subject="small",
                area="small",
                metric_name="ap_small",
                value=0.1,
                severity="high",
                action_candidates=["small_object_recipe", "bbox_loss_recipe"],
            )
        ],
    )

    result = AutoOptimizationLoopDriver().run(
        base_run_dir=base.run_dir,
        auto_rounds=1,
        execute=False,
        executor="dry-run",
        max_steps=4,
    )

    child_dir = run_root / "coco-yolo26n-r1"
    assert result.rounds
    assert result.rounds[0].run_id == "coco-yolo26n-r1"
    assert result.rounds[0].auto_round_summary_path.exists()
    assert (child_dir / "artifacts" / "llm_decision.yaml").exists()
    assert (child_dir / "artifacts" / "policy_evaluation.yaml").exists()
    assert result.summary_path.exists()
    assert result.full_candidate_recommendations_path.exists()
    recommendations = yaml.safe_load(result.full_candidate_recommendations_path.read_text(encoding="utf-8-sig"))
    assert recommendations["full_run_started"] is False
    assert result.stopped_reason in {"no_executable_candidates", "requested_rounds_completed"}
