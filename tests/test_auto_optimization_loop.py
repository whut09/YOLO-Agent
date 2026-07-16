"""Auto optimization loop driver tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from yolo_agent.agents.auto_optimization_loop import (
    AutoOptimizationLoopDriver,
    AutoRoundResult,
    _is_inheritable_metric_record,
    assess_candidate_execution,
)
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.loop_policy_evaluator import LoopPolicyEvaluation, LoopPolicyEvaluationReport
from yolo_agent.agents.llm_decision_advisor import LLMDecisionAdvisorResult
from yolo_agent.agents.optimize_runner import OptimizeRunner
from yolo_agent.agents.orchestrator import LoopOrchestrator
from yolo_agent.agents.policy_stage_runner import _synthetic_executable_pilot_policies
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


def test_verified_inherited_latency_can_continue_across_rounds() -> None:
    """Verified lineage metrics should not disappear after one child generation."""
    assert _is_inheritable_metric_record(
        {
            "metric_name": "latency_ms",
            "value": 44.27,
            "verified": True,
            "source": "inherited:coco-yolo26n-r1:benchmark",
        }
    )


def test_synthetic_pilot_uses_next_untried_parameter_variant(tmp_path: Path) -> None:
    """A tried action should advance its finite parameter ladder instead of disappearing."""
    data_yaml = _make_dataset(tmp_path / "dataset")
    result = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="ladder-run",
        run_root=tmp_path / "runs",
        profile="pilot",
        execute=False,
    )
    context = LoopOrchestrator.from_run_dir(result.run_dir).context
    policies = _synthetic_executable_pilot_policies(
        context,
        focus_items=[
            {
                "fact_type": "localization_heavy_class",
                "class_name": "person",
                "action_candidates": ["bbox_loss_recipe"],
            }
        ],
        allowed_actions={"bbox_loss_recipe", "increase_box_loss_gain"},
        tried_actions={"increase_box_loss_gain"},
        existing_policy_ids=set(),
    )

    by_id = {policy.policy_id: policy for policy in policies}
    assert "next_training_tune_box_loss_gain_8_25" in by_id
    assert by_id["next_training_tune_box_loss_gain_8_25"].train_overrides["box"] == 8.25


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


def test_auto_optimization_driver_stops_without_fake_executable_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A dry auto round should produce artifacts and stop if all candidates need adapters."""
    llm_calls = 0

    class FakeAdvisor:
        def propose(self, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal llm_calls
            llm_calls += 1
            assert kwargs["inherited_context"]["decision_context_hash"]
            return LLMDecisionAdvisorResult(
                status="failed",
                provider="test",
                model="test-model",
                warnings=["forced_test_fallback"],
            )

    monkeypatch.setattr("yolo_agent.agents.policy_stage_runner.LLMDecisionAdvisor", lambda: FakeAdvisor())
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
    assert (child_dir / "artifacts" / "paper_recipe_plan.yaml").exists()
    assert (child_dir / "artifacts" / "component_compatibility.yaml").exists()
    assert (child_dir / "artifacts" / "decision_ledger.jsonl").exists()
    assert (child_dir / "artifacts" / "policy_evaluation.yaml").exists()
    assert result.summary_path.exists()
    assert result.full_candidate_recommendations_path.exists()
    recommendations = yaml.safe_load(result.full_candidate_recommendations_path.read_text(encoding="utf-8-sig"))
    assert recommendations["full_run_started"] is False
    assert result.stopped_reason in {"no_executable_candidates", "requested_rounds_completed"}
    paper_plan = yaml.safe_load(
        (child_dir / "artifacts" / "paper_recipe_plan.yaml").read_text(encoding="utf-8-sig")
    )
    assert paper_plan["paper_claims_are_prior_only"] is True
    assert paper_plan["llm_status"] == "deferred_to_unified_decision_bundle"
    assert "recipe_critic_reports" in paper_plan
    assert "executable_pilot_policies" in paper_plan
    assert llm_calls == 1
    assert "Paper Intelligence" in result.summary_path.read_text(encoding="utf-8-sig")


def test_auto_optimization_driver_generates_executable_mosaic_pilot_from_background_fp(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Background FP facts should unlock a real Ultralytics pilot instead of stopping."""
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
                fact_type="background_false_positive_class",
                subject="person",
                class_name="person",
                count=1200,
                severity="high",
                action_candidates=[
                    "hard_negative_mining",
                    "background_only_sampling",
                    "precision_threshold_tuning",
                ],
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

    assert result.stopped_reason == "requested_rounds_completed"
    assert result.rounds[0].executable_count >= 1
    assessments = {item.policy_id: item for item in result.rounds[0].candidate_assessments}
    mosaic = assessments["next_augmentation_reduce_mosaic_strength"]
    assert mosaic.execution_class == "executable"
    assert "model=yolo26n.pt" in mosaic.command
    assert "mosaic=0.2" in mosaic.command
    assert "imgsz=640" in mosaic.command

    def fail_if_round_is_reexecuted(*args: object, **kwargs: object) -> object:
        raise AssertionError("completed auto round should be reused, not re-executed")

    monkeypatch.setattr(AutoOptimizationLoopDriver, "_run_one_round", fail_if_round_is_reexecuted)
    reused = AutoOptimizationLoopDriver().run(
        base_run_dir=base.run_dir,
        auto_rounds=1,
        execute=False,
        executor="dry-run",
        max_steps=4,
    )

    assert reused.rounds[0].run_id == "coco-yolo26n-r1"
    assert reused.rounds[0].status == "completed"
    assert reused.stopped_reason == "requested_rounds_completed"


def test_auto_optimization_execute_does_not_reuse_dry_run_round(tmp_path: Path, monkeypatch) -> None:
    """Execute mode must not treat a dry-run auto round as trained evidence."""
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
                fact_type="background_false_positive_class",
                subject="person",
                class_name="person",
                count=1200,
                severity="high",
                action_candidates=["hard_negative_mining"],
            )
        ],
    )
    AutoOptimizationLoopDriver().run(
        base_run_dir=base.run_dir,
        auto_rounds=1,
        execute=False,
        executor="dry-run",
        max_steps=4,
    )

    calls: list[str] = []

    def fake_execute_round(self: AutoOptimizationLoopDriver, **kwargs: object) -> AutoRoundResult:
        child = kwargs["child"]
        calls.append(child.context.run_id)
        return AutoRoundResult(
            round_index=1,
            run_id=child.context.run_id,
            run_dir=child.context.run_dir,
            parent_run_id=kwargs["parent"].context.run_id,
            status="completed",
            stop_reason="round_completed",
            auto_round_summary_path=child.context.artifact_path("auto_round_summary.yaml"),
        )

    monkeypatch.setattr(AutoOptimizationLoopDriver, "_run_one_round", fake_execute_round)
    executed = AutoOptimizationLoopDriver().run(
        base_run_dir=base.run_dir,
        auto_rounds=1,
        execute=True,
        executor="ultralytics-train",
        max_steps=4,
    )

    assert calls == ["coco-yolo26n-r1"]
    assert executed.rounds[0].run_id == "coco-yolo26n-r1"


def test_auto_optimization_execute_continues_after_completed_executed_round(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Execute mode should treat auto_rounds as additional work after completed executed rounds."""
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
                fact_type="background_false_positive_class",
                subject="person",
                class_name="person",
                count=1200,
                severity="high",
                action_candidates=["hard_negative_mining"],
            )
        ],
    )
    child_dir = run_root / "coco-yolo26n-r1"
    (child_dir / "artifacts").mkdir(parents=True)
    completed = AutoRoundResult(
        round_index=1,
        run_id="coco-yolo26n-r1",
        run_dir=child_dir,
        parent_run_id="coco-yolo26n",
        status="completed",
        stop_reason="round_completed",
        auto_round_summary_path=child_dir / "artifacts" / "auto_round_summary.yaml",
        training_loop={
            "run_id": "coco-yolo26n-r1",
            "profile": "pilot",
            "executor": "ultralytics-train",
            "auto_import": True,
            "max_steps": 1,
            "steps": [],
            "queue_counts": {"completed": 1},
            "stopped_reason": "complete",
            "completed": True,
        },
    )
    completed.auto_round_summary_path.write_text(
        yaml.safe_dump(completed.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )

    calls: list[str] = []

    def fake_execute_round(self: AutoOptimizationLoopDriver, **kwargs: object) -> AutoRoundResult:
        child = kwargs["child"]
        calls.append(child.context.run_id)
        return AutoRoundResult(
            round_index=2,
            run_id=child.context.run_id,
            run_dir=child.context.run_dir,
            parent_run_id=kwargs["parent"].context.run_id,
            status="completed",
            stop_reason="round_completed",
            auto_round_summary_path=child.context.artifact_path("auto_round_summary.yaml"),
        )

    monkeypatch.setattr(AutoOptimizationLoopDriver, "_run_one_round", fake_execute_round)
    result = AutoOptimizationLoopDriver().run(
        base_run_dir=base.run_dir,
        auto_rounds=1,
        execute=True,
        executor="ultralytics-train",
        max_steps=4,
    )

    assert calls == ["coco-yolo26n-r2"]
    assert result.rounds[0].round_index == 2
    assert result.rounds[0].run_id == "coco-yolo26n-r2"
