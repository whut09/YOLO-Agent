"""Matched-control scheduling and result lifecycle regression tests."""

from __future__ import annotations

import pytest

from yolo_agent.agents.asha_scheduler import ASHAScheduler
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.experiment_graph import ExperimentNode, MetricEvidence
from yolo_agent.core.matched_baseline import (
    assess_matched_control_plan,
    assess_matched_control_result,
)


def _node(
    candidate_id: str,
    *,
    control: bool = False,
    protocol: str = "protocol-a",
    dataset: str = "dataset-a",
    split: str = "val2017",
    fidelity: str = "pilot_3",
    seed: int = 42,
    imgsz: int = 640,
) -> ExperimentNode:
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data="coco.yaml",
        project="runs/ultralytics",
        name=candidate_id,
        epochs=3,
        imgsz=imgsz,
        seed=seed,
        metadata={
            "dataset_manifest_sha256": dataset,
            "split": split,
            "fidelity": fidelity,
            "seed_policy": str(seed),
            "protocol_hash": protocol,
            "matched_baseline_control": control,
            "matched_control_plan_required": True,
        },
    )
    return ExperimentNode(
        node_id=f"node-{candidate_id}",
        candidate_config=CandidateConfig(
            candidate_id=candidate_id,
            base_model="yolo26n.pt",
            scale="n",
            framework="ultralytics",
        ),
        data_version="coco2017",
        seed=seed,
        command=command.display(),
        command_spec=command,
        changed_variables={} if control else {"loss.weight": candidate_id},
    )


def _metric(node: ExperimentNode, value: float, *, control: bool) -> MetricEvidence:
    metadata = node.command_spec.metadata
    return MetricEvidence(
        candidate_id=node.candidate_config.candidate_id,
        node_id=node.node_id,
        run_id="paired-run",
        origin_run_id="paired-run",
        evidence_role="baseline_reference" if control else "current_observation",
        dataset_manifest_sha256=str(metadata["dataset_manifest_sha256"]),
        subset_manifest_sha256="pilot-subset",
        split=str(metadata["split"]),
        protocol_hash=str(metadata["protocol_hash"]),
        eval_protocol_hash="coco-eval",
        seed=node.seed,
        fidelity=str(metadata["fidelity"]),
        epochs=3,
        batch_policy_hash="batch-policy",
        ultralytics_version="test",
        imgsz=640,
        metric_name="map50_95",
        value=value,
        source="test",
        validator="test",
    )


def test_complete_control_plan_registers_without_historical_baseline_artifact() -> None:
    scheduler = ASHAScheduler.create("fresh-pair")
    candidate = _node("candidate")
    control = _node("baseline", control=True)

    trial = scheduler.register_trial(
        trial_id="candidate",
        candidate_id="candidate",
        source_run_id="fresh-pair",
        source_node=candidate,
        baseline_control_node=control,
    )

    assert trial.matched_control_plan_ready is True
    assert trial.matched_control_result_ready is False
    assert trial.matched_control_plan is not None
    assert trial.status == "waiting"


@pytest.mark.parametrize(
    ("field", "value", "blocker"),
    [
        ("protocol", "protocol-b", "matched_control_protocol_hash_mismatch"),
        ("dataset", "dataset-b", "matched_control_dataset_manifest_hash_mismatch"),
        ("split", "test2017", "matched_control_split_mismatch"),
        ("fidelity", "pilot_10", "matched_control_fidelity_mismatch"),
        ("seed", 43, "matched_control_seed_policy_mismatch"),
        ("imgsz", 1280, "matched_control_imgsz_must_be_640"),
    ],
)
def test_mismatched_control_plan_cannot_register(
    field: str,
    value: str | int,
    blocker: str,
) -> None:
    scheduler = ASHAScheduler.create("bad-pair")
    candidate = _node("candidate")
    control = _node("baseline", control=True, **{field: value})

    with pytest.raises(ValueError, match=blocker):
        scheduler.register_trial(
            trial_id="candidate",
            candidate_id="candidate",
            source_run_id="bad-pair",
            source_node=candidate,
            baseline_control_node=control,
        )


def test_one_sided_results_never_produce_paired_delta() -> None:
    candidate = _node("candidate")
    control = _node("baseline", control=True)
    plan = assess_matched_control_plan(candidate, control).plan
    assert plan is not None

    baseline_only = assess_matched_control_result(plan, [_metric(control, 0.39, control=True)])
    candidate_only = assess_matched_control_result(plan, [_metric(candidate, 0.40, control=False)])

    assert baseline_only.baseline_completed is True
    assert baseline_only.candidate_completed is False
    assert baseline_only.matched_control_result_ready is False
    assert baseline_only.paired_delta is None
    assert candidate_only.candidate_completed is True
    assert candidate_only.baseline_completed is False
    assert candidate_only.matched_control_result_ready is False
    assert candidate_only.paired_delta is None


def test_both_exact_results_make_paired_delta_ready() -> None:
    candidate = _node("candidate")
    control = _node("baseline", control=True)
    plan = assess_matched_control_plan(candidate, control).plan
    assert plan is not None

    result = assess_matched_control_result(
        plan,
        [_metric(control, 0.39, control=True), _metric(candidate, 0.41, control=False)],
    )

    assert result.matched_control_result_ready is True
    assert result.paired_delta is not None
    assert result.paired_delta.paired_delta == pytest.approx(0.02)


def test_control_plan_is_shareable_only_with_the_same_protocol() -> None:
    control = _node("baseline", control=True)
    first = assess_matched_control_plan(_node("candidate-a"), control).plan
    second = assess_matched_control_plan(_node("candidate-b"), control).plan
    assert first is not None and second is not None

    assert first.plan_hash != second.plan_hash
    assert first.protocol_fingerprint == second.protocol_fingerprint


def test_baseline_failure_isolated_by_protocol_fingerprint() -> None:
    scheduler = ASHAScheduler.create("failure-isolation")
    plans = []
    for suffix, protocol in (("a", "protocol-a"), ("b", "protocol-b")):
        trial = scheduler.register_trial(
            trial_id=f"candidate-{suffix}",
            candidate_id=f"candidate-{suffix}",
            source_run_id="failure-isolation",
            source_node=_node(f"candidate-{suffix}", protocol=protocol),
            baseline_control_node=_node(f"baseline-{suffix}", control=True, protocol=protocol),
        )
        assert trial.matched_control_plan is not None
        plans.append(trial.matched_control_plan)

    affected = scheduler.fail_matched_control(plans[0].protocol_fingerprint, "executor_failed")

    assert affected == ["candidate-a"]
    assert scheduler.study.trial("candidate-a").status == "needs_evidence"
    assert scheduler.study.trial("candidate-b").status == "waiting"
