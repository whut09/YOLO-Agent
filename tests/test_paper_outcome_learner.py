from __future__ import annotations

from pathlib import Path

from yolo_agent.agents.paper_outcome_learner import PaperOutcomeLearner, PaperRecipeOutcome
from yolo_agent.agents.policy_learner import PolicyLearner
from yolo_agent.core.policy_memory import PolicyMemoryStore, stable_negative_action_reasons


def _outcome(**updates) -> PaperRecipeOutcome:
    data = {
        "run_id": "run-1",
        "recipe_id": "paper-small-object",
        "recipe_version": "v1",
        "paper_ids": ["paper-1"],
        "component_ids": ["sampling.small_object"],
        "component_versions": {"sampling.small_object": "1.0"},
        "changed_variable": "data.sampler",
        "before_value": "uniform",
        "after_value": "small_object_weighted",
        "detector_family": "yolo26",
        "model_family": "yolo26n",
        "dataset_version": "coco2017",
        "dataset_signature": "coco-manifest-a",
        "protocol_hash": "protocol-640",
        "snapshot_hash": "snapshot-a",
        "fidelity": "pilot_10",
        "seed": 42,
        "paper_prior_effect": {"ap_small": "+1.2 paper claim"},
        "pilot_3_delta": 0.002,
        "pilot_10_delta": 0.006,
        "target_error_fact_delta": {"ap_small": 0.012, "small_fn": -12.0},
        "latency_delta": 0.1,
        "model_size_delta": 0.0,
        "paired_bootstrap_ci": (0.001, 0.011),
        "implementation_cost": {"engineer_hours": 2.0},
        "candidate_id": "candidate-1",
        "node_id": "node-1",
    }
    data.update(updates)
    return PaperRecipeOutcome.model_validate(data)


def test_single_seed_keeps_paper_prior_separate_and_marks_possible(tmp_path: Path) -> None:
    result = PaperOutcomeLearner(PolicyMemoryStore(tmp_path)).learn(_outcome())
    record = result.record
    assert result.local_posterior_status == "possible"
    assert record.confidence == "low"
    assert record.paper_prior_effect == {"ap_small": "+1.2 paper claim"}
    assert record.pilot_10_delta == 0.006
    assert record.effect_delta == 0.006
    assert record.action_fingerprint.paper_ids == ["paper-1"]
    assert record.action_fingerprint.component_ids == ["sampling.small_object"]
    assert record.action_fingerprint.snapshot_hash == "snapshot-a"


def test_multi_seed_with_paired_intervals_is_confirmed(tmp_path: Path) -> None:
    result = PolicyLearner(PolicyMemoryStore(tmp_path)).learn_paper_outcome(_outcome(
        fidelity="full",
        full_delta=0.021,
        seed_count=3,
        cross_seed_ci=(0.014, 0.028),
    ))
    assert result.local_posterior_status == "confirmed"
    assert result.record.confidence == "high"
    assert result.record.full_delta == 0.021
    assert "multi-seed" in result.record.confidence_reason


def test_duplicate_experiment_is_idempotent(tmp_path: Path) -> None:
    learner = PaperOutcomeLearner(PolicyMemoryStore(tmp_path))
    first = learner.learn(_outcome())
    second = learner.learn(_outcome())
    assert first.appended is True
    assert second.appended is False
    assert second.duplicate is True
    assert len(PolicyMemoryStore(tmp_path).read()) == 1


def test_dataset_and_snapshot_have_distinct_local_posteriors(tmp_path: Path) -> None:
    learner = PaperOutcomeLearner(PolicyMemoryStore(tmp_path))
    first = learner.learn(_outcome()).record
    other_dataset = learner.learn(_outcome(
        run_id="run-2",
        dataset_version="custom-v1",
        dataset_signature="custom-manifest",
        node_id="node-2",
    )).record
    other_snapshot = learner.learn(_outcome(
        run_id="run-3",
        snapshot_hash="snapshot-b",
        node_id="node-3",
    )).record
    assert first.action_fingerprint.posterior_sha256 == other_dataset.action_fingerprint.posterior_sha256
    assert first.action_fingerprint.posterior_sha256 != other_snapshot.action_fingerprint.posterior_sha256
    coco = PolicyMemoryStore(tmp_path).summarize_local(
        action="paper-small-object",
        dataset_signature="coco-manifest-a",
    )
    custom = PolicyMemoryStore(tmp_path).summarize_local(
        action="paper-small-object",
        dataset_signature="custom-manifest",
    )
    assert sum(item.record_count for item in coco) == 2
    assert sum(item.record_count for item in custom) == 1


def test_one_failure_lowers_local_confidence_without_banning_family(tmp_path: Path) -> None:
    store = PolicyMemoryStore(tmp_path)
    result = PaperOutcomeLearner(store).learn(_outcome(
        pilot_10_delta=-0.004,
        failure_reason="pilot regression",
    ))
    assert result.local_posterior_status == "failed"
    assert result.record.confidence == "low"
    assert result.record.failure_reason == "pilot regression"
    assert stable_negative_action_reasons(store.read(), {"paper-small-object"}) == []


def test_pilot_full_correlation_uses_same_dataset_protocol_and_snapshot(tmp_path: Path) -> None:
    learner = PaperOutcomeLearner(PolicyMemoryStore(tmp_path))
    learner.learn(_outcome(
        run_id="pair-1",
        fidelity="full",
        pilot_10_delta=0.004,
        full_delta=0.008,
        node_id="pair-node-1",
    ))
    result = learner.learn(_outcome(
        run_id="pair-2",
        fidelity="full",
        pilot_10_delta=0.008,
        full_delta=0.016,
        node_id="pair-node-2",
    ))
    assert result.record.pilot_full_correlation == 1.0
