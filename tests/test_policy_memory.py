"""Policy memory and learner tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.agents.policy_learner import PolicyLearner
from yolo_agent.agents.policy_memory_context import build_policy_memory_context
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.policy_memory import ActionFingerprint, PolicyMemoryRecord, PolicyMemoryStore
from tests.paired_result_helpers import verified_paired_result


def test_policy_memory_store_appends_idempotently_and_queries(tmp_path: Path) -> None:
    """Policy memory should be append-only but skip duplicate semantic records."""
    store = PolicyMemoryStore(tmp_path / "runs")
    record = PolicyMemoryRecord(
        run_id="child",
        parent_run_id="parent",
        dataset_version="coco2017",
        action="loss.bbox.nwd",
        target="area_metric:small:ap_small",
        metric_name="ap_small",
        before=0.214,
        after=0.229,
        delta=0.015,
        candidate_id="candidate_nwd",
        node_id="node_nwd",
        changed_variables={"bbox_loss": ["loss.bbox.nwd"]},
    )

    first = store.append([record])
    second = store.append([record])

    assert len(first) == 1
    assert second == []
    assert len(store.read()) == 1
    assert store.query(action="loss.bbox.nwd", metric_name="ap_small")[0].delta == 0.015
    assert store.query(action="loss.bbox.nwd", min_confidence="medium") == []


def test_policy_learner_records_action_effect_with_cost(tmp_path: Path) -> None:
    """Learner should convert comparable error deltas into action-effect memory."""
    evidence_store = EvidenceStore(tmp_path / "runs")
    evidence_store.log_candidate_metrics(
        "child",
        candidate_id="baseline",
        node_id="node_baseline",
        metrics={"latency_ms": 12.0, "model_size_mb": 5.0, "ap_small": 0.214},
        dataset_version="coco2017",
        evidence_role="baseline_reference",
        dataset_manifest_sha256="dataset",
        subset_manifest_sha256="subset",
        seed=1,
        epochs=10,
        fidelity="pilot_10",
        batch_policy_hash="batch",
        ultralytics_version="9.0.0",
        imgsz=640,
        eval_protocol_hash="eval",
    )
    evidence_store.log_candidate_metrics(
        "child",
        candidate_id="candidate_nwd",
        node_id="node_nwd_seed1",
        metrics={"latency_ms": 13.0, "model_size_mb": 5.5, "ap_small": 0.229},
        dataset_version="coco2017",
        dataset_manifest_sha256="dataset",
        subset_manifest_sha256="subset",
        seed=1,
        epochs=10,
        fidelity="pilot_10",
        batch_policy_hash="batch",
        ultralytics_version="9.0.0",
        imgsz=640,
        eval_protocol_hash="eval",
    )
    error_delta = {
        "improved_errors": [
            {
                "trend": "improved",
                "fact_type": "area_metric",
                "subject": "small",
                "area": "small",
                "metric_name": "ap_small",
                "parent_value": 0.214,
                "current_value": 0.229,
                "delta": 0.015,
                "action_candidates": ["small_object_recipe", "sahi_or_tiling_eval"],
                "candidate_id": "candidate_nwd",
                    "node_id": "node_nwd_seed1",
                    "matched_control_hash": "control-hash",
            }
        ],
        "regressed_errors": [],
        "unchanged_errors": [],
    }
    learner = PolicyLearner(PolicyMemoryStore(tmp_path / "runs"))

    records = learner.learn_from_error_delta(
        run_id="child",
        parent_run_id="parent",
        dataset_version="coco2017",
        error_delta=error_delta,
        current_evidence=evidence_store.load_run("child"),
        parent_evidence=None,
        changed_variables={"bbox_loss": ["loss.bbox.nwd"]},
        scenario="generic",
        paired_result=verified_paired_result(
            candidate_id="candidate_nwd",
            node_id="node_nwd_seed1",
            delta=0.015,
            target_improved=True,
            target_baseline_value=0.214,
            target_delta=0.015,
            latency_baseline=12.0,
            latency_delta=1.0,
            model_size_baseline=5.0,
            model_size_delta=0.5,
        ),
    )

    assert len(records) == 2
    primary = next(item for item in records if item.target == "metric:map50_95")
    assert primary.metric_name == "map50_95"
    assert primary.delta == 0.015
    assert primary.source == "paired_metric_delta"
    record = next(item for item in records if item.target == "area_metric:small:ap_small")
    assert record.action == "loss.bbox.nwd"
    assert record.target == "area_metric:small:ap_small"
    assert record.before == 0.214
    assert record.after == pytest.approx(0.229)
    assert record.delta == 0.015
    assert record.effect_delta == 0.015
    assert record.inferred_action is False
    assert record.confidence == "low"
    assert record.cost.latency_delta_pct == 8.333333
    assert record.cost.model_size_delta_pct == 10.0

    summary = PolicyMemoryStore(tmp_path / "runs").summarize(action="loss.bbox.nwd")
    assert summary[0].mean_effect_delta == 0.015
    assert summary[0].mean_latency_delta_pct == 8.333333


def test_policy_learner_marks_action_candidates_as_inferred(tmp_path: Path) -> None:
    """Without changed variables, learner must not pretend candidates were executed."""
    error_delta = {
        "improved_errors": [
            {
                "trend": "improved",
                "fact_type": "area_metric",
                "subject": "small",
                "area": "small",
                "metric_name": "ap_small",
                "parent_value": 0.2,
                "current_value": 0.21,
                "delta": 0.01,
                "action_candidates": ["small_object_recipe", "sahi_or_tiling_eval"],
                "candidate_id": "candidate",
                "node_id": "node",
            }
        ],
    }

    records = PolicyLearner(PolicyMemoryStore(tmp_path / "runs")).learn_from_error_delta(
        run_id="child",
        parent_run_id="parent",
        dataset_version="coco2017",
        error_delta=error_delta,
        changed_variables={},
    )

    assert records == []


def test_policy_memory_context_summarizes_relevant_history_for_llm(tmp_path: Path) -> None:
    """LLM context should include compact historical effects for relevant targets."""
    store = PolicyMemoryStore(tmp_path / "runs")
    store.append(
        [
            PolicyMemoryRecord(
                run_id="child1",
                parent_run_id="parent",
                dataset_version="coco2017",
                scenario="generic",
                action="small_object_oversampling",
                target="area_metric:small:ap_small",
                metric_name="ap_small",
                before=0.2,
                after=0.24,
                delta=0.04,
                changed_variables={"data_action": "small_object_oversampling"},
            ),
            PolicyMemoryRecord(
                run_id="child2",
                parent_run_id="parent",
                dataset_version="coco2017",
                action="sahi_or_tiling_eval",
                target="area_metric:small:ap_small",
                metric_name="ap_small",
                before=0.2,
                after=0.26,
                delta=0.06,
                changed_variables={"postprocess_action": "sahi_or_tiling_eval"},
            ),
            PolicyMemoryRecord(
                run_id="child3",
                parent_run_id="parent",
                dataset_version="other",
                action="unrelated",
                target="per_class_metric:car:per_class_ap",
                metric_name="per_class_ap",
                before=0.5,
                after=0.4,
                delta=-0.1,
            ),
        ]
    )

    context = build_policy_memory_context(
        store,
        dataset_version="coco2017",
        target_metrics=["ap_small"],
        target_actions=["small_object_oversampling"],
    )

    actions = {item["action"] for item in context["historical_effects"]}
    assert context["summary_count"] == 2
    assert "small_object_oversampling" in actions
    assert "sahi_or_tiling_eval" in actions
    assert "unrelated" not in actions
    assert all("interpretation" in item for item in context["historical_effects"])
    assert "Compatibility, evidence, budget, and single-variable gates override policy memory." in context["usage_rules"]


def test_action_fingerprint_normalizes_action_identity_and_posterior(tmp_path: Path) -> None:
    """Candidate names must not split observations of the same normalized action."""
    store = PolicyMemoryStore(tmp_path / "runs")
    records = []
    for index, effect in enumerate([0.01, 0.02, 0.03], start=1):
        records.append(
            PolicyMemoryRecord(
                run_id=f"candidate-run-{index}",
                action="reduce_mosaic_strength",
                action_fingerprint=ActionFingerprint(
                    action="reduce_mosaic_strength",
                    recipe_id="augmentation.reduce_mosaic",
                    component_versions={"augmentation.mosaic": "1.2"},
                    changed_variable="mosaic",
                    before_value=1.0,
                    after_value=0.5,
                    model_family="yolo26",
                    dataset_signature="coco-manifest",
                    protocol_hash="protocol-640",
                    fidelity="pilot",
                ),
                target="background_false_positive:all:count",
                metric_name="map50_95",
                effect_delta=effect,
                seed_count=1,
            )
        )
    store.append(records)

    summary = store.summarize(action="reduce_mosaic_strength")[0]
    assert summary.record_count == 3
    assert summary.seed_count == 3
    assert summary.mean_target_metric_gain == 0.02
    assert summary.effect_variance is not None
    assert summary.confidence_interval_95 is not None
    assert summary.confidence_interval_95[0] > 0
    assert summary.posterior_confidence == "high"
    assert summary.action_fingerprint is not None
    assert summary.action_fingerprint.changed_variable == "mosaic"


def test_policy_memory_tracks_pilot_to_full_transfer_and_cost_distribution(tmp_path: Path) -> None:
    store = PolicyMemoryStore(tmp_path / "runs")
    records = []
    for dataset, pilot_gain, full_gain, latency in [
        ("coco-a", 0.01, 0.02, 3.0),
        ("coco-b", 0.02, 0.04, 5.0),
    ]:
        for fidelity, gain in [("pilot", pilot_gain), ("full", full_gain)]:
            records.append(
                PolicyMemoryRecord(
                    run_id=f"{dataset}-{fidelity}",
                    action="small_object_sampling",
                    action_fingerprint=ActionFingerprint(
                        action="small_object_sampling",
                        recipe_id="sampling.small",
                        changed_variable="sampler",
                        before_value="uniform",
                        after_value="small_object_weighted",
                        model_family="yolo26",
                        dataset_signature=dataset,
                        protocol_hash=f"{dataset}-640",
                        fidelity=fidelity,
                    ),
                    target="area_metric:small:ap_small",
                    metric_name="ap_small",
                    effect_delta=gain,
                    cost={"latency_delta_pct": latency, "model_size_delta_pct": 0.0},
                )
            )
    store.append(records)

    summaries = store.summarize(action="small_object_sampling")
    pilot = next(item for item in summaries if item.action_fingerprint and item.action_fingerprint.fidelity == "pilot")
    assert pilot.pilot_to_full_correlation == 1.0
    assert pilot.pilot_to_full_gain_ratio == 2.0
    assert pilot.latency_cost_distribution.p50 == 4.0
    assert pilot.latency_cost_distribution.p90 == 4.8
    weighted = next(
        item
        for item in store.summarize(action="small_object_sampling", dataset_signature="coco-a")
        if item.action_fingerprint and item.action_fingerprint.fidelity == "pilot"
    )
    assert weighted.mean_effect_delta == 0.012
    assert weighted.mean_dataset_similarity_weight == 0.625


def test_policy_learner_writes_full_action_context(tmp_path: Path) -> None:
    records = PolicyLearner(PolicyMemoryStore(tmp_path / "runs")).learn_from_error_delta(
        run_id="child",
        parent_run_id="parent",
        dataset_version="coco2017",
        error_delta={
            "improved_errors": [
                {
                    "trend": "improved",
                        "fact_type": "area_metric",
                        "subject": "small",
                        "area": "small",
                        "metric_name": "ap_small",
                    "parent_value": 0.20,
                    "current_value": 0.22,
                        "delta": 0.02,
                        "matched_control_hash": "control-hash",
                }
            ]
        },
        changed_variables={"mosaic": 0.5},
        recipe_id="augmentation.reduce_mosaic",
        component_versions={"augmentation.mosaic": "1.2"},
        model_family="yolo26",
        dataset_signature="coco-manifest",
        protocol_hash="protocol-640",
        fidelity="pilot",
        action_before_values={"mosaic": 1.0},
        paired_result=verified_paired_result(
            candidate_id="candidate",
            node_id="node",
            delta=0.02,
            target_improved=True,
        ),
    )
    fingerprint = records[0].action_fingerprint
    assert fingerprint is not None
    assert fingerprint.recipe_id == "augmentation.reduce_mosaic"
    assert fingerprint.changed_variable == "mosaic"
    assert fingerprint.before_value == 1.0
    assert fingerprint.after_value == 0.5
    assert fingerprint.model_family == "yolo26"
    assert fingerprint.dataset_signature == "coco-manifest"
    assert fingerprint.protocol_hash == "protocol-640"
    assert fingerprint.fidelity == "pilot"
    assert fingerprint.recipe_version == "unknown"
    assert fingerprint.seed == "unknown"
    assert fingerprint.matched_control_hash == records[0].matched_control_hash
    assert fingerprint.matched_control_hash != "control-hash"


def test_policy_memory_predicts_full_gain_from_pilot_10_pairs(tmp_path: Path) -> None:
    store = PolicyMemoryStore(tmp_path / "runs")
    records = []
    for dataset, seed, pilot_3, pilot_10, full in [
        ("coco-a", 1, 0.005, 0.01, 0.02),
        ("coco-b", 2, 0.01, 0.02, 0.04),
    ]:
        for fidelity, gain in [("pilot_3", pilot_3), ("pilot_10", pilot_10), ("full", full)]:
            records.append(
                PolicyMemoryRecord(
                    run_id=f"{dataset}-{fidelity}",
                    action="sampling.small",
                    action_fingerprint=ActionFingerprint(
                        action="sampling.small",
                        recipe_id="sampling.small",
                        recipe_version="1.2.0",
                        component_versions={"sampling.small": "2.0.0"},
                        changed_variable="sampler",
                        after_value="weighted",
                        model_family="yolo26",
                        dataset_signature=dataset,
                        protocol_hash=f"{fidelity}-protocol",
                        fidelity=fidelity,
                        seed=seed,
                    ),
                    target="metric:map50_95",
                    metric_name="map50_95",
                    effect_delta=gain,
                    cost={"latency_delta_pct": 1.0, "model_size_delta_pct": 0.0},
                )
            )
    store.append(records)
    query = ActionFingerprint(
        action="sampling.small",
        recipe_id="sampling.small",
        recipe_version="1.2.0",
        component_versions={"sampling.small": "2.0.0"},
        changed_variable="sampler",
        after_value="weighted",
        model_family="yolo26",
        dataset_signature="coco-current",
        protocol_hash="pilot-10-current",
        fidelity="pilot_10",
        seed=3,
    )

    prediction = store.predict_full_gain(
        query,
        metric_name="map50_95",
        observed_pilot_delta=0.03,
    )

    assert prediction.expected_full_gain == 0.06
    assert prediction.pilot_full_correlation == 1.0
    assert prediction.pair_count == 2
    assert prediction.full_observation_count == 2
    assert prediction.confidence == 0.35
