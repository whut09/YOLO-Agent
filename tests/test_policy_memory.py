"""Policy memory and learner tests."""

from __future__ import annotations

from pathlib import Path

import pytest

import yolo_agent.core.policy_memory as policy_memory
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


def test_fingerprint_identity_properties_are_cached_and_stable() -> None:
    """fix 35: repeated identity reads must be cached, not recomputed.

    summarize re-reads transfer_sha256 for every group over the whole
    memory (80k+ records in a long run).  The uncached property rebuilt
    the payload, re-serialized it, and re-hashed it on every access, which
    turned the generate_loop_plan stage into an hours-long hot loop that
    grew with the memory size.
    """
    fingerprint = ActionFingerprint(
        action="reduce_mosaic_strength",
        recipe_id="augmentation.reduce_mosaic",
        component_versions={"augmentation.mosaic": "1.2"},
        changed_variable="mosaic",
        before_value=1.0,
        after_value=0.5,
        model_family="yolo26",
        dataset_signature="coco-manifest",
        protocol_hash="protocol-640",
        fidelity="pilot_3",
    )

    values = [fingerprint.transfer_sha256 for _ in range(4)]
    assert len(set(values)) == 1  # stable across repeated reads
    assert fingerprint.fingerprint_sha256 == fingerprint.execution_fingerprint_sha256
    assert values[0] != fingerprint.posterior_sha256  # different payloads hash differently

    # a fresh equal instance must produce the identical identity
    twin = fingerprint.model_copy()
    assert twin.transfer_sha256 == values[0]
    assert twin.posterior_sha256 == fingerprint.posterior_sha256


def _transfer_record(
    *,
    run_id: str,
    action: str,
    recipe_id: str | None,
    fidelity: str,
    effect: float | None,
    target: str = "area_metric:small:ap_small",
    metric_name: str | None = "ap_small",
    dataset_signature: str = "coco-a",
    seed: int | str = 1,
    snapshot_hash: str = "snapshot-1",
    with_fingerprint: bool = True,
    latency_delta_pct: float | None = 1.0,
    seed_count: int = 1,
    confidence: str = "low",
    scenario: str | None = None,
) -> PolicyMemoryRecord:
    """Build one memory record with a controllable transfer identity."""
    return PolicyMemoryRecord(
        run_id=run_id,
        dataset_version="coco2017",
        action=action,
        action_fingerprint=(
            ActionFingerprint(
                action=action,
                recipe_id=recipe_id,
                component_versions={"sampling.small": "2.0.0"},
                changed_variable="sampler",
                before_value="uniform",
                after_value="weighted",
                model_family="yolo26",
                dataset_signature=dataset_signature,
                protocol_hash="protocol-640",
                snapshot_hash=snapshot_hash,
                fidelity=fidelity,
                seed=seed,
            )
            if with_fingerprint
            else None
        ),
        target=target,
        metric_name=metric_name,
        effect_delta=effect,
        scenario=scenario,
        seed_count=seed_count,
        confidence=confidence,
        cost={"latency_delta_pct": latency_delta_pct, "model_size_delta_pct": 0.5},
    )


def _transfer_overlap_records() -> list[PolicyMemoryRecord]:
    """Corpus exercising both OR-branches of the transfer filter.

    Layout (action ``small_object_sampling`` unless noted):
    - wildcard records (recipe_id=None) share one transfer identity, so a
      wildcard-led group matches them through *both* filter branches;
    - a recipe_id==action record shares the wildcard transfer identity while
      only matching branch A;
    - a distinct recipe (``sampling.small``) forms its own transfer family that
      still pulls same-action wildcards through branch B;
    - full-fidelity records feed pilot groups' full means and vice versa;
    - a legacy record without an explicit fingerprint gets a synthesized
      wildcard fingerprint, matching branch B but never branch A;
    - a second action (``reduce_mosaic_strength``) with metric_name=None
      exercises the unfiltered-metric group path.
    """
    return [
        _transfer_record(  # wildcard pilot; group leader of the wildcard family
            run_id="w1", action="small_object_sampling", recipe_id=None,
            fidelity="pilot", effect=0.01, seed=1, seed_count=2,
        ),
        _transfer_record(  # recipe full; branch A only for the recipe family
            run_id="r2", action="small_object_sampling", recipe_id="sampling.small",
            fidelity="full", effect=0.04, seed=1, dataset_signature="coco-a",
        ),
        _transfer_record(  # wildcard pilot; matches both branches of the wildcard family
            run_id="w2", action="small_object_sampling", recipe_id=None,
            fidelity="pilot", effect=0.03, seed=2, dataset_signature="coco-b",
        ),
        _transfer_record(  # recipe_id == action: shares the wildcard transfer identity
            run_id="s1", action="small_object_sampling", recipe_id="small_object_sampling",
            fidelity="full", effect=0.05, seed=2, dataset_signature="coco-b",
        ),
        _transfer_record(  # second full in the wildcard family: second pilot/full pair
            run_id="s2", action="small_object_sampling", recipe_id="small_object_sampling",
            fidelity="full", effect=0.02, seed=1, dataset_signature="coco-a",
        ),
        _transfer_record(  # wildcard pilot_10 of the second action, distinct target
            run_id="m1", action="reduce_mosaic_strength", recipe_id=None,
            fidelity="pilot_10", effect=0.02, target="background_false_positive:all:count",
            metric_name="map50_95",
        ),
        _transfer_record(  # wildcard pilot with a None effect delta
            run_id="w3", action="small_object_sampling", recipe_id=None,
            fidelity="pilot", effect=None, seed=3, dataset_signature="coco-c",
        ),
        _transfer_record(  # recipe pilot; leader of the recipe family
            run_id="r1", action="small_object_sampling", recipe_id="sampling.small",
            fidelity="pilot", effect=0.02, seed=3, dataset_signature="coco-c",
            confidence="medium", latency_delta_pct=None,
        ),
        _transfer_record(  # full with metric_name=None: unfiltered-metric group
            run_id="m2", action="reduce_mosaic_strength", recipe_id=None,
            fidelity="full", effect=0.03, target="background_false_positive:all:count",
            metric_name=None,
        ),
        _transfer_record(  # full with the group metric; pulls m1/m2 via branch B
            run_id="m3", action="reduce_mosaic_strength", recipe_id=None,
            fidelity="full", effect=0.06, target="background_false_positive:all:count",
            metric_name="map50_95", seed_count=3,
        ),
        _transfer_record(  # legacy record without explicit fingerprint
            run_id="n1", action="small_object_sampling", recipe_id=None,
            fidelity="pilot", effect=0.09, with_fingerprint=False,
        ),
        _transfer_record(  # unrelated action wildcard: isolated family
            run_id="x1", action="unrelated_action", recipe_id=None,
            fidelity="full", effect=0.07, target="metric:map50_95",
        ),
    ]


class _FullScanTransferIndex:
    """Reference index reproducing the pre-fix-40a full-scan candidate set."""

    def __init__(self, records: list[PolicyMemoryRecord]) -> None:
        self._records = records

    def candidates(self, *, transfer_sha256: str, action: str) -> list[PolicyMemoryRecord]:
        return list(self._records)


def _brute_force_candidates(
    records: list[PolicyMemoryRecord],
    *,
    transfer_sha256: str,
    action: str,
) -> list[PolicyMemoryRecord]:
    """Direct transcription of the historical transfer OR-filter."""
    return [
        record
        for record in records
        if record.action_fingerprint is not None
        and (
            record.action_fingerprint.transfer_sha256 == transfer_sha256
            or (record.action == action and record.action_fingerprint.recipe_id is None)
        )
    ]


def test_transfer_matching_helper_preserves_historical_filter_semantics() -> None:
    """The extracted filter helper must keep both callers' exact semantics."""
    records = _transfer_overlap_records()
    fingerprint = next(
        item.action_fingerprint for item in records if item.run_id == "w1"
    )
    transfer = fingerprint.transfer_sha256

    lenient = policy_memory._transfer_matching(
        records, transfer_sha256=transfer, action="small_object_sampling",
        target=None, metric_name=None,
    )
    assert lenient == _brute_force_candidates(
        records, transfer_sha256=transfer, action="small_object_sampling"
    )
    # the legacy record (no explicit fingerprint) matches through branch B only:
    # the validator synthesizes a wildcard fingerprint whose transfer identity
    # differs from the wildcard family's snapshot identity
    assert "n1" in {item.run_id for item in lenient}
    legacy_fingerprint = next(
        item.action_fingerprint for item in records if item.run_id == "n1"
    )
    assert legacy_fingerprint.recipe_id is None
    assert legacy_fingerprint.transfer_sha256 != transfer

    strict = policy_memory._transfer_matching(
        records, transfer_sha256=transfer, action="small_object_sampling",
        target=None, metric_name=None, require_effect_delta=True,
    )
    assert strict == [item for item in lenient if item.effect_delta is not None]

    scoped = policy_memory._transfer_matching(
        records, transfer_sha256=transfer, action="small_object_sampling",
        target="area_metric:small:ap_small", metric_name="ap_small",
    )
    assert scoped == [
        item
        for item in lenient
        if item.target == "area_metric:small:ap_small"
        and item.metric_name in {"ap_small", None}
    ]


def test_transfer_candidate_index_dedupes_and_preserves_store_order() -> None:
    """Index candidates must equal the brute-force filter, once per record."""
    records = _transfer_overlap_records()
    index = policy_memory._TransferCandidateIndex(records)

    scenarios = [
        # wildcard-led family: wildcards match both OR-branches and must appear once
        (records[0].action_fingerprint.transfer_sha256, "small_object_sampling"),
        # recipe family: branch A plus same-action wildcards via branch B
        (
            next(
                item.action_fingerprint for item in records if item.run_id == "r1"
            ).transfer_sha256,
            "small_object_sampling",
        ),
        # empty transfer identity: only branch B wildcards can match
        ("", "small_object_sampling"),
        # unrelated action
        ("", "unrelated_action"),
        # unknown transfer identity
        ("0" * 64, "small_object_sampling"),
    ]
    for transfer_sha256, action in scenarios:
        candidates = index.candidates(transfer_sha256=transfer_sha256, action=action)
        expected = _brute_force_candidates(
            records, transfer_sha256=transfer_sha256, action=action
        )
        assert candidates == expected
        assert len({id(item) for item in candidates}) == len(candidates)

    # the wildcard family must dedupe a record that matches both branches
    wildcard_family = index.candidates(
        transfer_sha256=records[0].action_fingerprint.transfer_sha256,
        action="small_object_sampling",
    )
    assert "w2" in {item.run_id for item in wildcard_family}


def test_summarize_transfer_stats_match_full_scan_reference(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """fix 40a: indexed candidates must reproduce full-scan summaries exactly."""
    store = PolicyMemoryStore(tmp_path / "runs")
    store.append(_transfer_overlap_records())

    variants = [
        {},
        {"dataset_version": "coco2017"},
        {
            "action": "small_object_sampling",
            "dataset_version": "coco2017",
            "dataset_signature": "coco-a",
            "scenario": None,
            "model_family": "yolo26",
        },
        {"action": "reduce_mosaic_strength", "dataset_signature": "coco-b"},
    ]
    indexed = [store.summarize(**variant) for variant in variants]

    monkeypatch.setattr(policy_memory, "_TransferCandidateIndex", _FullScanTransferIndex)
    reference = [store.summarize(**variant) for variant in variants]

    for variant, indexed_summaries, reference_summaries in zip(variants, indexed, reference):
        dumped_indexed = [item.model_dump(mode="json") for item in indexed_summaries]
        dumped_reference = [item.model_dump(mode="json") for item in reference_summaries]
        assert dumped_indexed == dumped_reference, f"summaries diverged for {variant}"

    # the corpus must be non-trivial: transfer stats really were computed
    flat = [item for summaries in indexed for item in summaries]
    assert flat
    assert any(item.full_mean_effect_delta is not None for item in flat)
    assert any(item.pilot_mean_effect_delta is not None for item in flat)
    assert any(item.pilot_to_full_correlation is not None for item in flat)


def test_summarize_transfer_scan_volume_stays_subquadratic(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """fix 40a: per-group scans must not re-walk the whole store per group."""
    records: list[PolicyMemoryRecord] = []
    for action_index in range(30):
        action = f"action_{action_index}"
        for record_index in range(20):
            records.append(
                _transfer_record(
                    run_id=f"{action}-{record_index}",
                    action=action,
                    recipe_id=None if record_index % 2 else f"recipe_{action_index}",
                    fidelity=("pilot", "pilot_10", "full")[record_index % 3],
                    effect=0.01 + record_index * 0.001,
                    seed=record_index,
                    dataset_signature=f"coco-{record_index % 4}",
                )
            )
    store = PolicyMemoryStore(tmp_path / "runs")
    store.append(records)

    scanned: list[int] = []
    original = policy_memory._pilot_full_stats

    def counting(records_argument, **kwargs):  # type: ignore[no-untyped-def]
        scanned.append(len(records_argument))
        return original(records_argument, **kwargs)

    monkeypatch.setattr(policy_memory, "_pilot_full_stats", counting)
    indexed_summaries = store.summarize(dataset_version="coco2017")
    indexed_volume = sum(scanned)

    scanned.clear()
    monkeypatch.setattr(policy_memory, "_TransferCandidateIndex", _FullScanTransferIndex)
    store.summarize(dataset_version="coco2017")
    full_scan_volume = sum(scanned)

    assert len(indexed_summaries) >= 30  # many groups really were summarized
    assert full_scan_volume > indexed_volume * 4
