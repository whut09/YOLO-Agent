"""Three-round deterministic optimization loop (Prompt-18J §8).

The agent must demonstrably consume the *error delta* — not the headline
mAP — across three synthetic rounds:

* Round 1: small-object FN high → small-object candidates are eligible.
* Round 2: AP_small improves but latency violates the TaskSpec ceiling →
  the expensive candidate is rolled back, not promoted.
* Round 3: background FP increases → hard-negative / threshold /
  postprocess / classification-loss actions lead the next portfolio.

Everything runs on synthetic COCO-style numbers; no GPU, no training.
"""

from __future__ import annotations

from yolo_agent.agents.action_space import ActionCatalog
from yolo_agent.core.detection_error_delta import (
    ResourceSnapshot,
    build_detection_error_delta,
)
from yolo_agent.core.detection_error_profile import (
    DetectionErrorProfile,
    FalseNegativeFacts,
    FalsePositiveFacts,
    GlobalMetrics,
    LocalizationFacts,
    PerClassMetrics,
    ScaleMetrics,
)
from yolo_agent.core.error_next_round import build_next_round_portfolio
from yolo_agent.core.error_round_decision import decide_next_round
from yolo_agent.core.experiment_memory import ExperimentMemory
from yolo_agent.core.hpo_candidate_planner import plan_hpo_points
from yolo_agent.core.task_spec import MetricPriority, TaskSpec

_CATALOG = ActionCatalog.from_yaml("configs/actions/detection_action_catalog.yaml")


def _spec(max_latency_ms: float | None = None) -> TaskSpec:
    return TaskSpec(
        class_names=["person", "car"],
        primary_metric=MetricPriority(name="map50_95"),
        max_latency_ms=max_latency_ms,
    )


def _profile(
    profile_id: str,
    candidate_id: str,
    *,
    map50_95: float,
    ap_small: float,
    fn_small: int,
    fn_medium: int,
    background_fp: int,
    duplicate_fp: int,
    latency_ms: float | None = None,
) -> DetectionErrorProfile:
    fn_total = fn_small + fn_medium
    fp_total = background_fp + duplicate_fp
    return DetectionErrorProfile(
        profile_id=profile_id,
        run_id="run-loop",
        candidate_id=candidate_id,
        gt_artifact="gt-val.json",
        dataset_manifest_hash="manifest-loop-1",
        global_=GlobalMetrics(
            map50=map50_95 + 0.15,
            map50_95=map50_95,
            precision=round(1.0 - fp_total / (fp_total + 100), 4),
            recall=round(1.0 - fn_total / (fn_total + 100), 4),
        ),
        scale=ScaleMetrics(
            ap_small=ap_small,
            ap_medium=0.50,
            ap_large=0.60,
            recall_small=round(1.0 - fn_small / (fn_small + 50), 4),
        ),
        per_class=[
            PerClassMetrics(
                category_id=1,
                name="person",
                ap=map50_95,
                ap50=map50_95 + 0.1,
                precision=0.7,
                recall=0.6,
                support=40,
            ),
            PerClassMetrics(
                category_id=2,
                name="car",
                ap=map50_95 + 0.05,
                ap50=map50_95 + 0.12,
                precision=0.75,
                recall=0.62,
                support=60,
            ),
        ],
        false_negative=FalseNegativeFacts(
            total=fn_total,
            by_scale={"small": fn_small, "medium": fn_medium},
            by_class={"person": fn_small, "car": fn_medium},
        ),
        false_positive=FalsePositiveFacts(
            total=fp_total,
            background_fp=background_fp,
            duplicate_fp=duplicate_fp,
        ),
        localization=LocalizationFacts(
            matched_iou_distribution={"0.75-0.85": 20},
            mean_matched_iou=0.82,
            localization_error_count=2,
            ap50_vs_ap75_gap=0.12,
        ),
        resources=(
            ResourceSnapshot(latency_ms=latency_ms)
            if latency_ms is not None
            else None
        ),
    )


def _run_round(candidate, parent, memory, *, action_id, parameters, parent_fp, max_latency=None):
    delta = build_detection_error_delta(candidate, parent)
    decision = decide_next_round(delta, _spec(max_latency), budget_remaining=1)
    return delta, decision


def test_round1_small_fn_high_routes_to_small_object_actions() -> None:
    baseline = _profile(
        "prof-r1-base", "baseline",
        map50_95=0.280, ap_small=0.180, fn_small=45, fn_medium=10,
        background_fp=20, duplicate_fp=5,
    )
    # A candidate that only nudges everything without structural change.
    cand_a = _profile(
        "prof-r1-cand", "cand-a",
        map50_95=0.284, ap_small=0.190, fn_small=43, fn_medium=10,
        background_fp=20, duplicate_fp=5,
    )

    delta, decision = _run_round(
        cand_a, baseline, ExperimentMemory(),
        action_id="data.sampling.class_aware", parameters={},
        parent_fp="root",
    )

    # No regression anywhere and the objective improved.
    assert decision.decision == "promote"
    assert any("scale.ap_small" in item for item in decision.acknowledgements)

    # The next portfolio after this round must still contain
    # small-object-addressing families (the FN structure is open).
    portfolio = build_next_round_portfolio(cand_a, delta, decision, _CATALOG)
    top_families = {entry.family for entry in portfolio.entries[:5]}
    assert top_families & {
        "sampling", "input_resolution", "head", "annotation", "assignment",
        "auxiliary_loss", "inference",
    }


def test_round2_latency_breach_rolls_back_the_expensive_candidate() -> None:
    parent = _profile(
        "prof-r2-base", "baseline-r2",
        map50_95=0.284, ap_small=0.190, fn_small=43, fn_medium=10,
        background_fp=20, duplicate_fp=5, latency_ms=10.0,
    )
    expensive = _profile(
        "prof-r2-cand", "cand-expensive",
        map50_95=0.296, ap_small=0.260, fn_small=30, fn_medium=9,
        background_fp=19, duplicate_fp=5, latency_ms=13.2,
    )
    memory = ExperimentMemory()

    delta, decision = _run_round(
        expensive, parent, memory,
        action_id="train.head.p2_stack", parameters={"p2_channels": 128},
        parent_fp="parent-r2", max_latency=12.0,
    )

    # mAP +0.012 and AP_small +0.07 — but the TaskSpec ceiling is 12ms.
    assert decision.decision == "rollback"
    assert any(item.code == "latency_constraint_violated" for item in decision.objections)
    # The engine acknowledges the real gains it is refusing.
    assert any("scale.ap_small" in item for item in decision.acknowledgements)

    # And the same expensive experiment cannot simply be re-run next round.
    memory.record(
        profile=expensive,
        action_id="train.head.p2_stack",
        parameters={"p2_channels": 128},
        parent_fingerprint="parent-r2",
        outcome="rolled_back",
        candidate_id="cand-expensive",
    )
    blocked = memory.check_repeat(
        profile=expensive,
        action_id="train.head.p2_stack",
        parameters={"p2_channels": 128},
        parent_fingerprint="parent-r2",
    )
    assert blocked is not None and blocked.outcome == "rolled_back"


def test_round3_fp_regression_ranks_fp_actions_first() -> None:
    parent = _profile(
        "prof-r3-base", "baseline-r3",
        map50_95=0.296, ap_small=0.260, fn_small=30, fn_medium=9,
        background_fp=19, duplicate_fp=5, latency_ms=10.0,
    )
    fp_regressor = _profile(
        "prof-r3-cand", "cand-fp",
        map50_95=0.301, ap_small=0.262, fn_small=29, fn_medium=9,
        background_fp=31, duplicate_fp=6, latency_ms=10.1,
    )
    memory = ExperimentMemory()

    delta, decision = _run_round(
        fp_regressor, parent, memory,
        action_id="data.augmentation.small_copy_paste",
        parameters={"max_pastes_per_image": 4},
        parent_fp="parent-r3",
    )

    # Objective improved but background FP +12 and duplicate +1: refine.
    assert decision.decision == "refine"
    assert any(item.code == "non_objective_regressions" for item in decision.objections)

    portfolio = build_next_round_portfolio(fp_regressor, delta, decision, _CATALOG)
    assert portfolio.entries
    top = portfolio.entries[0]
    # §3: the FP regression round must lead with FP-addressing families,
    # not with more small-object capacity.
    assert top.family in {
        "sampling", "classification_loss", "threshold", "postprocess",
    }
    assert top.rationale.startswith("addresses regressed surface")

    # Bounded HPO for the leading action stays inside the declared grid and
    # skips remembered failures.
    spec = next(
        a for a in _CATALOG.actions if a.action_id == top.action_id
    )
    declared = next(iter(spec.search_space))
    grid = plan_hpo_points(
        spec,
        fp_regressor,
        "parent-r3",
        {declared: list(spec.search_space[declared])},
        memory,
        max_points=4,
    )
    assert grid.runnable_points
    for point in grid.runnable_points:
        assert set(point) == {declared}
        assert point[declared] in spec.search_space[declared]


def test_the_loop_consumes_delta_not_just_map() -> None:
    """Round 2 vs Round 3 have near-identical mAP movement but opposite
    verdicts — proof the decision reads the movement surface."""
    parent_r2 = _profile(
        "prof-x-base", "base",
        map50_95=0.284, ap_small=0.190, fn_small=43, fn_medium=10,
        background_fp=20, duplicate_fp=5, latency_ms=10.0,
    )
    cand_r2 = _profile(
        "prof-x-c2", "c2",
        map50_95=0.296, ap_small=0.260, fn_small=30, fn_medium=9,
        background_fp=19, duplicate_fp=5, latency_ms=13.2,
    )
    cand_r3 = _profile(
        "prof-x-c3", "c3",
        map50_95=0.296, ap_small=0.260, fn_small=30, fn_medium=9,
        background_fp=31, duplicate_fp=6, latency_ms=10.0,
    )

    _, decision_r2 = _run_round(
        cand_r2, parent_r2, ExperimentMemory(),
        action_id="a", parameters={}, parent_fp="p", max_latency=12.0,
    )
    _, decision_r3 = _run_round(
        cand_r3, parent_r2, ExperimentMemory(),
        action_id="a", parameters={}, parent_fp="p",
    )

    same_map_gain = (
        cand_r2.global_.map50_95 - parent_r2.global_.map50_95
    ) == (cand_r3.global_.map50_95 - parent_r2.global_.map50_95)
    assert same_map_gain
    assert decision_r2.decision == "rollback"
    assert decision_r3.decision == "refine"
