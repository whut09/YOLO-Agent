"""Prompt-15 bounded autonomous loop tests.

Pins the contract that the loop's next-round decision consumes the *full*
multi-metric error-delta profile (never a lone mAP scalar), that HPO is
bounded to ActionSpec-declared search spaces with train/eval routing, and
that stopping is honest: ``stop_exhausted`` fires only when no
evidence-supported action remains.  All orchestration runs on deterministic
fakes; no training executes here.
"""

from __future__ import annotations

import pytest

from yolo_agent.agents.action_space import ActionCatalog
from yolo_agent.agents.autonomous_loop import (
    BoundedAutonomousLoop,
    CompatibilityMaturityFilter,
)
from yolo_agent.agents.bounded_hpo import (
    EvalRouteTrial,
    HpoRequestRejected,
    hpo_scope_for_action,
    rank_eval_route_trials,
    request_hpo_points,
    search_space_from_action,
)
from yolo_agent.agents.error_delta_profile import (
    ErrorDeltaProfile,
    EvidenceGap,
    build_error_delta_profile,
)
from yolo_agent.agents.loop_decision_engine import (
    decide_next_round,
    pareto_score,
    pareto_weights_from_task_spec,
)
from yolo_agent.core.error_facts import (
    ErrorFact,
    build_evidence_incomplete_fact,
)
from yolo_agent.core.experiment_graph import MetricEvidence
from yolo_agent.core.task_spec import MetricPriority, TaskSpec


CATALOG_PATH = "configs/actions/detection_action_catalog.yaml"

_BASE_METRICS = {
    "map50_95": 0.40,
    "precision": 0.70,
    "recall": 0.60,
    "ap_small": 0.20,
    "ap_medium": 0.45,
    "ap_large": 0.55,
    "latency_ms": 10.0,
}


def _metric_records(values: dict[str, float], candidate_id: str, node_id: str) -> list[MetricEvidence]:
    return [
        MetricEvidence(candidate_id=candidate_id, node_id=node_id, metric_name=name, value=value)
        for name, value in values.items()
    ]


_DEFAULT_SPEC = TaskSpec(
    class_names=["a"],
    primary_metric=MetricPriority(name="ap_small", weight=2.0),
    max_latency_ms=12.0,
)


def profile_with(**overrides: float) -> ErrorDeltaProfile:
    """Build a delta profile against the shared baseline with overrides."""

    candidate = {**_BASE_METRICS, **overrides}
    return build_error_delta_profile(
        _metric_records(_BASE_METRICS, "base", "n_base"),
        _metric_records(candidate, "cand", "n_cand"),
        candidate_id="cand",
        node_id="n_cand",
        task_spec=_DEFAULT_SPEC,
    )


def small_fn_fact() -> ErrorFact:
    return ErrorFact(
        run_id="r",
        candidate_id="base",
        node_id="n",
        fact_type="area_metric",
        subject="small",
        area="small",
        metric_name="ap_small",
        value=0.20,
        severity="high",
    )


class FakeRunner:
    """Deterministic fake: profiles keyed by (round, action_id)."""

    def __init__(self, profiles: dict[tuple[int, str], ErrorDeltaProfile]) -> None:
        self.profiles = profiles
        self.calls: list[tuple[int, str, str]] = []

    def run_candidate(self, action, round_index, overrides):  # noqa: ANN001
        self.calls.append((round_index, action.action_id, action.family))
        return self.profiles[(round_index, action.action_id)]


@pytest.fixture(scope="module")
def catalog() -> ActionCatalog:
    return ActionCatalog.from_yaml(CATALOG_PATH)


# ------------------------------------------------- full delta profile tests --


def test_delta_profile_carries_full_multi_metric_state() -> None:
    profile = profile_with(ap_small=0.26, latency_ms=11.8)

    names = {delta.metric_name for delta in profile.deltas}
    assert {"map50_95", "precision", "recall", "ap_small", "latency_ms"} <= names
    scopes = {delta.metric_name: delta.scope for delta in profile.deltas}
    assert scopes["ap_small"] == "scale"
    assert scopes["map50_95"] == "global"
    assert scopes["latency_ms"] == "resource"

    small = profile.delta_for("ap_small")
    assert small is not None and small.improvement == pytest.approx(0.06)
    latency = profile.delta_for("latency_ms")
    assert latency is not None and latency.improvement == pytest.approx(-1.8)
    # The legacy single-scalar view exists only as an explicit projection.
    assert profile.primary_effect_delta() == pytest.approx(0.0)
    assert profile.is_evidence_complete()


def test_delta_profile_marks_missing_evidence_explicitly() -> None:
    baseline = _metric_records(_BASE_METRICS, "base", "n_base")
    candidate = _metric_records(
        {k: v for k, v in _BASE_METRICS.items() if k != "ap_small"},
        "cand",
        "n_cand",
    )
    profile = build_error_delta_profile(baseline, candidate, candidate_id="cand", node_id="n_cand")
    gaps = {gap.metric_name: gap.reason for gap in profile.evidence_incomplete}
    assert gaps.get("ap_small") == "missing_in_candidate"
    assert not profile.is_evidence_complete()


def test_delta_profile_flags_task_spec_hard_constraint_violation() -> None:
    profile = profile_with(latency_ms=12.5)
    assert profile.has_violations()
    violation = profile.hard_constraint_violations[0]
    assert violation.constraint == "max_latency_ms"
    assert violation.observed == pytest.approx(12.5)
    assert violation.limit == pytest.approx(12.0)


# ---------------------------------------------- error fact vocabulary tests --


def test_evidence_incomplete_fact_records_missing_metrics_verbatim() -> None:
    fact = build_evidence_incomplete_fact("r", "c", "n", ["ap_small", "latency_ms"])
    assert fact.fact_type == "evidence_incomplete"
    assert fact.evidence == {"ap_small": "missing", "latency_ms": "missing"}
    assert fact.action_candidates == ["collect_evidence"]


# -------------------------------------------------- decision engine tests ----


def test_decision_engine_promotes_multi_axis_gain_not_map_only() -> None:
    profile = profile_with(ap_small=0.26, latency_ms=10.2)
    decision = decide_next_round(
        profile,
        round_index=1,
        supported_action_ids=["other.action"],
        task_spec=TaskSpec(class_names=["a"], primary_metric=MetricPriority(name="ap_small", weight=2.0)),
    )
    assert decision.decision == "promote"
    # The Pareto axes must show the non-mAP gain the scalar view would hide.
    assert decision.pareto.per_axis["ap_small"] > 0
    assert "pareto_improvement" in decision.reasons[0]


def test_decision_engine_rolls_back_on_hard_constraint_violation() -> None:
    profile = profile_with(ap_small=0.26, latency_ms=12.5)
    decision = decide_next_round(
        profile,
        round_index=1,
        supported_action_ids=["other.action"],
        task_spec=_DEFAULT_SPEC,
    )
    assert decision.decision == "rollback"
    assert decision.rollback_candidate_id == "cand"


def test_decision_engine_collects_evidence_when_incomplete() -> None:
    profile = profile_with(ap_small=0.26)
    profile.evidence_incomplete.append(EvidenceGap(metric_name="ece", reason="missing_entirely"))
    decision = decide_next_round(profile, round_index=2, supported_action_ids=[])
    assert decision.decision == "collect_evidence"
    assert decision.missing_evidence == ["ece"]


def test_decision_engine_stops_exhausted_only_without_supported_actions() -> None:
    flat = profile_with()
    with_supported = decide_next_round(flat, round_index=3, supported_action_ids=["a.b"])
    assert with_supported.decision == "refine"

    without_supported = decide_next_round(flat, round_index=3, supported_action_ids=[])
    assert without_supported.decision == "stop_exhausted"
    assert without_supported.stop_reason == "no_evidence_supported_action"

    regressed = profile_with(ap_small=0.18)
    regressed_decision = decide_next_round(regressed, round_index=3, supported_action_ids=[])
    assert regressed_decision.decision == "stop_exhausted"


def test_pareto_axes_use_relative_improvements() -> None:
    weights = pareto_weights_from_task_spec(
        TaskSpec(class_names=["a"], primary_metric=MetricPriority(name="ap_small", weight=2.0))
    )
    assert weights.ap_small == 2.0
    profile = profile_with(ap_small=0.26, latency_ms=12.5)
    score = pareto_score(profile, weights)
    # +0.06/0.20 = 0.3 relative gain; latency 2.5/10.0 = 0.25 relative cost.
    assert score.per_axis["ap_small"] == pytest.approx(0.3)
    assert score.per_axis["latency"] == pytest.approx(-0.25)


# ------------------------------------------------------ bounded HPO tests ----


def test_bounded_hpo_rejects_params_outside_declared_space(catalog: ActionCatalog) -> None:
    action = next(a for a in catalog.actions if a.action_id == "inference.threshold.per_class")
    space = search_space_from_action(action)
    assert space.allows("conf_threshold") or space.allows("iou_threshold") or space.params, (
        "threshold action must declare a bounded search surface"
    )
    with pytest.raises(HpoRequestRejected) as excinfo:
        request_hpo_points(action, {"learning_rate": [0.1, 0.01]})
    assert any("param_not_in_declared_search_space" in reason for reason in excinfo.value.reasons)


def test_bounded_hpo_rejects_values_outside_declared_grid(catalog: ActionCatalog) -> None:
    action = next(a for a in catalog.actions if a.action_id == "inference.threshold.per_class")
    space = search_space_from_action(action)
    if not space.params:
        pytest.skip("action declares no search space")
    name = space.params[0].name
    with pytest.raises(HpoRequestRejected) as excinfo:
        request_hpo_points(action, {name: [0.123456]})
    assert any("values_outside_declared_grid" in reason for reason in excinfo.value.reasons)


def test_bounded_hpo_generates_grid_within_declared_surface(catalog: ActionCatalog) -> None:
    action = next(a for a in catalog.actions if a.action_id == "inference.threshold.per_class")
    space = search_space_from_action(action)
    if not space.params:
        pytest.skip("action declares no search space")
    name = space.params[0].name
    values = space.params[0].values[:2]
    points = request_hpo_points(action, {name: list(values)}, split="val")
    assert all(point[name] in values for point in points)
    assert len(points) == len(values)


def test_threshold_tuning_requires_validation_split(catalog: ActionCatalog) -> None:
    action = next(a for a in catalog.actions if a.action_id == "inference.threshold.per_class")
    space = search_space_from_action(action)
    if not space.params:
        pytest.skip("action declares no search space")
    name = space.params[0].name
    with pytest.raises(HpoRequestRejected) as excinfo:
        request_hpo_points(action, {name: list(space.params[0].values[:1])}, split="test")
    assert any("threshold_tuning_requires_validation_split" in reason for reason in excinfo.value.reasons)


def test_hpo_scope_routes_train_vs_eval(catalog: ActionCatalog) -> None:
    train_action = next(a for a in catalog.actions if a.action_id == "train.assigner.optimal_transport")
    eval_action = next(a for a in catalog.actions if a.action_id == "inference.tta.multi_scale")
    assert hpo_scope_for_action(train_action) == "asha_train_time"
    assert hpo_scope_for_action(eval_action) == "eval_route"


def test_eval_route_trials_never_authorize_training() -> None:
    trial = EvalRouteTrial(trial_id="t1", action_id="inference.tta.multi_scale", params={})
    assert trial.applies_to_training() is False
    ranked = rank_eval_route_trials(
        [
            EvalRouteTrial(trial_id="low", action_id="a", metrics={"map50_95": 0.40}),
            EvalRouteTrial(trial_id="high", action_id="a", metrics={"map50_95": 0.42}),
            EvalRouteTrial(
                trial_id="over_latency", action_id="a", metrics={"map50_95": 0.50, "latency_ms": 99.0}
            ),
        ],
        primary_metric="map50_95",
        max_latency=20.0,
    )
    assert [trial.trial_id for trial in ranked] == ["high", "low"]


# ----------------------------------------------------- loop state machine ----


def test_three_round_loop_is_delta_driven(catalog: ActionCatalog) -> None:
    """Round1 small-FN win, round2 constraint violation rollback, round3 FP fix."""

    runner = FakeRunner(
        {
            (1, "model.neck.rtmdet_large_kernel"): profile_with(ap_small=0.26),
            (2, "model.feature_fusion.gather_distribute"): profile_with(ap_small=0.24, latency_ms=12.5),
            (3, "train.assigner.optimal_transport"): profile_with(precision=0.73),
        }
    )
    spec = TaskSpec(
        class_names=["a"],
        primary_metric=MetricPriority(name="ap_small", weight=2.0),
        max_latency_ms=12.0,
    )
    filt = CompatibilityMaturityFilter(
        implementation_ready_ids=[
            "model.neck.rtmdet_large_kernel",
            "model.feature_fusion.gather_distribute",
            "train.assigner.optimal_transport",
        ]
    )
    loop = BoundedAutonomousLoop(catalog=catalog, runner=runner)
    report = loop.run(
        task_spec=spec,
        initial_facts=[small_fn_fact()],
        max_rounds=5,
        filter_=filt,
    )

    # Three executed rounds + the terminal exhaustion record (every eligible
    # action has been tried or banned by then — honest stop, no random params).
    assert [record.round_index for record in report.rounds] == [1, 2, 3, 4]
    assert report.rounds[0].decision == "promote"
    assert report.rounds[0].pareto_axes["ap_small"] > 0
    assert report.rounds[1].decision == "rollback"
    assert report.rounds[1].pareto_axes["latency"] < 0
    assert report.rounds[2].decision == "promote"
    assert report.rounds[2].pareto_axes["precision"] > 0
    assert report.rounds[3].decision == "stop_exhausted"
    assert report.rounds[3].executed_action_id is None
    assert report.final_decision == "stop_exhausted"
    assert report.stop_reason == "no_evidence_supported_action"
    assert runner.calls[0] == (1, "model.neck.rtmdet_large_kernel", "neck")
    assert len(runner.calls) == 3


def test_loop_round_2_sees_residual_profile_not_round_1_scalar(catalog: ActionCatalog) -> None:
    """Round-1's latency regression must be visible in round-2's decision trail."""

    runner = FakeRunner(
        {
            (1, "model.neck.rtmdet_large_kernel"): profile_with(ap_small=0.26),
            (2, "model.feature_fusion.gather_distribute"): profile_with(precision=0.66),
        }
    )
    spec = TaskSpec(class_names=["a"], primary_metric=MetricPriority(name="ap_small", weight=2.0))
    filt = CompatibilityMaturityFilter(
        implementation_ready_ids=["model.neck.rtmdet_large_kernel", "model.feature_fusion.gather_distribute"]
    )
    loop = BoundedAutonomousLoop(catalog=catalog, runner=runner)
    report = loop.run(task_spec=spec, initial_facts=[small_fn_fact()], max_rounds=4, filter_=filt)

    # Round 2 executed a *different* action because round 1 consumed one —
    # and its record carries the FP regression the full profile exposes.
    assert report.rounds[1].executed_action_id == "model.feature_fusion.gather_distribute"
    assert report.rounds[1].pareto_axes["precision"] < 0
    assert report.rounds[1].decision == "stop_exhausted"
    assert report.stop_reason == "no_evidence_supported_action"


def test_loop_stops_exhausted_without_evidence_supported_actions(catalog: ActionCatalog) -> None:
    runner = FakeRunner(
        {
            (1, "inference.tta.multi_scale"): profile_with(),
        }
    )
    filt = CompatibilityMaturityFilter(implementation_ready_ids=["inference.tta.multi_scale"])
    loop = BoundedAutonomousLoop(catalog=catalog, runner=runner)
    report = loop.run(
        task_spec=TaskSpec(class_names=["a"], primary_metric=MetricPriority(name="ap_small", weight=1.0)),
        initial_facts=[small_fn_fact()],
        max_rounds=4,
        filter_=filt,
    )
    assert report.final_decision == "stop_exhausted"
    assert report.stop_reason == "no_evidence_supported_action"
    assert report.rounds[-1].executed_action_id == "inference.tta.multi_scale"


def test_loop_filters_non_implementation_ready_actions(catalog: ActionCatalog) -> None:
    runner = FakeRunner({(1, "inference.tta.multi_scale"): profile_with(ap_small=0.24)})
    filt = CompatibilityMaturityFilter(
        implementation_ready_ids=["inference.tta.multi_scale"],
        synthetic_smoke_only_ids=["model.neck.rtmdet_large_kernel"],
    )
    loop = BoundedAutonomousLoop(catalog=catalog, runner=runner)
    report = loop.run(
        task_spec=TaskSpec(class_names=["a"], primary_metric=MetricPriority(name="ap_small", weight=1.0)),
        initial_facts=[small_fn_fact()],
        max_rounds=2,
        filter_=filt,
    )
    first = report.rounds[0]
    # The paper-lineage neck action is implementation-blocked and must appear
    # among the blocked ids regardless of which specific reason applies.
    assert "model.neck.rtmdet_large_kernel" in first.blocked_action_ids
    assert first.executed_action_id == "inference.tta.multi_scale"


def test_loop_goal_met_stops_with_goal_decision(catalog: ActionCatalog) -> None:
    runner = FakeRunner({(1, "inference.tta.multi_scale"): profile_with(ap_small=0.30)})
    filt = CompatibilityMaturityFilter(implementation_ready_ids=["inference.tta.multi_scale"])
    loop = BoundedAutonomousLoop(catalog=catalog, runner=runner)
    report = loop.run(
        task_spec=TaskSpec(class_names=["a"], primary_metric=MetricPriority(name="ap_small", weight=2.0)),
        initial_facts=[small_fn_fact()],
        max_rounds=3,
        filter_=filt,
        goal_met_round=1,
    )
    assert report.rounds[0].decision == "stop_goal_met"
    assert report.final_decision == "stop_goal_met"


def test_loop_stop_budget_when_budget_gone(catalog: ActionCatalog) -> None:
    runner = FakeRunner({(1, "inference.tta.multi_scale"): profile_with()})
    filt = CompatibilityMaturityFilter(implementation_ready_ids=["inference.tta.multi_scale"])
    loop = BoundedAutonomousLoop(catalog=catalog, runner=runner)
    report = loop.run(
        task_spec=TaskSpec(class_names=["a"], primary_metric=MetricPriority(name="ap_small", weight=1.0)),
        initial_facts=[small_fn_fact()],
        max_rounds=3,
        filter_=filt,
        budget_remaining_rounds=set(),
    )
    assert report.final_decision == "stop_budget"
    assert report.stop_reason == "budget_exhausted"
