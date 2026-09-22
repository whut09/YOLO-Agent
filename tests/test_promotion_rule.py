"""Four-way promotion rule: CONFIRMED / POSSIBLE / REJECTED / INCONCLUSIVE (§18K §4)."""

from __future__ import annotations

from types import SimpleNamespace

from yolo_agent.agents.promotion_rule import PromotionRule, apply_promotion_rule
from yolo_agent.agents.seed_confirmation import SeedConfirmationState


def _state(
    baseline: dict[str, list[float]],
    candidate: dict[str, list[float]],
) -> SeedConfirmationState:
    """Build a completed six-run state from per-metric per-seed lists."""
    state = SeedConfirmationState.create(
        confirmation_id="conf-rule", pilot_winner_id="trial-abc"
    )
    for role, table in (("baseline", baseline), ("candidate", candidate)):
        for index, seed in enumerate(state.baseline_seeds):
            run = state.run_for(role, seed)  # type: ignore[arg-type]
            run.status = "completed"
            for metric, series in table.items():
                run.metrics[metric] = series[index]
    return state


def _rule(**overrides: object) -> PromotionRule:
    values: dict[str, object] = {
        "target_delta": 0.01,
        "max_latency_regression": 0.05,
        "max_model_size_regression": None,
    }
    values.update(overrides)
    return PromotionRule.model_validate(values)


def test_consistent_improvement_confirms() -> None:
    state = _state(
        baseline={
            "map50_95": [0.30, 0.30, 0.30],
            "precision": [0.70, 0.70, 0.70],
            "latency_ms": [10.0, 10.0, 10.0],
        },
        candidate={
            "map50_95": [0.34, 0.35, 0.33],
            "precision": [0.72, 0.72, 0.72],
            "latency_ms": [10.4, 10.4, 10.4],
        },
    )

    decision = apply_promotion_rule(state, _rule())

    assert decision.verdict == "CONFIRMED"
    assert decision.is_confirmed
    assert any(r.startswith("ci_policy_satisfied") for r in decision.reasons)
    # Real secondary gains are acknowledged even on a confirmed verdict.
    assert any(
        b.startswith("secondary_improved:precision") for b in decision.acknowledged_benefits
    )


def test_high_variance_delta_is_possible_not_confirmed() -> None:
    """A large positive mean with a CI straddling zero must not confirm."""
    state = _state(
        baseline={
            "map50_95": [0.30, 0.30, 0.30],
            "latency_ms": [10.0, 10.0, 10.0],
        },
        candidate={
            "map50_95": [0.30, 0.60, 0.25],
            "latency_ms": [10.4, 10.4, 10.4],
        },
    )

    decision = apply_promotion_rule(state, _rule())

    assert decision.verdict == "POSSIBLE"
    assert any(r.startswith("ci_policy_unmet") for r in decision.reasons)


def test_one_bad_seed_is_inconclusive() -> None:
    """One collapsed seed drags the mean below the bar: cannot conclude."""
    state = _state(
        baseline={
            "map50_95": [0.30, 0.30, 0.30],
            "latency_ms": [10.0, 10.0, 10.0],
        },
        candidate={
            "map50_95": [0.35, 0.34, 0.21],
            "latency_ms": [10.4, 10.4, 10.4],
        },
    )

    decision = apply_promotion_rule(state, _rule())

    assert decision.verdict == "INCONCLUSIVE"
    assert any(r.startswith("effect_below_target") for r in decision.reasons)


def test_significant_primary_regression_rejects() -> None:
    state = _state(
        baseline={
            "map50_95": [0.30, 0.30, 0.30],
            "latency_ms": [10.0, 10.0, 10.0],
        },
        candidate={
            "map50_95": [0.25, 0.24, 0.25],
            "latency_ms": [10.4, 10.4, 10.4],
        },
    )

    decision = apply_promotion_rule(state, _rule())

    assert decision.verdict == "REJECTED"
    assert any(
        r.startswith("significant_primary_regression") for r in decision.reasons
    )


def test_latency_violation_rejects_despite_accuracy_gain() -> None:
    """Accuracy CONFIRMED conditions met, but +24% latency rejects outright."""
    state = _state(
        baseline={
            "map50_95": [0.30, 0.30, 0.30],
            "latency_ms": [10.0, 10.0, 10.0],
        },
        candidate={
            "map50_95": [0.34, 0.35, 0.33],
            "latency_ms": [12.4, 12.4, 12.4],
        },
    )

    decision = apply_promotion_rule(state, _rule())

    assert decision.verdict == "REJECTED"
    assert decision.hard_constraint_violations
    assert decision.hard_constraint_violations[0].startswith("latency_regression")
    assert any(
        r.startswith("hard_constraint_violation") for r in decision.reasons
    )


def test_missing_constraint_evidence_is_inconclusive() -> None:
    """No latency recorded -> cannot claim 'no regression' (fail-closed)."""
    state = _state(
        baseline={"map50_95": [0.30, 0.30, 0.30]},
        candidate={"map50_95": [0.34, 0.35, 0.33]},
    )

    decision = apply_promotion_rule(state, _rule())

    assert decision.verdict == "INCONCLUSIVE"
    assert any(
        r.startswith("constraint_evidence_missing") and "latency_ms" in r
        for r in decision.reasons
    )


def test_incomplete_matrix_is_inconclusive_not_a_verdict() -> None:
    state = SeedConfirmationState.create(
        confirmation_id="conf-partial", pilot_winner_id="trial-abc"
    )
    state.run_for("baseline", 42).status = "completed"  # type: ignore[arg-type]

    decision = apply_promotion_rule(state, _rule())

    assert decision.verdict == "INCONCLUSIVE"
    assert decision.reasons[0].startswith("evidence_incomplete")


def test_deployment_latency_cap_rejects() -> None:
    from yolo_agent.core.task_spec import TaskSpec

    state = _state(
        baseline={
            "map50_95": [0.30, 0.30, 0.30],
            "latency_ms": [10.0, 10.0, 10.0],
        },
        candidate={
            "map50_95": [0.34, 0.35, 0.33],
            "latency_ms": [10.4, 10.4, 10.4],
        },
    )
    spec = TaskSpec.model_validate(
        {
            "class_names": ["a"],
            "primary_metric": {"name": "map50_95"},
            "max_latency_ms": 10.2,
        }
    )

    decision = apply_promotion_rule(state, _rule(), task_spec=spec)

    assert decision.verdict == "REJECTED"
    assert any(v.startswith("latency_cap") for v in decision.hard_constraint_violations)


def test_minimize_primary_flips_the_sign() -> None:
    state = _state(
        baseline={
            "loss": [1.04, 1.04, 1.04],
            "latency_ms": [10.0, 10.0, 10.0],
        },
        candidate={
            "loss": [1.00, 1.00, 1.00],
            "latency_ms": [10.4, 10.4, 10.4],
        },
    )
    rule = _rule(primary_metric="loss", primary_goal="minimize")

    decision = apply_promotion_rule(state, rule)

    assert decision.verdict == "CONFIRMED"
    assert decision.mean_delta is not None and decision.mean_delta > 0


def test_from_objective_maps_declared_budgets() -> None:
    objective = SimpleNamespace(
        primary_metric="map50",
        target_absolute_delta=0.03,
        max_latency_regression=0.08,
        max_model_size_regression=0.20,
    )

    rule = PromotionRule.from_objective(objective)

    assert rule.primary_metric == "map50"
    assert rule.target_delta == 0.03
    assert rule.max_latency_regression == 0.08
    assert rule.max_model_size_regression == 0.20

    # An unset target falls back to the conservative default, not zero.
    bare = PromotionRule.from_objective(
        SimpleNamespace(primary_metric="map50_95", target_absolute_delta=None)
    )
    assert bare.target_delta == 0.02
