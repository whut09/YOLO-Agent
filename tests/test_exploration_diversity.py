from __future__ import annotations

from pathlib import Path

from yolo_agent.agents.exploration_diversity import (
    ExplorationDiversityPolicy,
    ExplorationHistoryEntry,
    ExplorationHistoryStore,
    describe_recipe,
    evaluate_diversity_stop,
    screen_proposal,
    semantic_distance,
)
from yolo_agent.agents.strategy_policy import CandidatePolicy


def _proposal(policy_id: str, **overrides: float) -> CandidatePolicy:
    return CandidatePolicy(
        policy_id=policy_id,
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
        action_domain="training" if "box" in overrides else "augmentation",
        action_id=policy_id,
        train_overrides={**overrides, "imgsz": 640},
    )


def _history(
    proposal: CandidatePolicy, *, round_index: int, effect: float | None = None,
) -> ExplorationHistoryEntry:
    descriptor = describe_recipe(proposal)
    return ExplorationHistoryEntry(
        run_id=f"run-r{round_index}", round_index=round_index,
        policy_id=proposal.policy_id, candidate_id=proposal.policy_id,
        recipe_fingerprint=descriptor.fingerprint,
        component_family=descriptor.component_family,
        changed_values=descriptor.changed_values, semantic_tokens=descriptor.semantic_tokens,
        bucket="exploitation" if effect and effect > 0 else "exploration",
        effect_delta=effect, improved=bool(effect and effect > 0.0005),
    )


def test_recipe_fingerprint_ignores_policy_name_and_fixed_imgsz() -> None:
    first = _proposal("tune_box_8", box=8.0)
    second = _proposal("another_name", box=8.0)
    assert describe_recipe(first).fingerprint == describe_recipe(second).fingerprint


def test_adjacent_numeric_recipes_have_low_semantic_distance() -> None:
    first = describe_recipe(_proposal("box_8", box=8.0))
    second = describe_recipe(_proposal("box_8_25", box=8.25))
    assert first.component_family == "loss:box"
    assert semantic_distance(first, second) < 0.15


def test_duplicate_cooldown_and_semantic_distance_are_explicit() -> None:
    policy = ExplorationDiversityPolicy(component_family_cooldown_rounds=2)
    previous = _proposal("box_8", box=8.0)
    history = [_history(previous, round_index=4)]
    duplicate = screen_proposal(previous, current_round=7, history=history, policy=policy)
    cooldown = screen_proposal(_proposal("box_9", box=9.0), current_round=5, history=history, policy=policy)
    nearby = screen_proposal(_proposal("box_8_25", box=8.25), current_round=7, history=history, policy=policy)
    assert duplicate.reason == "duplicate_recipe_fingerprint"
    assert cooldown.reason.startswith("component_family_cooldown:")
    assert nearby.reason.startswith("minimum_semantic_distance:")


def test_positive_family_history_changes_bucket_to_exploitation() -> None:
    previous = _proposal("box_8", box=8.0)
    decision = screen_proposal(
        _proposal("box_10", box=10.0), current_round=8,
        history=[_history(previous, round_index=2, effect=0.01)],
        policy=ExplorationDiversityPolicy(component_family_cooldown_rounds=1),
    )
    assert decision.status == "eligible"
    assert decision.bucket == "exploitation"


def test_family_exhaustion_and_no_improvement_patience_stop() -> None:
    policy = ExplorationDiversityPolicy(
        family_exhaustion_attempts=3, no_improvement_patience=3,
        minimum_families_for_exhaustion_stop=1,
    )
    history = [
        _history(_proposal(f"box_{value}", box=value), round_index=index)
        for index, value in enumerate((6.0, 8.0, 10.0), start=1)
    ]
    screened = screen_proposal(
        _proposal("box_12", box=12.0), current_round=7, history=history, policy=policy,
    )
    stopped = evaluate_diversity_stop(history, policy)
    assert screened.status == "exhausted"
    assert stopped.should_stop is True
    assert stopped.reason == "no_improvement_patience"


def test_history_store_is_idempotent(tmp_path: Path) -> None:
    entry = _history(_proposal("box_8", box=8.0), round_index=1)
    store = ExplorationHistoryStore(tmp_path / "exploration_history.jsonl")
    assert store.append([entry]) == [entry]
    assert store.append([entry]) == []
    assert store.read() == [entry]
