"""Experiment memory anti-repetition semantics (Prompt-18J §4)."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.core.detection_error_profile import (
    DetectionErrorProfile,
    FalseNegativeFacts,
    FalsePositiveFacts,
    GlobalMetrics,
    ScaleMetrics,
)
from yolo_agent.core.experiment_memory import ExperimentMemory


def _profile(profile_id: str, candidate_id: str, fn_total: int = 20) -> DetectionErrorProfile:
    return DetectionErrorProfile(
        profile_id=profile_id,
        run_id="run-18j",
        candidate_id=candidate_id,
        global_=GlobalMetrics(map50=0.5, map50_95=0.3, precision=0.7, recall=0.6),
        scale=ScaleMetrics(ap_small=0.3, ap_medium=0.5, ap_large=0.6),
        false_negative=FalseNegativeFacts(total=fn_total),
        false_positive=FalsePositiveFacts(total=5),
    )


def test_failed_experiment_blocks_identical_repeat() -> None:
    memory = ExperimentMemory()
    profile = _profile("p1", "cand")
    kwargs = {
        "action_id": "train.neck.p2",
        "parameters": {"epochs": 30},
        "parent_fingerprint": "parent-abc",
    }

    assert memory.check_repeat(profile=profile, **kwargs) is None

    memory.record(
        profile=profile, outcome="failed", candidate_id="cand-1", **kwargs
    )
    blocked = memory.check_repeat(profile=profile, **kwargs)
    assert blocked is not None
    assert blocked.outcome == "failed"


def test_blocking_is_run_and_round_id_independent() -> None:
    """The same problem + action + parameters + parent must not repeat even
    if the loop renames the run/candidate."""
    memory = ExperimentMemory()
    first = _profile("p1", "cand-a")
    second = _profile("p9", "cand-b")
    kwargs = {
        "action_id": "train.neck.p2",
        "parameters": {"epochs": 30},
        "parent_fingerprint": "parent-abc",
    }
    memory.record(profile=first, outcome="rejected", candidate_id="cand-a", **kwargs)

    assert memory.check_repeat(profile=second, **kwargs) is not None


def test_changed_parameters_are_a_new_experiment() -> None:
    memory = ExperimentMemory()
    profile = _profile("p1", "cand")
    memory.record(
        profile=profile,
        action_id="train.neck.p2",
        parameters={"epochs": 30},
        parent_fingerprint="parent-abc",
        outcome="failed",
        candidate_id="cand-1",
    )

    assert (
        memory.check_repeat(
            profile=profile,
            action_id="train.neck.p2",
            parameters={"epochs": 45},
            parent_fingerprint="parent-abc",
        )
        is None
    )


def test_successful_outcomes_do_not_block() -> None:
    memory = ExperimentMemory()
    profile = _profile("p1", "cand")
    kwargs = {
        "action_id": "train.neck.p2",
        "parameters": {"epochs": 30},
        "parent_fingerprint": "parent-abc",
    }
    memory.record(profile=profile, outcome="promoted", candidate_id="cand-1", **kwargs)

    assert memory.check_repeat(profile=profile, **kwargs) is None


def test_reset_explicitly_allows_retry() -> None:
    memory = ExperimentMemory()
    profile = _profile("p1", "cand")
    kwargs = {
        "action_id": "train.neck.p2",
        "parameters": {"epochs": 30},
        "parent_fingerprint": "parent-abc",
    }
    memory.record(profile=profile, outcome="failed", candidate_id="cand-1", **kwargs)

    assert memory.reset(profile=profile, **kwargs) is True
    assert memory.check_repeat(profile=profile, **kwargs) is None


def test_memory_persists_to_disk(tmp_path: Path) -> None:
    path = tmp_path / "experiment_memory.yaml"
    profile = _profile("p1", "cand")
    kwargs = {
        "action_id": "train.neck.p2",
        "parameters": {"epochs": 30},
        "parent_fingerprint": "parent-abc",
    }

    first = ExperimentMemory(path)
    first.record(profile=profile, outcome="failed", candidate_id="cand-1", **kwargs)

    reloaded = ExperimentMemory(path)
    assert reloaded.check_repeat(profile=profile, **kwargs) is not None
