from __future__ import annotations

import pytest

from yolo_agent.core.readiness_state import (
    legacy_readiness_state,
    validate_readiness_state,
)


def test_asha_eligible_requires_all_authorization_inputs() -> None:
    validate_readiness_state(
        "asha_eligible",
        cpu_checks_passed=True,
        runtime_checks_passed=True,
        matched_control_passed=True,
        inference_only=False,
        blocker=None,
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"cpu_checks_passed": False}, "CPU readiness"),
        ({"runtime_checks_passed": False}, "runtime readiness"),
        ({"matched_control_passed": False}, "matched baseline"),
        ({"inference_only": True}, "inference-only"),
        ({"blocker": "teacher_checkpoint_missing"}, "blocker"),
    ],
)
def test_asha_eligible_rejects_incomplete_evidence(kwargs: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "cpu_checks_passed": True,
        "runtime_checks_passed": True,
        "matched_control_passed": True,
        "inference_only": False,
        "blocker": None,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        validate_readiness_state("asha_eligible", **values)  # type: ignore[arg-type]


def test_pre_registered_is_not_training_authorization() -> None:
    validate_readiness_state(
        "pre_registered",
        cpu_checks_passed=False,
        runtime_checks_passed=False,
        matched_control_passed=False,
        inference_only=False,
        blocker=None,
    )


def test_blocked_and_incompatible_require_exact_blocker() -> None:
    for state in ("blocked", "incompatible"):
        with pytest.raises(ValueError, match="exact blocker"):
            validate_readiness_state(
                state,  # type: ignore[arg-type]
                cpu_checks_passed=False,
                runtime_checks_passed=False,
                matched_control_passed=False,
                inference_only=state == "incompatible",
                blocker=None,
            )


def test_legacy_runtime_ready_artifact_maps_conservatively() -> None:
    assert legacy_readiness_state(
        asha_eligibility=True,
        final_disposition="runtime_ready",
        exact_blocker=None,
    ) == "asha_eligible"
    assert legacy_readiness_state(
        asha_eligibility=False,
        final_disposition="runtime_ready",
        exact_blocker="cpu_certification_missing:neck",
    ) == "blocked"
