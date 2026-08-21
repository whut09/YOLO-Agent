"""Typed readiness states shared by paper preflight and ASHA admission.

The states in this module are authorization states, not accuracy claims.  In
particular, ``pre_registered`` records that a candidate has an identity in the
ledger; it never grants a training assignment.
"""

from __future__ import annotations

from typing import Literal


ReadinessState = Literal[
    "inventory_seen",
    "contract_ready",
    "cpu_ready",
    "runtime_ready",
    "asha_eligible",
    "pre_registered",
    "blocked",
    "incompatible",
]


READINESS_STATES: tuple[ReadinessState, ...] = (
    "inventory_seen",
    "contract_ready",
    "cpu_ready",
    "runtime_ready",
    "asha_eligible",
    "pre_registered",
    "blocked",
    "incompatible",
)


def validate_readiness_state(
    state: ReadinessState,
    *,
    cpu_checks_passed: bool,
    runtime_checks_passed: bool,
    matched_control_passed: bool,
    inference_only: bool,
    blocker: str | None,
) -> None:
    """Reject readiness states that grant more authorization than evidence allows."""
    if state == "asha_eligible":
        if inference_only:
            raise ValueError("inference-only candidate cannot be ASHA eligible")
        if not cpu_checks_passed:
            raise ValueError("ASHA eligibility requires CPU readiness")
        if not runtime_checks_passed:
            raise ValueError("ASHA eligibility requires runtime readiness")
        if not matched_control_passed:
            raise ValueError("ASHA eligibility requires a matched baseline")
        if blocker:
            raise ValueError("ASHA-eligible state cannot retain a blocker")
        return
    if state == "runtime_ready" and not runtime_checks_passed:
        raise ValueError("runtime-ready state requires runtime checks")
    if state == "cpu_ready" and not cpu_checks_passed:
        raise ValueError("CPU-ready state requires CPU checks")
    if state == "contract_ready" and not cpu_checks_passed:
        raise ValueError("contract-ready state requires contract checks")
    if state in {"blocked", "incompatible"} and not blocker:
        raise ValueError(f"{state} state requires an exact blocker")


def legacy_readiness_state(
    *,
    asha_eligibility: bool,
    final_disposition: str,
    exact_blocker: str | None,
) -> ReadinessState:
    """Map pre-state-machine artifacts to a conservative explicit state."""
    if asha_eligibility and final_disposition == "runtime_ready" and not exact_blocker:
        return "asha_eligible"
    if final_disposition == "incompatible":
        return "incompatible"
    if exact_blocker:
        return "blocked"
    if final_disposition == "runtime_ready":
        return "runtime_ready"
    return "pre_registered"


__all__ = [
    "READINESS_STATES",
    "ReadinessState",
    "legacy_readiness_state",
    "validate_readiness_state",
]
