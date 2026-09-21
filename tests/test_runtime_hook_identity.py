"""Prompt-18G: canonical RuntimeHookIdentity model tests.

These pin the auditable runtime identity that replaces ``unknown`` hook
strings in the preflight artifact.  No training, no GPU, no CUDA — every
probe imports real modules and hashes real source files.
"""

from __future__ import annotations

import pytest

from yolo_agent.research.runtime_hook_identity import (
    RUNTIME_HOOK_PHASES,
    RuntimeHookIdentity,
    RuntimeHookIdentityError,
)

_REAL = dict(
    hook_id="hook.assignment.assigner.optimal_transport",
    phase="assignment",
    implementation_path="yolo_agent.components.adapters.assigners.yolo26_assignment",
    class_name="YOLO26AssignmentAdapter",
    method_name="smoke_test",
    insertion_point="one_to_many_assignment",
    component_id="assigner.optimal_transport",
    paper_id="arxiv:2103.14259",
    paper_binding_id="arxiv:2103.14259/assigner.optimal_transport",
)


def test_real_contract_binding_resolves_with_source_hash() -> None:
    identity = RuntimeHookIdentity.resolve(**_REAL)
    assert identity.source_sha256, "the real adapter module must hash to a pin"
    assert len(identity.source_sha256) == 64
    assert identity.binding_key == (
        "arxiv:2103.14259::arxiv:2103.14259/assigner.optimal_transport"
    )


@pytest.mark.parametrize("bad", ["unknown", "hook.unknown", "loss_unknown", "UNKNOWN"])
def test_unknown_hook_ids_are_never_valid_identities(bad: str) -> None:
    fields = dict(_REAL, hook_id=bad)
    with pytest.raises(RuntimeHookIdentityError, match="unknown"):
        RuntimeHookIdentity.resolve(**fields)


def test_missing_callable_is_rejected() -> None:
    with pytest.raises(RuntimeHookIdentityError, match="missing callable"):
        RuntimeHookIdentity.resolve(**dict(_REAL, class_name="NoSuchAdapter"))
    with pytest.raises(RuntimeHookIdentityError, match="missing callable"):
        RuntimeHookIdentity.resolve(**dict(_REAL, method_name="no_such_method"))
    with pytest.raises(RuntimeHookIdentityError, match="missing callable"):
        RuntimeHookIdentity.resolve(
            **dict(_REAL, implementation_path="yolo_agent.research.no_such_module")
        )


def test_invalid_source_path_is_rejected() -> None:
    with pytest.raises(RuntimeHookIdentityError, match="dotted import path"):
        RuntimeHookIdentity.resolve(**dict(_REAL, implementation_path="not_a_module_path"))


def test_paper_specific_binding_is_mandatory() -> None:
    with pytest.raises(RuntimeHookIdentityError, match="paper-specific binding"):
        RuntimeHookIdentity.resolve(**dict(_REAL, paper_binding_id=""))


def test_invalid_phase_is_rejected() -> None:
    with pytest.raises(RuntimeHookIdentityError):
        RuntimeHookIdentity.resolve(**dict(_REAL, phase="quantum"))
    assert set(RUNTIME_HOOK_PHASES) == {
        "data",
        "preprocess",
        "model_graph",
        "loss",
        "assignment",
        "optimizer",
        "train_step",
        "postprocess",
        "inference",
        "evaluation",
    }


def test_shared_adapter_yields_two_independent_paper_bindings() -> None:
    """Same implementation_path, two papers -> two distinct audited identities."""
    shared = dict(
        hook_id="loss.quality",
        phase="loss",
        implementation_path="yolo_agent.components.adapters.losses.quality_alignment",
        class_name="QualityAlignmentAuxiliaryLossAdapter",
        method_name="smoke_test",
        insertion_point="trainer_loss",
    )
    first = RuntimeHookIdentity.resolve(
        **shared,
        component_id="loss.quality.pseudo_iou",
        paper_id="arxiv:2104.14082",
        paper_binding_id="arxiv:2104.14082/loss.quality",
    )
    second = RuntimeHookIdentity.resolve(
        **shared,
        component_id="loss.quality.correlation",
        paper_id="arxiv:2301.01019",
        paper_binding_id="arxiv:2301.01019/loss.quality",
    )
    assert first.binding_key != second.binding_key
    assert first.source_sha256 == second.source_sha256  # same code, distinct bindings


def test_verify_source_detects_drift() -> None:
    identity = RuntimeHookIdentity.resolve(**_REAL)  # pins the current source hash
    identity.verify_source()  # fresh pin must verify cleanly
    stale = RuntimeHookIdentity.resolve(
        **dict(_REAL, source_sha256="0" * 64)
    )
    assert identity.source_sha256 and identity.source_sha256 != stale.source_sha256
    with pytest.raises(RuntimeHookIdentityError, match="source drift"):
        stale.verify_source()


def test_round_trip_record_dict() -> None:
    identity = RuntimeHookIdentity.resolve(**_REAL)
    payload = identity.to_record_dict()
    assert payload["phase"] == "assignment"
    assert payload["paper_id"] == "arxiv:2103.14259"
    assert set(payload) == {
        "hook_id",
        "phase",
        "implementation_path",
        "class_name",
        "method_name",
        "insertion_point",
        "component_id",
        "paper_id",
        "source_sha256",
        "paper_binding_id",
    }
