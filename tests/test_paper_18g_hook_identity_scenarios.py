"""Prompt-18G: fail-closed scenarios for runtime hook identities.

Pins the contract that ``unknown`` is no longer an acceptable runtime
identity, exercising the report-level demotion gate and the executor-level
resolution failures.  Everything here is CPU-only and training-free.

Scenarios:

1. a PASS record whose hooks say ``unknown``            -> demoted to FAIL;
2. a PASS record without resolved identities            -> demoted to FAIL;
3. a contract whose callable cannot be imported          -> executor FAIL;
4. an implementation path that is not a dotted module    -> executor FAIL;
5. a shared adapter + two papers                         -> two independent
   paper bindings with the same source hash.
"""

from __future__ import annotations

import pytest

from yolo_agent.components.contracts import ComponentContract
from yolo_agent.research.paper_runtime_preflight import (
    PaperRuntimePreflightRecord,
    PaperRuntimePreflightRunner,
    RuntimePreflightFailure,
    _run_adapter_component,
)


def _pass_record(**overrides: object) -> PaperRuntimePreflightRecord:
    defaults: dict[str, object] = {
        "paper_id": "arxiv:2103.14259",
        "implementation_domain": "assigner",
        "adapter_ids": ["assigner.optimal_transport"],
        "runtime_hooks": ["hook.assignment.optimal_transport"],
        "runtime_hook_identities": [
            {
                "hook_id": "hook.assignment.optimal_transport",
                "phase": "assignment",
                "implementation_path": "yolo_agent.components.adapters.assigners.yolo26_assignment",
                "class_name": "YOLO26AssignmentAdapter",
                "method_name": "smoke_test",
                "insertion_point": "one_to_many_assignment",
                "component_id": "assigner.optimal_transport",
                "paper_id": "arxiv:2103.14259",
                "source_sha256": "0" * 64,
                "paper_binding_id": "arxiv:2103.14259/assigner.optimal_transport",
            }
        ],
        "materialized": True,
        "synthetic_forward": True,
        "synthetic_backward": True,
        "behavior_changed": True,
        "rollback_verified": True,
        "fingerprint": "abc123",
        "status": "PASS",
        "error": "",
    }
    defaults.update(overrides)
    return PaperRuntimePreflightRecord(**defaults)  # type: ignore[arg-type]


def _finalize_single(record: PaperRuntimePreflightRecord):
    runner = PaperRuntimePreflightRunner()
    return runner._finalize([str(record.paper_id)], [record])


# ---------------------------------------------------------------------------
# 1 + 2: report-level demotion gate
# ---------------------------------------------------------------------------


def test_unknown_hook_pass_record_is_demoted_to_fail() -> None:
    record = _pass_record(runtime_hooks=["unknown"], runtime_hook_identities=[])
    report = _finalize_single(record)
    assert report.failed == 1
    assert report.passed == 0
    assert report.unknown_runtime_hooks == 1
    assert report.runtime_preflight_passed is False
    demoted = report.records[0]
    assert demoted.status == "FAIL"
    assert "unresolvable" in demoted.error


def test_legacy_unknown_hook_string_is_demoted() -> None:
    """The old artifact vocabulary ``runtime_hooks: [unknown]`` cannot pass."""
    record = _pass_record(runtime_hooks=["unknown"])
    report = _finalize_single(record)
    assert report.records[0].status == "FAIL"
    assert report.unknown_runtime_hooks == 1


def test_pass_record_without_resolved_identity_is_demoted() -> None:
    record = _pass_record(runtime_hook_identities=[])
    report = _finalize_single(record)
    assert report.records[0].status == "FAIL"
    assert report.runtime_preflight_passed is False


def test_resolved_identity_record_stays_pass() -> None:
    report = _finalize_single(_pass_record())
    assert report.passed == 1
    assert report.failed == 0
    assert report.unknown_runtime_hooks == 0
    assert report.runtime_preflight_passed is True


# ---------------------------------------------------------------------------
# 3 + 4: executor-level resolution failures
# ---------------------------------------------------------------------------


def _contract(component_id: str, implementation_path: str, adapter_class: str) -> ComponentContract:
    return ComponentContract(
        component_id=component_id,
        display_name="Probe",
        category="loss",
        implementation_path=implementation_path,
        adapter_class=adapter_class,
        insertion_point="trainer_loss",
    )


def test_missing_callable_contract_fails_the_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = _contract(
        "loss.probe",
        "yolo_agent.components.adapters.losses.quality_alignment",
        "NoSuchAdapterClass",
    )
    monkeypatch.setattr(
        "yolo_agent.research.paper_runtime_preflight._load_component_contracts",
        lambda: {"loss.probe": broken},
    )
    with pytest.raises(RuntimePreflightFailure, match="runtime hook identity unresolvable"):
        _run_adapter_component("loss.probe", paper_id="arxiv:probe")


def test_invalid_source_path_fails_the_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = _contract("loss.probe2", "not_a_dotted_module", "SomeClass")
    monkeypatch.setattr(
        "yolo_agent.research.paper_runtime_preflight._load_component_contracts",
        lambda: {"loss.probe2": broken},
    )
    with pytest.raises(RuntimePreflightFailure, match="runtime hook identity unresolvable"):
        _run_adapter_component("loss.probe2", paper_id="arxiv:probe")


# ---------------------------------------------------------------------------
# 5: shared adapter, independent bindings
# ---------------------------------------------------------------------------


def test_shared_adapter_two_papers_two_independent_bindings() -> None:
    from yolo_agent.research.paper_runtime_preflight import _load_component_contracts
    from yolo_agent.research.runtime_hook_resolution import identity_for_contract

    contracts = _load_component_contracts()
    first = identity_for_contract(
        contracts["loss.quality.pseudo_iou"], paper_id="arxiv:2104.14082"
    )
    second = identity_for_contract(
        contracts["loss.quality.correlation"], paper_id="arxiv:2301.01019"
    )
    # Same implementation module and source hash...
    assert first.implementation_path == second.implementation_path
    assert first.source_sha256 == second.source_sha256
    assert first.source_sha256, "shared adapter source must be hashable"
    # ...but two auditable, paper-specific bindings.
    assert first.binding_key != second.binding_key
    assert first.paper_id == "arxiv:2104.14082"
    assert second.paper_id == "arxiv:2301.01019"
