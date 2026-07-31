"""Queue admission tests for component-specific GPU certification."""

from __future__ import annotations

from yolo_agent.certification.component_queue_gate import (
    ComponentQueueCertificationGate,
)
from tests.paper_materialization_fixtures import contract


def test_sampling_smoke_contract_allows_initial_pilot_without_pilot_10_report() -> None:
    component = contract(
        component_id="sampling.small_object",
        maturity="smoke_passed",
    )
    result = ComponentQueueCertificationGate().evaluate(
        component_ids=["sampling.small_object"],
        report_path=None,
        component_contracts={component.component_id: component},
    )

    assert result.allowed is True
    assert result.blockers == []
    assert result.checks["sampling.small_object:effective_smoke_passed"] is True


def test_missing_effective_contract_blocks_queue_admission() -> None:
    result = ComponentQueueCertificationGate().evaluate(
        component_ids=["sampling.small_object"],
        report_path=None,
    )

    assert result.allowed is False
    assert result.blockers == [
        "effective_maturity_contract_missing:sampling.small_object"
    ]


def test_pilot_report_cannot_replace_smoke_maturity() -> None:
    component = contract(
        component_id="sampling.small_object",
        maturity="adapter_implemented",
    )
    result = ComponentQueueCertificationGate().evaluate(
        component_ids=["sampling.small_object"],
        report_path="certification_report.yaml",
        component_contracts={component.component_id: component},
    )

    assert result.allowed is False
    assert result.blockers == [
        "effective_maturity_below_smoke_passed:sampling.small_object:"
        "adapter_implemented"
    ]


def test_unrelated_components_do_not_require_sampling_certification() -> None:
    component = contract(
        component_id="loss.quality.correlation",
        maturity="smoke_passed",
    )
    result = ComponentQueueCertificationGate().evaluate(
        component_ids=["loss.quality.correlation"],
        report_path=None,
        component_contracts={component.component_id: component},
    )

    assert result.allowed is True
