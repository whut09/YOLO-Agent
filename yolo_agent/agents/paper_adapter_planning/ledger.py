"""Decision Ledger integration for implementation queue planning."""

from __future__ import annotations

from yolo_agent.agents.paper_adapter_planning.schemas import PaperAdapterImplementationPlan
from yolo_agent.core.decision_ledger import DecisionLedger, DecisionLedgerRecord


def record_implementation_plan(
    ledger: DecisionLedger,
    *,
    run_id: str,
    plan: PaperAdapterImplementationPlan,
) -> DecisionLedgerRecord:
    actionable = [
        *plan.ready_to_materialize,
        *plan.implementation_queue,
        *plan.shadow_evaluation_queue,
    ]
    return ledger.append(DecisionLedgerRecord(
        run_id=run_id,
        policy_id="paper_adapter_implementation_planner",
        decision_type="paper_adapter_implementation_queue",
        proposal={
            "plan_hash": plan.plan_hash,
            "actionable_components": [item.component_id for item in actionable],
            "implementation_requests": [
                item.implementation_request.model_dump(mode="json")
                for item in plan.implementation_queue
                if item.implementation_request is not None
            ],
        },
        decision="planned" if actionable else "no_actionable_implementation",
        input_summary={
            "current_round": plan.current_round,
            "queue_counts": plan.summary,
            "auto_code_generation": plan.auto_code_generation,
        },
        blocked_by=[
            *[f"incompatible:{item.component_id}" for item in plan.incompatible],
            *[
                f"separate_detector_family:{item.component_id}"
                for item in plan.separate_detector_family
            ],
        ],
        rationale=(
            "Deterministic diagnosis-driven implementation planning; paper applicability "
            "does not grant execution and no adapter code was generated."
        ),
        policy_version="paper_adapter_implementation_plan.v1",
    ))


__all__ = ["record_implementation_plan"]
