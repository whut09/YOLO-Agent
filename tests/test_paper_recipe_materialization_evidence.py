from __future__ import annotations

from yolo_agent.agents.paper_recipe_materialization.evidence import (
    current_materialization_error_facts,
    evidence_recovery_for_facts,
)
from yolo_agent.agents.paper_recipe_materialization_gate import (
    PaperRecipeMaterializationGate,
)
from yolo_agent.core.error_facts import ErrorFact
from tests.paper_materialization_fixtures import (
    adapter_registry,
    candidate_input,
    error_fact,
    gate_kwargs,
)


def _fact(
    *,
    run_id: str = "run-1",
    protocol_hash: str = "protocol-1",
    evidence_role: str = "current_observation",
) -> ErrorFact:
    return ErrorFact(
        run_id=run_id,
        candidate_id="baseline",
        node_id="node-baseline",
        protocol_hash=protocol_hash,
        evidence_role=evidence_role,
        fact_type="area_metric",
        subject="small",
        metric_name="ap_small",
        value=0.2,
    )


def test_only_current_run_same_protocol_facts_can_materialize() -> None:
    facts = [
        _fact(evidence_role="inherited_context"),
        _fact(run_id="old-run"),
        _fact(protocol_hash="old-protocol"),
        _fact(),
    ]

    selected = current_materialization_error_facts(
        facts,
        run_id="run-1",
        protocol_hash="protocol-1",
    )

    assert selected == [facts[-1]]


def test_missing_current_protocol_fact_requires_non_training_recovery() -> None:
    recovery = evidence_recovery_for_facts(
        [_fact(evidence_role="inherited_context"), _fact(protocol_hash="old-protocol")],
        run_id="run-1",
        protocol_hash="protocol-1",
    )

    assert recovery is not None
    assert recovery.training_allowed is False
    assert recovery.required_evidence == [
        "current_node_coco_post_eval",
        "current_node_error_facts",
        "same_protocol_hash",
    ]


def test_gate_missing_facts_never_registers_or_queues_training(tmp_path) -> None:
    run_dir = tmp_path / "paper-run"
    gate = PaperRecipeMaterializationGate(
        run_dir,
        base_run_id="paper-run",
        adapter_registry=adapter_registry(),
    )
    values = gate_kwargs(tmp_path, candidates=[candidate_input()])
    values["current_error_facts"] = [
        error_fact(evidence_role="inherited_context"),
        error_fact(protocol_hash="old-protocol"),
    ]

    result = gate.materialize(**values)

    assert result.action == "evidence_recovery"
    assert result.evidence_recovery is not None
    assert result.evidence_recovery.training_allowed is False
    assert result.execution_queue is None
    assert not (run_dir / "execution_queue.yaml").exists()
    assert not (run_dir / "artifacts" / "asha_state.yaml").exists()
