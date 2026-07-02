"""Decision ledger tests."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.core.decision_ledger import DecisionLedger, DecisionLedgerRecord


def test_decision_ledger_writes_and_reads_jsonl(tmp_path: Path) -> None:
    """DecisionLedger should persist audit records as JSONL."""
    ledger = DecisionLedger(tmp_path / "decision_ledger.jsonl")
    record = DecisionLedgerRecord(
        run_id="run-001",
        policy_id="policy-001",
        proposal={"policy_id": "policy-001", "components": ["loss.bbox.nwd"]},
        decision="accepted",
        deployment_constraints=[{"name": "max_latency_ms", "value": 30, "hard": True}],
        compatibility_warnings=["p2 head may increase latency"],
        created_candidate_id="candidate-001",
        created_node_id="node-candidate-001",
    )

    ledger.write([record])
    records = ledger.read()

    assert len(records) == 1
    assert records[0].policy_id == "policy-001"
    assert records[0].decision == "accepted"
    assert records[0].proposal["components"] == ["loss.bbox.nwd"]
    assert records[0].created_candidate_id == "candidate-001"
