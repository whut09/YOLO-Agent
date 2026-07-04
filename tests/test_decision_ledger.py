"""Decision ledger tests."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.core.decision_ledger import (
    DecisionLedger,
    DecisionLedgerRecord,
    build_replay_snapshot,
    sha256_model,
    sha256_path,
)


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


def test_decision_replay_snapshot_hashes_inputs(tmp_path: Path) -> None:
    """Replay snapshots should store stable hashes for files, directories, and evidence gates."""
    task_path = tmp_path / "task.yaml"
    task_path.write_text("task_type: detect\n", encoding="utf-8")
    registry_dir = tmp_path / "components"
    registry_dir.mkdir()
    (registry_dir / "loss.yaml").write_text("id: loss.bbox.nwd\n", encoding="utf-8")
    loop_plan = tmp_path / "loop_plan.yaml"
    loop_plan.write_text("candidate_policies: []\n", encoding="utf-8")
    evidence_gate = {"trusted": False, "missing_required": ["map50"]}

    snapshot = build_replay_snapshot(
        task_spec_path=task_path,
        component_registry_path=registry_dir,
        loop_plan_path=loop_plan,
        evidence_gate=evidence_gate,
        policy_version="LoopPolicyEvaluator@test",
    )
    record = DecisionLedgerRecord(
        run_id="run-001",
        policy_id="policy-001",
        decision="needs_evidence",
        replay_snapshot=snapshot,
        task_spec_sha256=snapshot.task_spec_sha256,
        component_registry_sha256=snapshot.component_registry_sha256,
        loop_plan_sha256=snapshot.loop_plan_sha256,
        evidence_gate_sha256=snapshot.evidence_gate_sha256,
        policy_version=snapshot.policy_version,
    )

    DecisionLedger(tmp_path / "ledger.jsonl").write([record])
    restored = DecisionLedger(tmp_path / "ledger.jsonl").read()[0]

    assert restored.task_spec_sha256 == sha256_path(task_path)
    assert restored.component_registry_sha256 == sha256_path(registry_dir)
    assert restored.loop_plan_sha256 == sha256_path(loop_plan)
    assert restored.evidence_gate_sha256 == sha256_model(evidence_gate)
    assert restored.policy_version == "LoopPolicyEvaluator@test"
    assert restored.replay_snapshot is not None
    assert restored.replay_snapshot.evidence_gate_sha256 == restored.evidence_gate_sha256
