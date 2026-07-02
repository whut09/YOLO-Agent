"""Stage contract and event log tests."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.core.event_log import EventLog
from yolo_agent.core.stage_contract import LoopStageContracts, StageContract


def test_stage_contract_checks_required_items() -> None:
    """Stage contracts should report missing required inputs."""
    contract = StageContract(
        id="diagnose_errors",
        requires=["dataset_report", "detection_errors"],
        provides=["loop_diagnosis"],
        block_on_missing=True,
    )

    result = contract.check({"dataset_report"})

    assert result.ok is False
    assert result.missing_required == ["detection_errors"]
    assert result.warnings


def test_loop_policy_yaml_loads_executable_contracts() -> None:
    """Loop policy YAML should expose stage contracts, retry policy, and artifacts."""
    contracts = LoopStageContracts.from_yaml("configs/loop_policy.yaml")

    assert contracts.stage_order[0] == "init"
    diagnose = contracts.get("diagnose_errors")
    assert diagnose.requires == ["task_spec", "dataset_report", "detection_errors"]
    assert diagnose.provides == ["loop_diagnosis"]
    assert diagnose.block_on_missing is True
    assert diagnose.retry_policy.max_attempts == 1
    assert diagnose.producer_artifacts["loop_diagnosis"] == "artifacts/loop_diagnosis.json"


def test_event_log_appends_jsonl_entries(tmp_path: Path) -> None:
    """EventLog should persist append-only JSONL entries."""
    log = EventLog(tmp_path / "events.jsonl")

    log.append(
        run_id="run-1",
        event_type="stage_completed",
        stage="profile_data",
        status="completed",
        message="profile done",
        artifacts={"dataset_report": tmp_path / "dataset_report.json"},
    )

    entries = log.read()
    assert len(entries) == 1
    assert entries[0].run_id == "run-1"
    assert entries[0].stage == "profile_data"
    assert entries[0].artifacts["dataset_report"] == tmp_path / "dataset_report.json"
