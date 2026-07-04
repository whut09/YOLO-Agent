"""Stage contract and event log tests."""

from __future__ import annotations

from pathlib import Path

import json

from yolo_agent.core.artifact_manifest import ArtifactManifestEntry
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.event_log import EventLog
from yolo_agent.core.stage_contract import ArtifactContract, LoopStageContracts, StageContract


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
    assert contracts.stage_order[-3:] == ["mine_samples", "label_handoff", "dataset_promote"]
    diagnose = contracts.get("diagnose_errors")
    assert diagnose.requires == ["task_spec", "dataset_report", "detection_errors"]
    assert diagnose.provides == ["loop_diagnosis"]
    assert diagnose.block_on_missing is True
    assert diagnose.retry_policy.max_attempts == 1
    assert diagnose.producer_artifacts["loop_diagnosis"] == "artifacts/loop_diagnosis.json"
    assert diagnose.artifact_contract["dataset_report"].schema_name == "DatasetReport"
    assert diagnose.artifact_contract["dataset_report"].sha_required is True
    label_handoff = contracts.get("label_handoff")
    assert label_handoff.requires == ["active_learning_plan"]
    assert label_handoff.artifact_contract["active_learning_plan"].schema_name == "ActiveLearningPlan"
    dataset_promote = contracts.get("dataset_promote")
    assert dataset_promote.requires == ["labeling_manifest", "dataset_manifest"]
    assert dataset_promote.artifact_contract["labeling_manifest"].schema_name == "LabelingManifest"


def test_stage_contract_checks_required_evidence_when_evidence_is_supplied(tmp_path: Path) -> None:
    """evidence_required should be evaluated separately from raw input availability."""
    store = EvidenceStore(tmp_path / "runs")
    store.create_run("stage-gate")
    contract = StageContract(
        id="report",
        requires=["run_context"],
        evidence_required=["dataset_report", "map50"],
    )

    result = contract.check(
        {"run_context"},
        evidence=store.load_run("stage-gate"),
    )

    assert result.ok is False
    assert result.missing_required == []
    assert result.missing_evidence == ["dataset_report", "map50"]
    assert result.evidence_gate is not None
    assert result.evidence_gate.trusted is False


def test_stage_contract_accepts_required_evidence(tmp_path: Path) -> None:
    """Stage evidence gate should pass with present artifact and metric evidence."""
    store = EvidenceStore(tmp_path / "runs")
    run_dir = store.create_run("stage-gate-ok")
    (run_dir / "artifacts" / "dataset_report.json").write_text("{}", encoding="utf-8")
    store.log_metrics("stage-gate-ok", {"map50": 0.5})
    contract = StageContract(
        id="report",
        requires=["run_context"],
        evidence_required=["dataset_report", "map50"],
    )

    result = contract.check(
        {"run_context"},
        evidence=store.load_run("stage-gate-ok"),
    )

    assert result.ok is True
    assert result.missing_evidence == []


def test_stage_contract_can_warn_on_missing_evidence_without_blocking(tmp_path: Path) -> None:
    """Some stages, such as reporting, should run while carrying evidence warnings."""
    store = EvidenceStore(tmp_path / "runs")
    store.create_run("stage-gate-warning")
    contract = StageContract(
        id="report",
        requires=["run_context"],
        evidence_required=["map50"],
        block_on_missing=True,
        block_on_missing_evidence=False,
    )

    result = contract.check(
        {"run_context"},
        evidence=store.load_run("stage-gate-warning"),
    )

    assert result.ok is True
    assert result.missing_required == []
    assert result.missing_evidence == ["map50"]
    assert result.warnings == ["Missing required evidence for report: map50"]


def test_artifact_contract_validates_manifest_sha_schema_and_freshness(tmp_path: Path) -> None:
    """ArtifactContract should require a current-run manifest entry with valid content."""
    run_dir = tmp_path / "runs" / "run-1"
    artifact_path = run_dir / "artifacts" / "dataset_report.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(
        json.dumps(
            {
                "data_yaml": "data.yaml",
                "dataset_root": ".",
                "scene": "generic",
                "image_count": 1,
                "label_count": 0,
            }
        ),
        encoding="utf-8",
    )
    entry = ArtifactManifestEntry.from_path("dataset_report", artifact_path, "profile_data")
    contract = StageContract(
        id="diagnose_errors",
        requires=["dataset_report"],
        artifact_contract={"dataset_report": ArtifactContract(schema="DatasetReport")},
    )

    valid = contract.check({"dataset_report"}, [entry], run_dir)

    assert valid.ok is True

    artifact_path.write_text(
        json.dumps(
            {
                "data_yaml": "data.yaml",
                "dataset_root": ".",
                "scene": "generic",
                "image_count": 2,
                "label_count": 0,
            }
        ),
        encoding="utf-8",
    )
    tampered = contract.check({"dataset_report"}, [entry], run_dir)

    assert tampered.ok is False
    assert tampered.invalid_artifacts == ["dataset_report: sha256 verification failed"]


def test_artifact_contract_rejects_artifact_from_another_run(tmp_path: Path) -> None:
    """current_run freshness should reject a manifest path outside the run directory."""
    run_dir = tmp_path / "runs" / "run-1"
    other_path = tmp_path / "other" / "dataset_report.json"
    other_path.parent.mkdir(parents=True)
    other_path.write_text(
        json.dumps(
            {
                "data_yaml": "data.yaml",
                "dataset_root": ".",
                "image_count": 1,
                "label_count": 0,
            }
        ),
        encoding="utf-8",
    )
    entry = ArtifactManifestEntry.from_path("dataset_report", other_path, "profile_data")
    contract = StageContract(
        id="diagnose_errors",
        requires=["dataset_report"],
        artifact_contract={"dataset_report": ArtifactContract(schema="DatasetReport")},
    )

    result = contract.check({"dataset_report"}, [entry], run_dir)

    assert result.ok is False
    assert result.invalid_artifacts == ["dataset_report: artifact is not from current run"]


def test_artifact_contract_rejects_hash_mismatch(tmp_path: Path) -> None:
    """A tampered artifact file should fail sha256 verification."""
    run_dir = tmp_path / "runs" / "run-1"
    artifact_path = run_dir / "artifacts" / "dataset_report.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(
        json.dumps(
            {
                "data_yaml": "data.yaml",
                "dataset_root": ".",
                "scene": "generic",
                "image_count": 1,
                "label_count": 0,
            }
        ),
        encoding="utf-8",
    )
    entry = ArtifactManifestEntry.from_path("dataset_report", artifact_path, "profile_data")
    artifact_path.write_text(
        json.dumps(
            {
                "data_yaml": "data.yaml",
                "dataset_root": ".",
                "scene": "generic",
                "image_count": 2,
                "label_count": 0,
            }
        ),
        encoding="utf-8",
    )
    contract = StageContract(
        id="diagnose_errors",
        requires=["dataset_report"],
        artifact_contract={"dataset_report": ArtifactContract(schema="DatasetReport")},
    )

    result = contract.check({"dataset_report"}, [entry], run_dir)

    assert result.ok is False
    assert result.invalid_artifacts == ["dataset_report: sha256 verification failed"]


def test_artifact_contract_rejects_stale_artifact_from_other_run(tmp_path: Path) -> None:
    """A manifest entry outside the current run directory should be rejected."""
    run_dir = tmp_path / "runs" / "run-1"
    other_path = tmp_path / "other" / "dataset_report.json"
    other_path.parent.mkdir(parents=True)
    other_path.write_text(
        json.dumps(
            {
                "data_yaml": "data.yaml",
                "dataset_root": ".",
                "scene": "generic",
                "image_count": 1,
                "label_count": 0,
            }
        ),
        encoding="utf-8",
    )
    entry = ArtifactManifestEntry.from_path("dataset_report", other_path, "profile_data")
    contract = StageContract(
        id="diagnose_errors",
        requires=["dataset_report"],
        artifact_contract={"dataset_report": ArtifactContract(schema="DatasetReport")},
    )

    result = contract.check({"dataset_report"}, [entry], run_dir)

    assert result.ok is False
    assert result.invalid_artifacts == ["dataset_report: artifact is not from current run"]


def test_artifact_contract_rejects_unknown_schema_name(tmp_path: Path) -> None:
    """An unknown schema name should fail validation."""
    run_dir = tmp_path / "runs" / "run-1"
    artifact_path = run_dir / "artifacts" / "dataset_report.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(
        json.dumps(
            {
                "data_yaml": "data.yaml",
                "dataset_root": ".",
                "scene": "generic",
                "image_count": 1,
                "label_count": 0,
            }
        ),
        encoding="utf-8",
    )
    entry = ArtifactManifestEntry.from_path("dataset_report", artifact_path, "profile_data")
    contract = StageContract(
        id="diagnose_errors",
        requires=["dataset_report"],
        artifact_contract={"dataset_report": ArtifactContract(schema="UnknownSchema")},
    )

    result = contract.check({"dataset_report"}, [entry], run_dir)

    assert result.ok is False
    assert result.invalid_artifacts == ["dataset_report: unknown schema UnknownSchema"]


def test_artifact_contract_rejects_wrong_schema_content(tmp_path: Path) -> None:
    """Artifact content that fails pydantic schema validation should be blocked."""
    run_dir = tmp_path / "runs" / "run-1"
    artifact_path = run_dir / "artifacts" / "dataset_report.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(
        json.dumps(
            {
                "data_yaml": "data.yaml",
                "dataset_root": ".",
                "scene": "generic",
                "image_count": 1,
                "label_count": 0,
            }
        ),
        encoding="utf-8",
    )
    entry = ArtifactManifestEntry.from_path("dataset_report", artifact_path, "profile_data")
    artifact_path.write_text(
        json.dumps(
            {
                "data_yaml": 123,
                "dataset_root": ".",
                "scene": "generic",
                "image_count": 1,
                "label_count": 0,
            }
        ),
        encoding="utf-8",
    )
    contract = StageContract(
        id="diagnose_errors",
        requires=["dataset_report"],
        artifact_contract={
            "dataset_report": ArtifactContract(
                schema="DatasetReport",
                sha_required=False,
            )
        },
    )

    result = contract.check({"dataset_report"}, [entry], run_dir)

    assert result.ok is False
    assert len(result.invalid_artifacts) == 1
    assert result.invalid_artifacts[0].startswith("dataset_report: schema DatasetReport validation failed")


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
