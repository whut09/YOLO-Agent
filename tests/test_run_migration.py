"""Legacy run isolation and migration-report tests."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.agents.optimize_runner import OptimizeRunner
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.run_context import RunContext
from yolo_agent.core.run_migration import assess_run_protocol, write_migration_report
from yolo_agent.core.run_protocol import RunProtocolVersion


def _legacy_context(tmp_path: Path) -> RunContext:
    context = RunContext(
        run_id="legacy-run",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "data.yaml",
        dataset_version="coco2017",
    )
    context.ensure_dirs()
    context.to_yaml()
    context.to_json()
    return context


def test_legacy_run_report_quarantines_current_candidate_metrics(tmp_path: Path) -> None:
    context = _legacy_context(tmp_path)
    store = EvidenceStore(context.run_root)
    store.log_candidate_metrics(
        run_id=context.run_id,
        candidate_id="candidate",
        node_id="node-candidate",
        metrics={"map50_95": 0.4},
        protocol_hash="protocol",
        dataset_manifest_sha256="dataset",
        subset_manifest_sha256="subset",
        eval_protocol_hash="eval",
        seed=42,
        epochs=10,
        batch_policy_hash="batch",
        ultralytics_version="9.0.0",
        imgsz=640,
    )

    assessment = assess_run_protocol(context, store)
    assert assessment.legacy_run is True
    assert "missing_run_protocol" in assessment.reasons
    assert "missing_asha_state" in assessment.reasons
    assert "missing_objective_hash" in assessment.reasons
    assert assessment.latest_trusted_node_id == "node-candidate"

    report = write_migration_report(context, assessment)
    quarantined = store.load_run(context.run_id).metric_records[0]
    assert report.action == "start_new_run"
    assert report.suggested_run_id == "legacy-run-v2"
    assert quarantined.evidence_role == "inherited_context"
    assert quarantined.inheritance_depth == 1
    assert quarantined.source.startswith("legacy:")


def test_optimize_runner_blocks_legacy_run_and_suggests_new_id(tmp_path: Path, monkeypatch) -> None:
    context = _legacy_context(tmp_path)
    context.data_yaml.write_text("path: .\\ntrain: train\\nval: val\\nnames: []\\n", encoding="utf-8")
    monkeypatch.setattr("yolo_agent.agents.optimize_runner.optimize_preflight", lambda *args, **kwargs: [])

    result = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=context.data_yaml,
        run_id=context.run_id,
        run_root=context.run_root,
        execute=False,
    )

    assert result.ok is False
    assert result.migration_suggested_run_id == "legacy-run-v2"
    assert result.migration_report_path is not None and result.migration_report_path.is_file()
    assert any(check.name == "legacy_run" for check in result.preflight)


def test_run_with_unbound_post_eval_protocol_is_legacy(tmp_path: Path) -> None:
    context = _legacy_context(tmp_path)
    protocol = RunProtocolVersion(
        model="yolo26n.pt",
        dataset_version="coco2017",
        dataset_manifest_sha256="dataset",
        subset_manifest_sha256="subset",
        imgsz=640,
        epochs=10,
        seed=42,
        batch_policy={"mode": "auto"},
        batch_policy_hash="batch",
        ultralytics_version="9.0.0",
        eval_protocol={"split": "val", "save_json": True},
        eval_protocol_hash="eval",
        code_version="commit",
        profile="pilot",
    )
    protocol_path = protocol.to_yaml(context.artifact_path("run_protocol.yaml"))
    context.run_protocol_path = protocol_path
    context.run_protocol_hash = protocol.protocol_hash
    context.metadata["optimization_objective_hash"] = "objective"
    objective_path = context.artifact_path("optimization_objective.yaml")
    objective_path.write_text("schema_version: optimization_objective.v1\n", encoding="utf-8")
    context.metadata["optimization_objective_path"] = objective_path.as_posix()
    context.to_yaml()

    assessment = assess_run_protocol(context, EvidenceStore(context.run_root))

    assert assessment.legacy_run is True
    assert "missing_context_post_eval_protocol" in assessment.reasons
